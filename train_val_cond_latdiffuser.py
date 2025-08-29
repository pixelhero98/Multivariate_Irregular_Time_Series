import os, math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from latent_vae import LatentVAE
from cond_diffusion_utils import normalize_and_check
from lladit import LLapDiT  # your conditional diffusion backbone

# ----------------------------- config -----------------------------
DATA_MODULE   = "fin_data_prep_ratiosp_indexcache"
DATA_DIR      = "./ldt/crypto_data"
WINDOW        = 150          # panel context length K
HORIZON       = 40           # target length H
COVERAGE      = 0.85

BATCH_SIZE    = 64
EPOCHS        = 100
BASE_LR       = 5e-4
WARMUP_FRAC   = 0.05
WEIGHT_DECAY  = 1e-2
GRAD_CLIP     = 1.0
EARLY_STOP    = 20

LATENT_DIM    = 64
VAE_LAYERS    = 3
VAE_HEADS     = 4
VAE_FF        = 256
VAE_CKPT      = ""           # optional path

TIMESTEPS     = 1500
SCHEDULE      = "cosine"     # ["cosine","linear","sigmoid"]
PREDICT_TYPE  = "v"          # ["v","eps"]
DROP_COND_P   = 0.25         # classifier-free guidance (drop conditioning prob)
SELF_COND     = True
SELF_COND_P   = 0.50
SNR_CLIP      = 5.0

MODEL_WIDTH   = 256
NUM_LAYERS    = 4
NUM_HEADS     = 4
LAPLACE_K     = [24, 20, 16, 16]
GLOBAL_K      = 64
DROPOUT       = 0.0
ATTN_DROPOUT  = 0.0
CONTEXT_LEN   = HORIZON      # learned summary tokens

USE_EWMA      = True         # two-stage latent whitening: per-window EWMA, then global
EWMA_LAMBDA   = 0.94

CKPT_DIR      = "./ldt/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

# ----------------------------- utils -----------------------------
def set_torch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    return device

def make_warmup_cosine(optimizer, total_steps, warmup_frac=0.05, base_lr=5e-4, min_lr=1e-6):
    warmup_steps = max(1, int(total_steps * warmup_frac))
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr / base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

def ewma_std(x: torch.Tensor, lam: float = 0.94, eps: float = 1e-8) -> torch.Tensor:
    """x: [B,L,D] -> [B,1,D]"""
    B,L,D = x.shape
    var = x.new_zeros(B,D); mean = x.new_zeros(B,D)
    for t in range(L):
        xt = x[:,t,:]
        mean = lam*mean + (1-lam)*xt
        var  = lam*var  + (1-lam)*(xt-mean)**2
    return (var+eps).sqrt().unsqueeze(1)

def two_stage_norm(mu: torch.Tensor, use_ewma: bool, ewma_lambda: float,
                   mu_mean: torch.Tensor, mu_std: torch.Tensor) -> torch.Tensor:
    """Window-wise std (EWMA or plain), then global whitening; clamp outliers."""
    if use_ewma:
        s = ewma_std(mu, lam=ewma_lambda)
    else:
        s = mu.std(dim=1, keepdim=True).clamp_min(1e-6)
    mu_w = mu / s
    mu_g = (mu_w - mu_mean) / (mu_std + 1e-8)
    return mu_g.clamp(-5, 5)

def diffusion_loss(model: LLapDiT, scheduler, x0_lat_norm: torch.Tensor, t: torch.Tensor,
                   *, cond_summary: Optional[torch.Tensor], predict_type: str="v",
                   self_cond: bool=False, self_cond_p: float=0.0, snr_clip: float=5.0) -> torch.Tensor:
    """SNR-weighted MSE on v/eps; optional self-conditioning."""
    noise = torch.randn_like(x0_lat_norm)
    x_t, eps_true = scheduler.q_sample(x0_lat_norm, t, noise)
    sc_feat = None
    if self_cond and (torch.rand(()) < self_cond_p):
        with torch.no_grad():
            pred_sc = model(x_t, t, cond_summary=cond_summary)
            x0_sc   = scheduler.to_x0(x_t, t, pred_sc, param_type=predict_type)
        sc_feat = x0_sc
    pred = model(x_t, t, cond_summary=cond_summary, sc_feat=sc_feat)
    target = eps_true if predict_type == "eps" else scheduler.v_from_eps(x_t, t, eps_true)
    ab = scheduler._gather(scheduler.alpha_bars, t).view(-1,1,1)
    snr = (ab / (1 - ab + 1e-8)).clamp_max(snr_clip)
    return (snr * (pred - target) ** 2).mean()

@torch.no_grad()
def compute_latent_stats(vae: LatentVAE, dataloader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect μ across train set to get global mean/std (for whitening)."""
    all_mu = []
    for (_, y, meta) in tqdm(dataloader, desc="Collect μ stats"):
        m = meta["entity_mask"]             # [B,N] bool
        B, N, H = y.shape
        y_flat = y.reshape(B*N, H).unsqueeze(-1)  # [B*N, H, 1]
        m_flat = m.reshape(B*N)
        if not m_flat.any():
            continue
        y_in = y_flat[m_flat].to(device)
        _, mu, _ = vae(y_in)
        all_mu.append(mu.detach().cpu())
    if len(all_mu) == 0:
        mu_cat = torch.zeros(1, 1, LATENT_DIM)
    else:
        mu_cat = torch.cat(all_mu, dim=0)
    _, mu_mean, mu_std = normalize_and_check(mu_cat)
    return mu_mean.to(device), mu_std.to(device)

# ----------------------------- main -----------------------------
device = set_torch()

# data module
mod = __import__(DATA_MODULE, fromlist=['*'])

kept = mod.rebuild_window_index_only(data_dir=DATA_DIR, window=WINDOW, horizon=HORIZON)
print("new total windows indexed:", kept)

train_dl, val_dl, test_dl, sizes = mod.load_dataloaders_with_ratio_split(
    data_dir=DATA_DIR,
    train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
    n_entities=None,
    shuffle_train=False,
    coverage_per_window=COVERAGE,
    date_batching=True,
    dates_per_batch=BATCH_SIZE,
    collate_fn=None,
    window=WINDOW,
    horizon=HORIZON,
    norm_scope="train_only"
)
print("sizes:", sizes)

# infer dims from one batch
xb0, yb0, meta0 = next(iter(train_dl))
V0, T0 = xb0  # [B,N,K,Fv], [B,N,K,Ft]
B0, N0, K0, Fv = V0.shape
Ft = T0.shape[-1]
H  = yb0.shape[-1]
print("V:", V0.shape, "T:", T0.shape, "y:", yb0.shape)
assert Fv == Ft, f"Expected Fv == Ft, got {Fv} vs {Ft}"

# ---- VAE (encode y -> μ) ----
vae = LatentVAE(
    input_dim=1, seq_len=H, latent_dim=LATENT_DIM,
    enc_layers=VAE_LAYERS, enc_heads=VAE_HEADS, enc_ff=VAE_FF,
    dec_layers=VAE_LAYERS, dec_heads=VAE_HEADS, dec_ff=VAE_FF,
).to(device)
if VAE_CKPT and os.path.isfile(VAE_CKPT):
    ckpt = torch.load(VAE_CKPT, map_location=device)
    sd = ckpt.get("state_dict", ckpt)
    vae.load_state_dict(sd, strict=False)
    print("Loaded VAE checkpoint:", VAE_CKPT)

vae.eval()
for p in vae.encoder.parameters():  # freeze encoder (optional)
    p.requires_grad = False

mu_mean, mu_std = compute_latent_stats(vae, train_dl, device)

# ---- Conditional diffusion model ----
# IMPORTANT: ctx_dim = Fv (== Ft). LLapDiT/summary will *not* concat V and T;
# it derives both V- and T-like signals internally from a single feature panel.
ctx_dim = Fv
diff_model = LLapDiT(
    data_dim=LATENT_DIM, hidden_dim=MODEL_WIDTH,
    num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
    predict_type=PREDICT_TYPE, laplace_k=LAPLACE_K, global_k=GLOBAL_K,
    timesteps=TIMESTEPS, schedule=SCHEDULE,
    dropout=DROPOUT, attn_dropout=ATTN_DROPOUT,
    self_conditioning=SELF_COND,
    context_dim=ctx_dim, num_entities=N0, context_len=CONTEXT_LEN
).to(device)
scheduler = diff_model.scheduler

optimizer = torch.optim.AdamW(diff_model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
total_steps = EPOCHS * max(1, len(train_dl))
lr_sched = make_warmup_cosine(optimizer, total_steps, warmup_frac=WARMUP_FRAC, base_lr=BASE_LR)
scaler = GradScaler(enabled=(device.type=="cuda"))

best_val = float("inf"); patience = 0

for epoch in range(1, EPOCHS + 1):
    # ---------------- train ----------------
    diff_model.train()
    train_sum = 0.0; train_count = 0

    for xb, yb, meta in train_dl:
        V, T = xb                          # [B,N,K,F]
        mask_bn = meta["entity_mask"]      # [B,N] bool

        # series for summarizer: use a *single* feature panel (no concat)
        # Choose V by convention (T can be used equivalently since Fv == Ft).
        series = V.permute(0,2,1,3).to(device)   # [B,T(=K),N,F]
        mask_bn = mask_bn.to(device)

        # global conditional summary once per batch (mask applied inside)
        with torch.no_grad():
            cond_summary, _ = diff_model.context(series, entity_mask=mask_bn)  # [B,S,Hm]

        # targets: flatten entities, keep only valid ones
        y = yb.to(device)                         # [B,N,H]
        B,N,Hcur = y.shape
        y_flat = y.reshape(B*N, Hcur).unsqueeze(-1)  # [B*N, H, 1]
        m_flat = mask_bn.reshape(B*N)
        if not m_flat.any():
            continue
        y_in = y_flat[m_flat]                     # [B_eff, H, 1]

        # map cond_summary rows to samples
        batch_ids = torch.arange(B, device=device).unsqueeze(1).expand(B,N).reshape(B*N)[m_flat]
        cond_summary_flat = cond_summary[batch_ids]  # [B_eff,S,Hm]

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(device.type=="cuda")):
            _, mu, _ = vae(y_in)                      # [B_eff, H, D]
            mu_norm = two_stage_norm(mu, USE_EWMA, EWMA_LAMBDA, mu_mean, mu_std)
            t = torch.randint(0, scheduler.timesteps, (mu_norm.size(0),), device=device).long()
            # classifier-free drop
            use_cond = (torch.rand(()) >= DROP_COND_P)
            cs = cond_summary_flat if use_cond else None
            loss = diffusion_loss(diff_model, scheduler, mu_norm, t,
                                  cond_summary=cs, predict_type=PREDICT_TYPE,
                                  self_cond=SELF_COND, self_cond_p=SELF_COND_P,
                                  snr_clip=SNR_CLIP)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if GRAD_CLIP and GRAD_CLIP > 0:
            nn.utils.clip_grad_norm_(diff_model.parameters(), GRAD_CLIP)
        scaler.step(optimizer); scaler.update()
        lr_sched.step()

        train_sum += loss.item() * mu_norm.size(0)
        train_count += mu_norm.size(0)

    avg_train = train_sum / max(1, train_count)

    # ---------------- val ----------------
    diff_model.eval()
    val_sum = 0.0; val_count = 0
    with torch.no_grad():
        for xb, yb, meta in val_dl:
            V, T = xb
            mask_bn = meta["entity_mask"].to(device)
            series = V.permute(0,2,1,3).to(device)  # single panel, no concat
            cond_summary, _ = diff_model.context(series, entity_mask=mask_bn)

            y = yb.to(device)
            B,N,Hcur = y.shape
            y_flat = y.reshape(B*N, Hcur).unsqueeze(-1)
            m_flat = mask_bn.reshape(B*N)
            if not m_flat.any():
                continue
            y_in = y_flat[m_flat]
            batch_ids = torch.arange(B, device=device).unsqueeze(1).expand(B,N).reshape(B*N)[m_flat]
            cond_summary_flat = cond_summary[batch_ids]

            _, mu, _ = vae(y_in)
            mu_norm = two_stage_norm(mu, USE_EWMA, EWMA_LAMBDA, mu_mean, mu_std)
            t = torch.randint(0, scheduler.timesteps, (mu_norm.size(0),), device=device).long()
            loss = diffusion_loss(diff_model, scheduler, mu_norm, t,
                                  cond_summary=cond_summary_flat, predict_type=PREDICT_TYPE,
                                  self_cond=False, self_cond_p=0.0, snr_clip=SNR_CLIP)
            val_sum += loss.item() * mu_norm.size(0)
            val_count += mu_norm.size(0)

    avg_val = val_sum / max(1, val_count)
    print(f"Epoch {epoch:03d} | train: {avg_train:.6f} | val: {avg_val:.6f}")

    # checkpoint best
    if avg_val < best_val:
        best_val = avg_val; patience = 0
        ckpt_path = os.path.join(CKPT_DIR, f"best_latdiff_epoch_{epoch:03d}_val_{avg_val:.6f}.pt")
        torch.save({
                "epoch": epoch,
                "model_state": diff_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "mu_mean": mu_mean.detach().cpu(),
                "mu_std":  mu_std.detach().cpu(),
                "predict_type": PREDICT_TYPE,
                "timesteps": TIMESTEPS,
                "schedule": SCHEDULE,
        }, ckpt_path)
        print("Saved:", ckpt_path)
    else:
        patience += 1
        if patience >= EARLY_STOP:
            print("Early stopping.")
            break