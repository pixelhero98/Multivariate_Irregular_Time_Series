import os, math
import torch, importlib
import crypto_config
from torch import nn
from typing import Optional, Tuple
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from Latent_Space.latent_vae import LatentVAE
from Model.lladit import LLapDiT

# Try both locations for utils to be robust
try:
    from Model.cond_diffusion_utils import EMA, log_pole_health, _print_log
except Exception:
    from cond_diffusion_utils import EMA, log_pole_health, _print_log

# ============================= utils =============================
def set_torch() -> torch.device:
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
    """x: [B,L,D] -> [B,1,D] exponential window std"""
    B, L, D = x.shape
    var = x.new_zeros(B, D)
    mean = x.new_zeros(B, D)
    for t in range(L):
        xt = x[:, t, :]
        mean = lam * mean + (1 - lam) * xt
        var  = lam * var  + (1 - lam) * (xt - mean) ** 2
    return (var + eps).sqrt().unsqueeze(1)

def two_stage_norm(mu: torch.Tensor, *, use_ewma: bool, ewma_lambda: float,
                   mu_mean: torch.Tensor, mu_std: torch.Tensor) -> torch.Tensor:
    """Window-wise std (EWMA or plain), then global whitening; clamp outliers."""
    s = ewma_std(mu, lam=ewma_lambda) if use_ewma else mu.std(dim=1, keepdim=True).clamp_min(1e-6)
    mu_w = mu / s
    mu_g = (mu_w - mu_mean) / (mu_std + 1e-8)
    return mu_g.clamp(-5, 5)

def normalize_cond_per_batch(cs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """z-score over (B,S) for each feature dim; keeps gradients. cs: [B,S,Hc]"""
    m = cs.mean(dim=(0, 1), keepdim=True)
    v = cs.var (dim=(0, 1), keepdim=True, unbiased=False)
    return (cs - m) / (v.sqrt() + eps)

# ====================== common batch helpers ======================
def build_context(model: LLapDiT, V: torch.Tensor, T: torch.Tensor, mask_bn: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Returns normalized cond_summary: [B,S,Hm]"""
    with torch.no_grad():
        series_diff = T.permute(0, 2, 1, 3).to(device)  # [B,K,N,F]
        series      = V.permute(0, 2, 1, 3).to(device)  # [B,K,N,F]
        mask_bn     = mask_bn.to(device)
        cond_summary, _ = model.context(x=series, ctx_diff=series_diff, entity_mask=mask_bn)
        cond_summary = normalize_cond_per_batch(cond_summary)
    return cond_summary

def flatten_targets(yb: torch.Tensor, mask_bn: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """yb: [B,N,H] -> y_in: [Beff,H,1], batch_ids: [Beff] mapping to B for cond rows"""
    y = yb.to(device)
    B, N, Hcur = y.shape
    y_flat = y.reshape(B * N, Hcur).unsqueeze(-1)  # [B*N, H, 1]
    m_flat = mask_bn.to(device).reshape(B * N)
    if not m_flat.any():
        return None, None
    y_in = y_flat[m_flat]
    batch_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, N).reshape(B * N)[m_flat]
    return y_in, batch_ids

def encode_mu_norm(vae: LatentVAE, y_in: torch.Tensor, *, use_ewma: bool, ewma_lambda: float,
                   mu_mean: torch.Tensor, mu_std: torch.Tensor) -> torch.Tensor:
    """VAE encode then two-stage normalize; returns [Beff, H, Z]"""
    with torch.no_grad():
        _, mu, _ = vae(y_in)
        mu_norm = two_stage_norm(mu, use_ewma=use_ewma, ewma_lambda=ewma_lambda, mu_mean=mu_mean, mu_std=mu_std)
        mu_norm = torch.nan_to_num(mu_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return mu_norm

def sample_t_uniform(scheduler, n: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, scheduler.timesteps, (n,), device=device)

def diffusion_loss(model: LLapDiT, scheduler, x0_lat_norm: torch.Tensor, t: torch.Tensor,
                   *, cond_summary: Optional[torch.Tensor], predict_type: str = "v") -> torch.Tensor:
    """Plain MSE on v/eps; no weighting; cond_summary can be None."""
    noise = torch.randn_like(x0_lat_norm)
    x_t, eps_true = scheduler.q_sample(x0_lat_norm, t, noise)
    pred = model(x_t, t, cond_summary=cond_summary, sc_feat=None)
    target = eps_true if predict_type == "eps" else scheduler.v_from_eps(x_t, t, eps_true)
    return (pred - target).pow(2).mean()

# ============================ setup ============================
device = set_torch()

# data module (consistent use of PRED as horizon)
mod = importlib.import_module(crypto_config.DATA_MODULE)
kept = mod.rebuild_window_index_only(
    data_dir=crypto_config.DATA_DIR,
    window=crypto_config.WINDOW,
    horizon=crypto_config.PRED
)
print("new total windows indexed:", kept)

train_dl, val_dl, test_dl, sizes = mod.load_dataloaders_with_ratio_split(
    data_dir=crypto_config.DATA_DIR,
    train_ratio=crypto_config.train_ratio, val_ratio=crypto_config.val_ratio, test_ratio=crypto_config.test_ratio,
    n_entities=crypto_config.NUM_ENTITIES,
    shuffle_train=crypto_config.shuffle_train,
    coverage_per_window=crypto_config.COVERAGE,
    date_batching=crypto_config.date_batching,
    dates_per_batch=crypto_config.BATCH_SIZE,
    window=crypto_config.WINDOW,
    horizon=crypto_config.PRED,
    norm_scope=crypto_config.norm_scope
)
print("sizes:", sizes)

# infer dims from one batch
xb0, yb0, meta0 = next(iter(train_dl))
V0, T0 = xb0
B0, N0, K0, Fv = V0.shape
Ft = T0.shape[-1]
H  = yb0.shape[-1]
assert Fv == Ft, f"Expected Fv == Ft, got {Fv} vs {Ft}"
print("V:", V0.shape, "T:", T0.shape, "y:", yb0.shape)

# ---- VAE ----
VAE_CKPT = getattr(crypto_config, "VAE_CKPT", "")
vae = LatentVAE(
    input_dim=1, seq_len=crypto_config.PRED,
    latent_dim=crypto_config.VAE_LATENT_DIM,
    enc_layers=crypto_config.VAE_LAYERS, enc_heads=crypto_config.VAE_HEADS, enc_ff=crypto_config.VAE_FF,
    dec_layers=crypto_config.VAE_LAYERS, dec_heads=crypto_config.VAE_HEADS, dec_ff=crypto_config.VAE_FF
).to(device)
if VAE_CKPT and os.path.isfile(VAE_CKPT):
    ckpt = torch.load(VAE_CKPT, map_location=device)
    sd = ckpt.get("state_dict", ckpt)
    vae.load_state_dict(sd, strict=False)
    print("Loaded VAE checkpoint:", VAE_CKPT)

vae.eval()
for p in vae.parameters():
    p.requires_grad = False

# ---- Estimate global latent stats (uses same window-normalization) ----
@torch.no_grad()
def compute_latent_stats(dataloader):
    all_mu_w = []
    for (_, y, meta) in tqdm(dataloader, desc="Collect μ stats"):
        m = meta["entity_mask"]             # [B,N] bool
        B, N, Ht = y.shape
        y_flat = y.reshape(B * N, Ht).unsqueeze(-1)  # [B*N, H, 1]
        m_flat = m.reshape(B * N)
        if not m_flat.any():
            continue
        y_in = y_flat[m_flat].to(device)
        _, mu, _ = vae(y_in)
        s = ewma_std(mu, lam=getattr(crypto_config, "EWMA_LAMBDA", 0.94)) if getattr(crypto_config, "USE_EWMA", True) \
            else mu.std(dim=1, keepdim=True).clamp_min(1e-6)
        mu_w = (mu / s).clamp(-5, 5)
        all_mu_w.append(mu_w.detach().cpu())
    mu_cat = torch.cat(all_mu_w, dim=0) if all_mu_w else torch.zeros(1, 1, crypto_config.VAE_LATENT_DIM)
    mu_mean = mu_cat.mean(dim=(0, 1)).to(device)                 # [Z]
    mu_std  = mu_cat.std (dim=(0, 1)).clamp_min(1e-6).to(device) # [Z]
    return mu_mean, mu_std

mu_mean, mu_std = compute_latent_stats(train_dl)

# ---- Conditional diffusion model ----
ctx_dim = Fv
diff_model = LLapDiT(
    data_dim=crypto_config.VAE_LATENT_DIM, hidden_dim=crypto_config.MODEL_WIDTH,
    num_layers=crypto_config.NUM_LAYERS, num_heads=crypto_config.NUM_HEADS,
    predict_type=crypto_config.PREDICT_TYPE, laplace_k=crypto_config.LAPLACE_K, global_k=crypto_config.GLOBAL_K,
    timesteps=crypto_config.TIMESTEPS, schedule=crypto_config.SCHEDULE,
    dropout=crypto_config.DROPOUT, attn_dropout=crypto_config.ATTN_DROPOUT,
    self_conditioning=crypto_config.SELF_COND,
    context_dim=ctx_dim, num_entities=N0, context_len=crypto_config.CONTEXT_LEN
).to(device)

USE_EMA_EVAL = getattr(crypto_config, "USE_EMA_EVAL", True)
EMA_DECAY    = getattr(crypto_config, "EMA_DECAY", 0.999)
ema = EMA(diff_model, decay=EMA_DECAY) if USE_EMA_EVAL else None

scheduler = diff_model.scheduler
optimizer = torch.optim.AdamW(diff_model.parameters(),
                              lr=crypto_config.BASE_LR,
                              weight_decay=crypto_config.WEIGHT_DECAY)
total_steps = crypto_config.EPOCHS * max(1, len(train_dl))
lr_sched = make_warmup_cosine(optimizer, total_steps,
                              warmup_frac=crypto_config.WARMUP_FRAC,
                              base_lr=crypto_config.BASE_LR)
scaler = GradScaler(enabled=(device.type == "cuda"))

# ============================ train/val loops ============================
def train_one_epoch(epoch: int):
    diff_model.train()
    running_loss = 0.0; num_samples = 0
    global global_step

    for xb, yb, meta in tqdm(train_dl, desc=f"Train {epoch:03d}"):
        V, T = xb
        mask_bn = meta["entity_mask"]

        cond_summary = build_context(diff_model, V, T, mask_bn, device)
        y_in, batch_ids = flatten_targets(yb, mask_bn, device)
        if y_in is None:
            continue

        cond_summary_flat = cond_summary[batch_ids]  # [Beff,S,Hm]
        mu_norm = encode_mu_norm(
            vae, y_in,
            use_ewma=getattr(crypto_config, "USE_EWMA", True),
            ewma_lambda=getattr(crypto_config, "EWMA_LAMBDA", 0.94),
            mu_mean=mu_mean, mu_std=mu_std
        )

        # classifier-free guidance dropout (whole-batch mask)
        use_cond = (torch.rand(()) >= crypto_config.DROP_COND_P)
        cs = cond_summary_flat if use_cond else None

        t = sample_t_uniform(scheduler, mu_norm.size(0), device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(device.type == "cuda")):
            loss = diffusion_loss(
                diff_model, scheduler, mu_norm, t,
                cond_summary=cs, predict_type=crypto_config.PREDICT_TYPE
            )

        if not torch.isfinite(loss):
            print("[warn] non-finite loss detected; skipping step")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if getattr(crypto_config, "GRAD_CLIP", 0.0) and crypto_config.GRAD_CLIP > 0:
            nn.utils.clip_grad_norm_(diff_model.parameters(), crypto_config.GRAD_CLIP)
        scaler.step(optimizer); scaler.update()
        if ema is not None:
            ema.update(diff_model)
        lr_sched.step()

        global_step += 1
        # light-weight pole health logging
        if global_step % 5 == 0:
            log_pole_health([diff_model], lambda m, step: _print_log(m, step, csv_path=None),
                            step=global_step, tag_prefix="train/")

        running_loss += loss.item() * mu_norm.size(0)
        num_samples += mu_norm.size(0)

    return running_loss / max(1, num_samples)

@torch.no_grad()
def validate():
    diff_model.eval()
    total, count = 0.0, 0
    cond_gap_accum, cond_gap_batches = 0.0, 0

    if ema is not None:
        ema.store(diff_model)
        ema.copy_to(diff_model)
        log_pole_health([diff_model], lambda m, step: _print_log(m, step, csv_path=None),
                        step=global_step, tag_prefix="val/")

    for xb, yb, meta in tqdm(val_dl, desc="Val"):
        V, T = xb
        mask_bn = meta["entity_mask"]

        cond_summary = build_context(diff_model, V, T, mask_bn, device)
        y_in, batch_ids = flatten_targets(yb, mask_bn, device)
        if y_in is None:
            continue
        cond_summary_flat = cond_summary[batch_ids]
        mu_norm = encode_mu_norm(
            vae, y_in,
            use_ewma=getattr(crypto_config, "USE_EWMA", True),
            ewma_lambda=getattr(crypto_config, "EWMA_LAMBDA", 0.94),
            mu_mean=mu_mean, mu_std=mu_std
        )

        t = sample_t_uniform(scheduler, mu_norm.size(0), device)
        loss = diffusion_loss(
            diff_model, scheduler, mu_norm, t,
            cond_summary=cond_summary_flat, predict_type=crypto_config.PREDICT_TYPE
        )
        total += loss.item() * mu_norm.size(0)
        count += mu_norm.size(0)

        # Conditional gap probe
        probe_n = min(128, mu_norm.size(0))
        if probe_n > 0:
            mu_p = mu_norm[:probe_n]
            cs_p = cond_summary_flat[:probe_n]
            t_p  = sample_t_uniform(scheduler, probe_n, device)
            loss_cond = diffusion_loss(diff_model, scheduler, mu_p, t_p,
                                       cond_summary=cs_p, predict_type=crypto_config.PREDICT_TYPE).item()
            loss_unco = diffusion_loss(diff_model, scheduler, mu_p, t_p,
                                       cond_summary=None, predict_type=crypto_config.PREDICT_TYPE).item()
            cond_gap_accum += (loss_unco - loss_cond)
            cond_gap_batches += 1

    if ema is not None:
        ema.restore(diff_model)

    avg_val = total / max(1, count)
    cond_gap = (cond_gap_accum / cond_gap_batches) if cond_gap_batches > 0 else float("nan")
    return avg_val, cond_gap

# ============================ run ============================
best_val = float("inf")
patience = 0
current_best_path = None
global_step = 0

for epoch in range(1, crypto_config.EPOCHS + 1):
    train_loss = train_one_epoch(epoch)
    val_loss, cond_gap = validate()
    print(f"Epoch {epoch:03d} | train: {train_loss:.6f} | val: {val_loss:.6f} | cond_gap: {cond_gap:.6f}")

    # checkpoint best (with EMA state)
    if val_loss < best_val:
        best_val = val_loss; patience = 0
        if current_best_path and os.path.exists(current_best_path):
            os.remove(current_best_path)

        ckpt_path = os.path.join(
            crypto_config.CKPT_DIR,
            f"best_latdiff_epoch_{epoch:03d}_val_{val_loss:.6f}.pt"
        )
        save_payload = {
            "epoch": epoch,
            "model_state": diff_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "mu_mean": mu_mean.detach().cpu(),
            "mu_std":  mu_std.detach().cpu(),
            "predict_type": crypto_config.PREDICT_TYPE,
            "timesteps": crypto_config.TIMESTEPS,
            "schedule": crypto_config.SCHEDULE,
        }
        if ema is not None:
            save_payload["ema_state"] = ema.state_dict()
            save_payload["ema_decay"] = EMA_DECAY
        torch.save(save_payload, ckpt_path)
        print("Saved:", ckpt_path)
        current_best_path = ckpt_path
    else:
        patience += 1
        if patience >= crypto_config.EARLY_STOP:
            print("Early stopping.")
            break

# End of script
