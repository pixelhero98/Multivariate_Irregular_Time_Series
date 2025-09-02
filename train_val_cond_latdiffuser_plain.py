import os, math
import torch, importlib
import crypto_config
from torch import nn
from typing import Optional
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from Latent_Space.latent_vae import LatentVAE
from Model.lladit import LLapDiT
from Model.cond_diffusion_utils import normalize_and_check, EMA, log_pole_health, _print_log

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
    B, L, D = x.shape
    var = x.new_zeros(B, D)
    mean = x.new_zeros(B, D)
    for t in range(L):
        xt = x[:, t, :]
        mean = lam * mean + (1 - lam) * xt
        var  = lam * var  + (1 - lam) * (xt - mean) ** 2
    return (var + eps).sqrt().unsqueeze(1)

def two_stage_norm(mu: torch.Tensor, use_ewma: bool, ewma_lambda: float,
                   mu_mean: torch.Tensor, mu_std: torch.Tensor) -> torch.Tensor:
    """Window-wise std (EWMA or plain), then global whitening; clamp outliers."""
    s = ewma_std(mu, lam=ewma_lambda) if use_ewma else mu.std(dim=1, keepdim=True).clamp_min(1e-6)
    mu_w = mu / s
    mu_g = (mu_w - mu_mean) / (mu_std + 1e-8)
    return mu_g.clamp(-5, 5)

def diffusion_loss(model: LLapDiT, scheduler, x0_lat_norm: torch.Tensor, t: torch.Tensor,
                   *, cond_summary: Optional[torch.Tensor], predict_type: str = "v") -> torch.Tensor:
    """Plain MSE on v/eps; no extra weighting."""
    noise = torch.randn_like(x0_lat_norm)
    x_t, eps_true = scheduler.q_sample(x0_lat_norm, t, noise)
    pred = model(x_t, t, cond_summary=cond_summary, sc_feat=None)
    target = eps_true if predict_type == "eps" else scheduler.v_from_eps(x_t, t, eps_true)
    return (pred - target).pow(2).mean()

@torch.no_grad()
def compute_latent_stats(vae: LatentVAE, dataloader, device, use_ewma: bool, ewma_lambda: float):
    """Collect μ, apply the SAME window-wise step as training, then compute global per-dim mean/std."""
    all_mu_w = []
    for (_, y, meta) in tqdm(dataloader, desc="Collect μ stats"):
        m = meta["entity_mask"]             # [B,N] bool
        B, N, H = y.shape
        y_flat = y.reshape(B * N, H).unsqueeze(-1)  # [B*N, H, 1]
        m_flat = m.reshape(B * N)
        if not m_flat.any():
            continue
        y_in = y_flat[m_flat].to(device)

        _, mu, _ = vae(y_in)                # mu: [B_eff, H, Z]
        s = ewma_std(mu, lam=ewma_lambda) if use_ewma else mu.std(dim=1, keepdim=True).clamp_min(1e-6)
        mu_w = (mu / s).clamp(-5, 5)        # SAME clipping as training
        all_mu_w.append(mu_w.detach().cpu())

    mu_cat = torch.cat(all_mu_w, dim=0) if all_mu_w else torch.zeros(1, 1, crypto_config.VAE_LATENT_DIM)
    mu_mean = mu_cat.mean(dim=(0, 1)).to(device)                 # [Z]
    mu_std  = mu_cat.std (dim=(0, 1)).clamp_min(1e-6).to(device) # [Z]
    return mu_mean, mu_std

# --------- NEW: conditioning z-score & uniform log-SNR t-sampling ----------
def normalize_cond_per_batch(cs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    z-score over (B,S) for each feature dim; keeps gradients.
    cs: [B, S, Hc] -> same shape
    """
    m = cs.mean(dim=(0, 1), keepdim=True)
    v = cs.var (dim=(0, 1), keepdim=True, unbiased=False)
    return (cs - m) / (v.sqrt() + eps)

@torch.no_grad()
def sample_t_uniform_logsnr(scheduler, batch_size: int) -> torch.Tensor:
    """
    Sample timesteps uniformly in log-SNR by drawing s ~ U[min, max] and
    picking the nearest discrete t index.
    """
    # logSNR = log(alpha_bar / (1 - alpha_bar))
    ab = scheduler.alpha_bars
    logsnr = torch.log(ab / (1.0 - ab + 1e-12) + 1e-12)  # [T]
    s_min, s_max = logsnr[-1], logsnr[0]  # logsnr decreases with t
    u = torch.rand(batch_size, device=ab.device)
    s = s_min + (s_max - s_min) * u                       # [B]
    idx = (logsnr.view(1, -1) - s.view(-1, 1)).abs().argmin(dim=1)
    return idx.long()

# ----------------------------- main -----------------------------
device = set_torch()

# data module
mod = importlib.import_module(crypto_config.DATA_MODULE)
# use PRED consistently
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
    dates_per_batch=crypto_config.BATCH_SIZE,      # B == number of dates per batch
    window=crypto_config.WINDOW,
    horizon=crypto_config.PRED,
    norm_scope=crypto_config.norm_scope
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
VAE_CKPT = getattr(crypto_config, "VAE_CKPT", "")
vae = LatentVAE(
    input_dim=1, seq_len=crypto_config.PRED,  # univariate y per entity
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

mu_mean, mu_std = compute_latent_stats(
    vae, train_dl, device,
    use_ewma=crypto_config.USE_EWMA,
    ewma_lambda=crypto_config.EWMA_LAMBDA
)

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

best_val = float("inf"); patience = 0
current_best_path = None; global_step = 0

for epoch in range(1, crypto_config.EPOCHS + 1):
    # ---------------- train ----------------
    diff_model.train()
    train_sum = 0.0; train_count = 0

    for xb, yb, meta in train_dl:
        V, T = xb                          # [B,N,K,F]
        mask_bn = meta["entity_mask"]      # [B,N] bool

        # build context (no grads)
        with torch.no_grad():
            series_diff = T.permute(0, 2, 1, 3).to(device)  # [B,K,N,F]
            series = V.permute(0, 2, 1, 3).to(device)       # [B,K,N,F]
            mask_bn = mask_bn.to(device)
            cond_summary, _ = diff_model.context(
                x=series, ctx_diff=series_diff, entity_mask=mask_bn
            )  # [B,S,Hm]
            cond_summary = normalize_cond_per_batch(cond_summary)    # <<< NEW

        # targets: flatten entities, keep only valid ones
        y = yb.to(device)                         # [B,N,H]
        B, N, Hcur = y.shape
        y_flat = y.reshape(B * N, Hcur).unsqueeze(-1)  # [B*N, H, 1]
        m_flat = mask_bn.reshape(B * N)
        if not m_flat.any():
            continue
        y_in = y_flat[m_flat]                     # [B_eff, H, 1]

        # map cond_summary rows to samples
        batch_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, N).reshape(B * N)[m_flat]
        cond_summary_flat = cond_summary[batch_ids]  # [B_eff,S,Hm]

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(device.type == "cuda")):
            # encode with frozen VAE (no grads)
            with torch.no_grad():
                _, mu, _ = vae(y_in)                      # [B_eff, H, Z]
                mu_norm = two_stage_norm(mu, crypto_config.USE_EWMA, crypto_config.EWMA_LAMBDA, mu_mean, mu_std)
                mu_norm = torch.nan_to_num(mu_norm, nan=0.0, posinf=0.0, neginf=0.0)

            # --- timestep sampling: uniform in log-SNR (NEW) ---
            t = sample_t_uniform_logsnr(scheduler, mu_norm.size(0))

            # --- classifier-free dropout (whole-batch for efficiency) ---
            use_cond = (torch.rand(()) >= crypto_config.DROP_COND_P)
            cs = cond_summary_flat if use_cond else None

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
        if crypto_config.GRAD_CLIP and crypto_config.GRAD_CLIP > 0:
            nn.utils.clip_grad_norm_(diff_model.parameters(), crypto_config.GRAD_CLIP)
        scaler.step(optimizer); scaler.update()
        if ema is not None:
            ema.update(diff_model)
        lr_sched.step()

        # pole monitoring (train)
        global_step += 1
        if global_step % 5 == 0:
            log_pole_health([diff_model], lambda m, step: _print_log(m, step, csv_path=None),
                            step=global_step, tag_prefix="train/")

        train_sum += loss.item() * mu_norm.size(0)
        train_count += mu_norm.size(0)

    avg_train = train_sum / max(1, train_count)

    # ---------------- val ----------------
    diff_model.eval()
    val_sum = 0.0; val_count = 0
    cond_gap_accum = 0.0; cond_gap_batches = 0

    if ema is not None:
        ema.store(diff_model)
        ema.copy_to(diff_model)
        log_pole_health([diff_model], lambda m, step: _print_log(m, step, csv_path=None),
                        step=global_step, tag_prefix="val/")

    with torch.no_grad():
        for xb, yb, meta in val_dl:
            V, T = xb
            mask_bn = meta["entity_mask"].to(device)

            series_diff = T.permute(0, 2, 1, 3).to(device)
            series      = V.permute(0, 2, 1, 3).to(device)
            cond_summary, _ = diff_model.context(x=series, ctx_diff=series_diff, entity_mask=mask_bn)
            cond_summary = normalize_cond_per_batch(cond_summary)   # <<< NEW

            y = yb.to(device)
            B, N, Hcur = y.shape
            y_flat = y.reshape(B * N, Hcur).unsqueeze(-1)
            m_flat = mask_bn.reshape(B * N)
            if not m_flat.any():
                continue
            y_in = y_flat[m_flat]
            batch_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, N).reshape(B * N)[m_flat]
            cond_summary_flat = cond_summary[batch_ids]

            _, mu, _ = vae(y_in)
            mu_norm = two_stage_norm(mu, crypto_config.USE_EWMA, crypto_config.EWMA_LAMBDA, mu_mean, mu_std)
            mu_norm = torch.nan_to_num(mu_norm, nan=0.0, posinf=0.0, neginf=0.0)

            t = sample_t_uniform_logsnr(scheduler, mu_norm.size(0))
            loss = diffusion_loss(
                diff_model, scheduler, mu_norm, t,
                cond_summary=cond_summary_flat, predict_type=crypto_config.PREDICT_TYPE
            )
            val_sum += loss.item() * mu_norm.size(0)
            val_count += mu_norm.size(0)

            # ---- Conditional gap probe (cheap) ----
            probe_n = min(128, mu_norm.size(0))
            if probe_n > 0:
                mu_p = mu_norm[:probe_n]
                cs_p = cond_summary_flat[:probe_n]
                t_p  = sample_t_uniform_logsnr(scheduler, probe_n)
                loss_cond  = diffusion_loss(diff_model, scheduler, mu_p, t_p,
                                            cond_summary=cs_p, predict_type=crypto_config.PREDICT_TYPE).item()
                loss_unco  = diffusion_loss(diff_model, scheduler, mu_p, t_p,
                                            cond_summary=None, predict_type=crypto_config.PREDICT_TYPE).item()
                cond_gap_accum += (loss_unco - loss_cond)  # >0 means conditioning helps
                cond_gap_batches += 1

    if ema is not None:
        ema.restore(diff_model)

    avg_val = val_sum / max(1, val_count)
    cond_gap = (cond_gap_accum / cond_gap_batches) if cond_gap_batches > 0 else float("nan")
    print(f"Epoch {epoch:03d} | train: {avg_train:.6f} | val: {avg_val:.6f} | cond_gap: {cond_gap:.6f}")

    # ---------------- checkpoint best (with EMA) ----------------
    if avg_val < best_val:
        best_val = avg_val; patience = 0
        if current_best_path and os.path.exists(current_best_path):
            os.remove(current_best_path)

        ckpt_path = os.path.join(
            crypto_config.CKPT_DIR,
            f"best_latdiff_epoch_{epoch:03d}_val_{avg_val:.6f}.pt"
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

# ---------------- conditional generation (val & test) ----------------
# (left to your sampler; training script ends here)
