import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from latent_vae_utils import normalize_and_check
from latent_vae import LatentVAE
from lladit import LLapDiT  # updated model (multi-res + self-cond)

# =====================================================================================
# Config
# =====================================================================================
mod = importlib.import_module('fin_data_prep_ratiosp_indexcache')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

# Data
DATA_ROOT = "./ldt/crypto_data"
TICKER_LIST = "./CRYPTO_top.txt"

WINDOW   = 150    # context len
HORIZON  = 40     # prediction len

BATCH_SIZE  = 64
EPOCHS      = 500
BASE_LR     = 5e-4
WARMUP_FRAC = 0.05      # cosine warmup %
WEIGHT_DECAY = 0.01
GRAD_CLIP    = 1.0
EARLY_STOP_PATIENCE = 20

# VAE
vae_latent_dim = 64
vae_layers=3
vae_heads=4
vae_ff=256

# Diffusion
TOTAL_T        = 1500
SCHEDULE       = "cosine"
PREDICT_TYPE   = "v"     # "eps" or "v"  (pick one; both train & sample in same param)
DROP_COND_P    = 0.25
SELF_COND_P    = 0.50    # chance to use self-conditioning during training
SNR_CLIP       = 5.0     # cap weight for SNR-weighted loss

# Sampling (eval)
GUIDANCE_MINMAX_EVAL = (1.0, 3.0)  # schedule from 1.0 -> 3.0
GUIDANCE_POWER       = 0.3
DDIM_STEPS_EVAL      = 100
DDIM_ETA_EVAL        = 0.0
N_SAMPLES_EVAL       = 16  # MC samples (antithetic inside, so actual draws are ceil/paired)

# Model
MODEL_WIDTH  = 256
NUM_LAYERS   = 4
NUM_HEADS    = 4
# Multi-resolution Laplace: stem k0, then per-layer k
LAPLACE_K    = [24, 20, 16, 16]  
DROPOUT      = 0.0
ATTN_DROPOUT = 0.0
SELF_COND    = True  # enable self-conditioning path in the model

# Two-stage normalization
USE_EWMA_WINDOW_NORM = True
EWMA_LAMBDA          = 0.94

# Decoder fine-tune
FT_DEC_EPOCHS   = 3
FT_DEC_LR       = 1e-4
FT_DEC_WEIGHT_DECAY = 0.0

# IO
CHECKPOINT_DIR = './ldt/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =====================================================================================
# Data prep
# =====================================================================================
# ============== Re-index The Window & Future Horizon For Datasets ==============
kept = mod.rebuild_window_index_only(
    data_dir=DATA_DIR,
    window=WINDOW,
    horizon=PRED,
)
print("new total windows indexed:", kept)
# -------------------- Loaders --------------------
# Let the module’s default collate return panels + mask
train_dl, val_dl, test_dl, sizes = mod.load_dataloaders_with_ratio_split(
    data_dir=DATA_DIR,
    train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
    n_entities=N,
    shuffle_train=False,
    coverage_per_window=COVERAGE,
    date_batching=True,
    dates_per_batch=BATCH_SIZE,      # B == number of dates per batch
    collate_fn=None,                 # <— use the module’s collate
    window=WINDOW,
    horizon=PRED,
    norm_scope="train_only"
)
train_size, val_size, test_size = sizes
xb, yb, meta = next(iter(train_dl))
V, T = xb
M = meta["entity_mask"]
print("V:", V.shape, "T:", T.shape, "y:", yb.shape)         # -> [B,N,K,F], [B,N,K,F], [B,N,H]
print("min coverage:", float(M.float().mean(1).min().item()))
print("frac padded:", float((~M).float().mean().item()))

# =====================================================================================
# VAE (encoder for μ; decoder for final regression)
# =====================================================================================

vae = LatentVAE(
    input_dim=1,
    seq_len=HORIZON,
    latent_dim=vae_latent_dim,
    enc_layers=vae_layers, enc_heads=vae_heads, enc_ff=vae_ff,
    dec_layers=vae_layers, dec_heads=vae_heads, dec_ff=vae_ff,
).to(device)

vae_ckpt_path = './ldt/saved_model/recon_0.0559_epoch_4.pt'
vae.load_state_dict(torch.load(vae_ckpt_path, map_location=device))
vae.eval()
for p in vae.encoder.parameters():
    p.requires_grad = False

# =====================================================================================
# Latent normalization stats (global μ)
# =====================================================================================

@torch.no_grad()
def compute_latent_stats(dataloader):
    all_mu = []
    for _, y in tqdm(dataloader, desc="Collect μ stats"):
        y = y.to(device)
        _, mu, _ = vae(y)
        all_mu.append(mu.cpu())
    all_mu = torch.cat(all_mu, dim=0)
    _, mu_mean, mu_std = normalize_and_check(all_mu)
    return mu_mean.to(device), mu_std.to(device)

mu_mean, mu_std = compute_latent_stats(train_dl)
latent_dim = getattr(vae, "latent_dim", None)
if latent_dim is None:
    with torch.no_grad():
        dummy = torch.zeros(1, HORIZON, 1, device=device)
        _, mu_sample, _ = vae(dummy)
    latent_dim = mu_sample.shape[-1] # infer D

# =====================================================================================
# Helpers: normalization / antithetic / EMA / sched
# =====================================================================================

def ewma_std(x, lam=0.94, eps=1e-8):
    """
    x: [B,L,D] -> per-sample, per-feature EWMA std over L
    Returns std: [B,1,D] broadcastable across time.
    """
    B, L, D = x.shape
    var = torch.zeros(B, D, device=x.device)
    mean = torch.zeros(B, D, device=x.device)
    for t in range(L):
        xt = x[:, t, :]
        mean = lam * mean + (1 - lam) * xt
        var = lam * var + (1 - lam) * (xt - mean) ** 2
    std = torch.sqrt(var + eps).unsqueeze(1)  # [B,1,D]
    return std

def two_stage_norm(mu, mu_mean, mu_std, use_ewma=True, lam=0.94):
    """
    Per-window normalization (EWMA std or plain std) then global.
    """
    # window-wise
    if use_ewma:
        s = ewma_std(mu, lam=lam)              # [B,1,D]
    else:
        s = mu.std(dim=1, keepdim=True).clamp_min(1e-6)
    mu_w = mu / s
    # global
    mu_g = (mu_w - mu_mean) / (mu_std + 1e-8)
    return mu_g.clamp(min=-5.0, max=5.0), s  # also return window scale if needed

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
    @torch.no_grad()
    def load_into(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)

def make_warmup_cosine(optimizer, total_steps, warmup_frac=0.05, min_lr=1e-6):
    warmup_steps = max(1, int(total_steps * warmup_frac))
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr / BASE_LR, 0.5 * (1 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

def build_antithetic_noises(batch_size, L, D, n_samples, device):
    """
    Returns a list of x_T tensors with antithetic pairing.
    """
    tensors = []
    half = (n_samples + 1) // 2
    for i in range(half):
        z = torch.randn(batch_size, L, D, device=device)
        tensors.append(z)
        if len(tensors) < n_samples:
            tensors.append(-z)
    return tensors[:n_samples]

# =====================================================================================
# Diffusion model, optimizer, scheduler, EMA
# =====================================================================================

diff_model = LLapDiT(
    data_dim=latent_dim,
    hidden_dim=MODEL_WIDTH,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    predict_type=PREDICT_TYPE,
    laplace_k=LAPLACE_K,              # multi-resolution
    global_k=GLOBAL_K,
    timesteps=TOTAL_T,
    schedule=SCHEDULE,
    dropout=DROPOUT,
    attn_dropout=ATTN_DROPOUT,
    self_conditioning=SELF_COND,
    context_dim=ctx_dim,
    num_entities=len(TICKS),
    tgt_len=HORIZON
).to(device)

noise_scheduler = diff_model.scheduler

optimizer = torch.optim.AdamW(diff_model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
total_steps = EPOCHS * max(1, len(train_dl))
lr_sched = make_warmup_cosine(optimizer, total_steps, warmup_frac=WARMUP_FRAC)

scaler = GradScaler(enabled=(device.type == "cuda"))
ema = EMA(diff_model, decay=0.999)

# =====================================================================================
# Loss (SNR-weighted) and self-conditioning hook
# =====================================================================================

def diffusion_loss(model, scheduler, x0_lat_norm, t, cond_series, predict_type, use_self_cond=False):
    """
    Parameterization-consistent MSE with SNR weighting.
    """
    noise = torch.randn_like(x0_lat_norm)
    x_t, eps_true = scheduler.q_sample(x0_lat_norm, t, noise)

    sc_feat = None
    if use_self_cond and (torch.rand(()) < SELF_COND_P):
        with torch.no_grad():
            pred_sc = model(x_t, t, series=cond_series)  # native param
            x0_sc = scheduler.to_x0(x_t, t, pred_sc, param_type=predict_type)
        sc_feat = x0_sc

    pred = model(x_t, t, series=cond_series, sc_feat=sc_feat)  # native param
    if predict_type == "eps":
        target = eps_true
    elif predict_type == "v":
        target = scheduler.v_from_eps(x_t, t, eps_true)
    else:
        raise ValueError("predict_type must be 'eps' or 'v'")

    # SNR weighting
    ab = scheduler._gather(scheduler.alpha_bars, t)                 # [B]
    snr = (ab / (1 - ab + 1e-8)).clamp_max(SNR_CLIP).view(-1, 1, 1) # [B,1,1]
    loss = (snr * (pred - target) ** 2).mean()
    return loss

# =====================================================================================
# Training loop
# =====================================================================================

best_val_loss = float('inf')
best_ckpt_path = None
patience = 0
global_step = 0

for epoch in range(1, EPOCHS + 1):
    diff_model.train()
    train_loss_sum = 0.0

    for x_ctx, y_tgt in train_dl:
        x_ctx = x_ctx.to(device)  # [B,Lc,F] or [B,M,Lc,F]
        y_tgt = y_tgt.to(device)  # [B,L,1]

        # Encode to latent μ
        with torch.no_grad():
            _, mu, _ = vae(y_tgt)  # [B,L,D]
        # Two-stage normalization
        mu_norm, _win_scale = two_stage_norm(mu, mu_mean, mu_std,
                                             use_ewma=USE_EWMA_WINDOW_NORM, lam=EWMA_LAMBDA)

        # Timesteps + CFG drop
        t = torch.randint(0, noise_scheduler.timesteps, (mu_norm.size(0),), device=device).long()
        cond_series = None if (torch.rand(()) < DROP_COND_P) else x_ctx

        with autocast(enabled=(device.type == "cuda")):
            loss = diffusion_loss(diff_model, noise_scheduler, mu_norm, t, cond_series,
                                  predict_type=PREDICT_TYPE, use_self_cond=SELF_COND)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(diff_model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        lr_sched.step()

        # EMA after optimizer step
        ema.update(diff_model)

        train_loss_sum += loss.item() * mu_norm.size(0)
        global_step += 1

    avg_train_loss = train_loss_sum / max(1, N_TRAIN)

    # ----------------- Validation (EMA weights) -----------------
    diff_model.eval()
    saved_state = diff_model.state_dict()
    ema.load_into(diff_model)

    val_loss_sum = 0.0
    with torch.no_grad():
        for x_ctx, y_tgt in val_dl:
            x_ctx = x_ctx.to(device)
            y_tgt = y_tgt.to(device)
            _, mu, _ = vae(y_tgt)
            mu_norm, _ = two_stage_norm(mu, mu_mean, mu_std,
                                        use_ewma=USE_EWMA_WINDOW_NORM, lam=EWMA_LAMBDA)
            t = torch.randint(0, noise_scheduler.timesteps, (mu_norm.size(0),), device=device).long()
            cond_series = x_ctx
            loss = diffusion_loss(diff_model, noise_scheduler, mu_norm, t, cond_series,
                                  predict_type=PREDICT_TYPE, use_self_cond=SELF_COND)
            val_loss_sum += loss.item() * mu_norm.size(0)

    avg_val_loss = val_loss_sum / max(1, N_VAL)
    print(f"Epoch {epoch:03d} | train: {avg_train_loss:.6f} | val (EMA): {avg_val_loss:.6f}")

    # Restore live weights after EMA eval
    diff_model.load_state_dict(saved_state)

    # ----------------- Checkpoint / Early stop -----------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience = 0

        if best_ckpt_path and os.path.exists(best_ckpt_path):
            try: os.remove(best_ckpt_path)
            except OSError: pass

        best_ckpt_path = os.path.join(
            CHECKPOINT_DIR, f"best_latdiff_epoch_{epoch:03d}_val_{avg_val_loss:.6f}.pt"
        )
        # Save EMA weights for inference
        ema_ckpt = { "epoch": epoch,
                     "model_state": ema.shadow,
                     "optimizer_state": optimizer.state_dict(),
                     "mu_mean": mu_mean.cpu(),
                     "mu_std": mu_std.cpu(),
                     "predict_type": PREDICT_TYPE,
                     "timesteps": TOTAL_T,
                     "schedule": SCHEDULE }
        torch.save(ema_ckpt, best_ckpt_path)
        print(f"  -> saved EMA model: {best_ckpt_path}")
    else:
        patience += 1

    if patience >= EARLY_STOP_PATIENCE:
        print("\nEarly stopping. Loading best EMA model and running regression eval + decoder fine-tune...")
        break

# =====================================================================================
# Regression evaluation (EMA, antithetic MC, guidance scheduling)
# =====================================================================================

if best_ckpt_path and os.path.exists(best_ckpt_path):
    ckpt = torch.load(best_ckpt_path, map_location=device)
    diff_model.load_state_dict(ckpt["model_state"], strict=True)
    diff_model.eval()
    mu_mean_eval = ckpt["mu_mean"].to(device)
    mu_std_eval  = ckpt["mu_std"].to(device)

    @torch.no_grad()
    def eval_loader_regression(dataloader, n_samples=N_SAMPLES_EVAL):
        reg_sum = 0.0
        for x_ctx, y_true in tqdm(dataloader, desc="Regression eval (EMA)"):
            x_ctx = x_ctx.to(device)
            y_true = y_true.to(device)
            B = y_true.size(0)

            xT_list = build_antithetic_noises(B, HORIZON, latent_dim, n_samples, device)
            y_samples = []
            for x_T in xT_list:
                z_pred_norm = diff_model.generate(
                    shape=(B, HORIZON, latent_dim),
                    steps=DDIM_STEPS_EVAL,
                    guidance_strength=GUIDANCE_MINMAX_EVAL,   # scheduled guidance
                    guidance_power=GUIDANCE_POWER,
                    eta=DDIM_ETA_EVAL,
                    series=x_ctx,
                    x_T=x_T,
                    self_cond=SELF_COND
                )
                z_pred = z_pred_norm * mu_std_eval + mu_mean_eval
                y_hat = vae.decoder(z_pred)   # [B,L,1]
                y_samples.append(y_hat)

            y_samples = torch.stack(y_samples, dim=0)  # [S,B,L,1]
            y_pred = y_samples.mean(dim=0)             # MC mean
            reg_sum += F.mse_loss(y_pred, y_true, reduction="sum").item()
        return reg_sum / max(1, len(dataloader.dataset))

    val_reg_mse = eval_loader_regression(val_dl, n_samples=N_SAMPLES_EVAL)
    print(f"\nValidation regression MSE (EMA, {N_SAMPLES_EVAL} MC, antithetic): {val_reg_mse:.6f}")

    # =================================================================================
    # (Optional) Fine-tune decoder only on diffusion outputs (improves regression)
    # =================================================================================
    if FT_DEC_EPOCHS > 0:
        print(f"\nFine-tuning decoder for {FT_DEC_EPOCHS} epochs ...")
        for p in vae.encoder.parameters():  # keep encoder frozen
            p.requires_grad = False
        for p in vae.decoder.parameters():
            p.requires_grad = True

        dec_optim = torch.optim.AdamW(vae.decoder.parameters(), lr=FT_DEC_LR, weight_decay=FT_DEC_WEIGHT_DECAY)

        for e in range(1, FT_DEC_EPOCHS + 1):
            vae.decoder.train()
            ft_loss_sum = 0.0
            for x_ctx, y_true in tqdm(train_dl, desc=f"Decoder FT epoch {e}"):
                x_ctx = x_ctx.to(device)
                y_true = y_true.to(device)
                B = y_true.size(0)

                # small MC (antithetic pair) for training decoder
                xT_pair = build_antithetic_noises(B, HORIZON, latent_dim, 2, device)
                y_pair = []
                with torch.no_grad():
                    for x_T in xT_pair:
                        z_pred_norm = diff_model.generate(
                            shape=(B, HORIZON, latent_dim),
                            steps=DDIM_STEPS_EVAL//2,           # faster
                            guidance_strength=GUIDANCE_MINMAX_EVAL,
                            guidance_power=GUIDANCE_POWER,
                            eta=DDIM_ETA_EVAL,
                            series=x_ctx,
                            x_T=x_T,
                            self_cond=SELF_COND
                        )
                        z_pred = z_pred_norm * mu_std_eval + mu_mean_eval
                        y_pair.append(z_pred)
                z_pred_mean = torch.stack(y_pair, dim=0).mean(0).detach()  # [B,L,D]

                y_hat = vae.decoder(z_pred_mean)
                loss = F.mse_loss(y_hat, y_true)

                dec_optim.zero_grad(set_to_none=True)
                loss.backward()
                dec_optim.step()

                ft_loss_sum += loss.item() * B

            avg_ft = ft_loss_sum / max(1, N_TRAIN)
            print(f"  Decoder FT epoch {e}: train MSE {avg_ft:.6f}")

        # final validation after decoder FT
        vae.decoder.eval()
        val_reg_mse_ft = eval_loader_regression(val_dl, n_samples=N_SAMPLES_EVAL)
        print(f"\nValidation regression MSE after decoder FT: {val_reg_mse_ft:.6f}")
else:
    print("No best checkpoint found; skipping regression eval and decoder fine-tune.")







