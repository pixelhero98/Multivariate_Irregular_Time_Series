import os, torch
import crypto_config
from torch import nn
from typing import Optional, Tuple
from torch.cuda.amp import GradScaler, autocast
from Dataset.fin_dataset import run_experiment
from Latent_Space.latent_vae import LatentVAE
from Model.lladit import LLapDiT
from Model.cond_diffusion_utils import (EMA, set_torch,
                                        make_warmup_cosine,
                                        two_stage_norm, compute_latent_stats,
                                        normalize_cond_per_batch,
                                        flatten_targets, sample_t_uniform
                                        )

# ====================== Latent diffusion helpers ======================
def build_context(model: LLapDiT, V: torch.Tensor, T: torch.Tensor,
                  mask_bn: torch.Tensor, device: torch.device, norm: bool = True) -> torch.Tensor:
    """Returns normalized cond_summary: [B,S,Hm]"""
    with torch.no_grad():
        series_diff = T.permute(0, 2, 1, 3).to(device)  # [B,K,N,F]
        series      = V.permute(0, 2, 1, 3).to(device)  # [B,K,N,F]
        mask_bn     = mask_bn.to(device)
        cond_summary, _ = model.context(x=series, ctx_diff=series_diff, entity_mask=mask_bn)
        if norm:
            cond_summary = normalize_cond_per_batch(cond_summary)
    return cond_summary


def encode_mu_norm(vae: LatentVAE, y_in: torch.Tensor, *, use_ewma: bool, ewma_lambda: float,
                   mu_mean: torch.Tensor, mu_std: torch.Tensor) -> torch.Tensor:
    """VAE encode then two-stage normalize; returns [Beff, H, Z]"""
    with torch.no_grad():
        _, mu, _ = vae(y_in)
        mu_norm = two_stage_norm(mu, use_ewma=use_ewma, ewma_lambda=ewma_lambda, mu_mean=mu_mean, mu_std=mu_std)
        mu_norm = torch.nan_to_num(mu_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return mu_norm


def diffusion_loss(model: LLapDiT, scheduler, x0_lat_norm: torch.Tensor, t: torch.Tensor,
                   *, cond_summary: Optional[torch.Tensor], predict_type: str = "v",
                   weight_scheme: str = "none", minsnr_gamma: float = 5.0,
                   sc_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    MSE on v/eps with channel-invariant reduction.
    Optional min-SNR weighting.
    """
    noise = torch.randn_like(x0_lat_norm)
    x_t, eps_true = scheduler.q_sample(x0_lat_norm, t, noise)
    pred = model(x_t, t, cond_summary=cond_summary, sc_feat=sc_feat)
    target = eps_true if predict_type == "eps" else scheduler.v_from_eps(x_t, t, eps_true)

    # [B,H,Z] -> per-sample loss: mean over H, sum over Z  (scale-invariant to Z)
    err = (pred - target).pow(2)              # [B,H,Z]
    per_sample = err.mean(dim=1).sum(dim=1)   # [B]

    if weight_scheme == 'none':
        return per_sample.mean()
    elif weight_scheme == 'weighted_min_snr':
        abar = scheduler.alpha_bars[t]
        snr  = abar / (1.0 - abar).clamp_min(1e-8)
        gamma = minsnr_gamma
        w = torch.minimum(snr, torch.as_tensor(gamma, device=snr.device, dtype=snr.dtype))
        w = w / (snr + 1.0)
        return (w.detach() * per_sample).mean()


@torch.no_grad()
def calculate_v_variance(scheduler, dataloader, vae, device, latent_stats):
    """
    Calculates the variance of the v-prediction target over a given dataloader.
    """
    all_v_targets = []
    print("Calculating variance of v-prediction target...")

    # Unpack the pre-computed latent statistics
    mu_mean, mu_std = latent_stats

    # Loop through the validation set
    for xb, yb, meta in dataloader:
        # This block is the same as in your validate() function
        # It gets the normalized latent variable 'mu_norm' which is the x0 for diffusion
        y_in, _ = flatten_targets(yb, meta["entity_mask"], device)
        if y_in is None:
            continue

        mu_norm = encode_mu_norm(
            vae, y_in,
            use_ewma=crypto_config.USE_EWMA,
            ewma_lambda=crypto_config.EWMA_LAMBDA,
            mu_mean=mu_mean,
            mu_std=mu_std
        )

        # Now, simulate the process for creating the 'v' target
        # 1. Sample random timesteps
        t = sample_t_uniform(scheduler, mu_norm.size(0), device)

        # 2. Create the noise that would be added
        noise = torch.randn_like(mu_norm)

        # 3. Apply the forward process to get the noised latent x_t
        x_t, _ = scheduler.q_sample(mu_norm, t, noise)

        # 4. Calculate the ground-truth 'v' from x_t and the noise
        v_target = scheduler.v_from_eps(x_t, t, noise)
        all_v_targets.append(v_target.detach().cpu())

    # Concatenate all batches and compute the final variance
    if not all_v_targets:
        print("Warning: No valid data found to calculate variance.")
        return float('nan')

    all_v_targets_cat = torch.cat(all_v_targets, dim=0)
    v_variance = all_v_targets_cat.var().item()

    return v_variance
# ============================ Training setup ============================

device = set_torch()

train_dl, val_dl, test_dl, sizes = run_experiment(
    data_dir=crypto_config.DATA_DIR,
    date_batching=crypto_config.date_batching,
    dates_per_batch=crypto_config.BATCH_SIZE,
    K=crypto_config.WINDOW,
    H=crypto_config.PRED,
    coverage=crypto_config.COVERAGE
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

# ---- Estimate global latent stats (uses same window-normalization) ----
VAE_CKPT = crypto_config.VAE_CKPT
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

mu_mean, mu_std = compute_latent_stats(
    vae, train_dl, device,
    use_ewma=crypto_config.USE_EWMA,
    ewma_lambda=crypto_config.EWMA_LAMBDA
)


# ---- Conditional diffusion model ----
diff_model = LLapDiT(
    data_dim=crypto_config.VAE_LATENT_DIM, hidden_dim=crypto_config.MODEL_WIDTH,
    num_layers=crypto_config.NUM_LAYERS, num_heads=crypto_config.NUM_HEADS,
    predict_type=crypto_config.PREDICT_TYPE, laplace_k=crypto_config.LAPLACE_K, global_k=crypto_config.GLOBAL_K,
    timesteps=crypto_config.TIMESTEPS, schedule=crypto_config.SCHEDULE,
    dropout=crypto_config.DROPOUT, attn_dropout=crypto_config.ATTN_DROPOUT,
    self_conditioning=crypto_config.SELF_COND,
    context_dim=Fv, num_entities=N0, context_len=crypto_config.CONTEXT_LEN,
    lap_mode=crypto_config.LAP_MODE
).to(device)


# ---- Calculate the variance of the v-prediction target ----
# v_variance = calculate_v_variance(
#     scheduler=diff_model.scheduler,
#     dataloader=val_dl, # Use the validation set for a representative sample
#     vae=vae,
#     device=device,
#     latent_stats=(mu_mean, mu_std)
# )
# print(f"📈 Calculated V-Prediction Target Variance: {v_variance:.4f}")
# print("=========================================================")


ema = EMA(diff_model, decay=crypto_config.EMA_DECAY) if crypto_config.USE_EMA_EVAL else None

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

    for xb, yb, meta in train_dl:
        V, T = xb
        mask_bn = meta["entity_mask"]

        cond_summary = build_context(diff_model, V, T, mask_bn, device)
        y_in, batch_ids = flatten_targets(yb, mask_bn, device)
        if y_in is None:
            continue

        cond_summary_flat = cond_summary[batch_ids]  # [Beff,S,Hm]
        mu_norm = encode_mu_norm(
            vae, y_in,
            use_ewma=crypto_config.USE_EWMA,
            ewma_lambda=crypto_config.EWMA_LAMBDA,
            mu_mean=mu_mean, mu_std=mu_std
        )

        # classifier-free guidance dropout (whole-batch mask)
        use_cond = (torch.rand(()) >= crypto_config.DROP_COND_P)
        cs = cond_summary_flat if use_cond else None

        t = sample_t_uniform(scheduler, mu_norm.size(0), device)

        # ==================== SELF-CONDITIONING LOGIC START ====================
        sc_feat = None
        # Only apply self-conditioning after the specified epoch and with a given probability
        if epoch >= crypto_config.SELF_COND_START_EPOCH and torch.rand(()) < crypto_config.SELF_COND_P:
            with torch.no_grad():
                # 1. First pass: predict x0 with no gradients
                pred_no_grad = diff_model(mu_norm, t, cond_summary=cs, sc_feat=None)

                # 2. Convert model output to x0 and detach it from the graph
                sc_feat = scheduler.to_x0(
                    mu_norm, t, pred_no_grad, crypto_config.PREDICT_TYPE
                ).detach()
        # ===================== SELF-CONDITIONING LOGIC END =====================

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(device.type == "cuda")):
            loss = diffusion_loss(
                diff_model, scheduler, mu_norm, t,
                cond_summary=cs, predict_type=crypto_config.PREDICT_TYPE,
                weight_scheme='weighted_min_snr',
                sc_feat=sc_feat
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

        # global_step += 1
        # # light-weight pole health logging
        # if global_step % 500 == 0:
        #     log_pole_health([diff_model], lambda m, step: _print_log(m, step, csv_path=None),
        #                     step=global_step, tag_prefix="train/")

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
        # log_pole_health([diff_model], lambda m, step: _print_log(m, step, csv_path=None),
        #                 step=global_step, tag_prefix="val/")

    for xb, yb, meta in val_dl:
        V, T = xb
        mask_bn = meta["entity_mask"]

        cond_summary = build_context(diff_model, V, T, mask_bn, device)
        y_in, batch_ids = flatten_targets(yb, mask_bn, device)
        if y_in is None:
            continue
        cond_summary_flat = cond_summary[batch_ids]
        mu_norm = encode_mu_norm(
            vae, y_in,
            use_ewma=crypto_config.USE_EWMA,
            ewma_lambda=crypto_config.EWMA_LAMBDA,
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
    Z = crypto_config.VAE_LATENT_DIM
    print(f"Epoch {epoch:03d} | train: {train_loss:.6f} (/Z: {train_loss / Z:.6f}) "
          f"| val: {val_loss:.6f} (/Z: {val_loss / Z:.6f}) | cond_gap: {cond_gap:.6f}")

    # checkpoint best (with EMA state)
    if val_loss < best_val:
        best_val = val_loss; patience = 0
        if current_best_path and os.path.exists(current_best_path):
            os.remove(current_best_path)

        ckpt_path = os.path.join(
            crypto_config.CKPT_DIR,
            f"best_latdiff_epoch_{epoch:03d}_val_{val_loss / Z:.6f}.pt"
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
            save_payload["ema_decay"] = crypto_config.EMA_DECAY
        torch.save(save_payload, ckpt_path)
        print("Saved:", ckpt_path)
        current_best_path = ckpt_path
    else:
        patience += 1
        if patience >= crypto_config.EARLY_STOP:
            print("Early stopping.")
            break
