import os, torch, math
import crypto_config
from pathlib import Path
from torch import nn
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from Dataset.data_gen import run_experiment
from Latent_Space.latent_vae import LatentVAE
from Model.summarizer import LaplaceAE
from Model.llapdit import LLapDiT
from Model.llapdit_utils import (EMA, set_torch, encode_mu_norm, _flatten_for_mask,
                                 make_warmup_cosine, calculate_v_variance,
                                 compute_latent_stats, diffusion_loss,
                                 build_context, flatten_targets,
                                 sample_t_uniform, decode_latents_with_vae
                                 )

# ============================ Training setup ============================
# Baseline variance for validation comparisons (populated after initial computation)
BASELINE_V_VARIANCE = None
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
H = yb0.shape[-1]
assert Fv == Ft, f"Expected Fv == Ft, got {Fv} vs {Ft}"
print("V:", V0.shape, "T:", T0.shape, "y:", yb0.shape)

# ---- Load pretrained Laplace summarizer (shared with standalone training) ----
sum_lap_k = crypto_config.SUM_LAPLACE_K
sum_ctx_len = crypto_config.SUM_CONTEXT_LEN
sum_ctx_dim = crypto_config.SUM_CONTEXT_DIM
sum_tv_hidden = crypto_config.SUM_TV_HIDDEN
sum_dropout = crypto_config.SUM_DROPOUT

summarizer_ckpt = crypto_config.SUM_CKPT
if summarizer_ckpt is None:
    summarizer_dir = crypto_config.SUM_DIR
    summarizer_ckpt = Path(summarizer_dir) / f"{crypto_config.PRED}-{crypto_config.VAE_LATENT_CHANNELS}-summarizer.pt"
else:
    summarizer_ckpt = Path(summarizer_ckpt)

laplace_summarizer = LaplaceAE(
    num_entities=N0,
    feat_dim=Fv,
    window_size=crypto_config.WINDOW,
    lap_k=sum_lap_k,
    tv_hidden=sum_tv_hidden,
    out_len=sum_ctx_len,
    context_dim=sum_ctx_dim,
    dropout=sum_dropout,
).to(device)

if summarizer_ckpt.exists():
    state = torch.load(summarizer_ckpt, map_location="cpu")
    laplace_summarizer.load_state_dict(state.get("model", state))
    print(f"Loaded summarizer checkpoint: {summarizer_ckpt}")
else:
    print(f"[warn] Summarizer checkpoint not found at {summarizer_ckpt}; using randomly initialised weights.")

laplace_summarizer.eval()
for p in laplace_summarizer.parameters():
    p.requires_grad = False

# ---- Estimate global latent stats (uses same window-normalization) ----
vae = LatentVAE(
    seq_len=crypto_config.PRED,
    latent_dim=crypto_config.VAE_LATENT_DIM,
    latent_channel=crypto_config.VAE_LATENT_CHANNELS,
    enc_layers=crypto_config.VAE_LAYERS, enc_heads=crypto_config.VAE_HEADS, enc_ff=crypto_config.VAE_FF,
    dec_layers=crypto_config.VAE_LAYERS, dec_heads=crypto_config.VAE_HEADS, dec_ff=crypto_config.VAE_FF
).to(device)

if os.path.isfile(crypto_config.VAE_CKPT):
    ckpt = torch.load(crypto_config.VAE_CKPT, map_location=device)
    sd = ckpt.get("state_dict", ckpt)
    vae.load_state_dict(sd)
    print("Loaded VAE checkpoint:", crypto_config.VAE_CKPT)

vae.eval()
for p in vae.parameters():
    p.requires_grad = False

mu_mean, mu_std = compute_latent_stats(vae, train_dl, device)

# ---- Conditional diffusion model ----
diff_model = LLapDiT(
    data_dim=crypto_config.VAE_LATENT_CHANNELS, hidden_dim=crypto_config.MODEL_WIDTH,
    num_layers=crypto_config.NUM_LAYERS, num_heads=crypto_config.NUM_HEADS,
    predict_type=crypto_config.PREDICT_TYPE, laplace_k=crypto_config.LAPLACE_K,
    timesteps=crypto_config.TIMESTEPS, schedule=crypto_config.SCHEDULE,
    dropout=crypto_config.DROPOUT, attn_dropout=crypto_config.ATTN_DROPOUT,
    self_conditioning=crypto_config.SELF_COND,
    lap_mode_main=crypto_config.LAP_MODE
).to(device)

# ---- Calculate the variance of the v-prediction target ----
v_variance = calculate_v_variance(
    scheduler=diff_model.scheduler,
    dataloader=val_dl,
    vae=vae,
    device=device,
    latent_stats=(mu_mean, mu_std)
)
print(f"calculated V-Prediction Target Variance: {v_variance:.4f}")
print("=========================================================")
BASELINE_V_VARIANCE = float(v_variance)

ema = EMA(diff_model, decay=crypto_config.EMA_DECAY) if crypto_config.USE_EMA_EVAL else None

scheduler = diff_model.scheduler
optimizer = torch.optim.AdamW(diff_model.parameters(),
                              lr=crypto_config.BASE_LR,
                              weight_decay=crypto_config.WEIGHT_DECAY)
lr_sched = make_warmup_cosine(optimizer, crypto_config.EPOCHS * max(1, len(train_dl)),
                              warmup_frac=crypto_config.WARMUP_FRAC,
                              base_lr=crypto_config.BASE_LR,
                              min_lr=crypto_config.MIN_LR)
scaler = GradScaler(enabled=(device.type == "cuda"))


# ============================ train/val loops ============================
def train_one_epoch(epoch: int):
    diff_model.train()
    running_loss = 0.0
    num_samples = 0
    global global_step

    for xb, yb, meta in train_dl:
        V, T = xb
        mask_bn = meta["entity_mask"]

        cond_summary = build_context(laplace_summarizer, V, T, mask_bn, device)
        y_in, batch_ids = flatten_targets(yb, mask_bn, device)
        if y_in is None:
            continue

        cond_summary_flat = cond_summary[batch_ids]  # [Beff,S,Hm]
        mu_norm = encode_mu_norm(
            vae, y_in,
            mu_mean=mu_mean, mu_std=mu_std
        )

        # --------- per-sample classifier-free dropout mask ---------
        Beff = mu_norm.size(0)
        p_drop = float(crypto_config.DROP_COND_P)
        m_cond = (torch.rand(Beff, device=device) >= p_drop)
        idx_c = m_cond.nonzero(as_tuple=False).squeeze(1)
        idx_u = (~m_cond).nonzero(as_tuple=False).squeeze(1)

        # shared t / noise / q_sample for the whole batch
        t = sample_t_uniform(scheduler, Beff, device)
        noise = torch.randn_like(mu_norm)
        x_t, eps_true = scheduler.q_sample(mu_norm, t, noise)

        # --------- self-conditioning per branch (optional) ---------
        sc_feat_c = sc_feat_u = None
        use_sc = (epoch >= crypto_config.SELF_COND_START_EPOCH
                  and torch.rand(()) < crypto_config.SELF_COND_P) and crypto_config.SELF_COND
        if use_sc:
            with torch.no_grad():
                if idx_c.numel() > 0:
                    pred_ng_c = diff_model(x_t[idx_c], t[idx_c], cond_summary=cond_summary_flat[idx_c], sc_feat=None)
                    sc_feat_c = scheduler.to_x0(x_t[idx_c], t[idx_c], pred_ng_c, crypto_config.PREDICT_TYPE).detach()
                if idx_u.numel() > 0:
                    pred_ng_u = diff_model(x_t[idx_u], t[idx_u], cond_summary=None, sc_feat=None)
                    sc_feat_u = scheduler.to_x0(x_t[idx_u], t[idx_u], pred_ng_u, crypto_config.PREDICT_TYPE).detach()

        # --------- compute losses on each subset, weighted by fraction ---------
        optimizer.zero_grad(set_to_none=True)
        loss = 0.0

        with autocast(enabled=(device.type == "cuda")):
            if idx_c.numel() > 0:
                loss_c = diffusion_loss(
                    diff_model, scheduler, mu_norm[idx_c], t[idx_c],
                    cond_summary=cond_summary_flat[idx_c], predict_type=crypto_config.PREDICT_TYPE,
                    weight_scheme=crypto_config.LOSS_WEIGHT_SCHEME,
                    minsnr_gamma=crypto_config.MINSNR_GAMMA,
                    sc_feat=sc_feat_c,
                    reuse_xt_eps=(x_t[idx_c], eps_true[idx_c]),
                )
                loss = loss + loss_c * (idx_c.numel() / Beff)

            if idx_u.numel() > 0:
                loss_u = diffusion_loss(
                    diff_model, scheduler, mu_norm[idx_u], t[idx_u],
                    cond_summary=None, predict_type=crypto_config.PREDICT_TYPE,
                    weight_scheme=crypto_config.LOSS_WEIGHT_SCHEME,
                    minsnr_gamma=crypto_config.MINSNR_GAMMA,
                    sc_feat=sc_feat_u,
                    reuse_xt_eps=(x_t[idx_u], eps_true[idx_u]),
                )
                loss = loss + loss_u * (idx_u.numel() / Beff)

        # guard for numerical issues
        if not torch.isfinite(loss):
            print("[warn] non-finite loss detected; skipping step")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if crypto_config.GRAD_CLIP and crypto_config.GRAD_CLIP > 0:
            nn.utils.clip_grad_norm_(diff_model.parameters(), crypto_config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(diff_model)
        lr_sched.step()

        running_loss += float(loss.item()) * Beff
        num_samples += Beff

    return running_loss / max(1, num_samples)


@torch.inference_mode()
def validate():
    diff_model.eval()
    total, count = 0.0, 0
    cond_gap_accum, cond_gap_batches = 0.0, 0

    if ema is not None:
        ema.store(diff_model)
        ema.copy_to(diff_model)

    for xb, yb, meta in val_dl:
        V, T = xb
        mask_bn = meta["entity_mask"]

        cond_summary = build_context(laplace_summarizer, V, T, mask_bn, device)
        y_in, batch_ids = flatten_targets(yb, mask_bn, device)
        if y_in is None:
            continue
        cond_summary_flat = cond_summary[batch_ids]
        mu_norm = encode_mu_norm(
            vae, y_in,
            mu_mean=mu_mean, mu_std=mu_std
        )

        # main validation loss
        t = sample_t_uniform(scheduler, mu_norm.size(0), device)
        loss = diffusion_loss(
            diff_model, scheduler, mu_norm, t,
            cond_summary=cond_summary_flat, predict_type=crypto_config.PREDICT_TYPE
        )
        total += loss.item() * mu_norm.size(0)
        count += mu_norm.size(0)

        # low-variance cond_gap probe: reuse same (t, x_t, eps) for both branches
        probe_n = min(128, mu_norm.size(0))
        if probe_n > 0:
            mu_p = mu_norm[:probe_n]
            cs_p = cond_summary_flat[:probe_n]
            t_p = sample_t_uniform(scheduler, probe_n, device)
            noise_p = torch.randn_like(mu_p)
            x_t_p, eps_p = scheduler.q_sample(mu_p, t_p, noise_p)

            loss_cond = diffusion_loss(
                diff_model, scheduler, mu_p, t_p,
                cond_summary=cs_p, predict_type=crypto_config.PREDICT_TYPE,
                reuse_xt_eps=(x_t_p, eps_p)
            ).item()

            loss_unco = diffusion_loss(
                diff_model, scheduler, mu_p, t_p,
                cond_summary=None, predict_type=crypto_config.PREDICT_TYPE,
                reuse_xt_eps=(x_t_p, eps_p)
            ).item()

            cond_gap_accum += (loss_unco - loss_cond)
            cond_gap_batches += 1

    if ema is not None:
        ema.restore(diff_model)

    avg_val = total / max(1, count)
    cond_gap = (cond_gap_accum / cond_gap_batches) if cond_gap_batches > 0 else float("nan")
    return avg_val, cond_gap


# ============================ run ============================
skip_with_trained_model = crypto_config.TRAINED_LLapDiT

if skip_with_trained_model and os.path.exists(skip_with_trained_model) and crypto_config.downstream:
    print(f"Skipping training. Loading model from: {skip_with_trained_model}")
    ckpt = torch.load(skip_with_trained_model, map_location=device)

    # Load model weights
    diff_model.load_state_dict(ckpt["model_state"])
    print("Loaded model state.")

    # Load EMA weights if they exist in the checkpoint and are enabled
    if ema is not None and "ema_state" in ckpt:
        ema.load_state_dict(ckpt["ema_state"])
        print("Loaded EMA state.")

    # Load stats needed for downstream tasks
    mu_mean = ckpt["mu_mean"].to(device)
    mu_std = ckpt["mu_std"].to(device)

else:
    # --- TRAINING LOOP ---
    if skip_with_trained_model:

        ckpt = torch.load(skip_with_trained_model, map_location=device)

        # Load model weights
        diff_model.load_state_dict(ckpt["model_state"])
        print("Loaded model state.")

        # Load EMA weights if they exist in the checkpoint and are enabled
        if ema is not None and "ema_state" in ckpt:
            ema.load_state_dict(ckpt["ema_state"])
            print("Loaded EMA state.")
    else:
        print(f"Model path not found: {skip_with_trained_model}. Starting training from scratch.")

    best_val = float("inf")
    patience = 0
    current_best_path = None
    os.makedirs(crypto_config.CKPT_DIR, exist_ok=True)

    for epoch in tqdm(range(1, crypto_config.EPOCHS + 1), desc="Epochs"):
        train_loss = train_one_epoch(epoch)
        val_loss, cond_gap = validate()
        Z = crypto_config.VAE_LATENT_CHANNELS
        val_vs_baseline = (
            val_loss / BASELINE_V_VARIANCE
            if BASELINE_V_VARIANCE and BASELINE_V_VARIANCE > 0
            else float("inf")
        )
        improvement = (
            BASELINE_V_VARIANCE - val_loss
            if BASELINE_V_VARIANCE is not None
            else float("nan")
        )
        cond_gap_scaled = cond_gap * Z if math.isfinite(cond_gap) else float("nan")
        print(
            f"Epoch {epoch:03d} | train: {train_loss:.6f} "
            f"| val: {val_loss:.6f} | cond_gap: {cond_gap_scaled:.6f} "
            f"| val/v_var: {val_vs_baseline:.6f} | improvement: {improvement:.6f}"
        )

        # checkpoint best (with EMA state)
        if val_loss < best_val:
            best_val = val_loss;
            patience = 0
            if current_best_path and os.path.exists(current_best_path):
                os.remove(current_best_path)

            val_vs_baseline = (
                val_loss / BASELINE_V_VARIANCE
                if BASELINE_V_VARIANCE and BASELINE_V_VARIANCE > 0
                else float("inf")
            )
            ratio_str = (
                f"{val_vs_baseline:.6f}"
                if math.isfinite(val_vs_baseline)
                else ("inf" if val_vs_baseline > 0 else "nan")
            )

            ckpt_path = os.path.join(
                crypto_config.CKPT_DIR,
                f"mode-{crypto_config.LAP_MODE}-pred-{crypto_config.PRED}"
                f"-val-{val_loss:.6f}-cond-{cond_gap * Z:.6f}-ratio-{ratio_str}.pt"
            )
            save_payload = {
                "epoch": epoch,
                "model_state": diff_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "mu_mean": mu_mean.detach().cpu(),
                "mu_std": mu_std.detach().cpu(),
                "predict_type": crypto_config.PREDICT_TYPE,
                "timesteps": crypto_config.TIMESTEPS,
                "schedule": crypto_config.SCHEDULE,
            }
            if ema is not None:
                save_payload["ema_state"] = ema.state_dict()
                save_payload["ema_decay"] = crypto_config.EMA_DECAY
            torch.save(save_payload, ckpt_path)
            current_best_path = ckpt_path
        else:
            patience += 1
            if patience >= crypto_config.EARLY_STOP:
                print("Early stopping.")
                break


# ============================ Decoder finetune + regression eval ============================

def finetune_vae_decoder(
        vae,
        diff_model,
        train_dl,
        val_dl,
        device,
        *,
        mu_mean,
        mu_std,
        ema=None,
        epochs: int = 3,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        gen_steps: int = 36,
        guidance_strength: float = 2.0,
        guidance_power: float = 0.3,
        lambda_rec_anchor: float = 0.25,
):
    """
    Fine-tune ONLY the VAE decoder to adapt to the diffusion model's generated latent x0_norm.
    """
    print(f"[decoder-ft(gen)] epochs={epochs}, lr={lr}, steps={gen_steps}, "
          f"guidance={guidance_strength}, power={guidance_power}, anchor={lambda_rec_anchor}")

    # --- Freeze encoder; train decoder only ---
    for p in vae.encoder.parameters(): p.requires_grad = False
    for p in vae.decoder.parameters(): p.requires_grad = True
    for p in vae.mu_head.parameters(): p.requires_grad = False
    for p in vae.logvar_head.parameters(): p.requires_grad = False
    vae.train()

    # --- Prepare diffusion model (teacher) ---
    diff_model.eval()
    use_ema = (ema is not None)
    if use_ema:
        ema.store(diff_model)
        ema.copy_to(diff_model)

    opt = torch.optim.AdamW(vae.decoder.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    mse = torch.nn.MSELoss()

    def _run_epoch(loader, tag, train: bool):
        vae.train() if train else vae.eval()
        total_gen, total_anchor, total_all, count = 0.0, 0.0, 0.0, 0

        for xb, yb, meta in loader:
            V, T = xb
            mask_bn = meta["entity_mask"]
            cs_full = build_context(laplace_summarizer, V, T, mask_bn, device)
            y_true, batch_ids = _flatten_for_mask(yb, mask_bn, device)
            if y_true is None:
                continue
            cs = cs_full[batch_ids]

            Beff, Hcur, Z = y_true.size(0), y_true.size(1), mu_mean.numel()

            # ---- Generate x0_norm in latent space (no grads) ----
            with torch.no_grad():
                x0_norm_gen = diff_model.generate(
                    shape=(Beff, Hcur, Z), steps=gen_steps,
                    guidance_strength=guidance_strength, guidance_power=guidance_power,
                    cond_summary=cs, self_cond=crypto_config.SELF_COND, cfg_rescale=True,
                )

                # ---- Forward decoder on generated latents ----
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                y_hat_gen = decode_latents_with_vae(
                    vae, x0_norm_gen.detach(), mu_mean=mu_mean, mu_std=mu_std
                )
                loss_gen = mse(y_hat_gen, y_true)

                # ---- Teacher-forced anchor loss ----
                with torch.no_grad():
                    _, mu_enc, logvar_enc = vae(y_true)
                    z_enc = vae.reparameterize(mu_enc, logvar_enc)
                y_hat_rec = vae.decoder(z_enc.detach(), encoder_skips=None)
                loss_anchor = mse(y_hat_rec, y_true)

                loss = loss_gen + lambda_rec_anchor * loss_anchor

            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            bs = y_true.size(0)
            total_gen += loss_gen.item() * bs
            total_anchor += loss_anchor.item() * bs
            total_all += loss.item() * bs
            count += bs

        print(f"[decoder-ft(gen)] {tag}  loss_gen={total_gen / max(1, count):.6f}  "
              f"loss_anchor={total_anchor / max(1, count):.6f}  total={total_all / max(1, count):.6f}")

    for e in range(1, epochs + 1):
        _run_epoch(train_dl, f"train@{e}", train=True)
        _run_epoch(val_dl, f"val@{e}", train=False)

    # --- Restore original states ---
    for p in vae.decoder.parameters(): p.requires_grad = False
    if use_ema:
        ema.restore(diff_model)


@torch.inference_mode()
def evaluate_regression(
        diff_model, vae, dataloader, device, mu_mean, mu_std, config, ema=None,
        steps: int = 36, guidance_strength: float = 2.0, guidance_power: float = 0.3,
        aggregation_method: str = 'mean',
        quantiles: tuple = (0.1, 0.5, 0.9),
):
    """
    Evaluates the model by generating multiple samples and creating a probabilistic forecast.

    Metrics:
      - MAE / MSE on the aggregated point forecast (mean or median across samples) — exact per-element.
      - CRPS via all-pairs estimator: CRPS = E[|X - y|] - 0.5 * E[|X - X'|].
      - Pinball (quantile) loss for given quantiles:
            L_q(y, ŷ_q) = mean( max(q*(y - ŷ_q), (q-1)*(y - ŷ_q)) )
        where ŷ_q is the sample quantile of the predictive distribution at level q.

    Args:
        aggregation_method: 'mean' or 'median' for the point forecast.
        quantiles: tuple of quantile levels in (0,1), e.g., (0.1, 0.5, 0.9).

    Returns:
        dict with keys:
            crps, mae, mse, num_samples, aggregation, pinball (dict {q: loss})
    """
    if aggregation_method not in ['mean', 'median']:
        raise ValueError("aggregation_method must be either 'mean' or 'median'")
    if not all(0.0 < float(q) < 1.0 for q in quantiles):
        raise ValueError("All quantiles must be in the open interval (0, 1).")

    diff_model.eval()

    # Exact element-wise accumulators for deterministic metrics
    abs_sum, sq_sum, elts = 0.0, 0.0, 0
    # Batch-weighted CRPS accumulator (matches your previous semantics)
    crps_sum, n = 0.0, 0
    # Element-wise accumulators for pinball losses per quantile
    pinball_sums = {float(q): 0.0 for q in quantiles}

    num_samples = config.NUM_EVAL_SAMPLES
    if num_samples <= 1 and aggregation_method == 'median':
        print("Warning: Median aggregation is more meaningful with num_samples > 1.")

    # Use EMA weights if provided
    use_ema = (ema is not None)
    if use_ema:
        ema.store(diff_model)
        ema.copy_to(diff_model)

    warned_no_gt_scale = False  # one-time notice if config asks for GT-scale

    for xb, yb, meta in dataloader:
        V, T = xb
        mask_bn = meta["entity_mask"]

        cond_summary = build_context(laplace_summarizer, V, T, mask_bn, device)
        y_in, batch_ids = _flatten_for_mask(yb, mask_bn, device)
        if y_in is None:
            continue
        cs = cond_summary[batch_ids]

        Beff, Hcur, Z = y_in.size(0), y_in.size(1), config.VAE_LATENT_CHANNELS

        # ---- Generate multiple samples in latent space and decode ----

        all_y_hats = []
        for _ in range(num_samples):
            x0_norm = diff_model.generate(
                shape=(Beff, Hcur, Z), steps=steps,
                guidance_strength=guidance_strength, guidance_power=guidance_power,
                cond_summary=cs, self_cond=crypto_config.SELF_COND, cfg_rescale=True,
            )

            y_hat_sample = decode_latents_with_vae(
                vae, x0_norm, mu_mean=mu_mean, mu_std=mu_std
            )
            all_y_hats.append(y_hat_sample)

        # Stack samples: [S, B, H, C]
        all_samples = torch.stack(all_y_hats, dim=0)

        # ---- Point forecast via aggregation ----
        if aggregation_method == 'mean':
            point_forecast = all_samples.mean(dim=0)
        else:  # 'median'
            point_forecast = all_samples.median(dim=0).values

        # ---- Deterministic metrics (exact) ----
        res = point_forecast - y_in
        abs_sum += res.abs().sum().item()
        sq_sum += (res ** 2).sum().item()
        elts += res.numel()

        # ---- CRPS using all-pairs estimator ----
        # CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
        M = all_samples.shape[0]
        term1 = (all_samples - y_in.unsqueeze(0)).abs().mean(dim=0)  # [B,H,C]

        if M <= 1:
            term2 = torch.zeros_like(term1)
        else:
            diffs = (all_samples.unsqueeze(0) - all_samples.unsqueeze(1)).abs()  # [M,M,B,H,C]
            iu = torch.triu_indices(M, M, offset=1, device=diffs.device)
            diffs_ij = diffs[iu[0], iu[1], ...]  # [M*(M-1)/2,B,H,C]
            term2 = diffs_ij.mean(dim=0)  # [B,H,C]

        batch_crps = (term1 - 0.5 * term2).mean().item()
        crps_sum += batch_crps * Beff
        n += Beff

        # ---- Pinball (quantile) loss ----
        for q in quantiles:
            q = float(q)
            y_q = torch.quantile(all_samples, q, dim=0, interpolation="linear")  # [B,H,C]
            diff = y_in - y_q  # note: y - ŷ_q
            # pinball per element
            loss_q = torch.maximum(q * diff, (q - 1.0) * diff)  # [B,H,C]
            pinball_sums[q] += loss_q.sum().item()

    if use_ema:
        ema.restore(diff_model)

    mae = abs_sum / max(1, elts)
    mse = sq_sum / max(1, elts)
    crps = crps_sum / max(1, n)
    pinball = {q: (pinball_sums[q] / max(1, elts)) for q in pinball_sums.keys()}

    # Print a concise summary
    qs_fmt = ", ".join(f"{q:.2f}:{pinball[q]:.6f}" for q in sorted(pinball.keys()))
    print(f"[test ({num_samples} samples, aggregation: {aggregation_method})]")
    print(f"  CRPS: {crps:.6f} | MAE: {mae:.6f} | MSE: {mse:.6f} | Pinball[{qs_fmt}]")

    return {
        "crps": crps,
        "mae": mae,
        "mse": mse,
        "pinball": pinball,
        "num_samples": num_samples,
        "aggregation": aggregation_method,
    }


# ---------- Run decoder FT + test regression once training stops ----------
if True:
    # It's good practice to pass the whole config object
    # to avoid long argument lists and make adding new parameters easier.
    if crypto_config.DECODER_FT_EPOCHS > 0:
        torch.manual_seed(42)
        finetune_vae_decoder(
            vae, diff_model, train_dl, val_dl, device,
            mu_mean=mu_mean, mu_std=mu_std,
            ema=ema,
            epochs=crypto_config.DECODER_FT_EPOCHS,
            lr=crypto_config.DECODER_FT_LR,  # CRITICAL BUG FIX
            gen_steps=crypto_config.GEN_STEPS,
            guidance_strength=crypto_config.GUIDANCE_STRENGTH,
            guidance_power=crypto_config.GUIDANCE_POWER,
            lambda_rec_anchor=crypto_config.DECODER_FT_ANCHOR
        )

    # Evaluate conditional regression on test set
    evaluate_regression(
        diff_model, vae, test_dl, device, mu_mean, mu_std,
        config=crypto_config,  # Pass config object
        steps=crypto_config.GEN_STEPS,
        guidance_strength=crypto_config.GUIDANCE_STRENGTH,
        guidance_power=crypto_config.GUIDANCE_POWER,
        aggregation_method='mean',
        quantiles=(0.1, 0.5, 0.9)
    )
