import os, torch
import crypto_config
from torch import nn
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from Dataset.fin_dataset import run_experiment
from Model.llapdit import LLapDiT
from Model.llapdit_utils import (EMA, set_torch, _flatten_for_mask,
                                 make_warmup_cosine, diffusion_loss,
                                 build_context, flatten_targets,
                                 sample_t_uniform, simple_norm, invert_simple_norm
                                 )

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
H = yb0.shape[-1]
assert Fv == Ft, f"Expected Fv == Ft, got {Fv} vs {Ft}"
print("V:", V0.shape, "T:", T0.shape, "y:", yb0.shape)
all_y = []
for (_, y, meta) in train_dl:
    m = meta["entity_mask"]
    B, N, H = y.shape
    y_flat = y.reshape(B * N, H).unsqueeze(-1)
    m_flat = m.reshape(B * N)
    if not m_flat.any():
        continue
    y_in = y_flat[m_flat].to(device)

    all_y.append(y_in.detach().cpu())

mu_cat = torch.cat(all_y, dim=0)  # [sum(Beff), L, Z]
mu_mean = mu_cat.mean(dim=(0, 1)).to(device)
mu_std = mu_cat.std(dim=(0, 1), correction=0).clamp_min(1e-6).to(device)

# ---- Conditional diffusion model ----
diff_model = LLapDiT(
    data_dim=1, hidden_dim=crypto_config.MODEL_WIDTH,
    num_layers=crypto_config.NUM_LAYERS, num_heads=crypto_config.NUM_HEADS,
    predict_type=crypto_config.PREDICT_TYPE, laplace_k=crypto_config.LAPLACE_K, global_k=crypto_config.GLOBAL_K,
    timesteps=crypto_config.TIMESTEPS, schedule=crypto_config.SCHEDULE,
    dropout=crypto_config.DROPOUT, attn_dropout=crypto_config.ATTN_DROPOUT,
    self_conditioning=crypto_config.SELF_COND,
    context_dim=Fv, num_entities=N0, context_len=crypto_config.CONTEXT_LEN,
    lap_mode=crypto_config.LAP_MODE
).to(device)

# ---- Calculate the variance of the v-prediction target ----

all_v_targets = []
print("Calculating variance of v-prediction target...")

# Loop through the validation set
for xb, yb, meta in val_dl:
    # This block is the same as in your validate() function
    # It gets the normalized latent variable 'mu_norm' which is the x0 for diffusion
    y_in, _ = flatten_targets(yb, meta["entity_mask"], device)
    if y_in is None:
        continue

    # Now, simulate the process for creating the 'v' target
    # 1. Sample random timesteps
    t = sample_t_uniform(diff_model.scheduler, y_in.size(0), device)

    # 2. Create the noise that would be added
    noise = torch.randn_like(y_in)

    # 3. Apply the forward process to get the noised latent x_t
    x_t, _ = diff_model.scheduler.q_sample(y_in, t, noise)

    # 4. Calculate the ground-truth 'v' from x_t and the noise
    v_target = diff_model.scheduler.v_from_eps(x_t, t, noise)
    all_v_targets.append(v_target.detach().cpu())


all_v_targets_cat = torch.cat(all_v_targets, dim=0)
v_variance = all_v_targets_cat.var(correction=0).item()

print(f"calculated V-Prediction Target Variance: {v_variance:.4f}")
print("=========================================================")

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

        cond_summary = build_context(diff_model, V, T, mask_bn, device)
        y_in, batch_ids = flatten_targets(yb, mask_bn, device)
        if y_in is None:
            continue

        cond_summary_flat = cond_summary[batch_ids]  # [Beff,S,Hm]
        mu_norm = simple_norm(y_in,
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
        if getattr(crypto_config, "GRAD_CLIP", 0.0) and crypto_config.GRAD_CLIP > 0:
            nn.utils.clip_grad_norm_(diff_model.parameters(), crypto_config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(diff_model)
        lr_sched.step()

        running_loss += float(loss.item()) * Beff
        num_samples += Beff

    return running_loss / max(1, num_samples)


@torch.no_grad()
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

        cond_summary = build_context(diff_model, V, T, mask_bn, device)
        y_in, batch_ids = flatten_targets(yb, mask_bn, device)
        if y_in is None:
            continue
        cond_summary_flat = cond_summary[batch_ids]
        mu_norm = simple_norm(y_in,
            mu_mean=mu_mean, mu_std=mu_std
        )

        # main validation loss
        t = sample_t_uniform(scheduler, mu_norm.size(0), device)
        loss = diffusion_loss(
            diff_model, scheduler, mu_norm, t,
            cond_summary=cond_summary_flat, predict_type=crypto_config.PREDICT_TYPE,
            weight_scheme=crypto_config.LOSS_WEIGHT_SCHEME,
            minsnr_gamma=crypto_config.MINSNR_GAMMA
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
skip_with_trained_model = ""

if skip_with_trained_model and os.path.exists(skip_with_trained_model):
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
        print(f"Model path not found: {skip_with_trained_model}. Starting training from scratch.")

    best_val = float("inf")
    patience = 0
    current_best_path = None
    os.makedirs(crypto_config.CKPT_DIR, exist_ok=True)

    for epoch in tqdm(range(1, crypto_config.EPOCHS + 1), desc="Epochs"):
        train_loss = train_one_epoch(epoch)
        val_loss, cond_gap = validate()
        Z = 1
        print(f"Epoch {epoch:03d} | train: {train_loss:.6f} (/Z: {train_loss / Z:.6f}) "
              f"| val: {val_loss:.6f} (/Z: {val_loss / Z:.6f}) | cond_gap: {cond_gap:.6f}")

        # checkpoint best (with EMA state)
        if val_loss < best_val:
            best_val = val_loss;
            patience = 0
            if current_best_path and os.path.exists(current_best_path):
                os.remove(current_best_path)

            ckpt_path = os.path.join(
                crypto_config.CKPT_DIR,
                f"epoch_{epoch:03d}_val_{val_loss / Z:.6f}_cond_{cond_gap}.pt"
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
            # print("Saved:", ckpt_path)
            current_best_path = ckpt_path
        else:
            patience += 1
            if patience >= crypto_config.EARLY_STOP:
                print("Early stopping.")
                break

# ============================ Decoder finetune + regression eval ============================

@torch.no_grad()
def evaluate_regression(
    diff_model, dataloader, device, mu_mean, mu_std, config, ema=None,
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

        cond_summary = build_context(diff_model, V, T, mask_bn, device)
        y_in, batch_ids = _flatten_for_mask(yb, mask_bn, device)
        if y_in is None:
            continue
        cs = cond_summary[batch_ids]

        Beff, Hcur, Z = y_in.size(0), y_in.size(1), 1

        # ---- Generate multiple samples in latent space and decode ----

        all_y_hats = []
        for _ in range(num_samples):
            x0_norm = diff_model.generate(
                shape=(Beff, Hcur, Z), steps=steps,
                guidance_strength=guidance_strength, guidance_power=guidance_power,
                cond_summary=cs, self_cond=crypto_config.SELF_COND, cfg_rescale=True,
            )

            y_hat_sample = invert_simple_norm(x0_norm, mu_mean=mu_mean, mu_std=mu_std
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
        sq_sum  += (res ** 2).sum().item()
        elts    += res.numel()

        # ---- CRPS using all-pairs estimator ----
        # CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
        M = all_samples.shape[0]
        term1 = (all_samples - y_in.unsqueeze(0)).abs().mean(dim=0)  # [B,H,C]

        if M <= 1:
            term2 = torch.zeros_like(term1)
        else:
            diffs = (all_samples.unsqueeze(0) - all_samples.unsqueeze(1)).abs()  # [M,M,B,H,C]
            iu = torch.triu_indices(M, M, offset=1, device=diffs.device)
            diffs_ij = diffs[iu[0], iu[1], ...]                                   # [M*(M-1)/2,B,H,C]
            term2 = diffs_ij.mean(dim=0)                # [B,H,C]

        batch_crps = (term1 - 0.5 * term2).mean().item()
        crps_sum += batch_crps * Beff
        n += Beff

        # ---- Pinball (quantile) loss ----
        for q in quantiles:
            q = float(q)
            y_q = torch.quantile(all_samples, q, dim=0, interpolation="linear")   # [B,H,C]
            diff = y_in - y_q                                                     # note: y - ŷ_q
            # pinball per element
            loss_q = torch.maximum(q * diff, (q - 1.0) * diff)                    # [B,H,C]
            pinball_sums[q] += loss_q.sum().item()

    if use_ema:
        ema.restore(diff_model)

    mae = abs_sum / max(1, elts)
    mse = sq_sum  / max(1, elts)
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
    # Evaluate conditional regression on test set
    evaluate_regression(
        diff_model, test_dl, device, mu_mean, mu_std,
        config=crypto_config,  # Pass config object
        steps=crypto_config.GEN_STEPS,
        guidance_strength=crypto_config.GUIDANCE_STRENGTH,
        guidance_power=crypto_config.GUIDANCE_POWER,
        aggregation_method='mean',
        quantiles=(0.1, 0.5, 0.9)
    )
