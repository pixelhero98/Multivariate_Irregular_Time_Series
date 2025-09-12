import os, json, torch, math
import crypto_config
from torch import nn
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# --- project imports (unchanged) ---
from Dataset.fin_dataset import run_experiment
from Model.lladit import LLapDiT
from Model.cond_diffusion_utils import (
    EMA, set_torch, make_warmup_cosine, ewma_std,
    diffusion_loss, build_context,
    flatten_targets, sample_t_uniform,
    log_pole_health
)

# ========================= helpers =========================
def normalize_targets(y_in: torch.Tensor, use_ewma: bool, ewma_lambda: float, clip_val: float = 5.0):
    """
    Normalize target windows per sample.
      y_in: [Beff, H, C]
    Returns (y_norm, scale) where scale has shape [Beff, 1, C].
    """
    if use_ewma:
        s = ewma_std(y_in, lam=ewma_lambda)  # [Beff, 1, C]
    else:
        s = y_in.std(dim=1, keepdim=True, correction=0).clamp_min(1e-6)
    y_norm = (y_in / s).clamp(-clip_val, clip_val)
    return y_norm, s

def _print_log(metrics: dict, step: int, csv_path: str | None = None):
    kv = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    print(f"[step {step}] {kv}")
    if csv_path:
        hdr = (not os.path.exists(csv_path))
        with open(csv_path, "a") as f:
            if hdr:
                f.write("step," + ",".join(metrics.keys()) + "\n")
            f.write(str(step) + "," + ",".join(f"{v:.6f}" for v in metrics.values()) + "\n")

# ========================= evaluation =========================
@torch.no_grad()
def evaluate_regression_raw(
    diff_model, dataloader, device, config, ema=None,
    steps: int = 36, guidance_strength: float = 2.0, guidance_power: float = 0.3,
    aggregation_method: str = 'mean', quantiles: tuple = (0.1, 0.5, 0.9)
):
    """
    Raw-space evaluation mirroring the latent script:
      - Draw S samples per series: y_hat^{(s)} ~ p_theta(. | context)
      - Aggregation for point forecast: mean/median across samples
      - Metrics: MAE, MSE (exact), CRPS (all-pairs), Pinball losses at given quantiles
    """
    if aggregation_method not in ['mean', 'median']:
        raise ValueError("aggregation_method must be either 'mean' or 'median'")
    if not all(0.0 < float(q) < 1.0 for q in quantiles):
        raise ValueError("All quantiles must be in the open interval (0, 1).")

    diff_model.eval()

    abs_sum, sq_sum, elts = 0.0, 0.0, 0
    crps_sum, n = 0.0, 0
    pinball_sums = {float(q): 0.0 for q in quantiles}

    num_samples = int(getattr(config, "NUM_EVAL_SAMPLES", 8))

    use_ema = (ema is not None)
    if use_ema:
        ema.store(diff_model); ema.copy_to(diff_model)

    for xb, yb, meta in tqdm(dataloader, desc="test (raw)"):
        V, T = xb
        mask_bn = meta["entity_mask"]

        cond_summary = build_context(diff_model, V, T, mask_bn, device)
        y_in, batch_ids = flatten_targets(yb, mask_bn, device)
        if y_in is None:
            continue
        cs = cond_summary[batch_ids]

        B, Hcur, C = y_in.size()
        assert C == 1, "This raw script expects univariate targets (C==1)."

        # Draw multiple raw-space samples
        samples = []
        for _ in range(num_samples):
            y_hat = diff_model.generate(
                shape=(B, Hcur, C), steps=steps,
                guidance_strength=guidance_strength, guidance_power=guidance_power,
                cond_summary=cs, self_cond=config.SELF_COND, cfg_rescale=True
            )
            samples.append(y_hat)
        all_samples = torch.stack(samples, dim=0)  # [S,B,H,C]

        # Point forecast
        if aggregation_method == "mean":
            point_forecast = all_samples.mean(dim=0)
        else:
            point_forecast = all_samples.median(dim=0).values

        # Deterministic metrics
        res = point_forecast - y_in
        abs_sum += res.abs().sum().item()
        sq_sum  += (res ** 2).sum().item()
        elts    += res.numel()

        # CRPS
        M = all_samples.shape[0]
        term1 = (all_samples - y_in.unsqueeze(0)).abs().mean(dim=0)  # [B,H,C]
        if M <= 1:
            term2 = torch.zeros_like(term1)
        else:
            diffs = (all_samples.unsqueeze(0) - all_samples.unsqueeze(1)).abs()   # [M,M,B,H,C]
            iu = torch.triu_indices(M, M, offset=1, device=diffs.device)
            diffs_ij = diffs[iu[0], iu[1], ...]                                    # [M*(M-1)/2,B,H,C]
            term2 = (2.0 / (M * (M - 1))) * diffs_ij.mean(dim=0)                   # [B,H,C]
        batch_crps = (term1 - 0.5 * term2).mean().item()
        crps_sum += batch_crps * B
        n += B

        # Pinball
        for q in quantiles:
            q = float(q)
            y_q = torch.quantile(all_samples, q, dim=0, interpolation="linear")
            diff = y_in - y_q
            loss_q = torch.maximum(q * diff, (q - 1.0) * diff)
            pinball_sums[q] += loss_q.sum().item()

    if use_ema:
        ema.restore(diff_model)

    mae = abs_sum / max(1, elts)
    mse = sq_sum  / max(1, elts)
    crps = crps_sum / max(1, n)
    pinball = {q: (pinball_sums[q] / max(1, elts)) for q in pinball_sums.keys()}

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

# ========================= main =========================
def main():
    device = set_torch()
    torch.manual_seed(crypto_config.SEED)

    # ---- data ----
    train_dl, val_dl, test_dl = run_experiment(split="trainvaltest")  # keep your project helper

    # Peek to infer dims
    xb0, yb0, meta0 = next(iter(train_dl))
    V0, T0 = xb0
    B0, K0, N0, Fv = V0.shape   # context (values) [B,K,N,F]
    _,  _,  _, Ft = T0.shape    # context (diffs)  [B,K,N,F]
    assert Fv == Ft, f"Expected Fv == Ft, got {Fv} vs {Ft}"
    print("V:", V0.shape, "T:", T0.shape, "y:", yb0.shape)

    # ---- model ----
    diff_model = LLapDiT(
        data_dim=1,                             # raw target channels
        hidden_dim=crypto_config.MODEL_WIDTH,
        num_layers=crypto_config.NUM_LAYERS, num_heads=crypto_config.NUM_HEADS,
        predict_type=crypto_config.PREDICT_TYPE, laplace_k=crypto_config.LAPLACE_K, global_k=crypto_config.GLOBAL_K,
        timesteps=crypto_config.TIMESTEPS, schedule=crypto_config.SCHEDULE,
        dropout=crypto_config.DROPOUT, attn_dropout=crypto_config.ATTN_DROPOUT,
        self_conditioning=crypto_config.SELF_COND,
        context_dim=Fv, num_entities=N0, context_len=crypto_config.CONTEXT_LEN,
        lap_mode=crypto_config.LAP_MODE
    ).to(device)

    ema = EMA(diff_model, decay=crypto_config.EMA_DECAY) if crypto_config.USE_EMA_EVAL else None

    scheduler = diff_model.scheduler
    optimizer = torch.optim.AdamW(
        diff_model.parameters(), lr=crypto_config.BASE_LR,
        betas=(0.9, 0.999), weight_decay=crypto_config.WD
    )
    lr_sched = make_warmup_cosine(
        optimizer, warmup_steps=crypto_config.WARMUP_STEPS,
        total_steps=crypto_config.TOTAL_STEPS
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ========================= train/val fns =========================
    global_step = 0
    csv_path = os.path.join(crypto_config.OUT_DIR, "train_raw.csv")
    os.makedirs(crypto_config.OUT_DIR, exist_ok=True)

    def train_one_epoch(epoch: int):
        nonlocal global_step
        diff_model.train()
        running, n_seen = 0.0, 0

        p_drop = float(crypto_config.DROP_COND_P)
        use_sc_epoch = (epoch >= crypto_config.SELF_COND_START_EPOCH and crypto_config.SELF_COND)

        for xb, yb, meta in tqdm(train_dl, desc=f"train {epoch}"):
            V, T = xb
            mask_bn = meta["entity_mask"]
            cond_summary = build_context(diff_model, V, T, mask_bn, device)

            y_in, batch_ids = flatten_targets(yb, mask_bn, device)
            if y_in is None:
                continue
            cond_summary_flat = cond_summary[batch_ids]  # [Beff,S,Hm]

            # raw normalization
            y_norm, _ = normalize_targets(
                y_in, use_ewma=crypto_config.USE_EWMA, ewma_lambda=crypto_config.EWMA_LAMBDA
            )

            Beff = y_norm.size(0)
            m_cond = (torch.rand(Beff, device=device) >= p_drop)
            idx_c = m_cond.nonzero(as_tuple=False).squeeze(1)
            idx_u = (~m_cond).nonzero(as_tuple=False).squeeze(1)

            t = sample_t_uniform(scheduler, Beff, device)
            noise = torch.randn_like(y_norm)
            x_t, eps_true = scheduler.q_sample(y_norm, t, noise)

            # optional self-conditioning
            sc_feat_c = sc_feat_u = None
            if use_sc_epoch:
                with torch.no_grad():
                    if idx_c.numel() > 0:
                        pred_ng_c = diff_model(x_t[idx_c], t[idx_c],
                                               cond_summary=cond_summary_flat[idx_c], sc_feat=None)
                        x0_ng_c = scheduler.to_x0(x_t[idx_c], t[idx_c], pred_ng_c, predict_type=crypto_config.PREDICT_TYPE)
                        sc_feat_c = x0_ng_c.detach()
                    if idx_u.numel() > 0:
                        pred_ng_u = diff_model(x_t[idx_u], t[idx_u], cond_summary=None, sc_feat=None)
                        x0_ng_u = scheduler.to_x0(x_t[idx_u], t[idx_u], pred_ng_u, predict_type=crypto_config.PREDICT_TYPE)
                        sc_feat_u = x0_ng_u.detach()

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                loss_c = loss_u = torch.tensor(0.0, device=device)
                if idx_c.numel() > 0:
                    loss_c = diffusion_loss(
                        diff_model, scheduler, y_norm[idx_c], t[idx_c],
                        cond_summary=cond_summary_flat[idx_c], predict_type=crypto_config.PREDICT_TYPE,
                        weight_scheme=crypto_config.LOSS_WEIGHT_SCHEME,
                        minsnr_gamma=crypto_config.MINSNR_GAMMA, sc_feat=sc_feat_c,
                        reuse_xt_eps=(x_t[idx_c], eps_true[idx_c]),
                    )
                if idx_u.numel() > 0:
                    loss_u = diffusion_loss(
                        diff_model, scheduler, y_norm[idx_u], t[idx_u],
                        cond_summary=None, predict_type=crypto_config.PREDICT_TYPE,
                        weight_scheme=crypto_config.LOSS_WEIGHT_SCHEME,
                        minsnr_gamma=crypto_config.MINSNR_GAMMA, sc_feat=sc_feat_u,
                        reuse_xt_eps=(x_t[idx_u], eps_true[idx_u]),
                    )
                loss = (loss_c + loss_u)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(diff_model.parameters(), crypto_config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            lr_sched.step()

            running += loss.detach().item() * Beff
            n_seen += Beff
            global_step += 1

            # lightweight logging each log interval
            if global_step % crypto_config.LOG_EVERY_STEPS == 0:
                if ema is not None:
                    ema.store(diff_model); ema.copy_to(diff_model)
                log_pole_health([diff_model], lambda m: _print_log(m, global_step, csv_path=None), step=global_step, tag_prefix="train/")
                if ema is not None:
                    ema.restore(diff_model)

        return running / max(1, n_seen)

    @torch.no_grad()
    def validate(epoch: int):
        diff_model.eval()
        if ema is not None:
            ema.store(diff_model); ema.copy_to(diff_model)

        tot, n_seen = 0.0, 0
        for xb, yb, meta in tqdm(val_dl, desc=f"val {epoch}"):
            V, T = xb
            mask_bn = meta["entity_mask"]
            cond_summary = build_context(diff_model, V, T, mask_bn, device)

            y_in, batch_ids = flatten_targets(yb, mask_bn, device)
            if y_in is None:
                continue
            cond_summary_flat = cond_summary[batch_ids]

            y_norm, _ = normalize_targets(
                y_in, use_ewma=crypto_config.USE_EWMA, ewma_lambda=crypto_config.EWMA_LAMBDA
            )
            Beff = y_norm.size(0)
            t = sample_t_uniform(scheduler, Beff, device)
            noise = torch.randn_like(y_norm)
            x_t, eps_true = scheduler.q_sample(y_norm, t, noise)

            loss = diffusion_loss(
                diff_model, scheduler, y_norm, t,
                cond_summary=cond_summary_flat, predict_type=crypto_config.PREDICT_TYPE,
                weight_scheme=crypto_config.LOSS_WEIGHT_SCHEME,
                minsnr_gamma=crypto_config.MINSNR_GAMMA, sc_feat=None,
                reuse_xt_eps=(x_t, eps_true),
            )
            tot += loss.item() * Beff
            n_seen += Beff

        if ema is not None:
            ema.restore(diff_model)
        return tot / max(1, n_seen)

    # ========================= train loop =========================
    best = math.inf
    for epoch in range(crypto_config.EPOCHS):
        tr = train_one_epoch(epoch)
        va = validate(epoch)
        _print_log({"train/loss": tr, "val/loss": va}, step=epoch, csv_path=os.path.join(crypto_config.OUT_DIR, "raw_loss.csv"))
        if va < best and ema is not None:
            # Save EMA weights as best
            ema.store(diff_model); ema.copy_to(diff_model)
            torch.save(diff_model.state_dict(), os.path.join(crypto_config.OUT_DIR, "best_raw_ema.pt"))
            ema.restore(diff_model)
            best = va
        elif va < best:
            torch.save(diff_model.state_dict(), os.path.join(crypto_config.OUT_DIR, "best_raw.pt"))
            best = va

    # final checkpoint
    torch.save(diff_model.state_dict(), os.path.join(crypto_config.OUT_DIR, "last_raw.pt"))

    # ========================= downstream evaluation =========================
    results = evaluate_regression_raw(
        diff_model, test_dl, device, crypto_config, ema=ema,
        steps=crypto_config.GEN_STEPS,
        guidance_strength=crypto_config.GUIDANCE_STRENGTH,
        guidance_power=crypto_config.GUIDANCE_POWER,
        aggregation_method='median',
        quantiles=(0.1, 0.5, 0.9),
    )
    with open(os.path.join(crypto_config.OUT_DIR, "test_metrics_raw.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved test metrics to", os.path.join(crypto_config.OUT_DIR, "test_metrics_raw.json"))

if __name__ == "__main__":
    main()
