import os, torch
import crypto_config
from torch import nn
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from Dataset.fin_dataset import run_experiment
from Latent_Space.latent_vae import LatentVAE
from Model.lladit import LLapDiT
from Model.cond_diffusion_utils import (EMA, set_torch, encode_mu_norm,
                                        make_warmup_cosine,
                                        calculate_v_variance, compute_latent_stats,
                                        diffusion_loss, build_context,
                                        flatten_targets, sample_t_uniform,
                                        decode_latents_with_vae
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
v_variance = calculate_v_variance(
    scheduler=diff_model.scheduler,
    dataloader=train_dl,
    vae=vae,
    device=device,
    latent_stats=(mu_mean, mu_std),
    use_ewma=crypto_config.USE_EWMA,
    ewma_lambda=crypto_config.EWMA_LAMBDA
)
print(f"calculated V-Prediction Target Variance: {v_variance:.4f}")
print("=========================================================")


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
        # sample once, reuse for SC and loss
        noise = torch.randn_like(mu_norm)
        x_t, eps_true = scheduler.q_sample(mu_norm, t, noise)
        
        # ---- self-conditioning (optional) ----
        sc_feat = None
        if epoch >= crypto_config.SELF_COND_START_EPOCH and torch.rand(()) < crypto_config.SELF_COND_P:
            with torch.no_grad():
                pred_ng = diff_model(x_t, t, cond_summary=cs, sc_feat=None)  # <— x_t (no grads)
                sc_feat = scheduler.to_x0(x_t, t, pred_ng, crypto_config.PREDICT_TYPE).detach()
        
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(device.type == "cuda")):
            loss = diffusion_loss(
                diff_model, scheduler, mu_norm, t,
                cond_summary=cs, predict_type=crypto_config.PREDICT_TYPE,
                weight_scheme=crypto_config.LOSS_WEIGHT_SCHEME,
                minsnr_gamma=crypto_config.MINSNR_GAMMA,
                sc_feat=sc_feat,
                reuse_xt_eps=(x_t, eps_true),
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

for epoch in tqdm(range(1, crypto_config.EPOCHS + 1), desc="Epochs"):
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
        # print("Saved:", ckpt_path)
        current_best_path = ckpt_path
    else:
        patience += 1
        if patience >= crypto_config.EARLY_STOP:
            print("Early stopping.")
            break


# ============================ Decoder finetune + regression eval ============================
@torch.no_grad()
def _flatten_for_mask(yb, mask_bn, device):
    y_in, batch_ids = flatten_targets(yb, mask_bn, device)
    return y_in, batch_ids


def finetune_vae_decoder(vae, train_dl, val_dl, device,
                         epochs: int = 5, lr: float = 1e-4, weight_decay: float = 1e-6):
    print(f"[decoder-ft] epochs={epochs}, lr={lr}")
    # freeze encoder, unfreeze decoder
    for p in vae.parameters(): p.requires_grad = False
    for p in vae.decoder.parameters(): p.requires_grad = True

    vae.train()
    opt = torch.optim.AdamW(vae.decoder.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()

    def _run_epoch(loader, tag):
        total, count = 0.0, 0
        for xb, yb, meta in loader:
            y_in, _ = _flatten_for_mask(yb, meta["entity_mask"], device)
            if y_in is None: continue
            with torch.no_grad():
                _, mu, _ = vae(y_in)      # [Beff, H, Z]
                mu = mu.detach()
            pred = vae.decoder(mu, encoder_skips=None)     # [Beff, H, 1]
            loss = crit(pred, y_in)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * y_in.size(0)
            count += y_in.size(0)
        print(f"[decoder-ft] {tag} MSE: {total / max(1, count):.6f}")

    for e in range(1, epochs + 1):
        _run_epoch(train_dl, f"train@{e}")
        _run_epoch(val_dl,   f"val@{e}")
    vae.eval()
    # keep decoder trainable=false after FT
    for p in vae.decoder.parameters(): p.requires_grad = False


@torch.no_grad()
def evaluate_regression(diff_model, vae, dataloader, device, mu_mean, mu_std,
                        steps: int = 36, guidance_strength: float = 2.0, guidance_power: float = 0.3):
    diff_model.eval()
    mse_sum, mae_sum, n = 0.0, 0.0, 0

    use_ema = (ema is not None)
    if use_ema:
        ema.store(diff_model); ema.copy_to(diff_model)

    for xb, yb, meta in dataloader:
        V, T = xb
        mask_bn = meta["entity_mask"]
        cond_summary = build_context(diff_model, V, T, mask_bn, device)  # [B,S,Hc]
        y_in, batch_ids = _flatten_for_mask(yb, mask_bn, device)
        if y_in is None: continue
        cs = cond_summary[batch_ids]  # align to selected rows

        Beff, Hcur, Z = y_in.size(0), y_in.size(1), crypto_config.VAE_LATENT_DIM
        # Conditional generation in latent space (normalized x0)
        x0_norm = diff_model.generate(
            shape=(Beff, Hcur, Z),
            steps=steps,
            guidance_strength=guidance_strength,
            guidance_power=guidance_power,
            series=None,
            cond_summary=cs,
            self_cond=True,
            cfg_rescale=True,
        )  # [Beff, H, Z]

        # Invert normalization and decode
        use_gt_window_scale = getattr(crypto_config, "DECODE_USE_GT_SCALE", True)
        s = None
        if use_gt_window_scale:
            with torch.no_grad():
                _, mu_gt, _ = vae(y_in)  # latent of the true target window
            if crypto_config.USE_EWMA:
                s = ewma_std(mu_gt, lam=crypto_config.EWMA_LAMBDA)  # [Beff,1,Z]
            else:
                s = mu_gt.std(dim=1, keepdim=True, correction=0).clamp_min(1e-6)
        
        y_hat = decode_latents_with_vae(
          vae, x0_norm, mu_mean=mu_mean, mu_std=mu_std, window_scale=s) # [Beff, H, 1]

        # Metrics
        res = y_hat - y_in
        mae = res.abs().mean().item()
        mse = (res ** 2).mean().item()
        mae_sum += mae * Beff
        mse_sum += mse * Beff
        n += Beff

    if use_ema:
        ema.restore(diff_model)

    print(f"[test] MAE: {mae_sum / max(1, n):.6f} | MSE: {mse_sum / max(1, n):.6f}")


# ---------- Run decoder FT + test regression once training stops ----------
if True:
    # Fine-tune decoder on train+val (optional; set epochs=0 to skip)
    ft_epochs = crypto_config.DECODER_FT_EPOCHS
    ft_lr     = crypto_config.DECODER_FT_LR
    if ft_epochs > 0:
        finetune_vae_decoder(vae, train_dl, val_dl, device, epochs=ft_epochs, lr=ft_lr)

    # Evaluate conditional regression on test set
    gen_steps  = crypto_config.GEN_STEPS
    guidance_g = crypto_config.GUIDANCE_STRENGTH
    guidance_p = crypto_config.GUIDANCE_POWER
    evaluate_regression(diff_model, vae, test_dl, device, mu_mean, mu_std,
                        steps=gen_steps, guidance_strength=guidance_g, guidance_power=guidance_p)
