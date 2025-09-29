"""Train/validation script for LLapDiT on public long-range datasets."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import longrange_config as cfg
from Data_Prep.long_range_datasets import create_dataloaders
from Latent_Space.latent_vae import LatentVAE
from Model.llapdit import LLapDiT
from Model.llapdit_utils import (
    EMA,
    build_context,
    calculate_v_variance,
    compute_latent_stats,
    diffusion_loss,
    encode_mu_norm,
    flatten_targets,
    make_warmup_cosine,
    sample_t_uniform,
    set_torch,
)


device = set_torch()


def _resolve_data_root(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(Path(__file__).resolve().parent, path)


train_dl, val_dl, test_dl, dataset_info = create_dataloaders(
    dataset=cfg.DATASET_NAME,
    data_root=_resolve_data_root(cfg.DATA_ROOT),
    window=cfg.WINDOW,
    horizon=cfg.PRED,
    batch_size=cfg.BATCH_SIZE,
    ratios=cfg.RATIOS,
    stride=cfg.STRIDE,
    shuffle_train=cfg.SHUFFLE_TRAIN,
    num_workers=cfg.NUM_WORKERS,
    drop_last=cfg.DROP_LAST,
    scaling=cfg.SCALING,
)

print("Dataset:", dataset_info["description"], "path:", dataset_info["path"])
print("Split sizes (train/val/test):", dataset_info["sizes"])

xb0, yb0, meta0 = next(iter(train_dl))
V0, T0 = xb0
B0, N0, K0, Fv = V0.shape
Ft = T0.shape[-1]
H = yb0.shape[-1]
assert Fv == Ft, f"Expected V/T feature dims to match. Got {Fv} vs {Ft}."
print("Shapes -> V:", V0.shape, "T:", T0.shape, "y:", yb0.shape)


# ============================ Latent VAE ============================
vae = LatentVAE(
    seq_len=cfg.PRED,
    latent_dim=cfg.VAE_LATENT_DIM,
    latent_channel=cfg.VAE_LATENT_CHANNELS,
    enc_layers=cfg.VAE_LAYERS,
    enc_heads=cfg.VAE_HEADS,
    enc_ff=cfg.VAE_FF,
    dec_layers=cfg.VAE_LAYERS,
    dec_heads=cfg.VAE_HEADS,
    dec_ff=cfg.VAE_FF,
).to(device)

if os.path.isfile(cfg.VAE_CKPT):
    ckpt = torch.load(cfg.VAE_CKPT, map_location=device)
    vae.load_state_dict(ckpt.get("state_dict", ckpt))
    print("Loaded VAE checkpoint:", cfg.VAE_CKPT)

vae.eval()
for param in vae.parameters():
    param.requires_grad = False

mu_mean, mu_std = compute_latent_stats(vae, train_dl, device)


# ============================ Diffusion model ============================
diff_model = LLapDiT(
    data_dim=cfg.VAE_LATENT_CHANNELS,
    hidden_dim=cfg.MODEL_WIDTH,
    num_layers=cfg.NUM_LAYERS,
    num_heads=cfg.NUM_HEADS,
    predict_type=cfg.PREDICT_TYPE,
    laplace_k=cfg.LAPLACE_K,
    global_k=cfg.GLOBAL_K,
    timesteps=cfg.TIMESTEPS,
    schedule=cfg.SCHEDULE,
    dropout=cfg.DROPOUT,
    attn_dropout=cfg.ATTN_DROPOUT,
    self_conditioning=cfg.SELF_COND,
    context_dim=Fv,
    num_entities=N0,
    context_len=cfg.CONTEXT_LEN,
    lap_mode_main=cfg.LAP_MODE_main,
    lap_mode_cond=cfg.LAP_MODE_cond,
    zero_first_step=cfg.zero_first_step,
    add_guidance_tokens=cfg.add_guidance_tokens,
    summery_mode="EFF",
).to(device)


v_variance = calculate_v_variance(
    scheduler=diff_model.scheduler,
    dataloader=val_dl,
    vae=vae,
    device=device,
    latent_stats=(mu_mean, mu_std),
)
print(f"Calculated V-Prediction target variance: {v_variance:.4f}")
print("=========================================================")


ema = EMA(diff_model, decay=cfg.EMA_DECAY) if cfg.USE_EMA_EVAL else None

scheduler = diff_model.scheduler
optimizer = torch.optim.AdamW(
    diff_model.parameters(), lr=cfg.BASE_LR, weight_decay=cfg.WEIGHT_DECAY
)
lr_sched = make_warmup_cosine(
    optimizer,
    cfg.EPOCHS * max(1, len(train_dl)),
    warmup_frac=cfg.WARMUP_FRAC,
    base_lr=cfg.BASE_LR,
    min_lr=cfg.MIN_LR,
)
scaler = GradScaler(enabled=(device.type == "cuda"))

context_grad_checked = False


def train_one_epoch(epoch: int) -> float:
    global context_grad_checked
    diff_model.train()
    running_loss, num_samples = 0.0, 0

    for xb, yb, meta in train_dl:
        V, T = xb
        mask_bn = meta["entity_mask"]

        cond_summary = build_context(
            diff_model, V, T, mask_bn, device, requires_grad=True
        )
        y_in, batch_ids = flatten_targets(yb, mask_bn, device)
        if y_in is None:
            continue

        cond_summary_flat = cond_summary[batch_ids]
        mu_norm = encode_mu_norm(vae, y_in, mu_mean=mu_mean, mu_std=mu_std)

        Beff = mu_norm.size(0)
        p_drop = float(cfg.DROP_COND_P)
        m_cond = torch.rand(Beff, device=device) >= p_drop
        idx_c = m_cond.nonzero(as_tuple=False).squeeze(1)
        idx_u = (~m_cond).nonzero(as_tuple=False).squeeze(1)

        t = sample_t_uniform(scheduler, Beff, device)
        noise = torch.randn_like(mu_norm)
        x_t, eps_true = scheduler.q_sample(mu_norm, t, noise)

        optimizer.zero_grad(set_to_none=True)
        loss = 0.0

        with autocast(enabled=(device.type == "cuda")):
            if idx_c.numel() > 0:
                loss_c = diffusion_loss(
                    diff_model,
                    scheduler,
                    mu_norm[idx_c],
                    t[idx_c],
                    cond_summary=cond_summary_flat[idx_c],
                    predict_type=cfg.PREDICT_TYPE,
                    weight_scheme=cfg.LOSS_WEIGHT_SCHEME,
                    minsnr_gamma=cfg.MINSNR_GAMMA,
                    reuse_xt_eps=(x_t[idx_c], eps_true[idx_c]),
                )
                loss = loss + loss_c * (idx_c.numel() / Beff)

            if idx_u.numel() > 0:
                loss_u = diffusion_loss(
                    diff_model,
                    scheduler,
                    mu_norm[idx_u],
                    t[idx_u],
                    cond_summary=None,
                    predict_type=cfg.PREDICT_TYPE,
                    weight_scheme=cfg.LOSS_WEIGHT_SCHEME,
                    minsnr_gamma=cfg.MINSNR_GAMMA,
                    reuse_xt_eps=(x_t[idx_u], eps_true[idx_u]),
                )
                loss = loss + loss_u * (idx_u.numel() / Beff)

        if not torch.isfinite(loss):
            print("[warn] non-finite loss detected; skipping step")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if idx_c.numel() > 0 and not context_grad_checked:
            has_context_grad = any(
                p.grad is not None for p in diff_model.context.parameters()
            )
            if not has_context_grad:
                raise RuntimeError(
                    "Context parameters did not receive gradients during training."
                )
            context_grad_checked = True

        if getattr(cfg, "GRAD_CLIP", 0.0) and cfg.GRAD_CLIP > 0:
            nn.utils.clip_grad_norm_(diff_model.parameters(), cfg.GRAD_CLIP)

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

        cond_summary = build_context(
            diff_model, V, T, mask_bn, device, requires_grad=False
        )
        y_in, batch_ids = flatten_targets(yb, mask_bn, device)
        if y_in is None:
            continue
        cond_summary_flat = cond_summary[batch_ids]
        mu_norm = encode_mu_norm(vae, y_in, mu_mean=mu_mean, mu_std=mu_std)

        t = sample_t_uniform(scheduler, mu_norm.size(0), device)
        loss = diffusion_loss(
            diff_model,
            scheduler,
            mu_norm,
            t,
            cond_summary=cond_summary_flat,
            predict_type=cfg.PREDICT_TYPE,
        )
        total += loss.item() * mu_norm.size(0)
        count += mu_norm.size(0)

        probe_n = min(128, mu_norm.size(0))
        if probe_n > 0:
            mu_p = mu_norm[:probe_n]
            cs_p = cond_summary_flat[:probe_n]
            t_p = sample_t_uniform(scheduler, probe_n, device)
            noise_p = torch.randn_like(mu_p)
            x_t_p, eps_p = scheduler.q_sample(mu_p, t_p, noise_p)

            loss_cond = diffusion_loss(
                diff_model,
                scheduler,
                mu_p,
                t_p,
                cond_summary=cs_p,
                predict_type=cfg.PREDICT_TYPE,
                reuse_xt_eps=(x_t_p, eps_p),
            ).item()

            loss_unco = diffusion_loss(
                diff_model,
                scheduler,
                mu_p,
                t_p,
                cond_summary=None,
                predict_type=cfg.PREDICT_TYPE,
                reuse_xt_eps=(x_t_p, eps_p),
            ).item()

            cond_gap_accum += (loss_unco - loss_cond)
            cond_gap_batches += 1

    if ema is not None:
        ema.restore(diff_model)

    avg_val = total / max(1, count)
    cond_gap = (cond_gap_accum / cond_gap_batches) if cond_gap_batches > 0 else float("nan")
    return avg_val, cond_gap


skip_with_trained_model = cfg.TRAINED_LLapDiT

if skip_with_trained_model and os.path.exists(skip_with_trained_model) and cfg.downstream:
    print(f"Skipping training. Loading model from: {skip_with_trained_model}")
    ckpt = torch.load(skip_with_trained_model, map_location=device)
    diff_model.load_state_dict(ckpt["model_state"])
    print("Loaded model state.")
    if ema is not None and "ema_state" in ckpt:
        ema.load_state_dict(ckpt["ema_state"])
        print("Loaded EMA state.")
    mu_mean = ckpt["mu_mean"].to(device)
    mu_std = ckpt["mu_std"].to(device)
else:
    if skip_with_trained_model:
        lr_sched = make_warmup_cosine(
            optimizer,
            cfg.EPOCHS * max(1, len(train_dl)),
            warmup_frac=cfg.FT_WARMUP_FRAC,
            base_lr=cfg.FT_BASE_LR,
            min_lr=cfg.FT_MIN_LR,
        )
        ckpt = torch.load(skip_with_trained_model, map_location=device)
        diff_model.load_state_dict(ckpt["model_state"])
        print("Loaded model state.")
        if ema is not None and "ema_state" in ckpt:
            ema.load_state_dict(ckpt["ema_state"])
            print("Loaded EMA state.")
    else:
        print(
            f"Model path not found or downstream disabled: {skip_with_trained_model}. Starting training from scratch."
        )

    best_val = float("inf")
    patience = 0
    current_best_path = None
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)

    for epoch in tqdm(range(1, cfg.EPOCHS + 1), desc="Epochs"):
        train_loss = train_one_epoch(epoch)
        val_loss, cond_gap = validate()
        Z = cfg.VAE_LATENT_CHANNELS
        print(
            f"Epoch {epoch:03d} | train: {train_loss:.6f} | val: {val_loss:.6f} | cond_gap: {cond_gap * Z:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            if current_best_path and os.path.exists(current_best_path):
                os.remove(current_best_path)

            ckpt_path = os.path.join(
                cfg.CKPT_DIR,
                f"dataset-{cfg.DATASET_NAME}_pred-{cfg.PRED}_ch-{cfg.VAE_LATENT_CHANNELS}_val_{val_loss:.6f}_cond_{cond_gap * Z:.6f}.pt",
            )
            save_payload = {
                "epoch": epoch,
                "model_state": diff_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "mu_mean": mu_mean.detach().cpu(),
                "mu_std": mu_std.detach().cpu(),
                "predict_type": cfg.PREDICT_TYPE,
                "timesteps": cfg.TIMESTEPS,
                "schedule": cfg.SCHEDULE,
            }
            if ema is not None:
                save_payload["ema_state"] = ema.state_dict()
                save_payload["ema_decay"] = cfg.EMA_DECAY
            torch.save(save_payload, ckpt_path)
            current_best_path = ckpt_path
        else:
            patience += 1
            if patience >= cfg.EARLY_STOP:
                print("Early stopping.")
                break
