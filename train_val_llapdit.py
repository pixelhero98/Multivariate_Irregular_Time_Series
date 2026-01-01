import os, torch, math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import crypto_config
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from Dataset.data_gen import run_experiment
from Latent_Space.latent_vae import LatentVAE
from Model.summarizer import LaplaceAE
from Model.llapdit import LLapDiT
from Model.llapdit_utils import (
    EMA,
    set_torch,
    encode_mu_norm,
    _flatten_for_mask,
    make_warmup_cosine,
    calculate_target_variance,
    compute_latent_stats,
    diffusion_loss,
    build_context,
    flatten_targets,
    sample_t_uniform,
    decode_latents_with_vae,
    plot_laplace_poles,
)

LoaderTuple = Tuple[DataLoader, DataLoader, DataLoader]


def _nan_to_num(x: torch.Tensor) -> torch.Tensor:
    if torch.isfinite(x).all():
        return x
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _entity_finite_mask(x: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(x)
    for _ in range(x.dim() - 2):
        finite = finite.all(dim=-1)
    return finite


def _sanitize_batch(
    xb: Tuple[torch.Tensor, torch.Tensor],
    yb: torch.Tensor,
    mask_bn: torch.Tensor,
    device: torch.device,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    V, T = xb
    V = _nan_to_num(V).to(device)
    T = _nan_to_num(T).to(device)
    yb = _nan_to_num(yb).to(device)
    mask = mask_bn.to(device=device, dtype=torch.bool)
    mask = mask & _entity_finite_mask(V) & _entity_finite_mask(T) & _entity_finite_mask(yb)
    return (V, T), yb, mask


def _ensure_loaders(
    train_dl: Optional[DataLoader],
    val_dl: Optional[DataLoader],
    test_dl: Optional[DataLoader],
    sizes: Optional[Sequence[int]],
    config=crypto_config,
) -> Tuple[LoaderTuple, Optional[Tuple[int, int, int]]]:
    if any(loader is None for loader in (train_dl, val_dl, test_dl)):
        train_dl, val_dl, test_dl, sizes = run_experiment(
            data_dir=config.DATA_DIR,
            date_batching=config.date_batching,
            dates_per_batch=config.BATCH_SIZE,
            K=config.WINDOW,
            H=config.PRED,
            coverage=config.COVERAGE
        )
    elif sizes is None:
        try:
            sizes = tuple(len(dl.dataset) for dl in (train_dl, val_dl, test_dl))
        except Exception:
            sizes = None

    if train_dl is None or val_dl is None or test_dl is None:
        raise RuntimeError("Failed to obtain train/val/test dataloaders.")

    return (train_dl, val_dl, test_dl), sizes


def _summarize_dataset(train_dl: DataLoader, sizes: Optional[Sequence[int]]) -> Tuple[int, int, int, int]:
    if sizes is not None:
        print("sizes:", tuple(sizes))
    else:
        print("sizes: (unknown)")

    try:
        xb0, yb0, meta0 = next(iter(train_dl))
    except StopIteration as exc:  # pragma: no cover - defensive
        raise RuntimeError("Training dataloader produced no batches.") from exc

    V0, T0 = xb0
    B0, N0, K0, Fv = V0.shape
    Ft = T0.shape[-1]
    H = yb0.shape[-1]
    assert Fv == Ft, f"Expected Fv == Ft, got {Fv} vs {Ft}"
    print("V:", V0.shape, "T:", T0.shape, "y:", yb0.shape)
    return B0, N0, K0, Fv


def run(
    train_dl: Optional[DataLoader] = None,
    val_dl: Optional[DataLoader] = None,
    test_dl: Optional[DataLoader] = None,
    sizes: Optional[Sequence[int]] = None,
    config=crypto_config,
) -> Dict[str, object]:
    plot_poles_only = bool(getattr(config, "POLE_PLOT_ONLY", False))
    (train_dl, val_dl, test_dl), sizes = _ensure_loaders(train_dl, val_dl, test_dl, sizes, config)
    device = set_torch()

    _, N0, _, Fv = _summarize_dataset(train_dl, sizes)

    # ---- Load pretrained Laplace summarizer (shared with standalone training) ----
    sum_lap_k = config.SUM_LAPLACE_K
    sum_ctx_len = config.SUM_CONTEXT_LEN
    sum_ctx_dim = config.SUM_CONTEXT_DIM
    sum_tv_hidden = config.SUM_TV_HIDDEN
    sum_dropout = config.SUM_DROPOUT

    summarizer_ckpt = config.SUM_CKPT
    if summarizer_ckpt is None:
        summarizer_dir = config.SUM_DIR
        summarizer_ckpt = Path(summarizer_dir) / f"{config.PRED}-{config.VAE_LATENT_CHANNELS}-summarizer.pt"
    else:
        summarizer_ckpt = Path(summarizer_ckpt)

    laplace_summarizer = LaplaceAE(
        num_entities=N0,
        feat_dim=Fv,
        window_size=config.WINDOW,
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
        seq_len=config.PRED,
        latent_dim=config.VAE_LATENT_DIM,
        latent_channel=config.VAE_LATENT_CHANNELS,
        enc_layers=config.VAE_LAYERS, enc_heads=config.VAE_HEADS, enc_ff=config.VAE_FF,
        dec_layers=config.VAE_LAYERS, dec_heads=config.VAE_HEADS, dec_ff=config.VAE_FF
    ).to(device)

    if os.path.isfile(config.VAE_CKPT):
        ckpt = torch.load(config.VAE_CKPT, map_location=device)
        sd = ckpt.get("state_dict", ckpt)
        vae.load_state_dict(sd)
        print("Loaded VAE checkpoint:", config.VAE_CKPT)
    else:
        print(f"[warn] VAE checkpoint not found at {config.VAE_CKPT}; using randomly initialised weights.")

    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    mu_mean, mu_std = compute_latent_stats(vae, train_dl, device)

    # ---- Conditional diffusion model ----
    diff_model = LLapDiT(
        data_dim=config.VAE_LATENT_CHANNELS, hidden_dim=config.MODEL_WIDTH,
        num_layers=config.NUM_LAYERS, num_heads=config.NUM_HEADS,
        predict_type=config.PREDICT_TYPE, laplace_k=config.LAPLACE_K,
        timesteps=config.TIMESTEPS, schedule=config.SCHEDULE,
        dropout=config.DROPOUT, attn_dropout=config.ATTN_DROPOUT,
        self_conditioning=config.SELF_COND,
        lap_mode_main=config.LAP_MODE,
    ).to(device)

    # ---- Calculate the variance of the chosen prediction target ----
    baseline_target_variance = calculate_target_variance(
        predict_type=config.PREDICT_TYPE,
        scheduler=diff_model.scheduler,
        dataloader=val_dl,
        vae=vae,
        device=device,
        latent_stats=(mu_mean, mu_std),
        latent_encoder=lambda y_in: encode_mu_norm(
            vae,
            y_in,
            mu_mean=mu_mean,
            mu_std=mu_std,
        ),
    )
    print(f"Calculated target variance ({config.PREDICT_TYPE}-prediction): {baseline_target_variance:.4f}")
    print("=========================================================")

    ema = EMA(diff_model, decay=config.EMA_DECAY) if config.USE_EMA_EVAL else None

    scheduler = diff_model.scheduler
    optimizer = torch.optim.AdamW(diff_model.parameters(),
                                  lr=config.BASE_LR,
                                  weight_decay=config.WEIGHT_DECAY)
    lr_sched = make_warmup_cosine(optimizer, config.EPOCHS * max(1, len(train_dl)),
                                  warmup_frac=config.WARMUP_FRAC,
                                  base_lr=config.BASE_LR,
                                  min_lr=config.MIN_LR)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    pole_plot_dir = Path(getattr(config, "POLE_PLOT_DIR", Path(config.OUT_DIR) / "pole_plots"))
    if getattr(config, "SAVE_POLE_PLOTS", True):
        pole_plot_dir.mkdir(parents=True, exist_ok=True)

    def export_pole_plot(stage: str):
        if not getattr(config, "SAVE_POLE_PLOTS", True):
            return None
        use_ema = ema is not None and getattr(config, "USE_EMA_EVAL", False)
        if use_ema:
            ema.store(diff_model)
            ema.copy_to(diff_model)
        try:
            title = f"LLapDiT denoising poles ({stage}, pred={config.PRED})"
            fname = f"llapdit_poles_pred{config.PRED}_{stage}.pdf"
            return plot_laplace_poles(
                [diff_model.model],
                pole_plot_dir / fname,
                title=title,
                tag_prefix=f"{stage}-pred{config.PRED}-",
                prediction_length=config.PRED,
            )
        finally:
            if use_ema:
                ema.restore(diff_model)

    # ============================ train/val loops ============================
    def train_one_epoch(epoch: int) -> float:
        diff_model.train()
        running_loss = 0.0
        num_samples = 0

        for xb, yb, meta in train_dl:
            (V, T), yb, mask_bn = _sanitize_batch(xb, yb, meta["entity_mask"], device)

            if not mask_bn.any():
                continue

            cond_summary = build_context(laplace_summarizer, V, T, mask_bn, device)
            y_in, batch_ids = flatten_targets(yb, mask_bn, device)
            if y_in is None:
                continue

            cond_summary_flat = cond_summary[batch_ids]
            mu_norm = encode_mu_norm(
                vae, y_in,
                mu_mean=mu_mean, mu_std=mu_std
            )

            Beff = mu_norm.size(0)
            p_drop = float(config.DROP_COND_P)
            m_cond = (torch.rand(Beff, device=device) >= p_drop)
            idx_c = m_cond.nonzero(as_tuple=False).squeeze(1)
            idx_u = (~m_cond).nonzero(as_tuple=False).squeeze(1)

            t = sample_t_uniform(scheduler, Beff, device)
            noise = torch.randn_like(mu_norm)
            x_t, eps_true = scheduler.q_sample(mu_norm, t, noise)

            sc_feat_c = sc_feat_u = None
            use_sc = (epoch >= config.SELF_COND_START_EPOCH and torch.rand(()) < config.SELF_COND_P) and config.SELF_COND
            if use_sc:
                with torch.no_grad():
                    if idx_c.numel() > 0:
                        pred_ng_c = diff_model(x_t[idx_c], t[idx_c], cond_summary=cond_summary_flat[idx_c], sc_feat=None)
                        sc_feat_c = scheduler.to_x0(x_t[idx_c], t[idx_c], pred_ng_c, config.PREDICT_TYPE).detach()
                    if idx_u.numel() > 0:
                        pred_ng_u = diff_model(x_t[idx_u], t[idx_u], cond_summary=None, sc_feat=None)
                        sc_feat_u = scheduler.to_x0(x_t[idx_u], t[idx_u], pred_ng_u, config.PREDICT_TYPE).detach()

            optimizer.zero_grad(set_to_none=True)
            loss = 0.0

            with autocast(enabled=(device.type == "cuda")):
                if idx_c.numel() > 0:
                    loss_c = diffusion_loss(
                        diff_model, scheduler, mu_norm[idx_c], t[idx_c],
                        cond_summary=cond_summary_flat[idx_c], predict_type=config.PREDICT_TYPE,
                        weight_scheme=config.LOSS_WEIGHT_SCHEME,
                        minsnr_gamma=config.MINSNR_GAMMA,
                        sc_feat=sc_feat_c,
                        reuse_xt_eps=(x_t[idx_c], eps_true[idx_c]),
                    )
                    loss = loss + loss_c * (idx_c.numel() / Beff)

                if idx_u.numel() > 0:
                    loss_u = diffusion_loss(
                        diff_model, scheduler, mu_norm[idx_u], t[idx_u],
                        cond_summary=None, predict_type=config.PREDICT_TYPE,
                        weight_scheme=config.LOSS_WEIGHT_SCHEME,
                        minsnr_gamma=config.MINSNR_GAMMA,
                        sc_feat=sc_feat_u,
                        reuse_xt_eps=(x_t[idx_u], eps_true[idx_u]),
                    )
                    loss = loss + loss_u * (idx_u.numel() / Beff)

            if not torch.isfinite(loss):
                print("[warn] non-finite loss detected; skipping step")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if config.GRAD_CLIP and config.GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(diff_model.parameters(), config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(diff_model)
            lr_sched.step()

            running_loss += float(loss.item()) * Beff
            num_samples += Beff

        return running_loss / max(1, num_samples)
    
    
    @torch.inference_mode()
    def validate() -> Tuple[float, float]:
        diff_model.eval()
        total, count = 0.0, 0
        cond_gap_accum, cond_gap_batches = 0.0, 0
    
        if ema is not None:
            ema.store(diff_model)
            ema.copy_to(diff_model)
    
        for xb, yb, meta in val_dl:
            (V, T), yb, mask_bn = _sanitize_batch(xb, yb, meta["entity_mask"], device)
    
            if not mask_bn.any():
                continue
    
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
                cond_summary=cond_summary_flat, predict_type=config.PREDICT_TYPE
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
                    cond_summary=cs_p, predict_type=config.PREDICT_TYPE,
                    reuse_xt_eps=(x_t_p, eps_p)
                ).item()
    
                loss_unco = diffusion_loss(
                    diff_model, scheduler, mu_p, t_p,
                    cond_summary=None, predict_type=config.PREDICT_TYPE,
                    reuse_xt_eps=(x_t_p, eps_p)
                ).item()
    
                cond_gap_accum += (loss_unco - loss_cond)
                cond_gap_batches += 1
    
        if ema is not None:
            ema.restore(diff_model)
    
        avg_val = total / max(1, count)
        cond_gap = (cond_gap_accum / cond_gap_batches) if cond_gap_batches > 0 else float("nan")
        return avg_val, cond_gap
    
    
    # ============================ training control ============================
    skip_with_trained_model = config.TRAINED_LLapDiT
    loaded_checkpoint: Optional[str] = None
    best_checkpoint: Optional[str] = None
    best_val = float('inf')
    patience = 0

    skip_available = skip_with_trained_model and os.path.exists(skip_with_trained_model)
    if plot_poles_only and not skip_available:
        raise FileNotFoundError(
            "POLE_PLOT_ONLY is enabled but TRAINED_LLapDiT checkpoint was not found."
        )

    if skip_available and (config.downstream or plot_poles_only):
        print(f"Loading pretrained LLapDiT from: {skip_with_trained_model}")
        ckpt = torch.load(skip_with_trained_model, map_location=device)
        diff_model.load_state_dict(ckpt['model_state'])
        if ema is not None and 'ema_state' in ckpt:
            ema.load_state_dict(ckpt['ema_state'])
        mu_mean = ckpt['mu_mean'].to(device)
        mu_std = ckpt['mu_std'].to(device)
        loaded_checkpoint = skip_with_trained_model
    else:
        os.makedirs(config.CKPT_DIR, exist_ok=True)
        for epoch in tqdm(range(1, config.EPOCHS + 1), desc='Epochs'):
            train_loss = train_one_epoch(epoch)
            val_loss, cond_gap = validate()
            Z = config.VAE_LATENT_CHANNELS
            val_vs_baseline = (
                val_loss / baseline_target_variance
                if baseline_target_variance and baseline_target_variance > 0
                else float('inf')
            )
            improvement = (
                baseline_target_variance - val_loss
                if baseline_target_variance is not None
                else float('nan')
            )
            cond_gap_scaled = cond_gap * Z if math.isfinite(cond_gap) else float('nan')
            print(
                f"Epoch {epoch:03d} | train: {train_loss:.6f} "
                f"| val: {val_loss:.6f} | cond_gap: {cond_gap_scaled:.6f} "
                f"| val/v_var: {val_vs_baseline:.6f} | improvement: {improvement:.6f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                if best_checkpoint and os.path.exists(best_checkpoint):
                    os.remove(best_checkpoint)

                ratio_str = (
                    f"{val_vs_baseline:.6f}"
                    if math.isfinite(val_vs_baseline)
                    else ('inf' if val_vs_baseline > 0 else 'nan')
                )
                ckpt_path = os.path.join(
                    config.CKPT_DIR,
                    f"mode-{config.LAP_MODE}-pred-{config.PRED}"
                    f"-val-{val_loss:.6f}-cond-{cond_gap * Z:.6f}-ratio-{ratio_str}.pt"
                )
                save_payload = {
                    'epoch': epoch,
                    'model_state': diff_model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'mu_mean': mu_mean.detach().cpu(),
                    'mu_std': mu_std.detach().cpu(),
                    'predict_type': config.PREDICT_TYPE,
                    'timesteps': config.TIMESTEPS,
                    'schedule': config.SCHEDULE,
                    'val_loss': float(val_loss),
                }
                if ema is not None:
                    save_payload['ema_state'] = ema.state_dict()
                    save_payload['ema_decay'] = config.EMA_DECAY
                torch.save(save_payload, ckpt_path)
                best_checkpoint = ckpt_path
            else:
                patience += 1
                if patience >= config.EARLY_STOP:
                    print('Early stopping.')
                    break

        if best_checkpoint is not None:
            print(f"Loading best checkpoint: {best_checkpoint}")
            ckpt = torch.load(best_checkpoint, map_location=device)
            diff_model.load_state_dict(ckpt['model_state'])
            if ema is not None and 'ema_state' in ckpt:
                ema.load_state_dict(ckpt['ema_state'])
            mu_mean = ckpt['mu_mean'].to(device)
            mu_std = ckpt['mu_std'].to(device)
            loaded_checkpoint = best_checkpoint

    if plot_poles_only:
        export_pole_plot("pretrained")
        return {
            'train_loader': train_dl,
            'val_loader': val_dl,
            'test_loader': test_dl,
            'sizes': sizes,
            'baseline_target_variance': baseline_target_variance,
            'best_val': None,
            'best_checkpoint': best_checkpoint,
            'loaded_checkpoint': loaded_checkpoint,
            'eval_stats': None,
        }

    if config.DECODER_FT_EPOCHS > 0:
        torch.manual_seed(42)
        finetune_vae_decoder(
            vae,
            diff_model,
            laplace_summarizer,
            train_dl,
            val_dl,
            device,
            mu_mean=mu_mean,
            mu_std=mu_std,
            ema=ema,
            epochs=config.DECODER_FT_EPOCHS,
            lr=config.DECODER_FT_LR,
            gen_steps=config.GEN_STEPS,
            guidance_strength=config.GUIDANCE_STRENGTH,
            guidance_power=config.GUIDANCE_POWER,
            lambda_rec_anchor=config.DECODER_FT_ANCHOR,
            self_cond=config.SELF_COND,
        )

    eval_stats = evaluate_regression(
        diff_model,
        vae,
        laplace_summarizer,
        test_dl,
        device,
        mu_mean,
        mu_std,
        config=config,
        ema=ema,
        self_cond=config.SELF_COND,
        steps=config.GEN_STEPS,
        guidance_strength=config.GUIDANCE_STRENGTH,
        guidance_power=config.GUIDANCE_POWER,
        aggregation_method='mean',
        quantiles=(0.1, 0.5, 0.9),
        dynamic_thresh_p=0.995, 
        dynamic_thresh_max=1.0
    )

    if getattr(config, "SAVE_POLE_PLOTS", True):
        export_pole_plot("val")
        export_pole_plot("test")

    return {
        'train_loader': train_dl,
        'val_loader': val_dl,
        'test_loader': test_dl,
        'sizes': sizes,
        'baseline_target_variance': baseline_target_variance,
        'best_val': best_val if math.isfinite(best_val) else None,
        'best_checkpoint': best_checkpoint,
        'loaded_checkpoint': loaded_checkpoint,
        'eval_stats': eval_stats,
    }


def finetune_vae_decoder(
        vae,
        diff_model,
        summarizer,
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
        self_cond: bool = False,
        dynamic_thresh_p: float = 0.0,
        dynamic_thresh_max: float = 1.0
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
            (V, T), yb, mask_bn = _sanitize_batch(xb, yb, meta["entity_mask"], device)
            if not mask_bn.any():
                continue

            cs_full = build_context(summarizer, V, T, mask_bn, device)
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
                    cond_summary=cs, self_cond=self_cond, cfg_rescale=True,
                    dynamic_thresh_p, dynamic_thresh_max
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
        diff_model, vae, summarizer, dataloader, device, mu_mean, mu_std, config, ema=None,
        self_cond: bool = False,
        steps: int = 36, guidance_strength: float = 2.0, guidance_power: float = 0.3,
        aggregation_method: str = 'mean',
        quantiles: tuple = (0.1, 0.5, 0.9),
        dynamic_thresh_p: float = 0.995,
        dynamic_thresh_max: float = 1.0,
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

    for xb, yb, meta in dataloader:
        (V, T), yb, mask_bn = _sanitize_batch(xb, yb, meta["entity_mask"], device)
        if not mask_bn.any():
            continue

        cond_summary = build_context(summarizer, V, T, mask_bn, device)
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
                cond_summary=cs, self_cond=self_cond, cfg_rescale=True,
                dynamic_thresh_p, dynamic_thresh_max
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


if __name__ == "__main__":  # pragma: no cover
    run()
