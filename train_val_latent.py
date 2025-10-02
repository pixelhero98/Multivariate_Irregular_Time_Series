"""Training routine for the latent VAE used in the financial pipeline."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import crypto_config
from Dataset.fin_dataset import run_experiment
from Latent_Space.latent_vae import LatentVAE
from Latent_Space.latent_vae_utils import normalize_and_check


LoaderTuple = Tuple[DataLoader, DataLoader, DataLoader]


def _ensure_loaders(
    train_dl: Optional[DataLoader],
    val_dl: Optional[DataLoader],
    test_dl: Optional[DataLoader],
    sizes: Optional[Sequence[int]],
    config=crypto_config,
) -> Tuple[LoaderTuple, Optional[Tuple[int, int, int]]]:
    """Return dataloaders, creating them if necessary, and infer dataset sizes."""

    if any(loader is None for loader in (train_dl, val_dl, test_dl)):
        train_dl, val_dl, test_dl, sizes = run_experiment(
            data_dir=config.DATA_DIR,
            date_batching=config.date_batching,
            dates_per_batch=config.BATCH_SIZE,
            K=config.WINDOW,
            H=config.PRED,
            coverage=config.COVERAGE,
        )
    elif sizes is None:
        try:
            sizes = tuple(len(dl.dataset) for dl in (train_dl, val_dl, test_dl))
        except Exception:
            sizes = None

    if train_dl is None or val_dl is None or test_dl is None:
        raise RuntimeError("Failed to obtain train/val/test dataloaders.")

    return (train_dl, val_dl, test_dl), sizes


def _log_dataset_summary(train_loader: DataLoader, sizes: Optional[Sequence[int]]) -> None:
    if sizes is not None:
        print(f"sizes: {tuple(sizes)}")
    else:
        print("sizes: (unknown)")

    try:
        (xb, yb, meta) = next(iter(train_loader))
    except StopIteration as exc:  # pragma: no cover - defensive
        raise RuntimeError("Training dataloader produced no batches.") from exc

    V, T = xb
    mask = meta["entity_mask"]
    print("V:", tuple(V.shape), "T:", tuple(T.shape), "y:", tuple(yb.shape))
    mask_float = mask.float()
    min_cov = float(mask_float.mean(1).min().item())
    frac_padded = float((~mask).float().mean().item())
    print(f"min coverage: {min_cov:.4f}")
    print(f"frac padded: {frac_padded:.4f}")


def _prepare_latent_batch(y: torch.Tensor, mask: torch.Tensor) -> Optional[torch.Tensor]:
    """Select valid entity trajectories according to ``mask``."""

    if mask.dtype != torch.bool:
        mask = mask.to(dtype=torch.bool)
    mask = mask.to(device=y.device)
    y = y.to(device=mask.device)

    batch_size, num_entities, horizon = y.shape
    y_flat = y.reshape(batch_size * num_entities, horizon)
    mask_flat = mask.reshape(batch_size * num_entities)
    if not torch.any(mask_flat):
        return None
    return y_flat[mask_flat].unsqueeze(-1)


def _epoch_pass(
    loader: Iterable,
    model: LatentVAE,
    device: torch.device,
    beta: float,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    grad_clip: float = 1.0,
    amp_enabled: bool = False,
) -> Dict[str, float]:
    """Run one epoch step (train or eval) and accumulate statistics."""

    is_train = optimizer is not None
    totals = {
        "recon_sum": 0.0,
        "recon_elems": 0,
        "kl_sum": 0.0,
        "kl_count": 0,
    }

    grad_ctx_factory = nullcontext if is_train else torch.no_grad

    for (_, yb, meta) in loader:
        y = yb.to(device)
        mask = meta["entity_mask"].to(device)
        y_in = _prepare_latent_batch(y, mask)
        if y_in is None:
            continue

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with grad_ctx_factory():
            with autocast(enabled=amp_enabled):
                y_hat, mu, logvar = model(y_in)
                recon_loss = F.mse_loss(y_hat, y_in, reduction="mean")
                kl_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                kl_per_sample = kl_elem.sum(dim=-1)
                kl_loss = kl_per_sample.mean()
                loss = recon_loss + beta * kl_loss

        if is_train:
            if scaler is None:
                raise ValueError("GradScaler must be provided when training.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

        totals["recon_sum"] += recon_loss.item() * y_in.numel()
        totals["recon_elems"] += y_in.numel()
        totals["kl_sum"] += kl_per_sample.sum().item()
        totals["kl_count"] += kl_per_sample.numel()

    return totals


def _aggregate_metrics(totals: Dict[str, float]) -> Tuple[float, float]:
    recon = totals["recon_sum"] / max(1, totals["recon_elems"])
    kl = totals["kl_sum"] / max(1, totals["kl_count"])
    return recon, kl


def run(
    train_dl: Optional[DataLoader] = None,
    val_dl: Optional[DataLoader] = None,
    test_dl: Optional[DataLoader] = None,
    sizes: Optional[Sequence[int]] = None,
    config=crypto_config,
) -> Dict[str, object]:
    (train_dl, val_dl, test_dl), sizes = _ensure_loaders(train_dl, val_dl, test_dl, sizes, config)
    _log_dataset_summary(train_dl, sizes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    print(f"Using device: {device}")
    grad_clip = getattr(config, "GRAD_CLIP", 1.0)

    model = LatentVAE(
        seq_len=config.PRED,
        latent_dim=config.VAE_LATENT_DIM,
        latent_channel=config.VAE_LATENT_CHANNELS,
        enc_layers=config.VAE_LAYERS,
        enc_heads=config.VAE_HEADS,
        enc_ff=config.VAE_FF,
        dec_layers=config.VAE_LAYERS,
        dec_heads=config.VAE_HEADS,
        dec_ff=config.VAE_FF,
        skip=False,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.VAE_LEARNING_RATE,
        weight_decay=config.VAE_WEIGHT_DECAY,
    )
    scaler = GradScaler(enabled=amp_enabled)

    vae_beta = config.VAE_BETA
    warmup_epochs = config.VAE_WARMUP_EPOCHS

    model_dir = Path(config.VAE_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_val_elbo = float("inf")
    best_val_recon = float("inf")
    best_elbo_path: Optional[Path] = None
    best_recon_path: Optional[Path] = None
    patience_counter = 0
    max_patience = config.VAE_MAX_PATIENCE

    print("Starting training.")
    for epoch in range(1, config.EPOCHS + 1):
        beta = 0.0 if epoch <= warmup_epochs else vae_beta

        train_totals = _epoch_pass(
            train_dl,
            model,
            device,
            beta,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=grad_clip,
            amp_enabled=amp_enabled,
        )
        val_totals = _epoch_pass(
            val_dl,
            model,
            device,
            vae_beta,
            amp_enabled=amp_enabled,
        )

        train_recon, train_kl = _aggregate_metrics(train_totals)
        val_recon, val_kl = _aggregate_metrics(val_totals)

        train_elbo_beta = train_recon + vae_beta * train_kl
        val_elbo_beta = val_recon + vae_beta * val_kl

        print(
            f"Epoch {epoch:03d}/{config.EPOCHS:03d} - β={beta:.3f} | "
            f"Train β·ELBO {train_elbo_beta:.6f} [Recon {train_recon:.6f}, KL/sample {train_kl:.6f}] | "
            f"Val β·ELBO {val_elbo_beta:.6f} [Recon {val_recon:.6f}, KL/sample {val_kl:.6f}]"
        )

        improved_elbo = val_elbo_beta < 0.95 * best_val_elbo
        improved_recon = val_recon < 0.95 * best_val_recon

        if improved_elbo:
            best_val_elbo = val_elbo_beta
            best_elbo_path = model_dir / f"pred-{config.PRED}_ch-{config.VAE_LATENT_CHANNELS}_elbo.pt"
            torch.save(model.state_dict(), best_elbo_path)
            print("  -> Saved best β·ELBO checkpoint")

        if improved_recon:
            best_val_recon = val_recon
            best_recon_path = model_dir / f"pred-{config.PRED}_ch-{config.VAE_LATENT_CHANNELS}_recon.pt"
            torch.save(model.state_dict(), best_recon_path)
            print("  -> Saved best reconstruction checkpoint")

        patience_counter = 0 if improved_elbo else (patience_counter + 1)
        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch}: β·ELBO hasn't improved in {max_patience} epochs.")
            break

    checkpoint_to_load: Optional[Path] = best_elbo_path or best_recon_path
    if checkpoint_to_load is not None:
        print(f"Loading checkpoint: {checkpoint_to_load}")
        vae = LatentVAE(
            seq_len=config.PRED,
            latent_dim=config.VAE_LATENT_DIM,
            latent_channel=config.VAE_LATENT_CHANNELS,
            enc_layers=config.VAE_LAYERS,
            enc_heads=config.VAE_HEADS,
            enc_ff=config.VAE_FF,
            dec_layers=config.VAE_LAYERS,
            dec_heads=config.VAE_HEADS,
            dec_ff=config.VAE_FF,
        ).to(device)
        vae.load_state_dict(torch.load(checkpoint_to_load, map_location=device))
    else:
        print("No improved checkpoints saved; using the final training state.")
        vae = model

    vae.eval()

    for param in vae.encoder.parameters():
        param.requires_grad = False

    all_mu = []
    with torch.no_grad():
        for (_, yb, meta) in train_dl:
            y = yb.to(device)
            mask = meta["entity_mask"].to(device)
            y_in = _prepare_latent_batch(y, mask)
            if y_in is None:
                continue
            _, mu, _ = vae(y_in)
            all_mu.append(mu.cpu())

    if all_mu:
        latents = torch.cat(all_mu, dim=0)
        normalize_and_check(latents, plot=True)
    else:
        print("No latent means collected (empty dataloader batches?).")

    return {
        "train_loader": train_dl,
        "val_loader": val_dl,
        "test_loader": test_dl,
        "sizes": sizes,
        "best_elbo_path": str(best_elbo_path) if best_elbo_path else None,
        "best_recon_path": str(best_recon_path) if best_recon_path else None,
        "loaded_checkpoint": str(checkpoint_to_load) if checkpoint_to_load else None,
        "model": vae,
    }


if __name__ == "__main__":  # pragma: no cover
    run()
