"""Unified entrypoint to train the latent VAE followed by the summarizer."""

from __future__ import annotations

from typing import Tuple

import crypto_config
from Dataset.fin_dataset import run_experiment
import train_val_latent
import train_val_summarizer
from torch.utils.data import DataLoader


def prepare_dataloaders(
    config=crypto_config,
) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple[int, int, int]]:
    """Build train/val/test loaders using the shared configuration."""

    return run_experiment(
        data_dir=config.DATA_DIR,
        date_batching=config.date_batching,
        dates_per_batch=config.BATCH_SIZE,
        K=config.WINDOW,
        H=config.PRED,
        coverage=config.COVERAGE,
    )


def main() -> None:
    print("Preparing data loaders...")
    train_loader, val_loader, test_loader, sizes = prepare_dataloaders()

    print("\n=== Training VAE ===")
    vae_stats = train_val_latent.run(
        train_dl=train_loader,
        val_dl=val_loader,
        test_dl=test_loader,
        sizes=sizes,
        config=crypto_config,
    )
    print(
        "VAE checkpoints:\n"
        f"  loaded -> {vae_stats['loaded_checkpoint']}\n"
        f"  best β·ELBO -> {vae_stats['best_elbo_path']}\n"
        f"  best recon -> {vae_stats['best_recon_path']}"
    )

    print("\n=== Training Summarizer ===")
    summarizer_stats = train_val_summarizer.run(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        sizes=sizes,
        config=crypto_config,
    )
    print(
        "Summarizer results:\n"
        f"  best val loss -> {summarizer_stats['best_val']:.6f}\n"
        f"  test loss -> {summarizer_stats['test_loss']:.6f}\n"
        f"  checkpoint -> {summarizer_stats['checkpoint']}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
