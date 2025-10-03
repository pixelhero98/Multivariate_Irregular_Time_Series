"""Unified entrypoint to train the latent VAE, summarizer, and LLapDiT."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import crypto_config
from Dataset.fin_dataset import run_experiment
import train_val_latent
import train_val_summarizer
import train_val_llapdit
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


def _fmt_optional(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    if value is None:
        return "None"
    return str(value)


def _summarizer_ckpt_path(config=crypto_config) -> Path:
    ckpt = config.SUM_CKPT
    if ckpt:
        return Path(ckpt)
    return Path(config.SUM_DIR) / f"{config.PRED}-{config.VAE_LATENT_CHANNELS}-summarizer.pt"


def main() -> None:
    print("Preparing data loaders...")
    train_loader, val_loader, test_loader, sizes = prepare_dataloaders()

    print("\n=== Training VAE ===")
    vae_ckpt = Path(crypto_config.VAE_CKPT)
    if vae_ckpt.exists():
        print(f"Found existing VAE checkpoint at {vae_ckpt}; skipping training.")
        vae_stats: Dict[str, object] = {
            "loaded_checkpoint": str(vae_ckpt),
            "best_elbo_path": str(vae_ckpt),
            "best_recon_path": None,
        }
    else:
        vae_stats = train_val_latent.run(
            train_dl=train_loader,
            val_dl=val_loader,
            test_dl=test_loader,
            sizes=sizes,
            config=crypto_config,
        )
    print(
        "VAE checkpoints:\n"
        f"  loaded -> {vae_stats.get('loaded_checkpoint')}\n"
        f"  best β·ELBO -> {vae_stats.get('best_elbo_path')}\n"
        f"  best recon -> {vae_stats.get('best_recon_path')}"
    )

    print("\n=== Training Summarizer ===")
    summarizer_ckpt = _summarizer_ckpt_path(crypto_config)
    if summarizer_ckpt.exists():
        print(f"Found existing summarizer checkpoint at {summarizer_ckpt}; skipping training.")
        summarizer_stats: Dict[str, object] = {
            "best_val": None,
            "test_loss": None,
            "checkpoint": str(summarizer_ckpt),
        }
    else:
        summarizer_stats = train_val_summarizer.run(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            sizes=sizes,
            config=crypto_config,
        )
    print(
        "Summarizer results:\n"
        f"  best val loss -> {_fmt_optional(summarizer_stats.get('best_val'))}\n"
        f"  test loss -> {_fmt_optional(summarizer_stats.get('test_loss'))}\n"
        f"  checkpoint -> {summarizer_stats.get('checkpoint')}"
    )

    print("\n=== Training LLapDiT ===")
    llapdit_stats = train_val_llapdit.run(
        train_dl=train_loader,
        val_dl=val_loader,
        test_dl=test_loader,
        sizes=sizes,
        config=crypto_config,
    )
    eval_stats = llapdit_stats.get("eval_stats") or {}
    print(
        "LLapDiT results:\n"
        f"  baseline v-var -> {_fmt_optional(llapdit_stats.get('baseline_v_variance'))}\n"
        f"  best val loss -> {_fmt_optional(llapdit_stats.get('best_val'))}\n"
        f"  checkpoint -> {llapdit_stats.get('best_checkpoint') or llapdit_stats.get('loaded_checkpoint')}\n"
        f"  eval CRPS -> {_fmt_optional(eval_stats.get('crps'))}\n"
        f"  eval MAE -> {_fmt_optional(eval_stats.get('mae'))}\n"
        f"  eval MSE -> {_fmt_optional(eval_stats.get('mse'))}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
