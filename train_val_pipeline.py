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


def _update_config_for_pred(pred: int, config=crypto_config) -> None:
    """Mutate the shared configuration to reflect a new prediction horizon."""

    config.PRED = pred
    config.SUM_CONTEXT_LEN = pred
    config.CONTEXT_LEN = pred
    config.SUM_CKPT = str(Path(config.SUM_DIR) / f"{pred}-{config.VAE_LATENT_CHANNELS}-summarizer.pt")
    config.VAE_CKPT = str(
        Path(config.VAE_DIR)
        / f"pred-{pred}_ch-{config.VAE_LATENT_CHANNELS}_elbo.pt"
    )


def main() -> None:
    preds = (5, 20, 60, 100)
    log_path = Path("pred_performance_log.txt")
    print(f"Logging downstream performance to {log_path.resolve()}")

    with log_path.open("w", encoding="utf-8") as log_file:
        for pred in preds:
            print("\n========================================")
            print(f"Running training pipeline for PRED={pred}")
            log_file.write(f"PRED={pred}\n")

            _update_config_for_pred(pred)

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
                print(
                    f"Found existing summarizer checkpoint at {summarizer_ckpt}; "
                    "skipping training."
                )
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
            pinball_stats = eval_stats.get("pinball") if isinstance(eval_stats, dict) else None
            if isinstance(pinball_stats, dict) and pinball_stats:
                pinball_fmt = ", ".join(
                    f"q{float(q):.2f}: {_fmt_optional(val)}" for q, val in sorted(pinball_stats.items())
                )
            else:
                pinball_fmt = "None"

            print(
                "LLapDiT results:\n"
                f"  baseline v-var -> {_fmt_optional(llapdit_stats.get('baseline_v_variance'))}\n"
                f"  best val loss -> {_fmt_optional(llapdit_stats.get('best_val'))}\n"
                f"  checkpoint -> {llapdit_stats.get('best_checkpoint') or llapdit_stats.get('loaded_checkpoint')}\n"
                f"  eval CRPS -> {_fmt_optional(eval_stats.get('crps'))}\n"
                f"  eval MAE -> {_fmt_optional(eval_stats.get('mae'))}\n"
                f"  eval MSE -> {_fmt_optional(eval_stats.get('mse'))}\n"
                f"  eval Pinball -> {pinball_fmt}"
            )

            log_file.write(
                f"  baseline_v_variance={_fmt_optional(llapdit_stats.get('baseline_v_variance'))}\n"
            )
            log_file.write(f"  best_val={_fmt_optional(llapdit_stats.get('best_val'))}\n")
            log_file.write(
                f"  checkpoint={llapdit_stats.get('best_checkpoint') or llapdit_stats.get('loaded_checkpoint')}\n"
            )
            log_file.write(f"  eval_crps={_fmt_optional(eval_stats.get('crps'))}\n")
            log_file.write(f"  eval_mae={_fmt_optional(eval_stats.get('mae'))}\n")
            log_file.write(f"  eval_mse={_fmt_optional(eval_stats.get('mse'))}\n")

            if isinstance(pinball_stats, dict) and pinball_stats:
                for quantile, loss in sorted(pinball_stats.items()):
                    log_file.write(
                        f"  eval_pinball_q{float(quantile):.2f}={_fmt_optional(loss)}\n"
                    )
            else:
                log_file.write("  eval_pinball=None\n")
            log_file.write("\n")
            log_file.flush()


if __name__ == "__main__":  # pragma: no cover
    main()
