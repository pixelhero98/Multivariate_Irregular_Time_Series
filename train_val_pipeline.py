"""Unified training pipeline for the latent VAE and summarizer models."""

from Dataset.fin_dataset import run_experiment
import crypto_config
import train_val_latent
import train_val_summarizer


def prepare_dataloaders(config=crypto_config):
    """Build train/val/test loaders using the shared configuration."""
    return run_experiment(
        data_dir=config.DATA_DIR,
        date_batching=config.date_batching,
        dates_per_batch=config.BATCH_SIZE,
        K=config.WINDOW,
        H=config.PRED,
        coverage=config.COVERAGE,
    )


def main():
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
        f"VAE checkpoint loaded from: {vae_stats['loaded_checkpoint']}\n"
        f"Best β·ELBO checkpoint: {vae_stats['best_elbo_path']}\n"
        f"Best recon checkpoint: {vae_stats['best_recon_path']}"
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
        f"Summarizer best val loss: {summarizer_stats['best_val']:.6f}\n"
        f"Summarizer test loss: {summarizer_stats['test_loss']:.6f}\n"
        f"Checkpoint: {summarizer_stats['checkpoint']}"
    )


if __name__ == "__main__":
    main()
