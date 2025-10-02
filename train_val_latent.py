from Latent_Space.latent_vae_utils import normalize_and_check
from Latent_Space.latent_vae import LatentVAE
import torch.nn.functional as F
import os
import torch
import crypto_config
from Dataset.fin_dataset import run_experiment


def run(train_dl=None, val_dl=None, test_dl=None, sizes=None, config=crypto_config):
    owns_loaders = any(loader is None for loader in (train_dl, val_dl, test_dl))
    if owns_loaders:
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

    if sizes is not None:
        print("sizes:", sizes)
    else:
        print("sizes: (unknown)")

    xb, yb, meta = next(iter(train_dl))
    V, T = xb
    M = meta["entity_mask"]
    if sizes is not None:
        print("sizes:", sizes)
    print("V:", V.shape, "T:", T.shape, "y:", yb.shape)
    print("min coverage:", float(M.float().mean(1).min().item()))
    print("frac padded:", float((~M).float().mean().item()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
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

    learning_rate = config.VAE_LEARNING_RATE
    weight_decay = config.VAE_WEIGHT_DECAY
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    vae_beta = config.VAE_BETA
    warmup_epochs = config.VAE_WARMUP_EPOCHS

    model_dir = config.VAE_DIR
    os.makedirs(model_dir, exist_ok=True)
    last_ckpt = os.path.join(model_dir, "last.pt")

    best_val_elbo = float("inf")
    best_val_recon = float("inf")
    best_elbo_path, best_recon_path = None, None
    patience_counter = 0
    max_patience = config.VAE_MAX_PATIENCE

    print("Starting training.")
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        beta = 0.0 if epoch <= warmup_epochs else vae_beta

        train_recon_sum, train_kl_sum, train_elems = 0.0, 0.0, 0

        for (_, yb, meta) in train_dl:
            M = meta["entity_mask"].to(device)
            y = yb.to(device)

            B, N, H = y.shape
            y_flat = y.reshape(B * N, H)
            m_flat = M.reshape(B * N)
            if not m_flat.any():
                continue
            y_in = y_flat[m_flat].unsqueeze(-1)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                y_hat, mu, logvar = model(y_in)
                recon_loss = F.mse_loss(y_hat, y_in, reduction="mean")
                kl_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_elem.mean()
                loss = recon_loss + beta * kl_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            num_recon_elems = y_in.numel()
            num_kl_elems = mu.numel()
            train_recon_sum += recon_loss.item() * num_recon_elems
            train_kl_sum += kl_loss.item() * num_kl_elems / mu.size(-1)
            train_elems += num_recon_elems

        model.eval()
        val_recon_sum, val_kl_sum, val_elems = 0.0, 0.0, 0
        with torch.no_grad():
            for (_, yb, meta) in val_dl:
                M = meta["entity_mask"].to(device)
                y = yb.to(device)
                B, N, H = y.shape
                y_flat = y.reshape(B * N, H)
                m_flat = M.reshape(B * N)
                if not m_flat.any():
                    continue
                y_in = y_flat[m_flat].unsqueeze(-1)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    y_hat, mu, logvar = model(y_in)
                    recon_loss = F.mse_loss(y_hat, y_in, reduction="mean")
                    kl_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                    kl_loss = kl_elem.mean()

                num_recon_elems = y_in.numel()
                num_kl_elems = mu.numel()
                val_recon_sum += recon_loss.item() * num_recon_elems
                val_kl_sum += kl_loss.item() * num_kl_elems / mu.size(-1)
                val_elems += num_recon_elems

        per_elem_train_recon = train_recon_sum / max(1, train_elems)
        per_elem_val_recon = val_recon_sum / max(1, val_elems)
        per_elem_train_kl = train_kl_sum / max(1, train_elems)
        per_elem_val_kl = val_kl_sum / max(1, val_elems)

        beta = 0.0 if epoch <= warmup_epochs else vae_beta
        val_elbo_unweighted = per_elem_val_recon + per_elem_val_kl
        train_elbo_unweighted = per_elem_train_recon + per_elem_train_kl
        val_elbo_beta = per_elem_val_recon + vae_beta * per_elem_val_kl
        train_elbo_beta = per_elem_train_recon + vae_beta * per_elem_train_kl

        print(
            f"Epoch {epoch}/{config.EPOCHS} - β={beta:.3f} | "
            f"Train (β·ELBO): {train_elbo_beta:.6f}  [Recon {per_elem_train_recon:.6f}, KL/elem {per_elem_train_kl:.6f}] | "
            f"Val   (β·ELBO): {val_elbo_beta:.6f}    [Recon {per_elem_val_recon:.6f}, KL/elem {per_elem_val_kl:.6f}]"
        )

        improved_elbo = val_elbo_beta < 0.95 * best_val_elbo
        improved_recon = per_elem_val_recon < 0.95 * best_val_recon

        if improved_elbo:
            best_val_elbo = val_elbo_beta
            best_elbo_path = os.path.join(model_dir, f"pred-{config.PRED}_ch-{config.VAE_LATENT_CHANNELS}_elbo.pt")
            torch.save(model.state_dict(), best_elbo_path)
            print("  -> Saved best β·ELBO")

        if improved_recon:
            best_val_recon = per_elem_val_recon
            best_recon_path = os.path.join(model_dir, f"pred-{config.PRED}_ch-{config.VAE_LATENT_CHANNELS}_recon.pt")
            torch.save(model.state_dict(), best_recon_path)
            print("  -> Saved best Recon")

        torch.save(model.state_dict(), last_ckpt)

        patience_counter = 0 if improved_elbo else (patience_counter + 1)
        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch}: β·ELBO hasn't improved in {max_patience} epochs.")
            break

    to_load = best_elbo_path or best_recon_path or last_ckpt
    print(f"Loading checkpoint: {to_load}")
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
    vae.load_state_dict(torch.load(to_load, map_location=device))
    vae.eval()

    for p in vae.encoder.parameters():
        p.requires_grad = False

    all_mu = []
    with torch.no_grad():
        for (_, yb, meta) in train_dl:
            M = meta["entity_mask"].to(device)
            y = yb.to(device)
            B, N, H = y.shape
            y_flat = y.reshape(B * N, H)
            m_flat = M.reshape(B * N)
            if not m_flat.any():
                continue
            y_in = y_flat[m_flat].unsqueeze(-1)
            _, mu, _ = vae(y_in)
            all_mu.append(mu.cpu())
    if all_mu:
        all_mu = torch.cat(all_mu, dim=0)
        _normed, mu_d, std_d = normalize_and_check(all_mu, plot=True)
    else:
        print("No μ collected (empty dataloader batch?).")

    return {
        "train_loader": train_dl,
        "val_loader": val_dl,
        "test_loader": test_dl,
        "sizes": sizes,
        "best_elbo_path": best_elbo_path,
        "best_recon_path": best_recon_path,
        "loaded_checkpoint": to_load,
        "model": vae,
    }


if __name__ == "__main__":
    run()
