from Model.cond_diffusion_utils import normalize_and_check
from Latent_Space.latent_vae import LatentVAE
import torch.nn.functional as F
import os
import torch, importlib
import crypto_config

# -------------------- Config --------------------
mod = importlib.import_module(crypto_config.DATA_MODULE)

# ============== Re-index The Window & Future Horizon For Datasets ==============
kept = mod.rebuild_window_index_only(
    data_dir=crypto_config.DATA_DIR,
    window=crypto_config.WINDOW,
    horizon=crypto_config.PRED,
)
print("new total windows indexed:", kept)
# -------------------- Loaders --------------------
# Let the module’s default collate return panels + mask
train_dl, val_dl, test_dl, sizes = mod.load_dataloaders_with_ratio_split(
    data_dir=crypto_config.DATA_DIR,
    train_ratio=crypto_config.train_ratio, val_ratio=crypto_config.val_ratio, test_ratio=crypto_config.test_ratio,
    n_entities=crypto_config.NUM_ENTITIES,
    shuffle_train=crypto_config.shuffle_train,
    coverage_per_window=crypto_config.COVERAGE,
    date_batching=crypto_config.date_batching,
    dates_per_batch=crypto_config.BATCH_SIZE,      # B == number of dates per batch
    window=crypto_config.WINDOW,
    horizon=crypto_config.PRED,
    norm_scope=crypto_config.norm_scope
)
train_size, val_size, test_size = sizes
xb, yb, meta = next(iter(train_dl))
V, T = xb
M = meta["entity_mask"]
print("sizes:", sizes)
print("V:", V.shape, "T:", T.shape, "y:", yb.shape)         # -> [B,N,K,F], [B,N,K,F], [B,N,H]
print("min coverage:", float(M.float().mean(1).min().item()))
print("frac padded:", float((~M).float().mean().item()))

# -------------------- Model --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = LatentVAE(
    input_dim=1, seq_len=crypto_config.PRED, # Since target sequences are univariate, though entities are far greater than 1.
    latent_dim=crypto_config.VAE_LATENT_DIM,
    enc_layers=crypto_config.VAE_LAYERS, enc_heads=crypto_config.VAE_HEADS, enc_ff=crypto_config.VAE_FF,
    dec_layers=crypto_config.VAE_LAYERS, dec_heads=crypto_config.VAE_HEADS, dec_ff=crypto_config.VAE_FF
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=crypto_config.LEARNING_RATE)

# AMP scaler (safe on CPU too; it’ll just be disabled)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

# -------------------- Checkpointing --------------------
model_dir = './ldt/saved_model'
os.makedirs(model_dir, exist_ok=True)  # <- fix: import os

current_best_recon_path = None
best_val_recon = float('inf')
patience_counter_recon = 0
warm_up_epochs = 200
print("Starting training.")
# -------------------- Train Loop --------------------
for epoch in range(1, crypto_config.EPOCHS + 1):
    model.train()
    total_recon, total_kl = 0.0, 0.0
    total_valid = 0        # number of valid entity sequences seen this epoch

    annealing_period = 2000
    if epoch <= warm_up_epochs:
        beta = 0
    else:
        beta = min(0.2, (epoch - warm_up_epochs) / annealing_period)
    for (_, yb, meta) in train_dl:
        # yb: [B, N, H], mask: [B, N]
        M = meta["entity_mask"].to(device)               # [B, N] bool
        y = yb.to(device)                                # [B, N, H]
        # Flatten panels, select valid entities only
        B, N, H = y.shape
        y_flat = y.reshape(B * N, H)                     # [B*N, H]
        m_flat = M.reshape(B * N)                        # [B*N]
        if m_flat.any():
            y_in = y_flat[m_flat].unsqueeze(-1)          # [B_eff, H, 1]
        else:
            continue  # no valid entities in this mini-batch (unlikely if coverage>0)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            y_hat, mu, logvar = model(y_in)              # shapes depend on your model; y_hat ~ y_in

            # Reconstruction over valid sequences only
            recon_loss = F.mse_loss(y_hat, y_in, reduction='sum')

            # KL per element (handles either [B_eff, Z] or [B_eff, H, Z])
            mu32, logvar32 = mu.float(), logvar.float()
            kl_elem = -0.5 * (1 + logvar32 - mu32.pow(2) - logvar32.exp())
            kl_div = kl_elem.sum()

            # Soft free-bits (average across batch dims, keep time/latent dims if present)
            # Collapse batch axis only:
            reduce_axes = tuple(i for i in range(kl_elem.dim()) if i == 0)
            kl_per_elem_mean = kl_elem.mean(dim=reduce_axes)  # [(H,)Z] or [Z]
            excess_per_elem = F.softplus(
                (kl_per_elem_mean - crypto_config.FREE_BITS_PER_ELEM) / crypto_config.SOFTNESS
            ) * crypto_config.SOFTNESS
            kl_div_eff = excess_per_elem.sum() * y_in.size(0)  # scale back by batch size

            loss = (recon_loss + beta * kl_div_eff) / y_in.size(0)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer); scaler.update()

        total_recon += recon_loss.item()
        total_kl += kl_div.item()
        total_valid += y_in.size(0)

    # -------------------- Validation --------------------
    model.eval()
    val_recon_sum, val_kl_sum, val_valid = 0.0, 0.0, 0
    with torch.no_grad():
        for (_, yb, meta) in val_dl:
            M = meta["entity_mask"].to(device)           # [B, N]
            y = yb.to(device)                            # [B, N, H]
            B, N, H = y.shape
            y_flat = y.reshape(B * N, H)
            m_flat = M.reshape(B * N)
            if not m_flat.any():
                continue
            y_in = y_flat[m_flat].unsqueeze(-1)          # [B_eff, H, 1]

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                y_hat, mu, logvar = model(y_in)
                val_recon_sum += F.mse_loss(y_hat, y_in, reduction='sum').item()
                val_kl_sum += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).item()
                val_valid += y_in.size(0)

    # -------------------- Logging --------------------
    # Averages per *sequence* (entity) using valid counts
    train_recon_avg = total_recon / max(1, total_valid)
    train_kl_avg = total_kl / max(1, total_valid)
    val_recon_avg = val_recon_sum / max(1, val_valid)
    val_kl_avg = val_kl_sum / max(1, val_valid)

    # Per-element (time × feature=1)
    per_elem_train_recon = train_recon_avg / (crypto_config.PRED * 1)
    per_elem_val_recon   = val_recon_avg   / (crypto_config.PRED * 1)
    per_elem_train_kl    = train_kl_avg    / (crypto_config.PRED * crypto_config.VAE_LATENT_DIM)
    per_elem_val_kl      = val_kl_avg      / (crypto_config.PRED * crypto_config.VAE_LATENT_DIM)

    train_elbo = train_recon_avg + train_kl_avg
    val_elbo   = val_recon_avg   + val_kl_avg

    print(
        f"Epoch {epoch}/{crypto_config.EPOCHS} - β={beta:.3f} | "
        f"Train ELBO: {train_elbo:.4f} (Recon: {train_recon_avg:.4f} /elem: {per_elem_train_recon:.6f}, "
        f"KL: {train_kl_avg:.4f} /elem: {per_elem_train_kl:.6f}) | "
        f"Val ELBO: {val_elbo:.4f} (Recon: {val_recon_avg:.4f} /elem: {per_elem_val_recon:.6f}, "
        f"KL: {val_kl_avg:.4f} /elem: {per_elem_val_kl:.6f})"
    )

    # -------------------- Checkpointing + Early Stop --------------------
    if val_recon_avg < best_val_recon and per_elem_val_kl <= 2.0:
        if current_best_recon_path and os.path.exists(current_best_recon_path):
            os.remove(current_best_recon_path)
        best_val_recon = val_recon_avg
        patience_counter_recon = 0
        new_path = os.path.join(model_dir, f"hidden_{crypto_config.VAE_LATENT_DIM}_recon_{val_recon_avg:.4f}_kl_{val_kl_avg}.pt")
        torch.save(model.state_dict(), new_path)
        current_best_recon_path = new_path
        print(f"  -> New best Recon model saved: {os.path.basename(new_path)}")
    else:
        patience_counter_recon += 1

    if patience_counter_recon >= crypto_config.MAX_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}: Recon hasn't improved in {crypto_config.MAX_PATIENCE} epochs.")
        break


# --- Load pre-trained VAE ---
vae = LatentVAE(
    input_dim=1, seq_len=crypto_config.PRED, # Since target sequences are univariate, though entities are far greater than 1.
    latent_dim=crypto_config.VAE_LATENT_DIM,
    enc_layers=crypto_config.VAE_LAYERS, enc_heads=crypto_config.VAE_HEADS, enc_ff=crypto_config.VAE_FF,
    dec_layers=crypto_config.VAE_LAYERS, dec_heads=crypto_config.VAE_HEADS, dec_ff=crypto_config.VAE_FF
).to(device)

vae.load_state_dict(torch.load(
    current_best_recon_path,
    map_location=device
))
vae.eval()
# Freeze encoder parameters
for param in vae.encoder.parameters():
    param.requires_grad = False

# --- Compute global μ statistics for normalization ---
all_mu = []
with torch.no_grad():
    for (_, y, m) in train_dl:
        M = meta["entity_mask"].to(device)  # [B, N]
        y = yb.to(device)  # [B, N, H]
        B, N, H = y.shape
        y_flat = y.reshape(B * N, H)
        m_flat = M.reshape(B * N)
        if not m_flat.any():
            continue
        y_in = y_flat[m_flat].unsqueeze(-1)
        _, mu, _ = vae(y_in)
        all_mu.append(mu.cpu())
all_mu = torch.cat(all_mu, dim=0)  # [N, L, D]
_, mu_d, std_d = normalize_and_check(all_mu, plot=True)


