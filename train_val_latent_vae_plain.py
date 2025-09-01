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

# -------------------- Data -------------------
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
    input_dim=1, seq_len=crypto_config.PRED,  # univariate y with horizon=PRED
    latent_dim=crypto_config.VAE_LATENT_DIM,
    enc_layers=crypto_config.VAE_LAYERS, enc_heads=crypto_config.VAE_HEADS, enc_ff=crypto_config.VAE_FF,
    dec_layers=crypto_config.VAE_LAYERS, dec_heads=crypto_config.VAE_HEADS, dec_ff=crypto_config.VAE_FF
).to(device)

# ---- optimizer (WD helps rein in KL a bit) ----
LEARNING_RATE = getattr(crypto_config, "LEARNING_RATE", 1e-3)
WEIGHT_DECAY  = getattr(crypto_config, "WEIGHT_DECAY", 1e-4)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# AMP scaler (safe on CPU too; it’ll just be disabled)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

# -------------------- β schedule (simple & effective) --------------------
VAE_BETA = float(getattr(crypto_config, "VAE_BETA", 6.0))             # ← pick 4–8
WARMUP_EPOCHS = int(getattr(crypto_config, "VAE_WARMUP_EPOCHS", 10))  # ← short warm-up

# -------------------- Checkpointing --------------------
model_dir = './ldt/saved_model'
os.makedirs(model_dir, exist_ok=True)

best_val_elbo = float('inf')
best_val_recon = float('inf')
best_elbo_path, best_recon_path = None, None
patience_counter = 0
MAX_PATIENCE = int(getattr(crypto_config, "MAX_PATIENCE", 50))

print("Starting training.")
# -------------------- Train Loop --------------------
for epoch in range(1, crypto_config.EPOCHS + 1):
    model.train()
    beta = 0.0 if epoch <= WARMUP_EPOCHS else VAE_BETA

    # running sums to report per-element means
    train_recon_sum, train_kl_sum, train_elems = 0.0, 0.0, 0

    for (_, yb, meta) in train_dl:
        # yb: [B, N, H], mask: [B, N]
        M = meta["entity_mask"].to(device)               # [B, N] bool
        y = yb.to(device)                                # [B, N, H]

        # Flatten panels, select valid entities only
        B, N, H = y.shape
        y_flat = y.reshape(B * N, H)                     # [B*N, H]
        m_flat = M.reshape(B * N)                        # [B*N]
        if not m_flat.any():
            continue
        y_in = y_flat[m_flat].unsqueeze(-1)              # [B_eff, H, 1]

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            y_hat, mu, logvar = model(y_in)

            # ------- per-element MEAN losses (train) -------
            recon_loss = F.mse_loss(y_hat, y_in, reduction='mean')  # mean over B*H*D
            kl_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B_eff,H,Z]
            kl_loss  = kl_elem.mean()                                # mean over B*H*Z
            loss = recon_loss + beta * kl_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer); scaler.update()

        # accumulate for logging as sums (to compute means later)
        num_recon_elems = y_in.numel()                               # B_eff*H*1
        num_kl_elems    = mu.numel()                                 # B_eff*H*Z
        train_recon_sum += recon_loss.item() * num_recon_elems
        train_kl_sum    += kl_loss.item() * num_kl_elems / mu.size(-1)  # put KL on /elem basis of (B*H), not per-dim
        train_elems     += num_recon_elems

    # -------------------- Validation --------------------
    model.eval()
    val_recon_sum, val_kl_sum, val_elems = 0.0, 0.0, 0
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
                recon_loss = F.mse_loss(y_hat, y_in, reduction='mean')
                kl_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_elem.mean()

            num_recon_elems = y_in.numel()
            num_kl_elems    = mu.numel()
            val_recon_sum += recon_loss.item() * num_recon_elems
            val_kl_sum    += kl_loss.item() * num_kl_elems / mu.size(-1)
            val_elems     += num_recon_elems

    # -------------------- Epoch metrics (per-element means) --------------------
    per_elem_train_recon = train_recon_sum / max(1, train_elems)
    per_elem_val_recon   = val_recon_sum   / max(1, val_elems)
    # KL we logged on a (B*H) element basis; keep it comparable to recon:
    per_elem_train_kl = train_kl_sum / max(1, train_elems)
    per_elem_val_kl   = val_kl_sum   / max(1, val_elems)

    train_elbo = per_elem_train_recon + per_elem_train_kl
    val_elbo   = per_elem_val_recon   + per_elem_val_kl

    print(
        f"Epoch {epoch}/{crypto_config.EPOCHS} - β={beta:.3f} | "
        f"Train ELBO: {train_elbo:.6f} (Recon: {per_elem_train_recon:.6f}, KL/elem: {per_elem_train_kl:.6f}) | "
        f"Val   ELBO: {val_elbo:.6f} (Recon: {per_elem_val_recon:.6f}, KL/elem: {per_elem_val_kl:.6f})"
    )

    # -------------------- Checkpointing + Early Stop --------------------
    # best ELBO
    if val_elbo < best_val_elbo:
        best_val_elbo = val_elbo
        best_elbo_path = os.path.join(model_dir, f"best_elbo.pt")
        torch.save(model.state_dict(), best_elbo_path)
        print("  -> Saved best ELBO")

    # best Recon
    if per_elem_val_recon < best_val_recon:
        best_val_recon = per_elem_val_recon
        best_recon_path = os.path.join(model_dir, f"best_recon.pt")
        torch.save(model.state_dict(), best_recon_path)
        print("  -> Saved best Recon")

    # always save last
    last_path = os.path.join(model_dir, f"last.pt")
    torch.save(model.state_dict(), last_path)

    # simple patience on ELBO
    if val_elbo + 1e-8 < best_val_elbo:
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= MAX_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}: ELBO hasn't improved in {MAX_PATIENCE} epochs.")
        break

# --- Load the preferred checkpoint for downstream use ---
to_load = best_elbo_path or best_recon_path or os.path.join(model_dir, "last.pt")
print(f"Loading checkpoint: {to_load}")
vae = LatentVAE(
    input_dim=1, seq_len=crypto_config.PRED,
    latent_dim=crypto_config.VAE_LATENT_DIM,
    enc_layers=crypto_config.VAE_LAYERS, enc_heads=crypto_config.VAE_HEADS, enc_ff=crypto_config.VAE_FF,
    dec_layers=crypto_config.VAE_LAYERS, dec_heads=crypto_config.VAE_HEADS, dec_ff=crypto_config.VAE_FF
).to(device)
vae.load_state_dict(torch.load(to_load, map_location=device))
vae.eval()

# Freeze encoder parameters if you want to reuse the encoder as-is
for p in vae.encoder.parameters():
    p.requires_grad = False

# --- (Optional) gather μ for sanity checks / whitening stats ---
all_mu = []
with torch.no_grad():
    for (_, yb, meta) in train_dl:
        M = meta["entity_mask"].to(device)  # [B, N]
        y = yb.to(device)                   # [B, N, H]
        B, N, H = y.shape
        y_flat = y.reshape(B * N, H)
        m_flat = M.reshape(B * N)
        if not m_flat.any():
            continue
        y_in = y_flat[m_flat].unsqueeze(-1)   # [B_eff, H, 1]
        _, mu, _ = vae(y_in)                  # mu: [B_eff, H, Z]
        all_mu.append(mu.cpu())
if all_mu:
    all_mu = torch.cat(all_mu, dim=0)         # [N_seq, H, Z]
    _normed, mu_d, std_d = normalize_and_check(all_mu, plot=True)
    # Save stats for diffusion if you like:
    # torch.save({"mu": mu_d, "std": std_d}, os.path.join(model_dir, "latent_stats.pt"))
else:
    print("No μ collected (empty dataloader batch?).")
