from Latent_Space.latent_vae_utils import normalize_and_check
from Latent_Space.latent_vae import LatentVAE
import torch.nn.functional as F
import os
import torch
import crypto_config
from Dataset.fin_dataset import run_experiment


train_dl, val_dl, test_dl, sizes = run_experiment(
    data_dir=crypto_config.DATA_DIR,
    date_batching=crypto_config.date_batching,
    dates_per_batch=crypto_config.BATCH_SIZE,
    K=crypto_config.WINDOW,
    H=crypto_config.PRED,
    coverage=crypto_config.COVERAGE
)
print("sizes:", sizes)

train_size, val_size, test_size = sizes
xb, yb, meta = next(iter(train_dl))
V, T = xb
M = meta["entity_mask"]
print("sizes:", sizes)
print("V:", V.shape, "T:", T.shape, "y:", yb.shape)  # -> [B,N,K,F], [B,N,K,F], [B,N,H]
print("min coverage:", float(M.float().mean(1).min().item()))
print("frac padded:", float((~M).float().mean().item()))
# -------------------- Model --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = LatentVAE(
    seq_len=crypto_config.PRED,  # multi-y with horizon=PRED
    latent_dim=crypto_config.VAE_LATENT_DIM,
    latent_channel=crypto_config.VAE_LATENT_CHANNELS,
    enc_layers=crypto_config.VAE_LAYERS, enc_heads=crypto_config.VAE_HEADS, enc_ff=crypto_config.VAE_FF,
    dec_layers=crypto_config.VAE_LAYERS, dec_heads=crypto_config.VAE_HEADS, dec_ff=crypto_config.VAE_FF,
    skip=False
).to(device)

# ---- optimizer (WD helps rein in KL a bit) ----
LEARNING_RATE = crypto_config.VAE_LEARNING_RATE
WEIGHT_DECAY = crypto_config.VAE_WEIGHT_DECAY
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# AMP scaler (safe on CPU too; it’ll just be disabled)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

# -------------------- β schedule (simple & effective) --------------------
VAE_BETA = crypto_config.VAE_BETA
WARMUP_EPOCHS = crypto_config.VAE_WARMUP_EPOCHS

# -------------------- Checkpointing --------------------
model_dir = crypto_config.VAE_DIR
os.makedirs(model_dir, exist_ok=True)

best_val_elbo = float('inf')
best_val_recon = float('inf')
best_elbo_path, best_recon_path = None, None
patience_counter = 0
MAX_PATIENCE = crypto_config.VAE_MAX_PATIENCE

print("Starting training.")
# -------------------- Train Loop --------------------
for epoch in range(1, crypto_config.EPOCHS + 1):
    model.train()
    beta = 0.0 if epoch <= WARMUP_EPOCHS else VAE_BETA

    # running sums to report per-element means
    train_recon_sum, train_kl_sum, train_elems = 0.0, 0.0, 0

    for (_, yb, meta) in train_dl:
        # yb: [B, N, H], mask: [B, N]
        M = meta["entity_mask"].to(device)  # [B, N] bool
        y = yb.to(device)  # [B, N, H]

        # Flatten panels, select valid entities only
        B, N, H = y.shape
        y_flat = y.reshape(B * N, H)  # [B*N, H]
        m_flat = M.reshape(B * N)  # [B*N]
        if not m_flat.any():
            continue
        y_in = y_flat[m_flat].unsqueeze(-1)  # [B_eff, H, 1]

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            y_hat, mu, logvar = model(y_in)

            # ------- per-element MEAN losses (train) -------
            recon_loss = F.mse_loss(y_hat, y_in, reduction='mean')  # mean over B*H*D
            kl_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B_eff,H,Z]
            kl_loss = kl_elem.mean()  # mean over B*H*Z
            loss = recon_loss + beta * kl_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer);
        scaler.update()

        # accumulate for logging as sums (to compute means later)
        num_recon_elems = y_in.numel()  # B_eff*H*1
        num_kl_elems = mu.numel()  # B_eff*H*Z
        train_recon_sum += recon_loss.item() * num_recon_elems
        train_kl_sum += kl_loss.item() * num_kl_elems / mu.size(-1)  # put KL on /elem basis of (B*H), not per-dim
        train_elems += num_recon_elems

    # -------------------- Validation --------------------
    model.eval()
    val_recon_sum, val_kl_sum, val_elems = 0.0, 0.0, 0
    with torch.no_grad():
        for (_, yb, meta) in val_dl:
            M = meta["entity_mask"].to(device)  # [B, N]
            y = yb.to(device)  # [B, N, H]
            B, N, H = y.shape
            y_flat = y.reshape(B * N, H)
            m_flat = M.reshape(B * N)
            if not m_flat.any():
                continue
            y_in = y_flat[m_flat].unsqueeze(-1)  # [B_eff, H, 1]

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                y_hat, mu, logvar = model(y_in)
                recon_loss = F.mse_loss(y_hat, y_in, reduction='mean')
                kl_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_elem.mean()

            num_recon_elems = y_in.numel()
            num_kl_elems = mu.numel()
            val_recon_sum += recon_loss.item() * num_recon_elems
            val_kl_sum += kl_loss.item() * num_kl_elems / mu.size(-1)
            val_elems += num_recon_elems

    # -------------------- Epoch metrics (per-element means) --------------------
    per_elem_train_recon = train_recon_sum / max(1, train_elems)
    per_elem_val_recon = val_recon_sum / max(1, val_elems)

    # KL we tracked per (B*H); align denominators with recon
    per_elem_train_kl = train_kl_sum / max(1, train_elems)
    per_elem_val_kl = val_kl_sum / max(1, val_elems)

    # Report both classical (beta=1) and training (beta=β) objectives
    beta = 0.0 if epoch <= WARMUP_EPOCHS else VAE_BETA
    val_elbo_unweighted = per_elem_val_recon + per_elem_val_kl
    train_elbo_unweighted = per_elem_train_recon + per_elem_train_kl
    val_elbo_beta = per_elem_val_recon + VAE_BETA * per_elem_val_kl
    train_elbo_beta = per_elem_train_recon + VAE_BETA * per_elem_train_kl

    print(
        f"Epoch {epoch}/{crypto_config.EPOCHS} - β={beta:.3f} | "
        f"Train (β·ELBO): {train_elbo_beta:.6f}  [Recon {per_elem_train_recon:.6f}, KL/elem {per_elem_train_kl:.6f}] | "
        f"Val   (β·ELBO): {val_elbo_beta:.6f}    [Recon {per_elem_val_recon:.6f}, KL/elem {per_elem_val_kl:.6f}]"
    )

    # -------------------- Checkpointing + Early Stop --------------------
    # Use β·ELBO for selection, to match training objective
    improved_elbo = (val_elbo_beta < 0.95 * best_val_elbo)
    improved_recon = (per_elem_val_recon < 0.95 * best_val_recon)

    if improved_elbo:
        best_val_elbo = val_elbo_beta
        best_elbo_path = os.path.join(model_dir, f"pred-{crypto_config.PRED}_ch-{crypto_config.VAE_LATENT_CHANNELS}_elbo.pt")
        torch.save(model.state_dict(), best_elbo_path)
        print("  -> Saved best β·ELBO")

    if improved_recon:
        best_val_recon = per_elem_val_recon
        best_recon_path = os.path.join(model_dir, f"pred-{crypto_config.PRED}_ch-{crypto_config.VAE_LATENT_CHANNELS}_recon.pt")
        torch.save(model.state_dict(), best_recon_path)
        print("  -> Saved best Recon")

    patience_counter = 0 if improved_elbo else (patience_counter + 1)
    if patience_counter >= MAX_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}: β·ELBO hasn't improved in {MAX_PATIENCE} epochs.")
        break

# --- Load the preferred checkpoint for downstream use ---
to_load = best_elbo_path or best_recon_path or os.path.join(model_dir, "last.pt")
print(f"Loading checkpoint: {to_load}")
vae = LatentVAE(seq_len=crypto_config.PRED,
    latent_dim=crypto_config.VAE_LATENT_DIM,
    latent_channel=crypto_config.VAE_LATENT_CHANNELS,
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
        y = yb.to(device)  # [B, N, H]
        B, N, H = y.shape
        y_flat = y.reshape(B * N, H)
        m_flat = M.reshape(B * N)
        if not m_flat.any():
            continue
        y_in = y_flat[m_flat].unsqueeze(-1)  # [B_eff, H, 1]
        _, mu, _ = vae(y_in)  # mu: [B_eff, H, Z]
        all_mu.append(mu.cpu())
if all_mu:
    all_mu = torch.cat(all_mu, dim=0)  # [N_seq, H, Z]
    _normed, mu_d, std_d = normalize_and_check(all_mu, plot=True)
else:
    print("No μ collected (empty dataloader batch?).")
