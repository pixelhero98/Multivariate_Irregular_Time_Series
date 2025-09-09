
from Latent_Space.latent_vae_utils import normalize_and_check  # plotting/stats helper (unchanged)
from Latent_Space.latent_vae_2d import VAE2D
import torch.nn.functional as F
import os
import torch
import crypto_config
from Dataset.fin_dataset import run_experiment

# -------------------- Data --------------------
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
print("V:", V.shape, "T:", T.shape, "y:", yb.shape)         # -> [B,N,K,F], [B,N,K,F], [B,N,H]
print("min coverage:", float(M.float().mean(1).min().item()))
print("frac padded:", float((~M).float().mean().item()))

# -------------------- Model --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparams for 2D VAE
LATENT_C = int(getattr(crypto_config, "VAE_LATENT_CHANNELS", 8))
PATCH_N  = int(getattr(crypto_config, "VAE_PATCH_N", 1))   # no entity compression by default
PATCH_H  = int(getattr(crypto_config, "VAE_PATCH_H", 5))   # e.g., weekly patches

model = VAE2D(
    d_model=crypto_config.VAE_LATENT_DIM,
    C=LATENT_C,
    p_n=PATCH_N,
    p_h=PATCH_H,
    n_layers=crypto_config.VAE_LAYERS,
    n_heads=crypto_config.VAE_HEADS,
    ff=crypto_config.VAE_FF,
    dropout=getattr(crypto_config, "VAE_DROPOUT", 0.1),
    use_cross=True,
).to(device)

# ---- optimizer ----
LEARNING_RATE = float(getattr(crypto_config, "LEARNING_RATE", 1e-3))
WEIGHT_DECAY  = float(getattr(crypto_config, "WEIGHT_DECAY", 1e-4))
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# AMP scaler
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

# -------------------- β schedule --------------------
VAE_BETA = float(getattr(crypto_config, "VAE_BETA", 6.0))
WARMUP_EPOCHS = int(getattr(crypto_config, "VAE_WARMUP_EPOCHS", 10))

# -------------------- Checkpointing --------------------
model_dir = './ldt/saved_model_2d'
os.makedirs(model_dir, exist_ok=True)
best_val_elbo = float('inf')
best_val_recon = float('inf')
best_elbo_path, best_recon_path = None, None
patience_counter = 0
MAX_PATIENCE = int(getattr(crypto_config, "MAX_PATIENCE", 50))

def masked_mse(y_hat, y, mask_2d):
    """
    y_hat: [B,N,H,1], y: [B,N,H], mask_2d: [B,N] (True=valid entity)
    Applies entity mask across all H steps.
    """
    B, N, H = y.shape
    m = mask_2d.to(y_hat.dtype).unsqueeze(-1).expand(B, N, H)  # [B,N,H]
    diff2 = (y_hat.squeeze(-1) - y)**2 * m
    denom = m.sum().clamp_min(1.)
    return diff2.sum() / denom

def kl_weighted(mu_g, logvar_g, cov=None):
    """
    mu_g, logvar_g: [B,N',H',C]
    cov: optional coverage [B,1,N',H'] in [0,1]; when provided, weight token KL by coverage.
    returns mean KL per element of (B·N'·H'·C) or coverage-weighted mean.
    """
    kl = 0.5 * (mu_g.pow(2) + logvar_g.exp() - logvar_g - 1.0)  # [B,N',H',C]
    if cov is None:
        return kl.mean()
    # weight by coverage (broadcast to C)
    w = cov.permute(0,2,3,1)  # [B,N',H',1]
    klw = kl * w
    denom = (w.sum() * mu_g.size(-1)).clamp_min(1.)
    return klw.sum() / denom

print("Starting training.")
# -------------------- Train Loop --------------------
for epoch in range(1, crypto_config.EPOCHS + 1):
    model.train()
    beta = 0.0 if epoch <= WARMUP_EPOCHS else VAE_BETA

    train_recon_sum, train_kl_sum, train_elems = 0.0, 0.0, 0.0

    for (_, yb, meta) in train_dl:
        # yb: [B,N,H], entity_mask: [B,N] (True=valid ticker in this batch window)
        M = meta["entity_mask"].to(device)         # [B,N] bool
        y = yb.to(device)                          # [B,N,H]

        # prepare inputs for 2D VAE
        x_in   = y.unsqueeze(-1).to(device)        # [B,N,H,1]
        m_2d   = M                                  # [B,N] bool
        m_full = m_2d.unsqueeze(-1).expand_as(y)    # [B,N,H] (broadcast mask across time)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            # forward (coverage computed inside if mask provided)
            y_hat, mu_g, logvar_g, z_grid, cov, grid, orig = model(x_in, mask=m_full)

            # losses
            recon_loss = masked_mse(y_hat, y, m_2d)           # entity-masked MSE
            kl_loss    = kl_weighted(mu_g, logvar_g, cov)     # coverage-weighted KL over tokens
            loss = recon_loss + beta * kl_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer); scaler.update()

        # logging accumulators: per-element means over observed y
        elems = m_full.sum().item()                           # number of observed scalar targets
        train_recon_sum += float(recon_loss.item()) * elems
        train_kl_sum    += float(kl_loss.item()) * elems
        train_elems     += elems

    # -------------------- Validation --------------------
    model.eval()
    val_recon_sum, val_kl_sum, val_elems = 0.0, 0.0, 0.0
    with torch.no_grad():
        for (_, yb, meta) in val_dl:
            M = meta["entity_mask"].to(device)      # [B,N]
            y = yb.to(device)                       # [B,N,H]
            x_in   = y.unsqueeze(-1)
            m_full = M.unsqueeze(-1).expand_as(y)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                y_hat, mu_g, logvar_g, z_grid, cov, grid, orig = model(x_in, mask=m_full)
                recon_loss = masked_mse(y_hat, y, M)
                kl_loss    = kl_weighted(mu_g, logvar_g, cov)

            elems = m_full.sum().item()
            val_recon_sum += float(recon_loss.item()) * elems
            val_kl_sum    += float(kl_loss.item()) * elems
            val_elems     += elems

    # -------------------- Epoch metrics (per observed scalar) --------------------
    per_elem_train_recon = train_recon_sum / max(1.0, train_elems)
    per_elem_val_recon   = val_recon_sum   / max(1.0, val_elems)
    per_elem_train_kl    = train_kl_sum    / max(1.0, train_elems)
    per_elem_val_kl      = val_kl_sum      / max(1.0, val_elems)

    beta = 0.0 if epoch <= WARMUP_EPOCHS else VAE_BETA
    val_elbo_beta   = per_elem_val_recon   + beta * per_elem_val_kl
    train_elbo_beta = per_elem_train_recon + beta * per_elem_train_kl

    print(
        f"Epoch {epoch}/{crypto_config.EPOCHS} - β={beta:.3f} | "
        f"Train (β·ELBO): {train_elbo_beta:.6f}  [Recon {per_elem_train_recon:.6f}, KL/elem {per_elem_train_kl:.6f}] | "
        f"Val   (β·ELBO): {val_elbo_beta:.6f}    [Recon {per_elem_val_recon:.6f}, KL/elem {per_elem_val_kl:.6f}]"
    )

    # -------------------- Checkpointing + Early Stop --------------------
    improved_elbo  = (val_elbo_beta < best_val_elbo - 1e-9)
    improved_recon = (per_elem_val_recon < best_val_recon - 1e-9)

    if improved_elbo:
        best_val_elbo = val_elbo_beta
        best_elbo_path = os.path.join(model_dir, f"best_elbo.pt")
        torch.save(model.state_dict(), best_elbo_path)
        print("  -> Saved best β·ELBO")

    if improved_recon:
        best_val_recon = per_elem_val_recon
        best_recon_path = os.path.join(model_dir, f"best_recon.pt")
        torch.save(model.state_dict(), best_recon_path)
        print("  -> Saved best Recon")

    torch.save(model.state_dict(), os.path.join(model_dir, "last.pt"))

    patience_counter = 0 if improved_elbo else (patience_counter + 1)
    if patience_counter >= MAX_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}: β·ELBO hasn't improved in {MAX_PATIENCE} epochs.")
        break

# --- Load the preferred checkpoint for downstream use ---
to_load = best_elbo_path or best_recon_path or os.path.join(model_dir, "last.pt")
print(f"Loading checkpoint: {to_load}")
vae = VAE2D(
    d_model=crypto_config.VAE_LATENT_DIM,
    C=LATENT_C,
    p_n=PATCH_N,
    p_h=PATCH_H,
    n_layers=crypto_config.VAE_LAYERS,
    n_heads=crypto_config.VAE_HEADS,
    ff=crypto_config.VAE_FF,
    dropout=getattr(crypto_config, "VAE_DROPOUT", 0.1),
    use_cross=True,
).to(device)
vae.load_state_dict(torch.load(to_load, map_location=device))
vae.eval()

# Freeze encoder parameters if you want to reuse the encoder as-is
for p in vae.encoder.parameters():
    p.requires_grad = False

# --- (Optional) gather μ for sanity checks / whitening stats ---
# We flatten (B, N') into one pseudo-batch and keep H' and C for the checker: [num, H', C]
all_mu = []
with torch.no_grad():
    for (_, yb, meta) in train_dl:
        M = meta["entity_mask"].to(device)   # [B,N]
        y = yb.to(device)                    # [B,N,H]
        x_in   = y.unsqueeze(-1)
        m_full = M.unsqueeze(-1).expand_as(y)
        _, mu_g, _, _, _, grid, _ = vae(x_in, mask=m_full)   # μ: [B,N',H',C]
        B, Np, Hp, Cc = mu_g.shape
        mu_seq = mu_g.view(B*Np, Hp, Cc).cpu()               # -> [num, H', C]
        all_mu.append(mu_seq)
if all_mu:
    all_mu = torch.cat(all_mu, dim=0)
    _normed, mu_d, std_d = normalize_and_check(all_mu, plot=True)
else:
    print("No μ collected (empty dataloader batch?).")

print("Done. This VAE is 2D (p_n=%d, p_h=%d). Diffusion-ready latent shape per batch: [B, C, N', H']." % (PATCH_N, PATCH_H))
