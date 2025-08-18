import torch, os
from latent_vae_utils import prepare_data_and_cache, load_dataloaders_from_cache, normalize_and_check
from fin_data_prep_f16_ultramem import (
    prepare_stock_windows_and_cache_v2,
    load_dataloaders_with_meta_v2,
    load_dataloaders_with_ratio_split,
    FeatureConfig, CalendarConfig
)
from latent_vae import LatentVAE
from cond_latent_diffuser import LapDiT
import numpy as np
from tqdm import tqdm


# Datasets & Training Preparation
BATCH_SIZE = 64
EPOCHS = 500
LR = 5e-4
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.55, 0.05, 0.4
DEV, DATA_DIR, TCK_PATH = torch.device("cuda" if torch.cuda.is_available() else "cpu"), './ldt/data', './CRYPTO_top.txt'
CHECKPOINT_DIR = './ldt/checkpoints'; os.makedirs(CHECKPOINT_DIR, exist_ok=True)
TCKS, WINDOW, PRED = [], 150, 40

with open(TCK_PATH, 'r') as file:
  TCKS = [line.strip() for line in file]
cfg = FeatureConfig(
    price_fields=["Open", "High", "Low", "Close"],      # returns computed for each listed field
    returns_mode="log",          # 'log' or 'pct'
    include_rvol=True,           # realized vol over returns
    rvol_span=20,
    rvol_on="Close",
    include_dlv=True,            # Δ log volume
    market_proxy="SPY",          # extra MKT factor from SPY
    include_oc=False,
    include_gap=False,
    include_hl_range=False,
    target_field="Close",        # predict future return on this field
    calendar=CalendarConfig(     # calendar features on/off
        include_dow=True,
        include_dom=True,
        include_moy=True,
    ),
)
prepare_stock_windows_and_cache_v2(
    tickers=TCKS,
    start="2017-01-01", # equities "2015-01-01"
    val_start="2023-01-01", # equities "2021-01-01"
    test_start="2024-01-01", # equities "2022-01-01"
    end="2025-06-30",
    window=WINDOW,
    horizon=PRED,
    data_dir="./ldt/crypto_data", # equities "./ldt/data"
    feature_cfg=cfg,
    normalize_per_ticker=True,
    min_train_coverage=0.85, # equities 0.9
    liquidity_rank_window=None,
    top_n_by_dollar_vol=None,  # optional extra filter inside the prep step
    regression=True
)
train_dl, val_dl, test_dl, sizes = load_dataloaders_with_ratio_split(
    data_dir="./ldt/crypto_data", # equities "./ldt/data"
    train_ratio=0.55,
    val_ratio=0.05,
    test_ratio=0.4,
    batch_size=64,
    regression=True,
    per_asset=True,        # keep time order per asset
    shuffle_train=True,
    num_workers=0,
    seed=42,
)

# --- Load pre-trained VAE ---
vae = LatentVAE(
    input_dim=1, seq_len=PRED, # Since our target field is only close, so input_dim is 1.
    latent_dim=64,
    enc_layers=3, enc_heads=4, enc_ff=256,
    dec_layers=3, dec_heads=4, dec_ff=256,
).to(device)
vae.load_state_dict(torch.load(
    './ldt/saved_model/recon_0.0559_epoch_4.pt',
    map_location=device
))
vae.eval()
# Freeze encoder parameters
for param in vae.encoder.parameters():
    param.requires_grad = False

# --- Compute global μ statistics for normalization ---
all_mu = []
with torch.no_grad():
    for _, y in train_loader:
        y = y.to(device)
        _, mu, _ = vae(y)
        all_mu.append(mu.cpu())
all_mu = torch.cat(all_mu, dim=0)  # [N, L, D]
_, mu_d, std_d = normalize_and_check(all_mu)
mu_d, std_d = mu_d.to(device), std_d.to(device)
# -------------------------------------------
latent_dim = all_mu.size(-1)
time_embed_dim = 256  # embedding dimension for timesteps
lap_kernel = 32
num_layers = 4
num_heads = 4
TOTAL_T = 1500 
PATIENCE = 20
N_INF_SAMPLES = 50
# --- Instantiate diffusion model ---
diff_model = LapDiT(
    latent_dim=latent_dim,
    time_embed_dim=time_embed_dim,
    horizon=PRED,
    lap_kernel=lap_kernel,
    num_layers=num_layers,
    num_heads=num_heads,
    mlp_dim=latent_dim,
    max_timesteps=TOTAL_T
).to(device)
noise_scheduler = diff_model.scheduler
optimizer = torch.optim.AdamW(diff_model.parameters(), lr=LR)
val_patience = 0
best_val_loss = np.inf
current_best_ckpt_path = ""
# --- Training loop ---
scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
for epoch in range(1, EPOCHS + 1):
    diff_model.train()
    train_loss = 0.0
    p_unconditional = 0.25

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            _, mu, _ = vae(y)
        mu = (mu - mu_d) / std_d
        mu = torch.clamp(mu, min=-5, max=5)

        timesteps = torch.randint(
            0, noise_scheduler.timesteps,
            (mu.size(0),), device=device
        ).long()
        noise = torch.randn_like(mu)

        if torch.rand(1).item() < p_unconditional:
            cond_tensor = None
        else:
            cond_tensor = x

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            noisy_mu, actual_noise = noise_scheduler.q_sample(mu, timesteps, noise)
            noise_pred = diff_model(noisy_mu, timesteps, series=cond_tensor)  # <- be explicit
            loss = torch.nn.functional.mse_loss(noise_pred, actual_noise)
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # <-- unscale before clipping
        torch.nn.utils.clip_grad_norm_(diff_model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item() * mu.size(0)

    avg_train_loss = train_loss / train_size
    # --- Validation ---
    val_loss = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            _, mu, _ = vae(y)
            mu = (mu - mu_d) / std_d
            mu = torch.clamp(mu, min=-5, max=5)

            t = torch.randint(0, noise_scheduler.timesteps, (mu.size(0),), device=device).long()
            noisy_mu, actual_noise = noise_scheduler.q_sample(mu, t, torch.randn_like(mu))
            noise_pred = diff_model(noisy_mu, t, series=x)
            val_loss += torch.nn.functional.mse_loss(noise_pred, actual_noise).item() * mu.size(0)

    avg_val_loss = val_loss / val_size
    print(f"Epoch {epoch}  train loss: {avg_train_loss:.6f} |  val loss: {avg_val_loss:.6f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        val_patience = 0
    
        if os.path.exists(current_best_ckpt_path):
            os.remove(current_best_ckpt_path)
    
        best_ckpt_path = os.path.join(
            CHECKPOINT_DIR,
            f"best_model_epoch_{epoch:03d}_val_{avg_val_loss:.4f}.pt"
        )
    
        print(f"  -> New best model saved to {os.path.basename(best_ckpt_path)}")
        torch.save({
            'epoch': epoch,
            'model_state': diff_model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'mu_mean': mu_d,
            'mu_std': std_d
        }, best_ckpt_path)
        current_best_ckpt_path = best_ckpt_path
    else:
        val_patience += 1

    if val_patience >= PATIENCE:
        if not os.path.exists(current_best_ckpt_path):
            print("Error: Best checkpoint path not found. Cannot perform final validation.")
        else:
            best_checkpoint = torch.load(current_best_ckpt_path, map_location=device)
            diff_model.load_state_dict(best_checkpoint['model_state'])
            diff_model.eval()
    
            # fixed seeds for reproducible multi-sample eval
            EVAL_SEEDS = [1000 + i for i in range(max(1, N_INF_SAMPLES))]
    
            val_reg_loss = 0.0
            with torch.no_grad():
                for x, y_true in tqdm(val_loader, desc="Validating"):
                    x = x.to(device)
                    y_true = y_true.to(device)
    
                    mu_d_device  = best_checkpoint['mu_mean'].to(device)
                    std_d_device = best_checkpoint['mu_std'].to(device)
    
                    # generate K samples with different seeds, decode each
                    y_samples = []
                    for s in EVAL_SEEDS:
                        z_pred_norm = diff_model.generate(
                            context_series=x,
                            horizon=y_true.size(1),
                            num_inference_steps=100,
                            guidance_strength=0.1,
                            eta=0.0,          # deterministic DDIM given seed
                            seed=s,           # different seed per sample
                        )  # [B, L, D_latent]
    
                        z_pred_raw = z_pred_norm * std_d_device + mu_d_device
                        y_samples.append(vae.decoder(z_pred_raw))   # [B, L, D_out]
    
                    y_samples = torch.stack(y_samples, dim=0)       # [S, B, L, D_out]
    
                    # For MSE, aggregate with the MEAN in output space
                    y_pred = y_samples.mean(dim=0)                  # [B, L, D_out]
    
                    # accumulate sum over batch to average later
                    val_reg_loss += torch.nn.functional.mse_loss(
                        y_pred, y_true, reduction='sum'
                    ).item()
    
            avg_val_reg_loss = val_reg_loss / val_size
            print(f"\n--- Final Validation Results ---")
            print(f"Using Best Model from Epoch: {best_checkpoint['epoch']}")
            print(f"Final Validation Regression MSE: {avg_val_reg_loss:.6f}")
            print("---------------------------------")
    
        print("Stopping training due to early stopping.")
        break






