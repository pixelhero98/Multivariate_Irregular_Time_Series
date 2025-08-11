import torch, os
from latent_vae_utils import prepare_data_and_cache, load_dataloaders_from_cache, normalize_and_check
from latent_vae import LatentVAE
from cond_latent_diffuser import LapDiT
from cond_diffusion_utils import NoiseScheduler
import numpy as np
from tqdm import tqdm


TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'GOOGL', 'GOOG', 'TSLA', 'AVGO', 'ADBE',
    'ADSK', 'ADP', 'ALGN', 'AMD', 'AMGN', 'AMAT', 'ANSS', 'ASML',
    'AZN', 'BKNG', 'BKR', 'CDNS', 'CEG', 'CHTR', 'CMCSA', 'COST', 'CPRT',
    'CRWD', 'CSCO', 'CSX', 'CTAS', 'CTSH', 'DASH', 'DDOG', 'DLTR', 'DXCM', 'EA',
    'EBAY', 'ENPH', 'EXC', 'FANG', 'FAST', 'FTNT', 'GEHC', 'GFS', 'GILD',
    'HON', 'IDXX', 'ILMN', 'INTC', 'INTU', 'ISRG', 'KDP', 'KHC', 'KLAC', 'LRCX',
    'LULU', 'MAR', 'MCHP', 'MDLZ', 'MELI', 'MNST', 'MRNA', 'MRVL', 'MU', 'NFLX',
    'ODFL', 'ON', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD', 'PEP', 'PYPL', 'QCOM',
    'REGN', 'ROP', 'ROST', 'SBUX', 'SIRI', 'SNPS', 'TEAM', 'TMUS', 'TXN', 'VRSK',
    'VRTX', 'WBA', 'WBD', 'WDAY', 'XEL', 'ZM', 'ZS'
]
START, END = "2020-01-01", "2024-12-31"
WINDOW = 60
PRED = 10
FEATURES = ['Open', 'High', 'Low', 'Close']
BATCH_SIZE = 64
EPOCHS = 500
LR = 5e-4
VAL_FRAC = 0.1
TEST_FRAC = 0.4
DATA_DIR = './ldt/data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = './ldt/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Data Preparation ---
prepare_data_and_cache(
  TICKERS, START, END, FEATURES, WINDOW,
  data_dir=DATA_DIR,
  close_feature='Close',
  val_ratio=VAL_FRAC,
  test_ratio=TEST_FRAC,
  horizon=PRED
)

train_loader, val_loader, test_loader, (train_size, val_size, test_size) = \
  load_dataloaders_from_cache(
    batch_size=BATCH_SIZE,
    data_dir=DATA_DIR,
  )

# --- Load pre-trained VAE ---
vae = LatentVAE(
    input_dim=1, seq_len=PRED,
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

# -------------------------------------------
latent_dim = all_mu.size(-1)
time_embed_dim = 256  # embedding dimension for timesteps
lap_kernel = 128
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
current_best_ckpt_path = "./ldt/checkpoints/best_model_epoch_107_val_0.0120.pt"
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
        mu = (mu - mu_d.to(device)) / std_d.to(device)
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
            noise_pred = diff_model(noisy_mu, timesteps, cond_tensor)
            loss = torch.nn.functional.mse_loss(noise_pred, actual_noise)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(diff_model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.step()
        train_loss += loss.item() * mu.size(0)

    avg_train_loss = train_loss / train_size
    # --- Validation ---
    val_loss = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            _, mu, _ = vae(y)
            mu = (mu - mu_d.to(device)) / std_d.to(device)
            mu = torch.clamp(mu, min=-5, max=5)

            cond_tensor = x
            t = torch.randint(0, noise_scheduler.timesteps, (mu.size(0),), device=device)
            noise = torch.randn_like(mu)
            noisy_mu, actual_noise = noise_scheduler.q_sample(mu, t, noise)
            noise_pred = diff_model(noisy_mu, t, cond_tensor)
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
    
            val_reg_loss = 0.0
            with torch.no_grad():
                for x, y_true in tqdm(val_loader, desc="Validating"):
                    x = x.to(device)
                    y_true = y_true.to(device)
    
                    # Multiple samples for robustness
                    output_samples_latent = []
                    for _ in range(N_INF_SAMPLES):
                        z_pred_sample = diff_model.generate(
                            context_series=x,
                            horizon=y_true.size(1),
                            num_inference_steps=100,
                            guidance_strength=0.2,
                        )
                        output_samples_latent.append(z_pred_sample)
    
                    stacked = torch.stack(output_samples_latent)  # [N_INF_SAMPLES, B, L, D]
                    z_pred_norm = stacked.median(dim=0).values    # [B, L, D]
                    mu_d_device = best_checkpoint['mu_mean'].to(device)
                    std_d_device = best_checkpoint['mu_std'].to(device)
                    z_pred_raw = z_pred_norm * std_d_device + mu_d_device
                    y_pred = vae.decoder(z_pred_raw)
    
                    val_reg_loss += torch.nn.functional.mse_loss(y_pred, y_true).item() * y_true.size(0)
    
            avg_val_reg_loss = val_reg_loss / val_size
            print(f"\n--- Final Validation Results ---")
            print(f"Using Best Model from Epoch: {best_checkpoint['epoch']}")
            print(f"Final Validation Regression MSE: {avg_val_reg_loss:.6f}")
            print("---------------------------------")
    
        print("Stopping training due to early stopping.")
        break

