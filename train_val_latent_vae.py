from latent_vae_utils import *
from latent_vae import LatentVAE
import torch.nn.functional as F
import torch


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
max_patience = 20
VAL_FRAC = 0.1
TEST_FRAC = 0.4
DATA_DIR = './ldt/data'
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

# --- Model Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = LatentVAE(
    input_dim=1, seq_len=PRED,
    latent_dim=64,
    enc_layers=3, enc_heads=4, enc_ff=256,
    dec_layers=3, dec_heads=4, dec_ff=256
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
# --- Training Setup ---
# Define a directory to store the models
model_dir = './ldt/saved_model'
os.makedirs(model_dir, exist_ok=True) # Ensure the directory exists

# Track the paths of the current best models to delete them later
current_best_elbo_path = None
current_best_recon_path = None

# Best values and patience counters for each metric
best_val_recon= float('inf')
patience_counter_recon = 0

print("Starting training...")
# --- Training Loop ---
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_recon, total_kl = 0.0, 0.0
    # Linear KL weight annealing (β-VAE)
    annealing_period = 500  # The number of epochs to reach the full KL penalty
    beta = min(0.5, epoch / annealing_period)

    for _, y in train_loader:
        y = y.to(device)
        optimizer.zero_grad()
        y_hat, mu, logvar = model(y)

        # VAE Loss Calculation
        recon_loss = F.mse_loss(y_hat, y, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = (recon_loss + beta * kl_div) / y.size(0)

        loss.backward()
        optimizer.step()

        total_recon += recon_loss.item()
        total_kl += kl_div.item()

    # --- Validation ---
    model.eval()
    val_recon_sum, val_kl_sum = 0.0, 0.0
    with torch.no_grad():
        for _, y in val_loader:
            y = y.to(device)
            y_hat, mu, logvar = model(y)
            val_recon_sum += F.mse_loss(y_hat, y, reduction='sum').item()
            val_kl_sum += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).item()

    # --- Logging and Checkpointing ---
    train_recon_avg = total_recon / train_size
    train_kl_avg = total_kl / train_size
    val_recon_avg = val_recon_sum / val_size
    val_kl_avg = val_kl_sum / val_size

    train_elbo = train_recon_avg + train_kl_avg
    val_elbo = val_recon_avg + val_kl_avg

    print(
        f"Epoch {epoch}/{EPOCHS} - β={beta:.3f} | Train ELBO: {train_elbo:.4f} (Recon: {train_recon_avg:.4f} / per-dim: {train_recon_avg/y_hat.shape[-1]/y_hat.shape[-2]:.4f}, KL: {train_kl_avg:.4f} / per-dim: {train_kl_avg/64:.4f}) | "
        f"Val ELBO: {val_elbo:.4f} (Recon: {val_recon_avg:.4f}, KL: {val_kl_avg:.4f})")

    # --- DYNAMIC CHECKPOINTING AND EARLY STOPPING ---
    if val_recon_avg < best_val_recon:# and val_kl_avg >= 0.01:
        if current_best_recon_path and os.path.exists(current_best_recon_path):
            os.remove(current_best_recon_path)  # Delete old best model

        best_val_recon = val_recon_avg
        patience_counter_recon = 0
        new_path = os.path.join(model_dir, f"recon_{val_recon_avg:.4f}_epoch_{epoch}.pt")
        torch.save(model.state_dict(), new_path)
        current_best_recon_path = new_path  # Update path to the new best
        print(f"  -> New best Recon model saved: {os.path.basename(new_path)}")
    else:
        patience_counter_recon += 1

    # 3. Check the early stopping condition
    if patience_counter_recon >= max_patience:
        print(f"\nEarly stopping triggered at epoch {epoch}: Both metrics haven't improved in {max_patience} epochs.")
        break

print(f"\nTraining complete. Best Val Recon: {best_val_recon:.4f}")

