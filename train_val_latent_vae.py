from latent_vae_utils import *
from latent_vae import LatentVAE
import torch.nn.functional as F
import torch
import os

# -------------------- Config --------------------
TICKERS = [
    'AAPL','MSFT','AMZN','NVDA','META','GOOGL','GOOG','TSLA','AVGO','ADBE',
    'ADSK','ADP','ALGN','AMD','AMGN','AMAT','ANSS','ASML','AZN','BKNG','BKR','CDNS','CEG','CHTR','CMCSA','COST','CPRT',
    'CRWD','CSCO','CSX','CTAS','CTSH','DASH','DDOG','DLTR','DXCM','EA','EBAY','ENPH','EXC','FANG','FAST','FTNT','GEHC',
    'GFS','GILD','HON','IDXX','ILMN','INTC','INTU','ISRG','KDP','KHC','KLAC','LRCX','LULU','MAR','MCHP','MDLZ','MELI',
    'MNST','MRNA','MRVL','MU','NFLX','ODFL','ON','ORLY','PANW','PAYX','PCAR','PDD','PEP','PYPL','QCOM','REGN','ROP',
    'ROST','SBUX','SIRI','SNPS','TEAM','TMUS','TXN','VRSK','VRTX','WBA','WBD','WDAY','XEL','ZM','ZS'
]
START, END = "2020-01-01", "2024-12-31"
WINDOW = 60
PRED = 10
FEATURES = ['Open','High','Low','Close']
PRED_FEAT = ['Close']
BATCH_SIZE = 64
EPOCHS = 500
LR = 5e-4
max_patience = 20
VAL_FRAC = 0.1
TEST_FRAC = 0.4
DATA_DIR = './ldt/data'

# -------------------- Data --------------------
prepare_data_and_cache(
  TICKERS, START, END, FEATURES, WINDOW,
  data_dir=DATA_DIR, close_feature='Close',
  val_ratio=VAL_FRAC, test_ratio=TEST_FRAC, horizon=PRED
)

train_loader, val_loader, test_loader, (train_size, val_size, test_size) = \
  load_dataloaders_from_cache(batch_size=BATCH_SIZE, data_dir=DATA_DIR)

# -------------------- Model --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = LatentVAE(
    input_dim=len(PRED_FEAT), seq_len=PRED,
    latent_dim=64,
    enc_layers=3, enc_heads=4, enc_ff=256,
    dec_layers=3, dec_heads=4, dec_ff=256
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# AMP scaler (safe on CPU too; it’ll just be disabled)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

# -------------------- Checkpointing --------------------
model_dir = './ldt/saved_model'
os.makedirs(model_dir, exist_ok=True)  # <- fix: import os

current_best_recon_path = None
best_val_recon = float('inf')
patience_counter_recon = 0

print("Starting training.")
# -------------------- Train Loop --------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_recon, total_kl = 0.0, 0.0

    # β-VAE annealing toward 0.5 over 500 epochs
    annealing_period = 500
    beta = min(0.5, epoch / annealing_period)

    for _, y in train_loader:
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            y_hat, mu, logvar = model(y)
            # losses (sum over all elements; divide by B for average per-sample)
            recon_loss = F.mse_loss(y_hat, y, reduction='sum')
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (recon_loss + beta * kl_div) / y.size(0)

        scaler.scale(loss).backward()
        # unscale before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_recon += recon_loss.item()
        total_kl += kl_div.item()

    # -------------------- Validation --------------------
    model.eval()
    val_recon_sum, val_kl_sum = 0.0, 0.0
    with torch.no_grad():
        for _, y in val_loader:
            y = y.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                y_hat, mu, logvar = model(y)
                val_recon_sum += F.mse_loss(y_hat, y, reduction='sum').item()
                val_kl_sum += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).item()

    # -------------------- Logging --------------------
    train_recon_avg = total_recon / train_size
    train_kl_avg = total_kl / train_size
    val_recon_avg = val_recon_sum / val_size
    val_kl_avg = val_kl_sum / val_size

    # per-element numbers (output is PRED x 1)
    per_elem_train_recon = train_recon_avg / (PRED * 1)
    per_elem_val_recon = val_recon_avg / (PRED * 1)
    per_elem_train_kl = train_kl_avg / (PRED * 64)  # per (time, latent-dim)
    per_elem_val_kl = val_kl_avg / (PRED * 64)

    train_elbo = train_recon_avg + train_kl_avg
    val_elbo = val_recon_avg + val_kl_avg

    print(
        f"Epoch {epoch}/{EPOCHS} - β={beta:.3f} | "
        f"Train ELBO: {train_elbo:.4f} (Recon: {train_recon_avg:.4f} /elem: {per_elem_train_recon:.6f}, "
        f"KL: {train_kl_avg:.4f} /elem: {per_elem_train_kl:.6f}) | "
        f"Val ELBO: {val_elbo:.4f} (Recon: {val_recon_avg:.4f} /elem: {per_elem_val_recon:.6f}, "
        f"KL: {val_kl_avg:.4f} /elem: {per_elem_val_kl:.6f})"
    )

    # -------------------- Checkpointing + Early Stop --------------------
    if val_recon_avg < best_val_recon:
        # delete previous best
        if current_best_recon_path and os.path.exists(current_best_recon_path):
            os.remove(current_best_recon_path)

        best_val_recon = val_recon_avg
        patience_counter_recon = 0
        new_path = os.path.join(model_dir, f"recon_{val_recon_avg:.4f}_epoch_{epoch}.pt")
        torch.save(model.state_dict(), new_path)
        current_best_recon_path = new_path
        print(f"  -> New best Recon model saved: {os.path.basename(new_path)}")
    else:
        patience_counter_recon += 1

    if patience_counter_recon >= max_patience:
        print(f"\nEarly stopping at epoch {epoch}: Recon hasn't improved in {max_patience} epochs.")
        break

print(f"\nTraining complete. Best Val Recon: {best_val_recon:.4f}")

