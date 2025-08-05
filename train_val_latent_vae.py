from latent_vae_utils import *
from latent_vae import LatentVAE
import torch.nn.functional as F
import torch


TICKERS = [
    'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ', 'V',
    'NVDA', 'HD', 'PG', 'MA', 'DIS', 'BAC', 'XOM', 'PFE', 'ADBE', 'CSCO'
]
START, END = "2020-01-01", "2024-12-31"
WINDOW = 60
FEATURES = ['Open', 'High', 'Low', 'Close']
BATCH_SIZE = 32
EPOCHS = 200
LR = 5e-4
VAL_FRAC = 0.1
TEST_FRAC = 0.4
DATA_DIR = './ldt/data'
# --- Data Preparation ---
prepare_data_and_cache(
  TICKERS, START, END, FEATURES, WINDOW,
  data_dir=DATA_DIR,
  close_feature='Close',
  classification=True,
  val_ratio=VAL_FRAC,
  test_ratio=TEST_FRAC
)

train_loader, val_loader, test_loader, (train_size, val_size, test_size) = \
  load_dataloaders_from_cache(
    batch_size=BATCH_SIZE,
    data_dir=DATA_DIR,
    classification=True
  )

# --- Model Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = LatentVAE(
    input_dim=len(FEATURES), seq_len=WINDOW,
    latent_dim=64,
    enc_layers=3, enc_heads=4, enc_ff=256,
    dec_layers=3, dec_heads=4, dec_ff=256,
    lap_k=32
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# --- Training Setup ---
# Define a directory to store the models
model_dir = './ldt/saved_model'
os.makedirs(model_dir, exist_ok=True) # Ensure the directory exists

# Track the paths of the current best models to delete them later
current_best_elbo_path = None
current_best_recon_path = None

# Best values and patience counters for each metric
best_val_elbo = float('inf')
best_val_recon = float('inf')
max_patience = 5
patience_counter_elbo = 0
patience_counter_recon = 0

print("Starting training...")
# --- Training Loop ---
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_recon, total_kl = 0.0, 0.0
    # Linear KL weight annealing (β-VAE)
    beta = min(1, epoch / 500) # Assuming an annealing period of 60 epochs

    for x, y in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        x_hat, mu, logvar = model(x)

        # VAE Loss Calculation
        recon_loss = F.mse_loss(x_hat, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = (recon_loss + beta * kl_div) / x.size(0)

        loss.backward()
        optimizer.step()

        total_recon += recon_loss.item()
        total_kl += kl_div.item()

    # --- Validation ---
    model.eval()
    val_recon_sum, val_kl_sum = 0.0, 0.0
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            val_recon_sum += F.mse_loss(x_hat, x, reduction='sum').item()
            val_kl_sum += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).item()

    # --- Logging and Checkpointing ---
    train_recon_avg = total_recon / train_size
    train_kl_avg = total_kl / train_size
    val_recon_avg = val_recon_sum / val_size
    val_kl_avg = val_kl_sum / val_size

    train_elbo = train_recon_avg + train_kl_avg
    val_elbo = val_recon_avg + val_kl_avg

    print(
        f"Epoch {epoch}/{EPOCHS} - β={beta:.3f} | Train ELBO: {train_elbo:.4f} (Recon: {train_recon_avg:.4f} / per-dim: {train_recon_avg/x_hat.shape[-1]/x_hat.shape[-2]:.4f}, KL: {train_kl_avg:.4f} / per-dim: {train_kl_avg/64:.4f}) | "
        f"Val ELBO: {val_elbo:.4f} (Recon: {val_recon_avg:.4f}, KL: {val_kl_avg:.4f})")

    # --- DYNAMIC CHECKPOINTING AND EARLY STOPPING ---

    # 1. Check and save based on ELBO
    # if val_elbo < best_val_elbo:
    #     if current_best_elbo_path and os.path.exists(current_best_elbo_path):
    #         os.remove(current_best_elbo_path)  # Delete old best model
    #
    #     best_val_elbo = val_elbo
    #     patience_counter_elbo = 0
    #     new_path = os.path.join(model_dir, f"elbo_{val_elbo:.4f}_epoch_{epoch}.pt")
    #     torch.save(model.state_dict(), new_path)
    #     current_best_elbo_path = new_path  # Update path to the new best
    #     print(f"  -> New best ELBO model saved: {os.path.basename(new_path)}")
    # else:
    #     patience_counter_elbo += 1

    # 2. Check and save based on Reconstruction Loss
    if val_recon_avg < best_val_recon:
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
    if patience_counter_elbo >= max_patience or patience_counter_recon >= max_patience:
        print(f"\nEarly stopping triggered at epoch {epoch}: Both metrics haven't improved in {max_patience} epochs.")
        break

print(f"\nTraining complete. Best Val ELBO: {best_val_elbo:.4f}, Best Val Recon: {best_val_recon:.4f}")


# model.load_state_dict(torch.load(model_path, map_location=device))
# model.eval()
# # ------Testing--------
# test_recon_sum, test_kl_sum = 0.0, 0.0
# with torch.no_grad():
#     for x, y in test_loader:
#         x = x.to(device)
#         x_hat, mu, logvar = model(x)
#         test_recon_sum += F.mse_loss(x_hat, x, reduction='sum').item()
#         test_kl_sum += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).item()
# test_elbo = (test_recon_sum + test_kl_sum) / test_size
# print(f"Val ELBO: {test_elbo:.4f} (Recon: {test_recon_sum / test_size:.4f}, KL: {test_kl_sum / test_size:.4f})")

# # ------Visualizing on Test Sets--------
# x_batch, _ = next(iter(test_loader))
# x_batch = x_batch.to(device)
# with torch.no_grad():
#     x_hat, mu, logvar = model(x_batch)
# feature_idx = FEATURES.index('Open')
# n = min(100, x_batch.size(0))
# for i in range(n):
#     fig = plt.figure()            # now plt.figure exists
#     plt.plot(
#         x_batch[i, :, feature_idx].cpu().numpy(),
#         label='Original'
#     )
#     plt.plot(
#         x_hat[i, :, feature_idx].cpu().numpy(),
#         label='Reconstruction'
#     )
#     plt.title(f'Sample {i} – Close Price Reconstruction')
#     plt.xlabel('Time step')
#     plt.ylabel('Price')
#     plt.legend()
#     plt.show()