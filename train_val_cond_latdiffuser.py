import torch, os
from latent_vae_utils import prepare_data_and_cache, load_dataloaders_from_cache, latent_space_check, normalize_and_check
from latent_vae import LatentVAE
from cond_latent_diffuser import LapDiT
from cond_diffusion_utils import NoiseScheduler, classify_latent, regress_latent_grad
from tqdm import tqdm


TICKERS = [
    'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ', 'V',
    'NVDA', 'HD', 'PG', 'MA', 'DIS', 'BAC', 'XOM', 'PFE', 'ADBE', 'CSCO'
]
START, END = "2020-01-01", "2024-12-31"
WINDOW = 60
FEATURES = ['Open', 'High', 'Low', 'Close']
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-3
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

# --- Load pre-trained VAE ---
vae = LatentVAE(
    input_dim=len(FEATURES), seq_len=WINDOW,
    latent_dim=64,
    enc_layers=3, enc_heads=4, enc_ff=256,
    dec_layers=3, dec_heads=4, dec_ff=256,
    lap_k=32
).to(device)
vae.load_state_dict(torch.load(
    './ldt/saved_model/recon_0.4007_epoch_15.pt',
    map_location=device
))
vae.eval()
# Freeze encoder parameters
for param in vae.encoder.parameters():
    param.requires_grad = False

# --- Compute global μ statistics for normalization ---
all_mu = []
with torch.no_grad():
    for x, _ in train_loader:
        x = x.to(device)
        _, mu, _ = vae(x)
        all_mu.append(mu.cpu())
all_mu = torch.cat(all_mu, dim=0)  # [N, L, D]
_, mu_d, std_d = normalize_and_check(all_mu)
# --- Diffusion model hyperparameters ---
latent_dim = all_mu.size(-1)
cond_embed_dim = 128  # embedding dimension for class labels
time_embed_dim = 128  # embedding dimension for timesteps
lap_kernel = 32
num_layers = 4
num_heads = 8

# --- Instantiate diffusion model ---
diff_model = LapDiT(
    latent_dim=latent_dim,
    time_embed_dim=time_embed_dim,
    lap_kernel= lap_kernel,
    cond_embed_dim=cond_embed_dim,
    num_layers=num_layers,
    num_heads=num_heads,
    num_classes=2
).to(device)

# Noise scheduler with default 1000 timesteps
noise_scheduler = NoiseScheduler(timesteps=1200)
optimizer = torch.optim.Adam(diff_model.parameters(), lr=LR)

# --- Training loop ---
for epoch in range(1, EPOCHS + 1):
    diff_model.train()
    train_loss = 0.0
    p_uncond = max(0.3, epoch / 60)
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
        x = x.to(device)
        y = y.to(device).long()

        # 1) Encode and normalize μ
        with torch.no_grad():
            _, mu, _ = vae(x)
        mu = (mu - mu_d.to(device)) / std_d.to(device)

        # 2) Sample timesteps & noise
        timesteps = torch.randint(
            0, noise_scheduler.timesteps,
            (mu.size(0),), device=device
        ).long()
        noise = torch.randn_like(mu)

        # 3) Create noisy input
        noisy_mu, actual_noise = noise_scheduler.q_sample(mu, timesteps, noise)

        # 4) Decide whether to drop conditioning this batch
        if torch.rand(1).item() < p_uncond:
            cond_labels = None
        else:
            cond_labels = y

        # 5) Predict noise with (maybe) dropped labels
        noise_pred = diff_model(noisy_mu, timesteps, class_labels=cond_labels)

        # 6) Compute loss & step
        loss = torch.nn.functional.mse_loss(noise_pred, actual_noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * mu.size(0)

    avg_train_loss = train_loss / train_size
    print(f"Epoch {epoch} train loss: {avg_train_loss:.6f}")

    # --- Validation ---
    # 1) Diffusion validation loss
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device).long()

            # encode + normalize
            _, mu, _ = vae(x)
            mu = (mu - mu_d.to(device)) / std_d.to(device)

            # sample noise & timestep
            t = torch.randint(0, noise_scheduler.timesteps, (mu.size(0),), device=device)
            noise = torch.randn_like(mu)
            noisy_mu, actual_noise = noise_scheduler.q_sample(mu, t, noise)

            # always condition during val-loss computation
            noise_pred = diff_model(noisy_mu, t, class_labels=y)
            val_loss += torch.nn.functional.mse_loss(noise_pred, actual_noise).item() * mu.size(0)

    avg_val_loss = val_loss / val_size


    # 2) Classification accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            # 2a) encode + normalize
            _, mu, _ = vae(x)
            mu = (mu - mu_d.to(device)) / std_d.to(device)

            # 2b) predict class by lowest denoising error
            preds = classify_latent(mu, noise_scheduler, diff_model, num_class=2, num_trials=20)

            correct += (preds == y).sum().item()
            total += y.size(0)

    val_acc = correct / total

    print(
        f"Epoch {epoch}  train loss: {avg_train_loss:.6f} |  val acc: {val_acc:.4f}") #val loss: {avg_val_loss:.6f}"
#
#     # 3) Regression performance
#     val_reg_loss = 0
#     mean_y = (
#             sum(y.sum().item() for _, y in val_loader)
#             / sum(y.numel() for _, y in val_loader)
#     )
#     for x, y in tqdm(val_loader, desc="Validating"):
#         x = x.to(device)
#         y = y.to(device)
#
#         # 2a) encode + normalize
#         with torch.no_grad():
#             _, mu, _ = vae(x)
#             mu = (mu - mu_mean.to(device)) / mu_std.to(device)
#
#         # 2b) predict class by lowest denoising error
#         preds = regress_latent_grad(
#             mu,
#             noise_scheduler,
#             diff_model,
#             init_y=mean_y,
#             show_progress=False
#         )
#
#         val_reg_loss += torch.nn.functional.mse_loss(y, preds).item() * mu.size(0)
#
#     avg_val_reg_loss = val_reg_loss / val_size
#
#     print(
#         f"Epoch {epoch}  train loss: {avg_train_loss:.6f}  |  val loss: {avg_val_loss:.6f}  |  val reg loss: {avg_val_reg_loss:.4f}")
#
#     # --- Checkpoint ---
#     ckpt_path = os.path.join(CHECKPOINT_DIR, f"diff_epoch_{epoch:03d}_train_{avg_train_loss:.4f}_val_{avg_val_loss:.4f}.pt")
#     torch.save({
#         'epoch': epoch,
#         'model_state': diff_model.state_dict(),
#         'optimizer_state': optimizer.state_dict(),
#         'mu_mean': mu_mean,
#         'mu_std': mu_std
#     }, ckpt_path)
#
# print("Training complete.")

