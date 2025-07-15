import yfinance as yf
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Config: list company tickers
TICKERS = [
    'AAPL','MSFT','GOOG','AMZN','META','TSLA','BRK-B','JPM','JNJ','V',
    'NVDA','HD','PG','MA','DIS','BAC','XOM','PFE','ADBE','CSCO'
]
START, END = "2020-01-01", "2024-12-31"
WINDOW = 30
FEATURES = ['Open', 'High', 'Low', 'Close']
BATCH_SIZE = 16
EPOCHS = 200
LR = 1e-3

# Download & preprocess
raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True)[FEATURES]
data = []
for ticker in TICKERS:
    df = raw.xs(ticker, axis=1, level=1)
    arr = np.log1p(df.values)
    mu, sigma = arr.mean(0, keepdims=True), arr.std(0, keepdims=True) + 1e-6
    arr_norm = (arr - mu) / sigma
    for i in range(len(arr_norm) - WINDOW + 1):
        data.append(arr_norm[i:i+WINDOW])
data = np.stack(data)  # [N, WINDOW, F]

class TimeSeriesDataset(Dataset):
    def __init__(self, array): self.array = array
    def __len__(self): return len(self.array)
    def __getitem__(self, idx): return torch.from_numpy(self.array[idx]).float()

dataset = TimeSeriesDataset(data)
dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True)

# Model, device, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate VAE with Transformer Decoder and GT processor
model = VAEWithTransformerDecoder(
    input_dim=len(FEATURES), seq_len=WINDOW,
    latent_dim=64,
    enc_layers=3, enc_heads=4, enc_ff=256,
    gt_layers=2, gt_heads=8, gt_k=16,
    dec_layers=2, dec_heads=4, dec_ff=256
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# Training loop with KL annealing
model.train()
for epoch in range(1, EPOCHS+1):
    total_recon = 0.0
    total_kl = 0.0
    # Linear KL weight annealing over first 100 epochs
    beta = epoch / 100.0 if epoch < 100 else 1.0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        x_hat, mu, logvar = model(batch)
        # compute losses
        recon = F.mse_loss(x_hat, batch, reduction='sum') / batch.size(0)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(0)
        loss = recon + beta * kl
        loss.backward()
        optimizer.step()
        total_recon += recon.item() * batch.size(0)
        total_kl += kl.item() * batch.size(0)
    N = len(dataset)
    avg_recon = total_recon / N
    avg_kl = total_kl / N
    avg_elbo = avg_recon + avg_kl
    print(f"Epoch {epoch}/{EPOCHS} - β={beta:.2f} Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}, ELBO: {avg_elbo:.4f}")
