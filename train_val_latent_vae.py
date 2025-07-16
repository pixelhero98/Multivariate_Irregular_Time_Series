import yfinance as yf
import numpy as np
import torch.nn.functional as F
import torch
import os
import json
from torch.utils.data import Dataset, DataLoader
from latent_vae import LatentGTVAE


class WindowDataset(Dataset):
        def __init__(self, array): self.array = array
        def __len__(self): return len(self.array)
        def __getitem__(self, idx): return torch.from_numpy(self.array[idx]).float()

def prepare_dataloaders(
    tickers, start, end, features, window, batch_size,
    val_fraction=0.1, test_fraction=0.1,
    data_dir='./data'):
    """
    Downloads and preprocesses financial data or loads from cached .npy files.
    Constructs sliding windows, splits chronologically, normalizes based on the
    training set, and returns DataLoaders and split sizes.

    Args:
        tickers (list of str): List of ticker symbols.
        start, end (str): Date strings for yfinance.
        features (list of str): OHLC feature names.
        window (int): Sliding window length.
        batch_size (int): DataLoader batch size.
        val_fraction, test_fraction (float): Fractions for splits.
        data_dir (str): Directory path to load/save dataset files.
    """
    os.makedirs(data_dir, exist_ok=True)
    # Define file paths for data and metadata
    train_file = os.path.join(data_dir, 'train_data.npy')
    val_file = os.path.join(data_dir, 'val_data.npy')
    test_file = os.path.join(data_dir, 'test_data.npy')
    meta_file = os.path.join(data_dir, 'meta.json')

    # Check if all necessary cached files exist
    if all(os.path.exists(f) for f in [train_file, val_file, test_file, meta_file]):
        print("Loading pre-split and normalized data from cache...")
        train_array = np.load(train_file)
        val_array = np.load(val_file)
        test_array = np.load(test_file)
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        
        # --- Integrity Checks ---
        # Ensure cached data parameters match current parameters
        assert meta['window'] == window, "Cached window size mismatch."
        assert meta['features'] == features, "Cached features mismatch."
        
        # **IMPROVED**: Verify that the number of loaded samples matches the number from when the cache was created
        total_loaded = train_array.shape[0] + val_array.shape[0] + test_array.shape[0]
        assert meta['total_windows'] == total_loaded, (
            f"Cached data sample count mismatch. Expected {meta['total_windows']}, "
            f"but loaded {total_loaded}."
        )
        print("Cached data loaded successfully and passed integrity checks.")

    else:
        print("Processing data from scratch: downloading, splitting, and normalizing...")
        # Download data for all tickers
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True)[features]
        
        windows = []
        for ticker in tickers:
            # Extract data for a single ticker. Use .xs for robust multi-index access.
            df = raw.xs(ticker, axis=1, level=1) if len(tickers) > 1 else raw
            df = df.dropna() # Drop rows with missing values for the current ticker
            
            dates = df.index
            # Apply log transform to stabilize variance and handle price scales
            arr = np.log1p(df.values)

            # Create sliding windows if there's enough data
            if len(arr) >= window:
                for i in range(len(arr) - window + 1):
                    # Append a tuple of (window_data, end_date_of_window)
                    window_end_date = dates[i + window - 1]
                    windows.append((arr[i:i+window], window_end_date))
        
        # Sort all windows from all tickers chronologically by their end date
        # This is crucial for a correct time-based train/val/test split
        windows.sort(key=lambda x: x[1])
        data_array = np.stack([item[0] for item in windows])

        # --- Chronological Split ---
        total_windows = len(data_array)
        val_size = int(val_fraction * total_windows)
        test_size = int(test_fraction * total_windows)
        train_size = total_windows - val_size - test_size
        print(f"Total windows: {total_windows} | Train: {train_size}, Val: {val_size}, Test: {test_size}")

        train_array = data_array[:train_size]
        val_array = data_array[train_size : train_size + val_size]
        test_array = data_array[train_size + val_size:]

        # --- Normalization ---
        # Calculate mean and std ONLY from the training data to prevent data leakage
        mu, sigma = train_array.mean((0, 1), keepdims=True), train_array.std((0, 1), keepdims=True) + 1e-6
        
        # Apply the calculated normalization to all splits
        train_array = (train_array - mu) / sigma
        val_array = (val_array - mu) / sigma
        test_array = (test_array - mu) / sigma
        
        # --- Save to Cache ---
        print("Saving processed data to cache...")
        np.save(train_file, train_array)
        np.save(val_file, val_array)
        np.save(test_file, test_array)
        # Save metadata for future integrity checks
        meta = {'window': window, 'features': features, 'total_windows': total_windows}
        with open(meta_file, 'w') as f:
            json.dump(meta, f)

    # Create datasets and dataloaders
    train_ds = WindowDataset(train_array)
    val_ds = WindowDataset(val_array)
    test_ds = WindowDataset(test_array)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader, (len(train_ds), len(val_ds), len(test_ds))

if __name__ == '__main__':
    # --- Configuration ---
    TICKERS = [
        'AAPL','MSFT','GOOG','AMZN','META','TSLA','BRK-B','JPM','JNJ','V',
        'NVDA','HD','PG','MA','DIS','BAC','XOM','PFE','ADBE','CSCO'
    ]
    START, END = "2020-01-01", "2024-12-31"
    WINDOW = 30
    FEATURES = ['Open', 'High', 'Low', 'Close']
    BATCH_SIZE = 32
    EPOCHS = 200
    LR = 1e-3
    VAL_FRAC = 0.15
    TEST_FRAC = 0.15
    DATA_DIR = './data'

    # --- Data Preparation ---
    train_loader, val_loader, test_loader, (train_size, val_size, test_size) = \
        prepare_dataloaders(
            TICKERS, START, END, FEATURES, WINDOW, BATCH_SIZE,
            val_fraction=VAL_FRAC, test_fraction=TEST_FRAC, data_dir=DATA_DIR
        )

    # --- Model Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # **FIXED**: Instantiate the correct model class `LatentGTVAE`
    model = LatentGTVAE(
        input_dim=len(FEATURES), seq_len=WINDOW,
        latent_dim=64,
        enc_layers=3, enc_heads=4, enc_ff=256,
        gt_layers=2, gt_heads=8, gt_k=8,
        dec_layers=2, dec_heads=4, dec_ff=256
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    
    # --- Training Setup ---
    best_val_elbo = float('inf')
    patience, max_patience = 0, 20
    model_path = 'best_model.pt'

    print("Starting training...")
    # --- Training Loop ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_recon, total_kl = 0.0, 0.0
        # Linear KL weight annealing (β-VAE)
        beta = min(1.0, epoch / 100.0)
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(batch)
            
            # VAE Loss Calculation
            recon_loss = F.mse_loss(x_hat, batch, reduction='sum')
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (recon_loss + beta * kl_div) / batch.size(0)
            
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            optimizer.step()
            
            total_recon += recon_loss.item()
            total_kl += kl_div.item()

        # --- Validation ---
        model.eval()
        val_recon_sum, val_kl_sum = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x_hat, mu, logvar = model(batch)
                val_recon_sum += F.mse_loss(x_hat, batch, reduction='sum').item()
                val_kl_sum += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).item()

        # --- Logging and Checkpointing ---
        train_recon_avg = total_recon / train_size
        train_kl_avg = total_kl / train_size
        val_recon_avg = val_recon_sum / val_size
        val_kl_avg = val_kl_sum / val_size
        
        train_elbo = train_recon_avg + train_kl_avg
        val_elbo = val_recon_avg + val_kl_avg
        
        print(f"Epoch {epoch}/{EPOCHS} - β={beta:.2f} | Train ELBO: {train_elbo:.4f} (Recon: {train_recon_avg:.4f}, KL: {train_kl_avg:.4f}) | "
              f"Val ELBO: {val_elbo:.4f} (Recon: {val_recon_avg:.4f}, KL: {val_kl_avg:.4f})")

        if val_elbo < best_val_elbo:
            best_val_elbo = val_elbo
            torch.save(model.state_dict(), model_path)
            patience = 0
            print(f"  -> New best model saved (Val ELBO {val_elbo:.4f})")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    print(f"\nTraining complete. Best Val ELBO: {best_val_elbo:.4f}")
