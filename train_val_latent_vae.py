import yfinance as yf
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from latent_vae import LatentGTVAE
import os


class WindowDataset(Dataset):
        def __init__(self, array): self.array = array
        def __len__(self): return len(self.array)
        def __getitem__(self, idx): return torch.from_numpy(self.array[idx]).float()


def prepare_dataloaders(
    tickers, start, end, features, window, batch_size,
    val_fraction=0.1, test_fraction=0.1, seed=None,
    data_dir='./data'):
    """
    Download and preprocess financial data or load from cached .npy files in data_dir.
    Constructs sliding windows, splits into train/val/test, and returns DataLoaders and split sizes.

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
    # Filenames
    train_file = os.path.join(data_dir, 'train_data.npy')
    val_file = os.path.join(data_dir, 'val_data.npy')
    test_file = os.path.join(data_dir, 'test_data.npy')

    # Check for cached files\    
    if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
        print("Loading pre-split and normalized data from cache...")
        # Load preprocessed arrays
        train_array = np.load(train_file)
        val_array = np.load(val_file)
        test_array = np.load(test_file)
        # Integrity checks: shapes should match expected dimensions
        for arr, name in [(train_array, 'train'), (val_array, 'val'), (test_array, 'test')]:
            assert arr.ndim == 3, f"{name}_data.npy must be 3D [N, window, features]."
            assert arr.shape[1] == window, f"{name}_data.npy second dimension (window) mismatch."
            assert arr.shape[2] == len(features), f"{name}_data.npy third dimension (features) mismatch."
        # Check non-empty
        assert train_array.shape[0] > 0, "train_data.npy is empty."
        assert val_array.shape[0] > 0, "val_data.npy is empty."
        assert test_array.shape[0] > 0, "test_data.npy is empty."
        # Verify total samples matches expected windows per ticker
        raw_check = yf.download(tickers, start=start, end=end, auto_adjust=True)[features]
        expected = 0
        for ticker in tickers:
            df_check = raw_check.xs(ticker, axis=1, level=1)
            expected += max(0, len(df_check) - window + 1)
        total_loaded = train_array.shape[0] + val_array.shape[0] + test_array.shape[0]
        assert total_loaded == expected, f"Cached data total {total_loaded} does not match expected {expected}."
    else:
        print("Processing data from scratch: downloading, splitting chronologically, and normalizing...")
        # Download & preprocess
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True)[features]
        windows = []
        for ticker in tickers:
            df = raw.xs(ticker, axis=1, level=1)
            dates = df.index
            arr = np.log1p(df.values)

            if len(arr) >= window:
                for i in range(len(arr) - window + 1):
                    window_end_date = dates[i + window - 1]
                    windows.append((arr[i:i+window], window_end_date))
                        
        windows.sort(key=lambda x: x[1])
        data_array = np.stack([item[0] for item in windows])

        # Determine split sizes
        total = len(data_array)
        val_size = int(val_fraction * total)
        test_size = int(test_fraction * total)
        train_size = total - val_size - test_size
        print(f"Total windows: {total_windows} | Train: {train_size}, Val: {val_size}, Test: {test_size}")

        # Shuffle indices
        train_array = data_array[:train_size]
        val_array = data_array[train_size : train_size + val_size]
        test_array = data_array[train_size + val_size : ]
        mu, sigma = train_array.mean(0, keepdims=True), train_array.std(0, keepdims=True) + 1e-6
        arr_norm = (arr - mu) / sigma
        train_array = (train_array - mu) / sigma
        val_array = (val_array - mu) / sigma
        test_array = (test_array - mu) / sigma
        # Save for future
        np.save(train_file, train_array)
        np.save(val_file, val_array)
        np.save(test_file, test_array)

    # Create datasets and loaders
    train_ds = WindowDataset(train_array)
    val_ds = WindowDataset(val_array)
    test_ds = WindowDataset(test_array)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Sizes
    train_size = len(train_ds)
    val_size = len(val_ds)
    test_size = len(test_ds)
            
    return train_loader, val_loader, test_loader, (train_size, val_size, test_size)


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
    VAL_FRAC = 0.1
    TEST_FRAC = 0.45
    SEED = 42
    PATH = './data'

    # Prepare data loaders
    train_loader, val_loader, test_loader, (train_size, val_size, test_size) = \
        prepare_dataloaders(
            TICKERS, START, END, FEATURES, WINDOW, BATCH_SIZE,
            val_fraction=VAL_FRAC, test_fraction=TEST_FRAC, seed=SEED, data_dir=PATH
        )

    # Device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate VAE with Transformer Decoder and GT processor
    model = VAEWithTransformerDecoder(
        input_dim=len(FEATURES), seq_len=WINDOW,
        latent_dim=64,
        enc_layers=3, enc_heads=4, enc_ff=256,
        gt_layers=2, gt_heads=8, gt_k=8,
        dec_layers=2, dec_heads=4, dec_ff=256
    ).to(device)

    # Optimizer with optional weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)  # weight_decay can help prevent overfitting

    # Early stopping and best-model saving setup
    best_val_elbo = float('inf')
    patience, max_patience = 0, 20  # stop if no improvement for 20 epochs

    # Training loop with KL annealing, validation, and checkpointing
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_recon, total_kl = 0.0, 0.0
        # Linear KL weight annealing over first 100 epochs
        beta = epoch / 100.0 if epoch < 100 else 1.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(batch)
            recon = F.mse_loss(x_hat, batch, reduction='sum') / batch.size(0)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(0)
            loss = recon + beta * kl
            loss.backward()
            optimizer.step()
            total_recon += recon.item() * batch.size(0)
            total_kl += kl.item() * batch.size(0)
        # Compute train metrics
        train_recon = total_recon / train_size
        train_kl = total_kl / train_size

        # Validation
        model.eval()
        with torch.no_grad():
            val_recon_sum, val_kl_sum = 0.0, 0.0
            for batch in val_loader:
                batch = batch.to(device)
                x_hat, mu, logvar = model(batch)
                recon = F.mse_loss(x_hat, batch, reduction='sum') / batch.size(0)
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(0)
                val_recon_sum += recon.item() * batch.size(0)
                val_kl_sum += kl.item() * batch.size(0)
            val_recon = val_recon_sum / val_size
            val_kl = val_kl_sum / val_size
            train_elbo = train_recon + train_kl
            val_elbo = val_recon + val_kl
        # Print detailed metrics
        # Corrected print statement
        print(f"Epoch {epoch}/{EPOCHS} - β={beta:.2f} | Train ELBO: {train_elbo:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}) | "
              f"Val ELBO: {val_elbo:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})")
        # Checkpointing
        if val_elbo < best_val_elbo:
            best_val_elbo = val_elbo
            torch.save(model.state_dict(), 'best_model.pt')
            patience = 0
            print(f"  --> New best model saved (Val ELBO {val_elbo:.4f})")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    print("Training complete. Best Val ELBO: {:.4f}".format(best_val_elbo))
