import yfinance as yf
import numpy as np
import torch
import os
import json
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def compute_per_dim_stats(all_mu: torch.Tensor):
    """
    Compute per–latent‐dimension mean & std from raw latents.
    Args:
        all_mu: [N, L, D] raw μ outputs of your encoder.
    Returns:
        mu_per_dim:  [D] tensor of per-dim means
        std_per_dim: [D] tensor of per-dim stds
    """
    mu_per_dim  = all_mu.mean(dim=(0, 1))                  # [D]
    std_per_dim = all_mu.std(dim=(0, 1)).clamp(min=1e-6)  # [D]
    return mu_per_dim, std_per_dim

def normalize_and_check(all_mu: torch.Tensor):
    """
    Normalize per‐dimension and print out global & per‐dim stats.
    Args:
        all_mu: [N, L, D] raw μ outputs of your encoder.
    Returns:
        all_mu_norm: [N, L, D] normalized per‐dim (zero‐mean, unit‐var)
        mu_per_dim:  [D] original per‐dim means
        std_per_dim: [D] original per‐dim stds
    """
    # compute per‐dim stats
    mu_per_dim, std_per_dim = compute_per_dim_stats(all_mu)

    # normalize per‐dimension
    mu_b  = mu_per_dim.view(1, 1, -1)
    std_b = std_per_dim.view(1, 1, -1)
    all_mu_norm = (all_mu - mu_b) / std_b

    # 1) global stats
    all_vals = all_mu_norm.view(-1)
    print(f"Global mean (post‑norm): {all_vals.mean().item():.6f}")
    print(f"Global std  (post‑norm): {all_vals.std().item():.6f}")

    # 2) per‑dim stats
    per_dim_mean = all_mu_norm.mean(dim=(0,1))  # [D]
    per_dim_std  = all_mu_norm.std(dim=(0,1))   # [D]
    D = all_mu_norm.size(-1)
    print("\nPer‑dim stats (first 10 dims):")
    for i in range(D):
        print(f" dim {i:2d}: mean={per_dim_mean[i]:6.3f}, std={per_dim_std[i]:6.3f}")

    # 3) optional histogram
    plt.clf()
    plt.figure(figsize=(4,3))
    plt.hist(all_vals.cpu().numpy(), bins=500, range=(-5, 5))
    plt.title("Histogram of normalized μ values")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.show()
    print(f"Number of NaNs: {torch.isnan(all_mu_norm).sum().item()}")
    print(f"Number of Infs: {torch.isinf(all_mu_norm).sum().item()}")
    print(f"Min value in normalized data: {all_mu_norm.min().item()}")
    print(f"Max value in normalized data: {all_mu_norm.max().item()}")
    return all_mu_norm, mu_per_dim, std_per_dim


class LabeledWindowDataset(Dataset):
    """
    Dataset for sliding windows with labels for regression or classification.
    Returns (x, y) where x is window tensor and y is label tensor.
    """
    def __init__(
        self,
        data_array: np.ndarray,
        labels_array: np.ndarray,
        regression: bool
    ):
        self.data_array = data_array
        self.labels_array = labels_array
        self.regression = regression

    def __len__(self) -> int:
        return len(self.data_array)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.data_array[idx]).float()
        y_np = self.labels_array[idx]
        if not self.regression:
            y = torch.tensor(int(y_np), dtype=torch.int64)
        else:
            y = torch.from_numpy(y_np).float().unsqueeze(-1)
        return x, y

def _cache_sliding_windows(
    tickers,
    start,
    end,
    features,
    window,
    regression,
    close_feature,
    data_dir,
    val_ratio: float = 0.1,
    test_ratio: float = 0.4,
    horizon: int = 24
):
    """
    Generate (data_array, labels_array) and cache them in data_dir.
    Splits data chronologically according to val_ratio and test_ratio.
    Features and labels are raw returns: (x_{t+1} - x_t) / x_t.
    Windows are shaped (n_features, window) with stride=1.
    Classification: positive return → 1, else 0.
    """
    os.makedirs(data_dir, exist_ok=True)
    cache = {
        'train': os.path.join(data_dir, 'train.npy'),
        'val':   os.path.join(data_dir, 'val.npy'),
        'test':  os.path.join(data_dir, 'test.npy'),
        'train_lbl': os.path.join(data_dir, 'train_lbl.npy'),
        'val_lbl':   os.path.join(data_dir, 'val_lbl.npy'),
        'test_lbl':  os.path.join(data_dir, 'test_lbl.npy'),
        'meta':      os.path.join(data_dir, 'meta.json')
    }

    # Check cache
    if all(os.path.exists(p) for p in cache.values()):
        with open(cache['meta'], 'r') as f:
            meta = json.load(f)
        if (
            meta.get('tickers') == tickers and
            meta.get('start') == start and
            meta.get('end') == end and
            meta.get('features') == features and
            meta.get('window') == window and
            meta.get('regression') == regression and
            meta.get('close_feature') == close_feature and
            meta.get('val_ratio') == val_ratio and
            meta.get('test_ratio') == test_ratio and
            meta.get('horizon') == horizon
        ):
            train = np.load(cache['train'])
            val   = np.load(cache['val'])
            test  = np.load(cache['test'])
            tl    = np.load(cache['train_lbl'])
            vl    = np.load(cache['val_lbl'])
            te    = np.load(cache['test_lbl'])
            return (train, tl), (val, vl), (test, te)

    # Fetch raw data and compute returns
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)[features]
    windows = []
    for t in tickers:
        df = raw.xs(t, axis=1, level=1) if len(tickers) > 1 else raw
        df = df.dropna()
        # compute raw returns for all features
        df_ret = df.pct_change().dropna() * 100
        arr = df_ret.values
        closes = df_ret[close_feature].values

        # Slide windows with stride=1
        for i in range(len(arr) - window - horizon + 1):
            w = arr[i:i + window]  # shape: (n_features, window)
            # target return for close feature
            if regression:
                lbl = closes[i + window : i + window + horizon]
            else:
                raise NotImplementedError("Multi-step regression label is not defined.")
            windows.append((w, lbl))

    # Chronological split
    data = np.stack([w for w, _ in windows])
    labels = np.array([lbl for _, lbl in windows])
    n = len(data)
    v = int(n * val_ratio)
    t = int(n * test_ratio)
    tr = n - v - t
    splits = {
        'train': (data[:tr], labels[:tr]),
        'val':   (data[tr:tr + v], labels[tr:tr + v]),
        'test':  (data[tr + v:], labels[tr + v:])
    }

    # --- MODIFICATION START ---
    # Standardize the data after splitting to prevent data leakage.
    # 1. Calculate the mean and standard deviation ONLY from the training data.
    train_data_x, train_data_y = splits['train']

    # For context data X (shape: [N, window, features])
    mean_x = np.mean(train_data_x, axis=(0, 1), keepdims=True)  # -> [1, 1, features]
    std_x = np.std(train_data_x, axis=(0, 1), keepdims=True)
    std_x[std_x == 0] = 1.0  # Avoid division by zero for flat features

    # For target data Y (shape: [N, horizon]) - assuming it's a single feature (close_feature)
    mean_y = np.mean(train_data_y, axis=0, keepdims=True)  # -> [1, horizon]
    std_y = np.std(train_data_y, axis=0, keepdims=True)
    std_y[std_y == 0] = 1.0

    # 2. Apply this normalization to all splits (train, val, and test).
    for split_name in ['train', 'val', 'test']:
        data_x, data_y = splits[split_name]
        # Normalize context X
        normalized_data_x = (data_x - mean_x) / std_x
        # Normalize target Y
        normalized_data_y = (data_y - mean_y) / std_y
        # Update the splits dictionary with normalized data
        splits[split_name] = (normalized_data_x, normalized_data_y)

    # Also save the normalization stats so you can de-normalize predictions later
    norm_stats = {
        'mean_x': mean_x.tolist(), 'std_x': std_x.tolist(),
        'mean_y': mean_y.tolist(), 'std_y': std_y.tolist()
    }
    # --- MODIFICATION END ---


    # Save splits
    for key, (da, la) in splits.items():
        np.save(cache[key], da)
        np.save(cache[f"{key}_lbl"], la)

    meta = {
        'tickers': tickers,
        'start': start,
        'end': end,
        'features': features,
        'window': window,
        'regression': regression,
        'close_feature': close_feature,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'horizon': horizon
    }
    with open(cache['meta'], 'w') as f:
        json.dump(meta, f)

    return splits['train'], splits['val'], splits['test']


def load_dataloaders_from_cache(
    batch_size: int,
    data_dir: str = './data',
    regression: bool = True,
    pin_memory: bool = True,
    shuffle_train: bool = True
):
    """
    Loads the cached arrays and returns DataLoaders.
    """
    arrs = {}
    for split in ('train', 'val', 'test'):
        arrs[split], arrs[f"{split}_lbl"] = (
            np.load(os.path.join(data_dir, f"{split}.npy")),
            np.load(os.path.join(data_dir, f"{split}_lbl.npy"))
        )

    loaders = {}
    for split in ('train', 'val', 'test'):
        ds = LabeledWindowDataset(
            arrs[split], arrs[f"{split}_lbl"], regression
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == 'train' and shuffle_train),
            pin_memory=pin_memory
        )

    sizes = tuple(len(loaders[s].dataset) for s in ('train', 'val', 'test'))
    return loaders['train'], loaders['val'], loaders['test'], sizes


def prepare_data_and_cache(
    tickers,
    start,
    end,
    features,
    window,
    data_dir: str = './data',
    close_feature: str = 'Close',
    regression: bool = True,
    val_ratio: float = 0.1,
    test_ratio: float = 0.4,
    horizon: int = 24,
):
    """
    Call this once up‑front to generate & cache your windows.
    """
    return _cache_sliding_windows(
        tickers, start, end, features, window,
        regression, close_feature, data_dir,
        val_ratio, test_ratio, horizon
    )