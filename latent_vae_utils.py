import yfinance as yf
import numpy as np
import torch
import os
import json
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def check_samples_per_class(loader: DataLoader) -> None:
    """
    Quickly count and print the number of samples per class in a classification DataLoader.

    Args:
        loader (DataLoader): Yields (x, y) batches, where y contains integer class labels.
    """
    counts = None
    for _, y in loader:
        # ensure labels on CPU and flattened 1D
        if torch.is_tensor(y):
            labels = y.detach().cpu().flatten()
        else:
            labels = torch.tensor(y, dtype=torch.long).flatten()
        # initialize counts tensor based on max class
        if counts is None:
            num_classes = int(labels.max().item()) + 1
            counts = torch.zeros(num_classes, dtype=torch.long)
        # accumulate batch counts
        batch_counts = torch.bincount(labels, minlength=counts.size(0))
        counts += batch_counts
    # print results
    for cls, cnt in enumerate(counts.tolist()):
        print(f"Class {cls}: {cnt} samples")

def label_check(train_loader):
    all_labels = []
    # collect every label
    for _, labels in train_loader:
        all_labels.append(labels.view(-1))

    # concatenate into one big vector
    all_labels = torch.cat(all_labels)

    # get each unique class and its count
    unique_classes, counts = torch.unique(all_labels, return_counts=True)

    print("Classes:", unique_classes.tolist())
    print("Counts :", counts.tolist())

def latent_space_check(train_z):
    # --- sanity checks ---
    # 1) Shape & dtype
    print("train_mus.shape:", train_z.shape)
    print("dtype:", train_z.dtype)

    # 2) Global mean & std (should be ≈0, ≈1 if your encoder outputs are normalized)
    all_vals = train_z.view(-1)
    print("global mean:", all_vals.mean().item())
    print("global std: ", all_vals.std().item())

    # 3) Per‑dimension stats
    per_dim_mean = train_z.mean(dim=(0, 1))  # → [D]
    per_dim_std = train_z.std(dim=(0, 1))  # → [D]
    for i in range(min(100, train_z.size(-1))):
        print(f"dim {i:2d}: mean={per_dim_mean[i]:6.3f}, std={per_dim_std[i]:6.3f}")

    # 4) Histogram (optional, but very informative)
    plt.hist(all_vals.numpy(), bins=500)
    plt.title("Histogram of μ values")
    plt.xlabel("μ value")
    plt.ylabel("count")
    plt.show()

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
    plt.figure(figsize=(4,3))
    plt.hist(all_vals.cpu().numpy(), bins=200)
    plt.title("Histogram of normalized μ values")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.show()

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
        classification: bool
    ):
        self.data_array = data_array
        self.labels_array = labels_array
        self.classification = classification

    def __len__(self) -> int:
        return len(self.data_array)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.data_array[idx]).float()
        y_np = self.labels_array[idx]
        if self.classification:
            y = torch.tensor(int(y_np), dtype=torch.int64)
        else:
            y = torch.tensor(float(y_np), dtype=torch.float32)
        return x, y

def _cache_sliding_windows(
    tickers,
    start,
    end,
    features,
    window,
    classification,
    close_feature,
    data_dir,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
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
            meta.get('classification') == classification and
            meta.get('close_feature') == close_feature and
            meta.get('val_ratio') == val_ratio and
            meta.get('test_ratio') == test_ratio
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
        for i in range(len(arr) - window):
            w = arr[i:i + window]  # shape: (n_features, window)
            # target return for close feature
            ret = closes[i + window]
            if classification:
                lbl = int(ret > 0)
            else:
                lbl = float(ret)
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
        'classification': classification,
        'close_feature': close_feature,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio
    }
    with open(cache['meta'], 'w') as f:
        json.dump(meta, f)

    return splits['train'], splits['val'], splits['test']


def load_dataloaders_from_cache(
    batch_size: int,
    data_dir: str = './data',
    classification: bool = False,
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
            arrs[split], arrs[f"{split}_lbl"], classification
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
    classification: bool = False,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
):
    """
    Call this once up‑front to generate & cache your windows.
    """
    return _cache_sliding_windows(
        tickers, start, end, features, window,
        classification, close_feature, data_dir,
        val_ratio, test_ratio
    )