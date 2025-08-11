# latent_vae_utils.py
# Compatible with your old pipeline (x: [window, F], y: [horizon, 1])
# + per-ticker chronological split, train-only normalization,
# + robust yfinance handling, cached norm stats, and safer loaders.

from __future__ import annotations

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# yfinance is optional; only needed for prepare_data_and_cache
try:
    import yfinance as yf
    import pandas as pd
except Exception:
    yf = None
    pd = None


# ------------------------- Latent stats helpers -------------------------

def compute_per_dim_stats(all_mu: torch.Tensor):
    """
    all_mu: [N, L, D]
    returns:
      mu_per_dim  [D], std_per_dim [D]  (std is clamped to >= 1e-6)
    """
    mu_per_dim  = all_mu.mean(dim=(0, 1))                         # [D]
    std_per_dim = all_mu.std(dim=(0, 1)).clamp(min=1e-6)          # [D]
    return mu_per_dim, std_per_dim


def normalize_and_check(all_mu: torch.Tensor, plot: bool = False):
    """
    Per-dimension normalize and (optionally) plot a histogram.

    returns:
      all_mu_norm: [N, L, D]
      mu_per_dim:  [D]
      std_per_dim: [D]
    """
    mu_per_dim, std_per_dim = compute_per_dim_stats(all_mu)
    mu_b  = mu_per_dim.view(1, 1, -1)
    std_b = std_per_dim.view(1, 1, -1)
    all_mu_norm = (all_mu - mu_b) / std_b

    # global check
    all_vals = all_mu_norm.reshape(-1)
    print(f"Global mean (post-norm): {all_vals.mean().item():.6f}")
    print(f"Global std  (post-norm): {all_vals.std().item():.6f}")

    # per-dim printout (first few)
    per_dim_mean = all_mu_norm.mean(dim=(0, 1))
    per_dim_std  = all_mu_norm.std(dim=(0, 1))
    D = all_mu_norm.size(-1)
    print("\nPer-dim stats (first 10 dims or D if smaller):")
    for i in range(min(10, D)):
        print(f"  dim {i:2d}: mean={per_dim_mean[i]:7.4f}, std={per_dim_std[i]:7.4f}")

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4, 3))
        plt.hist(all_vals.cpu().numpy(), bins=500, range=(-5, 5))
        plt.title("Histogram of normalized μ values")
        plt.xlabel("Value"); plt.ylabel("Count")
        plt.show()

    print(f"NaNs: {torch.isnan(all_mu_norm).sum().item()} | Infs: {torch.isinf(all_mu_norm).sum().item()}")
    print(f"Min: {all_mu_norm.min().item():.6f} | Max: {all_mu_norm.max().item():.6f}")
    return all_mu_norm, mu_per_dim, std_per_dim


# ------------------------- Dataset (keeps your y: [H,1]) -------------------------

class LabeledWindowDataset(Dataset):
    """
    Returns (x, y) where:
      x: [window, F]  float32
      y: [horizon, 1] float32   (for regression=True)
    """
    def __init__(self, data_array: np.ndarray, labels_array: np.ndarray, regression: bool):
        assert data_array.ndim == 3, f"X must be [N, window, F], got {data_array.shape}"
        self.data_array = data_array.astype(np.float32, copy=False)
        self.labels_array = labels_array
        self.regression = regression

    def __len__(self) -> int:
        return self.data_array.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.data_array[idx]).float()             # [window, F]
        y_np = self.labels_array[idx]
        if not self.regression:
            y = torch.tensor(int(y_np), dtype=torch.int64)
        else:
            # ensure [H,1]
            y = torch.from_numpy(y_np).float()
            if y.dim() == 1:
                y = y.unsqueeze(-1)                                    # [H] -> [H,1]
        return x, y


# ------------------------- Cache loaders -------------------------

def load_dataloaders_from_cache(
    batch_size: int,
    data_dir: str = './data',
    regression: bool = True,
    pin_memory: bool | None = None,
    shuffle_train: bool = True,
    num_workers: int = 2,
):
    """
    Loads cached arrays and wraps them in DataLoaders.
    Returns: train_dl, val_dl, test_dl, (n_train, n_val, n_test)
    """
    paths = {
        'train':     os.path.join(data_dir, 'train.npy'),
        'train_lbl': os.path.join(data_dir, 'train_lbl.npy'),
        'val':       os.path.join(data_dir, 'val.npy'),
        'val_lbl':   os.path.join(data_dir, 'val_lbl.npy'),
        'test':      os.path.join(data_dir, 'test.npy'),
        'test_lbl':  os.path.join(data_dir, 'test_lbl.npy'),
    }
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing cache file: {p}. Run prepare_data_and_cache(...) first.")

    Xtr = np.load(paths['train']);     Ytr = np.load(paths['train_lbl'])
    Xva = np.load(paths['val']);       Yva = np.load(paths['val_lbl'])
    Xte = np.load(paths['test']);      Yte = np.load(paths['test_lbl'])

    ds_tr = LabeledWindowDataset(Xtr, Ytr, regression=regression)
    ds_va = LabeledWindowDataset(Xva, Yva, regression=regression)
    ds_te = LabeledWindowDataset(Xte, Yte, regression=regression)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    def _mk(dl_ds, split: str):
        return DataLoader(
            dl_ds,
            batch_size=batch_size,
            shuffle=(split == 'train' and shuffle_train),
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )

    train_dl = _mk(ds_tr, 'train')
    val_dl   = _mk(ds_va, 'val')
    test_dl  = _mk(ds_te, 'test')
    return train_dl, val_dl, test_dl, (len(ds_tr), len(ds_va), len(ds_te))


def load_norm_stats(data_dir: str = './data') -> dict:
    """
    Load normalization stats saved by the cache builders below.
    returns numpy arrays (float32) or None where not applicable.
    """
    path = os.path.join(data_dir, 'norm_stats.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"norm_stats.json not found in {data_dir}")
    with open(path, 'r') as f:
        stats = json.load(f)
    for k in ('mean_x','std_x','mean_y','std_y'):
        stats[k] = None if stats.get(k) is None else np.array(stats[k], dtype=np.float32)
    return stats


# ------------------------- yfinance-based builder (chronological) -------------------------

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
    Builds sliding windows per ticker in chronological order and caches:
      train.npy, val.npy, test.npy, *_lbl.npy, meta.json, norm_stats.json

    X windows are [window, F]; y labels are [horizon] (normalized later and returned as [horizon,1] by the Dataset).
    """
    if yf is None:
        raise ImportError("yfinance is not installed. `pip install yfinance`")

    os.makedirs(data_dir, exist_ok=True)
    cache = {
        'train':     os.path.join(data_dir, 'train.npy'),
        'train_lbl': os.path.join(data_dir, 'train_lbl.npy'),
        'val':       os.path.join(data_dir, 'val.npy'),
        'val_lbl':   os.path.join(data_dir, 'val_lbl.npy'),
        'test':      os.path.join(data_dir, 'test.npy'),
        'test_lbl':  os.path.join(data_dir, 'test_lbl.npy'),
        'meta':      os.path.join(data_dir, 'meta.json'),
        'norm':      os.path.join(data_dir, 'norm_stats.json'),
    }

    # short-circuit if cache exists and meta matches
    if all(os.path.exists(cache[k]) for k in ('train','val','test','train_lbl','val_lbl','test_lbl','meta')):
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

    # --- Download & per-ticker windows (chronological) ---
    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, group_by="column", progress=False
    )[features]

    # robust MultiIndex handling
    if isinstance(raw.columns, pd.MultiIndex):
        # Expect (feature, ticker); if not, swap
        if not set(features).issubset(set(raw.columns.get_level_values(0))):
            raw = raw.swaplevel(0, 1, axis=1).sort_index(axis=1)

    per_ticker = {t: {'X': [], 'Y': []} for t in tickers}

    for t in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            try:
                df = raw.xs(t, axis=1, level=1)
            except Exception:
                df = raw.xs(t, axis=1, level=0)
        else:
            df = raw
        df = df.sort_index().dropna()
        if df.shape[0] < window + horizon + 1:
            continue

        # percent change * 100 (features + target)
        df_feat = df[features].pct_change().mul(100.0)
        tgt = df[close_feature].pct_change().mul(100.0)
        aligned = pd.concat([df_feat, tgt.rename("__TARGET__")], axis=1).dropna()

        arr = aligned[features].to_numpy(dtype=np.float32)            # [T, F]
        closes = aligned["__TARGET__"].to_numpy(dtype=np.float32)     # [T]

        T = arr.shape[0]
        for i in range(T - window - horizon + 1):
            w = arr[i:i + window]                                     # [window, F]
            if regression:
                lbl = closes[i + window : i + window + horizon]       # [horizon]
            else:
                future = closes[i + window : i + window + horizon]
                lbl = int(np.sign(future.sum()) > 0)
            per_ticker[t]['X'].append(w)
            per_ticker[t]['Y'].append(lbl)

    # --- Chronological split per ticker, then concat across tickers ---
    splits_lists = { 'train': {'X': [], 'Y': []},
                     'val':   {'X': [], 'Y': []},
                     'test':  {'X': [], 'Y': []} }

    def split_counts(n, v_ratio, te_ratio):
        v = int(n * v_ratio); te = int(n * te_ratio); tr = n - v - te
        if n > 0 and tr == 0:
            if v > 0: v -= 1; tr += 1
            elif te > 0: te -= 1; tr += 1
        return tr, v, te

    for t in tickers:
        X = per_ticker[t]['X']; Y = per_ticker[t]['Y']
        if len(X) == 0: continue
        X = np.stack(X, axis=0)                                      # [Nt, window, F]
        Y = np.stack(Y, axis=0) if regression else np.array(Y)       # [Nt, horizon] / [Nt]
        n = len(X)
        tr, v, te = split_counts(n, val_ratio, test_ratio)
        splits_lists['train']['X'].append(X[:tr]);         splits_lists['train']['Y'].append(Y[:tr])
        splits_lists['val']['X'].append(X[tr:tr+v]);       splits_lists['val']['Y'].append(Y[tr:tr+v])
        splits_lists['test']['X'].append(X[tr+v:]);        splits_lists['test']['Y'].append(Y[tr+v:])

    F = len(features)
    splits = {}
    for split in ('train','val','test'):
        xs, ys = splits_lists[split]['X'], splits_lists[split]['Y']
        if len(xs) == 0:
            Xc = np.empty((0, window, F), dtype=np.float32)
            Yc = np.empty((0, horizon), dtype=np.float32) if regression else np.empty((0,), dtype=np.int64)
        else:
            Xc = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
            Yc = np.concatenate(ys, axis=0)
            if regression:
                Yc = Yc.astype(np.float32, copy=False)
        splits[split] = (Xc, Yc)

    # --- Train-only normalization, then apply to all splits ---
    train_x, train_y = splits['train']
    if train_x.shape[0] == 0:
        raise RuntimeError("No training data after per-ticker split. Check dates/ratios.")

    mean_x = np.mean(train_x, axis=(0, 1), keepdims=True)           # [1,1,F]
    std_x  = np.std(train_x, axis=(0, 1), keepdims=True); std_x[std_x == 0] = 1.0

    if regression:
        mean_y = np.mean(train_y, axis=0, keepdims=True)            # [1,H]
        std_y  = np.std(train_y, axis=0, keepdims=True); std_y[std_y == 0] = 1.0
    else:
        mean_y = None; std_y = None

    for split_name in ('train','val','test'):
        data_x, data_y = splits[split_name]
        data_x = (data_x - mean_x) / std_x
        if regression:
            data_y = (data_y - mean_y) / std_y
        splits[split_name] = (data_x, data_y)

    # --- Save arrays (float32) + meta + norm stats ---
    for key in ('train','val','test'):
        da, la = splits[key]
        np.save(cache[key],     da.astype(np.float32, copy=False))
        np.save(cache[f"{key}_lbl"], la.astype(np.float32, copy=False) if regression else la)

    meta = {
        'tickers': tickers, 'start': start, 'end': end,
        'features': features, 'window': window, 'regression': regression,
        'close_feature': close_feature, 'val_ratio': val_ratio, 'test_ratio': test_ratio,
        'horizon': horizon
    }
    with open(cache['meta'], 'w') as f:
        json.dump(meta, f)

    norm_stats = {
        'mean_x': mean_x.tolist(), 'std_x': std_x.tolist(),
        'mean_y': None if mean_y is None else mean_y.tolist(),
        'std_y':  None if std_y  is None else std_y.tolist(),
    }
    with open(cache['norm'], 'w') as f:
        json.dump(norm_stats, f)

    return splits['train'], splits['val'], splits['test']
