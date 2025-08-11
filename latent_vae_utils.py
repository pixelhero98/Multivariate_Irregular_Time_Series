import yfinance as yf
import numpy as np
import torch
import os
import json
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


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

    CHANGE: splits are now **per-ticker chronological**:
      - For each ticker, we build sliding windows in time order.
      - We split that ticker's windows into train/val/test by ratio.
      - We then concatenate across tickers for each split.

    Shapes:
      X window = [window, n_features]
      y label  = [horizon]   (regression=True)
    """
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

    # If cache exists AND config matches, load & return
    if all(os.path.exists(cache[k]) for k in ('train', 'val', 'test', 'train_lbl', 'val_lbl', 'test_lbl', 'meta')):
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

    # --- Build windows per ticker (time order), returns ×100 ---
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)[features]
    per_ticker = {t: {'X': [], 'Y': []} for t in tickers}

    for t in tickers:
        df = raw.xs(t, axis=1, level=1) if len(tickers) > 1 else raw
        df = df.dropna()
        if df.shape[0] < window + horizon + 1:
            continue  # not enough history for this ticker

        # compute returns for all features and for the target close
        df_ret = df.pct_change().dropna() * 100.0
        arr = df_ret.values.astype(np.float32)                         # [T, F]
        closes = df_ret[close_feature].values.astype(np.float32)       # [T]

        # slide windows with stride=1
        for i in range(len(arr) - window - horizon + 1):
            w = arr[i:i + window]                                      # [window, F]
            if regression:
                lbl = closes[i + window : i + window + horizon]        # [horizon]
            else:
                # (classification path if you have one)
                future = closes[i + window : i + window + horizon]     # [horizon]
                lbl = int(np.sign(future.sum()) > 0)
            per_ticker[t]['X'].append(w)
            per_ticker[t]['Y'].append(lbl)

    # --- Chronological split per ticker, then concatenate across tickers ---
    splits_lists = {
        'train': {'X': [], 'Y': []},
        'val':   {'X': [], 'Y': []},
        'test':  {'X': [], 'Y': []},
    }

    def split_counts(n, val_ratio, test_ratio):
        v = int(n * val_ratio)
        te = int(n * test_ratio)
        tr = n - v - te
        # ensure at least 1 train sample if possible
        if n > 0 and tr == 0:
            if v > 0:
                v -= 1; tr += 1
            elif te > 0:
                te -= 1; tr += 1
        return tr, v, te

    for t in tickers:
        X = per_ticker[t]['X']; Y = per_ticker[t]['Y']
        if len(X) == 0:
            continue
        X = np.stack(X, axis=0)                     # [Nt, window, F]
        Y = np.stack(Y, axis=0) if regression else np.array(Y)  # [Nt, horizon] or [Nt]

        n = len(X)
        tr, v, te = split_counts(n, val_ratio, test_ratio)

        splits_lists['train']['X'].append(X[:tr])
        splits_lists['train']['Y'].append(Y[:tr])
        splits_lists['val']['X'].append(X[tr:tr+v])
        splits_lists['val']['Y'].append(Y[tr:tr+v])
        splits_lists['test']['X'].append(X[tr+v:])
        splits_lists['test']['Y'].append(Y[tr+v:])

    # concatenate across tickers
    splits = {}
    for split in ('train', 'val', 'test'):
        if len(splits_lists[split]['X']) == 0:
            splits[split] = (np.empty((0, window, len(features)), dtype=np.float32),
                             np.empty((0, horizon), dtype=np.float32) if regression else np.empty((0,), dtype=np.int64))
        else:
            Xc = np.concatenate(splits_lists[split]['X'], axis=0).astype(np.float32, copy=False)
            Yc = np.concatenate(splits_lists[split]['Y'], axis=0)
            if regression:
                Yc = Yc.astype(np.float32, copy=False)
            splits[split] = (Xc, Yc)

    # --- Normalize using TRAIN ONLY, then apply to all splits ---
    train_data_x, train_data_y = splits['train']
    if train_data_x.shape[0] == 0:
        raise RuntimeError("No training data after per-ticker split. Check dates/ratios.")

    mean_x = np.mean(train_data_x, axis=(0, 1), keepdims=True)     # [1, 1, F]
    std_x  = np.std(train_data_x, axis=(0, 1), keepdims=True)
    std_x[std_x == 0] = 1.0

    if regression:
        mean_y = np.mean(train_data_y, axis=0, keepdims=True)      # [1, horizon]
        std_y  = np.std(train_data_y, axis=0, keepdims=True)
        std_y[std_y == 0] = 1.0
    else:
        mean_y = None; std_y = None

    for split_name in ['train', 'val', 'test']:
        data_x, data_y = splits[split_name]
        data_x = (data_x - mean_x) / std_x
        if regression:
            data_y = (data_y - mean_y) / std_y
        splits[split_name] = (data_x, data_y)

    # --- Save arrays (float32) and meta/norm stats ---
    for key in ('train', 'val', 'test'):
        da, la = splits[key]
        np.save(cache[key],     da.astype(np.float32, copy=False))
        np.save(cache[f"{key}_lbl"], la.astype(np.float32, copy=False) if regression else la)

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

    norm_stats = {
        'mean_x': mean_x.tolist(), 'std_x': std_x.tolist(),
        'mean_y': mean_y.tolist() if mean_y is not None else None,
        'std_y':  std_y.tolist()  if std_y  is not None else None
    }
    with open(cache['norm'], 'w') as f:
        json.dump(norm_stats, f)

    return splits['train'], splits['val'], splits['test']


def load_norm_stats(data_dir: str = './data') -> dict:
    """
    Load train-only normalization stats saved by prepare_data_and_cache().
    Returns:
        {
          'mean_x': np.ndarray [1, 1, F],
          'std_x':  np.ndarray [1, 1, F],
          'mean_y': np.ndarray [1, H] or None,
          'std_y':  np.ndarray [1, H] or None,
        }
    """
    path = os.path.join(data_dir, 'norm_stats.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"norm_stats.json not found in {data_dir}. "
                                "Run prepare_data_and_cache(...) first.")
    with open(path, 'r') as f:
        stats = json.load(f)

    # Convert to float32 numpy arrays
    for k in ('mean_x', 'std_x', 'mean_y', 'std_y'):
        if k in stats and stats[k] is not None:
            stats[k] = np.array(stats[k], dtype=np.float32)
        else:
            stats[k] = None
    return stats
