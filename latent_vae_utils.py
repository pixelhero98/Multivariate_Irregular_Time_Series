# latent_vae_utils.py
# Utilities for preparing multivariate time-series datasets, caching splits,
# building DataLoaders, and simple normalization helpers.

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# If you use yfinance-based preparation:
try:
    import yfinance as yf
except Exception:
    yf = None


# ----------------------------- Dataset -----------------------------

class TSWindowDataset(Dataset):
    """
    Simple window dataset returning (X, y):
      X: [window, F]  float32
      y: [horizon]    float32 (regression)  or []/int (classification)
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, regression: bool = True):
        assert X.ndim == 3, f"X must be [N, window, F], got {X.shape}"
        self.X = X.astype(np.float32, copy=False)
        self.Y = Y.astype(np.float32, copy=False) if regression else Y
        self.regression = regression

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])  # [window, F]
        y = torch.from_numpy(self.Y[idx]) if self.regression else self.Y[idx]
        return x, y


# ---------------------------- I/O Helpers ----------------------------

def load_dataloaders_from_cache(
    batch_size: int,
    data_dir: str = "./data",
    regression: bool = True,
    pin_memory: bool = None,
    shuffle_train: bool = True,
    num_workers: int = 2,
):
    """
    Load cached arrays (train/val/test and labels) and wrap them in DataLoaders.
    Returns: (train_loader, val_loader, test_loader, (n_train, n_val, n_test))
    """
    paths = {
        "train": os.path.join(data_dir, "train.npy"),
        "train_lbl": os.path.join(data_dir, "train_lbl.npy"),
        "val": os.path.join(data_dir, "val.npy"),
        "val_lbl": os.path.join(data_dir, "val_lbl.npy"),
        "test": os.path.join(data_dir, "test.npy"),
        "test_lbl": os.path.join(data_dir, "test_lbl.npy"),
    }
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing cache file: {p}. Run prepare_data_and_cache(...) first.")

    Xtr = np.load(paths["train"]); Ytr = np.load(paths["train_lbl"])
    Xva = np.load(paths["val"]);   Yva = np.load(paths["val_lbl"])
    Xte = np.load(paths["test"]);  Yte = np.load(paths["test_lbl"])

    ds_tr = TSWindowDataset(Xtr, Ytr, regression=regression)
    ds_va = TSWindowDataset(Xva, Yva, regression=regression)
    ds_te = TSWindowDataset(Xte, Yte, regression=regression)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # persistent_workers requires num_workers > 0
    loaders = {}
    for split, ds in (("train", ds_tr), ("val", ds_va), ("test", ds_te)):
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train" and shuffle_train),
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )

    sizes = (len(ds_tr), len(ds_va), len(ds_te))
    return loaders["train"], loaders["val"], loaders["test"], sizes


def load_norm_stats(data_dir: str = "./data") -> dict:
    """
    Load normalization stats saved by dataset preparation.
    Returns dict with numpy arrays or None where not applicable:
      'mean_x': [1,1,F], 'std_x': [1,1,F], 'mean_y': [1,H] or None, 'std_y': [1,H] or None
    """
    path = os.path.join(data_dir, "norm_stats.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"norm_stats.json not found in {data_dir}. Run prepare_data_and_cache(...) first.")
    with open(path, "r") as f:
        stats = json.load(f)
    for k in ("mean_x", "std_x", "mean_y", "std_y"):
        stats[k] = None if stats.get(k) is None else np.array(stats[k], dtype=np.float32)
    return stats


# ------------------------- Normalization check -------------------------

def normalize_and_check(all_mu: torch.Tensor, clip: float = 5.0, plot: bool = False):
    """
    Compute mean/std over dataset latents and (optionally) check distribution.

    all_mu: [N, L, D] torch.Tensor
    Returns:
      all_vals: flattened normalized values (for optional inspection)
      mu_mean:  [1, 1, D] mean
      mu_std:   [1, 1, D] std (zeros guarded to 1.0)
    """
    assert all_mu.ndim == 3, f"Expected all_mu [N,L,D], got {list(all_mu.shape)}"
    mu_mean = all_mu.mean(dim=(0, 1), keepdim=True)  # [1,1,D]
    mu_std = all_mu.std(dim=(0, 1), keepdim=True)
    mu_std[mu_std == 0] = 1.0

    normed = torch.clamp((all_mu - mu_mean) / mu_std, -clip, clip)
    all_vals = normed.reshape(-1).detach()

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 3))
        plt.hist(all_vals.cpu().numpy(), bins=400, range=(-clip, clip))
        plt.title("Histogram of normalized μ values")
        plt.xlabel("z"); plt.ylabel("count")
        plt.show()

    return all_vals, mu_mean, mu_std


# -------------------- yfinance-based preparation --------------------

def prepare_data_and_cache(
    tickers: list[str],
    start: str,
    end: str,
    features: list[str],
    window: int,
    data_dir: str = "./data",
    *,
    close_feature: str = "Close",
    val_ratio: float = 0.1,
    test_ratio: float = 0.4,
    horizon: int = 24,
    regression: bool = True,
):
    """
    Convenience wrapper for yfinance-driven preparation.
    """
    if yf is None:
        raise ImportError("yfinance is not installed. `pip install yfinance`")
    return _cache_sliding_windows(
        tickers=tickers,
        start=start,
        end=end,
        features=features,
        window=window,
        regression=regression,
        close_feature=close_feature,
        data_dir=data_dir,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        horizon=horizon,
    )


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
    horizon: int = 24,
):
    """
    Generate (data_array, labels_array) per-ticker in chronological order and cache to disk.
    Windows: X=[window, F], labels: y=[horizon] for regression.
    """
    os.makedirs(data_dir, exist_ok=True)
    cache = {
        "train":     os.path.join(data_dir, "train.npy"),
        "train_lbl": os.path.join(data_dir, "train_lbl.npy"),
        "val":       os.path.join(data_dir, "val.npy"),
        "val_lbl":   os.path.join(data_dir, "val_lbl.npy"),
        "test":      os.path.join(data_dir, "test.npy"),
        "test_lbl":  os.path.join(data_dir, "test_lbl.npy"),
        "meta":      os.path.join(data_dir, "meta.json"),
        "norm":      os.path.join(data_dir, "norm_stats.json"),
    }

    # short-circuit if cache exists and meta matches
    if all(os.path.exists(cache[k]) for k in ("train", "val", "test", "train_lbl", "val_lbl", "test_lbl", "meta")):
        with open(cache["meta"], "r") as f:
            meta = json.load(f)
        if (
            meta.get("tickers") == tickers
            and meta.get("start") == start
            and meta.get("end") == end
            and meta.get("features") == features
            and meta.get("window") == window
            and meta.get("regression") == regression
            and meta.get("close_feature") == close_feature
            and meta.get("val_ratio") == val_ratio
            and meta.get("test_ratio") == test_ratio
            and meta.get("horizon") == horizon
        ):
            train = np.load(cache["train"])
            val   = np.load(cache["val"])
            test  = np.load(cache["test"])
            tl    = np.load(cache["train_lbl"])
            vl    = np.load(cache["val_lbl"])
            te    = np.load(cache["test_lbl"])
            return (train, tl), (val, vl), (test, te)

    # ---- Download & prepare per ticker (chronological) ----
    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, group_by="column", progress=False
    )[features]

    per_ticker = {t: {"X": [], "Y": []} for t in tickers}

    # robust MultiIndex handling
    if isinstance(raw.columns, pd.MultiIndex):
        # Expect level-0 = feature, level-1 = ticker; fix if not
        if not set(features).issubset(set(raw.columns.get_level_values(0))):
            raw = raw.swaplevel(0, 1, axis=1).sort_index(axis=1)

    for t in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            # after group_by="column", columns are (feature, ticker)
            try:
                df = raw.xs(t, axis=1, level=1)
            except Exception:
                # If tickers are on level-0
                df = raw.xs(t, axis=1, level=0)
        else:
            df = raw  # single-ticker case

        df = df.sort_index().dropna()
        if df.shape[0] < window + horizon + 1:
            continue  # not enough history

        # Percent-change * 100 for all features and target
        df_feat = df[features].pct_change().mul(100.0)
        tgt = df[close_feature].pct_change().mul(100.0)
        aligned = pd.concat([df_feat, tgt.rename("__TARGET__")], axis=1).dropna()

        arr = aligned[features].to_numpy(dtype=np.float32)           # [T, F]
        closes = aligned["__TARGET__"].to_numpy(dtype=np.float32)    # [T]

        # slide windows (stride=1)
        T = arr.shape[0]
        for i in range(T - window - horizon + 1):
            w = arr[i : i + window]                                  # [window, F]
            if regression:
                lbl = closes[i + window : i + window + horizon]      # [horizon]
            else:
                future = closes[i + window : i + window + horizon]
                lbl = int(np.sign(future.sum()) > 0)
            per_ticker[t]["X"].append(w)
            per_ticker[t]["Y"].append(lbl)

    # ---- Per-ticker chronological split, then concat across tickers ----
    splits_lists = {"train": {"X": [], "Y": []}, "val": {"X": [], "Y": []}, "test": {"X": [], "Y": []}}

    def split_counts(n, v_ratio, te_ratio):
        v = int(n * v_ratio); te = int(n * te_ratio); tr = n - v - te
        if n > 0 and tr == 0:
            if v > 0: v -= 1; tr += 1
            elif te > 0: te -= 1; tr += 1
        return tr, v, te

    for t in tickers:
        X = per_ticker[t]["X"]; Y = per_ticker[t]["Y"]
        if len(X) == 0:
            continue
        X = np.stack(X, axis=0)                                      # [Nt, window, F]
        Y = np.stack(Y, axis=0) if regression else np.array(Y)       # [Nt, horizon] or [Nt]

        n = len(X)
        tr, v, te = split_counts(n, val_ratio, test_ratio)

        splits_lists["train"]["X"].append(X[:tr])
        splits_lists["train"]["Y"].append(Y[:tr])
        splits_lists["val"]["X"].append(X[tr : tr + v])
        splits_lists["val"]["Y"].append(Y[tr : tr + v])
        splits_lists["test"]["X"].append(X[tr + v :])
        splits_lists["test"]["Y"].append(Y[tr + v :])

    # concat across tickers
    F = len(features)
    splits = {}
    for split in ("train", "val", "test"):
        xs, ys = splits_lists[split]["X"], splits_lists[split]["Y"]
        if len(xs) == 0:
            Xc = np.empty((0, window, F), dtype=np.float32)
            Yc = np.empty((0, horizon), dtype=np.float32) if regression else np.empty((0,), dtype=np.int64)
        else:
            Xc = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
            Yc = np.concatenate(ys, axis=0)
            if regression:
                Yc = Yc.astype(np.float32, copy=False)
        splits[split] = (Xc, Yc)

    # ---- Train-only normalization, then apply to all splits ----
    train_x, train_y = splits["train"]
    if train_x.shape[0] == 0:
        raise RuntimeError("No training data after per-ticker split. Check dates/ratios.")

    mean_x = np.mean(train_x, axis=(0, 1), keepdims=True)           # [1,1,F]
    std_x = np.std(train_x, axis=(0, 1), keepdims=True); std_x[std_x == 0] = 1.0

    if regression:
        mean_y = np.mean(train_y, axis=0, keepdims=True)            # [1,H]
        std_y = np.std(train_y, axis=0, keepdims=True); std_y[std_y == 0] = 1.0
    else:
        mean_y = None; std_y = None

    for split_name in ("train", "val", "test"):
        Xc, Yc = splits[split_name]
        Xc = (Xc - mean_x) / std_x
        if regression:
            Yc = (Yc - mean_y) / std_y
        splits[split_name] = (Xc, Yc)

    # ---- Save arrays and metadata ----
    for key in ("train", "val", "test"):
        da, la = splits[key]
        np.save(cache[key], da.astype(np.float32, copy=False))
        np.save(cache[f"{key}_lbl"], la.astype(np.float32, copy=False) if regression else la)

    meta = {
        "tickers": tickers, "start": start, "end": end,
        "features": features, "window": window, "horizon": horizon,
        "regression": regression, "close_feature": close_feature,
        "val_ratio": val_ratio, "test_ratio": test_ratio,
    }
    with open(cache["meta"], "w") as f:
        json.dump(meta, f)

    norm_stats = {
        "mean_x": mean_x.tolist(), "std_x": std_x.tolist(),
        "mean_y": None if mean_y is None else mean_y.tolist(),
        "std_y":  None if std_y  is None else std_y.tolist(),
    }
    with open(cache["norm"], "w") as f:
        json.dump(norm_stats, f)

    return splits["train"], splits["val"], splits["test"]


# ------------------ Generic DataFrame-based preparation ------------------

def prepare_dataframe_and_cache(
    df: pd.DataFrame,
    entity_col: str,
    date_col: str,
    feature_cols: list,
    target_col: str,
    window: int,
    horizon: int,
    data_dir: str = "./data",
    *,
    regression: bool = True,
    val_ratio: float = 0.1,
    test_ratio: float = 0.4,
    feature_transform: str = "none",   # "none" | "pct" | callable(df)->df
    target_transform: str = "pct",     # "none" | "pct" | callable(s)->s
    embargo: int = 0,                              # gap between splits to avoid overlap
    min_points: int = None,
    save_meta_name: str = "meta.json",
    save_norm_name: str = "norm_stats.json",
    verbose: bool = True,
):
    """
    Build windows per-entity in chronological order and cache to disk.
    Produces the same files used by load_dataloaders_from_cache().
    """
    os.makedirs(data_dir, exist_ok=True)
    cache = {
        "train":     os.path.join(data_dir, "train.npy"),
        "train_lbl": os.path.join(data_dir, "train_lbl.npy"),
        "val":       os.path.join(data_dir, "val.npy"),
        "val_lbl":   os.path.join(data_dir, "val_lbl.npy"),
        "test":      os.path.join(data_dir, "test.npy"),
        "test_lbl":  os.path.join(data_dir, "test_lbl.npy"),
        "meta":      os.path.join(data_dir, save_meta_name),
        "norm":      os.path.join(data_dir, save_norm_name),
    }

    def _apply_feat_transform(df_feat: pd.DataFrame) -> pd.DataFrame:
        if callable(feature_transform):
            out = feature_transform(df_feat.copy())
        elif feature_transform == "pct":
            out = df_feat.pct_change().mul(100.0)
        elif feature_transform == "none":
            out = df_feat.copy()
        else:
            raise ValueError(f"Unknown feature_transform: {feature_transform}")
        return out

    def _apply_tgt_transform(s: pd.Series) -> pd.Series:
        if callable(target_transform):
            out = target_transform(s.copy())
        elif target_transform == "pct":
            out = s.pct_change().mul(100.0)
        elif target_transform == "none":
            out = s.copy()
        else:
            raise ValueError(f"Unknown target_transform: {target_transform}")
        return out

    def _split_counts(n, v_ratio, te_ratio):
        v = int(n * v_ratio); te = int(n * te_ratio); tr = n - v - te
        if n > 0 and tr == 0:
            if v > 0: v -= 1; tr += 1
            elif te > 0: te -= 1; tr += 1
        return tr, v, te

    if verbose:
        print(f"[prepare_dataframe_and_cache] entities={df[entity_col].nunique()}, "
              f"window={window}, horizon={horizon}, feat_transform={feature_transform}, tgt_transform={target_transform}")

    per_entity = {}
    for ent, g in df.groupby(entity_col, sort=False):
        g = g.sort_values(date_col)
        g = g[[date_col] + feature_cols + [target_col]].dropna()

        if min_points is not None and g.shape[0] < min_points:
            continue

        feat = _apply_feat_transform(g[feature_cols])
        tgt = _apply_tgt_transform(g[target_col])

        aligned = pd.concat([feat, tgt.rename("__TARGET__")], axis=1).dropna()
        if aligned.shape[0] < window + horizon + 1:
            continue

        arrX = aligned[feature_cols].to_numpy(dtype=np.float32)   # [T, F]
        arrY = aligned["__TARGET__"].to_numpy(dtype=np.float32)   # [T]

        X_list, Y_list = [], []
        T = arrX.shape[0]
        for i in range(T - window - horizon + 1):
            win = arrX[i : i + window]                            # [window, F]
            if regression:
                lbl = arrY[i + window : i + window + horizon]     # [horizon]
            else:
                future = arrY[i + window : i + window + horizon]
                lbl = int(np.sign(future.sum()) > 0)
            X_list.append(win); Y_list.append(lbl)

        if len(X_list) == 0:
            continue
        per_entity[ent] = {"X": X_list, "Y": Y_list}

    if len(per_entity) == 0:
        raise RuntimeError("No entities produced windows. Check columns/transforms/window/horizon.")

    splits_lists = {"train": {"X": [], "Y": []}, "val": {"X": [], "Y": []}, "test": {"X": [], "Y": []}}
    for ent, d in per_entity.items():
        X = np.stack(d["X"], axis=0)
        Y = np.stack(d["Y"], axis=0) if regression else np.array(d["Y"])
        n = len(X)
        tr, v, te = _split_counts(n, val_ratio, test_ratio)

        g = max(0, embargo)
        X_tr, Y_tr = X[:tr], Y[:tr]
        X_v,  Y_v  = X[tr+g : tr+g+v], Y[tr+g : tr+g+v]
        X_te, Y_te = X[tr+g+v+g :],    Y[tr+g+v+g :]

        splits_lists["train"]["X"].append(X_tr); splits_lists["train"]["Y"].append(Y_tr)
        splits_lists["val"]["X"].append(X_v);   splits_lists["val"]["Y"].append(Y_v)
        splits_lists["test"]["X"].append(X_te); splits_lists["test"]["Y"].append(Y_te)

    # concat
    F = len(feature_cols)
    splits = {}
    for split in ("train", "val", "test"):
        xs, ys = splits_lists[split]["X"], splits_lists[split]["Y"]
        if len(xs) == 0:
            Xc = np.empty((0, window, F), dtype=np.float32)
            Yc = np.empty((0, horizon), dtype=np.float32) if regression else np.empty((0,), dtype=np.int64)
        else:
            Xc = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
            Yc = np.concatenate(ys, axis=0)
            if regression:
                Yc = Yc.astype(np.float32, copy=False)
        splits[split] = (Xc, Yc)

    # train-only normalization
    train_x, train_y = splits["train"]
    if train_x.shape[0] == 0:
        raise RuntimeError("No training data after split; adjust ratios/embargo/min_points.")

    mean_x = np.mean(train_x, axis=(0, 1), keepdims=True)         # [1,1,F]
    std_x  = np.std(train_x,  axis=(0, 1), keepdims=True); std_x[std_x == 0] = 1.0

    if regression:
        mean_y = np.mean(train_y, axis=0, keepdims=True)          # [1,H]
        std_y  = np.std(train_y,  axis=0, keepdims=True); std_y[std_y == 0] = 1.0
    else:
        mean_y = None; std_y = None

    for split_name in ("train", "val", "test"):
        Xc, Yc = splits[split_name]
        Xc = (Xc - mean_x) / std_x
        if regression:
            Yc = (Yc - mean_y) / std_y
        splits[split_name] = (Xc, Yc)

    # save arrays + meta + norm
    np.save(cache["train"],     splits["train"][0].astype(np.float32, copy=False))
    np.save(cache["train_lbl"], splits["train"][1].astype(np.float32, copy=False) if regression else splits["train"][1])
    np.save(cache["val"],       splits["val"][0].astype(np.float32, copy=False))
    np.save(cache["val_lbl"],   splits["val"][1].astype(np.float32, copy=False)   if regression else splits["val"][1])
    np.save(cache["test"],      splits["test"][0].astype(np.float32, copy=False))
    np.save(cache["test_lbl"],  splits["test"][1].astype(np.float32, copy=False)  if regression else splits["test"][1])

    meta = {
        "entity_col": entity_col, "date_col": date_col,
        "feature_cols": feature_cols, "target_col": target_col,
        "window": window, "horizon": horizon,
        "regression": regression, "val_ratio": val_ratio, "test_ratio": test_ratio,
        "feature_transform": "callable" if callable(feature_transform) else feature_transform,
        "target_transform":  "callable" if callable(target_transform)  else target_transform,
        "embargo": embargo,
    }
    with open(cache["meta"], "w") as f:
        json.dump(meta, f)

    norm_stats = {
        "mean_x": mean_x.tolist(), "std_x": std_x.tolist(),
        "mean_y": None if mean_y is None else mean_y.tolist(),
        "std_y":  None if std_y  is None else std_y.tolist(),
    }
    with open(cache["norm"], "w") as f:
        json.dump(norm_stats, f)

    if verbose:
        nx = {k: splits[k][0].shape[0] for k in splits}
        print(f"[prepare_dataframe_and_cache] done. N per split: {nx}. Saved to {data_dir}")

    return splits["train"], splits["val"], splits["test"]

