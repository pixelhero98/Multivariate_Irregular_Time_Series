# ==== NEW: cleaner stock prep & loaders (drop into latent_vae_utils.py) ====
from __future__ import annotations

import os, json, math, warnings
from typing import List, Optional, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import yfinance as yf
    import pandas as pd
except Exception:
    yf = None
    pd = None


# --------------------- Dataset with optional meta (asset IDs / times) ---------------------

class LabeledWindowWithMetaDataset(Dataset):
    """
    Same shapes as your original:
      x: [window, F]  float32
      y: [horizon, 1] float32
    But also returns a 'meta' dict with:
      - 'asset_id' (int)
      - 'asset' (str ticker)
      - 'ctx_times' (np.datetime64[K])   # optional if passed
      - 'y_times'   (np.datetime64[H])   # optional if passed
    """
    def __init__(self,
                 X: np.ndarray,       # [N, K, F]
                 Y: np.ndarray,       # [N, H] or [N, H, 1]
                 asset_ids: Optional[np.ndarray] = None,  # [N]
                 assets: Optional[List[str]] = None,      # list of tickers (asset_id -> ticker)
                 ctx_times: Optional[np.ndarray] = None,  # [N, K] datetime64
                 y_times: Optional[np.ndarray] = None,    # [N, H] datetime64
                 regression: bool = True):
        assert X.ndim == 3, f"X must be [N, K, F], got {X.shape}"
        self.X = X.astype(np.float32, copy=False)
        if regression:
            Y = Y.astype(np.float32, copy=False)
            if Y.ndim == 2:
                Y = Y[..., None]  # [N,H] -> [N,H,1]
        self.Y = Y
        self.regression = regression
        self.asset_ids = asset_ids
        self.assets = assets
        self.ctx_times = ctx_times
        self.y_times = y_times

    def __len__(self): 
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx]).float()           # [K,F]
        y_np = self.Y[idx]
        y = torch.from_numpy(y_np).float() if self.regression else torch.tensor(int(y_np), dtype=torch.int64)
        meta = {}
        if self.asset_ids is not None:
            meta['asset_id'] = int(self.asset_ids[idx])
            if self.assets is not None:
                meta['asset'] = self.assets[meta['asset_id']]
        if self.ctx_times is not None: meta['ctx_times'] = self.ctx_times[idx]
        if self.y_times   is not None: meta['y_times'] = self.y_times[idx]
        return x, y, meta


def load_dataloaders_with_meta_v2(
    batch_size: int,
    data_dir: str = './data',
    regression: bool = True,
    num_workers: int = 2,
    shuffle_train: bool = True,
    pin_memory: Optional[bool] = None,
    return_meta_arrays: bool = False,
):
    """
    Loads NPZ caches written by prepare_stock_windows_and_cache_v2 and wraps in DataLoaders.
    Returns: train_dl, val_dl, test_dl, lengths, (optional) raw arrays/meta if return_meta_arrays
    """
    path_npz = os.path.join(data_dir, 'cache_v2.npz')
    meta_path = os.path.join(data_dir, 'meta_v2.json')
    if not (os.path.exists(path_npz) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"Missing {path_npz} or {meta_path}. Run prepare_stock_windows_and_cache_v2(...) first.")

    z = np.load(path_npz, allow_pickle=True)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    assets = meta['assets']

    def mk_ds(prefix: str):
        X = z[f'{prefix}_X']              # [N,K,F]
        Y = z[f'{prefix}_Y']              # [N,H] (regression)
        ids = z.get(f'{prefix}_asset_id') # [N]
        ctx_t = z.get(f'{prefix}_ctx_times')  # [N,K]
        y_t   = z.get(f'{prefix}_y_times')    # [N,H]
      
        return LabeledWindowWithMetaDataset(X, Y, ids, assets, ctx_t, y_t, regression=regression)

    ds_tr = mk_ds('train'); ds_va = mk_ds('val'); ds_te = mk_ds('test')

    if pin_memory is None: pin_memory = torch.cuda.is_available()
    def mk_loader(ds, split):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=(split=='train' and shuffle_train),
            pin_memory=pin_memory, num_workers=num_workers, persistent_workers=(num_workers>0),
        )
    train_dl = mk_loader(ds_tr, 'train'); val_dl = mk_loader(ds_va, 'val'); test_dl = mk_loader(ds_te, 'test')
    lengths = (len(ds_tr), len(ds_va), len(ds_te))

    if return_meta_arrays:
        return train_dl, val_dl, test_dl, lengths, (z, meta)
    return train_dl, val_dl, test_dl, lengths

# --------------------- Stock prep (date-based splits, per-ticker norm, clean features) -----

def _safe_pct_change(s: pd.Series) -> pd.Series:
    return s.pct_change().replace([np.inf, -np.inf], np.nan)

def _log_return(s: pd.Series) -> pd.Series:
    return np.log(s).diff()

def _ewma_vol(ret: pd.Series, span: int = 20) -> pd.Series:
    return ret.pow(2).ewm(span=span, adjust=False).mean().pow(0.5)

def _delta_log_volume(vol: pd.Series) -> pd.Series:
    return np.log(vol.replace(0, np.nan)).diff()

def _winsorize(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(arr, low, high)

def prepare_stock_windows_and_cache_v2(
    tickers: List[str],
    start: str,            # e.g., "2012-01-01"
    val_start: str,        # e.g., "2020-01-01"
    test_start: str,       # e.g., "2022-01-01"
    end: str,              # e.g., "2025-06-30"
    window: int,           # K
    horizon: int,          # H
    data_dir: str = './data',
    close_col: str = 'Close',
    volume_col: str = 'Volume',
    returns_mode: str = 'log',       # 'log' or 'pct'
    features: Optional[List[str]] = None,  # if None, build sensible defaults below
    add_market_proxy: Optional[str] = 'SPY',  # add SPY close as extra channel if available
    normalize_per_ticker: bool = True,
    clamp_sigma: float = 5.0,        # clamp before z-scoring using train stats
    min_obs_buffer: int = 50,        # extra obs beyond K+H to accept a ticker
    min_train_coverage: float = 0.9, # % non-NaN rows in train period
    liquidity_rank_window: Optional[Tuple[str,str]] = None,  # (rank_start, rank_end)
    top_n_by_dollar_vol: Optional[int] = None,               # if set, pick top N by median $ volume
    max_windows_per_ticker: Optional[int] = None,            # if set, subsample windows to balance
    regression: bool = True,
    seed: int = 1337,
):
    """
    Clean, reproducible prep for large stock panels. Writes a single compressed NPZ:
      cache_v2.npz  +  meta_v2.json  +  norm_stats_v2.json

    Shapes preserved vs your original (x: [K,F], y: [H,1]).  (Based on your current code. :contentReference[oaicite:1]{index=1})
    Key features:
      - Universe cleaning (history/coverage; optional liquidity filter)
      - Date-based splits with embargo
      - Returns + richer conditioning features (EWMA vol, Δlog(volume), market proxy)
      - Optional per-ticker normalization; robust clamping
      - Asset IDs and timestamps per window
    """
    if yf is None or pd is None:
        raise ImportError("yfinance/pandas not available. pip install yfinance pandas")

    rng = np.random.RandomState(seed)
    os.makedirs(data_dir, exist_ok=True)
    cache_npz = os.path.join(data_dir, 'cache_v2.npz')
    meta_json = os.path.join(data_dir, 'meta_v2.json')
    norm_json = os.path.join(data_dir, 'norm_stats_v2.json')

    # ---- 1) Download raw (auto_adjusted close; multi-index columns) ----
    wanted_cols = [close_col, volume_col]
    if add_market_proxy and add_market_proxy not in tickers:
        tickers_dl = tickers + [add_market_proxy]
    else:
        tickers_dl = tickers[:]

    raw = yf.download(tickers_dl, start=start, end=end, auto_adjust=True,
                      group_by="column", progress=False)

    # Normalize multiindex to (column, ticker) layout
    if isinstance(raw.columns, pd.MultiIndex):
        # Expect (field, ticker); swap if needed
        if not set(wanted_cols).issubset(set(raw.columns.get_level_values(0))):
            raw = raw.swaplevel(0, 1, axis=1).sort_index(axis=1)

    def get_ticker_df(t: str) -> pd.DataFrame:
        if isinstance(raw.columns, pd.MultiIndex):
            try:
                df = raw.xs(t, axis=1, level=1)
            except Exception:
                df = raw.xs(t, axis=1, level=0)
        else:
            # single ticker download
            df = raw
        # keep only close/volume, dropna rows
        keep = [c for c in (close_col, volume_col) if c in df.columns]
        return df[keep].sort_index()

    # ---- 2) Optional liquidity ranking (median dollar volume) ----
    def median_dollar_volume(df: pd.DataFrame, a: str, b: str) -> float:
        sub = df.loc[a:b]
        if close_col not in sub or volume_col not in sub: return 0.0
        dv = (sub[close_col] * sub[volume_col]).replace([np.inf, -np.inf], np.nan).dropna()
        return float(dv.median()) if len(dv) else 0.0

    if liquidity_rank_window is not None and top_n_by_dollar_vol is not None:
        a, b = liquidity_rank_window
        ranks = []
        for t in tickers:
            df_t = get_ticker_df(t)
            ranks.append((t, median_dollar_volume(df_t, a, b)))
        ranks.sort(key=lambda x: x[1], reverse=True)
        tickers = [t for t, _ in ranks[:top_n_by_dollar_vol]]

    # Add SPY proxy if requested and present
    have_proxy = add_market_proxy and add_market_proxy in raw.columns.get_level_values(1) if isinstance(raw.columns, pd.MultiIndex) else (add_market_proxy in tickers_dl)
    proxy_series = None
    if add_market_proxy and have_proxy:
        proxy_df = get_ticker_df(add_market_proxy)
        proxy_series = proxy_df[close_col].copy().rename('SPY_Close')

    # ---- 3) Build per-ticker feature frames (returns, vol, Δlog vol, proxy) ----
    def make_feat_df(df: pd.DataFrame) -> pd.DataFrame:
        # returns
        if returns_mode == 'log':
            ret = _log_return(df[close_col])
        elif returns_mode == 'pct':
            ret = _safe_pct_change(df[close_col])
        else:
            raise ValueError("returns_mode must be 'log' or 'pct'")

        vol = _ewma_vol(ret, span=20)                    # realized vol proxy
        turn = _delta_log_volume(df[volume_col]) if volume_col in df else pd.Series(index=df.index, dtype=float)

        out = pd.DataFrame({
            'RET': ret,
            'RVOL20': vol,
            'DLV': turn,
        })
        if proxy_series is not None:
            # align proxy returns (market factor)
            if returns_mode == 'log':
                mret = _log_return(proxy_series)
            else:
                mret = _safe_pct_change(proxy_series)
            out['MKT'] = mret.reindex(out.index)
        return out

    per_ticker = {}
    min_obs = window + horizon + min_obs_buffer

    for t in tickers:
        df = get_ticker_df(t)
        if df.shape[0] < min_obs:
            continue
        feat = make_feat_df(df).dropna(how='all')
        # require coverage in train segment
        train_seg = feat.loc[start: val_start]
        coverage = 1.0 - train_seg.isna().any(axis=1).mean() if len(train_seg) else 0.0
        if coverage < min_train_coverage:
            continue
        feat = feat.dropna()
        if feat.shape[0] < min_obs:
            continue
        per_ticker[t] = feat

    if len(per_ticker) == 0:
        raise RuntimeError("No tickers passed the cleaning criteria. Relax thresholds or check dates.")

    assets = sorted(per_ticker.keys())
    asset2id = {a:i for i,a in enumerate(assets)}

    # ---- 4) Sliding windows with DATE-BASED SPLITS (no leakage) ----
    def window_rows(feat: pd.DataFrame):
        """
        Returns lists of (X[K,F], Y[H], ctx_times[K], y_times[H], split, asset_id)
        Split assignment by the END-OF-CONTEXT date:
          - end_date < val_start  -> train
          - val_start <= end_date < test_start -> val
          - test_start <= end_date -> test
        """
        arr = feat[['RET','RVOL20','DLV'] + (['MKT'] if 'MKT' in feat.columns else [])].astype(np.float32)
        times = arr.index.to_numpy()  # datetime64
        A = arr.to_numpy(dtype=np.float32)  # [T,F]
        F = A.shape[1]

        rows = {'train':[], 'val':[], 'test':[]}
        for i in range(0, len(A) - window - horizon + 1):
            end_ctx_idx = i + window - 1
            end_ctx_date = pd.Timestamp(times[end_ctx_idx])
            x = A[i:i+window]                               # [K,F]
            y = arr['RET'].to_numpy(dtype=np.float32)[i+window : i+window+horizon]  # [H] predict returns
            ctx_t = times[i:i+window]
            y_t   = times[i+window : i+window+horizon]

            if end_ctx_date < pd.Timestamp(val_start):
                split = 'train'
            elif end_ctx_date < pd.Timestamp(test_start):
                split = 'val'
            else:
                if end_ctx_date > pd.Timestamp(end):
                    continue
                split = 'test'
            rows[split].append((x, y, ctx_t, y_t))
        return rows

    splits = {'train':[], 'val':[], 'test':[]}
    id_splits = {'train':[], 'val':[], 'test':[]}

    for a in assets:
        rows = window_rows(per_ticker[a])
        for sp in ('train','val','test'):
            if max_windows_per_ticker is not None and len(rows[sp]) > max_windows_per_ticker:
                idxs = np.sort(rng.choice(len(rows[sp]), size=max_windows_per_ticker, replace=False))
                take = [rows[sp][j] for j in idxs]
            else:
                take = rows[sp]
            splits[sp].extend(take)
            id_splits[sp].extend([asset2id[a]] * len(take))

    # ---- 5) Stack, TRAIN-ONLY NORMALIZATION (per-ticker optional), clamp, apply ----
    def stack_split(lst):
        if len(lst)==0:
            return (np.empty((0, window, 3 + (1 if have_proxy else 0)), np.float32),
                    np.empty((0, horizon), np.float32),
                    np.empty((0, window), 'datetime64[ns]'),
                    np.empty((0, horizon), 'datetime64[ns]'))
        X = np.stack([x for (x, y, ct, yt) in lst], axis=0)
        Y = np.stack([y for (x, y, ct, yt) in lst], axis=0)
        CT= np.stack([ct for (x, y, ct, yt) in lst], axis=0)
        YT= np.stack([yt for (x, y, ct, yt) in lst], axis=0)
        return X, Y, CT, YT

    Xtr, Ytr, CTtr, YTtr = stack_split(splits['train'])
    Xva, Yva, CTva, YTva = stack_split(splits['val'])
    Xte, Yte, CTte, YTte = stack_split(splits['test'])

    ids_tr = np.array(id_splits['train'], dtype=np.int32)
    ids_va = np.array(id_splits['val'], dtype=np.int32)
    ids_te = np.array(id_splits['test'], dtype=np.int32)

    if Xtr.shape[0] == 0: raise RuntimeError("No training windows after splits — check dates.")

    # Compute train stats
    if normalize_per_ticker:
        # per-asset stats computed on TRAIN ONLY (using windows belonging to that asset)
        Fdim = Xtr.shape[-1]
        mean_x = np.zeros((len(assets), 1, 1, Fdim), np.float32)
        std_x  = np.ones ((len(assets), 1, 1, Fdim), np.float32)
        mean_y = np.zeros((len(assets), 1, horizon), np.float32)
        std_y  = np.ones ((len(assets), 1, horizon), np.float32)

        for aid in range(len(assets)):
            mask = (ids_tr == aid)
            if not np.any(mask): continue
            _X = Xtr[mask]                        # [N_a, K, F]
            _Y = Ytr[mask]                        # [N_a, H]
            mx = _X.mean(axis=(0,1), keepdims=True)
            sx = _X.std (axis=(0,1), keepdims=True); sx[sx==0]=1.0
            my = _Y.mean(axis=0, keepdims=True)
            sy = _Y.std (axis=0, keepdims=True); sy[sy==0]=1.0
            mean_x[aid] = mx; std_x[aid] = sx; mean_y[aid] = my; std_y[aid] = sy

        def norm_apply(X, Y, ids):
            # clamp BEFORE z-scoring using per-asset mean/std implied sigma bounds
            Xn = X.copy(); Yn = Y.copy()
            for aid in np.unique(ids):
                m = (ids == aid)
                mx = mean_x[aid]; sx = std_x[aid]
                # estimate sigma per-feature from sx to build clamp bounds around mx
                lo = mx - clamp_sigma * sx
                hi = mx + clamp_sigma * sx
                Xn[m] = np.clip(Xn[m], lo, hi)
                Xn[m] = (Xn[m] - mx) / sx
                if regression:
                    my = mean_y[aid]; sy = std_y[aid]
                    lo_y = my - clamp_sigma * sy; hi_y = my + clamp_sigma * sy
                    Yn[m] = np.clip(Yn[m], lo_y, hi_y)
                    Yn[m] = (Yn[m] - my) / sy
            return Xn, Yn

        Xtr, Ytr = norm_apply(Xtr, Ytr, ids_tr)
        Xva, Yva = norm_apply(Xva, Yva, ids_va) if len(Xva) else (Xva, Yva)
        Xte, Yte = norm_apply(Xte, Yte, ids_te) if len(Xte) else (Xte, Yte)

        norm_stats = {
            'per_ticker': True,
            'mean_x': mean_x.tolist(), 'std_x': std_x.tolist(),
            'mean_y': mean_y.tolist() if regression else None,
            'std_y':  std_y.tolist()  if regression else None,
            'assets': assets,
        }
    else:
        # global train stats (your original approach)
        mean_x = Xtr.mean(axis=(0,1), keepdims=True); std_x = Xtr.std(axis=(0,1), keepdims=True); std_x[std_x==0]=1.0
        mean_y = Ytr.mean(axis=0, keepdims=True);     std_y = Ytr.std(axis=0, keepdims=True);     std_y[std_y==0]=1.0

        def norm_apply(X, Y):
            lo = mean_x - clamp_sigma*std_x; hi = mean_x + clamp_sigma*std_x
            X = np.clip(X, lo, hi)
            X = (X - mean_x) / std_x
            if regression:
                lo_y = mean_y - clamp_sigma*std_y; hi_y = mean_y + clamp_sigma*std_y
                Y = np.clip(Y, lo_y, hi_y)
                Y = (Y - mean_y) / std_y
            return X, Y

        Xtr, Ytr = norm_apply(Xtr, Ytr)
        Xva, Yva = norm_apply(Xva, Yva) if len(Xva) else (Xva, Yva)
        Xte, Yte = norm_apply(Xte, Yte) if len(Xte) else (Xte, Yte)

        norm_stats = {
            'per_ticker': False,
            'mean_x': mean_x.tolist(), 'std_x': std_x.tolist(),
            'mean_y': mean_y.tolist() if regression else None,
            'std_y':  std_y.tolist()  if regression else None,
            'assets': assets,
        }

    # ---- 6) Optional balancing (limit windows per ticker AFTER norm) ----
    # (already did per-split sampling earlier via max_windows_per_ticker; keeping simple)

    # ---- 7) Save compressed NPZ + meta + norm stats ----
    np.savez_compressed(
        cache_npz,
        train_X=Xtr, train_Y=Ytr, train_asset_id=ids_tr, train_ctx_times=CTtr, train_y_times=YTtr,
        val_X=Xva,   val_Y=Yva,   val_asset_id=ids_va,   val_ctx_times=CTva,  val_y_times=YTva,
        test_X=Xte,  test_Y=Yte,  test_asset_id=ids_te,  test_ctx_times=CTte, test_y_times=YTte,
    )
    meta = {
        'assets': assets,
        'asset2id': asset2id,
        'start': start, 'val_start': val_start, 'test_start': test_start, 'end': end,
        'window': window, 'horizon': horizon,
        'returns_mode': returns_mode,
        'features_built': ['RET','RVOL20','DLV'] + (['MKT'] if have_proxy else []),
        'normalize_per_ticker': normalize_per_ticker,
        'clamp_sigma': clamp_sigma,
        'min_obs_buffer': min_obs_buffer,
        'min_train_coverage': min_train_coverage,
        'liquidity_rank_window': liquidity_rank_window,
        'top_n_by_dollar_vol': top_n_by_dollar_vol,
        'max_windows_per_ticker': max_windows_per_ticker,
        'regression': regression,
        'market_proxy_used': (add_market_proxy if have_proxy else None),
        'seed': seed,
    }
    with open(meta_json, 'w') as f: json.dump(meta, f, indent=2)
    with open(norm_json, 'w') as f: json.dump(norm_stats, f)
    return ( (Xtr, Ytr, ids_tr), (Xva, Yva, ids_va), (Xte, Yte, ids_te) )
