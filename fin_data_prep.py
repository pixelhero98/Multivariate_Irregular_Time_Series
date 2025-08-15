# ==== NEW: cleaner stock prep & loaders (drop into latent_vae_utils.py) ====
from dataclasses import dataclass, field
import os, json, math, warnings
from typing import List, Optional, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Lazy-friendly imports: don't hard-fail at module import time.
try:
    import yfinance as yf  # may be None if not installed
    import pandas as pd    # may be None if not installed
except Exception:
    yf = None
    pd = None


def collate_keep_meta(batch):
    """
    Collate (x, y, meta) while keeping meta as a dict of Python lists.
    - x: stacked to [B, K, F] tensor
    - y: stacked to [B, H, 1] (or [B, ...]) tensor
    - meta: dict where each value is a list (e.g., list of np.datetime64 arrays)
    This avoids default_collate trying to convert datetime64 to tensors.
    """
    xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]
    ms = [b[2] for b in batch]

    x = torch.stack(xs, dim=0)
    # y is already a tensor per sample in your dataset; stacking is safe
    y = torch.stack(ys, dim=0)

    # aggregate meta into dict of lists (do NOT convert to tensors)
    keys = set().union(*(m.keys() for m in ms)) if ms else set()
    meta = {k: [m.get(k, None) for m in ms] for k in keys}
    return x, y, meta

# --------------------- Dataset with optional meta (asset IDs / times) ---------------------
@dataclass
class CalendarConfig:
    include_dow: bool = True     # time-of-week
    include_dom: bool = True     # day-of-month
    include_moy: bool = True     # month-of-year

    # Periods for cyclical encodings
    dow_period: int = 7          # 7 works for equities & crypto; change to 5 if you insist on 5-day cycle
    moy_period: int = 12         # 12 months

    # Output columns (you can rename if you want)
    dow_sin_name: str = "DOW_SIN"
    dow_cos_name: str = "DOW_COS"
    dom_sin_name: str = "DOM_SIN"
    dom_cos_name: str = "DOM_COS"
    moy_sin_name: str = "MOY_SIN"
    moy_cos_name: str = "MOY_COS"


@dataclass
class FeatureConfig:
    # Which price fields to compute returns for
    price_fields: List[str] = field(default_factory=lambda: ['Close'])
    returns_mode: str = 'log'     # 'log' or 'pct'

    # Volatility features
    include_rvol: bool = True
    rvol_span: int = 20
    rvol_on: str = 'Close'

    # Volume feature
    include_dlv: bool = True

    # Market proxy
    market_proxy: Optional[str] = 'SPY'

    # Candle/intraday structure
    include_oc: bool = False
    include_gap: bool = False
    include_hl_range: bool = False

    # Target price field
    target_field: str = 'Close'

    # Use default_factory for nested dataclass
    calendar: CalendarConfig = field(default_factory=CalendarConfig)


def _cyclical_from_int(values: np.ndarray, period: int):
    """Return (sin, cos) with exact alignment to the given integer-period."""
    ang = 2.0 * np.pi * (values.astype(np.float32) / float(period))
    return np.sin(ang).astype(np.float32), np.cos(ang).astype(np.float32)

def build_calendar_frame(idx: 'pd.DatetimeIndex', cfg: CalendarConfig) -> 'pd.DataFrame':
    """
    Deterministic, complete calendar features; no NaNs are produced.
    Works for any business-day or 7-day index.
    """
    # Lazy import so loaders can work without pandas installed
    try:
        import pandas as pd  # local import; safe even if global pd is None
    except Exception as e:
        raise ImportError("pandas is required for build_calendar_frame(...). Install with: pip install pandas") from e

    cols = {}
    if cfg.include_dow:
        # 0=Mon ... 6=Sun; equities simply won't have Sat/Sun rows
        dow = idx.dayofweek.values  # int [0..6]
        s, c = _cyclical_from_int(dow, cfg.dow_period)
        cols[cfg.dow_sin_name] = s
        cols[cfg.dow_cos_name] = c

    if cfg.include_dom:
        # day-of-month ∈ {1..28-31}, with varying month length
        dom = idx.day.values
        dim = idx.days_in_month.values
        # cyclical DOM (scale by actual days in that month)
        # use (dom-1) so day 1 and day end are not artificially close
        ang = 2.0 * np.pi * ((dom - 1).astype(np.float32) / dim.astype(np.float32).clip(min=1))
        cols[cfg.dom_sin_name] = np.sin(ang).astype(np.float32)
        cols[cfg.dom_cos_name] = np.cos(ang).astype(np.float32)

    if cfg.include_moy:
        # month-of-year ∈ {1..12} → shift to 0..11 for clean phase
        moy = (idx.month.values - 1)
        s, c = _cyclical_from_int(moy, cfg.moy_period)
        cols[cfg.moy_sin_name] = s
        cols[cfg.moy_cos_name] = c

    return pd.DataFrame(cols, index=idx)


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

    def __len__(self): return self.X.shape[0]

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
    seed: int = 1337,
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

    # Reproducible shuffling like ratio-split loader
    gen = torch.Generator()
    gen.manual_seed(seed)

    def mk_loader(ds, split):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=(split=='train' and shuffle_train),
            pin_memory=pin_memory, num_workers=num_workers, persistent_workers=(num_workers>0),
            generator=gen,
            collate_fn=collate_keep_meta,  # ensure datetime meta stays as Python lists
        )
    train_dl = mk_loader(ds_tr, 'train'); val_dl = mk_loader(ds_va, 'val'); test_dl = mk_loader(ds_te, 'test')
    lengths = (len(ds_tr), len(ds_va), len(ds_te))

    if return_meta_arrays:
        return train_dl, val_dl, test_dl, lengths, (z, meta)
    return train_dl, val_dl, test_dl, lengths


# --------------------- Stock prep (date-based splits, per-ticker norm, clean features) -----

def _safe_pct_change(s: 'pd.Series') -> 'pd.Series':
    return s.pct_change().replace([np.inf, -np.inf], np.nan)

def _log_return(s: 'pd.Series') -> 'pd.Series':
    return np.log(s).diff()

def _ewma_vol(ret: 'pd.Series', span: int = 20) -> 'pd.Series':
    return ret.pow(2).ewm(span=span, adjust=False).mean().pow(0.5)

def _delta_log_volume(vol: 'pd.Series') -> 'pd.Series':
    return np.log(vol.replace(0, np.nan)).diff()

def _winsorize(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(arr, low, high)

def prepare_stock_windows_and_cache_v2(
    tickers: List[str],
    start: str,
    val_start: str,
    test_start: str,
    end: str,
    window: int,
    horizon: int,
    data_dir: str = './data',
    # NEW: pass a FeatureConfig to tune features & target
    feature_cfg: FeatureConfig = FeatureConfig(),
    normalize_per_ticker: bool = True,
    clamp_sigma: float = 5.0,
    min_obs_buffer: int = 50,
    min_train_coverage: float = 0.9,
    liquidity_rank_window: Optional[tuple] = None,
    top_n_by_dollar_vol: Optional[int] = None,
    max_windows_per_ticker: Optional[int] = None,
    regression: bool = True,
    seed: int = 1337,
):
    # Ensure required deps are available only when this function is used
    try:
        import pandas as pd  # local import
    except Exception as e:
        raise ImportError("pandas is required for prepare_stock_windows_and_cache_v2. Install with: pip install pandas") from e
    try:
        import yfinance as yf  # local import
    except Exception as e:
        raise ImportError("yfinance is required for prepare_stock_windows_and_cache_v2. Install with: pip install yfinance") from e

    import numpy as np, json, os
    rng = np.random.RandomState(seed)
    os.makedirs(data_dir, exist_ok=True)
    cache_npz = os.path.join(data_dir, 'cache_v2.npz')
    meta_json = os.path.join(data_dir, 'meta_v2.json')
    norm_json = os.path.join(data_dir, 'norm_stats_v2.json')

    # ---- Helpers ----
    def _safe_pct_change(s: pd.Series) -> pd.Series:
        return s.pct_change().replace([np.inf, -np.inf], np.nan)

    def _log_return(s: pd.Series) -> pd.Series:
        return np.log(s).diff()

    def _ret(s: pd.Series) -> pd.Series:
        return _log_return(s) if feature_cfg.returns_mode == 'log' else _safe_pct_change(s)

    def _ewma_vol(ret: pd.Series, span: int) -> pd.Series:
        return ret.pow(2).ewm(span=span, adjust=False).mean().pow(0.5)

    def _delta_log_volume(vol: pd.Series) -> pd.Series:
        return np.log(vol.replace(0, np.nan)).diff()

    # ---- 1) Download raw (auto-adjust OHLCV) ----
    # figure out which raw columns we need (OHLC for selected features; Volume if DLV/gap)
    need_price_cols = set(feature_cfg.price_fields) | {feature_cfg.target_field}
    if feature_cfg.include_oc or feature_cfg.include_gap:
        need_price_cols |= {'Open', 'Close'}
    if feature_cfg.include_hl_range:
        need_price_cols |= {'High', 'Low'}
    wanted_cols = sorted(list(need_price_cols | ({'Volume'} if feature_cfg.include_dlv else set())))

    tickers_dl = tickers[:]
    if feature_cfg.market_proxy and feature_cfg.market_proxy not in tickers_dl:
        tickers_dl.append(feature_cfg.market_proxy)

    raw = yf.download(tickers_dl, start=start, end=end, auto_adjust=True,
                      group_by="column", progress=False)

    # normalize multiindex to (column, ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        if not set(wanted_cols).issubset(set(raw.columns.get_level_values(0))):
            raw = raw.swaplevel(0, 1, axis=1).sort_index(axis=1)

    def get_ticker_df(t: str) -> pd.DataFrame:
        if isinstance(raw.columns, pd.MultiIndex):
            try:
                df = raw.xs(t, axis=1, level=1)
            except Exception:
                df = raw.xs(t, axis=1, level=0)
        else:
            df = raw
        keep = [c for c in wanted_cols if c in df.columns]
        return df[keep].sort_index()

    # proxy (market) series
    proxy_ret = None
    if feature_cfg.market_proxy:
        try:
            proxy_df = get_ticker_df(feature_cfg.market_proxy)
            if 'Close' in proxy_df:
                proxy_ret = _ret(proxy_df['Close']).rename('MKT')
        except Exception:
            proxy_ret = None

    # ---- 2) Optional liquidity ranking ----
    def median_dollar_volume(df: pd.DataFrame, a: str, b: str) -> float:
        sub = df.loc[a:b]
        if 'Close' not in sub or 'Volume' not in sub: return 0.0
        dv = (sub['Close'] * sub['Volume']).replace([np.inf, -np.inf], np.nan).dropna()
        return float(dv.median()) if len(dv) else 0.0

    if liquidity_rank_window and top_n_by_dollar_vol:
        a, b = liquidity_rank_window
        ranks = []
        for t in tickers:
            df_t = get_ticker_df(t)
            ranks.append((t, median_dollar_volume(df_t, a, b)))
        ranks.sort(key=lambda x: x[1], reverse=True)
        tickers = [t for t, _ in ranks[:top_n_by_dollar_vol]]

    # ---- 3) Build per-ticker *tunable* feature frames ----
    def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
        feat = {}  # column -> Series aligned on df.index

        # Returns for each requested price field
        for field in feature_cfg.price_fields:
            if field in df:
                feat[f'RET_{field.upper()}'] = _ret(df[field])

        # Candle/intraday features
        if feature_cfg.include_oc and 'Open' in df and 'Close' in df:
            # intraday open->close return
            oc = (np.log(df['Close']) - np.log(df['Open'])) if feature_cfg.returns_mode == 'log' \
                 else (df['Close'] / df['Open'] - 1.0)
            feat['OC_RET'] = oc

        if feature_cfg.include_gap and 'Open' in df and 'Close' in df:
            # overnight gap: today's open vs yesterday's close
            gap = (np.log(df['Open']) - np.log(df['Close'].shift(1))) if feature_cfg.returns_mode == 'log' \
                  else (df['Open'] / df['Close'].shift(1) - 1.0)
            feat['GAP_RET'] = gap

        if feature_cfg.include_hl_range and 'High' in df and 'Low' in df:
            # intraday range magnitude
            hlr = (np.log(df['High']) - np.log(df['Low'])) if feature_cfg.returns_mode == 'log' \
                  else (df['High'] / df['Low'] - 1.0)
            feat['HL_RANGE'] = hlr

        # Volume feature
        if feature_cfg.include_dlv and 'Volume' in df:
            feat['DLV'] = _delta_log_volume(df['Volume'])

        # Realized volatility (based on RET_<rvol_on>)
        if feature_cfg.include_rvol:
            base_col = f'RET_{feature_cfg.rvol_on.upper()}'
            if base_col in feat:
                feat[f'RVOL{feature_cfg.rvol_span}_{feature_cfg.rvol_on.upper()}'] = \
                    _ewma_vol(feat[base_col], span=feature_cfg.rvol_span)

        # Market proxy
        if proxy_ret is not None:
            feat['MKT'] = proxy_ret.reindex(df.index)
        out = pd.DataFrame(feat)
        cal = build_calendar_frame(out.index, feature_cfg.calendar)
        out = pd.concat([out, cal], axis=1)
        # NOTE: Do NOT dropna() here so coverage computation remains meaningful upstream.
        return out

    per_ticker = {}
    min_obs = window + horizon + min_obs_buffer

    for t in tickers:
        if t == feature_cfg.market_proxy:  # skip proxy as an asset
            continue
        df = get_ticker_df(t)
        if df.shape[0] < min_obs:
            continue
        feat_df = build_feature_frame(df)
        # require coverage in train BEFORE dropping NaNs
        train_seg = feat_df.loc[start:val_start]
        coverage = 1.0 - train_seg.isna().any(axis=1).mean() if len(train_seg) else 0.0
        if coverage < min_train_coverage:
            continue
        feat_df = feat_df.dropna()
        if feat_df.shape[0] < min_obs:
            continue
        per_ticker[t] = feat_df

    if not per_ticker:
        raise RuntimeError("No tickers passed the cleaning criteria. Relax thresholds or check dates.")

    assets = sorted(per_ticker.keys())
    asset2id = {a:i for i,a in enumerate(assets)}

    # Determine feature columns & target column (order is fixed for all assets)
    # Use intersection across assets to be safe (so F consistent)
    cols_sets = [set(df.columns) for df in per_ticker.values()]
    common_cols = set.intersection(*cols_sets)
    # target RET_<FIELD>
    target_col = f"RET_{feature_cfg.target_field.upper()}"
    if target_col not in common_cols:
        raise ValueError(f"Target column '{target_col}' not available in common features. "
                         f"Add '{feature_cfg.target_field}' to cfg.price_fields.")
    # Remove target from X later? No: keep it in X as history is useful; Y uses future values of the same
    feature_cols = sorted(list(common_cols))

    # ---- 4) Sliding windows with date-based splits ----
    def window_rows(feat: pd.DataFrame):
        arr = feat[feature_cols].astype(np.float32)
        times = arr.index.to_numpy()
        A = arr.to_numpy(dtype=np.float32)  # [T,F]
        ret_target = feat[target_col].to_numpy(dtype=np.float32)  # [T]
        rows = {'train':[], 'val':[], 'test':[]}
        for i in range(0, len(A) - window - horizon + 1):
            end_ctx_idx = i + window - 1
            end_ctx_date = pd.Timestamp(times[end_ctx_idx])
            x = A[i:i+window]                                          # [K,F]
            y = ret_target[i+window : i+window+horizon]                # [H]
            ctx_t = times[i:i+window]; y_t = times[i+window : i+window+horizon]
            if end_ctx_date < pd.Timestamp(val_start): split = 'train'
            elif end_ctx_date < pd.Timestamp(test_start): split = 'val'
            else:
                if end_ctx_date > pd.Timestamp(end): continue
                split = 'test'
            rows[split].append((x, y, ctx_t, y_t))
        return rows

    splits = {'train':[], 'val':[], 'test':[]}
    id_splits = {'train':[], 'val':[], 'test':[]}

    for a in assets:
        rows = window_rows(per_ticker[a])
        for sp in ('train','val','test'):
            take = rows[sp]
            if max_windows_per_ticker and len(take) > max_windows_per_ticker:
                idxs = np.sort(rng.choice(len(take), size=max_windows_per_ticker, replace=False))
                take = [take[j] for j in idxs]
            splits[sp].extend(take)
            id_splits[sp].extend([asset2id[a]] * len(take))

    def stack_split(lst):
        if len(lst)==0:
            return (np.empty((0, window, len(feature_cols)), np.float32),
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

    # ---- 5) Train-only normalization (per-ticker optional) + clamping ----
    if normalize_per_ticker:
        Fdim = Xtr.shape[-1]
        mean_x = np.zeros((len(assets), 1, 1, Fdim), np.float32)
        std_x  = np.ones ((len(assets), 1, 1, Fdim), np.float32)
        mean_y = np.zeros((len(assets), 1, horizon), np.float32)
        std_y  = np.ones ((len(assets), 1, horizon), np.float32)

        for aid in range(len(assets)):
            m = (ids_tr == aid)
            if not np.any(m): continue
            _X, _Y = Xtr[m], Ytr[m]
            mx = _X.mean(axis=(0,1), keepdims=True); sx = _X.std(axis=(0,1), keepdims=True); sx[sx==0]=1.0
            my = _Y.mean(axis=0, keepdims=True);     sy = _Y.std(axis=0, keepdims=True);     sy[sy==0]=1.0
            mean_x[aid], std_x[aid], mean_y[aid], std_y[aid] = mx, sx, my, sy

        def norm_apply(X, Y, ids):
            Xn = X.copy(); Yn = Y.copy()
            for aid in np.unique(ids):
                m = (ids == aid)
                mx, sx = mean_x[aid], std_x[aid]
                lo, hi = mx - clamp_sigma*sx, mx + clamp_sigma*sx
                Xn[m] = np.clip(Xn[m], lo, hi)
                Xn[m] = (Xn[m] - mx) / sx
                my, sy = mean_y[aid], std_y[aid]
                lo_y, hi_y = my - clamp_sigma*sy, my + clamp_sigma*sy
                Yn[m] = np.clip(Yn[m], lo_y, hi_y)
                Yn[m] = (Yn[m] - my) / sy
            return Xn, Yn

        Xtr, Ytr = norm_apply(Xtr, Ytr, ids_tr)
        if len(Xva): Xva, Yva = norm_apply(Xva, Yva, ids_va)
        if len(Xte): Xte, Yte = norm_apply(Xte, Yte, ids_te)

        norm_stats = {
            'per_ticker': True,
            'mean_x': mean_x.tolist(), 'std_x': std_x.tolist(),
            'mean_y': mean_y.tolist(), 'std_y': std_y.tolist(),
            'assets': assets,
        }
    else:
        mean_x = Xtr.mean(axis=(0,1), keepdims=True); std_x = Xtr.std(axis=(0,1), keepdims=True); std_x[std_x==0]=1.0
        mean_y = Ytr.mean(axis=0, keepdims=True);     std_y = Ytr.std(axis=0, keepdims=True);     std_y[std_y==0]=1.0
        lo, hi = mean_x - clamp_sigma*std_x, mean_x + clamp_sigma*std_x
        Xtr = np.clip(Xtr, lo, hi); Xtr = (Xtr - mean_x) / std_x
        if len(Xva): Xva = np.clip(Xva, lo, hi); Xva = (Xva - mean_x) / std_x
        if len(Xte): Xte = np.clip(Xte, lo, hi); Xte = (Xte - mean_x) / std_x
        lo_y, hi_y = mean_y - clamp_sigma*std_y, mean_y + clamp_sigma*std_y
        Ytr = np.clip(Ytr, lo_y, hi_y); Ytr = (Ytr - mean_y) / std_y
        if len(Yva): Yva = np.clip(Yva, lo_y, hi_y); Yva = (Yva - mean_y) / std_y
        if len(Yte): Yte = np.clip(Yte, lo_y, hi_y); Yte = (Yte - mean_y) / std_y
        norm_stats = {
            'per_ticker': False,
            'mean_x': mean_x.tolist(), 'std_x': std_x.tolist(),
            'mean_y': mean_y.tolist(), 'std_y': std_y.tolist(),
            'assets': assets,
        }

    # --- add these sanity checks right before np.savez_compressed(...) ---
    def _assert_no_nans(name, arr):
        if arr.size == 0:
            return
        if np.isnan(arr.astype(np.float32)).any():
            raise ValueError(f"{name} contains NaNs; check feature construction and dropna().")

    _assert_no_nans("train_X", Xtr);
    _assert_no_nans("train_Y", Ytr)
    _assert_no_nans("val_X", Xva);
    _assert_no_nans("val_Y", Yva)
    _assert_no_nans("test_X", Xte);
    _assert_no_nans("test_Y", Yte)


    # ---- 6) Save compressed NPZ + meta ----

    np.savez_compressed(
        cache_npz,
        train_X=Xtr, train_Y=Ytr, train_asset_id=ids_tr, train_ctx_times=CTtr, train_y_times=YTtr,
        val_X=Xva,   val_Y=Yva,   val_asset_id=ids_va,   val_ctx_times=CTva,  val_y_times=YTva,
        test_X=Xte,  test_Y=Yte,  test_asset_id=ids_te,  test_ctx_times=CTte, test_y_times=YTte,
    )
    meta = {
        'assets': assets, 'asset2id': asset2id,
        'start': start, 'val_start': val_start, 'test_start': test_start, 'end': end,
        'window': window, 'horizon': horizon,
        'feature_cols': feature_cols, 'target_col': target_col,
        'feature_cfg': {
            'price_fields': feature_cfg.price_fields,
            'returns_mode': feature_cfg.returns_mode,
            'include_rvol': feature_cfg.include_rvol,
            'rvol_span': feature_cfg.rvol_span,
            'rvol_on': feature_cfg.rvol_on,
            'include_dlv': feature_cfg.include_dlv,
            'market_proxy': feature_cfg.market_proxy,
            'include_oc': feature_cfg.include_oc,
            'include_gap': feature_cfg.include_gap,
            'include_hl_range': feature_cfg.include_hl_range,
            'target_field': feature_cfg.target_field,
        },
        'normalize_per_ticker': normalize_per_ticker,
        'clamp_sigma': clamp_sigma,
        'min_obs_buffer': min_obs_buffer,
        'min_train_coverage': min_train_coverage,
        'liquidity_rank_window': liquidity_rank_window,
        'top_n_by_dollar_vol': top_n_by_dollar_vol,
        'max_windows_per_ticker': max_windows_per_ticker,
        'regression': regression,
        'seed': seed,
    }
    with open(meta_json, 'w') as f: json.dump(meta, f, indent=2)
    with open(norm_json, 'w') as f: json.dump(norm_stats, f)

    return ((Xtr, Ytr, ids_tr), (Xva, Yva, ids_va), (Xte, Yte, ids_te))


import os, json, random
import numpy as np
import torch
from torch.utils.data import DataLoader

def load_dataloaders_with_ratio_split(
    data_dir='./data',
    train_ratio=0.55,
    val_ratio=0.05,
    test_ratio=0.4,
    batch_size=64,
    regression=True,
    per_asset=True,            # split *within each asset* chronologically
    shuffle_train=True,
    num_workers=2,
    pin_memory=None,
    seed=1337,
):
    """
    Chronological ratio split loader.
    Uses cache_v2.npz + meta_v2.json; splits *per asset* (or globally) by ratios.
    Returns: train_dl, val_dl, test_dl, (n_train, n_val, n_test)
    """
    cache_npz = os.path.join(data_dir, 'cache_v2.npz')
    meta_json = os.path.join(data_dir, 'meta_v2.json')
    if not (os.path.exists(cache_npz) and os.path.exists(meta_json)):
        raise FileNotFoundError(f"Missing {cache_npz} or {meta_json}.")

    z = np.load(cache_npz, allow_pickle=True)
    with open(meta_json, 'r') as f:
        meta = json.load(f)
    assets = meta.get('assets', None)

    # ---- pull and recombine all splits ----
    parts = []
    for p in ('train', 'val', 'test'):
        X = z.get(f'{p}_X'); Y = z.get(f'{p}_Y')
        ids = z.get(f'{p}_asset_id')
        ctx_t = z.get(f'{p}_ctx_times'); y_t = z.get(f'{p}_y_times')
        if X is not None and X.size:
            parts.append((X, Y, ids, ctx_t, y_t))
    if not parts:
        raise RuntimeError("No data found in cache_v2.npz")

    X_all   = np.concatenate([p[0] for p in parts], axis=0)
    Y_all   = np.concatenate([p[1] for p in parts], axis=0)
    IDS_all = np.concatenate([p[2] for p in parts], axis=0).astype(np.int32)
    CT_all  = np.concatenate([p[3] for p in parts], axis=0)  # datetime64[ns]
    YT_all  = np.concatenate([p[4] for p in parts], axis=0)  # datetime64[ns]
    if Y_all.ndim == 3 and Y_all.shape[-1] == 1:
        Y_all = Y_all[..., 0]

    # ---- chronological order: (asset_id, end_of_context_time) ----
    end_ctx = CT_all[:, -1]
    order = np.lexsort((end_ctx, IDS_all))
    X_all, Y_all, IDS_all, CT_all, YT_all = X_all[order], Y_all[order], IDS_all[order], CT_all[order], YT_all[order]

    def _stack(lst):
        if not lst:
            K, F, H = X_all.shape[1], X_all.shape[2], Y_all.shape[1]
            return (np.empty((0, K, F), np.float32),
                    np.empty((0, H), np.float32),
                    np.empty((0,), np.int32),
                    np.empty((0, K), 'datetime64[ns]'),
                    np.empty((0, H), 'datetime64[ns]'))
        X = np.stack([t[0] for t in lst], axis=0)
        Y = np.stack([t[1] for t in lst], axis=0)
        I = np.stack([t[2] for t in lst], axis=0)
        C = np.stack([t[3] for t in lst], axis=0)
        T = np.stack([t[4] for t in lst], axis=0)
        return X, Y, I, C, T

    def _split_counts(n, tr, vr, te):
        s = float(tr + vr + te)
        trn = int(np.floor(n * (tr / s)))
        van = int(np.floor(n * (vr / s)))
        ten = n - trn - van
        if n >= 3:
            if trn == 0: trn, ten = 1, ten - 1
            if van == 0 and n - trn >= 2: van, ten = 1, ten - 1
            if ten == 0: ten = 1
        return trn, van, ten

    train_rows, val_rows, test_rows = [], [], []
    if per_asset:
        for aid in np.unique(IDS_all):
            m = (IDS_all == aid)
            Xa, Ya, Ia, Ca, Ta = X_all[m], Y_all[m], IDS_all[m], CT_all[m], YT_all[m]
            n = Xa.shape[0]
            if n == 0: continue
            trn, van, ten = _split_counts(n, train_ratio, val_ratio, test_ratio)
            train_rows.extend([(Xa[i], Ya[i], Ia[i], Ca[i], Ta[i]) for i in range(0, trn)])
            val_rows.extend(  [(Xa[i], Ya[i], Ia[i], Ca[i], Ta[i]) for i in range(trn, trn+van)])
            test_rows.extend( [(Xa[i], Ya[i], Ia[i], Ca[i], Ta[i]) for i in range(trn+van, n)])
    else:
        n = X_all.shape[0]
        trn, van, ten = _split_counts(n, train_ratio, val_ratio, test_ratio)
        train_rows = [(X_all[i], Y_all[i], IDS_all[i], CT_all[i], YT_all[i]) for i in range(0, trn)]
        val_rows   = [(X_all[i], Y_all[i], IDS_all[i], CT_all[i], YT_all[i]) for i in range(trn, trn+van)]
        test_rows  = [(X_all[i], Y_all[i], IDS_all[i], CT_all[i], YT_all[i]) for i in range(trn+van, n)]

    Xtr, Ytr, IDtr, CTtr, YTtr = _stack(train_rows)
    Xva, Yva, IDva, CTva, YTva = _stack(val_rows)
    Xte, Yte, IDte, CTte, YTe  = _stack(test_rows)

    def _assert_no_nans(name, arr):
        if arr.size and np.isnan(arr.astype(np.float32)).any():
            raise ValueError(f"{name} has NaNs after ratio split; check upstream prep.")
    _assert_no_nans("train_X", Xtr); _assert_no_nans("train_Y", Ytr)
    _assert_no_nans("val_X",   Xva); _assert_no_nans("val_Y",   Yva)
    _assert_no_nans("test_X",  Xte); _assert_no_nans("test_Y",  YTe)

    # Reproducible shuffling in DataLoader
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    gen = torch.Generator()
    gen.manual_seed(seed)

    ds_tr = LabeledWindowWithMetaDataset(Xtr, Ytr, IDtr, assets, CTtr, YTtr, regression=regression)
    ds_va = LabeledWindowWithMetaDataset(Xva, Yva, IDva, assets, CTva, YTva, regression=regression)
    ds_te = LabeledWindowWithMetaDataset(Xte, YTe, IDte, assets, CTte, YTe,  regression=regression)

    def _mk(ds, split):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == 'train' and shuffle_train),
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            generator=gen,
            collate_fn=collate_keep_meta,  # keep datetime meta intact
        )

    train_dl = _mk(ds_tr, 'train')
    val_dl   = _mk(ds_va, 'val')
    test_dl  = _mk(ds_te, 'test')
    return train_dl, val_dl, test_dl, (len(ds_tr), len(ds_va), len(ds_te))