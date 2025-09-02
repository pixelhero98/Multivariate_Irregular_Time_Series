from __future__ import annotations
from dataclasses import dataclass, field
import os, json, gc
from math import ceil as _ceil
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler as _Sampler

# --------------------- Public configs (kept compatible) ---------------------

@dataclass
class CalendarConfig:
    include_dow: bool = True
    include_dom: bool = True
    include_moy: bool = True
    dow_period: int = 7
    moy_period: int = 12
    dow_sin_name: str = "DOW_SIN"
    dow_cos_name: str = "DOW_COS"
    dom_sin_name: str = "DOM_SIN"
    dom_cos_name: str = "DOM_COS"
    moy_sin_name: str = "MOY_SIN"
    moy_cos_name: str = "MOY_COS"

@dataclass
class FeatureConfig:
    price_fields: List[str] = field(default_factory=lambda: ["Close"])  # which to convert to returns
    returns_mode: str = "log"       # 'log' or 'pct'
    include_rvol: bool = True
    rvol_span: int = 20
    rvol_on: str = "Close"
    include_dlv: bool = True
    market_proxy: Optional[str] = "SPY"
    include_oc: bool = False
    include_gap: bool = False
    include_hl_range: bool = False
    target_field: str = "Close"     # used for Y
    calendar: CalendarConfig = field(default_factory=CalendarConfig)
    include_entity_id_feature: bool = False

# --------------------- Lightweight stores ---------------------

def _indexcache_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "cache_ratio_index")

def _features_dir(data_dir: str) -> str:
    return os.path.join(_indexcache_dir(data_dir), "features_fp16")

def _targets_dir(data_dir: str) -> str:
    return os.path.join(_indexcache_dir(data_dir), "targets_fp16")

def _times_dir(data_dir: str) -> str:
    return os.path.join(_indexcache_dir(data_dir), "times")

def _windows_dir(data_dir: str) -> str:
    return os.path.join(_indexcache_dir(data_dir), "windows")

def _meta_path(data_dir: str) -> str:
    return os.path.join(_indexcache_dir(data_dir), "meta.json")

def _norm_path(data_dir: str) -> str:
    return os.path.join(_indexcache_dir(data_dir), "norm_stats.json")

# --------------------- Feature engineering (same semantics) ---------------------

_EPS = 1e-6

def _mask_nonpos(s: 'pd.Series') -> 'pd.Series':
    return s.where((s > 0) & np.isfinite(s))

def _safe_log_series(s: 'pd.Series') -> 'pd.Series':
    return np.log(_mask_nonpos(s))

def _safe_log1p_series(s: 'pd.Series') -> 'pd.Series':
    return np.log1p(s.clip(lower=-1 + _EPS))

def _safe_pct_change(s: 'pd.Series') -> 'pd.Series':
    return s.pct_change().replace([np.inf, -np.inf], np.nan)

def _log_return(s: 'pd.Series') -> 'pd.Series':
    return _safe_log_series(s).diff()

def _ewma_vol(ret: 'pd.Series', span: int = 20) -> 'pd.Series':
    return ret.pow(2).ewm(span=span, adjust=False).mean().pow(0.5)

def _delta_log_volume(vol: 'pd.Series') -> 'pd.Series':
    v = vol.replace([0, np.inf, -np.inf], np.nan)
    return _safe_log_series(v).diff()


def _cyclical_from_int(values: np.ndarray, period: int):
    ang = 2.0 * np.pi * (values.astype(np.float32) / float(period))
    return np.sin(ang).astype(np.float32), np.cos(ang).astype(np.float32)


def build_calendar_frame(idx: 'pd.DatetimeIndex', cfg: CalendarConfig) -> 'pd.DataFrame':
    import pandas as pd
    cols = {}
    if cfg.include_dow:
        dow = idx.dayofweek.values
        s, c = _cyclical_from_int(dow, cfg.dow_period)
        cols[cfg.dow_sin_name] = s
        cols[cfg.dow_cos_name] = c
    if cfg.include_dom:
        dom = idx.day.values
        dim = idx.days_in_month.values
        ang = 2.0 * np.pi * ((dom - 1).astype(np.float32) / dim.astype(np.float32).clip(min=1))
        cols[cfg.dom_sin_name] = np.sin(ang).astype(np.float32)
        cols[cfg.dom_cos_name] = np.cos(ang).astype(np.float32)
    if cfg.include_moy:
        moy = (idx.month.values - 1)
        s, c = _cyclical_from_int(moy, cfg.moy_period)
        cols[cfg.moy_sin_name] = s
        cols[cfg.moy_cos_name] = c
    out = pd.DataFrame(cols, index=idx)
    for c in out.columns:
        if out[c].dtype == np.float64:
            out[c] = out[c].astype(np.float32)
    return out

# --------------------- Compact cache builder ---------------------

def prepare_features_and_index_cache(
    tickers: List[str],
    start: str,
    end: str,
    window: int,
    horizon: int,
    data_dir: str = "./data",
    feature_cfg: FeatureConfig = FeatureConfig(),
    normalize_per_ticker: bool = True,
    clamp_sigma: float = 5.0,
    min_obs_buffer: int = 50,
    min_train_coverage: float = 0.9,
    liquidity_rank_window: Optional[Tuple[str,str]] = None,
    top_n_by_dollar_vol: Optional[int] = None,
    max_windows_per_ticker: Optional[int] = None,  # applies at *index* build time
    regression: bool = True,
    seed: int = 1337,
    keep_time_meta: str = "end",  # "full" | "end" | "none"
):
    """Builds a *compact* cache: per-ticker feature matrices + global window index.
    - No split-by-date; splitting is performed later (ratio-based) in the loader.
    - Stores float16 features/targets to halve disk usage again.
    """
    try:
        import pandas as pd
        import yfinance as yf
    except Exception as e:
        raise ImportError("pandas + yfinance required to prepare cache. pip install pandas yfinance") from e

    rng = np.random.RandomState(seed)
    os.makedirs(data_dir, exist_ok=True)
    root = _indexcache_dir(data_dir)
    os.makedirs(root, exist_ok=True)
    os.makedirs(_features_dir(data_dir), exist_ok=True)
    os.makedirs(_targets_dir(data_dir), exist_ok=True)
    os.makedirs(_times_dir(data_dir), exist_ok=True)
    os.makedirs(_windows_dir(data_dir), exist_ok=True)

    # ---- download / load features per ticker (reuses your feature logic) ----
    need_price_cols = set(feature_cfg.price_fields) | {feature_cfg.target_field}
    if feature_cfg.include_oc or feature_cfg.include_gap:
        need_price_cols |= {"Open", "Close"}
    if feature_cfg.include_hl_range:
        need_price_cols |= {"High", "Low"}
    wanted_cols = sorted(list(need_price_cols | ({'Volume'} if feature_cfg.include_dlv else set())))

    tickers_dl = tickers[:]
    if feature_cfg.market_proxy and feature_cfg.market_proxy not in tickers_dl:
        tickers_dl.append(feature_cfg.market_proxy)

    raw = yf.download(tickers_dl, start=start, end=end, auto_adjust=True, group_by="column", progress=False)
    if hasattr(raw, "columns") and getattr(raw.columns, "nlevels", 1) > 1:
        # normalize to (field, ticker)
        raw = raw.swaplevel(0, 1, axis=1).sort_index(axis=1)

    def get_ticker_df(t: str) -> 'pd.DataFrame':
        if getattr(raw, "columns", None) is not None and getattr(raw.columns, "nlevels", 1) > 1:
            try:
                df = raw.xs(t, axis=1, level=1)
            except Exception:
                df = raw.xs(t, axis=1, level=0)
        else:
            df = raw
        keep = [c for c in wanted_cols if c in df.columns]
        return df[keep].sort_index()

    proxy_ret = None
    if feature_cfg.market_proxy:
        try:
            proxy_df = get_ticker_df(feature_cfg.market_proxy)
            if 'Close' in proxy_df:
                proxy_ret = (_log_return(proxy_df['Close']) if feature_cfg.returns_mode == 'log' else _safe_pct_change(proxy_df['Close'])).rename('MKT')
        except Exception:
            proxy_ret = None

    def median_dollar_volume(df: 'pd.DataFrame', a: str, b: str) -> float:
        sub = df.loc[a:b]
        if 'Close' not in sub or 'Volume' not in sub:
            return 0.0
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

    def build_feature_frame(df: 'pd.DataFrame') -> 'pd.DataFrame':
        feat = {}
        for field in feature_cfg.price_fields:
            if field in df:
                if feature_cfg.returns_mode == 'log':
                    feat[f'RET_{field.upper()}'] = _log_return(df[field])
                else:
                    feat[f'RET_{field.upper()}'] = _safe_pct_change(df[field])
        if feature_cfg.include_oc and 'Open' in df and 'Close' in df:
            oc = (_safe_log_series(df['Close']) - _safe_log_series(df['Open'])
                  if feature_cfg.returns_mode == 'log' else (df['Close'] / df['Open'] - 1.0))
            feat['OC_RET'] = oc
        if feature_cfg.include_gap and 'Open' in df and 'Close' in df:
            gap = (_safe_log_series(df['Open']) - _safe_log_series(df['Close'].shift(1))
                   if feature_cfg.returns_mode == 'log' else (df['Open'] / df['Close'].shift(1) - 1.0))
            feat['GAP_RET'] = gap
        if feature_cfg.include_hl_range and 'High' in df and 'Low' in df:
            hlr = (_safe_log_series(df['High']) - _safe_log_series(df['Low'])
                   if feature_cfg.returns_mode == 'log' else (df['High'] / df['Low'] - 1.0))
            feat['HL_RANGE'] = hlr
        if feature_cfg.include_dlv and 'Volume' in df:
            feat['DLV'] = _delta_log_volume(df['Volume'])
        if feature_cfg.include_rvol:
            base_col = f"RET_{feature_cfg.rvol_on.upper()}"
            if base_col in feat:
                feat[f'RVOL{feature_cfg.rvol_span}_{feature_cfg.rvol_on.upper()}'] = _ewma_vol(feat[base_col], span=feature_cfg.rvol_span)
        if proxy_ret is not None:
            feat['MKT'] = proxy_ret.reindex(df.index)
        out = pd.DataFrame(feat)
        cal = build_calendar_frame(out.index, feature_cfg.calendar)
        out = pd.concat([out, cal], axis=1)
        out = out.dropna()
        # enforce float32 now, will save to float16 later
        for c in out.columns:
            if out[c].dtype == np.float64:
                out[c] = out[c].astype(np.float32)
        return out

    # ---- Build per-ticker features ----
    per_ticker: Dict[str, 'pd.DataFrame'] = {}
    min_obs = window + horizon + min_obs_buffer
    for t in tickers:
        if t == feature_cfg.market_proxy:
            continue
        df_raw = get_ticker_df(t)
        if df_raw.shape[0] < min_obs:
            continue
        feat_df = build_feature_frame(df_raw)
        if feat_df.shape[0] < min_obs:
            continue
        # coverage check using first 80% of sample as proxy for train adequacy
        train_like = feat_df.iloc[: max(1, int(0.8 * len(feat_df)))]
        coverage = 1.0 - train_like.isna().any(axis=1).mean() if len(train_like) else 0.0
        if coverage < min_train_coverage:
            continue
        per_ticker[t] = feat_df

    if not per_ticker:
        raise RuntimeError("No tickers passed the cleaning criteria. Relax thresholds or check dates.")

    # ---- Align feature columns across tickers ----
    assets = sorted(per_ticker.keys())
    asset2id = {a: i for i, a in enumerate(assets)}

    if getattr(feature_cfg, 'include_entity_id_feature', False):
        denom = max(1, len(assets) - 1)
        for a in assets:
            aid = asset2id[a]
            val = np.float32(aid / denom) if denom > 0 else np.float32(0.0)
            per_ticker[a]['ENTITY_ID'] = np.full((len(per_ticker[a])), val, dtype=np.float32)

    col_sets = [set(df.columns) for df in per_ticker.values()]
    feature_cols = sorted(list(set.intersection(*col_sets)))
    target_col = f"RET_{feature_cfg.target_field.upper()}"
    if target_col not in feature_cols:
        raise ValueError(f"Target column '{target_col}' not in common features. Add '{feature_cfg.target_field}' to cfg.price_fields.")

    # ---- Save compact arrays ----
    # Per-ticker matrices: X[t] shape [T,F] float16, Y[t] shape [T] float16, times[t] datetime64
    for a in assets:
        df = per_ticker[a]
        X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        Y = df[target_col].to_numpy(dtype=np.float32, copy=False)
        times = df.index.to_numpy()
        # Write as float16 (2× smaller) – okay post-normalization/standardization
        np.save(os.path.join(_features_dir(data_dir), f"{asset2id[a]}.npy"), X.astype(np.float16))
        np.save(os.path.join(_targets_dir(data_dir),  f"{asset2id[a]}.npy"), Y.astype(np.float16))
        np.save(os.path.join(_times_dir(data_dir),    f"{asset2id[a]}.npy"), times.astype('datetime64[ns]'))

    # ---- Precompute a *global* window index (small) ----
    # This is optional but speeds up loader start; also lets us cap max_windows_per_ticker deterministically
    pairs: List[np.ndarray] = []   # (aid, start_idx)
    ends:  List[np.ndarray] = []   # end-of-context times
    for a in assets:
        aid = asset2id[a]
        times = np.load(os.path.join(_times_dir(data_dir), f"{aid}.npy"))
        T = times.shape[0]
        if T < (window + horizon):
            continue
        start_idxs = np.arange(0, T - window - horizon + 1, dtype=np.int32)
        if max_windows_per_ticker is not None and start_idxs.size > max_windows_per_ticker:
            start_idxs = start_idxs[:max_windows_per_ticker]
        end_times = times[start_idxs + window - 1]
        pairs.append(np.stack([np.full_like(start_idxs, aid), start_idxs], axis=1))
        ends.append(end_times.astype('datetime64[ns]'))

    if not pairs:
        raise RuntimeError("No valid windows across assets after indexing.")

    global_pairs = np.concatenate(pairs, axis=0).astype(np.int32)    # [M,2]
    end_times    = np.concatenate(ends,  axis=0).astype('datetime64[ns]')  # [M]

    # Persist tiny index
    np.save(os.path.join(_windows_dir(data_dir), "global_pairs.npy"), global_pairs)
    np.save(os.path.join(_windows_dir(data_dir), "end_times.npy"), end_times)

    # ---- Norm stats (per-ticker, scalar Y) ----
    norm_stats = {
        'per_ticker': normalize_per_ticker,
        'assets': assets,
        'mean_x': [], 'std_x': [],   # list per asset -> [1,1,F] if per_ticker else [1,1,F]
        'mean_y': [], 'std_y': [],
    }
    if normalize_per_ticker:
        for a in assets:
            aid = asset2id[a]
            Xf = np.load(os.path.join(_features_dir(data_dir), f"{aid}.npy"), mmap_mode='r')
            Yf = np.load(os.path.join(_targets_dir(data_dir),  f"{aid}.npy"), mmap_mode='r')
            mx = Xf.astype(np.float32).mean(axis=0, keepdims=True)[None, ...]   # [1,1,F]
            sx = Xf.astype(np.float32).std(axis=0, keepdims=True)[None, ...]
            sx[sx == 0] = 1.0
            my = float(Yf.astype(np.float32).mean())
            sy = float(Yf.astype(np.float32).std()); sy = (1.0 if sy == 0 else sy)
            norm_stats['mean_x'].append(mx.tolist()); norm_stats['std_x'].append(sx.tolist())
            norm_stats['mean_y'].append(my);          norm_stats['std_y'].append(sy)
    else:
        # global stats shared by all assets
        Xs, Ys = [], []
        for a in assets:
            aid = asset2id[a]
            Xs.append(np.load(os.path.join(_features_dir(data_dir), f"{aid}.npy"), mmap_mode='r').astype(np.float32))
            Ys.append(np.load(os.path.join(_targets_dir(data_dir),  f"{aid}.npy"), mmap_mode='r').astype(np.float32))
        Xcat = np.concatenate(Xs, axis=0)
        Ycat = np.concatenate(Ys, axis=0)
        mx = Xcat.mean(axis=0, keepdims=True)[None, ...]
        sx = Xcat.std(axis=0, keepdims=True)[None, ...]; sx[sx == 0] = 1.0
        my = float(Ycat.mean()); sy = float(Ycat.std()); sy = (1.0 if sy == 0 else sy)
        norm_stats['mean_x'] = mx.tolist(); norm_stats['std_x'] = sx.tolist()
        norm_stats['mean_y'] = my;           norm_stats['std_y'] = sy
        del Xcat, Ycat
    with open(_norm_path(data_dir), 'w') as f:
        json.dump(norm_stats, f)

    # ---- Meta
    meta = {
        'format': 'indexcache_v1',
        'assets': assets,
        'asset2id': {a:i for i,a in enumerate(assets)},
        'start': start, 'end': end,
        'window': int(window), 'horizon': int(horizon),
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
        'normalize_per_ticker': bool(normalize_per_ticker),
        'clamp_sigma': float(clamp_sigma),
        'min_obs_buffer': int(min_obs_buffer),
        'liquidity_rank_window': liquidity_rank_window,
        'top_n_by_dollar_vol': top_n_by_dollar_vol,
        'max_windows_per_ticker': max_windows_per_ticker,
        'regression': bool(regression),
        'seed': int(seed),
        'keep_time_meta': keep_time_meta,
    }
    with open(_meta_path(data_dir), 'w') as f:
        json.dump(meta, f, indent=2)

    # Free big refs
    del raw, per_ticker
    gc.collect()

    return True

# --------------------- Reindex-only ---------------------------------------------------------
def rebuild_window_index_only(
    data_dir: str,
    window: int,
    horizon: int,
    max_windows_per_ticker: Optional[int] = None,
    update_meta: bool = True,
    backup_old: bool = True,
) -> int:
    """
    Rebuilds windows/global_pairs.npy and windows/end_times.npy for a NEW (K,H)
    using existing per-ticker times. Fast: does NOT touch features/targets.

    Returns the total number of indexed windows.
    """
    import shutil
    base = _indexcache_dir(data_dir)
    times_dir = _times_dir(data_dir)
    windows_dir = _windows_dir(data_dir)
    meta_path = _meta_path(data_dir)

    with open(meta_path, "r") as f:
        meta = json.load(f)
    assets = meta["assets"]

    pairs_list, ends_list = [], []
    for aid in range(len(assets)):
        tp = os.path.join(times_dir, f"{aid}.npy")
        if not os.path.exists(tp):
            continue
        times = np.load(tp)  # datetime64[ns]
        T = int(times.shape[0])
        if T < window + horizon:
            continue
        starts = np.arange(0, T - window - horizon + 1, dtype=np.int32)
        if max_windows_per_ticker is not None and starts.size > max_windows_per_ticker:
            starts = starts[:max_windows_per_ticker]
        end_times = times[starts + window - 1]
        pairs_list.append(np.stack([np.full_like(starts, aid), starts], axis=1))
        ends_list.append(end_times.astype("datetime64[ns]"))

    if not pairs_list:
        raise RuntimeError("No windows with the requested (window,horizon). Try smaller values.")

    global_pairs = np.concatenate(pairs_list, axis=0).astype(np.int32)
    end_times = np.concatenate(ends_list, axis=0).astype("datetime64[ns]")

    os.makedirs(windows_dir, exist_ok=True)
    gp_path = os.path.join(windows_dir, "global_pairs.npy")
    et_path = os.path.join(windows_dir, "end_times.npy")
    if backup_old:
        for p in (gp_path, et_path):
            if os.path.exists(p):
                shutil.move(p, p + ".bak")

    np.save(gp_path, global_pairs)
    np.save(et_path, end_times)

    if update_meta:
        meta["window"] = int(window)
        meta["horizon"] = int(horizon)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    return int(global_pairs.shape[0])

# --------------------- Utility samplers & collates (grouping preserved) ---------------------

class _ListBatchSampler(_Sampler):
    def __init__(self, batches: Sequence[Sequence[int]]):
        self.batches = [list(b) for b in batches if len(b)]
    def __iter__(self):
        for b in self.batches:
            yield b
    def __len__(self):
        return len(self.batches)

def make_collate_level_and_firstdiff(
    n_entities: int = 1,
    return_entity_mask: bool = True,
    **_ignored,  # absorbs unused kwargs from the loader
):
    """
    Panel collate that builds date-wise panels of size [B, N, K, F] where N = n_entities
    (i.e., the full asset universe). Slots are addressed by integer asset_id in [0, N-1].
    Missing entities per date are zero-filled and masked out in meta['entity_mask'].

    Returns:
      [V, T], Y, meta
        V: [B, N, K, F]  levels
        T: [B, N, K, F]  first differences along time
        Y: [B, N, H]
        meta: {
          'entity_mask': [B, N] bool,
          'dates': list of python dates (len B),
          'ctx_times': list[list[Any]] of length [B][N] (None for missing)
        }
    """
    import numpy as _np
    import torch as _torch
    import datetime as _dt

    def _to_day_key(ctx_times):
        # ctx_times can be scalar or array-like; use last (end-of-context)
        if isinstance(ctx_times, (list, tuple, _np.ndarray)):
            ctx_times = ctx_times[-1]
        try:
            d64 = _np.datetime64(ctx_times, 'ns')
        except Exception:
            d64 = _np.datetime64(str(ctx_times))
        return int(d64.astype('datetime64[D]').astype(_np.int64))

    def _key_to_pydate(day_key: int):
        d64 = _np.datetime64(day_key, 'D')
        y, m, d = map(int, _np.datetime_as_string(d64).split('-'))
        return _dt.date(y, m, d)

    def _first_diff(x: _torch.Tensor):
        t = _torch.zeros_like(x)
        t[1:] = x[1:] - x[:-1]
        return t

    def collate(batch):
        # ---- discover unique end dates (B) ----
        day_keys = [_to_day_key(meta.get("ctx_times")) for _, _, meta in batch]
        date_to_idx, dates_order = {}, []
        for dk in day_keys:
            if dk not in date_to_idx:
                date_to_idx[dk] = len(dates_order)
                dates_order.append(dk)
        B = len(dates_order)

        # ---- infer shapes & dtypes from first sample ----
        x0, y0, _ = batch[0]
        x0 = _torch.as_tensor(x0)
        K, F = int(x0.shape[0]), int(x0.shape[1])
        y0 = _torch.as_tensor(y0)
        H = int(y0.shape[-1]) if y0.ndim else 1
        x_dtype, y_dtype = x0.dtype, y0.dtype

        N = int(n_entities)  # full universe width

        # ---- allocate (zeros are fine — will be masked) ----
        V = _torch.zeros((B, N, K, F), dtype=x_dtype)
        T = _torch.zeros_like(V)
        Y = _torch.zeros((B, N, H), dtype=y_dtype)
        M = _torch.zeros((B, N), dtype=_torch.bool)
        ctx_times = [[None] * N for _ in range(B)]  # <— make this BEFORE the fill loop

        # ---- fill panels ----
        for x, y, meta in batch:
            b = date_to_idx[_to_day_key(meta.get("ctx_times"))]
            aid = int(meta["asset_id"])  # must be in [0, N-1]
            if 0 <= aid < N:
                n = aid  # slot index
                xt = _torch.as_tensor(x, dtype=x_dtype)
                V[b, n] = xt
                T[b, n] = _first_diff(xt)

                yt = _torch.as_tensor(y, dtype=y_dtype)
                if yt.ndim == 0:
                    Y[b, n, 0] = yt
                else:
                    Y[b, n, :yt.shape[-1]] = yt

                M[b, n] = True
                ctx_times[b][n] = meta.get("ctx_times")  # <— now indices exist
            else:
                # asset outside universe: ignore (optional: log a warning)
                pass

        # ---- meta ----
        meta_out = {
            "entity_mask": M if return_entity_mask else None,
            "dates": [_key_to_pydate(k) for k in dates_order],
            "ctx_times": ctx_times,
        }
        return [V, T], Y, meta_out

    return collate

# --------------------- Datasets (index-backed on-the-fly windows) ---------------------

class _IndexBackedDataset(Dataset):
    """On-the-fly slicer from compact per-ticker arrays using stored (asset_id, start_idx).
    Applies (optional) per-ticker normalization + clamp.
    """
    def __init__(self,
                 pairs: np.ndarray,               # [N,2] int32 (aid, start)
                 assets: List[str],
                 data_dir: str,
                 window: int,
                 horizon: int,
                 regression: bool,
                 keep_time_meta: str,
                 norm_stats: dict,
                 clamp_sigma: float,
                 ): 
        self.pairs = pairs
        self.assets = assets
        self.data_dir = data_dir
        self.window = int(window)
        self.horizon = int(horizon)
        self.regression = bool(regression)
        self.keep_time_meta = keep_time_meta
        self.clamp_sigma = float(clamp_sigma)

        # mmap per-ticker arrays lazily
        self._X: Dict[int, np.ndarray] = {}
        self._Y: Dict[int, np.ndarray] = {}
        self._T: Dict[int, np.ndarray] = {}

        self.per_ticker = bool(norm_stats.get('per_ticker', True))
        self.mean_x = norm_stats['mean_x']
        self.std_x  = norm_stats['std_x']
        self.mean_y = norm_stats['mean_y']
        self.std_y  = norm_stats['std_y']

    def __len__(self):
        return self.pairs.shape[0]

    def _get_arrays(self, aid: int):
        if aid not in self._X:
            self._X[aid] = np.load(os.path.join(_features_dir(self.data_dir), f"{aid}.npy"), mmap_mode='r')
            self._Y[aid] = np.load(os.path.join(_targets_dir(self.data_dir),  f"{aid}.npy"), mmap_mode='r')
            self._T[aid] = np.load(os.path.join(_times_dir(self.data_dir),    f"{aid}.npy"), mmap_mode='r')
        return self._X[aid], self._Y[aid], self._T[aid]

    def __getitem__(self, i: int):
        aid, start = self.pairs[i]
        Xf, Yf, Tf = self._get_arrays(int(aid))
        s = int(start); e = s + self.window
        x = Xf[s:e, :].astype(np.float32)  # [K,F]
        y_vec = Yf[e:e+self.horizon].astype(np.float32)  # [H]

        raw_last_for_label = float(y_vec[-1])
        # Normalize + clamp (train-style). Use per-ticker or global stats
        if self.per_ticker:
            mx = np.array(self.mean_x[aid], dtype=np.float32)   # [1,1,F]
            sx = np.array(self.std_x[aid],  dtype=np.float32)
            my = float(self.mean_y[aid]) if isinstance(self.mean_y, list) else float(self.mean_y)
            sy = float(self.std_y[aid])  if isinstance(self.std_y, list)  else float(self.std_y)
        else:
            mx = np.array(self.mean_x, dtype=np.float32)        # [1,1,F]
            sx = np.array(self.std_x,  dtype=np.float32)
            my = float(self.mean_y)
            sy = float(self.std_y)
        lo, hi = mx - self.clamp_sigma * sx, mx + self.clamp_sigma * sx
        x = np.clip(x, lo[0,0], hi[0,0], out=x)
        x = (x - mx[0,0]) / sx[0,0]
        lo_y, hi_y = my - self.clamp_sigma * sy, my + self.clamp_sigma * sy
        y_vec = np.clip(y_vec, lo_y, hi_y, out=y_vec)
        y_vec = (y_vec - my) / sy

        # Torch tensors expected by collate: X->float32, Y->float32 or int
        x_t = torch.tensor(x, dtype=torch.float32)
        if self.regression:
            y_t = torch.tensor(y_vec, dtype=torch.float32)
        else:
            y_t = torch.tensor(int(raw_last_for_label > 0.0), dtype=torch.int64)  # simple example for classification

        meta = {'asset_id': int(aid), 'asset': self.assets[int(aid)]}
        if self.keep_time_meta != 'none':
            if self.keep_time_meta == 'full':
                meta['ctx_times'] = Tf[s:e]
                meta['y_times'] = Tf[e:e+self.horizon]
            else:
                meta['ctx_times'] = Tf[e-1]
                meta['y_times']  = Tf[e+self.horizon-1]
        return x_t, y_t, meta

# --------------------- Date grouping helpers for ratio-split ---------------------

def _normalize_to_day(int64_ns: np.ndarray) -> np.ndarray:
    a = int64_ns.astype('datetime64[D]').astype(np.int64)
    return a

def _build_date_batches_from_pairs(order_pairs: np.ndarray,
                                   end_times: np.ndarray,
                                   dates_per_batch: int,
                                   min_real_entities: int) -> List[np.ndarray]:
    # order_pairs: [M,2] (aid, start) sorted by end_times
    days = _normalize_to_day(end_times)
    order = np.argsort(days, kind='mergesort')
    days_sorted = days[order]
    _, starts = np.unique(days_sorted, return_index=True)
    groups = np.split(order, starts[1:])
    dense_groups = [g for g in groups if g.size >= int(min_real_entities)]
    batches = []
    for k in range(0, len(dense_groups), dates_per_batch):
        chunk = dense_groups[k:k+dates_per_batch]
        if chunk:
            batches.append(np.concatenate(chunk, axis=0))
    return batches

def _compute_train_only_norm_stats(
    data_dir: str,
    assets: List[str],
    tr_pairs: np.ndarray,    # [Nt,2] (aid, start)
    window: int,
    per_ticker: bool,
    feature_dim: int,
) -> dict | None:
    """
    Compute mean/std for X and Y using ONLY rows that can appear in TRAIN contexts.
    For asset a: use prefix up to max_train_end_idx[a] = max(start + window - 1) across train windows.
    Returns a dict like norm_stats.json or None if no train rows exist.
    """
    import numpy as _np
    import os as _os

    last_end = _np.full(len(assets), -1, dtype=_np.int64)
    if tr_pairs.size > 0:
        aids = tr_pairs[:, 0].astype(_np.int64)
        ends = tr_pairs[:, 1].astype(_np.int64) + (window - 1)
        for a, e in zip(aids, ends):
            if e > last_end[a]:
                last_end[a] = e

    has_train = last_end >= 0

    if per_ticker:
        mean_x, std_x, mean_y, std_y = [], [], [], []

        g_count = 0
        g_sum = _np.zeros((feature_dim,), dtype=_np.float64)
        g_sumsq = _np.zeros((feature_dim,), dtype=_np.float64)
        g_y_sum = 0.0
        g_y_sumsq = 0.0

        for aid in range(len(assets)):
            fp = _os.path.join(_features_dir(data_dir), f"{aid}.npy")
            yp = _os.path.join(_targets_dir(data_dir),  f"{aid}.npy")
            if not (_os.path.exists(fp) and _os.path.exists(yp)):
                mean_x.append(_np.zeros((1,1,feature_dim), dtype=_np.float32).tolist())
                std_x.append(_np.ones((1,1,feature_dim), dtype=_np.float32).tolist())
                mean_y.append(0.0); std_y.append(1.0)
                continue

            if has_train[aid]:
                Xf = _np.load(fp, mmap_mode='r').astype(_np.float32)
                Yf = _np.load(yp, mmap_mode='r').astype(_np.float32)
                upto = int(last_end[aid]) + 1
                Xp = Xf[:upto, :]
                Yp = Yf[:upto]

                mx = Xp.mean(axis=0, keepdims=True)[None, ...]
                vx = Xp.var(axis=0, keepdims=True, ddof=0)[None, ...]
                sx = _np.sqrt(_np.maximum(vx, 1e-12)); sx[sx == 0] = 1.0
                my = float(Yp.mean())
                sy = float(Yp.std()); sy = (1.0 if sy == 0 else sy)

                mean_x.append(mx.tolist()); std_x.append(sx.tolist())
                mean_y.append(my);         std_y.append(sy)

                g_count += Xp.shape[0]
                g_sum   += Xp.sum(axis=0, dtype=_np.float64)
                g_sumsq += (Xp.astype(_np.float64) ** 2).sum(axis=0)
                g_y_sum   += float(Yp.sum())
                g_y_sumsq += float((Yp.astype(_np.float64) ** 2).sum())
            else:
                mean_x.append(None); std_x.append(None)
                mean_y.append(None); std_y.append(None)

        if g_count > 0:
            g_mx = (g_sum / g_count).astype(_np.float32)
            g_vx = (g_sumsq / g_count) - (g_mx.astype(_np.float64) ** 2)
            g_vx = _np.maximum(g_vx, 1e-12)
            g_sx = _np.sqrt(g_vx).astype(_np.float32); g_sx[g_sx == 0] = 1.0
            g_my = float(g_y_sum / g_count)
            g_vy = max((g_y_sumsq / g_count) - (g_my ** 2), 1e-12)
            g_sy = float(_np.sqrt(g_vy)); g_sy = (1.0 if g_sy == 0 else g_sy)

            for aid in range(len(assets)):
                if mean_x[aid] is None:
                    mean_x[aid] = g_mx.reshape(1,1,-1).tolist()
                    std_x[aid]  = g_sx.reshape(1,1,-1).tolist()
                    mean_y[aid] = g_my; std_y[aid] = g_sy
        else:
            return None

        return {
            'per_ticker': True,
            'mean_x': mean_x, 'std_x': std_x,
            'mean_y': mean_y, 'std_y': std_y,
        }

    # global stats
    g_count = 0
    g_sum = _np.zeros((feature_dim,), dtype=_np.float64)
    g_sumsq = _np.zeros((feature_dim,), dtype=_np.float64)
    g_y_sum = 0.0
    g_y_sumsq = 0.0

    for aid in range(len(assets)):
        if not has_train[aid]:
            continue
        Xf = _np.load(_os.path.join(_features_dir(data_dir), f"{aid}.npy"), mmap_mode='r').astype(_np.float32)
        Yf = _np.load(_os.path.join(_targets_dir(data_dir),  f"{aid}.npy"), mmap_mode='r').astype(_np.float32)
        upto = int(last_end[aid]) + 1
        Xp = Xf[:upto, :]
        Yp = Yf[:upto]
        g_count += Xp.shape[0]
        g_sum   += Xp.sum(axis=0, dtype=_np.float64)
        g_sumsq += (Xp.astype(_np.float64) ** 2).sum(axis=0)
        g_y_sum   += float(Yp.sum())
        g_y_sumsq += float((Yp.astype(_np.float64) ** 2).sum())

    if g_count == 0:
        return None

    g_mx = (g_sum / g_count).astype(_np.float32)
    g_vx = (g_sumsq / g_count) - (g_mx.astype(_np.float64) ** 2)
    g_vx = _np.maximum(g_vx, 1e-12)
    g_sx = _np.sqrt(g_vx).astype(_np.float32); g_sx[g_sx == 0] = 1.0
    g_my = float(g_y_sum / g_count)
    g_vy = max((g_y_sumsq / g_count) - (g_my ** 2), 1e-12)
    g_sy = float(_np.sqrt(g_vy)); g_sy = (1.0 if g_sy == 0 else g_sy)

    return {
        'per_ticker': False,
        'mean_x': g_mx.reshape(1,1,-1).tolist(),
        'std_x':  g_sx.reshape(1,1,-1).tolist(),
        'mean_y': g_my,
        'std_y':  g_sy,
    }

# --------------------- Ratio-split loader (ONLY) ---------------------

def load_dataloaders_with_ratio_split(
    data_dir: str = './data',
    train_ratio: float = 0.55,
    val_ratio: float = 0.05,
    test_ratio: float = 0.40,
    batch_size: int = 64,
    regression: bool = True,
    per_asset: bool = True,
    norm_scope: str = "train_only",
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    seed: int = 1337,
    n_entities: int = 8,
    pad_incomplete: str = 'zeros',   # kept for API compat
    collate_fn=None,
    coverage_per_window: float = 0.0,
    date_batching: Optional[bool] = None,
    dates_per_batch: int = 4,
    window: Optional[int] = None,
    horizon: Optional[int] = None
):
    # Read meta + norm
    with open(_meta_path(data_dir), 'r') as f:
        meta = json.load(f)
    assets = meta['assets']
    base_window = int(meta['window']);base_horizon = int(meta['horizon'])
    window = int(window if window is not None else base_window)
    horizon = int(horizon if horizon is not None else base_horizon)
    if window > base_window or horizon > base_horizon:
        raise ValueError(f"Requested (window={window}, horizon={horizon}) exceed cached meta "
             f"({base_window}, {base_horizon}). Call rebuild_window_index_only(...) first."
            )

    keep_time_meta = meta.get('keep_time_meta', 'end')
    with open(_norm_path(data_dir), 'r') as f:
        norm_stats = json.load(f)

    # Collate (levels + first-diff), grouped-by-end if requested later
    if collate_fn is None:
        collate_fn = make_collate_level_and_firstdiff(
            n_entities=len(assets),
            return_entity_mask = True
        )

    # Load small global index
    pairs = np.load(os.path.join(_windows_dir(data_dir), 'global_pairs.npy'))  # [M,2]
    end_times = np.load(os.path.join(_windows_dir(data_dir), 'end_times.npy')) # [M]

    # Ensure chronological ordering within asset (pairs might already be grouped, but do it anyway)
    # We'll sort by (asset_id, end_time)
    aid = pairs[:, 0].astype(np.int32)
    if per_asset:
        order = np.lexsort((end_times.astype('datetime64[ns]').astype(np.int64), aid))
    else:
        order = np.argsort(end_times.astype('datetime64[ns]').astype(np.int64))
    pairs = pairs[order]
    end_times = end_times[order]

    # Ratio assignment
    def _split_counts(n, tr, vr, te):
        s = float(tr + vr + te)
        trn = int(np.floor(n * (tr / s)))
        van = int(np.floor(n * (vr / s)))
        ten = n - trn - van
        if n >= 3:
            if trn == 0:
                trn, ten = 1, ten - 1
            if van == 0 and n - trn >= 2:
                van, ten = 1, ten - 1
            if ten == 0:
                ten = 1
        return trn, van, ten

    assign = np.empty(pairs.shape[0], dtype=np.uint8)
    if per_asset:
        for a in np.unique(aid):
            idx = np.nonzero(aid == a)[0]
            na = idx.size
            trn, van, ten = _split_counts(na, train_ratio, val_ratio, test_ratio)
            assign[idx[:trn]] = 0
            assign[idx[trn:trn+van]] = 1
            assign[idx[trn+van:]] = 2
    else:
        n = pairs.shape[0]
        trn, van, ten = _split_counts(n, train_ratio, val_ratio, test_ratio)
        assign[:trn] = 0; assign[trn:trn+van] = 1; assign[trn+van:] = 2

    tr_pairs = pairs[assign == 0]
    va_pairs = pairs[assign == 1]
    te_pairs = pairs[assign == 2]

    # ---- Train-only normalization (optional) ----
    if norm_scope.lower() == "train_only":
        per_ticker_flag = bool(norm_stats.get('per_ticker', True))
        F = len(meta['feature_cols'])
        tr_norm = _compute_train_only_norm_stats(
            data_dir=data_dir,
            assets=assets,
            tr_pairs=tr_pairs,
            window=window,
            per_ticker=per_ticker_flag,
            feature_dim=F,
        )
        if tr_norm is not None:
            norm_stats = tr_norm


    # Build datasets
    ds_tr = _IndexBackedDataset(tr_pairs, assets, data_dir, window, horizon, regression,
                                keep_time_meta, norm_stats, clamp_sigma=float(meta.get('clamp_sigma', 5.0)))
    ds_va = _IndexBackedDataset(va_pairs, assets, data_dir, window, horizon, regression,
                                keep_time_meta, norm_stats, clamp_sigma=float(meta.get('clamp_sigma', 5.0)))
    ds_te = _IndexBackedDataset(te_pairs, assets, data_dir, window, horizon, regression,
                                keep_time_meta, norm_stats, clamp_sigma=float(meta.get('clamp_sigma', 5.0)))

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    gen = torch.Generator(); gen.manual_seed(seed)

    # Optional date-aware batching (uses end_times filtered by split)
    if date_batching is None:
        date_batching = (coverage_per_window > 0.0)
    if date_batching:
        min_real = max(1, int(_ceil(coverage_per_window * len(assets)))) if coverage_per_window > 0 else 1
        # recover split-specific end_times by masking
        tr_mask = (assign == 0); va_mask = (assign == 1); te_mask = (assign == 2)
        batches_tr = _build_date_batches_from_pairs(tr_pairs, end_times[tr_mask], dates_per_batch, min_real)
        batches_va = _build_date_batches_from_pairs(va_pairs, end_times[va_mask], dates_per_batch, min_real)
        batches_te = _build_date_batches_from_pairs(te_pairs, end_times[te_mask], dates_per_batch, min_real)
        train_dl = DataLoader(ds_tr, batch_sampler=_ListBatchSampler(batches_tr), pin_memory=pin_memory,
                              num_workers=num_workers, persistent_workers=False, generator=gen,
                              collate_fn=collate_fn)
        val_dl   = DataLoader(ds_va, batch_sampler=_ListBatchSampler(batches_va), pin_memory=pin_memory,
                              num_workers=num_workers, persistent_workers=False, generator=gen,
                              collate_fn=collate_fn)
        test_dl  = DataLoader(ds_te, batch_sampler=_ListBatchSampler(batches_te), pin_memory=pin_memory,
                              num_workers=num_workers, persistent_workers=False, generator=gen,
                              collate_fn=collate_fn)
    else:
        def _mk(ds, split):
            return DataLoader(
                ds, batch_size=batch_size, shuffle=(split == 'train' and shuffle_train),
                pin_memory=pin_memory, num_workers=num_workers, persistent_workers=False,
                generator=gen, collate_fn=collate_fn,
            )
        train_dl = _mk(ds_tr, 'train')
        val_dl   = _mk(ds_va, 'val')
        test_dl  = _mk(ds_te, 'test')

    return train_dl, val_dl, test_dl, (len(ds_tr), len(ds_va), len(ds_te))
