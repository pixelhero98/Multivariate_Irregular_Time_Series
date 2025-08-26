# fin_data_prep_f16_ultramem.py — cleaned
# Ultra memory-lean variant. Same public APIs:
#   - prepare_stock_windows_and_cache_v2(...)
#   - load_dataloaders_with_meta_v2(...)
#   - load_dataloaders_with_ratio_split(...)
#
# Key goals of this pass:
#   • Remove unused imports and dead/commented code paths
#   • Keep identical I/O and behavior
#   • Fix date-batching scope in ratio-split loader when memmap is absent
#   • Light refactors for clarity (no functional changes)

from dataclasses import dataclass, field
import os, json, gc, shutil

# ---------- Date-aware batching helpers (built-in) ----------
import numpy as _np
from math import ceil as _ceil
from torch.utils.data import Dataset, DataLoader, Sampler as _Sampler
from typing import List, Optional, Tuple, Dict, Sequence
import numpy as np
import torch

# Lazy-friendly imports: don't hard-fail at module import time.
try:
    import yfinance as yf  # may be None if not installed
    import pandas as pd    # may be None if not installed
except Exception:
    yf = None
    pd = None


class _ListBatchSampler(_Sampler):
    """Batch sampler that yields precomputed lists of dataset indices."""
    def __init__(self, batches):
        self.batches = [list(b) for b in batches if len(b)]
    def __iter__(self):
        for b in self.batches:
            yield b
    def __len__(self):
        return len(self.batches)


def _normalize_to_day(int64_ns):
    a = int64_ns.astype('datetime64[D]').astype(_np.int64)
    return a


def _build_date_batches_for_dataset(ds, dates_per_batch, min_real_entities, keep_time_meta: str):
    CT = getattr(ds, 'ctx_times', None)
    if CT is None or getattr(CT, 'size', 0) == 0:
        raise RuntimeError("ctx_times missing; rebuild cache with keep_time_meta != 'none'.")
    if keep_time_meta == "full" and getattr(CT, 'ndim', 1) == 2:
        end = CT[:, -1]
    else:
        end = CT
    end_ns = end.astype('datetime64[ns]').astype(_np.int64)
    days = _normalize_to_day(end_ns)

    order = _np.argsort(days, kind='mergesort')
    days_sorted = days[order]
    _, starts = _np.unique(days_sorted, return_index=True)
    groups = _np.split(order, starts[1:])
    dense_groups = [g for g in groups if g.size >= int(min_real_entities)]
    batches = []
    for k in range(0, len(dense_groups), dates_per_batch):
        chunk = dense_groups[k:k+dates_per_batch]
        if chunk:
            batches.append(_np.concatenate(chunk, axis=0))
    return batches


def _build_date_batches_for_order(order_arr, splits, dates_per_batch, min_real_entities, keep_time_meta: str):
    order = order_arr
    M = order.shape[0]
    days = _np.empty((M,), dtype=_np.int64)

    sids = order[:, 0].astype(_np.int16)
    for sid in _np.unique(sids):
        mask = (sids == sid)
        if not mask.any():
            continue
        idxs = order[mask, 1]
        CT = splits[sid][3]
        if CT is None or getattr(CT, 'size', 0) == 0:
            raise RuntimeError("ctx_times missing; rebuild cache with keep_time_meta != 'none'.")
        if keep_time_meta == "full" and getattr(CT, 'ndim', 1) == 2:
            end = CT[idxs, -1]
        else:
            end = CT[idxs]
        end_ns = end.astype('datetime64[ns]').astype(_np.int64)
        days[mask] = _normalize_to_day(end_ns)

    order_idx = _np.argsort(days, kind='mergesort')
    days_sorted = days[order_idx]
    _, starts = _np.unique(days_sorted, return_index=True)
    groups = _np.split(order_idx, starts[1:])
    dense_groups = [g for g in groups if g.size >= int(min_real_entities)]
    batches = []
    for k in range(0, len(dense_groups), dates_per_batch):
        chunk = dense_groups[k:k+dates_per_batch]
        if chunk:
            batches.append(_np.concatenate(chunk, axis=0))
    return batches


def collate_keep_meta(batch):
    xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]
    ms = [b[2] for b in batch]
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    keys = set().union(*(m.keys() for m in ms)) if ms else set()
    meta = {k: [m.get(k, None) for m in ms] for k in keys}
    return x, y, meta


# --------------------- Collate factory (levels + first differences) ---------------------

def make_collate_level_and_firstdiff(
    n_entities: int = 1,
    group_by_end_time: bool = False,
    pad_incomplete: str = "zeros",
    target_mode: str = "all",
    target_index: int = 0,
    return_entity_mask: bool = True,
    require_all_entities: bool = False,
    coverage_per_window: float = 0.0,
):
    import numpy as _np
    import torch as _torch

    def _diff_along_time(v: _torch.Tensor) -> _torch.Tensor:
        t = _torch.zeros_like(v)
        t[..., 1:, :] = v[..., 1:, :] - v[..., :-1, :]
        return t

    def _end_time_from_meta(m):
        ct = m.get('ctx_times', None)
        if ct is None:
            return None
        val = ct[-1] if isinstance(ct, _np.ndarray) else ct
        import pandas as _pd
        d = _pd.Timestamp(val).normalize()  # group by DATE regardless of time-of-day
        return int(_np.datetime64(d, 'ns').astype(_np.int64))

    def _stack_meta(ms):
        keys = set().union(*(m.keys() for m in ms)) if ms else set()
        return {k: [m.get(k, None) for m in ms] for k in keys}

    def _normalize_y(y):
        return y if isinstance(y, _torch.Tensor) else _torch.as_tensor(y)

    def _pack_group(samples):
        xs = [_torch.as_tensor(s[0]) for s in samples]     # list of [K,F]
        ys = [_normalize_y(s[1]) for s in samples]
        ms = [s[2] for s in samples]
        xN = _torch.stack(xs, dim=0).unsqueeze(0)          # [1,N,K,F]
        def _sq(y):
            if y.ndim == 2 and y.shape[-1] == 1:
                return y.squeeze(-1)
            return y
        yN = _torch.stack([_sq(y) for y in ys], dim=0)     # [N,...]
        mN = _stack_meta(ms)
        return xN, yN, mN

    def collate(batch):
        if not group_by_end_time or n_entities == 1:
            xs = [_torch.as_tensor(b[0]) for b in batch]
            ys = [_normalize_y(b[1]) for b in batch]
            ms = [b[2] for b in batch]
            V = _torch.stack(xs, dim=0).unsqueeze(1)          # [B,1,K,F]
            T = _diff_along_time(V)
            def _sq(y):
                return y if not (y.ndim == 2 and y.shape[-1] == 1) else y.squeeze(-1)
            y = _torch.stack([_sq(y) for y in ys], dim=0)
            meta = _stack_meta(ms)
            if return_entity_mask:
                meta['entity_mask'] = _torch.ones(V.shape[0], 1, dtype=_torch.bool)
            return [V, T], y, meta

        buckets = {}
        for x, y, m in batch:
            k = _end_time_from_meta(m)
            if k is None:
                return make_collate_level_and_firstdiff(1, False)(batch)
            buckets.setdefault(k, []).append((x, y, m))

        V_panels, T_panels, Y_panels, meta_panels, masks = [], [], [], [], []
        for k in sorted(buckets.keys()):
            # Robust dedupe by asset name first, then asset_id, then row pointer
            by_key = {}
            for x_, y_, m_ in buckets[k]:
                a_name = m_.get('asset', None)
                a_id   = m_.get('asset_id', None)
                if a_name is not None:
                    key = ('asset', a_name)
                elif a_id is not None:
                    key = ('asset_id', int(a_id))
                else:
                    key = ('rowptr', id(m_))
                if key not in by_key:
                    by_key[key] = (x_, y_, m_)
            g = list(by_key.values())
            real_n = len(g)

            # Require full panel?
            if require_all_entities and real_n < n_entities:
                if pad_incomplete == "drop":
                    continue
                raise RuntimeError(f"Incomplete panel at end_time={k}: have {real_n} < required {n_entities} entities.")
            if real_n == 0:
                continue

            # Coverage threshold (fraction of desired entities present)
            if coverage_per_window and (real_n / float(max(1, n_entities)) < coverage_per_window):
                continue

            # Pack & (maybe) pad
            V1, Y1, M1 = _pack_group(g[:n_entities])
            real_n = min(real_n, n_entities)
            if real_n < n_entities:
                Klen = V1.shape[2]; Fdim = V1.shape[3]; pad_n = n_entities - real_n
                Vpad = _torch.zeros((1, pad_n, Klen, Fdim), dtype=V1.dtype, device=V1.device)
                V1 = _torch.cat([V1, Vpad], dim=1)
                y_tail = Y1.shape[1:]
                Ypad = _torch.zeros((pad_n, *y_tail), dtype=Y1.dtype, device=Y1.device)
                Y1 = _torch.cat([Y1, Ypad], dim=0)
                for kmeta in list(M1.keys()):
                    vals = M1[kmeta]
                    if isinstance(vals, list):
                        M1[kmeta] = vals + [None]*pad_n
                mask = _torch.zeros(n_entities, dtype=_torch.bool); mask[:real_n] = True
            else:
                mask = _torch.ones(n_entities, dtype=_torch.bool)

            T1 = _diff_along_time(V1)
            V_panels.append(V1); T_panels.append(T1); Y_panels.append(Y1); meta_panels.append(M1); masks.append(mask)

        if not V_panels:
            return make_collate_level_and_firstdiff(1, False)(batch)

        Vb = _torch.cat(V_panels, dim=0)
        Tb = _torch.cat(T_panels, dim=0)
        yb_all = _torch.stack(Y_panels, dim=0)
        yb = yb_all if target_mode == "all" else yb_all[:, target_index, ...]

        def _merge(Ms, key):
            out = []
            for M in Ms:
                vals = M.get(key, None)
                out.append(vals if isinstance(vals, list) else [vals])
            return out
        meta_out = {
            'grouped_by': 'end_ctx_time',
            'asset_id': _merge(meta_panels, 'asset_id'),
            'asset': _merge(meta_panels, 'asset'),
            'ctx_times': _merge(meta_panels, 'ctx_times')
        }
        if return_entity_mask:
            meta_out['entity_mask'] = _torch.stack(masks, dim=0)
        return [Vb, Tb], yb, meta_out

    return collate


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
    price_fields: List[str] = field(default_factory=lambda: ['Close'])
    returns_mode: str = 'log'     # 'log' or 'pct'
    include_rvol: bool = True
    rvol_span: int = 20
    rvol_on: str = 'Close'
    include_dlv: bool = True
    market_proxy: Optional[str] = 'SPY'
    include_oc: bool = False
    include_gap: bool = False
    include_hl_range: bool = False
    target_field: str = 'Close'
    calendar: CalendarConfig = field(default_factory=CalendarConfig)
    include_entity_id_feature: bool = False


def _cyclical_from_int(values: np.ndarray, period: int):
    ang = 2.0 * np.pi * (values.astype(np.float32) / float(period))
    return np.sin(ang).astype(np.float32), np.cos(ang).astype(np.float32)


def build_calendar_frame(idx: 'pd.DatetimeIndex', cfg: CalendarConfig) -> 'pd.DataFrame':
    try:
        import pandas as pd  # local import
    except Exception as e:
        raise ImportError("pandas is required for build_calendar_frame(...). Install with: pip install pandas") from e

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


class LabeledWindowWithMetaDataset(Dataset):
    def __init__(self,
                 X: np.ndarray,       # [N, K, F] (can be memmap)
                 Y: np.ndarray,       # [N, H] or [N, H, 1]
                 asset_ids: Optional[np.ndarray] = None,
                 assets: Optional[List[str]] = None,
                 ctx_times: Optional[np.ndarray] = None,  # 1D or 2D
                 y_times: Optional[np.ndarray] = None,    # 1D or 2D
                 regression: bool = True):
        assert X.ndim == 3, f"X must be [N,K,F], got {X.shape}"
        self.X = X  # keep dtype as-is to avoid memmap copies
        if regression and Y.ndim == 2:
            Y = Y[..., None]
        self.Y = Y
        self.regression = regression
        self.asset_ids = asset_ids
        self.assets = assets
        self.ctx_times = ctx_times
        self.y_times = y_times

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y_np = self.Y[idx]
        y = torch.tensor(y_np, dtype=torch.float32) if self.regression else torch.tensor(int(y_np), dtype=torch.int64)
        meta = {}
        if self.asset_ids is not None:
            aid = int(self.asset_ids[idx])
            meta['asset_id'] = aid
            if self.assets is not None:
                meta['asset'] = self.assets[aid]
        if self.ctx_times is not None:
            meta['ctx_times'] = self.ctx_times[idx]
        if self.y_times is not None:
            meta['y_times'] = self.y_times[idx]
        return x, y, meta


# --------------------- Paths & helpers ---------------------

def _memmap_dir(data_dir: str):
    return os.path.join(data_dir, "cache_v2_memmap")

def _npz_path(data_dir: str):
    return os.path.join(data_dir, "cache_v2.npz")

def _meta_path(data_dir: str):
    return os.path.join(data_dir, "meta_v2.json")

def _norm_path(data_dir: str):
    return os.path.join(data_dir, "norm_stats_v2.json")

def _open_memmap(path: str, shape: Tuple[int, ...], dtype, mode='w+'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return np.lib.format.open_memmap(path, mode=mode, dtype=dtype, shape=shape)


# --------------------- Stock prep (memmap-backed) ---------------------
_EPS = 1e-6  # small positive, float32-friendly


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


def prepare_stock_windows_and_cache_v2(
    tickers: List[str],
    start: str,
    val_start: str,
    test_start: str,
    end: str,
    window: int,
    horizon: int,
    data_dir: str = './data',
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
    keep_time_meta: str = "end",  # "full" | "end" | "none"
):
    try:
        import pandas as pd
    except Exception as e:
        raise ImportError("pandas is required for prepare_stock_windows_and_cache_v2. Install with: pip install pandas") from e
    try:
        import yfinance as yf
    except Exception as e:
        raise ImportError("yfinance is required for prepare_stock_windows_and_cache_v2. Install with: pip install yfinance") from e

    rng = np.random.RandomState(seed)
    os.makedirs(data_dir, exist_ok=True)
    mem_dir = _memmap_dir(data_dir)
    meta_json = _meta_path(data_dir)
    norm_json = _norm_path(data_dir)

    # ---- 1) Raw columns we need if we download ----
    need_price_cols = set(feature_cfg.price_fields) | {feature_cfg.target_field}
    if feature_cfg.include_oc or feature_cfg.include_gap:
        need_price_cols |= {'Open', 'Close'}
    if feature_cfg.include_hl_range:
        need_price_cols |= {'High', 'Low'}
    wanted_cols = sorted(list(need_price_cols | ({'Volume'} if feature_cfg.include_dlv else set())))

    tickers_dl = tickers[:]
    if feature_cfg.market_proxy and feature_cfg.market_proxy not in tickers_dl:
        tickers_dl.append(feature_cfg.market_proxy)

    # ---- Try cached per-ticker features ----
    features_dir = os.path.join(data_dir, "features")
    loaded_from_cache = False
    per_ticker: Dict[str, 'pd.DataFrame'] = {}

    if os.path.isdir(features_dir):
        try:
            for t in tickers:
                p = os.path.join(features_dir, f"{t}.parquet")
                if os.path.exists(p):
                    df = pd.read_parquet(p)
                    for c in df.columns:
                        if df[c].dtype == np.float64:
                            df[c] = df[c].astype(np.float32)
                    per_ticker[t] = df
            loaded_from_cache = len(per_ticker) > 0
        except Exception:
            loaded_from_cache = False
            per_ticker = {}

    # ---- Raw path: download + build features, then save parquet for next time ----
    if not loaded_from_cache:
        raw = yf.download(
            tickers_dl, start=start, end=end, auto_adjust=True,
            group_by="column", progress=False
        )

        if isinstance(raw.columns, pd.MultiIndex):
            if not set(wanted_cols).issubset(set(raw.columns.get_level_values(0))):
                raw = raw.swaplevel(0, 1, axis=1).sort_index(axis=1)

        def get_ticker_df(t: str) -> 'pd.DataFrame':
            if isinstance(raw.columns, pd.MultiIndex):
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
                      if feature_cfg.returns_mode == 'log'
                      else (df['Close'] / df['Open'] - 1.0))
                feat['OC_RET'] = oc

            if feature_cfg.include_gap and 'Open' in df and 'Close' in df:
                gap = (_safe_log_series(df['Open']) - _safe_log_series(df['Close'].shift(1))
                       if feature_cfg.returns_mode == 'log'
                       else (df['Open'] / df['Close'].shift(1) - 1.0))
                feat['GAP_RET'] = gap

            if feature_cfg.include_hl_range and 'High' in df and 'Low' in df:
                hlr = (_safe_log_series(df['High']) - _safe_log_series(df['Low'])
                       if feature_cfg.returns_mode == 'log'
                       else (df['High'] / df['Low'] - 1.0))
                feat['HL_RANGE'] = hlr

            if feature_cfg.include_dlv and 'Volume' in df:
                feat['DLV'] = _delta_log_volume(df['Volume'])
            if feature_cfg.include_rvol:
                base_col = f'RET_{feature_cfg.rvol_on.upper()}'
                if base_col in feat:
                    feat[f'RVOL{feature_cfg.rvol_span}_{feature_cfg.rvol_on.upper()}'] = _ewma_vol(feat[base_col], span=feature_cfg.rvol_span)
            if proxy_ret is not None:
                feat['MKT'] = proxy_ret.reindex(df.index)

            out = pd.DataFrame(feat)
            cal = build_calendar_frame(out.index, feature_cfg.calendar)
            out = pd.concat([out, cal], axis=1)
            for c in out.columns:
                if out[c].dtype == np.float64:
                    out[c] = out[c].astype(np.float32)
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
            train_seg = feat_df.loc[start:val_start]
            coverage = 1.0 - train_seg.isna().any(axis=1).mean() if len(train_seg) else 0.0
            if coverage < min_train_coverage:
                continue
            feat_df = feat_df.dropna()
            if feat_df.shape[0] < min_obs:
                continue
            for c in feat_df.columns:
                if feat_df[c].dtype == np.float64:
                    feat_df[c] = feat_df[c].astype(np.float32)
            per_ticker[t] = feat_df

        if not per_ticker:
            raise RuntimeError("No tickers passed the cleaning criteria. Relax thresholds or check dates.")
        os.makedirs(features_dir, exist_ok=True)
        for t, df in per_ticker.items():
            df.to_parquet(os.path.join(features_dir, f"{t}.parquet"))
        del raw; gc.collect()
    else:
        min_obs = window + horizon + min_obs_buffer
        per_ticker = {t: df for t, df in per_ticker.items() if t != feature_cfg.market_proxy and df.shape[0] >= min_obs}
        for t, df in list(per_ticker.items()):
            for c in df.columns:
                if df[c].dtype == np.float64:
                    df[c] = df[c].astype(np.float32)
        if not per_ticker:
            raise RuntimeError("Cached features found, but none meet min_obs. Rebuild with smaller K/H or new dates.")

    # ---- 2) Assemble asset list & feature column set ----
    assets = sorted(per_ticker.keys())
    asset2id = {a: i for i, a in enumerate(assets)}
    if getattr(feature_cfg, 'include_entity_id_feature', False):
        denom = max(1, len(assets) - 1)
        for a in assets:
            aid = asset2id[a]
            val = np.float32(aid / denom) if denom > 0 else np.float32(0.0)
            per_ticker[a]['ENTITY_ID'] = np.full((len(per_ticker[a])), val, dtype=np.float32)

    cols_sets = [set(df.columns) for df in per_ticker.values()]
    common_cols = set.intersection(*cols_sets)
    target_col = f"RET_{feature_cfg.target_field.upper()}"
    if target_col not in common_cols:
        raise ValueError(f"Target column '{target_col}' not in common features. Add '{feature_cfg.target_field}' to cfg.price_fields.")
    feature_cols = sorted(list(common_cols))

    # ---- 3) Sliding windows with date-based splits (two-pass, pre-count) ----
    starts = {a: {'train': [], 'val': [], 'test': []} for a in assets}
    lengths = {'train': 0, 'val': 0, 'test': 0}

    def _iter_indices(feat_df: 'pd.DataFrame'):
        arr = feat_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        times = feat_df.index.to_numpy()
        tgt = feat_df[target_col].to_numpy(dtype=np.float32, copy=False)
        n = len(arr)
        for i in range(0, n - window - horizon + 1):
            end_ctx_idx = i + window - 1
            end_ctx_date = pd.Timestamp(times[end_ctx_idx])
            if end_ctx_date < pd.Timestamp(val_start):
                yield 'train', i
            elif end_ctx_date < pd.Timestamp(test_start):
                yield 'val', i
            else:
                if end_ctx_date > pd.Timestamp(end):
                    continue
                yield 'test', i

    for a in assets:
        for split, i in _iter_indices(per_ticker[a]):
            if max_windows_per_ticker and len(starts[a][split]) >= max_windows_per_ticker:
                continue
            starts[a][split].append(i)
            lengths[split] += 1

    Ntr, Nva, Nte = lengths['train'], lengths['val'], lengths['test']
    K = window; H = horizon; Fdim = len(feature_cols)
    if Ntr == 0:
        raise RuntimeError("No training windows after splits — check dates.")

    # Prepare memmap directory fresh
    mem_dir = _memmap_dir(data_dir)
    if os.path.isdir(mem_dir):
        shutil.rmtree(mem_dir)
    os.makedirs(mem_dir, exist_ok=True)

    def _alloc_memmap(prefix: str, N: int):
        Xp = _open_memmap(os.path.join(mem_dir, f"{prefix}_X.npy"), (N, K, Fdim), np.float32, mode='w+')
        Yp = _open_memmap(os.path.join(mem_dir, f"{prefix}_Y.npy"), (N, H), np.float32, mode='w+')
        Ip = _open_memmap(os.path.join(mem_dir, f"{prefix}_asset_id.npy"), (N,), np.int32, mode='w+')
        if keep_time_meta != "none":
            if keep_time_meta == "full":
                CTp = _open_memmap(os.path.join(mem_dir, f"{prefix}_ctx_times.npy"), (N, K), 'datetime64[ns]', mode='w+')
                YTp = _open_memmap(os.path.join(mem_dir, f"{prefix}_y_times.npy"), (N, H), 'datetime64[ns]', mode='w+')
            else:
                CTp = _open_memmap(os.path.join(mem_dir, f"{prefix}_ctx_times.npy"), (N,), 'datetime64[ns]', mode='w+')
                YTp = _open_memmap(os.path.join(mem_dir, f"{prefix}_y_times.npy"), (N,), 'datetime64[ns]', mode='w+')
        else:
            CTp = YTp = None
        return Xp, Yp, Ip, CTp, YTp

    Xtr, Ytr, IDtr, CTtr, YTtr = _alloc_memmap('train', Ntr)
    Xva, Yva, IDva, CTva, YTva = _alloc_memmap('val',   Nva)
    Xte, Yte, IDte, CTte, YTte = _alloc_memmap('test',  Nte)

    p = {'train': 0, 'val': 0, 'test': 0}
    for a in assets:
        aid = asset2id[a]
        feat_df = per_ticker[a]
        arr = feat_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        times = feat_df.index.to_numpy()
        tgt = feat_df[target_col].to_numpy(dtype=np.float32, copy=False)

        for split in ('train', 'val', 'test'):
            idxs = starts[a][split]
            if not idxs:
                continue
            idxs = np.array(idxs, dtype=np.int64)
            s = p[split]; e = s + len(idxs); p[split] = e
            if split == 'train':
                Xtr[s:e] = np.stack([arr[i:i+K] for i in idxs], axis=0)
                Ytr[s:e] = np.stack([tgt[i+K:i+K+H] for i in idxs], axis=0)
                IDtr[s:e] = aid
                if CTtr is not None:
                    if keep_time_meta == "full":
                        CTtr[s:e] = np.stack([times[i:i+K] for i in idxs], axis=0)
                        YTtr[s:e] = np.stack([times[i+K:i+K+H] for i in idxs], axis=0)
                    else:
                        CTtr[s:e] = times[idxs + K - 1]
                        YTtr[s:e] = times[idxs + K + H - 1]
            elif split == 'val':
                Xva[s:e] = np.stack([arr[i:i+K] for i in idxs], axis=0)
                Yva[s:e] = np.stack([tgt[i+K:i+K+H] for i in idxs], axis=0)
                IDva[s:e] = aid
                if CTva is not None:
                    if keep_time_meta == "full":
                        CTva[s:e] = np.stack([times[i:i+K] for i in idxs], axis=0)
                        YTva[s:e] = np.stack([times[i+K:i+K+H] for i in idxs], axis=0)
                    else:
                        CTva[s:e] = times[idxs + K - 1]
                        YTva[s:e] = times[idxs + K + H - 1]
            else:
                Xte[s:e] = np.stack([arr[i:i+K] for i in idxs], axis=0)
                Yte[s:e] = np.stack([tgt[i+K:i+K+H] for i in idxs], axis=0)
                IDte[s:e] = aid
                if CTte is not None:
                    if keep_time_meta == "full":
                        CTte[s:e] = np.stack([times[i:i+K] for i in idxs], axis=0)
                        YTte[s:e] = np.stack([times[i+K:i+K+H] for i in idxs], axis=0)
                    else:
                        CTte[s:e] = times[idxs + K - 1]
                        YTte[s:e] = times[idxs + K + H - 1]

    # ---- 4) Train-only normalization (per-ticker optional) + clamping ----
    if normalize_per_ticker:
        Fdim = Xtr.shape[-1]
        mean_x = np.zeros((len(assets), 1, 1, Fdim), np.float32)
        std_x  = np.ones ((len(assets), 1, 1, Fdim), np.float32)
        mean_y = np.zeros((len(assets), 1, H), np.float32)
        std_y  = np.ones ((len(assets), 1, H), np.float32)

        for aid in range(len(assets)):
            m = (IDtr == aid)
            if not np.any(m):
                continue
            _X, _Y = Xtr[m], Ytr[m]
            mx = _X.mean(axis=(0,1), keepdims=True); sx = _X.std(axis=(0,1), keepdims=True); sx[sx==0]=1.0
            my = _Y.mean(axis=0, keepdims=True);     sy = _Y.std(axis=0, keepdims=True);     sy[sy==0]=1.0
            mean_x[aid], std_x[aid], mean_y[aid], std_y[aid] = mx, sx, my, sy

        def norm_apply(X, Y, ids):
            for aid in np.unique(ids):
                m = (ids == aid)
                mx, sx = mean_x[aid], std_x[aid]
                lo, hi = mx - clamp_sigma*sx, mx + clamp_sigma*sx
                X[m] = np.clip(X[m], lo, hi)
                X[m] = (X[m] - mx) / sx
                my, sy = mean_y[aid], std_y[aid]
                lo_y, hi_y = my - clamp_sigma*sy, my + clamp_sigma*sy
                Y[m] = np.clip(Y[m], lo_y, hi_y)
                Y[m] = (Y[m] - my) / sy
            return X, Y

        Xtr, Ytr = norm_apply(Xtr, Ytr, IDtr)
        if len(Xva):
            Xva, Yva = norm_apply(Xva, Yva, IDva)
        if len(Xte):
            Xte, Yte = norm_apply(Xte, Yte, IDte)

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
        Xtr[...] = np.clip(Xtr, lo, hi); Xtr[...] = (Xtr - mean_x) / std_x
        if len(Xva):
            Xva[...] = np.clip(Xva, lo, hi); Xva[...] = (Xva - mean_x) / std_x
        if len(Xte):
            Xte[...] = np.clip(Xte, lo, hi); Xte[...] = (Xte - mean_x) / std_x
        lo_y, hi_y = mean_y - clamp_sigma*std_y, mean_y + clamp_sigma*std_y
        Ytr[...] = np.clip(Ytr, lo_y, hi_y); Ytr[...] = (Ytr - mean_y) / std_y
        if len(Yva):
            Yva[...] = np.clip(Yva, lo_y, hi_y); Yva[...] = (Yva - mean_y) / std_y
        if len(Yte):
            Yte[...] = np.clip(Yte, lo_y, hi_y); Yte[...] = (Yte - mean_y) / std_y
        norm_stats = {
            'per_ticker': False,
            'mean_x': mean_x.tolist(), 'std_x': std_x.tolist(),
            'mean_y': mean_y.tolist(), 'std_y': std_y.tolist(),
            'assets': assets,
        }

    # --- sanity checks ---
    def _assert_no_nans(name, arr):
        if arr.size == 0:
            return
        if np.issubdtype(arr.dtype, np.floating) and np.isnan(np.asarray(arr, dtype=np.float32)).any():
            raise ValueError(f"{name} contains NaNs; check feature construction and dropna().")
    _assert_no_nans("train_X", Xtr); _assert_no_nans("train_Y", Ytr)
    _assert_no_nans("val_X",   Xva); _assert_no_nans("val_Y",   Yva)
    _assert_no_nans("test_X",  Xte); _assert_no_nans("test_Y",  Yte)

    # ---- 5) Save meta/manifests (no big in-memory NPZ; memmap is the cache) ----
    manifest = {
        'format': 'memmap_v1',
        'shapes': {
            'train': [int(Ntr), int(K), int(Fdim)],
            'val':   [int(Nva), int(K), int(Fdim)],
            'test':  [int(Nte), int(K), int(Fdim)],
            'horizon': int(H),
        },
        'dtypes': {'X': 'float32', 'Y': 'float32', 'asset_id': 'int32', 'time': 'datetime64[ns]'},
        'keep_time_meta': keep_time_meta,
    }
    with open(os.path.join(mem_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

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
        'keep_time_meta': keep_time_meta,
    }
    with open(_meta_path(data_dir), 'w') as f:
        json.dump(meta, f, indent=2)
    with open(_norm_path(data_dir), 'w') as f:
        json.dump(norm_stats, f)

    # Legacy marker for older code expecting an NPZ file.
    np.savez_compressed(os.path.join(data_dir, 'cache_v2_marker.npz'),
                        note="Data stored in memmap directory 'cache_v2_memmap/'. Use loaders in this module.")
    return ((Xtr, Ytr, IDtr), (Xva, Yva, IDva), (Xte, Yte, IDte))


# --------------------- Loaders (prefer memmap; fall back to legacy NPZ) ---------------------

def _memmap_exists(data_dir: str) -> bool:
    return os.path.isdir(_memmap_dir(data_dir)) and os.path.exists(os.path.join(_memmap_dir(data_dir), 'manifest.json'))


def _open_split_memmaps(mem_dir: str, prefix: str):
    X = np.lib.format.open_memmap(os.path.join(mem_dir, f"{prefix}_X.npy"), mode='r')
    Y = np.lib.format.open_memmap(os.path.join(mem_dir, f"{prefix}_Y.npy"), mode='r')
    I = np.lib.format.open_memmap(os.path.join(mem_dir, f"{prefix}_asset_id.npy"), mode='r')
    CT = None; YT = None
    ctx_p = os.path.join(mem_dir, f"{prefix}_ctx_times.npy")
    yt_p  = os.path.join(mem_dir, f"{prefix}_y_times.npy")
    if os.path.exists(ctx_p):
        CT = np.lib.format.open_memmap(ctx_p, mode='r')
    if os.path.exists(yt_p):
        YT = np.lib.format.open_memmap(yt_p, mode='r')
    return X, Y, I, CT, YT


def _load_legacy_npz(data_dir: str):
    path_npz = _npz_path(data_dir)
    if not os.path.exists(path_npz):
        raise FileNotFoundError(f"Missing {path_npz}. Build cache with prepare_stock_windows_and_cache_v2(...)")
    z = np.load(path_npz, allow_pickle=True)
    return z


def load_dataloaders_with_meta_v2(
    batch_size: int,
    data_dir: str = './data',
    regression: bool = True,
    num_workers: int = 0,
    shuffle_train: bool = True,
    pin_memory: Optional[bool] = None,
    return_meta_arrays: bool = False,
    seed: int = 1337,
    n_entities: int = 8,
    pad_incomplete: str = 'zeros',
    collate_fn=None,
    coverage_per_window: float = 0.0,
    date_batching: Optional[bool] = None,
    dates_per_batch: int = 4,
):
    # Default to grouped-all collate if none provided
    if collate_fn is None:
        collate_fn = make_collate_level_and_firstdiff(
            n_entities=n_entities,
            group_by_end_time=True,
            pad_incomplete=pad_incomplete,
            target_mode='all',
            return_entity_mask=True,
            coverage_per_window=coverage_per_window,
        )
    meta_path = _meta_path(data_dir)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}. Run prepare_stock_windows_and_cache_v2(...) first.")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    assets = meta['assets']

    if _memmap_exists(data_dir):
        mem_dir = _memmap_dir(data_dir)

        def mk_ds(prefix: str):
            X, Y, I, CT, YT = _open_split_memmaps(mem_dir, prefix)
            return LabeledWindowWithMetaDataset(X, Y, I, assets, CT, YT, regression=regression)

        ds_tr = mk_ds('train'); ds_va = mk_ds('val'); ds_te = mk_ds('test')
        pin = torch.cuda.is_available() if pin_memory is None else pin_memory
        gen = torch.Generator(); gen.manual_seed(seed)

        def mk_loader(ds, split):
            return DataLoader(
                ds, batch_size=batch_size, shuffle=(split=='train' and shuffle_train),
                pin_memory=pin, num_workers=num_workers, persistent_workers=False,
                generator=gen, collate_fn=collate_fn,
            )
        train_dl = mk_loader(ds_tr, 'train'); val_dl = mk_loader(ds_va, 'val'); test_dl = mk_loader(ds_te, 'test')
        lengths = (len(ds_tr), len(ds_va), len(ds_te))
        if return_meta_arrays:
            return train_dl, val_dl, test_dl, lengths, (None, meta)
        return train_dl, val_dl, test_dl, lengths

    # Fallback to legacy NPZ (RAM heavy, not recommended).
    z = _load_legacy_npz(data_dir)

    def mk_ds(prefix: str):
        X = z[f'{prefix}_X']; Y = z[f'{prefix}_Y']; ids = z.get(f'{prefix}_asset_id')
        ctx_t = z.get(f'{prefix}_ctx_times'); y_t = z.get(f'{prefix}_y_times')
        return LabeledWindowWithMetaDataset(X, Y, ids, assets, ctx_t, y_t, regression=regression)

    ds_tr = mk_ds('train'); ds_va = mk_ds('val'); ds_te = mk_ds('test')
    pin = torch.cuda.is_available() if pin_memory is None else pin_memory
    gen = torch.Generator(); gen.manual_seed(seed)

    def mk_loader(ds, split):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=(split=='train' and shuffle_train),
            pin_memory=pin, num_workers=num_workers, persistent_workers=False,
            generator=gen, collate_fn=collate_fn,
        )
    train_dl = mk_loader(ds_tr, 'train'); val_dl = mk_loader(ds_va, 'val'); test_dl = mk_loader(ds_te, 'test')
    lengths = (len(ds_tr), len(ds_va), len(ds_te))
    if return_meta_arrays:
        return train_dl, val_dl, test_dl, lengths, (z, meta)
    return train_dl, val_dl, test_dl, lengths


# --------------------- Chronological ratio split loader (memmap-backed preferred) ------------------

class _ConcatIndexDataset(Dataset):
    """Read-only view over multiple underlying splits by index tuples (split_id, idx)."""
    def __init__(self,
                 splits: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]],
                 split_ids: Sequence[int],
                 order: np.ndarray,
                 assets: List[str],
                 regression: bool = True):
        self.splits = splits
        self.split_ids = np.asarray(split_ids, dtype=np.int16)
        self.order = order
        self.assets = assets
        self.regression = regression

    def __len__(self):
        return self.order.shape[0]

    def __getitem__(self, i: int):
        sid, local_idx = self.order[i]
        X, Y, IDs, CT, YT = self.splits[sid]
        x = torch.tensor(X[local_idx], dtype=torch.float32)
        y_np = Y[local_idx]
        y = torch.tensor(y_np, dtype=torch.float32) if self.regression else torch.tensor(int(y_np), dtype=torch.int64)
        meta = {}
        aid = int(IDs[local_idx])
        meta['asset_id'] = aid; meta['asset'] = self.assets[aid]
        if CT is not None:
            meta['ctx_times'] = CT[local_idx]
        if YT is not None:
            meta['y_times']  = YT[local_idx]
        return x, y, meta


def load_dataloaders_with_ratio_split(
    data_dir='./data',
    train_ratio=0.55,
    val_ratio=0.05,
    test_ratio=0.4,
    batch_size=64,
    regression=True,
    per_asset=True,
    shuffle_train=True,
    num_workers=0,
    pin_memory=None,
    seed=1337,
    n_entities: int = 8,
    pad_incomplete: str = 'zeros',
    collate_fn=None,
    coverage_per_window: float = 0.0,
    date_batching: Optional[bool] = None,
    dates_per_batch: int = 4,
):
    # Default to grouped-all collate if none provided
    if collate_fn is None:
        collate_fn = make_collate_level_and_firstdiff(
            n_entities=n_entities,
            group_by_end_time=True,
            pad_incomplete=pad_incomplete,
            target_mode="all",
            return_entity_mask=True,
        )
    meta_path = _meta_path(data_dir)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}. Build cache first.")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    assets = meta.get('assets', None)

    keep_time_meta = meta.get('keep_time_meta', 'end')

    # Prefer memmap cache
    if _memmap_exists(data_dir):
        mem_dir = _memmap_dir(data_dir)
        parts = []
        end_parts = []
        for sid, name in enumerate(('train', 'val', 'test')):
            X, Y, IDs, CT, YT = _open_split_memmaps(mem_dir, name)
            parts.append((X, Y, IDs, CT, YT))
            # 1D end-of-context times for ordering
            if CT is not None and CT.size:
                end_ctx = CT[:, -1] if getattr(CT, 'ndim', 1) == 2 else CT
            else:
                end_ctx = None
            end_parts.append(end_ctx)

        if not parts:
            raise RuntimeError("No memmap data found")

        # Build flattened (split_id, local_idx) list without concatenating arrays
        global_list = []  # will hold tuples (split_id, local_idx)
        asset_ids   = []  # parallel list of asset ids
        end_times   = []  # parallel list of end-of-context times

        for sid, (X, Y, IDs, CT, YT) in enumerate(parts):
            N = X.shape[0]
            idxs = np.arange(N, dtype=np.int64)
            global_list.append(np.stack([np.full(N, sid, dtype=np.int16), idxs], axis=1))  # [N,2]
            asset_ids.append(IDs.astype(np.int32))
            end_ctx = end_parts[sid]
            if end_ctx is None:
                raise RuntimeError("End-of-context times missing; rebuild cache with keep_time_meta != 'none'.")
            end_times.append(end_ctx)

        global_idx = np.concatenate(global_list, axis=0)      # [M,2]
        all_ids    = np.concatenate(asset_ids, axis=0)        # [M]
        all_ends   = np.concatenate(end_times, axis=0)        # [M]

        order = np.lexsort((all_ends, all_ids))               # chronological within asset
        global_idx = global_idx[order]
        all_ids = all_ids[order]

        # Compute assignments (vectorized) without materializing data
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

        assign = np.empty(global_idx.shape[0], dtype=np.uint8)
        if per_asset:
            for aid in np.unique(all_ids):
                idx = np.nonzero(all_ids == aid)[0]
                na = idx.size
                trn, van, ten = _split_counts(na, train_ratio, val_ratio, test_ratio)
                assign[idx[:trn]] = 0
                assign[idx[trn:trn+van]] = 1
                assign[idx[trn+van:]] = 2
        else:
            n = global_idx.shape[0]
            trn, van, ten = _split_counts(n, train_ratio, val_ratio, test_ratio)
            assign[:trn] = 0; assign[trn:trn+van] = 1; assign[trn+van:] = 2

        # Build datasets as index views
        tr_order = global_idx[assign==0]
        va_order = global_idx[assign==1]
        te_order = global_idx[assign==2]

        ds_tr = _ConcatIndexDataset(parts, (0,1,2), tr_order, assets, regression=regression)
        ds_va = _ConcatIndexDataset(parts, (0,1,2), va_order, assets, regression=regression)
        ds_te = _ConcatIndexDataset(parts, (0,1,2), te_order, assets, regression=regression)

        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        gen = torch.Generator(); gen.manual_seed(seed)

        # Date-aware batching (memmap branch)
        if date_batching is None:
            date_batching = (coverage_per_window > 0.0)
        if date_batching:
            min_real = max(1, int(_ceil(coverage_per_window * n_entities))) if coverage_per_window > 0 else 1
            batches_tr = _build_date_batches_for_order(tr_order, parts, dates_per_batch, min_real, keep_time_meta)
            batches_va = _build_date_batches_for_order(va_order, parts, dates_per_batch, min_real, keep_time_meta)
            batches_te = _build_date_batches_for_order(te_order, parts, dates_per_batch, min_real, keep_time_meta)
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
                    ds, batch_size=batch_size, shuffle=(split=='train' and shuffle_train),
                    pin_memory=pin_memory, num_workers=num_workers, persistent_workers=False,
                    generator=gen, collate_fn=collate_fn,
                )
            train_dl = _mk(ds_tr, 'train')
            val_dl   = _mk(ds_va, 'val')
            test_dl  = _mk(ds_te, 'test')
        return train_dl, val_dl, test_dl, (len(ds_tr), len(ds_va), len(ds_te))

    # Fallback to legacy NPZ (RAM heavy). Compatibility path.
    z = _load_legacy_npz(data_dir)

    parts = []
    end_parts = []
    for p in ('train', 'val', 'test'):
        X = z.get(f'{p}_X'); Y = z.get(f'{p}_Y'); ids = z.get(f'{p}_asset_id')
        ctx_t = z.get(f'{p}_ctx_times'); y_t = z.get(f'{p}_y_times')
        if X is None or X.size == 0:
            continue
        if Y.ndim == 3 and Y.shape[-1] == 1:
            Y = Y[..., 0]
        parts.append((X, Y, ids, ctx_t, y_t))
        end_ctx = ctx_t[:, -1] if (ctx_t is not None and getattr(ctx_t, 'ndim', 1) == 2) else ctx_t
        end_parts.append(end_ctx)

    if not parts:
        raise RuntimeError("No data found in cache_v2.npz")

    X_all   = np.concatenate([p[0] for p in parts], axis=0)
    Y_all   = np.concatenate([p[1] for p in parts], axis=0)
    IDS_all = np.concatenate([p[2] for p in parts], axis=0).astype(np.int32)
    CT_all  = np.concatenate([p[3] for p in parts], axis=0) if (parts[0][3] is not None and parts[0][3].size) else None
    YT_all  = np.concatenate([p[4] for p in parts], axis=0) if (parts[0][4] is not None and parts[0][4].size) else None
    end_ctx_all = np.concatenate(end_parts, axis=0) if (end_parts and end_parts[0] is not None) else None
    if end_ctx_all is None:
        raise RuntimeError("End-of-context times missing; rebuild cache with keep_time_meta != 'none'.")
    order = np.lexsort((end_ctx_all, IDS_all))

    X_all, Y_all, IDS_all = X_all[order], Y_all[order], IDS_all[order]
    if CT_all is not None:
        CT_all = CT_all[order]
    if YT_all is not None:
        YT_all = YT_all[order]

    n = X_all.shape[0]
    assign = np.empty(n, dtype=np.uint8)

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

    if per_asset:
        for aid in np.unique(IDS_all):
            idx = np.nonzero(IDS_all == aid)[0]
            na = idx.size
            if na == 0:
                continue
            trn, van, ten = _split_counts(na, train_ratio, val_ratio, test_ratio)
            assign[idx[:trn]] = 0
            assign[idx[trn:trn+van]] = 1
            assign[idx[trn+van:]] = 2
    else:
        trn, van, ten = _split_counts(n, train_ratio, val_ratio, test_ratio)
        assign[:trn] = 0; assign[trn:trn+van] = 1; assign[trn+van:] = 2

    m_tr = (assign == 0); m_va = (assign == 1); m_te = (assign == 2)

    ds_tr = LabeledWindowWithMetaDataset(X_all[m_tr], Y_all[m_tr], IDS_all[m_tr], assets,
                                         CT_all[m_tr] if CT_all is not None else None,
                                         YT_all[m_tr] if YT_all is not None else None,
                                         regression=regression)
    ds_va = LabeledWindowWithMetaDataset(X_all[m_va], Y_all[m_va], IDS_all[m_va], assets,
                                         CT_all[m_va] if CT_all is not None else None,
                                         YT_all[m_va] if YT_all is not None else None,
                                         regression=regression)
    ds_te = LabeledWindowWithMetaDataset(X_all[m_te], Y_all[m_te], IDS_all[m_te], assets,
                                         CT_all[m_te] if CT_all is not None else None,
                                         YT_all[m_te] if YT_all is not None else None,
                                         regression=regression)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    gen = torch.Generator(); gen.manual_seed(seed)

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
