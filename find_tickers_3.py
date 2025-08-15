
# find_tickers.py  — US-only (NASDAQ + NYSE)
# Keeps the original selection/ranking logic while removing non‑US markets.
# Writes three files of tickers:
#   1) NASDAQ_top.txt  — top N NASDAQ tickers by median dollar volume (w/ viability filters)
#   2) NYSE_top.txt    — top N NYSE   tickers by median dollar volume (w/ viability filters)
#   3) US_stock.txt    — remaining available US tickers (NASDAQ+NYSE) after subtracting both tops
#
# Notes:
# - Still uses Yahoo Finance (yfinance) to check availability and compute stats.
# - 'top' selections also write full stats CSVs like: NASDAQ_Dynamic{topN}_ratio.csv
# - Keep parameters (K/H, ratios, windows) consistent with your downstream pipeline.

import os, time, re
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple

# ------------------------ Common name-based filter ------------------------
BAD_NAME = re.compile(
    r'(?:WARRANT|RIGHTS?|UNITS?|PREF(?:ERRED)?|DEPOSITARY|CONVERTIBLE|NOTE|BOND|FUND|ETF|ETN|TRUST|SPAC|ACQUISITION|TRACKING)',
    re.I
)

def _read_pipe(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, sep='|')
    if 'File Creation Time' in df.columns:  # drop footer row if present
        df = df.iloc[:-1]
    return df

def _to_yahoo(sym):
    if sym is None:
        return None
    s = str(sym).strip()
    if not s or s.lower() == "nan":
        return None
    return s.replace('.', '-')   # BRK.A -> BRK-A

def candidates_us_by_exchange() -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """
    Returns:
      by_mkt: {'NASDAQ': [...], 'NYSE': [...], ...}
      clean : DataFrame with columns ['symbol','secname','market','yahoo']
    """
    nasdaq = _read_pipe("https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt")
    other  = _read_pipe("https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt")

    nasdaq.columns = [c.strip().lower() for c in nasdaq.columns]
    other.columns  = [c.strip().lower() for c in other.columns]

    # file-level filters
    if 'etf' in nasdaq: nasdaq = nasdaq[nasdaq['etf'].astype(str).str.upper().eq('N')]
    if 'test issue' in nasdaq: nasdaq = nasdaq[nasdaq['test issue'].astype(str).str.upper().eq('N')]
    if 'financial status' in nasdaq: nasdaq = nasdaq[nasdaq['financial status'].astype(str).str.upper().eq('N')]

    if 'etf' in other: other = other[other['etf'].astype(str).str.upper().eq('N')]
    if 'test issue' in other: other = other[other['test issue'].astype(str).str.upper().eq('N')]

    nas = nasdaq[['symbol','security name']].rename(columns={'security name':'secname'})
    oth = other[['act symbol','security name','exchange']].rename(
        columns={'act symbol':'symbol','security name':'secname','exchange':'ex'}
    )

    nas['market'] = 'NASDAQ'
    exch_map = {'N':'NYSE','A':'NYSE American','P':'NYSE Arca','Z':'Cboe BZX','V':'IEX'}
    oth['market'] = oth['ex'].map(exch_map).fillna('Other')

    all_syms = pd.concat([nas[['symbol','secname','market']], oth[['symbol','secname','market']]], ignore_index=True)

    # common-stock name filter
    all_syms = all_syms.dropna(subset=['symbol']).copy()
    all_syms['symbol']  = all_syms['symbol'].astype(str).str.strip()
    all_syms['secname'] = all_syms['secname'].astype(str)
    clean = all_syms[~all_syms['secname'].fillna('').str.contains(BAD_NAME, regex=True, na=False)].copy()

    # Yahoo mapping + cleanup
    clean['yahoo'] = clean['symbol'].map(_to_yahoo)
    clean = clean.dropna(subset=['yahoo'])
    clean = clean[~clean['yahoo'].str.upper().str.endswith(('-W', '-WS', '-WT', '-U', '-R'))]
    clean = clean[~clean['yahoo'].str.contains(r'[^\w\-\^]')]
    clean = clean.drop_duplicates(subset=['yahoo', 'market'])

    by_mkt = {m: clean.loc[clean['market'].eq(m), 'yahoo'].tolist()
              for m in sorted(clean['market'].unique())}
    return by_mkt, clean

# ------------------------ Train/val/test window math ------------------------
def _coverage_nonempty(s: pd.Series, a: str, b: str) -> float:
    seg = s.loc[a:b]
    if len(seg) == 0: return 0.0
    # yfinance daily data only has trading days, so coverage ~1.0 unless missing chunks
    return float(1.0 - seg.isna().mean())

def _window_counts_by_ratio(L: int, K: int, H: int, tr: float, vr: float, te: float):
    if L is None or L <= 0:
        return 0, 0, 0
    # end-of-context bounds overall
    ec_min = K - 1
    ec_max = L - H - 1
    if ec_max < ec_min:
        return 0, 0, 0

    s = float(tr + vr + te)
    tr_frac = tr / s; vr_frac = vr / s
    t1 = int(np.floor(L * tr_frac)) - 1
    t2 = int(np.floor(L * (tr_frac + vr_frac))) - 1

    def count(lo, hi):
        lo = max(lo, ec_min); hi = min(hi, ec_max)
        return max(0, hi - lo + 1)

    n_tr = count(ec_min, t1)
    n_va = count(t1 + 1, t2)
    n_te = count(t2 + 1, L - 1)
    return n_tr, n_va, n_te

# ------------------------ Core selector (unchanged logic) ------------------------
def build_universe_by_ratios(
    tickers,
    start="2015-01-01", end="2025-06-30",
    train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
    rank_start="2016-01-01", rank_end="2018-12-31",
    K=512, H=20, buffer_days=50,
    min_price=3.0,
    presence_mode="stable",               # "stable" or "dynamic"
    min_windows_train=128,
    min_windows_val=8,
    min_windows_test=32,
    topN=500,
    min_cov_train=0.90,
    batch=200,
    out_csv="universe_by_ratios.csv"
):
    # 0) name-based exclusion (common stocks only)
    BAD_PAT = re.compile(r'(ETF|ETN|PREF|PFD|PREFERRED|WARRANT|RIGHT|UNIT|NOTE|BOND|FUND|TRUST|SPAC|ACQUISITION)', re.I)
    cands = []
    for t in tickers:
        if isinstance(t, (list, tuple)): t = t[0]
        if t and not BAD_PAT.search(str(t)):
            cands.append(str(t).strip())

    stats = []
    dl_start = min(start, rank_start)
    dl_end   = end

    for i in range(0, len(cands), batch):
        chunk = cands[i:i+batch]
        try:
            df = yf.download(chunk, start=dl_start, end=dl_end,
                             auto_adjust=True, group_by="column",
                             progress=False, threads=True)
        except Exception:
            continue

        if not isinstance(df.columns, pd.MultiIndex):
            df = pd.concat({chunk[0]: df}, axis=1).swaplevel(0,1,axis=1)

        if 'Close' not in df.columns.get_level_values(0) or 'Volume' not in df.columns.get_level_values(0):
            continue

        closes = df['Close']; vols = df['Volume']

        for t in chunk:
            if t not in closes or t not in vols:
                continue

            px = closes[t].dropna()
            vo = vols[t].replace(0, np.nan).dropna()
            idx = px.index.intersection(vo.index)
            if len(idx) == 0:
                continue

            px = px.reindex(idx)
            vo = vo.reindex(idx)
            dv = (px * vo)

            px_span = px.loc[start:end]
            if len(px_span) == 0:
                continue

            dv_rank = dv.loc[rank_start:rank_end].dropna()
            if len(dv_rank) == 0:
                continue
            med_dv = float(dv_rank.median())
            med_px = float(px.loc[rank_start:rank_end].median()) if len(px.loc[rank_start:rank_end]) else np.nan

            first_date = str(px.index.min().date())
            last_date  = str(px.index.max().date())
            L = int(len(px))

            # Quick train coverage sanity
            cov_tr = _coverage_nonempty(px, start, "2019-12-31")

            need_days = K + H + buffer_days
            if L < need_days:
                continue

            n_tr, n_va, n_te = _window_counts_by_ratio(L, K, H, train_ratio, val_ratio, test_ratio)

            if presence_mode == "stable":
                end_tol_days = 5
                first_ok = (pd.to_datetime(first_date) <= pd.to_datetime(start))
                last_ok = (pd.to_datetime(last_date) >= pd.to_datetime(end) - pd.Timedelta(days=end_tol_days))
                cond_presence = first_ok and last_ok
            else:
                cond_presence = (n_te >= 1)

            if (
                (med_px is not np.nan) and (med_px >= min_price) and
                (cov_tr >= min_cov_train) and
                cond_presence and
                (n_tr >= min_windows_train) and
                (n_va >= min_windows_val) and
                (n_te >= min_windows_test)
            ):
                stats.append({
                    "ticker": t,
                    "median_dollar_vol": med_dv,
                    "median_price": med_px,
                    "first_date": first_date,
                    "last_date": last_date,
                    "L_days": L,
                    "win_train": int(n_tr),
                    "win_val": int(n_va),
                    "win_test": int(n_te),
                })

    if not stats:
        raise RuntimeError("No candidates passed the ratio-based viability filters — consider relaxing min_windows_* or K/H.")

    df = pd.DataFrame(stats).sort_values("median_dollar_vol", ascending=False)
    top = df.head(topN).copy()

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    top.to_csv(out_csv, index=False)
    return top["ticker"].tolist(), df

# ------------------------ Availability check (unchanged) ------------------------
def available_in_period(tickers, start="2015-01-01", end="2025-06-30",
                        batch=100, max_retries=2, backoff=15.0):
    ok, missing = [], []

    def _has_close(df):
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return False
        if isinstance(df.columns, pd.MultiIndex):
            try:
                return 'Close' in df and df['Close'].notna().any().any()
            except Exception:
                return False
        return ('Close' in df.columns) and df['Close'].notna().any()

    i = 0
    while i < len(tickers):
        chunk = tickers[i:i+batch]
        df = None
        try:
            df = yf.download(chunk, start=start, end=end,
                             auto_adjust=True, group_by="column",
                             progress=False, threads=False)
        except Exception:
            time.sleep(backoff)

        if df is not None and not isinstance(df.columns, pd.MultiIndex):
            df = pd.concat({chunk[0]: df}, axis=1).swaplevel(0,1,axis=1)

        for t in chunk:
            got = False
            for r in range(max_retries + 1):
                try:
                    if df is not None and 'Close' in df and t in df['Close']:
                        got = df['Close'][t].notna().any()
                    else:
                        d1 = yf.download(t, start=start, end=end, auto_adjust=True,
                                         group_by="column", progress=False, threads=False)
                        got = _has_close(d1)
                except Exception:
                    time.sleep(backoff * (r + 1))
                    continue
                break
            (ok if got else missing).append(t)
        i += batch

    return sorted(set(ok)), sorted(set(missing))

# ------------------------ US-only builder ------------------------
def _write_list(path: str, tickers: List[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # one ticker per line, no header
    with open(path, "w", encoding="utf-8") as f:
        for t in tickers:
            f.write(f"{t}\n")

def build_us_only_universes(
    topk: Dict[str, int],
    # modeling / split params
    start="2015-01-01", end="2025-06-30",
    train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
    rank_start="2016-01-01", rank_end="2018-12-31",
    K=120, H=10, buffer_days=50,
    presence_mode="dynamic",
    min_windows_train=128, min_windows_val=8, min_windows_test=32,
    out_dir="./ldt"
) -> Dict[str, Tuple[List[str], "pd.DataFrame"]]:
    """
    Builds NASDAQ + NYSE only, then writes:
      - NASDAQ_top.txt
      - NYSE_top.txt
      - US_stock.txt  (all available NASDAQ+NYSE minus both tops)
    Returns a dict: {'NASDAQ': (sel_list, stats_df), 'NYSE': (sel_list, stats_df), 'US': (remaining_list, None)}
    """
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    by_mkt, us_df = candidates_us_by_exchange()
    nas_raw = by_mkt.get('NASDAQ', [])
    nyse_raw = by_mkt.get('NYSE', [])

    print(f"[NASDAQ] raw candidates: {len(nas_raw)}")
    print(f"[NYSE]   raw candidates: {len(nyse_raw)}")

    nas_ok, _ = available_in_period(nas_raw, start=start, end=end, batch=100)
    nys_ok, _ = available_in_period(nyse_raw, start=start, end=end, batch=100)

    print(f"[NASDAQ] available OK: {len(nas_ok)}")
    print(f"[NYSE]   available OK: {len(nys_ok)}")

    nas_topN = int(topk.get('NASDAQ', 500))
    nys_topN = int(topk.get('NYSE', 500))

    nas_sel, nas_stats = build_universe_by_ratios(
        tickers=nas_ok,
        start=start, end=end,
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        rank_start=rank_start, rank_end=rank_end,
        K=K, H=H, buffer_days=buffer_days,
        presence_mode=presence_mode,
        min_windows_train=min_windows_train, min_windows_val=min_windows_val, min_windows_test=min_windows_test,
        topN=nas_topN,
        out_csv=os.path.join(out_dir, f"NASDAQ_Dynamic{nas_topN}_ratio.csv")
    )
    _write_list(os.path.join(out_dir, "NASDAQ_top.txt"), nas_sel)
    results['NASDAQ'] = (nas_sel, nas_stats)

    nys_sel, nys_stats = build_universe_by_ratios(
        tickers=nys_ok,
        start=start, end=end,
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        rank_start=rank_start, rank_end=rank_end,
        K=K, H=H, buffer_days=buffer_days,
        presence_mode=presence_mode,
        min_windows_train=min_windows_train, min_windows_val=min_windows_val, min_windows_test=min_windows_test,
        topN=nys_topN,
        out_csv=os.path.join(out_dir, f"NYSE_Dynamic{nys_topN}_ratio.csv")
    )
    _write_list(os.path.join(out_dir, "NYSE_top.txt"), nys_sel)
    results['NYSE'] = (nys_sel, nys_stats)

    # Combined US available minus both tops
    all_ok = sorted(set(nas_ok).union(nys_ok))
    top_set = set(nas_sel).union(nys_sel)
    us_remaining = [t for t in all_ok if t not in top_set]
    _write_list(os.path.join(out_dir, "US_stock.txt"), us_remaining)
    results['US'] = (us_remaining, None)

    print(f"[US] remaining (available minus tops): {len(us_remaining)}")
    print(f"Wrote files in: {os.path.abspath(out_dir)}")
    print(" - NASDAQ_top.txt")
    print(" - NYSE_top.txt")
    print(" - US_stock.txt")
    return results

# ------------------------ CLI entry ------------------------
# if __name__ == "__main__":
#     # Example defaults; adjust as needed.
#     params = dict(
#         topk={'NASDAQ': 500, 'NYSE': 500},
#         start="2015-01-01", end="2025-06-30",
#         train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
#         rank_start="2016-01-01", rank_end="2018-12-31",
#         K=120, H=10, buffer_days=50,
#         presence_mode="dynamic",
#         min_windows_train=128, min_windows_val=8, min_windows_test=32,
#         out_dir="./ldt"
#     )
#     build_us_only_universes(**params)
