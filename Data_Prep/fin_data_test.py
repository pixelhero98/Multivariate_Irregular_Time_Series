import json, importlib, torch
import pandas as pd
import numpy as np


mod = importlib.import_module('fin_data_prep_ratio_indexcache')
Tickers = []
file_path = './The_Ticker_file.txt'
with open(file_path, 'r') as file:
  Tickers = [line.strip() for line in file]

# ============== Maximum Window & Future Horizon ==============
K, H = 150, 40

# ============== Crypto Build ==============
start      = "2019-01-01"
val_start  = "2021-01-01"
test_start = "2022-01-01"
end        = "2024-12-31"
# fcfg = mod.FeatureConfig(
#     price_fields=['Open','High','Low','Close'],
#     returns_mode='log',
#     include_rvol=True, rvol_span=20, rvol_on='Close',
#     include_dlv=False,
#     market_proxy='BTC-USD',
#     target_field='Close',
# )
# ============== Equity Build ==============
# start      = "2016-01-01"
# val_start  = "2021-01-01"
# test_start = "2022-01-01"
# end        = "2025-06-30"
# fcfg = mod.FeatureConfig(
#     price_fields=['Open','High','Low','Close'],     # features from these price columns
#     returns_mode="log",         # log or pct
#     include_rvol=True, rvol_span=20, rvol_on="Close",
#     include_dlv=True,           # Δlog(volume)
#     market_proxy="SPY",         # adds market return factor
#     target_field="Close"        # target column becomes RET_CLOSE
# )
# ============== Dataset Directory ==============
DATA_DIR   = "./ldt/crypto_data"     # crypto_data / equity_data
FEATURES_DIR = f"{DATA_DIR}/features"  # your per-ticker parquet/pickle files live here

# ============== Download & Cache Dataset ==============
# mod.prepare_features_and_index_cache(
#     tickers=Tickers,
#     start=start, end=end,
#     window=K, horizon=H,
#     data_dir=DATA_DIR,
#     feature_cfg=fcfg,
#     normalize_per_ticker=True,    # or False if you prefer not useing per-asset stats
#     keep_time_meta="end",
# )

# ============== Statistic Check of Loaded Dataset ==============
with open(f"{DATA_DIR}/cache_ratio_index/meta.json", "r") as f:
    assets = json.load(f)["assets"]
N = len(assets)
print("assets in cache:", N)
print("first few assets:", assets[:10])

def make_panel_collate(assets):
    asset_count = len(assets)
    def collate(batch):
        # find unique dates in the batch (based on end-of-context time)
        def to_date(t):
            if isinstance(t, (list, tuple, np.ndarray)):  # 'full' mode may give arrays
                t = t[-1]
            return pd.Timestamp(t).normalize().date()

        dates = [to_date(m[2]["ctx_times"]) for m in batch]
        unique_dates, date_to_idx = [], {}
        for d in dates:
            if d not in date_to_idx:
                date_to_idx[d] = len(unique_dates)
                unique_dates.append(d)
        B = len(unique_dates)

        K, F = batch[0][0].shape
        H = batch[0][1].shape[-1] if batch[0][1].ndim else 1

        V = torch.zeros(B, asset_count, K, F)
        T = torch.zeros_like(V)
        Y = torch.zeros(B, asset_count, H)
        M = torch.zeros(B, asset_count, dtype=torch.bool)
        ctx_times = [[None]*asset_count for _ in range(B)]

        for x, y, meta in batch:
            aid = int(meta["asset_id"])
            b = date_to_idx[to_date(meta["ctx_times"])]
            V[b, aid] = x
            td = torch.zeros_like(x); td[1:] = x[1:] - x[:-1]  # first-diff
            T[b, aid] = td
            if y.ndim == 0:
                Y[b, aid, 0] = y
            else:
                Y[b, aid, :y.shape[-1]] = y
            M[b, aid] = True
            ctx_times[b][aid] = meta["ctx_times"]

        meta_out = {"entity_mask": M, "ctx_times": ctx_times, "dates": unique_dates}
        return [V, T], Y, meta_out
    return collate

# ============== Re-index The Window & Future Horizon ==============
kept = mod.rebuild_window_index_only(
    data_dir=DATA_DIR,
    window=120,
    horizon=10,
)
print("new total windows indexed:", kept)

# Use the panel collate + date batching to get grouped-by-date panels
panel_collate = make_panel_collate(assets)
train_dl, val_dl, test_dl, sizes = mod.load_dataloaders_with_ratio_split(
    data_dir=DATA_DIR,
    train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
    n_entities=N,
    shuffle_train=False,
    coverage_per_window=0.8,     # enforce min entities per date group, date below which will be dropped
    date_batching=True,
    dates_per_batch=64,          # enforce diversity in batch
    collate_fn=panel_collate,
    window=120,
    horizon=10
)
print("sizes (train,val,test):", sizes)

xb, yb, meta = next(iter(train_dl))
V, T = xb
M = meta["entity_mask"]
print("V:", V.shape, "T:", T.shape, "y:", yb.shape)         # -> [B,N,K,F], [B,N,K,F], [B,N,H]
print("min coverage:", float(M.float().mean(1).min().item()))
print("frac padded:", float((~M).float().mean().item()))

# Your date checks now match the shapes you expect:
def dates_seen(dl):
    seen = set()
    for xb, yb, meta in dl:
        M = meta["entity_mask"]
        for b in range(M.shape[0]):
            for n in range(M.shape[1]):
                if bool(M[b, n]):
                    d = pd.Timestamp(meta["ctx_times"][b][n]).normalize().date()
                    seen.add(d)
                    break
    return seen

S = dates_seen(train_dl)
print("dates seen in this split:", len(S))

ok = all(
    len({pd.Timestamp(meta["ctx_times"][b][n]).normalize().date()
         for n in range(M.shape[1]) if bool(M[b,n])}) == 1
    for b in range(M.shape[0])
)
print("same DATE per panel?", ok)
