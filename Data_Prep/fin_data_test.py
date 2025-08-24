import json, importlib
import pandas as pd
mod = importlib.import_module('fin_data_prep_f16_ultramem_autobatch')


Tickers = []
file_path = './NASDAQ_top.txt'
with open(file_path, 'r') as file:
  Tickers = [line.strip() for line in file]
K, H = 150, 40
# Crypto time span
# start      = "2019-01-01"
# val_start  = "2021-01-01"
# test_start = "2022-01-01"
# end        = "2024-12-31"

# Equity time span
start      = "2016-01-01"
val_start  = "2021-01-01"
test_start = "2022-01-01"
end        = "2025-06-30"

# Data cache
DATA_DIR   = "./ldt"     # crypto_data / equity_data
FEATURES_DIR = f"{DATA_DIR}/features"  # your per-ticker parquet/pickle files live here

# Crypto construction
# fcfg = mod.FeatureConfig(
#     price_fields=['Open','High','Low','Close'],
#     returns_mode='log',
#     include_rvol=True, rvol_span=20, rvol_on='Close',
#     include_dlv=False,
#     market_proxy='BTC-USD',
#     target_field='Close',
# )

# Equity construction
fcfg = mod.FeatureConfig(
    price_fields=['Open','High','Low','Close'],     # features from these price columns
    returns_mode="log",         # log or pct
    include_rvol=True, rvol_span=20, rvol_on="Close",
    include_dlv=True,           # Δlog(volume)
    market_proxy="SPY",         # adds market return factor
    target_field="Close"        # target column becomes RET_CLOSE
)

(Xtr, Ytr, IDtr), (Xva, Yva, IDva), (Xte, Yte, IDte) = mod.prepare_stock_windows_and_cache_v2(
    tickers=Tickers,
    start=start, val_start=val_start, test_start=test_start, end=end,
    window=K, horizon=H,
    data_dir=DATA_DIR,
    feature_cfg=fcfg,
    normalize_per_ticker=False,   # or True if you prefer per-asset normalization
    min_obs_buffer=0,
    keep_time_meta="end",         # keep the end timestamp for grouping; "full" keeps full timeline
)

with open(f"{DATA_DIR}/meta_v2.json","r") as f:
    assets = json.load(f)["assets"]
N = len(assets)

train_dl, val_dl, test_dl, sizes = mod.load_dataloaders_with_ratio_split(
    data_dir=DATA_DIR,
    train_ratio=0.55, val_ratio=0.05, test_ratio=0.40,
    batch_size=64,                 # ignored when date_batching=True
    n_entities=N,
    pad_incomplete="zeros",
    shuffle_train=False,
    coverage_per_window=0.85,
    date_batching=True,
    dates_per_batch=30,            # if date_batching=True, then it is the effective batch size
)

xb, yb, meta = next(iter(train_dl))
V, T = xb
M = meta["entity_mask"]  # [B, N] bool

print("V:", V.shape, "T:", T.shape, "y:", yb.shape)            # expect [B,N,K,F], [B,N,K,F], [B,N,H]
print("min coverage:", float(M.float().mean(1).min().item()))   # >= coverage_per_window
print("frac padded:", float((~M).float().mean().item()))        # goes down as you raise coverage
def dates_seen(dl):
    seen = set()
    for xb, yb, meta in dl:
        M = meta["entity_mask"]  # [B,N]
        for b in range(M.shape[0]):
            # take any real entity to read the panel date
            for n in range(M.shape[1]):
                if bool(M[b, n]):
                    d = pd.Timestamp(meta["ctx_times"][b][n]).normalize().date()
                    seen.add(d)
                    break
    return seen

S = dates_seen(train_dl)
print("dates seen in this split:", len(S))
# All real entities share the same DATE per panel?
import pandas as pd
ok = all(
    len({pd.Timestamp(meta["ctx_times"][b][n]).normalize().date()
         for n in range(M.shape[1]) if bool(M[b,n])}) == 1
    for b in range(M.shape[0])
)
print("same DATE per panel?", ok)
# raw unique dates in the split (before any coverage filtering)
import pandas as pd
def raw_unique_dates(dl):
    ds = dl.dataset
    CT = getattr(ds, "ctx_times", None)
    if CT is None:     # _ConcatIndexDataset path
        # just iterate once over the whole dataset (not the loader) to count dates
        seen = set()
        for i in range(len(ds)):
            _, _, m = ds[i]
            t = m["ctx_times"][-1] if isinstance(m["ctx_times"], (list, tuple)) else m["ctx_times"]
            seen.add(pd.Timestamp(t).normalize().date())
        return len(seen)
    # memmap/RAM path
    end = CT[:, -1] if (getattr(CT, "ndim", 1) == 2) else CT
    return len(pd.to_datetime(end).normalize().unique())

raw = raw_unique_dates(train_dl)
kept = len({pd.Timestamp(m).normalize().date()
            for xb, yb, meta in train_dl
            for b in range(meta["entity_mask"].shape[0])
            for n in range(meta["entity_mask"].shape[1])
            if bool(meta["entity_mask"][b, n]) and (m := meta["ctx_times"][b][n])})
print(f"dates raw={raw}, kept={kept}, drop_rate={(raw-kept)/max(1,raw):.1%}")
