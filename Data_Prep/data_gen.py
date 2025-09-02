from fin_dataset import (
    FeatureConfig,
    prepare_features_and_index_cache,
    rebuild_window_index_only,
    load_dataloaders_with_ratio_split,
)
import pandas as pd
import numpy as np


DATA_DIR = "./ldt/data"
start      = "2019-01-01"
end        = "2024-12-31"

# # Read universe
# with open("./CRYPTO_top.txt", "r") as f:
#     Tickers = [line.strip() for line in f]
#
# # If using BTC as the market proxy, remove it from the asset universe (recommended)
# if "BTC-USD" in Tickers:
#     Tickers = [t for t in Tickers if t != "BTC-USD"]
#
# fc = FeatureConfig(
#     returns_mode="log",
#     price_fields=("Open","High","Low","Close","Volume"),
#     include_rvol=True, rvol_span=10, rvol_on="Close",
#     include_dlv=True,                 # usually True for crypto/equities
#     market_proxy="BTC-USD",           # cross-asset proxy
#     include_oc=True, include_gap=True, include_hl_range=True,
#     target_field="Close",
#     include_entity_id_feature=False,
# )
#
# # 1) Build cache at your MAX intended K/H
# ok = prepare_features_and_index_cache(
#     tickers=Tickers,
#     start=start, end=end,
#     window=200, horizon=100,          # K_max, H_max
#     data_dir=DATA_DIR,
#     feature_cfg=fc,
#     keep_time_meta="end",
#     normalize_per_ticker=False,
#     clamp_sigma=5.0,
#     min_obs_buffer=10,
#     regression=True,
#     seed=123,
# )
# print("cache built:", ok)


def distinct_end_dates(dl, max_batches=None):
    """Return a sorted list of unique end-dates (YYYY-MM-DD) seen in a DataLoader."""
    seen = set()
    b = 0
    for (_, _), _, meta in dl:
        # Try common meta keys; fall back to computing day keys
        if "date_keys" in meta:  # int days since epoch
            days = pd.to_datetime(meta["date_keys"].cpu().numpy(), unit="D").date
        elif "dates" in meta:    # string/np.datetime64 list
            days = pd.to_datetime(np.array(meta["dates"])).date
        elif "ctx_times" in meta:  # np.datetime64 ns per sample
            days = pd.to_datetime(np.array(meta["ctx_times"])).normalize().date
        elif "end_times" in meta:
            days = pd.to_datetime(np.array(meta["end_times"])).normalize().date
        else:
            # Robust fallback: if no date info is exposed, skip
            days = []

        for d in np.unique(days):
            seen.add(str(d))
        b += 1
        if max_batches and b >= max_batches:
            break
    return sorted(seen)

def run_experiment(
    data_dir: str,
    K: int,
    H: int,
    *,
    ratios=(0.7, 0.1, 0.2),
    per_asset=True,
    date_batching=True,
    coverage=0.85,
    dates_per_batch=30,
    batch_size=64,
    norm="train_only",    # default is "train_only" in the patched loader; set "cache" if you want fixed μ/σ
    reindex=True,         # set False if NOT using date batching and you don’t need to rebuild end_times
):
    """
    Builds loaders for a given (K, H) using the already-downloaded cache.
    - If date_batching=True, reindex first so day end_times align with K/H.
    - Coverage threshold is applied per day; panel width is auto-detected from the cache.
    """
    if reindex:
        rebuild_window_index_only(DATA_DIR, window=K, horizon=H, update_meta=False, backup_old=False)

    train_dl, val_dl, test_dl, lengths = load_dataloaders_with_ratio_split(
        data_dir=data_dir,
        train_ratio=ratios[0],
        val_ratio=ratios[1],
        test_ratio=ratios[2],
        batch_size=batch_size,
        regression=True,         # set False for classification
        per_asset=per_asset,     # freely tunable each run
        norm_scope=norm,         # "train_only" (recommended) or "cache"
        date_batching=date_batching,
        coverage_per_window=coverage,
        dates_per_batch=dates_per_batch,
        window=K,
        horizon=H,
    )
    return train_dl, val_dl, test_dl, lengths

# 3) Load loaders (chronological per asset; train-only normalization)
train_dl, val_dl, test_dl, sizes = run_experiment(data_dir=DATA_DIR,
                                                K=120,
                                                H=50,
                                                per_asset=False,
                                                date_batching=True,
                                                dates_per_batch=30,
                                                coverage=0.80)
train_size, val_size, test_size = sizes
xb, yb, meta = next(iter(train_dl))
V, T = xb
M = meta["entity_mask"]
print("sizes:", sizes)
print("V:", V.shape, "T:", T.shape, "y:", yb.shape)         # -> [B,N,K,F], [B,N,K,F], [B,N,H]
print("min coverage:", float(M.float().mean(1).min().item()))
print("frac padded:", float((~M).float().mean().item()))
# Example
train_days = distinct_end_dates(train_dl)      # all train dates
val_days   = distinct_end_dates(val_dl)
test_days  = distinct_end_dates(test_dl)

print("train distinct dates:", len(train_days))
