from fin_dataset_gen import (
    FeatureConfig,
    prepare_features_and_index_cache,
    run_experiment
)
import pandas as pd


DATA_DIR = "./ldt/data"
start      = "2019-01-01"
end        = "2024-12-31"

# Read universe
with open("./CRYPTO_top.txt", "r") as f:
    Tickers = [line.strip() for line in f]

# If using BTC as the market proxy, remove it from the asset universe (recommended)
if "BTC-USD" in Tickers:
    Tickers = [t for t in Tickers if t != "BTC-USD"]

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


# 3) Load loaders (chronological per asset; train-only normalization)
train_dl, val_dl, test_dl, sizes = run_experiment(data_dir=DATA_DIR,
                                                K=120,
                                                H=40,
                                                per_asset=False,
                                                date_batching=True,
                                                dates_per_batch=30,
                                                coverage=0.85)

# 4) Quick peek
X, Y, M = next(iter(train_dl))
levels_T, diffs_T = X
print("K used:", levels_T.shape[-2], "F:", levels_T.shape[-1])
print("H used:", Y.shape[-1])
print(levels_T.shape)
# Rough normalization check (should be ~0/1 with train_only)
import torch
batch_mu = levels_T.mean(dim=(0,1,2))
batch_sd = levels_T.std (dim=(0,1,2))
print("approx batch mean (first 5 features):", batch_mu[:5])
print("approx batch std  (first 5 features):", batch_sd[:5])


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
print(sizes)