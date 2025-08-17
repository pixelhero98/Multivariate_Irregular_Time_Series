from fin_data_prep_f16_ultramem import (
    prepare_stock_windows_and_cache_v2,
    load_dataloaders_with_meta_v2,
    load_dataloaders_with_ratio_split,
    FeatureConfig, CalendarConfig
)
import numpy as np
from find_tickers import build_us_only_universes


Tickers = []
win, hor = 150, 40
# Define the path to your text file
file_path = './CRYPTO_top.txt'

# Use a 'with' block to safely open and close the file
with open(file_path, 'r') as file:
  # Create a list where each line is an element, with whitespace/newlines removed
  Tickers = [line.strip() for line in file]
# Tickers = Tickers[:300]
cfg = FeatureConfig(
    price_fields=['Close'],
    returns_mode='log',
    include_rvol=True, rvol_span=20, rvol_on='Close',
    include_dlv=False,                 # safer for crypto initially
    market_proxy='BTC-USD',            # use BTC as market factor
    include_oc=False, include_gap=False, include_hl_range=False
)

prepare_stock_windows_and_cache_v2(
    tickers=Tickers,
    start="2017-01-01",
    val_start="2023-01-01",
    test_start="2024-01-01",
    end="2025-06-30",
    window=win,
    horizon=hor,
    data_dir="./ldt/crypto_data",
    feature_cfg=cfg,
    normalize_per_ticker=True,
    min_train_coverage=0.85,
    liquidity_rank_window=None,
    top_n_by_dollar_vol=None,  # optional extra filter inside the prep step
    regression=True
)


# feat_cfg = FeatureConfig(
#     price_fields=["Close"],      # returns computed for each listed field
#     returns_mode="log",          # 'log' or 'pct'
#     include_rvol=True,           # realized vol over returns
#     rvol_span=20,
#     rvol_on="Close",
#     include_dlv=True,            # Δ log volume
#     market_proxy="SPY",          # extra MKT factor from SPY
#     include_oc=False,
#     include_gap=False,
#     include_hl_range=False,
#     target_field="Close",# predict future return on this field
#     calendar=CalendarConfig(     # calendar features on/off
#         include_dow=True,
#         include_dom=True,
#         include_moy=True,
#     ),
# )
#
# # --- (B) build the on-disk cache once (writes ./data/cache_v2.npz + meta jsons) ---
# prepare_stock_windows_and_cache_v2(
#     tickers=Tickers,
#     start="2015-01-01",
#     val_start="2021-01-01",
#     test_start="2022-01-01",
#     end="2025-06-30",
#     window=150,                   # context length (K)
#     horizon=40,                   # prediction horizon (H)
#     data_dir="./ldt/data",
#     feature_cfg=feat_cfg,
#     normalize_per_ticker=True,   # train-only, per-ticker normalization
#     clamp_sigma=5.0,
#     min_obs_buffer=50,
#     min_train_coverage=0.9,
#     liquidity_rank_window=None,  # e.g. ("2020-01-01","2020-12-31") with top_n_by_dollar_vol=50
#     top_n_by_dollar_vol=None,
#     max_windows_per_ticker=None,
#     regression=True,
#     seed=1337,
#     keep_time_meta="end",
# )

# train_dl, val_dl, test_dl, sizes = load_dataloaders_with_ratio_split(
#     data_dir="./ldt/crypto_data",
#     train_ratio=0.55,
#     val_ratio=0.05,
#     test_ratio=0.4,
#     batch_size=64,
#     regression=True,
#     per_asset=True,        # keep time order per asset
#     shuffle_train=True,
#     num_workers=0,
#     seed=42,
# )
#
# print("sizes:", sizes)
# xb, yb, meta = next(iter(train_dl))
# print(xb.shape, yb.shape, meta["asset"][:10])
# print(xb[0])
# B = xb.size(0)
# for i in range(B):
#     aid     = meta["asset_id"][i]     # int id
#     ticker  = meta["asset"][i]        # string ticker
#     ctx_end = meta["ctx_times"][i][-1]  # end-of-context timestamp
#     y_range = (meta["y_times"][i][0], meta["y_times"][i][-1])
#
#     print(f"batch[{i}]: aid={aid:3d} ticker={ticker:5s} "
#           f"ctx_end={ctx_end}  y={y_range[0]}→{y_range[1]}")

# choose markets & top-K (user-defined)

# topk = {"NASDAQ":1000, "NYSE":1000}
#
# results = build_us_only_universes(
#     topk=topk,
#     start="2015-01-01", end="2025-06-30",
#     train_ratio=0.55, val_ratio=0.05, test_ratio=0.4,
#     rank_start="2016-01-01", rank_end="2017-12-31",
#     K=120, H=30, buffer_days=50,
#     presence_mode="dynamic",
#     min_windows_train=128, min_windows_val=8, min_windows_test=32,
#     out_dir="./ldt"
# )
#
# # results is a dict: market -> (selected_tickers, stats_df)
# for m, (sel, stats) in results.items():
#     print(m, len(sel))

