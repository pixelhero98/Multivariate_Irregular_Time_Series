"""BMS (Beijing Multi-Site) air quality dataset utilities.

This module mirrors the public APIs exposed by :mod:`fin_dataset` so the rest of
the training pipeline can treat financial and air-quality data
interchangeably. In particular it produces the same compact cache layout
consisting of per-entity feature/target arrays accompanied by a global window
index. The resulting cache can therefore be consumed by the existing
ratio-split dataloader from :mod:`fin_dataset`, which also yields the familiar
``entity_mask`` metadata.

Targets ``Y`` correspond to future values of the ``PM2.5`` pollutant while the
context tensors (``V`` levels and ``T`` temporal first-differences) include the
full set of historical features *including* past ``PM2.5`` readings.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
import requests

from .fin_dataset import (
    CachePaths,
    load_dataloaders_with_ratio_split as _load_fin_ratio_split,
    rebuild_window_index_only as _rebuild_window_index_only,
)

# ---------------------------------------------------------------------------
# Public constants & configuration helpers

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/"
    "PRSA2017_Data_20130301-20170228.zip"
)
ARCHIVE_ROOT = "PRSA_Data_20130301-20170228"
TARGET_COLUMN = "PM2.5"

if TYPE_CHECKING:
    from os import PathLike as _PathLike

    PathLike = Union[str, _PathLike[str]]
else:
    PathLike = Union[str, os.PathLike]


@dataclass
class BMSCacheConfig:
    """Configuration used when building the compact cache."""

    window: int
    horizon: int
    data_dir: PathLike = "./bms_air_cache"
    raw_data_dir: Optional[PathLike] = "./bms_air_quality_data"
    normalize_per_station: bool = True
    clamp_sigma: float = 5.0
    keep_time_meta: str = "end"  # "full" | "end" | "none"
    freq: str = "H"  # hourly resolution
    stations: Optional[Sequence[str]] = None
    overwrite: bool = False


# ---------------------------------------------------------------------------
# Downloading & ingestion utilities

def download_bms_air_dataset(
    dest_path: PathLike = "./bms_air_quality_data",
    url: str = DATA_URL,
) -> Path:
    """Download and extract the raw dataset if it is not already present."""

    dest = Path(dest_path).expanduser().resolve()
    extracted = dest / ARCHIVE_ROOT
    if extracted.exists():
        return extracted

    dest.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(dest)
    return extracted


def load_raw_bms_frames(data_dir: PathLike) -> pd.DataFrame:
    """Load every monitoring station CSV into a single dataframe."""

    root = Path(data_dir)
    csv_files = sorted(root.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{root}'. Did you download the dataset?"
        )

    frames = []
    for csv in csv_files:
        df = pd.read_csv(csv)
        if "station" not in df.columns:
            raise ValueError(f"File '{csv}' does not contain the 'station' column.")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _encode_wind_direction(series: pd.Series) -> pd.Series:
    """Convert cardinal wind direction strings into stable float codes."""

    codes = series.astype("category").cat.codes
    codes = codes.replace(-1, np.nan)
    return codes.astype(np.float32)


def clean_bms_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform light cleaning and station-wise forward-filling of the dataset."""

    df = df.copy()
    # Build a timestamp using the provided components.
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])

    # Drop redundant index columns.
    df = df.drop(columns=["No", "year", "month", "day", "hour"], errors="ignore")

    numeric_cols = [
        "PM2.5",
        "PM10",
        "SO2",
        "NO2",
        "CO",
        "O3",
        "TEMP",
        "PRES",
        "DEWP",
        "RAIN",
        "WSPM",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "wd" in df.columns:
        df["WD_CODE"] = _encode_wind_direction(df["wd"])
        df = df.drop(columns="wd")

    df = df.sort_values(["station", "datetime"])  # ensure chronological order

    fill_cols = [c for c in df.columns if c not in {"station", "datetime"}]
    df[fill_cols] = df.groupby("station", sort=False)[fill_cols].ffill()

    return df


def _build_station_frames(
    df: pd.DataFrame,
    freq: str = "H",
    stations: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Transform the long dataframe into per-station panels aligned on time."""

    if stations is not None:
        df = df[df["station"].isin(set(stations))].copy()

    if df.empty:
        raise ValueError("No data available after applying station filters.")

    feature_cols = [c for c in df.columns if c not in {"station", "datetime"}]
    global_index = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq=freq)

    panels: Dict[str, pd.DataFrame] = {}
    for station, group in df.groupby("station"):
        station_df = group.set_index("datetime")[feature_cols]
        station_df = station_df.reindex(global_index)
        station_df = station_df.ffill()
        station_df = station_df.dropna()
        if not station_df.empty:
            panels[str(station)] = station_df.astype(np.float32)

    if not panels:
        raise ValueError("All station panels became empty after reindexing.")

    return panels


class _NormStatsAccumulator:
    """Accumulate normalization statistics for per-station or global modes."""

    def __init__(self, num_assets: int, feature_dim: int, per_asset: bool) -> None:
        self.per_asset = per_asset
        self.num_assets = int(num_assets)
        self.feature_dim = int(feature_dim)

        if per_asset:
            self.mean_x: List[Optional[np.ndarray]] = [None] * self.num_assets
            self.std_x: List[Optional[np.ndarray]] = [None] * self.num_assets
            self.mean_y: List[Optional[float]] = [None] * self.num_assets
            self.std_y: List[Optional[float]] = [None] * self.num_assets
        else:
            self.count = 0
            self.sum_x = np.zeros(self.feature_dim, dtype=np.float64)
            self.sumsq_x = np.zeros(self.feature_dim, dtype=np.float64)
            self.total_y = 0.0
            self.total_y_sq = 0.0
            self.total_y_count = 0

    def update(self, aid: int, features: np.ndarray, targets: np.ndarray) -> None:
        if self.per_asset:
            f64 = features.astype(np.float64, copy=False)
            t64 = targets.astype(np.float64, copy=False)
            mx = f64.mean(axis=0)
            sx = f64.std(axis=0)
            sx = np.where(sx == 0.0, 1.0, sx)
            my = float(t64.mean()) if t64.size else 0.0
            sy = float(t64.std()) if t64.size else 1.0
            sy = 1.0 if sy == 0.0 else sy
            self.mean_x[aid] = mx.reshape(1, 1, -1).astype(np.float32)
            self.std_x[aid] = sx.reshape(1, 1, -1).astype(np.float32)
            self.mean_y[aid] = my
            self.std_y[aid] = sy
        else:
            f64 = features.astype(np.float64, copy=False)
            t64 = targets.astype(np.float64, copy=False)
            self.count += f64.shape[0]
            self.sum_x += f64.sum(axis=0)
            self.sumsq_x += np.square(f64).sum(axis=0)
            self.total_y += float(t64.sum())
            self.total_y_sq += float(np.square(t64).sum())
            self.total_y_count += t64.shape[0]

    def finalize(self, assets: Iterable[str]) -> Dict[str, object]:
        assets_list = list(assets)
        if self.per_asset:
            if not all(
                m is not None and s is not None and my is not None and sy is not None
                for m, s, my, sy in zip(self.mean_x, self.std_x, self.mean_y, self.std_y)
            ):
                raise RuntimeError("Normalization statistics missing for some assets.")
            return {
                "per_ticker": True,
                "assets": assets_list,
                "mean_x": [mx.tolist() for mx in self.mean_x],
                "std_x": [sx.tolist() for sx in self.std_x],
                "mean_y": [float(my) for my in self.mean_y],
                "std_y": [float(sy) for sy in self.std_y],
            }

        if self.count == 0:
            raise RuntimeError("Unable to compute normalization statistics (no samples).")

        mean_x = (self.sum_x / self.count).astype(np.float32)
        var_x = (self.sumsq_x / self.count) - np.square(mean_x.astype(np.float64))
        var_x = np.maximum(var_x, 1e-12)
        std_x = np.sqrt(var_x).astype(np.float32)
        std_x[std_x == 0.0] = 1.0

        if self.total_y_count > 0:
            mean_y = float(self.total_y / self.total_y_count)
            var_y = max((self.total_y_sq / self.total_y_count) - (mean_y ** 2), 1e-12)
            std_y = float(np.sqrt(var_y))
            std_y = 1.0 if std_y == 0.0 else std_y
        else:
            mean_y = 0.0
            std_y = 1.0

        return {
            "per_ticker": False,
            "assets": assets_list,
            "mean_x": mean_x.reshape(1, 1, -1).tolist(),
            "std_x": std_x.reshape(1, 1, -1).tolist(),
            "mean_y": mean_y,
            "std_y": std_y,
        }


# ---------------------------------------------------------------------------
# Cache builder

def prepare_bms_air_cache(cfg: BMSCacheConfig) -> Mapping[str, List[str]]:
    """Prepare the compact cache with context features and PM2.5 targets.

    The resulting directory layout matches :func:`fin_dataset.prepare_features_and_index_cache`.

    Returns
    -------
    Mapping[str, List[str]]
        Metadata describing the assets and feature columns that were persisted.
    """

    if cfg.window <= 0:
        raise ValueError("window must be a positive integer")
    if cfg.horizon < 0:
        raise ValueError("horizon must be non-negative")
    keep_time_meta = cfg.keep_time_meta.lower()
    if keep_time_meta not in {"full", "end", "none"}:
        raise ValueError("keep_time_meta must be one of {'full', 'end', 'none'}")

    raw_root = cfg.raw_data_dir or "./bms_air_quality_data"
    raw_path = download_bms_air_dataset(raw_root)
    raw_df = load_raw_bms_frames(raw_path)
    clean_df = clean_bms_data(raw_df)

    panels = _build_station_frames(clean_df, freq=cfg.freq, stations=cfg.stations)

    assets = sorted(panels.keys())
    asset_to_id = {asset: idx for idx, asset in enumerate(assets)}

    feature_cols = list(next(iter(panels.values())).columns)
    if TARGET_COLUMN not in feature_cols:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from features.")
    # Ensure the target is the first column to make downstream inspection easier.
    feature_cols = [TARGET_COLUMN] + [c for c in feature_cols if c != TARGET_COLUMN]

    data_dir = Path(cfg.data_dir).expanduser().resolve()
    paths = CachePaths.from_dir(data_dir)
    if cfg.overwrite and paths.cache_root.exists():
        shutil.rmtree(paths.cache_root)
    paths.ensure()

    pairs: List[np.ndarray] = []
    end_times: List[np.ndarray] = []
    start_times: List[np.datetime64] = []
    stop_times: List[np.datetime64] = []

    # Normalization statistics accumulator.
    feature_dim = len(feature_cols)
    norm_acc = _NormStatsAccumulator(
        num_assets=len(assets),
        feature_dim=feature_dim,
        per_asset=cfg.normalize_per_station,
    )

    for asset in assets:
        aid = asset_to_id[asset]
        panel = panels[asset][feature_cols]
        total_rows = panel.shape[0]
        if total_rows < cfg.window + cfg.horizon:
            raise ValueError(
                f"Station '{asset}' has insufficient history for the requested window/horizon."
            )

        features = panel.to_numpy(dtype=np.float32, copy=True)
        targets = panel[TARGET_COLUMN].to_numpy(dtype=np.float32, copy=True)
        times = panel.index.to_numpy(dtype="datetime64[ns]")

        np.save(paths.features / f"{aid}.npy", features.astype(np.float16, copy=False))
        np.save(paths.targets / f"{aid}.npy", targets.astype(np.float16, copy=False))
        np.save(paths.times / f"{aid}.npy", times)

        norm_acc.update(aid, features, targets)

        max_start = total_rows - (cfg.window + cfg.horizon) + 1
        if max_start <= 0:
            continue
        starts = np.arange(0, max_start, dtype=np.int32)
        window_end_times = times[starts + cfg.window - 1].astype("datetime64[ns]")
        pairs.append(np.stack([np.full_like(starts, aid), starts], axis=1))
        end_times.append(window_end_times)
        start_times.append(times[0])
        stop_times.append(times[-1])

    if not pairs:
        raise RuntimeError("No valid context windows available across stations.")

    global_pairs = np.concatenate(pairs, axis=0).astype(np.int32)
    global_end_times = np.concatenate(end_times, axis=0).astype("datetime64[ns]")
    np.save(paths.windows / "global_pairs.npy", global_pairs)
    np.save(paths.windows / "end_times.npy", global_end_times)

    norm_stats = norm_acc.finalize(assets)

    dataset_start = min(start_times).astype("datetime64[s]") if start_times else None
    dataset_end = max(stop_times).astype("datetime64[s]") if stop_times else None

    meta = {
        "dataset": "bms_air_quality",
        "format": "indexcache_v1",
        "assets": assets,
        "asset2id": asset_to_id,
        "feature_cols": feature_cols,
        "target_col": TARGET_COLUMN,
        "window": int(cfg.window),
        "horizon": int(cfg.horizon),
        "keep_time_meta": keep_time_meta,
        "normalize_per_ticker": cfg.normalize_per_station,
        "clamp_sigma": cfg.clamp_sigma,
        "freq": cfg.freq,
        "start": str(dataset_start) if dataset_start is not None else None,
        "end": str(dataset_end) if dataset_end is not None else None,
    }

    with paths.meta.open("w") as f:
        json.dump(meta, f, indent=2)

    with paths.norm_stats.open("w") as f:
        json.dump(norm_stats, f, indent=2)

    return {"assets": assets, "feature_cols": feature_cols}


# ---------------------------------------------------------------------------
# Loader wrappers

def load_bms_dataloaders_with_ratio_split(
    data_dir: PathLike,
    **loader_kwargs,
):
    """Wrapper around the financial ratio-split loader for BMS datasets."""

    return _load_fin_ratio_split(data_dir=data_dir, **loader_kwargs)


def _validate_bms_cache(paths: CachePaths) -> Dict[str, object]:
    """Read and validate the cache metadata, returning the parsed JSON."""

    if not paths.meta.exists():
        raise FileNotFoundError(
            f"Cache meta file not found at '{paths.meta}'. Did you run prepare_bms_air_cache()?"
        )

    with paths.meta.open("r") as f:
        meta: Dict[str, object] = json.load(f)

    dataset = meta.get("dataset")
    if dataset not in {"bms_air_quality", "bms_air_dataset"}:
        raise ValueError(
            f"The cache located at '{paths.cache_root}' does not correspond to the BMS "
            "air-quality dataset."
        )

    return meta


def run_experiment(
    data_dir: PathLike,
    K: int,
    H: int,
    *,
    ratios=(0.7, 0.1, 0.2),
    per_asset: bool = True,
    date_batching: bool = True,
    coverage: float = 0.85,
    dates_per_batch: int = 30,
    batch_size: int = 64,
    norm: str = "train_only",
    reindex: bool = True,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
):
    """Build train/val/test loaders for the prepared BMS cache.

    This mirrors :func:`Dataset.fin_dataset.run_experiment` so the training
    pipeline can swap between the financial and BMS datasets transparently.
    """

    paths = CachePaths.from_dir(data_dir)
    meta = _validate_bms_cache(paths)
    base_window = int(meta.get("window", K))
    base_horizon = int(meta.get("horizon", H))
    if K > base_window or H > base_horizon:
        raise ValueError(
            "Requested (window, horizon) exceed the cached configuration. "
            "Re-run prepare_bms_air_cache with larger values first."
        )

    if reindex:
        _rebuild_window_index_only(
            data_dir,
            window=K,
            horizon=H,
            update_meta=False,
            backup_old=False,
        )

    train_dl, val_dl, test_dl, lengths = load_bms_dataloaders_with_ratio_split(
        data_dir=data_dir,
        train_ratio=ratios[0],
        val_ratio=ratios[1],
        test_ratio=ratios[2],
        batch_size=batch_size,
        regression=True,
        per_asset=per_asset,
        norm_scope=norm,
        date_batching=date_batching,
        coverage_per_window=coverage,
        dates_per_batch=dates_per_batch,
        window=K,
        horizon=H,
        shuffle_train=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_dl, val_dl, test_dl, lengths


__all__ = [
    "BMSCacheConfig",
    "download_bms_air_dataset",
    "load_raw_bms_frames",
    "clean_bms_data",
    "prepare_bms_air_cache",
    "load_bms_dataloaders_with_ratio_split",
    "run_experiment",
]

