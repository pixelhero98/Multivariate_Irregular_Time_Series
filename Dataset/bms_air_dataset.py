"""BMS (Beijing Multi-Site) air quality dataset utilities.

This module mirrors the public APIs exposed by :mod:`fin_dataset` so the rest of
the training pipeline can treat financial and air-quality data interchangeably.
In particular it produces the same compact cache layout consisting of per-entity
feature/target arrays accompanied by a global window index. The resulting cache
can therefore be consumed by the existing ratio-split dataloader from
``fin_dataset`` which also yields the familiar ``entity_mask`` metadata.

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
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import requests

from .fin_dataset import (
    CachePaths,
    load_dataloaders_with_ratio_split as _load_fin_ratio_split,
)

# ---------------------------------------------------------------------------
# Public constants & configuration helpers

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/"
    "PRSA2017_Data_20130301-20170228.zip"
)
ARCHIVE_ROOT = "PRSA_Data_20130301-20170228"
TARGET_COLUMN = "PM2.5"

PathLike = Union[str, os.PathLike[str]]


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

def download_bms_air_dataset(dest_path: PathLike = "./bms_air_quality_data",
                             url: str = DATA_URL) -> Path:
    """Download and extract the raw dataset if it is not already present.

    Parameters
    ----------
    dest_path:
        Directory where the archive should be unpacked. The extracted CSV files
        will live under ``<dest_path> / ARCHIVE_ROOT``.
    url:
        Download location for the zipped dataset. Overridable for testing.

    Returns
    -------
    Path
        Path pointing to the extracted directory that contains all CSV files.
    """

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

    codes = series.astype("category").cat.codes.astype(np.float32)
    codes = codes.replace(-1, np.nan)
    return codes


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


def _build_station_frames(df: pd.DataFrame, freq: str = "H",
                          stations: Optional[Sequence[str]] = None) -> Dict[str, pd.DataFrame]:
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

    raw_root = cfg.raw_data_dir
    if raw_root is None:
        raw_root = "./bms_air_quality_data"
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

    # Persist per-station arrays.
    for asset in assets:
        aid = asset_to_id[asset]
        panel = panels[asset][feature_cols]
        if panel.shape[0] < cfg.window + cfg.horizon:
            raise ValueError(
                f"Station '{asset}' has insufficient history for the requested window/horizon."
            )

        features = panel.to_numpy(dtype=np.float32, copy=True)
        targets = panel[TARGET_COLUMN].to_numpy(dtype=np.float32, copy=True)
        times = panel.index.to_numpy(dtype="datetime64[ns]")

        np.save(paths.features / f"{aid}.npy", features.astype(np.float16))
        np.save(paths.targets / f"{aid}.npy", targets.astype(np.float16))
        np.save(paths.times / f"{aid}.npy", times)

    # Construct the global window index.
    pairs: List[np.ndarray] = []
    end_times: List[np.ndarray] = []
    for asset in assets:
        aid = asset_to_id[asset]
        times = np.load(paths.times / f"{aid}.npy")
        total = times.shape[0]
        max_start = total - (cfg.window + cfg.horizon) + 1
        if max_start <= 0:
            continue
        starts = np.arange(0, max_start, dtype=np.int32)
        ends = times[starts + cfg.window - 1].astype("datetime64[ns]")
        pairs.append(np.stack([np.full_like(starts, aid), starts], axis=1))
        end_times.append(ends)

    if not pairs:
        raise RuntimeError("No valid context windows available across stations.")

    global_pairs = np.concatenate(pairs, axis=0).astype(np.int32)
    global_end_times = np.concatenate(end_times, axis=0).astype("datetime64[ns]")
    np.save(paths.windows / "global_pairs.npy", global_pairs)
    np.save(paths.windows / "end_times.npy", global_end_times)

    # Compute normalization statistics (per-station or global).
    norm_stats: Dict[str, object]
    if cfg.normalize_per_station:
        norm_stats = {
            "per_ticker": True,
            "assets": assets,
            "mean_x": [],
            "std_x": [],
            "mean_y": [],
            "std_y": [],
        }
        for asset in assets:
            aid = asset_to_id[asset]
            features = np.load(paths.features / f"{aid}.npy").astype(np.float32)
            targets = np.load(paths.targets / f"{aid}.npy").astype(np.float32)
            mx = features.mean(axis=0, dtype=np.float64)
            sx = features.std(axis=0, dtype=np.float64)
            sx = np.where(sx == 0.0, 1.0, sx)
            my = float(targets.mean(dtype=np.float64))
            sy = float(targets.std(dtype=np.float64))
            sy = 1.0 if sy == 0.0 else sy
            norm_stats["mean_x"].append(mx.reshape(1, 1, -1).astype(np.float32).tolist())
            norm_stats["std_x"].append(sx.reshape(1, 1, -1).astype(np.float32).tolist())
            norm_stats["mean_y"].append(my)
            norm_stats["std_y"].append(sy)
    else:
        total_count = 0
        sum_x = None
        sumsq_x = None
        total_y = 0.0
        total_y_sq = 0.0
        total_y_count = 0
        for asset in assets:
            aid = asset_to_id[asset]
            features = np.load(paths.features / f"{aid}.npy").astype(np.float32)
            targets = np.load(paths.targets / f"{aid}.npy").astype(np.float32)
            if sum_x is None:
                sum_x = np.zeros(features.shape[1], dtype=np.float64)
                sumsq_x = np.zeros(features.shape[1], dtype=np.float64)
            sum_x += features.sum(axis=0, dtype=np.float64)
            sumsq_x += np.square(features.astype(np.float64)).sum(axis=0)
            total_count += features.shape[0]
            total_y += float(targets.sum(dtype=np.float64))
            total_y_sq += float(np.square(targets.astype(np.float64)).sum())
            total_y_count += targets.shape[0]

        if total_count == 0:
            raise RuntimeError("Unable to compute normalization statistics (no samples).")

        mean_x = (sum_x / total_count).astype(np.float32)
        var_x = (sumsq_x / total_count) - np.square(mean_x.astype(np.float64))
        var_x = np.maximum(var_x, 1e-12)
        std_x = np.sqrt(var_x).astype(np.float32)
        std_x[std_x == 0.0] = 1.0

        mean_y = float(total_y / max(1, total_y_count))
        var_y = max((total_y_sq / max(1, total_y_count)) - (mean_y ** 2), 1e-12)
        std_y = float(np.sqrt(var_y))
        std_y = 1.0 if std_y == 0.0 else std_y

        norm_stats = {
            "per_ticker": False,
            "assets": assets,
            "mean_x": mean_x.reshape(1, 1, -1).tolist(),
            "std_x": std_x.reshape(1, 1, -1).tolist(),
            "mean_y": mean_y,
            "std_y": std_y,
        }

    meta = {
        "dataset": "bms_air_quality",
        "assets": assets,
        "asset2id": asset_to_id,
        "feature_cols": feature_cols,
        "target_col": TARGET_COLUMN,
        "window": int(cfg.window),
        "horizon": int(cfg.horizon),
        "keep_time_meta": cfg.keep_time_meta,
        "normalize_per_ticker": cfg.normalize_per_station,
        "clamp_sigma": cfg.clamp_sigma,
        "freq": cfg.freq,
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


__all__ = [
    "BMSCacheConfig",
    "download_bms_air_dataset",
    "load_raw_bms_frames",
    "clean_bms_data",
    "prepare_bms_air_cache",
    "load_bms_dataloaders_with_ratio_split",
]

