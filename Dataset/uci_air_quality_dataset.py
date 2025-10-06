"""UCI Air Quality dataset utilities.

This module mirrors the public API provided by :mod:`Dataset.fin_dataset` and
the other dataset wrappers living in the :mod:`Dataset` package.  It prepares
the UCI Air Quality dataset using the compact "ratio index" cache format so the
existing training and evaluation pipelines can consume the data without any
additional glue code.

The dataset contains hourly air quality measurements collected by a single
monitoring station in an Italian city.  We model the future concentration of
``C6H6(GT)`` (benzene) as the regression target while the context features
include the remaining pollutant and meteorological readings provided by the
original dataset.  Missing values are denoted with ``-200`` in the raw files;
the cleaning pipeline replaces those entries with ``NaN`` and performs
time-based interpolation followed by forward/backward filling.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
import requests

from .fin_dataset import (
    CachePaths,
    load_dataloaders_with_ratio_split as _load_fin_ratio_split,
    rebuild_window_index_only as _rebuild_window_index_only,
)

if TYPE_CHECKING:
    from os import PathLike as _PathLike

    PathLike = Union[str, _PathLike[str]]
else:
    PathLike = Union[str, os.PathLike]


# ---------------------------------------------------------------------------
# Public constants & configuration helpers

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
ARCHIVE_CSV_NAME = "AirQualityUCI.csv"
TARGET_COLUMN = "C6H6(GT)"
DEFAULT_FEATURE_COLUMNS: Tuple[str, ...] = (
    "CO(GT)",
    "PT08.S1(CO)",
    "NMHC(GT)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
)
ASSET_NAME = "uci_air_monitor"


@dataclass
class UCIAirCacheConfig:
    """Configuration used when preparing the compact UCI cache."""

    window: int
    horizon: int
    data_dir: PathLike = "./uci_air_cache"
    raw_data_dir: Optional[PathLike] = "./uci_air_quality_raw"
    freq: str = "H"
    target_column: str = TARGET_COLUMN
    feature_columns: Optional[Sequence[str]] = None
    normalize_per_asset: bool = False
    clamp_sigma: float = 5.0
    keep_time_meta: str = "end"  # "full" | "end" | "none"
    overwrite: bool = False


# ---------------------------------------------------------------------------
# Helpers


def _resolve_feature_columns(
    feature_columns: Optional[Sequence[str]],
    target_column: str,
) -> Tuple[str, ...]:
    """Return a clean feature column ordering with the target in front."""

    if feature_columns is None:
        cols = list(DEFAULT_FEATURE_COLUMNS)
    else:
        cols = [str(c) for c in feature_columns]

    if target_column not in cols:
        cols.insert(0, target_column)
    else:
        cols = [target_column] + [c for c in cols if c != target_column]

    return tuple(cols)


def _find_existing_member(root: Path, filenames: Sequence[str]) -> Optional[Path]:
    """Return the first existing file under ``root`` matching ``filenames``."""

    for name in filenames:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def download_uci_air_quality_dataset(
    dest_path: PathLike = "./uci_air_quality_raw",
    url: str = DATA_URL,
) -> Path:
    """Download the Air Quality dataset archive and extract the CSV file.

    Parameters
    ----------
    dest_path:
        Directory where the extracted CSV will be stored.  If the file already
        exists, it is reused as-is.
    url:
        Download URL for the UCI archive.
    """

    dest = Path(dest_path).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    existing = _find_existing_member(dest, (ARCHIVE_CSV_NAME, "AirQualityUCI.xlsx"))
    if existing is not None:
        return existing

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not members:
            members = [m for m in zf.namelist() if m.lower().endswith(('.xlsx', '.xls'))]
        if not members:
            raise RuntimeError("Unable to locate a CSV/XLSX file inside the UCI archive.")

        member = members[0]
        target_path = dest / Path(member).name
        with zf.open(member) as source, target_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)

    return target_path


def load_raw_uci_air_quality(data_path: PathLike) -> pd.DataFrame:
    """Load the raw Air Quality dataset into a :class:`pandas.DataFrame`."""

    path = Path(data_path)
    if path.is_dir():
        existing = _find_existing_member(path, (ARCHIVE_CSV_NAME, "AirQualityUCI.xlsx"))
        if existing is None:
            raise FileNotFoundError(
                f"No AirQualityUCI CSV/XLSX file found in '{path}'. Did you download the dataset?"
            )
        path = existing

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(
            path,
            sep=";",
            decimal=",",
            na_values=[-200, "-200"],
            parse_dates=False,
            engine="python",
        )
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, na_values=[-200])
    else:
        raise ValueError(f"Unsupported Air Quality file format: '{path.suffix}'.")

    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def clean_uci_air_quality(
    df: pd.DataFrame,
    *,
    freq: str = "H",
    feature_columns: Optional[Sequence[str]] = None,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Clean the raw Air Quality dataframe and return a panel of features.

    The resulting dataframe is indexed by timestamps sampled at ``freq`` and the
    columns contain the numeric features used for training.  The target column
    is included in the returned frame to simplify downstream cache creation.
    """

    if "Date" not in df.columns or "Time" not in df.columns:
        raise ValueError("Raw UCI Air Quality dataframe must contain 'Date' and 'Time' columns.")

    frame = df.copy()
    timestamp = pd.to_datetime(
        frame["Date"].astype(str).str.strip() + " " + frame["Time"].astype(str).str.strip(),
        dayfirst=True,
        errors="coerce",
    )
    frame["datetime"] = timestamp
    frame = frame.dropna(subset=["datetime"])

    frame = frame.set_index("datetime").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]

    feature_cols = _resolve_feature_columns(feature_columns, target_column)
    missing = [col for col in feature_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected columns in UCI Air Quality data: {missing}")

    numeric = frame[feature_cols].apply(pd.to_numeric, errors="coerce")

    if freq:
        numeric = numeric.resample(freq).mean()

    numeric = numeric.interpolate(method="time", limit_direction="both")
    numeric = numeric.ffill().bfill()
    numeric = numeric.dropna(how="any")

    return numeric.astype(np.float32)


class _NormStatsAccumulator:
    """Accumulate normalization statistics for per-asset or global modes."""

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

        if getattr(self, "count", 0) == 0:
            raise RuntimeError("Unable to compute normalization statistics (no samples).")

        mean_x = (self.sum_x / self.count).astype(np.float32)
        var_x = (self.sumsq_x / self.count) - np.square(mean_x.astype(np.float64))
        var_x = np.maximum(var_x, 1e-12)
        std_x = np.sqrt(var_x).astype(np.float32)
        std_x[std_x == 0.0] = 1.0

        if getattr(self, "total_y_count", 0) > 0:
            mean_y = float(self.total_y / self.total_y_count)
            var_y = max((self.total_y_sq / self.total_y_count) - (mean_y**2), 1e-12)
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


def prepare_uci_air_cache(cfg: UCIAirCacheConfig) -> Mapping[str, List[str]]:
    """Prepare the compact cache with context features and benzene targets."""

    if cfg.window <= 0:
        raise ValueError("window must be a positive integer")
    if cfg.horizon < 0:
        raise ValueError("horizon must be non-negative")

    keep_time_meta = cfg.keep_time_meta.lower()
    if keep_time_meta not in {"full", "end", "none"}:
        raise ValueError("keep_time_meta must be one of {'full', 'end', 'none'}")

    feature_cols = _resolve_feature_columns(cfg.feature_columns, cfg.target_column)

    raw_root = cfg.raw_data_dir or "./uci_air_quality_raw"
    raw_path = download_uci_air_quality_dataset(raw_root)
    raw_df = load_raw_uci_air_quality(raw_path)
    clean_df = clean_uci_air_quality(
        raw_df,
        freq=cfg.freq,
        feature_columns=feature_cols,
        target_column=cfg.target_column,
    )

    data_dir = Path(cfg.data_dir).expanduser().resolve()
    paths = CachePaths.from_dir(data_dir)
    if cfg.overwrite and paths.cache_root.exists():
        shutil.rmtree(paths.cache_root)
    paths.ensure()

    assets = [ASSET_NAME]
    asset_to_id = {ASSET_NAME: 0}

    feature_cols = [cfg.target_column] + [c for c in clean_df.columns if c != cfg.target_column]

    panel = clean_df[feature_cols]
    total_rows = panel.shape[0]
    if total_rows < cfg.window + cfg.horizon:
        raise ValueError("Insufficient history for the requested window/horizon configuration.")

    features = panel.to_numpy(dtype=np.float32, copy=True)
    targets = panel[cfg.target_column].to_numpy(dtype=np.float32, copy=True)
    times = panel.index.to_numpy(dtype="datetime64[ns]")

    np.save(paths.features / "0.npy", features.astype(np.float16, copy=False))
    np.save(paths.targets / "0.npy", targets.astype(np.float16, copy=False))
    np.save(paths.times / "0.npy", times)

    norm_acc = _NormStatsAccumulator(
        num_assets=len(assets),
        feature_dim=len(feature_cols),
        per_asset=cfg.normalize_per_asset,
    )
    norm_acc.update(0, features, targets)

    max_start = total_rows - (cfg.window + cfg.horizon) + 1
    if max_start <= 0:
        raise RuntimeError("No valid context windows available for the requested configuration.")

    starts = np.arange(0, max_start, dtype=np.int32)
    window_end_times = times[starts + cfg.window - 1].astype("datetime64[ns]")
    global_pairs = np.stack([np.zeros_like(starts), starts], axis=1).astype(np.int32)
    np.save(paths.windows / "global_pairs.npy", global_pairs)
    np.save(paths.windows / "end_times.npy", window_end_times)

    norm_stats = norm_acc.finalize(assets)

    dataset_start = times[0].astype("datetime64[s]") if times.size else None
    dataset_end = times[-1].astype("datetime64[s]") if times.size else None

    meta = {
        "dataset": "uci_air_quality",
        "format": "indexcache_v1",
        "assets": assets,
        "asset2id": asset_to_id,
        "feature_cols": feature_cols,
        "target_col": cfg.target_column,
        "window": int(cfg.window),
        "horizon": int(cfg.horizon),
        "keep_time_meta": keep_time_meta,
        "normalize_per_ticker": cfg.normalize_per_asset,
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


def load_uci_dataloaders_with_ratio_split(
    data_dir: PathLike,
    **loader_kwargs,
):
    """Wrapper around the financial ratio-split loader for the UCI dataset."""

    return _load_fin_ratio_split(data_dir=data_dir, **loader_kwargs)


def _validate_uci_cache(paths: CachePaths) -> Dict[str, object]:
    """Read and validate the cache metadata, returning the parsed JSON."""

    if not paths.meta.exists():
        raise FileNotFoundError(
            f"Cache meta file not found at '{paths.meta}'. Did you run prepare_uci_air_cache()?"
        )

    with paths.meta.open("r") as f:
        meta: Dict[str, object] = json.load(f)

    dataset = meta.get("dataset")
    if dataset not in {"uci_air_quality", "uci_air_dataset"}:
        raise ValueError(
            f"The cache located at '{paths.cache_root}' does not correspond to the UCI Air Quality dataset."
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
    """Build train/val/test loaders for the prepared UCI Air Quality cache."""

    paths = CachePaths.from_dir(data_dir)
    meta = _validate_uci_cache(paths)
    base_window = int(meta.get("window", K))
    base_horizon = int(meta.get("horizon", H))
    if K > base_window or H > base_horizon:
        raise ValueError(
            "Requested (window, horizon) exceed the cached configuration. "
            "Re-run prepare_uci_air_cache with larger values first."
        )

    if reindex:
        _rebuild_window_index_only(
            data_dir,
            window=K,
            horizon=H,
            update_meta=False,
            backup_old=False,
        )

    train_dl, val_dl, test_dl, lengths = load_uci_dataloaders_with_ratio_split(
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
    "UCIAirCacheConfig",
    "download_uci_air_quality_dataset",
    "load_raw_uci_air_quality",
    "clean_uci_air_quality",
    "prepare_uci_air_cache",
    "load_uci_dataloaders_with_ratio_split",
    "run_experiment",
]

