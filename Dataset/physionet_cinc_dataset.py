"""PhysioNet 2012 (CinC Challenge) dataset utilities.

This module mirrors the public API provided by :mod:`Dataset.fin_dataset` and
the auxiliary air-quality datasets so that the existing training pipeline can
consume intensive-care unit (ICU) time-series data without additional glue
code.  The helpers create the same compact cache layout used throughout the
repository: per-entity feature matrices stored as ``float16`` together with a
global window index.  Each entity corresponds to a single patient admission and
the target series ``Y`` represents future values of a chosen vital sign stored
under the unified column name ``future_vital``.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING, Union

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

DATA_URLS = {
    "set-a": "https://physionet.org/static/published-projects/challenge-2012/"
    "physionet-challenge-2012-set-a-1.0.0.zip",
    "set-b": "https://physionet.org/static/published-projects/challenge-2012/"
    "physionet-challenge-2012-set-b-1.0.0.zip",
    "set-c": "https://physionet.org/static/published-projects/challenge-2012/"
    "physionet-challenge-2012-set-c-1.0.0.zip",
}

OUTCOME_FILENAMES = {
    "set-a": "Outcomes-a.txt",
    "set-b": "Outcomes-b.txt",
    "set-c": "Outcomes-c.txt",
}

TARGET_COLUMN = "future_vital"

# PhysioNet measurements are recorded every hour for the challenge; align to that by default.
DEFAULT_FREQ = "1H"

# Keep the cache bounded so ``run_experiment`` can rebuild smaller indices cheaply.
MAX_WINDOW = 48
MAX_HORIZON = 24


@dataclass
class PhysioNetCacheConfig:
    """Configuration used when building the compact PhysioNet cache."""

    window: int = MAX_WINDOW
    horizon: int = MAX_HORIZON
    data_dir: PathLike = "./physionet_cinc_cache"
    raw_data_dir: Optional[PathLike] = "./physionet_cinc_raw"
    subset: str = "set-a"
    target_parameter: str = "HR"
    parameters: Optional[Sequence[str]] = None
    normalize_per_patient: bool = True
    clamp_sigma: float = 5.0
    keep_time_meta: str = "end"  # "full" | "end" | "none"
    freq: str = DEFAULT_FREQ
    min_minutes: int = 60
    min_coverage: float = 0.6
    max_patients: Optional[int] = None
    include_outcomes: bool = True
    overwrite: bool = False


# ---------------------------------------------------------------------------
# Downloading & ingestion utilities

def download_physionet_cinc_dataset(
    dest_path: PathLike = "./physionet_cinc_raw",
    subset: str = "set-a",
) -> Path:
    """Download and extract a PhysioNet CinC subset if not already available."""

    subset_key = subset.lower()
    if subset_key not in DATA_URLS:
        raise ValueError(f"Unknown subset '{subset}'. Expected one of {sorted(DATA_URLS)}.")

    dest = Path(dest_path).expanduser().resolve()
    subset_dir = dest / subset_key
    outcomes_file = subset_dir / OUTCOME_FILENAMES[subset_key]
    if outcomes_file.exists() and any(subset_dir.glob("*.txt")):
        return subset_dir

    dest.mkdir(parents=True, exist_ok=True)

    url = DATA_URLS[subset_key]
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(dest)

    if not outcomes_file.exists():
        raise FileNotFoundError(
            f"Outcome file '{OUTCOME_FILENAMES[subset_key]}' not found after extracting '{url}'."
        )

    return subset_dir


def _read_patient_file(path: Path) -> Tuple[str, pd.DataFrame]:
    """Read a single patient text file into a cleaned time-indexed dataframe."""

    df = pd.read_csv(path)
    expected_cols = {"RecordID", "Time", "Parameter", "Value"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"File '{path}' is missing required columns: {sorted(missing)}")

    record_id = str(df["RecordID"].iloc[0])

    # Convert the ``HH:MM`` strings to elapsed minutes since ICU admission.
    time_delta = pd.to_timedelta(df["Time"].astype(str) + ":00")
    df["minutes"] = (time_delta / pd.Timedelta(minutes=1)).astype(int)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    pivot = df.pivot_table(index="minutes", columns="Parameter", values="Value", aggfunc="last")
    pivot = pivot.sort_index()

    # Treat minutes as offsets from a fixed origin to obtain a DatetimeIndex.
    origin = pd.Timestamp("2000-01-01")
    pivot.index = origin + pd.to_timedelta(pivot.index, unit="m")
    pivot = pivot.ffill()
    pivot = pivot.dropna(how="all")

    return record_id, pivot


def load_physionet_patient_panels(
    data_dir: PathLike,
    *,
    parameters: Optional[Sequence[str]] = None,
    freq: str = "1H",
    min_minutes: int = 60,
    min_coverage: float = 0.6,
    max_patients: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Load and align per-patient panels from raw PhysioNet text files."""

    root = Path(data_dir)
    txt_files = sorted(p for p in root.glob("*.txt") if not p.name.lower().startswith("outcomes"))
    if max_patients is not None:
        txt_files = txt_files[: int(max_patients)]

    panels: Dict[str, pd.DataFrame] = {}
    keep_params = set(str(p) for p in parameters) if parameters is not None else None

    for file_path in txt_files:
        record_id, pivot = _read_patient_file(file_path)

        if keep_params is not None:
            cols = [c for c in pivot.columns if c in keep_params]
            pivot = pivot[cols]

        if pivot.empty or pivot.shape[0] < 2:
            continue

        # Regularize to the requested sampling frequency.
        if freq:
            start, end = pivot.index.min(), pivot.index.max()
            range_minutes = (end - start) / pd.Timedelta(minutes=1)
            if range_minutes < max(min_minutes, 1):
                continue
            full_index = pd.date_range(start, end, freq=freq)
            pivot = pivot.reindex(full_index)
        pivot = pivot.ffill()
        pivot = pivot.dropna(thresh=int(np.ceil(min_coverage * pivot.shape[1])))

        if pivot.empty:
            continue

        panels[record_id] = pivot.astype(np.float32)

    if not panels:
        raise RuntimeError(
            "No valid patient panels were produced. Check the frequency/min_coverage configuration."
        )

    return panels


def load_physionet_outcomes(path: PathLike) -> pd.DataFrame:
    """Load static outcome information associated with each patient."""

    df = pd.read_csv(path)
    if "RecordID" not in df.columns:
        raise ValueError(f"Outcome file '{path}' is missing the 'RecordID' column.")
    df["RecordID"] = df["RecordID"].astype(str)
    df = df.set_index("RecordID")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


class _NormStatsAccumulator:
    """Accumulate normalization statistics for PhysioNet panels."""

    def __init__(self, num_assets: int, feature_dim: int, per_asset: bool) -> None:
        self.per_asset = bool(per_asset)
        self.num_assets = int(num_assets)
        self.feature_dim = int(feature_dim)

        if per_asset:
            self.sum_x = [np.zeros(feature_dim, dtype=np.float64) for _ in range(num_assets)]
            self.sumsq_x = [np.zeros(feature_dim, dtype=np.float64) for _ in range(num_assets)]
            self.count_x = [0 for _ in range(num_assets)]
            self.sum_y = [0.0 for _ in range(num_assets)]
            self.sumsq_y = [0.0 for _ in range(num_assets)]
            self.count_y = [0 for _ in range(num_assets)]
        else:
            self.sum_x = np.zeros(feature_dim, dtype=np.float64)
            self.sumsq_x = np.zeros(feature_dim, dtype=np.float64)
            self.count_x = 0
            self.sum_y = 0.0
            self.sumsq_y = 0.0
            self.count_y = 0

    def update(self, aid: int, features: np.ndarray, targets: np.ndarray) -> None:
        if self.per_asset:
            self.sum_x[aid] += features.sum(axis=0, dtype=np.float64)
            self.sumsq_x[aid] += (features.astype(np.float64) ** 2).sum(axis=0)
            self.count_x[aid] += features.shape[0]
            self.sum_y[aid] += float(targets.sum(dtype=np.float64))
            self.sumsq_y[aid] += float((targets.astype(np.float64) ** 2).sum())
            self.count_y[aid] += targets.shape[0]
        else:
            self.sum_x += features.sum(axis=0, dtype=np.float64)
            self.sumsq_x += (features.astype(np.float64) ** 2).sum(axis=0)
            self.count_x += features.shape[0]
            self.sum_y += float(targets.sum(dtype=np.float64))
            self.sumsq_y += float((targets.astype(np.float64) ** 2).sum())
            self.count_y += targets.shape[0]

    def finalize(self, assets: Sequence[str]) -> Dict[str, object]:
        def _safe_stats(sum_x, sumsq_x, count_x):
            if count_x <= 0:
                return np.zeros(self.feature_dim, dtype=np.float32), np.ones(self.feature_dim, dtype=np.float32)
            mean = (sum_x / count_x).astype(np.float32)
            var = (sumsq_x / count_x) - (mean.astype(np.float64) ** 2)
            var = np.maximum(var, 1e-12)
            std = np.sqrt(var).astype(np.float32)
            std[std == 0] = 1.0
            return mean, std

        if self.per_asset:
            mean_x, std_x, mean_y, std_y = [], [], [], []
            for aid in range(len(assets)):
                mx, sx = _safe_stats(self.sum_x[aid], self.sumsq_x[aid], self.count_x[aid])
                if self.count_y[aid] > 0:
                    my = float(self.sum_y[aid] / self.count_y[aid])
                    vy = max(self.sumsq_y[aid] / self.count_y[aid] - my ** 2, 1e-12)
                    sy = float(np.sqrt(vy))
                else:
                    my = 0.0
                    sy = 1.0
                mean_x.append(mx.reshape(1, 1, -1).tolist())
                std_x.append(sx.reshape(1, 1, -1).tolist())
                mean_y.append(my)
                std_y.append(sy if sy != 0 else 1.0)
            return {
                "per_ticker": True,
                "mean_x": mean_x,
                "std_x": std_x,
                "mean_y": mean_y,
                "std_y": std_y,
            }

        mx, sx = _safe_stats(self.sum_x, self.sumsq_x, self.count_x)
        if self.count_y > 0:
            my = float(self.sum_y / self.count_y)
            vy = max(self.sumsq_y / self.count_y - my ** 2, 1e-12)
            sy = float(np.sqrt(vy))
        else:
            my = 0.0
            sy = 1.0
        return {
            "per_ticker": False,
            "mean_x": mx.reshape(1, 1, -1).tolist(),
            "std_x": sx.reshape(1, 1, -1).tolist(),
            "mean_y": my,
            "std_y": sy if sy != 0 else 1.0,
        }


# ---------------------------------------------------------------------------
# Cache preparation

def _validate_window_and_horizon(window: int, horizon: int) -> Tuple[int, int]:
    """Clamp obvious configuration errors early and return normalized values."""

    if window <= 0:
        raise ValueError("window must be a positive integer")
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if window > MAX_WINDOW:
        raise ValueError(f"window={window} exceeds the supported maximum of {MAX_WINDOW}.")
    if horizon > MAX_HORIZON:
        raise ValueError(f"horizon={horizon} exceeds the supported maximum of {MAX_HORIZON}.")
    return int(window), int(horizon)


def _ensure_future_vital_target(panel: pd.DataFrame, target_parameter: str) -> pd.DataFrame:
    """Append the continuous target column expected by the downstream pipeline."""

    if target_parameter not in panel.columns:
        raise ValueError(f"Target parameter '{target_parameter}' missing for patient panel.")

    target_series = panel[target_parameter].ffill().bfill()
    if target_series.isna().any():
        raise ValueError(
            "Target parameter contains gaps even after forward/backward filling; "
            "cannot create a continuous future vital series."
        )

    panel = panel.drop(columns=[target_parameter])
    panel[TARGET_COLUMN] = target_series.astype(np.float32)
    return panel


def prepare_physionet_cinc_cache(cfg: PhysioNetCacheConfig) -> Mapping[str, object]:
    """Prepare the compact cache compatible with the financial pipeline."""

    window, horizon = _validate_window_and_horizon(cfg.window, cfg.horizon)

    keep_time_meta = cfg.keep_time_meta.lower()
    if keep_time_meta not in {"full", "end", "none"}:
        raise ValueError("keep_time_meta must be one of {'full', 'end', 'none'}")

    subset_dir = download_physionet_cinc_dataset(cfg.raw_data_dir or "./physionet_cinc_raw", cfg.subset)
    panels = load_physionet_patient_panels(
        subset_dir,
        parameters=cfg.parameters,
        freq=cfg.freq,
        min_minutes=cfg.min_minutes,
        min_coverage=cfg.min_coverage,
        max_patients=cfg.max_patients,
    )

    outcomes: Optional[pd.DataFrame] = None
    if cfg.include_outcomes:
        outcomes = load_physionet_outcomes(subset_dir / OUTCOME_FILENAMES[cfg.subset.lower()])

    assets = sorted(panels.keys())
    asset_to_id = {asset: idx for idx, asset in enumerate(assets)}

    feature_frames: Dict[str, pd.DataFrame] = {}
    for asset in assets:
        panel = _ensure_future_vital_target(panels[asset].copy(), cfg.target_parameter)

        if outcomes is not None and asset in outcomes.index:
            static_values = outcomes.loc[asset]
            for col, value in static_values.items():
                panel[f"OUTCOME_{col.upper()}"] = np.float32(value) if pd.notna(value) else np.nan
            panel = panel.ffill()  # ensure broadcast columns remain constant

        panel = panel.dropna(how="all")
        feature_frames[asset] = panel.astype(np.float32)

    if not feature_frames:
        raise RuntimeError("No patient panels remaining after applying target/outcome processing.")

    feature_cols = list(next(iter(feature_frames.values())).columns)
    if TARGET_COLUMN not in feature_cols:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from feature columns.")

    data_dir = Path(cfg.data_dir).expanduser().resolve()
    paths = CachePaths.from_dir(data_dir)
    if cfg.overwrite and paths.cache_root.exists():
        shutil.rmtree(paths.cache_root)
    paths.ensure()

    pairs: List[np.ndarray] = []
    end_times: List[np.ndarray] = []
    start_times: List[np.datetime64] = []
    stop_times: List[np.datetime64] = []

    norm_acc = _NormStatsAccumulator(
        num_assets=len(assets),
        feature_dim=len(feature_cols),
        per_asset=cfg.normalize_per_patient,
    )

    min_required = window + horizon
    for asset in assets:
        aid = asset_to_id[asset]
        panel = feature_frames[asset][feature_cols]
        if panel.shape[0] < min_required:
            raise ValueError(
                f"Patient '{asset}' has insufficient length ({panel.shape[0]} rows) "
                f"for window={window} and horizon={horizon}."
            )

        features = panel.to_numpy(dtype=np.float32, copy=True)
        targets = panel[TARGET_COLUMN].to_numpy(dtype=np.float32, copy=True)
        times = panel.index.to_numpy(dtype="datetime64[ns]")

        np.save(paths.features / f"{aid}.npy", features.astype(np.float16, copy=False))
        np.save(paths.targets / f"{aid}.npy", targets.astype(np.float16, copy=False))
        np.save(paths.times / f"{aid}.npy", times)

        norm_acc.update(aid, features, targets)

        max_start = panel.shape[0] - min_required + 1
        starts = np.arange(0, max_start, dtype=np.int32)
        window_end_times = times[starts + window - 1].astype("datetime64[ns]")
        pairs.append(np.stack([np.full_like(starts, aid), starts], axis=1))
        end_times.append(window_end_times)
        start_times.append(times[0])
        stop_times.append(times[-1])

    global_pairs = np.concatenate(pairs, axis=0).astype(np.int32)
    global_end_times = np.concatenate(end_times, axis=0).astype("datetime64[ns]")
    np.save(paths.windows / "global_pairs.npy", global_pairs)
    np.save(paths.windows / "end_times.npy", global_end_times)

    norm_stats = norm_acc.finalize(assets)

    dataset_start = min(start_times).astype("datetime64[s]") if start_times else None
    dataset_end = max(stop_times).astype("datetime64[s]") if stop_times else None

    meta = {
        "dataset": "physionet_cinc",
        "subset": cfg.subset,
        "window": window,
        "horizon": horizon,
        "max_window": int(MAX_WINDOW),
        "max_horizon": int(MAX_HORIZON),
        "assets": assets,
        "feature_cols": feature_cols,
        "target_column": TARGET_COLUMN,
        "target_parameter": cfg.target_parameter,
        "normalize_per_patient": bool(cfg.normalize_per_patient),
        "clamp_sigma": float(cfg.clamp_sigma),
        "keep_time_meta": keep_time_meta,
        "freq": cfg.freq or DEFAULT_FREQ,
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

def load_physionet_dataloaders_with_ratio_split(
    data_dir: PathLike,
    **loader_kwargs,
):
    """Wrapper around the financial ratio-split loader for PhysioNet datasets."""

    return _load_fin_ratio_split(data_dir=data_dir, **loader_kwargs)


def _validate_physionet_cache(paths: CachePaths) -> Dict[str, object]:
    if not paths.meta.exists():
        raise FileNotFoundError(
            f"Cache metadata not found at '{paths.meta}'. Did you run prepare_physionet_cinc_cache()?"
        )

    with paths.meta.open("r") as f:
        meta: Dict[str, object] = json.load(f)

    dataset = meta.get("dataset")
    if dataset not in {"physionet_cinc", "physionet_cinc_dataset"}:
        raise ValueError(
            f"The cache at '{paths.cache_root}' does not correspond to the PhysioNet CinC dataset."
        )

    return meta


def run_experiment(
    data_dir: PathLike,
    K: int,
    H: int,
    *,
    ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
    per_asset: bool = True,
    date_batching: bool = True,
    coverage: float = 0.8,
    dates_per_batch: int = 14,
    batch_size: int = 64,
    norm: str = "train_only",
    reindex: bool = True,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
):
    """Mirror :func:`Dataset.fin_dataset.run_experiment` for PhysioNet caches."""

    paths = CachePaths.from_dir(data_dir)
    meta = _validate_physionet_cache(paths)
    base_window = min(int(meta.get("window", K)), MAX_WINDOW)
    base_horizon = min(int(meta.get("horizon", H)), MAX_HORIZON)
    max_window = min(int(meta.get("max_window", MAX_WINDOW)), MAX_WINDOW)
    max_horizon = min(int(meta.get("max_horizon", MAX_HORIZON)), MAX_HORIZON)

    # Sanity check: old caches might predate the tighter maxima.
    base_window, base_horizon = _validate_window_and_horizon(base_window, base_horizon)

    if K > max_window or H > max_horizon:
        raise ValueError(
            "Requested (window, horizon) exceed the cached configuration. "
            "Re-run prepare_physionet_cinc_cache with larger values first."
        )

    if reindex and (K != base_window or H != base_horizon):
        _validate_window_and_horizon(K, H)
        _rebuild_window_index_only(
            data_dir,
            window=K,
            horizon=H,
            update_meta=True,
            backup_old=False,
        )
        base_window, base_horizon = K, H

    elif not reindex and (K > base_window or H > base_horizon):
        raise ValueError(
            "Requested (window, horizon) exceed the currently indexed configuration. "
            "Enable reindex=True to rebuild within the cached maximums."
        )

    train_dl, val_dl, test_dl, lengths = load_physionet_dataloaders_with_ratio_split(
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
    "PhysioNetCacheConfig",
    "download_physionet_cinc_dataset",
    "load_physionet_patient_panels",
    "load_physionet_outcomes",
    "prepare_physionet_cinc_cache",
    "load_physionet_dataloaders_with_ratio_split",
    "run_experiment",
]