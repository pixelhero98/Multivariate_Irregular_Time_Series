"""Utility helpers for long-range time-series datasets.

This module provides a light-weight dataset and dataloader builder that
covers common academic benchmarks such as ETT, Solar-Energy, Electricity
Load, and PEMS-BAY Traffic.  The code standardises the CSV loading,
normalisation, and sliding-window sampling so that the training scripts
can focus on the modelling logic.

Usage
-----
```
from Data_Prep.long_range_datasets import create_dataloaders

train_dl, val_dl, test_dl, info = create_dataloaders(
    dataset="ettm1",
    data_root="./data",
    window=96,
    horizon=24,
    batch_size=32,
)
```

The returned ``info`` dictionary contains the actual split sizes, the
per-feature normalisation statistics, and a short textual description of
the dataset that can be used for logging.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class DatasetSpec:
    """Registry entry describing a canonical long-range dataset."""

    file: str
    description: str
    window: int
    horizon: int
    ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2)
    stride: int = 1
    batch_size: int = 32
    shuffle_train: bool = True
    num_workers: int = 0
    drop_last: bool = False
    scaling: str = "standard"
    vae_channels: int = 24
    model_width: int = 256


def _standard_dataset_specs() -> Dict[str, DatasetSpec]:
    """Build a registry with sensible defaults for popular datasets."""

    day = 24
    week = 7 * day
    quarter = 3 * 30 * day

    return {
        # ETT (Electricity Transformer Temperature)
        "etth1": DatasetSpec(
            file=os.path.join("ETT", "ETTh1.csv"),
            description="ETT hourly, subset 1",
            window=96,
            horizon=24,
            batch_size=32,
        ),
        "etth2": DatasetSpec(
            file=os.path.join("ETT", "ETTh2.csv"),
            description="ETT hourly, subset 2",
            window=96,
            horizon=24,
            batch_size=32,
        ),
        "ettm1": DatasetSpec(
            file=os.path.join("ETT", "ETTm1.csv"),
            description="ETT 15 min, subset 1",
            window=96,
            horizon=24,
            batch_size=64,
        ),
        "ettm2": DatasetSpec(
            file=os.path.join("ETT", "ETTm2.csv"),
            description="ETT 15 min, subset 2",
            window=96,
            horizon=24,
            batch_size=64,
        ),
        # Other multivariate long-range datasets
        "solar": DatasetSpec(
            file=os.path.join("solar", "solar_AL.csv"),
            description="Solar Energy production",
            window=day * 7,
            horizon=day,
            batch_size=32,
            stride=1,
            vae_channels=32,
        ),
        "electricity": DatasetSpec(
            file=os.path.join("electricity", "electricity.csv"),
            description="Electricity consumption",
            window=week,
            horizon=day,
            batch_size=32,
            stride=1,
        ),
        "traffic": DatasetSpec(
            file=os.path.join("traffic", "PEMS-BAY.csv"),
            description="Traffic (PEMS-BAY)",
            window=week * 2,
            horizon=12,
            batch_size=32,
            stride=1,
        ),
    }


DATASET_REGISTRY: Dict[str, DatasetSpec] = _standard_dataset_specs()


class LongRangeWindowDataset(Dataset):
    """Sliding-window dataset for long-range forecasting benchmarks."""

    def __init__(
        self,
        data: torch.Tensor,
        *,
        window: int,
        horizon: int,
        start_indices: Sequence[int],
    ) -> None:
        if window <= 0 or horizon <= 0:
            raise ValueError("window and horizon must be positive integers")
        self.data = data
        self.window = int(window)
        self.horizon = int(horizon)
        self.start_indices = list(int(s) for s in start_indices)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.start_indices)

    def __getitem__(self, idx: int):
        start = self.start_indices[idx]
        end_ctx = start + self.window
        end_tgt = end_ctx + self.horizon

        ctx = self.data[start:end_ctx]  # [K, D]
        tgt = self.data[end_ctx:end_tgt]  # [H, D]

        if ctx.shape[0] != self.window or tgt.shape[0] != self.horizon:
            raise IndexError("Sample outside dataset bounds. Check split indices.")

        # Treat each dimension as an "entity" with a single feature channel.
        # [K, D] -> [D, K, 1]
        ctx_series = ctx.transpose(0, 1).unsqueeze(-1).contiguous()
        tgt_series = tgt.transpose(0, 1).contiguous()

        mask = torch.ones(ctx_series.shape[0], dtype=torch.bool)
        meta = {"entity_mask": mask}
        return (ctx_series, ctx_series.clone()), tgt_series, meta


def _load_csv_values(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    # Keep numeric columns only (timestamps are typically non-numeric).
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError(f"No numeric columns found in dataset: {path}")

    filled = numeric_df.fillna(method="ffill").fillna(method="bfill")
    values = filled.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values


def _compute_split_indices(
    total_len: int, window: int, horizon: int, ratios: Sequence[float], stride: int
) -> Tuple[List[int], List[int], List[int]]:
    if total_len <= window + horizon:
        raise ValueError(
            "Dataset is too short for the requested window/horizon combination."
        )

    seq_len = window + horizon
    available = total_len - seq_len + 1
    if available <= 0:
        raise ValueError("No valid start positions after accounting for window/horizon")

    ratios = list(ratios)
    if len(ratios) != 3:
        raise ValueError("ratios must be a sequence of three floats (train/val/test)")

    if any(r < 0 for r in ratios):
        raise ValueError("ratios must be non-negative")

    total = sum(ratios)
    if total <= 0:
        raise ValueError("Sum of ratios must be positive")

    ratios = [r / total for r in ratios]
    train_cut = int(math.floor(ratios[0] * available))
    val_cut = int(math.floor((ratios[0] + ratios[1]) * available))

    train_idx = list(range(0, train_cut, stride))
    val_idx = list(range(train_cut, val_cut, stride))
    test_idx = list(range(val_cut, available, stride))

    return train_idx, val_idx, test_idx


def _build_dataloader(
    data: torch.Tensor,
    *,
    window: int,
    horizon: int,
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool,
) -> DataLoader:
    dataset = LongRangeWindowDataset(
        data, window=window, horizon=horizon, start_indices=indices
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


def create_dataloaders(
    *,
    dataset: str,
    data_root: str,
    window: int,
    horizon: int,
    batch_size: int,
    ratios: Sequence[float] = (0.7, 0.1, 0.2),
    stride: int = 1,
    shuffle_train: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    scaling: str = "standard",
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, object]]:
    """Create train/validation/test dataloaders for long-range datasets."""

    dataset_key = dataset.lower()
    spec = DATASET_REGISTRY.get(dataset_key)

    data_path = spec.file if spec else dataset
    if not os.path.isabs(data_path):
        data_path = os.path.join(data_root, data_path)

    values = _load_csv_values(data_path)
    total_len, num_channels = values.shape

    train_idx, val_idx, test_idx = _compute_split_indices(
        total_len, window, horizon, ratios, max(1, int(stride))
    )

    if not train_idx:
        raise ValueError("Training split produced zero samples; adjust ratios or window.")
    if not val_idx:
        raise ValueError("Validation split produced zero samples; adjust ratios or window.")
    if not test_idx:
        raise ValueError("Test split produced zero samples; adjust ratios or window.")

    # Normalisation statistics from the training portion only
    train_end = train_idx[-1] + window + horizon
    train_slice = values[:train_end]
    mean = train_slice.mean(axis=0)
    std = train_slice.std(axis=0)
    std[std == 0.0] = 1.0

    if scaling == "standard":
        norm_values = (values - mean) / std
    elif scaling in (None, "none"):
        norm_values = values
    else:
        raise ValueError(f"Unknown scaling mode: {scaling}")

    tensor_data = torch.from_numpy(norm_values.astype(np.float32))

    train_dl = _build_dataloader(
        tensor_data,
        window=window,
        horizon=horizon,
        indices=train_idx,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    val_dl = _build_dataloader(
        tensor_data,
        window=window,
        horizon=horizon,
        indices=val_idx,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_dl = _build_dataloader(
        tensor_data,
        window=window,
        horizon=horizon,
        indices=test_idx,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    info = {
        "sizes": (len(train_idx), len(val_idx), len(test_idx)),
        "num_channels": num_channels,
        "window": window,
        "horizon": horizon,
        "scaler": {"mean": mean, "std": std},
        "dataset": dataset_key,
        "path": data_path,
        "description": spec.description if spec else dataset,
    }

    return train_dl, val_dl, test_dl, info


__all__ = ["DATASET_REGISTRY", "DatasetSpec", "create_dataloaders"]
