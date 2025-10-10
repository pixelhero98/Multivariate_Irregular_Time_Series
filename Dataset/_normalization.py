"""Utilities for accumulating normalization statistics across datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class _PerAssetState:
    mean_x: Optional[np.ndarray]
    std_x: Optional[np.ndarray]
    mean_y: Optional[float]
    std_y: Optional[float]


class NormalizationStatsAccumulator:
    """Accumulate per-feature normalization statistics.

    The accumulator mirrors the behaviour previously duplicated across several
    dataset helpers.  It supports both per-asset and global normalization
    strategies and outputs JSON-friendly dictionaries ready to be serialized in
    ``norm_stats.json``.
    """

    def __init__(self, num_assets: int, feature_dim: int, *, per_asset: bool) -> None:
        if num_assets <= 0:
            raise ValueError("num_assets must be a positive integer")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")

        self._per_asset = bool(per_asset)
        self._feature_dim = int(feature_dim)

        if self._per_asset:
            self._state: List[_PerAssetState] = [
                _PerAssetState(None, None, None, None) for _ in range(int(num_assets))
            ]
        else:
            self._count = 0
            self._sum_x = np.zeros(self._feature_dim, dtype=np.float64)
            self._sumsq_x = np.zeros(self._feature_dim, dtype=np.float64)
            self._sum_y = 0.0
            self._sumsq_y = 0.0
            self._count_y = 0

    # ------------------------------------------------------------------
    # Update & finalization helpers

    def update(self, asset_id: int, features: np.ndarray, targets: np.ndarray) -> None:
        """Update statistics with a new mini-batch for ``asset_id``."""

        features = np.asarray(features, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)

        if features.ndim != 2 or features.shape[1] != self._feature_dim:
            raise ValueError(
                "features must be a 2D array with shape (N, feature_dim); "
                f"received {features.shape}"
            )
        if targets.ndim != 1:
            targets = targets.reshape(-1)
        if targets.shape[0] != features.shape[0]:
            raise ValueError("targets must align with features along axis 0")

        if self._per_asset:
            if not (0 <= asset_id < len(self._state)):
                raise IndexError("asset_id is out of range for the accumulator state")

            state = self._state[asset_id]
            f64 = features.astype(np.float64, copy=False)
            t64 = targets.astype(np.float64, copy=False)

            mean_x = f64.mean(axis=0)
            std_x = f64.std(axis=0)
            std_x = np.where(std_x == 0.0, 1.0, std_x)
            mean_y = float(t64.mean()) if t64.size else 0.0
            std_y = float(t64.std()) if t64.size else 1.0
            std_y = 1.0 if std_y == 0.0 else std_y

            state.mean_x = mean_x.reshape(1, 1, -1).astype(np.float32)
            state.std_x = std_x.reshape(1, 1, -1).astype(np.float32)
            state.mean_y = mean_y
            state.std_y = std_y
            return

        # Global statistics mode
        f64 = features.astype(np.float64, copy=False)
        t64 = targets.astype(np.float64, copy=False)
        self._count += f64.shape[0]
        self._sum_x += f64.sum(axis=0)
        self._sumsq_x += np.square(f64).sum(axis=0)
        self._sum_y += float(t64.sum())
        self._sumsq_y += float(np.square(t64).sum())
        self._count_y += t64.shape[0]

    # ------------------------------------------------------------------

    def finalize(self, assets: Sequence[str]) -> dict:
        """Return the JSON-serialisable statistics dictionary."""

        assets_list = list(assets)
        if len(assets_list) == 0:
            raise ValueError("assets must contain at least one entry")

        if self._per_asset:
            if len(assets_list) != len(self._state):
                raise ValueError("Number of assets does not match the accumulator state")

            if not all(s.mean_x is not None and s.std_x is not None for s in self._state):
                raise RuntimeError("Normalization statistics missing for some assets.")

            return {
                "per_ticker": True,
                "assets": assets_list,
                "mean_x": [np.asarray(s.mean_x).tolist() for s in self._state],
                "std_x": [np.asarray(s.std_x).tolist() for s in self._state],
                "mean_y": [float(s.mean_y) if s.mean_y is not None else 0.0 for s in self._state],
                "std_y": [float(s.std_y) if s.std_y is not None else 1.0 for s in self._state],
            }

        if getattr(self, "_count", 0) == 0:
            raise RuntimeError("Unable to compute normalization statistics (no samples).")

        mean_x = (self._sum_x / self._count).astype(np.float32)
        var_x = (self._sumsq_x / self._count) - np.square(mean_x.astype(np.float64))
        var_x = np.maximum(var_x, 1e-12)
        std_x = np.sqrt(var_x).astype(np.float32)
        std_x[std_x == 0.0] = 1.0

        if getattr(self, "_count_y", 0) > 0:
            mean_y = float(self._sum_y / self._count_y)
            var_y = max((self._sumsq_y / self._count_y) - (mean_y**2), 1e-12)
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


__all__ = ["NormalizationStatsAccumulator"]

