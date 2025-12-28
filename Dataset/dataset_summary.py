"""Summarize prepared dataset caches.

This script inspects the compact cache produced by the dataset utilities in
this repository and reports:

- Number of input channels (feature columns)
- Number of entities
- Train/validation/test steps (split using the ratio-based logic)
- Coverage statistics computed per day across the cached windows

Example:
    python Dataset/dataset_summary.py --data-dir ./data --coverage 0.85
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

# Provide a clear dependency hint without wrapping imports in try/except.
if importlib.util.find_spec("numpy") is None:
    raise SystemExit("numpy is required to run this script. Please install numpy.")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Dataset.fin_dataset import CachePaths, _normalize_to_day


@dataclass(frozen=True)
class CoverageSummary:
    days: int
    min: float
    mean: float
    median: float
    max: float


def _load_meta(paths: CachePaths) -> Dict[str, object]:
    if not paths.meta.exists():
        raise FileNotFoundError(
            f"Cache metadata not found at '{paths.meta}'. Did you prepare the dataset cache?"
        )
    with paths.meta.open("r") as f:
        return json.load(f)


def _summarize_coverage(per_day: np.ndarray) -> CoverageSummary:
    if per_day.size == 0:
        return CoverageSummary(days=0, min=0.0, mean=0.0, median=0.0, max=0.0)
    return CoverageSummary(
        days=int(per_day.size),
        min=float(per_day.min()),
        mean=float(per_day.mean()),
        median=float(np.median(per_day)),
        max=float(per_day.max()),
    )


def _compute_day_coverage(
    pairs: np.ndarray, end_times: np.ndarray, num_assets: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coverage per day along with inverse day indices and unique days."""
    if pairs.size == 0 or end_times.size == 0:
        return np.array([]), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    day_keys = _normalize_to_day(end_times)
    unique_days, inverse = np.unique(day_keys, return_inverse=True)

    day_asset = np.stack(
        [inverse.astype(np.int64), pairs[:, 0].astype(np.int64)], axis=1
    )
    day_asset_unique = np.unique(day_asset, axis=0)
    counts = np.bincount(day_asset_unique[:, 0], minlength=len(unique_days))
    coverage = counts / float(max(1, num_assets))
    return coverage, inverse, unique_days


def _split_counts(n: int, tr: float, vr: float, te: float) -> Tuple[int, int, int]:
    s = float(tr + vr + te)
    trn = int(np.floor(n * (tr / s))) if s > 0 else 0
    van = int(np.floor(n * (vr / s))) if s > 0 else 0
    ten = n - trn - van
    if n >= 3:
        if trn == 0:
            trn, ten = 1, ten - 1
        if van == 0 and n - trn >= 2:
            van, ten = 1, ten - 1
        if ten == 0:
            ten = 1
    return trn, van, ten


def _apply_split(
    pairs: np.ndarray,
    end_times: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    per_asset: bool,
) -> Tuple[int, int, int]:
    """Replicate the ratio split logic from the dataloader helper."""
    if pairs.size == 0:
        return 0, 0, 0

    aids = pairs[:, 0].astype(np.int32)
    if per_asset:
        order = np.lexsort(
            (end_times.astype("datetime64[ns]").astype(np.int64), aids)
        )
    else:
        order = np.argsort(end_times.astype("datetime64[ns]").astype(np.int64))
    pairs = pairs[order]
    end_times = end_times[order]
    aids = pairs[:, 0].astype(np.int32)

    assign = np.empty(pairs.shape[0], dtype=np.uint8)
    if per_asset:
        for aid in np.unique(aids):
            idx = np.nonzero(aids == aid)[0]
            na = idx.size
            trn, van, ten = _split_counts(na, train_ratio, val_ratio, test_ratio)
            assign[idx[:trn]] = 0
            assign[idx[trn : trn + van]] = 1
            assign[idx[trn + van :]] = 2
    else:
        n = pairs.shape[0]
        trn, van, ten = _split_counts(n, train_ratio, val_ratio, test_ratio)
        assign[:trn] = 0
        assign[trn : trn + van] = 1
        assign[trn + van :] = 2

    return int((assign == 0).sum()), int((assign == 1).sum()), int((assign == 2).sum())


def summarize_dataset(
    data_dir: Path,
    coverage_threshold: float,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    per_asset: bool,
) -> None:
    paths = CachePaths.from_dir(data_dir)
    meta = _load_meta(paths)
    assets = meta.get("assets", [])
    feature_cols = meta.get("feature_cols", [])
    window = int(meta.get("window", meta.get("max_window", 0)))
    horizon = int(meta.get("horizon", meta.get("max_horizon", 0)))

    pairs = np.load(paths.windows / "global_pairs.npy")
    end_times = np.load(paths.windows / "end_times.npy")

    base_cov, inverse_days, unique_days = _compute_day_coverage(
        pairs, end_times, len(assets)
    )
    base_cov_summary = _summarize_coverage(base_cov)

    kept_pairs = pairs
    kept_end_times = end_times
    filtered_cov_summary = base_cov_summary
    if coverage_threshold > 0.0 and base_cov.size > 0:
        keep_days = base_cov >= coverage_threshold
        keep_mask = keep_days[inverse_days]
        kept_pairs = pairs[keep_mask]
        kept_end_times = end_times[keep_mask]
        filtered_cov, *_ = _compute_day_coverage(
            kept_pairs, kept_end_times, len(assets)
        )
        filtered_cov_summary = _summarize_coverage(filtered_cov)

    tr_steps, va_steps, te_steps = _apply_split(
        kept_pairs,
        kept_end_times,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        per_asset=per_asset,
    )

    print(f"Data directory : {paths.cache_root}")
    print(f"Dataset        : {meta.get('dataset', 'unknown')}")
    print(f"Entities       : {len(assets)}")
    print(f"Input channels : {len(feature_cols)}")
    print(f"Window/Horizon : {window}/{horizon}")
    print()
    print("Coverage per day (before filtering):")
    print(
        f"  days={base_cov_summary.days} "
        f"min={base_cov_summary.min:.3f} "
        f"mean={base_cov_summary.mean:.3f} "
        f"median={base_cov_summary.median:.3f} "
        f"max={base_cov_summary.max:.3f}"
    )
    if coverage_threshold > 0.0:
        kept_days = filtered_cov_summary.days
        print(
            f"Applied coverage >= {coverage_threshold:.2f}: kept {kept_days} / {base_cov_summary.days} days"
        )
        print(
            f"  min={filtered_cov_summary.min:.3f} "
            f"mean={filtered_cov_summary.mean:.3f} "
            f"median={filtered_cov_summary.median:.3f} "
            f"max={filtered_cov_summary.max:.3f}"
        )
    print()
    print("Steps by split (ratio-based):")
    print(f"  train={tr_steps}  val={va_steps}  test={te_steps}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize prepared dataset caches.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the dataset directory containing cache_ratio_index.",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.0,
        help="Minimum per-day coverage required (0.0 disables filtering).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train ratio used when computing step counts.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio used when computing step counts.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test ratio used when computing step counts.",
    )
    parser.add_argument(
        "--per-asset",
        action="store_true",
        help="Split chronologically within each asset (matches loader default).",
    )
    parser.add_argument(
        "--global-order",
        dest="per_asset",
        action="store_false",
        help="Split by global chronological order instead of per-asset.",
    )
    parser.set_defaults(per_asset=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summarize_dataset(
        data_dir=args.data_dir,
        coverage_threshold=args.coverage,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        per_asset=args.per_asset,
    )
