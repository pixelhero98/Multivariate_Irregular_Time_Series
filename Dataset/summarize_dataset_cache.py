"""Summarize cached datasets produced by :mod:`Dataset.fin_dataset`.

The script prints basic metadata such as the feature dimension (input channels),
entity count, number of indexed windows assigned to each split, and panel
coverage statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from Dataset.fin_dataset import CachePaths, load_dataloaders_with_ratio_split


def _apply_coverage_filter(
    pairs: np.ndarray, end_times: np.ndarray, n_entities: int, coverage: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Mirror the coverage-per-window filtering used by the dataloader."""

    if coverage <= 0.0:
        return pairs, end_times

    days = end_times.astype("datetime64[D]")
    uniq_days, inv = np.unique(days, return_inverse=True)

    day_asset = np.stack(
        [inv.astype(np.int64), pairs[:, 0].astype(np.int64)], axis=1
    )
    unique_day_asset = np.unique(day_asset, axis=0)
    counts = np.bincount(unique_day_asset[:, 0], minlength=len(uniq_days))

    min_real = int(np.ceil(coverage * n_entities))
    keep_days = counts >= max(1, min_real)
    keep_mask = keep_days[inv]

    return pairs[keep_mask], end_times[keep_mask]


def _coverage_stats(
    pairs: np.ndarray, end_times: np.ndarray, n_entities: int
) -> Dict[str, float]:
    if pairs.size == 0 or n_entities <= 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}

    days = end_times.astype("datetime64[D]")
    uniq_days = np.unique(days)
    coverage = []
    for d in uniq_days:
        mask = days == d
        assets = np.unique(pairs[mask][:, 0]) if mask.any() else []
        coverage.append(len(assets) / float(n_entities))

    cov = np.asarray(coverage, dtype=np.float32)
    return {
        "mean": float(cov.mean()),
        "min": float(cov.min(initial=0.0)),
        "max": float(cov.max(initial=0.0)),
    }


def summarize_cache(
    data_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    per_asset: bool,
    coverage_per_window: float,
) -> None:
    paths = CachePaths.from_dir(data_dir)
    if not paths.meta.exists():
        raise FileNotFoundError(
            f"No cache found at {paths.meta}. Prepare the dataset before summarizing."
        )

    with paths.meta.open("r") as f:
        meta = json.load(f)

    feature_dim = len(meta.get("feature_cols", []))
    assets = meta.get("assets", [])
    n_entities = len(assets)

    global_pairs = np.load(paths.windows / "global_pairs.npy")
    end_times = np.load(paths.windows / "end_times.npy")
    filtered_pairs, filtered_end_times = _apply_coverage_filter(
        global_pairs, end_times, n_entities, coverage_per_window
    )
    cov_stats = _coverage_stats(filtered_pairs, filtered_end_times, n_entities)

    _, _, _, lengths = load_dataloaders_with_ratio_split(
        data_dir=str(data_dir),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        batch_size=1,
        regression=True,
        per_asset=per_asset,
        norm_scope="cache",
        shuffle_train=False,
        num_workers=0,
        pin_memory=False,
        coverage_per_window=coverage_per_window,
        date_batching=False,
    )

    tr_steps, va_steps, te_steps = lengths

    print(f"Cache root: {paths.cache_root}")
    print(f"Entities: {n_entities}")
    print(f"Input channels: {feature_dim}")
    print("Window counts:")
    print(f"  total indexed windows: {filtered_pairs.shape[0]}")
    print(f"  train windows: {tr_steps}")
    print(f"  val windows:   {va_steps}")
    print(f"  test windows:  {te_steps}")
    print("Coverage (unique assets per day / total entities):")
    print(
        f"  mean={cov_stats['mean']:.3f}, min={cov_stats['min']:.3f}, max={cov_stats['max']:.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Directory containing cache_ratio_index and metadata.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.55)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.40)
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Treat all windows jointly instead of splitting per entity.",
    )
    parser.add_argument(
        "--coverage-per-window",
        type=float,
        default=0.0,
        help="Optional coverage filter applied before counting windows (0-1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize_cache(
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        per_asset=not args.aggregate,
        coverage_per_window=args.coverage_per_window,
    )


if __name__ == "__main__":
    main()
