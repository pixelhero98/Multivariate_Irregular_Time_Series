"""Summarize cached datasets built by the helpers in :mod:`Dataset`.

The script inspects the compact ``cache_ratio_index`` layout produced by
``prepare_*_cache`` utilities and reports:
* number of input channels (feature columns)
* number of entities
* train/val/test steps (windows) for a ratio split
* average entity coverage per distinct end date

Coverage is computed as the mean fraction of entities with at least one window
per unique end date after the optional coverage filter is applied, mirroring
``coverage_per_window`` in :func:`Dataset.fin_dataset.load_dataloaders_with_ratio_split`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from Dataset.fin_dataset import CachePaths, _normalize_to_day


@dataclass
class SplitSummary:
    steps: int
    unique_days: int
    coverage_mean: float
    coverage_min: float
    coverage_max: float


def _load_cache(data_dir: Path) -> Tuple[Dict[str, object], np.ndarray, np.ndarray]:
    """Load cache metadata and window indices."""

    paths = CachePaths.from_dir(data_dir)

    with paths.meta.open("r") as f:
        meta = json.load(f)

    pairs = np.load(paths.windows / "global_pairs.npy")
    end_times = np.load(paths.windows / "end_times.npy")
    return meta, pairs, end_times


def _apply_coverage_filter(
    pairs: np.ndarray, end_times: np.ndarray, coverage: float, num_assets: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter pairs/end_times using the same day-level rule as the loader."""

    if coverage <= 0.0:
        return pairs, end_times

    days = _normalize_to_day(end_times)
    uniq_days, inv = np.unique(days, return_inverse=True)

    # unique rows of (day_idx, asset_id) to count active entities per day
    day_asset = np.stack([inv.astype(np.int64), pairs[:, 0].astype(np.int64)], axis=1)
    uniq_da = np.unique(day_asset, axis=0)
    counts = np.bincount(uniq_da[:, 0], minlength=len(uniq_days))

    min_real = int(np.ceil(coverage * num_assets))
    keep_days = counts >= max(1, min_real)
    keep_mask = keep_days[inv]

    return pairs[keep_mask], end_times[keep_mask]


def _split_counts(n: int, tr: float, va: float, te: float) -> Tuple[int, int, int]:
    """Replica of the split logic in ``load_dataloaders_with_ratio_split``."""

    s = float(tr + va + te)
    trn = int(np.floor(n * (tr / s)))
    van = int(np.floor(n * (va / s)))
    ten = n - trn - van
    if n >= 3:
        if trn == 0:
            trn, ten = 1, ten - 1
        if van == 0 and n - trn >= 2:
            van, ten = 1, ten - 1
        if ten == 0:
            ten = 1
    return trn, van, ten


def _split_pairs(
    pairs: np.ndarray,
    end_times: np.ndarray,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    per_asset: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform the same deterministic ratio split as the loader."""

    aid = pairs[:, 0].astype(np.int32)
    if per_asset:
        order = np.lexsort((end_times.astype("datetime64[ns]").astype(np.int64), aid))
    else:
        order = np.argsort(end_times.astype("datetime64[ns]").astype(np.int64))
    pairs = pairs[order]
    end_times = end_times[order]

    assign = np.empty(pairs.shape[0], dtype=np.uint8)
    if per_asset:
        for a in np.unique(aid):
            idx = np.nonzero(aid == a)[0]
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

    tr_pairs = pairs[assign == 0]
    va_pairs = pairs[assign == 1]
    te_pairs = pairs[assign == 2]
    tr_end = end_times[assign == 0]
    va_end = end_times[assign == 1]
    te_end = end_times[assign == 2]
    return (tr_pairs, tr_end), (va_pairs, va_end), (te_pairs, te_end)


def _summarize_split(
    pairs: np.ndarray, end_times: np.ndarray, num_assets: int
) -> SplitSummary:
    """Compute step counts and coverage statistics for a split."""

    if pairs.size == 0 or end_times.size == 0:
        return SplitSummary(0, 0, 0.0, 0.0, 0.0)

    day_keys = _normalize_to_day(end_times)
    uniq_days, inv = np.unique(day_keys, return_inverse=True)
    day_asset = np.stack([inv.astype(np.int64), pairs[:, 0].astype(np.int64)], axis=1)
    uniq_da = np.unique(day_asset, axis=0)
    per_day_counts = np.bincount(uniq_da[:, 0], minlength=len(uniq_days))
    coverage = per_day_counts / float(num_assets)

    return SplitSummary(
        steps=int(pairs.shape[0]),
        unique_days=int(len(uniq_days)),
        coverage_mean=float(coverage.mean()),
        coverage_min=float(coverage.min()),
        coverage_max=float(coverage.max()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a prepared dataset cache.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing the cache_ratio_index folder.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.55)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.40)
    parser.add_argument(
        "--coverage-per-window",
        type=float,
        default=0.0,
        help="Minimum fraction of entities required per end date (0.0 disables filtering).",
    )
    parser.add_argument(
        "--global-split",
        dest="per_asset",
        action="store_false",
        help="Split by chronological order across all assets instead of per-asset ordering.",
    )
    parser.add_argument(
        "--as-json",
        action="store_true",
        help="Emit the summary as JSON instead of human-readable text.",
    )
    parser.set_defaults(per_asset=True)
    args = parser.parse_args()

    meta, pairs, end_times = _load_cache(args.data_dir)
    num_assets = len(meta.get("assets", []))
    num_features = len(meta.get("feature_cols", []))

    pairs, end_times = _apply_coverage_filter(
        pairs, end_times, float(args.coverage_per_window), num_assets
    )
    (tr_pairs, tr_end), (va_pairs, va_end), (te_pairs, te_end) = _split_pairs(
        pairs,
        end_times,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        per_asset=bool(args.per_asset),
    )

    overall_summary = _summarize_split(pairs, end_times, num_assets)
    train_summary = _summarize_split(tr_pairs, tr_end, num_assets)
    val_summary = _summarize_split(va_pairs, va_end, num_assets)
    test_summary = _summarize_split(te_pairs, te_end, num_assets)

    summary = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "entities": num_assets,
        "input_channels": num_features,
        "coverage_filter": float(args.coverage_per_window),
        "ratios": {
            "train": float(args.train_ratio),
            "val": float(args.val_ratio),
            "test": float(args.test_ratio),
        },
        "splits": {
            "overall": overall_summary.__dict__,
            "train": train_summary.__dict__,
            "val": val_summary.__dict__,
            "test": test_summary.__dict__,
        },
    }

    if args.as_json:
        print(json.dumps(summary, indent=2))
        return

    print(f"Cache: {summary['data_dir']}")
    print(f"Entities: {summary['entities']}")
    print(f"Input channels: {summary['input_channels']}")
    print(
        "Ratios (train/val/test): "
        f"{summary['ratios']['train']} / {summary['ratios']['val']} / {summary['ratios']['test']}"
    )
    print(f"Coverage filter: {summary['coverage_filter']}")
    print("Steps per split (train / val / test): "
          f"{train_summary.steps} / {val_summary.steps} / {test_summary.steps}")
    print("Average entity coverage per day:")
    print(f"  overall: {overall_summary.coverage_mean:.3f} "
          f"(min={overall_summary.coverage_min:.3f}, max={overall_summary.coverage_max:.3f})")
    print(f"  train:   {train_summary.coverage_mean:.3f} "
          f"(min={train_summary.coverage_min:.3f}, max={train_summary.coverage_max:.3f})")
    print(f"  val:     {val_summary.coverage_mean:.3f} "
          f"(min={val_summary.coverage_min:.3f}, max={val_summary.coverage_max:.3f})")
    print(f"  test:    {test_summary.coverage_mean:.3f} "
          f"(min={test_summary.coverage_min:.3f}, max={test_summary.coverage_max:.3f})")


if __name__ == "__main__":
    main()
