"""Plot qualitative missingness examples for prepared dataset caches.

This script builds a 2x2 figure illustrating two datasets (NOAA ISD and
PhysioNet CinC) with consecutive timestamps that exhibit lower per-timestamp
coverage. Missing regions are shaded to make gaps visually obvious. Each row
shares a y-axis so the two subplots in the row are directly comparable for the
same entity/time span.

Example usage:
    python Viz/plot_missingness_examples.py \\
        --noaa-dir ./nnoa_isd_cache \\
        --physionet-dir ./physionet_cinc_cache \\
        --output ./missingness_examples.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Dataset.dataset_summary import _compute_timestamp_coverage
from Dataset.fin_dataset import CachePaths


@dataclass(frozen=True)
class ExampleSlice:
    dataset: str
    asset_name: str
    feature_names: Sequence[str]
    timestamps: pd.DatetimeIndex
    values: Mapping[str, np.ndarray]
    coverage_range: Tuple[pd.Timestamp, pd.Timestamp]


def _load_meta(paths: CachePaths) -> Dict[str, object]:
    meta_path = paths.meta
    if not meta_path.exists():
        alt = paths.cache_root / "meta.json"
        if alt.exists():
            meta_path = alt
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Cache metadata not found at '{paths.meta}' (also tried '{paths.cache_root / 'meta.json'}')."
        )
    with meta_path.open("r") as f:
        return json.load(f)


def _longest_true_run(mask: np.ndarray) -> Tuple[Optional[int], int]:
    best_start: Optional[int] = None
    best_len = 0
    run_start: Optional[int] = None
    for idx, flag in enumerate(mask.tolist()):
        if flag:
            if run_start is None:
                run_start = idx
        else:
            if run_start is not None:
                run_len = idx - run_start
                if run_len > best_len:
                    best_start, best_len = run_start, run_len
                run_start = None
    if run_start is not None:
        run_len = len(mask) - run_start
        if run_len > best_len:
            best_start, best_len = run_start, run_len
    return best_start, best_len


def _pick_coverage_block(
    coverage: np.ndarray, unique_t: np.ndarray, quantile: float, min_block: int
) -> np.ndarray:
    threshold = np.quantile(coverage, quantile)
    low_mask = coverage <= threshold
    run_start, run_len = _longest_true_run(low_mask)

    if run_start is None or run_len == 0:
        anchor = int(np.argmin(coverage))
        run_start = max(0, anchor - max(1, min_block // 2))
        run_len = 1

    if run_len < min_block:
        run_len = min_block
        if run_start + run_len > unique_t.shape[0]:
            run_start = max(0, unique_t.shape[0] - run_len)

    end_idx = min(unique_t.shape[0], run_start + run_len)
    return np.arange(run_start, end_idx, dtype=np.int64)


def _feature_indices(all_features: Sequence[str], requested: Optional[Sequence[str]]) -> List[int]:
    if requested:
        names = [name for name in requested if name in all_features]
    else:
        names = list(all_features[:2])

    if not names:
        raise ValueError("No valid feature names found to plot.")
    return [all_features.index(name) for name in names]


def _select_asset_slice(
    paths: CachePaths,
    meta: Mapping[str, object],
    feature_ids: Sequence[int],
    quantile: float,
    min_block: int,
    max_points: int,
) -> ExampleSlice:
    pairs = np.load(paths.windows / "global_pairs.npy")
    end_times = np.load(paths.windows / "end_times.npy")

    assets = list(meta.get("assets", []))
    window = int(meta.get("window", meta.get("max_window", 0)))
    if window <= 0:
        raise ValueError("The cache metadata does not specify a valid window size.")

    coverage, inverse_t, unique_t = _compute_timestamp_coverage(pairs, end_times, len(assets))
    if coverage.size == 0:
        raise RuntimeError("Coverage array is empty; ensure the cache contains windowed data.")

    block_indices = _pick_coverage_block(coverage, unique_t, quantile=quantile, min_block=min_block)
    block_times = unique_t[block_indices]
    coverage_range = (pd.to_datetime(block_times[0]), pd.to_datetime(block_times[-1]))

    window_mask = np.isin(inverse_t, block_indices)
    block_pairs = pairs[window_mask]
    if block_pairs.size == 0:
        raise RuntimeError("Unable to find windows that align with the selected low-coverage block.")

    minlength = max(int(block_pairs[:, 0].max()) + 1, len(assets))
    counts = np.bincount(block_pairs[:, 0].astype(int), minlength=minlength)
    asset_id = int(np.argmax(counts))
    asset_name = assets[asset_id] if asset_id < len(assets) else f"asset-{asset_id}"

    asset_pairs = block_pairs[block_pairs[:, 0] == asset_id]
    start_indices = asset_pairs[:, 1].astype(int)
    slice_start = int(start_indices.min())
    slice_end = int(start_indices.max() + window)

    features = np.load(paths.features / f"{asset_id}.npy")
    times = np.load(paths.times / f"{asset_id}.npy")

    slice_end = min(slice_end, features.shape[0])
    if max_points > 0 and slice_end - slice_start > max_points:
        mid = slice_start + (slice_end - slice_start) // 2
        half = max_points // 2
        slice_start = max(0, mid - half)
        slice_end = min(features.shape[0], slice_start + max_points)

    timestamps = pd.to_datetime(times[slice_start:slice_end])
    selected = features[slice_start:slice_end, :].astype(np.float32)

    feature_names = [meta.get("feature_cols", [])[idx] for idx in feature_ids]
    values = {name: selected[:, fid] for name, fid in zip(feature_names, feature_ids)}

    return ExampleSlice(
        dataset=str(meta.get("dataset", "unknown")),
        asset_name=asset_name,
        feature_names=feature_names,
        timestamps=pd.DatetimeIndex(timestamps),
        values=values,
        coverage_range=coverage_range,
    )


def _mask_to_spans(mask: np.ndarray) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def _shade_missing(ax, timestamps: pd.DatetimeIndex, values: np.ndarray) -> None:
    missing = np.isnan(values)
    if not missing.any():
        return
    spans = _mask_to_spans(missing)
    ymin, ymax = ax.get_ylim()
    label_added = False
    for start, end in spans:
        ax.axvspan(
            timestamps[start],
            timestamps[min(end, len(timestamps) - 1)],
            color="grey",
            alpha=0.2,
            label="missing" if not label_added else None,
        )
        label_added = True
    ax.set_ylim(ymin, ymax)


def _shared_ylim(values: Iterable[np.ndarray], pad: float = 0.05) -> Tuple[float, float]:
    finite_vals = np.concatenate([np.asarray(v)[np.isfinite(v)] for v in values if v.size])
    if finite_vals.size == 0:
        return 0.0, 1.0
    vmin, vmax = float(finite_vals.min()), float(finite_vals.max())
    if vmin == vmax:
        return vmin - 1.0, vmax + 1.0
    delta = vmax - vmin
    return vmin - pad * delta, vmax + pad * delta


def _plot_examples(noaa: ExampleSlice, physio: ExampleSlice, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    formatter = mdates.DateFormatter("%Y-%m-%d\n%H:%M")
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    rows = [(noaa, axes[0]), (physio, axes[1])]
    for row_idx, (example, row_axes) in enumerate(rows):
        ylim = _shared_ylim(example.values.values())
        for col_idx, (ax, (fname, series)) in enumerate(zip(row_axes, example.values.items())):
            color = color_cycle[col_idx % len(color_cycle)] if color_cycle else None
            ax.plot(example.timestamps, series, label=fname, color=color)
            ax.set_ylim(*ylim)
            _shade_missing(ax, example.timestamps, series)
            ax.xaxis.set_major_formatter(formatter)
            ax.set_title(f"{example.dataset} • {fname}")
            ax.axvline(example.coverage_range[0], color="red", linestyle="--", alpha=0.6)
            ax.axvline(example.coverage_range[1], color="red", linestyle="--", alpha=0.6)
            ax.grid(True, linestyle="--", alpha=0.3)

        row_axes[0].set_ylabel(f"{example.asset_name}")
        row_axes[0].legend(loc="upper right")
        row_axes[1].legend(loc="upper right")

    fig.suptitle(
        "Qualitative missingness across consecutive, low-coverage timestamps\n"
        "(rows: same case and y-scale; shaded regions denote missing observations)"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Saved figure to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 2x2 qualitative plot showing missingness for NOAA ISD and PhysioNet CinC caches."
    )
    parser.add_argument("--noaa-dir", type=Path, required=True, help="Path to prepared NOAA ISD cache directory.")
    parser.add_argument(
        "--physionet-dir", type=Path, required=True, help="Path to prepared PhysioNet CinC cache directory."
    )
    parser.add_argument(
        "--noaa-features",
        nargs="*",
        help="Feature names to plot for NOAA (default: first two feature columns recorded in the cache).",
    )
    parser.add_argument(
        "--physio-features",
        nargs="*",
        help="Feature names to plot for PhysioNet (default: first two feature columns recorded in the cache).",
    )
    parser.add_argument(
        "--coverage-quantile",
        type=float,
        default=0.35,
        help="Quantile used to pick consecutive timestamps with lower coverage.",
    )
    parser.add_argument(
        "--min-block",
        type=int,
        default=32,
        help="Minimum number of consecutive timestamps to include from the low-coverage region.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=160,
        help="Maximum number of points to display per subplot (controls the time-span shown).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("missingness_examples.png"),
        help="Where to save the resulting figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    noaa_paths = CachePaths.from_dir(args.noaa_dir)
    physio_paths = CachePaths.from_dir(args.physionet_dir)

    noaa_meta = _load_meta(noaa_paths)
    physio_meta = _load_meta(physio_paths)

    noaa_feats = _feature_indices(list(noaa_meta.get("feature_cols", [])), args.noaa_features)
    physio_feats = _feature_indices(list(physio_meta.get("feature_cols", [])), args.physio_features)

    noaa_example = _select_asset_slice(
        noaa_paths,
        noaa_meta,
        noaa_feats,
        quantile=args.coverage_quantile,
        min_block=args.min_block,
        max_points=args.max_points,
    )
    physio_example = _select_asset_slice(
        physio_paths,
        physio_meta,
        physio_feats,
        quantile=args.coverage_quantile,
        min_block=args.min_block,
        max_points=args.max_points,
    )

    _plot_examples(noaa_example, physio_example, args.output)


if __name__ == "__main__":
    main()
