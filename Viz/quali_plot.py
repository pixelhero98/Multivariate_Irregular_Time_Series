"""Plot qualitative missingness examples for prepared dataset caches.

This script builds a 2x2 figure illustrating two datasets (NOAA ISD and
PhysioNet CinC) with consecutive timestamps that exhibit lower per-timestamp
coverage. Missing regions are shaded to make gaps visually obvious. Each row
shares a y-axis so the two subplots in the row are directly comparable for the
same entity/time span. The requested horizon/window are enforced via each
dataset module's ``run_experiment`` entry point so the plot matches the
training configuration.

Example usage:
    python Viz/plot_missingness_examples.py \\
        --noaa-dir ./nnoa_isd_cache \\
        --physionet-dir ./physionet_cinc_cache \\
        --output ./missingness_examples.png
"""

from __future__ import annotations


import json
from dataclasses import dataclass
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset_summary import _compute_timestamp_coverage
from Dataset.fin_dataset import CachePaths
from pathlib import Path
from typing import Optional, Sequence
import matplotlib as mpl

mpl.rcParams.update({
    # Font: Times New Roman (with safe fallbacks)
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
    "mathtext.fontset": "stix",   # math matches Times-like look

    # Larger font sizes (tune as you like)
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,

    # Embed TrueType fonts nicely in PDF/PS (good for ICML)
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
mpl.rcParams.update({
    "lines.linewidth": 2.0,
    "lines.markersize": 5.0,
})




@dataclass(frozen=True)
class ExampleSlice:
    dataset: str
    asset_name: str
    feature_names: Sequence[str]
    timestamps: pd.DatetimeIndex
    values: Mapping[str, np.ndarray]
    panel_coverage: pd.Series
    coverage_threshold: float
    coverage_range: Tuple[pd.Timestamp, pd.Timestamp]
    window: int
    horizon: int

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


def _coerce_freq_to_timedelta(freq: Optional[object]) -> Optional[pd.Timedelta]:
    """Best-effort conversion of a stored freq string to a Timedelta."""
    if freq is None:
        return None
    try:
        return pd.to_timedelta(str(freq))
    except (ValueError, TypeError):
        # Allow unit-only strings like "H" or "D".
        try:
            return pd.to_timedelta(f"1{str(freq)}")
        except Exception:
            return None


def _pick_coverage_block(
    coverage: np.ndarray,
    unique_t: np.ndarray,
    quantile: float,
    min_block: int,
    expected_step: Optional[pd.Timedelta] = None,
) -> Tuple[np.ndarray, float]:
    """Pick a low-coverage time block.

    The returned block indices are consecutive *in time* when expected_step is provided;
    otherwise, they are consecutive in the unique_t index.
    """
    threshold = float(np.quantile(coverage, quantile))
    low_mask = coverage <= threshold

    if expected_step is not None and unique_t.shape[0] >= 2:
        ut = pd.to_datetime(unique_t)
        dt = ut.to_series().diff().fillna(expected_step)
        # consecutive-in-time check with tolerance
        is_step = (dt >= (expected_step * 0.5)) & (dt <= (expected_step * 1.5))
        low_mask = low_mask & is_step.to_numpy(dtype=bool)

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
    return np.arange(run_start, end_idx, dtype=np.int64), threshold



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

def _feature_indices(all_features: Sequence[str], requested: Optional[Sequence[str]]) -> List[int]:
    if requested:
        names = [name for name in requested if name in all_features]
    else:
        names = list(all_features[:2])

    if not names:
        raise ValueError("No valid feature names found to plot.")
    return [all_features.index(name) for name in names]


def _dataset_key(meta: Mapping[str, object]) -> str:
    return str(meta.get("dataset", "")).lower()


def _prepare_index_with_run_experiment(
    paths: CachePaths,
    meta: Mapping[str, object],
    window_override: Optional[int],
    horizon_override: Optional[int],
) -> Tuple[Mapping[str, object], int, int]:
    dataset = _dataset_key(meta)
    window = int(window_override if window_override is not None else meta.get("window", meta.get("max_window", 0)))
    horizon = int(horizon_override if horizon_override is not None else meta.get("horizon", meta.get("max_horizon", 0)))

    cov_meta = meta.get("min_coverage", meta.get("coverage", None))
    common_kwargs = dict(
        ratios=(0.7, 0.1, 0.2),
        per_asset=True,
        date_batching=True,
        dates_per_batch=1,
        batch_size=1,
        norm="train_only",
        reindex=True,
        shuffle_train=False,
        num_workers=0,
        pin_memory=False,
    )
    if cov_meta is not None:
        try:
            common_kwargs["coverage"] = float(cov_meta)
        except (TypeError, ValueError):
            pass

    # --- actually rebuild index via dataset run_experiment ---
    if ("noaa" in dataset) or ("isd" in dataset) or ("nnoa" in dataset):
        from Dataset.noaa_isd_dataset import run_experiment as _run_experiment
    elif "physio" in dataset or "cinc" in dataset:
        from Dataset.physionet_cinc_dataset import run_experiment as _run_experiment
    else:
        from Dataset.fin_dataset import run_experiment as _run_experiment
        # fin_dataset run_experiment often doesn't accept the loader flags:
        common_kwargs.pop("shuffle_train", None)
        common_kwargs.pop("num_workers", None)
        common_kwargs.pop("pin_memory", None)

    _run_experiment(
        data_dir=str(paths.data_dir),
        K=window,
        H=horizon,
        **common_kwargs,
    )

    refreshed_meta = _load_meta(paths)
    window = int(refreshed_meta.get("window", window))
    horizon = int(refreshed_meta.get("horizon", horizon))
    if window <= 0:
        raise ValueError("The cache metadata does not specify a valid window size after run_experiment.")
    return refreshed_meta, window, horizon



def _select_asset_forecast_slices(
    paths: CachePaths,
    meta: Mapping[str, object],
    feature_id: int,               # single feature to plot (e.g., temperature or HR)
    horizons: Tuple[int, int],      # (short, long)
    quantile: float,
    min_block: int,
    window: int,
) -> Tuple[ExampleSlice, ExampleSlice]:
    pairs = np.load(paths.windows / "global_pairs.npy")
    end_times = np.load(paths.windows / "end_times.npy")

    assets = list(meta.get("assets", []))
    coverage, inverse_t, unique_t = _compute_timestamp_coverage(pairs, end_times, len(assets))
    if coverage.size == 0:
        raise RuntimeError("Coverage array is empty; ensure the cache contains windowed data.")

    expected_step = _coerce_freq_to_timedelta(meta.get("freq"))
    block_indices, threshold = _pick_coverage_block(
        coverage,
        unique_t,
        quantile=quantile,
        min_block=min_block,
        expected_step=expected_step,
    )
    if block_indices.size == 0:
        raise RuntimeError("Selected low-coverage block is empty (unexpected).")

    block_times = unique_t[block_indices]
    coverage_range = (pd.to_datetime(block_times[0]), pd.to_datetime(block_times[-1]))

    # Find windows whose end_time falls inside the selected low-coverage block
    window_mask = np.isin(inverse_t, block_indices)
    block_pairs = pairs[window_mask]
    if block_pairs.size == 0:
        raise RuntimeError("Unable to find windows that align with the selected low-coverage block.")

    # Pick the asset most present in that low-coverage block
    minlength = max(int(block_pairs[:, 0].max()) + 1, len(assets))
    counts = np.bincount(block_pairs[:, 0].astype(int), minlength=minlength)
    asset_id = int(np.argmax(counts))
    asset_name = assets[asset_id] if asset_id < len(assets) else f"asset-{asset_id}"

    # Choose ONE forecast start index (so both horizons share the same start)
    asset_pairs = block_pairs[block_pairs[:, 0] == asset_id]
    start_indices = asset_pairs[:, 1].astype(int)
    if start_indices.size == 0:
        raise RuntimeError("No windows found for selected asset inside low-coverage block.")

    start_idx = int(np.median(start_indices))  # stable choice; could also pick closest to block center

    features = np.load(paths.features / f"{asset_id}.npy")
    times = np.load(paths.times / f"{asset_id}.npy")

    max_h = max(horizons)
    # Ensure we can slice window+max_h; if not, shift left
    max_end = start_idx + window + max_h
    if max_end > features.shape[0]:
        start_idx = max(0, features.shape[0] - (window + max_h))
        max_end = start_idx + window + max_h

    # Coverage time series for shading (aligned to timestamps)
    coverage_series = pd.Series(coverage, index=pd.to_datetime(unique_t)).sort_index()

    feature_cols = list(meta.get("feature_cols", []))
    fname = feature_cols[feature_id] if feature_id < len(feature_cols) else f"feat-{feature_id}"

    def _make_slice(h: int) -> ExampleSlice:
        end = start_idx + window + h
        t = pd.DatetimeIndex(pd.to_datetime(times[start_idx:end]))
        sel = features[start_idx:end, :].astype(np.float32)
        y = sel[:, feature_id].astype(float)

        if expected_step is not None:
            cov_slice = coverage_series.reindex(t, method="nearest", tolerance=expected_step / 2)
        else:
            cov_slice = coverage_series.reindex(t)

        return ExampleSlice(
            dataset=str(meta.get("dataset", "unknown")),
            asset_name=asset_name,
            feature_names=[fname],
            timestamps=t,
            values={fname: y},
            panel_coverage=cov_slice,
            coverage_threshold=float(threshold),
            coverage_range=coverage_range,
            window=window,
            horizon=h,
        )

    short_h, long_h = horizons
    return _make_slice(short_h), _make_slice(long_h)




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


def _shade_low_coverage(
    ax,
    x,
    panel_coverage: pd.Series,
    threshold: float,
) -> None:
    """Shade spans where per-timestep panel coverage is below `threshold`.

    `x` can be datetime-like (DatetimeIndex) or numeric (relative step indices).
    """
    x_arr = np.asarray(x)
    cov = panel_coverage.to_numpy(dtype=float)
    low = np.isfinite(cov) & (cov <= threshold)
    if not low.any():
        return
    spans = _mask_to_spans(low)
    ymin, ymax = ax.get_ylim()
    label_added = False
    for start, end in spans:
        ax.axvspan(
            x_arr[start],
            x_arr[min(end, len(x_arr) - 1)],
            color="grey",
            alpha=0.2,
            label="low coverage" if not label_added else None,
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


def _plot_examples_2x2_horizons(
    noaa_short: ExampleSlice,
    noaa_long: ExampleSlice,
    physio_short: ExampleSlice,
    physio_long: ExampleSlice,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 6.8), constrained_layout=True)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    series_color = color_cycle[0] if color_cycle else None

    grid = [
        [noaa_short, noaa_long],
        [physio_short, physio_long],
    ]

    def _set_panel_ylim(ax, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=float)
        finite = np.isfinite(y)
        if not finite.any():
            return
        lo, hi = np.quantile(y[finite], [0.01, 0.99])
        pad = 0.05 * (hi - lo + 1e-8)
        ax.set_ylim(lo - pad, hi + pad)

    for r in range(2):
        for c in range(2):
            ex = grid[r][c]
            ax = axes[r, c]
            fname, y = next(iter(ex.values.items()))

            # relative x-axis: 0..(K+h-1)
            x = np.arange(len(ex.timestamps))
            K = ex.window

            cov = ex.panel_coverage.to_numpy(dtype=float)
            low = np.isfinite(cov) & (cov <= ex.coverage_threshold)

            y = np.asarray(y, dtype=float)
            y_gap = y.copy()
            y_gap[low] = np.nan

            # Plot series (no legend label here; legend is shared)
            ax.plot(x, y_gap, color=series_color, linewidth=2.0, label="_nolegend_")
            ax.scatter(x[~low], y[~low], s=1, alpha=0.75, color=series_color,
                       edgecolors="none", label="_nolegend_")

            # Forecast start boundary
            # ax.axvline(K, color="k", linestyle=":", alpha=0.7, linewidth=1.5, label="_nolegend_")

            _set_panel_ylim(ax, y)
            _shade_low_coverage(ax, x, ex.panel_coverage, ex.coverage_threshold)

            # Short title
            ds = str(ex.dataset).replace("_", "-")
            ax.set_title(f"{ds} (h={ex.horizon})")

            # Clean ticks: 0, K, K+h
            ticks = [0, K, K + ex.horizon]
            ticks = [t for t in ticks if 0 <= t <= x[-1]]
            ax.set_xticks(sorted(set(ticks)))

            ax.grid(True, linestyle="--", alpha=0.25)
            ax.margins(x=0.01)

            # Only left column shows y tick labels
            if c == 1:
                ax.tick_params(axis="y", labelleft=False)

            # Only bottom row shows x labels
            if r == 0:
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xlabel("time step")

        # Put asset id/name on left side of each row
        axes[r, 0].set_ylabel(str(grid[r][0].asset_name))

    # Shared legend (generic entries, readable)
    legend_handles = [
        Line2D([0], [0], color=series_color, linewidth=2.0, label="series"),
        Patch(facecolor="grey", edgecolor="none", alpha=0.2, label="low coverage"),
        # Line2D([0], [0], color="k", linestyle=":", linewidth=1.5, label="forecast start"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output}")

def save_qual_2x2_datasets_horizons(
    noaa_dir: Path,
    physionet_dir: Path,
    out_path: Path = Path("./plot/qual_2x2_datasets_horizons.png"),
    noaa_feature: str = "temperature",
    physio_feature: str = "HR",
    noaa_horizons: Tuple[int, int] = (24, 168),
    physio_horizons: Tuple[int, int] = (4, 12),
    noaa_window: int = 336,
    physio_window: int = 24,
    coverage_quantile: list = [0.0, 0.3],
    min_block: int = 256,
) -> Path:
    noaa_paths = CachePaths.from_dir(noaa_dir)
    physio_paths = CachePaths.from_dir(physionet_dir)

    noaa_meta = _load_meta(noaa_paths)
    physio_meta = _load_meta(physio_paths)

    noaa_meta, noaa_window_eff, _ = _prepare_index_with_run_experiment(
        noaa_paths, noaa_meta, window_override=noaa_window, horizon_override=max(noaa_horizons)
    )
    physio_meta, physio_window_eff, _ = _prepare_index_with_run_experiment(
        physio_paths, physio_meta, window_override=physio_window, horizon_override=max(physio_horizons)
    )

    noaa_feat = _feature_indices(list(noaa_meta.get("feature_cols", [])), [noaa_feature])[0]
    physio_feat = _feature_indices(list(physio_meta.get("feature_cols", [])), [physio_feature])[0]

    noaa_short, noaa_long = _select_asset_forecast_slices(
        noaa_paths, noaa_meta, feature_id=noaa_feat,
        horizons=noaa_horizons, quantile=coverage_quantile[0],
        min_block=min_block, window=noaa_window_eff
    )
    physio_short, physio_long = _select_asset_forecast_slices(
        physio_paths, physio_meta, feature_id=physio_feat,
        horizons=physio_horizons, quantile=coverage_quantile[-1],
        min_block=min_block, window=physio_window_eff
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_examples_2x2_horizons(noaa_short, noaa_long, physio_short, physio_long, out_path)
    return out_path



if __name__ == "__main__":
    save_qual_2x2_datasets_horizons(
        noaa_dir=Path("./ldt/noaa_isd_uk_data/noaa_isd_uk"),
        physionet_dir=Path("./ldt/physionet_cinc_data/physionet_cinc_cache"),
    )
