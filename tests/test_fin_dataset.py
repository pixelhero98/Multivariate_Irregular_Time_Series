import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")


from Data_Prep.fin_dataset import load_dataloaders_with_ratio_split


def _write_minimal_cache(base: Path, *, feature_values, target_values, times, norm_stats):
    root = base / "cache_ratio_index"
    (root / "features_fp16").mkdir(parents=True, exist_ok=True)
    (root / "targets_fp16").mkdir(parents=True, exist_ok=True)
    (root / "times").mkdir(parents=True, exist_ok=True)
    (root / "windows").mkdir(parents=True, exist_ok=True)

    np.save(root / "features_fp16" / "0.npy", np.asarray(feature_values, dtype=np.float16))
    np.save(root / "targets_fp16" / "0.npy", np.asarray(target_values, dtype=np.float16))
    np.save(root / "times" / "0.npy", np.asarray(times, dtype="datetime64[ns]"))

    global_pairs = np.array([[0, 0], [0, 1]], dtype=np.int32)
    end_times = np.asarray(times)[np.array([1, 2])]
    np.save(root / "windows" / "global_pairs.npy", global_pairs)
    np.save(root / "windows" / "end_times.npy", end_times.astype("datetime64[ns]"))

    meta = {
        "format": "indexcache_v1",
        "assets": ["AAA"],
        "asset2id": {"AAA": 0},
        "start": "2020-01-01",
        "end": "2020-01-04",
        "window": 2,
        "horizon": 1,
        "feature_cols": ["RET_CLOSE"],
        "target_col": "RET_CLOSE",
        "feature_cfg": {},
        "normalize_per_ticker": True,
        "clamp_sigma": 5.0,
        "regression": True,
        "seed": 0,
        "keep_time_meta": "end",
    }
    with open(root / "meta.json", "w") as fh:
        json.dump(meta, fh)

    with open(root / "norm_stats.json", "w") as fh:
        json.dump(norm_stats, fh)


def _dataset_norm_summary(dl):
    ds = dl.dataset
    mean_x = np.array(ds.mean_x[0], dtype=np.float32).reshape(-1)[0]
    std_x = np.array(ds.std_x[0], dtype=np.float32).reshape(-1)[0]
    mean_y = float(ds.mean_y[0] if isinstance(ds.mean_y, list) else ds.mean_y)
    std_y = float(ds.std_y[0] if isinstance(ds.std_y, list) else ds.std_y)
    return mean_x, std_x, mean_y, std_y


def test_train_only_norm_scope_recomputes_stats(tmp_path):
    base = tmp_path / "train_only"
    feature_values = np.array([[0.0], [0.1], [0.2], [0.3]], dtype=np.float32)
    target_values = np.array([0.0, 0.01, 0.02, 0.03], dtype=np.float32)
    times = np.array([
        "2020-01-01",
        "2020-01-02",
        "2020-01-03",
        "2020-01-04",
    ], dtype="datetime64[ns]")
    cached_norm = {
        "per_ticker": True,
        "mean_x": [[[10.0]]],
        "std_x": [[[2.0]]],
        "mean_y": [5.0],
        "std_y": [3.0],
    }
    _write_minimal_cache(base, feature_values=feature_values, target_values=target_values,
                         times=times, norm_stats=cached_norm)

    train_dl, _, _, _ = load_dataloaders_with_ratio_split(
        data_dir=str(base),
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        batch_size=1,
        shuffle_train=False,
        num_workers=0,
        pin_memory=False,
        date_batching=False,
        norm_scope="train_only",
    )

    mean_x, std_x, mean_y, std_y = _dataset_norm_summary(train_dl)
    assert mean_x == pytest.approx(0.05, rel=0, abs=1e-6)
    assert std_x == pytest.approx(0.05, rel=0, abs=1e-6)
    assert mean_y == pytest.approx(0.01, rel=0, abs=1e-6)
    assert std_y == pytest.approx(0.0081649658, rel=0, abs=1e-6)


def test_cache_norm_scope_preserves_persisted_stats(tmp_path):
    base = tmp_path / "cache_scope"
    feature_values = np.array([[0.0], [0.1], [0.2], [0.3]], dtype=np.float32)
    target_values = np.array([0.0, 0.01, 0.02, 0.03], dtype=np.float32)
    times = np.array([
        "2021-01-01",
        "2021-01-02",
        "2021-01-03",
        "2021-01-04",
    ], dtype="datetime64[ns]")
    cached_norm = {
        "per_ticker": True,
        "mean_x": [[[1.25]]],
        "std_x": [[[0.75]]],
        "mean_y": [0.2],
        "std_y": [0.1],
    }
    _write_minimal_cache(base, feature_values=feature_values, target_values=target_values,
                         times=times, norm_stats=cached_norm)

    train_dl, _, _, _ = load_dataloaders_with_ratio_split(
        data_dir=str(base),
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        batch_size=1,
        shuffle_train=False,
        num_workers=0,
        pin_memory=False,
        date_batching=False,
        norm_scope="cache",
    )

    mean_x, std_x, mean_y, std_y = _dataset_norm_summary(train_dl)
    assert mean_x == pytest.approx(1.25)
    assert std_x == pytest.approx(0.75)
    assert mean_y == pytest.approx(0.2)
    assert std_y == pytest.approx(0.1)
