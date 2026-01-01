"""Training and evaluation loop for the LaplaceAE summarizer model."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import crypto_config
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from Dataset.fin_dataset import run_experiment
from Model.summarizer import LaplaceAE


LoaderTuple = Tuple[DataLoader, DataLoader, DataLoader]


def set_seed(seed: int = 42) -> None:
    """Seed all relevant RNGs for reproducible runs."""

    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def save_ckpt(path: Path, model: nn.Module, stats: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "stats": stats}, path)


def _ensure_loaders(
    train_loader: Optional[DataLoader],
    val_loader: Optional[DataLoader],
    test_loader: Optional[DataLoader],
    sizes: Optional[Sequence[int]],
    config=crypto_config,
) -> Tuple[LoaderTuple, Optional[Tuple[int, int, int]]]:
    if any(loader is None for loader in (train_loader, val_loader, test_loader)):
        train_loader, val_loader, test_loader, sizes = run_experiment(
            data_dir=config.DATA_DIR,
            date_batching=config.date_batching,
            dates_per_batch=config.BATCH_SIZE,
            K=config.WINDOW,
            H=config.PRED,
            coverage=config.COVERAGE,
        )
    elif sizes is None:
        try:
            sizes = tuple(len(dl.dataset) for dl in (train_loader, val_loader, test_loader))
        except Exception:
            sizes = None

    if train_loader is None or val_loader is None or test_loader is None:
        raise RuntimeError("Failed to obtain train/val/test dataloaders.")

    return (train_loader, val_loader, test_loader), sizes


def _summarize_dataset(train_loader: DataLoader, sizes: Optional[Sequence[int]]) -> Tuple[int, int]:
    if sizes is not None:
        print(f"sizes: {tuple(sizes)}")
    else:
        print("sizes: (unknown)")

    try:
        (xb, yb, meta) = next(iter(train_loader))
    except StopIteration as exc:  # pragma: no cover - defensive
        raise RuntimeError("Training dataloader produced no batches.") from exc

    V, T = xb
    _, num_entities, _, feat_dim = V.shape
    print("V:", tuple(V.shape), "T:", tuple(T.shape), "y:", tuple(yb.shape))
    return num_entities, feat_dim


def _permute_to_seq_first(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1, 3).contiguous()


def _nan_to_num(x: torch.Tensor) -> torch.Tensor:
    if torch.isfinite(x).all():
        return x
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _entity_finite_mask(x: torch.Tensor) -> torch.Tensor:
    """Return per-entity mask marking sequences with all finite values."""

    finite = torch.isfinite(x)
    for _ in range(x.dim() - 2):
        finite = finite.all(dim=-1)
    return finite


def _apply_entity_mask(series: torch.Tensor, mask_bn: torch.Tensor) -> torch.Tensor:
    if mask_bn.dtype != torch.bool:
        mask_bn = mask_bn.to(dtype=torch.bool)
    if mask_bn.shape[0] != series.shape[0] or mask_bn.shape[1] != series.shape[2]:
        raise ValueError(
            f"Mask shape {tuple(mask_bn.shape)} incompatible with series shape {tuple(series.shape)}"
        )
    mask = mask_bn[:, None, :, None].to(device=series.device, dtype=series.dtype)
    return series * mask


def _batch_elements(mask: torch.Tensor, steps: int) -> float:
    return mask.float().sum().item() * float(steps)


def _prepare_batch(
    batch: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    (V, T), _, meta = batch
    V = _nan_to_num(_permute_to_seq_first(V)).to(device)
    T = _nan_to_num(_permute_to_seq_first(T)).to(device)
    mask = meta["entity_mask"].to(device=device, dtype=torch.bool)
    mask = mask & _entity_finite_mask(V) & _entity_finite_mask(T)
    V = _apply_entity_mask(V, mask)
    T = _apply_entity_mask(T, mask)
    elems = _batch_elements(mask, V.size(1))
    return V, T, mask, elems


def _run_epoch(
    loader: Iterable,
    model: LaplaceAE,
    device: torch.device,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    grad_clip: float = 0.0,
    amp: bool = False,
) -> float:
    is_train = optimizer is not None
    total_loss = 0.0
    total_elems = 0.0

    for batch in loader:
        V, T, mask, elems = _prepare_batch(batch, device)
        if elems == 0.0:
            continue

        if is_train:
            if scaler is None:
                raise ValueError("GradScaler must be provided when training.")
            optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp):
            _, aux = model(V, ctx_diff=T)
            loss = model.recon_loss(aux, mask)

        if is_train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * elems
        total_elems += elems

    if total_elems == 0.0:
        return 0.0 if is_train else float("inf")
    return total_loss / total_elems


def run(
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    sizes: Optional[Sequence[int]] = None,
    config=crypto_config,
) -> Dict[str, object]:
    (train_loader, val_loader, test_loader), sizes = _ensure_loaders(
        train_loader, val_loader, test_loader, sizes, config
    )
    num_entities, feat_dim = _summarize_dataset(train_loader, sizes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config.SUM_AMP and device.type == "cuda")
    grad_clip = getattr(config, "GRAD_CLIP", 0.0)
    print(f"Using device: {device}")

    set_seed(config.SEED)

    model = LaplaceAE(
        num_entities=num_entities,
        feat_dim=feat_dim,
        window_size=config.WINDOW,
        lap_k=config.SUM_LAPLACE_K,
        tv_hidden=config.SUM_TV_HIDDEN,
        out_len=config.SUM_CONTEXT_LEN,
        context_dim=config.SUM_CONTEXT_DIM,
        dropout=config.SUM_DROPOUT,
    ).to(device)
    print(f"Model params: {count_params(model) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.SUM_LR, weight_decay=config.SUM_WEIGHT_DECAY)
    scaler = GradScaler(enabled=amp)

    epochs = config.SUM_EPOCHS
    patience = config.SUM_PATIENCE
    min_delta = config.SUM_MIN_DELTA

    ckpt_path = Path(config.SUM_DIR) / f"{config.PRED}-{config.VAE_LATENT_CHANNELS}-summarizer.pt"

    best_val = math.inf
    best_epoch = 0
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = _run_epoch(
            train_loader,
            model,
            device,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=grad_clip,
            amp=amp,
        )

        model.eval()
        with torch.no_grad():
            val_loss = _run_epoch(val_loader, model, device, amp=amp)

        elapsed = time.time() - epoch_start
        improved = val_loss < (best_val - min_delta)
        if improved:
            best_val = val_loss
            best_epoch = epoch
            patience_ctr = 0
            save_ckpt(ckpt_path, model, {"epoch": epoch, "val_loss": val_loss})
        else:
            patience_ctr += 1

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | train {train_loss:.6f} | val {val_loss:.6f} | "
            f"best {best_val:.6f} @ {best_epoch:03d} | patience {patience_ctr}/{patience} | {elapsed:.1f}s"
        )

        if patience_ctr >= patience:
            print(f"\nEarly stopping at epoch {epoch}: validation loss plateaued.")
            break

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        best_val = state.get("stats", {}).get("val_loss", best_val)

    model.eval()
    with torch.no_grad():
        test_loss = _run_epoch(test_loader, model, device, amp=amp)

    print(f"Best val loss: {best_val:.6f} | Test loss: {test_loss:.6f}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "sizes": sizes,
        "best_val": best_val,
        "test_loss": test_loss,
        "checkpoint": str(ckpt_path),
    }


if __name__ == "__main__":  # pragma: no cover
    run()
