import math
import time
from pathlib import Path
from typing import Dict
import crypto_config
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from Dataset.fin_dataset import run_experiment
from Model.summarizer import LaplaceAE


def set_seed(seed: int = 42) -> None:
    """Seed all relevant RNGs for reproducible runs."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def save_ckpt(path: Path, model: nn.Module, stats: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "stats": stats}, path)


def _batch_elements(mask: torch.Tensor, steps: int) -> float:
    """Return number of valid entity/time elements represented by ``mask``."""

    # mask: [B, N] bool
    valid_entities = mask.float().sum().item()
    return valid_entities * float(steps)


def _permute_to_seq_first(x: torch.Tensor) -> torch.Tensor:
    """Convert tensors from ``[B, N, K, F]`` to ``[B, K, N, F]`` layout."""
    return x.permute(0, 2, 1, 3).contiguous()


def run() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Hyper-parameters ----
    lap_k = crypto_config.SUM_LAPLACE_K
    out_len = crypto_config.SUM_CONTEXT_LEN
    ctx_dim = crypto_config.SUM_CONTEXT_DIM
    tv_hidden = crypto_config.SUM_TV_HIDDEN
    dropout = crypto_config.SUM_DROPOUT

    lr = crypto_config.SUM_LR
    wd = crypto_config.SUM_WEIGHT_DECAY
    epochs = crypto_config.SUM_EPOCHS
    grad_clip = crypto_config.GRAD_CLIP
    amp = crypto_config.SUM_AMP
    patience = crypto_config.SUM_PATIENCE
    min_delta = crypto_config.SUM_MIN_DELTA

    model_dir = crypto_config.SUM_DIR
    ckpt_path = Path(model_dir) / f"{crypto_config.PRED}-{crypto_config.VAE_LATENT_CHANNELS}-summarizer.pt"

    set_seed(crypto_config.SEED)

    # ---- Data loaders ----
    train_loader, val_loader, test_loader, sizes = run_experiment(
        data_dir=crypto_config.DATA_DIR,
        date_batching=crypto_config.date_batching,
        dates_per_batch=crypto_config.BATCH_SIZE,
        K=crypto_config.WINDOW,
        H=crypto_config.PRED,
        coverage=crypto_config.COVERAGE,
    )
    print('sizes:', sizes)

    # Probe one batch for dims
    (xb, yb, meta0) = next(iter(train_loader))
    V0, T0 = xb
    _, N0, _, F0 = V0.shape
    print("V:", V0.shape, "T:", T0.shape, "y:", yb.shape)

    # ---- Model ----
    model = LaplaceAE(
        num_entities=N0,
        feat_dim=F0,
        lap_k=lap_k,
        tv_hidden=tv_hidden,
        out_len=out_len,
        context_dim=ctx_dim,
        dropout=dropout,
    ).to(device)

    print(f"Model params: {count_params(model)/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = GradScaler(enabled=amp)

    best_val = math.inf
    best_epoch = 0
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_sum = 0.0
        train_elems = 0.0

        for (V, T), _, meta in train_loader:
            V = _permute_to_seq_first(V).to(device)
            T = _permute_to_seq_first(T).to(device)
            mask = meta["entity_mask"].to(device)
            elems = _batch_elements(mask, V.size(1))
            if elems == 0.0:  # no valid entities
                continue

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                _, aux = model(V, ctx_diff=T, entity_mask=mask)
                loss = model.recon_loss(aux, mask)
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_sum += loss.item() * elems
            train_elems += elems

        train_loss = train_sum / train_elems if train_elems > 0 else 0.0

        # ---- Validation ----
        model.eval()
        val_sum = 0.0
        val_elems = 0.0
        with torch.no_grad():
            for (V, T), _, meta in val_loader:
                V = _permute_to_seq_first(V).to(device)
                T = _permute_to_seq_first(T).to(device)
                mask = meta["entity_mask"].to(device)
                elems = _batch_elements(mask, V.size(1))
                if elems == 0.0:
                    continue

                with autocast(enabled=amp):
                    _, aux = model(V, ctx_diff=T, entity_mask=mask)
                    loss = model.recon_loss(aux, mask)

                val_sum += loss.item() * elems
                val_elems += elems

        val_loss = val_sum / val_elems if val_elems > 0 else float("inf")

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

    # Final save for reference
        if patience_ctr >= patience:
            print(f"\nEarly stopping at epoch {epoch}: validation loss plateaued.")
            break

    # Always save a final snapshot for reproducibility
    save_ckpt(ckpt_path.with_suffix(".final.pt"), model, {"best_val": best_val, "best_epoch": best_epoch})

    # Reload the best checkpoint for downstream evaluation convenience
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        best_val = state.get("stats", {}).get("val_loss", best_val)

    model.eval()
    test_sum = 0.0
    test_elems = 0.0
    with torch.no_grad():
        for (V, T), _, meta in test_loader:
            V = _permute_to_seq_first(V).to(device)
            T = _permute_to_seq_first(T).to(device)
            mask = meta["entity_mask"].to(device)
            elems = _batch_elements(mask, V.size(1))
            if elems == 0.0:
                continue

            with autocast(enabled=amp):
                _, aux = model(V, ctx_diff=T, entity_mask=mask)
                loss = model.recon_loss(aux, mask)

            test_sum += loss.item() * elems
            test_elems += elems

    test_loss = test_sum / test_elems if test_elems > 0 else float("nan")
    print(f"Best val loss: {best_val:.6f} | Test loss: {test_loss:.6f}")



if __name__ == "__main__":
    run()
