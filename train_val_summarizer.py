import os
import math
import time
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from Dataset.fin_dataset import run_experiment
from Model.summarizer import LaplaceAE
import crypto_config


def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def save_ckpt(path: Path, model: nn.Module, stats: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'stats': stats}, path)


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Hyperparams (read from crypto_config to mirror VAE script) ----
    lap_k     = getattr(crypto_config, 'LAPLACE_K', 8)
    out_len   = getattr(crypto_config, 'CTX_OUT_LEN', 16)
    ctx_dim   = getattr(crypto_config, 'CTX_DIM', 256)
    tv_hidden = getattr(crypto_config, 'TV_HIDDEN', 32)
    dropout   = getattr(crypto_config, 'CTX_DROPOUT', 0.0)

    lr        = getattr(crypto_config, 'BASE_LR', 3e-4)
    wd        = getattr(crypto_config, 'WEIGHT_DECAY', 1e-4)
    epochs    = getattr(crypto_config, 'EPOCHS', 200)
    grad_clip = getattr(crypto_config, 'GRAD_CLIP', 1.0)
    amp       = getattr(crypto_config, 'AMP', True)

    model_dir = Path(getattr(crypto_config, 'SUM_DIR', './ldt/saved_model/SUMMARIZER_EFF'))
    ckpt_path = model_dir / 'summarizer_laplaceAE.pt'

    set_seed(getattr(crypto_config, 'SEED', 42))

    # ---- Data (identical entrypoint to VAE) ----
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
    V0, T0 = xb  # [B,N,K,F]
    B0, N0, K0, F0 = V0.shape
    print('V:', V0.shape, 'T:', T0.shape, 'y:', yb.shape)

    # ---- Model ----
    model = LaplaceAE(
        num_entities=N0, feat_dim=F0,
        lap_k=lap_k, tv_hidden=tv_hidden, out_len=out_len, context_dim=ctx_dim, dropout=dropout
    ).to(device)

    print(f"Model params: {count_params(model)/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = GradScaler(enabled=amp)

    best_val = float('inf')
    history = {}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for (V, T), y, meta in train_loader:
            # V,T come as [B,N,K,F] → convert to [B,K,N,F]
            V = V.permute(0, 2, 1, 3).contiguous().to(device)
            T = T.permute(0, 2, 1, 3).contiguous().to(device)
            mask = meta['entity_mask'].to(device)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                _, aux = model(V, ctx_diff=T, entity_mask=mask)
                loss = model.recon_loss(aux, mask)
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            tr_loss += loss.item() * V.size(0)
            n_tr += V.size(0)

        tr_loss /= max(1, n_tr)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad(), autocast(enabled=amp):
            for (V, T), y, meta in val_loader:
                V = V.permute(0, 2, 1, 3).contiguous().to(device)
                T = T.permute(0, 2, 1, 3).contiguous().to(device)
                mask = meta['entity_mask'].to(device)
                _, aux = model(V, ctx_diff=T, entity_mask=mask)
                loss = model.recon_loss(aux, mask)
                val_loss += loss.item() * V.size(0)
                n_val += V.size(0)
        val_loss /= max(1, n_val)

        dt = time.time() - t0
        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            save_ckpt(ckpt_path, model, {'epoch': epoch, 'val_loss': val_loss})

        print(
            f"Epoch {epoch:03d} | train: {tr_loss:.6f} | val: {val_loss:.6f} | "
            f"best: {best_val:.6f} | {'*' if improved else ' '} | {dt:.1f}s"
        )

    # Final save for reference
    save_ckpt(ckpt_path.with_suffix('.final.pt'), model, {'best_val': best_val, 'epochs': epochs})


if __name__ == "__main__":
    run()
