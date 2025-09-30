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

from Dataset.data_gen import build_datasets  # assumed helper used across repo
from Model.summarizer import LaplaceAE
from crypto_config import CONFIG


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
    cfg = CONFIG  # project-wide config

    # ---- Hyperparams (lightweight; feel free to tune in CONFIG) ----
    num_entities = cfg['NUM_ENTITIES']         # e.g., 200
    feat_dim     = cfg.get('ENTITY_FEAT_DIM', 16)
    seq_len      = cfg.get('SEQ_LEN', 118)
    lap_k        = cfg.get('LAPLACE_K', 8)
    out_len      = cfg.get('CTX_OUT_LEN', 16)
    ctx_dim      = cfg.get('CTX_DIM', 256)
    tv_hidden    = cfg.get('TV_HIDDEN', 32)
    dropout      = cfg.get('CTX_DROPOUT', 0.0)

    batch_size   = cfg.get('BATCH_SIZE', 20)
    lr           = cfg.get('LR', 3e-4)
    wd           = cfg.get('WEIGHT_DECAY', 1e-4)
    epochs       = cfg.get('EPOCHS', 200)
    grad_clip    = cfg.get('GRAD_CLIP', 1.0)
    amp          = cfg.get('AMP', True)

    save_root    = Path(cfg.get('SAVE_DIR', './ldt/saved_model/CRYPTO_130'))
    ckpt_path    = save_root / 'summarizer_laplaceAE.pt'

    set_seed(cfg.get('SEED', 42))

    # ---- Data ----
    train_ds, val_ds = build_datasets(cfg)  # should return items with keys: V,T,mask
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=cfg.get('NUM_WORKERS', 4), pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=cfg.get('NUM_WORKERS', 4), pin_memory=True)

    # ---- Model ----
    model = LaplaceAE(
        num_entities=num_entities, feat_dim=feat_dim,
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
        for batch in train_loader:
            # Expect keys: 'V': [B,K,N,D], 'T': [B,K,N,D], 'entity_mask': [B,N]
            V = batch['V'].to(device)
            T = batch['T'].to(device)
            mask = batch['entity_mask'].to(device)

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
            for batch in val_loader:
                V = batch['V'].to(device)
                T = batch['T'].to(device)
                mask = batch['entity_mask'].to(device)
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
