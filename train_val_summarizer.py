import os
import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import crypto_config
from Dataset.fin_dataset import run_experiment

from Model.summarizer import EffSummarizerAE

# -------------------- Data --------------------
train_dl, val_dl, test_dl, sizes = run_experiment(
    data_dir=crypto_config.DATA_DIR,
    date_batching=crypto_config.date_batching,
    dates_per_batch=crypto_config.BATCH_SIZE,
    K=crypto_config.WINDOW,
    H=crypto_config.PRED,
    coverage=crypto_config.COVERAGE,
)
print("sizes:", sizes)
train_size, val_size, test_size = sizes

# Peek a batch to infer shapes
xb, yb, meta = next(iter(train_dl))
V, T = xb  # [B,N,K,D], [B,N,K,D]
M = meta["entity_mask"]  # [B,N]
B, N, K, D = V.shape
print("V:", V.shape, "T:", T.shape)

# -------------------- Model --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

LAP_K = getattr(crypto_config, "SUM_LAP_K", 8)
TV_DIM = getattr(crypto_config, "SUM_TV_DIM", 32)
USE_RES_MLP = getattr(crypto_config, "SUM_USE_RES_MLP", True)

model = EffSummarizerAE(num_entities=N, feat_dim=D, lap_k=LAP_K, tv_dim=TV_DIM, use_residual_mlp=USE_RES_MLP).to(device)

# -------------------- Optimizer & Schedules --------------------
LR = getattr(crypto_config, "SUM_LR", 2e-3)
WD = getattr(crypto_config, "SUM_WD", 1e-4)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

SCALER = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

EPOCHS = getattr(crypto_config, "SUM_EPOCHS", 200)
MAX_PATIENCE = getattr(crypto_config, "SUM_MAX_PATIENCE", 20)
W_V = getattr(crypto_config, "SUM_W_V", 1.0)
W_T = getattr(crypto_config, "SUM_W_T", 1.0)

# -------------------- Checkpointing --------------------
model_dir = getattr(crypto_config, "SUM_DIR", os.path.join("./ldt/saved_model", "SUMMARIZER_EFF"))
os.makedirs(model_dir, exist_ok=True)
best_val = float("inf")
patience = 0
best_path = None

print("Starting summarizer pretraining.")

# -------------------- Training loop --------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_sum, train_elems = 0.0, 0

    for (xb, _, meta) in train_dl:
        V, T = xb  # [B,N,K,D]
        M = meta["entity_mask"].to(device)  # [B,N]
        V = V.permute(0, 2, 1, 3).contiguous().to(device)  # -> [B,K,N,D]
        T = T.permute(0, 2, 1, 3).contiguous().to(device)  # -> [B,K,N,D]

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            loss, parts = model.recon_loss(V, T, entity_mask=M, w_v=W_V, w_t=W_T)
        SCALER.scale(loss).backward()
        SCALER.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        SCALER.step(optimizer)
        SCALER.update()

        # Track per-element mean over [B*T*N]
        elems = (M.sum(dim=1, keepdim=True).clamp_min(1).expand(M.size(0), K).sum()).item()
        train_sum += loss.item() * max(1, int(elems))
        train_elems += max(1, int(elems))

    train_mean = train_sum / max(1, train_elems)

    # --------------- Validation ---------------
    model.eval()
    val_sum, val_elems = 0.0, 0
    with torch.no_grad():
        for (xb, _, meta) in val_dl:
            V, T = xb
            M = meta["entity_mask"].to(device)
            V = V.permute(0, 2, 1, 3).contiguous().to(device)  # [B,K,N,D]
            T = T.permute(0, 2, 1, 3).contiguous().to(device)  # [B,K,N,D]
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                loss, parts = model.recon_loss(V, T, entity_mask=M, w_v=W_V, w_t=W_T)
            elems = (M.sum(dim=1, keepdim=True).clamp_min(1).expand(M.size(0), K).sum()).item()
            val_sum += loss.item() * max(1, int(elems))
            val_elems += max(1, int(elems))

    val_mean = val_sum / max(1, val_elems)

    print(f"Epoch {epoch}/{EPOCHS} | Train {train_mean:.6f} | Val {val_mean:.6f}")

    improved = val_mean < best_val * 0.995
    if improved:
        best_val = val_mean
        best_path = os.path.join(model_dir, f"summarizer_eff_k{LAP_K}.pt")
        torch.save(model.state_dict(), best_path)
        print("  -> Saved best checkpoint")
        patience = 0
    else:
        patience += 1
        if patience >= MAX_PATIENCE:
            print(f"Early stopping at epoch {epoch} (no val improvement in {MAX_PATIENCE} epochs).")
            break

# -------------------- Export encoder stats (μ,σ) for whitening if desired) --------------------
@torch.no_grad()
def export_feature_stats(dloader) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    running_sum = None
    running_sq = None
    count = 0
    for (xb, _, meta) in dloader:
        V, T = xb
        M = meta["entity_mask"].to(device)
        V = V.permute(0, 2, 1, 3).contiguous().to(device)
        T = T.permute(0, 2, 1, 3).contiguous().to(device)
        L = model.summarize(V, T, entity_mask=M)  # [B,K,4K]
        B, TT, C = L.shape
        L = L.view(B * TT, C)
        if running_sum is None:
            running_sum = L.sum(dim=0)
            running_sq = (L ** 2).sum(dim=0)
        else:
            running_sum += L.sum(dim=0)
            running_sq += (L ** 2).sum(dim=0)
        count += L.size(0)
    mu = running_sum / max(1, count)
    var = running_sq / max(1, count) - mu ** 2
    std = var.clamp_min(1e-8).sqrt()
    return mu.cpu(), std.cpu()

mu, std = export_feature_stats(train_dl)
if best_path is None:
    best_path = os.path.join(model_dir, "last.pt")
    torch.save(model.state_dict(), best_path)

stats_path = os.path.join(model_dir, f"summarizer_eff_k{LAP_K}_stats.pt")
torch.save({"mu": mu, "std": std}, stats_path)
print(f"Saved summarizer to: {best_path}\nSaved feature stats to: {stats_path}")

