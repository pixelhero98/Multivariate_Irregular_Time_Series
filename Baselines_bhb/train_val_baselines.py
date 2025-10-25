import argparse
from contextlib import nullcontext
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from torch.cuda.amp import GradScaler, autocast

import crypto_config
from Dataset.fin_dataset import run_experiment
from Baselines_bhb.DLinear import DLinear
from Baselines_bhb.FEDformer import FEDformer


MODEL_REGISTRY = {"dlinear": DLinear, "fedformer": FEDformer}


def device_of():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = device_of()


MODEL_NAME_OVERRIDE = 'fedformer'
model_name = (MODEL_NAME_OVERRIDE or getattr(crypto_config, "BASELINE_MODEL", "dlinear")).lower()
if model_name not in MODEL_REGISTRY:
    raise SystemExit(f"Unknown baseline '{model_name}'. Available: {sorted(MODEL_REGISTRY)}")


def autocast_context():
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)

    return nullcontext()





# ---------------- data ----------------
train_dl, val_dl, test_dl, sizes = run_experiment(
    data_dir=crypto_config.DATA_DIR,

    K=crypto_config.WINDOW,   # past length
    H=crypto_config.PRED,     # forecast horizon

    ratios=(
        getattr(crypto_config, "train_ratio", 0.7),
        getattr(crypto_config, "val_ratio", 0.1),
        getattr(crypto_config, "test_ratio", 0.2),
    ),
    date_batching=getattr(crypto_config, "date_batching", True),
    coverage=getattr(crypto_config, "COVERAGE", 0.85),
    dates_per_batch=getattr(crypto_config, "BATCH_SIZE", 30),
    batch_size=getattr(crypto_config, "ENTITY_BATCH_SIZE", 64),
    norm=getattr(crypto_config, "BASELINE_NORM_SCOPE", "train_only"),
    per_asset=getattr(crypto_config, "BASELINE_PER_ASSET", True),
)
print("sizes:", sizes)
(xb0, yb0, meta0) = next(iter(train_dl))
V0, T0 = xb0
B0, N0, K, F = V0.shape
H = yb0.shape[-1]
print("V:", V0.shape, "T:", T0.shape, "y:", yb0.shape)

# ---------------- model ----------------
if model_name == "dlinear":
    model = MODEL_REGISTRY[model_name](
        seq_len=K,
        pred_len=H,
        moving_avg=getattr(crypto_config, "DLINEAR_MA", 25),
    ).to(device)
    lr = getattr(crypto_config, "DLINEAR_LR", 1e-3)
    weight_decay = getattr(crypto_config, "DLINEAR_WD", 1e-4)
    epochs = getattr(crypto_config, "DLINEAR_EPOCHS", 50)
    patience = getattr(crypto_config, "DLINEAR_EARLY_STOP", 20)
    ckpt_subdir = "dlinear"
elif model_name == "fedformer":
    if "fedformer" not in MODEL_REGISTRY:
        raise SystemExit("FEDformer baseline is unavailable. Please ensure Baselines/FEDformer.py imports correctly.")
    FEDformer = MODEL_REGISTRY[model_name]
    label_len_default = max(1, min(H, K // 2))
    model = FEDformer(
        seq_len=K,
        pred_len=H,
        label_len=getattr(crypto_config, "FEDFORMER_LABEL_LEN", label_len_default),
        d_model=getattr(crypto_config, "FEDFORMER_D_MODEL", 32),
        d_ff=getattr(crypto_config, "FEDFORMER_D_FF", 64),
        dropout=getattr(crypto_config, "FEDFORMER_DROPOUT", 0.1),
        moving_avg=getattr(crypto_config, "FEDFORMER_MOVING_AVG", 17),
        modes=getattr(crypto_config, "FEDFORMER_MODES", 8),
        enc_layers=getattr(crypto_config, "FEDFORMER_ENC_LAYERS", 1),
        dec_layers=getattr(crypto_config, "FEDFORMER_DEC_LAYERS", 1),
        c_in=1,
        c_out=1,
    ).to(device)
    lr = getattr(crypto_config, "FEDFORMER_LR", 3e-4)
    weight_decay = getattr(crypto_config, "FEDFORMER_WD", 1e-4)
    epochs = getattr(crypto_config, "FEDFORMER_EPOCHS", 50)
    patience = getattr(crypto_config, "FEDFORMER_EARLY_STOP", 15)
    ckpt_subdir = "fedformer"
else:  # pragma: no cover - safety net
    raise SystemExit(f"Unsupported model '{model_name}'")

opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda")) if device.type == "cuda" else None

ckpt_dir = os.path.join(getattr(crypto_config, "CKPT_DIR", "./checkpoints"), ckpt_subdir)

os.makedirs(ckpt_dir, exist_ok=True)
best_val, best_path = float("inf"), None

# ---------------- helpers ----------------
def _flatten_mask(yb, mask_bn, device):
    """yb: [B,N,H], mask_bn: [B,N] -> y_true: [Beff,H,1], sel_idx: [Beff]"""
    B, N, H = yb.shape
    y_flat = yb.reshape(B * N, H)
    m_flat = mask_bn.reshape(B * N)
    if not m_flat.any():
        return None, None
    y_true = y_flat[m_flat].unsqueeze(-1).to(device)  # [Beff,H,1]
    sel = m_flat.nonzero(as_tuple=False).squeeze(1)
    return y_true, sel

def _build_x_hist_from_T(T, sel_idx):
    """T: [B,N,K,F]; if F==1 use it directly; else take the first channel.
       Returns x_hist: [Beff,K,1] (no target-field arg needed)."""
    B, N, K, F = T.shape
    if F == 1:
        t = T[..., 0]                   # [B,N,K]
    else:
        t = T[..., 0]                   # <— uses the first channel as target history
    t_flat = t.reshape(B * N, K)        # [B*N,K]
    return t_flat[sel_idx].unsqueeze(-1)

def run_epoch(dl, train: bool):
    model.train() if train else model.eval()
    tot, cnt = 0.0, 0
    for xb, yb, meta in dl:
        V, T = xb
        mask_bn = meta["entity_mask"]
        y_true, sel = _flatten_mask(yb, mask_bn, device)
        if y_true is None:
            continue
        x_hist = _build_x_hist_from_T(T, sel).to(device)  # [Beff,K,1]

        with autocast_context():
            y_hat = model(x_hist)                         # [Beff,H,1]
            loss = F_nn.mse_loss(y_hat, y_true)

        bs = y_true.size(0)
        if train:
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()


        tot += loss.item() * bs
        cnt += bs
    return tot / max(1, cnt)

# ---------------- train ----------------
E = epochs
pat_max = patience
pat = 0
for e in range(1, E + 1):
    tr = run_epoch(train_dl, True)
    va = run_epoch(val_dl, False)
    # print(f"[DLinear] epoch {e:03d}  train:{tr:.6f}  val:{va:.6f}")
    print(f"[{model_name.upper()}] epoch {e:03d}  train:{tr:.6f}  val:{va:.6f}")
    if va + 1e-12 < best_val:
        best_val, pat = va, 0
        # best_path = os.path.join(ckpt_dir, f"dlinear_best_{va:.6f}.pt")
        best_path = os.path.join(ckpt_dir, f"{model_name}_best_{va:.6f}.pt")
        torch.save({"state_dict": model.state_dict(), "K": K, "H": H}, best_path)
    else:
        pat += 1
        if pat >= pat_max:
            # print("[DLinear] early stopping.")
            print(f"[{model_name.upper()}] early stopping.")
            break

# ---------------- test ----------------
if best_path:
    model.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
model.eval()

abs_sum = sq_sum = 0.0
elts = 0
pinball_qs = getattr(crypto_config, "PINBALL_QS", (0.1, 0.5, 0.9))
pinball_sums = {float(q): 0.0 for q in pinball_qs}

with torch.no_grad():
    for xb, yb, meta in test_dl:
        V, T = xb
        mask_bn = meta["entity_mask"]
        y_true, sel = _flatten_mask(yb, mask_bn, device)
        if y_true is None:
            continue
        x_hist = _build_x_hist_from_T(T, sel).to(device)
        y_hat = model(x_hist)  # [Beff,H,1]

        res = y_hat - y_true
        abs_sum += res.abs().sum().item()
        sq_sum  += (res ** 2).sum().item()
        elts    += res.numel()

        for q in pinball_qs:
            qf = float(q)
            diff = y_true - y_hat
            loss_q = torch.maximum(qf * diff, (qf - 1.0) * diff)
            pinball_sums[qf] += loss_q.sum().item()

mae = abs_sum / max(1, elts)
mse = sq_sum  / max(1, elts)
pinball = {q: pinball_sums[q] / max(1, elts) for q in pinball_sums}
qs_fmt = ", ".join(f"{q:.2f}:{pinball[q]:.6f}" for q in sorted(pinball))
# print(f"[DLinear][test]  MAE: {mae:.6f} | MSE: {mse:.6f} | Pinball[{qs_fmt}]")
print(f"[{model_name.upper()}][test]  MAE: {mae:.6f} | MSE: {mse:.6f} | Pinball[{qs_fmt}]")