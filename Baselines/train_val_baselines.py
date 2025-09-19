import os
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from torch.cuda.amp import GradScaler, autocast

import crypto_config
from Dataset.fin_dataset import run_experiment
from Baselines.dlinear import DLinear

def device_of():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    return torch.device("cpu")

device = device_of()

# ---------------- data ----------------
train_dl, val_dl, test_dl, sizes = run_experiment(
    data_dir=crypto_config.DATA_DIR,
    date_batching=crypto_config.date_batching,
    dates_per_batch=crypto_config.BATCH_SIZE,
    K=crypto_config.WINDOW,   # past length
    H=crypto_config.PRED,     # forecast horizon
    coverage=crypto_config.COVERAGE,
)
print("sizes:", sizes)
(xb0, yb0, meta0) = next(iter(train_dl))
V0, T0 = xb0
B0, N0, K, F = V0.shape
H = yb0.shape[-1]
print("V:", V0.shape, "T:", T0.shape, "y:", yb0.shape)

# ---------------- model ----------------
model = DLinear(seq_len=K, pred_len=H, moving_avg=getattr(crypto_config, "DLINEAR_MA", 25)).to(device)
opt = torch.optim.AdamW(model.parameters(),
                        lr=getattr(crypto_config, "DLINEAR_LR", 1e-3),
                        weight_decay=getattr(crypto_config, "DLINEAR_WD", 1e-4))
scaler = GradScaler(enabled=(device.type == "cuda"))

ckpt_dir = os.path.join(getattr(crypto_config, "CKPT_DIR", "./checkpoints"), "dlinear")
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

        with autocast(enabled=(device.type == "cuda")):
            y_hat = model(x_hist)                         # [Beff,H,1]
            loss = F_nn.mse_loss(y_hat, y_true)

        bs = y_true.size(0)
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

        tot += loss.item() * bs
        cnt += bs
    return tot / max(1, cnt)

# ---------------- train ----------------
E = getattr(crypto_config, "DLINEAR_EPOCHS", 50)
pat_max = getattr(crypto_config, "DLINEAR_EARLY_STOP", 20)
pat = 0
for e in range(1, E + 1):
    tr = run_epoch(train_dl, True)
    va = run_epoch(val_dl, False)
    print(f"[DLinear] epoch {e:03d}  train:{tr:.6f}  val:{va:.6f}")
    if va + 1e-12 < best_val:
        best_val, pat = va, 0
        best_path = os.path.join(ckpt_dir, f"dlinear_best_{va:.6f}.pt")
        torch.save({"state_dict": model.state_dict(), "K": K, "H": H}, best_path)
    else:
        pat += 1
        if pat >= pat_max:
            print("[DLinear] early stopping.")
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
print(f"[DLinear][test]  MAE: {mae:.6f} | MSE: {mse:.6f} | Pinball[{qs_fmt}]")
