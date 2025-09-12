
# train_val_baseline.py
import os, json, math, argparse, inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Flexible imports for your repo structure
try:
    import crypto_config
except Exception:
    class _Cfg: pass
    crypto_config = _Cfg()
    # sensible fallbacks
    crypto_config.SEED = 1337
    crypto_config.PRED = 36
    crypto_config.BASE_LR = 1e-3
    crypto_config.WEIGHT_DECAY = 1e-4
    crypto_config.WARMUP_STEPS = 100
    crypto_config.TOTAL_STEPS = 10000
    crypto_config.EPOCHS = 10
    crypto_config.MODEL_WIDTH = 256
    crypto_config.DECODER_OUT = 32
    crypto_config.TIDE_NE = 2
    crypto_config.TIDE_ND = 2
    crypto_config.TEMPORAL_HIDDEN = 64
    crypto_config.DROPOUT = 0.1
    crypto_config.MAX_GRAD_NORM = 1.0
    crypto_config.OUT_DIR = "./out"
    crypto_config.MOVING_AVG = 25
    crypto_config.NUM_EVAL_SAMPLES = 0  # set >1 to enable MC-dropout
    crypto_config.EVAL_HORIZON_STEPS = [5, 20, 63, 126]
    crypto_config.EVAL_HORIZON_NAMES = ["1w","1m","1q","2q"]

# dataset / utils
try:
    from Dataset.fin_dataset import run_experiment
except Exception:
    raise

try:
    from Model.cond_diffusion_utils import set_torch, make_warmup_cosine, flatten_targets
except Exception:
    from cond_diffusion_utils import set_torch, make_warmup_cosine, flatten_targets

# models
from DLinear import DLinear
from models.tide_small import TiDESmall

def build_x_hist_from_T(T: torch.Tensor, mask_bn: torch.Tensor, device: torch.device):
    B, N, K, F = T.shape
    t0 = T[..., 0].reshape(B * N, K)
    m_flat = mask_bn.reshape(B * N)
    x_hist = t0[m_flat].unsqueeze(-1).to(device)  # [Beff,K,1]
    return x_hist, K

def build_cov_from_V(V: torch.Tensor, mask_bn: torch.Tensor, device: torch.device):
    B, N, K, Fv = V.shape
    v_flat = V.reshape(B * N, K * Fv)
    m_flat = mask_bn.reshape(B * N)
    x_cov = v_flat[m_flat].to(device)  # [Beff, K*Fv]
    return x_cov, K, Fv

def evaluate_point(y_true: torch.Tensor, y_pred: torch.Tensor):
    err = (y_pred - y_true)
    mae = err.abs().mean().item()
    mse = (err**2).mean().item()
    return mae, mse

def evaluate_by_horizon_deterministic(y_true, y_pred, horizon_steps, horizon_names):
    import math
    B, H, C = y_true.size()
    if len(horizon_names) != len(horizon_steps):
        horizon_names = [str(h) for h in horizon_steps]
    out = {}
    for name, h in zip(horizon_names, horizon_steps):
        if h < 1 or h > H: 
            continue
        idx = h - 1
        y = y_true[:, idx, :]
        yhat = y_pred[:, idx, :]
        err = (yhat - y)
        mae = err.abs().mean().item()
        rmse = math.sqrt((err**2).mean().item())
        crps = mae  # degenerate distribution
        out[name] = {"crps": crps, "mae": mae, "rmse": rmse}
    return out

@torch.no_grad()
def evaluate_prob_mc_dropout(model: nn.Module, test_dl, device, num_samples: int = 32, use_cov: bool = False):
    model.train()  # keep dropout ON
    crps_sum, n = 0.0, 0
    abs_sum, sq_sum, elts = 0.0, 0.0, 0
    pin_qs = (0.1, 0.5, 0.9)
    pinball_sums = {q: 0.0 for q in pin_qs}

    had_dropout = any(isinstance(m, nn.Dropout) and m.p > 0 for m in model.modules())

    for xb, yb, meta in tqdm(test_dl, desc="test (MC)"):
        V, T = xb
        mask_bn = meta["entity_mask"]
        y_true, _ = flatten_targets(yb, mask_bn, device)
        if y_true is None:
            continue
        x_hist, _ = build_x_hist_from_T(T, mask_bn, device)
        if use_cov:
            x_cov, _, _ = build_cov_from_V(V, mask_bn, device)
        else:
            x_cov = None

        # Draw samples (if no dropout, this will be identical; still okay)
        samples = []
        for _ in range(num_samples):
            if x_cov is None:
                y_pred = model(x_hist)
            else:
                # detect if model accepts x_cov
                import inspect as _inspect
                if "x_cov" in _inspect.signature(model.forward).parameters:
                    y_pred = model(x_hist, x_cov=x_cov)
                else:
                    y_pred = model(x_hist)
            samples.append(y_pred)
        all_samples = torch.stack(samples, dim=0)  # [S,B,H,1]

        # point (median for MAE, mean for RMSE)
        med = all_samples.median(dim=0).values
        mean = all_samples.mean(dim=0)

        res = med - y_true
        abs_sum += res.abs().sum().item()
        sq_sum  += ((mean - y_true) ** 2).sum().item()
        elts    += res.numel()

        # CRPS
        M = all_samples.shape[0]
        term1 = (all_samples - y_true.unsqueeze(0)).abs().mean(dim=0)  # [B,H,1]
        if M <= 1:
            term2 = torch.zeros_like(term1)
        else:
            diffs = (all_samples.unsqueeze(0) - all_samples.unsqueeze(1)).abs()  # [M,M,B,H,1]
            iu = torch.triu_indices(M, M, offset=1, device=diffs.device)
            diffs_ij = diffs[iu[0], iu[1], ...]                                   # [M*(M-1)/2,B,H,1]
            term2 = (2.0 / (M * (M - 1))) * diffs_ij.mean(dim=0)                  # [B,H,1]
        batch_crps = (term1 - 0.5 * term2).mean().item()
        crps_sum += batch_crps * y_true.size(0)
        n += y_true.size(0)

        # Pinball
        for q in pin_qs:
            y_q = torch.quantile(all_samples, q, dim=0, interpolation="linear")
            diff = y_true - y_q
            qv = torch.tensor(q, device=device)
            loss_q = torch.maximum(qv * diff, (qv - 1.0) * diff)
            pinball_sums[q] += loss_q.sum().item()

    mae = abs_sum / max(1, elts)
    mse = sq_sum  / max(1, elts)
    crps = crps_sum / max(1, n)
    pinball = {q: pinball_sums[q] / max(1, elts) for q in pin_qs}
    return {"crps": crps, "mae": mae, "mse": mse, "pinball": pinball, "had_dropout": had_dropout}

def main():
    import inspect
    import torch
    import torch.nn as nn
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dlinear", choices=["dlinear", "tide"], help="baseline model to train")
    args = parser.parse_args()

    device = set_torch()
    torch.manual_seed(getattr(crypto_config, "SEED", 1337))

    # Data
    train_dl, val_dl, test_dl = run_experiment(split="trainvaltest")

    # Peek batch to infer dims and build model
    xb0, yb0, meta0 = next(iter(train_dl))
    V0, T0 = xb0
    B0, N0, K, F = T0.shape
    H = yb0.shape[-1]
    assert H == getattr(crypto_config, "PRED", H), f"PRED mismatch: {H} vs {getattr(crypto_config, 'PRED', H)}"

    if args.model == "dlinear":
        model = DLinear(seq_len=K, pred_len=H, moving_avg=getattr(crypto_config, "MOVING_AVG", 25)).to(device)
        use_cov = False
    else:
        Fv = V0.shape[-1]
        cov_dim = K * Fv
        model = TiDESmall(
            lookback=K,
            horizon=H,
            d_model=getattr(crypto_config, "MODEL_WIDTH", 256),
            decoder_out=getattr(crypto_config, "DECODER_OUT", 32),
            ne=getattr(crypto_config, "TIDE_NE", 2),
            nd=getattr(crypto_config, "TIDE_ND", 2),
            temporal_hidden=getattr(crypto_config, "TEMPORAL_HIDDEN", 64),
            p_drop=getattr(crypto_config, "DROPOUT", 0.1),
            cov_dim=int(cov_dim),
        ).to(device)
        use_cov = True

    # Optim & sched
    optim = torch.optim.AdamW(model.parameters(), lr=getattr(crypto_config, "BASE_LR", 1e-3),
                              weight_decay=getattr(crypto_config, "WEIGHT_DECAY", 1e-4))
    sched = make_warmup_cosine(optim, warmup_steps=getattr(crypto_config, "WARMUP_STEPS", 100),
                               total_steps=getattr(crypto_config, "TOTAL_STEPS", 10000))

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    out_dir = getattr(crypto_config, "OUT_DIR", "./out")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_best = os.path.join(out_dir, f"{args.model}_best.pt")

    def run_epoch(dataloader, train: bool):
        model.train(train)
        total_loss, n_rows = 0.0, 0
        for xb, yb, meta in tqdm(dataloader, desc="train" if train else "val"):
            V, T = xb
            mask_bn = meta["entity_mask"]
            x_hist, _ = build_x_hist_from_T(T, mask_bn, device)  # [Beff,K,1]
            if use_cov:
                x_cov, _, _ = build_cov_from_V(V, mask_bn, device)  # [Beff,K*Fv]
            else:
                x_cov = None
            y_true, _ = flatten_targets(yb, mask_bn, device)     # [Beff,H,1]
            if y_true is None:
                continue

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                if x_cov is None:
                    y_pred = model(x_hist)
                else:
                    if "x_cov" in inspect.signature(model.forward).parameters:
                        y_pred = model(x_hist, x_cov=x_cov)
                    else:
                        y_pred = model(x_hist)
                loss = F.mse_loss(y_pred, y_true)

            if train:
                optim.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), getattr(crypto_config, "MAX_GRAD_NORM", 1.0))
                scaler.step(optim)
                scaler.update()
                sched.step()

            total_loss += loss.detach().item() * y_true.size(0)
            n_rows += y_true.size(0)
        return total_loss / max(1, n_rows)

    best = float("inf")
    for epoch in range(getattr(crypto_config, "EPOCHS", 10)):
        tr = run_epoch(train_dl, train=True)
        va = run_epoch(val_dl,   train=False)
        print(f"[{args.model}][epoch {epoch}] train_mse={tr:.6f} | val_mse={va:.6f}")
        if va < best:
            torch.save(model.state_dict(), ckpt_best)
            best = va

    # Load best and eval
    if os.path.exists(ckpt_best):
        model.load_state_dict(torch.load(ckpt_best, map_location=device))

    model.eval()
    preds_all, trues_all = [], []
    with torch.no_grad():
        for xb, yb, meta in tqdm(test_dl, desc="test (point)"):
            V, T = xb
            mask_bn = meta["entity_mask"]
            x_hist, _ = build_x_hist_from_T(T, mask_bn, device)
            if use_cov:
                x_cov, _, _ = build_cov_from_V(V, mask_bn, device)
            else:
                x_cov = None
            y_true, _ = flatten_targets(yb, mask_bn, device)
            if y_true is None:
                continue
            if x_cov is None:
                y_pred = model(x_hist)
            else:
                if "x_cov" in inspect.signature(model.forward).parameters:
                    y_pred = model(x_hist, x_cov=x_cov)
                else:
                    y_pred = model(x_hist)
            preds_all.append(y_pred.cpu())
            trues_all.append(y_true.cpu())

    if preds_all:
        y_pred = torch.cat(preds_all, dim=0)
        y_true = torch.cat(trues_all, dim=0)
        mae, mse = evaluate_point(y_true, y_pred)
        rmse = math.sqrt(mse)
    else:
        mae = mse = rmse = float("nan")

    print(f"[{args.model}][test] MAE={mae:.6f} | MSE={mse:.6f} | RMSE={rmse:.6f}")

    # Optional MC-dropout probabilistic eval
    num_samples = int(getattr(crypto_config, "NUM_EVAL_SAMPLES", 0))
    results = {"mae": mae, "mse": mse, "rmse": rmse}
    if num_samples and num_samples > 1:
        prob = evaluate_prob_mc_dropout(model, test_dl, device, num_samples=num_samples, use_cov=use_cov)
        results.update(prob)
        if prob.get("had_dropout", False):
            print(f"[{args.model}][test-MC] CRPS={prob['crps']:.6f} | Pinball(0.1/0.5/0.9)="
                  f"{prob['pinball'][0.1]:.6f}/{prob['pinball'][0.5]:.6f}/{prob['pinball'][0.9]:.6f}")
        else:
            print(f"[{args.model}] Model has no dropout; MC samples will be identical (deterministic).")

    # Per-horizon deterministic metrics
    steps = list(getattr(crypto_config, "EVAL_HORIZON_STEPS", [5,20,63,126]))
    names = list(getattr(crypto_config, "EVAL_HORIZON_NAMES", ["1w","1m","1q","2q"]))
    by_h = evaluate_by_horizon_deterministic(y_true, y_pred, steps, names) if preds_all else {}

    # Save metrics
    os.makedirs(crypto_config.OUT_DIR, exist_ok=True)
    out_main = os.path.join(crypto_config.OUT_DIR, f"test_metrics_{args.model}.json")
    out_h    = os.path.join(crypto_config.OUT_DIR, f"test_metrics_{args.model}_by_h.json")
    with open(out_main, "w") as f:
        json.dump(results, f, indent=2)
    with open(out_h, "w") as f:
        json.dump({"by_horizon": by_h}, f, indent=2)
    print("Saved:", out_main)
    print("Saved:", out_h)

if __name__ == "__main__":
    main()
