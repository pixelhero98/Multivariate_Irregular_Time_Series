
import os, json, math, torch
import crypto_config
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

from Dataset.fin_dataset import run_experiment
from Model.cond_diffusion_utils import (
    set_torch, make_warmup_cosine, flatten_targets
)

# ================= TiDE-small (paper-aligned MLP encoder-decoder, with past covariates) =================
class ResidualBlock(nn.Module):
    \"\"\"
    Paper-style residual block:
      - Linear(in, hidden) -> ReLU
      - Dropout
      - Linear(hidden, out)
      - Linear skip from input to out (projection if dims differ)
      - LayerNorm at output
    \"\"\"
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, p_drop: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(p_drop)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        y = h + (self.skip(x) if not isinstance(self.skip, nn.Identity) else x)
        return self.ln(y)

class TiDESmallPaperAligned(nn.Module):
    \"\"\"
    TiDE-style encoder-decoder with temporal decoder and global linear residual.

    We implement the channel-independent variant (per-series) and *no future covariates* case.
    Past covariates from V (history only) are supported and fused in the encoder.

    Shapes:
      Target history x_hist: [B, L, 1]  -> flatten over time -> [B, L]
      Past covariates x_cov: [B, L*Fv]  (flattened history features)
      Encoder input: proj_hist(x_hist) [+ proj_cov(x_cov)] -> [B, d_model]
      Decoder core:  -> [B, H*p] -> reshape [B, H, p]
      Temporal decoder (shared): p -> 1 per step
      Global linear residual: Linear(L, H) applied to x_hist (flattened) -> [B, H, 1]
    \"\"\"
    def __init__(
        self,
        lookback: int,
        horizon: int,
        d_model: int = 256,
        decoder_out: int = 32,      # p in paper
        ne: int = 2,                # encoder layers (in/out d_model)
        nd: int = 2,                # decoder layers (in/out d_model before final proj)
        temporal_hidden: int = 64,
        p_drop: float = 0.1,
        cov_dim: int = 0,           # flattened past covariates dim (K * Fv)
    ):
        super().__init__()
        self.L = int(lookback)
        self.H = int(horizon)
        self.p = int(decoder_out)
        self.d_model = int(d_model)
        self.cov_dim = int(cov_dim)

        # Project target history [B,L] -> [B,d_model]
        self.proj_hist = ResidualBlock(self.L, self.d_model, hidden_dim=self.d_model, p_drop=p_drop)
        # Optional covariate projector [B, cov_dim] -> [B,d_model]
        self.proj_cov  = ResidualBlock(self.cov_dim, self.d_model, hidden_dim=self.d_model, p_drop=p_drop) if self.cov_dim > 0 else None

        # Encoder: stack ne residual blocks with in/out = d_model
        enc = []
        for _ in range(max(ne, 0)):
            enc.append(ResidualBlock(self.d_model, self.d_model, hidden_dim=self.d_model, p_drop=p_drop))
        self.encoder = nn.Sequential(*enc) if enc else nn.Identity()

        # Decoder: nd-1 residual blocks in d_model space, then Linear to H*p
        dec = []
        for _ in range(max(nd - 1, 0)):
            dec.append(ResidualBlock(self.d_model, self.d_model, hidden_dim=self.d_model, p_drop=p_drop))
        self.decoder_core = nn.Sequential(*dec) if dec else nn.Identity()
        self.decoder_out = nn.Linear(self.d_model, self.H * self.p)

        # Temporal decoder: per-step residual block mapping p -> 1 (shared across steps)
        self.temporal = ResidualBlock(self.p, 1, hidden_dim=temporal_hidden, p_drop=p_drop)

        # Global linear residual (lookback -> horizon)
        self.global_skip = nn.Linear(self.L, self.H)

    def forward(self, x_hist: torch.Tensor, x_cov: torch.Tensor = None) -> torch.Tensor:
        # x_hist: [B,L,1] -> flatten time
        x = x_hist.squeeze(-1)       # [B,L]
        # Projections
        h = self.proj_hist(x)        # [B,d_model]
        if (self.proj_cov is not None) and (x_cov is not None):
            h = h + self.proj_cov(x_cov)  # fuse past covariates

        # Encoder
        e = self.encoder(h)          # [B,d_model]

        # Decoder dense -> [B,H,p]
        dcore = self.decoder_core(e) # [B,d_model]
        g = self.decoder_out(dcore)  # [B,H*p]
        D = g.view(-1, self.H, self.p)  # [B,H,p]

        # Temporal decoder per step (shared weights): flatten B*H, apply block p->1
        y_local = self.temporal(D.reshape(-1, self.p)).view(-1, self.H, 1)  # [B,H,1]

        # Global linear residual
        y_skip = self.global_skip(x).unsqueeze(-1)  # [B,H,1]

        return y_local + y_skip

# ================= Helpers =================
def build_x_hist_from_T(T: torch.Tensor, mask_bn: torch.Tensor, device: torch.device):
    \"\"\"
    Build univariate history inputs from first-difference feature (T[...,0]).
    Returns x_hist [Beff, L, 1]
    \"\"\"
    B, N, K, F = T.shape
    t0 = T[..., 0].reshape(B * N, K)
    m_flat = mask_bn.reshape(B * N)
    x_hist = t0[m_flat].unsqueeze(-1).to(device)  # [Beff,L,1]
    return x_hist, K

def build_cov_from_V(V: torch.Tensor, mask_bn: torch.Tensor, device: torch.device):
    \"\"\"
    Build flattened past covariates from V history (all features, all times).
    Returns x_cov [Beff, L*Fv], and (L, Fv)
    \"\"\"
    B, N, K, Fv = V.shape
    v_flat = V.reshape(B * N, K * Fv)
    m_flat = mask_bn.reshape(B * N)
    x_cov = v_flat[m_flat].to(device)  # [Beff, L*Fv]
    return x_cov, K, Fv

def evaluate_point(y_true: torch.Tensor, y_pred: torch.Tensor):
    \"\"\"
    y_true, y_pred: [Beff,H,1]
    Returns MAE, MSE
    \"\"\"
    err = (y_pred - y_true)
    mae = err.abs().mean().item()
    mse = (err**2).mean().item()
    return mae, mse

@torch.no_grad()
def evaluate_prob_mc_dropout(model: nn.Module, test_dl, device, num_samples: int = 32):
    \"\"\"
    Monte Carlo dropout sampling to get distributional metrics:
    - CRPS via all-pairs estimator
    - Pinball at (0.1, 0.5, 0.9)
    \"\"\"
    model.train()  # keep dropout ON for MC sampling
    crps_sum, n = 0.0, 0
    abs_sum, sq_sum, elts = 0.0, 0.0, 0
    pin_qs = (0.1, 0.5, 0.9)
    pinball_sums = {q: 0.0 for q in pin_qs}

    for xb, yb, meta in tqdm(test_dl, desc="test (TiDE-MC)"):
        V, T = xb
        mask_bn = meta["entity_mask"]

        y_true, _ = flatten_targets(yb, mask_bn, device)  # [Beff,H,1]
        if y_true is None:
            continue
        B, H, C = y_true.size()

        x_hist, _ = build_x_hist_from_T(T, mask_bn, device)           # [Beff,L,1]
        x_cov, _, _ = build_cov_from_V(V, mask_bn, device)            # [Beff,L*Fv]

        # Draw samples
        samples = []
        for _ in range(num_samples):
            y_pred = model(x_hist, x_cov=x_cov)  # [Beff,H,1]
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
        crps_sum += batch_crps * B
        n += B

        # Pinball
        for q in pin_qs:
            y_q = torch.quantile(all_samples, q, dim=0, interpolation="linear")
            diff = y_true - y_q
            loss_q = torch.maximum(torch.tensor(q, device=device) * diff,
                                   torch.tensor(q-1.0, device=device) * diff)
            pinball_sums[q] += loss_q.sum().item()

    mae = abs_sum / max(1, elts)
    mse = sq_sum  / max(1, elts)
    crps = crps_sum / max(1, n)
    pinball = {q: pinball_sums[q] / max(1, elts) for q in pin_qs}
    return {"crps": crps, "mae": mae, "mse": mse, "pinball": pinball}

# ================= Train/Val/Test =================
def main():
    device = set_torch()
    torch.manual_seed(crypto_config.SEED)

    # Data
    train_dl, val_dl, test_dl = run_experiment(split="trainvaltest")

    # Infer dims (and cov_dim) from a peek batch
    xb0, yb0, meta0 = next(iter(train_dl))
    V0, T0 = xb0
    B0, N0, K, F = T0.shape
    H = yb0.shape[-1]
    Fv = V0.shape[-1]
    cov_dim = K * Fv

    assert H == crypto_config.PRED, f"PRED mismatch: {H} vs {crypto_config.PRED}"
    print(f"TiDE-small (paper-aligned) | L={K}, H={H}, cov_dim={cov_dim}")

    # Model
    model = TiDESmallPaperAligned(
        lookback=K,
        horizon=H,
        d_model=getattr(crypto_config, 'MODEL_WIDTH', 256),
        decoder_out=getattr(crypto_config, 'DECODER_OUT', 32),
        ne=getattr(crypto_config, 'TIDE_NE', 2),
        nd=getattr(crypto_config, 'TIDE_ND', 2),
        temporal_hidden=getattr(crypto_config, 'TEMPORAL_HIDDEN', 64),
        p_drop=getattr(crypto_config, 'DROPOUT', 0.1),
        cov_dim=int(cov_dim),
    ).to(device)

    # Optim & sched
    optim = torch.optim.AdamW(model.parameters(), lr=crypto_config.BASE_LR, weight_decay=crypto_config.WEIGHT_DECAY)
    sched = make_warmup_cosine(optim, warmup_steps=crypto_config.WARMUP_STEPS, total_steps=crypto_config.TOTAL_STEPS)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    out_dir = getattr(crypto_config, "OUT_DIR", "./out")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_best = os.path.join(out_dir, "tide_small_paper_best.pt")

    def run_epoch(dataloader, train: bool):
        model.train(train)
        total_loss, n_rows = 0.0, 0
        for xb, yb, meta in tqdm(dataloader, desc="train" if train else "val"):
            V, T = xb
            mask_bn = meta["entity_mask"]
            x_hist, _ = build_x_hist_from_T(T, mask_bn, device)     # [Beff,L,1]
            x_cov, _, _ = build_cov_from_V(V, mask_bn, device)      # [Beff,L*Fv]
            y_true, _ = flatten_targets(yb, mask_bn, device)        # [Beff,H,1]
            if y_true is None:
                continue

            with autocast(enabled=(device.type == "cuda")):
                y_pred = model(x_hist, x_cov=x_cov)                 # [Beff,H,1]
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
    for epoch in range(crypto_config.EPOCHS):
        tr = run_epoch(train_dl, train=True)
        va = run_epoch(val_dl,   train=False)
        print(f"[epoch {epoch}] train_mse={tr:.6f} | val_mse={va:.6f}")
        if va < best:
            torch.save(model.state_dict(), ckpt_best)
            best = va

    # Load best
    if os.path.exists(ckpt_best):
        model.load_state_dict(torch.load(ckpt_best, map_location=device))

    # Test deterministic metrics
    model.eval()
    mae_sum = mse_sum = 0.0
    n_rows = 0
    with torch.no_grad():
        for xb, yb, meta in tqdm(test_dl, desc="test (point)"):
            V, T = xb
            mask_bn = meta["entity_mask"]
            x_hist, _ = build_x_hist_from_T(T, mask_bn, device)
            x_cov, _, _ = build_cov_from_V(V, mask_bn, device)
            y_true, _ = flatten_targets(yb, mask_bn, device)
            if y_true is None:
                continue
            y_pred = model(x_hist, x_cov=x_cov)
            mae, mse = evaluate_point(y_true, y_pred)
            mae_sum += mae * y_true.size(0)
            mse_sum += mse * y_true.size(0)
            n_rows  += y_true.size(0)
    mae = mae_sum / max(1, n_rows)
    mse = mse_sum / max(1, n_rows)
    print(f"[TiDE-small][test] MAE={mae:.6f} | MSE={mse:.6f} | RMSE={math.sqrt(mse):.6f}")

    # Optional probabilistic eval via MC-dropout
    num_samples = int(getattr(crypto_config, "NUM_EVAL_SAMPLES", 0))
    mc_samples = num_samples if num_samples and num_samples > 1 else 0
    results = {"mae": mae, "mse": mse, "rmse": math.sqrt(mse)}
    if mc_samples:
        prob = evaluate_prob_mc_dropout(model, test_dl, device, num_samples=mc_samples)
        results.update(prob)
        print(f"[TiDE-small][test-MC] CRPS={prob['crps']:.6f} | Pinball(0.1/0.5/0.9)="
              f"{prob['pinball'][0.1]:.6f}/{prob['pinball'][0.5]:.6f}/{prob['pinball'][0.9]:.6f}")

    with open(os.path.join(out_dir, "test_metrics_tide_small.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved:", os.path.join(out_dir, "test_metrics_tide_small.json"))

if __name__ == "__main__":
    main()
