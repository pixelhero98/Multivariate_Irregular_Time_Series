import math, torch
import torch.nn as nn
import torch.nn.functional as F

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
