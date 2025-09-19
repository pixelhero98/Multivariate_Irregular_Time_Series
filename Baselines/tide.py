
# models/tide_small.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Linear -> ReLU -> Dropout -> Linear + residual (projection if needed) -> LayerNorm
    """
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

class TiDESmall(nn.Module):
    """
    Paper-aligned TiDE-style MLP (small) with past-covariate support.

    Inputs:
      x_hist : [B, L, 1]         (target first-diff history)
      x_cov  : [B, L*Fv] or None (flattened past covariates V)

    Output:
      y_hat  : [B, H, 1]
    """
    def __init__(
        self,
        lookback: int,
        horizon: int,
        d_model: int = 256,
        decoder_out: int = 32,
        ne: int = 2,
        nd: int = 2,
        temporal_hidden: int = 64,
        p_drop: float = 0.1,
        cov_dim: int = 0,
    ):
        super().__init__()
        self.L = int(lookback)
        self.H = int(horizon)
        self.p = int(decoder_out)
        self.d_model = int(d_model)
        self.cov_dim = int(cov_dim)

        # Projectors
        self.proj_hist = ResidualBlock(self.L, self.d_model, hidden_dim=self.d_model, p_drop=p_drop)
        self.proj_cov  = ResidualBlock(self.cov_dim, self.d_model, hidden_dim=self.d_model, p_drop=p_drop) if self.cov_dim > 0 else None

        # Encoder blocks
        enc = []
        for _ in range(max(ne, 0)):
            enc.append(ResidualBlock(self.d_model, self.d_model, hidden_dim=self.d_model, p_drop=p_drop))
        self.encoder = nn.Sequential(*enc) if enc else nn.Identity()

        # Decoder core + projection to H*p
        dec = []
        for _ in range(max(nd - 1, 0)):
            dec.append(ResidualBlock(self.d_model, self.d_model, hidden_dim=self.d_model, p_drop=p_drop))
        self.decoder_core = nn.Sequential(*dec) if dec else nn.Identity()
        self.decoder_out = nn.Linear(self.d_model, self.H * self.p)

        # Temporal decoder p -> 1 (shared across steps)
        self.temporal = ResidualBlock(self.p, 1, hidden_dim=temporal_hidden, p_drop=p_drop)

        # Global linear residual (lookback -> horizon)
        self.global_skip = nn.Linear(self.L, self.H)

    def forward(self, x_hist: torch.Tensor, x_cov: torch.Tensor = None) -> torch.Tensor:
        x = x_hist.squeeze(-1)  # [B,L]
        h = self.proj_hist(x)
        if (self.proj_cov is not None) and (x_cov is not None):
            h = h + self.proj_cov(x_cov)
        e = self.encoder(h)
        dcore = self.decoder_core(e)
        g = self.decoder_out(dcore)             # [B,H*p]
        D = g.view(-1, self.H, self.p)          # [B,H,p]
        y_local = self.temporal(D.reshape(-1, self.p)).view(-1, self.H, 1)
        y_skip  = self.global_skip(x).unsqueeze(-1)
        return y_local + y_skip
