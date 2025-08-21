import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# --------------------------------------------------------------
# Laplace basis (normalized time) with dt or optional modulation
# --------------------------------------------------------------

class LearnableLaplacianBasis(nn.Module):
    """
    x:[B, T, D] -> Laplace features:[B, T, 2k] using learnable complex poles.
    Normalized time t in [0,1], with learnable global timescale τ.
    Supports per-step dt and optional pole modulation (A & C changes).
    """
    def __init__(self, k: int, feat_dim: int, alpha_min: float = 1e-6):
        super().__init__()
        self.k = k
        self.alpha_min = alpha_min

        self.s_real = nn.Parameter(torch.empty(k))
        self.s_imag = nn.Parameter(torch.empty(k))
        self.reset_parameters()

        self.proj = spectral_norm(
            nn.Linear(feat_dim, k, bias=True),
            n_power_iterations=1, eps=1e-7
        )
        self._tau = nn.Parameter(torch.tensor(0.0))  # softplus -> positive scale, no exploding patterns

    def reset_parameters(self):
        nn.init.uniform_(self.s_real, 0.01, 0.2)    # α > 0
        nn.init.uniform_(self.s_imag, -math.pi, math.pi)  # ω init

    def forward(self, x: torch.Tensor,
                dt: torch.Tensor | None = None,
                alpha_mod: torch.Tensor | None = None,
                omega_mod: torch.Tensor | None = None,
                tau_mod: torch.Tensor | None = None) -> torch.Tensor:
        """
        Stable, driven damped-sinusoid bank (linear time-*varying* in complex-domain):
          z_t = [c_t, s_t],   z_t = rho_t * R(theta_t) * z_{t-1} + [u_t, 0]
        Returns [B, T, 2k] = concat([c_t, s_t]) per pole k.

        Args:
            x:          [B,T,D] input drive
            dt:         optional [B,T] or [T] step sizes. If None, uses 1/(T-1).
            alpha_mod:  optional [B,T,1] multiplier in log-space (exp applied)
            omega_mod:  optional [B,T,1] multiplier in log-space (exp applied)
            tau_mod:    optional [B,T,1] shared scale mod (exp applied)
        """
        B, T, D = x.shape
        device = x.device
        k = self.k

        # Base pole params with learnable time scale
        tau = F.softplus(self._tau) + 1e-3             # scalar > 0
        alpha0 = self.s_real.clamp_min(self.alpha_min) * tau   # [k]
        omega0 = self.s_imag * tau                              # [k]

        # Prepare dt to shape [B,T,1]
        if dt is None:
            base = (1.0 / max(T - 1, 1)) if T > 1 else 1.0
            dt_bt1 = x.new_full((B, T, 1), base)
        else:
            if dt.dim() == 1:   # [T] -> [B,T,1]
                dt_bt1 = dt.view(1, T, 1).to(x).expand(B, T, 1)
            elif dt.dim() == 2: # [B,T] -> [B,T,1]
                dt_bt1 = dt.unsqueeze(-1).to(x)
            else:
                raise ValueError("dt must be [T] or [B,T] if provided")

        # Expand poles to [B,T,k] and apply optional modulation (log-space)
        alpha = alpha0.view(1,1,k).expand(B,T,k)
        omega = omega0.view(1,1,k).expand(B,T,k)
        if alpha_mod is not None:
            alpha = alpha * alpha_mod.to(x).exp()
        if omega_mod is not None:
            omega = omega * omega_mod.to(x).exp()
        if tau_mod is not None:
            scale = tau_mod.to(x).exp()
            alpha = alpha * scale
            omega = omega * scale

        # Per-step decay & rotation
        rho   = torch.exp(-alpha * dt_bt1)            # [B,T,k]
        theta = omega * dt_bt1                        # [B,T,k]
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)

        # Drive (shared linear projection)
        u = self.proj(x)  # [B, T, k]

        # Recurrence with time-varying parameters
        c = torch.zeros(B, k, device=device, dtype=u.dtype)
        s = torch.zeros(B, k, device=device, dtype=u.dtype)
        out_c, out_s = [], []
        for t in range(T):
            rt = rho[:, t, :]      # [B,k]
            ct = cos_t[:, t, :]
            st = sin_t[:, t, :]
            c_new = rt * (c * ct - s * st) + u[:, t, :]
            s_new = rt * (c * st + s * ct)
            c, s = c_new, s_new
            out_c.append(c)
            out_s.append(s)
        C = torch.stack(out_c, dim=1)  # [B,T,k]
        S = torch.stack(out_s, dim=1)  # [B,T,k]
        return torch.cat([C, S], dim=2).contiguous()  # [B,T,2k]


class LearnableInverseLaplacianBasis(nn.Module):
    """
    Readout from Laplace states [B,T,2k] -> time features [B,T,D].
    Not an exact inverse; it learns whatever mapping helps downstream.
    """
    def __init__(self, laplace_basis: LearnableLaplacianBasis):
        super().__init__()
        self.lap = laplace_basis
        D  = laplace_basis.proj.in_features
        k  = laplace_basis.k
        C  = 2 * k

        # Light, well-conditioned projector (LN + MLP)
        self.norm = nn.LayerNorm(C)
        self.fc1  = spectral_norm(nn.Linear(C, max(C, D)), n_power_iterations=1, eps=1e-7)
        self.act  = nn.GELU()
        self.fc2  = spectral_norm(nn.Linear(max(C, D), D), n_power_iterations=1, eps=1e-7)

        # -------- Warm start that survives SpectralNorm --------
        # - fc1: pass-through for the first C channels (cos/sin)
        # - fc2: copy analysis proj^T into the first k columns (cosine block)
        with torch.no_grad():
            # If SN wrapped these layers, write to weight_orig (not weight)
            w1 = getattr(self.fc1, "weight_orig", self.fc1.weight)  # [H, C], H = max(C, D)
            w2 = getattr(self.fc2, "weight_orig", self.fc2.weight)  # [D, H]

            if getattr(self.fc2, "bias", None) is not None:
                self.fc2.bias.zero_()

            # Zero then seed
            w1.zero_()
            w2.zero_()

            # fc1: identity pass-through on the first C channels
            H = w1.shape[0]
            eye = torch.eye(C, device=w1.device, dtype=w1.dtype)
            w1[:min(H, C), :C].copy_(eye[:min(H, C), :])

            # fc2: put analysis.proj^T in the first k columns (cos block)
            # Note: due to spectral_norm scaling, the effective weight may be a scaled version, which is fine.
            W = self.lap.proj.weight.data  # [k, D]
            if w2.shape[1] >= k:
                w2[:, :k].copy_(W.t())

    def forward(self, lap_feats: torch.Tensor) -> torch.Tensor:
        # lap_feats: [B,T,2k]
        h = self.norm(lap_feats)
        h = self.act(self.fc1(h))
        x_hat = self.fc2(h)  # [B,T,D]
        return x_hat


# if __name__ == "__main__":
#     torch.manual_seed(0)
#     B, T, D, k = 2, 5, 16, 8
#
#     lap = LearnableLaplacianBasis(k=k, feat_dim=D)
#     inv = LearnableInverseLaplacianBasis(lap)
#
#     x = torch.randn(B, T, D)
#     lap_feats = lap(x)           # [B, T, 2k]
#     x_hat     = inv(lap_feats)   # [B, T, D]
#
#     print("Laplace features shape:", lap_feats.shape)  # torch.Size([2, 5, 16])
#     print("Inverse output shape  :", x_hat.shape)      # torch.Size([2, 5, 16])
#
#     # Gradient flow check
#     loss = x_hat.pow(2).mean()
#     loss.backward()
#     print("Backward OK (no errors)")
