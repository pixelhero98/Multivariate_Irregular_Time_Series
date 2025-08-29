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

        # Reparameterized real part: raw (unconstrained) -> positive via softplus + alpha_min
        self._s_real_raw = nn.Parameter(torch.empty(k))
        self.s_imag = nn.Parameter(torch.empty(k))
        self.reset_parameters()

        self.proj = spectral_norm(
            nn.Linear(feat_dim, k, bias=True),
            n_power_iterations=1, eps=1e-7
        )
        self._tau = nn.Parameter(torch.tensor(0.0))  # softplus -> positive scale, no exploding patterns

    # Expose positive alpha via property (backward flows to _s_real_raw)
    @property
    def s_real(self) -> torch.Tensor:
        return F.softplus(self._s_real_raw) + self.alpha_min  # strictly > alpha_min

    def reset_parameters(self):
        # Initialize s_imag as before
        nn.init.uniform_(self.s_imag, -math.pi, math.pi)  # ω init

        # Initialize raw real-part so that s_real ≈ U[0.01, 0.2]
        with torch.no_grad():
            target_alpha = torch.empty_like(self._s_real_raw).uniform_(0.01, 0.2)
            y = (target_alpha - self.alpha_min).clamp_min(1e-8)  # softplus(raw) = y > 0
            self._s_real_raw.copy_(torch.log(torch.expm1(y)))    # raw = softplus^{-1}(y)

    def forward(self, x: torch.Tensor,
                dt: torch.Tensor | None = None,
                alpha_mod: torch.Tensor | None = None,
                omega_mod: torch.Tensor | None = None,
                tau_mod: torch.Tensor | None = None) -> torch.Tensor:
        """
        Stable, driven damped-sinusoid bank (linear time-*varying* in complex-domain):
          z_t = [c_t, s_t],   z_t = rho_t * R(theta_t) * z_{t-1} + [u_t, 0]
        Returns [B, T, 2k] = concat([c_t, s_t]) per pole k.
        """
        B, T, D = x.shape
        k = self.k

        # Base pole params with learnable time scale
        tau = F.softplus(self._tau) + 1e-3              # scalar > 0
        alpha0 = self.s_real * tau                       # [k], strictly > alpha_min * tau
        omega0 = self.s_imag * tau                       # [k] (unconstrained sign)

        # Prepare dt to shape [B,T,1]
        if dt is None:
            base = (1.0 / max(T - 1, 1)) if T > 1 else 1.0
            dt_bt1 = x.new_full((B, T, 1), base)
        else:
            if dt.dim() == 1:   # [T] -> [B,T,1]
                dt_bt1 = dt.view(1, T, 1).to(dtype=x.dtype, device=x.device).expand(B, T, 1)
            elif dt.dim() == 2: # [B,T] -> [B,T,1]
                dt_bt1 = dt.unsqueeze(-1).to(dtype=x.dtype, device=x.device)
            else:
                raise ValueError("dt must be [T] or [B,T] if provided")

        # Expand poles to [B,T,k] and apply optional modulation (log-space)
        alpha = alpha0.view(1,1,k).expand(B,T,k)
        omega = omega0.view(1,1,k).expand(B,T,k)
        if alpha_mod is not None:
            alpha = alpha * alpha_mod.to(dtype=x.dtype, device=x.device).exp()
        if omega_mod is not None:
            omega = omega * omega_mod.to(dtype=x.dtype, device=x.device).exp()
        if tau_mod is not None:
            scale = tau_mod.to(dtype=x.dtype, device=x.device).exp()
            alpha = alpha * scale
            omega = omega * scale

        # Per-step decay & rotation
        rho   = torch.exp(-alpha * dt_bt1)            # [B,T,k]
        theta = omega * dt_bt1                        # [B,T,k]
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)

        # Drive (shared linear projection)
        u = self.proj(x)  # [B, T, k]

        # Recurrence with time-varying parameters
        C = torch.empty(B, T, k, dtype=u.dtype, device=x.device)
        S = torch.empty(B, T, k, dtype=u.dtype, device=x.device)
        c = torch.zeros(B, k, device=x.device, dtype=u.dtype)
        s = torch.zeros(B, k, device=x.device, dtype=u.dtype)
        # before the loop, keep c_prev, s_prev aliases
        for t in range(T):
            c_prev, s_prev = c, s
            rt = rho[:, t, :];
            ct = cos_t[:, t, :];
            st = sin_t[:, t, :]
            c = rt * (c_prev * ct - s_prev * st) + u[:, t, :]
            s = rt * (c_prev * st + s_prev * ct)
            C[:, t, :] = c
            S[:, t, :] = s

        # for t in range(T):
        #     rt = rho[:, t, :]
        #     ct = cos_t[:, t, :]
        #     st = sin_t[:, t, :]
        #     c = rt * (c * ct - s * st) + u[:, t, :]
        #     s = rt * (c * st + s * ct)
        #     C[:, t, :] = c
        #     S[:, t, :] = s
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