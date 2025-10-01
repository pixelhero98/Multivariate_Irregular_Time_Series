import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

"""
This module implements a learnable Laplace transform encoder and a corresponding
decoder, based on the theory of dt-marginalization.

- LearnableLaplaceTrans: Can operate in two modes:
    1. 'effective': A fast, parallel, dt-agnostic path that projects onto a
       basis of damped exponentials defined by "effective" poles. These poles
       have statistically absorbed the effect of irregular time gaps.
    2. 'exact': A slower, recurrent path that computes the exact Zero-Order-Hold
       (ZOH) solution for each specific time gap, making it dt-sensitive.

- PseudoInverse: A complex-aware linear head that maps the Laplace features
  from the encoder back to the original feature space.
"""

class LearnableLaplaceTrans(nn.Module):
    """
    Learnable Laplace transform basis producing 2*k channels (cos & sin).

    This encoder implements the core logic of transforming a time series into
    the Laplace domain using learnable poles.

    Args:
        k:         Number of complex poles (output channels = 2*k).
        feat_dim:  Input feature dimension.
        mode:      'effective' (dt-agnostic parallel path) or
                   'exact' (dt-sensitive recurrent path).
        alpha_min: A small positive floor for the real part of the poles to
                   ensure stability (decay).
        omega_max: Clamp for the imaginary part (frequency) of the poles.
    """
    def __init__(
        self,
        k: int,
        feat_dim: int,
        mode: str = "effective",
        alpha_min: float = 1e-6,
        omega_max: float = math.pi,
    ) -> None:
        super().__init__()
        if mode.lower() not in {"effective", "exact"}:
            raise ValueError("mode must be 'effective' or 'exact'.")
        self.mode = mode.lower()
        self.k = int(k)
        self.feat_dim = int(feat_dim)
        self.alpha_min = float(alpha_min)
        self.omega_max = float(omega_max)

        # Trainable pole parameters: s = -alpha + i*omega
        # Raw parameter for the real part (decay), becomes > 0 via softplus.
        self._s_real_raw = nn.Parameter(torch.empty(k))
        # Imaginary part (frequency).
        self.s_imag = nn.Parameter(torch.empty(k))

        # Optional learnable global time-scale for the 'exact' recurrent path.
        self._tau = nn.Parameter(torch.tensor(0.0))

        # Linear layer to project input features to the k modes.
        self.proj = spectral_norm(nn.Linear(self.feat_dim, k, bias=True))

        # Per-mode nonnegative input gain (b_k >= 0).
        init_val = math.log(math.e - 1.0)  # softplus(init_val) ≈ 1
        self.b_param = nn.Parameter(torch.full((1, 1, k), init_val))

        self.reset_parameters()

    @property
    def s_real(self) -> torch.Tensor:
        """Returns the strictly positive real part of the poles (decay rate alpha)."""
        return F.softplus(self._s_real_raw) + self.alpha_min

    def reset_parameters(self) -> None:
        """Initializes the model parameters."""
        with torch.no_grad():
            # Init frequency in [-π, π]
            nn.init.uniform_(self.s_imag, -math.pi, math.pi)
            # Init decay rate to target a range of [0.01, 0.2]
            target_alpha = torch.empty_like(self._s_real_raw).uniform_(0.01, 0.2)
            y = (target_alpha - self.alpha_min).clamp_min(1e-8)
            self._s_real_raw.copy_(torch.log(torch.expm1(y)))  # Inverse softplus

            # Standard init for the projection layer.
            w = getattr(self.proj, "weight_orig", self.proj.weight)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            if self.proj.bias is not None:
                bound = 1 / math.sqrt(self.proj.in_features)
                nn.init.uniform_(self.proj.bias, -bound, bound)

    def forward(
        self,
        x: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert x.dim() == 3 and x.size(-1) == self.feat_dim, f"x must be [B, T, {self.feat_dim}]"
        B, T, _ = x.shape
        k = self.k

        # --------------------------------------------------------------------
        # --- 'effective' mode: dt-agnostic parallel projection (Fast Path) ---
        # --------------------------------------------------------------------
        if self.mode == "effective":
            # This path models the system using effective poles that have
            # absorbed the statistics of the time gaps (dt-marginalization).
            # It is dt-agnostic at runtime and computes in parallel.
            alpha = self.s_real.to(x.dtype)
            omega = self.s_imag.clamp(-self.omega_max, self.omega_max).to(x.dtype)
            s = torch.complex(-alpha, omega)  # Pole: s = -α + iω

            # Create the exponential basis functions: exp(s*t)
            t_idx = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(1)
            expo_basis = torch.exp(t_idx * s.unsqueeze(0)) # [T, k]
            re_basis, im_basis = expo_basis.real, expo_basis.imag

            # Project input onto the basis.
            proj_feats = self.proj(x)  # [B, T, k]
            
            # Real part is projection on cosine, Imaginary part on sine.
            C = proj_feats * re_basis.unsqueeze(0)
            S = proj_feats * im_basis.unsqueeze(0)
            return torch.cat([C, S], dim=2).contiguous()

        # ------------------------------------------------------------------
        # --- 'exact' mode: dt-sensitive recurrent update (Precise Path) ---
        # ------------------------------------------------------------------
        # This path computes the exact Zero-Order Hold (ZOH) solution for
        # each irregular time step dt. It is computationally
        # more expensive due to its sequential nature.
        tau = F.softplus(self._tau) + 1e-3
        alpha = (self.s_real * tau).to(x.dtype)
        omega = (self.s_imag * tau).clamp(-self.omega_max, self.omega_max).to(x.dtype)

        # Prepare dt to be [B, T, 1] for broadcasting.
        if dt is None:
            dt_val = 1.0 / max(T - 1, 1)
            dt = x.new_full((B, T, 1), dt_val)
        else:
            dt = dt.to(x.dtype).view(dt.shape[0] if dt.dim() > 1 else 1, T, 1)
            if dt.shape[0] == 1: dt = dt.expand(B, -1, -1)
        
        # Per-timestep drive signal.
        u = self.proj(x)  # [B, T, k]

        # Step-wise decay (rho) and rotation (theta) based on exact dt.
        rho = torch.exp(-alpha.view(1, 1, k) * dt)
        theta = omega.view(1, 1, k) * dt
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)

        # Calculate the exact ZOH input map coefficients (Psi).
        alpha_sq = alpha**2
        omega_sq = omega**2
        den = (alpha_sq + omega_sq).clamp_min(1e-6).view(1, 1, k)
        
        # This corresponds to integrating exp(A*t)*B from 0 to dt.
        psi_c = (-omega + rho * (alpha * sin_t + omega * cos_t)) / den
        psi_s = ( alpha - rho * (alpha * cos_t - omega * sin_t)) / den

        # Apply nonnegative input gain.
        b = F.softplus(self.b_param).to(x.dtype) + 1e-8
        u_eff = u * b

        # Sequential rollout.
        c_hist, s_hist = [], []
        c = torch.zeros(B, k, device=x.device, dtype=x.dtype)
        s = torch.zeros(B, k, device=x.device, dtype=x.dtype)
        
        for t in range(T):
            # Homogeneous part: state evolution without input.
            c_new = rho[:, t] * (c * cos_t[:, t] - s * sin_t[:, t])
            s_new = rho[:, t] * (c * sin_t[:, t] + s * cos_t[:, t])
            # Input part: add the effect of the input over the interval.
            c = c_new + psi_c[:, t] * u_eff[:, t]
            s = s_new + psi_s[:, t] * u_eff[:, t]
            c_hist.append(c)
            s_hist.append(s)

        C = torch.stack(c_hist, dim=1)
        S = torch.stack(s_hist, dim=1)
        return torch.cat([C, S], dim=2).contiguous()

# =====================================
# PseudoInverser (NOT a strict inverse)
# =====================================
class PseudoInverse(nn.Module):
    """
    Decoder that maps Laplace features back to the original feature space.
    This acts as a pseudo-inverse to the LearnableLaplaceTrans.

    Args:
        encoder:          The paired LearnableLaplaceTrans instance.
        hidden_dim:       Width of the optional residual MLP.
        num_layers:       Depth of the optional residual MLP.
        use_mlp_residual: If True, adds an MLP on top of the linear head.
    """

    def __init__(
        self,
        encoder: LearnableLaplaceTrans,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        use_mlp_residual: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.use_mlp_residual = bool(use_mlp_residual)

        k = encoder.k
        D = encoder.feat_dim
        in_dim = 2 * k
        H = int(hidden_dim if hidden_dim is not None else max(in_dim, D))

        # --- Complex-aware linear head ---
        # This performs a linear transform on a complex number (C + iS)
        # by applying separate weights to the real (C) and imaginary (S) parts.
        # The output is y = head_c(C) + head_s(S).
        self.head_c = spectral_norm(nn.Linear(k, D, bias=False))
        self.head_s = spectral_norm(nn.Linear(k, D, bias=False))

        # --- Optional residual MLP for non-linear corrections ---
        if self.use_mlp_residual:
            self.norm = nn.LayerNorm(in_dim)
            layers = [spectral_norm(nn.Linear(in_dim, H)), nn.GELU()]
            for _ in range(num_layers - 2):
                layers += [spectral_norm(nn.Linear(H, H)), nn.GELU()]
            layers += [spectral_norm(nn.Linear(H, D))]
            self.mlp = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes decoder parameters."""
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.head_c.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.head_s.weight, a=math.sqrt(5))

    def forward(self, lap_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lap_feats: [B, T, 2k] tensor from the LaplaceEncoder.
        Returns:
            y: [B, T, D] tensor in the original feature space.
        """
        k = self.encoder.k
        C, S = lap_feats.chunk(2, dim=-1)  # Split into cos/sin channels

        # Linear readout combines the two channels.
        y_lin = self.head_c(C) + self.head_s(S)

        if not self.use_mlp_residual:
            return y_lin

        # Add non-linear residual correction.
        y_mlp = self.mlp(self.norm(lap_feats))
        return y_lin + y_mlp
