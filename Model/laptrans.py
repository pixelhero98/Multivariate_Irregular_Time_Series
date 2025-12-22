import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

"""
Learnable Laplace transform encoder and a matching pseudo-inverse decoder.

This version implements an *effective* (dt-agnostic, parallel) encoder that
projects a sequence onto a basis of damped sinusoids with learnable poles and
learnable per-mode gains.

Key design choices:
- No exact rollout / explicit convolution logic.
- Supports irregular sampling by evaluating the basis at provided timestamps
  (or by cumulatively summing provided time deltas).
"""

__all__ = ["LaplaceTransformEncoder", "LaplacePseudoInverse"]


class LaplaceTransformEncoder(nn.Module):
    """
    Learnable Laplace transform basis producing 2*k channels (cos & sin).

    Basis functions (per mode k):
        phi_k(t) = exp(-alpha_k * t) * cos(omega_k * t)
        psi_k(t) = exp(-alpha_k * t) * sin(omega_k * t)

    Poles are parameterized as s_k = -alpha_k + i*omega_k with alpha_k > 0.

    The encoder first projects input features into k modal channels and then
    multiplies them by the basis functions. Learnable per-mode gains (c_k, b_k)
    scale the cosine and sine channels.

    Args:
        k:         Number of complex poles (output channels = 2*k).
        feat_dim:  Input feature dimension.
        alpha_min: Positive floor for decay rates to ensure stability.
        omega_max: Clamp for frequencies (imag parts), in rad/unit-time.
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
        if mode.lower() != "effective":
            raise ValueError("Only mode='effective' is supported.")
        self.mode = "effective"
        self.k = int(k)
        self.feat_dim = int(feat_dim)
        self.alpha_min = float(alpha_min)
        self.omega_max = float(omega_max)

        # Trainable pole parameters: s = -alpha + i*omega.
        # alpha is constrained positive via softplus.
        self._s_real_raw = nn.Parameter(torch.empty(self.k))
        self.s_imag = nn.Parameter(torch.empty(self.k))

        # Projection from feature space -> k modal channels.
        self.proj = spectral_norm(nn.Linear(self.feat_dim, self.k, bias=True))

        # Learnable per-mode gains for cosine/sine channels.
        # Keep unconstrained to allow signed gains; initialize near 1.
        self.c_param = nn.Parameter(torch.ones(1, 1, self.k))  # cosine gain c_k
        self.b_param = nn.Parameter(torch.ones(1, 1, self.k))  # sine gain  b_k

        self.reset_parameters()

    @property
    def s_real(self) -> torch.Tensor:
        """Strictly positive decay rate alpha."""
        return F.softplus(self._s_real_raw) + self.alpha_min

    def reset_parameters(self) -> None:
        """Initializes pole and projection parameters."""
        with torch.no_grad():
            # Init frequency in [-π, π]
            nn.init.uniform_(self.s_imag, -math.pi, math.pi)

            # Init decay rate to target a range of [0.01, 0.2]
            target_alpha = torch.empty_like(self._s_real_raw).uniform_(0.01, 0.2)
            y = (target_alpha - self.alpha_min).clamp_min(1e-8)
            self._s_real_raw.copy_(torch.log(torch.expm1(y)))  # inverse softplus

            # Standard init for projection layer.
            w = getattr(self.proj, "weight_orig", self.proj.weight)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            if self.proj.bias is not None:
                bound = 1 / math.sqrt(self.proj.in_features)
                nn.init.uniform_(self.proj.bias, -bound, bound)

            # Gains start at 1; leave as is.

    @staticmethod
    def _prepare_time_like(
        v: torch.Tensor,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Normalize a time-like tensor to shape [B, T, 1] on (device, dtype).

        Accepts shapes: [B,T], [B,T,1], [T], [T,1].
        """
        if v.dim() == 1:
            if v.numel() != seq_len:
                raise ValueError(f"time input has length {v.numel()} but expected T={seq_len}.")
            v = v.view(1, seq_len).expand(batch_size, seq_len)

        elif v.dim() == 2:
            if v.size(1) != seq_len:
                raise ValueError(f"time input has T={v.size(1)} but expected T={seq_len}.")
            if v.size(0) == 1 and batch_size > 1:
                v = v.expand(batch_size, seq_len)
            elif v.size(0) != batch_size:
                raise ValueError(f"time input has B={v.size(0)} but expected B={batch_size}.")

        elif v.dim() == 3:
            if v.size(-1) != 1:
                raise ValueError("time input with 3 dims must have trailing dimension 1 (i.e., [B,T,1]).")
            v = v.squeeze(-1)
            if v.size(1) != seq_len:
                raise ValueError(f"time input has T={v.size(1)} but expected T={seq_len}.")
            if v.size(0) == 1 and batch_size > 1:
                v = v.expand(batch_size, seq_len)
            elif v.size(0) != batch_size:
                raise ValueError(f"time input has B={v.size(0)} but expected B={batch_size}.")

        else:
            raise ValueError("time input must have shape [B,T], [B,T,1], [T] or [T,1].")

        return v.to(device=device, dtype=dtype).unsqueeze(-1)  # [B,T,1]

    def forward(
        self,
        x: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encodes an input sequence into Laplace-domain features.

        Args:
            x:  Input tensor of shape [B, T, D], D == feat_dim.
            dt: Optional time deltas between samples. Shapes: [B,T], [B,T,1], [T], [T,1].
                When provided, we compute relative timestamps by cumulative summation.
            t:  Optional absolute timestamps. Same supported shapes as dt.
                If provided, it takes precedence over dt.

        Returns:
            Tensor of shape [B, T, 2k] (cosine and sine channels concatenated).
        """
        if x.dim() != 3 or x.size(-1) != self.feat_dim:
            raise ValueError(f"x must be of shape [B, T, {self.feat_dim}]")

        B, T, _ = x.shape

        # Prepare relative times in shape [B,T,1]
        if t is not None:
            t_rel = self._prepare_time_like(t, B, T, x.dtype, x.device)
            t_rel = t_rel - t_rel[:, :1]  # shift origin
        elif dt is not None:
            dt_ = self._prepare_time_like(dt, B, T, x.dtype, x.device)
            t_rel = torch.cumsum(dt_, dim=1)
            t_rel = t_rel - t_rel[:, :1]  # ensure starts at 0
        else:
            # Fall back to uniform index grid: 0..T-1
            t_rel = torch.arange(T, device=x.device, dtype=x.dtype).view(1, T, 1).expand(B, T, 1)

        alpha = self.s_real.to(dtype=x.dtype, device=x.device)                       # [k]
        omega = self.s_imag.clamp(-self.omega_max, self.omega_max).to(x.dtype)       # [k]

        # Project features -> k modal channels.
        proj_feats = self.proj(x)  # [B, T, k]

        # Evaluate basis at times.
        # Broadcasting: t_rel [B,T,1] * alpha [k] -> [B,T,k]
        decay = torch.exp(-t_rel * alpha.view(1, 1, -1))
        angle = t_rel * omega.view(1, 1, -1)
        cos_basis = decay * torch.cos(angle)
        sin_basis = decay * torch.sin(angle)

        cosine_response = proj_feats * cos_basis * self.c_param.to(x.dtype)
        sine_response = proj_feats * sin_basis * self.b_param.to(x.dtype)

        return torch.cat([cosine_response, sine_response], dim=-1).contiguous()


class LaplacePseudoInverse(nn.Module):
    """
    Decoder that maps Laplace features back to the original feature space.
    Acts as a pseudo-inverse to LaplaceTransformEncoder.

    Args:
        encoder:          The paired LaplaceTransformEncoder instance.
        hidden_dim:       Width of the optional residual MLP.
        num_layers:       Depth of the optional residual MLP (>= 2 if enabled).
        use_mlp_residual: If True, adds an MLP residual on top of the linear head.
    """

    def __init__(
        self,
        encoder: LaplaceTransformEncoder,
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

        # Complex-aware linear head: y = head_c(C) + head_s(S)
        self.head_c = spectral_norm(nn.Linear(k, D, bias=False))
        self.head_s = spectral_norm(nn.Linear(k, D, bias=False))

        if self.use_mlp_residual:
            if num_layers < 2:
                raise ValueError("num_layers must be >= 2 when use_mlp_residual=True.")
            self.norm = nn.LayerNorm(in_dim)
            layers = [spectral_norm(nn.Linear(in_dim, H)), nn.GELU()]
            for _ in range(num_layers - 2):
                layers += [spectral_norm(nn.Linear(H, H)), nn.GELU()]
            layers += [spectral_norm(nn.Linear(H, D))]
            self.mlp = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            w = getattr(self.head_c, "weight_orig", self.head_c.weight)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            w = getattr(self.head_s, "weight_orig", self.head_s.weight)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))

    def forward(self, lap_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lap_feats: [B, T, 2k] tensor produced by LaplaceTransformEncoder.
        Returns:
            [B, T, D] tensor in the original feature space.
        """
        if lap_feats.dim() != 3 or lap_feats.size(-1) != 2 * self.encoder.k:
            raise ValueError("lap_feats must be of shape [B, T, 2k]")

        k = self.encoder.k
        C, S = lap_feats[..., :k], lap_feats[..., k:]

        y_lin = self.head_c(C) + self.head_s(S)

        if not self.use_mlp_residual:
            return y_lin

        y_mlp = self.mlp(self.norm(lap_feats))
        return y_lin + y_mlp
