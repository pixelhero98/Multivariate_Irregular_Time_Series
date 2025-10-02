import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

"""
This module implements a learnable Laplace transform encoder and a matching
decoder based on dt-marginalisation.

- :class:`LaplaceTransformEncoder`:
    1. ``"effective"`` – a dt-agnostic, parallel encoder that projects onto a
       basis of damped exponentials whose poles have absorbed statistics of the
       sampling gaps.
    2. ``"exact"`` – a dt-sensitive, sequential encoder that integrates the
       Zero-Order-Hold (ZOH) solution for each provided time step.

- :class:`LaplacePseudoInverse`: A complex-aware linear head (optionally
  augmented by an MLP) that maps the Laplace-domain representation back to the
  original feature space.
"""

__all__ = ["LaplaceTransformEncoder", "LaplacePseudoInverse"]


class LaplaceTransformEncoder(nn.Module):
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
        """Encodes an input sequence into Laplace-domain features.

        Args:
            x: Input tensor of shape ``[B, T, D]`` where ``D`` equals
                ``feat_dim``.
            dt: Optional tensor describing the timestep between successive
                samples. Supported shapes are ``[B, T]``, ``[B, T, 1]``,
                ``[T]`` or ``[T, 1]``. When ``None`` a uniform grid is assumed.

        Returns:
            Tensor of shape ``[B, T, 2k]`` containing stacked cosine and sine
            responses for each pole.
        """

        if x.dim() != 3 or x.size(-1) != self.feat_dim:
            raise ValueError(f"x must be of shape [B, T, {self.feat_dim}]")

        B, T, _ = x.shape
        k = self.k

        if self.mode == "effective":
            return self._forward_effective(x, T)

        return self._forward_exact(x, dt, B, T, k)

    # ------------------------------------------------------------------
    # --- helper paths -------------------------------------------------
    # ------------------------------------------------------------------
    def _forward_effective(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Parallel, dt-agnostic projection using effective poles."""

        alpha = self.s_real.to(x.dtype)
        omega = self.s_imag.clamp(-self.omega_max, self.omega_max).to(x.dtype)

        proj_feats = self.proj(x)  # Shape: [B, T, k]

        # --- Basis Calculation ---
        time_index = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(1)

        # Deconstruct the complex exponentiation using Euler's formula and torch.sincos.
        # This avoids creating a ComplexHalf tensor and is more efficient.
        # e^(t*s) = e^(t*(-α+iω)) = e^(-tα) * (cos(tω) + i*sin(tω))
        t_alpha = time_index * alpha.unsqueeze(0)
        t_omega = time_index * omega.unsqueeze(0)

        exp_decay = torch.exp(-t_alpha)
        cos_basis = exp_decay * torch.cos(t_omega)
        sin_basis = exp_decay * torch.sin(t_omega)

        # --- Calculate Final Response ---
        cosine_response = proj_feats * cos_basis.unsqueeze(0)
        sine_response = proj_feats * sin_basis.unsqueeze(0)
        return torch.cat([cosine_response, sine_response], dim=2).contiguous()

    def _forward_exact(
            self,
            x: torch.Tensor,
            dt: Optional[torch.Tensor],
            batch_size: int,
            seq_len: int,
            num_poles: int,
    ) -> torch.Tensor:
        """Sequential, dt-sensitive Zero-Order Hold rollout."""

        dtype = x.dtype

        tau = F.softplus(self._tau) + 1e-3
        alpha = (self.s_real * tau).to(dtype)
        omega = (self.s_imag * tau).clamp(-self.omega_max, self.omega_max).to(dtype)

        dt = self._prepare_dt(dt, batch_size, seq_len, dtype, x.device)

        drive = self.proj(x)  # [B, T, k]

        alpha = alpha.view(1, 1, num_poles)
        omega = omega.view(1, 1, num_poles)

        rho = torch.exp(-alpha * dt)
        theta = omega * dt
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)

        denom = (alpha.square() + omega.square()).clamp_min(1e-6)

        psi_cos = (-omega + rho * (alpha * sin_theta + omega * cos_theta)) / denom
        psi_sin = (alpha - rho * (alpha * cos_theta - omega * sin_theta)) / denom

        gain = F.softplus(self.b_param).to(dtype) + 1e-8
        drive = drive * gain

        cos_hist, sin_hist = [], []
        cos_state = torch.zeros(batch_size, num_poles, device=x.device, dtype=dtype)
        sin_state = torch.zeros(batch_size, num_poles, device=x.device, dtype=dtype)

        for t in range(seq_len):
            cos_new = rho[:, t] * (cos_state * cos_theta[:, t] - sin_state * sin_theta[:, t])
            sin_new = rho[:, t] * (cos_state * sin_theta[:, t] + sin_state * cos_theta[:, t])

            cos_state = cos_new + psi_cos[:, t] * drive[:, t]
            sin_state = sin_new + psi_sin[:, t] * drive[:, t]

            cos_hist.append(cos_state)
            sin_hist.append(sin_state)

        cosine_response = torch.stack(cos_hist, dim=1)
        sine_response = torch.stack(sin_hist, dim=1)
        return torch.cat([cosine_response, sine_response], dim=2).contiguous()

    def _prepare_dt(
            self,
            dt: Optional[torch.Tensor],
            batch_size: int,
            seq_len: int,
            dtype: torch.dtype,
            device: torch.device,
    ) -> torch.Tensor:
        """Normalises dt tensors to the broadcastable ``[B, T, 1]`` shape."""

        if dt is None:
            default_dt = 1.0 / max(seq_len - 1, 1)
            return torch.full((batch_size, seq_len, 1), default_dt, dtype=dtype, device=device)

        dt = dt.to(dtype=dtype, device=device)

        if dt.dim() == 1:
            dt = dt.view(1, seq_len, 1)
        elif dt.dim() == 2:
            dt = dt.view(dt.size(0), seq_len, 1)
        elif dt.dim() == 3:
            if dt.size(-1) != 1:
                raise ValueError("dt tensors must have last dimension equal to 1")
            dt = dt.view(dt.size(0), seq_len, 1)
        else:
            raise ValueError("dt must have 1, 2 or 3 dimensions")

        if dt.size(0) == 1 and batch_size > 1:
            dt = dt.expand(batch_size, -1, -1)
        elif dt.size(0) != batch_size:
            raise ValueError("dt batch dimension must either be 1 or match x")

        return dt


class LaplacePseudoInverse(nn.Module):
    """
    Decoder that maps Laplace features back to the original feature space.
    This acts as a pseudo-inverse to the :class:`LaplaceTransformEncoder`.

    Args:
        encoder:          The paired :class:`LaplaceTransformEncoder` instance.
        hidden_dim:       Width of the optional residual MLP.
        num_layers:       Depth of the optional residual MLP.
        use_mlp_residual: If True, adds an MLP on top of the linear head.
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
        """Decodes Laplace-domain features back to the data space.

        Args:
            lap_feats: ``[B, T, 2k]`` tensor produced by the encoder.

        Returns:
            ``[B, T, D]`` tensor in the original feature space.
        """

        if lap_feats.dim() != 3 or lap_feats.size(-1) != 2 * self.encoder.k:
            raise ValueError("lap_feats must be of shape [B, T, 2k]")

        k = self.encoder.k
        C, S = lap_feats.chunk(2, dim=-1)  # Split into cos/sin channels

        # Linear readout combines the two channels.
        y_lin = self.head_c(C) + self.head_s(S)

        if not self.use_mlp_residual:
            return y_lin

        # Add non-linear residual correction.
        y_mlp = self.mlp(self.norm(lap_feats))
        return y_lin + y_mlp
