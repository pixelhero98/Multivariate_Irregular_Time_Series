"""
Learnable Laplace analysis + decoder stack.

- LearnableLaplacianBasis:
    Projects sequences to a 2*k-channel Laplace feature space using either
    a parallel complex-exponential basis or a time-varying recurrent update.

- LearnableInverseLaplacianBasis (decoder):
    Learns to map Laplace features back to the original feature space.
    This is NOT a strict mathematical inverse; it is a small MLP trained
    end-to-end for the downstream objective.

- LaplaceBlock:
    Convenience wrapper combining analysis + decoder.

Contract (as requested):
    The decoder is always used with its paired basis:
      input_dim  = 2 * k
      output_dim = feat_dim  (original input dim to the basis)
    The only optional size is the decoder's hidden_dim.
"""
import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


__all__ = [
    "LearnableLaplacianBasis",
    "LearnableInverseLaplacianBasis"
]


# =========================
# Analysis: Laplace features
# =========================
class LearnableLaplacianBasis(nn.Module):
    """
    Learnable Laplace transform basis producing 2*k channels (cos & sin branches).

    Args:
        k:         number of complex poles (output channels = 2*k)
        feat_dim:  input feature dimension (last dim of x)
        mode:      'parallel' (fixed basis) or 'recurrent' (time-varying recurrence)
                   Aliases: 'static'→'parallel' (deprecated), 'tv'→'recurrent' (deprecated)
        alpha_min: strictly positive floor for the real part (decay)
        omega_max: clamp for imaginary part (frequency) in 'parallel' mode

    Forward:
        x: [B, T, D]

        For mode='recurrent' you may also pass:
          dt:        [T] or [B, T] step sizes; None -> uniform over [0,1]
          alpha_mod: [B, T, k] log-scale modulation of decay (multiplies alpha)
          omega_mod: [B, T, k] log-scale modulation of frequency (multiplies omega)
          tau_mod:   [B, T, 1] or [B, T, k] log-scale global timescale modulation

    Returns:
        lap_feats: [B, T, 2*k]  (concat of cosine-like and sine-like channels)
    """

    def __init__(
        self,
        k: int,
        feat_dim: int,
        mode: str = "parallel",
        alpha_min: float = 1e-6,
        omega_max: float = math.pi,
    ) -> None:
        super().__init__()
        self.mode = self._canonicalize_mode(mode)
        self.k = int(k)
        self.feat_dim = int(feat_dim)
        self.alpha_min = float(alpha_min)
        self.omega_max = float(omega_max)

        # Trainable pole parameters
        self._s_real_raw = nn.Parameter(torch.empty(k))  # mapped via softplus -> positive
        self.s_imag = nn.Parameter(torch.empty(k))       # frequency
        self._tau = nn.Parameter(torch.tensor(0.0))      # global timescale (recurrent)

        # Learned projection feat_dim -> k (shared across modes)
        self.proj = spectral_norm(nn.Linear(self.feat_dim, k, bias=True), n_power_iterations=1, eps=1e-6)

        self.reset_parameters()

    @staticmethod
    def _canonicalize_mode(mode: str) -> str:
        m = mode.lower()
        if m in {"parallel"}:
            return "parallel"
        if m in {"recurrent"}:
            return "recurrent"
        if m in {"static"}:
            warnings.warn("mode='static' is deprecated; use 'parallel'.", DeprecationWarning, stacklevel=3)
            return "parallel"
        if m in {"tv", "timevarying", "time-varying"}:
            warnings.warn("mode='tv' is deprecated; use 'recurrent'.", DeprecationWarning, stacklevel=3)
            return "recurrent"
        raise ValueError("mode must be one of {'parallel','recurrent'} (or deprecated {'static','tv'}).")

    @property
    def s_real(self) -> torch.Tensor:
        """Strictly positive real part (decay): softplus(raw) + alpha_min."""
        return F.softplus(self._s_real_raw) + self.alpha_min

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.uniform_(self.s_imag, -math.pi, math.pi)

            target_alpha = torch.empty_like(self._s_real_raw).uniform_(0.01, 0.2)
            y = (target_alpha - self.alpha_min).clamp_min(1e-8)  # softplus target
            self._s_real_raw.copy_(torch.log(torch.expm1(y)))    # softplus^{-1}(y)

            w = getattr(self.proj, "weight_orig", self.proj.weight)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            if self.proj.bias is not None:
                bound = 1 / math.sqrt(self.proj.in_features)
                nn.init.uniform_(self.proj.bias, -bound, bound)

    def forward(
        self,
        x: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
        alpha_mod: Optional[torch.Tensor] = None,
        omega_mod: Optional[torch.Tensor] = None,
        tau_mod: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert x.dim() == 3 and x.size(-1) == self.feat_dim, f"x must be [B, T, {self.feat_dim}]"
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype
        k = self.k

        if self.mode == "parallel":
            # Complex exponential basis (fast path)
            alpha = self.s_real
            beta = self.s_imag.clamp(-self.omega_max, self.omega_max)

            t_idx = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)  # [T,1]
            s = torch.complex(-alpha.float(), beta.float())                            # [k]
            expo = torch.exp(t_idx * s.unsqueeze(0))                                   # [T,k] complex
            re_basis, im_basis = expo.real.to(dtype), expo.imag.to(dtype)             # [T,k] each

            proj_feats = self.proj(x)                              # [B,T,k]
            real_proj = proj_feats * re_basis.unsqueeze(0)         # [B,T,k]
            imag_proj = proj_feats * im_basis.unsqueeze(0)         # [B,T,k]
            return torch.cat([real_proj, imag_proj], dim=2).contiguous()

        # Recurrent (time-varying) path
        tau = F.softplus(self._tau) + 1e-3
        alpha0 = self.s_real * tau                 # [k]
        omega0 = self.s_imag * tau                 # [k]

        # dt -> [B, T, 1]
        if dt is None:
            base = (1.0 / max(T - 1, 1)) if T > 1 else 1.0
            dt_bt1 = x.new_full((B, T, 1), base)
        else:
            if dt.dim() == 1:
                if dt.numel() != T:
                    raise ValueError(f"dt shape {tuple(dt.shape)} incompatible with T={T}.")
                dt_bt1 = dt.view(1, T, 1).to(dtype=x.dtype, device=device).expand(B, T, 1)
            elif dt.dim() == 2:
                if dt.shape != (B, T):
                    raise ValueError(f"dt must be [B, T]={B,T} if 2D; got {tuple(dt.shape)}.")
                dt_bt1 = dt.unsqueeze(-1).to(dtype=x.dtype, device=device)
            else:
                raise ValueError("dt must be [T] or [B, T] if provided")

        # Expand poles to [B, T, k] and apply optional log-space modulation
        alpha = alpha0.view(1, 1, k).expand(B, T, k)
        omega = omega0.view(1, 1, k).expand(B, T, k)

        if alpha_mod is not None:
            if alpha_mod.shape != (B, T, k):
                raise ValueError(f"alpha_mod must be [B, T, k]={B,T,k}; got {tuple(alpha_mod.shape)}.")
            alpha = alpha * alpha_mod.to(dtype=x.dtype, device=device).exp()

        if omega_mod is not None:
            if omega_mod.shape != (B, T, k):
                raise ValueError(f"omega_mod must be [B, T, k]={B,T,k}; got {tuple(omega_mod.shape)}.")
            omega = omega * omega_mod.to(dtype=x.dtype, device=device).exp()

        if tau_mod is not None:
            if tau_mod.shape[:2] != (B, T) or tau_mod.shape[-1] not in (1, k):
                raise ValueError(f"tau_mod must be [B, T, 1] or [B, T, k]; got {tuple(tau_mod.shape)}.")
            scale = tau_mod.to(dtype=x.dtype, device=device).exp()
            if scale.shape[-1] == 1:
                scale = scale.expand(B, T, k)
            alpha = alpha * scale
            omega = omega * scale

        # Step-wise decay & rotation
        rho = torch.exp(-alpha * dt_bt1)     # [B,T,k]
        theta = omega * dt_bt1               # [B,T,k]
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)

        # Drive (real only)
        u = self.proj(x)                     # [B,T,k]
        
        # z_t = rho_t * e^{i theta_t} * z_{t-1} + u_t

        Z = torch.empty(B, T, k, dtype=torch.complex64, device=device)
        z = torch.zeros(B, k, dtype=torch.complex64, device=device)
        
        for t in range(T):
            update_factor = rho[:, t, :] * torch.exp(1j * theta[:, t, :])
            z = update_factor * z + u[:, t, :]
            Z[:, t, :] = z
        
        return torch.cat([Z.real.to(x.dtype), Z.imag.to(x.dtype)], dim=2).contiguous()



# ==============================
# Decoder (NOT a strict inverse)
# ==============================
class LearnableInverseLaplacianBasis(nn.Module):
    """
    Decoder that maps Laplace features back to the original feature space.

    Always paired with a LearnableLaplacianBasis:
        input_dim  = 2 * basis.k
        output_dim = basis.feat_dim

    Implements a complex-aware linear head:
        y_lin = C @ Wc - S @ Ws
    Optionally adds a small MLP residual on top for non-linear corrections.

    Args:
        basis:               the paired LearnableLaplacianBasis
        hidden_dim:          width of the residual MLP; default = max(2k, D)
        num_layers:          residual MLP depth (>=2 recommended if enabled)
        use_sn:              apply spectral norm to linear layers
        use_mlp_residual:    add MLP residual on top of the complex-aware head
    """

    def __init__(
        self,
        basis: LearnableLaplacianBasis,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        use_sn: bool = True,
        use_mlp_residual: bool = True,
    ) -> None:
        super().__init__()
        assert isinstance(basis, LearnableLaplacianBasis), "basis must be a LearnableLaplacianBasis"
        self.basis = basis
        self.use_mlp_residual = bool(use_mlp_residual)

        k = basis.k
        D = basis.feat_dim
        in_dim = 2 * k
        H = int(hidden_dim if hidden_dim is not None else max(in_dim, D))

        def maybe_sn(linear: nn.Linear) -> nn.Linear:
            return spectral_norm(linear, n_power_iterations=1, eps=1e-6) if use_sn else linear

        # --- Complex-aware linear head: y = C @ Wc - S @ Ws ---
        self.head_c = maybe_sn(nn.Linear(k, D, bias=False))
        self.head_s = maybe_sn(nn.Linear(k, D, bias=False))

        # --- Optional residual MLP over [C,S] ---
        if self.use_mlp_residual:
            self.norm = nn.LayerNorm(in_dim)
            layers = [maybe_sn(nn.Linear(in_dim, H)), nn.GELU()]
            for _ in range(max(0, num_layers - 2)):
                layers += [maybe_sn(nn.Linear(H, H)), nn.GELU()]
            layers += [maybe_sn(nn.Linear(H, D))]
            self.mlp = nn.Sequential(*layers)

        # Init
        with torch.no_grad():
            for lin in (self.head_c, self.head_s):
                w = getattr(lin, "weight_orig", lin.weight)
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            if self.use_mlp_residual:
                for m in self.mlp:
                    if isinstance(m, nn.Linear):
                        w = getattr(m, "weight_orig", m.weight)
                        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
                        if m.bias is not None:
                            bound = 1 / math.sqrt(m.in_features)
                            nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, lap_feats: torch.Tensor) -> torch.Tensor:
        # lap_feats: [B, T, 2k] with [C | S] halves
        B, T, Ctot = lap_feats.shape
        k = self.basis.k
        assert Ctot == 2 * k, f"lap_feats must be [B, T, {2*k}]"

        C, S = lap_feats[..., :k], lap_feats[..., k:]  # [B,T,k] each

        # Complex-aware linear readout (no LayerNorm to preserve amplitude/phase scale)
        y_lin = self.head_c(C) - self.head_s(S)       # [B,T,D]

        if not self.use_mlp_residual:
            return y_lin

        # Residual MLP over concatenated [C,S]
        h = self.norm(lap_feats)
        y_mlp = self.mlp(h)                            # [B,T,D]
        return y_lin + y_mlp
