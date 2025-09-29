import math
import warnings
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class LearnableLaplacianBasis(nn.Module):
    """
    Learnable Laplace transform basis producing 2*k channels (cos & sin).

    Args:
        k:         number of complex poles (output channels = 2*k)
        feat_dim:  input feature dimension (last dim of x)
        mode:      'parallel' (fixed basis) or 'recurrent' (irregular-step recurrence)
        alpha_min: strictly positive floor added to softplus(real part)
        omega_max: clamp for imaginary part in 'parallel' mode

    Forward:
        x:  [B, T, D]
        dt: [T] or [B, T] step sizes; None -> uniform over [0,1]

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

        # Learned projection feat_dim -> k (shared)
        self.proj = spectral_norm(
            nn.Linear(self.feat_dim, k, bias=True),
            n_power_iterations=1,
            eps=1e-6,
        )

        # Per-mode positive input gain shared across cos/sin (exp -> >= 0)
        self.b_log = nn.Parameter(torch.zeros(1, 1, k))

        self.reset_parameters()

    @staticmethod
    def _canonicalize_mode(mode: str) -> str:
        m = mode.lower()
        if m in {"parallel"}:
            return "parallel"
        if m in {"recurrent"}:
            return "recurrent"
        if m in {"static"}:
            warnings.warn("mode='static' is deprecated; use 'parallel'.",
                          DeprecationWarning, stacklevel=3)
            return "parallel"
        if m in {"tv", "timevarying", "time-varying"}:
            warnings.warn("mode='tv' is deprecated; use 'recurrent'.",
                          DeprecationWarning, stacklevel=3)
            return "recurrent"
        raise ValueError("mode must be one of {'parallel','recurrent'} (or deprecated {'static','tv'}).")

    @property
    def s_real(self) -> torch.Tensor:
        """Strictly positive real part (decay): softplus(raw) + alpha_min."""
        return F.softplus(self._s_real_raw) + self.alpha_min

    def reset_parameters(self) -> None:
        with torch.no_grad():
            # Imag part in [-pi, pi]
            nn.init.uniform_(self.s_imag, -math.pi, math.pi)

            # Real part softplus target in [0.01, 0.2] plus alpha_min
            target_alpha = torch.empty_like(self._s_real_raw).uniform_(0.01, 0.2)
            y = (target_alpha - self.alpha_min).clamp_min(1e-8)  # softplus target
            self._s_real_raw.copy_(torch.log(torch.expm1(y)))    # softplus^{-1}(y)

            # Projection init
            w = getattr(self.proj, "weight_orig", self.proj.weight)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            if self.proj.bias is not None:
                bound = 1 / math.sqrt(self.proj.in_features)
                nn.init.uniform_(self.proj.bias, -bound, bound)

            # Input gain b=exp(0)=1
            self.b_log.zero_()

    def forward(
        self,
        x: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert x.dim() == 3 and x.size(-1) == self.feat_dim, f"x must be [B, T, {self.feat_dim}]"
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype
        k = self.k

        if self.mode == "parallel":
            # Complex exponential basis (fast path), dtype-safe for AMP
            alpha = self.s_real.to(dtype)                                   # [k]
            beta  = self.s_imag.clamp(-self.omega_max, self.omega_max).to(dtype)  # [k]
            t_idx = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)      # [T,1]
            # Complex exponent requires float32; cast minimally then back to dtype
            s = torch.complex((-alpha).float(), beta.float())                     # [k] complex
            expo = torch.exp(t_idx.float() * s.unsqueeze(0))                      # [T,k] complex
            re_basis, im_basis = expo.real.to(dtype), expo.imag.to(dtype)         # [T,k]

            proj_feats = self.proj(x)                                             # [B,T,k]
            out_re = proj_feats * re_basis.unsqueeze(0)                           # [B,T,k]
            out_im = proj_feats * im_basis.unsqueeze(0)                           # [B,T,k]
            return torch.cat([out_re, out_im], dim=2).contiguous()                # [B,T,2k]

        # ----- recurrent (irregular-step) path -----

        alpha0 = self.s_real.to(dtype)                                            # [k]
        omega0 = self.s_imag.clamp(-self.omega_max, self.omega_max).to(dtype)     # [k]

        # dt -> [B, T, 1]
        if dt is None:
            base = (1.0 / max(T - 1, 1)) if T > 1 else 1.0
            dt_bt1 = x.new_full((B, T, 1), base)
        else:
            if dt.dim() == 1:
                if dt.numel() != T:
                    raise ValueError(f"dt shape {tuple(dt.shape)} incompatible with T={T}.")
                dt_bt1 = dt.view(1, T, 1).to(dtype=dtype, device=device).expand(B, T, 1)
            elif dt.dim() == 2:
                if dt.shape != (B, T):
                    raise ValueError(f"dt must be [B, T]={B,T} if 2D; got {tuple(dt.shape)}.")
                dt_bt1 = dt.unsqueeze(-1).to(dtype=dtype, device=device)
            else:
                raise ValueError("dt must be [T] or [B, T] if provided")

        # Expand poles to [B, T, k]
        alpha = alpha0.view(1, 1, k).expand(B, T, k)
        omega = omega0.view(1, 1, k).expand(B, T, k)

        # Per-step decay & rotation (for Phi)
        rho   = torch.exp(-alpha * dt_bt1)           # [B,T,k]
        theta = omega * dt_bt1                       # [B,T,k]
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)

        # Projection to per-mode scalar drive
        u = self.proj(x)  # [B,T,k]

        # Per-mode positive input gain (shared across cos/sin)
        b = torch.exp(self.b_log).to(dtype=dtype, device=device)  # [1,1,k]

        # Preallocate rollout tensors
        C = torch.empty(B, T, k, dtype=dtype, device=device)
        S = torch.empty(B, T, k, dtype=dtype, device=device)
        c = torch.zeros(B, k, dtype=dtype, device=device)
        s = torch.zeros(B, k, dtype=dtype, device=device)

        # Time loop
        for t in range(T):
            rt = rho[:, t, :]
            ct = cos_t[:, t, :]
            st = sin_t[:, t, :]

            # Drift (rotation+decay)
            c_new = rt * (c * ct - s * st)
            s_new = rt * (c * st + s * ct)

            # Ultra-cheap, gap-aware injection to both channels
            inj = (1.0 - rt) * (b * u[:, t, :])  # [B,k]

            c = c_new + inj
            s = s_new + inj

            C[:, t, :], S[:, t, :] = c, s

        lap_feats = torch.cat([C, S], dim=2).contiguous()  # [B,T,2k]
        return lap_feats

# ==============================
# Decoder (NOT a strict inverse)
# ==============================
class LearnablepesudoInverse(nn.Module):
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
        basis: LearnableLaplaceBasis,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        use_sn: bool = True,
        use_mlp_residual: bool = True,
    ) -> None:
        super().__init__()
        assert isinstance(basis, LearnableLaplaceBasis), "basis must be a LearnableLaplacianBasis"
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





# class LearnableLaplacianBasis(nn.Module):
#     """
#     Learnable Laplace transform basis producing 2*k channels (cos & sin).

#     Args:
#         k:         number of complex poles (output channels = 2*k)
#         feat_dim:  input feature dimension (last dim of x)
#         mode:      'parallel' (fixed basis) or 'recurrent' (irregular-step recurrence)
#         alpha_min: strictly positive floor added to softplus(real part)
#         omega_max: clamp for imaginary part in 'parallel' mode

#     Forward:
#         x:  [B, T, D]
#         dt: [T] or [B, T] step sizes; None -> uniform over [0,1]

#     Returns:
#         lap_feats: [B, T, 2*k]  (concat of cosine-like and sine-like channels)
#     """
#     def __init__(
#         self,
#         k: int,
#         feat_dim: int,
#         mode: str = "parallel",
#         alpha_min: float = 1e-6,
#         omega_max: float = math.pi,
#     ) -> None:
#         super().__init__()
#         self.mode = self._canonicalize_mode(mode)
#         self.k = int(k)
#         self.feat_dim = int(feat_dim)
#         self.alpha_min = float(alpha_min)
#         self.omega_max = float(omega_max)

#         # Trainable pole parameters
#         self._s_real_raw = nn.Parameter(torch.empty(k))  # softplus -> positive, then + alpha_min
#         self.s_imag = nn.Parameter(torch.empty(k))       # frequency (can be negative)

#         # Optional global time-scale for recurrent path
#         self._tau = nn.Parameter(torch.tensor(0.0))      # softplus(~0) ≈ 1 at init

#         # Projection feat_dim -> k (shared across modes)
#         self.proj = spectral_norm(
#             nn.Linear(self.feat_dim, k, bias=True),
#             n_power_iterations=1, eps=1e-6
#         )

#         # Per-mode nonnegative input gain shared across cos/sin (softplus → ≥0)
#         init_val = math.log(math.e - 1.0)   # softplus(init_val) ≈ 1
#         self.b_param = nn.Parameter(torch.full((1, 1, k), init_val))

#         self.reset_parameters()

#     @staticmethod
#     def _canonicalize_mode(mode: str) -> str:
#         m = mode.lower()
#         if m in {"parallel"}:  return "parallel"
#         if m in {"recurrent"}: return "recurrent"
#         if m in {"static"}:
#             warnings.warn("mode='static' is deprecated; use 'parallel'.", DeprecationWarning, stacklevel=3)
#             return "parallel"
#         if m in {"tv", "timevarying", "time-varying"}:
#             warnings.warn("mode='tv' is deprecated; use 'recurrent'.", DeprecationWarning, stacklevel=3)
#             return "recurrent"
#         raise ValueError("mode must be one of {'parallel','recurrent'} (or deprecated {'static','tv'}).")

#     @property
#     def s_real(self) -> torch.Tensor:
#         """Strictly positive real part (decay): softplus(raw) + alpha_min."""
#         return F.softplus(self._s_real_raw) + self.alpha_min

#     def reset_parameters(self) -> None:
#         with torch.no_grad():
#             # Imag part in [-π, π]
#             nn.init.uniform_(self.s_imag, -math.pi, math.pi)
#             # Real part target in [0.01, 0.2] (then + alpha_min)
#             target_alpha = torch.empty_like(self._s_real_raw).uniform_(0.01, 0.2)
#             y = (target_alpha - self.alpha_min).clamp_min(1e-8)
#             self._s_real_raw.copy_(torch.log(torch.expm1(y)))  # softplus^{-1}(y)

#             # Projection init
#             w = getattr(self.proj, "weight_orig", self.proj.weight)
#             nn.init.kaiming_uniform_(w, a=math.sqrt(5))
#             if self.proj.bias is not None:
#                 bound = 1 / math.sqrt(self.proj.in_features)
#                 nn.init.uniform_(self.proj.bias, -bound, bound)

#             # Input gain b=1 at init; tau≈1 at init
#             self.b_param.zero_()
#             self._tau.zero_()

#     def forward(
#         self,
#         x: torch.Tensor,
#         dt: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         assert x.dim() == 3 and x.size(-1) == self.feat_dim, f"x must be [B, T, {self.feat_dim}]"
#         B, T, _ = x.shape
#         device, dtype = x.device, x.dtype
#         k = self.k

#         if self.mode == "parallel":
#             # Complex exponential basis (fast path)
#             alpha = self.s_real.to(dtype)                                  # [k]
#             beta  = self.s_imag.clamp(-self.omega_max, self.omega_max).to(dtype)
#             t_idx = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)  # [T,1]
#             s = torch.complex((-alpha).float(), beta.float())              # [k] complex
#             expo = torch.exp(t_idx.float() * s.unsqueeze(0))               # [T,k] complex
#             re_basis, im_basis = expo.real.to(dtype), expo.imag.to(dtype)  # [T,k]
#             proj_feats = self.proj(x)                                      # [B,T,k]
#             return torch.cat([proj_feats * re_basis.unsqueeze(0),
#                               proj_feats * im_basis.unsqueeze(0)], dim=2).contiguous()

#         # ----- recurrent (irregular) path -----
#         tau = F.softplus(self._tau) + 1e-3
#         alpha0 = self.s_real * tau                 # [k]
#         omega0 = self.s_imag * tau                 # [k]
#         omega0 = omega0.clamp(-self.omega_max, self.omega_max)

#         # dt -> [B, T, 1]
#         if dt is None:
#             base = (1.0 / max(T - 1, 1)) if T > 1 else 1.0
#             dt_bt1 = x.new_full((B, T, 1), base)
#         else:
#             if dt.dim() == 1:
#                 if dt.numel() != T:
#                     raise ValueError(f"dt shape {tuple(dt.shape)} incompatible with T={T}.")
#                 dt_bt1 = dt.view(1, T, 1).to(dtype=dtype, device=device).expand(B, T, 1)
#             elif dt.dim() == 2:
#                 if dt.shape != (B, T):
#                     raise ValueError(f"dt must be [B, T]={B,T} if 2D; got {tuple(dt.shape)}.")
#                 dt_bt1 = dt.unsqueeze(-1).to(dtype=dtype, device=device)
#             else:
#                 raise ValueError("dt must be [T] or [B, T] if provided")

#         # Expand poles to [B, T, k]
#         alpha = alpha0.view(1, 1, k).expand(B, T, k).to(dtype)
#         omega = omega0.view(1, 1, k).expand(B, T, k).to(dtype)

#         # Per-timestep per-mode drive
#         u = self.proj(x)  # [B,T,k]

#         # Step-wise decay & rotation
#         rho   = torch.exp(-alpha * dt_bt1)     # [B,T,k]
#         theta = omega * dt_bt1                  # [B,T,k]
#         cos_t, sin_t = torch.cos(theta), torch.sin(theta)

#         # Exact ZOH input map Ψ(Δ) for B = [0, 1]^T (2×1).
#         # If other parts assume B=[0, -1]^T, flip the sign of u upstream.
#         den   = (alpha**2 + omega**2).clamp_min(1e-6)                     # [B,T,k]
#         psi_c = (-omega + rho * (alpha * sin_t + omega * cos_t)) / den    # [B,T,k]
#         psi_s = ( alpha - rho * (alpha * cos_t - omega * sin_t)) / den    # [B,T,k]

#         # Nonnegative per-mode input gain
#         b = F.softplus(self.b_param).to(dtype=dtype, device=device) + 1e-8  # [1,1,k]
#         u_eff = u * b                                                        # [B,T,k]

#         # Recurrent rollout
#         c_hist, s_hist = [], []
#         c = torch.zeros(B, k, dtype=dtype, device=device)
#         s = torch.zeros(B, k, dtype=dtype, device=device)
#         for t in range(T):
#             rt, ct, st = rho[:, t, :], cos_t[:, t, :], sin_t[:, t, :]
#             # homogeneous drift
#             c_new = rt * (c * ct - s * st)
#             s_new = rt * (c * st + s * ct)
#             # exact ZOH input
#             c = c_new + psi_c[:, t, :] * u_eff[:, t, :]
#             s = s_new + psi_s[:, t, :] * u_eff[:, t, :]
#             c_hist.append(c); s_hist.append(s)

#         C = torch.stack(c_hist, dim=1)  # [B,T,k]
#         S = torch.stack(s_hist, dim=1)  # [B,T,k]
#         return torch.cat([C, S], dim=2).contiguous()  # [B,T,2k]
