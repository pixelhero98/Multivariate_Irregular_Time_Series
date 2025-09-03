import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class LearnableLaplacianBasis(nn.Module):
    """
    Unified Laplace Transform with two modes:
      - mode='parallel': fixed complex-exponential basis (O(1), fast, simple for long sequences).
      - mode='recurrent': time-varying, driven damped-sinusoid recurrence with optional per-step dt & modulation,
      (O(T), can be slow if sequences are long, but supports irregular sampling step).
      
    Args:
        k:         number of complex poles (outputs are 2*k due to cos/sin parts)
        feat_dim:  input feature dimension (last dim of x)
        mode:      'parallel' or 'recurrent' (plus aliases above)
        alpha_min: lower stability bound on damping (softplus(..) + alpha_min)
        omega_max: soft bound on frequency for parallel mode (clamped to [-omega_max, omega_max])

    Forward:
        x: [B, T, D]
        For mode='recurrent' you may also pass:
          dt:        [T] or [B,T] step sizes (if None, uses a uniform grid on [0,1])
          alpha_mod: [B,T,k] optional log-scale modulation of decay  (multiplies alpha)
          omega_mod: [B,T,k] optional log-scale modulation of frequency (multiplies omega)
          tau_mod:   [B,T,1] or [B,T,k] optional log-scale global timescale modulation
          
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
        import warnings
        self.mode = self._canonicalize_mode(mode)
        if mode.lower() in {"static", "tv"}:
            warnings.warn(
                f"mode='{mode}' is deprecated; use 'parallel' or 'recurrent' instead",
                DeprecationWarning,
                stacklevel=2,
            )
        self.k = int(k)
        self.alpha_min = float(alpha_min)
        self.omega_max = float(omega_max)

        # --- Trainable pole parameters ---
        # Real part is stored as raw and mapped by softplus to (0, +inf)
        self._s_real_raw = nn.Parameter(torch.empty(k))  # raw -> softplus + alpha_min (pole, or decay rate)
        self.s_imag = nn.Parameter(torch.empty(k))       # imaginary part (frequency)

        # Optional global timescale used by 'recurrent' mode
        self._tau = nn.Parameter(torch.tensor(0.0))

        # Learned projection feat_dim -> k (shared across modes)
        self.proj = spectral_norm(
            nn.Linear(feat_dim, k, bias=True), n_power_iterations=1, eps=1e-6
        )

        self.reset_parameters()

    @staticmethod
    def _canonicalize_mode(mode: str) -> str:
        m = mode.lower()
        if m in {"parallel"}:
            return "parallel"
        if m in {"recurrent"}:
            return "recurrent"
        raise ValueError("mode must be one of {'parallel','recurrent'} or their aliases")

    # Expose positive real-part after softplus (used in both modes)
    @property
    def s_real(self) -> torch.Tensor:
        return F.softplus(self._s_real_raw) + self.alpha_min

    def reset_parameters(self) -> None:
        # Frequencies ~ U[-pi, pi]; Real part targeted to ~U[0.01, 0.2] for stable starts
        with torch.no_grad():
            nn.init.uniform_(self.s_imag, -math.pi, math.pi)

            target_alpha = torch.empty_like(self._s_real_raw).uniform_(0.01, 0.2)
            # raw := softplus^{-1}(target_alpha - alpha_min)
            y = (target_alpha - self.alpha_min).clamp_min(1e-8)
            self._s_real_raw.copy_(torch.log(torch.expm1(y)))

            # Init projection well
            w = getattr(self.proj, "weight_orig", self.proj.weight)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            if self.proj.bias is not None:
                fan_in = self.proj.in_features
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.proj.bias, -bound, bound)

    def forward(
        self,
        x: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
        alpha_mod: Optional[torch.Tensor] = None,
        omega_mod: Optional[torch.Tensor] = None,
        tau_mod: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert x.dim() == 3, "x must be [B,T,D]"
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype
        k = self.k

        if self.mode == "parallel":
            # ---- Parallel complex exponential basis (fast path) ----
            alpha = self.s_real                     # [k], strictly > alpha_min
            beta  = self.s_imag.clamp(-self.omega_max, self.omega_max)

            # time indices in float32 for stable complex exp; cast back to input dtype
            t_idx = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)  # [T,1]

            s = torch.complex(-alpha.float(), beta.float())            # [k]
            expo = torch.exp(t_idx * s.unsqueeze(0))                   # [T,k] complex64
            re_basis = expo.real.to(dtype)
            im_basis = expo.imag.to(dtype)

            proj_feats = self.proj(x)                                  # [B,T,k]
            real_proj = proj_feats * re_basis.unsqueeze(0)             # [B,T,k]
            imag_proj = proj_feats * im_basis.unsqueeze(0)             # [B,T,k]
            return torch.cat([real_proj, imag_proj], dim=2).contiguous()

        # ---- Recurrent (time-varying) driven recurrence with per-step dt & modulation ----
        tau = F.softplus(self._tau) + 1e-3          # > 0
        alpha0 = self.s_real * tau                  # [k]
        omega0 = self.s_imag * tau                  # [k]

        # Shape dt -> [B,T,1]
        if dt is None:
            base = (1.0 / max(T - 1, 1)) if T > 1 else 1.0
            dt_bt1 = x.new_full((B, T, 1), base)
        else:
            if dt.dim() == 1:      # [T] -> [B,T,1]
                dt_bt1 = dt.view(1, T, 1).to(dtype=x.dtype, device=device).expand(B, T, 1)
            elif dt.dim() == 2:    # [B,T] -> [B,T,1]
                dt_bt1 = dt.unsqueeze(-1).to(dtype=x.dtype, device=device)
            else:
                raise ValueError("dt must be [T] or [B,T] if provided")

        # Expand poles to [B,T,k] and apply optional log-space modulation
        alpha = alpha0.view(1, 1, k).expand(B, T, k)
        omega = omega0.view(1, 1, k).expand(B, T, k)
        if alpha_mod is not None:
            alpha = alpha * alpha_mod.to(dtype=x.dtype, device=device).exp()
        if omega_mod is not None:
            omega = omega * omega_mod.to(dtype=x.dtype, device=device).exp()
        if tau_mod is not None:
            scale = tau_mod.to(dtype=x.dtype, device=device).exp()
            # allow scale to be [B,T,1] or [B,T,k]
            if scale.shape[-1] == 1:
                scale = scale.expand(B, T, k)
            alpha = alpha * scale
            omega = omega * scale

        # Per-step decay & rotation
        rho   = torch.exp(-alpha * dt_bt1)          # [B,T,k]
        theta = omega * dt_bt1                      # [B,T,k]
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)

        # Drive
        u = self.proj(x)  # [B, T, k]

        # Recurrence (use previous c,s to update both)
        C = torch.empty(B, T, k, dtype=u.dtype, device=device)
        S = torch.empty(B, T, k, dtype=u.dtype, device=device)
        c = torch.zeros(B, k, device=device, dtype=u.dtype)
        s = torch.zeros(B, k, device=device, dtype=u.dtype)
        for t in range(T):
            c_prev, s_prev = c, s
            rt = rho[:, t, :]
            ct = cos_t[:, t, :]
            st = sin_t[:, t, :]
            c = rt * (c_prev * ct - s_prev * st) + u[:, t, :]
            s = rt * (c_prev * st + s_prev * ct)
            C[:, t, :] = c
            S[:, t, :] = s

        return torch.cat([C, S], dim=2).contiguous()


class LearnableInverseLaplacianBasis(nn.Module):
    def __init__(self, laplace_basis: LearnableLaplacianBasis) -> None:
        super().__init__()
        self.lap = laplace_basis
        D = laplace_basis.proj.in_features
        C = 2 * laplace_basis.k

        self.norm = nn.LayerNorm(C)
        self.fc1 = spectral_norm(nn.Linear(C, max(C, D)), n_power_iterations=1, eps=1e-6)
        self.act = nn.GELU()
        self.fc2 = spectral_norm(nn.Linear(max(C, D), D), n_power_iterations=1, eps=1e-6)

        with torch.no_grad():
            w1 = getattr(self.fc1, "weight_orig", self.fc1.weight)
            w2 = getattr(self.fc2, "weight_orig", self.fc2.weight)
            if getattr(self.fc2, "bias", None) is not None:
                self.fc2.bias.zero_()
            w1.zero_(); w2.zero_()

            H = w1.shape[0]
            eye = torch.eye(C, device=w1.device, dtype=w1.dtype)
            w1[:min(H, C), :C].copy_(eye[:min(H, C), :])

            W = self.lap.proj.weight.data  # [k, D]
            if w2.shape[1] >= self.lap.k:
                w2[:, : self.lap.k].copy_(W.t())

    def forward(self, lap_feats: torch.Tensor) -> torch.Tensor:
        h = self.norm(lap_feats)
        h = self.act(self.fc1(h))
        return self.fc2(h)


class LaplaceBlock(nn.Module):
    def __init__(self, k: int, feat_dim: int, mode: str = "parallel") -> None:
        super().__init__()
        self.analysis = LearnableLaplacianBasis(k=k, feat_dim=feat_dim, mode=mode)
        self.synthesis = LearnableInverseLaplacianBasis(self.analysis)

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.analysis(x, **kwargs)

    def decode(self, feats: torch.Tensor) -> torch.Tensor:
        return self.synthesis(feats)
