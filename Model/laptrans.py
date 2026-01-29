import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

__all__ = ["ModalPredictor", "ModalSynthesizer"]


class ModalPredictor(nn.Module):
    """Modal analysis that maps a time sequence to modal residues.

    This module implements the paper-aligned notion of *effective* modal parameters
    conditioned on a history summary. We treat poles (rho, omega) as history- and
    diffusion-conditioned (stable by construction), and obtain cosine/sine residues
    (c, b) as *modal coefficients* (one vector in R^{D} per mode).

    Computational modes:
        - use_time_attn=True: learned spectral cross-attention (mode queries attend
          over time keys, values from x).
        - use_time_attn=False: fast diagonal projection using the analytic basis
          (normalized correlations; no ridge solve).

    Output:
        theta: [B, 2K, D]  (first K cosine residues, last K sine residues)
        rho:   [B, K]
        omega: [B, K]
    """

    def __init__(
        self,
        k: int,
        feat_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        alpha_min: float = 1e-6,
        omega_max: float = math.pi,
        cond_dim: Optional[int] = None,
        rho_perturb_scale: float = 0.5,
        omega_perturb_scale: float = 0.5,
        proj_eps: float = 1e-6,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.feat_dim = int(feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.alpha_min = float(alpha_min)
        self.omega_max = float(omega_max)
        self.cond_dim = cond_dim
        self.rho_perturb_scale = float(rho_perturb_scale)
        self.omega_perturb_scale = float(omega_perturb_scale)
        self.proj_eps = float(proj_eps)

        if self.hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )

        # Base poles
        self._rho_raw = nn.Parameter(torch.empty(self.k))
        self._omega_raw = nn.Parameter(torch.empty(self.k))

        # Conditioned bounded perturbations for poles
        if cond_dim is not None:
            self.rho_refine = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.k),
            )
            self.omega_refine = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.k),
            )
            nn.init.zeros_(self.rho_refine[-1].weight)
            nn.init.zeros_(self.rho_refine[-1].bias)
            nn.init.zeros_(self.omega_refine[-1].weight)
            nn.init.zeros_(self.omega_refine[-1].bias)

        # --- Spectral cross-attention path (optional) ---
        # Queries from (rho, omega, component_id) where component_id=0 (cos) or 1 (sin).
        # A learned query bank captures mode-specific priors.
        self.mode_queries = nn.Parameter(torch.randn(1, 2 * self.k, hidden_dim) * 0.02)
        self.pole_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Time embedding used to inject ordering into attention keys.
        self.time_emb = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Shared projection for keys/values from x.
        self.input_proj = nn.Linear(feat_dim, hidden_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(hidden_dim, feat_dim)

        # Norms help stability of attention when T is large
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.k_norm = nn.LayerNorm(hidden_dim)
        self.v_norm = nn.LayerNorm(hidden_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            # rho init in (0.01, 0.2)
            target_rho = torch.empty_like(self._rho_raw).uniform_(0.01, 0.2)
            y = (target_rho - self.alpha_min).clamp_min(1e-8)
            self._rho_raw.copy_(torch.log(torch.expm1(y)))

            # omega init in [0.01*omega_max, 0.95*omega_max]
            low_log = math.log(0.01 * self.omega_max)
            high_log = math.log(0.95 * self.omega_max)
            target_omega = torch.exp(
                torch.empty_like(self._omega_raw).uniform_(low_log, high_log)
            )
            p = (target_omega / self.omega_max).clamp(1e-4, 1 - 1e-4)
            self._omega_raw.copy_(torch.log(p) - torch.log1p(-p))

    def _base_poles(self, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        rho = F.softplus(self._rho_raw.to(device=device, dtype=dtype)) + self.alpha_min
        omega = self.omega_max * torch.sigmoid(self._omega_raw.to(device=device, dtype=dtype))
        return rho, omega

    def _merge_condition(
        self,
        cond: Optional[torch.Tensor],
        diffusion_time_emb: Optional[torch.Tensor],
        history_context: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if cond is not None:
            return cond
        if history_context is None and diffusion_time_emb is None:
            return None
        if history_context is None:
            global_ctx = diffusion_time_emb
        elif history_context.dim() == 3:
            global_ctx = history_context.mean(dim=1)
        else:
            global_ctx = history_context
        if diffusion_time_emb is not None:
            if diffusion_time_emb.dim() == 3:
                diffusion_time_emb = diffusion_time_emb.squeeze(1)
            global_ctx = global_ctx + diffusion_time_emb
        return global_ctx

    def effective_poles(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        cond: Optional[torch.Tensor] = None,
        diffusion_time_emb: Optional[torch.Tensor] = None,
        history_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-sample effective poles rho, omega with stability constraints."""
        rho0, omega0 = self._base_poles(dtype, device)  # [K], [K]
        rho = rho0.unsqueeze(0).expand(batch_size, self.k).contiguous()
        omega = omega0.unsqueeze(0).expand(batch_size, self.k).contiguous()

        cond = self._merge_condition(cond, diffusion_time_emb, history_context)
        if self.cond_dim is not None and cond is not None:
            d_rho = self.rho_perturb_scale * torch.tanh(self.rho_refine(cond))
            d_omega = self.omega_perturb_scale * torch.tanh(self.omega_refine(cond))

            rho = F.softplus(rho0.unsqueeze(0) + d_rho) + self.alpha_min

            p0 = (omega0 / self.omega_max).clamp(1e-4, 1 - 1e-4)
            logit0 = torch.log(p0) - torch.log1p(-p0)
            omega = self.omega_max * torch.sigmoid(logit0.unsqueeze(0) + d_omega)

        return rho, omega  # [B,K], [B,K]

    @staticmethod
    def relative_time(
        B: int,
        T: int,
        dtype: torch.dtype,
        device: torch.device,
        dt: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return relative time t_rel with t_rel[:,0]=0, shape [B,T,1]."""
        if t is not None:
            t = t.to(device=device, dtype=dtype)
            if t.dim() == 2:
                t = t.unsqueeze(-1)
            return t - t[:, :1]
        if dt is not None:
            dt = dt.to(device=device, dtype=dtype)
            if dt.dim() == 2:
                dt = dt.unsqueeze(-1)
            t_abs = torch.cumsum(dt, dim=1)
            return t_abs - t_abs[:, :1]
        return torch.arange(T, device=device, dtype=dtype).view(1, T, 1).expand(B, T, 1)

    @staticmethod
    def basis_matrix(t_rel: torch.Tensor, rho: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """Compute damped cosine/sine basis matrix A_lap, shape [B,T,2K]."""
        rho_ = rho.unsqueeze(1)      # [B,1,K]
        omega_ = omega.unsqueeze(1)  # [B,1,K]
        decay = torch.exp(-t_rel * rho_)
        angle = t_rel * omega_
        cos_basis = decay * torch.cos(angle)
        sin_basis = decay * torch.sin(angle)
        return torch.cat([cos_basis, sin_basis], dim=-1).contiguous()

    def _theta_diag_projection(self, x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """Fast approximate projection (no solve): theta = (A^T x) / (diag(A^T A)+eps)."""
        # x: [B,T,D], basis: [B,T,2K]
        Bt = basis.transpose(1, 2)  # [B,2K,T]
        num = torch.bmm(Bt, x)      # [B,2K,D]
        den = (basis * basis).sum(dim=1).clamp_min(self.proj_eps)  # [B,2K]
        return (num / den.unsqueeze(-1)).contiguous()

    def _theta_time_attention(self, x: torch.Tensor, t_rel: torch.Tensor, rho: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """Learned spectral cross-attention to obtain residues."""
        B, T, _ = x.shape
        # Build 2K queries from poles + component id
        rho2 = rho.repeat(1, 2).unsqueeze(-1)   # [B,2K,1]
        omg2 = omega.repeat(1, 2).unsqueeze(-1) # [B,2K,1]
        comp = torch.cat(
            [
                torch.zeros(B, self.k, 1, device=x.device, dtype=x.dtype),
                torch.ones(B, self.k, 1, device=x.device, dtype=x.dtype),
            ],
            dim=1,
        )  # [B,2K,1]
        pole_feat = torch.cat([rho2, omg2, comp], dim=-1)  # [B,2K,3]
        q = self.q_norm(self.mode_queries + self.pole_embedding(pole_feat))  # [B,2K,H]

        # Keys/values from x with time embedding for ordering.
        z_emb = self.input_proj(x) + self.time_emb(t_rel)
        k = self.k_norm(z_emb)
        v = self.v_norm(z_emb)

        out, _ = self.attention(q, k, v, need_weights=False)  # [B,2K,H]
        theta = self.out_proj(out)  # [B,2K,D]
        return theta.contiguous()

    def forward(
        self,
        x: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        diffusion_time_emb: Optional[torch.Tensor] = None,
        history_context: Optional[torch.Tensor] = None,
        use_time_attn: bool = True,
        poles: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_t_rel: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute modal residues and effective poles.

        Args:
            x: [B,T,D]
            dt/t: timing info
            cond: [B,cond_dim]
            diffusion_time_emb: [B,cond_dim] or [B,1,cond_dim]
            history_context: [B,cond_dim] or [B,T_hist,cond_dim]
            use_time_attn: whether to use spectral cross-attention (True) or
                           diagonal projection (False)
            poles: optional precomputed (rho, omega), each [B,K]
            return_t_rel: if True, also returns t_rel [B,T,1]

        Returns:
            theta: [B,2K,D]
            rho:   [B,K]
            omega: [B,K]
            t_rel: [B,T,1] if return_t_rel else None
        """
        if x.dim() != 3 or x.size(-1) != self.feat_dim:
            raise ValueError(f"Input x must be [B, T, {self.feat_dim}]")
        B, T, _ = x.shape

        t_rel = self.relative_time(B, T, x.dtype, x.device, dt=dt, t=t)
        if poles is None:
            rho, omega = self.effective_poles(
                B,
                x.dtype,
                x.device,
                cond=cond,
                diffusion_time_emb=diffusion_time_emb,
                history_context=history_context,
            )
        else:
            rho, omega = poles
            if rho.shape != (B, self.k) or omega.shape != (B, self.k):
                raise ValueError("poles must be (rho, omega) with shape [B,K]")

        if use_time_attn:
            theta = self._theta_time_attention(x, t_rel, rho, omega)
        else:
            basis = self.basis_matrix(t_rel, rho, omega)
            theta = self._theta_diag_projection(x, basis)

        return theta, rho, omega, (t_rel if return_t_rel else None)


class ModalSynthesizer(nn.Module):
    """Explicit synthesis from residues and effective poles.

    Computes y(t) = A_lap(t; rho, omega) @ theta, then optionally applies a small
    residual MLP to capture transients.
    """

    def __init__(
        self,
        encoder: ModalPredictor,
        hidden_dim: Optional[int] = None,
        use_mlp_residual: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.use_mlp_residual = bool(use_mlp_residual)
        D = encoder.feat_dim
        H = int(hidden_dim if hidden_dim is not None else D * 2)

        if self.use_mlp_residual:
            self.norm = nn.LayerNorm(D)
            self.mlp_in = spectral_norm(nn.Linear(D, H * 2))
            self.mlp_out = spectral_norm(nn.Linear(H, D))
            nn.init.zeros_(self.mlp_out.weight)
            nn.init.zeros_(self.mlp_out.bias)

    def forward(
        self,
        theta: torch.Tensor,  # [B,2K,D]
        rho: torch.Tensor,    # [B,K]
        omega: torch.Tensor,  # [B,K]
        dt: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        target_T: Optional[int] = None,
    ) -> torch.Tensor:
        if theta.dim() != 3:
            raise ValueError("theta must be [B,2K,D]")
        B = theta.shape[0]

        if t is not None:
            T = t.shape[1]
        elif dt is not None:
            T = dt.shape[1]
        elif target_T is not None:
            T = int(target_T)
        else:
            raise ValueError("Provide t or dt or target_T to determine output length")

        t_rel = self.encoder.relative_time(B, T, theta.dtype, theta.device, dt=dt, t=t)
        basis = self.encoder.basis_matrix(t_rel, rho, omega)  # [B,T,2K]
        y = torch.bmm(basis, theta)  # [B,T,D]

        if not self.use_mlp_residual:
            return y

        res = self.norm(y)
        gate, val = self.mlp_in(res).chunk(2, dim=-1)
        res = val * F.gelu(gate)
        return y + self.mlp_out(res)
