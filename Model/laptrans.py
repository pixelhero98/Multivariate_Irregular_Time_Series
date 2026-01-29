import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

__all__ = ["ModalPredictor", "ModalSynthesizer"]


def inv_softplus(y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Inverse of softplus function:
        softplus(x) = log(1 + exp(x))
        x = log(exp(y) - 1)   (for y > 0)

    Used for precise initialization of parameters constrained by Softplus.
    """
    return torch.log(torch.expm1(y.clamp_min(eps)))


class ModalPredictor(nn.Module):
    """
    Modal analysis encoder: maps a time sequence to modal residues.

    - Poles (rho, omega): global dictionary + instance-specific refinement.
      Constrained via Softplus/Sigmoid to ensure stability (rho > 0 => Re(s) < 0).
    - Residues (theta): spectral cross-attention (modes query time).

    Output:
        theta: [B, 2K, D]  (first K cosine coeffs, last K sine coeffs)
        rho:   [B, K]
        omega: [B, K]
    """

    def __init__(
        self,
        k: int,
        feat_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        rho_min: float = 1e-6,
        omega_max: float = math.pi,
        cond_dim: Optional[int] = None,
        rho_perturb_scale: float = 0.5,
        omega_perturb_scale: float = 0.5,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        time_scale: float = 1.0,        # scale timestamps to avoid exp underflow
        learn_time_scale: bool = False, # optionally learn the time scale
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.feat_dim = int(feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.rho_min = float(rho_min)
        self.omega_max = float(omega_max)
        self.cond_dim = cond_dim
        self.rho_perturb_scale = float(rho_perturb_scale)
        self.omega_perturb_scale = float(omega_perturb_scale)

        if self.hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        # --- Time Scaling (log-param ensures positivity) ---
        init_scale = math.log(max(time_scale, 1e-6))
        if learn_time_scale:
            self.log_time_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        else:
            self.register_buffer("log_time_scale", torch.tensor(init_scale, dtype=torch.float32))

        # --- Base Poles (global learnable dictionary) ---
        self._rho_raw = nn.Parameter(torch.empty(self.k))   # pre-softplus
        self._omega_raw = nn.Parameter(torch.empty(self.k)) # pre-sigmoid (logit)

        # --- Context Refinement Heads ---
        if cond_dim is not None:
            self.cond_norm = nn.LayerNorm(cond_dim)

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
            # Zero-init: start training at base poles
            nn.init.zeros_(self.rho_refine[-1].weight)
            nn.init.zeros_(self.rho_refine[-1].bias)
            nn.init.zeros_(self.omega_refine[-1].weight)
            nn.init.zeros_(self.omega_refine[-1].bias)

        # --- Spectral Cross-Attention ---
        self.mode_queries = nn.Parameter(torch.randn(1, 2 * self.k, hidden_dim) * 0.02)

        self.pole_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.time_emb = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(hidden_dim, feat_dim)

        self.q_norm = nn.LayerNorm(hidden_dim)
        self.k_norm = nn.LayerNorm(hidden_dim)
        self.v_norm = nn.LayerNorm(hidden_dim)

        # NOTE: precompute component ids (cos/sin) to avoid allocating every forward.
        comp = torch.cat([torch.zeros(1, self.k, 1), torch.ones(1, self.k, 1)], dim=1)  # [1,2K,1]
        self.register_buffer("_comp_id", comp)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize poles using inverse activations to guarantee specific start ranges."""
        with torch.no_grad():
            # rho ~ U(0.01, 0.2) for stable decay
            target_rho = torch.empty_like(self._rho_raw).uniform_(0.01, 0.2)
            self._rho_raw.copy_(inv_softplus((target_rho - self.rho_min).clamp_min(1e-6)))

            # omega ~ LogUniform(0.01*max, 0.95*max) for frequency diversity
            low_log = math.log(0.01 * self.omega_max)
            high_log = math.log(0.95 * self.omega_max)
            target_omega = torch.exp(torch.empty_like(self._omega_raw).uniform_(low_log, high_log))
            p = (target_omega / self.omega_max).clamp(1e-4, 1 - 1e-4)
            self._omega_raw.copy_(torch.log(p) - torch.log1p(-p))  # inverse sigmoid

    def time_scale_value(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.exp(self.log_time_scale.to(device=device, dtype=dtype)).clamp_min(1e-6)

    def _merge_condition(
        self,
        diffusion_time_emb: Optional[torch.Tensor],
        history_context: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Merge diffusion embedding and history context into a single conditioning vector [B, cond_dim]."""
        global_ctx = None

        if history_context is not None:
            global_ctx = history_context.mean(dim=1) if history_context.dim() == 3 else history_context

        if diffusion_time_emb is not None:
            dt_emb = diffusion_time_emb.squeeze(1) if diffusion_time_emb.dim() == 3 else diffusion_time_emb
            global_ctx = dt_emb if global_ctx is None else (global_ctx + dt_emb)

        return global_ctx

    def modal_poles(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        diffusion_time_emb: Optional[torch.Tensor] = None,
        history_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute effective poles (rho, omega), refining in raw/logit space before activation."""
        rho_raw0 = self._rho_raw.to(device=device, dtype=dtype)        # [K]
        omega_logit0 = self._omega_raw.to(device=device, dtype=dtype)  # [K]

        rho_raw = rho_raw0.unsqueeze(0).expand(batch_size, self.k).contiguous()
        omega_logit = omega_logit0.unsqueeze(0).expand(batch_size, self.k).contiguous()

        cond = self._merge_condition(diffusion_time_emb, history_context)
        if self.cond_dim is not None and cond is not None:
            if cond.shape[-1] != self.cond_dim:
                raise ValueError(f"cond last dim {cond.shape[-1]} != cond_dim {self.cond_dim}")

            cond = self.cond_norm(cond)

            d_rho = self.rho_perturb_scale * torch.tanh(self.rho_refine(cond))       # [B,K]
            d_omega = self.omega_perturb_scale * torch.tanh(self.omega_refine(cond)) # [B,K]

            rho_raw = rho_raw + d_rho
            omega_logit = omega_logit + d_omega

        rho = F.softplus(rho_raw) + self.rho_min
        omega = self.omega_max * torch.sigmoid(omega_logit)
        return rho, omega

    def relative_time(
        self,
        B: int,
        T: int,
        dtype: torch.dtype,
        device: torch.device,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute scaled relative time t_rel normalized to start at 0.
        Returns: [B, T, 1]
        """
        scale = self.time_scale_value(dtype, device)

        if t is not None:
            t = t.to(device=device, dtype=dtype)
            if t.dim() == 2:
                t = t.unsqueeze(-1)
            t_rel = t - t[:, :1]
            return t_rel / scale

        return torch.arange(T, device=device, dtype=dtype).view(1, T, 1).expand(B, T, 1) / scale

    @staticmethod
    def basis_matrix(t_rel: torch.Tensor, rho: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """Compute damped sinusoidal basis matrix. Returns: [B, T, 2K]"""
        rho_ = rho.unsqueeze(1)      # [B,1,K]
        omega_ = omega.unsqueeze(1)  # [B,1,K]
        decay = torch.exp(-t_rel * rho_)
        angle = t_rel * omega_
        return torch.cat([decay * torch.cos(angle), decay * torch.sin(angle)], dim=-1).contiguous()

    def _theta_time_attention(
        self,
        x: torch.Tensor,
        t_rel: torch.Tensor,
        rho: torch.Tensor,
        omega: torch.Tensor,
    ) -> torch.Tensor:
        """Extract residues via spectral cross-attention."""
        B, _, _ = x.shape

        # Pole features for each of 2K modes: [rho, omega, comp_id]
        rho2 = rho.repeat(1, 2).unsqueeze(-1)     # [B,2K,1]
        omg2 = omega.repeat(1, 2).unsqueeze(-1)   # [B,2K,1]
        comp = self._comp_id.expand(B, -1, -1).to(device=x.device, dtype=x.dtype)  # [B,2K,1]
        pole_feat = torch.cat([rho2, omg2, comp], dim=-1)  # [B,2K,3]

        # Queries: learned embeddings + pole embedding (broadcast to B)
        mq = self.mode_queries.to(device=x.device, dtype=x.dtype).expand(B, -1, -1)  # [B,2K,H]
        q = self.q_norm(mq + self.pole_embedding(pole_feat))

        # Keys/values: projected input + time embedding
        z = self.input_proj(x) + self.time_emb(t_rel)
        z = self.proj_drop(z)
        k = self.k_norm(z)
        v = self.v_norm(z)

        out, _ = self.attention(q, k, v, need_weights=False)
        return self.out_proj(out).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        diffusion_time_emb: Optional[torch.Tensor] = None,
        history_context: Optional[torch.Tensor] = None,
        poles: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_t_rel: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if x.dim() != 3 or x.size(-1) != self.feat_dim:
            raise ValueError(f"Input x must be [B, T, {self.feat_dim}]")

        B, T, _ = x.shape
        t_rel = self.relative_time(B, T, x.dtype, x.device, t=t)

        if poles is None:
            rho, omega = self.modal_poles(
                B, x.dtype, x.device,
                diffusion_time_emb=diffusion_time_emb,
                history_context=history_context,
            )
        else:
            # NOTE: ensure dtype/device correctness when poles are cached or produced elsewhere.
            rho, omega = poles
            rho = rho.to(device=x.device, dtype=x.dtype)
            omega = omega.to(device=x.device, dtype=x.dtype)
            if rho.shape != (B, self.k) or omega.shape != (B, self.k):
                raise ValueError(f"poles must be [B,K]; got rho {rho.shape}, omega {omega.shape}")

        theta = self._theta_time_attention(x, t_rel, rho, omega)
        return theta, rho, omega, (t_rel if return_t_rel else None)


class ModalSynthesizer(nn.Module):
    """Explicit synthesis: y(t) = A_lap(t) @ theta + MLP(residual)."""

    def __init__(
        self,
        encoder: ModalPredictor,
        hidden_dim: Optional[int] = None,
        use_mlp_residual: bool = True,
        residual_dropout: float = 0.0,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.use_mlp_residual = bool(use_mlp_residual)
        D = encoder.feat_dim
        H = int(hidden_dim if hidden_dim is not None else D * 2)

        self.res_drop = nn.Dropout(residual_dropout)

        if self.use_mlp_residual:
            self.norm = nn.LayerNorm(D)
            lin1 = nn.Linear(D, H * 2)
            lin2 = nn.Linear(H, D)

            if use_spectral_norm:
                lin1 = spectral_norm(lin1)
                lin2 = spectral_norm(lin2)

            self.mlp_in = lin1
            self.mlp_out = lin2
            nn.init.zeros_(self.mlp_out.weight)
            nn.init.zeros_(self.mlp_out.bias)

    def forward(
        self,
        theta: torch.Tensor,  # [B, 2K, D]
        rho: torch.Tensor,    # [B, K]
        omega: torch.Tensor,  # [B, K]
        t: Optional[torch.Tensor] = None,
        target_T: Optional[int] = None,
    ) -> torch.Tensor:
        if theta.dim() != 3:
            raise ValueError("theta must be [B, 2K, D]")

        B = theta.shape[0]

        if t is not None:
            T = t.shape[1]
        elif target_T is not None:
            T = int(target_T)
        else:
            raise ValueError("Provide t or target_T")

        # Use encoder's method to ensure consistent time scaling
        t_rel = self.encoder.relative_time(B, T, theta.dtype, theta.device, t=t)

        basis = self.encoder.basis_matrix(t_rel, rho, omega)
        y = torch.bmm(basis, theta)

        if not self.use_mlp_residual:
            return y

        res = self.norm(y)
        gate, val = self.mlp_in(res).chunk(2, dim=-1)
        res = val * F.gelu(gate)
        res = self.res_drop(res)
        return y + self.mlp_out(res)
