import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

__all__ = ["ModalPredictor", "ModalSynthesizer"]


class ModalPredictor(nn.Module):
    """
    Modal analysis encoder: Maps a time sequence to modal residues.

    Implements paper-aligned *effective* modal parameterization.
    - Poles (rho, omega): History-conditioned, stable by construction (Real(s) < 0).
    - Residues (theta): Extracted via spectral cross-attention.

    Output:
        theta: [B, 2K, D]  (First K: cosine coeffs, Last K: sine coeffs)
        rho:   [B, K]      (Decay rates)
        omega: [B, K]      (Frequencies)
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

        # --- Base Poles (Global Learnable Dictionary) ---
        self._rho_raw = nn.Parameter(torch.empty(self.k))
        self._omega_raw = nn.Parameter(torch.empty(self.k))

        # --- Context Refinement Heads ---
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
            # Init to zero: start training with base poles
            nn.init.zeros_(self.rho_refine[-1].weight)
            nn.init.zeros_(self.rho_refine[-1].bias)
            nn.init.zeros_(self.omega_refine[-1].weight)
            nn.init.zeros_(self.omega_refine[-1].bias)

        # --- Spectral Cross-Attention ---
        # Queries: Mode-specific learned embeddings + Pole embeddings
        self.mode_queries = nn.Parameter(torch.randn(1, 2 * self.k, hidden_dim) * 0.02)
        self.pole_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Keys/Values: Input features + Time embeddings
        self.time_emb = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(feat_dim, hidden_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(hidden_dim, feat_dim)

        # Normalization
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.k_norm = nn.LayerNorm(hidden_dim)
        self.v_norm = nn.LayerNorm(hidden_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize poles to ensure stability and frequency coverage."""
        with torch.no_grad():
            # rho ~ U(0.01, 0.2)
            target_rho = torch.empty_like(self._rho_raw).uniform_(0.01, 0.2)
            y = (target_rho - self.rho_min).clamp_min(1e-8)
            self._rho_raw.copy_(torch.log(torch.expm1(y)))

            # omega ~ LogUniform(0.01*max, 0.95*max)
            low_log = math.log(0.01 * self.omega_max)
            high_log = math.log(0.95 * self.omega_max)
            target_omega = torch.exp(torch.empty_like(self._omega_raw).uniform_(low_log, high_log))
            p = (target_omega / self.omega_max).clamp(1e-4, 1 - 1e-4)
            self._omega_raw.copy_(torch.log(p) - torch.log1p(-p))

    def _base_poles(self, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        rho = F.softplus(self._rho_raw.to(device=device, dtype=dtype)) + self.rho_min
        omega = self.omega_max * torch.sigmoid(self._omega_raw.to(device=device, dtype=dtype))
        return rho, omega

    def _merge_condition(
            self,
            diffusion_time_emb: Optional[torch.Tensor],
            history_context: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Merges diffusion embedding and history context efficiently."""
        if history_context is None:
            global_ctx = None
        elif history_context.dim() == 3:
            global_ctx = history_context.mean(dim=1)
        else:
            global_ctx = history_context

        if diffusion_time_emb is not None:
            dt_emb = diffusion_time_emb.squeeze(1) if diffusion_time_emb.dim() == 3 else diffusion_time_emb
            global_ctx = dt_emb if global_ctx is None else global_ctx + dt_emb

        return global_ctx

    def modal_poles(
            self,
            batch_size: int,
            dtype: torch.dtype,
            device: torch.device,
            diffusion_time_emb: Optional[torch.Tensor] = None,
            history_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute effective poles (rho, omega) with refinement."""
        rho0, omega0 = self._base_poles(dtype, device)

        # [B, K] expansion
        rho = rho0.unsqueeze(0).expand(batch_size, self.k).contiguous()
        omega = omega0.unsqueeze(0).expand(batch_size, self.k).contiguous()

        # Apply Context Refinement
        cond = self._merge_condition(diffusion_time_emb, history_context)

        if self.cond_dim is not None and cond is not None:
            d_rho = self.rho_perturb_scale * torch.tanh(self.rho_refine(cond))
            d_omega = self.omega_perturb_scale * torch.tanh(self.omega_refine(cond))

            rho = F.softplus(rho0.unsqueeze(0) + d_rho) + self.rho_min

            p0 = (omega0 / self.omega_max).clamp(1e-4, 1 - 1e-4)
            logit0 = torch.log(p0) - torch.log1p(-p0)
            omega = self.omega_max * torch.sigmoid(logit0.unsqueeze(0) + d_omega)

        return rho, omega

    @staticmethod
    def relative_time(
            B: int,
            T: int,
            dtype: torch.dtype,
            device: torch.device,
            t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute relative time t_rel, normalized to start at 0.
        Returns: [B, T, 1]
        """
        if t is not None:
            t = t.to(device=device, dtype=dtype)
            if t.dim() == 2:
                t = t.unsqueeze(-1)
            # Normalize: t_rel = t - t_start
            return t - t[:, :1]

        # Fallback: Integer grid
        return torch.arange(T, device=device, dtype=dtype).view(1, T, 1).expand(B, T, 1)

    @staticmethod
    def basis_matrix(t_rel: torch.Tensor, rho: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """
        Compute damped sinusoidal basis matrix A_lap.
        Returns: [B, T, 2K]
        """
        # Broadcasting: [B, T, K]
        rho_ = rho.unsqueeze(1)
        omega_ = omega.unsqueeze(1)

        decay = torch.exp(-t_rel * rho_)
        angle = t_rel * omega_

        # Concat [Cos | Sin] -> [B, T, 2K]
        return torch.cat([decay * torch.cos(angle), decay * torch.sin(angle)], dim=-1).contiguous()

    def _theta_time_attention(self, x: torch.Tensor, t_rel: torch.Tensor, rho: torch.Tensor,
                              omega: torch.Tensor) -> torch.Tensor:
        """Extract residues via spectral cross-attention."""
        B, T, _ = x.shape

        # --- 1. Construct Queries (Spectral Modes) ---
        # Features: [rho, omega, type_id]
        rho2 = rho.repeat(1, 2).unsqueeze(-1)
        omg2 = omega.repeat(1, 2).unsqueeze(-1)
        comp = torch.cat([
            torch.zeros(B, self.k, 1, device=x.device, dtype=x.dtype),  # Cosine
            torch.ones(B, self.k, 1, device=x.device, dtype=x.dtype),  # Sine
        ], dim=1)

        pole_feat = torch.cat([rho2, omg2, comp], dim=-1)  # [B, 2K, 3]

        # Q = Learned Queries + Pole Embeddings
        q = self.q_norm(self.mode_queries + self.pole_embedding(pole_feat))

        # --- 2. Construct Keys/Values (Temporal Data) ---
        # K, V = Project(Input) + Embed(Time)
        z_emb = self.input_proj(x) + self.time_emb(t_rel)
        k = self.k_norm(z_emb)
        v = self.v_norm(z_emb)

        # --- 3. Attention ---
        # [B, 2K, H] -> [B, 2K, D]
        out, _ = self.attention(q, k, v, need_weights=False)
        return self.out_proj(out).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        t_rel: Optional[torch.Tensor] = None,          # NEW
        diffusion_time_emb: Optional[torch.Tensor] = None,
        history_context: Optional[torch.Tensor] = None,
        poles: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_t_rel: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if x.dim() != 3 or x.size(-1) != self.feat_dim:
            raise ValueError(f"Input x must be [B, T, {self.feat_dim}]")

        B, T, _ = x.shape

        # --- Unify: use provided t_rel or compute once here ---
        if t_rel is None:
            t_rel = self.relative_time(B, T, x.dtype, x.device, t=t)
        else:
            if t_rel.shape != (B, T, 1):
                raise ValueError(f"t_rel must be [B, T, 1]; got {t_rel.shape}")
            t_rel = t_rel.to(device=x.device, dtype=x.dtype)

        # Poles
        if poles is None:
            rho, omega = self.modal_poles(
                B, x.dtype, x.device,
                diffusion_time_emb=diffusion_time_emb,
                history_context=history_context,
            )
        else:
            rho, omega = poles
            rho = rho.to(device=x.device, dtype=x.dtype)
            omega = omega.to(device=x.device, dtype=x.dtype)
            if rho.shape != (B, self.k) or omega.shape != (B, self.k):
                raise ValueError(f"poles must be [B,K]; got rho {rho.shape}, omega {omega.shape}")

        # Residues
        theta = self._theta_time_attention(x, t_rel, rho, omega)
        return theta, rho, omega, (t_rel if return_t_rel else None)


class ModalSynthesizer(nn.Module):
    """
    Explicit synthesis: y(t) = A_lap(t) @ theta + MLP(residual).
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
        theta: torch.Tensor,  # [B, 2K, D]
        rho: torch.Tensor,    # [B, K]
        omega: torch.Tensor,  # [B, K]
        t: Optional[torch.Tensor] = None,
        t_rel: Optional[torch.Tensor] = None,          # NEW
        target_T: Optional[int] = None,
    ) -> torch.Tensor:
        if theta.dim() != 3:
            raise ValueError("theta must be [B, 2K, D]")
        B = theta.shape[0]

        # Determine T and t_rel
        if t_rel is None:
            if t is not None:
                T = t.shape[1]
            elif target_T is not None:
                T = int(target_T)
            else:
                raise ValueError("Provide t, t_rel, or target_T")
            t_rel = self.encoder.relative_time(B, T, theta.dtype, theta.device, t=t)
        else:
            if t_rel.dim() != 3 or t_rel.size(0) != B or t_rel.size(-1) != 1:
                raise ValueError(f"t_rel must be [B, T, 1]; got {t_rel.shape}")
            t_rel = t_rel.to(device=theta.device, dtype=theta.dtype)

        # Basis + synthesis
        rho = rho.to(device=theta.device, dtype=theta.dtype)
        omega = omega.to(device=theta.device, dtype=theta.dtype)
        basis = self.encoder.basis_matrix(t_rel, rho, omega)  # [B,T,2K]
        y = torch.bmm(basis, theta)                            # [B,T,D]

        if not self.use_mlp_residual:
            return y

        # Residual refinement
        res = self.norm(y)
        gate, val = self.mlp_in(res).chunk(2, dim=-1)
        res = val * F.gelu(gate)
        res = self.res_drop(res)
        return y + self.mlp_out(res)
