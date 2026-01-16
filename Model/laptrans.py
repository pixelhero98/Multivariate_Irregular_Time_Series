import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

__all__ = ["LaplaceTransformEncoder", "LaplacePseudoInverse"]


class LaplaceTransformEncoder(nn.Module):
    """
    Learned Modal Projection via Spectral Cross-Attention (no ridge solve).

    Output:
        theta: [B, 2K, D]  (first K -> cosine residues, last K -> sine residues)
        rho:   [B, K]
        omega: [B, K]

    Key design choice for efficiency/stability:
        - Keys depend ONLY on timestamps (time features).
        - Values come from x (projected).
        - Queries come from effective poles (rho, omega) + component id (cos/sin).
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
        time_nfreq: int = 16,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.feat_dim = int(feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.alpha_min = float(alpha_min)
        self.omega_max = float(omega_max)
        self.rho_perturb_scale = float(rho_perturb_scale)
        self.omega_perturb_scale = float(omega_perturb_scale)
        self.cond_dim = cond_dim

        if self.hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        # --- 1) Base poles + conditioned perturbations ---
        self._rho_raw = nn.Parameter(torch.empty(self.k))
        self._omega_raw = nn.Parameter(torch.empty(self.k))

        if cond_dim is not None:
            self.to_poles = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, 2 * self.k),
            )
            nn.init.zeros_(self.to_poles[-1].weight)
            nn.init.zeros_(self.to_poles[-1].bias)

        # --- 2) Queries from (rho, omega, component_id) ---
        # component_id: 0 for cos, 1 for sin
        self.pole_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.comp_emb = nn.Parameter(torch.zeros(1, 2 * self.k, hidden_dim))
        nn.init.normal_(self.comp_emb, mean=0.0, std=0.02)

        # --- 3) Keys from time only; Values from x only ---
        self.time_nfreq = int(time_nfreq)
        # Fixed log-spaced frequencies for Fourier time features (buffer, not learned)
        # We include frequencies up to ~omega_max (you can tune if your time units differ).
        freqs = torch.logspace(
            math.log10(1.0),
            math.log10(max(2.0, float(self.omega_max))),
            steps=self.time_nfreq,
        )
        self.register_buffer("time_freqs", freqs, persistent=False)

        # time features dim = 1 (t) + 2*time_nfreq (sin/cos)
        time_feat_dim = 1 + 2 * self.time_nfreq
        self.time_key_proj = nn.Sequential(
            nn.Linear(time_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.value_proj = nn.Linear(feat_dim, hidden_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.out_proj = nn.Linear(hidden_dim, feat_dim)

        # Optional norms (helpful for stability)
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.k_norm = nn.LayerNorm(hidden_dim)
        self.v_norm = nn.LayerNorm(hidden_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            # Initialize rho in (0.01, 0.2)
            target_rho = torch.empty_like(self._rho_raw).uniform_(0.01, 0.2)
            y = (target_rho - self.alpha_min).clamp_min(1e-8)
            self._rho_raw.copy_(torch.log(torch.expm1(y)))

            # Initialize omega in ~[0.01*omega_max, 0.95*omega_max]
            low_log = math.log(0.01 * self.omega_max)
            high_log = math.log(0.95 * self.omega_max)
            target_omega = torch.exp(torch.empty_like(self._omega_raw).uniform_(low_log, high_log))
            p = (target_omega / self.omega_max).clamp(1e-4, 1 - 1e-4)
            self._omega_raw.copy_(torch.log(p) - torch.log1p(-p))

    def _base_poles(self, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        rho = F.softplus(self._rho_raw.to(device=device, dtype=dtype)) + self.alpha_min
        omega = self.omega_max * torch.sigmoid(self._omega_raw.to(device=device, dtype=dtype))
        return rho, omega

    def effective_poles(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rho0, omega0 = self._base_poles(dtype, device)  # [K], [K]
        rho = rho0.unsqueeze(0).expand(batch_size, self.k).contiguous()
        omega = omega0.unsqueeze(0).expand(batch_size, self.k).contiguous()

        if self.cond_dim is not None and cond is not None:
            delta = self.to_poles(cond).view(batch_size, 2, self.k)
            d_rho = self.rho_perturb_scale * torch.tanh(delta[:, 0])
            d_omega = self.omega_perturb_scale * torch.tanh(delta[:, 1])

            rho = F.softplus(rho0.unsqueeze(0) + d_rho) + self.alpha_min

            p0 = (omega0 / self.omega_max).clamp(1e-4, 1 - 1e-4)
            logit0 = torch.log(p0) - torch.log1p(-p0)
            omega = self.omega_max * torch.sigmoid(logit0.unsqueeze(0) + d_omega)

        return rho, omega  # [B,K], [B,K]

    @staticmethod
    def _relative_time(
        B: int, T: int, dtype: torch.dtype, device: torch.device,
        dt: Optional[torch.Tensor], t: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # Returns [B, T, 1] relative time with t[:,0]=0
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

    def _time_features(self, t_rel: torch.Tensor) -> torch.Tensor:
        # t_rel: [B,T,1] -> features: [B,T, 1 + 2F]
        # Use fixed Fourier features; cheap and expressive.
        # angles: [B,T,F]
        angles = t_rel * self.time_freqs.view(1, 1, -1).to(t_rel.dtype)
        return torch.cat([t_rel, torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    [B, T, D]
            dt/t: timing information
            cond: [B, cond_dim] (e.g., concat(diffusion_step_embed, pooled_summary))

        Returns:
            theta: [B, 2K, D]
            rho:   [B, K]
            omega: [B, K]
        """
        B, T, D = x.shape
        t_rel = self._relative_time(B, T, x.dtype, x.device, dt, t)  # [B,T,1]

        # 1) Effective poles
        rho, omega = self.effective_poles(B, x.dtype, x.device, cond)  # [B,K], [B,K]

        # 2) Build 2K pole queries with explicit component id
        rho2 = rho.repeat(1, 2).unsqueeze(-1)     # [B,2K,1]
        omg2 = omega.repeat(1, 2).unsqueeze(-1)   # [B,2K,1]
        comp = torch.cat(
            [
                torch.zeros(B, self.k, 1, device=x.device, dtype=x.dtype),
                torch.ones(B, self.k, 1, device=x.device, dtype=x.dtype),
            ],
            dim=1,
        )  # [B,2K,1]
        pole_feat = torch.cat([rho2, omg2, comp], dim=-1)  # [B,2K,3]
        queries = self.pole_embedding(pole_feat) + self.comp_emb  # [B,2K,H]
        queries = self.q_norm(queries)

        # 3) Keys from time only; values from x only
        time_feat = self._time_features(t_rel)          # [B,T,1+2F]
        keys = self.time_key_proj(time_feat)            # [B,T,H]
        keys = self.k_norm(keys)

        values = self.value_proj(x)                     # [B,T,H]
        values = self.v_norm(values)

        # 4) Cross-attention: modal queries attend over time
        attn_out, _ = self.attention(queries, keys, values, need_weights=False)  # [B,2K,H]

        # 5) Project back to residues in latent dimension
        theta = self.out_proj(attn_out)  # [B,2K,D]

        return theta, rho, omega


class LaplacePseudoInverse(nn.Module):
    """
    Explicit synthesis from residues and effective poles:
        y(t) = [phi(t), psi(t)] @ theta

    We compute basis on the fly to avoid passing [B,T,2K] around.
    """

    def __init__(
        self,
        encoder: LaplaceTransformEncoder,
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

    @staticmethod
    def _compute_basis(t_rel: torch.Tensor, rho: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        # t_rel: [B,T,1], rho/omega: [B,K] -> basis: [B,T,2K]
        rho_ = rho.unsqueeze(1)      # [B,1,K]
        omega_ = omega.unsqueeze(1)  # [B,1,K]
        decay = torch.exp(-t_rel * rho_)
        angle = t_rel * omega_
        return torch.cat([decay * torch.cos(angle), decay * torch.sin(angle)], dim=-1)

    def forward(
        self,
        theta: torch.Tensor,  # [B,2K,D]
        rho: torch.Tensor,    # [B,K]
        omega: torch.Tensor,  # [B,K]
        dt: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        target_T: Optional[int] = None,
    ) -> torch.Tensor:
        """
        If dt/t are not provided, target_T must be provided.
        """
        B = theta.shape[0]

        if t is not None:
            T = t.shape[1]
        elif dt is not None:
            T = dt.shape[1]
        elif target_T is not None:
            T = int(target_T)
        else:
            raise ValueError("LaplacePseudoInverse: provide t or dt or target_T (cannot infer T from theta).")

        t_rel = self.encoder._relative_time(B, T, theta.dtype, theta.device, dt, t)  # [B,T,1]
        basis = self._compute_basis(t_rel, rho, omega)  # [B,T,2K]

        y = torch.bmm(basis, theta)  # [B,T,D]

        if self.use_mlp_residual:
            res = self.norm(y)
            gate, val = self.mlp_in(res).chunk(2, dim=-1)
            res = val * F.gelu(gate)
            y = y + self.mlp_out(res)

        return y
