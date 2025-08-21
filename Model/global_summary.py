import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Tuple
import torch.nn.functional as F
from pos_time_emb import get_sinusoidal_pos_emb as _pos_emb
from laptrans import LearnableLaplacianBasis

# -------------------------------
# System Context Summarizer
# -------------------------------

# ---------- Tiny heads: [B,T,N,F] -> [B,T,N] ----------
class TVHead(nn.Module):
    """Per-entity scalar readout from features at each time."""
    def __init__(self, feat_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,T,N,F]
        return self.net(x).squeeze(-1)                   #    [B,T,N]


# ---------- Pole-wise derivative in Laplace-feature space ----------
class PolewiseDiff(nn.Module):
    """
    Given Laplace features L(x) ∈ [B,T,2K] as K blocks [cos, sin],
    approximate L(dx/dt) via K independent 2×2 blocks.

    Mode A (free):      K learnable 2×2 matrices.
    Mode B (physics):   Initialize from poles (α, ω, τ) as
                        τ * [[-α, -ω],
                             [ ω, -α]]  + small residual (learnable).
    """
    def __init__(self, k: int, physics_tied: bool = True, residual_scale: float = 0.05):
        super().__init__()
        self.k = k
        self.physics_tied = physics_tied
        self.lap_ref = None

        if physics_tied:
            # Buffers hold the analytic part; residual is learnable.
            self.register_buffer("alpha0", torch.ones(k))
            self.register_buffer("omega0", torch.zeros(k))
            self.register_buffer("tau0",   torch.ones(1))
            self.residual = nn.Parameter(torch.zeros(k, 2, 2))
            nn.init.normal_(self.residual, std=residual_scale)
        else:
            self.blocks = nn.Parameter(torch.zeros(k, 2, 2))
            nn.init.normal_(self.blocks, std=residual_scale)

    def bind_from_basis(self, lap: "LearnableLaplacianBasis"):
        if not self.physics_tied:
            return self
        self.lap_ref = lap
        with torch.no_grad():
            self.alpha0.copy_(lap.s_real.clamp_min(lap.alpha_min))
            self.omega0.copy_(lap.s_imag)
            self.tau0.copy_(F.softplus(lap._tau) + 1e-3)

        return self

    def forward(self, L: torch.Tensor) -> torch.Tensor:  # [B,T,2K]
        B, T, twoK = L.shape
        K = twoK // 2
        L = L.view(B, T, K, 2)

        if self.physics_tied:
            # τ * [[-α, -ω],[ω, -α]] + residual (live-tied to basis if available)
            if hasattr(self, "lap_ref") and (self.lap_ref is not None):
                alpha0 = self.lap_ref.s_real.clamp_min(self.lap_ref.alpha_min)  # [K] or scalar
                omega0 = self.lap_ref.s_imag  # [K] or scalar
                tau0 = F.softplus(self.lap_ref._tau) + 1e-3  # [K] or scalar

                # Build A from α, ω, then scale by τ and add residual
                A = torch.zeros(K, 2, 2, device=L.device, dtype=L.dtype)
                A[:, 0, 0] = -alpha0
                A[:, 0, 1] = -omega0
                A[:, 1, 0] = omega0
                A[:, 1, 1] = -alpha0
                A = A * tau0.view(-1, 1, 1) + self.residual  # [K,2,2]
            else:
            # Fallback: build A from tied buffers (alpha0, omega0, tau0)
                alpha0 = self.alpha0.to(L.device, L.dtype)
                omega0 = self.omega0.to(L.device, L.dtype)
                tau0 = self.tau0.to(L.device, L.dtype)
                A = torch.zeros(K, 2, 2, device=L.device, dtype=L.dtype)
                A[:, 0, 0] = -alpha0
                A[:, 0, 1] = -omega0
                A[:, 1, 0] = omega0
                A[:, 1, 1] = -alpha0
                A = A * tau0.view(-1, 1, 1) + self.residual  # [K,2,2]
        else:
            # Untied (free) case: use learnable per-pole 2×2 blocks
            A = self.blocks.to(device=L.device, dtype=L.dtype)

        out = torch.einsum('kij,btkj->btki', A, L)  # [B,T,K,2]
        return out.reshape(B, T, 2 * K)


# ---------- 2nd-order PDE + Laplace combiner ----------
class SecondOrderLaplaceCombinerPolewise(nn.Module):
    """
    Implements the 2nd-order (damped) premise in Laplace space:
        H(s) ~ G(s) * [ (a0 + a1*s) T(s) + (b0 + b1*s) V(s) ]
    where G(s) (Green's function) is represented by the learnable Laplace basis.
    We never discretize the ODE—everything is learned stably via poles.

    Inputs:  T_sig, V_sig ∈ [B,T,N]
    Output:  L ∈ [B,T,2K]  (guidance features)
    """
    def __init__(self, num_entities: int, k: int, physics_tied: bool = True, residual_scale: float = 0.05):
        super().__init__()
        self.lap = LearnableLaplacianBasis(k=k, feat_dim=num_entities)  # [B,T,N] -> [B,T,2K]
        self.diff = PolewiseDiff(k, physics_tied=physics_tied, residual_scale=residual_scale)

        # Optionally bind derivative blocks to poles from the basis (no hard dep).
        if physics_tied:
            _ = self.diff.bind_from_basis(self.lap)

        # Tiny conditioner for time-varying numerator [a0, a1, b0, b1]
        self.num_head = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, 8), nn.GELU(),
            nn.Linear(8, 4)
        )

    @staticmethod
    def _forward_diff(x: torch.Tensor) -> torch.Tensor:
        # Only used to make the coefficient conditioner aware of trend changes; not for the derivative path.
        return torch.diff(x, dim=1, prepend=x[:, :1])

    def forward(self, T_sig: torch.Tensor, V_sig: torch.Tensor, dt: torch.Tensor | None = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Stats for a tiny time-varying numerator conditioner (shared across entities)
        stats = torch.stack([
            T_sig.mean(-1),                           # [B,T]
            V_sig.mean(-1),                           # [B,T]
            self._forward_diff(T_sig).mean(-1),       # [B,T]
            self._forward_diff(V_sig).mean(-1),       # [B,T]
        ], dim=-1)                                    # [B,T,4]
        coeff = self.num_head(stats)                  # [B,T,4]
        a0, a1, b0, b1 = torch.unbind(coeff, dim=-1)  # each [B,T]

        # Laplace features + pole-wise time-derivatives (no raw differencing)
        LT  = self.lap(T_sig, dt=dt)                         # [B,T,2K]
        LV  = self.lap(V_sig, dt=dt)                         # [B,T,2K]
        LdT = self.diff(LT)                           # [B,T,2K]
        LdV = self.diff(LV)                           # [B,T,2K]

        # Combine per numerator (broadcast over 2K)
        L = (a0.unsqueeze(-1) * LT
           + a1.unsqueeze(-1) * LdT
           + b0.unsqueeze(-1) * LV
           + b1.unsqueeze(-1) * LdV)                  # [B,T,2K]

        aux = {"LT": LT, "LV": LV, "LdT": LdT, "LdV": LdV, "coeff": coeff}
        return L, aux


class ODELaplaceGuidedSummarizer(nn.Module):
    """
    Global summary via cross-attention guided by ODE+Laplace features.

    Inputs:
      x:        [B, T, N, F]
      pad_mask: optional [B, T]  (True/1 = ignore these padded time steps)
    Outputs:
      summary: [B, Lq, H]
      aux:     dict of intermediates for diagnostics
    """
    def __init__(self,
                 num_entities: int,
                 feat_dim: int,
                 hidden_dim: int,
                 out_len: int,
                 num_heads: int = 4,
                 lap_k: int = 8,
                 dropout: float = 0.0,
                 add_guidance_tokens: bool = True,
                 physics_tied_derivative: bool = True,
                 residual_scale: float = 0.05):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.N, self.F, self.H = num_entities, feat_dim, hidden_dim
        self.num_heads = num_heads
        self.add_guidance_tokens = add_guidance_tokens

        # Context projection: [B,T,NF] -> [B,T,H]
        self.ctx_proj = nn.Linear(num_entities * feat_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        # Tiny scalar heads for V (level) and T (from feature difference)
        self.v_head = TVHead(feat_dim)
        self.t_head = TVHead(feat_dim)

        # PDE + Laplace combiner (pole-wise derivative)
        self.pde_lap = SecondOrderLaplaceCombinerPolewise(
            num_entities=num_entities, k=lap_k,
            physics_tied=physics_tied_derivative, residual_scale=residual_scale
        )

        # FiLM + per-head bias
        self.lap_to_film = nn.Linear(2 * lap_k, 2 * hidden_dim)  # -> gamma, beta
        self.lap_to_bias = nn.Linear(2 * lap_k, num_heads)       # -> per-head bias over time

        # Optional: project guidance to tokens and concatenate to memory
        if add_guidance_tokens:
            self.lap_token_proj = nn.Linear(2 * lap_k, hidden_dim)

        # Learned queries
        self.queries = nn.Parameter(torch.randn(out_len, hidden_dim) / math.sqrt(hidden_dim))
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                         dropout=dropout, batch_first=True)

    @staticmethod
    def _time_diff_feats(x: torch.Tensor) -> torch.Tensor:
        # Forward diff on feature tensor for T-head input (leading copy)
        return torch.diff(x, dim=1, prepend=x[:, :1])

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None, dt: torch.Tensor | None = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x:        [B,T,N,F]
        pad_mask: [B,T] (True/1 = ignore padded steps) or None
        """
        B, T, N, F = x.shape
        device = x.device

        # 1) Context memory (flatten N,F -> project -> add time pos)
        ctx = x.reshape(B, T, N * F)                        # [B,T,NF]
        ctx_h = self.ctx_proj(ctx)                          # [B,T,H]
        pos = _pos_emb(T, self.H, device=device).unsqueeze(0)  # [1,T,H]
        K = Vv = ctx_h + pos                                # [B,T,H]

        # 2) Per-entity scalars
        V_sig = self.v_head(x)                              # [B,T,N]   (level)
        T_sig = self.t_head(self._time_diff_feats(x))       # [B,T,N]   (from feature diff)

        # 3) PDE + Laplace guidance in [B,T,2K]
        L, lap_aux = self.pde_lap(T_sig, V_sig, dt=dt)             # [B,T,2K]

        # 4) FiLM on keys/values + per-head time bias
        film = self.lap_to_film(L)                          # [B,T,2H]
        gamma, beta = torch.chunk(film, 2, dim=-1)          # [B,T,H] each
        K = (1.0 + gamma) * K + beta
        Vv = (1.0 + gamma) * Vv + beta

        # Per-head additive time bias for attention logits.
        # attn_mask expects float; shape can be (B*heads, Lq, S) for per-batch biases.
        bias_ht = self.lap_to_bias(L)                       # [B,T,heads]
        Lq = self.queries.shape[0]
        attn_bias = bias_ht.permute(0, 2, 1).unsqueeze(2).expand(B, self.num_heads, Lq, T)
        attn_bias = attn_bias.reshape(B * self.num_heads, Lq, T).to(K.dtype)  # [(B*heads), Lq, T]

        # Optional: append guidance tokens
        if self.add_guidance_tokens:
            lap_tokens = self.lap_token_proj(L)             # [B,T,H]
            memory = torch.cat([K, lap_tokens], dim=1)      # [B, 2T, H]
            values = torch.cat([Vv, lap_tokens], dim=1)     # [B, 2T, H]
            zeros_bias = torch.zeros(B * self.num_heads, Lq, T, device=device, dtype=attn_bias.dtype)
            attn_bias = torch.cat([attn_bias, zeros_bias], dim=-1)  # [(B*heads), Lq, 2T]
            key_padding_mask = None if pad_mask is None else torch.cat(
                [pad_mask.to(torch.bool), torch.zeros(B, T, device=device, dtype=torch.bool)], dim=1
            )
        else:
            memory, values = K, Vv
            key_padding_mask = None if pad_mask is None else pad_mask.to(torch.bool)

        # 5) Cross-attention from learned queries
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)     # [B,Lq,H]
        summary, _ = self.mha(Q, memory, values,
                              key_padding_mask=key_padding_mask,
                              attn_mask=attn_bias)          # [B,Lq,H]
        summary = self.norm(self.dropout(summary) + Q)      # residual on queries

        aux = {
            "T": T_sig, "V": V_sig,
            "lap_guidance": L,
            **lap_aux
        }
        return summary, aux


# ---------------------------
# Minimal usage (example)
# ---------------------------
# N, F, T = 20, 6, 64
# H, Lq, K, heads = 256, 16, 8, 4
# model = ODELaplaceGuidedSummarizer(
#     num_entities=N, feat_dim=F, hidden_dim=H, out_len=Lq,
#     num_heads=heads, lap_k=K, dropout=0.1,
#     add_guidance_tokens=True, physics_tied_derivative=True
# )
# # (Optional) tie derivative blocks to basis poles currently in laptrans:
# model.tie_derivative_to_basis()
#
# x = torch.randn(8, T, N, F)
# pad_mask = torch.zeros(8, T, dtype=torch.bool)  # or None
# summary, aux = model(x, pad_mask)
# print(summary.shape)  # -> [8, Lq, H]