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
class TinyScalarHead(nn.Module):
    """Per-entity scalar readout from features at each time."""
    def __init__(self, feat_dim: int, hidden: int = 32):
        super().__init__()
        h = min(hidden, max(8, feat_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, h),
            nn.GELU(),
            nn.Linear(h, 1),
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

        if physics_tied:
            # Buffers hold the analytic part; residual is learnable.
            self.register_buffer("alpha0", torch.ones(k))
            self.register_buffer("omega0", torch.zeros(k))
            self.register_buffer("tau0",   torch.ones(1))
            self.residual = nn.Parameter(torch.zeros(k, 2, 2))
            nn.init.normal_(self.residual, std=residual_scale)
        else:
            self.blocks = nn.Parameter(torch.zeros(k, 2, 2))
            nn.init.normal_(self.blocks, std=0.02)

    @torch.no_grad()
    def set_poles(self, alpha: torch.Tensor, omega: torch.Tensor, tau: torch.Tensor | float = 1.0):
        """
        Manually set poles for physics-tied mode. Shapes:
          alpha: [K], omega: [K], tau: scalar or [K] (will be broadcast).
        """
        assert self.physics_tied, "set_poles is only for physics-tied mode."
        self.alpha0.copy_(alpha.reshape(-1))
        self.omega0.copy_(omega.reshape(-1))
        self.tau0.copy_(torch.as_tensor(tau).reshape(1))

    @torch.no_grad()
    def bind_from_basis(self, basis) -> bool:
        """
        Try to read poles from LearnableLaplacianBasis.
        Expects either .get_poles() -> (alpha[K], omega[K], tau[1 or K])
        or attributes with reasonable names. Returns True if success.
        """
        if not self.physics_tied:
            return False

        if hasattr(basis, "get_poles"):
            alpha, omega, tau = basis.get_poles()
            self.set_poles(alpha, omega, tau)
            return True

        # Heuristic attribute names (edit if needed to match your laptrans.py)
        got = False
        for a_name in ("alpha", "alphas", "a", "real"):
            if hasattr(basis, a_name):
                self.alpha0.copy_(getattr(basis, a_name).detach().abs().reshape(-1))
                got = True
                break
        for w_name in ("omega", "omegas", "w", "imag"):
            if hasattr(basis, w_name):
                self.omega0.copy_(getattr(basis, w_name).detach().reshape(-1))
        for t_name in ("tau", "time_scale", "scale", "dt"):
            if hasattr(basis, t_name):
                self.tau0.copy_(torch.as_tensor(getattr(basis, t_name)).reshape(1))
        return got

    def _physics_blocks(self) -> torch.Tensor:
        # Build τ * [[-α, -ω], [ω, -α]]  per pole
        a = self.alpha0 * self.tau0
        w = self.omega0 * self.tau0
        M = torch.zeros(self.k, 2, 2, device=a.device, dtype=a.dtype)
        M[:, 0, 0] = -a; M[:, 0, 1] = -w
        M[:, 1, 0] =  w; M[:, 1, 1] = -a
        return M

    def forward(self, lap_feats: torch.Tensor) -> torch.Tensor:
        """
        lap_feats: [B,T,2K]  (pairs [cos, sin] per pole)
        returns:   [B,T,2K]  (approx. derivative features)
        """
        B, T, C = lap_feats.shape
        K = C // 2
        x = lap_feats.view(B, T, K, 2)   # [B,T,K,2]
        if self.physics_tied:
            M = self._physics_blocks() + self.residual  # [K,2,2]
        else:
            M = self.blocks                              # [K,2,2]
        y = torch.einsum('btkc,kcd->btkd', x, M)         # [B,T,K,2]
        return y.view(B, T, 2*K)


# ---------- Second-order numerator combiner (pole-wise) ----------
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

    def forward(self, T_sig: torch.Tensor, V_sig: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
        LT  = self.lap(T_sig)                         # [B,T,2K]
        LV  = self.lap(V_sig)                         # [B,T,2K]
        LdT = self.diff(LT)                           # [B,T,2K]
        LdV = self.diff(LV)                           # [B,T,2K]

        # Combine per numerator (broadcast over 2K)
        L = (a0.unsqueeze(-1) * LT) + (a1.unsqueeze(-1) * LdT) + \
            (b0.unsqueeze(-1) * LV) + (b1.unsqueeze(-1) * LdV)   # [B,T,2K]

        aux = {"coeff": coeff, "LT": LT, "LdT": LdT, "LV": LV, "LdV": LdV}
        return L, aux


# ---------- Full global summarizer with FiLM + (optional) tokens ----------
class PDELaplaceGuidedSummarizer(nn.Module):
    """
    Global summary via cross-attention guided by PDE+Laplace features.

    Inputs:
      x:    [B, T, N, F]
      mask: optional [B, T]  (True/1 = ignore these time steps)
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
        self.v_head = TinyScalarHead(feat_dim)
        self.t_head = TinyScalarHead(feat_dim)

        # PDE + Laplace combiner (pole-wise derivative)
        self.pde_lap = SecondOrderLaplaceCombinerPolewise(
            num_entities=num_entities, k=lap_k,
            physics_tied=physics_tied_derivative, residual_scale=residual_scale
        )

        # Map guidance L ∈ [B,T,2K] to FiLM params and per-head time bias
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
        # Forward diff on feature tensor for T-head input (leading zero frame)
        dx = x[:, 1:] - x[:, :-1]                           # [B,T-1,N,F]
        zero = torch.zeros_like(x[:, :1])                   # [B,1,N,F]
        return torch.cat([zero, dx], dim=1)                 # [B,T,N,F]

    @torch.no_grad()
    def tie_derivative_to_basis(self) -> bool:
        """
        (Optional) Bind the derivative blocks to the current Laplace poles
        from the internal LearnableLaplacianBasis. Returns True if successful.
        """
        return self.pde_lap.diff.bind_from_basis(self.pde_lap.lap)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x:    [B,T,N,F]
        mask: [B,T] (True/1 = ignore) or None
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
        L, lap_aux = self.pde_lap(T_sig, V_sig)             # [B,T,2K]

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
            key_padding_mask = None if mask is None else torch.cat(
                [mask.to(torch.bool), torch.zeros(B, T, device=device, dtype=torch.bool)], dim=1
            )
        else:
            memory, values = K, Vv
            key_padding_mask = None if mask is None else mask.to(torch.bool)

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
# model = PDELaplaceGuidedSummarizer(
#     num_entities=N, feat_dim=F, hidden_dim=H, out_len=Lq,
#     num_heads=heads, lap_k=K, dropout=0.1,
#     add_guidance_tokens=True, physics_tied_derivative=True
# )
# # (Optional) tie derivative blocks to basis poles currently in laptrans:
# model.tie_derivative_to_basis()
#
# x = torch.randn(8, T, N, F)
# mask = torch.zeros(8, T, dtype=torch.bool)  # or None
# summary, aux = model(x, mask)
# print(summary.shape)  # -> [8, Lq, H]