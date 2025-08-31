import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Tuple
import torch.nn.functional as F
from Model.pos_time_emb import get_sinusoidal_pos_emb as _pos_emb
from Model.laptrans import LearnableLaplacianBasis

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

class ODELaplaceGuidedSummarizer(nn.Module):
    """
    Simplified global summarizer using Laplace features over time.
      • Build per-entity scalars V(t) and T(t) via tiny heads
      • Apply simple Laplace basis: L_v = lap(V), L_t = lap(T)
      • Learnable scalars a,b mix streams: L = a*L_v + b*L_t
      • Project Laplace tokens and concatenate to context memory
      • Cross-attend learned queries to produce [B,Lq,H]
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
                 **_):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.N, self.F, self.H = num_entities, feat_dim, hidden_dim
        self.num_heads = num_heads
        self.add_guidance_tokens = add_guidance_tokens

        # Context projection: [B,T,NF] → [B,T,H]
        self.ctx_proj = nn.Linear(num_entities * feat_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        # Per-entity scalar heads and Laplace basis over time
        self.v_head = TVHead(feat_dim)
        self.t_head = TVHead(feat_dim)
        self.lap = LearnableLaplacianBasis(k=lap_k, feat_dim=num_entities)

        # Learnable mix scalars a, b (positive via softplus) init≈1.0
        init_raw = math.log(math.e - 1.0)  # ~0.5413 → softplus=1.0
        self.w_v_raw = nn.Parameter(torch.tensor(init_raw))
        self.w_t_raw = nn.Parameter(torch.tensor(init_raw))

        if add_guidance_tokens:
            self.lap_token_proj = nn.Linear(2 * lap_k, hidden_dim)

        # Learned queries + MHA
        self.queries = nn.Parameter(torch.randn(out_len, hidden_dim) / math.sqrt(hidden_dim))
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                         dropout=dropout, batch_first=True)

    def forward(self,
                x: torch.Tensor,                    # [B,T,N,F]
                pad_mask: Optional[torch.Tensor] = None,  # [B,T] True=ignore
                dt: torch.Tensor = None,            # accepted but ignored (compat)
                ctx_diff: torch.Tensor = None,      # [B,T,N,F]
                entity_mask: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, N, F = x.shape
        device = x.device

        # mask entities if provided
        mN = None
        if entity_mask is not None:
            mN = entity_mask.to(device=x.device, dtype=x.dtype)
            if mN.dim() == 2:  # [B,N] -> [B,1,N] -> [B,T,N]
                mN = mN.unsqueeze(1).expand(B, T, N)
            x = x * mN.unsqueeze(-1)

        # context memory from raw features
        ctx = x.reshape(B, T, N * F)
        K = self.ctx_proj(ctx) + _pos_emb(T, self.H, device=device)

        # per-entity scalars for V and T
        V_sig = self.v_head(x)                               # [B,T,N]
        if ctx_diff is None:
            x_diff = torch.zeros_like(x)
            x_diff[:, 1:] = x[:, 1:] - x[:, :-1]
        else:
            x_diff = ctx_diff.to(device)
        if mN is not None:
            x_diff = x_diff * mN.unsqueeze(-1)
        T_sig = self.t_head(x_diff)                          # [B,T,N]

        # Laplace over time on both streams & learnable mix
        L_v = self.lap(V_sig)                                # [B,T,2K]
        L_t = self.lap(T_sig)                                # [B,T,2K]
        a = F.softplus(self.w_v_raw)                         # ≥0
        b = F.softplus(self.w_t_raw)                         # ≥0
        L = a * L_v + b * L_t                                # [B,T,2K]

        # build memory (concat lap tokens)
        if self.add_guidance_tokens:
            lap_tokens = self.lap_token_proj(L)              # [B,T,H]
            memory = torch.cat([K, lap_tokens], dim=1)       # [B,2T,H]
            values = memory
            key_padding_mask = (None if pad_mask is None else
                                torch.cat([pad_mask.to(torch.bool),
                                           torch.zeros(B, T, device=device, dtype=torch.bool)], dim=1))
        else:
            memory = values = K
            key_padding_mask = None if pad_mask is None else pad_mask.to(torch.bool)

        # cross-attend learned queries to memory
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)
        summary, _ = self.mha(Q, memory, values, key_padding_mask=key_padding_mask)
        summary = self.norm(self.dropout(summary) + Q)

        aux = {"V_sig": V_sig, "T_sig": T_sig, "lap_V": L_v, "lap_T": L_t, "w_v": a.detach(), "w_t": b.detach()}
        return summary, aux

# # ---------- Pole-wise derivative in Laplace-feature space ----------
# class PolewiseDiff(nn.Module):
#     """
#     Given Laplace features L(x) ∈ [B,T,2K] as K blocks [cos, sin],
#     approximate L(dx/dt) via K independent 2×2 blocks.

#     Mode A (free):      K learnable 2×2 matrices.
#     Mode B (physics):   Initialize from poles (α, ω, τ) as
#                         τ * [[-α, -ω],
#                              [ ω, -α]]  + small residual (learnable).
#     """
#     def __init__(self, k: int, physics_tied: bool = True, residual_scale: float = 0.05):
#         super().__init__()
#         self.k = k
#         self.physics_tied = physics_tied
#         self.lap_ref = None

#         if physics_tied:
#             # Buffers hold the analytic part; residual is learnable.
#             self.register_buffer("alpha0", torch.ones(k))
#             self.register_buffer("omega0", torch.zeros(k))
#             self.register_buffer("tau0",   torch.ones(1))
#             self.residual = nn.Parameter(torch.zeros(k, 2, 2))
#             nn.init.normal_(self.residual, std=residual_scale)
#         else:
#             self.blocks = nn.Parameter(torch.zeros(k, 2, 2))
#             nn.init.normal_(self.blocks, std=residual_scale)

#     def bind_from_basis(self, lap: "LearnableLaplacianBasis"):
#         if not self.physics_tied:
#             return self
#         self.lap_ref = lap
#         with torch.no_grad():
#             self.alpha0.copy_(lap.s_real.clamp_min(lap.alpha_min))
#             self.omega0.copy_(lap.s_imag)
#             self.tau0.copy_(F.softplus(lap._tau) + 1e-3)

#         return self

#     def forward(self, L: torch.Tensor) -> torch.Tensor:  # [B,T,2K]
#         B, T, twoK = L.shape
#         K = twoK // 2
#         L = L.view(B, T, K, 2)

#         if self.physics_tied:
#             # τ * [[-α, -ω],[ω, -α]] + residual (live-tied to basis if available)
#             if hasattr(self, "lap_ref") and (self.lap_ref is not None):
#                 alpha0 = self.lap_ref.s_real.clamp_min(self.lap_ref.alpha_min)  # [K] or scalar
#                 omega0 = self.lap_ref.s_imag  # [K] or scalar
#                 tau0 = F.softplus(self.lap_ref._tau) + 1e-3  # [K] or scalar

#                 # Build A from α, ω, then scale by τ and add residual
#                 A = torch.zeros(K, 2, 2, device=L.device, dtype=L.dtype)
#                 A[:, 0, 0] = -alpha0
#                 A[:, 0, 1] = -omega0
#                 A[:, 1, 0] = omega0
#                 A[:, 1, 1] = -alpha0
#                 A = A * tau0.view(-1, 1, 1) + self.residual  # [K,2,2]
#             else:
#             # Fallback: build A from tied buffers (alpha0, omega0, tau0)
#                 alpha0 = self.alpha0.to(L.device, L.dtype)
#                 omega0 = self.omega0.to(L.device, L.dtype)
#                 tau0 = self.tau0.to(L.device, L.dtype)
#                 A = torch.zeros(K, 2, 2, device=L.device, dtype=L.dtype)
#                 A[:, 0, 0] = -alpha0
#                 A[:, 0, 1] = -omega0
#                 A[:, 1, 0] = omega0
#                 A[:, 1, 1] = -alpha0
#                 A = A * tau0.view(-1, 1, 1) + self.residual  # [K,2,2]
#         else:
#             # Untied (free) case: use learnable per-pole 2×2 blocks
#             A = self.blocks

#         out = torch.einsum('kij,btkj->btki', A, L)  # [B,T,K,2]
#         return out.reshape(B, T, 2 * K)


# # ---------- 2nd-order ODE + Laplace combiner ----------
# class SecondOrderLaplaceCombinerPolewise(nn.Module):
#     """
#     Implements the 2nd-order (damped) premise in Laplace space:
#         H(s) ~ G(s) * [ (a0 + a1*s) T(s) + (b0 + b1*s) V(s) ]
#     where G(s) (Green's function) is represented by the learnable Laplace basis.
#     We never discretize the ODE—everything is learned stably via poles.

#     Inputs:  T_sig, V_sig ∈ [B,T,N]
#     Output:  L ∈ [B,T,2K]  (guidance features)
#     """
#     def __init__(self, num_entities: int, k: int,
#                  physics_tied: bool = True, residual_scale: float = 0.05,
#                  renorm_by_fill: bool = True):  # <- NEW arg (default on)
#         super().__init__()
#         self.lap = LearnableLaplacianBasis(k=k, feat_dim=num_entities)
#         self.diff = PolewiseDiff(k, physics_tied=physics_tied, residual_scale=residual_scale)
#         self.num_entities = num_entities  # <- for renorm
#         self.renorm_by_fill = renorm_by_fill

#         # Optionally bind derivative blocks to poles from the basis (no hard dep).
#         if physics_tied:
#             _ = self.diff.bind_from_basis(self.lap)

#         # Tiny conditioner for time-varying numerator [a0, a1, b0, b1]
#         self.num_head = nn.Sequential(
#             nn.LayerNorm(4),
#             nn.Linear(4, 8), nn.GELU(),
#             nn.Linear(8, 4)
#         )

#     @staticmethod
#     def _masked_mean(x: torch.Tensor, m: torch.Tensor, dim: int = -1, eps: float = 1e-6):
#         """Mean over `dim` with mask m∈{0,1} broadcastable to x."""
#         num = (x * m).sum(dim=dim)
#         den = m.sum(dim=dim).clamp_min(eps)
#         return num / den

#     @staticmethod
#     def _forward_diff(x: torch.Tensor) -> torch.Tensor:
#         # Only used to make the coefficient conditioner aware of trend changes; not for the derivative path.
#         return torch.diff(x, dim=1, prepend=x[:, :1])

#     def forward(self,
#                 T_sig: torch.Tensor,  # [B,T,N]
#                 V_sig: torch.Tensor,  # [B,T,N]
#                 dt: torch.Tensor = None,
#                 entity_mask: torch.Tensor = None  # <- NEW (optional)
#                 ) -> Tuple[torch.Tensor, Dict]:
#         B, T, N = T_sig.shape
#         device = T_sig.device
#         dtype = T_sig.dtype

#         # Build [B,T,N] mask if provided
#         mN = None
#         if entity_mask is not None:
#             mN = entity_mask.to(device=device, dtype=dtype)  # [B,N] or [B,1,N] or [B,T,N]
#             if mN.dim() == 2:  # [B,N] -> [B,1,N]
#                 mN = mN.unsqueeze(1)
#             if mN.dim() == 3 and mN.shape[1] == 1:  # [B,1,N] -> [B,T,N]
#                 mN = mN.expand(B, T, N)
#             elif mN.dim() == 3 and mN.shape[1] == T:
#                 pass  # already [B,T,N]
#             else:
#                 raise ValueError("entity_mask must be [B,N], [B,1,N], or [B,T,N]")

#             # Zero out padded entities conservatively
#             T_sig = T_sig * mN
#             V_sig = V_sig * mN

#         # ---- Tiny numerator conditioner (masked stats over entity axis) ----
#         dT = self._forward_diff(T_sig)  # [B,T,N]
#         dV = self._forward_diff(V_sig)  # [B,T,N]

#         if mN is None:
#             s_T = T_sig.mean(dim=-1)  # [B,T]
#             s_V = V_sig.mean(dim=-1)
#             s_dT = dT.mean(dim=-1)
#             s_dV = dV.mean(dim=-1)
#         else:
#             s_T = self._masked_mean(T_sig, mN, dim=-1)
#             s_V = self._masked_mean(V_sig, mN, dim=-1)
#             s_dT = self._masked_mean(dT, mN, dim=-1)
#             s_dV = self._masked_mean(dV, mN, dim=-1)

#         stats = torch.stack([s_T, s_V, s_dT, s_dV], dim=-1)  # [B,T,4]
#         coeff = self.num_head(stats)  # [B,T,4]
#         a0, a1, b0, b1 = torch.unbind(coeff, dim=-1)  # each [B,T]

#         # ---- Optional renorm by fill ratio (stabilize amplitude across fill) ----
#         if (mN is not None) and self.renorm_by_fill and (self.num_entities > 0):
#             real = mN.sum(dim=-1, keepdim=True).clamp_min(1.0)  # [B,T,1] count of real entities
#             scale = (float(self.num_entities) / real).to(dtype=dtype)  # [B,T,1]
#             T_in = T_sig * scale
#             V_in = V_sig * scale
#         else:
#             T_in, V_in = T_sig, V_sig

#         # ---- Laplace features + pole-wise derivative ----
#         LT = self.lap(T_in, dt=dt)  # [B,T,2K]
#         LV = self.lap(V_in, dt=dt)  # [B,T,2K]
#         LdT = self.diff(LT)  # [B,T,2K]
#         LdV = self.diff(LV)  # [B,T,2K]

#         # Combine per numerator (broadcast over 2K)
#         L = (a0.unsqueeze(-1) * LT
#              + a1.unsqueeze(-1) * LdT
#              + b0.unsqueeze(-1) * LV
#              + b1.unsqueeze(-1) * LdV)  # [B,T,2K]

#         aux = {"LT": LT, "LV": LV, "LdT": LdT, "LdV": LdV, "coeff": coeff}
#         return L, aux


# class ODELaplaceGuidedSummarizer(nn.Module):
#     """
#     Global summary via cross-attention guided by ODE+Laplace features.

#     Inputs:
#       x:        [B, T, N, F]
#       pad_mask: optional [B, T]  (True/1 = ignore these padded time steps)
#     Outputs:
#       summary: [B, Lq, H]
#       aux:     dict of intermediates for diagnostics
#     """
#     def __init__(self,
#                  num_entities: int,
#                  feat_dim: int,
#                  hidden_dim: int,
#                  out_len: int,
#                  num_heads: int = 4,
#                  lap_k: int = 8,
#                  dropout: float = 0.0,
#                  add_guidance_tokens: bool = True,
#                  physics_tied_derivative: bool = True,
#                  residual_scale: float = 0.05):
#         super().__init__()
#         assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
#         self.N, self.F, self.H = num_entities, feat_dim, hidden_dim
#         self.num_heads = num_heads
#         self.add_guidance_tokens = add_guidance_tokens

#         # Context projection: [B,T,NF] -> [B,T,H]
#         self.ctx_proj = nn.Linear(num_entities * feat_dim, hidden_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.norm = nn.LayerNorm(hidden_dim)

#         # Tiny scalar heads for V (level) and T (from feature difference)
#         self.v_head = TVHead(feat_dim)
#         self.t_head = TVHead(feat_dim)

#         # PDE + Laplace combiner (pole-wise derivative)
#         self.ode_lap = SecondOrderLaplaceCombinerPolewise(
#             num_entities=num_entities, k=lap_k,
#             physics_tied=physics_tied_derivative, residual_scale=residual_scale
#         )

#         # FiLM + per-head bias
#         self.lap_to_film = nn.Linear(2 * lap_k, 2 * hidden_dim)  # -> gamma, beta
#         self.lap_to_bias = nn.Linear(2 * lap_k, num_heads)       # -> per-head bias over time

#         # Optional: project guidance to tokens and concatenate to memory
#         if add_guidance_tokens:
#             self.lap_token_proj = nn.Linear(2 * lap_k, hidden_dim)

#         # Learned queries
#         self.queries = nn.Parameter(torch.randn(out_len, hidden_dim) / math.sqrt(hidden_dim))
#         self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
#                                          dropout=dropout, batch_first=True)

#     @staticmethod
#     def _time_diff_feats(x: torch.Tensor) -> torch.Tensor:
#         # Forward diff on feature tensor for T-head input (leading copy)
#         return torch.diff(x, dim=1, prepend=x[:, :1])

#     def forward(
#             self,
#             x: torch.Tensor,  # [B,T,N,F]
#             pad_mask: Optional[torch.Tensor] = None,  # [B,T] (True=ignore)
#             dt: torch.Tensor = None,
#             ctx_diff: torch.Tensor = None,
#             entity_mask: Optional[torch.Tensor] = None,  # [B,N] or [B,1,N] or [B,T,N]
#     ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

#         """
#         x:        [B,T,N,F]
#         pad_mask: [B,T] (True/1 = ignore padded steps) or None
#         """
#         B, T, N, F = x.shape
#         device = x.device

#         mN = None
#         if entity_mask is not None:
#             mN = entity_mask.to(device=x.device, dtype=x.dtype)
#             if mN.dim() == 2:  # [B,N] -> [B,1,N]
#                 mN = mN.unsqueeze(1)
#             if mN.size(1) == 1:  # [B,1,N] -> [B,T,N]
#                 mN = mN.expand(B, T, N)
#             x = x * mN.unsqueeze(-1)  # zero padded entities in features

#         ctx = x.reshape(B, T, N * F)  # [B,T,NF]
#         ctx_h = self.ctx_proj(ctx)  # [B,T,H]
#         pos = _pos_emb(T, self.H, device=device)  # [1,T,H]
#         K = Vv = ctx_h + pos  # [B,T,H]

#         # ----------------------------------------
#         # 2) Per-entity scalars (masked where pad)
#         # ----------------------------------------
#         V_sig = self.v_head(x)  # [B,T,N]
#         Tfeat = ctx_diff if ctx_diff is not None else self._time_diff_feats(x)
#         # If ctx_diff was precomputed upstream (unmasked), mask it here:
#         if mN is not None and ctx_diff is not None:
#             if Tfeat.dim() == 4:  # [B,T,N,*]
#                 Tfeat = Tfeat * mN.unsqueeze(-1)
#             else:  # [B,T,N]
#                 Tfeat = Tfeat * mN
#         T_sig = self.t_head(Tfeat)  # [B,T,N]

#         if mN is not None:
#             V_sig = V_sig * mN
#             T_sig = T_sig * mN

#         # 3) PDE + Laplace guidance in [B,T,2K]
#         try:
#             L, lap_aux = self.ode_lap(T_sig, V_sig, dt=dt, entity_mask=mN)
#         except TypeError as e:
#             if "unexpected keyword argument 'entity_mask'" in str(e):
#                 L, lap_aux = self.ode_lap(T_sig, V_sig, dt=dt)
#             else:
#                 raise

#         # ----------------------------------------
#         # 4) FiLM + per-head time bias
#         # ----------------------------------------
#         film = self.lap_to_film(L)  # [B,T,2H]
#         gamma, beta = torch.chunk(film, 2, dim=-1)  # [B,T,H] each
#         K = (1.0 + gamma) * K + beta
#         Vv = (1.0 + gamma) * Vv + beta

#         # Per-head additive time bias for attention logits
#         bias_ht = self.lap_to_bias(L)  # [B,T,heads]
#         Lq = self.queries.shape[0]
#         attn_bias = bias_ht.permute(0, 2, 1).unsqueeze(2).expand(B, self.num_heads, Lq, T)
#         attn_bias = attn_bias.reshape(B * self.num_heads, Lq, T).to(K.dtype)  # [(B*heads), Lq, T]

#         # ----------------------------------------
#         # 5) Optional guidance tokens branch
#         # ----------------------------------------
#         if getattr(self, "add_guidance_tokens", False):
#             lap_tokens = self.lap_token_proj(L)  # [B,T,H]
#             memory = torch.cat([K, lap_tokens], dim=1)  # [B, 2T, H]
#             values = torch.cat([Vv, lap_tokens], dim=1)  # [B, 2T, H]

#             zeros_bias = torch.zeros(B * self.num_heads, Lq, T, device=device, dtype=attn_bias.dtype)
#             attn_bias = torch.cat([attn_bias, zeros_bias], dim=-1)  # [(B*heads), Lq, 2T]

#             key_padding_mask = (None if pad_mask is None else
#                                 torch.cat([pad_mask.to(torch.bool),
#                                            torch.zeros(B, T, device=device, dtype=torch.bool)], dim=1))
#         else:
#             memory, values = K, Vv
#             key_padding_mask = None if pad_mask is None else pad_mask.to(torch.bool)

#         # ----------------------------------------
#         # 6) Cross-attention from learned queries
#         # ----------------------------------------
#         Q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B,Lq,H]
#         summary, _ = self.mha(Q, memory, values,
#                               key_padding_mask=key_padding_mask,
#                               attn_mask=attn_bias)  # [B,Lq,H]
#         summary = self.norm(self.dropout(summary) + Q)  # residual on queries

#         aux = {
#             "T": T_sig, "V": V_sig,
#             "lap_guidance": L,
#             **lap_aux
#         }

#         return summary, aux


