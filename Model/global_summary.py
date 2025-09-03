
import math
from typing import Optional, Dict, Tuple
from Model.laptrans import LearnableLaplacianBasis
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Positional embedding (fallback) ---
try:
    from Model.pos_time_emb import get_sinusoidal_pos_emb as _pos_emb
except Exception:
    def _pos_emb(T: int, H: int, device=None):
        device = device or torch.device("cpu")
        position = torch.arange(T, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, H, 2, device=device) * (-math.log(10000.0) / H))
        pe = torch.zeros(T, H, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

def _canon_mode(mode: str) -> str:
    m = mode.lower()
    if m in {"parallel", "static", "global"}:
        return "parallel"
    if m in {"recurrent", "tv", "time_varying", "time-varying"}:
        return "recurrent"
    raise ValueError("lap_mode must be one of {'parallel','recurrent'} or their aliases")


class TVHead(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ParallelLaplaceSummarizer(nn.Module):
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
        self.N, self.D, self.H = num_entities, feat_dim, hidden_dim
        self.num_heads = num_heads
        self.add_guidance_tokens = add_guidance_tokens

        self.ctx_proj = nn.Linear(num_entities * feat_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        self.v_head = TVHead(feat_dim)
        self.t_head = TVHead(feat_dim)
        self.lap = LearnableLaplacianBasis(k=lap_k, feat_dim=num_entities, mode="parallel")

        init_raw = math.log(math.e - 1.0)
        self.w_v_raw = nn.Parameter(torch.tensor(init_raw))
        self.w_t_raw = nn.Parameter(torch.tensor(init_raw))

        if add_guidance_tokens:
            self.lap_token_proj = nn.Linear(2 * lap_k, hidden_dim)

        self.queries = nn.Parameter(torch.randn(out_len, hidden_dim) / math.sqrt(hidden_dim))
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                         dropout=dropout, batch_first=True)

    def forward(self,
                x: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None,
                dt: torch.Tensor = None,
                ctx_diff: torch.Tensor = None,
                entity_mask: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, N, D = x.shape
        device = x.device

        mN = None
        if entity_mask is not None:
            mN = entity_mask.to(device=x.device, dtype=x.dtype)
            if mN.dim() == 2:
                mN = mN.unsqueeze(1).expand(B, T, N)
            x = x * mN.unsqueeze(-1)

        ctx = x.reshape(B, T, N * D)
        K = self.ctx_proj(ctx) + _pos_emb(T, self.H, device=device)

        V_sig = self.v_head(x)
        if ctx_diff is None:
            x_diff = torch.zeros_like(x)
            x_diff[:, 1:] = x[:, 1:] - x[:, :-1]
        else:
            x_diff = ctx_diff.to(device)
        if mN is not None:
            x_diff = x_diff * mN.unsqueeze(-1)
        T_sig = self.t_head(x_diff)

        L_v = self.lap(V_sig)
        L_t = self.lap(T_sig)
        a = F.softplus(self.w_v_raw)
        b = F.softplus(self.w_t_raw)
        L = a * L_v + b * L_t

        if self.add_guidance_tokens:
            lap_tokens = self.lap_token_proj(L)
            memory = torch.cat([K, lap_tokens], dim=1)
            values = memory
            key_padding_mask = (None if pad_mask is None else
                                torch.cat([pad_mask.to(torch.bool),
                                           torch.zeros(B, T, device=device, dtype=torch.bool)], dim=1))
        else:
            memory = values = K
            key_padding_mask = None if pad_mask is None else pad_mask.to(torch.bool)

        Q = self.queries.unsqueeze(0).expand(B, -1, -1)
        summary, _ = self.mha(Q, memory, values, key_padding_mask=key_padding_mask)
        summary = self.norm(self.dropout(summary) + Q)

        aux = {"V_sig": V_sig, "T_sig": T_sig, "lap": L, "w_v": a.detach(), "w_t": b.detach()}
        return summary, aux


class PolewiseDiff(nn.Module):
    def __init__(self, k: int, physics_tied: bool = True, residual_scale: float = 0.05):
        super().__init__()
        self.k = k
        self.physics_tied = physics_tied
        self.lap_ref = None

        if physics_tied:
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
            if hasattr(lap, "_tau"):
                self.tau0.copy_(F.softplus(lap._tau) + 1e-3)
            else:
                self.tau0.fill_(1.0)
        return self

    def forward(self, L: torch.Tensor) -> torch.Tensor:
        B, T, twoK = L.shape
        K = twoK // 2
        L = L.view(B, T, K, 2)

        if self.physics_tied:
            if hasattr(self, "lap_ref") and (self.lap_ref is not None):
                alpha0 = self.lap_ref.s_real.clamp_min(self.lap_ref.alpha_min)
                omega0 = self.lap_ref.s_imag
                tau0 = F.softplus(self.lap_ref._tau) + 1e-3
            else:
                alpha0 = self.alpha0
                omega0 = self.omega0
                tau0 = self.tau0

            A = torch.zeros(K, 2, 2, device=L.device, dtype=L.dtype)
            A[:, 0, 0] = -alpha0
            A[:, 0, 1] = -omega0
            A[:, 1, 0] = omega0
            A[:, 1, 1] = -alpha0
            A = A * tau0.view(-1, 1, 1) + self.residual
        else:
            A = self.blocks

        out = torch.einsum('kij,btkj->btki', A, L)
        return out.reshape(B, T, 2 * K)


class SecondOrderLaplaceCombinerPolewise(nn.Module):
    def __init__(self, num_entities: int, k: int,
                 physics_tied: bool = True, residual_scale: float = 0.05,
                 renorm_by_fill: bool = True):
        super().__init__()
        self.lap = LearnableLaplacianBasis(k=k, feat_dim=num_entities, mode="recurrent")
        self.diff = PolewiseDiff(k, physics_tied=physics_tied, residual_scale=residual_scale)
        self.num_entities = num_entities
        self.renorm_by_fill = renorm_by_fill
        if physics_tied:
            _ = self.diff.bind_from_basis(self.lap)

        self.num_head = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, 8), nn.GELU(),
            nn.Linear(8, 4)
        )

    @staticmethod
    def _masked_mean(x: torch.Tensor, m: torch.Tensor, dim: int = -1, eps: float = 1e-6):
        num = (x * m).sum(dim=dim)
        den = m.sum(dim=dim).clamp_min(eps)
        return num / den

    @staticmethod
    def _forward_diff(x: torch.Tensor) -> torch.Tensor:
        return torch.diff(x, dim=1, prepend=x[:, :1])

    def forward(self,
                T_sig: torch.Tensor,
                V_sig: torch.Tensor,
                dt: torch.Tensor = None,
                entity_mask: torch.Tensor = None
                ) -> Tuple[torch.Tensor, Dict]:
        B, T, N = T_sig.shape
        device = T_sig.device
        dtype = T_sig.dtype

        mN = None
        if entity_mask is not None:
            mN = entity_mask.to(device=device, dtype=dtype)
            if mN.dim() == 2:
                mN = mN.unsqueeze(1)
            if mN.dim() == 3 and mN.shape[1] == 1:
                mN = mN.expand(B, T, N)
            elif mN.dim() == 3 and mN.shape[1] == T:
                pass
            else:
                raise ValueError("entity_mask must be [B,N], [B,1,N], or [B,T,N]")
            T_sig = T_sig * mN
            V_sig = V_sig * mN

        dT = self._forward_diff(T_sig)
        dV = self._forward_diff(V_sig)

        if mN is None:
            s_T = T_sig.mean(dim=-1)
            s_V = V_sig.mean(dim=-1)
            s_dT = dT.mean(dim=-1)
            s_dV = dV.mean(dim=-1)
        else:
            s_T = self._masked_mean(T_sig, mN, dim=-1)
            s_V = self._masked_mean(V_sig, mN, dim=-1)
            s_dT = self._masked_mean(dT, mN, dim=-1)
            s_dV = self._masked_mean(dV, mN, dim=-1)

        stats = torch.stack([s_T, s_V, s_dT, s_dV], dim=-1)
        coeff = self.num_head(stats)
        a0, a1, b0, b1 = torch.unbind(coeff, dim=-1)

        if (mN is not None) and self.renorm_by_fill and (self.num_entities > 0):
            real = mN.sum(dim=-1, keepdim=True).clamp_min(1.0)
            scale = (float(self.num_entities) / real).to(dtype=dtype)
            T_in = T_sig * scale
            V_in = V_sig * scale
        else:
            T_in, V_in = T_sig, V_sig

        LT = self.lap(T_in, dt=dt)
        LV = self.lap(V_in, dt=dt)
        LdT = self.diff(LT)
        LdV = self.diff(LV)

        L = (a0.unsqueeze(-1) * LT
             + a1.unsqueeze(-1) * LdT
             + b0.unsqueeze(-1) * LV
             + b1.unsqueeze(-1) * LdV)

        aux = {"LT": LT, "LV": LV, "LdT": LdT, "LdV": LdV, "coeff": coeff}
        return L, aux


class RecurrentLaplaceSummarizer(nn.Module):
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
        self.N, self.D, self.H = num_entities, feat_dim, hidden_dim
        self.num_heads = num_heads
        self.add_guidance_tokens = add_guidance_tokens

        self.ctx_proj = nn.Linear(num_entities * feat_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        self.v_head = TVHead(feat_dim)
        self.t_head = TVHead(feat_dim)

        self.ode_lap = SecondOrderLaplaceCombinerPolewise(
            num_entities=num_entities, k=lap_k,
            physics_tied=physics_tied_derivative, residual_scale=residual_scale
        )

        self.lap_to_film = nn.Linear(2 * lap_k, 2 * hidden_dim)
        self.lap_to_bias = nn.Linear(2 * lap_k, num_heads)

        if add_guidance_tokens:
            self.lap_token_proj = nn.Linear(2 * lap_k, hidden_dim)

        self.queries = nn.Parameter(torch.randn(out_len, hidden_dim) / math.sqrt(hidden_dim))
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                         dropout=dropout, batch_first=True)

    @staticmethod
    def _time_diff_feats(x: torch.Tensor) -> torch.Tensor:
        return torch.diff(x, dim=1, prepend=x[:, :1])

    def forward(self,
            x: torch.Tensor,
            pad_mask: Optional[torch.Tensor] = None,
            dt: torch.Tensor = None,
            ctx_diff: torch.Tensor = None,
            entity_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        B, T, N, D = x.shape
        device = x.device

        mN = None
        if entity_mask is not None:
            mN = entity_mask.to(device=x.device, dtype=x.dtype)
            if mN.dim() == 2:
                mN = mN.unsqueeze(1)
            if mN.size(1) == 1:
                mN = mN.expand(B, T, N)
            x = x * mN.unsqueeze(-1)

        ctx = x.reshape(B, T, N * D)
        ctx_h = self.ctx_proj(ctx)
        pos = _pos_emb(T, self.H, device=device)
        K = Vv = ctx_h + pos

        V_sig = self.v_head(x)
        Tfeat = ctx_diff if ctx_diff is not None else self._time_diff_feats(x)
        if mN is not None and ctx_diff is not None:
            if Tfeat.dim() == 4:
                Tfeat = Tfeat * mN.unsqueeze(-1)
            else:
                Tfeat = Tfeat * mN
        T_sig = self.t_head(Tfeat)
        if mN is not None:
            V_sig = V_sig * mN
            T_sig = T_sig * mN

        L, lap_aux = self.ode_lap(T_sig, V_sig, dt=dt, entity_mask=mN)

        film = self.lap_to_film(L)
        gamma, beta = torch.chunk(film, 2, dim=-1)
        K = (1.0 + gamma) * K + beta
        Vv = (1.0 + gamma) * Vv + beta

        bias_ht = self.lap_to_bias(L)
        Lq = self.queries.shape[0]
        attn_bias = bias_ht.permute(0, 2, 1).unsqueeze(2).expand(B, self.num_heads, Lq, T)
        attn_bias = attn_bias.reshape(B * self.num_heads, Lq, T).to(K.dtype)

        if getattr(self, "add_guidance_tokens", False):
            lap_tokens = self.lap_token_proj(L)
            memory = torch.cat([K, lap_tokens], dim=1)
            values = torch.cat([Vv, lap_tokens], dim=1)

            zeros_bias = torch.zeros(B * self.num_heads, Lq, T, device=device, dtype=attn_bias.dtype)
            attn_bias = torch.cat([attn_bias, zeros_bias], dim=-1)

            key_padding_mask = (None if pad_mask is None else
                                torch.cat([pad_mask.to(torch.bool),
                                           torch.zeros(B, T, device=device, dtype=torch.bool)], dim=1))
        else:
            memory, values = K, Vv
            key_padding_mask = None if pad_mask is None else pad_mask.to(torch.bool)

        Q = self.queries.unsqueeze(0).expand(B, -1, -1)
        summary, _ = self.mha(Q, memory, values,
                              key_padding_mask=key_padding_mask,
                              attn_mask=attn_bias)
        summary = self.norm(self.dropout(summary) + Q)

        aux = {"T": T_sig, "V": V_sig, "lap_guidance": L, **lap_aux}
        return summary, aux


class UnifiedGlobalSummarizer(nn.Module):
    def __init__(self,
                 lap_mode: str,
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
        mode = _canon_mode(lap_mode)
        if mode == "parallel":
            self.impl = ParallelLaplaceSummarizer(
                num_entities=num_entities, feat_dim=feat_dim, hidden_dim=hidden_dim,
                out_len=out_len, num_heads=num_heads, lap_k=lap_k, dropout=dropout,
                add_guidance_tokens=add_guidance_tokens,
            )
        else:
            self.impl = RecurrentLaplaceSummarizer(
                num_entities=num_entities, feat_dim=feat_dim, hidden_dim=hidden_dim,
                out_len=out_len, num_heads=num_heads, lap_k=lap_k, dropout=dropout,
                add_guidance_tokens=add_guidance_tokens,
                physics_tied_derivative=physics_tied_derivative, residual_scale=residual_scale,
            )

    def forward(self, *args, **kwargs):
        return self.impl(*args, **kwargs)
