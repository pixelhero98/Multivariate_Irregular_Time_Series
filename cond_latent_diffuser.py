import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Optional, Tuple, Union, List
from cond_diffusion_utils import NoiseScheduler


# -------------------------------
# Positional / timestep embeddings
# -------------------------------

def get_sinusoidal_pos_emb(L: int, dim: int, device: torch.device) -> torch.Tensor:
    """
    Standard 1D sinusoidal positional embeddings.
    Returns: [1, L, dim]
    """
    if dim % 2 != 0:
        raise ValueError("pos_emb dim must be even")
    pos = torch.arange(L, device=device).unsqueeze(1).float()            # [L, 1]
    i   = torch.arange(dim // 2, device=device).float()                  # [dim/2]
    denom = torch.exp((i / (dim // 2)) * math.log(10000.0))              # [dim/2]
    angles = pos / denom                                                 # [L, dim/2]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)       # [L, dim]
    return emb.unsqueeze(0)                                              # [1, L, dim]

def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    From DiT/ADM: create [B, dim] embedding for integer timesteps.
    """
    if dim % 2 != 0:
        raise ValueError("timestep embedding dim must be even")
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device).float() / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    return emb

# -------------------------------
# Context summarizer (kept)
# -------------------------------

class GlobalContextSummarizer(nn.Module):
    """
    Supports:
      - per-entity context:     [B, Lc, F]
      - global multi-entity:    [B, M, Lc, F] (+ optional entity_ids:[B, M])
    Produces fixed-length summary [B, Lh, H] via cross-attention.
    """
    def __init__(self, input_dim: int, output_seq_len: int, hidden_dim: int,
                 num_heads: int = 4, num_entities: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.input_dim = input_dim
        self.output_seq_len = output_seq_len
        self.hidden_dim = hidden_dim
        self.num_entities = num_entities

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.entity_emb = None
        if num_entities is not None:
            self.entity_emb = nn.Embedding(num_entities, hidden_dim)

        self.summary_queries = nn.Parameter(torch.randn(1, output_seq_len, hidden_dim))
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                context_series: torch.Tensor,
                context_mask: Optional[torch.Tensor] = None,
                entity_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = context_series
        B = x.size(0)
        device = x.device

        if x.dim() == 3:
            B, Lc, F = x.shape
            x = self.input_proj(x)  # [B, Lc, H]
            x = x + get_sinusoidal_pos_emb(Lc, self.hidden_dim, device)
            key_padding_mask = context_mask  # [B, Lc] or None
        elif x.dim() == 4:
            B, M, Lc, F = x.shape
            x = self.input_proj(x)  # [B, M, Lc, H]
            pos = get_sinusoidal_pos_emb(Lc, self.hidden_dim, device)
            x = x + pos.unsqueeze(1)  # [B, M, Lc, H]
            if self.entity_emb is not None and entity_ids is not None:
                ent = self.entity_emb(entity_ids)  # [B, M, H]
                x = x + ent.unsqueeze(2)          # [B, M, Lc, H]
            x = x.reshape(B, M * Lc, self.hidden_dim)
            key_padding_mask = context_mask.reshape(B, M * Lc) if context_mask is not None else None
        else:
            raise ValueError("context_series must be [B,Lc,F] or [B,M,Lc,F]")

        queries = self.summary_queries.expand(B, -1, -1)  # [B, Lh, H]
        summary, _ = self.cross_attn(queries, x, x, key_padding_mask=key_padding_mask)
        return self.norm(self.dropout(summary) + queries)
                    
# -------------------------------
# Laplace basis (normalized time)
# -------------------------------

class LearnableLaplacianBasis(nn.Module):
    """
    x:[B, T, D] -> Laplace features:[B, T, 2k] using learnable complex poles.
    Normalized time t in [0,1], with learnable global timescale τ.
    """
    def __init__(self, k: int, feat_dim: int, alpha_min: float = 1e-6):
        super().__init__()
        self.k = k
        self.alpha_min = alpha_min

        self.s_real = nn.Parameter(torch.empty(k))
        self.s_imag = nn.Parameter(torch.empty(k))
        self.reset_parameters()

        self.proj = spectral_norm(nn.Linear(feat_dim, k, bias=True), n_power_iterations=1, eps=1e-6)
        self._tau = nn.Parameter(torch.tensor(0.0))  # softplus -> positive scale

    def reset_parameters(self):
        nn.init.uniform_(self.s_real, 0.01, 0.2)  # α > 0
        nn.init.uniform_(self.s_imag, -math.pi, math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        device = x.device
        t_idx = torch.linspace(0, 1, T, device=device, dtype=torch.float32).unsqueeze(1)  # [T,1]
        tau = F.softplus(self._tau) + 1e-3
        s = (-self.s_real.clamp_min(self.alpha_min) + 1j * self.s_imag) * tau  # [k]

        expo = torch.exp(t_idx * s.unsqueeze(0))  # [T,k] complex
        re_basis, im_basis = expo.real, expo.imag

        proj_feats = self.proj(x)                                  # [B,T,k]
        real_proj = proj_feats * re_basis.unsqueeze(0)             # [B,T,k]
        imag_proj = proj_feats * im_basis.unsqueeze(0)             # [B,T,k]
        return torch.cat([real_proj, imag_proj], dim=2)            # [B,T,2k]


class LearnableInverseLaplacianBasis(nn.Module):
    """ Maps Laplace features [B, T, 2k] back to feature space [B, T, D]. """
    def __init__(self, laplace_basis: LearnableLaplacianBasis):
        super().__init__()
        feat_dim = laplace_basis.proj.in_features                  # D
        self.inv_proj = spectral_norm(nn.Linear(2 * laplace_basis.k, feat_dim, bias=True),
                                      n_power_iterations=1, eps=1e-6)

    def forward(self, lap_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lap_feats: [B, T, 2k]
        Returns:
            x_hat: [B, T, D]
        """
        return self.inv_proj(lap_feats)

# -------------------------------
# Transformer blocks
# -------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=attn_dropout)
        self.drop_path1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.drop_path2 = nn.Dropout(dropout)

        # Residual scaling init for stability (helps regression calibration)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None, attn_bias: Optional[torch.Tensor] = None):
        h = self.norm1(x)
        combined_mask = None
        if attn_mask is not None and attn_bias is not None:
            combined_mask = attn_mask + attn_bias
        elif attn_mask is not None:
            combined_mask = attn_mask
        else:
            combined_mask = attn_bias

        h, _ = self.attn(h, h, h, attn_mask=combined_mask, key_padding_mask=key_padding_mask)
        x = x + self.drop_path1(h)

        h = self.norm2(x)
        h = self.mlp(h)
        x = x + self.drop_path2(h)
        return x

class CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=attn_dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, kv: torch.Tensor, kv_mask: Optional[torch.Tensor] = None):
        q = self.norm_q(x)
        k = self.norm_kv(kv)
        h, _ = self.cross(q, k, k, key_padding_mask=kv_mask)
        return x + self.drop(h)


# -------------------------------
# LapFormer (multi-resolution + self-conditioning)
# -------------------------------

class LaplaceSandwichBlock(nn.Module):
    """
    time x:[B,L,D] --(LearnableLaplacianBasis k)-> z:[B,L,2k]
      -> Linear(2k->H) + pos + time (+ optional self-cond add in H)
      -> TransformerBlock(H) with RPB over time axis
      -> CrossAttn(H  <-- summary2lap->H)
      -> Linear(H->2k) residual in Laplace domain
      -> LearnableInverseLaplacianBasis -> y:[B,L,D]
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, k: int,
                 dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        # analysis / synthesis in Laplace domain
        self.analysis  = LearnableLaplacianBasis(k=k, feat_dim=input_dim)     # [B,L,2k]
        self.synthesis = LearnableInverseLaplacianBasis(self.analysis)         # 2k -> D
        # Laplace <-> hidden
        self.lap2hid = nn.Linear(2 * k, hidden_dim)
        self.hid2lap = nn.Linear(hidden_dim, 2 * k)
        # core attention blocks
        self.self_blk  = TransformerBlock(hidden_dim, num_heads, mlp_ratio=4.0,
                                          dropout=dropout, attn_dropout=attn_dropout)
        self.cross_blk = CrossAttnBlock(hidden_dim, num_heads, dropout=dropout,
                                        attn_dropout=attn_dropout)
        # start near-identity for stability
        nn.init.zeros_(self.hid2lap.weight); nn.init.zeros_(self.hid2lap.bias)

    def forward(self,
                x_time: torch.Tensor,
                pos_emb: torch.Tensor,
                t_vec: torch.Tensor,
                attn_bias: torch.Tensor,
                summary_kv_H: Optional[torch.Tensor] = None,
                kv_mask: Optional[torch.Tensor] = None,
                sc_add_H: Optional[torch.Tensor] = None) -> torch.Tensor:
        # time -> Laplace
        z = self.analysis(x_time)                                 # [B,L,2k]
        # Laplace -> hidden
        h = self.lap2hid(z) + pos_emb + t_vec.unsqueeze(1)        # [B,L,H]
        if sc_add_H is not None:
            h = h + sc_add_H
        # self-attn (time axis) with shared RPB
        h = self.self_blk(h, attn_mask=None, key_padding_mask=None, attn_bias=attn_bias)
        # cross-attn to context summary (already width H after summary2lap proj)
        if summary_kv_H is not None:
            h = self.cross_blk(h, summary_kv_H, kv_mask=kv_mask)
        # hidden -> Laplace (residual), Laplace -> time
        z_upd  = z + self.hid2lap(h)                              # [B,L,2k]
        y_time = self.synthesis(z_upd)                            # [B,L,D]
        return y_time

class LapFormer(nn.Module):
    """
    Stack of Laplace-sandwich blocks (per-block k) with per-block summary2lap conditioning.
    Self-conditioning supported (no gates). Public API unchanged.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int,
                 laplace_k: Union[int, List[int]] = 16, dropout: float = 0.0, attn_dropout: float = 0.0,
                 self_conditioning: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.self_conditioning = self_conditioning

        # ---------- per-block k parsing ----------
        # accepts: int -> broadcast; list -> one per block (if longer, use first num_layers after an optional stem)
        if isinstance(laplace_k, (list, tuple)):
            k_list = list(laplace_k)
            if len(k_list) == 0:
                raise ValueError("laplace_k list must be non-empty")
            if len(k_list) >= num_layers + 1:
                per_layer_k = k_list[1:1+num_layers]            # drop optional stem k0
            elif len(k_list) >= num_layers:
                per_layer_k = k_list[:num_layers]
            else:
                per_layer_k = k_list + [k_list[-1]] * (num_layers - len(k_list))
        else:
            per_layer_k = [int(laplace_k)] * num_layers

        # optional self-conditioning: project time-domain sc_feat to H (added pre-attn in H)
        if self.self_conditioning:
            self.self_cond_proj = nn.Linear(input_dim, hidden_dim)

        # timestep embedding -> H (kept)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # per-block summary2lap: [B,Lh,H] -> Laplace(2k_i) -> H
        self.summary2lap = nn.ModuleList([
            LearnableLaplacianBasis(k=k_i, feat_dim=hidden_dim) for k_i in per_layer_k
        ])
        self.summary2hid = nn.ModuleList([
            nn.Linear(2 * k_i, hidden_dim) for k_i in per_layer_k
        ])

        # Laplace-sandwich blocks over time
        self.blocks = nn.ModuleList([
            LaplaceSandwichBlock(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=num_heads,
                                 k=per_layer_k[i], dropout=dropout, attn_dropout=attn_dropout)
            for i in range(num_layers)
        ])

        # shared RPB and small head in TIME domain, zero-init
        self.head_norm = nn.LayerNorm(input_dim)
        self.head_proj = nn.Linear(input_dim, input_dim)
        nn.init.zeros_(self.head_proj.weight); nn.init.zeros_(self.head_proj.bias)

    def forward(self, x_tokens: torch.Tensor, t_emb: torch.Tensor,
                cond_summary: Optional[torch.Tensor] = None,
                cond_mask: Optional[torch.Tensor] = None,
                sc_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x_tokens: [B, L, D] time-domain latents
        returns:  [B, L, D] native-param prediction in time domain
        """
        B, L, D = x_tokens.shape
        device = x_tokens.device

        # positional & time embeddings (shared across blocks)
        pos = get_sinusoidal_pos_emb(L, self.hidden_dim, device)        # [1,L,H]
        t_vec = self.time_mlp(t_emb)                                    # [B,H]

        # self-conditioning add in H (optional)
        sc_add_H = self.self_cond_proj(sc_feat) if (self.self_conditioning and sc_feat is not None) else None

        # precompute per-block summary2lap -> H
        kvs = [None] * len(self.blocks)
        if cond_summary is not None:
            for i in range(len(self.blocks)):
                s_lap = self.summary2lap[i](cond_summary)              # [B,Lh,2k_i]
                kvs[i] = self.summary2hid[i](s_lap)                    # [B,Lh,H]

        # run the sandwich stack; each block returns TIME-domain [B,L,D]
        h_time = x_tokens
        for i, blk in enumerate(self.blocks):
            h_time = blk(h_time, pos, t_vec, None, kvs[i], cond_mask, sc_add_H)

        # tiny head in time domain (identity at init)
        out = self.head_proj(self.head_norm(h_time))
        return out


class LapDiT(nn.Module):
    """
    Latent conditional diffusion model for multivariate time series.
    - Global multi-entity conditioning
    - Positional encodings in context & target
    - Native parameterization throughout ('eps' or 'v')
    - Self-conditioning (optional)
    - Multi-resolution Laplace (optional via list)
    """
    def __init__(self,
                 data_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 laplace_k: Union[int, List[int]] = 16,   # int or list: [k_stem, k1, k2, ...]
                 cond_len: int = 32,
                 num_entities: Optional[int] = None,
                 dropout: float = 0.1,
                 attn_dropout: float = 0.0,
                 predict_type: str = "eps",               # "eps" or "v"
                 timesteps: int = 1000,
                 schedule: str = "cosine",
                 self_conditioning: bool = False,
                 context_dim: int = None):
        super().__init__()
        assert predict_type in {"eps", "v"}
        self.predict_type = predict_type
        self.self_conditioning = self_conditioning

        self.scheduler = NoiseScheduler(timesteps=timesteps, schedule=schedule)
        ctx_dim = context_dim if context_dim is not None else data_dim
        self.context = GlobalContextSummarizer(
            input_dim=ctx_dim,
            output_seq_len=cond_len,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_entities=num_entities,
            dropout=dropout
        )

        self.model = LapFormer(
            input_dim=data_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            laplace_k=laplace_k,
            dropout=dropout,
            attn_dropout=attn_dropout,
            self_conditioning=self_conditioning
        )

        self.time_dim = hidden_dim

    def _time_embed(self, t: torch.Tensor) -> torch.Tensor:
        te = timestep_embedding(t, self.time_dim)
        te = F.silu(te)
        return te

    def _maybe_build_cond(self,
                          series: Optional[torch.Tensor],
                          series_mask: Optional[torch.Tensor],
                          cond_summary: Optional[torch.Tensor],
                          entity_ids: Optional[torch.Tensor]):
        if cond_summary is not None:
            return cond_summary, None
        if series is None:
            return None, None
        ctx = self.context(series, context_mask=series_mask, entity_ids=entity_ids)
        return ctx, None

    def forward(self,
                x_t: torch.Tensor,
                t: torch.Tensor,
                series: Optional[torch.Tensor] = None,
                series_mask: Optional[torch.Tensor] = None,
                cond_summary: Optional[torch.Tensor] = None,
                entity_ids: Optional[torch.Tensor] = None,
                sc_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns native-param prediction: [B, L, D] in 'eps' or 'v'
        """
        cond_summary, cond_mask = self._maybe_build_cond(series, series_mask, cond_summary, entity_ids)
        t_emb = self._time_embed(t)
        raw = self.model(x_t, t_emb, cond_summary, cond_mask, sc_feat=sc_feat)
        return raw

    @torch.no_grad()
    def generate(self,
                 shape: Tuple[int, int, int],
                 steps: int = 50,
                 guidance_strength: Union[float, Tuple[float, float]] = 1.5,
                 guidance_power: float = 0.3,
                 eta: float = 0.0,
                 series: Optional[torch.Tensor] = None,
                 series_mask: Optional[torch.Tensor] = None,
                 cond_summary: Optional[torch.Tensor] = None,
                 entity_ids: Optional[torch.Tensor] = None,
                 y_obs: Optional[torch.Tensor] = None,
                 obs_mask: Optional[torch.Tensor] = None,
                 x_T: Optional[torch.Tensor] = None,
                 self_cond: bool = False) -> torch.Tensor:
        """
        DDIM sampling with classifier-free guidance (scheduled) and optional inpainting.
        Guidance scheduling:
          - if guidance_strength is float G: g_t = 1 + (G-1) * (1 - alpha_bar_t)^{guidance_power}
          - if (g_min,g_max): g_t = g_min + (g_max-g_min) * (1 - alpha_bar_t)^{guidance_power}
        """
        device = next(self.parameters()).device
        B, L, D = shape

        x_t = torch.randn(B, L, D, device=device) if x_T is None else x_T.to(device)

        built_cond, built_mask = self._maybe_build_cond(series, series_mask, cond_summary, entity_ids)
        cond_summary = built_cond
        cond_mask = built_mask

        if (y_obs is not None) and (obs_mask is not None):
            y_obs = y_obs.to(device)
            obs_mask = obs_mask.to(device)

        total_T = self.scheduler.timesteps
        steps = max(1, min(steps, total_T))
        step_indices = torch.linspace(0, total_T - 1, steps, dtype=torch.long, device=device).flip(0)
        ts_prev = torch.cat([step_indices[1:], step_indices[-1:].clone()])

        for t_i, t_prev_i in zip(step_indices, ts_prev):
            t_b = t_i.repeat(B)
            tprev_b = t_prev_i.repeat(B)

            # optional self-conditioning (teacher is previous prediction of x0)
            sc_feat = None
            if self.self_conditioning and self_cond:
                with torch.no_grad():
                    pred_sc = self.forward(x_t, t_b, series=series, series_mask=series_mask,
                                           cond_summary=cond_summary, entity_ids=entity_ids)
                    x0_sc = self.scheduler.to_x0(x_t, t_b, pred_sc, param_type=self.predict_type)
                    sc_feat = x0_sc  # feed estimated x0 back in

            # unconditional / conditional predictions (native param)
            pred_uncond = self.forward(x_t, t_b, series=None, series_mask=None, cond_summary=None, sc_feat=sc_feat)
            pred_cond   = self.forward(x_t, t_b, series=series, series_mask=series_mask,
                                       cond_summary=cond_summary, entity_ids=entity_ids, sc_feat=sc_feat)

            # scheduled classifier-free guidance
            ab_t = self.scheduler._gather(self.scheduler.alpha_bars, t_b)  # [B]
            if isinstance(guidance_strength, (tuple, list)):
                g_min, g_max = guidance_strength
            else:
                g_min, g_max = 1.0, float(guidance_strength)
            w = torch.pow(1.0 - ab_t, guidance_power).view(-1, 1, 1)
            g_t = g_min + (g_max - g_min) * w  # [B,1,1]

            pred = pred_uncond + g_t * (pred_cond - pred_uncond)

            # DDIM step with native param
            x_t = self.scheduler.ddim_step_from(x_t, t_b, tprev_b, pred, param_type=self.predict_type, eta=eta)

            if (y_obs is not None) and (obs_mask is not None):
                x_t_obs, _ = self.scheduler.q_sample(y_obs, t_b)
                x_t = obs_mask * x_t_obs + (1.0 - obs_mask) * x_t

        return x_t  # x_0
