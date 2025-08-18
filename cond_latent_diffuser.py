import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Optional, Tuple
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
    pos = torch.arange(L, device=device).unsqueeze(1).float()   # [L, 1]
    i   = torch.arange(dim // 2, device=device).float()         # [dim/2]
    angles = pos / (10000 ** (i / (dim // 2)))                  # [L, dim/2]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [L, dim]
    return emb.unsqueeze(0)  # [1, L, dim]

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


class RelativePositionBias(nn.Module):
    """
    T5-style bucketed relative bias (symmetric; non-causal).
    Produces a float mask [L, L] that is ADDED to attention logits.
    Shared across heads (simple and stable).
    """
    def __init__(self, n_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.n_buckets = n_buckets
        self.max_distance = max_distance
        self.bias = nn.Parameter(torch.zeros(n_buckets))  # learnable scalar per bucket

    def _bucket(self, rel: torch.Tensor) -> torch.Tensor:
        # rel = |j - i|, shape [L, L]
        n = self.n_buckets
        max_d = self.max_distance
        # first half: exact for small distances
        small = rel < (n // 2)
        # remaining buckets: log-scaled
        log_rel = torch.log(rel.float() / (n // 2) + 1e-6)
        log_max = math.log(max(max_d / (n // 2), 1.0))
        scaled = (log_rel / (log_max + 1e-6)) * (n - n // 2 - 1)
        large_bucket = (n // 2 + scaled.floor().clamp(min=0, max=n - n // 2 - 1)).long()
        return torch.where(small, rel.long(), large_bucket)

    def forward(self, L: int, device: torch.device) -> torch.Tensor:
        i = torch.arange(L, device=device)[:, None]
        j = torch.arange(L, device=device)[None, :]
        rel = (j - i).abs()  # [L, L]
        buckets = self._bucket(rel)  # [L, L] integers in [0, n_buckets)
        return self.bias[buckets]    # [L, L] float mask to add to logits
        
# -------------------------------
# Context summarizers
# -------------------------------

class GlobalContextSummarizer(nn.Module):
    """
    Flexible summarizer that can ingest either:
      - per-entity context:     [B, Lc, F]
      - global multi-entity:    [B, M, Lc, F] with optional entity_ids:[B, M]
    Produces a fixed-length summary [B, Lh, H] via cross-attention from
    learnable queries to temporally encoded context tokens.

    Notes:
      * Adds sinusoidal positional encodings along time (Lc).
      * If entity_ids are provided and num_entities is set, adds an entity embedding.
      * Concatenates entities along the sequence dimension (M * Lc) and uses a single cross-attn.
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
        """
        Args:
            context_series:
                [B, Lc, F]  OR  [B, M, Lc, F]
            context_mask:
                [B, Lc]     OR  [B, M, Lc]  (True = PAD/ignore, like nn.MultiheadAttention expects)
            entity_ids:
                [B, M] or None (only used when context is [B,M,Lc,F] and num_entities provided)
        Returns:
            summary: [B, Lh, H]
        """
        x = context_series
        B = x.size(0)
        device = x.device

        if x.dim() == 3:
            # [B, Lc, F]
            B, Lc, F = x.shape
            x = self.input_proj(x)  # [B, Lc, H]
            # add temporal pos enc
            x = x + get_sinusoidal_pos_emb(Lc, self.hidden_dim, device)
            key_padding_mask = context_mask  # [B, Lc] or None
        elif x.dim() == 4:
            # [B, M, Lc, F] -> flatten entities into time
            B, M, Lc, F = x.shape
            x = self.input_proj(x)  # [B, M, Lc, H]
            # add temporal pos enc per entity
            pos = get_sinusoidal_pos_emb(Lc, self.hidden_dim, device)  # [1, Lc, H]
            x = x + pos.unsqueeze(1)  # [B, M, Lc, H]
            if self.entity_emb is not None and entity_ids is not None:
                ent = self.entity_emb(entity_ids)  # [B, M, H]
                x = x + ent.unsqueeze(2)          # broadcast to [B, M, Lc, H]
            x = x.reshape(B, M * Lc, self.hidden_dim)  # [B, M*Lc, H]
            if context_mask is not None:
                key_padding_mask = context_mask.reshape(B, M * Lc)  # [B, M*Lc]
            else:
                key_padding_mask = None
        else:
            raise ValueError("context_series must be [B,Lc,F] or [B,M,Lc,F]")

        queries = self.summary_queries.expand(B, -1, -1)  # [B, Lh, H]
        # Cross-attention
        summary, _ = self.cross_attn(
            queries,         # Q: [B, Lh, H]
            x,               # K: [B, Lctx, H]
            x,               # V: [B, Lctx, H]
            key_padding_mask=key_padding_mask
        )
        return self.norm(self.dropout(summary) + queries)  # residual to stabilize
                    
# -------------------------------
# Laplace basis (time normalized)
# -------------------------------

class LearnableLaplacianBasis(nn.Module):
    """
    Projects x:[B, T, D] -> Laplace features:[B, T, 2k] using learnable complex poles.
    Uses normalized time index t_norm in [0,1] and a learnable time scale to avoid
    over-/under-decay when T varies.
    """
    def __init__(self, k: int, feat_dim: int, alpha_min: float = 1e-6):
        super().__init__()
        self.k = k
        self.alpha_min = alpha_min

        # trainable poles s = -α + iβ
        self.s_real = nn.Parameter(torch.empty(k))
        self.s_imag = nn.Parameter(torch.empty(k))
        self.reset_parameters()

        # D -> k (spectral norm for stability)
        self.proj = spectral_norm(nn.Linear(feat_dim, k, bias=True), n_power_iterations=1, eps=1e-6)

        # learnable global time scale (positive via softplus)
        self._tau = nn.Parameter(torch.tensor(0.0))  # softplus(0)=~0.693 -> scale ~1.0

    def reset_parameters(self):
        nn.init.uniform_(self.s_real, 0.01, 0.2)  # α > 0 (decay)
        nn.init.uniform_(self.s_imag, -math.pi, math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            laplace_feats: [B, T, 2k]
        """
        B, T, D = x.shape
        device = x.device
        # normalized time [0,1]
        t_idx = torch.linspace(0, 1, T, device=device, dtype=torch.float32).unsqueeze(1)  # [T,1]
        tau = F.softplus(self._tau) + 1e-3  # strictly positive
        s = (-self.s_real.clamp_min(self.alpha_min) + 1j * self.s_imag) * tau  # [k] complex

        # basis over time: exp(t * s)
        expo = torch.exp(t_idx * s.unsqueeze(0))                   # [T, k] complex
        re_basis, im_basis = expo.real, expo.imag                  # [T, k] each

        proj_feats = self.proj(x)                                  # [B, T, k]
        real_proj = proj_feats * re_basis.unsqueeze(0)             # [B, T, k]
        imag_proj = proj_feats * im_basis.unsqueeze(0)             # [B, T, k]
        return torch.cat([real_proj, imag_proj], dim=2)            # [B, T, 2k]


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

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None, attn_bias: Optional[torch.Tensor] = None):
        """
        attn_mask: standard mask (e.g., causal). Shape [L, L] or [B, L, L].
        attn_bias: additive bias to logits (same shapes as attn_mask). We sum them if both provided.
        """
        h = self.norm1(x)
        # Combine mask + bias if both provided
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
        """
        x:  [B, L, H]
        kv: [B, Lc, H]
        kv_mask: [B, Lc] (True = ignore)
        """
        q = self.norm_q(x)
        k = self.norm_kv(kv)
        h, _ = self.cross(q, k, k, key_padding_mask=kv_mask)
        return x + self.drop(h)


class LapFormer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int,
                 laplace_k: int = 16, dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lap = LearnableLaplacianBasis(k=laplace_k, feat_dim=input_dim)
        self.lap_proj = nn.Linear(2 * laplace_k, hidden_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_ratio=4.0, dropout=dropout, attn_dropout=attn_dropout)
            for _ in range(num_layers)
        ])
        self.cross_blocks = nn.ModuleList([
            CrossAttnBlock(hidden_dim, num_heads, dropout=dropout, attn_dropout=attn_dropout)
            for _ in range(num_layers)
        ])

        # >>> NEW: shared relative position bias for self-attn <<<
        self.rel_bias = RelativePositionBias(n_buckets=32, max_distance=128)

        self.norm_out = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x_tokens: torch.Tensor, t_emb: torch.Tensor,
                cond_summary: Optional[torch.Tensor] = None,
                cond_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x_tokens.shape
        device = x_tokens.device

        h = self.input_proj(x_tokens) + self.lap_proj(self.lap(x_tokens))
        h = h + get_sinusoidal_pos_emb(L, h.size(-1), device)

        t_add = self.time_mlp(t_emb).unsqueeze(1)
        h = h + t_add

        # Precompute RPB once per forward; shape [L, L]
        attn_bias = self.rel_bias(L, device)  # small float mask added to logits

        for blk, cblk in zip(self.blocks, self.cross_blocks):
            h = blk(h, attn_mask=attn_bias, key_padding_mask=None, attn_bias=None)  # use bias as mask
            if cond_summary is not None:
                h = cblk(h, cond_summary, kv_mask=cond_mask)

        h = self.norm_out(h)
        out = self.out_proj(h)
        return out

# -------------------------------
# Full model
# -------------------------------

class LapDiT(nn.Module):
    """
    Latent conditional diffusion model for multivariate time series.
    - Supports global multi-entity conditioning.
    - Positional encodings in both context and target paths.
    - Optional v-parameterization internally (converted to ε for sampling if needed).
    """
    def __init__(self,
                 data_dim: int,                  # feature dimension of the target series (D)
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 laplace_k: int = 16,
                 cond_len: int = 32,             # Lh: length of conditioning summary
                 num_entities: Optional[int] = None,   # total entities for global context (optional)
                 dropout: float = 0.1,
                 attn_dropout: float = 0.0,
                 predict_type: str = "eps",      # "eps" or "v"
                 timesteps: int = 1000,
                 schedule: str = "cosine"):
        super().__init__()
        assert predict_type in {"eps", "v"}
        self.predict_type = predict_type

        self.scheduler = NoiseScheduler(timesteps=timesteps, schedule=schedule)

        # context summarizer (can handle global context)
        self.context = GlobalContextSummarizer(
            input_dim=data_dim,
            output_seq_len=cond_len,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_entities=num_entities,
            dropout=dropout
        )

        # main DiT stack
        self.model = LapFormer(
            input_dim=data_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            laplace_k=laplace_k,
            dropout=dropout,
            attn_dropout=attn_dropout
        )

        # time embedding dim = hidden_dim
        self.time_dim = hidden_dim

    # ----- helpers -----

    def _time_embed(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B] integer timesteps
        returns: [B, H]
        """
        te = timestep_embedding(t, self.time_dim)    # [B, H]
        te = F.silu(te)
        return te

    def _maybe_build_cond(self,
                          series: Optional[torch.Tensor],
                          series_mask: Optional[torch.Tensor],
                          cond_summary: Optional[torch.Tensor],
                          entity_ids: Optional[torch.Tensor]):
        """
        Builds cond_summary if not provided.
        """
        if cond_summary is not None:
            return cond_summary, None
        if series is None:
            return None, None
        # series can be [B,Lc,F] or [B,M,Lc,F]
        ctx = self.context(series, context_mask=series_mask, entity_ids=entity_ids)  # [B, Lh, H]
        return ctx, None

    # ----- main forward -----

    def forward(self,
                x_t: torch.Tensor,                # [B, L, D] noisy input at time t
                t: torch.Tensor,                  # [B] int timesteps
                series: Optional[torch.Tensor] = None,          # context: [B,Lc,F] or [B,M,Lc,F]
                series_mask: Optional[torch.Tensor] = None,     # [B, Lc] or [B,M,Lc]
                cond_summary: Optional[torch.Tensor] = None,    # [B, Lh, H]
                entity_ids: Optional[torch.Tensor] = None,       # [B, M] if series is [B,M,Lc,F]
                out_type: str = "eps"                            # NEW: "eps" | "v" | "param"
                ) -> torch.Tensor:
        """
        Returns:
            eps_pred: [B, L, D] (epsilon prediction)
        """
        cond_summary, cond_mask = self._maybe_build_cond(series, series_mask, cond_summary, entity_ids)
        t_emb = self._time_embed(t)                       # [B, H]
        raw = self.model(x_t, t_emb, cond_summary, cond_mask)  # model’s native param (= predict_type)  [B, L, D]

        if out_type == "param":
            return raw
        elif out_type == "eps":
            if self.predict_type == "eps":
                return raw
            else:
                return self.scheduler.pred_eps_from_v(x_t, t, raw)
        elif out_type == "v":
            if self.predict_type == "v":
                return raw
            else:
                return self.scheduler.v_from_eps(x_t, t, raw)
        else:
            raise ValueError("out_type must be 'eps', 'v', or 'param'")

    # ----- generation with CFG + imputation inpainting -----

    @torch.no_grad()
    def generate(self,
                 shape: Tuple[int, int, int],  # (B, L, D)
                 steps: int = 50,
                 guidance_strength: float = 1.5,
                 eta: float = 0.0,
                 series: Optional[torch.Tensor] = None,
                 series_mask: Optional[torch.Tensor] = None,
                 cond_summary: Optional[torch.Tensor] = None,
                 entity_ids: Optional[torch.Tensor] = None,
                 y_obs: Optional[torch.Tensor] = None,
                 obs_mask: Optional[torch.Tensor] = None,
                 x_T: Optional[torch.Tensor] = None
                 ) -> torch.Tensor:
        """
        DDIM sampling with classifier-free guidance and optional known-value replacement (imputation).
        """
        device = next(self.parameters()).device
        B, L, D = shape

        if x_T is None:
            x_t = torch.randn(B, L, D, device=device)
        else:
            x_t = x_T.to(device)

        # build conditional summary once (static context)
        built_cond, built_mask = self._maybe_build_cond(series, series_mask, cond_summary, entity_ids)
        cond_summary = built_cond
        cond_mask = built_mask

        # prepare observed values diffusion for inpainting
        if (y_obs is not None) and (obs_mask is not None):
            y_obs = y_obs.to(device)
            obs_mask = obs_mask.to(device)

        # choose an evenly-spaced subset of timesteps for DDIM
        total_T = self.scheduler.timesteps
        steps = max(1, min(steps, total_T))
        step_indices = torch.linspace(0, total_T - 1, steps, dtype=torch.long, device=device).flip(0)
        ts_prev = torch.cat([step_indices[1:], step_indices[-1:].clone()])

        for t_i, t_prev_i in zip(step_indices, ts_prev):
            t_b = t_i.repeat(B)
            tprev_b = t_prev_i.repeat(B)

            # Unconditional path (drop cond)
            eps_uncond = self.forward(x_t, t_b, series=None, series_mask=None, cond_summary=None, out_type="eps")

            # Conditional path
            eps_cond = self.forward(x_t, t_b, series=series, series_mask=series_mask,
                                    cond_summary=cond_summary, entity_ids=entity_ids,
                                    out_type="eps")

            # CFG combine in epsilon-space
            eps = eps_uncond + guidance_strength * (eps_cond - eps_uncond)

            # DDIM step
            x_t = self.scheduler.ddim_sample(x_t, t_b, tprev_b, eps, eta)

            # Hard replace known observations (inpaint) at current noise level
            if (y_obs is not None) and (obs_mask is not None):
                x_t_obs, _ = self.scheduler.q_sample(y_obs, t_b)
                x_t = obs_mask * x_t_obs + (1.0 - obs_mask) * x_t

        return x_t  # x_0 sample
