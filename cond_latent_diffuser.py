import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from cond_diffusion_utils import NoiseScheduler


def get_sinusoidal_pos_emb(L: int, dim: int, device: torch.device) -> torch.Tensor:
    """
    Returns:
        pos_emb: [1, L, dim]
    """
    pos = torch.arange(L, device=device).unsqueeze(1).float()   # [L, 1]
    i   = torch.arange(dim, device=device).unsqueeze(0).float() # [1, dim]
    angle_rates = 1 / (10000 ** (2 * (i // 2) / dim))
    angles = pos * angle_rates                                  # [L, dim]

    pe = torch.zeros(L, dim, device=device)                     # [L, dim]
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe.unsqueeze(0)                                      # [1, L, dim]


class ContextSummarizer(nn.Module):
    """
    Encodes a variable-length context series [B, Lc, F] to a fixed-length summary
    [B, Lh, H] using learnable queries and cross-attention.
    """

    def __init__(self, input_dim, output_seq_len, hidden_dim, num_heads=4):
        super().__init__()
        # MHA sanity check
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        self.input_dim = input_dim
        self.output_seq_len = output_seq_len
        self.hidden_dim = hidden_dim

        # F -> H
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # learnable queries determine output length Lh
        self.summary_queries = nn.Parameter(torch.randn(1, output_seq_len, hidden_dim))  # [1, Lh, H]

        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, context_series: torch.Tensor, context_mask: torch.Tensor | None = None):
        """
        Args:
            context_series: [B, Lc, F]
            context_mask:   [B, Lc] bool (True = PAD/ignore)
        Returns:
            summary: [B, Lh, H]
        """
        B, Lc, F = context_series.shape
        assert F == self.input_dim, f"context_series last dim {F} != expected {self.input_dim}"

        context_embedding = self.input_proj(context_series)          # [B, Lc, H]
        queries = self.summary_queries.expand(B, -1, -1)             # [B, Lh, H]

        summary, _ = self.cross_attn(
            queries,                                                # Q: [B, Lh, H]
            context_embedding,                                      # K: [B, Lc, H]
            context_embedding,                                      # V: [B, Lc, H]
            key_padding_mask=context_mask                           # mask over K/V (Lc)
        )
        return self.norm(summary)                                    # [B, Lh, H]


class LearnableLaplacianBasis(nn.Module):
    """
    Projects x:[B, T, D] to Laplace features:[B, T, 2k] using learnable complex poles.
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

    def reset_parameters(self):
        nn.init.uniform_(self.s_real, a=0.0, b=0.1)
        nn.init.uniform_(self.s_imag, a=-math.pi, b=math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            lap_feats: [B, T, 2k]
        """
        B, T, _ = x.shape
        device = x.device

        # complex poles (-α + iβ), complex64
        alpha = F.softplus(self.s_real) + self.alpha_min           # [k]
        beta = self.s_imag                                         # [k]
        s = torch.complex(-alpha, beta)                            # [k], complex

        # time indices 0..T-1, cast to complex to match s
        t_idx = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1).to(s.dtype)  # [T,1] complex

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


class LapFormer(nn.Module):
    """
    One transformer block operating partly in Laplace space.

    Shapes through the block:
      x (in):           [B, T, H]
      cross-attn out:   [B, T, H]   (keys/values are 'cond')
      Lap(x):           [B, T, 2k]
      self-attn Lap:    [B, T, 2k]
      InvLap:           [B, T, H]
      MLP out:          [B, T, H]
    """
    def __init__(self, dim, num_heads, k, mlp_dim=None):
        super().__init__()
        # attention divisibility guards
        assert dim % num_heads == 0, f"dim ({dim}) % num_heads ({num_heads}) != 0"
        assert (2 * k) % num_heads == 0, f"2*k ({2*k}) % num_heads ({num_heads}) != 0"

        # Cross-attention to conditioning
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        # Laplace-domain self-attention
        self.Lap = LearnableLaplacianBasis(k, dim)
        self.norm_lap = nn.LayerNorm(2 * k)
        self.self_attn = nn.MultiheadAttention(embed_dim=2 * k, num_heads=num_heads, batch_first=True)
        self.norm_inv = nn.LayerNorm(2 * k)
        self.InvLap = LearnableInverseLaplacianBasis(self.Lap)

        # MLP in time-domain
        mlp_dim = mlp_dim or 4 * dim
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))

    def forward(self, x, cond=None,
                tgt_mask: torch.Tensor | None = None,   # [B, T] (True = PAD)
                cond_mask: torch.Tensor | None = None): # [B, Tcond] (True = PAD)
        """
        Args:
            x:         [B, T, H]
            cond:      [B, Tcond, H] or None
            tgt_mask:  [B, T] bool for self-attn (over x)
            cond_mask: [B, Tcond] bool for cross-attn (over cond)
        """
        if cond is not None:
            x = x + self.cross_attn(self.norm_cross(x), cond, cond, key_padding_mask=cond_mask)[0]  # [B, T, H]

        lap = self.Lap(x)                                                                              # [B, T, 2k]
        lap = lap + self.self_attn(self.norm_lap(lap), lap, lap, key_padding_mask=tgt_mask)[0]         # [B, T, 2k]
        x_rec = self.InvLap(self.norm_inv(lap))                                                        # [B, T, H]
        x = x + x_rec                                                                                  # [B, T, H]
        x = x + self.mlp(self.norm_mlp(x))                                                             # [B, T, H]
        return x


class LapDiT(nn.Module):
    """
    Conditional diffusion model.

    Notation:
      B = batch, L = target length (horizon), Lc = context length, F = his_dim,
      D = latent_dim, H = hidden_dim = D + time_embed_dim, k = lap_kernel
    """

    def __init__(self,
                 latent_dim: int,
                 time_embed_dim: int,
                 horizon: int = 10,
                 his_dim: int = 4,
                 lap_kernel: int = 32,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 mlp_dim: int = None,
                 max_timesteps: int = 1000):
        super().__init__()
        self.latent_dim = latent_dim
        self.his_dim = his_dim
        self.horizon = horizon

        # t in [0, 1] -> [B, 1, time_embed_dim]
        self.time_embed = nn.Sequential(nn.Linear(1, time_embed_dim), nn.GELU(), nn.Linear(time_embed_dim, time_embed_dim))

        max_L = 512
        hidden_dim = latent_dim + time_embed_dim                   # H

        # divisibility checks for top-level dims
        assert hidden_dim % num_heads == 0, \
            f"(latent_dim + time_embed_dim) ({hidden_dim}) % num_heads ({num_heads}) != 0"
        assert (2 * lap_kernel) % num_heads == 0, \
            f"2*lap_kernel ({2*lap_kernel}) % num_heads ({num_heads}) != 0"

        # context encoder -> summary length == horizon
        self.conditioning_encoder = ContextSummarizer(
            input_dim=self.his_dim,
            output_seq_len=horizon,                                 # Lh == horizon
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )

        self.k = lap_kernel
        self.layers = nn.ModuleList([
            LapFormer(dim=hidden_dim, num_heads=num_heads, k=lap_kernel, mlp_dim=mlp_dim)
            for _ in range(num_layers)
        ])

        # position embeddings [1, max_L, H]
        self.register_buffer('pos_table', get_sinusoidal_pos_emb(max_L, hidden_dim, torch.device('cpu')))

        # head back to latent space: [B, L, H] -> [B, L, D]
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

        # diffusion scheduler
        self.scheduler = NoiseScheduler(timesteps=max_timesteps)

    def forward(self,
                x_latent: torch.Tensor,                             # [B, L, D]
                t: torch.Tensor,                                    # [B]
                series: torch.Tensor | None = None,                 # [B, Lc, F]
                cond_summary: torch.Tensor | None = None,           # [B, L, H]
                context_mask: torch.Tensor | None = None,           # [B, Lc] (True = PAD)
                tgt_mask: torch.Tensor | None = None):              # [B, L]  (True = PAD)
        """
        Returns:
            eps_pred: [B, L, D]
        """
        B, L, D = x_latent.shape
        device = x_latent.device

        # guard: positional table length
        assert L <= self.pos_table.size(1), \
            f"Sequence length L={L} exceeds pos_table size {self.pos_table.size(1)}"

        # t -> [B, 1] -> embed -> [B, 1, Te] -> repeat to [B, L, Te]
        t_norm = t.float() / self.scheduler.timesteps
        t_emb = self.time_embed(t_norm.view(-1, 1)).unsqueeze(1).repeat(1, L, 1)  # [B, L, Te]

        # concat latent + time: [B, L, D] + [B, L, Te] -> [B, L, H]
        h = torch.cat([x_latent, t_emb], dim=-1)                                    # [B, L, H]
        h = h + self.pos_table[:, :L, :].to(device)                                 # [B, L, H]

        # Only one of (series, cond_summary) should be provided
        assert not (series is not None and cond_summary is not None), \
            "Pass either 'series' or 'cond_summary', not both."

        cond_tensor = None
        cond_mask_for_layers = None  # mask aligned with 'cond_tensor' length

        if cond_summary is None and series is not None:
            # context -> summary: [B, Lc, F] -> [B, L, H]
            assert series.size(-1) == self.his_dim, \
                f"series last dim {series.size(-1)} != his_dim {self.his_dim}"
            cond_summary = self.conditioning_encoder(series, context_mask=context_mask)  # [B, L, H]
            cond_mask_for_layers = None  # summary already length L; no mask by default

        if cond_summary is not None:
            # Expect [B, L, H]
            assert cond_summary.shape[0] == B and cond_summary.shape[1] == L, \
                f"cond_summary shape {tuple(cond_summary.shape)} must be [B={B}, L={L}, H]"
            h = h + cond_summary
            cond_tensor = cond_summary
            # if caller wishes to mask target steps, we can reuse tgt_mask for cond_summary as well
            cond_mask_for_layers = tgt_mask

        # transformer stack
        for layer in self.layers:
            h = layer(h, cond=cond_tensor, tgt_mask=tgt_mask, cond_mask=cond_mask_for_layers)  # [B, L, H]

        return self.output_proj(h)                                                               # [B, L, D]

    @torch.no_grad()
    def generate(self,
                 context_series: torch.Tensor,                        # [B, Lc, F]
                 horizon: int,
                 num_inference_steps: int = 50,
                 guidance_strength: float = 3.0,
                 eta: float = 0.0,
                 context_mask: torch.Tensor | None = None):          # [B, Lc] (True = PAD)
        """
        DDIM sampling with classifier-free guidance.

        Returns:
            x_0: [B, L=horizon, D]
        """
        self.eval()
        device = self.time_embed[0].weight.device
        B = context_series.size(0)

        # horizon must match the summarizer's output length
        assert horizon == self.horizon, \
            f"generate(horizon={horizon}) must equal model.horizon={self.horizon}"

        # safe timesteps: [S] decreasing ints from T-1 to 0
        T = self.scheduler.timesteps
        assert 1 <= num_inference_steps <= T, "num_inference_steps must be within [1, total timesteps]"
        timesteps = torch.linspace(T - 1, 0, num_inference_steps, device=device).long()  # [S]

        # precompute conditioning summary once: [B, L, H]
        cond_summary = self.conditioning_encoder(context_series, context_mask=context_mask)

        # start from noise: [B, L, D]
        x_t = torch.randn(B, horizon, self.latent_dim, device=device)

        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else torch.tensor(-1, device=device)
            t_b = t.expand(B)        # [B]
            tprev_b = t_prev.expand(B)

            # uncond path (no cond)
            noise_uncond = self.forward(x_t, t_b, series=None, cond_summary=None)                 # [B, L, D]
            # cond path (reuse summary); we could pass tgt_mask if using padded targets
            noise_cond   = self.forward(x_t, t_b, series=None, cond_summary=cond_summary)         # [B, L, D]

            # classifier-free guidance
            noise_pred = noise_uncond + guidance_strength * (noise_cond - noise_uncond)           # [B, L, D]

            # DDIM step handled by scheduler
            x_t = self.scheduler.ddim_sample(x_t, t_b, tprev_b, noise_pred, eta)                  # [B, L, D]

        return x_t
