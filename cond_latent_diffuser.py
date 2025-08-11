import torch
import torch.nn as nn
from cond_diffusion_utils import NoiseScheduler
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math


def get_sinusoidal_pos_emb(L: int, dim: int, device: torch.device) -> torch.Tensor:
    """
    Create a [1, L, dim] sinusoidal positional embedding.
    """
    # position indices (L,1) and dimension indices (1,dim)
    pos = torch.arange(L, device=device).unsqueeze(1).float()
    i   = torch.arange(dim, device=device).unsqueeze(0).float()
    # compute the angles: 1/10000^(2*(i//2)/dim)
    angle_rates = 1 / (10000 ** (2 * (i // 2) / dim))
    angles = pos * angle_rates  # (L, dim)

    pe = torch.zeros(L, dim, device=device)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe.unsqueeze(0)  # (1, L, dim)


class ContextSummarizer(nn.Module):
    """
    Encodes and summarizes a context series to a fixed sequence length.
    Uses attention pooling with learnable queries.
    """

    def __init__(self, input_dim, output_seq_len, hidden_dim, num_heads=4):
        super().__init__()
        # Project input features (e.g., 4 for OHLC) into the model's hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # The learnable queries that will "ask" for summary information.
        # Their number determines the output sequence length.
        self.summary_queries = nn.Parameter(torch.randn(1, output_seq_len, hidden_dim))

        # Standard cross-attention layer
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, context_series):
        """
        Args:
            context_series: Tensor of shape [B, context_L, input_dim] (e.g., [B, 60, 4])
        Returns:
            Tensor of shape [B, output_seq_len, hidden_dim] (e.g., [B, 30, 192])
        """
        # 1. Project context features into the hidden dimension
        context_embedding = self.input_proj(context_series)  # -> [B, 60, hidden_dim]

        # 2. Expand queries to match the batch size
        B = context_embedding.size(0)
        queries = self.summary_queries.expand(B, -1, -1)

        # 3. The queries attend to the context to create the summary
        summary, _ = self.cross_attn(queries, context_embedding, context_embedding)

        return self.norm(summary)


class LearnableLaplacianBasis(nn.Module):
    def __init__(self, k: int, feat_dim: int, alpha_min: float = 1e-6):
        """
        Args:
            k: number of complex Laplacian basis elements
            feat_dim: feature dimension of x
            alpha_min: minimum decay rate (small positive)
        """
        super().__init__()
        self.k = k
        self.alpha_min = alpha_min

        # trainable poles α_raw, β
        self.s_real = nn.Parameter(torch.empty(k))
        self.s_imag = nn.Parameter(torch.empty(k))
        self.reset_parameters()

        # learned projection from feat_dim → k with spectral normalization
        self.proj = spectral_norm(
            nn.Linear(feat_dim, k, bias=True),
            n_power_iterations=1,
            eps=1e-6
        )

    def reset_parameters(self):
        nn.init.uniform_(self.s_real, a=0.0, b=0.1)
        nn.init.uniform_(self.s_imag, a=-math.pi, b=math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, feat_dim)
        Returns:
            lap_feats: (B, T, 2*k)
        """
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype

        # build decaying/oscillatory poles
        alpha = F.softplus(self.s_real) + self.alpha_min
        beta = self.s_imag
        s = torch.complex(-alpha, beta)

        # time indices 0…T-1 → (T,1) cast to complex
        t_idx = torch.arange(T, device=device, dtype=dtype).unsqueeze(1).to(s.dtype)

        # compute Laplace basis kernels → (T, k)
        expo = torch.exp(t_idx * s.unsqueeze(0))
        re_basis, im_basis = expo.real, expo.imag

        # project features → (B, T, k)
        proj_feats = self.proj(x)

        # modulate and concat real+imag → (B, T, 2*k)
        real_proj = proj_feats * re_basis.unsqueeze(0)
        imag_proj = proj_feats * im_basis.unsqueeze(0)
        return torch.cat([real_proj, imag_proj], dim=2)


class LearnableInverseLaplacianBasis(nn.Module):
    def __init__(self, laplace_basis: LearnableLaplacianBasis):
        """
        Learnable inverse map (not strict inverse) from Laplace features → input space.
        """
        super().__init__()
        feat_dim = laplace_basis.proj.in_features
        self.inv_proj = spectral_norm(
            nn.Linear(2 * laplace_basis.k, feat_dim, bias=True),
            n_power_iterations=1,
            eps=1e-6
        )

    def forward(self, lap_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lap_feats: (B, T, 2*k)
        Returns:
            x_hat: (B, T, feat_dim)
        """
        return self.inv_proj(lap_feats)


class LapFormer(nn.Module):
    """
    Transformer block with:
      - Laplace transform → Norm → Self‑Attention → Cross‑Attention
      - Norm → Learned Inverse → Residual‑Lap skip → MLP
    """
    def __init__(self, dim, num_heads, k, mlp_dim=None):
        super().__init__()
        # 1. Conditioning module (Cross-Attention first)
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim,
                                                num_heads=num_heads,
                                                batch_first=True)
        # 2. Laplace-domain modules
        self.Lap = LearnableLaplacianBasis(k, dim)
        self.norm_lap = nn.LayerNorm(2 * k)
        self.self_attn = nn.MultiheadAttention(embed_dim=2 * k,
                                               num_heads=num_heads,
                                               batch_first=True)
        self.norm_inv = nn.LayerNorm(2 * k)
        self.InvLap = LearnableInverseLaplacianBasis(self.Lap)

        # 3. Time-domain MLP
        mlp_dim = mlp_dim or 4 * dim
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x, cond=None):
        if cond is not None:
            # 1) Cross-Attention with pre-norm and residual
            x = x + self.cross_attn(self.norm_cross(x), cond, cond)[0]

        # 2) Laplace processing on the conditioned data
        lap = self.Lap(x)                       # → (B, T, 2k)
        lap = lap + self.self_attn(self.norm_lap(lap), lap, lap)[0]
        x_rec = self.InvLap(self.norm_inv(lap)) # → (B, T, dim)

        # 4) Residual and MLP
        x = x + x_rec
        x = x + self.mlp(self.norm_mlp(x))
        return x


class LapDiT(nn.Module):
    """
    Conditional diffusion model using a transformer backbone.
    """

    def __init__(
            self,
            latent_dim: int,
            time_embed_dim: int,
            horizon: int = 10,
            his_dim: int = 4,
            lap_kernel: int = 32,
            num_layers: int = 4,
            num_heads: int = 8,
            mlp_dim: int = None,
            max_timesteps: int = 1000,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.his_dim = his_dim
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        max_L = 512
        hidden_dim = latent_dim + time_embed_dim
        # Masked time-series embedding (project per-feature)
        # Assume input series shape (B, L, F)
        self.conditioning_encoder = ContextSummarizer(
            input_dim=self.his_dim,
            output_seq_len=horizon,  # Output length matches prediction horizon
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
        self.k = lap_kernel
        self.layers = nn.ModuleList([
            LapFormer(
                dim=hidden_dim,
                num_heads=num_heads,
                k=lap_kernel,
                mlp_dim=mlp_dim
            ) for _ in range(num_layers)
        ])

        self.register_buffer(
            'pos_table',
            get_sinusoidal_pos_emb(max_L, hidden_dim, torch.device('cpu'))
        )

        # Project back to latent space
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        # Noise scheduler
        self.scheduler = NoiseScheduler(timesteps=max_timesteps)

    def forward(self, x_latent, t, series=None):
        """
        x_latent: [B, L, latent_dim]
        t: [B] integer timesteps
        class_labels: [B] (optional)
        series: [B, L, latent_dim] (optional)
        scalars: [B] (optional)
        """
        B, L, _ = x_latent.shape
        device = x_latent.device
        # Time embedding
        t_norm = t.float() / self.scheduler.timesteps
        t_emb = self.time_embed(t_norm.view(-1, 1))  # [B, time_embed_dim]
        t_emb = t_emb.unsqueeze(1).repeat(1, L, 1)  # [B, L, time_embed_dim]

        h = torch.cat([x_latent, t_emb], dim=-1)
        pos_emb = self.pos_table[:, :L, :].to(device)  # (1,L,hidden_dim)
        h = h + pos_emb

        cond_tensor = None
        if series is not None:
            summarized_context = self.conditioning_encoder(series)  # -> [B, L, hidden_dim]
            h = h + summarized_context
            cond_tensor = summarized_context

        # Pass through transformer layers
        for layer in self.layers:
            h = layer(h, cond=cond_tensor)

        # Project to predict noise
        out = self.output_proj(h)
        # Predict noise of shape [B,L,latent_dim]
        return out

    # Replace the existing generate method in your LapDiT class

    @torch.no_grad()
    def generate(
            self,
            context_series: torch.Tensor,
            horizon: int,
            num_inference_steps: int = 50,
            guidance_strength: float = 3.0,
            eta: float = 0.0
    ):
        """
        Generate a future sequence using the faster DDIM sampler.
        """
        # This method does not need to change, as the logic inside the forward pass
        # has been updated to handle the new conditioning scheme.
        self.eval()
        device = self.time_embed[0].weight.device
        B = context_series.size(0)

        step_ratio = self.scheduler.timesteps // num_inference_steps
        timesteps = (torch.arange(0, num_inference_steps) * step_ratio).round().long().flip(0).to(device)

        x_t = torch.randn(B, horizon, self.latent_dim, device=device)

        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else torch.tensor(-1, device=device)

            noise_uncond = self.forward(x_t, t.expand(B), series=None)
            noise_cond = self.forward(x_t, t.expand(B), series=context_series)
            noise_pred = noise_uncond + guidance_strength * (noise_cond - noise_uncond)

            x_t = self.scheduler.ddim_sample(x_t, t.expand(B), t_prev.expand(B), noise_pred, eta)

        return x_t