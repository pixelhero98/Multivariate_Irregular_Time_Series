import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseScheduler:
    """
    Implements a simple linear beta schedule and corresponding alphas for diffusion.
    """
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.LongTensor, noise: torch.Tensor = None):
        """
        Diffuse the clean sample x0 at timestep t by adding noise.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        device = x0.device
        # move the alpha schedule to the data device before indexing
        alpha_bars = self.alpha_bars.to(device)  # → [T]
        # gather the right scalars per batch
        ab_t = alpha_bars[t]  # → [B]
        sqrt_ab = ab_t.sqrt().view(-1, *([1] * (x0.dim() - 1)))
        sqrt_1_ab = (1.0 - ab_t).sqrt().view(-1, *([1] * (x0.dim() - 1)))
        return sqrt_ab * x0 + sqrt_1_ab * noise, noise


class CrossAttentionBlock(nn.Module):
    """
    A single transformer block with self-attention + cross-attention for conditioning.
    """
    def __init__(self, dim, num_heads, cond_dim=None, mlp_dim=None):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True) if cond_dim else None
        if self.cross_attn:
            self.norm_cross = nn.LayerNorm(dim)
        mlp_dim = mlp_dim or 4*dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, cond=None):
        # Self-attention
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h)[0]
        # Cross-attention
        if cond is not None and self.cross_attn:
            h2 = self.norm_cross(x)
            x = x + self.cross_attn(h2, cond, cond)[0]
        # MLP
        h3 = self.norm2(x)
        x = x + self.mlp(h3)
        return x

class DiffusionTransformer(nn.Module):
    """
    Conditional diffusion model using a transformer backbone.
    """

    def __init__(
            self,
            latent_dim: int,
            cond_embed_dim: int,
            time_embed_dim: int,
            num_layers: int = 4,
            num_heads: int = 8,
            mlp_dim: int = None,
            max_timesteps: int = 1000,
            num_classes: int = None
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Optional class embedding
        self.class_embed = nn.Embedding(num_classes, cond_embed_dim) if num_classes else None

        # Regression scalar embedding
        self.scalar_embed = nn.Linear(1, cond_embed_dim)

        # Masked time-series embedding (project per-feature)
        # Assume input series shape (B, L, C)
        self.series_proj = nn.Linear(latent_dim, cond_embed_dim)

        # Combined hidden dimension (latent + time)
        hidden_dim = latent_dim + time_embed_dim
        self.hidden_dim = hidden_dim

        # Position embedding for the concatenated hidden dimension
        self.pos_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Project condition embeddings to hidden dimension
        self.cond_proj = nn.Linear(cond_embed_dim, hidden_dim)

        # Build transformer layers
        self.layers = nn.ModuleList([
            CrossAttentionBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                cond_dim=hidden_dim,
                mlp_dim=mlp_dim
            ) for _ in range(num_layers)
        ])

        # Project back to latent space
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

        # Noise scheduler
        self.scheduler = NoiseScheduler(timesteps=max_timesteps)

    def forward(self, x_latent, t, class_labels=None, series=None, scalars=None):
        """
        x_latent: [B, L, latent_dim]
        t: [B] integer timesteps
        class_labels: [B] (optional)
        series: [B, L, latent_dim] (optional)
        scalars: [B] (optional)
        """
        B, L, _ = x_latent.shape
        # Time embedding
        t_norm = t.float() / self.scheduler.timesteps
        t_emb = self.time_embed(t_norm.view(-1, 1))  # [B, time_embed_dim]
        t_emb = t_emb.unsqueeze(1).repeat(1, L, 1)  # [B, L, time_embed_dim]

        # Prepare conditioning tensor
        conds = []
        if class_labels is not None and self.class_embed:
            conds.append(self.class_embed(class_labels).unsqueeze(1))  # [B,1,cond_dim]
        if series is not None:
            conds.append(self.series_proj(series))  # [B,L,cond_dim]
        if scalars is not None:
            conds.append(self.scalar_embed(scalars.view(-1, 1)).unsqueeze(1))  # [B,1,cond_dim]
        cond_tensor = torch.cat(conds, dim=1) if conds else None  # [B, C', cond_dim]

        # Lift cond embeddings to hidden size
        if cond_tensor is not None:
            cond_tensor = self.cond_proj(cond_tensor)  # [B, C', hidden_dim]

        # Input projection + concat time embedding
        h = torch.cat([x_latent, t_emb], dim=-1)  # [B, L, hidden_dim]
        # Add positional embedding (broadcasted)
        h = h + self.pos_embed

        # Pass through transformer layers
        for layer in self.layers:
            h = layer(h, cond=cond_tensor)

        # Project to predict noise
        out = self.output_proj(h)
        # Predict noise of shape [B,L,latent_dim]
        return out


# ----------------- Inference for Classification -----------------
@torch.no_grad()
def classify_latent(
    mu: torch.Tensor,
    scheduler: NoiseScheduler,
    model: DiffusionTransformer,
    num_class: int = 2,
    num_trials: int = 100,
) -> torch.LongTensor:
    """
    Classify each latent μ by choosing the class (0…num_class-1)
    that yields lowest average denoising error.
    """
    model.eval()
    B = mu.size(0)
    device = mu.device

    # accumulator of shape (num_class, B)
    errors = torch.zeros(num_class, B, device=device)

    for _ in range(num_trials):
        # sample random timestep and noise
        t = torch.randint(0, scheduler.timesteps, (B,), device=device)
        noise = torch.randn_like(mu)
        x_noisy, actual_noise = scheduler.q_sample(mu, t, noise)

        # prepare to batch over classes
        # cls_labels: (num_class * B,)
        cls_labels = (
            torch.arange(num_class, device=device)
            .unsqueeze(1)
            .expand(num_class, B)
            .reshape(-1)
        )
        # repeat x_noisy and actual_noise
        x_rep = x_noisy.unsqueeze(0).expand(num_class, B, *x_noisy.shape[1:]) \
                    .reshape(-1, *x_noisy.shape[1:])
        t_rep = t.unsqueeze(0).expand(num_class, B).reshape(-1)

        # single forward pass for all classes
        pred_noise = model(x_rep, t_rep, cls_labels)
        # reshape back: (num_class, B, *)
        pred_noise = pred_noise.view(num_class, B, *pred_noise.shape[1:])
        actual_noise = actual_noise.unsqueeze(0) \
                                  .expand(num_class, B, *actual_noise.shape[1:]) \
                                  .reshape(num_class, B, *actual_noise.shape[1:])

        # squared error per sample, per class
        # adjust dims (here assuming noise has shape [B, C, L])
        err = ((pred_noise - actual_noise) ** 2).mean(dim=(2,3))
        errors += err

    # average
    errors /= num_trials

    # pick class with lowest error
    preds = errors.argmin(dim=0)
    return preds
