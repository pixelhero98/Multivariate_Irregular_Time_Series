import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ------- Unconditional Latent Diffusion -------
class LatentDiffusion(nn.Module):
    def __init__(self, latent_dim, d_model=128, num_layers=6, num_heads=8):
        super().__init__()
        # Transformer-based noise predictor
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.input_proj = nn.Linear(latent_dim, d_model)
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.to_eps = nn.Linear(d_model, latent_dim)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # z_t: (batch, latent_dim), t: (batch, 1) with values in [0,1]
        h = self.input_proj(z_t)
        h = h + self.time_embed(t)
        # shape for transformer: (seq_len=1, batch, d_model)
        h = h.unsqueeze(0)
        h = self.encoder(h)
        h = h.squeeze(0)
        return self.to_eps(h)

# ------- Conditional Latent Diffusion (Classifier-Free) -------
class ConditionedDiffusion(LatentDiffusion):
    def __init__(self, latent_dim, num_classes, d_model=128, num_layers=6, num_heads=8):
        super().__init__(latent_dim, d_model, num_layers, num_heads)
        # override label projection
        self.label_proj = nn.Embedding(num_classes + 1, d_model)  # +1 for null
        self.num_classes = num_classes

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        # y: (batch,) int labels or None
        h = self.input_proj(z_t)
        h = h + self.time_embed(t)
        if y is None:
            # null class index = num_classes
            y_idx = torch.full((z_t.size(0),), self.num_classes, device=z_t.device, dtype=torch.long)
        else:
            y_idx = y
        h = h + self.label_proj(y_idx)
        h = h.unsqueeze(0)
        h = self.encoder(h)
        h = h.squeeze(0)
        return self.to_eps(h)

# ------- Beta Schedules & Sampling -------
def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    t = torch.linspace(0, T, steps) / T
    alphas_cum = torch.cos(((t + s) / (1 + s)) * (math.pi / 2)) ** 2
    alphas_cum = alphas_cum / alphas_cum[0]
    betas = 1 - (alphas_cum[1:] / alphas_cum[:-1])
    return betas.clamp(max=0.999)

def q_sample(z0: torch.Tensor, t: torch.Tensor, betas: torch.Tensor) -> tuple:
    # Forward diffusion: add noise to z0 at timestep t
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)[t]
    noise = torch.randn_like(z0)
    z_t = alpha_bar.sqrt().unsqueeze(-1) * z0 + (1 - alpha_bar).sqrt().unsqueeze(-1) * noise
    return z_t, noise

# ------- Training Loops -------
def train_diffusion(
    vae: nn.Module,
    diffusion: LatentDiffusion,
    dataset: Dataset,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    T: int = 100,
    schedule: str = 'linear'
):
    """
    Unconditional diffusion pre-training.

    Args:
        vae: pretrained VAE encoder (with .encode and .reparameterize)
        diffusion: LatentDiffusion instance
        dataset: Dataset yielding raw x
        epochs, batch_size, lr: training hyperparameters
        T: number of diffusion steps
        schedule: 'linear' or 'cosine'
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae.to(device).eval()
    diffusion.to(device).train()

    betas = (
        cosine_beta_schedule(T) if schedule == 'cosine' else linear_beta_schedule(T)
    ).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=lr)

    for epoch in range(epochs):
        for x in loader:
            x = x.to(device)
            with torch.no_grad():
                mu, logvar = vae.encode(x)
                z0 = vae.reparameterize(mu, logvar)
            t = torch.randint(0, T, (z0.size(0),), device=device)
            z_t, noise = q_sample(z0, t, betas)
            t_norm = t.float().unsqueeze(-1) / T
            pred_noise = diffusion(z_t, t_norm)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[Diffusion Pre-train] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


def train_conditional_diffusion(
    vae: nn.Module,
    diffusion: ConditionedDiffusion,
    dataset: Dataset,
    num_classes: int,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    T: int = 100,
    schedule: str = 'linear',
    p_uncond: float = 0.1
):
    """
    Conditional diffusion fine-tuning with classifier-free guidance.

    Args:
        vae: pretrained VAE encoder
        diffusion: ConditionedDiffusion instance
        dataset: Dataset yielding (x, y)
        num_classes: number of classes
        epochs, batch_size, lr: hyperparameters
        T: diffusion steps
        schedule: 'linear' or 'cosine'
        p_uncond: probability to drop conditioning
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae.to(device).eval()
    diffusion.to(device).train()

    # ensure proper label embedding size
    diffusion.label_proj = nn.Embedding(num_classes + 1, diffusion.d_model).to(device)

    betas = (
        cosine_beta_schedule(T) if schedule == 'cosine' else linear_beta_schedule(T)
    ).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=lr)

    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                mu, logvar = vae.encode(x)
                z0 = vae.reparameterize(mu, logvar)
            t = torch.randint(0, T, (z0.size(0),), device=device)
            z_t, noise = q_sample(z0, t, betas)
            t_norm = t.float().unsqueeze(-1) / T

            # classifier-free masking
            mask = torch.rand(z0.size(0), device=device) < p_uncond
            y_input = y.clone()
            y_input[mask] = num_classes

            pred_noise = diffusion(z_t, t_norm, y_input)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[Conditional Fine-tune] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
