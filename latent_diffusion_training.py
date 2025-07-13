import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ------- Unconditional Latent Diffusion -------
class LatentDiffusion(nn.Module):
    def __init__(self, latent_dim, d_model=128, num_layers=6, num_heads=8):
        super().__init__()
        # A simple Transformer as denoiser
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_model*4, activation='gelu')
        self.model = nn.TransformerEncoder(layer, num_layers)
        self.to_eps = nn.Linear(d_model, latent_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.input_proj = nn.Linear(latent_dim, d_model)
        self.d_model = d_model

    def forward(self, z_t, t):
        # z_t: (batch, latent_dim); t: (batch,1) in [0,1]
        h = self.input_proj(z_t).unsqueeze(1)  # (b,1,d)
        h = h + self.time_embed(t).unsqueeze(1)
        h = self.model(h.transpose(0,1)).transpose(0,1).squeeze(1)
        return self.to_eps(h)

# Schedule utilities
def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

def cosine_beta_schedule(T, s: float = 0.008):
    steps = T + 1
    t = torch.linspace(0, T, steps) / T
    alphas_cum = torch.cos(((t + s) / (1 + s)) * (math.pi / 2)) ** 2
    alphas_cum = alphas_cum / alphas_cum[0]
    betas = 1 - (alphas_cum[1:] / alphas_cum[:-1])
    return betas.clamp(max=0.999)

def q_sample(z0, t, betas):
    # sample z_t = sqrt(alpha_bar)*z0 + sqrt(1-alpha_bar)*eps
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)[t]
    noise = torch.randn_like(z0)
    return alpha_bar.sqrt().unsqueeze(-1) * z0 + (1 - alpha_bar).sqrt().unsqueeze(-1) * noise, noise

# Training Loop
def train_diffusion(vae, diffusion, dataset, epochs=10, batch_size=32, lr=1e-4, T=100, schedule_type='linear'):
    """
    Train the diffusion model using either 'linear' or 'cosine' beta schedule.

    Args:
        vae: pretrained TransformerVAE (frozen during diffusion training)
        diffusion: LatentDiffusion model to train
        dataset: Dataset yielding time-series tensors
        epochs: number of training epochs
        batch_size: DataLoader batch size
        lr: learning rate
        T: total diffusion timesteps
        schedule_type: 'linear' or 'cosine'
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae.to(device).eval()
    diffusion.to(device).train()

    # Choose schedule
    if schedule_type == 'cosine':
        betas = cosine_beta_schedule(T).to(device)
    else:
        betas = linear_beta_schedule(T).to(device)

    optim = torch.optim.Adam(diffusion.parameters(), lr=lr)
    for ep in range(epochs):
        for x, _ in loader:
            x = x.to(device)
            with torch.no_grad():
                mu, logvar = vae.encode(x)
                z0 = vae.reparameterize(mu, logvar)
            t = torch.randint(0, T, (x.size(0),), device=device)
            z_t, noise = q_sample(z0, t, betas)
            t_norm = t.float().unsqueeze(-1) / T
            pred_noise = diffusion(z_t, t_norm)
            loss = F.mse_loss(pred_noise, noise)
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f"Epoch {ep+1}, Loss: {loss.item():.4f}")

# Example Dataset placeholder
class TimeSeriesDataset(Dataset):
    def __init__(self, X):
        self.X = X  # Tensor of shape (N, seq_len, input_dim)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], 0

# Usage:
# vae = TransformerVAE(input_dim=D, seq_len=T)
# diffusion = LatentDiffusion(latent_dim=vae.fc_mu.out_features)
# dataset = TimeSeriesDataset(your_data_tensor)
# train_diffusion(vae, diffusion, dataset, schedule_type='cosine')

# End of module
def train_diffusion(vae, diffusion, dataset, epochs=10, batch_size=32, lr=1e-4, T=100):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae.to(device).eval()  # freeze VAE
    diffusion.to(device).train()

    betas = linear_beta_schedule(T).to(device)
    optim = torch.optim.Adam(diffusion.parameters(), lr=lr)
    for ep in range(epochs):
        for x, _ in loader:
            x = x.to(device)
            with torch.no_grad():
                mu, logvar = vae.encode(x)
                z0 = vae.reparameterize(mu, logvar)
            t = torch.randint(0, T, (x.size(0),), device=device)
            z_t, noise = q_sample(z0, t, betas)
            t_norm = t.float().unsqueeze(-1) / T
            pred_noise = diffusion(z_t, t_norm)
            loss = F.mse_loss(pred_noise, noise)
            optim.zero_grad(); loss.backward(); optim.step()
        print(f"Epoch {ep+1}, Loss: {loss.item():.4f}")

# Usage:
# vae = TransformerVAE(input_dim=D, seq_len=T)
# diffusion = LatentDiffusion(latent_dim=vae.fc_mu.out_features)
# dataset = TimeSeriesDataset(your_data_tensor)
# train_diffusion(vae, diffusion, dataset, schedule_type='cosine')
