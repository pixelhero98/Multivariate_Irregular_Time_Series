import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionedDiffusion(nn.Module):
    def __init__(self, latent_dim, d_model=128, num_layers=6,
                 num_heads=8):
        super().__init__()
        # Transformer denoiser
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_model*4, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.to_eps = nn.Linear(d_model, latent_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.input_proj = nn.Linear(latent_dim, d_model)
        self.label_proj = nn.Embedding(1, d_model)  # placeholder
        self.d_model = d_model

    def forward(self, z_t, t, y=None):
        h = self.input_proj(z_t)
        te = self.time_embed(t)
        h = h + te
        if y is not None:
            lb = self.label_proj(y)
        else:
            lb = torch.zeros_like(h)
        h = h + lb
        h = h.unsqueeze(1).transpose(0,1)
        h = self.encoder(h)
        h = h.transpose(0,1).squeeze(1)
        return self.to_eps(h)


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T, s: float = 0.008):
    steps = T + 1
    t = torch.linspace(0, T, steps) / T
    alphas_cum = torch.cos(((t + s) / (1 + s)) * (math.pi / 2))**2
    alphas_cum = alphas_cum / alphas_cum[0]
    betas = 1 - (alphas_cum[1:] / alphas_cum[:-1])
    return betas.clamp(max=0.999)


def q_sample(z0, t, betas):
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)[t]
    noise = torch.randn_like(z0)
    return alpha_bar.sqrt().unsqueeze(-1)*z0 + (1-alpha_bar).sqrt().unsqueeze(-1)*noise, noise


def train_conditional_diffusion(vae, diffusion, dataset, num_classes,
                                epochs=10, batch_size=32, lr=1e-4,
                                T=100, schedule='linear', p_uncond=0.1):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae.to(device).eval()
    diffusion.to(device).train()
    diffusion.label_proj = nn.Embedding(num_classes, diffusion.d_model).to(device)
    betas = (cosine_beta_schedule(T) if schedule=='cosine' else linear_beta_schedule(T)).to(device)
    optim = torch.optim.Adam(diffusion.parameters(), lr=lr)
    for ep in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                mu, logvar = vae.encode(x)
                z0 = vae.reparameterize(mu, logvar)
            t = torch.randint(0, T, (x.size(0),), device=device)
            z_t, noise = q_sample(z0, t, betas)
            t_norm = t.float().unsqueeze(-1) / T
            mask = torch.rand(x.size(0), device=device) < p_uncond
            y_input = y.clone()
            y_input[mask] = num_classes
            pred_noise = diffusion(z_t, t_norm, y_input)
            loss = F.mse_loss(pred_noise, noise)
            optim.zero_grad(); loss.backward(); optim.step()
        print(f"Epoch {ep+1}, Loss: {loss.item():.4f}")
