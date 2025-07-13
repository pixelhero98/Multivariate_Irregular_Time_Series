import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Assumes the following are defined and imported:
# - TransformerVAE
# - ConditionedDiffusion
# - ClassificationHead


def ddim_sampling(
    diffusion: ConditionedDiffusion,
    vae: torch.nn.Module,
    label: int,
    num_steps: int = 50,
    guidance_weight: float = 2.0,
    eta: float = 0.0,  # eta=0 for deterministic DDIM
    device: torch.device = None,
):
    """
    Perform DDIM sampling with classifier-free guidance.

    Args:
        diffusion: fine-tuned conditional diffusion model
        vae: pretrained TransformerVAE decoder
        label: integer class label
        num_steps: number of DDIM steps (<= T)
        guidance_weight: strength of guidance (w)
        eta: noise parameter (0 for deterministic)
        device: torch device
    Returns:
        Generated sample x_rec: torch.Tensor of shape (1, seq_len, input_dim)
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion.to(device).eval()
    vae.to(device).eval()

    # Retrieve original diffusion hyperparams
    T = diffusion.num_timesteps if hasattr(diffusion, 'num_timesteps') else num_steps
    betas = diffusion.betas if hasattr(diffusion, 'betas') else torch.linspace(1e-4, 0.02, T).to(device)
    alphas = 1 - betas
    alpha_cum = torch.cumprod(alphas, dim=0)

    # Select ddim timesteps
    ddim_timesteps = torch.linspace(0, T-1, num_steps, dtype=torch.long)

    # Initial z_T ~ N(0, I)
    latent_dim = diffusion.input_proj.in_features
    z_t = torch.randn(1, latent_dim, device=device)

    for i in reversed(range(num_steps)):
        t = ddim_timesteps[i]
        t_norm = torch.tensor([[t / (T-1)]], device=device)

        # Predict eps at conditional and unconditional
        eps_cond = diffusion(z_t, t_norm, torch.full((1,), label, device=device))
        eps_uncond = diffusion(z_t, t_norm, None)
        eps = (1 + guidance_weight) * eps_cond - guidance_weight * eps_uncond

        alpha_t = alpha_cum[t]
        alpha_prev = alpha_cum[ddim_timesteps[i-1]] if i > 0 else torch.tensor(1.0, device=device)

        # DDIM inversion
        sigma = eta * ((1 - alpha_prev) / (1 - alpha_t)).sqrt() * (1 - alpha_t / alpha_prev).sqrt()
        pred_z0 = (z_t - (1 - alpha_t).sqrt().unsqueeze(-1) * eps) / alpha_t.sqrt().unsqueeze(-1)
        dir_term = (1 - alpha_prev - sigma**2).sqrt().unsqueeze(-1) * eps
        noise = sigma.unsqueeze(-1) * torch.randn_like(z_t)
        z_t = alpha_prev.sqrt().unsqueeze(-1) * pred_z0 + dir_term + noise

    # Decode to data space
    with torch.no_grad():
        x_rec = vae.decode(z_t)
    return x_rec.cpu()


def classify_latent(
    z: torch.Tensor,
    classifier: ClassificationHead,
    device: torch.device = None,
) -> int:
    """
    Perform classification on a latent vector z using a trained head.

    Args:
        z: latent tensor of shape (batch, latent_dim)
        classifier: trained ClassificationHead
    Returns:
        predicted class index
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier.to(device).eval()
    z = z.to(device)
    with torch.no_grad():
        logits = classifier(z)
        pred = logits.argmax(dim=-1)
    return pred.item()
