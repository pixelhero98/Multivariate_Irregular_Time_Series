# Standalone loss function for reuse across models
def vae_loss(x: torch.Tensor, x_rec: torch.Tensor,
              mu: torch.Tensor, logvar: torch.Tensor,
              beta: float = 1.0) -> torch.Tensor:
    """
    Compute VAE loss as reconstruction MSE plus beta-weighted KL divergence.
    """
    recon_loss = F.mse_loss(x_rec, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

