import torch
import torch.nn as nn
from tqdm import tqdm
from cond_latent_diffuser import DiffusionTransformer

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

    # pick class with minimum error
    preds = errors.argmin(dim=0)

    return preds


def regress_latent_grad(
    mu: torch.Tensor,
    scheduler,
    model: nn.Module,
    init_y: float | torch.Tensor,
    lr: float = 1e-2,
    steps: int = 20,
    trials: int = 10,
    clamp_range: tuple[float, float] | None = None,
    show_progress: bool = True,             # new flag
) -> torch.Tensor:
    """
    Perform gradient-based inversion to regress scalar y from latent means mu.
    """
    model.eval()
    device, dtype = mu.device, mu.dtype

    # Initialize y
    if isinstance(init_y, (float, int)):
        y = torch.full((mu.size(0),), float(init_y), device=device, dtype=dtype)
    elif torch.is_tensor(init_y):
        y = init_y.to(device=device, dtype=dtype).clone()
        if y.ndim == 0 or y.numel() == 1:
            y = y.expand(mu.size(0))
    else:
        raise ValueError(f"init_y must be float or Tensor, got {type(init_y)}")

    y.requires_grad_(True)
    optimizer = torch.optim.Adam([y], lr=lr)

    # choose iterator based on show_progress
    iterator = (
        tqdm(range(steps), desc="Regressing y", leave=False)
        if show_progress else
        range(steps)
    )
    for _ in iterator:
        optimizer.zero_grad()
        total_loss = 0.0

        for _ in range(trials):
            t = torch.randint(0, scheduler.timesteps, (mu.size(0),), device=device)
            noise = torch.randn_like(mu)
            x_noisy, actual_noise = scheduler.q_sample(mu, t, noise)

            pred_noise = model(x_noisy, t, scalars=y)
            mse = ((pred_noise - actual_noise) ** 2).mean(
                dim=list(range(1, pred_noise.ndim))
            )
            total_loss += mse

        loss = (total_loss / trials).mean()
        loss.backward()
        optimizer.step()

        if clamp_range is not None:
            y.data.clamp_(clamp_range[0], clamp_range[1])

    return y.detach()