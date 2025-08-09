import torch


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

    # def p_sample(self, xt: torch.Tensor, t: torch.LongTensor, noise_pred: torch.Tensor):
    #     """
    #     Performs one step of the reverse diffusion process (DDPM sampling).
    #     """
    #     device = xt.device
    #     # Move schedules to the correct device
    #     betas_t = self.betas.to(device)[t]
    #     sqrt_one_minus_alpha_bars_t = (1.0 - self.alpha_bars.to(device)[t]).sqrt()
    #     sqrt_recip_alphas_t = (1.0 / self.alphas.to(device)[t]).sqrt()
    #
    #     # Reshape for broadcasting
    #     sqrt_recip_alphas_t = sqrt_recip_alphas_t.view(-1, *([1] * (xt.dim() - 1)))
    #     sqrt_one_minus_alpha_bars_t = sqrt_one_minus_alpha_bars_t.view(-1, *([1] * (xt.dim() - 1)))
    #     betas_t = betas_t.view(-1, *([1] * (xt.dim() - 1)))
    #
    #     # Calculate the mean of the distribution for x_{t-1}
    #     model_mean = sqrt_recip_alphas_t * (xt - betas_t * noise_pred / sqrt_one_minus_alpha_bars_t)
    #
    #     # Add noise to get the final sample, unless it's the last step
    #     posterior_variance = betas_t
    #     noise = torch.randn_like(xt) if t[0] > 0 else 0
    #
    #     return model_mean + (posterior_variance ** 0.5) * noise

    def ddim_sample(self, xt: torch.Tensor, t: torch.LongTensor, t_prev: torch.LongTensor, noise_pred: torch.Tensor,
                    eta: float = 0.1):
        """
        Performs one step of the reverse diffusion process using DDIM.
        """
        device = xt.device

        # Get alpha schedules for current and previous timesteps
        alpha_bar_t = self.alpha_bars.to(device)[t]
        alpha_bar_t_prev = self.alpha_bars.to(device)[t_prev] if t_prev[0] >= 0 else torch.tensor(1.0, device=device)

        # Reshape for broadcasting
        alpha_bar_t = alpha_bar_t.view(-1, *([1] * (xt.dim() - 1)))
        alpha_bar_t_prev = alpha_bar_t_prev.view(-1, *([1] * (xt.dim() - 1)))

        # 1. Predict the original sample (x0) from xt and noise_pred
        pred_x0 = (xt - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()

        # 2. Calculate the variance and direction for the step
        sigma = eta * ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)).sqrt()
        direction = (1 - alpha_bar_t_prev - sigma ** 2).sqrt() * noise_pred

        # 3. Add noise for stochasticity (if eta > 0)
        noise = torch.randn_like(xt) if t[0] > 0 else 0

        # 4. Calculate x_{t-1}
        xt_prev = alpha_bar_t_prev.sqrt() * pred_x0 + direction + sigma * noise
        return xt_prev
