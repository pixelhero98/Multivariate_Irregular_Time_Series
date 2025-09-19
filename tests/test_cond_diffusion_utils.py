import math
import pytest


torch = pytest.importorskip("torch")


from Model import cond_diffusion_utils as utils


def test_noise_scheduler_cosine_schedule_shapes_and_monotonicity():
    scheduler = utils.NoiseScheduler(timesteps=32, schedule="cosine")

    assert scheduler.betas.shape == (32,)
    assert scheduler.alphas.shape == (32,)
    assert scheduler.alpha_bars.shape == (32,)

    # alpha bars should start near 1 and decrease monotonically
    alpha_bars = scheduler.alpha_bars
    assert math.isclose(alpha_bars[0].item(), 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert torch.all(alpha_bars[1:] <= alpha_bars[:-1])
    assert alpha_bars[-1] < alpha_bars[-2]

    # buffers for square roots should match shapes and ranges
    assert scheduler.sqrt_alpha_bars.shape == (32,)
    assert scheduler.sqrt_one_minus_alpha_bars.shape == (32,)
    assert torch.all(scheduler.sqrt_alpha_bars >= 0)
    assert torch.all(scheduler.sqrt_one_minus_alpha_bars >= 0)

    # final beta should approach 1 (cosine schedule ends at pure noise)
    assert scheduler.betas[-1] > 0.5


def test_diffusion_loss_zero_when_model_matches_target_eps_and_v():
    torch.manual_seed(0)
    scheduler = utils.NoiseScheduler(timesteps=16, schedule="linear")
    x0 = torch.randn(4, 3)
    t = torch.tensor([0, 1, 5, 10], dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t, eps_true = scheduler.q_sample(x0, t, noise=noise)

    def model_eps(x_t_in, t_in, cond_summary=None, sc_feat=None):
        return eps_true

    loss_eps = utils.diffusion_loss(
        model_eps,
        scheduler,
        x0,
        t,
        cond_summary=None,
        predict_type="eps",
        reuse_xt_eps=(x_t, eps_true),
    )
    assert torch.allclose(loss_eps, torch.zeros((), device=loss_eps.device), atol=1e-6)

    v_true = scheduler.v_from_eps(x_t, t, eps_true)

    def model_v(x_t_in, t_in, cond_summary=None, sc_feat=None):
        return v_true

    loss_v = utils.diffusion_loss(
        model_v,
        scheduler,
        x0,
        t,
        cond_summary=None,
        predict_type="v",
        reuse_xt_eps=(x_t, eps_true),
    )
    assert torch.allclose(loss_v, torch.zeros((), device=loss_v.device), atol=1e-6)


def test_diffusion_loss_min_snr_weights_match_manual_computation():
    torch.manual_seed(1)
    scheduler = utils.NoiseScheduler(timesteps=8, schedule="linear")
    x0 = torch.randn(3, 2)
    t = torch.tensor([0, 2, 5], dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t, eps_true = scheduler.q_sample(x0, t, noise=noise)

    def zero_model(x_t_in, t_in, cond_summary=None, sc_feat=None):
        return torch.zeros_like(x_t_in)

    loss = utils.diffusion_loss(
        zero_model,
        scheduler,
        x0,
        t,
        cond_summary=None,
        predict_type="eps",
        weight_scheme="weighted_min_snr",
        minsnr_gamma=3.5,
        reuse_xt_eps=(x_t, eps_true),
    )

    per_sample = ((eps_true).pow(2).mean(dim=1))
    abar = scheduler.alpha_bars.index_select(0, t).clamp(1e-6, 1.0 - 1e-6)
    snr = abar / (1.0 - abar)
    gamma = torch.as_tensor(3.5, device=snr.device, dtype=snr.dtype)
    weights = torch.minimum(snr, gamma) / (snr + 1.0)
    w_mean = weights.mean().clamp_min(1e-8)
    expected = (weights * per_sample).mean() / w_mean

    assert torch.allclose(loss, expected, atol=1e-6)
