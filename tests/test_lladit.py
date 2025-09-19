import types

import pytest

torch = pytest.importorskip("torch")

from Model.lladit import LLapDiT


def test_cfg_mask_scaling_applies_without_rescale():
    model = LLapDiT(
        data_dim=3,
        hidden_dim=4,
        num_layers=1,
        num_heads=1,
        laplace_k=2,
        global_k=2,
        timesteps=8,
        schedule="linear",
        dropout=0.0,
        attn_dropout=0.0,
        self_conditioning=False,
        context_dim=3,
        num_entities=1,
        context_len=1,
        lap_mode="parallel",
    ).eval()

    cond_summary = torch.zeros(1, 1, model.time_dim)

    def fake_forward(self, x_t, t, *, cond_summary=None, **_):
        base = torch.zeros_like(x_t)
        if cond_summary is None:
            return base
        return base + 2.0

    model.forward = types.MethodType(fake_forward, model)

    model.scheduler.to_x0 = lambda x_t, t, pred, param_type: pred
    model.scheduler.ddim_step_from = lambda x_t, t, t_prev, pred, param_type, eta: pred
    model.scheduler.q_sample = lambda y, t: (y, torch.zeros_like(y))

    obs_mask = torch.tensor([[1.0, 0.0]])
    expected = torch.tensor([[[2.0, 2.0, 2.0], [6.0, 6.0, 6.0]]])

    out = model.generate(
        shape=(1, 2, 3),
        steps=1,
        guidance_strength=3.0,
        eta=0.0,
        series=None,
        cond_summary=cond_summary,
        obs_mask=obs_mask,
        y_obs=torch.zeros_like(expected),
        cfg_rescale=False,
    )

    assert torch.allclose(out, expected)
