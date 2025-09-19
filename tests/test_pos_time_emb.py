import math
import pytest


torch = pytest.importorskip("torch")


from Model.pos_time_emb import get_sinusoidal_pos_emb, timestep_embedding


def test_get_sinusoidal_pos_emb_matches_reference():
    L, dim = 4, 6
    emb = get_sinusoidal_pos_emb(L=L, dim=dim, device=torch.device("cpu"))
    assert emb.shape == (1, L, dim)

    pos = torch.arange(L, dtype=torch.float32).unsqueeze(1)
    half = dim // 2
    idx = torch.arange(half, dtype=torch.float32)
    denom = torch.exp((idx / half) * math.log(10000.0))
    expected = torch.cat([torch.sin(pos / denom), torch.cos(pos / denom)], dim=1)

    assert torch.allclose(emb.squeeze(0), expected)


def test_timestep_embedding_matches_manual_formula():
    t = torch.tensor([0, 3, 7], dtype=torch.int64)
    dim = 8
    max_period = 1000

    emb = timestep_embedding(t, dim=dim, max_period=max_period)
    assert emb.shape == (t.shape[0], dim)

    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device).float() / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    expected = torch.cat([torch.cos(args), torch.sin(args)], dim=1)

    assert torch.allclose(emb, expected)
