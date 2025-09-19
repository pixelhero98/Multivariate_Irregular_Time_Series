import pytest


torch = pytest.importorskip("torch")


from Model.lapformer import LapFormer


@pytest.mark.parametrize("lap_mode", ["parallel", "recurrent"])
def test_lapformer_forward_backward_with_conditioning(lap_mode: str):
    torch.manual_seed(0)
    model = LapFormer(
        input_dim=4,
        hidden_dim=8,
        num_layers=2,
        num_heads=2,
        laplace_k=3,
        lap_mode=lap_mode,
        self_conditioning=True,
    )

    x = torch.randn(2, 5, 4)
    t_vec = torch.randn(2, 8)
    cond_summary = torch.randn(2, 3, 8)
    sc_feat = torch.randn(2, 5, 4)
    dt = torch.rand(5) if lap_mode == "recurrent" else None

    out = model(x, t_vec, cond_summary=cond_summary, sc_feat=sc_feat, dt=dt)

    assert out.shape == (2, 5, 4)
    assert out.dtype == x.dtype

    loss = out.sum()
    loss.backward()

    first_block = model.blocks[0]
    assert first_block.analysis.proj.weight.grad is not None
    assert model.head_proj.weight.grad is not None
    if lap_mode == "recurrent":
        assert first_block.analysis._s_real_raw.grad is not None


def test_lapformer_positional_cache_extends_and_tracks_dtype():
    torch.manual_seed(0)
    model = LapFormer(
        input_dim=3,
        hidden_dim=6,
        num_layers=1,
        num_heads=2,
        laplace_k=2,
    ).double()

    x = torch.randn(1, 1500, 3, dtype=torch.double)
    t_vec = torch.randn(1, 6, dtype=torch.double)

    out = model(x, t_vec)

    assert out.dtype == torch.double
    assert model.pos_cache.dtype == torch.double
    assert model.pos_cache.shape[1] >= 1500
