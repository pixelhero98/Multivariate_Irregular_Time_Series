import pytest


torch = pytest.importorskip("torch")


from Model.laptrans import LearnableLaplacianBasis


def test_recurrent_laplacian_basis_backpropagates_to_parameters():
    torch.manual_seed(0)
    basis = LearnableLaplacianBasis(k=3, feat_dim=5, mode="recurrent")
    x = torch.randn(2, 4, 5)

    output = basis(x)
    loss = output.sum()
    loss.backward()

    assert basis.proj.weight.grad is not None
    assert basis._s_real_raw.grad is not None
    assert basis.proj.weight.grad.abs().sum().item() > 0.0
    assert basis._s_real_raw.grad.abs().sum().item() > 0.0
