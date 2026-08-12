import pytest
import torch

from tensorfold.layers import TensorFoldLinear
from tensorfold.fused_backend import fused_available, fused_forward


pytestmark = pytest.mark.skipif(
    not fused_available(),
    reason="Fused CPU extension could not be built in this environment",
)


@pytest.mark.parametrize("batch_size", [1, 7, 16, 64, 256])
@pytest.mark.parametrize("in_features,out_features,rank", [
    (16, 8, 4),
    (784, 512, 100),
    (512, 256, 64),
    (256, 10, 8),
    (100, 100, 1),   # rank=1 edge case
])
def test_fused_matches_torch_backend(batch_size, in_features, out_features, rank):
    torch.manual_seed(0)

    layer_torch = TensorFoldLinear(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        bias=True,
        backend="torch",
    )
    layer_torch.eval()

    x = torch.randn(batch_size, in_features)

    with torch.no_grad():
        y_torch = layer_torch(x)
        y_fused = fused_forward(x, layer_torch.U, layer_torch.V, layer_torch.bias)

    assert y_fused.shape == y_torch.shape
    assert torch.allclose(y_fused, y_torch, atol=1e-4, rtol=1e-4)


def test_fused_matches_torch_backend_no_bias():
    torch.manual_seed(0)

    layer = TensorFoldLinear(
        in_features=128,
        out_features=64,
        rank=16,
        bias=False,
        backend="torch",
    )
    layer.eval()

    x = torch.randn(32, 128)

    with torch.no_grad():
        y_torch = layer(x)
        y_fused = fused_forward(x, layer.U, layer.V, None)

    assert torch.allclose(y_fused, y_torch, atol=1e-4, rtol=1e-4)


def test_backend_fused_flag_routes_through_forward():
    """
    TensorFoldLinear(backend="fused") should produce the same output
    as backend="torch" when called in eval mode with no autograd.
    """
    torch.manual_seed(1)

    layer_torch = TensorFoldLinear(
        in_features=64, out_features=32, rank=8, backend="torch"
    )
    layer_torch.eval()

    layer_fused = TensorFoldLinear(
        in_features=64, out_features=32, rank=8, backend="fused"
    )
    layer_fused.eval()

    # Give both layers identical weights
    with torch.no_grad():
        layer_fused.U.copy_(layer_torch.U)
        layer_fused.V.copy_(layer_torch.V)
        layer_fused.bias.copy_(layer_torch.bias)

    x = torch.randn(50, 64)

    with torch.no_grad():
        y_torch = layer_torch(x)
        y_fused = layer_fused(x)

    assert torch.allclose(y_fused, y_torch, atol=1e-4, rtol=1e-4)


def test_backend_fused_falls_back_during_training():
    """
    backend="fused" must not be used while the module is in training
    mode (it would silently skip autograd tracking). forward() should
    fall back to the torch path so gradients still flow.
    """
    layer = TensorFoldLinear(
        in_features=32, out_features=16, rank=4, backend="fused"
    )
    layer.train()

    x = torch.randn(8, 32, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.U.grad is not None
    assert layer.V.grad is not None
