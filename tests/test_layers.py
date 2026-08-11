import pytest
import torch
import torch.nn as nn
from typing import cast

from tensorfold.layers import TensorFoldLinear


def test_tensorfold_linear_forward_shape():
    torch.manual_seed(42)

    layer = TensorFoldLinear(
        in_features=16,
        out_features=8,
        rank=4,
    )

    x = torch.randn(
        4,
        16,
    )

    y = layer(x)

    assert y.shape == (4, 8)


def test_tensorfold_linear_backward():
    torch.manual_seed(42)

    layer = TensorFoldLinear(
        in_features=16,
        out_features=8,
        rank=4,
    )

    x = torch.randn(
        4,
        16,
        requires_grad=True,
    )

    y = layer(x)

    loss = y.sum()

    loss.backward()

    assert x.grad is not None
    assert layer.U.grad is not None
    assert layer.V.grad is not None

    if layer.bias is not None:
        assert layer.bias.grad is not None


def test_tensorfold_linear_no_bias():
    layer = TensorFoldLinear(
        in_features=16,
        out_features=8,
        rank=4,
        bias=False,
    )

    assert layer.bias is None

    x = torch.randn(
        4,
        16,
    )

    y = layer(x)

    assert y.shape == (4, 8)


def test_tensorfold_linear_matches_manual_factorization():
    torch.manual_seed(42)

    layer = TensorFoldLinear(
        in_features=8,
        out_features=4,
        rank=2,
        bias=True,
    )

    x = torch.randn(
        3,
        8,
    )

    expected = (
        (x @ layer.U) @ layer.V
        + layer.bias
    )

    actual = layer(x)

    assert torch.allclose(
        actual,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


def test_tensorfold_linear_from_linear():
    torch.manual_seed(42)

    linear = nn.Linear(
        32,
        16,
    )

    layer = TensorFoldLinear.from_linear(
        linear,
        rank=8,
    )

    assert layer.in_features == 32
    assert layer.out_features == 16
    assert layer.rank == 8

    assert layer.U.shape == (32, 8)
    assert layer.V.shape == (8, 16)

    x = torch.randn(
        4,
        32,
    )

    original_output = linear(x)
    factorized_output = layer(x)

    assert original_output.shape == factorized_output.shape


def test_tensorfold_linear_from_linear_approximates_original():
    torch.manual_seed(42)

    linear = nn.Linear(
        64,
        32,
    )

    layer = TensorFoldLinear.from_linear(
        linear,
        rank=32,
    )

    x = torch.randn(
        8,
        64,
    )

    original_output = linear(x)
    factorized_output = layer(x)

    assert torch.allclose(
        factorized_output,
        original_output,
        atol=1e-5,
        rtol=1e-5,
    )


def test_tensorfold_linear_invalid_rank():
    with pytest.raises(ValueError):
        TensorFoldLinear(
            in_features=16,
            out_features=8,
            rank=0,
        )

    with pytest.raises(ValueError):
        TensorFoldLinear(
            in_features=16,
            out_features=8,
            rank=17,
        )


def test_tensorfold_linear_from_linear_invalid_layer():
    invalid_layer = torch.randn(8, 16)

    invalid_layer_as_linear = cast(
        nn.Linear,
        invalid_layer,
    )

    with pytest.raises(TypeError):
        TensorFoldLinear.from_linear(
            invalid_layer_as_linear,
            rank=4,
        )


def test_tensorfold_linear_from_linear_invalid_rank():
    linear = nn.Linear(
        16,
        8,
    )

    with pytest.raises(ValueError):
        TensorFoldLinear.from_linear(
            linear,
            rank=0,
        )

    with pytest.raises(ValueError):
        TensorFoldLinear.from_linear(
            linear,
            rank=17,
        )
