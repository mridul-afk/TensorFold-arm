import pytest
import torch
import torch.nn as nn

from tensorfold.layers import TensorFoldLinear


def parameter_count(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def test_tensorfold_linear_shape():
    torch.manual_seed(42)

    layer = TensorFoldLinear(
        in_features=64,
        out_features=32,
        rank=16,
    )

    x = torch.randn(
        8,
        64,
    )

    y = layer(x)

    assert y.shape == (8, 32)


def test_tensorfold_linear_parameter_count():
    layer = TensorFoldLinear(
        in_features=64,
        out_features=32,
        rank=16,
    )

    expected = (
        64 * 16
        + 16 * 32
        + 32
    )

    actual = parameter_count(layer)

    assert actual == expected


def test_tensorfold_linear_has_fewer_parameters_than_dense():
    dense = nn.Linear(
        256,
        128,
    )

    tensorfold = TensorFoldLinear(
        in_features=256,
        out_features=128,
        rank=16,
    )

    dense_parameters = parameter_count(dense)
    tensorfold_parameters = parameter_count(
        tensorfold
    )

    assert tensorfold_parameters < dense_parameters


def test_tensorfold_linear_parameter_reduction():
    in_features = 256
    out_features = 128
    rank = 16

    dense_parameters = (
        in_features * out_features
        + out_features
    )

    tensorfold_parameters = (
        in_features * rank
        + rank * out_features
        + out_features
    )

    reduction = (
        1
        - tensorfold_parameters / dense_parameters
    ) * 100

    assert reduction > 0

    assert reduction == pytest.approx(
        80.933852,
        abs=0.01,
    )


def test_tensorfold_linear_invalid_rank():
    with pytest.raises(ValueError):
        TensorFoldLinear(
            in_features=64,
            out_features=32,
            rank=0,
        )

    with pytest.raises(ValueError):
        TensorFoldLinear(
            in_features=64,
            out_features=32,
            rank=33,
        )


def test_tensorfold_linear_from_linear():
    torch.manual_seed(42)

    dense = nn.Linear(
        16,
        8,
    )

    compressed = TensorFoldLinear.from_linear(
        dense,
        rank=8,
    )

    x = torch.randn(
        4,
        16,
    )

    y_dense = dense(x)
    y_compressed = compressed(x)

    assert y_dense.shape == y_compressed.shape

    assert torch.allclose(
        y_dense,
        y_compressed,
        atol=1e-5,
        rtol=1e-5,
    )


def test_tensorfold_linear_from_linear_preserves_bias():
    torch.manual_seed(42)

    dense = nn.Linear(
        16,
        8,
        bias=True,
    )

    compressed = TensorFoldLinear.from_linear(
        dense,
        rank=8,
    )

    assert compressed.bias is not None

    assert torch.allclose(
        compressed.bias,
        dense.bias,
        atol=1e-6,
        rtol=1e-6,
    )


def test_tensorfold_linear_from_linear_without_bias():
    torch.manual_seed(42)

    dense = nn.Linear(
        16,
        8,
        bias=False,
    )

    compressed = TensorFoldLinear.from_linear(
        dense,
        rank=8,
    )

    assert compressed.bias is None

    x = torch.randn(
        4,
        16,
    )

    assert torch.allclose(
        compressed(x),
        dense(x),
        atol=1e-5,
        rtol=1e-5,
    )


def test_tensorfold_linear_no_bias():
    layer = TensorFoldLinear(
        in_features=16,
        out_features=8,
        rank=4,
        bias=False,
    )

    assert layer.bias is None

    x = torch.randn(
        2,
        16,
    )

    y = layer(x)

    assert y.shape == (2, 8)
