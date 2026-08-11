import torch
import torch.nn as nn

from tensorfold import TensorFoldLinear, compress


def make_low_rank_linear(
    in_features=256,
    out_features=128,
    rank=8,
):
    layer = nn.Linear(
        in_features,
        out_features,
    )

    torch.manual_seed(42)

    left = torch.randn(
        out_features,
        rank,
    )

    right = torch.randn(
        rank,
        in_features,
    )

    weight = left @ right

    with torch.no_grad():
        layer.weight.copy_(weight)

    return layer


def test_compress_linear():
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
    )

    compressed = compress(
        model,
        energy=0.90,
    )

    assert isinstance(compressed, nn.Module)

    x = torch.randn(4, 32)

    y = compressed(x)

    assert y.shape == (4, 8)


def test_compress_replaces_beneficial_linear_layers():
    model = nn.Sequential(
        make_low_rank_linear(),
        nn.ReLU(),
        make_low_rank_linear(
            in_features=128,
            out_features=64,
            rank=8,
        ),
    )

    compressed = compress(
        model,
        energy=0.90,
    )

    assert isinstance(
        compressed,
        nn.Sequential,
    )

    assert isinstance(
        compressed[0],
        TensorFoldLinear,
    )

    assert isinstance(
        compressed[2],
        TensorFoldLinear,
    )


def test_compress_does_not_modify_original_model():
    model = nn.Sequential(
        make_low_rank_linear(),
        nn.ReLU(),
        make_low_rank_linear(
            in_features=128,
            out_features=64,
            rank=8,
        ),
    )

    original_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    compressed = compress(
        model,
        energy=0.90,
    )

    assert compressed is not model

    for original, current in zip(
        original_parameters,
        model.parameters(),
    ):
        assert torch.equal(
            original,
            current.detach(),
        )

    assert isinstance(
        compressed,
        nn.Sequential,
    )

    assert isinstance(
        compressed[0],
        TensorFoldLinear,
    )

    assert isinstance(
        compressed[2],
        TensorFoldLinear,
    )
