import pytest
import torch

from tensorfold.decomposition import (
    low_rank_svd,
    select_rank,
    analyze_linear,
)


def test_low_rank_svd():
    torch.manual_seed(42)

    weight = torch.randn(32, 16)

    U, S, Vh = low_rank_svd(
        weight,
        rank=8,
    )

    assert U.shape == (32, 8)
    assert S.shape == (8,)
    assert Vh.shape == (8, 16)


def test_low_rank_svd_reconstruction():
    torch.manual_seed(42)

    weight = torch.randn(32, 16)

    U, S, Vh = low_rank_svd(
        weight,
        rank=16,
    )

    reconstructed = (
        U
        @ torch.diag(S)
        @ Vh
    )

    assert torch.allclose(
        reconstructed,
        weight,
        atol=1e-5,
        rtol=1e-5,
    )


def test_low_rank_svd_truncated_reconstruction():
    torch.manual_seed(42)

    weight = torch.randn(32, 16)

    U, S, Vh = low_rank_svd(
        weight,
        rank=4,
    )

    reconstructed = (
        U
        @ torch.diag(S)
        @ Vh
    )

    assert reconstructed.shape == weight.shape

    # A truncated SVD should not reproduce the
    # original matrix exactly.
    assert not torch.allclose(
        reconstructed,
        weight,
        atol=1e-6,
        rtol=1e-6,
    )


def test_low_rank_svd_invalid_input_dimension():
    weight = torch.randn(4, 4, 4)

    with pytest.raises(ValueError):
        low_rank_svd(
            weight,
            rank=2,
        )


def test_low_rank_svd_invalid_rank():
    weight = torch.randn(16, 16)

    with pytest.raises(ValueError):
        low_rank_svd(
            weight,
            rank=0,
        )

    with pytest.raises(ValueError):
        low_rank_svd(
            weight,
            rank=17,
        )


def test_select_rank():
    torch.manual_seed(42)

    weight = torch.randn(32, 16)

    rank = select_rank(
        weight,
        energy=0.95,
    )

    assert isinstance(rank, int)
    assert 1 <= rank <= 16


def test_select_rank_meets_energy_target():
    torch.manual_seed(42)

    weight = torch.randn(32, 16)

    energy_target = 0.90

    rank = select_rank(
        weight,
        energy=energy_target,
    )

    _, singular_values, _ = torch.linalg.svd(
        weight,
        full_matrices=False,
    )

    energies = singular_values ** 2

    cumulative_energy = torch.cumsum(
        energies,
        dim=0,
    )

    total_energy = energies.sum()

    explained_energy = (
        cumulative_energy / total_energy
    )

    selected_energy = explained_energy[
        rank - 1
    ].item()

    assert selected_energy >= energy_target


def test_select_rank_is_monotonic_with_energy():
    torch.manual_seed(42)

    weight = torch.randn(64, 32)

    rank_80 = select_rank(
        weight,
        energy=0.80,
    )

    rank_90 = select_rank(
        weight,
        energy=0.90,
    )

    rank_99 = select_rank(
        weight,
        energy=0.99,
    )

    assert rank_80 <= rank_90
    assert rank_90 <= rank_99


def test_select_rank_invalid_dimension():
    weight = torch.randn(4, 4, 4)

    with pytest.raises(ValueError):
        select_rank(
            weight,
            energy=0.95,
        )


def test_select_rank_invalid_energy():
    weight = torch.randn(16, 16)

    with pytest.raises(ValueError):
        select_rank(
            weight,
            energy=0,
        )

    with pytest.raises(ValueError):
        select_rank(
            weight,
            energy=1.1,
        )


def test_analyze_linear():
    torch.manual_seed(42)

    weight = torch.randn(256, 512)

    result = analyze_linear(
        weight,
        energy=0.95,
    )

    assert result["in_features"] == 512
    assert result["out_features"] == 256

    assert (
        result["original_parameters"]
        == 512 * 256
    )

    assert result["rank"] >= 1

    assert (
        result["rank"]
        <= min(512, 256)
    )

    assert result["compressed_parameters"] > 0

    assert "parameter_reduction" in result
    assert "compression_possible" in result


def test_analyze_linear_parameter_calculation():
    torch.manual_seed(42)

    weight = torch.randn(128, 256)

    result = analyze_linear(
        weight,
        energy=0.90,
    )

    expected_original = (
        128 * 256
    )

    expected_compressed = (
        256 * result["rank"]
        + result["rank"] * 128
    )

    assert (
        result["original_parameters"]
        == expected_original
    )

    assert (
        result["compressed_parameters"]
        == expected_compressed
    )


def test_analyze_linear_invalid_dimension():
    weight = torch.randn(4, 4, 4)

    with pytest.raises(ValueError):
        analyze_linear(
            weight,
            energy=0.95,
        )
