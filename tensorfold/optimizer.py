import copy

import torch.nn as nn

from .layers import TensorFoldLinear
from .decomposition import analyze_linear


def compress(
    model: nn.Module,
    energy: float = 0.95,
    backend: str = "torch",
) -> nn.Module:
    """
    Convert beneficial nn.Linear layers into
    TensorFoldLinear layers.

    The original model is not modified.

    backend: forward-pass backend for the created TensorFoldLinear
        layers. "torch" (default) uses two plain matmuls. "fused"
        uses the custom fused CPU kernel (see tensorfold/fused_backend.py)
        which avoids materializing the intermediate [batch, rank]
        tensor; falls back to "torch" automatically wherever it isn't
        applicable (training mode, autograd, non-CPU, non-float32).
    """

    model = copy.deepcopy(model)

    _compress_module(
        model,
        energy=energy,
        backend=backend,
    )

    return model


def _compress_module(
    module: nn.Module,
    energy: float,
    backend: str = "torch",
):
    for name, child in list(
        module.named_children()
    ):

        # Already TensorFold
        if isinstance(
            child,
            TensorFoldLinear
        ):
            continue

        # Linear layer
        if isinstance(
            child,
            nn.Linear
        ):

            analysis = analyze_linear(
                child.weight,
                energy=energy,
            )

            rank = analysis["rank"]

            compressed_parameters = (
                child.in_features * rank
                + rank * child.out_features
            )

            if child.bias is not None:
                compressed_parameters += (
                    child.out_features
                )

            original_parameters = sum(
                p.numel()
                for p in child.parameters()
            )

            # Only replace when parameters decrease
            if compressed_parameters < original_parameters:

                new_layer = (
                    TensorFoldLinear.from_linear(
                        child,
                        rank=rank,
                        backend=backend,
                    )
                )

                setattr(
                    module,
                    name,
                    new_layer,
                )

        else:
            # Recursively process nested modules
            _compress_module(
                child,
                energy=energy,
                backend=backend,
            )
