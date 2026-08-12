import warnings

import torch
import torch.nn as nn

from .fused_backend import fused_available, fused_forward


class TensorFoldLinear(nn.Module):
    """
    Factorized Linear layer.

    Instead of storing a dense weight matrix W:

        Y = XW + b

    TensorFold stores:

        U: [in_features, rank]
        V: [rank, out_features]

    and computes:

        Y = (X @ U) @ V + b

    Two backends are available for the forward pass:

        backend="torch" (default): two plain `@` matmuls. Correct on
            any platform/dtype, but on CPU this pays for two separate
            kernel dispatches and materializes the [batch, rank]
            intermediate in memory. Benchmarks show this can be
            *slower* than a dense nn.Linear at large batch sizes
            (see benchmarks/results/arm64_results.md).

        backend="fused": a custom CPU kernel (tensorfold/csrc/fused_linear.cpp)
            that computes the same result in a single pass without
            materializing the intermediate. JIT-compiled on first use.
            Falls back to backend="torch" automatically (with a
            warning) if compilation fails or a non-CPU/non-float32
            tensor is passed.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        backend: str = "torch",
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError(
                "rank must be greater than 0"
            )

        if rank > min(in_features, out_features):
            raise ValueError(
                "rank cannot exceed "
                "min(in_features, out_features)"
            )

        if backend not in ("torch", "fused"):
            raise ValueError(
                'backend must be "torch" or "fused"'
            )

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.backend = backend

        self.U = nn.Parameter(
            torch.empty(
                in_features,
                rank
            )
        )

        self.V = nn.Parameter(
            torch.empty(
                rank,
                out_features
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features)
            )
        else:
            self.register_parameter(
                "bias",
                None
            )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(
            self.U,
            a=5 ** 0.5
        )

        nn.init.kaiming_uniform_(
            self.V,
            a=5 ** 0.5
        )

        if self.bias is not None:
            bound = 1 / self.in_features ** 0.5

            nn.init.uniform_(
                self.bias,
                -bound,
                bound
            )

    def _can_use_fused(self, x: torch.Tensor) -> bool:
        return (
            self.backend == "fused"
            and not self.training
            and not x.requires_grad
            and not self.U.requires_grad
            and not self.V.requires_grad
            and x.device.type == "cpu"
            and x.dtype == torch.float32
            and fused_available()
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        if self._can_use_fused(x):
            return fused_forward(x, self.U, self.V, self.bias)

        if self.backend == "fused":
            # Fused backend was requested but isn't usable for this
            # call (training mode, autograd, non-CPU, non-fp32, or
            # the extension failed to build). fused_backend already
            # warns on build failure; silently fall back here so
            # training/autograd always works.
            pass

        y = x @ self.U
        y = y @ self.V

        if self.bias is not None:
            y = y + self.bias

        return y

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        rank: int,
        backend: str = "torch",
    ):
        """
        Convert a PyTorch nn.Linear layer into
        a TensorFoldLinear layer using truncated SVD.
        """

        if not isinstance(layer, nn.Linear):
            raise TypeError(
                "from_linear expects an nn.Linear layer"
            )

        if rank <= 0:
            raise ValueError(
                "rank must be greater than 0"
            )

        if rank > min(
            layer.in_features,
            layer.out_features
        ):
            raise ValueError(
                "rank cannot exceed "
                "min(in_features, out_features)"
            )

        # PyTorch weight:
        #
        # [out_features, in_features]
        #
        # TensorFold wants:
        #
        # [in_features, out_features]

        weight = layer.weight.data.T

        # Wᵀ ≈ U @ diag(S) @ Vᵀ

        U, S, Vh = torch.linalg.svd(
            weight,
            full_matrices=False
        )

        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]

        # Fold singular values into U.
        #
        # U:  [in_features, rank]
        # S:  [rank]

        U = U * S.unsqueeze(0)

        # Vh:
        #
        # [rank, out_features]

        V = Vh

        # Create TensorFold layer

        new_layer = cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            rank=rank,
            bias=layer.bias is not None,
            backend=backend,
        )

        # Copy factors

        new_layer.U.data.copy_(U)
        new_layer.V.data.copy_(V)

        # Copy bias

        if layer.bias is not None:
            new_layer.bias.data.copy_(
                layer.bias.data
            )

        return new_layer
