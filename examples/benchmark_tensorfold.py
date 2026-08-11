import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn

from tensorfold import compress


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.network(x)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device("cpu")

THREADS = 1
WARMUP = 100
ITERATIONS = 500
REPEATS = 10

ENERGY = 0.90

BATCH_SIZES = [
    1,
    16,
    32,
    64,
    256,
]


# --------------------------------------------------
# Model path
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "mnist_mlp.pt"


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def parameter_count(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def benchmark(model, batch_size):
    x = torch.randn(
        batch_size,
        784,
        device=DEVICE,
    )

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)

    measurements = []

    for _ in range(REPEATS):
        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(ITERATIONS):
                model(x)

        end = time.perf_counter()

        latency = (
            (end - start)
            / ITERATIONS
            * 1000
        )

        measurements.append(latency)

    return {
        "mean": statistics.mean(measurements),
        "median": statistics.median(measurements),
        "std": statistics.stdev(measurements),
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    torch.set_num_threads(THREADS)

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = MLP().to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
        )
    )

    model.eval()

    # --------------------------------------------------
    # Compress model
    # --------------------------------------------------

    tensorfold_model = compress(
        model,
        energy=ENERGY,
    )

    tensorfold_model.eval()

    # --------------------------------------------------
    # Parameter statistics
    # --------------------------------------------------

    dense_parameters = parameter_count(model)

    tensorfold_parameters = parameter_count(
        tensorfold_model
    )

    parameter_reduction = (
        1
        - tensorfold_parameters
        / dense_parameters
    ) * 100

    # --------------------------------------------------
    # System information
    # --------------------------------------------------

    print("TensorFold CPU Benchmark")
    print("========================")
    print()

    print(f"Platform: {platform.system()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print()

    print(f"Threads: {THREADS}")
    print(f"Energy target: {ENERGY:.0%}")
    print(f"Warmup: {WARMUP}")
    print(f"Repeats: {REPEATS}")
    print(f"Iterations: {ITERATIONS}")
    print()

    print(
        f"Dense parameters: "
        f"{dense_parameters:,}"
    )

    print(
        f"TensorFold parameters: "
        f"{tensorfold_parameters:,}"
    )

    print(
        f"Parameter reduction: "
        f"{parameter_reduction:.2f}%"
    )

    print()

    # --------------------------------------------------
    # Benchmark results
    # --------------------------------------------------

    print(
        f"{'Batch':>8} "
        f"{'Model':<18} "
        f"{'Mean':>12} "
        f"{'Median':>12} "
        f"{'Std':>12} "
        f"{'Speedup':>12}"
    )

    print("-" * 80)

    for batch_size in BATCH_SIZES:

        dense = benchmark(
            model,
            batch_size,
        )

        tensorfold = benchmark(
            tensorfold_model,
            batch_size,
        )

        speedup = (
            dense["mean"]
            / tensorfold["mean"]
        )

        print(
            f"{batch_size:>8} "
            f"{'Dense':<18} "
            f"{dense['mean']:>12.4f} "
            f"{dense['median']:>12.4f} "
            f"{dense['std']:>12.4f} "
            f"{'-':>12}"
        )

        print(
            f"{'':>8} "
            f"{'TensorFold 90%':<18} "
            f"{tensorfold['mean']:>12.4f} "
            f"{tensorfold['median']:>12.4f} "
            f"{tensorfold['std']:>12.4f} "
            f"{speedup:>11.3f}x"
        )

        print()


if __name__ == "__main__":
    main()
