"""
Three-way ARM64 benchmark: Dense vs TensorFold (torch backend) vs
TensorFold (fused backend), on the same MNIST MLP used in
examples/benchmark_tensorfold.py, with identical methodology
(threads=1, 100 warmup, 500 iterations, 10 repeats) so results are
directly comparable to benchmarks/results/arm64_results.md.

This answers the question bench_fused_local.py could not: does the
fused kernel close the gap to dense at the batch sizes where the
original torch-backend TensorFold benchmark showed a slowdown
(batch 64: 0.910x, batch 256: 0.599x)?
"""

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


DEVICE = torch.device("cpu")
THREADS = 1
WARMUP = 100
ITERATIONS = 500
REPEATS = 10
ENERGY = 0.90
BATCH_SIZES = [1, 16, 32, 64, 256]

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "mnist_mlp.pt"


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def benchmark(model, batch_size):
    x = torch.randn(batch_size, 784, device=DEVICE)

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
        measurements.append((end - start) / ITERATIONS * 1000)

    return {
        "mean": statistics.mean(measurements),
        "median": statistics.median(measurements),
        "std": statistics.stdev(measurements),
    }


def main():
    torch.set_num_threads(THREADS)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    dense_model = MLP().to(DEVICE)
    dense_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    dense_model.eval()

    tensorfold_torch = compress(dense_model, energy=ENERGY, backend="torch")
    tensorfold_torch.eval()

    tensorfold_fused = compress(dense_model, energy=ENERGY, backend="fused")
    tensorfold_fused.eval()

    dense_params = parameter_count(dense_model)
    tf_torch_params = parameter_count(tensorfold_torch)
    tf_fused_params = parameter_count(tensorfold_fused)

    print("TensorFold three-way benchmark: Dense vs torch vs fused")
    print("=========================================================")
    print(f"Platform: {platform.system()} / {platform.machine()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Threads: {THREADS}, Energy: {ENERGY:.0%}")
    print()
    print(f"Dense parameters:            {dense_params:,}")
    print(f"TensorFold (torch) params:   {tf_torch_params:,}")
    print(f"TensorFold (fused) params:   {tf_fused_params:,}")
    print()

    header = (
        f"{'Batch':>6} {'Dense(ms)':>11} {'Torch(ms)':>11} "
        f"{'Fused(ms)':>11} {'Torch/Dense':>12} {'Fused/Dense':>12}"
    )
    print(header)
    print("-" * len(header))

    for batch_size in BATCH_SIZES:
        dense_r = benchmark(dense_model, batch_size)
        torch_r = benchmark(tensorfold_torch, batch_size)
        fused_r = benchmark(tensorfold_fused, batch_size)

        speedup_torch = dense_r["mean"] / torch_r["mean"]
        speedup_fused = dense_r["mean"] / fused_r["mean"]

        print(
            f"{batch_size:>6} "
            f"{dense_r['mean']:>11.4f} "
            f"{torch_r['mean']:>11.4f} "
            f"{fused_r['mean']:>11.4f} "
            f"{speedup_torch:>11.3f}x "
            f"{speedup_fused:>11.3f}x"
        )

    print()
    print("Torch/Dense and Fused/Dense > 1.0x means TensorFold is faster than dense.")


if __name__ == "__main__":
    main()
