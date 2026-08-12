"""
Local sanity-check benchmark for the fused backend, mirroring
examples/benchmark_tensorfold.py's methodology (threads=1, 100 warmup,
500 iterations, 10 repeats) but comparing backend="torch" vs
backend="fused" on a single TensorFoldLinear layer shaped like the
first layer of the MNIST MLP (784 -> 512, rank chosen for ~90% energy
on a random matrix as a stand-in).

Run this on the ARM64 CI runner (not just here) before trusting the
numbers -- this box is x86_64, so it tells us the fused kernel is
*correct* and *doesn't regress*, not what the real Arm speedup is.
"""

import platform
import statistics
import time

import torch

from tensorfold.layers import TensorFoldLinear
from tensorfold.fused_backend import fused_available, platform_summary

torch.set_num_threads(1)
torch.manual_seed(0)

IN_FEATURES = 784
OUT_FEATURES = 512
RANK = 246  # matches the worked example in README.md for this shape
WARMUP = 100
ITERATIONS = 500
REPEATS = 10
BATCH_SIZES = [1, 16, 32, 64, 256]


def bench(layer, batch_size):
    x = torch.randn(batch_size, IN_FEATURES)
    layer.eval()

    with torch.no_grad():
        for _ in range(WARMUP):
            layer(x)

    measurements = []
    with torch.no_grad():
        for _ in range(REPEATS):
            start = time.perf_counter()
            for _ in range(ITERATIONS):
                layer(x)
            end = time.perf_counter()
            measurements.append((end - start) / ITERATIONS * 1000)

    return {
        "mean": statistics.mean(measurements),
        "median": statistics.median(measurements),
        "std": statistics.stdev(measurements),
    }


def main():
    print("TensorFold fused-kernel local benchmark")
    print("========================================")
    print(platform_summary())
    print(f"Platform: {platform.system()} / {platform.machine()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Fused extension available: {fused_available()}")
    print()

    layer_torch = TensorFoldLinear(
        in_features=IN_FEATURES, out_features=OUT_FEATURES, rank=RANK,
        backend="torch",
    )
    layer_fused = TensorFoldLinear(
        in_features=IN_FEATURES, out_features=OUT_FEATURES, rank=RANK,
        backend="fused",
    )
    with torch.no_grad():
        layer_fused.U.copy_(layer_torch.U)
        layer_fused.V.copy_(layer_torch.V)
        layer_fused.bias.copy_(layer_torch.bias)

    print(f"{'Batch':>8} {'torch (ms)':>12} {'fused (ms)':>12} {'speedup':>10}")
    print("-" * 48)

    for batch_size in BATCH_SIZES:
        r_torch = bench(layer_torch, batch_size)
        r_fused = bench(layer_fused, batch_size)
        speedup = r_torch["mean"] / r_fused["mean"]
        print(
            f"{batch_size:>8} "
            f"{r_torch['mean']:>12.4f} "
            f"{r_fused['mean']:>12.4f} "
            f"{speedup:>9.3f}x"
        )


if __name__ == "__main__":
    main()
