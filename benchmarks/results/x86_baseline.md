# TensorFold x86 CPU Reference

> This benchmark was run manually on a local Windows/AMD64 machine, not in either CI workflow (both `ci.yml` and `arm-test.yml` run on Linux). It's included as a secondary reference point. The project's primary target platform is ARM64 — see `benchmarks/results/arm64_results.md` and `RESULTS.md` for the authoritative, CI-verified results.

## Model

MNIST MLP:

784 → 512 → 256 → 10

## TensorFold configuration

Energy: 90%

## Accuracy

Dense: 97.79%
TensorFold: 97.06%

Accuracy change: -0.73 percentage points

## Parameters

Dense: 535,818
TensorFold: 279,940

Parameter reduction: 47.75%

## Single-thread CPU latency

| Batch | Dense (ms) | TensorFold (ms) | Speedup |
| ------: | -----------: | ----------------: | --------: |
| 1 | 0.0852 | 0.0820 | 1.039x |
| 16 | 0.3182 | 0.2401 | 1.325x |
| 32 | 0.5031 | 0.3335 | 1.509x |
| 64 | 0.8549 | 0.5702 | 1.499x |
| 256 | 3.0489 | 1.9401 | 1.572x |
