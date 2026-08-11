# TensorFold-arm ARM64 Benchmark Results

## Platform

| Property | Value |
| --- | --- |
| Operating System | Linux |
| Architecture | ARM64 (`aarch64`) |
| Python | 3.11.15 |
| PyTorch | 2.13.0+cu130 |
| Threads | 1 |
| Energy Target | 90% |
| Warmup Iterations | 100 |
| Benchmark Repeats | 10 |
| Iterations per Repeat | 500 |

---

## Model

The benchmark uses the MNIST MLP:

```
784 → 512 → 256 → 10
```

The model contains three dense `nn.Linear` layers.

TensorFold replaces beneficial dense layers with low-rank factorized layers selected using SVD-based energy preservation.

### Parameter Count

| Model | Parameters |
| --- | --- |
| Dense | 535,818 |
| TensorFold | 279,940 |
| **Parameter Reduction** | **47.75%** |

---

## ARM64 Inference Results

Latency is reported in milliseconds per inference.

| Batch Size | Model | Mean (ms) | Median (ms) | Std (ms) | Speedup |
| --- | --- | --- | --- | --- | --- |
| 1 | Dense | 0.1244 | 0.1244 | 0.0002 | — |
| 1 | TensorFold 90% | 0.1111 | 0.1109 | 0.0012 | 1.119× |
| 16 | Dense | 0.6872 | 0.6909 | 0.0138 | — |
| 16 | TensorFold 90% | 0.4862 | 0.4853 | 0.0022 | 1.413× |
| 32 | Dense | 0.7642 | 0.7602 | 0.0096 | — |
| 32 | TensorFold 90% | 0.6319 | 0.6317 | 0.0011 | 1.209× |
| 64 | Dense | 0.9771 | 0.9737 | 0.0150 | — |
| 64 | TensorFold 90% | 1.0739 | 1.0737 | 0.0020 | 0.910× |
| 256 | Dense | 2.0406 | 2.0376 | 0.0100 | — |
| 256 | TensorFold 90% | 3.4052 | 3.4045 | 0.0048 | 0.599× |

---

## Results Summary

TensorFold achieved a **47.75% reduction** in model parameters.

The inference speedup varies with batch size:

| Batch Size | Speedup | Result |
| --- | --- | --- |
| 1 | 1.119× | Faster |
| 16 | 1.413× | Faster |
| 32 | 1.209× | Faster |
| 64 | 0.910× | Slower |
| 256 | 0.599× | Slower |

The best measured ARM64 result was at batch size 16: **1.413× speedup**.

---

## Interpretation

Parameter reduction and inference speedup are related, but they are not equivalent.

A dense linear layer computes:

$$Y = XW + b$$

where $W \in \mathbb{R}^{\text{in\_features} \times \text{out\_features}}$.

TensorFold factorizes the weight matrix into two smaller matrices:

$$W \approx UV$$

and computes:

$$Y = (XU)V + b$$

This reduces the number of stored parameters and can reduce the amount of arithmetic required. However, TensorFold performs two matrix multiplications instead of one.

Therefore, the performance benefit depends on:

- Batch size
- Matrix dimensions
- Selected rank
- CPU architecture
- Memory access patterns
- Matrix multiplication implementation
- Framework overhead

The ARM64 benchmark demonstrates this directly.

For smaller batch sizes, the reduced computation can outweigh the additional factorized operation. For larger batch sizes, the second matrix multiplication and associated memory operations can become more expensive than the original dense operation.

Therefore, TensorFold should be viewed primarily as a model compression and parameter-reduction technique, with inference speed improvements depending on the workload.

---

## Benchmark Methodology

The benchmark uses:

| Setting | Value |
| --- | --- |
| Threads | 1 |
| Warmup | 100 iterations |
| Repeats | 10 |
| Iterations | 500 per repeat |

For every batch size:

1. A random input tensor is created.
2. Both models are warmed up.
3. The dense model is executed repeatedly.
4. The TensorFold model is executed repeatedly.
5. Mean latency is calculated.
6. Median latency is calculated.
7. Standard deviation is calculated.
8. TensorFold speedup is calculated relative to the dense model.

The speedup is:

$$\text{Speedup} = \frac{\text{Dense Mean Latency}}{\text{TensorFold Mean Latency}}$$

A value greater than 1.0× means TensorFold is faster. A value below 1.0× means the dense model is faster.

---

## ARM64 Validation

The benchmark was executed inside the project's GitHub Actions ARM64 workflow.

The workflow verifies the architecture before running the tests:

```python
platform.machine() → aarch64
```

The same ARM64 workflow then:

1. Installs Python
2. Installs PyTorch dependencies
3. Installs TensorFold-arm
4. Verifies the TensorFold installation
5. Verifies `mnist_mlp.pt`
6. Runs the test suite
7. Runs the ARM64 benchmark

This ensures that the benchmark results are produced on an actual ARM64 environment rather than being inferred from an AMD64 machine.

---

## Reproducibility

The benchmark can be executed with:

```bash
python examples/benchmark_tensorfold.py
```

The benchmark configuration is defined in `examples/benchmark_tensorfold.py`.

The model used by the benchmark is `mnist_mlp.pt`.

The compression target is **90% singular-value energy**.

---

## Conclusion

The ARM64 results demonstrate that TensorFold can substantially reduce the parameter count of dense neural network layers.

For the benchmarked MNIST MLP:

| Metric | Value |
| --- | --- |
| Dense parameters | 535,818 |
| TensorFold parameters | 279,940 |
| Parameter reduction | 47.75% |

TensorFold also produced measurable inference speedups for batch sizes 1, 16, and 32, with the highest measured speedup of **1.413×** at batch size 16.

At larger batch sizes, the factorized implementation was slower than the dense implementation. This highlights an important engineering trade-off: low-rank compression does not guarantee faster inference for every workload.

The benchmark therefore provides evidence for TensorFold as a practical parameter and model-size reduction technique, while also identifying opportunities for future ARM-specific kernel and matrix-multiplication optimizations.
