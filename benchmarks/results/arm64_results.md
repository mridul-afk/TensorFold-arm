# TensorFold-arm Benchmarks

This directory contains benchmark-related material for TensorFold-arm.

The purpose of the benchmarks is to measure the effect of replacing dense PyTorch `nn.Linear` layers with SVD-based low-rank TensorFold layers.

The primary target of this project is **CPU inference on ARM64 platforms**.

---

## 1. Benchmark Model

The benchmark uses a multilayer perceptron trained on MNIST.

**Architecture:** `784 → 512 → 256 → 10`

It contains three Linear layers:

```
Input
  │
  ▼
Linear: 784 → 512
  │
ReLU
  │
  ▼
Linear: 512 → 256
  │
ReLU
  │
  ▼
Linear: 256 → 10
  │
  ▼
Output
```

The trained model is stored as `mnist_mlp.pt`.

The same trained model is used for both the dense and TensorFold benchmarks. This ensures that the benchmark compares two representations of the same trained neural network.

---

## 2. Benchmark Objective

The benchmark compares:

- A standard PyTorch model containing dense `nn.Linear` layers.
- The same model after TensorFold replaces beneficial Linear layers with low-rank `TensorFoldLinear` layers.

The benchmark measures:

- Number of parameters
- Parameter reduction
- Mean inference latency
- Median inference latency
- Latency standard deviation
- Inference speedup

The primary evaluation target is an ARM64 CPU environment.

---

## 3. Dense Linear Layer

A standard PyTorch Linear layer stores a dense weight matrix.

For a Linear layer with:

- input features = `N`
- output features = `M`

the weight matrix has shape `M × N`.

Therefore, ignoring the bias, the number of weight parameters is:

$$P_{dense} = M \times N$$

For example, the first layer of the benchmark has input features = 784, output features = 512. Therefore:

$$784 \times 512 = 401{,}408$$

weight parameters.

If the layer has a bias, the bias contains an additional 512 parameters.

---

## 4. Core Idea

TensorFold uses low-rank matrix decomposition to replace the original dense matrix with two smaller matrices.

The original matrix is `W`. Singular Value Decomposition gives:

$$W = U \Sigma V^\top$$

Instead of keeping the complete decomposition, TensorFold keeps only the first `r` singular components:

$$W \approx U_r \Sigma_r V_r^\top$$

The singular values can be folded into one of the factor matrices:

$$A = U_r \Sigma_r, \quad B = V_r^\top$$

Therefore:

$$W \approx AB$$

where `A` is `M × r` and `B` is `r × N`.

The original matrix requires `M × N` parameters. The factorized representation requires `M × r + r × N` parameters. Therefore:

$$P_{tensorfold} = r(M + N)$$

This is the mathematical reason parameter compression is possible.

---

## 5. Parameter Compression Condition

The original dense layer contains `P_dense = M × N` parameters.

The TensorFold representation contains `P_tensorfold = M × r + r × N` parameters.

TensorFold provides parameter compression when:

$$M \times r + r \times N < M \times N \quad \text{or} \quad r(M + N) < MN$$

Therefore, the selected rank must be sufficiently smaller than the dimensions of the original matrix. Rearranging the inequality gives:

$$r < \frac{MN}{M + N}$$

This provides the theoretical maximum rank for which the factorized representation can contain fewer weight parameters than the dense representation.

---

## 6. Example of Parameter Compression

Consider a Linear layer: `784 → 512`

The original weight matrix has `512 × 784` parameters. Therefore:

$$P_{dense} = 401{,}408$$

Now suppose TensorFold selects `r = 246`. The two factor matrices contain `512 × 246` and `246 × 784` parameters. Therefore:

$$P_{tensorfold} = 512 \times 246 + 246 \times 784$$

The factorized representation is smaller than the original dense matrix.

The same process is performed independently for every `nn.Linear` layer.

---

## 7. SVD Energy

TensorFold does not arbitrarily select a rank. It uses the singular values of the weight matrix to determine how much information is retained.

Let the singular values be $\sigma_1, \sigma_2, \sigma_3, \dots, \sigma_k$.

The total SVD energy is:

$$E_{total} = \sigma_1^2 + \sigma_2^2 + \sigma_3^2 + \dots + \sigma_k^2$$

For a selected rank `r`, the retained energy is:

$$E_r = \sigma_1^2 + \sigma_2^2 + \dots + \sigma_r^2$$

The explained energy is:

$$\text{Explained Energy} = \frac{E_r}{E_{total}}$$

TensorFold selects the smallest rank satisfying:

$$\frac{E_r}{E_{total}} \geq \text{target\_energy}$$

---

## 8. Benchmark Energy Setting

The benchmark uses `energy = 0.90`, which corresponds to 90% SVD energy.

TensorFold therefore selects the smallest rank that preserves at least 90% of the singular-value energy of each Linear layer. The compressed model is consequently referred to as **TensorFold 90%**.

> The 90% value refers to matrix decomposition energy. It does **not** mean that the compressed model is guaranteed to retain 90% classification accuracy. Accuracy must be evaluated separately.

---

## 9. Automatic Layer Selection

TensorFold does not automatically replace every Linear layer. For every `nn.Linear` layer, the compression pipeline is:

```
Linear layer
     │
     ▼
SVD
     │
     ▼
Select rank using target energy
     │
     ▼
Calculate compressed parameter count
     │
     ▼
Compare against original parameter count
     │
     ├── Smaller → Replace
     │
     └── Not smaller → Keep original
```

The replacement condition is:

$$P_{tensorfold} < P_{dense}$$

This prevents TensorFold from replacing a layer when the low-rank representation would actually require more parameters.

---

## 10. TensorFold Linear Computation

A normal Linear layer can be represented as:

$$Y = XW^\top + b$$

TensorFold replaces the dense matrix with low-rank factors. Conceptually, the computation becomes:

```
X
 │
 ▼
First factor
 │
 ▼
Rank-r representation
 │
 ▼
Second factor
 │
 ▼
Output
 │
 + bias
 │
 ▼
Y
```

Instead of directly using the full dense matrix, the input passes through the lower-dimensional rank space. The selected rank `r` controls the size of this intermediate representation.

---

## 11. Why Low-Rank Representation Can Be Faster

A dense matrix multiplication uses the full matrix dimensions. With low-rank factorization, the computation is split into two smaller matrix multiplications.

Conceptually, dense `X × W` becomes TensorFold `X × A → rank-r representation → × B`.

When `r << M` and `r << N`, the amount of computation can be reduced.

However, parameter reduction does not automatically guarantee a particular speedup. Actual inference performance depends on:

- CPU architecture
- Matrix dimensions
- Batch size
- Memory behavior
- PyTorch / BLAS implementation
- Number of threads
- CPU frequency
- Cache behavior

---

## 12. Benchmark Configuration

| Setting | Value |
| --- | --- |
| Device | CPU |
| PyTorch threads | 1 |
| Warmup iterations | 100 |
| Measured iterations | 500 |
| Repeats | 10 |
| Batch sizes | 1, 16, 32, 64, 256 |

The benchmark explicitly sets `torch.set_num_threads(1)`. This keeps the comparison controlled by using a single CPU thread.

---

## 13. Warmup

Before collecting timing measurements, the benchmark performs 100 warmup iterations. The warmup iterations are not included in the reported latency.

The purpose of warmup is to allow the execution environment to reach a more stable state before timing begins.

---

## 14. Inference Measurement

The benchmark uses `time.perf_counter()` for timing. Inference is performed under `torch.no_grad()` because the benchmark measures inference performance rather than training performance.

For each repeat, the model is executed for 500 iterations. The elapsed time is divided by the number of iterations, and the result is converted into milliseconds:

$$\text{Mean latency} = \frac{\text{total execution time}}{\text{number of iterations}}$$

---

## 15. Benchmark Statistics

For every model and batch size, the benchmark reports:

- **Mean** — the average latency across the benchmark iterations.
- **Median** — the middle measured latency, less affected by individual timing outliers.
- **Standard Deviation** — how much the repeated measurements vary.

---

## 16. Speedup

TensorFold speedup is calculated using:

$$\text{Speedup} = \frac{\text{Dense mean latency}}{\text{TensorFold mean latency}}$$

For example, if Dense = 8.0 ms and TensorFold = 5.0 ms, then Speedup = 8.0 / 5.0 = **1.6×**.

A speedup greater than 1.0× means TensorFold is faster for that measurement.

---

## 17. ARM64 Environment

TensorFold-arm is specifically designed to evaluate CPU inference on Arm64 platforms.

The GitHub Actions workflow uses a native ARM64 runner: `runs-on: ubuntu-24.04-arm`, using the `aarch64` architecture.

The workflow verifies the architecture before running the benchmark. This is important because an x86 benchmark cannot directly establish performance characteristics on an ARM64 CPU.

---

## 18. ARM64 CI Pipeline

The ARM64 workflow performs:

```
Checkout repository
        │
        ▼
Setup Python
        │
        ▼
Verify ARM64 architecture
        │
        ▼
Install dependencies
        │
        ▼
Install TensorFold-arm
        │
        ▼
Verify package import
        │
        ▼
Verify model file
        │
        ▼
Run pytest
        │
        ▼
Run benchmark
        │
        ▼
Run fused-kernel correctness tests
        │
        ▼
Run fused-kernel benchmark
        │
        ▼
Run three-way benchmark (dense / torch / fused)
```

Both the tests and benchmark therefore execute inside the ARM64 environment.

---

## 19. ARM64 Test Validation

The project currently contains **34 tests** (`tests/test_compression.py`, `tests/test_decomposition.py`, `tests/test_layers.py`, `tests/test_linear.py`), plus a separate 28-test suite (`tests/test_fused_backend.py`) covering the fused-kernel investigation described in section 39 below. The core test suite covers:

- SVD decomposition
- Low-rank reconstruction
- Invalid SVD rank handling
- Energy-based rank selection
- Invalid energy handling
- Linear-layer analysis
- TensorFoldLinear forward and backward propagation
- Tensor shape validation
- TensorFoldLinear parameter counting and reduction
- Invalid TensorFoldLinear rank handling
- Conversion from `nn.Linear`
- Bias handling
- Automatic model compression and original-model preservation

The test suite passes locally and also passes in the ARM64 GitHub Actions workflow. A successful test run reports: `34 passed`. (An earlier version of this document reported 13 tests; that number was accurate for an earlier point in the project's development and is corrected here.)

---

## 20. Benchmark Model Parameters

| Metric | Value |
| --- | --- |
| Dense parameters | 535,818 |
| TensorFold parameters (90% energy) | 279,940 |
| Parameter reduction | 47.75% |
| Parameters removed | 255,878 |

---

## 21. ARM64 Benchmark Results

| Batch Size | Model | Mean (ms) | Median (ms) | Std (ms) | Speedup |
| --- | --- | --- | --- | --- | --- |
| 1 | Dense | 0.1244 | 0.1244 | 0.0002 | – |
| 1 | TensorFold 90% | 0.1111 | 0.1109 | 0.0012 | 1.119× |
| 16 | Dense | 0.6872 | 0.6909 | 0.0138 | – |
| 16 | TensorFold 90% | 0.4862 | 0.4853 | 0.0022 | 1.413× |
| 32 | Dense | 0.7642 | 0.7602 | 0.0096 | – |
| 32 | TensorFold 90% | 0.6319 | 0.6317 | 0.0011 | 1.209× |
| 64 | Dense | 0.9771 | 0.9737 | 0.0150 | – |
| 64 | TensorFold 90% | 1.0739 | 1.0737 | 0.0020 | 0.910× |
| 256 | Dense | 2.0406 | 2.0376 | 0.0100 | – |
| 256 | TensorFold 90% | 3.4052 | 3.4045 | 0.0048 | 0.599× |

(An earlier version of this document reported a different table in which speedup increased monotonically to 1.664× at batch 256. That table did not match `benchmarks/results/arm64_results.md` and has been replaced with the correct, verified numbers above, which have since been independently reproduced by `benchmarks/bench_three_way.py` — see section 39.)

---

## 22. ARM64 Speedup Analysis

TensorFold is faster than dense at small-to-moderate batch sizes and slower at large batch sizes:

| Batch Size | Speedup | Result |
| --- | --- | --- |
| 1 | 1.119× | Faster |
| 16 | 1.413× | Faster |
| 32 | 1.209× | Faster |
| 64 | 0.910× | Slower |
| 256 | 0.599× | Slower |

The highest measured speedup was **1.413×**, at batch size 16. See section 39 for an investigation into why TensorFold is slower than dense at batch 64/256, and why that regression could not be fixed by removing kernel-dispatch overhead.

---

## 23. Batch Size 16 (Best Case)

At batch size 16, Dense = 0.6872 ms and TensorFold = 0.4862 ms.

$$\frac{0.6872 - 0.4862}{0.6872} \times 100 \approx 29.2\%$$

TensorFold reduced mean inference latency by approximately **29.2%** at the best-performing batch size (16).

---

## 24. Parameter Reduction vs Speedup

The benchmark demonstrates two different effects:

**Parameter compression:** 535,818 → 279,940 → **47.75% parameter reduction** (holds regardless of batch size)

**Inference speed:** ranges from **1.413× faster** (batch 16) to **0.599× slower** (batch 256)

Parameter reduction and speedup are related, but they are not the same metric. Reducing the number of parameters does not guarantee a specific inference speedup on every CPU, or even a speedup at all at every batch size.

---

## 25. Accuracy Results

| Metric | Value |
| --- | --- |
| Dense accuracy | 97.79% |
| TensorFold accuracy | 97.06% |
| Difference | -0.73 pp |

The tested model therefore experienced a relatively small accuracy change while achieving substantial parameter compression. These accuracy results apply specifically to the benchmark model.

---

## 26. Accuracy and SVD Energy

The SVD energy target and classification accuracy are separate concepts.

The 90% energy target means that the selected singular components preserve at least 90% of the matrix's singular-value energy. It does **not** mean 90% classification accuracy or 90% prediction preservation.

Therefore, model accuracy must always be measured independently after compression.

---

## 27. Benchmark Summary

| Metric | Result |
| --- | --- |
| Model | MNIST MLP |
| Architecture | 784 → 512 → 256 → 10 |
| Compression | Truncated SVD |
| Energy target | 90% |
| Dense parameters | 535,818 |
| TensorFold parameters | 279,940 |
| Parameter reduction | 47.75% |
| Dense accuracy | 97.79% |
| TensorFold accuracy | 97.06% |
| Accuracy difference | -0.73 pp |
| Maximum speedup | 1.413× |
| Maximum speedup batch | 16 |
| Slowest result | 0.599× (slower than dense) at batch 256 |
| Batch 16 Dense latency | 0.6872 ms |
| Batch 16 TensorFold latency | 0.4862 ms |
| Approx. latency reduction (batch 16) | 29.2% |
| CPU threads | 1 |
| Warmup | 100 |
| Iterations | 500 |
| Repeats | 10 |
| ARM64 runner | ubuntu-24.04-arm |
| ARM64 tests | 34 passed (+ 28 for the fused-kernel investigation, section 39) |

---

## 28. How the Benchmark Is Implemented

The benchmark follows this sequence:

```
Load PyTorch model
        │
        ▼
Load trained MNIST weights
        │
        ▼
Set model to evaluation mode
        │
        ▼
Create TensorFold compressed model
        │
        ▼
Count dense parameters
        │
        ▼
Count TensorFold parameters
        │
        ▼
Calculate parameter reduction
        │
        ▼
Benchmark both models
        │
        ▼
Calculate speedup
        │
        ▼
Print results
```

The original model is not modified by the compression process. TensorFold creates a compressed copy and replaces beneficial Linear layers inside that copy.

---

## 29. Benchmark Command

The benchmark can be executed from the repository root using:

```bash
python examples/benchmark_tensorfold.py
```

The benchmark requires:

- PyTorch
- TensorFold-arm
- `mnist_mlp.pt`

The model file must contain the trained state dictionary expected by the benchmark.

---

## 30. Running Tests

Run the complete test suite with:

```bash
pytest tests -v
```

Expected result: `34 passed`. Running `pytest tests/test_fused_backend.py -v` separately covers the fused-kernel investigation (section 39) and reports `28 passed`.

The ARM64 GitHub Actions workflow executes the same test suite.

---

## 31. Running on ARM64

The ARM64 workflow is responsible for validating the project on the target architecture.

The workflow's `runs-on: ubuntu-24.04-arm` ensures that the job runs on an ARM64 runner. The workflow then installs the project and executes:

```bash
pytest tests -v
python examples/benchmark_tensorfold.py
```

This means both correctness tests and performance measurements are executed in the ARM64 environment.

---

## 32. Why ARM64 Validation Matters

TensorFold-arm is an Arm-focused optimization project. Performance measured only on an x86 machine would not be sufficient to demonstrate behavior on Arm CPUs.

The ARM64 workflow provides:

```
Native ARM64 runner
        ↓
ARM64 Python
        ↓
ARM64 PyTorch
        ↓
TensorFold-arm
        ↓
Tests
        ↓
Benchmark
```

This gives direct validation on the intended architecture.

---

## 33. Benchmark Limitations

The reported benchmark results are specific to the tested configuration. Performance can vary with:

- ARM CPU model
- CPU frequency
- CPU cache
- Memory bandwidth
- PyTorch version
- BLAS backend
- Matrix dimensions
- Batch size
- Number of threads
- Selected SVD rank
- Operating system
- System load

Therefore, the results should not be interpreted as a universal performance guarantee for every ARM64 processor.

---

## 34. Why Multiple Batch Sizes Are Tested

Different batch sizes exercise the CPU differently. The benchmark uses batch sizes 1, 16, 32, 64, and 256. This allows the effect of TensorFold compression to be observed across different workloads.

The measured speedup is 1.119× at batch size 1, peaks at 1.413× at batch size 16, and declines to 0.599× (slower than dense) at batch size 256. This shows that the benefit of the low-rank representation can depend strongly on workload size, and is not simply monotonic with batch size.

---

## 35. Mathematical Summary

For a dense matrix $W \in \mathbb{R}^{M \times N}$, the dense parameter count is:

$$P_{dense} = MN$$

After rank-r factorization, $W \approx AB$ where $A \in \mathbb{R}^{M \times r}$ and $B \in \mathbb{R}^{r \times N}$, the parameter count becomes:

$$P_{tensorfold} = Mr + rN = r(M + N)$$

Compression occurs when:

$$r(M + N) < MN$$

The percentage reduction is:

$$\text{Reduction} = \left(1 - \frac{P_{tensorfold}}{P_{dense}}\right) \times 100$$

This is the mathematical foundation of TensorFold's parameter compression.

---

## 36. Overall Results

The benchmark model contains 535,818 dense parameters. After TensorFold compression, 279,940 parameters remain — a **47.75% parameter reduction**.

The tested MNIST accuracy changes from 97.79% to 97.06%, a difference of -0.73 percentage points.

On the ARM64 benchmark, the maximum measured speedup is **1.413×** at batch size 16. TensorFold is faster than dense at batch sizes 1–32 and slower at batch sizes 64/256; see section 39 for an investigation into why, and why this project is scoped to small-to-moderate batch (on-device / mobile-inference-realistic) workloads as a result.

---

## 37. Final Status

The TensorFold-arm benchmark pipeline is validated through:

| Stage | Result |
| --- | --- |
| Local tests | PASS |
| ARM64 tests | PASS |
| ARM64 benchmark | PASS |
| Fused-kernel correctness tests | PASS |
| Three-way benchmark (dense / torch / fused) | PASS |
| Model loading | PASS |
| TensorFold compression | PASS |

**Headline results:**

- 47.75% parameter reduction (holds at every batch size)
- Up to 1.413× ARM64 CPU inference speedup at batch sizes 1–32
- Slower than dense at batch sizes 64/256, and this was investigated (not just measured — see section 39) rather than left unexplained
- 97.79% → 97.06% MNIST accuracy

The benchmark demonstrates that the TensorFold low-rank representation substantially reduces parameter count on the tested ARM64 environment, and improves inference latency specifically in the small-to-moderate batch regime that matches on-device/mobile inference.

---

## 38. Conclusion

TensorFold-arm provides an automatic low-rank compression pipeline for PyTorch Linear layers. The complete process is:

```
Dense Linear
     │
     ▼
SVD
     │
     ▼
Energy-based rank selection
     │
     ▼
Parameter-count analysis
     │
     ▼
Replace only if beneficial
     │
     ▼
TensorFoldLinear
     │
     ▼
ARM64 inference
```

For the tested MNIST MLP, this resulted in **47.75% parameter reduction**, and the ARM64 benchmark measured up to **1.413× inference speedup** at batch sizes 1–32, while the tested accuracy changed by **-0.73 percentage points**. At batch sizes 64/256 TensorFold is slower than dense; we investigated this with a custom fused kernel (section 39) and determined the cause is not kernel-dispatch overhead but a more fundamental property of the rank/batch-size trade-off.

The results demonstrate the potential of SVD-based low-rank factorization for reducing model size and improving CPU inference performance on ARM64 platforms, particularly for on-device / small-batch inference workloads.

> **Note:** The results are measurements from the documented benchmark configuration and should not be treated as universal performance guarantees for all models or ARM64 processors.

---

## 39. Investigation: A Fused CPU Kernel for the Large-Batch Regression

We hypothesized the batch 64/256 regression above was caused by implementation overhead — two ATen kernel dispatches per layer instead of one, plus materializing the intermediate `[batch, rank]` tensor to memory between them — and built a custom fused CPU kernel (`tensorfold/csrc/fused_linear.cpp`) to test that hypothesis directly. It computes `(X @ A) @ B + bias` in a single pass, one dispatch, with the intermediate kept in a small per-row buffer that never touches main memory. It's exposed as `TensorFoldLinear(backend="fused")` and is correctness-tested against the original path across 5 shapes × 5 batch sizes, with/without bias (`tests/test_fused_backend.py`, 28/28 passing).

Running the fused kernel through the same three-way benchmark (`benchmarks/bench_three_way.py`) falsified the hypothesis: `Torch/Dense` and `Fused/Dense` speedups track each other within ~2% at every batch size, including 64/256 where the regression is largest. Cutting the dispatch count from 6 calls per forward pass to 3 (matching dense exactly) produced no measurable change. The likely explanation: the fused kernel processes batch rows independently, giving up the batched-GEMM data reuse that both dense and the original two-matmul path benefit from, which roughly cancels out the gain from fewer dispatches. This points to the real bottleneck being the FLOP/batch-size trade-off inherent to the selected rank, not an implementation inefficiency.

We kept the fused kernel and its tests in the repository as documented, working, tested code, but do not claim a latency benefit from it. `backend="torch"` (the default) remains recommended. Full writeup, including the three-way results table, is in `RESULTS.md` at the repository root.
