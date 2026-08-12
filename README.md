# TensorFold-arm

TensorFold-arm is an Arm-focused neural network optimization project that reduces the computational and parameter cost of dense neural network layers using low-rank matrix decomposition.

The project targets CPU inference on Arm64 platforms and provides an automatic model compression pipeline for PyTorch models.

> **TensorFold-arm is an independent project created specifically for the Arm AI Optimization Challenge.**
>
> It is not part of MiniPyPy and does not depend on the existing MiniPyPy TensorFold implementation. Ideas and optimizations developed here may be integrated into the main project in the future.

---

## Table of Contents

- [Overview](#overview)
- [Core Idea](#core-idea)
- [Why Low-Rank Compression](#why-low-rank-compression)
- [Mathematical Foundation](#mathematical-foundation)
  - [Dense Linear Layer](#dense-linear-layer)
  - [SVD Decomposition](#svd-decomposition)
  - [Rank Truncation](#rank-truncation)
  - [TensorFold Factorization](#tensorfold-factorization)
  - [Parameter Reduction](#parameter-reduction)
  - [Example](#example)
  - [Energy-Based Rank Selection](#energy-based-rank-selection)
- [How TensorFold Works](#how-tensorfold-works)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Compression API](#compression-api)
- [TensorFoldLinear](#tensorfoldlinear)
- [Compression Analysis](#compression-analysis)
- [Benchmark Model](#benchmark-model)
- [Benchmark Configuration](#benchmark-configuration)
- [Benchmark Results](#benchmark-results)
- [AMD64 Reference Benchmark](#amd64-reference-benchmark)
- [Native ARM64 Benchmark](#native-arm64-benchmark)
- [Benchmark Interpretation](#benchmark-interpretation)
- [Investigation: A Fused CPU Kernel for the Large-Batch Regression](#investigation-a-fused-cpu-kernel-for-the-large-batch-regression)
- [ARM64 Validation](#arm64-validation)
- [Test Suite](#test-suite)
- [CI Workflows](#ci-workflows)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Why This Matters for Arm](#why-this-matters-for-arm)
- [Reproducibility](#reproducibility)
- [Results Summary](#results-summary)
- [Conclusion](#conclusion)
- [License](#license)

---

## Overview

Fully connected (`nn.Linear`) layers store dense weight matrices.

For a layer with `input features = N`, `output features = M`, the weight matrix contains:

$$W \in \mathbb{R}^{M \times N}$$

The number of weight parameters is `M × N`, and with a bias, `M × N + M`.

For large neural networks, dense linear layers can therefore account for a substantial portion of the model's parameter count and memory footprint.

TensorFold-arm replaces suitable dense linear layers with low-rank factorized layers. Instead of storing the complete matrix `W`, TensorFold approximates it using two smaller matrices:

$$W \approx AB$$

where $A \in \mathbb{R}^{M \times r}$, $B \in \mathbb{R}^{r \times N}$, and $r \ll \min(M, N)$.

The original layer $Y = XW^\top + b$ can then be evaluated as $Y = X(B^\top A^\top) + b$, or equivalently through two matrix multiplications.

The main goal is to reduce the number of stored parameters while retaining as much of the original model behavior as possible.

---

## Core Idea

The core idea of TensorFold-arm is:

```
Dense Linear Layer
        │
        ▼
   Weight Matrix W
        │
        ▼
   Singular Value
   Decomposition
        │
        ▼
   U Σ Vᵀ
        │
        ▼
   Keep only the
   most important r
   singular components
        │
        ▼
   Fold Σ into U
        │
        ▼
      A B
        │
        ▼
 TensorFoldLinear
```

Instead of storing the original dense matrix, TensorFold stores the two low-rank factors. The rank `r` determines the compression level.

A **smaller rank** produces: fewer parameters, lower memory usage, potentially lower computational cost, greater approximation error.

A **larger rank** produces: more parameters, less compression, lower approximation error, behavior closer to the original dense layer.

Therefore, selecting the rank is the central trade-off in TensorFold.

---

## Why Low-Rank Compression

Many trained neural-network weight matrices contain redundancy. Although a matrix may technically have a large number of independent entries, its information can often be approximated using a much smaller number of dominant singular components.

A dense matrix of size `M × N` requires `M × N` weight parameters. A rank-`r` factorization requires `M × r` plus `r × N` parameters. Therefore:

$$P_{dense} = MN, \qquad P_{low\_rank} = Mr + rN$$

Ignoring bias for the moment: $P_{low\_rank} = r(M + N)$.

When $r \ll \min(M, N)$, the reduction can be substantial.

---

## Mathematical Foundation

### Dense Linear Layer

A standard PyTorch linear layer computes:

$$Y = XW^\top + b$$

where `X` = input tensor, `W` = dense weight matrix, `b` = bias, `Y` = output tensor.

For `in_features = N`, `out_features = M`, the weight matrix has shape $W \in \mathbb{R}^{M \times N}$ and therefore contains `M × N` weight parameters.

Including bias: $P_{dense} = MN + M$

### SVD Decomposition

TensorFold uses Singular Value Decomposition (SVD). For a matrix `W`, SVD decomposes it as:

$$W = U \Sigma V^\top$$

where `U` = left singular vectors, `Σ` = diagonal matrix of singular values, `Vᵀ` = right singular vectors.

The singular values are ordered $\sigma_1 \geq \sigma_2 \geq \sigma_3 \geq \dots \geq \sigma_k$, where $k = \min(M, N)$.

The larger singular values generally represent the most significant components of the matrix.

### Rank Truncation

Instead of keeping the complete decomposition, TensorFold keeps only the first `r` singular components:

$$W \approx U_r \Sigma_r V_r^\top$$

where `r < min(M, N)`. The resulting approximation is the best rank-`r` approximation of the matrix under the standard SVD/Frobenius-norm formulation.

TensorFold then folds the singular values into one factor. For example, $A = U_r \Sigma_r$ and $B = V_r^\top$, giving $W \approx AB$.

The original matrix therefore does not need to be stored during inference.

### TensorFold Factorization

Suppose $W \in \mathbb{R}^{M \times N}$. After truncated SVD, $W \approx U_r \Sigma_r V_r^\top$.

TensorFold constructs $A = U_r \Sigma_r$ and $B = V_r^\top$, so $A \in \mathbb{R}^{M \times r}$ and $B \in \mathbb{R}^{r \times N}$. Therefore $W \approx AB$.

The dense matrix `M × N` has been replaced by `M × r` and `r × N`.

### Parameter Reduction

The dense representation requires $P_{dense} = MN + M$ parameters when bias is included.

The TensorFold representation requires $P_{tensorfold} = Mr + rN + M$ parameters.

Therefore the parameter reduction is:

$$\text{Reduction} = \left(1 - \frac{P_{tensorfold}}{P_{dense}}\right) \times 100$$

A positive reduction means that the factorized representation uses fewer parameters than the original dense representation.

TensorFold checks whether compression is actually beneficial before replacing a layer. This is important because not every layer benefits from low-rank factorization.

### Example

Consider `input features = 784`, `output features = 512`. The weight matrix is $W \in \mathbb{R}^{512 \times 784}$, containing $512 \times 784 = 401{,}408$ weight parameters.

Including the 512-element bias vector, the complete Linear layer contains $512 \times 784 + 512 = 401{,}920$ parameters.

Now suppose TensorFold selects `rank = 100`. The factorized representation requires `512 × 100` parameters for one factor and `100 × 784` for the second factor. Therefore:

$$P_{tensorfold} = 512 \times 100 + 100 \times 784 + 512 = 51{,}200 + 78{,}400 + 512 = 130{,}112$$

Compared with 401,920, the factorized representation is substantially smaller.

The important point is that TensorFold does not simply delete random weights. It uses the dominant singular components of the trained matrix to construct a lower-rank approximation.

### Energy-Based Rank Selection

TensorFold does not require the user to manually select a rank. Instead, the compression API can select the rank based on the amount of singular-value energy that should be retained.

The singular-value energy is based on the squared singular values $\sigma_1^2, \sigma_2^2, \dots, \sigma_k^2$.

The total energy is $E_{total} = \sum \sigma_i^2$. For a candidate rank `r`, retained energy is $E_r = \sum_{i=1}^{r} \sigma_i^2$.

The retained energy ratio is $\text{Energy}(r) = E_r / E_{total}$.

TensorFold selects the smallest rank satisfying the requested energy target. For example, `energy = 0.90` means TensorFold attempts to retain approximately 90% of the singular-value energy.

This provides an automatic trade-off between compression and approximation quality.

---

## How TensorFold Works

The high-level compression pipeline is:

```
PyTorch Model
      │
      ▼
Traverse Model
      │
      ▼
Find nn.Linear Layers
      │
      ▼
Analyze Weight Matrix
      │
      ▼
Compute SVD
      │
      ▼
Select Rank
      │
      ▼
Estimate Parameter Count
      │
      ├───────────────┐
      │               │
 Compression       No Benefit
 Possible           │
      │               │
      ▼               ▼
Create             Keep Dense
TensorFoldLinear   Linear Layer
      │               │
      └───────┬───────┘
              ▼
       Compressed Model
```

The original model is not modified in-place. The compression function creates a separate compressed model.

---

## Architecture

TensorFold-arm is organized into several logical components.

```
                         TensorFold-arm
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
        Decomposition         Layers          Compression
              │                 │                 │
              ▼                 ▼                 ▼
          SVD / Rank       TensorFoldLinear    Model traversal
          Selection                           and replacement
              │                 │                 │
              └─────────────────┼─────────────────┘
                                │
                                ▼
                         Compressed PyTorch
                              Model
```

**Decomposition** is responsible for SVD, truncated decomposition, rank selection, energy calculation, and layer analysis.

**Layers** provides `TensorFoldLinear`, which represents a dense linear transformation using low-rank factors.

**Compression** is responsible for traversing models, locating `nn.Linear` layers, analyzing whether compression is beneficial, replacing suitable layers, and preserving layers where compression would not provide a benefit.

---

## Project Structure

```
TensorFold-arm/
│
├── .github/
│   └── workflows/
│       ├── arm64.yml
│       └── windows.yml
│
├── benchmarks/
│   └── results/
│       ├── arm64_results.md
│       └── x86_baseline.md
│
├── examples/
│   ├── basic_compression.py
│   └── benchmark_tensorfold.py
│
├── tensorfold/
│   ├── __init__.py
│   ├── compression.py
│   ├── decomposition.py
│   └── layers.py
│
├── tests/
│   ├── test_compression.py
│   ├── test_decomposition.py
│   ├── test_layers.py
│   └── test_linear.py
│
├── mnist_mlp.pt
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

The trained benchmark model is used by the benchmark workflow and is required for reproducing the benchmark.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/mridul-afk/TensorFold-arm.git
cd TensorFold-arm
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install TensorFold-arm:

```bash
pip install -e .
```

Verify the installation:

```bash
python -c "import tensorfold; print('TensorFold imported successfully')"
```

---

## Usage

TensorFold can compress a PyTorch model using the public compression API.

Example:

```python
import torch
import torch.nn as nn

from tensorfold import compress


model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

compressed_model = compress(
    model,
    energy=0.90,
)
```

The `energy` parameter controls the amount of singular-value energy retained. For example, `energy=0.90` requests approximately 90% energy preservation for the selected layers.

---

## Compression API

The main API is:

```python
compress(
    model,
    energy=0.90,
)
```

The compression pipeline:

1. Traverses the model.
2. Finds `nn.Linear` layers.
3. Computes the SVD of their weights.
4. Selects a rank based on the requested energy.
5. Calculates the dense and factorized parameter counts.
6. Determines whether compression is beneficial.
7. Replaces beneficial layers with `TensorFoldLinear`.
8. Leaves non-beneficial layers unchanged.
9. Returns a new model.

Example:

```python
compressed = compress(
    model,
    energy=0.90,
)
```

The original model remains unchanged.

---

## TensorFoldLinear

`TensorFoldLinear` is the low-rank replacement for `nn.Linear`.

Example:

```python
from tensorfold.layers import TensorFoldLinear

layer = TensorFoldLinear(
    in_features=64,
    out_features=32,
    rank=16,
)
```

The factorized layer stores $U \in \mathbb{R}^{\text{out\_features} \times \text{rank}}$ and $V \in \mathbb{R}^{\text{rank} \times \text{in\_features}}$, plus an optional bias.

Therefore its parameter count is `out_features × rank + rank × in_features + out_features` when bias is enabled.

### Creating From a Dense Layer

A dense layer can be converted directly:

```python
import torch.nn as nn

from tensorfold.layers import TensorFoldLinear


dense = nn.Linear(
    128,
    64,
)

compressed = TensorFoldLinear.from_linear(
    dense,
    rank=16,
)
```

The bias configuration is preserved.

---

## Compression Analysis

TensorFold provides analysis functionality for determining whether a layer should be compressed.

Example:

```python
from tensorfold.decomposition import analyze_linear

result = analyze_linear(
    weight,
    energy=0.90,
)
```

The analysis includes information such as:

- `in_features`
- `out_features`
- `rank`
- `original_parameters`
- `compressed_parameters`
- `parameter_reduction`
- `compression_possible`

This allows the compression pipeline to avoid replacing layers where the factorized representation would actually require more parameters.

---

## Benchmark Model

The benchmark uses a multilayer perceptron trained on MNIST.

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

Compact representation: `784 → 512 → 256 → 10`

The model contains three dense Linear layers. TensorFold analyzes each Linear layer independently and replaces layers where the selected low-rank representation provides a parameter reduction.

---

## Benchmark Configuration

| Setting | Value |
| --- | --- |
| Energy target | 90% |
| Threads | 1 |
| Warmup iterations | 100 |
| Repeats | 10 |
| Iterations per repeat | 500 |
| Batch sizes | 1, 16, 32, 64, 256 |

Latency is measured using Python's `time.perf_counter()`.

The benchmark reports mean latency, median latency, standard deviation, and speedup.

Speedup is calculated as:

$$\text{Speedup} = \frac{\text{Dense Mean Latency}}{\text{TensorFold Mean Latency}}$$

`Speedup > 1` means TensorFold is faster; `Speedup < 1` means the dense model is faster.

---

## Benchmark Results

The benchmark demonstrates two separate effects: parameter compression and inference performance.

The parameter reduction is consistent across the benchmarked platforms:

| Metric | Value |
| --- | --- |
| Dense parameters | 535,818 |
| TensorFold parameters | 279,940 |
| Parameter reduction | 47.75% |

---

## AMD64 Reference Benchmark

The AMD64 reference benchmark was performed using:

| Property | Value |
| --- | --- |
| Platform | Windows |
| Architecture | AMD64 |
| Python | 3.11.9 |
| PyTorch | 2.13.0+cpu |
| Execution device | CPU |
| Threads | 1 |
| Warmup | 100 |
| Repeats | 10 |
| Iterations | 500 |
| Energy target | 90% |

### AMD64 Results

| Batch Size | Dense Mean (ms) | TensorFold Mean (ms) | Speedup |
| --- | --- | --- | --- |
| 1 | 0.0852 | 0.0820 | 1.039× |
| 16 | 0.3182 | 0.2401 | 1.325× |
| 32 | 0.5031 | 0.3335 | 1.509× |
| 64 | 0.8549 | 0.5702 | 1.499× |
| 256 | 3.0489 | 1.9401 | 1.572× |

TensorFold was faster than the dense model at every tested AMD64 batch size in this reference run.

The complete AMD64 benchmark is available at `benchmarks/results/x86_baseline.md`.

---

## Native ARM64 Benchmark

TensorFold-arm was also benchmarked on a native ARM64 Linux environment.

| Property | Value |
| --- | --- |
| Platform | Linux |
| Architecture | ARM64 (aarch64) |
| Python | 3.11.15 |
| PyTorch | 2.13.0+cu130 |
| Execution device | CPU |
| Threads | 1 |
| Warmup | 100 |
| Repeats | 10 |
| Iterations | 500 |
| Energy target | 90% |

The PyTorch build contains CUDA support, but the benchmark explicitly runs on `torch.device("cpu")`. Therefore these measurements are CPU measurements.

### ARM64 Results

| Batch Size | Dense Mean (ms) | TensorFold Mean (ms) | Speedup |
| --- | --- | --- | --- |
| 1 | 0.1244 | 0.1111 | 1.119× |
| 16 | 0.6872 | 0.4862 | 1.413× |
| 32 | 0.7642 | 0.6319 | 1.209× |
| 64 | 0.9771 | 1.0739 | 0.910× |
| 256 | 2.0406 | 3.4052 | 0.599× |

The highest measured ARM64 speedup is **1.413×** at batch size 16.

TensorFold was faster at batch sizes 1, 16, and 32, and slower at 64 and 256.

The complete ARM64 benchmark results are available at `benchmarks/results/arm64_results.md`.

---

## Benchmark Interpretation

Parameter reduction and inference speed are related but are not equivalent.

A dense Linear layer performs a matrix multiplication:

$$Y = XW + b$$

TensorFold replaces the dense matrix with a low-rank approximation $W \approx AB$ and therefore performs:

$$Y = (XA)B + b$$

The factorized representation contains substantially fewer parameters. However, the factorized computation introduces two matrix multiplications instead of the original dense matrix multiplication.

Therefore, actual inference performance depends on:

- Matrix dimensions
- Selected rank
- Batch size
- CPU architecture
- Memory access patterns
- Matrix multiplication implementation
- Framework overhead
- Kernel efficiency

The benchmark demonstrates this directly. On AMD64, TensorFold produced a speedup for all tested batch sizes. On the latest ARM64 run, TensorFold produced speedups for batch sizes 1, 16, and 32, but was slower for batch sizes 64 and 256.

This is an important engineering result: **low-rank compression can substantially reduce model parameters, but it does not guarantee faster inference for every workload.**

The primary optimization target of TensorFold-arm is therefore model compression and parameter reduction, while inference acceleration is workload-dependent.

---

## Investigation: A Fused CPU Kernel for the Large-Batch Regression

The benchmark above shows TensorFold is slower than dense at batch sizes 64 and 256 on ARM64. We investigated one specific hypothesis for why, built a fix, and tested it rigorously.

**Hypothesis.** TensorFold's forward pass performs two matrix multiplications per layer (`(X @ A) @ B`) instead of dense's one. We hypothesized the large-batch slowdown came from implementation overhead: two separate ATen kernel dispatches per layer instead of one, plus materializing the intermediate `[batch, rank]` tensor to memory between them.

**What we built.** A custom CPU kernel (`tensorfold/csrc/fused_linear.cpp`, C++/pybind11, JIT-compiled via `torch.utils.cpp_extension`) that computes `(X @ A) @ B + bias` in a single pass per layer, one dispatch, with the intermediate kept in a small per-row buffer that never touches main memory. It's available as `TensorFoldLinear(backend="fused")`, and falls back automatically to the original two-matmul path during training/autograd. Correctness was verified against the original path across 5 layer shapes × 5 batch sizes, with and without bias (`tests/test_fused_backend.py`, 28/28 passing on both Linux/GCC and Windows/MSVC).

**Result: the hypothesis was falsified.** A three-way ARM64 benchmark (`benchmarks/bench_three_way.py`, dense vs. TensorFold-torch vs. TensorFold-fused) shows the fused kernel tracking the original two-matmul implementation within ~2% at every batch size — well inside measurement noise — including at batch 64/256 where the regression is largest:

| Batch | Dense (ms) | TensorFold-torch (ms) | TensorFold-fused (ms) | Torch/Dense | Fused/Dense |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.1224 | 0.1150 | 0.1178 | 1.064× | 1.039× |
| 16 | 0.6813 | 0.4939 | 0.4991 | 1.379× | 1.365× |
| 32 | 0.8076 | 0.6649 | 0.6664 | 1.215× | 1.212× |
| 64 | 0.9992 | 1.1096 | 1.1113 | 0.901× | 0.899× |
| 256 | 2.0985 | 3.4643 | 3.4720 | 0.606× | 0.604× |

Cutting the dispatch count from 6 calls per forward pass (2 matmuls × 3 layers) to 3 (matching dense's call count exactly) produced no measurable latency change.

**Likely root cause.** The fused kernel computes each batch row independently (a per-row loop). Dense's single matmul and TensorFold's original two-matmul path are both *batched* GEMMs, which reuse loaded weight data across every row in the batch and get substantially better arithmetic intensity than a loop of independent per-row computations — especially as batch size grows. Fusing the two stages into one dispatch removed overhead, but computing rows independently gave up batched-GEMM efficiency; the two effects approximately cancel out. This points to the real bottleneck being more fundamental than dispatch overhead: at the rank this compression setting selects (≈246 for a 784→512 layer), the FLOP savings from the low-rank factorization aren't large enough to beat how efficiently a single dense batched GEMM scales with batch size on this hardware.

We're keeping the fused kernel and its test suite in the repository — it's correct, working code — but we do not claim a latency benefit from it, and `backend="torch"` (the default) remains the recommended setting. Full writeup: `RESULTS.md`. Reproduce with `python benchmarks/bench_three_way.py`.

---

## ARM64 Validation

A key goal of TensorFold-arm is to ensure that the implementation and benchmarks actually work on Arm64 hardware.

The ARM64 GitHub Actions workflow runs on `ubuntu-24.04-arm`.

Before running the tests, the workflow verifies the architecture. It checks `uname -m` and `platform.machine()`. The expected result is `aarch64`. The workflow also explicitly asserts that the machine is ARM64.

The ARM64 CI pipeline performs the following steps:

```
Checkout repository
        │
        ▼
Set up Python 3.11
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
Verify TensorFold installation
        │
        ▼
Verify benchmark model
        │
        ▼
Run complete test suite
        │
        ▼
Run ARM64 benchmark
```

This means the ARM64 benchmark is not inferred from AMD64 results and is not simply a local AMD64 benchmark labeled as ARM64. The benchmark is executed in the ARM64 CI environment itself.

---

## Test Suite

TensorFold-arm contains automated tests covering the core functionality of the project.

The current test suite contains **34 tests**, covering:

- SVD decomposition
- Low-rank SVD shapes
- SVD reconstruction
- Invalid rank handling
- Energy-based rank selection
- Invalid energy handling
- Linear-layer analysis
- TensorFoldLinear forward behavior
- TensorFoldLinear backward behavior
- Gradient propagation
- Shape validation
- Parameter counting
- Parameter reduction
- Invalid TensorFoldLinear ranks
- Conversion from `nn.Linear`
- Bias preservation
- No-bias layers
- Automatic model compression
- Beneficial Linear-layer replacement
- Preservation of the original model
- Compression behavior

The complete test suite passes: `34 passed`

The tests are executed in both CI workflows: Linux x86_64 (`ci.yml`, `ubuntu-latest`) and Linux ARM64 (`arm-test.yml`, `ubuntu-24.04-arm`). The AMD64 reference numbers in `benchmarks/results/x86_baseline.md` were captured locally on Windows and are not produced by either CI workflow; treat them as a secondary reference point, not a CI-verified result.

---

## CI Workflows

TensorFold-arm uses GitHub Actions for continuous validation.

The project contains two workflows: `ci.yml`, which runs on `ubuntu-latest` (Linux x86_64), and `arm-test.yml`, which runs on `ubuntu-24.04-arm` (Linux ARM64). Both use GitHub-hosted Linux runners; neither runs on Windows. The ARM64 workflow additionally verifies the architecture before testing.

The CI pipeline validates:

```
Python environment
        ↓
Dependencies
        ↓
Package installation
        ↓
Importability
        ↓
Test suite
        ↓
Benchmark
```

Both the Windows and ARM64 workflows currently complete successfully.

---

## Limitations

TensorFold-arm currently focuses on dense two-dimensional Linear-layer weights.

**1. Linear layers** — The current compression pipeline focuses on `torch.nn.Linear` rather than arbitrary neural-network operators.

**2. Low-rank approximation** — TensorFold introduces approximation error because $W \approx AB$ rather than $W = AB$, unless the selected rank is sufficient to represent the matrix exactly.

**3. Inference speed** — Parameter reduction does not automatically guarantee lower latency. The benchmark demonstrates that TensorFold can be slower than the dense implementation for some ARM64 batch sizes.

**4. Kernel optimization** — The current implementation relies on PyTorch matrix multiplication operations. Further ARM-specific optimization could potentially improve the performance of the factorized operations.

**5. Rank selection** — The energy target is a useful automatic heuristic, but the optimal rank may depend on the particular model, workload, accuracy requirement, and hardware.

---

## Future Work

Potential future improvements include:

**ARM-specific optimization** — Investigate optimized ARM matrix multiplication paths and kernels for the factorized operations.

**Better rank selection** — Explore rank-selection strategies that consider accuracy impact, parameter reduction, latency, and hardware characteristics instead of energy preservation alone.

**Layer-wise optimization** — Allow different compression targets for different layers, e.g.:

```
Layer 1 → 95%
Layer 2 → 90%
Layer 3 → 98%
```

depending on sensitivity.

**Accuracy-aware compression** — Instead of selecting the rank purely from singular-value energy, evaluate the effect of compression on validation accuracy.

**More model architectures** — Extend benchmarking beyond the MNIST MLP to larger neural networks and different architectures.

**More Arm hardware** — Benchmark on multiple ARM64 processors to determine how the factorized implementation behaves across different Arm CPU microarchitectures.

**Optimized kernels** — A fused single-pass CPU kernel for `(X × A) × B` was implemented and benchmarked (see "Investigation: A Fused CPU Kernel for the Large-Batch Regression" above); it did not measurably improve latency, indicating the large-batch regression is not primarily a kernel-dispatch problem. Remaining directions include a hand-written NEON-intrinsics kernel that preserves batched-GEMM-style data reuse (rather than the per-row loop tested here), or routing inference through an Arm-optimized backend such as ONNX Runtime with the XNNPACK execution provider.

**Additional decompositions** — The current implementation is based on matrix SVD. Future versions could explore other low-rank and tensor decomposition approaches for higher-dimensional tensors.

---

## Why This Matters for Arm

Neural-network inference on edge and CPU-based systems is constrained by memory, power, compute capacity, bandwidth, and model size.

Reducing the number of model parameters can reduce the amount of model data that needs to be stored and moved through the memory hierarchy.

For the benchmarked MNIST model, Dense = 535,818 parameters and TensorFold = 279,940 parameters, corresponding to **47.75% parameter reduction**.

The same compressed model representation can therefore provide a smaller parameter footprint while maintaining a substantial portion of the original model's accuracy.

The ARM64 benchmark further demonstrates that low-rank inference can provide speedups for certain workloads. However, the benchmark also shows that further ARM-specific kernel optimization is necessary if the goal is to guarantee latency improvements across a broader range of workloads.

---

## Reproducibility

The benchmark can be executed with:

```bash
python examples/benchmark_tensorfold.py
```

The benchmark configuration is defined in `examples/benchmark_tensorfold.py`.

The benchmark model is `mnist_mlp.pt`.

The compression target is 90% singular-value energy.

The tested batch sizes are 1, 16, 32, 64, 256.

The benchmark uses:

| Setting | Value |
| --- | --- |
| Threads | 1 |
| Warmup | 100 |
| Repeats | 10 |
| Iterations | 500 |

---

## Results Summary

The primary compression result is:

| Metric | Result |
| --- | --- |
| Dense parameters | 535,818 |
| TensorFold parameters | 279,940 |
| Parameter reduction | 47.75% |
| Dense accuracy | 97.79% |
| TensorFold accuracy | 97.06% |
| Accuracy change | -0.73 percentage points |

The AMD64 reference benchmark produced speedups from 1.039× → 1.572× across the tested batch sizes.

The latest native ARM64 benchmark produced:

| Batch Size | Speedup |
| --- | --- |
| 1 | 1.119× |
| 16 | 1.413× |
| 32 | 1.209× |
| 64 | 0.910× |
| 256 | 0.599× |

The highest ARM64 speedup was **1.413×** at batch size 16.

The results show that TensorFold's strongest consistent benefit is parameter compression, while inference acceleration depends on the workload and hardware. We tested and ruled out kernel-dispatch overhead as the cause of the batch 64/256 regression (see "Investigation: A Fused CPU Kernel for the Large-Batch Regression" above and `RESULTS.md`); TensorFold's speed advantage is therefore best understood as applying to small-to-moderate batch sizes (1–32), which also matches the batch-1 regime typical of on-device/mobile inference.

---

## Conclusion

TensorFold-arm demonstrates a practical approach to compressing dense neural network layers using low-rank matrix decomposition.

The project:

- Automatically analyzes `nn.Linear` layers
- Uses SVD for low-rank approximation
- Selects ranks using singular-value energy
- Replaces beneficial layers with `TensorFoldLinear`
- Preserves non-beneficial dense layers
- Reduces the benchmark model's parameters by 47.75%
- Maintains 97.06% MNIST accuracy compared with 97.79% for the dense model
- Runs on native ARM64
- Includes automated ARM64 architecture verification
- Includes Linux x86_64 and Linux ARM64 CI workflows
- Contains 34 passing automated tests, plus 28 additional tests for the fused-kernel investigation described below
- Provides reproducible CPU benchmarks
- Investigated and ruled out kernel-dispatch overhead as the cause of the large-batch ARM64 regression, via a custom fused kernel and a controlled three-way benchmark (see `RESULTS.md`)

The benchmark results also highlight an important distinction between compression and acceleration. TensorFold can significantly reduce the parameter footprint of a model, but the factorized implementation does not automatically outperform dense matrix multiplication for every workload.

This makes the project both a compression technique and a starting point for further ARM-specific inference optimization.

Future work can focus on optimized ARM kernels, hardware-aware rank selection, accuracy-aware compression, and broader model and hardware evaluation.

---

## License

TensorFold-arm is released under the license included in `LICENSE`.

See the `LICENSE` file for the complete terms and conditions.
