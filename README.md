# TensorFold-arm

TensorFold-arm is an Arm-focused neural network optimization project that reduces the parameter and computational cost of dense neural-network `Linear` layers using SVD-based low-rank matrix decomposition.

The project targets CPU inference on **Arm64 platforms** and provides an automatic compression pipeline for PyTorch models.

> **TensorFold-arm is an independent project created specifically for the Arm AI Optimization Challenge.**
>
> It is not part of MiniPyPy and does not depend on the existing MiniPyPy TensorFold implementation.

---

## Table of Contents

- [Overview](#overview)
- [The Problem](#the-problem)
- [Core Idea](#core-idea)
- [Mathematical Foundation](#mathematical-foundation)
- [Parameter Compression](#parameter-compression)
- [SVD-Based Decomposition](#svd-based-decomposition)
- [Energy-Based Rank Selection](#energy-based-rank-selection)
- [Inference Computation](#inference-computation)
- [Architecture](#architecture)
- [How TensorFold Works](#how-tensorfold-works)
- [Code Architecture](#code-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Automatic Model Compression](#automatic-model-compression)
- [Running Tests](#running-tests)
- [Benchmark Methodology](#benchmark-methodology)
- [ARM64 Benchmark Results](#arm64-benchmark-results)
- [x86 Reference Benchmark](#x86-reference-benchmark)
- [Accuracy Results](#accuracy-results)
- [Results Summary](#results-summary)
- [ARM64 Validation](#arm64-validation)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Reproducibility](#reproducibility)
- [License](#license)

---

## Overview

Modern neural networks contain many dense fully connected layers.

A PyTorch `nn.Linear` layer stores its weights as a dense matrix:

$$W \in \mathbb{R}^{M \times N}$$

where `N` = number of input features, `M` = number of output features.

The number of weight parameters is therefore `M × N`.

For large neural networks, these dense matrices can account for a significant portion of the model's parameters and inference computation.

TensorFold-arm exploits approximate low-rank structure in these matrices. Instead of storing the complete matrix `W`, TensorFold approximates it using two smaller matrices:

$$W \approx A \times B$$

where $A \in \mathbb{R}^{M \times r}$, $B \in \mathbb{R}^{r \times N}$, and $r \ll \min(M, N)$.

This changes the number of stored weight parameters from `M × N` to:

$$M \times r + r \times N = r(M + N)$$

When `r` is sufficiently small, this produces a significant reduction in the number of parameters and can also reduce inference computation.

---

## The Problem

A dense Linear layer performs:

$$y = Wx + b$$

For a batch of inputs:

$$Y = XW^\top + b$$

If the weight matrix is large, the model has to store and multiply using the complete dense matrix.

For example, consider input features = 784, output features = 512. The dense weight matrix contains:

$$784 \times 512 = 401{,}408$$

weight parameters.

TensorFold asks a simple question: **can the same transformation be represented accurately enough using a much smaller low-rank representation?**

---

## Core Idea

TensorFold-arm uses Singular Value Decomposition (SVD) to find a low-rank approximation of a Linear layer's weight matrix.

For a matrix `W`:

$$W = U \Sigma V^\top$$

Instead of keeping every singular component, TensorFold keeps only the first `r` components:

$$W \approx U_r \Sigma_r V_r^\top$$

The singular values are ordered from largest to smallest:

$$\sigma_1 \geq \sigma_2 \geq \sigma_3 \geq \dots \geq \sigma_k$$

Large singular values represent the dominant directions of the matrix. If most of the matrix's energy is concentrated in the first few singular values, the remaining components can be discarded.

TensorFold therefore replaces `W` with $U_r \Sigma_r V_r^\top$ and folds the singular values into one factor:

$$A = U_r \Sigma_r, \quad B = V_r^\top$$

giving $W \approx AB$.

---

## Mathematical Foundation

### 1. Dense Matrix

For a Linear layer $W \in \mathbb{R}^{M \times N}$, the number of weight parameters is:

$$P_{dense} = M \times N$$

### 2. SVD

The Singular Value Decomposition of `W` is:

$$W = U \Sigma V^\top$$

where `U` = left singular vectors, `Σ` = singular values, `V` = right singular vectors.

The singular values are sorted in descending order $\sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_k$, where $k = \min(M, N)$.

### 3. Truncated SVD

TensorFold retains only the first `r` singular components:

$$W \approx U_r \Sigma_r V_r^\top$$

where `r < min(M, N)`. The approximation has rank at most `r`. The discarded singular values represent the information removed from the matrix.

The squared Frobenius reconstruction error is:

$$\|W - W_r\|_F^2 = \sigma_{r+1}^2 + \sigma_{r+2}^2 + \dots + \sigma_k^2$$

The truncated SVD provides the best rank-`r` approximation of the matrix under the Frobenius norm.

---

## Parameter Compression

The main reason TensorFold can compress a Linear layer is the difference between the parameter counts.

A dense matrix contains `P_dense = M × N` parameters. The factorized representation, with $A \in \mathbb{R}^{M \times r}$ and $B \in \mathbb{R}^{r \times N}$, contains:

$$P_{tensorfold} = M \times r + r \times N = r(M + N)$$

Therefore compression is beneficial when:

$$r(M + N) < MN \quad \text{or} \quad r < \frac{MN}{M + N}$$

This is the fundamental parameter-compression condition used by TensorFold-arm.

### Worked Example

Consider a Linear layer: `784 → 512`

The dense weight matrix is $W \in \mathbb{R}^{512 \times 784}$.

Dense parameters: $512 \times 784 = 401{,}408$

Now suppose SVD rank selection chooses `r = 246`. The two TensorFold factors contain $A \in \mathbb{R}^{512 \times 246}$ and $B \in \mathbb{R}^{246 \times 784}$.

Their parameters are:

- $512 \times 246 = 125{,}952$
- $246 \times 784 = 192{,}864$

Total: $125{,}952 + 192{,}864 = 318{,}816$

| | Parameters |
| --- | --- |
| Dense | 401,408 |
| TensorFold | 318,816 |
| Reduction | 82,592 (≈ 20.58%) |

This illustrates how replacing one large matrix with two smaller matrices can reduce parameter storage.

---

## SVD-Based Decomposition

TensorFold-arm implements SVD decomposition in `tensorfold/decomposition.py`.

The main function is `low_rank_svd(weight, rank)`. It:

1. Checks that the input is a 2D matrix.
2. Validates the requested rank.
3. Computes reduced SVD.
4. Keeps only the first `rank` components.
5. Returns $U_r$, $\Sigma_r$, $V_r^\top$.

The implementation uses:

```python
torch.linalg.svd(
    weight,
    full_matrices=False
)
```

### Folding the Singular Values

TensorFold does not need to store three separate matrices. Starting from $W \approx U_r \Sigma_r V_r^\top$, we define:

$$A = U_r \Sigma_r, \quad B = V_r^\top$$

Therefore $W \approx AB$.

Because $\Sigma_r$ is diagonal, multiplying $U_r$ by $\Sigma_r$ simply scales the columns of $U_r$. This is performed by `TensorFoldLinear.from_linear()`.

---

## Energy-Based Rank Selection

Choosing a rank manually for every layer would be inconvenient. TensorFold-arm therefore provides automatic rank selection using singular-value energy.

The total matrix energy is:

$$E_{total} = \sigma_1^2 + \sigma_2^2 + \dots + \sigma_k^2$$

The energy retained by rank `r` is:

$$E_r = \sigma_1^2 + \sigma_2^2 + \dots + \sigma_r^2$$

The explained energy is $E_r / E_{total}$.

TensorFold chooses the smallest rank satisfying:

$$\frac{E_r}{E_{total}} \geq \text{target\_energy}$$

For example, with `energy = 0.90`, TensorFold selects the smallest rank that preserves at least 90% of the matrix's SVD energy.

This is implemented by `select_rank(weight, energy)`.

### Energy Is Not Accuracy

An important distinction: **90% SVD energy does not mean 90% model accuracy.**

SVD energy measures how much of the original matrix's squared singular value energy is retained. Model accuracy measures how well the complete neural network performs on its task.

Therefore, TensorFold evaluates matrix compression and model accuracy separately.

---

## Inference Computation

A normal Linear layer performs:

$$Y = XW^\top + b$$

After TensorFold decomposition, $W^\top \approx AB$, therefore:

$$Y \approx XAB + b$$

Matrix multiplication is associative: $XAB = (XA)B$. So TensorFold computes:

$$Y = (XA)B + b$$

instead of multiplying directly by the complete dense matrix.

### Computational Complexity

For a batch size `B`:

**Dense** — the approximate matrix multiplication cost is $O(BMN)$.

**TensorFold** — the factorized computation performs `X × A` followed by `(XA) × B`, giving an approximate cost of:

$$O(BMr + BrN) = O(Br(M + N))$$

Therefore the theoretical compute ratio is approximately:

$$\frac{r(M + N)}{MN}$$

When `r` is sufficiently smaller than both `M` and `N`, the factorized representation can reduce computation.

However, theoretical operation counts do not automatically guarantee a real-world speedup. Actual inference performance depends on:

- CPU architecture
- Matrix dimensions
- Cache behavior
- Memory movement
- PyTorch kernels
- Threading
- Batch size

Therefore TensorFold-arm benchmarks the actual implementation on ARM64 hardware.

---

## Architecture

TensorFold-arm operates as a compression layer around an existing PyTorch model.

```
                PyTorch Model
                     │
                     ▼
              ┌─────────────┐
              │ nn.Linear   │
              └─────────────┘
                     │
                     ▼
               SVD Analysis
                     │
                     ▼
              Rank Selection
                     │
                     ▼
          Is Compression Beneficial?
               /             \
             No               Yes
             │                 │
             ▼                 ▼
      Keep nn.Linear     TensorFoldLinear
                               │
                               ▼
                         Low-Rank Factors
                            A and B
```

During inference:

**Dense Linear:**

```
Input
  │
  ▼
┌───────────────┐
│      W        │
│     M × N     │
└───────────────┘
  │
  ▼
Output
```

**TensorFold:**

```
Input
  │
  ▼
┌───────────────┐
│      A        │
│     M × r     │
└───────────────┘
  │
  ▼
┌───────────────┐
│      B        │
│     r × N     │
└───────────────┘
  │
  ▼
Output
```

The external input and output dimensions remain unchanged. Only the internal representation of the weight transformation changes.

---

## How TensorFold Works

The complete compression pipeline is:

```
Original PyTorch Model
          │
          ▼
    Find nn.Linear
          │
          ▼
     Extract weight
          │
          ▼
       Compute SVD
          │
          ▼
      Select rank r
          │
          ▼
 Calculate compressed parameters
          │
          ▼
 Is compressed model smaller?
       /          \
     No            Yes
     │              │
     ▼              ▼
Keep Linear   TensorFoldLinear
                     │
                     ▼
               Copy bias
                     │
                     ▼
             Return compressed
                  model
```

The original model is not modified. `compress()` creates a deep copy before replacing layers.

---

## Code Architecture

The project is divided into a small number of focused components.

### `tensorfold/decomposition.py`

Contains the mathematical decomposition logic.

Main functions:

- `low_rank_svd()` — Computes a truncated SVD.
- `select_rank()` — Chooses the smallest rank satisfying the requested energy threshold.
- `analyze_linear()` — Calculates input features, output features, selected rank, energy, original parameters, compressed parameters, parameter reduction, and compression decision.

### `tensorfold/layers.py`

Contains `TensorFoldLinear`.

The layer stores:

- `U` / `A`: `in_features × rank`
- `V` / `B`: `rank × out_features`

The forward pass is:

```python
y = x @ self.U
y = y @ self.V
```

followed by bias addition.

The class method `TensorFoldLinear.from_linear()` converts an existing PyTorch `nn.Linear` layer using truncated SVD.

### `tensorfold/optimizer.py`

Contains `compress()`.

The optimizer:

1. Deep-copies the model.
2. Recursively traverses its modules.
3. Finds `nn.Linear` layers.
4. Performs SVD-based analysis.
5. Selects an energy-based rank.
6. Calculates compressed parameter count.
7. Replaces the layer only when compression actually reduces parameters.

This prevents TensorFold from replacing layers where the low-rank representation would not provide a parameter benefit.

### `tensorfold/__init__.py`

Exports the public TensorFold API:

- `TensorFoldLinear`
- `low_rank_svd`
- `select_rank`
- `analyze_linear`
- `compress`

This allows users to write `from tensorfold import compress` instead of importing individual internal modules.

---

## Project Structure

```
TensorFold-arm/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── arm-test.yml
│
├── benchmarks/
│
├── data/
│
├── examples/
│   ├── basic_compression.py
│   └── benchmark_tensorfold.py
│
├── tensorfold/
│   ├── __init__.py
│   ├── decomposition.py
│   ├── layers.py
│   └── optimizer.py
│
├── tests/
│   ├── test_compression.py
│   ├── test_decomposition.py
│   ├── test_layers.py
│   └── test_linear.py
│
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```

The trained model used for the benchmark is kept out of the repository when configured through `.gitignore`.

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

Activate it.

**Linux / macOS**

```bash
source venv/bin/activate
```

**Windows PowerShell**

```powershell
venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install TensorFold-arm in editable mode:

```bash
pip install -e .
```

---

## Usage

### Create a TensorFold Linear Layer

```python
from tensorfold import TensorFoldLinear

layer = TensorFoldLinear(
    in_features=784,
    out_features=512,
    rank=128
)
```

The layer stores two low-rank matrices rather than one dense matrix.

### Convert an Existing Linear Layer

```python
import torch.nn as nn
from tensorfold import TensorFoldLinear

dense = nn.Linear(784, 512)

compressed = TensorFoldLinear.from_linear(
    dense,
    rank=128
)
```

The bias is copied from the original layer.

### Analyze a Linear Layer

```python
from tensorfold import analyze_linear

analysis = analyze_linear(
    dense.weight,
    energy=0.90
)

print(analysis)
```

The analysis returns information including:

- `rank`
- `energy`
- `original_parameters`
- `compressed_parameters`
- `parameter_reduction`
- `compression_possible`

---

## Automatic Model Compression

The main high-level API is:

```python
from tensorfold import compress

compressed_model = compress(
    model,
    energy=0.90
)
```

TensorFold recursively searches the model for `nn.Linear` layers. Only layers where `compressed_parameters < original_parameters` are replaced. The original model remains unchanged.

---

## Running Tests

Run the complete test suite:

```bash
pytest tests -v
```

The current test suite contains 13 tests covering:

- SVD decomposition
- Low-rank reconstruction
- Invalid ranks
- Energy-based rank selection
- Invalid energy values
- Linear-layer analysis
- TensorFoldLinear forward/backward behavior
- Parameter counts
- Shape validation
- Conversion from `nn.Linear`
- Bias and no-bias behavior
- Automatic compression

The test suite currently passes: `13 passed`

---

## Benchmark Methodology

TensorFold-arm benchmarks dense and TensorFold versions of the same MNIST MLP.

The model architecture is `784 → 512 → 256 → 10`.

The benchmark compares a Dense PyTorch model against a TensorFold model using a 90% SVD energy target.

Benchmark configuration:

| Setting | Value |
| --- | --- |
| Threads | 1 |
| Repeats | 10 |
| Iterations | 500 |

The benchmark reports mean latency, median latency, standard deviation, speedup, parameter count, and parameter reduction.

The benchmark is intentionally run with a single thread so that the comparison is controlled and reproducible.

---

## ARM64 Benchmark Results

The primary performance evaluation was performed on a native ARM64 GitHub Actions runner.

The ARM64 workflow uses `runs-on: ubuntu-24.04-arm`. The workflow verifies the architecture before running the benchmark.

The resulting ARM64 measurements are:

| Batch Size | Dense Mean (ms) | TensorFold 90% Mean (ms) | Speedup |
| --- | --- | --- | --- |
| 1 | 0.1914 | 0.1759 | 1.088× |
| 16 | 0.7866 | 0.5991 | 1.313× |
| 32 | 1.2195 | 0.8667 | 1.407× |
| 64 | 2.1408 | 1.4104 | 1.518× |
| 256 | 7.5964 | 4.5649 | 1.664× |

The highest measured speedup is **1.664×** at batch size 256. This corresponds to approximately **39.9%** lower measured mean latency for that benchmark configuration.

### Parameter Compression Results

For the benchmark model:

| Metric | Value |
| --- | --- |
| Dense parameters | 535,818 |
| TensorFold parameters (90% energy) | 279,940 |
| Parameter reduction | 47.75% |

Therefore the compressed model stores less than 53% of the original parameter count for this benchmark configuration.

---

## x86 Reference Benchmark

For comparison, the same benchmark was also run on the development environment.

| Batch Size | Dense Mean (ms) | TensorFold 90% Mean (ms) | Speedup |
| --- | --- | --- | --- |
| 1 | 0.0852 | 0.0820 | 1.039× |
| 16 | 0.3182 | 0.2401 | 1.325× |
| 32 | 0.5031 | 0.3335 | 1.509× |
| 64 | 0.8549 | 0.5702 | 1.499× |
| 256 | 3.0489 | 1.9401 | 1.572× |

The x86 measurements are provided as a reference. The primary target of TensorFold-arm is Arm64 CPU inference.

---

## Accuracy Results

The benchmark model was evaluated before and after TensorFold compression.

| Model | Accuracy |
| --- | --- |
| Dense | 97.79% |
| TensorFold | 97.06% |
| Difference | -0.73 pp |

The compression therefore produced a substantial reduction in parameter count while maintaining similar model accuracy for the tested MNIST model.

Accuracy and SVD energy should not be interpreted as the same metric.

---

## Results Summary

The main results are:

| Metric | Result |
| --- | --- |
| Dense parameters | 535,818 |
| TensorFold parameters | 279,940 |
| Parameter reduction | 47.75% |
| Dense accuracy | 97.79% |
| TensorFold accuracy | 97.06% |
| Accuracy difference | -0.73 pp |
| Maximum ARM64 speedup | 1.664× |
| Maximum speedup batch size | 256 |

The results demonstrate that the low-rank representation can reduce both parameter count and inference latency for the tested model.

---

## ARM64 Validation

TensorFold-arm is not only benchmarked locally. The project includes a dedicated ARM64 GitHub Actions workflow.

The workflow runs on `ubuntu-24.04-arm` and verifies:

- Machine architecture
- CPU information
- Python architecture
- PyTorch installation
- TensorFold installation

The workflow then executes the TensorFold benchmark directly on the ARM64 environment. This provides a reproducible ARM64 execution path rather than relying only on x86 development results.

### Continuous Integration

The repository also contains a standard CI workflow. The CI pipeline verifies that:

```
Python environment
       ↓
Dependencies
       ↓
TensorFold installation
       ↓
Package import
       ↓
Pytest
```

complete successfully.

The ARM64 workflow extends this validation to the target architecture and additionally runs the benchmark.

---

## Limitations

The current implementation intentionally focuses on 2D matrices.

**Supported:**

- PyTorch `nn.Linear`
- 2D weight matrices
- SVD-based decomposition
- Low-rank factorization
- Energy-based rank selection
- Automatic Linear-layer replacement
- CPU inference
- ARM64 benchmarking

The current implementation does not yet provide native tensor decompositions for higher-order tensors. Methods such as Tucker, CP, and Tensor Train are therefore outside the current implementation scope.

The current TensorFold-arm implementation uses SVD because the weights being compressed are 2D matrices.

---

## Future Work

Possible future improvements include:

### Latency-Aware Rank Selection

Instead of selecting rank only from SVD energy:

```
Energy target
      ↓
Candidate ranks
      ↓
Benchmark candidates
      ↓
Select best latency / accuracy tradeoff
```

This could make rank selection hardware-aware.

### Higher-Order Tensor Decompositions

Future versions could extend beyond 2D matrices and support Tucker, CP, and Tensor Train for higher-order neural-network tensors.

### More ARM Platforms

Future benchmarking could evaluate different ARM64 CPUs, different core counts, different batch sizes, and different PyTorch versions to determine how low-rank inference behaves across the ARM ecosystem.

### Larger Models

Future evaluations can include larger neural networks and transformer models to determine how the approach scales beyond the current MNIST MLP benchmark.

---

## Reproducibility

To reproduce the benchmark:

```bash
python examples/benchmark_tensorfold.py
```

The benchmark configuration is:

| Setting | Value |
| --- | --- |
| Threads | 1 |
| Repeats | 10 |
| Iterations | 500 |

To reproduce the test suite:

```bash
pytest tests -v
```

For ARM64 validation, use the GitHub Actions ARM64 workflow included in `.github/workflows/arm-test.yml`.

---

## Design Philosophy

TensorFold-arm is intentionally implemented as a small, focused optimization layer rather than a complete neural-network framework.

The project operates on existing PyTorch models:

```
Existing PyTorch Model
          │
          ▼
       TensorFold
          │
          ▼
Compressed PyTorch Model
```

This allows the optimization to be applied without requiring users to rewrite their neural-network architecture.

The core transformation is simple:

```
Dense:

W ∈ R^(M × N)

        ↓ SVD

W ≈ UᵣΣᵣVᵣᵀ

        ↓ fold Σ

W ≈ AB

        ↓

A ∈ R^(M × r)
B ∈ R^(r × N)
```

The result is a smaller representation of the original Linear-layer weight matrix.

---

## Summary

TensorFold-arm demonstrates an SVD-based approach to neural-network inference optimization on Arm64 CPUs.

The central transformation is $W \approx AB$, which changes the parameter count from `MN` to `r(M + N)` when $r \ll \min(M, N)$.

For the tested MNIST MLP: 535,818 → 279,940 parameters, giving **47.75% parameter reduction**, while model accuracy changed from **97.79%** to **97.06%**, and the native ARM64 benchmark achieved a maximum measured speedup of **1.664×** at batch size 256.

The project demonstrates that low-rank matrix decomposition can be used as a practical model-compression technique for CPU inference while retaining the original PyTorch model interface.

---

## License

TensorFold-arm is released under the MIT License. See `LICENSE` for the complete license text.

---

## Project Status

TensorFold-arm currently provides:

- SVD-based Linear-layer decomposition
- Energy-based rank selection
- Automatic model compression
- Low-rank `TensorFoldLinear`
- PyTorch integration
- Unit tests
- Continuous integration
- Native ARM64 testing
- ARM64 performance benchmarking
- Parameter reduction analysis
- Accuracy evaluation

The project was developed as an independent implementation for the Arm AI Optimization Challenge.
