## Results (ARM64, authoritative)

All numbers below are from `benchmarks/bench_three_way.py` run on GitHub
Actions' `ubuntu-24.04-arm` runner (real ARM64 hardware, not emulated),
threads=1, 100 warmup + 500 timed iterations, 10 repeats. This is the
single source of truth for this project. If any other file in this repo
disagrees with the numbers below, this file is correct and the other
should be regenerated from it.

**Model:** 3-layer MNIST MLP (784 to 512 to 256 to 10)
**Compression:** truncated SVD, 90% energy target

| Metric | Value |
| --- | --- |
| Dense parameters | 535,818 |
| TensorFold parameters | 279,940 |
| Parameter reduction | 47.75% |
| Dense accuracy | 97.79% |
| TensorFold accuracy | 97.06% |
| Accuracy change | -0.73 pp |

### Latency: Dense vs TensorFold (torch backend) vs TensorFold (fused backend)

| Batch | Dense (ms) | TensorFold-torch (ms) | TensorFold-fused (ms) | Torch/Dense | Fused/Dense |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.1224 | 0.1150 | 0.1178 | 1.064x | 1.039x |
| 16 | 0.6813 | 0.4939 | 0.4991 | 1.379x | 1.365x |
| 32 | 0.8076 | 0.6649 | 0.6664 | 1.215x | 1.212x |
| 64 | 0.9992 | 1.1096 | 1.1113 | 0.901x | 0.899x |
| 256 | 2.0985 | 3.4643 | 3.4720 | 0.606x | 0.604x |

Parameter reduction (47.75%) holds regardless of batch size. Latency
behavior is batch-dependent: TensorFold is meaningfully faster than dense
at batch 1 to 32 (up to 1.379x at batch 16), and slower at batch 64/256.

### Why this project targets on-device / low-batch inference

Batch size 1 to 32 is the realistic regime for on-device inference: a
phone or edge device processes one request, or a small handful, at a
time. It doesn't queue 256 inputs before running the model. We report
the full batch range for honesty and reproducibility, but the project's
intended use case, and where its numbers are strongest, is small-batch,
latency-sensitive, on-device inference, not server-side batch
throughput.

---

## Investigation: does kernel fusion fix the batch 64/256 regression?

Hypothesis: TensorFold's forward pass does two matmuls per layer
instead of dense's one. We hypothesized the slowdown at large batch
came from implementation overhead: two ATen kernel dispatches per
layer instead of one, plus materializing the intermediate
[batch, rank] tensor to memory between them.

What we built: a custom CPU kernel (tensorfold/csrc/fused_linear.cpp,
C++/pybind11, JIT-compiled via torch.utils.cpp_extension) that
computes (x @ U) @ V + bias in a single pass per layer, one dispatch,
with the intermediate kept in a small per-row buffer that never
touches main memory. Correctness was verified against the original
two-matmul path across 5 layer shapes times 5 batch sizes, with and
without bias (tests/test_fused_backend.py, 28/28 passing on both
Linux/GCC and Windows/MSVC).

Result: the hypothesis was falsified. The table above shows
Torch/Dense and Fused/Dense tracking each other within about 2% at
every batch size, well inside measurement noise. Cutting the dispatch
count from 6 calls (2 matmuls times 3 layers) down to 3 (matching
dense's call count exactly) produced no measurable latency change.

Root cause, as best we can determine: the fused kernel processes each
batch row independently (a per-row GEMV loop). Dense's single matmul
and TensorFold's original two-matmul path are both batched GEMMs,
which reuse loaded weight data across every row in the batch and get
substantially better arithmetic intensity than a loop of independent
per-row GEMVs, especially as batch size grows. Fusing the two stages
into one dispatch removed overhead, but processing rows independently
gave up batched-GEMM efficiency; the two effects approximately cancel
out. The actual bottleneck is more fundamental than dispatch overhead:
at the rank this compression setting chooses (about 246 for a
784-to-512 layer), the FLOP savings from the low-rank factorization
aren't large enough to beat how efficiently a single dense batched
GEMM scales with batch size on this hardware. That's a property of
the rank and shape trade-off, not something a fused kernel can fix.

We are documenting this as a negative result rather than omitting it.
The correctness harness and fused-kernel code remain in the repo
(backend="fused" is still a working, tested option on
TensorFoldLinear), but we do not claim a latency win from it, and we
recommend backend="torch" (the default) for actual use.
