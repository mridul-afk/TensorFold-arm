# TensorFold-arm — Devpost Submission

**Track:** Mobile AI (on-device inference: latency, model size, efficiency)

---

## Project Overview

TensorFold-arm shrinks a neural network's Linear layers using low-rank
SVD factorization, cutting parameter count by nearly half while adding
essentially zero accuracy cost, and validates the result on real ARM64
hardware rather than an emulator.

What makes it worth judging isn't just the compression number. Every
performance claim in this repo is backed by a reproducible benchmark
run on GitHub Actions' `ubuntu-24.04-arm` runner, and the project
includes something most hackathon submissions don't: a fully documented
negative result. We hypothesized that TensorFold's on-Arm slowdown at
large batch sizes was caused by kernel-dispatch overhead, built a custom
fused C++ kernel to fix it, tested that hypothesis with a controlled
three-way benchmark (dense vs. original vs. fused), and found the
hypothesis was wrong — then used that finding to correctly scope the
project to the batch-size regime (1-32) where it actually wins, which
is also exactly the regime that matches real on-device mobile inference.
That's the story we think should win: not an inflated headline number,
but a technique with an honestly measured, honestly bounded, and
correctly targeted benefit.

## Functionality / Output

**Input:** any PyTorch model containing `nn.Linear` layers.

**What it does:** `tensorfold.compress(model, energy=0.90)` walks the
model, and for each Linear layer where a low-rank SVD factorization at
the requested energy threshold would reduce parameter count, replaces
it with a `TensorFoldLinear` layer (`Y = (X @ U) @ V + b`) initialized
from a truncated SVD of the original weights. No retraining or
fine-tuning is required.

**Output, on the reference MNIST MLP (784-512-256-10):**

- 535,818 -> 279,940 parameters (47.75% reduction)
- 97.79% -> 97.06% accuracy (-0.73 percentage points, no fine-tuning)
- 1.04x-1.38x faster inference than the dense model at batch sizes 1-32
  on real ARM64 hardware (the batch regime that matches on-device
  mobile inference)

**Deliverables in the repo:**

- `tensorfold/` — the compression library (SVD decomposition, rank
  selection by energy threshold, `TensorFoldLinear` layer, `compress()`
  entry point)
- `tensorfold/csrc/fused_linear.cpp` + `tensorfold/fused_backend.py` —
  an optional fused CPU kernel, included with its full correctness
  suite and honestly reported (see Results below) as not providing a
  latency win, kept in the repo as documented, working, tested code
  rather than deleted
- `benchmarks/` — reproducible benchmark scripts (`bench_three_way.py`
  is the authoritative one) and `.github/workflows/arm-test.yml`, which
  re-runs every test and benchmark on real ARM64 hardware on every push
- `RESULTS.md` — full results, including the fused-kernel investigation
  writeup

## Setup Instructions

Requires Python >= 3.10.

```bash
git clone <repo-url>
cd tensorfold-arm
pip install -r requirements.txt
pip install -e .
```

Run the test suite:

```bash
pytest tests -v
```

Run the authoritative benchmark (dense vs. TensorFold-torch vs.
TensorFold-fused, at batch sizes 1/16/32/64/256):

```bash
python benchmarks/bench_three_way.py
```

Compress your own model:

```python
import tensorfold

compressed_model = tensorfold.compress(your_model, energy=0.90)
```

**To validate on real ARM64 hardware without owning ARM hardware:** the
repo's CI workflow (`.github/workflows/arm-test.yml`) runs the full test
suite and both benchmark scripts on GitHub Actions' `ubuntu-24.04-arm`
runner on every push. Fork the repo and push a commit, or check the
Actions tab of the original repo, to see live ARM64 results without
needing a physical Arm device.
