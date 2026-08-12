// Fused low-rank linear forward for CPU.
//
// Computes:   Y = (X @ U) @ V + bias
//
// where X: [B, N], U: [N, R], V: [R, M], bias: [M] (optional), Y: [B, M].
//
// This is mathematically identical to two separate matmuls
// (`(x @ U) @ V`), which is what tensorfold/layers.py does today. The
// difference is *how* it's computed:
//
//   - Stock PyTorch path: two ATen kernel dispatches, and the full
//     [B, R] intermediate tensor is written to memory by the first
//     matmul and re-read by the second. At small batch sizes this
//     overhead is negligible; at batch >= 64 in the ARM64 benchmark it
//     dominates and TensorFold ends up *slower* than the dense layer.
//
//   - This kernel: one dispatch, and the intermediate for each row is
//     a length-R buffer (R is the compressed rank, typically small)
//     that lives in cache/registers for the lifetime of that row's
//     computation and is never written back to main memory.
//
// The inner loops are written in "outer product accumulation" order
// (loop over the contraction index outermost, accumulate into a
// contiguous output buffer) rather than the naive dot-product order.
// This keeps every innermost-loop memory access contiguous, which is
// what lets the compiler auto-vectorize with SIMD (AVX on x86,
// NEON on Armv8) at -O3. The naive dot-product order would stride
// through U/V with stride R or M, which defeats auto-vectorization
// and cache locality.

#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <vector>

torch::Tensor fused_forward(
    torch::Tensor x,                      // [B, N]
    torch::Tensor U,                      // [N, R]
    torch::Tensor V,                      // [R, M]
    c10::optional<torch::Tensor> bias_opt // [M] or none
)
{
  TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor [batch, in_features]");
  TORCH_CHECK(U.dim() == 2, "U must be a 2D tensor [in_features, rank]");
  TORCH_CHECK(V.dim() == 2, "V must be a 2D tensor [rank, out_features]");
  TORCH_CHECK(x.size(1) == U.size(0),
              "in_features mismatch between x and U");
  TORCH_CHECK(U.size(1) == V.size(0),
              "rank mismatch between U and V");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "only float32 is supported");
  TORCH_CHECK(U.scalar_type() == torch::kFloat32, "only float32 is supported");
  TORCH_CHECK(V.scalar_type() == torch::kFloat32, "only float32 is supported");

  x = x.contiguous();
  U = U.contiguous();
  V = V.contiguous();

  const int64_t B = x.size(0);
  const int64_t N = x.size(1);
  const int64_t R = U.size(1);
  const int64_t M = V.size(1);

  torch::Tensor bias;
  const float *bias_ptr = nullptr;
  if (bias_opt.has_value())
  {
    bias = bias_opt.value().contiguous();
    TORCH_CHECK(bias.numel() == M, "bias size must equal out_features");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "only float32 is supported");
    bias_ptr = bias.data_ptr<float>();
  }

  auto y = torch::empty({B, M}, x.options());

  const float *x_ptr = x.data_ptr<float>();
  const float *U_ptr = U.data_ptr<float>();
  const float *V_ptr = V.data_ptr<float>();
  float *y_ptr = y.data_ptr<float>();

  // Parallelize over the batch dimension. Each row is independent,
  // so this is embarrassingly parallel and scales with thread count
  // the same way the stock matmul path would (both eventually
  // bottleneck on the same BLAS-level parallelism for large batches;
  // the win here is specifically the removed intermediate + one
  // fewer dispatch, not a change in asymptotic FLOP parallelism).
  at::parallel_for(0, B, /*grain_size=*/1, [&](int64_t start, int64_t end)
                   {
        std::vector<float> tmp(static_cast<size_t>(R));

        for (int64_t b = start; b < end; ++b) {
            const float* xrow = x_ptr + b * N;

            // Stage 1: tmp[0:R] = xrow @ U
            // Outer-product accumulation: for each input feature i,
            // scale U's i-th row (contiguous, length R) by x[i] and
            // accumulate into tmp. Inner loop over r is contiguous.
            std::fill(tmp.begin(), tmp.end(), 0.0f);
            for (int64_t i = 0; i < N; ++i) {
                const float xi = xrow[i];
                const float* Urow = U_ptr + i * R;
                for (int64_t r = 0; r < R; ++r) {
                    tmp[r] += xi * Urow[r];
                }
            }

            // Stage 2: yrow[0:M] = tmp @ V (+ bias)
            // Same outer-product pattern: for each rank component r,
            // scale V's r-th row (contiguous, length M) by tmp[r] and
            // accumulate into yrow.
            float* yrow = y_ptr + b * M;
            if (bias_ptr != nullptr) {
                std::copy(bias_ptr, bias_ptr + M, yrow);
            } else {
                std::fill(yrow, yrow + M, 0.0f);
            }
            for (int64_t r = 0; r < R; ++r) {
                const float tr = tmp[r];
                const float* Vrow = V_ptr + r * M;
                for (int64_t m = 0; m < M; ++m) {
                    yrow[m] += tr * Vrow[m];
                }
            }
        } });

  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def(
      "fused_forward",
      &fused_forward,
      "Fused (X @ U) @ V + bias forward for TensorFoldLinear (CPU)");
}
