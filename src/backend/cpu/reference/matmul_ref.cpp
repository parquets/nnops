/// @file matmul_ref.cpp
/// @brief Naive CPU reference implementation of matrix multiplication.
///
/// Computes: C = A × B  (optionally transposed)

#include "nnops/ops/matmul.hpp"
#include "nnops/core/parallel_for.hpp"

namespace nnops::backend::cpu::reference {

void matmul_ref(const MatMulAttributes& attrs,
                const TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* /*workspace*/)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    const int64_t M = attrs.transpose_a ? a.shape(1) : a.shape(0);
    const int64_t K = attrs.transpose_a ? a.shape(0) : a.shape(1);
    const int64_t N = attrs.transpose_b ? b.shape(0) : b.shape(1);

    const auto* a_ptr = a.data_as<float>();
    const auto* b_ptr = b.data_as<float>();
    auto* c_ptr = output.data_as<float>();

    // Parallel over M
    const auto compute_row = [&](int64_t m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                const float a_val = attrs.transpose_a
                    ? a_ptr[k * M + m]
                    : a_ptr[m * K + k];
                const float b_val = attrs.transpose_b
                    ? b_ptr[n * K + k]
                    : b_ptr[k * N + n];
                sum += a_val * b_val;
            }
            c_ptr[m * N + n] = apply_epilogue(attrs.epilogue, sum, n);
        }
    };

    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, M,
            [&](int64_t m) { compute_row(m); });
    } else {
        for (int64_t m = 0; m < M; ++m) {
            compute_row(m);
        }
    }
}

}  // namespace nnops::backend::cpu::reference
