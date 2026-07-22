/// @file matmul_ref.cpp
/// @brief Naive CPU reference implementation of matrix multiplication.
///
/// Computes: C = A × B  (optionally transposed)

#include "nnops/ops/matmul.hpp"
#include "nnops/core/parallel_for.hpp"

namespace nnops::backend::cpu::reference {

void matmul_ref(const MatMulAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* /*workspace*/)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    const int64_t M = attrs.transpose_a ? a.shape(1) : a.shape(0);
    const int64_t K = attrs.transpose_a ? a.shape(0) : a.shape(1);
    const int64_t N = attrs.transpose_b ? b.shape(0) : b.shape(1);

    const auto* a_ptr = a.ptr<float>();
    const auto* b_ptr = b.ptr<float>();
    auto* c_ptr = output.ptr<float>();

    // Row strides in elements (from pitch, >= innermost dim size)
    const int64_t a_row_stride = a.row_stride_elems();
    const int64_t b_row_stride = b.row_stride_elems();
    const int64_t c_row_stride = output.row_stride_elems();

    // Parallel over M
    const auto compute_row = [&](int64_t m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                const float a_val = attrs.transpose_a
                    ? a_ptr[k * a_row_stride + m]
                    : a_ptr[m * a_row_stride + k];
                const float b_val = attrs.transpose_b
                    ? b_ptr[n * b_row_stride + k]
                    : b_ptr[k * b_row_stride + n];
                sum += a_val * b_val;
            }
            float val = apply_epilogue(attrs.epilogue, sum, n);
            c_ptr[m * c_row_stride + n] = attrs.add_to ? c_ptr[m * c_row_stride + n] + val : val;
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
