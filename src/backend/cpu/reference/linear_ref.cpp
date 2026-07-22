/// @file linear_ref.cpp
/// @brief Naive CPU reference implementation of Linear / fully-connected layer.
///
/// Computes: output = input × weight^T + bias
///   input:  [M, K], weight: [N, K], bias: [N] (optional), output: [M, N]

#include "nnops/ops/linear.hpp"
#include "nnops/core/parallel_for.hpp"

namespace nnops::backend::cpu::reference {

void linear_ref(const LinearAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* /*workspace*/)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    // Determine M and K from input
    const int64_t M = (input.rank() > 1) ? input.shape(0) : 1;
    const int64_t K = (input.rank() > 1) ? input.shape(1) : input.shape(0);

    // Weight: [N, K]
    const int64_t N = weight.shape(0);

    const auto* in_ptr = input.ptr<float>();
    const auto* w_ptr  = weight.ptr<float>();
    const auto* b_ptr  = has_bias ? inputs[2].ptr<float>() : nullptr;
    auto* out_ptr = output.ptr<float>();

    // Row strides in elements (from pitch)
    const int64_t in_row_stride  = input.row_stride_elems();   // >= K
    const int64_t w_row_stride   = weight.row_stride_elems();  // >= K
    const int64_t out_row_stride = output.row_stride_elems();  // >= N

    // Parallel over M (batch dimension)
    const auto compute_row = [&](int64_t m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += in_ptr[m * in_row_stride + k] * w_ptr[n * w_row_stride + k];
            }
            if (has_bias) {
                sum += b_ptr[n];
            }
            float val = apply_epilogue(attrs.epilogue, sum, n);
            out_ptr[m * out_row_stride + n] = attrs.add_to ? out_ptr[m * out_row_stride + n] + val : val;
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
