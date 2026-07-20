/// @file linear_ref.cpp
/// @brief Naive CPU reference implementation of Linear / fully-connected layer.
///
/// Computes: output = input × weight^T + bias
///   input:  [M, K], weight: [N, K], bias: [N] (optional), output: [M, N]

#include "nnops/ops/linear.hpp"
#include "nnops/core/parallel_for.hpp"

namespace nnops::backend::cpu::reference {

void linear_ref(const LinearAttributes& attrs,
                const TensorView& output,
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

    const auto* in_ptr = input.data_as<float>();
    const auto* w_ptr  = weight.data_as<float>();
    const auto* b_ptr  = has_bias ? inputs[2].data_as<float>() : nullptr;
    auto* out_ptr = output.data_as<float>();

    // Parallel over M (batch dimension)
    const auto compute_row = [&](int64_t m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += in_ptr[m * K + k] * w_ptr[n * K + k];  // weight^T access
            }
            if (has_bias) {
                sum += b_ptr[n];
            }
            out_ptr[m * N + n] = apply_epilogue(attrs.epilogue, sum, n);
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
