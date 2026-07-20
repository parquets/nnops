/// @file softmax_ref.cpp
/// @brief Naive CPU reference implementation of softmax / log-softmax.
///
/// Uses the max-subtraction trick for numerical stability.

#include "nnops/ops/softmax.hpp"
#include "nnops/core/parallel_for.hpp"

#include <cmath>
#include <algorithm>
#include <cfloat>

namespace nnops::backend::cpu::reference {

void softmax_ref(const SoftmaxAttributes& attrs,
                 const TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* /*workspace*/)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    NNOPS_ASSERT(rank >= 1);

    // Normalize axis
    int64_t axis = attrs.axis;
    if (axis < 0) axis += rank;
    NNOPS_ASSERT(axis >= 0 && axis < rank);

    // Compute outer and inner sizes
    int64_t outer_size = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer_size *= input.shape(i);
    }
    const int64_t D = input.shape(axis);  // normalization dimension
    int64_t inner_size = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        inner_size *= input.shape(i);
    }

    const auto* in_ptr  = input.data_as<float>();
    auto* out_ptr = output.data_as<float>();
    const bool log_softmax = attrs.log_softmax;

    // Process each row of size D
    const auto process_row = [&](int64_t outer) {
        int64_t base = outer * D * inner_size;
        for (int64_t s = 0; s < inner_size; ++s) {
            // Collect row elements (stride = inner_size between consecutive elements)
            // We need stride between elements along the axis.
            // Elements at positions: base + k * inner_size + s for k = 0..D-1

            // Step 1: Find max for numerical stability
            float max_val = -FLT_MAX;
            for (int64_t k = 0; k < D; ++k) {
                float v = in_ptr[base + k * inner_size + s];
                if (v > max_val) max_val = v;
            }

            // Step 2: Compute sum of exp(x - max)
            float sum_exp = 0.0f;
            for (int64_t k = 0; k < D; ++k) {
                float v = in_ptr[base + k * inner_size + s];
                sum_exp += std::exp(v - max_val);
            }

            if (log_softmax) {
                // log_softmax = (x - max) - log(sum_exp)
                float log_sum = std::log(sum_exp);
                for (int64_t k = 0; k < D; ++k) {
                    float v = in_ptr[base + k * inner_size + s];
                    float val = (v - max_val) - log_sum;
                    out_ptr[base + k * inner_size + s] =
                        attrs.add_to ? out_ptr[base + k * inner_size + s] + val : val;
                }
            } else {
                // softmax = exp(x - max) / sum_exp
                float inv_sum = 1.0f / sum_exp;
                for (int64_t k = 0; k < D; ++k) {
                    float v = in_ptr[base + k * inner_size + s];
                    float val = std::exp(v - max_val) * inv_sum;
                    out_ptr[base + k * inner_size + s] =
                        attrs.add_to ? out_ptr[base + k * inner_size + s] + val : val;
                }
            }
        }
    };

    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, outer_size, process_row);
    } else {
        for (int64_t i = 0; i < outer_size; ++i) {
            process_row(i);
        }
    }
}

}  // namespace nnops::backend::cpu::reference
