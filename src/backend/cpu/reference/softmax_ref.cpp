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
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* /*workspace*/)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    NNOPS_ASSERT(rank >= 1);

    // Normalize axis
    int64_t axis = attrs.axis;
    if (axis < 0) { axis += rank; }
    NNOPS_ASSERT(axis >= 0 && axis < rank);

    // Compute outer size and axis stride
    int64_t outer_size = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer_size *= input.shape(i);
    }
    const int64_t D = input.shape(axis);  // normalization dimension
    const int64_t axis_elems = input.stride_elems(axis);  // element stride along axis (accounts for pitch)
    const int64_t inner_elems = input.stride_elems(rank - 1);  // always 1

    const auto* in_ptr  = input.ptr<float>();
    auto* out_ptr = output.ptr<float>();
    const bool log_softmax = attrs.log_softmax;

    // Process each row of size D
    const auto process_row = [&](int64_t outer) {
        int64_t base = outer * D * axis_elems;
        // Iterate over inner elements (elements "between" axis values)
        // The inner stride is always 1 (contiguous in the innermost dimension)
        // But the "inner" here refers to the dimensions after axis.
        // We need to iterate over them properly accounting for pitch.
        //
        // For a [B, C, H, W] tensor with axis=1 (C), axis_elems = H * row_stride.
        // The elements at position (outer, k, ...) are at: base + k * axis_elems + inner_off
        // where inner_off traverses the (H, W) sub-grid.
        //
        // We compute inner_total = product of dims[axis+1..rank-1]
        // and iterate over inner using offset_within_dims.
        int64_t inner_total = 1;
        for (int64_t i = axis + 1; i < rank; ++i) {
            inner_total *= input.shape(i);
        }

        for (int64_t s = 0; s < inner_total; ++s) {
            // Compute offset of inner position s within dims[axis+1..rank-1]
            int64_t inner_off = 0;
            int64_t rem = s;
            for (int64_t i = rank - 1; i > axis; --i) {
                int64_t dim = input.shape(i);
                inner_off += (rem % dim) * input.stride_elems(i);
                rem /= dim;
            }

            // Step 1: Find max for numerical stability
            float max_val = -FLT_MAX;
            for (int64_t k = 0; k < D; ++k) {
                float v = in_ptr[base + k * axis_elems + inner_off];
                if (v > max_val) { max_val = v; }
            }

            // Step 2: Compute sum of exp(x - max)
            float sum_exp = 0.0f;
            for (int64_t k = 0; k < D; ++k) {
                float v = in_ptr[base + k * axis_elems + inner_off];
                sum_exp += std::exp(v - max_val);
            }

            if (log_softmax) {
                // log_softmax = (x - max) - log(sum_exp)
                float log_sum = std::log(sum_exp);
                for (int64_t k = 0; k < D; ++k) {
                    float v = in_ptr[base + k * axis_elems + inner_off];
                    float val = (v - max_val) - log_sum;
                    out_ptr[base + k * axis_elems + inner_off] = val;
                }
            } else {
                // softmax = exp(x - max) / sum_exp
                float inv_sum = 1.0f / sum_exp;
                for (int64_t k = 0; k < D; ++k) {
                    float v = in_ptr[base + k * axis_elems + inner_off];
                    float val = std::exp(v - max_val) * inv_sum;
                    out_ptr[base + k * axis_elems + inner_off] = val;
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
