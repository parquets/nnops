/// @file cumsum_ref.cpp
/// @brief Naive CPU reference implementation of cumulative sum.
///
/// Decomposes the tensor along axis into upper/lower dims and applies
/// recurrence: out[i] = in[i-1] + out[i-1] (exclusive) or + in[i] (inclusive).

#include "nnops/ops/cumsum.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/assert.hpp"

#include <numeric>

namespace nnops::backend::cpu::reference {

void cumsum_ref(const CumSumAttributes& attrs,
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

    const int64_t dim = input.shape(axis);

    // Number of slices before the axis
    int64_t upper_dim_count = 1;
    for (int64_t i = 0; i < axis; ++i) {
        upper_dim_count *= input.shape(i);
    }

    // Size of the contiguous tail after the axis (treated as 1D vectors)
    int64_t lower_dim_size = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        lower_dim_size *= input.shape(i);
    }

    const auto* in_ptr  = input.ptr<float>();
    auto* out_ptr = output.ptr<float>();

    // Stride between consecutive elements along the cumulative axis
    // Uses stride_elems(axis) which accounts for pitch padding.
    const int64_t axis_stride = input.stride_elems(axis);

    if (!attrs.reverse) {
        // Forward cumulative sum
        const auto process_slice = [&](int64_t outer) {
            const int64_t slice_start = outer * dim * lower_dim_size;

            if (attrs.exclusive) {
                // exclusive: out[0] = 0, out[i] = in[i-1] + out[i-1]
                float running = 0.0f;
                for (int64_t k = 0; k < dim; ++k) {
                    for (int64_t s = 0; s < lower_dim_size; ++s) {
                        int64_t idx = slice_start + k * axis_stride + s;
                        float val = running;  // exclusive: result does not include current
                        out_ptr[idx] = val;
                        running += in_ptr[idx];  // update after writing
                    }
                }
            } else {
                // inclusive: out[0] = in[0], out[i] = in[i] + out[i-1]
                float running = 0.0f;
                for (int64_t k = 0; k < dim; ++k) {
                    for (int64_t s = 0; s < lower_dim_size; ++s) {
                        int64_t idx = slice_start + k * axis_stride + s;
                        running += in_ptr[idx];
                        out_ptr[idx] = running;
                    }
                }
            }
        };

        ctx.cpu.run(0, upper_dim_count, process_slice);
    } else {
        // Reverse cumulative sum (start from the end)
        const auto process_slice = [&](int64_t outer) {
            const int64_t slice_start = outer * dim * lower_dim_size;

            if (attrs.exclusive) {
                // exclusive reverse: out[dim-1] = 0, out[i] = in[i+1] + out[i+1]
                float running = 0.0f;
                for (int64_t k = dim - 1; k >= 0; --k) {
                    for (int64_t s = 0; s < lower_dim_size; ++s) {
                        int64_t idx = slice_start + k * axis_stride + s;
                        float val = running;  // exclusive: result does not include current
                        out_ptr[idx] = val;
                        running += in_ptr[idx];  // update after writing
                    }
                }
            } else {
                // inclusive reverse: out[dim-1] = in[dim-1], out[i] = in[i] + out[i+1]
                float running = 0.0f;
                for (int64_t k = dim - 1; k >= 0; --k) {
                    for (int64_t s = 0; s < lower_dim_size; ++s) {
                        int64_t idx = slice_start + k * axis_stride + s;
                        running += in_ptr[idx];
                        out_ptr[idx] = running;
                    }
                }
            }
        };

        ctx.cpu.run(0, upper_dim_count, process_slice);
    }
}

}  // namespace nnops::backend::cpu::reference
