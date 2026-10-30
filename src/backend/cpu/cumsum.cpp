/// @file cumsum.cpp
/// @brief CPU CumSum along one axis, forward or reverse, inclusive or exclusive.
///
/// The tensor decomposes into slices above the axis, one task each; a slice
/// makes one run of lower_dim_size elements per axis step, the steps
/// stride_elems(axis) apart. Each slice carries its own running sum; reverse
/// runs the same recurrence from the far end. Exclusive emits the sum before
/// folding in the current element, inclusive after.

#include "nnops/ops/cumsum.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops::backend::cpu {

Status cumsum_cpu(const CumSumAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* /*workspace*/)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    NNOPS_ASSERT(rank >= 1);

    int64_t axis = attrs.axis;
    if (axis < 0) { axis += rank; }
    NNOPS_ASSERT(axis >= 0 && axis < rank);

    const int64_t dim = input.shape(axis);

    int64_t upper_dim_count = 1;
    for (int64_t i = 0; i < axis; ++i) {
        upper_dim_count *= input.shape(i);
    }

    int64_t lower_dim_size = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        lower_dim_size *= input.shape(i);
    }

    const auto* in_ptr  = input.ptr<float>();
    auto* out_ptr = output.ptr<float>();

    // Pitch-aware: the axis step can exceed lower_dim_size.
    const int64_t axis_stride = input.stride_elems(axis);

    if (!attrs.reverse) {
        const auto process_slice = [&](int64_t outer) {
            const int64_t slice_start = outer * dim * lower_dim_size;

            if (attrs.exclusive) {
                float running = 0.0f;
                for (int64_t k = 0; k < dim; ++k) {
                    for (int64_t s = 0; s < lower_dim_size; ++s) {
                        int64_t idx = slice_start + k * axis_stride + s;
                        float val = running;
                        out_ptr[idx] = val;
                        running += in_ptr[idx];
                    }
                }
            } else {
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
        const auto process_slice = [&](int64_t outer) {
            const int64_t slice_start = outer * dim * lower_dim_size;

            if (attrs.exclusive) {
                float running = 0.0f;
                for (int64_t k = dim - 1; k >= 0; --k) {
                    for (int64_t s = 0; s < lower_dim_size; ++s) {
                        int64_t idx = slice_start + k * axis_stride + s;
                        float val = running;
                        out_ptr[idx] = val;
                        running += in_ptr[idx];
                    }
                }
            } else {
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
    return Status::ok;
}

}  // namespace nnops::backend::cpu
