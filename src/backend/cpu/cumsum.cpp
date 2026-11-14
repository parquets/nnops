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
#include "nnops/detail/simd/simd.hpp"
#include "common/dtype_dispatch.hpp"

namespace nnops::backend::cpu {

/// The running sum lives in f32 whatever T is, so a bf16 or f16 run does not
/// accumulate the rounding of every step into the next.
template <typename T>
Status cumsum_impl(const CumSumAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx)
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

    const auto* in_ptr  = input.ptr<T>();
    auto* out_ptr = output.ptr<T>();

    // Pitch-aware: the axis step can exceed lower_dim_size.
    const int64_t axis_stride = input.stride_elems(axis);

    // One scan per lower-dim position: the lower dims are lanes that each carry
    // their own running sum. Sharing one sum across them scans the row instead of
    // the axis, which only agrees with this when the axis is the last one.
    if (!attrs.reverse) {
        const auto process_slice = [&](int64_t outer) {
            const int64_t slice_start = outer * dim * axis_stride;

            for (int64_t s = 0; s < lower_dim_size; ++s) {
                float running = 0.0f;
                for (int64_t k = 0; k < dim; ++k) {
                    const int64_t idx = slice_start + k * axis_stride + s;
                    if (attrs.exclusive) {
                        simd::s_store(out_ptr + idx, running);
                    }
                    running += simd::s_load(in_ptr + idx);
                    if (!attrs.exclusive) {
                        simd::s_store(out_ptr + idx, running);
                    }
                }
            }
        };

        ctx.cpu.run(0, upper_dim_count, process_slice);
    } else {
        const auto process_slice = [&](int64_t outer) {
            const int64_t slice_start = outer * dim * axis_stride;

            for (int64_t s = 0; s < lower_dim_size; ++s) {
                float running = 0.0f;
                for (int64_t k = dim - 1; k >= 0; --k) {
                    const int64_t idx = slice_start + k * axis_stride + s;
                    if (attrs.exclusive) {
                        simd::s_store(out_ptr + idx, running);
                    }
                    running += simd::s_load(in_ptr + idx);
                    if (!attrs.exclusive) {
                        simd::s_store(out_ptr + idx, running);
                    }
                }
            }
        };

        ctx.cpu.run(0, upper_dim_count, process_slice);
    }
    return Status::ok;
}

Status cumsum_cpu(const CumSumAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx,
                  void* /*workspace*/)
{
    dispatch_f32_f16(inputs[0].data_type(), "cumsum_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        cumsum_impl<T>(attrs, output, inputs, ctx);
    });
    return Status::ok;
}

}  // namespace nnops::backend::cpu
