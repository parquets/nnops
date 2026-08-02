/// @file argminmax.cpp
/// @brief SIMD-optimized CPU implementation of ArgMax / ArgMin.
///
/// Supports both f32 and f16 via templated implementation.
/// Uses scalar reduction (argminmax is inherently sequential along the axis).
/// Data loading uses generic s_load for dtype abstraction.

#include "nnops/ops/argminmax.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include <limits>

namespace nnops::backend::cpu {

using nnops::simd::s_load;

template <typename T>
void argminmax_impl(const ArgMinMaxAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs)
{
    const auto& in = inputs[0];
    const int64_t rank = in.rank();
    const int64_t ax = (attrs.axis < 0) ? attrs.axis + rank : attrs.axis;
    const int64_t axis_dim = in.shape(ax);
    const bool is_max = (attrs.type == ArgMinMaxType::Max);

    int64_t outer_dims = 1;
    for (int64_t d = 0; d < ax; ++d) {
        outer_dims *= in.shape(d);
    }
    int64_t inner_size = 1;
    for (int64_t d = ax + 1; d < rank; ++d) {
        inner_size *= in.shape(d);
    }

    const int64_t axis_stride = in.stride_elems(ax);
    const auto* i_ptr = in.ptr<T>();
    auto* o_ptr = output.ptr<int64_t>();

    int64_t o_outer_stride;
    if (attrs.keepdims) {
        o_outer_stride = output.stride_elems(ax);
    } else {
        o_outer_stride = inner_size;
    }

    const int64_t total = outer_dims * inner_size;

    if (total > 0) {
        for (int64_t outer = 0; outer < outer_dims; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                const int64_t i_base = outer * axis_stride * axis_dim + inner;

                int64_t best_idx = 0;
                float   best_val = s_load(i_ptr + i_base);

                for (int64_t k = 1; k < axis_dim; ++k) {
                    float val = s_load(i_ptr + i_base + k * axis_stride);
                    bool better = is_max ? (val > best_val) : (val < best_val);
                    if (better) {
                        best_val = val;
                        best_idx = k;
                    }
                }

                int64_t o_pos = outer * o_outer_stride + inner;
                o_ptr[o_pos] = best_idx;
            }
        }
    }
}

void argmax_cpu(const ArgMinMaxAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& /*ctx*/,
                 void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        argminmax_impl<float>(attrs, output, inputs);
        return;
    case DataType::f16:
        argminmax_impl<half>(attrs, output, inputs);
        return;
    default:
        NNOPS_ASSERT(!"argmax_cpu: unsupported data type (only f32 and f16)");
    }
}

void argmin_cpu(const ArgMinMaxAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* workspace)
{
    ArgMinMaxAttributes a = attrs;
    a.type = ArgMinMaxType::Min;
    argmax_cpu(a, output, inputs, ctx, workspace);
}

}  // namespace nnops::backend::cpu
