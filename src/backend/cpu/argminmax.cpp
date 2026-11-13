/// @file argminmax.cpp
/// @brief SIMD-optimized CPU implementation of ArgMax / ArgMin.
///
/// Supports both f32 and f16 via templated implementation.
/// Uses scalar reduction (argminmax is inherently sequential along the axis).
///
/// The walk is described by strides alone — an outer group, the axis, an inner
/// group — so the layout never enters the loop: a padded row, a channel-last
/// view and a dense planar tensor all reduce to the same two groups.

#include "nnops/ops/argminmax.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "nnops/detail/axis_walk.hpp"
#include "common/dtype_dispatch.hpp"

namespace nnops::backend::cpu {

using nnops::simd::s_load;
using detail::AxisWalk;
using detail::axis_group_offset;

template <typename T>
void argminmax_impl(const ArgMinMaxAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs)
{
    const auto& in = inputs[0];
    const int64_t rank = in.rank();
    const int64_t ax = (attrs.axis < 0) ? attrs.axis + rank : attrs.axis;
    const bool is_max = (attrs.type == ArgMinMaxType::Max);
    const AxisWalk w = detail::make_axis_walk(in, output, ax, attrs.keepdims);

    if (w.axis_dim == 0 || w.outer == 0 || w.inner == 0) {
        return;
    }

    const auto* i_ptr = in.ptr<T>();
    auto* o_ptr = output.ptr<int64_t>();

    for (int64_t outer = 0; outer < w.outer; ++outer) {
        const int64_t i_base = axis_group_offset(outer, w.pre);
        const int64_t o_base = axis_group_offset(outer, w.out_pre);
        for (int64_t inner = 0; inner < w.inner; ++inner) {
            const int64_t base = i_base + axis_group_offset(inner, w.post);

            int64_t best_idx = 0;
            float   best_val = s_load(i_ptr + base);

            for (int64_t k = 1; k < w.axis_dim; ++k) {
                float val = s_load(i_ptr + base + k * w.axis_stride);
                bool better = is_max ? (val > best_val) : (val < best_val);
                if (better) {
                    best_val = val;
                    best_idx = k;
                }
            }

            o_ptr[o_base + axis_group_offset(inner, w.out_post)] = best_idx;
        }
    }
}

Status argmax_cpu(const ArgMinMaxAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& /*ctx*/,
                 void* /*workspace*/)
{
    dispatch_f32_f16(inputs[0].data_type(), "argmax_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        argminmax_impl<T>(attrs, output, inputs);
    });
    return Status::ok;
}

Status argmin_cpu(const ArgMinMaxAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* workspace)
{
    ArgMinMaxAttributes a = attrs;
    a.type = ArgMinMaxType::Min;
    argmax_cpu(a, output, inputs, ctx, workspace);
    return Status::ok;
}

}  // namespace nnops::backend::cpu
