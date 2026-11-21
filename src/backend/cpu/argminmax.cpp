/// @file argminmax.cpp
/// @brief SIMD-optimized CPU implementation of ArgMax / ArgMin.
///
/// The axis is reduced in two passes: an 8-lane max/min over the contiguous run
/// (four accumulators, NaN never winning), then a scalar scan for the first
/// element that holds it. The index scan is what keeps ties and NaN behaviour
/// identical to the element-at-a-time form; the reduction itself is what makes
/// a long axis cheap — a [1, 150k] arg-max used to be a 150k-step dependency
/// chain on one core.
///
/// The walk is described by strides alone — an outer group, the axis, an inner
/// group — so the layout never enters the loop: a padded row, an NHWC
/// view and a dense tensor all reduce to the same two groups.

#include "nnops/ops/argminmax.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "nnops/detail/axis_walk.hpp"
#include "common/dtype_dispatch.hpp"

namespace nnops::backend::cpu {

using namespace nnops::simd;
using nnops::simd::s_load;
using detail::AxisWalk;
using detail::axis_group_offset;

template <typename T>
void argminmax_impl(const ArgMinMaxAttributes& attrs, TensorView& output,
                    std::span<const TensorView> inputs, const ComputeContext& ctx)
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

    // One task per (outer, inner) position: an arg-max over the channel axis of
    // a feature map has inner == 1 and one position per pixel, which used to run
    // on a single thread.
    ctx.cpu.run(0, w.outer * w.inner, [&](int64_t t) {
        const int64_t outer = t / w.inner;
        const int64_t inner = t % w.inner;
        const int64_t base = axis_group_offset(outer, w.pre) + axis_group_offset(inner, w.post);
        const T* row = i_ptr + base;

        int64_t best_idx = 0;
        if (w.axis_stride == 1) {
            // v_max(x, acc) / v_min(x, acc) keep the accumulatee when the
            // element is NaN, so a NaN never wins — the same rule the scalar
            // compare applies.
            constexpr int L = 8;
            const int64_t n = w.axis_dim;
            int64_t k = 0;
            v_f32x8 a0, a1, a2, a3;
            if (n >= 4 * L) {
                a0 = v_load_f32(row);
                a1 = v_load_f32(row + L);
                a2 = v_load_f32(row + 2 * L);
                a3 = v_load_f32(row + 3 * L);
                for (k = 4 * L; k + 4 * L <= n; k += 4 * L) {
                    a0 = is_max ? v_max(v_load_f32(row + k), a0) : v_min(v_load_f32(row + k), a0);
                    a1 = is_max ? v_max(v_load_f32(row + k + L), a1) :
                                  v_min(v_load_f32(row + k + L), a1);
                    a2 = is_max ? v_max(v_load_f32(row + k + 2 * L), a2) :
                                  v_min(v_load_f32(row + k + 2 * L), a2);
                    a3 = is_max ? v_max(v_load_f32(row + k + 3 * L), a3) :
                                  v_min(v_load_f32(row + k + 3 * L), a3);
                }
                a0 = is_max ? v_max(v_max(a0, a1), v_max(a2, a3))
                            : v_min(v_min(a0, a1), v_min(a2, a3));
                for (; k + L <= n; k += L) {
                    a0 = is_max ? v_max(v_load_f32(row + k), a0) : v_min(v_load_f32(row + k), a0);
                }
                best_idx = 0;
                const float best = is_max ? v_reduce_max(a0) : v_reduce_min(a0);
                for (int64_t j = 0; j < n; ++j) {
                    if (s_load(row + j) == best) {
                        best_idx = j;
                        break;
                    }
                }
            } else {
                float best_val = s_load(row);
                for (int64_t j = 1; j < n; ++j) {
                    const float val = s_load(row + j);
                    if (is_max ? (val > best_val) : (val < best_val)) {
                        best_val = val;
                        best_idx = j;
                    }
                }
            }
        } else {
            float best_val = s_load(row);
            for (int64_t k = 1; k < w.axis_dim; ++k) {
                const float val = s_load(row + k * w.axis_stride);
                if (is_max ? (val > best_val) : (val < best_val)) {
                    best_val = val;
                    best_idx = k;
                }
            }
        }

        o_ptr[axis_group_offset(outer, w.out_pre) + axis_group_offset(inner, w.out_post)] =
            best_idx;
    });
}

Status argmax_cpu(const ArgMinMaxAttributes& attrs, TensorView& output,
                  std::span<const TensorView> inputs, const ComputeContext& ctx,
                  void* /*workspace*/)
{
    if (!dispatch_f32_f16(inputs[0].data_type(), "argmax_cpu", [&](auto tag) {
            using T = typename decltype(tag)::type;
            argminmax_impl<T>(attrs, output, inputs, ctx);
        })) {
        return Status::unsupported;
    }
    return Status::ok;
}

Status argmin_cpu(const ArgMinMaxAttributes& attrs, TensorView& output,
                  std::span<const TensorView> inputs, const ComputeContext& ctx, void* workspace)
{
    ArgMinMaxAttributes a = attrs;
    a.type = ArgMinMaxType::Min;
    argmax_cpu(a, output, inputs, ctx, workspace);
    return Status::ok;
}

} // namespace nnops::backend::cpu
