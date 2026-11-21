/// @file softmax.cpp
/// @brief SIMD-optimized CPU implementation of softmax / log-softmax on NHWC.
///
/// One templated implementation covers f32 and f16; the operator refuses the
/// packed weight layouts (Softmax::validate). Two dispatch paths, chosen by
/// where the normalization axis sits in memory:
///   - The axis is the innermost axis of the storage — NHWC's C (or the
///     last dim when C == 1) — *and* the view is compact, so every group is
///     `D` contiguous elements and the next group is exactly D further on. Each
///     group is one kernel::softmax_process_standard_row call (horizontal
///     reduction).
///   - Every other case, including an innermost axis in a row-pitched view:
///     each group is `D` elements strided by the axis' element stride, so
///     kernel::softmax_per_lane runs L groups at a time, the lanes being the
///     channels of the tensor.
///
/// Pitch-aware via stride_elems() / row_stride_elems(). Patterned on
/// onnxruntime's MlasComputeSoftmax / MlasReduceMaximumF32Kernel /
/// MlasComputeSumExpF32Kernel / MlasComputeSoftmaxOutputF32Kernel.

#include "nnops/ops/softmax.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "common/dtype_dispatch.hpp"
#include "common/strided_walk.hpp"
#include "simd_kernel/simd_softmax.hpp"

#include <cmath>
#include <cfloat>
#include <cstdint>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

/// True when the axis is the innermost axis of the storage: NHWC's C is
/// one contiguous run, and the runs are a constant stride apart, so the
/// contiguous-row kernel can take the whole axis.
bool axis_is_innermost(const TensorView& x, int64_t axis, int64_t& unit_stride)
{
    const int64_t rank = x.rank();
    if (rank == 1) {
        unit_stride = 1;
        return true;
    }
    // A channel-last tensor's innermost axis is C, so a softmax over the last
    // dimension walks a contiguous run.
    if (axis == rank - 1) {
        unit_stride = x.shape(rank - 1);
        return true;
    }
    // C == 1 leaves the spatial axis as the innermost one as well.
    if (axis == rank - 2 && x.shape(rank - 1) == 1) {
        unit_stride = x.shape(rank - 2);
        return true;
    }
    return false;
}

/// One normalization group, scalar: D elements `stride` apart.
template <typename T>
void softmax_group_scalar(const T* x, T* y, int64_t D, int64_t stride, int64_t stride_y,
                          bool log_softmax, float inv_T)
{
    float max_val = -std::numeric_limits<float>::infinity();
    for (int64_t k = 0; k < D; ++k) {
        const float v = s_load(&x[k * stride]);
        if (v > max_val) {
            max_val = v;
        }
    }

    float sum_exp = 0.0f;
    for (int64_t k = 0; k < D; ++k) {
        sum_exp += std::exp((s_load(&x[k * stride]) - max_val) * inv_T);
    }

    if (log_softmax) {
        const float log_sum = std::log(sum_exp);
        for (int64_t k = 0; k < D; ++k) {
            const float v = (s_load(&x[k * stride]) - max_val) * inv_T - log_sum;
            s_store(&y[k * stride_y], v);
        }
    } else {
        const float inv_sum = 1.0f / sum_exp;
        for (int64_t k = 0; k < D; ++k) {
            const float v = std::exp((s_load(&x[k * stride]) - max_val) * inv_T) * inv_sum;
            s_store(&y[k * stride_y], v);
        }
    }
}

/// The dims whose flat index selects a group's position — everything but the
/// softmax axis, and (channel-last) the channel axis, which supplies the lanes.
struct PositionWalk {
    int64_t count = 1;
    int64_t n = 0;
    int64_t shape[TensorView::kMaxRank] = {};
    int64_t stride[TensorView::kMaxRank] = {};

    void add(const TensorView& x, int64_t dim)
    {
        shape[n] = x.shape(dim);
        stride[n] = x.stride_elems(dim);
        count *= shape[n];
        ++n;
    }

    int64_t offset_of(int64_t flat) const
    {
        int64_t off = 0;
        for (int64_t i = n - 1; i >= 0; --i) {
            off += (flat % shape[i]) * stride[i];
            flat /= shape[i];
        }
        return off;
    }
};

// ============================================================
// General-axis softmax — per-lane SIMD along the axis
// ============================================================

template <typename T>
void softmax_strided_axis(const T* x_ptr, T* y_ptr, const TensorView& X, const TensorView& Y,
                          int64_t axis, bool log_softmax, float inv_T, const ComputeContext& ctx)
{
    constexpr int64_t L = simd_lane_for<T>;
    const int64_t rank = X.rank();
    const int64_t D = X.shape(axis);
    const int64_t axis_stride = X.stride_elems(axis);
    const int64_t axis_stride_y = Y.stride_elems(axis);

    // Lanes: the channels of a channel-last tensor are one vector.
    const int64_t channel_axis = rank - 1;
    const bool lanes_are_channels = rank > 1 && axis != channel_axis;
    const int64_t nlanes = lanes_are_channels ? X.shape(channel_axis) : 1;

    // Both sides walk the same positions, each over its own strides: the output
    // may be pitched differently from the input.
    PositionWalk pos;
    PositionWalk pos_y;
    for (int64_t d = 0; d < rank; ++d) {
        if (d == axis || (lanes_are_channels && d == channel_axis)) {
            continue;
        }
        pos.add(X, d);
        pos_y.add(Y, d);
    }

    const int64_t lane_blocks = (nlanes + L - 1) / L;

    ctx.cpu.run(0, pos.count * lane_blocks, [&](int64_t task) {
        const int64_t base = pos.offset_of(task / lane_blocks);
        const int64_t base_y = pos_y.offset_of(task / lane_blocks);
        const int64_t lane0 = (task % lane_blocks) * L;

        if (lane0 + L <= nlanes) {
            kernel::softmax_per_lane<T>(x_ptr + base + lane0, y_ptr + base_y + lane0, D,
                                        axis_stride, log_softmax, inv_T, axis_stride_y);
        } else {
            for (int64_t lane = lane0; lane < nlanes; ++lane) {
                softmax_group_scalar<T>(x_ptr + base + lane, y_ptr + base_y + lane, D, axis_stride,
                                        axis_stride_y, log_softmax, inv_T);
            }
        }
    });
}

// ============================================================
// Innermost-axis softmax — one contiguous run per group
// ============================================================

template <typename T>
void softmax_contiguous_axis(const T* x_ptr, T* y_ptr, const TensorView& X, int64_t axis,
                             int64_t unit_stride, int64_t unit_stride_y, bool log_softmax,
                             float inv_T, const ComputeContext& ctx)
{
    const int64_t D = X.shape(axis);
    const int64_t units = (D > 0) ? X.numel() / D : 0;

    ctx.cpu.run(0, units, [&](int64_t u) {
        kernel::softmax_process_standard_row<T>(x_ptr + u * unit_stride, y_ptr + u * unit_stride_y,
                                                D, log_softmax, inv_T);
    });
}

} // anonymous namespace

// ---- Templated implementation (f32 and f16) ----

template <typename T>
void softmax_impl(const SoftmaxAttributes& attrs, TensorView& output,
                  std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& X = inputs[0];

    const int64_t rank = X.rank();
    NNOPS_ASSERT(rank >= 1);

    const bool log_softmax = attrs.log_softmax;
    const float temperature = attrs.temperature;
    NNOPS_ASSERT(temperature > 0.0f);
    const float inv_T = 1.0f / temperature;

    int64_t axis = attrs.axis;
    if (axis < 0) {
        axis += rank;
    }
    NNOPS_ASSERT(axis >= 0 && axis < rank);

    const auto* x_ptr = X.ptr<T>();
    auto* y_ptr = output.ptr<T>();

    int64_t unit_stride = 0;
    // The contiguous kernel steps from one group to the next by the group width,
    // which is only the group's true distance when the view carries no row pitch
    // — a padded row jumps further along. A pitched view therefore takes the
    // strided walk (correct at any stride, one scalar group at a time).
    if (axis_is_innermost(X, axis, unit_stride) && is_compact_view(X) &&
        is_compact_view(output)) {
        // Same shape on both sides, so the output answers the same question over
        // its own pitch — which a caller is free to give it differently.
        int64_t unit_stride_y = 0;
        (void)axis_is_innermost(output, axis, unit_stride_y);
        softmax_contiguous_axis<T>(x_ptr, y_ptr, X, axis, unit_stride, unit_stride_y, log_softmax,
                                   inv_T, ctx);
        return;
    }

    softmax_strided_axis<T>(x_ptr, y_ptr, X, output, axis, log_softmax, inv_T, ctx);
}

Status softmax_cpu(const SoftmaxAttributes& attrs, TensorView& output,
                   std::span<const TensorView> inputs, const ComputeContext& ctx,
                   void* /*workspace*/)
{
    if (!dispatch_f32_f16(inputs[0].data_type(), "softmax_cpu", [&](auto tag) {
            using T = typename decltype(tag)::type;
            softmax_impl<T>(attrs, output, inputs, ctx);
        })) {
        return Status::unsupported;
    }
    return Status::ok;
}

} // namespace nnops::backend::cpu
