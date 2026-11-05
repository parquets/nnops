/// @file softmax.cpp
/// @brief SIMD-optimized CPU implementation of softmax / log-softmax.
///
/// One templated implementation covers f32 and f16 over the two activation
/// layout families — channel-first and channel-last; the
/// operator refuses channel-packed layouts (Softmax::validate). Two dispatch
/// paths, chosen by where the normalization axis sits in memory:
///   - The axis is the innermost axis of the storage — channel-first's last dim, or
///     channel-last's C. Every group is then `D` contiguous elements and the
///     groups are a constant stride apart, so each one is one
///     kernel::softmax_process_standard_row call (horizontal reduction).
///   - Every other axis: each group is `D` elements strided by the axis' element
///     stride, so kernel::softmax_per_lane runs L groups at a time. Lanes are
///     the channels of a channel-last tensor, or the dims below a channel-first axis
///     when the pitch keeps them contiguous; otherwise one group per lane.
///
/// Pitch-aware via stride_elems() / row_stride_elems(). Patterned on
/// onnxruntime's MlasComputeSoftmax / MlasReduceMaximumF32Kernel /
/// MlasComputeSumExpF32Kernel / MlasComputeSoftmaxOutputF32Kernel.

#include "nnops/ops/softmax.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "nnops/detail/half.hpp"
#include "common/memory_pool.hpp"
#include "common/dtype_dispatch.hpp"
#include "simd_kernel/simd_softmax.hpp"

#if defined(NNOPS_ARCH_X86_64)
#include "x86_64/quant.hpp"
#elif defined(NNOPS_ARCH_AARCH64)
#include "aarch64/quant.hpp"
#else
#error "softmax: unsupported architecture for quantization kernels"
#endif

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <vector>

namespace nnops::backend::cpu {

using namespace nnops::simd;

#if defined(NNOPS_ARCH_X86_64)
namespace quant_kernel = nnops::backend::cpu::x86_64;
#elif defined(NNOPS_ARCH_AARCH64)
namespace quant_kernel = nnops::backend::cpu::aarch64;
#endif

namespace {

/// True when the axis is the innermost axis of the storage: channel-first's last dim
/// and channel-last's C are one contiguous run, and the runs are a constant
/// stride apart, so the contiguous-row kernel can take the whole axis.
bool axis_is_innermost(const TensorView& x, int64_t axis, int64_t& unit_stride)
{
    const int64_t rank = x.rank();
    if (rank == 1) {
        unit_stride = 1;
        return true;
    }
    if (is_channel_last(x.layout())) {
        if (axis == 1) {
            unit_stride = x.shape(1);
            return true;
        }
        // C == 1 leaves the spatial axis as the innermost one as well.
        if (axis == rank - 1 && x.shape(1) == 1) {
            unit_stride = x.shape(rank - 1);
            return true;
        }
        return false;
    }
    if (axis == rank - 1) {
        unit_stride = x.row_stride_elems();
        return true;
    }
    return false;
}

/// One normalization group, scalar: D elements `stride` apart.
template <typename T>
void softmax_group_scalar(const T* x, T* y, int64_t D, int64_t stride,
                          bool log_softmax, float inv_T)
{
    float max_val = -std::numeric_limits<float>::infinity();
    for (int64_t k = 0; k < D; ++k) {
        const float v = s_load(&x[k * stride]);
        if (v > max_val) { max_val = v; }
    }

    float sum_exp = 0.0f;
    for (int64_t k = 0; k < D; ++k) {
        sum_exp += std::exp((s_load(&x[k * stride]) - max_val) * inv_T);
    }

    if (log_softmax) {
        const float log_sum = std::log(sum_exp);
        for (int64_t k = 0; k < D; ++k) {
            const float v = (s_load(&x[k * stride]) - max_val) * inv_T - log_sum;
            s_store(&y[k * stride], v);
        }
    } else {
        const float inv_sum = 1.0f / sum_exp;
        for (int64_t k = 0; k < D; ++k) {
            const float v = std::exp((s_load(&x[k * stride]) - max_val) * inv_T) * inv_sum;
            s_store(&y[k * stride], v);
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

    void add(const TensorView& x, int64_t dim) {
        shape[n] = x.shape(dim);
        stride[n] = x.stride_elems(dim);
        count *= shape[n];
        ++n;
    }

    int64_t offset_of(int64_t flat) const {
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
void softmax_strided_axis(const T* x_ptr, T* y_ptr, const TensorView& X,
                          int64_t axis, bool log_softmax, float inv_T,
                          const ComputeContext& ctx)
{
    constexpr int64_t L = simd_lane_for<T>;
    const int64_t rank = X.rank();
    const int64_t D = X.shape(axis);
    const int64_t axis_stride = X.stride_elems(axis);

    // Lanes: the channels of a channel-last tensor are one vector, and so are
    // the dims below a channel-first axis when the pitch leaves them contiguous.
    const bool channel_last = is_channel_last(X.layout()) && rank > 1;
    const bool lanes_are_channels = channel_last && axis != 1;
    int64_t nlanes = 1;
    if (lanes_are_channels) {
        nlanes = X.shape(1);
    } else if (axis + 1 < rank && X.stride_elems(rank - 1) == 1 &&
               (axis == rank - 2 ||
                X.pitch() == static_cast<int64_t>(
                    data_type_row_bytes(X.data_type(), X.shape(rank - 1))))) {
        for (int64_t d = axis + 1; d < rank; ++d) { nlanes *= X.shape(d); }
    }

    PositionWalk pos;
    for (int64_t d = 0; d < rank; ++d) {
        if (d == axis) { continue; }
        if (nlanes > 1) {
            if (lanes_are_channels ? (d == 1) : (d > axis)) { continue; }
        }
        pos.add(X, d);
    }

    const int64_t lane_blocks = (nlanes + L - 1) / L;

    ctx.cpu.run(0, pos.count * lane_blocks, [&](int64_t task) {
        const int64_t base = pos.offset_of(task / lane_blocks);
        const int64_t lane0 = (task % lane_blocks) * L;

        if (lane0 + L <= nlanes) {
            kernel::softmax_per_lane<T>(
                x_ptr + base + lane0, y_ptr + base + lane0,
                D, axis_stride, log_softmax, inv_T);
        } else {
            for (int64_t lane = lane0; lane < nlanes; ++lane) {
                softmax_group_scalar<T>(
                    x_ptr + base + lane, y_ptr + base + lane,
                    D, axis_stride, log_softmax, inv_T);
            }
        }
    });
}

// ============================================================
// Innermost-axis softmax — one contiguous run per group
// ============================================================

template <typename T>
void softmax_contiguous_axis(const T* x_ptr, T* y_ptr, const TensorView& X,
                             int64_t axis, int64_t unit_stride,
                             bool log_softmax, float inv_T,
                             const ComputeContext& ctx)
{
    const int64_t D = X.shape(axis);
    const int64_t units = (D > 0) ? X.numel() / D : 0;

    ctx.cpu.run(0, units, [&](int64_t u) {
        kernel::softmax_process_standard_row<T>(
            x_ptr + u * unit_stride, y_ptr + u * unit_stride,
            D, log_softmax, inv_T);
    });
}

}  // anonymous namespace

// ---- Templated implementation (f32 and f16) ----

template <typename T>
void softmax_impl(const SoftmaxAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx)
{
    const auto& X = inputs[0];

    const int64_t rank = X.rank();
    NNOPS_ASSERT(rank >= 1);

    const bool log_softmax = attrs.log_softmax;
    const float temperature = attrs.temperature;
    NNOPS_ASSERT(temperature > 0.0f);
    const float inv_T = 1.0f / temperature;

    int64_t axis = attrs.axis;
    if (axis < 0) { axis += rank; }
    NNOPS_ASSERT(axis >= 0 && axis < rank);

    const auto* x_ptr = X.ptr<T>();
    auto* y_ptr = output.ptr<T>();

    int64_t unit_stride = 0;
    if (axis_is_innermost(X, axis, unit_stride)) {
        softmax_contiguous_axis<T>(x_ptr, y_ptr, X, axis, unit_stride,
                                   log_softmax, inv_T, ctx);
        return;
    }

    softmax_strided_axis<T>(x_ptr, y_ptr, X, axis, log_softmax, inv_T, ctx);
}

// ---- Quantized input (s8/u8) → float output (f32/f16) ----
//
// Dequantize to contiguous f32 scratch (arch quant.hpp per-row kernels), run f32
// softmax on it, then write back f32 or f16 (probabilities are never
// re-quantized). PerTensor / PerToken only, channel-first layouts only.

Status softmax_quant_input_impl(const SoftmaxAttributes& attrs,
                                TensorView& output,
                                std::span<const TensorView> inputs,
                                const ComputeContext& ctx,
                                void* workspace)
{
    const auto& X = inputs[0];
    const int64_t numel = X.numel();
    const int64_t rank = X.rank();
    NNOPS_ASSERT(rank >= 1);   // Softmax::validate

    const DataType in_dtype  = X.data_type();             // s8 or u8
    const DataType out_dtype = output.data_type();        // f32 or f16
    NNOPS_ASSERT(is_quantized_dtype(in_dtype));
    NNOPS_ASSERT(out_dtype == DataType::f32 || out_dtype == DataType::f16);

    // Quantized softmax is only defined over channel-first layouts (Softmax::validate).
    NNOPS_ASSERT(is_channel_first_spatial(X.layout()));
    if (workspace == nullptr) { return Status::out_of_memory; }

    const QuantParams& qp = X.quant_params();
    const bool per_token = qp.granularity == QuantGranularity::PerToken
                        && qp.scale_data != nullptr;

    // Row shape: one (scale, zero_point) per innermost row for PerToken, one
    // parameter for the whole tensor for PerTensor.
    const int64_t last_dim = X.shape(rank - 1);
    const int64_t M = numel / last_dim;  // number of `last_dim`-element rows

    // Carve the pooled scratch (allocated once in softmax_cpu; getWorkspaceSize
    // returns 0):
    //   [ x_f32 : numel ][ scale : M ][ zero : M ][ y_f32 : numel (f16 only) ]
    float* x_f32 = static_cast<float*>(workspace);
    float* scale = x_f32 + numel;
    float* zero  = scale + M;

    // Quant kernels take int dimensions/strides (see x86_64/quant.hpp).
    const uint8_t* in_base = X.ptr<uint8_t>();
    const int sr_i = static_cast<int>(X.row_stride_elems());
    const int m_i = static_cast<int>(M);
    const int n_i = static_cast<int>(last_dim);

    for (int64_t i = 0; i < M; ++i) {
        if (per_token) {
            scale[i] = qp.scale_data[i];
            zero[i]  = qp.zero_point_data != nullptr
                ? static_cast<float>(qp.zero_point_data[i]) : 0.0f;
        } else {
            scale[i] = qp.scale;
            zero[i]  = static_cast<float>(qp.zero_point);
        }
    }

    if (in_dtype == DataType::s8) {
        quant_kernel::dequantization<int8_t>(m_i, n_i, x_f32, n_i,
            reinterpret_cast<const int8_t*>(in_base), sr_i, scale, zero);
    } else {
        quant_kernel::dequantization<uint8_t>(m_i, n_i, x_f32, n_i,
            in_base, sr_i, scale, zero);
    }

    TensorView x_view(X.shape_span(), DataType::f32, x_f32, TensorLayout::NCHW);
    const TensorView ins[] = {x_view};

    if (out_dtype == DataType::f32) {
        softmax_impl<float>(attrs, output, ins, ctx);
    } else {
        // f32 staging, downcast to f16 on store (probabilities are not re-quantized).
        float* y_f32 = zero + M;
        TensorView y_view(X.shape_span(), DataType::f32, y_f32, TensorLayout::NCHW);
        softmax_impl<float>(attrs, y_view, ins, ctx);
        half* out = output.ptr<half>();
        for (int64_t i = 0; i < numel; ++i) {
            s_store(&out[i], y_f32[i]);
        }
    }
    return Status::ok;
}

Status softmax_cpu(const SoftmaxAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx,
                  void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    if (is_quantized_dtype(dtype)) {
        // softmax probabilities are dequantized float, never int
        NNOPS_ASSERT(!is_quantized_dtype(output.data_type()));

        // Dequantization scratch is pooled internally (getWorkspaceSize returns 0):
        //   [ x_f32 : numel ][ scale : M ][ zero : M ][ y_f32 : numel (f16) ].
        const int64_t numel    = inputs[0].numel();
        const int64_t rank     = inputs[0].rank();
        const int64_t last_dim = inputs[0].shape(rank - 1);
        const int64_t M        = numel / last_dim;
        size_t ws = static_cast<size_t>(numel) * sizeof(float)          // x_f32
                  + 2 * static_cast<size_t>(M) * sizeof(float);         // scale + zero
        if (output.data_type() == DataType::f16) {
            ws += static_cast<size_t>(numel) * sizeof(float);           // y_f32 staging
        }
        PoolPtr scratch(ws);

        return softmax_quant_input_impl(attrs, output, inputs, ctx,
                                        scratch.get());
    }
    dispatch_f32_f16(dtype, "softmax_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        softmax_impl<T>(attrs, output, inputs, ctx);
    });
    return Status::ok;
}

}  // namespace nnops::backend::cpu
