/// @file norm.cpp
/// @brief SIMD-optimized CPU implementation of normalization operators.
///
/// Covers BatchNorm, LayerNorm, RMSNorm, GroupNorm, and L2Norm, dispatched
/// on NormAttributes::type. All share the consolidated SIMD kernels in
/// simd_kernel/simd_norm.hpp.
///
/// Supports f32 and f16 via a single templated implementation per type.
///
/// Also supports s8/u8 (quantized) input for LayerNorm / RMSNorm / L2Norm over
/// the trailing axis — see the "Quantized input" section below.

#include "nnops/ops/norm.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "simd_kernel/simd_norm.hpp"
#include "common/index.hpp"
#include "common/dtype_dispatch.hpp"
#include "common/memory_pool.hpp"

#if defined(NNOPS_ARCH_X86_64)
#include "x86_64/quant.hpp"
#elif defined(NNOPS_ARCH_AARCH64)
#include "aarch64/quant.hpp"
#else
#error "norm: unsupported architecture for quantization kernels"
#endif

#include <cmath>
#include <vector>

namespace nnops::backend::cpu {

using namespace nnops::simd;

#if defined(NNOPS_ARCH_X86_64)
namespace quant_kernel = nnops::backend::cpu::x86_64;
#elif defined(NNOPS_ARCH_AARCH64)
namespace quant_kernel = nnops::backend::cpu::aarch64;
#endif

namespace {

// ============================================================
// Shared helpers
// ============================================================

/// Pre-compute element offsets for non-contiguous normalization axes.
/// Maps a flattened inner index (0..norm_size-1) to its stride offset
/// within the tensor, handling arbitrary multi-dimensional layouts.
inline std::vector<int64_t> compute_inner_offsets(
    const TensorView& X, int64_t axis, int64_t norm_size)
{
    return build_offsets(X, axis, X.rank() - 1, norm_size);
}

/// Normalized axis dimensions for row-wise norm dispatch.
struct NormDims {
    int64_t axis;               ///< Normalized axis (0 <= axis < rank)
    int64_t num_rows;           ///< Number of independent rows to process
    int64_t norm_size;          ///< Elements per normalization row
    bool is_contiguous_tail;   ///< True when axis == rank-1 (SIMD fast path)
};

inline NormDims compute_norm_dims(const TensorView& X, int64_t raw_axis) {
    const int64_t rank = X.rank();
    int64_t axis = raw_axis;
    if (axis < 0) { axis += rank; }
    NNOPS_ASSERT(axis >= 0 && axis < rank);

    int64_t num_rows = 1;
    for (int64_t i = 0; i < axis; ++i) { num_rows *= X.shape(i); }
    int64_t norm_size = 1;
    for (int64_t i = axis; i < rank; ++i) { norm_size *= X.shape(i); }

    return {axis, num_rows, norm_size, axis == rank - 1};
}

// ============================================================
// BatchNorm
// ============================================================

template <typename T>
void batch_norm_impl(const NormAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx)
{
    const auto& X     = inputs[0];
    const auto& scale = inputs[1];
    const auto& bias  = inputs[2];
    const auto& mean  = inputs[3];
    const auto& var   = inputs[4];

    const int64_t rank = X.rank();
    const float epsilon = attrs.epsilon;

    int64_t N, C;
    if (rank == 1) { N = X.shape(0); C = 1; }
    else           { N = X.shape(0); C = X.shape(1); }

    const auto* x_ptr = X.ptr<T>();
    const auto* s_ptr = scale.ptr<T>();
    const auto* b_ptr = bias.ptr<T>();
    const auto* m_ptr = mean.ptr<T>();
    const auto* v_ptr = var.ptr<T>();
    auto* y_ptr = output.ptr<T>();

    if (attrs.spatial) {
        std::vector<float> ns_f(static_cast<size_t>(C));
        std::vector<float> nb_f(static_cast<size_t>(C));
        for (int64_t c = 0; c < C; ++c) {
            float inv_std = 1.0f / std::sqrt(s_load(&v_ptr[c]) + epsilon);
            ns_f[static_cast<size_t>(c)] = inv_std * s_load(&s_ptr[c]);
            nb_f[static_cast<size_t>(c)] = s_load(&b_ptr[c]) - s_load(&m_ptr[c]) * ns_f[static_cast<size_t>(c)];
        }

        const int64_t pack = X.channel_pack_size();

        if (pack > 1) {
            const int64_t c8_blocks = X.num_channel_blocks();
            std::vector<T> ns_packed(static_cast<size_t>(c8_blocks * pack), T(0));
            std::vector<T> nb_packed(static_cast<size_t>(c8_blocks * pack), T(0));
            for (int64_t c = 0; c < C; ++c) {
                s_store(&ns_packed[static_cast<size_t>(c)], ns_f[static_cast<size_t>(c)]);
                s_store(&nb_packed[static_cast<size_t>(c)], nb_f[static_cast<size_t>(c)]);
            }

            const int64_t num_rows = X.total_rows();
            const int64_t last_dim = X.shape(rank - 1) * pack;
            const int64_t x_rs = X.row_stride_elems();
            const int64_t y_rs = output.row_stride_elems();

            int64_t rows_per_c8 = 1;
            for (int64_t d = 2; d < rank - 1; ++d) {
                rows_per_c8 *= X.shape(d);
            }

            const auto process_row = [&](int64_t r) {
                const int64_t c8 = (r / rows_per_c8) % c8_blocks;
                kernel::batch_norm_process_packed_row<T>(
                    x_ptr + r * x_rs, y_ptr + r * y_rs,
                    ns_packed.data(), nb_packed.data(),
                    c8, pack, last_dim, attrs.add_to);
            };

            ctx.cpu.run(0, num_rows, process_row);

        } else {
            int64_t sample_size = 1;
            if (rank >= 2) {
                for (int64_t i = 2; i < rank; ++i) {
                    sample_size *= X.shape(i);
                }
            }

            const int64_t x_n_stride = X.stride_elems(0);
            const int64_t x_c_stride = (rank >= 2) ? X.stride_elems(1) : 1;
            const int64_t y_n_stride = output.stride_elems(0);
            const int64_t y_c_stride = (rank >= 2) ? output.stride_elems(1) : 1;

            const int64_t last_dim = (rank >= 3) ? X.shape(rank - 1) : 1;
            const int64_t num_rows = (rank >= 3) ? sample_size / last_dim : 1;
            const int64_t x_row_stride = X.row_stride_elems();
            const int64_t y_row_stride = output.row_stride_elems();

            const auto compute_sample = [&](int64_t n) {
                for (int64_t c = 0; c < C; ++c) {
                    const float ns = ns_f[static_cast<size_t>(c)];
                    const float nb = nb_f[static_cast<size_t>(c)];
                    const int64_t x_ch_base = n * x_n_stride + c * x_c_stride;
                    const int64_t y_ch_base = n * y_n_stride + c * y_c_stride;

                    for (int64_t hh = 0; hh < num_rows; ++hh) {
                        kernel::batch_norm_process_planar_row<T>(
                            x_ptr + x_ch_base + hh * x_row_stride,
                            y_ptr + y_ch_base + hh * y_row_stride,
                            last_dim, ns, nb, attrs.add_to);
                    }
                }
            };

            ctx.cpu.run(0, N, compute_sample);
        }

    } else {
        const int64_t total = X.numel();
        kernel::batch_norm_process_nonspatial_block<T>(
            x_ptr, y_ptr, s_ptr, b_ptr, m_ptr, v_ptr,
            0, total, epsilon, attrs.add_to);
    }
}

// ============================================================
// LayerNorm
// ============================================================

template <typename T>
void layer_norm_general_scalar(
    const T* x_ptr, const T* s_ptr, const T* b_ptr,
    T* y_ptr,
    int64_t num_rows, int64_t norm_size,
    int64_t axis, const TensorView& X, const TensorView& scale,
    float epsilon, bool add_to, bool has_bias,
    const ComputeContext& ctx)
{
    const auto inner_offsets = compute_inner_offsets(X, axis, norm_size);
    const int64_t outer_stride = (axis > 0) ? X.stride_elems(axis - 1) : 0;
    const bool scale_is_scalar = (scale.numel() == 1);

    const auto process_row = [&](int64_t row) {
        const int64_t row_base = row * outer_stride;

        float mean_val = 0.0f;
        float M2 = 0.0f;
        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t off = row_base + inner_offsets[static_cast<size_t>(i)];
            float x = s_load(&x_ptr[off]);
            float delta = x - mean_val;
            mean_val += delta / static_cast<float>(i + 1);
            float delta2 = x - mean_val;
            M2 += delta * delta2;
        }
        const float var_val = M2 / static_cast<float>(norm_size);
        const float inv_std = 1.0f / std::sqrt(var_val + epsilon);

        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t off = row_base + inner_offsets[static_cast<size_t>(i)];
            int64_t s_idx = scale_is_scalar ? 0 : i;
            float s = s_load(&s_ptr[s_idx]);
            float b = (has_bias && b_ptr) ? s_load(&b_ptr[s_idx]) : 0.0f;
            float val = (s_load(&x_ptr[off]) - mean_val) * inv_std * s + b;
            s_store_add(&y_ptr[off], val, add_to);
        }
    };

    ctx.cpu.run(0, num_rows, process_row);
}

template <typename T>
void layer_norm_impl(const NormAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx)
{
    const auto& X     = inputs[0];
    const auto& scale = inputs[1];
    const bool has_bias = (inputs.size() >= 3 && !inputs[2].is_empty());

    const float epsilon = attrs.epsilon;

    const auto* x_ptr  = X.ptr<T>();
    const auto* s_ptr  = scale.ptr<T>();
    const auto* b_ptr  = (has_bias && !inputs[2].is_empty()) ? inputs[2].ptr<T>() : nullptr;
    auto* y_ptr = output.ptr<T>();

    const bool add_to = attrs.add_to;
    const auto dims = compute_norm_dims(X, attrs.axis);

    if (!dims.is_contiguous_tail) {
        layer_norm_general_scalar<T>(
            x_ptr, s_ptr, b_ptr, y_ptr,
            dims.num_rows, dims.norm_size, dims.axis, X, scale,
            epsilon, add_to, has_bias, ctx);
        return;
    }

    const int64_t x_row_stride = X.row_stride_elems();
    const bool scale_is_scalar = (scale.numel() == 1);

    const auto process_row = [&](int64_t row) {
        const int64_t row_off = row * x_row_stride;
        auto [sum, sum_sq] = kernel::norm_reduce_sum_sq<T>(x_ptr + row_off, dims.norm_size);

        const float inv_n = 1.0f / static_cast<float>(dims.norm_size);
        const float mean_val = sum * inv_n;
        float var_val = sum_sq * inv_n - mean_val * mean_val;
        if (var_val < 0.0f) { var_val = 0.0f; }
        const float inv_std = 1.0f / std::sqrt(var_val + epsilon);

        if (has_bias && b_ptr) {
            kernel::norm_apply_row<T, true, true, true>(
                x_ptr + row_off, y_ptr + row_off, dims.norm_size,
                mean_val, inv_std, s_ptr, b_ptr, scale_is_scalar, add_to);
        } else {
            kernel::norm_apply_row<T, true, true, false>(
                x_ptr + row_off, y_ptr + row_off, dims.norm_size,
                mean_val, inv_std, s_ptr, nullptr, scale_is_scalar, add_to);
        }
    };

    ctx.cpu.run(0, dims.num_rows, process_row);
}

// ============================================================
// RMSNorm
// ============================================================

template <typename T>
void rms_norm_general_scalar(
    const T* x_ptr, const T* s_ptr,
    T* y_ptr,
    int64_t num_rows, int64_t norm_size,
    int64_t axis, const TensorView& X, const TensorView& scale,
    float epsilon, bool add_to,
    const ComputeContext& ctx)
{
    const auto inner_offsets = compute_inner_offsets(X, axis, norm_size);
    const int64_t outer_stride = (axis > 0) ? X.stride_elems(axis - 1) : 0;
    const bool scale_is_scalar = (scale.numel() == 1);

    const auto process_row = [&](int64_t row) {
        const int64_t row_base = row * outer_stride;

        float sum_sq = 0.0f;
        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t off = row_base + inner_offsets[static_cast<size_t>(i)];
            float x = s_load(&x_ptr[off]);
            sum_sq += x * x;
        }

        const float rms = std::sqrt(sum_sq / static_cast<float>(norm_size) + epsilon);
        const float inv_rms = 1.0f / rms;

        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t off = row_base + inner_offsets[static_cast<size_t>(i)];
            int64_t s_idx = scale_is_scalar ? 0 : i;
            float s = s_load(&s_ptr[s_idx]);
            float val = s_load(&x_ptr[off]) * inv_rms * s;
            s_store_add(&y_ptr[off], val, add_to);
        }
    };

    ctx.cpu.run(0, num_rows, process_row);
}

template <typename T>
void rms_norm_impl(const NormAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx)
{
    const auto& X     = inputs[0];
    const auto& scale = inputs[1];

    const float epsilon = attrs.epsilon;

    const auto* x_ptr = X.ptr<T>();
    const auto* s_ptr = scale.ptr<T>();
    auto* y_ptr = output.ptr<T>();

    const bool add_to = attrs.add_to;
    const auto dims = compute_norm_dims(X, attrs.axis);

    if (!dims.is_contiguous_tail) {
        rms_norm_general_scalar<T>(
            x_ptr, s_ptr, y_ptr,
            dims.num_rows, dims.norm_size, dims.axis, X, scale,
            epsilon, add_to, ctx);
        return;
    }

    const int64_t x_row_stride = X.row_stride_elems();
    const bool scale_is_scalar = (scale.numel() == 1);

    const auto process_row = [&](int64_t row) {
        const int64_t row_off = row * x_row_stride;
        auto [sum, sum_sq] = kernel::norm_reduce_sum_sq<T>(x_ptr + row_off, dims.norm_size);
        (void)sum;
        const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(dims.norm_size) + epsilon);
        kernel::norm_apply_row<T, false, true, false>(
            x_ptr + row_off, y_ptr + row_off, dims.norm_size,
            0.0f, inv_rms, s_ptr, nullptr, scale_is_scalar, add_to);
    };

    ctx.cpu.run(0, dims.num_rows, process_row);
}

// ============================================================
// L2Norm
// ============================================================

template <typename T>
void l2_norm_general_scalar(
    const T* x_ptr, T* y_ptr,
    int64_t num_rows, int64_t norm_size,
    int64_t axis, const TensorView& X,
    float epsilon, bool add_to,
    const ComputeContext& ctx)
{
    const auto inner_offsets = compute_inner_offsets(X, axis, norm_size);
    const int64_t outer_stride = (axis > 0) ? X.stride_elems(axis - 1) : 0;

    const auto process_row = [&](int64_t row) {
        const int64_t row_base = row * outer_stride;

        float sum_sq = 0.0f;
        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t off = row_base + inner_offsets[static_cast<size_t>(i)];
            float x = s_load(&x_ptr[off]);
            sum_sq += x * x;
        }

        const float norm_val = std::sqrt(sum_sq + epsilon);
        const float inv_norm = 1.0f / norm_val;

        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t off = row_base + inner_offsets[static_cast<size_t>(i)];
            float val = s_load(&x_ptr[off]) * inv_norm;
            s_store_add(&y_ptr[off], val, add_to);
        }
    };

    ctx.cpu.run(0, num_rows, process_row);
}

template <typename T>
void l2_norm_impl(const NormAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx)
{
    const auto& X = inputs[0];

    const float epsilon = attrs.epsilon;

    const auto* x_ptr = X.ptr<T>();
    auto* y_ptr = output.ptr<T>();

    const bool add_to = attrs.add_to;
    const auto dims = compute_norm_dims(X, attrs.axis);

    if (!dims.is_contiguous_tail) {
        l2_norm_general_scalar<T>(
            x_ptr, y_ptr,
            dims.num_rows, dims.norm_size, dims.axis, X,
            epsilon, add_to, ctx);
        return;
    }

    const int64_t x_row_stride = X.row_stride_elems();

    const auto process_row = [&](int64_t row) {
        const int64_t row_off = row * x_row_stride;
        auto [sum, sum_sq] = kernel::norm_reduce_sum_sq<T>(x_ptr + row_off, dims.norm_size);
        (void)sum;
        const float inv_norm = 1.0f / std::sqrt(sum_sq + epsilon);
        kernel::norm_apply_row<T, false, false, false>(
            x_ptr + row_off, y_ptr + row_off, dims.norm_size,
            0.0f, inv_norm, nullptr, nullptr, false, add_to);
    };

    ctx.cpu.run(0, dims.num_rows, process_row);
}

// ============================================================
// GroupNorm
// ============================================================

template <typename T>
void group_norm_impl(const NormAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx)
{
    const auto& X     = inputs[0];
    const auto& scale = inputs[1];
    const bool has_bias = (inputs.size() >= 3 && !inputs[2].is_empty());

    const int64_t rank = X.rank();
    NNOPS_ASSERT(rank >= 2);

    const int64_t N = X.shape(0);
    const int64_t C = X.shape(1);
    int64_t G = attrs.num_groups;
    if (G <= 0) {
        G = 1;
    }
    NNOPS_ASSERT(C % G == 0);

    const int64_t channels_per_group = C / G;

    int64_t spatial_size = 1;
    for (int64_t i = 2; i < rank; ++i) {
        spatial_size *= X.shape(i);
    }
    const int64_t W = (rank >= 3) ? X.shape(rank - 1) : int64_t(1);
    const int64_t rows_per_channel = spatial_size / W;

    const int64_t norm_size = channels_per_group * spatial_size;
    const float inv_norm = 1.0f / static_cast<float>(norm_size);
    const float epsilon = attrs.epsilon;

    const auto* x_ptr  = X.ptr<T>();
    const auto* s_ptr  = scale.ptr<T>();
    const auto* b_ptr  = (has_bias && !inputs[2].is_empty()) ? inputs[2].ptr<T>() : nullptr;
    auto* y_ptr = output.ptr<T>();

    const int64_t stride_N = X.stride_elems(0);
    const int64_t stride_C = X.stride_elems(1);
    const int64_t row_stride = X.row_stride_elems();

    const int64_t total_groups = N * G;
    const int64_t total_rows_per_group = channels_per_group * rows_per_channel;
    const bool add_to = attrs.add_to;

    const auto process_group = [&](int64_t ng) {
        const int64_t n = ng / G;
        const int64_t g = ng % G;
        const int64_t c_start = g * channels_per_group;

        const int64_t group_offset = n * stride_N + c_start * stride_C;

        float sum = 0.0f;
        float sum_sq = 0.0f;

        for (int64_t row = 0; row < total_rows_per_group; ++row) {
            const int64_t row_off = group_offset + row * row_stride;
            auto [s, sq] = kernel::norm_reduce_sum_sq<T>(x_ptr + row_off, W);
            sum += s;
            sum_sq += sq;
        }

        const float mean_val = sum * inv_norm;
        float var_val = sum_sq * inv_norm - mean_val * mean_val;
        if (var_val < 0.0f) { var_val = 0.0f; }
        const float inv_std = 1.0f / std::sqrt(var_val + epsilon);

        for (int64_t row = 0; row < total_rows_per_group; ++row) {
            const int64_t c_global = c_start + row / rows_per_channel;
            const int64_t row_off = group_offset + row * row_stride;

            const float s = s_load(&s_ptr[c_global]);
            const float b = (has_bias && b_ptr) ? s_load(&b_ptr[c_global]) : 0.0f;

            kernel::norm_apply_affine_row<T>(x_ptr + row_off, y_ptr + row_off, W,
                                  mean_val, inv_std, s, b, add_to);
        }
    };

    ctx.cpu.run(0, total_groups, process_group);
}

// ============================================================
// Quantized input (s8/u8): LayerNorm / RMSNorm / L2Norm
// ============================================================
//
// The integer row is dequantized to f32 by the arch quant.hpp per-row kernel and
// the existing SIMD norm row kernels then run on that row buffer, so no new
// kernel is needed. The compute type is always f32 — an f16 output is narrowed
// on store, matching what the f16 float path does internally anyway.
//
// This is only defined for axis == rank-1, where one normalization row is
// exactly one quantization row (the unit of a PerToken scale/zero_point) —
// that is what lets a single row be dequantized, normalized and written back
// without staging the whole tensor. PerChannel / PerBlock have no meaning over
// the trailing axis, and BatchNorm / GroupNorm are not supported because their
// reduction units don't line up with quantization rows.
//
// Row buffer: an f32 output is dequantized and normalized in place in the output
// row, so it needs no scratch at all. An f16 or s8/u8 output stages each row in
// a pooled f32 buffer and converts on write-back — requantizing with the output
// TensorView's own (scale, zero_point). The staging buffer comes from the CPU
// memory pool, so getWorkspaceSize stays 0.

/// Resolve the (scale, zero_point) of row @p row: the per-row buffers when
/// present, else the per-tensor scalars. Matches TensorView's PerToken /
/// PerTensor convention (a missing zero_point buffer means zero).
inline void row_qparam(const QuantParams& qp, int64_t row, float& scale, float& zp)
{
    if (qp.scale_data != nullptr) {
        scale = qp.scale_data[row];
        zp = (qp.zero_point_data != nullptr)
                 ? static_cast<float>(qp.zero_point_data[row])
                 : 0.0f;
    } else {
        scale = qp.scale;
        zp = static_cast<float>(qp.zero_point);
    }
}

/// Dequantize one contiguous row of `n` s8/u8 values into a T row.
template <typename T, typename InT>
inline void dequant_row(T* dst, const InT* src, int64_t n, float scale, float zp)
{
    const int n_i = static_cast<int>(n);
    quant_kernel::dequantization<InT>(1, n_i, dst, n_i, src, n_i, &scale, &zp);
}

/// Requantize one contiguous row of `n` f32 values into an s8/u8 row.
template <typename OutT>
inline void requant_row(OutT* dst, const float* src, int64_t n, float scale, float zp)
{
    const int n_i = static_cast<int>(n);
    quant_kernel::quantization<OutT>(1, n_i, dst, n_i, src, n_i, &scale, &zp);
}

/// Per-row fused kernel: dequantize row -> normalize in place -> write back.
/// `HasMean` / `HasScale` select the LayerNorm / RMSNorm / L2Norm variants.
template <bool HasMean, bool HasScale>
void norm_quant_rows(const NormAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     bool has_bias)
{
    const auto& X = inputs[0];

    // L2Norm has no scale input; HasScale is false for it, so s_ptr and
    // scale_is_scalar are never used there — but they must not be formed from
    // an out-of-range inputs[1].
    const bool has_scale = (inputs.size() >= 2);
    const auto* s_ptr = has_scale ? inputs[1].ptr<float>() : nullptr;
    const bool scale_is_scalar = has_scale && (inputs[1].numel() == 1);

    const DataType x_dtype = X.data_type();
    const DataType y_dtype = output.data_type();
    const bool out_is_f32  = (y_dtype == DataType::f32);
    const bool out_is_f16  = (y_dtype == DataType::f16);
    const bool is_i8       = (x_dtype == DataType::s8);

    const QuantParams& x_qp = X.quant_params();

    const float epsilon = attrs.epsilon;
    const auto dims = compute_norm_dims(X, attrs.axis);
    NNOPS_ASSERT(dims.is_contiguous_tail);

    const int64_t n = dims.norm_size;

    // scale / bias are f32 on this path (asserted in norm_quant_impl).
    const auto* b_ptr = (has_bias && !inputs[2].is_empty()) ? inputs[2].ptr<float>() : nullptr;

    const auto process_row = [&](int64_t row) {
        const size_t row_idx = static_cast<size_t>(row);

        float x_scale, x_zp;
        row_qparam(x_qp, row, x_scale, x_zp);

        // f32 output: dequantize and normalize straight in the output row, no
        // scratch. f16 / s8 / u8 output: stage the row in a pooled f32 buffer.
        PoolPtr rowbuf_owner;
        float* rowbuf;
        if (out_is_f32) {
            rowbuf = output.ptr<float>(row_idx);
        } else {
            rowbuf_owner = PoolPtr(static_cast<size_t>(n) * sizeof(float));
            rowbuf = rowbuf_owner.as<float>();
        }

        const void* x_row = X.ptr<uint8_t>(row_idx);
        if (is_i8) {
            dequant_row<float>(rowbuf, static_cast<const int8_t*>(x_row), n, x_scale, x_zp);
        } else {
            dequant_row<float>(rowbuf, static_cast<const uint8_t*>(x_row), n, x_scale, x_zp);
        }

        auto [sum, sum_sq] = kernel::norm_reduce_sum_sq<float>(rowbuf, n);
        const float inv_n = 1.0f / static_cast<float>(n);
        float mean_val = 0.0f;
        float inv_norm;
        if constexpr (HasMean) {          // LayerNorm
            mean_val = sum * inv_n;
            float var_val = sum_sq * inv_n - mean_val * mean_val;
            if (var_val < 0.0f) { var_val = 0.0f; }
            inv_norm = 1.0f / std::sqrt(var_val + epsilon);
        } else if constexpr (HasScale) {  // RMSNorm
            inv_norm = 1.0f / std::sqrt(sum_sq * inv_n + epsilon);
        } else {                          // L2Norm
            inv_norm = 1.0f / std::sqrt(sum_sq + epsilon);
        }

        // Normalize in place — x and y share the row buffer. add_to is rejected
        // on the quantized path, so the plain (overwrite) store is used.
        if constexpr (HasScale) {
            if (b_ptr != nullptr) {
                kernel::norm_apply_row<float, HasMean, true, true>(
                    rowbuf, rowbuf, n, mean_val, inv_norm, s_ptr, b_ptr, scale_is_scalar, false);
            } else {
                kernel::norm_apply_row<float, HasMean, true, false>(
                    rowbuf, rowbuf, n, mean_val, inv_norm, s_ptr, nullptr, scale_is_scalar, false);
            }
        } else {
            kernel::norm_apply_row<float, HasMean, false, false>(
                rowbuf, rowbuf, n, mean_val, inv_norm, nullptr, nullptr, false, false);
        }

        if (out_is_f32) { return; }  // already written in place

        if (out_is_f16) {
            half* y_row = output.ptr<half>(row_idx);
            for (int64_t i = 0; i < n; ++i) {
                s_store(&y_row[i], rowbuf[i]);
            }
            return;
        }

        float y_scale, y_zp;
        row_qparam(output.quant_params(), row, y_scale, y_zp);
        if (y_dtype == DataType::s8) {
            requant_row<int8_t>(output.ptr<int8_t>(row_idx), rowbuf, n, y_scale, y_zp);
        } else {
            requant_row<uint8_t>(output.ptr<uint8_t>(row_idx), rowbuf, n, y_scale, y_zp);
        }
    };

    ctx.cpu.run(0, dims.num_rows, process_row);
}

/// Dispatch the quantized path on the norm type (LayerNorm / RMSNorm / L2Norm).
void norm_quant_dispatch(const NormAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         bool has_bias)
{
    switch (attrs.type) {
    case NormType::LayerNorm:
        norm_quant_rows<true, true>(attrs, output, inputs, ctx, has_bias);
        return;
    case NormType::RMSNorm:
        norm_quant_rows<false, true>(attrs, output, inputs, ctx, false);
        return;
    case NormType::L2Norm:
        norm_quant_rows<false, false>(attrs, output, inputs, ctx, false);
        return;
    default:
        break;
    }
    NNOPS_ASSERT(!"norm_cpu: quantized input is only supported for LayerNorm/RMSNorm/L2Norm");
}

/// Validate a quantized-input call and dispatch it.
void norm_quant_impl(const NormAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx)
{
    const auto& X = inputs[0];

    // add_to would require dequantizing, accumulating and requantizing the
    // pre-existing output — not supported on this path.
    NNOPS_ASSERT(!attrs.add_to);

    NNOPS_ASSERT(X.is_quantized());
    NNOPS_ASSERT(X.channel_pack_size() == 1);  // planar layouts only

    const auto& x_qp = X.quant_params();
    NNOPS_ASSERT(x_qp.granularity == QuantGranularity::PerTensor
              || x_qp.granularity == QuantGranularity::PerToken);

    const auto dims = compute_norm_dims(X, attrs.axis);
    NNOPS_ASSERT_MSG(dims.is_contiguous_tail,
                     "norm_cpu: quantized input requires axis == rank-1");
    if (x_qp.granularity == QuantGranularity::PerToken) {
        NNOPS_ASSERT(x_qp.scale_data != nullptr);
        NNOPS_ASSERT(x_qp.num_scales >= dims.num_rows);
    }

    const DataType y_dtype = output.data_type();
    NNOPS_ASSERT(y_dtype == DataType::f32 || y_dtype == DataType::f16
              || is_quantized_dtype(y_dtype));
    if (is_quantized_dtype(y_dtype)) {
        NNOPS_ASSERT(output.is_quantized());
        const auto& y_qp = output.quant_params();
        NNOPS_ASSERT(y_qp.granularity == QuantGranularity::PerTensor
                  || y_qp.granularity == QuantGranularity::PerToken);
        if (y_qp.granularity == QuantGranularity::PerToken) {
            NNOPS_ASSERT(y_qp.scale_data != nullptr);
            NNOPS_ASSERT(y_qp.num_scales >= dims.num_rows);
        }
    }

    // scale (LayerNorm / RMSNorm) and bias stay f32 on this path; L2Norm has
    // neither, so inputs[1] may not exist.
    if (inputs.size() >= 2) { NNOPS_ASSERT(inputs[1].data_type() == DataType::f32); }
    const bool has_bias = (inputs.size() >= 3 && !inputs[2].is_empty());
    if (has_bias) { NNOPS_ASSERT(inputs[2].data_type() == DataType::f32); }

    // The compute type is always f32; the output dtype only picks the store.
    norm_quant_dispatch(attrs, output, inputs, ctx, has_bias);
}

}  // anonymous namespace

// ============================================================
// Dispatcher + entry point with dtype dispatch
// ============================================================

template <typename T>
void norm_impl(const NormAttributes& attrs,
               TensorView& output,
               std::span<const TensorView> inputs,
               const ComputeContext& ctx)
{
    switch (attrs.type) {
    case NormType::BatchNorm:
        batch_norm_impl<T>(attrs, output, inputs, ctx);
        return;
    case NormType::LayerNorm:
        layer_norm_impl<T>(attrs, output, inputs, ctx);
        return;
    case NormType::RMSNorm:
        rms_norm_impl<T>(attrs, output, inputs, ctx);
        return;
    case NormType::GroupNorm:
        group_norm_impl<T>(attrs, output, inputs, ctx);
        return;
    case NormType::L2Norm:
        l2_norm_impl<T>(attrs, output, inputs, ctx);
        return;
    }
    NNOPS_ASSERT(!"norm_cpu: unknown norm type");
}

void norm_cpu(const NormAttributes& attrs,
              TensorView& output,
              std::span<const TensorView> inputs,
              const ComputeContext& ctx,
              void* /*workspace*/)
{
    const DataType x_dtype = inputs[0].data_type();

    // s8/u8 input: dequantize per row and run the float norm kernels on it.
    // Scratch is pooled internally, so getWorkspaceSize stays 0.
    if (is_quantized_dtype(x_dtype)) {
        norm_quant_impl(attrs, output, inputs, ctx);
        return;
    }

    dispatch_f32_f16(x_dtype, "norm_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        norm_impl<T>(attrs, output, inputs, ctx);
    });
}

}  // namespace nnops::backend::cpu
