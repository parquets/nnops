/// @file norm.cpp
/// @brief SIMD-optimized CPU implementation of normalization operators.
///
/// Covers BatchNorm, LayerNorm, RMSNorm, GroupNorm, and L2Norm, dispatched
/// on NormAttributes::type. All share the consolidated SIMD kernels in
/// simd_kernel/simd_norm.hpp.
///
/// Supports f32 and f16 via a single templated implementation per type.

#include "nnops/ops/norm.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "simd_kernel/simd_norm.hpp"

#include <cmath>
#include <vector>

namespace nnops::backend::cpu {

using namespace nnops::simd;

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
    const int64_t rank = X.rank();
    std::vector<int64_t> offsets(static_cast<size_t>(norm_size));
    for (int64_t flat = 0; flat < norm_size; ++flat) {
        int64_t off = 0;
        int64_t rem = flat;
        for (int64_t d = rank - 1; d >= axis; --d) {
            int64_t dim = X.shape(d);
            off += (rem % dim) * X.stride_elems(d);
            rem /= dim;
        }
        offsets[static_cast<size_t>(flat)] = off;
    }
    return offsets;
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
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        norm_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        norm_impl<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"norm_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
