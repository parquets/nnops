/// @file group_norm.cpp
/// @brief SIMD-optimized CPU implementation of group normalization.
///
/// Supports both f32 and f16 via a single templated implementation.
///
/// Algorithm per (N, G) normalization unit:
///   Pass 1 — SIMD reduce sum/sum_sq over all channel-rows within the group.
///            Each row is processed with 4-wide multi-accumulator SIMD reduction.
///   Pass 2 — SIMD normalize each row: y = (x-mean)*inv_std*scale_c + bias_c.
///
/// The data within a group is contiguous in planar layout:
///   - Group data = channels_per_group consecutive channel planes
///   - Each channel plane = spatial_rows × row_stride elements
///   - Each row has W contiguous elements (the innermost dimension)
///
/// Parallel dispatch: cpu_parallel_for over N*G groups.

#include "nnops/ops/group_norm.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "simd_kernel/simd_norm.hpp"

#include <cmath>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

template <typename T>
void group_norm_impl(const GroupNormAttributes& attrs,
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

    // Compute spatial size and row decomposition
    int64_t spatial_size = 1;
    for (int64_t i = 2; i < rank; ++i) {
        spatial_size *= X.shape(i);
    }
    // W = innermost dimension size (contiguous elements per row)
    const int64_t W = (rank >= 3) ? X.shape(rank - 1) : int64_t(1);
    const int64_t rows_per_channel = spatial_size / W;

    const int64_t norm_size = channels_per_group * spatial_size;
    const float inv_norm = 1.0f / static_cast<float>(norm_size);
    const float epsilon = attrs.epsilon;

    const auto* x_ptr  = X.ptr<T>();
    const auto* s_ptr  = scale.ptr<T>();
    const auto* b_ptr  = (has_bias && !inputs[2].is_empty()) ? inputs[2].ptr<T>() : nullptr;
    auto* y_ptr = output.ptr<T>();

    // Strides
    const int64_t stride_N = X.stride_elems(0);
    const int64_t stride_C = X.stride_elems(1);
    const int64_t row_stride = X.row_stride_elems();

    const int64_t total_groups = N * G;
    const int64_t total_rows_per_group = channels_per_group * rows_per_channel;
    const bool add_to = attrs.add_to;

    // ---- Per-group processing (parallel over N*G) ----
    const auto process_group = [&](int64_t ng) {
        const int64_t n = ng / G;
        const int64_t g = ng % G;
        const int64_t c_start = g * channels_per_group;

        const int64_t group_offset = n * stride_N + c_start * stride_C;

        // ---- Pass 1: SIMD reduction over all rows in the group ----
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

        // ---- Pass 2: SIMD normalize each row with per-channel scale/bias ----
        for (int64_t row = 0; row < total_rows_per_group; ++row) {
            const int64_t c_global = c_start + row / rows_per_channel;
            const int64_t row_off = group_offset + row * row_stride;

            const float s = s_load(&s_ptr[c_global]);
            const float b = (has_bias && b_ptr) ? s_load(&b_ptr[c_global]) : 0.0f;

            kernel::norm_apply_affine_row<T>(x_ptr + row_off, y_ptr + row_off, W,
                                  mean_val, inv_std, s, b, add_to);
        }
    };

    // ---- Parallel dispatch over N*G ----
    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, total_groups, process_group);
    } else {
        for (int64_t i = 0; i < total_groups; ++i) {
            process_group(i);
        }
    }
}

}  // anonymous namespace

// ============================================================
// Entry point with dtype dispatch
// ============================================================

void group_norm_cpu(const GroupNormAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        group_norm_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        group_norm_impl<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"group_norm_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
