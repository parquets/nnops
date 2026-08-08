/// @file group_norm_ref.cpp
/// @brief Naive CPU reference implementation of group normalization.
///
/// Uses Welford's online algorithm for numerically stable mean/variance.
/// Normalizes over group-of-channels + spatial dims, broadcasts scale/bias per-channel.

#include "nnops/ops/group_norm.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"

#include <cmath>

namespace nnops::backend::cpu::reference {

namespace {

void group_norm_ref_impl(const GroupNormAttributes& attrs,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx)
{
    const auto& X = inputs[0];
    const auto& scale = inputs[1];
    const bool has_bias = (inputs.size() >= 3 && !inputs[2].is_empty());

    const int64_t rank = X.rank();
    const int64_t N = X.shape(0);
    const int64_t C = X.shape(1);
    int64_t G = attrs.num_groups;
    if (G <= 0) {
        G = 1;
    }
    NNOPS_ASSERT(C % G == 0);

    const int64_t channels_per_group = C / G;
    const float epsilon = attrs.epsilon;

    // Compute spatial size
    int64_t spatial_size = 1;
    for (int64_t i = 2; i < rank; ++i) {
        spatial_size *= X.shape(i);
    }
    const int64_t norm_size = channels_per_group * spatial_size;
    const float inv_norm = 1.0f / static_cast<float>(norm_size);

    const auto* x_ptr  = X.ptr<float>();
    const auto* s_ptr  = scale.ptr<float>();
    const auto* b_ptr  = has_bias ? inputs[2].ptr<float>() : nullptr;
    auto* y_ptr = output.ptr<float>();

    // Stride of a channel plane in elements
    const int64_t stride_C = X.stride_elems(1);
    const int64_t stride_N = X.stride_elems(0);
    const int64_t row_stride = X.row_stride_elems();

    // Number of rows per channel plane (H for 2D, D*H for 3D)
    int64_t rows_per_channel = 1;
    for (int64_t i = 2; i < rank - 1; ++i) {
        rows_per_channel *= X.shape(i);
    }
    const int64_t W = (rank >= 3) ? X.shape(rank - 1) : 1;

    const int64_t total_groups = N * G;

    const auto process_group = [&](int64_t ng) {
        const int64_t n = ng / G;
        const int64_t g = ng % G;
        const int64_t c_start = g * channels_per_group;
        const int64_t x_offset = n * stride_N + c_start * stride_C;

        // ---- Pass 1: Welford online mean/variance ----
        float mean_val = 0.0f;
        float M2 = 0.0f;
        int64_t count = 0;

        for (int64_t c = 0; c < channels_per_group; ++c) {
            const int64_t c_off = c * stride_C;
            for (int64_t r = 0; r < rows_per_channel; ++r) {
                const int64_t row_off = x_offset + c_off + r * row_stride;
                for (int64_t w = 0; w < W; ++w) {
                    float x = x_ptr[row_off + w];
                    count++;
                    float delta = x - mean_val;
                    mean_val += delta / static_cast<float>(count);
                    float delta2 = x - mean_val;
                    M2 += delta * delta2;
                }
            }
        }
        const float var_val = M2 * inv_norm;
        const float inv_std = 1.0f / std::sqrt(var_val + epsilon);

        // ---- Pass 2: normalize ----
        for (int64_t c = 0; c < channels_per_group; ++c) {
            const int64_t c_global = c_start + c;
            const int64_t c_off = c * stride_C;
            const float s = s_ptr[c_global];
            const float b = (has_bias && b_ptr) ? b_ptr[c_global] : 0.0f;

            for (int64_t r = 0; r < rows_per_channel; ++r) {
                const int64_t row_off = x_offset + c_off + r * row_stride;
                for (int64_t w = 0; w < W; ++w) {
                    const int64_t idx = row_off + w;
                    float val = (x_ptr[idx] - mean_val) * inv_std * s + b;
                    if (attrs.add_to) {
                        y_ptr[idx] += val;
                    } else {
                        y_ptr[idx] = val;
                    }
                }
            }
        }
    };

    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, total_groups, process_group);
    } else {
        for (int64_t i = 0; i < total_groups; ++i) {
            process_group(i);
        }
    }
}

}  // anonymous namespace

void group_norm_ref(const GroupNormAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* /*workspace*/)
{
    // Reference uses f32 regardless of input dtype for simplicity
    group_norm_ref_impl(attrs, output, inputs, ctx);
}

}  // namespace nnops::backend::cpu::reference
