/// @file batch_norm_ref.cpp
/// @brief Naive CPU reference implementation of batch normalization (inference only).
///
/// Uses the fused formula:
///   new_scale = inv_std * scale
///   new_bias  = bias - mean * new_scale
///   y = x * new_scale + new_bias

#include "nnops/ops/batch_norm.hpp"
#include "nnops/core/parallel_for.hpp"

#include <cmath>
#include <algorithm>
#include <vector>

namespace nnops::backend::cpu::reference {

void batch_norm_ref(const BatchNormAttributes& attrs,
                    const TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* /*workspace*/)
{
    const auto& X     = inputs[0];
    const auto& scale = inputs[1];
    const auto& bias  = inputs[2];
    const auto& mean  = inputs[3];
    const auto& var   = inputs[4];

    const int64_t rank = X.rank();
    const float epsilon = attrs.epsilon;

    // Compute N, C, and sample_size
    int64_t N, C, sample_size;
    if (rank == 1) {
        N = X.shape(0);
        C = 1;
        sample_size = 1;
    } else {
        N = X.shape(0);
        C = X.shape(1);
        sample_size = 1;
        for (int64_t i = 2; i < rank; ++i) {
            sample_size *= X.shape(i);
        }
    }

    const auto* x_ptr  = X.data_as<float>();
    const auto* s_ptr  = scale.data_as<float>();
    const auto* b_ptr  = bias.data_as<float>();
    const auto* m_ptr  = mean.data_as<float>();
    const auto* v_ptr  = var.data_as<float>();
    auto* y_ptr = output.data_as<float>();

    if (attrs.spatial) {
        // Spatial mode: per-channel statistics
        // Precompute new_scale and new_bias per channel
        std::vector<float> new_scale(static_cast<size_t>(C));
        std::vector<float> new_bias(static_cast<size_t>(C));
        for (int64_t c = 0; c < C; ++c) {
            float inv_std = 1.0f / std::sqrt(v_ptr[c] + epsilon);
            new_scale[static_cast<size_t>(c)] = inv_std * s_ptr[c];
            new_bias[static_cast<size_t>(c)]  = b_ptr[c] - m_ptr[c] * new_scale[static_cast<size_t>(c)];
        }

        // Use stride_elems to compute offsets (accounts for pitch)
        const int64_t x_n_stride = X.stride_elems(0);
        const int64_t x_c_stride = (rank >= 2) ? X.stride_elems(1) : 1;
        const int64_t y_n_stride = output.stride_elems(0);
        const int64_t y_c_stride = (rank >= 2) ? output.stride_elems(1) : 1;

        // Spatial dims (for rank >= 3)
        const int64_t last_dim = (rank >= 3) ? X.shape(rank - 1) : 1;
        const int64_t num_rows = (rank >= 3) ? sample_size / last_dim : 1;
        const int64_t x_row_stride = X.row_stride_elems();
        const int64_t y_row_stride = output.row_stride_elems();

        const int64_t spatial_total = N * C;
        // Per-channel compute
        const auto body = [&](int64_t idx) {
            int64_t n = idx / C;
            int64_t c = idx % C;
            const float ns = new_scale[static_cast<size_t>(c)];
            const float nb = new_bias[static_cast<size_t>(c)];
            const int64_t x_ch_base = n * x_n_stride + c * x_c_stride;
            const int64_t y_ch_base = n * y_n_stride + c * y_c_stride;
            for (int64_t hh = 0; hh < num_rows; ++hh) {
                const int64_t x_row = x_ch_base + hh * x_row_stride;
                const int64_t y_row = y_ch_base + hh * y_row_stride;
                for (int64_t w = 0; w < last_dim; ++w) {
                    float val = x_ptr[x_row + w] * ns + nb;
                    y_ptr[y_row + w] = attrs.add_to ? y_ptr[y_row + w] + val : val;
                }
            }
        };

        if (ctx.cpu_parallel_for) {
            ctx.cpu_parallel_for(0, spatial_total, body);
        } else {
            for (int64_t i = 0; i < spatial_total; ++i) body(i);
        }
    } else {
        // Non-spatial: per-element statistics (scale/bias/mean/var have same shape as X)
        const int64_t total = X.numel();
        const auto body = [&](int64_t i) {
            float inv_std = 1.0f / std::sqrt(v_ptr[i] + epsilon);
            float ns = inv_std * s_ptr[i];
            float nb = b_ptr[i] - m_ptr[i] * ns;
            float val = x_ptr[i] * ns + nb;
            y_ptr[i] = attrs.add_to ? y_ptr[i] + val : val;
        };

        if (ctx.cpu_parallel_for) {
            ctx.cpu_parallel_for(0, total, body);
        } else {
            for (int64_t i = 0; i < total; ++i) body(i);
        }
    }
}

}  // namespace nnops::backend::cpu::reference
