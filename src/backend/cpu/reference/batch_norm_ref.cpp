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

        const int64_t spatial_total = N * C * sample_size;
        const auto body = [&](int64_t i) {
            int64_t c = (i / sample_size) % C;
            y_ptr[i] = x_ptr[i] * new_scale[static_cast<size_t>(c)] + new_bias[static_cast<size_t>(c)];
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
            y_ptr[i] = x_ptr[i] * ns + nb;
        };

        if (ctx.cpu_parallel_for) {
            ctx.cpu_parallel_for(0, total, body);
        } else {
            for (int64_t i = 0; i < total; ++i) body(i);
        }
    }
}

}  // namespace nnops::backend::cpu::reference
