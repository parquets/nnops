/// @file batch_norm_ref.cpp
/// @brief Naive CPU reference implementation of batch normalization (inference only).
///
/// Supports planar (NCHW/NCDHW) and packed (NCHWC8/NCDHWC8) layouts.
/// Fused formula: new_scale = inv_std * scale, new_bias = bias - mean * new_scale.
///                y = x * new_scale + new_bias

#include "nnops/ops/batch_norm.hpp"
#include "nnops/core/parallel_for.hpp"

#include <cmath>
#include <algorithm>
#include <vector>

namespace nnops::backend::cpu::reference {

void batch_norm_ref(const BatchNormAttributes& attrs,
                    TensorView& output,
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

    int64_t N, C;
    if (rank == 1) { N = X.shape(0); C = 1; }
    else           { N = X.shape(0); C = X.shape(1); }

    const auto* x_ptr = X.ptr<float>();
    const auto* s_ptr = scale.ptr<float>();
    const auto* b_ptr = bias.ptr<float>();
    const auto* m_ptr = mean.ptr<float>();
    const auto* v_ptr = var.ptr<float>();
    auto* y_ptr = output.ptr<float>();

    if (attrs.spatial) {
        // ---- Spatial mode: per-channel statistics ----
        std::vector<float> ns_f(static_cast<size_t>(C));
        std::vector<float> nb_f(static_cast<size_t>(C));
        for (int64_t c = 0; c < C; ++c) {
            float inv_std = 1.0f / std::sqrt(v_ptr[c] + epsilon);
            ns_f[static_cast<size_t>(c)] = inv_std * s_ptr[c];
            nb_f[static_cast<size_t>(c)] = b_ptr[c] - m_ptr[c] * ns_f[static_cast<size_t>(c)];
        }

        const int64_t pack = X.channel_pack_size();

        if (pack > 1) {
            // ---- Packed layout path (NCHWC8, NCDHWC8) ----
            const int64_t c8_blocks = X.num_channel_blocks();
            std::vector<float> ns_packed(static_cast<size_t>(c8_blocks * pack), 0.0f);
            std::vector<float> nb_packed(static_cast<size_t>(c8_blocks * pack), 0.0f);
            for (int64_t c = 0; c < C; ++c) {
                ns_packed[static_cast<size_t>(c)] = ns_f[static_cast<size_t>(c)];
                nb_packed[static_cast<size_t>(c)] = nb_f[static_cast<size_t>(c)];
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
                const float* ns = &ns_packed[c8 * pack];
                const float* nb = &nb_packed[c8 * pack];
                const float* x_row = x_ptr + r * x_rs;
                float* y_row = y_ptr + r * y_rs;
                for (int64_t i = 0; i < last_dim; ++i) {
                    float val = x_row[i] * ns[i % pack] + nb[i % pack];
                    y_row[i] = attrs.add_to ? y_row[i] + val : val;
                }
            };

            if (ctx.cpu_parallel_for) {
                ctx.cpu_parallel_for(0, num_rows, process_row);
            }
            else {
                for (int64_t r = 0; r < num_rows; ++r) {
                    process_row(r);
                }
            }

        } else {
            // ---- Planar layout path (NCHW, NCDHW) ----
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

            const int64_t spatial_total = N * C;
            const auto body = [&](int64_t idx) {
                int64_t n = idx / C;
                int64_t c = idx % C;
                const float ns = ns_f[static_cast<size_t>(c)];
                const float nb = nb_f[static_cast<size_t>(c)];
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
                for (int64_t i = 0; i < spatial_total; ++i) { body(i); }
            }
        }
    } else {
        // ---- Non-spatial: per-element statistics ----
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
            for (int64_t i = 0; i < total; ++i) { body(i); }
        }
    }
}

}  // namespace nnops::backend::cpu::reference
