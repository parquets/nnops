/// @file batch_norm.cpp
/// @brief SIMD-optimized CPU implementation of batch normalization (inference only).
///
/// Supports f32 and f16. Respects pitch via row_stride_elems().
///
/// Spatial mode (per-channel statistics):
///   Planar layouts (NCHW/NCDHW): delegates to kernel::batch_norm_process_planar_row.
///   Packed layouts (NCHWC8/NCDHWC8): delegates to kernel::batch_norm_process_packed_row.
///
/// Non-spatial mode: delegates to kernel::batch_norm_process_nonspatial_block.
///
/// Fused formula: new_scale = inv_std * scale, new_bias = bias - mean * new_scale,
///                y = x * new_scale + new_bias

#include "nnops/ops/batch_norm.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "simd_kernel/simd_norm.hpp"

#include <cmath>
#include <vector>

namespace nnops::backend::cpu {

using namespace nnops::simd;

// ============================================================
// Templated implementation (f32 and f16)
// ============================================================

template <typename T>
void batch_norm_impl(const BatchNormAttributes& attrs,
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
    constexpr int L = simd_lane_for<T>;

    // Compute N, C
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
        // ---- Spatial mode: per-channel statistics ----
        std::vector<float> ns_f(static_cast<size_t>(C));
        std::vector<float> nb_f(static_cast<size_t>(C));
        for (int64_t c = 0; c < C; ++c) {
            float inv_std = 1.0f / std::sqrt(s_load(&v_ptr[c]) + epsilon);
            ns_f[static_cast<size_t>(c)] = inv_std * s_load(&s_ptr[c]);
            nb_f[static_cast<size_t>(c)] = s_load(&b_ptr[c]) - s_load(&m_ptr[c]) * ns_f[static_cast<size_t>(c)];
        }

        const int64_t pack = X.channel_pack_size();

        if (pack > 1) {
            // ---- Packed layout path (NCHWC8, NCDHWC8) ----
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

            if (ctx.cpu_parallel_for) {
                ctx.cpu_parallel_for(0, N, compute_sample);
            }
            else {
                for (int64_t n = 0; n < N; ++n) {
                    compute_sample(n);
                }
            }
        }

    } else {
        // ---- Non-spatial mode: per-element statistics ----
        const int64_t total = X.numel();
        kernel::batch_norm_process_nonspatial_block<T>(
            x_ptr, y_ptr, s_ptr, b_ptr, m_ptr, v_ptr,
            0, total, epsilon, attrs.add_to);
    }
}

// ============================================================
// Entry point with dtype dispatch
// ============================================================

void batch_norm_cpu(const BatchNormAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        batch_norm_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        batch_norm_impl<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"batch_norm_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
