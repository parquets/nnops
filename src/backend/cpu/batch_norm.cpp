/// @file batch_norm.cpp
/// @brief SIMD-optimized CPU implementation of batch normalization (inference only).
///
/// Vectorized with the nnops SIMD abstraction layer. On x86_64 this compiles to
/// AVX2+FMA (8-wide) by default; on AArch64 to NEON (v_f32x8 emulated); on RISC-V
/// to V extension.  simd_default_lane_f32 replaces hardcoded width for portability.
///
/// Fused formula:
///   new_scale = inv_std * scale
///   new_bias  = bias - mean * new_scale
///   y = x * new_scale + new_bias

#include "nnops/ops/batch_norm.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <cmath>
#include <vector>

namespace nnops::backend::cpu {

void batch_norm_cpu(const BatchNormAttributes& attrs,
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

    using namespace nnops::simd;

    constexpr int L = simd_default_lane_f32;  // 8

    if (attrs.spatial) {
        // ---- Spatial mode: per-channel statistics ----
        // Precompute new_scale and new_bias per channel.
        std::vector<float> new_scale(static_cast<size_t>(C));
        std::vector<float> new_bias(static_cast<size_t>(C));
        for (int64_t c = 0; c < C; ++c) {
            float inv_std = 1.0f / std::sqrt(v_ptr[c] + epsilon);
            new_scale[static_cast<size_t>(c)] = inv_std * s_ptr[c];
            new_bias[static_cast<size_t>(c)]  = b_ptr[c] - m_ptr[c] * new_scale[static_cast<size_t>(c)];
        }

        const int64_t x_n_stride = X.stride_elems(0);
        const int64_t x_c_stride = (rank >= 2) ? X.stride_elems(1) : 1;
        const int64_t y_n_stride = output.stride_elems(0);
        const int64_t y_c_stride = (rank >= 2) ? output.stride_elems(1) : 1;

        // Spatial dims (for rank >= 3)
        const int64_t last_dim = (rank >= 3) ? X.shape(rank - 1) : 1;
        const int64_t num_rows = (rank >= 3) ? sample_size / last_dim : 1;
        const int64_t x_row_stride = X.row_stride_elems();
        const int64_t y_row_stride = output.row_stride_elems();

        // Per-sample compute: process each (n, c), row-by-row SIMD over spatial dims.
        const auto compute_sample = [&](int64_t n) {
            for (int64_t c = 0; c < C; ++c) {
                const float ns = new_scale[static_cast<size_t>(c)];
                const float nb = new_bias[static_cast<size_t>(c)];
                const int64_t x_ch_base = n * x_n_stride + c * x_c_stride;
                const int64_t y_ch_base = n * y_n_stride + c * y_c_stride;

                for (int64_t hh = 0; hh < num_rows; ++hh) {
                    const int64_t x_off = x_ch_base + hh * x_row_stride;
                    const int64_t y_off = y_ch_base + hh * y_row_stride;
                    int64_t i = 0;

                    if (attrs.add_to) {
                        const v_f32x8 scale8 = v_set1_f32x8(ns);
                        const v_f32x8 bias8  = v_set1_f32x8(nb);
                        for (; i + L <= last_dim; i += L) {
                            const v_f32x8 x8 = v_load_f32x8(x_ptr + x_off + i);
                            const v_f32x8 y8 = v_load_f32x8(y_ptr + y_off + i);
                            v_store(y_ptr + y_off + i, v_add(y8, v_fmadd(scale8, x8, bias8)));
                        }
                        for (; i < last_dim; ++i) {
                            y_ptr[y_off + i] += x_ptr[x_off + i] * ns + nb;
                        }
                    } else {
                        const v_f32x8 scale8 = v_set1_f32x8(ns);
                        const v_f32x8 bias8  = v_set1_f32x8(nb);
                        for (; i + L <= last_dim; i += L) {
                            const v_f32x8 x8 = v_load_f32x8(x_ptr + x_off + i);
                            v_store(y_ptr + y_off + i, v_fmadd(scale8, x8, bias8));
                        }
                        for (; i < last_dim; ++i) {
                            y_ptr[y_off + i] = x_ptr[x_off + i] * ns + nb;
                        }
                    }
                }
            }
        };

        if (ctx.cpu_parallel_for) {
            ctx.cpu_parallel_for(0, N, compute_sample);
        } else {
            for (int64_t n = 0; n < N; ++n) compute_sample(n);
        }
    } else {
        // ---- Non-spatial mode: per-element statistics ----
        const int64_t total = X.numel();

        const auto process_block = [&](int64_t i_begin, int64_t i_end) {
            int64_t i = i_begin;

            if (attrs.add_to) {
                const v_f32x8 eps8 = v_set1_f32x8(epsilon);
                const v_f32x8 one8 = v_set1_f32x8(1.0f);

                for (; i + L <= i_end; i += L) {
                    const v_f32x8 x8 = v_load_f32x8(x_ptr + i);
                    const v_f32x8 s8 = v_load_f32x8(s_ptr + i);
                    const v_f32x8 b8 = v_load_f32x8(b_ptr + i);
                    const v_f32x8 m8 = v_load_f32x8(m_ptr + i);
                    const v_f32x8 v8 = v_load_f32x8(v_ptr + i);

                    const v_f32x8 inv_std = v_div(one8, v_sqrt(v_add(v8, eps8)));
                    const v_f32x8 ns = v_mul(inv_std, s8);
                    const v_f32x8 nb = v_sub(b8, v_mul(m8, ns));
                    const v_f32x8 y8 = v_load_f32x8(y_ptr + i);
                    v_store(y_ptr + i, v_add(y8, v_fmadd(ns, x8, nb)));
                }

                for (; i < i_end; ++i) {
                    float inv_std = 1.0f / std::sqrt(v_ptr[i] + epsilon);
                    float ns = inv_std * s_ptr[i];
                    float nb = b_ptr[i] - m_ptr[i] * ns;
                    y_ptr[i] += x_ptr[i] * ns + nb;
                }
            } else {
                const v_f32x8 eps8 = v_set1_f32x8(epsilon);
                const v_f32x8 one8 = v_set1_f32x8(1.0f);

                for (; i + L <= i_end; i += L) {
                    const v_f32x8 x8 = v_load_f32x8(x_ptr + i);
                    const v_f32x8 s8 = v_load_f32x8(s_ptr + i);
                    const v_f32x8 b8 = v_load_f32x8(b_ptr + i);
                    const v_f32x8 m8 = v_load_f32x8(m_ptr + i);
                    const v_f32x8 v8 = v_load_f32x8(v_ptr + i);

                    const v_f32x8 inv_std = v_div(one8, v_sqrt(v_add(v8, eps8)));
                    const v_f32x8 ns = v_mul(inv_std, s8);
                    const v_f32x8 nb = v_sub(b8, v_mul(m8, ns));
                    v_store(y_ptr + i, v_fmadd(ns, x8, nb));
                }

                for (; i < i_end; ++i) {
                    float inv_std = 1.0f / std::sqrt(v_ptr[i] + epsilon);
                    float ns = inv_std * s_ptr[i];
                    float nb = b_ptr[i] - m_ptr[i] * ns;
                    y_ptr[i] = x_ptr[i] * ns + nb;
                }
            }
        };

        process_block(0, total);
    }
}

}  // namespace nnops::backend::cpu
