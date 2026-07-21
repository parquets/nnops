/// @file batch_norm.cpp
/// @brief SIMD-optimized CPU implementation of batch normalization (inference only).
///
/// Vectorized with the nnops SIMD abstraction layer. On x86_64 this compiles to
/// SSE4.1 by default; compile with /arch:AVX2 (MSVC) or -mavx2 -mfma (GCC/Clang)
/// to enable native AVX2+FMA 8-wide processing.
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

        // Per-sample compute: for each (n, c), SIMD over the spatial block.
        const auto compute_sample = [&](int64_t n) {
            for (int64_t c = 0; c < C; ++c) {
                const float ns = new_scale[static_cast<size_t>(c)];
                const float nb = new_bias[static_cast<size_t>(c)];
                const int64_t base = (n * C + c) * sample_size;

                int64_t i = 0;

                if (attrs.add_to) {
                    // add_to path: load existing output, accumulate
                    const v_f32x8 scale8 = v_set1_f32x8(ns);
                    const v_f32x8 bias8  = v_set1_f32x8(nb);
                    for (; i + 8 <= sample_size; i += 8) {
                        const v_f32x8 x8 = v_load_f32x8(x_ptr + base + i);
                        const v_f32x8 y8 = v_load_f32x8(y_ptr + base + i);
                        v_store(y_ptr + base + i, v_add(y8, v_fmadd(scale8, x8, bias8)));
                    }

                    const v_f32x4 scale4 = v_set1_f32x4(ns);
                    const v_f32x4 bias4  = v_set1_f32x4(nb);
                    for (; i + 4 <= sample_size; i += 4) {
                        const v_f32x4 x4 = v_load_f32x4(x_ptr + base + i);
                        const v_f32x4 y4 = v_load_f32x4(y_ptr + base + i);
                        v_store(y_ptr + base + i, v_add(y4, v_fmadd(scale4, x4, bias4)));
                    }

                    for (; i < sample_size; ++i) {
                        y_ptr[base + i] += x_ptr[base + i] * ns + nb;
                    }
                } else {
                    // Overwrite path (original)
                    const v_f32x8 scale8 = v_set1_f32x8(ns);
                    const v_f32x8 bias8  = v_set1_f32x8(nb);
                    for (; i + 8 <= sample_size; i += 8) {
                        const v_f32x8 x8 = v_load_f32x8(x_ptr + base + i);
                        v_store(y_ptr + base + i, v_fmadd(scale8, x8, bias8));
                    }

                    const v_f32x4 scale4 = v_set1_f32x4(ns);
                    const v_f32x4 bias4  = v_set1_f32x4(nb);
                    for (; i + 4 <= sample_size; i += 4) {
                        const v_f32x4 x4 = v_load_f32x4(x_ptr + base + i);
                        v_store(y_ptr + base + i, v_fmadd(scale4, x4, bias4));
                    }

                    for (; i < sample_size; ++i) {
                        y_ptr[base + i] = x_ptr[base + i] * ns + nb;
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
        // Scale/bias/mean/var have the same shape as X.
        const int64_t total = X.numel();

        // Use a block-based body for single-threaded execution.
        // When parallel_for is active, we chunk at a coarser granularity.
        const auto process_block = [&](int64_t i_begin, int64_t i_end) {
            int64_t i = i_begin;

            if (attrs.add_to) {
                // add_to path: load existing output, accumulate
                const v_f32x8 eps8 = v_set1_f32x8(epsilon);
                const v_f32x8 one8 = v_set1_f32x8(1.0f);
                const v_f32x4 eps4 = v_set1_f32x4(epsilon);
                const v_f32x4 one4 = v_set1_f32x4(1.0f);

                // 8-wide SIMD
                for (; i + 8 <= i_end; i += 8) {
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

                // 4-wide tail
                for (; i + 4 <= i_end; i += 4) {
                    const v_f32x4 x4 = v_load_f32x4(x_ptr + i);
                    const v_f32x4 s4 = v_load_f32x4(s_ptr + i);
                    const v_f32x4 b4 = v_load_f32x4(b_ptr + i);
                    const v_f32x4 m4 = v_load_f32x4(m_ptr + i);
                    const v_f32x4 v4 = v_load_f32x4(v_ptr + i);

                    const v_f32x4 inv_std = v_div(one4, v_sqrt(v_add(v4, eps4)));
                    const v_f32x4 ns = v_mul(inv_std, s4);
                    const v_f32x4 nb = v_sub(b4, v_mul(m4, ns));
                    const v_f32x4 y4 = v_load_f32x4(y_ptr + i);
                    v_store(y_ptr + i, v_add(y4, v_fmadd(ns, x4, nb)));
                }

                // Scalar tail
                for (; i < i_end; ++i) {
                    float inv_std = 1.0f / std::sqrt(v_ptr[i] + epsilon);
                    float ns = inv_std * s_ptr[i];
                    float nb = b_ptr[i] - m_ptr[i] * ns;
                    y_ptr[i] += x_ptr[i] * ns + nb;
                }
            } else {
                // Overwrite path (original)
                const v_f32x8 eps8 = v_set1_f32x8(epsilon);
                const v_f32x8 one8 = v_set1_f32x8(1.0f);
                const v_f32x4 eps4 = v_set1_f32x4(epsilon);
                const v_f32x4 one4 = v_set1_f32x4(1.0f);

                // 8-wide SIMD
                for (; i + 8 <= i_end; i += 8) {
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

                // 4-wide tail
                for (; i + 4 <= i_end; i += 4) {
                    const v_f32x4 x4 = v_load_f32x4(x_ptr + i);
                    const v_f32x4 s4 = v_load_f32x4(s_ptr + i);
                    const v_f32x4 b4 = v_load_f32x4(b_ptr + i);
                    const v_f32x4 m4 = v_load_f32x4(m_ptr + i);
                    const v_f32x4 v4 = v_load_f32x4(v_ptr + i);

                    const v_f32x4 inv_std = v_div(one4, v_sqrt(v_add(v4, eps4)));
                    const v_f32x4 ns = v_mul(inv_std, s4);
                    const v_f32x4 nb = v_sub(b4, v_mul(m4, ns));
                    v_store(y_ptr + i, v_fmadd(ns, x4, nb));
                }

                // Scalar tail
                for (; i < i_end; ++i) {
                    float inv_std = 1.0f / std::sqrt(v_ptr[i] + epsilon);
                    float ns = inv_std * s_ptr[i];
                    float nb = b_ptr[i] - m_ptr[i] * ns;
                    y_ptr[i] = x_ptr[i] * ns + nb;
                }
            }
        };

        // Single-threaded SIMD: per-element statistics diverge across elements,
        // making the per-element parallel_for API a poor fit for SIMD chunking.
        // Non-spatial mode is rare in practice; spatial mode is the hot path.
        process_block(0, total);
    }
}

}  // namespace nnops::backend::cpu
