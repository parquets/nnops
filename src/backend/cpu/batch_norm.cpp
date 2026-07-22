/// @file batch_norm.cpp
/// @brief SIMD-optimized CPU implementation of batch normalization (inference only).
///
/// Supports both f32 and f16 via a single templated implementation that uses the
/// generic v_load/v_store/v_set1/s_load/s_store SIMD API. Respects pitch through
/// row_stride_elems() and stride_elems().
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

using namespace nnops::simd;

// ============================================================
// Templated implementation (f32 and f16)
// ============================================================

template <typename T>
void batch_norm_impl(const BatchNormAttributes& attrs,
                      const TensorView& output,
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

    const auto* x_ptr  = X.data_as<T>();
    const auto* s_ptr  = scale.data_as<T>();
    const auto* b_ptr  = bias.data_as<T>();
    const auto* m_ptr  = mean.data_as<T>();
    const auto* v_ptr  = var.data_as<T>();
    auto* y_ptr = output.data_as<T>();

    constexpr int L = simd_lane_for<T>;

    if (attrs.spatial) {
        // ---- Spatial mode: per-channel statistics ----
        // Precompute new_scale and new_bias per channel (in float for precision).
        std::vector<float> new_scale(static_cast<size_t>(C));
        std::vector<float> new_bias(static_cast<size_t>(C));
        for (int64_t c = 0; c < C; ++c) {
            float inv_std = 1.0f / std::sqrt(s_load(&v_ptr[c]) + epsilon);
            new_scale[static_cast<size_t>(c)] = inv_std * s_load(&s_ptr[c]);
            new_bias[static_cast<size_t>(c)]  = s_load(&b_ptr[c]) - s_load(&m_ptr[c]) * new_scale[static_cast<size_t>(c)];
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

                    const auto scale8 = v_set1(x_ptr, ns);
                    const auto bias8  = v_set1(x_ptr, nb);
                    for (; i + L <= last_dim; i += L) {
                        const auto x8 = v_load(x_ptr + x_off + i);
                        auto rv = v_fmadd(scale8, x8, bias8);
                        if (attrs.add_to) {
                            v_store(y_ptr + y_off + i,
                                    v_add(v_load(y_ptr + y_off + i), rv));
                        } else {
                            v_store(y_ptr + y_off + i, rv);
                        }
                    }
                    for (; i < last_dim; ++i) {
                        float rv = s_load(&x_ptr[x_off + i]) * ns + nb;
                        if (attrs.add_to) {
                            s_store(&y_ptr[y_off + i],
                                    s_load(&y_ptr[y_off + i]) + rv);
                        } else {
                            s_store(&y_ptr[y_off + i], rv);
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

            const auto eps8 = v_set1(x_ptr, epsilon);
            const auto one8 = v_set1(x_ptr, 1.0f);

            for (; i + L <= i_end; i += L) {
                const auto x8 = v_load(x_ptr + i);
                const auto s8 = v_load(s_ptr + i);
                const auto b8 = v_load(b_ptr + i);
                const auto m8 = v_load(m_ptr + i);
                const auto v8 = v_load(v_ptr + i);

                const auto inv_std = v_div(one8, v_sqrt(v_add(v8, eps8)));
                const auto ns = v_mul(inv_std, s8);
                const auto nb = v_sub(b8, v_mul(m8, ns));
                auto rv = v_fmadd(ns, x8, nb);
                if (attrs.add_to) {
                    v_store(y_ptr + i, v_add(v_load(y_ptr + i), rv));
                } else {
                    v_store(y_ptr + i, rv);
                }
            }

            for (; i < i_end; ++i) {
                float inv_std_val = 1.0f / std::sqrt(s_load(&v_ptr[i]) + epsilon);
                float ns = inv_std_val * s_load(&s_ptr[i]);
                float nb = s_load(&b_ptr[i]) - s_load(&m_ptr[i]) * ns;
                float rv = s_load(&x_ptr[i]) * ns + nb;
                if (attrs.add_to) {
                    s_store(&y_ptr[i], s_load(&y_ptr[i]) + rv);
                } else {
                    s_store(&y_ptr[i], rv);
                }
            }
        };

        process_block(0, total);
    }
}

// ============================================================
// Entry point with dtype dispatch
// ============================================================

void batch_norm_cpu(const BatchNormAttributes& attrs,
                     const TensorView& output,
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
