/// @file norm_ref.cpp
/// @brief Naive CPU reference implementations of normalization operators.
///
/// Covers BatchNorm, LayerNorm, RMSNorm, GroupNorm, and L2Norm, dispatched
/// on NormAttributes::type. All use f32 regardless of input dtype for
/// simplicity (matching the test harness's reference-vs-SIMD comparison).

#include "nnops/ops/norm.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"

#include <cmath>
#include <vector>

namespace nnops::backend::cpu::reference {

namespace {

// ============================================================
// BatchNorm
// ============================================================

void batch_norm_ref_impl(const NormAttributes& attrs,
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

            ctx.cpu.run(0, num_rows, process_row);

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

            ctx.cpu.run(0, spatial_total, body);
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

        ctx.cpu.run(0, total, body);
    }
}

// ============================================================
// LayerNorm
// ============================================================

void layer_norm_ref_impl(const NormAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx)
{
    const auto& X     = inputs[0];
    const auto& scale = inputs[1];
    const bool has_bias = (inputs.size() >= 3 && !inputs[2].is_empty());

    const int64_t rank = X.rank();
    const float epsilon = attrs.epsilon;

    int64_t axis = attrs.axis;
    if (axis < 0) { axis += rank; }
    NNOPS_ASSERT(axis >= 0 && axis < rank);

    int64_t num_rows = 1;
    for (int64_t i = 0; i < axis; ++i) {
        num_rows *= X.shape(i);
    }
    int64_t norm_size = 1;
    for (int64_t i = axis; i < rank; ++i) {
        norm_size *= X.shape(i);
    }

    const auto* x_ptr  = X.ptr<float>();
    const auto* s_ptr  = scale.ptr<float>();
    const auto* b_ptr  = has_bias ? inputs[2].ptr<float>() : nullptr;
    auto* y_ptr = output.ptr<float>();

    const int64_t outer_stride = (axis > 0) ? X.stride_elems(axis - 1) : 0;

    auto inner_offset = [&](int64_t flat_idx) -> int64_t {
        int64_t off = 0;
        int64_t rem = flat_idx;
        for (int64_t i = rank - 1; i >= axis; --i) {
            int64_t dim = X.shape(i);
            off += (rem % dim) * X.stride_elems(i);
            rem /= dim;
        }
        return off;
    };

    auto scale_offset = [&](int64_t flat_idx) -> int64_t {
        if (scale.numel() == 1) { return 0; }
        if (scale.rank() == 1 && norm_size == scale.shape(0)) { return flat_idx; }
        return flat_idx;
    };

    const auto process_row = [&](int64_t row) {
        int64_t row_base = row * outer_stride;

        // Welford's online algorithm for mean and variance
        float mean_val = 0.0f;
        float M2 = 0.0f;
        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t off = row_base + inner_offset(i);
            float x = x_ptr[off];
            float delta = x - mean_val;
            mean_val += delta / static_cast<float>(i + 1);
            float delta2 = x - mean_val;
            M2 += delta * delta2;
        }
        float var_val = M2 / static_cast<float>(norm_size);
        float inv_std = 1.0f / std::sqrt(var_val + epsilon);

        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t off = row_base + inner_offset(i);
            int64_t s_idx = scale_offset(i);
            float s = s_ptr[s_idx];
            float b = b_ptr ? b_ptr[s_idx] : 0.0f;
            float val = (x_ptr[off] - mean_val) * inv_std * s + b;
            y_ptr[off] = attrs.add_to ? y_ptr[off] + val : val;
        }
    };

    ctx.cpu.run(0, num_rows, process_row);
}

// ============================================================
// RMSNorm
// ============================================================

void rms_norm_ref_impl(const NormAttributes& attrs,
                       TensorView& output,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx)
{
    const auto& X     = inputs[0];
    const auto& scale = inputs[1];

    const int64_t rank = X.rank();
    const float epsilon = attrs.epsilon;

    int64_t axis = attrs.axis;
    if (axis < 0) { axis += rank; }
    NNOPS_ASSERT(axis >= 0 && axis < rank);

    int64_t num_rows = 1;
    for (int64_t i = 0; i < axis; ++i) {
        num_rows *= X.shape(i);
    }
    int64_t norm_size = 1;
    for (int64_t i = axis; i < rank; ++i) {
        norm_size *= X.shape(i);
    }

    const auto* x_ptr = X.ptr<float>();
    const auto* s_ptr = scale.ptr<float>();
    auto* y_ptr = output.ptr<float>();

    const int64_t outer_stride = (axis > 0) ? X.stride_elems(axis - 1) : 0;

    auto inner_offset = [&](int64_t flat_idx) -> int64_t {
        int64_t off = 0;
        int64_t rem = flat_idx;
        for (int64_t i = rank - 1; i >= axis; --i) {
            int64_t dim = X.shape(i);
            off += (rem % dim) * X.stride_elems(i);
            rem /= dim;
        }
        return off;
    };

    const bool scale_is_scalar = (scale.numel() == 1);

    const auto process_row = [&](int64_t row) {
        int64_t row_base = row * outer_stride;

        float sum_sq = 0.0f;
        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t off = row_base + inner_offset(i);
            float x = x_ptr[off];
            sum_sq += x * x;
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(norm_size) + epsilon);
        float inv_rms = 1.0f / rms;

        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t off = row_base + inner_offset(i);
            float s = scale_is_scalar ? s_ptr[0] : s_ptr[i];
            float val = x_ptr[off] * inv_rms * s;
            y_ptr[off] = attrs.add_to ? y_ptr[off] + val : val;
        }
    };

    ctx.cpu.run(0, num_rows, process_row);
}

// ============================================================
// L2Norm
// ============================================================

void l2_norm_ref_impl(const NormAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx)
{
    const auto& X = inputs[0];

    const int64_t rank = X.rank();
    const float epsilon = attrs.epsilon;

    int64_t axis = attrs.axis;
    if (axis < 0) { axis += rank; }
    NNOPS_ASSERT(axis >= 0 && axis < rank);

    int64_t num_rows = 1;
    for (int64_t i = 0; i < axis; ++i) {
        num_rows *= X.shape(i);
    }
    int64_t norm_size = 1;
    for (int64_t i = axis; i < rank; ++i) {
        norm_size *= X.shape(i);
    }

    const auto* x_ptr = X.ptr<float>();
    auto* y_ptr = output.ptr<float>();

    const int64_t outer_stride = (axis > 0) ? X.stride_elems(axis - 1) : 0;

    auto inner_offset = [&](int64_t flat_idx) -> int64_t {
        int64_t off = 0;
        int64_t rem = flat_idx;
        for (int64_t i = rank - 1; i >= axis; --i) {
            int64_t dim = X.shape(i);
            off += (rem % dim) * X.stride_elems(i);
            rem /= dim;
        }
        return off;
    };

    const auto process_row = [&](int64_t row) {
        int64_t row_base = row * outer_stride;

        float sum_sq = 0.0f;
        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t off = row_base + inner_offset(i);
            float x = x_ptr[off];
            sum_sq += x * x;
        }
        float inv_norm = 1.0f / std::sqrt(sum_sq + epsilon);

        for (int64_t i = 0; i < norm_size; ++i) {
            int64_t off = row_base + inner_offset(i);
            float val = x_ptr[off] * inv_norm;
            y_ptr[off] = attrs.add_to ? y_ptr[off] + val : val;
        }
    };

    ctx.cpu.run(0, num_rows, process_row);
}

// ============================================================
// GroupNorm
// ============================================================

void group_norm_ref_impl(const NormAttributes& attrs,
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

    const int64_t stride_C = X.stride_elems(1);
    const int64_t stride_N = X.stride_elems(0);
    const int64_t row_stride = X.row_stride_elems();

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

    ctx.cpu.run(0, total_groups, process_group);
}

}  // anonymous namespace

// ============================================================
// Entry point — dispatch on NormAttributes::type
// ============================================================

void norm_ref(const NormAttributes& attrs,
              TensorView& output,
              std::span<const TensorView> inputs,
              const ComputeContext& ctx,
              void* /*workspace*/)
{
    switch (attrs.type) {
    case NormType::BatchNorm:
        batch_norm_ref_impl(attrs, output, inputs, ctx);
        return;
    case NormType::LayerNorm:
        layer_norm_ref_impl(attrs, output, inputs, ctx);
        return;
    case NormType::RMSNorm:
        rms_norm_ref_impl(attrs, output, inputs, ctx);
        return;
    case NormType::GroupNorm:
        group_norm_ref_impl(attrs, output, inputs, ctx);
        return;
    case NormType::L2Norm:
        l2_norm_ref_impl(attrs, output, inputs, ctx);
        return;
    }
    NNOPS_ASSERT(!"norm_ref: unknown norm type");
}

}  // namespace nnops::backend::cpu::reference
