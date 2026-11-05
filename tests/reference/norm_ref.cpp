/// @file norm_ref.cpp
/// @brief Naive CPU references for BatchNorm, LayerNorm, RMSNorm, GroupNorm and
/// L2Norm, dispatched on NormAttributes::type.
///
/// Every type reads and writes f32 whatever the input dtype: the tests
/// dequantize to f32, run this, and requantize. Addressing goes through
/// stride_elems(), so channel-first and channel-last inputs are both covered by the
/// same code; only the logical axis matters.

#include "nnops/ops/norm.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"

#include <cmath>
#include <vector>

namespace nnops::backend::cpu::reference {

namespace {

/// Offset of logical flat index @p flat over dims [start, end), row-major.
int64_t flat_offset(const TensorView& t, int64_t start, int64_t end, int64_t flat) {
    int64_t off = 0;
    int64_t rem = flat;
    for (int64_t d = end - 1; d >= start; --d) {
        const int64_t dim = t.shape(d);
        off += (rem % dim) * t.stride_elems(d);
        rem /= dim;
    }
    return off;
}

int64_t outer_base(const TensorView& t, int64_t row, int64_t axis) {
    return flat_offset(t, 0, axis, row);
}

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

    const int64_t N = X.shape(0);
    const int64_t C = (rank >= 2) ? X.shape(1) : 1;

    const auto* x_ptr = X.ptr<float>();
    const auto* s_ptr = scale.ptr<float>();
    const auto* b_ptr = bias.ptr<float>();
    const auto* m_ptr = mean.ptr<float>();
    const auto* v_ptr = var.ptr<float>();
    auto* y_ptr = output.ptr<float>();

    if (!attrs.spatial) {
        const int64_t total = X.numel();
        const auto body = [&](int64_t i) {
            float inv_std = 1.0f / std::sqrt(v_ptr[i] + epsilon);
            float ns = inv_std * s_ptr[i];
            float nb = b_ptr[i] - m_ptr[i] * ns;
            float val = x_ptr[i] * ns + nb;
            y_ptr[i] = attrs.add_to ? y_ptr[i] + val : val;
        };
        ctx.cpu.run(0, total, body);
        return;
    }

    // Spatial: statistics per channel. One odometer over the whole logical
    // index space feeds both layouts; channel is always logical dim 1.
    std::vector<float> ns_f(static_cast<size_t>(C));
    std::vector<float> nb_f(static_cast<size_t>(C));
    for (int64_t c = 0; c < C; ++c) {
        float inv_std = 1.0f / std::sqrt(v_ptr[c] + epsilon);
        ns_f[static_cast<size_t>(c)] = inv_std * s_ptr[c];
        nb_f[static_cast<size_t>(c)] = b_ptr[c] - m_ptr[c] * ns_f[static_cast<size_t>(c)];
    }

    const int64_t total = X.numel();
    const int64_t spatial_size = (rank >= 2 && N > 0 && C > 0)
        ? total / (N * C) : 1;
    const auto body = [&](int64_t flat) {
        const int64_t c = (rank >= 2) ? (flat / spatial_size) % C : 0;
        const int64_t x_off = flat_offset(X, 0, rank, flat);
        const int64_t y_off = flat_offset(output, 0, rank, flat);
        const float ns = ns_f[static_cast<size_t>(c)];
        const float nb = nb_f[static_cast<size_t>(c)];
        float val = x_ptr[x_off] * ns + nb;
        y_ptr[y_off] = attrs.add_to ? y_ptr[y_off] + val : val;
    };

    ctx.cpu.run(0, total, body);
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

    std::vector<int64_t> x_inner(static_cast<size_t>(norm_size));
    std::vector<int64_t> y_inner(static_cast<size_t>(norm_size));
    for (int64_t i = 0; i < norm_size; ++i) {
        x_inner[static_cast<size_t>(i)] = flat_offset(X, axis, rank, i);
        y_inner[static_cast<size_t>(i)] = flat_offset(output, axis, rank, i);
    }

    const auto process_row = [&](int64_t row) {
        const int64_t x_base = outer_base(X, row, axis);
        const int64_t y_base = outer_base(output, row, axis);

        // Welford's online algorithm for mean and variance.
        float mean_val = 0.0f;
        float M2 = 0.0f;
        for (int64_t i = 0; i < norm_size; ++i) {
            float x = x_ptr[x_base + x_inner[static_cast<size_t>(i)]];
            float delta = x - mean_val;
            mean_val += delta / static_cast<float>(i + 1);
            float delta2 = x - mean_val;
            M2 += delta * delta2;
        }
        float var_val = M2 / static_cast<float>(norm_size);
        float inv_std = 1.0f / std::sqrt(var_val + epsilon);

        for (int64_t i = 0; i < norm_size; ++i) {
            const int64_t xi = x_base + x_inner[static_cast<size_t>(i)];
            const int64_t yi = y_base + y_inner[static_cast<size_t>(i)];
            float s = s_ptr[i];
            float b = b_ptr ? b_ptr[i] : 0.0f;
            float val = (x_ptr[xi] - mean_val) * inv_std * s + b;
            y_ptr[yi] = attrs.add_to ? y_ptr[yi] + val : val;
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

    std::vector<int64_t> x_inner(static_cast<size_t>(norm_size));
    std::vector<int64_t> y_inner(static_cast<size_t>(norm_size));
    for (int64_t i = 0; i < norm_size; ++i) {
        x_inner[static_cast<size_t>(i)] = flat_offset(X, axis, rank, i);
        y_inner[static_cast<size_t>(i)] = flat_offset(output, axis, rank, i);
    }

    const bool scale_is_scalar = (scale.numel() == 1);

    const auto process_row = [&](int64_t row) {
        const int64_t x_base = outer_base(X, row, axis);
        const int64_t y_base = outer_base(output, row, axis);

        float sum_sq = 0.0f;
        for (int64_t i = 0; i < norm_size; ++i) {
            float x = x_ptr[x_base + x_inner[static_cast<size_t>(i)]];
            sum_sq += x * x;
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(norm_size) + epsilon);
        float inv_rms = 1.0f / rms;

        for (int64_t i = 0; i < norm_size; ++i) {
            const int64_t xi = x_base + x_inner[static_cast<size_t>(i)];
            const int64_t yi = y_base + y_inner[static_cast<size_t>(i)];
            float s = scale_is_scalar ? s_ptr[0] : s_ptr[i];
            float val = x_ptr[xi] * inv_rms * s;
            y_ptr[yi] = attrs.add_to ? y_ptr[yi] + val : val;
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

    std::vector<int64_t> x_inner(static_cast<size_t>(norm_size));
    std::vector<int64_t> y_inner(static_cast<size_t>(norm_size));
    for (int64_t i = 0; i < norm_size; ++i) {
        x_inner[static_cast<size_t>(i)] = flat_offset(X, axis, rank, i);
        y_inner[static_cast<size_t>(i)] = flat_offset(output, axis, rank, i);
    }

    const auto process_row = [&](int64_t row) {
        const int64_t x_base = outer_base(X, row, axis);
        const int64_t y_base = outer_base(output, row, axis);

        float sum_sq = 0.0f;
        for (int64_t i = 0; i < norm_size; ++i) {
            float x = x_ptr[x_base + x_inner[static_cast<size_t>(i)]];
            sum_sq += x * x;
        }
        float inv_norm = 1.0f / std::sqrt(sum_sq + epsilon);

        for (int64_t i = 0; i < norm_size; ++i) {
            const int64_t xi = x_base + x_inner[static_cast<size_t>(i)];
            const int64_t yi = y_base + y_inner[static_cast<size_t>(i)];
            float val = x_ptr[xi] * inv_norm;
            y_ptr[yi] = attrs.add_to ? y_ptr[yi] + val : val;
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
    const int64_t y_stride_C = output.stride_elems(1);
    const int64_t y_stride_N = output.stride_elems(0);

    std::vector<int64_t> x_spatial(static_cast<size_t>(spatial_size));
    std::vector<int64_t> y_spatial(static_cast<size_t>(spatial_size));
    for (int64_t s = 0; s < spatial_size; ++s) {
        x_spatial[static_cast<size_t>(s)] = flat_offset(X, 2, rank, s);
        y_spatial[static_cast<size_t>(s)] = flat_offset(output, 2, rank, s);
    }

    const int64_t total_groups = N * G;

    const auto process_group = [&](int64_t ng) {
        const int64_t n = ng / G;
        const int64_t g = ng % G;
        const int64_t c_start = g * channels_per_group;
        const int64_t x_offset = n * stride_N + c_start * stride_C;
        const int64_t y_offset = n * y_stride_N + c_start * y_stride_C;

        // ---- Pass 1: Welford online mean/variance ----
        float mean_val = 0.0f;
        float M2 = 0.0f;
        int64_t count = 0;

        for (int64_t c = 0; c < channels_per_group; ++c) {
            const int64_t c_off = c * stride_C;
            for (int64_t s = 0; s < spatial_size; ++s) {
                float x = x_ptr[x_offset + c_off + x_spatial[static_cast<size_t>(s)]];
                count++;
                float delta = x - mean_val;
                mean_val += delta / static_cast<float>(count);
                float delta2 = x - mean_val;
                M2 += delta * delta2;
            }
        }
        const float var_val = M2 * inv_norm;
        const float inv_std = 1.0f / std::sqrt(var_val + epsilon);

        // ---- Pass 2: normalize ----
        for (int64_t c = 0; c < channels_per_group; ++c) {
            const int64_t c_global = c_start + c;
            const int64_t c_off = c * stride_C;
            const int64_t yc_off = c * y_stride_C;
            const float s = s_ptr[c_global];
            const float b = (has_bias && b_ptr) ? b_ptr[c_global] : 0.0f;

            for (int64_t sp = 0; sp < spatial_size; ++sp) {
                const int64_t xi = x_offset + c_off + x_spatial[static_cast<size_t>(sp)];
                const int64_t yi = y_offset + yc_off + y_spatial[static_cast<size_t>(sp)];
                float val = (x_ptr[xi] - mean_val) * inv_std * s + b;
                if (attrs.add_to) {
                    y_ptr[yi] += val;
                } else {
                    y_ptr[yi] = val;
                }
            }
        }
    };

    ctx.cpu.run(0, total_groups, process_group);
}

}  // anonymous namespace

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
