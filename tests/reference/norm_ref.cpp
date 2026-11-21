/// @file norm_ref.cpp
/// @brief Naive CPU references for BatchNorm, LayerNorm, RMSNorm, GroupNorm and
/// L2Norm, dispatched on NormAttributes::type.
///
/// Every type reads and writes f32 whatever the input dtype: the tests
/// dequantize to f32, run this, and requantize. Addressing goes through
/// stride_elems(), so dense and NHWC inputs are both covered by the
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

/// Base offset of row `row` in an AdaLN/FiLM conditioning tensor that
/// broadcasts NumPy-style against X's leading dims.
int64_t mod_row_base(const TensorView& mod, const TensorView& x, int64_t row, int64_t axis) {
    const int64_t xr = x.rank();
    const int64_t mr = mod.rank();
    int64_t off = 0;
    int64_t rem = row;
    for (int64_t d = axis - 1; d >= 0; --d) {
        const int64_t dim = x.shape(d);
        if (dim <= 0) { continue; }
        const int64_t coord = rem % dim;
        rem /= dim;
        if (coord == 0) { continue; }
        const int64_t md = d - (xr - mr);
        if (md < 0 || mod.shape(md) == 1) { continue; }
        off += coord * mod.stride_elems(md);
    }
    return off;
}

/// Scalar activation for the gated form (RMSNormGated): act(gate).
float ref_gate_act(EpilogueActivateType a, float g) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    switch (a) {
    case EpilogueActivateType::Silu:    return g / (1.0f + std::exp(-g));
    case EpilogueActivateType::Gelu:    return 0.5f * g *
        (1.0f + std::tanh(kSqrt2OverPi * (g + 0.044715f * g * g * g)));
    case EpilogueActivateType::Relu:    return g > 0.0f ? g : 0.0f;
    case EpilogueActivateType::Sigmoid: return 1.0f / (1.0f + std::exp(-g));
    case EpilogueActivateType::Tanh:    return std::tanh(g);
    default:                            return g;
    }
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
    const auto& X = inputs[0];

    // Modulation (AdaLN / FiLM) owns the last 2 or 3 slots, after the type's.
    const bool modulate = attrs.modulate;
    const bool mod_gate = modulate && attrs.modulation_gate;
    const int64_t n_trail = modulate ? (mod_gate ? 3 : 2) : 0;
    const int64_t base_n = static_cast<int64_t>(inputs.size()) - n_trail;
    const bool has_scale = modulate ? (base_n >= 2 && inputs[1].rank() > 0) : true;
    const bool has_bias = base_n >= 3 && !inputs[2].is_empty();
    const TensorView* mod_scale = modulate ? &inputs[static_cast<size_t>(base_n)] : nullptr;
    const TensorView* mod_shift = modulate ? &inputs[static_cast<size_t>(base_n + 1)] : nullptr;
    const TensorView* mod_gate_v = mod_gate ? &inputs[static_cast<size_t>(base_n + 2)] : nullptr;
    const float mod_offset = attrs.modulation_scale_offset;

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
    const auto* s_ptr  = has_scale ? inputs[1].ptr<float>() : nullptr;
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

        const int64_t ms_base = modulate ? mod_row_base(*mod_scale, X, row, axis) : 0;
        const int64_t mt_base = modulate ? mod_row_base(*mod_shift, X, row, axis) : 0;
        const int64_t mg_base = mod_gate ? mod_row_base(*mod_gate_v, X, row, axis) : 0;
        const auto* ms_ptr = modulate ? mod_scale->ptr<float>() + ms_base : nullptr;
        const auto* mt_ptr = modulate ? mod_shift->ptr<float>() + mt_base : nullptr;
        const auto* mg_ptr = mod_gate ? mod_gate_v->ptr<float>() + mg_base : nullptr;

        for (int64_t i = 0; i < norm_size; ++i) {
            const int64_t xi = x_base + x_inner[static_cast<size_t>(i)];
            const int64_t yi = y_base + y_inner[static_cast<size_t>(i)];
            float s = has_scale ? s_ptr[i] + (attrs.one_plus_weight ? 1.0f : 0.0f) : 1.0f;
            float b = b_ptr ? b_ptr[i] : 0.0f;
            float val = (x_ptr[xi] - mean_val) * inv_std * s + b;
            if (modulate) {
                val = val * (ms_ptr[i] + mod_offset) + mt_ptr[i];
                if (mod_gate) { val *= mg_ptr[i]; }
            }
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
    const auto& X = inputs[0];

    const bool modulate = attrs.modulate;
    const bool mod_gate = modulate && attrs.modulation_gate;
    const int64_t n_trail = modulate ? (mod_gate ? 3 : 2) : 0;
    const int64_t base_n = static_cast<int64_t>(inputs.size()) - n_trail;
    const bool has_scale = modulate ? (base_n >= 2 && inputs[1].rank() > 0) : true;
    const TensorView* mod_scale = modulate ? &inputs[static_cast<size_t>(base_n)] : nullptr;
    const TensorView* mod_shift = modulate ? &inputs[static_cast<size_t>(base_n + 1)] : nullptr;
    const TensorView* mod_gate_v = mod_gate ? &inputs[static_cast<size_t>(base_n + 2)] : nullptr;
    const float mod_offset = attrs.modulation_scale_offset;

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
    const auto* s_ptr = has_scale ? inputs[1].ptr<float>() : nullptr;
    auto* y_ptr = output.ptr<float>();

    // The gate is the trailing input (RMSNormGated); a residual, when present,
    // sits before it.
    const bool gated = attrs.gate_activation != EpilogueActivateType::None;
    const TensorView* gate = gated ? &inputs.back() : nullptr;
    const auto* g_ptr = gated ? gate->ptr<float>() : nullptr;

    std::vector<int64_t> x_inner(static_cast<size_t>(norm_size));
    std::vector<int64_t> y_inner(static_cast<size_t>(norm_size));
    std::vector<int64_t> g_inner(static_cast<size_t>(norm_size));
    for (int64_t i = 0; i < norm_size; ++i) {
        x_inner[static_cast<size_t>(i)] = flat_offset(X, axis, rank, i);
        y_inner[static_cast<size_t>(i)] = flat_offset(output, axis, rank, i);
        if (gated) {
            g_inner[static_cast<size_t>(i)] = flat_offset(*gate, axis, rank, i);
        }
    }

    const bool scale_is_scalar = has_scale && (inputs[1].numel() == 1);

    const auto process_row = [&](int64_t row) {
        const int64_t x_base = outer_base(X, row, axis);
        const int64_t y_base = outer_base(output, row, axis);
        const int64_t g_base = gated ? outer_base(*gate, row, axis) : 0;

        float sum_sq = 0.0f;
        for (int64_t i = 0; i < norm_size; ++i) {
            float x = x_ptr[x_base + x_inner[static_cast<size_t>(i)]];
            sum_sq += x * x;
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(norm_size) + epsilon);
        float inv_rms = 1.0f / rms;

        const int64_t ms_base = modulate ? mod_row_base(*mod_scale, X, row, axis) : 0;
        const int64_t mt_base = modulate ? mod_row_base(*mod_shift, X, row, axis) : 0;
        const int64_t mg_base = mod_gate ? mod_row_base(*mod_gate_v, X, row, axis) : 0;
        const auto* ms_ptr = modulate ? mod_scale->ptr<float>() + ms_base : nullptr;
        const auto* mt_ptr = modulate ? mod_shift->ptr<float>() + mt_base : nullptr;
        const auto* mg_ptr = mod_gate ? mod_gate_v->ptr<float>() + mg_base : nullptr;

        for (int64_t i = 0; i < norm_size; ++i) {
            const int64_t xi = x_base + x_inner[static_cast<size_t>(i)];
            const int64_t yi = y_base + y_inner[static_cast<size_t>(i)];
            float s = has_scale ? (scale_is_scalar ? s_ptr[0] : s_ptr[i]) : 1.0f;
            if (has_scale && attrs.one_plus_weight) { s += 1.0f; }
            float val = x_ptr[xi] * inv_rms * s;
            if (gated) {
                val *= ref_gate_act(
                    attrs.gate_activation,
                    g_ptr[g_base + g_inner[static_cast<size_t>(i)]]);
            }
            if (modulate) {
                val = val * (ms_ptr[i] + mod_offset) + mt_ptr[i];
                if (mod_gate) { val *= mg_ptr[i]; }
            }
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

    const bool modulate = attrs.modulate;
    const bool mod_gate = modulate && attrs.modulation_gate;
    const int64_t n_trail = modulate ? (mod_gate ? 3 : 2) : 0;
    const int64_t base_n = static_cast<int64_t>(inputs.size()) - n_trail;
    const TensorView* mod_scale = modulate ? &inputs[static_cast<size_t>(base_n)] : nullptr;
    const TensorView* mod_shift = modulate ? &inputs[static_cast<size_t>(base_n + 1)] : nullptr;
    const TensorView* mod_gate_v = mod_gate ? &inputs[static_cast<size_t>(base_n + 2)] : nullptr;
    const float mod_offset = attrs.modulation_scale_offset;

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

        const int64_t ms_base = modulate ? mod_row_base(*mod_scale, X, row, axis) : 0;
        const int64_t mt_base = modulate ? mod_row_base(*mod_shift, X, row, axis) : 0;
        const int64_t mg_base = mod_gate ? mod_row_base(*mod_gate_v, X, row, axis) : 0;
        const auto* ms_ptr = modulate ? mod_scale->ptr<float>() + ms_base : nullptr;
        const auto* mt_ptr = modulate ? mod_shift->ptr<float>() + mt_base : nullptr;
        const auto* mg_ptr = mod_gate ? mod_gate_v->ptr<float>() + mg_base : nullptr;

        for (int64_t i = 0; i < norm_size; ++i) {
            const int64_t xi = x_base + x_inner[static_cast<size_t>(i)];
            const int64_t yi = y_base + y_inner[static_cast<size_t>(i)];
            float val = x_ptr[xi] * inv_norm;
            if (modulate) {
                val = val * (ms_ptr[i] + mod_offset) + mt_ptr[i];
                if (mod_gate) { val *= mg_ptr[i]; }
            }
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

    const bool modulate = attrs.modulate;
    const bool mod_gate = modulate && attrs.modulation_gate;
    const int64_t n_trail = modulate ? (mod_gate ? 3 : 2) : 0;
    const int64_t base_n = static_cast<int64_t>(inputs.size()) - n_trail;
    const bool has_bias = base_n >= 3 && !inputs[2].is_empty();
    const TensorView* mod_scale_v = modulate ? &inputs[static_cast<size_t>(base_n)] : nullptr;
    const TensorView* mod_shift_v = modulate ? &inputs[static_cast<size_t>(base_n + 1)] : nullptr;
    const TensorView* mod_gate_v = mod_gate ? &inputs[static_cast<size_t>(base_n + 2)] : nullptr;

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
    const auto* s_ptr  = inputs[1].ptr<float>();
    const auto* b_ptr  = has_bias ? inputs[2].ptr<float>() : nullptr;
    auto* y_ptr = output.ptr<float>();

    const auto* ms_ptr = modulate ? mod_scale_v->ptr<float>() : nullptr;
    const auto* mt_ptr = modulate ? mod_shift_v->ptr<float>() : nullptr;
    const auto* mg_ptr = mod_gate ? mod_gate_v->ptr<float>() : nullptr;
    const auto mod_n_stride = [](const TensorView* v) {
        return (v != nullptr && v->rank() == 2 && v->shape(0) != 1) ? v->stride_elems(0) : 0;
    };
    const int64_t ms_n = mod_n_stride(mod_scale_v);
    const int64_t mt_n = mod_n_stride(mod_shift_v);
    const int64_t mg_n = mod_n_stride(mod_gate_v);
    const float mod_offset = attrs.modulation_scale_offset;

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
                if (modulate) {
                    val = val * (ms_ptr[ms_n * n + c_global] + mod_offset) +
                          mt_ptr[mt_n * n + c_global];
                    if (mod_gate) { val *= mg_ptr[mg_n * n + c_global]; }
                }
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
