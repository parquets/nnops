/// @file norm.cpp
/// @brief SIMD CPU normalization — BatchNorm, LayerNorm, RMSNorm, GroupNorm,
///        L2Norm — dispatched on NormAttributes::type.
///
/// f32 and f16 share one templated implementation per type, over the SIMD row
/// kernels in simd_kernel/simd_norm.hpp. LayerNorm / RMSNorm / L2Norm run the
/// vectorized row path when the normalized axes are the physically contiguous
/// tail (dense axis == rank-1) and otherwise walk precomputed offsets, so a
/// logical axis means the same thing in dense and spatial layouts.
/// BatchNorm walks rows and applies the per-channel factors pixel by pixel in
/// NHWC/NDHWC; GroupNorm reduces one (sample, group) at a time.
///
/// s8/u8 input is supported for LayerNorm / RMSNorm / L2Norm over the trailing
/// axis: each integer row is dequantized to f32 by the arch quant.hpp per-row
/// kernel and the same SIMD row kernels then run on it — no new kernel is
/// needed. The unit is one row, which is exactly one quantization row (what a
/// PerToken scale/zero_point addresses), and that is what lets a row be
/// dequantized, normalized and written back without staging the tensor.
/// PerChannel has no meaning over the trailing axis, and BatchNorm / GroupNorm
/// are out because their reduction units do not line up with quantization rows.
/// The compute type is always f32 and scale / bias stay f32; an f16 output is
/// narrowed on store. An f32 output is dequantized and normalized in place in
/// its own row; f16 and s8/u8 outputs stage each row in a pooled f32 buffer and
/// convert on write-back — for s8/u8, requantizing with the output view's own
/// (scale, zero_point). The buffer is pooled, so getWorkspaceSize stays 0.
/// add_to is rejected here — it would mean accumulating into a pre-existing
/// quantized output.

#include "nnops/ops/norm.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "simd_kernel/simd_norm.hpp"
#include "common/gate_combine.hpp"
#include "common/index.hpp"
#include "common/dtype_dispatch.hpp"
#include "common/memory_pool.hpp"

#if defined(NNOPS_ARCH_X86_64)
#include "x86_64/quant.hpp"
#elif defined(NNOPS_ARCH_AARCH64)
#include "aarch64/quant.hpp"
#else
#error "norm: unsupported architecture for quantization kernels"
#endif

#include <cmath>
#include <vector>

namespace nnops::backend::cpu {

using namespace nnops::simd;

#if defined(NNOPS_ARCH_X86_64)
namespace quant_kernel = nnops::backend::cpu::x86_64;
#elif defined(NNOPS_ARCH_AARCH64)
namespace quant_kernel = nnops::backend::cpu::aarch64;
#endif

namespace {

// ============================================================
// Shared helpers
// ============================================================

/// Flattened inner index -> stride offset (non-trailing axis).
inline PoolArray<int64_t> compute_inner_offsets(const TensorView& X, int64_t axis,
                                                int64_t norm_size)
{
    return build_offsets(X, axis, X.rank() - 1, norm_size);
}

/// Base offset of row `row` in the outer dims [0, axis-1].
inline int64_t outer_base(const TensorView& t, int64_t row, int64_t axis)
{
    return decompose_flat_offset(row, t, 0, axis - 1);
}

// ============================================================
// RMSNormGated's fused post step
// ============================================================

/// The normalized value is multiplied by act(gate) in the same row pass, so
/// the hidden is written once: the unfused form stored the row, then read it
/// back to scale it and stored it again.
template <EpilogueActivateType Act, typename T> struct GatePost {
    const T* gate;
    template <class V> V vec(const V& v, int64_t i) const
    {
        return v_mul(v, gate_act_vec<Act>(v_load(gate + i), gate, 1.0f));
    }
    float scalar(float v, int64_t i) const
    {
        return v * gate_act_scalar<Act>(s_load(gate + i), 1.0f);
    }
};

/// Runtime activation -> the compile-time row kernel. RMSNormGated is the only
/// gated form (validation), so the shape is fixed: no mean, a scale, no bias.
template <typename T, bool WithResidual, EpilogueActivateType Act>
inline void rms_norm_apply_gated_act(const T* x, const T* r, const T* g, T* y, int64_t n,
                                     float inv_rms, const T* scale, bool scale_is_scalar,
                                     bool add_to, bool one_plus)
{
    GatePost<Act, T> post{g};
    if constexpr (WithResidual) {
        kernel::norm_apply_row_add_post_dispatch<T, false, true, false>(
            x, r, y, n, 0.0f, inv_rms, scale, nullptr, scale_is_scalar, add_to, one_plus, post);
    } else {
        kernel::norm_apply_row_post_dispatch<T, false, true, false>(
            x, y, n, 0.0f, inv_rms, scale, nullptr, scale_is_scalar, add_to, one_plus, post);
    }
}

template <typename T, bool WithResidual>
inline void rms_norm_apply_gated(EpilogueActivateType act, const T* x, const T* r, const T* g, T* y,
                                 int64_t n, float inv_rms, const T* scale, bool scale_is_scalar,
                                 bool add_to, bool one_plus)
{
#define NNOPS_NORM_GATE_CASE(name)                                             \
    case EpilogueActivateType::name:                                           \
        rms_norm_apply_gated_act<T, WithResidual, EpilogueActivateType::name>( \
            x, r, g, y, n, inv_rms, scale, scale_is_scalar, add_to, one_plus); \
        return
    switch (act) {
        NNOPS_NORM_GATE_CASE(Silu);
        NNOPS_NORM_GATE_CASE(Gelu);
        NNOPS_NORM_GATE_CASE(Relu);
        NNOPS_NORM_GATE_CASE(Sigmoid);
        NNOPS_NORM_GATE_CASE(Tanh);
    default:
        rms_norm_apply_gated_act<T, WithResidual, EpilogueActivateType::None>(
            x, r, g, y, n, inv_rms, scale, scale_is_scalar, add_to, one_plus);
        return;
    }
#undef NNOPS_NORM_GATE_CASE
}

// ============================================================
// Adaptive normalization (AdaLN / AdaLN-Zero / FiLM)
// ============================================================

/// The row kernel has already applied the norm's own scale/bias; this folds the
/// conditioning in before the store:
///   y = gate? · ((mod_scale + offset) · y + mod_shift)
/// so DiT's re-scaling, FiLM's shift and AdaLN-Zero's gate cost no extra pass
/// over the hidden — the unfused form stored the row, read it back, and stored
/// it again once per conditioning term.
template <typename T, bool WithGate> struct ModulationPost {
    const T* scale;
    const T* shift;
    const T* gate;  ///< WithGate only
    float offset;

    template <class V> V vec(const V& v, int64_t i) const
    {
        const auto vs = v_add(v_load(scale + i), v_set1(scale, offset));
        auto rv = v_fmadd(v, vs, v_load(shift + i));
        if constexpr (WithGate) {
            rv = v_mul(rv, v_load(gate + i));
        }
        return rv;
    }
    float scalar(float v, int64_t i) const
    {
        float rv = v * (s_load(&scale[i]) + offset) + s_load(&shift[i]);
        if constexpr (WithGate) {
            rv *= s_load(&gate[i]);
        }
        return rv;
    }
};

/// One modulated row. HasMean is the norm type; scale and bias are runtime
/// (AdaLN's LayerNorm has neither), the residual and the gate compile-time.
template <typename T, bool HasMean, bool WithResidual, bool WithGate>
inline void norm_apply_modulated_gate(const T* x, const T* r, T* y, int64_t n, float mean_val,
                                      float inv, const T* scale, const T* bias, bool has_scale,
                                      bool has_bias, bool scale_is_scalar, bool add_to,
                                      bool one_plus, const T* mod_scale, const T* mod_shift,
                                      const T* gate, float offset)
{
    ModulationPost<T, WithGate> post{mod_scale, mod_shift, gate, offset};
    if (has_scale) {
        if (has_bias) {
            if constexpr (WithResidual) {
                kernel::norm_apply_row_add_post_dispatch<T, HasMean, true, true>(
                    x, r, y, n, mean_val, inv, scale, bias, scale_is_scalar, add_to, one_plus,
                    post);
            } else {
                kernel::norm_apply_row_post_dispatch<T, HasMean, true, true>(
                    x, y, n, mean_val, inv, scale, bias, scale_is_scalar, add_to, one_plus, post);
            }
        } else {
            if constexpr (WithResidual) {
                kernel::norm_apply_row_add_post_dispatch<T, HasMean, true, false>(
                    x, r, y, n, mean_val, inv, scale, nullptr, scale_is_scalar, add_to, one_plus,
                    post);
            } else {
                kernel::norm_apply_row_post_dispatch<T, HasMean, true, false>(
                    x, y, n, mean_val, inv, scale, nullptr, scale_is_scalar, add_to, one_plus,
                    post);
            }
        }
        return;
    }
    if constexpr (WithResidual) {
        kernel::norm_apply_row_add_post_dispatch<T, HasMean, false, false>(
            x, r, y, n, mean_val, inv, nullptr, nullptr, false, add_to, false, post);
    } else {
        kernel::norm_apply_row_post_dispatch<T, HasMean, false, false>(
            x, y, n, mean_val, inv, nullptr, nullptr, false, add_to, false, post);
    }
}

/// Runtime dispatch over the residual / gate shape.
template <typename T, bool HasMean>
inline void norm_apply_modulated_row(const T* x, const T* r, T* y, int64_t n, float mean_val,
                                     float inv, const T* scale, const T* bias, bool has_scale,
                                     bool has_bias, bool scale_is_scalar, bool add_to,
                                     bool one_plus, const T* mod_scale, const T* mod_shift,
                                     const T* gate, float offset, bool with_gate,
                                     bool with_residual)
{
    if (with_residual) {
        if (with_gate) {
            norm_apply_modulated_gate<T, HasMean, true, true>(
                x, r, y, n, mean_val, inv, scale, bias, has_scale, has_bias, scale_is_scalar,
                add_to, one_plus, mod_scale, mod_shift, gate, offset);
        } else {
            norm_apply_modulated_gate<T, HasMean, true, false>(
                x, r, y, n, mean_val, inv, scale, bias, has_scale, has_bias, scale_is_scalar,
                add_to, one_plus, mod_scale, mod_shift, nullptr, offset);
        }
    } else {
        if (with_gate) {
            norm_apply_modulated_gate<T, HasMean, false, true>(
                x, nullptr, y, n, mean_val, inv, scale, bias, has_scale, has_bias, scale_is_scalar,
                add_to, one_plus, mod_scale, mod_shift, gate, offset);
        } else {
            norm_apply_modulated_gate<T, HasMean, false, false>(
                x, nullptr, y, n, mean_val, inv, scale, bias, has_scale, has_bias, scale_is_scalar,
                add_to, one_plus, mod_scale, mod_shift, nullptr, offset);
        }
    }
}

/// Base offset of row `row` in a modulation tensor that broadcasts NumPy-style
/// against X's leading dims: a missing or size-1 axis contributes nothing, and
/// the shared last axis (the normalized width) is addressed per element.
inline int64_t modulation_row_offset(const TensorView& mod, const TensorView& x, int64_t row,
                                     int64_t axis)
{
    const int64_t xr = x.rank();
    const int64_t mr = mod.rank();
    int64_t off = 0;
    int64_t rem = row;
    for (int64_t d = axis - 1; d >= 0; --d) {
        const int64_t dim = x.shape(d);
        if (dim <= 0) {
            continue;
        }
        const int64_t coord = rem % dim;
        rem /= dim;
        if (coord == 0) {
            continue;
        }
        const int64_t md = d - (xr - mr);
        if (md < 0 || mod.shape(md) == 1) {
            continue;
        }
        off += coord * mod.stride_elems(md);
    }
    return off;
}

/// Normalized axis dimensions for row-wise norm dispatch.
struct NormDims {
    int64_t axis;            ///< Normalized axis (0 <= axis < rank)
    int64_t num_rows;        ///< Number of independent rows to process
    int64_t norm_size;       ///< Elements per normalization row
    bool is_contiguous_tail; ///< True when axis == rank-1 (SIMD fast path)
};

inline NormDims compute_norm_dims(const TensorView& X, int64_t raw_axis)
{
    const int64_t rank = X.rank();
    int64_t axis = raw_axis;
    if (axis < 0) {
        axis += rank;
    }
    NNOPS_ASSERT(axis >= 0 && axis < rank);

    int64_t num_rows = 1;
    for (int64_t i = 0; i < axis; ++i) {
        num_rows *= X.shape(i);
    }
    int64_t norm_size = 1;
    for (int64_t i = axis; i < rank; ++i) {
        norm_size *= X.shape(i);
    }

    // Only a dense tensor's trailing axis is one physically contiguous run.
    const bool contiguous_tail = (axis == rank - 1) && !is_channel_last(X.layout());
    return {axis, num_rows, norm_size, contiguous_tail};
}

// ============================================================
// BatchNorm
// ============================================================

template <typename T>
void batch_norm_impl(const NormAttributes& attrs, TensorView& output,
                     std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& X = inputs[0];
    const auto& scale = inputs[1];
    const auto& bias = inputs[2];
    const auto& mean = inputs[3];
    const auto& var = inputs[4];

    const int64_t rank = X.rank();
    const float epsilon = attrs.epsilon;

    int64_t N, C;
    if (rank == 1) {
        N = X.shape(0);
        C = 1;
    } else {
        N = X.shape(0);
        C = X.shape(X.rank() - 1);
    }

    const auto* x_ptr = X.ptr<T>();
    const auto* s_ptr = scale.ptr<T>();
    const auto* b_ptr = bias.ptr<T>();
    const auto* m_ptr = mean.ptr<T>();
    const auto* v_ptr = var.ptr<T>();
    auto* y_ptr = output.ptr<T>();

    if (attrs.spatial) {
        PoolArray<float> ns_f(static_cast<size_t>(C));
        PoolArray<float> nb_f(static_cast<size_t>(C));
        for (int64_t c = 0; c < C; ++c) {
            float inv_std = 1.0f / std::sqrt(s_load(&v_ptr[c]) + epsilon);
            ns_f[static_cast<size_t>(c)] = inv_std * s_load(&s_ptr[c]);
            nb_f[static_cast<size_t>(c)] =
                s_load(&b_ptr[c]) - s_load(&m_ptr[c]) * ns_f[static_cast<size_t>(c)];
        }

        // One row is a spatial line of W pixels × C channels (channel-last, so
        // W is the second-innermost axis); each pixel takes the same C factors.
        // Rank 2 has C as its only row.
        const int64_t pixels = (rank >= 3) ? X.shape(rank - 2) : 1;
        const int64_t num_rows = (rank >= 3) ? X.total_rows() : N;
        const int64_t x_rs = X.row_stride_elems();
        const int64_t y_rs = output.row_stride_elems();

        // A pixel is C contiguous channels, so the per-channel factors are one
        // vector load each and the whole pixel normalizes 8 lanes at a time.
        // The math stays f32 for every storage dtype (v_load_f32 widens, and
        // rounds once on the store), as the scalar form did.
        const auto process_row = [&](int64_t r) {
            const T* xr = x_ptr + r * x_rs;
            T* yr = y_ptr + r * y_rs;
            for (int64_t p = 0; p < pixels; ++p) {
                const int64_t base = p * C;
                int64_t c = 0;
                for (; c + 8 <= C; c += 8) {
                    const auto ns = v_load(ns_f.data() + c);
                    const auto nb = v_load(nb_f.data() + c);
                    const auto rv = v_add(v_mul(simd::v_load_f32(xr + base + c), ns), nb);
                    if (attrs.add_to) {
                        v_store_f32(yr + base + c, v_add(simd::v_load_f32(yr + base + c), rv));
                    } else {
                        v_store_f32(yr + base + c, rv);
                    }
                }
                for (; c < C; ++c) {
                    const int64_t idx = base + c;
                    const float v = s_load(&xr[idx]) * ns_f[static_cast<size_t>(c)] +
                                    nb_f[static_cast<size_t>(c)];
                    s_store_add(&yr[idx], v, attrs.add_to);
                }
            }
        };

        ctx.cpu.run(0, num_rows, process_row);
    } else {
        const int64_t total = X.numel();
        kernel::batch_norm_process_nonspatial_block<T>(x_ptr, y_ptr, s_ptr, b_ptr, m_ptr, v_ptr, 0,
                                                       total, epsilon, attrs.add_to);
    }
}

// ============================================================
// LayerNorm
// ============================================================

template <typename T>
void layer_norm_general_scalar(const T* x_ptr, const T* s_ptr, const T* b_ptr, T* y_ptr,
                               int64_t num_rows, int64_t norm_size, int64_t axis,
                               const TensorView& X, const TensorView& Y, const TensorView& scale,
                               float epsilon, bool add_to, bool has_bias, bool one_plus,
                               const ComputeContext& ctx)
{
    const auto inner_offsets = compute_inner_offsets(X, axis, norm_size);
    const auto y_inner_offsets = compute_inner_offsets(Y, axis, norm_size);
    const bool scale_is_scalar = (scale.numel() == 1);

    const auto process_row = [&](int64_t row) {
        const int64_t x_base = outer_base(X, row, axis);
        const int64_t y_base = outer_base(Y, row, axis);

        float mean_val = 0.0f;
        float M2 = 0.0f;
        for (int64_t i = 0; i < norm_size; ++i) {
            float x = s_load(&x_ptr[x_base + inner_offsets[static_cast<size_t>(i)]]);
            float delta = x - mean_val;
            mean_val += delta / static_cast<float>(i + 1);
            float delta2 = x - mean_val;
            M2 += delta * delta2;
        }
        const float var_val = M2 / static_cast<float>(norm_size);
        const float inv_std = 1.0f / std::sqrt(var_val + epsilon);

        for (int64_t i = 0; i < norm_size; ++i) {
            const int64_t x_off = x_base + inner_offsets[static_cast<size_t>(i)];
            const int64_t y_off = y_base + y_inner_offsets[static_cast<size_t>(i)];
            int64_t s_idx = scale_is_scalar ? 0 : i;
            float s = s_load(&s_ptr[s_idx]) + (one_plus ? 1.0f : 0.0f);
            float b = (has_bias && b_ptr) ? s_load(&b_ptr[s_idx]) : 0.0f;
            float val = (s_load(&x_ptr[x_off]) - mean_val) * inv_std * s + b;
            s_store_add(&y_ptr[y_off], val, add_to);
        }
    };

    ctx.cpu.run(0, num_rows, process_row);
}

template <typename T>
void layer_norm_impl(const NormAttributes& attrs, TensorView& output,
                     std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& X = inputs[0];

    // The modulation pair (then AdaLN-Zero's gate) is the last group of slots,
    // so the type's own slots keep their indices.
    const bool modulate = attrs.modulate;
    const bool mod_gate = modulate && attrs.modulation_gate;
    const int64_t n_trail = modulate ? (mod_gate ? 3 : 2) : 0;
    const int64_t base_n = static_cast<int64_t>(inputs.size()) - n_trail;

    // AdaLN's LayerNorm has no learned affine; an empty scale slot says so.
    const bool has_scale = modulate ? (base_n >= 2 && inputs[1].rank() > 0) : true;
    const bool has_bias = base_n >= 3 && !inputs[2].is_empty();

    const float epsilon = attrs.epsilon;

    const auto* x_ptr = X.ptr<T>();
    const auto* s_ptr = has_scale ? inputs[1].ptr<T>() : nullptr;
    const auto* b_ptr = has_bias ? inputs[2].ptr<T>() : nullptr;
    auto* y_ptr = output.ptr<T>();

    // The pre-add residual is the trailing slot, after the optional bias.
    const bool has_res = attrs.add_residual && base_n >= 4 && !inputs[3].is_empty();
    const auto* r_ptr = has_res ? inputs[3].ptr<T>() : nullptr;

    const TensorView* mod_scale = modulate ? &inputs[static_cast<size_t>(base_n)] : nullptr;
    const TensorView* mod_shift = modulate ? &inputs[static_cast<size_t>(base_n + 1)] : nullptr;
    const TensorView* mod_gate_v = mod_gate ? &inputs[static_cast<size_t>(base_n + 2)] : nullptr;

    const bool add_to = attrs.add_to;
    const auto dims = compute_norm_dims(X, attrs.axis);

    if (!dims.is_contiguous_tail) {
        NNOPS_ASSERT(!has_res && "norm residual needs the contiguous tail");
        layer_norm_general_scalar<T>(x_ptr, s_ptr, b_ptr, y_ptr, dims.num_rows, dims.norm_size,
                                     dims.axis, X, output, inputs[1], epsilon, add_to, has_bias,
                                     attrs.one_plus_weight, ctx);
        return;
    }

    const int64_t x_row_stride = X.row_stride_elems();
    const int64_t y_row_stride = output.row_stride_elems();
    const int64_t r_row_stride = has_res ? inputs[3].row_stride_elems() : 0;
    const bool scale_is_scalar = has_scale && (inputs[1].numel() == 1);
    const bool one_plus = attrs.one_plus_weight;
    const float offset = attrs.modulation_scale_offset;

    const auto process_row = [&](int64_t row) {
        const int64_t x_off = row * x_row_stride;
        const int64_t r_off = row * r_row_stride;
        const int64_t y_off = row * y_row_stride;
        const auto [sum, sum_sq] =
            has_res ?
                kernel::norm_reduce_sum_sq_add<T>(x_ptr + x_off, r_ptr + r_off, dims.norm_size) :
                kernel::norm_reduce_sum_sq<T>(x_ptr + x_off, dims.norm_size);

        const float inv_n = 1.0f / static_cast<float>(dims.norm_size);
        const float mean_val = sum * inv_n;
        float var_val = sum_sq * inv_n - mean_val * mean_val;
        if (var_val < 0.0f) {
            var_val = 0.0f;
        }
        const float inv_std = 1.0f / std::sqrt(var_val + epsilon);

        if (modulate) {
            const int64_t ms_off = modulation_row_offset(*mod_scale, X, row, dims.axis);
            const int64_t mt_off = modulation_row_offset(*mod_shift, X, row, dims.axis);
            const int64_t mg_off =
                mod_gate ? modulation_row_offset(*mod_gate_v, X, row, dims.axis) : 0;
            norm_apply_modulated_row<T, true>(
                x_ptr + x_off, has_res ? r_ptr + r_off : nullptr, y_ptr + y_off, dims.norm_size,
                mean_val, inv_std, s_ptr, b_ptr, has_scale, has_bias, scale_is_scalar, add_to,
                one_plus, mod_scale->ptr<T>() + ms_off, mod_shift->ptr<T>() + mt_off,
                mod_gate ? mod_gate_v->ptr<T>() + mg_off : nullptr, offset, mod_gate, has_res);
            return;
        }

        if (has_res) {
            if (b_ptr) {
                kernel::norm_apply_row_add_dispatch<T, true, true, true>(
                    x_ptr + x_off, r_ptr + r_off, y_ptr + y_off, dims.norm_size, mean_val, inv_std,
                    s_ptr, b_ptr, scale_is_scalar, add_to, one_plus);
            } else {
                kernel::norm_apply_row_add_dispatch<T, true, true, false>(
                    x_ptr + x_off, r_ptr + r_off, y_ptr + y_off, dims.norm_size, mean_val, inv_std,
                    s_ptr, nullptr, scale_is_scalar, add_to, one_plus);
            }
        } else if (b_ptr) {
            kernel::norm_apply_row_dispatch<T, true, true, true>(
                x_ptr + x_off, y_ptr + y_off, dims.norm_size, mean_val, inv_std, s_ptr, b_ptr,
                scale_is_scalar, add_to, one_plus);
        } else {
            kernel::norm_apply_row_dispatch<T, true, true, false>(
                x_ptr + x_off, y_ptr + y_off, dims.norm_size, mean_val, inv_std, s_ptr, nullptr,
                scale_is_scalar, add_to, one_plus);
        }
    };

    ctx.cpu.run(0, dims.num_rows, process_row);
}

// ============================================================
// RMSNorm
// ============================================================

template <typename T>
void rms_norm_general_scalar(const T* x_ptr, const T* s_ptr, T* y_ptr, int64_t num_rows,
                             int64_t norm_size, int64_t axis, const TensorView& X,
                             const TensorView& Y, const TensorView& scale, float epsilon,
                             bool add_to, bool one_plus, const ComputeContext& ctx)
{
    const auto inner_offsets = compute_inner_offsets(X, axis, norm_size);
    const auto y_inner_offsets = compute_inner_offsets(Y, axis, norm_size);
    const bool scale_is_scalar = (scale.numel() == 1);

    const auto process_row = [&](int64_t row) {
        const int64_t x_base = outer_base(X, row, axis);
        const int64_t y_base = outer_base(Y, row, axis);

        float sum_sq = 0.0f;
        for (int64_t i = 0; i < norm_size; ++i) {
            float x = s_load(&x_ptr[x_base + inner_offsets[static_cast<size_t>(i)]]);
            sum_sq += x * x;
        }

        const float rms = std::sqrt(sum_sq / static_cast<float>(norm_size) + epsilon);
        const float inv_rms = 1.0f / rms;

        for (int64_t i = 0; i < norm_size; ++i) {
            const int64_t x_off = x_base + inner_offsets[static_cast<size_t>(i)];
            const int64_t y_off = y_base + y_inner_offsets[static_cast<size_t>(i)];
            int64_t s_idx = scale_is_scalar ? 0 : i;
            float s = s_load(&s_ptr[s_idx]) + (one_plus ? 1.0f : 0.0f);
            float val = s_load(&x_ptr[x_off]) * inv_rms * s;
            s_store_add(&y_ptr[y_off], val, add_to);
        }
    };

    ctx.cpu.run(0, num_rows, process_row);
}

template <typename T>
void rms_norm_impl(const NormAttributes& attrs, TensorView& output,
                   std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& X = inputs[0];

    const bool modulate = attrs.modulate;
    const bool mod_gate = modulate && attrs.modulation_gate;
    const int64_t n_trail = modulate ? (mod_gate ? 3 : 2) : 0;
    const int64_t base_n = static_cast<int64_t>(inputs.size()) - n_trail;
    const bool has_scale = modulate ? (base_n >= 2 && inputs[1].rank() > 0) : true;

    const float epsilon = attrs.epsilon;

    const auto* x_ptr = X.ptr<T>();
    const auto* s_ptr = has_scale ? inputs[1].ptr<T>() : nullptr;
    auto* y_ptr = output.ptr<T>();

    // The gate is the trailing slot; the residual, when present, sits before
    // it. With neither, the third input would be the gate, not a residual —
    // so the residual slot is counted back from the end rather than assumed.
    const bool has_gate = attrs.gate_activation != EpilogueActivateType::None;
    const int64_t trailing = (has_gate ? 1 : 0) + n_trail;
    const int64_t res_idx = static_cast<int64_t>(inputs.size()) - 1 - trailing;
    const bool has_res =
        attrs.add_residual && res_idx >= 2 && !inputs[static_cast<size_t>(res_idx)].is_empty();
    const auto* r_ptr = has_res ? inputs[static_cast<size_t>(res_idx)].ptr<T>() : nullptr;
    const auto* g_ptr = has_gate ? inputs.back().ptr<T>() : nullptr;

    const TensorView* mod_scale = modulate ? &inputs[static_cast<size_t>(base_n)] : nullptr;
    const TensorView* mod_shift = modulate ? &inputs[static_cast<size_t>(base_n + 1)] : nullptr;
    const TensorView* mod_gate_v = mod_gate ? &inputs[static_cast<size_t>(base_n + 2)] : nullptr;

    const bool add_to = attrs.add_to;
    const auto dims = compute_norm_dims(X, attrs.axis);

    if (!dims.is_contiguous_tail) {
        NNOPS_ASSERT(!has_res && "norm residual needs the contiguous tail");
        rms_norm_general_scalar<T>(x_ptr, s_ptr, y_ptr, dims.num_rows, dims.norm_size, dims.axis, X,
                                   output, inputs[1], epsilon, add_to, attrs.one_plus_weight, ctx);
        return;
    }

    const int64_t x_row_stride = X.row_stride_elems();
    const int64_t y_row_stride = output.row_stride_elems();
    const int64_t r_row_stride =
        has_res ? inputs[static_cast<size_t>(res_idx)].row_stride_elems() : 0;
    const int64_t g_row_stride = has_gate ? inputs.back().row_stride_elems() : 0;
    const bool scale_is_scalar = has_scale && (inputs[1].numel() == 1);
    const bool one_plus = attrs.one_plus_weight;
    const EpilogueActivateType gate_act = attrs.gate_activation;
    const float offset = attrs.modulation_scale_offset;

    const auto process_row = [&](int64_t row) {
        const int64_t x_off = row * x_row_stride;
        const int64_t r_off = row * r_row_stride;
        const int64_t y_off = row * y_row_stride;
        const auto [sum, sum_sq] =
            has_res ?
                kernel::norm_reduce_sum_sq_add<T>(x_ptr + x_off, r_ptr + r_off, dims.norm_size) :
                kernel::norm_reduce_sum_sq<T>(x_ptr + x_off, dims.norm_size);
        (void)sum;
        const float inv_rms =
            1.0f / std::sqrt(sum_sq / static_cast<float>(dims.norm_size) + epsilon);
        if (modulate) {
            const int64_t ms_off = modulation_row_offset(*mod_scale, X, row, dims.axis);
            const int64_t mt_off = modulation_row_offset(*mod_shift, X, row, dims.axis);
            const int64_t mg_off =
                mod_gate ? modulation_row_offset(*mod_gate_v, X, row, dims.axis) : 0;
            norm_apply_modulated_row<T, false>(
                x_ptr + x_off, has_res ? r_ptr + r_off : nullptr, y_ptr + y_off, dims.norm_size,
                0.0f, inv_rms, s_ptr, nullptr, has_scale, false, scale_is_scalar, add_to, one_plus,
                mod_scale->ptr<T>() + ms_off, mod_shift->ptr<T>() + mt_off,
                mod_gate ? mod_gate_v->ptr<T>() + mg_off : nullptr, offset, mod_gate, has_res);
            return;
        }
        if (has_gate) {
            // act(gate) multiplies the normalized value inside the row kernel.
            const T* g_row = g_ptr + row * g_row_stride;
            if (has_res) {
                rms_norm_apply_gated<T, true>(gate_act, x_ptr + x_off, r_ptr + r_off, g_row,
                                              y_ptr + y_off, dims.norm_size, inv_rms, s_ptr,
                                              scale_is_scalar, add_to, one_plus);
            } else {
                rms_norm_apply_gated<T, false>(gate_act, x_ptr + x_off, nullptr, g_row,
                                               y_ptr + y_off, dims.norm_size, inv_rms, s_ptr,
                                               scale_is_scalar, add_to, one_plus);
            }
            return;
        }
        if (has_res) {
            kernel::norm_apply_row_add_dispatch<T, false, true, false>(
                x_ptr + x_off, r_ptr + r_off, y_ptr + y_off, dims.norm_size, 0.0f, inv_rms, s_ptr,
                nullptr, scale_is_scalar, add_to, one_plus);
        } else {
            kernel::norm_apply_row_dispatch<T, false, true, false>(
                x_ptr + x_off, y_ptr + y_off, dims.norm_size, 0.0f, inv_rms, s_ptr, nullptr,
                scale_is_scalar, add_to, one_plus);
        }
    };

    ctx.cpu.run(0, dims.num_rows, process_row);
}

// ============================================================
// L2Norm
// ============================================================

template <typename T>
void l2_norm_general_scalar(const T* x_ptr, T* y_ptr, int64_t num_rows, int64_t norm_size,
                            int64_t axis, const TensorView& X, const TensorView& Y, float epsilon,
                            bool add_to, const ComputeContext& ctx)
{
    const auto inner_offsets = compute_inner_offsets(X, axis, norm_size);
    const auto y_inner_offsets = compute_inner_offsets(Y, axis, norm_size);

    const auto process_row = [&](int64_t row) {
        const int64_t x_base = outer_base(X, row, axis);
        const int64_t y_base = outer_base(Y, row, axis);

        float sum_sq = 0.0f;
        for (int64_t i = 0; i < norm_size; ++i) {
            float x = s_load(&x_ptr[x_base + inner_offsets[static_cast<size_t>(i)]]);
            sum_sq += x * x;
        }

        const float norm_val = std::sqrt(sum_sq + epsilon);
        const float inv_norm = 1.0f / norm_val;

        for (int64_t i = 0; i < norm_size; ++i) {
            const int64_t x_off = x_base + inner_offsets[static_cast<size_t>(i)];
            const int64_t y_off = y_base + y_inner_offsets[static_cast<size_t>(i)];
            float val = s_load(&x_ptr[x_off]) * inv_norm;
            s_store_add(&y_ptr[y_off], val, add_to);
        }
    };

    ctx.cpu.run(0, num_rows, process_row);
}

template <typename T>
void l2_norm_impl(const NormAttributes& attrs, TensorView& output,
                  std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& X = inputs[0];

    const bool modulate = attrs.modulate;
    const bool mod_gate = modulate && attrs.modulation_gate;
    const int64_t n_trail = modulate ? (mod_gate ? 3 : 2) : 0;
    const int64_t base_n = static_cast<int64_t>(inputs.size()) - n_trail;

    const float epsilon = attrs.epsilon;

    const auto* x_ptr = X.ptr<T>();
    auto* y_ptr = output.ptr<T>();

    // L2Norm has no scale, so the residual is the second input.
    const bool has_res = attrs.add_residual && base_n >= 2 && !inputs[1].is_empty();
    const auto* r_ptr = has_res ? inputs[1].ptr<T>() : nullptr;

    const TensorView* mod_scale = modulate ? &inputs[static_cast<size_t>(base_n)] : nullptr;
    const TensorView* mod_shift = modulate ? &inputs[static_cast<size_t>(base_n + 1)] : nullptr;
    const TensorView* mod_gate_v = mod_gate ? &inputs[static_cast<size_t>(base_n + 2)] : nullptr;

    const bool add_to = attrs.add_to;
    const auto dims = compute_norm_dims(X, attrs.axis);

    if (!dims.is_contiguous_tail) {
        NNOPS_ASSERT(!has_res && "norm residual needs the contiguous tail");
        l2_norm_general_scalar<T>(x_ptr, y_ptr, dims.num_rows, dims.norm_size, dims.axis, X, output,
                                  epsilon, add_to, ctx);
        return;
    }

    const int64_t x_row_stride = X.row_stride_elems();
    const int64_t y_row_stride = output.row_stride_elems();
    const int64_t r_row_stride = has_res ? inputs[1].row_stride_elems() : 0;

    const auto process_row = [&](int64_t row) {
        const int64_t x_off = row * x_row_stride;
        const int64_t r_off = row * r_row_stride;
        const int64_t y_off = row * y_row_stride;
        const auto [sum, sum_sq] =
            has_res ?
                kernel::norm_reduce_sum_sq_add<T>(x_ptr + x_off, r_ptr + r_off, dims.norm_size) :
                kernel::norm_reduce_sum_sq<T>(x_ptr + x_off, dims.norm_size);
        (void)sum;
        const float inv_norm = 1.0f / std::sqrt(sum_sq + epsilon);
        if (modulate) {
            const int64_t ms_off = modulation_row_offset(*mod_scale, X, row, dims.axis);
            const int64_t mt_off = modulation_row_offset(*mod_shift, X, row, dims.axis);
            const int64_t mg_off =
                mod_gate ? modulation_row_offset(*mod_gate_v, X, row, dims.axis) : 0;
            norm_apply_modulated_row<T, false>(
                x_ptr + x_off, has_res ? r_ptr + r_off : nullptr, y_ptr + y_off, dims.norm_size,
                0.0f, inv_norm, nullptr, nullptr, false, false, false, add_to, false,
                mod_scale->ptr<T>() + ms_off, mod_shift->ptr<T>() + mt_off,
                mod_gate ? mod_gate_v->ptr<T>() + mg_off : nullptr, attrs.modulation_scale_offset,
                mod_gate, has_res);
            return;
        }
        if (has_res) {
            kernel::norm_apply_row_add<T, false, false, false>(
                x_ptr + x_off, r_ptr + r_off, y_ptr + y_off, dims.norm_size, 0.0f, inv_norm,
                nullptr, nullptr, false, add_to);
        } else {
            kernel::norm_apply_row<T, false, false, false>(x_ptr + x_off, y_ptr + y_off,
                                                           dims.norm_size, 0.0f, inv_norm, nullptr,
                                                           nullptr, false, add_to);
        }
    };

    ctx.cpu.run(0, dims.num_rows, process_row);
}

// ============================================================
// GroupNorm
// ============================================================

template <typename T>
void group_norm_impl(const NormAttributes& attrs, TensorView& output,
                     std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& X = inputs[0];
    const auto& scale = inputs[1];

    const bool modulate = attrs.modulate;
    const bool mod_gate = modulate && attrs.modulation_gate;
    const int64_t n_trail = modulate ? (mod_gate ? 3 : 2) : 0;
    const int64_t base_n = static_cast<int64_t>(inputs.size()) - n_trail;
    const bool has_bias = base_n >= 3 && !inputs[2].is_empty();

    const int64_t rank = X.rank();
    NNOPS_ASSERT(rank >= 2);

    const int64_t N = X.shape(0);
    const int64_t C = X.shape(X.rank() - 1);
    int64_t G = attrs.num_groups;
    if (G <= 0) {
        G = 1;
    }
    NNOPS_ASSERT(C % G == 0);

    const int64_t channels_per_group = C / G;

    int64_t spatial_size = 1;
    for (int64_t i = 1; i < rank - 1; ++i) {
        spatial_size *= X.shape(i);
    }
    const int64_t norm_size = channels_per_group * spatial_size;
    const float inv_norm = 1.0f / static_cast<float>(norm_size);
    const float epsilon = attrs.epsilon;

    const auto* x_ptr = X.ptr<T>();
    const auto* s_ptr = scale.ptr<T>();
    const auto* b_ptr = (has_bias && !inputs[2].is_empty()) ? inputs[2].ptr<T>() : nullptr;
    auto* y_ptr = output.ptr<T>();
    const bool add_to = attrs.add_to;

    // FiLM: one conditioning pair per channel, optionally one per sample.
    const TensorView* mod_scale_v = modulate ? &inputs[static_cast<size_t>(base_n)] : nullptr;
    const TensorView* mod_shift_v = modulate ? &inputs[static_cast<size_t>(base_n + 1)] : nullptr;
    const TensorView* mod_gate_v = mod_gate ? &inputs[static_cast<size_t>(base_n + 2)] : nullptr;
    const auto* ms_ptr = modulate ? mod_scale_v->ptr<T>() : nullptr;
    const auto* mt_ptr = modulate ? mod_shift_v->ptr<T>() : nullptr;
    const auto* mg_ptr = mod_gate ? mod_gate_v->ptr<T>() : nullptr;
    // [N, C] steps by a sample; a size-1 leading axis (or [C]) does not.
    const auto mod_n_stride = [](const TensorView* v) {
        return (v != nullptr && v->rank() == 2 && v->shape(0) != 1) ? v->stride_elems(0) : 0;
    };
    const int64_t ms_n = mod_n_stride(mod_scale_v);
    const int64_t mt_n = mod_n_stride(mod_shift_v);
    const int64_t mg_n = mod_n_stride(mod_gate_v);
    const float mod_offset = attrs.modulation_scale_offset;

    // Channels are contiguous inside each pixel, so a group is
    // channels_per_group consecutive elements at every spatial offset.
    const auto x_sp = build_offsets(X, 1, rank - 2, spatial_size);
    const auto y_sp = build_offsets(output, 1, rank - 2, spatial_size);
    const int64_t x_n = X.stride_elems(0);
    const int64_t y_n = output.stride_elems(0);

    const auto process_group = [&](int64_t ng) {
        const int64_t n = ng / G;
        const int64_t g = ng % G;
        const int64_t c_start = g * channels_per_group;

        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (int64_t p = 0; p < spatial_size; ++p) {
            const int64_t base = x_n * n + x_sp[static_cast<size_t>(p)] + c_start;
            for (int64_t k = 0; k < channels_per_group; ++k) {
                const float v = s_load(&x_ptr[base + k]);
                sum += v;
                sum_sq += v * v;
            }
        }

        const float mean_val = sum * inv_norm;
        float var_val = sum_sq * inv_norm - mean_val * mean_val;
        if (var_val < 0.0f) {
            var_val = 0.0f;
        }
        const float inv_std = 1.0f / std::sqrt(var_val + epsilon);

        for (int64_t p = 0; p < spatial_size; ++p) {
            const int64_t xo = x_n * n + x_sp[static_cast<size_t>(p)] + c_start;
            const int64_t yo = y_n * n + y_sp[static_cast<size_t>(p)] + c_start;
            for (int64_t k = 0; k < channels_per_group; ++k) {
                const int64_t c_global = c_start + k;
                const float s = s_load(&s_ptr[c_global]);
                const float b = (has_bias && b_ptr) ? s_load(&b_ptr[c_global]) : 0.0f;
                float val = (s_load(&x_ptr[xo + k]) - mean_val) * inv_std * s + b;
                if (modulate) {
                    // The last axis is the channel, so its stride is 1.
                    const int64_t mo = c_global;
                    val = val * (s_load(&ms_ptr[ms_n * n + mo]) + mod_offset) +
                          s_load(&mt_ptr[mt_n * n + mo]);
                    if (mod_gate) {
                        val *= s_load(&mg_ptr[mg_n * n + mo]);
                    }
                }
                s_store_add(&y_ptr[yo + k], val, add_to);
            }
        }
    };

    ctx.cpu.run(0, N * G, process_group);
}

// ============================================================
// Quantized input (s8/u8): LayerNorm / RMSNorm / L2Norm
// ============================================================

/// Row (scale, zero_point): the per-row buffers when present, else the
/// per-tensor scalars; a missing zero_point buffer means zero.
inline void row_qparam(const QuantParams& qp, int64_t row, float& scale, float& zp)
{
    if (qp.scale_data != nullptr) {
        scale = qp.scale_data[row];
        zp = (qp.zero_point_data != nullptr) ? static_cast<float>(qp.zero_point_data[row]) : 0.0f;
    } else {
        scale = qp.scale;
        zp = static_cast<float>(qp.zero_point);
    }
}

template <typename T, typename InT>
inline void dequant_row(T* dst, const InT* src, int64_t n, float scale, float zp)
{
    const int n_i = static_cast<int>(n);
    quant_kernel::dequantization<InT>(1, n_i, dst, n_i, src, n_i, &scale, &zp);
}

template <typename OutT>
inline void requant_row(OutT* dst, const float* src, int64_t n, float scale, float zp)
{
    const int n_i = static_cast<int>(n);
    quant_kernel::quantization<OutT>(1, n_i, dst, n_i, src, n_i, &scale, &zp);
}

/// Per row: dequantize -> normalize in place -> write back; HasMean / HasScale
/// pick the LayerNorm / RMSNorm / L2Norm variants.
template <bool HasMean, bool HasScale>
void norm_quant_rows(const NormAttributes& attrs, TensorView& output,
                     std::span<const TensorView> inputs, const ComputeContext& ctx, bool has_bias)
{
    const auto& X = inputs[0];

    // L2Norm has no scale input, so inputs[1] must not be formed blindly.
    const bool has_scale = (inputs.size() >= 2);
    const auto* s_ptr = has_scale ? inputs[1].ptr<float>() : nullptr;
    const bool scale_is_scalar = has_scale && (inputs[1].numel() == 1);

    const DataType x_dtype = X.data_type();
    const DataType y_dtype = output.data_type();
    const bool out_is_f32 = (y_dtype == DataType::f32);
    const bool out_is_f16 = (y_dtype == DataType::f16); // or bf16, narrowed below
    const bool is_i8 = (x_dtype == DataType::s8);

    const QuantParams& x_qp = X.quant_params();

    const float epsilon = attrs.epsilon;
    const auto dims = compute_norm_dims(X, attrs.axis);
    NNOPS_ASSERT(dims.is_contiguous_tail);

    const int64_t n = dims.norm_size;
    const bool one_plus = attrs.one_plus_weight;

    const auto* b_ptr = (has_bias && !inputs[2].is_empty()) ? inputs[2].ptr<float>() : nullptr;

    const auto process_row = [&](int64_t row) {
        const size_t row_idx = static_cast<size_t>(row);

        float x_scale, x_zp;
        row_qparam(x_qp, row, x_scale, x_zp);

        PoolPtr rowbuf_owner;
        float* rowbuf;
        if (out_is_f32) {
            rowbuf = output.ptr<float>(row_idx);
        } else {
            rowbuf_owner = PoolPtr(static_cast<size_t>(n) * sizeof(float));
            rowbuf = rowbuf_owner.as<float>();
        }

        const void* x_row = X.ptr<uint8_t>(row_idx);
        if (is_i8) {
            dequant_row<float>(rowbuf, static_cast<const int8_t*>(x_row), n, x_scale, x_zp);
        } else {
            dequant_row<float>(rowbuf, static_cast<const uint8_t*>(x_row), n, x_scale, x_zp);
        }

        auto [sum, sum_sq] = kernel::norm_reduce_sum_sq<float>(rowbuf, n);
        const float inv_n = 1.0f / static_cast<float>(n);
        float mean_val = 0.0f;
        float inv_norm;
        if constexpr (HasMean) { // LayerNorm
            mean_val = sum * inv_n;
            float var_val = sum_sq * inv_n - mean_val * mean_val;
            if (var_val < 0.0f) {
                var_val = 0.0f;
            }
            inv_norm = 1.0f / std::sqrt(var_val + epsilon);
        } else if constexpr (HasScale) { // RMSNorm
            inv_norm = 1.0f / std::sqrt(sum_sq * inv_n + epsilon);
        } else { // L2Norm
            inv_norm = 1.0f / std::sqrt(sum_sq + epsilon);
        }

        // In place: rowbuf is source and destination; add_to is rejected here.
        if constexpr (HasScale) {
            if (b_ptr != nullptr) {
                kernel::norm_apply_row_dispatch<float, HasMean, true, true>(
                    rowbuf, rowbuf, n, mean_val, inv_norm, s_ptr, b_ptr, scale_is_scalar, false,
                    one_plus);
            } else {
                kernel::norm_apply_row_dispatch<float, HasMean, true, false>(
                    rowbuf, rowbuf, n, mean_val, inv_norm, s_ptr, nullptr, scale_is_scalar, false,
                    one_plus);
            }
        } else {
            kernel::norm_apply_row<float, HasMean, false, false>(
                rowbuf, rowbuf, n, mean_val, inv_norm, nullptr, nullptr, false, false);
        }

        if (out_is_f32) {
            return;
        }

        if (out_is_f16 || y_dtype == DataType::bf16) {
            dispatch_f32_f16(y_dtype, "norm_cpu", [&](auto tag) {
                using U = typename decltype(tag)::type;
                U* y_row = output.ptr<U>(row_idx);
                for (int64_t i = 0; i < n; ++i) {
                    s_store(&y_row[i], rowbuf[i]);
                }
            });
            return;
        }

        float y_scale, y_zp;
        row_qparam(output.quant_params(), row, y_scale, y_zp);
        if (y_dtype == DataType::s8) {
            requant_row<int8_t>(output.ptr<int8_t>(row_idx), rowbuf, n, y_scale, y_zp);
        } else {
            requant_row<uint8_t>(output.ptr<uint8_t>(row_idx), rowbuf, n, y_scale, y_zp);
        }
    };

    ctx.cpu.run(0, dims.num_rows, process_row);
}

/// Dispatch the quantized path on the norm type; BatchNorm has no such form.
Status norm_quant_dispatch(const NormAttributes& attrs, TensorView& output,
                           std::span<const TensorView> inputs, const ComputeContext& ctx,
                           bool has_bias)
{
    switch (attrs.type) {
    case NormType::LayerNorm:
        norm_quant_rows<true, true>(attrs, output, inputs, ctx, has_bias);
        return Status::ok;
    case NormType::RMSNorm:
        norm_quant_rows<false, true>(attrs, output, inputs, ctx, false);
        return Status::ok;
    case NormType::L2Norm:
        norm_quant_rows<false, false>(attrs, output, inputs, ctx, false);
        return Status::ok;
    default:
        return Status::unsupported;
    }
}

/// Validate the quantized-input call before dispatching it.
Status norm_quant_impl(const NormAttributes& attrs, TensorView& output,
                       std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& X = inputs[0];

    // add_to would mean accumulating into a pre-existing quantized output.
    if (attrs.add_to) {
        return Status::unsupported;
    }

    // Quant params live on the view, so descriptor-only validate() cannot see
    // them; check them here, before anything is dereferenced.
    if (!X.is_quantized()) {
        return Status::invalid_argument;
    }

    const auto& x_qp = X.quant_params();
    if (x_qp.granularity != QuantGranularity::PerTensor &&
        x_qp.granularity != QuantGranularity::PerToken) {
        return Status::unsupported;
    }

    const auto dims = compute_norm_dims(X, attrs.axis);
    if (!dims.is_contiguous_tail) {
        // The per-row dequantization walks the row as one contiguous run.
        return Status::unsupported;
    }
    if (x_qp.granularity == QuantGranularity::PerToken &&
        (x_qp.scale_data == nullptr || x_qp.num_scales < dims.num_rows)) {
        return Status::invalid_argument;
    }

    const DataType y_dtype = output.data_type();
    if (!is_float_dtype(y_dtype) && !is_quantized_dtype(y_dtype)) {
        return Status::unsupported;
    }
    if (is_quantized_dtype(y_dtype)) {
        if (!output.is_quantized()) {
            return Status::invalid_argument;
        }
        const auto& y_qp = output.quant_params();
        if (y_qp.granularity != QuantGranularity::PerTensor &&
            y_qp.granularity != QuantGranularity::PerToken) {
            return Status::unsupported;
        }
        if (y_qp.granularity == QuantGranularity::PerToken &&
            (y_qp.scale_data == nullptr || y_qp.num_scales < dims.num_rows)) {
            return Status::invalid_argument;
        }
    }

    // L2Norm has no scale or bias, so inputs[1] may not exist.
    if (inputs.size() >= 2 && inputs[1].data_type() != DataType::f32) {
        return Status::invalid_argument;
    }
    const bool has_bias = (inputs.size() >= 3 && !inputs[2].is_empty());
    if (has_bias && inputs[2].data_type() != DataType::f32) {
        return Status::invalid_argument;
    }

    return norm_quant_dispatch(attrs, output, inputs, ctx, has_bias);
}

} // anonymous namespace

// ============================================================
// Dispatcher + entry point with dtype dispatch
// ============================================================

template <typename T>
void norm_impl(const NormAttributes& attrs, TensorView& output, std::span<const TensorView> inputs,
               const ComputeContext& ctx)
{
    switch (attrs.type) {
    case NormType::BatchNorm:
        batch_norm_impl<T>(attrs, output, inputs, ctx);
        return;
    case NormType::LayerNorm:
        layer_norm_impl<T>(attrs, output, inputs, ctx);
        return;
    case NormType::RMSNorm:
        rms_norm_impl<T>(attrs, output, inputs, ctx);
        return;
    case NormType::GroupNorm:
        group_norm_impl<T>(attrs, output, inputs, ctx);
        return;
    case NormType::L2Norm:
        l2_norm_impl<T>(attrs, output, inputs, ctx);
        return;
    }
    NNOPS_ASSERT(!"norm_cpu: unknown norm type");
}

Status norm_cpu(const NormAttributes& attrs, TensorView& output, std::span<const TensorView> inputs,
                const ComputeContext& ctx, void* /*workspace*/)
{
    const DataType x_dtype = inputs[0].data_type();

    if (is_quantized_dtype(x_dtype)) {
        return norm_quant_impl(attrs, output, inputs, ctx);
    }

    dispatch_f32_f16(x_dtype, "norm_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        norm_impl<T>(attrs, output, inputs, ctx);
    });
    return Status::ok;
}

} // namespace nnops::backend::cpu
