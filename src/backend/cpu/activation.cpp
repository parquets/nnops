/// @file activation.cpp
/// @brief SIMD-optimized CPU implementation of activation functions.
///
/// All 8 activation types are vectorized with the nnops SIMD abstraction layer.
/// Supports both f32 (v_f32x8) and f16 (v_f16x8) via a single templated implementation.
/// Processing is tiled in groups of TILE_M rows and dispatched via
/// ComputeContext::cpu.run.
///
/// Quantized input/output (s8/u8 → s8/u8) is supported through a fused path: the
/// input is dequantized (int → f32), the activation is applied in f32, and the
/// result is re-quantized (f32 → int). Both input and output must be quantized
/// (no float↔int mixing). Only PerTensor / PerToken granularity is supported
/// (activation quantization); the quant/dequant work is delegated to the
/// raw-intrinsic arch kernels in `x86_64/quant.hpp` / `aarch64/quant.hpp` (NOT
/// `simd_kernel/simd_quant.hpp`).

#include "nnops/ops/activation.hpp"
#include "nnops/detail/assert.hpp"
#include "simd_kernel/simd_activation.hpp"
#include "common/elementwise.hpp"
#include "common/dtype_dispatch.hpp"

#include <algorithm>
#include <cstdint>

namespace nnops::backend::cpu {

namespace {

/// Apply a floating-point activation to an M×N tile, dispatching on the type.
/// Shared by the f32/f16 paths (in/out strides may differ) and the quantized
/// path (in == out, stride == N, add_to == false).
template <typename T>
void apply_activation_flt(ActivationType type, const T* in, T* out,
                          int64_t M, int64_t N, int64_t in_stride, int64_t out_stride,
                          bool add_to, float alpha, float beta) {
    switch (type) {
    case ActivationType::Relu:        kernel::relu<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case ActivationType::LeakyRelu:   kernel::leaky_relu<T>(in, out, M, N, in_stride, out_stride, add_to, alpha); break;
    case ActivationType::Sigmoid:     kernel::sigmoid<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case ActivationType::Tanh:        kernel::tanh<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case ActivationType::Gelu:        kernel::gelu<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case ActivationType::Silu:        kernel::silu<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case ActivationType::HardSwish:   kernel::hard_swish<T>(in, out, M, N, in_stride, out_stride, add_to, beta); break;
    case ActivationType::Elu:         kernel::elu<T>(in, out, M, N, in_stride, out_stride, add_to, alpha); break;
    case ActivationType::HardSigmoid: kernel::hard_sigmoid<T>(in, out, M, N, in_stride, out_stride, add_to, alpha, beta); break;
    case ActivationType::Softplus:    kernel::softplus<T>(in, out, M, N, in_stride, out_stride, add_to, beta); break;
    case ActivationType::Selu:        kernel::selu<T>(in, out, M, N, in_stride, out_stride, add_to, alpha, beta); break;
    case ActivationType::Mish:        kernel::mish<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case ActivationType::QuickGelu:   kernel::quick_gelu<T>(in, out, M, N, in_stride, out_stride, add_to, alpha); break;
    case ActivationType::GeluExact:   kernel::gelu_exact<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case ActivationType::PRelu:       break;  // channel-mixing; see prelu_impl
    }
}

}  // anonymous namespace

/// PRelu: one slope per logical channel (dim 1). Channel-first strides a whole plane
/// per slope, so a row is one constant-slope LeakyRelu; channel-last packs C
/// inside every pixel, so the slope vector loads alongside the pixel vector.
template <typename T>
void prelu_impl(const ActivationAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx)
{
    using namespace simd;
    const auto& x = inputs[0];
    const auto& slope = inputs[1];
    const int64_t C = x.shape(1);
    const int64_t slope_n = slope.numel();
    const T* sp = slope.ptr<T>();
    const T* in = x.ptr<T>();
    T* out = output.ptr<T>();
    const int64_t in_stride  = x.row_stride_elems();
    const int64_t out_stride = output.row_stride_elems();
    const bool add_to = attrs.add_to;
    const T* type_tag = nullptr;

    const int64_t rows_total = x.total_rows();
    if (rows_total == 0) { return; }
    const int64_t rows_per_c = (C > 0) ? rows_total / C : 0;

    // Rank-2 keeps C on the innermost axis in either layout; rank ≥ 3 channel-first
    // gives each channel a contiguous run of rows.
    const bool per_row_slope = is_channel_first_spatial(x.layout()) && x.rank() >= 3;

    constexpr int64_t kTileRows = 32;
    const int64_t num_tiles = (rows_total + kTileRows - 1) / kTileRows;

    if (per_row_slope && rows_per_c > 0) {
        const int64_t cols = x.shape(x.rank() - 1);
        ctx.cpu.run(0, num_tiles, [&](int64_t ti) {
            const int64_t r0 = ti * kTileRows;
            const int64_t m = std::min(kTileRows, rows_total - r0);
            for (int64_t j = 0; j < m; ++j) {
                const int64_t r = r0 + j;
                const int64_t c = (slope_n == 1) ? 0 : r / rows_per_c;
                kernel::leaky_relu<T>(in + r * in_stride, out + r * out_stride,
                                      1, cols, in_stride, out_stride, add_to,
                                      s_load(&sp[c]));
            }
        });
        return;
    }

    // Pixel-major: channel = column % C. A channel-last row is W pixels of C
    // channels; a rank-2 row is the channel vector itself.
    constexpr int L = simd_lane_for<T>;
    const int64_t last = (C > 0) ? x.shape(x.rank() - 1) : 0;
    const int64_t cols = is_channel_last(x.layout()) ? C * last : last;
    const bool c_aligned = (C % L) == 0;

    // The slope vector for the L lanes starting at column @p c0.
    const auto slope_vec = [&](int64_t c0) {
        if (slope_n == 1) { return v_set1(type_tag, s_load(sp)); }
        if (c_aligned) { return v_load(&sp[c0 % C]); }
        // The channel repeats every C, so one vector can straddle two pixels;
        // gather its lanes.
        T lane_alpha[L];
        for (int l = 0; l < L; ++l) {
            s_store(&lane_alpha[l], s_load(&sp[(c0 + l) % C]));
        }
        return v_load(lane_alpha);
    };

    ctx.cpu.run(0, num_tiles, [&](int64_t ti) {
        const int64_t r0 = ti * kTileRows;
        const int64_t m = std::min(kTileRows, rows_total - r0);
        for (int64_t j = 0; j < m; ++j) {
            const int64_t r = r0 + j;
            const T* src = in + r * in_stride;
            T* dst = out + r * out_stride;
            int64_t c0 = 0;
            for (; c0 + L <= cols; c0 += L) {
                const auto xv = v_load(src + c0);
                const auto av = slope_vec(c0);
                const auto y = kernel::v_leaky_relu(xv, v_zero(type_tag), av);
                v_store_add(dst + c0, y, add_to);
            }
            for (; c0 < cols; ++c0) {
                const float a = (slope_n == 1) ? s_load(sp) : s_load(&sp[c0 % C]);
                const float v = s_load(&src[c0]);
                const float y = v > 0.0f ? v : a * v;
                s_store_add(&dst[c0], y, add_to);
            }
        }
    });
}

template <typename T>
void activation_impl(const ActivationAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx)
{
    tiled_float_transform<T>(inputs[0], output, ctx,
        [&](const T* in, T* out, int64_t m, int64_t n, int64_t in_stride, int64_t out_stride) {
            apply_activation_flt<T>(attrs.type, in, out, m, n, in_stride, out_stride,
                                    attrs.add_to, attrs.alpha, attrs.beta);
        });
}

Status activation_quant_impl(const ActivationAttributes& attrs,
                              TensorView& output,
                              std::span<const TensorView> inputs,
                              const ComputeContext& ctx)
{
    // add_to is only meaningful on the float (non-quantized) path.
    if (attrs.add_to) { return Status::unsupported; }

    quantized_unary_transform(inputs[0], output, ctx,
        [&](float* scratch, int h, int w) {
            apply_activation_flt<float>(attrs.type, scratch, scratch, h, w, w, w, false,
                                        attrs.alpha, attrs.beta);
        });
    return Status::ok;
}

Status activation_cpu(const ActivationAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    const bool in_is_int  = is_quantized_dtype(dtype);
    const bool out_is_int = is_quantized_dtype(output.data_type());

    if (!in_is_int && !out_is_int) {
        dispatch_f32_f16(dtype, "activation_cpu", [&](auto tag) {
            using T = typename decltype(tag)::type;
            if (attrs.type == ActivationType::PRelu) {
                prelu_impl<T>(attrs, output, inputs, ctx);
            } else {
                activation_impl<T>(attrs, output, inputs, ctx);
            }
        });
        return Status::ok;
    }

    // Quantized path: both input and output must be s8/u8 (no float↔int mixing).
    NNOPS_ASSERT(in_is_int && out_is_int);

    // A per-channel slope crossing channels would need the fused dequant to
    // carry the channel index; keep the float route as the only one.
    if (attrs.type == ActivationType::PRelu) { return Status::unsupported; }

    return activation_quant_impl(attrs, output, inputs, ctx);
}

}  // namespace nnops::backend::cpu
