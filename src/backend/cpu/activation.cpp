/// @file activation.cpp
/// @brief SIMD-optimized CPU implementation of activation functions.
///
/// All 8 activation types are vectorized with the nnops SIMD abstraction layer.
/// Supports both f32 (v_f32x8) and f16 (v_f16x8) via a single templated implementation.
/// Processing is tiled in groups of 32 rows and dispatched via
/// ComputeContext::cpu.parallel_for when available.
///
/// Design:
///   1. Rows are grouped into tiles of TILE_M (32) for SIMD-friendly blocking
///   2. Each tile calls into tiled_activation kernel library
///   3. Tile iteration is parallelized via ctx.cpu.run
///   4. Op-type dispatch lives in the shared apply_activation_flt helper
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
#include "nnops/detail/half.hpp"
#include "simd_kernel/simd_activation.hpp"
#include "simd_kernel/activation_kernels.hpp"
#include "common/elementwise.hpp"
#include "common/dtype_dispatch.hpp"

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
    case ActivationType::Relu:      kernel::relu<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case ActivationType::LeakyRelu: kernel::leaky_relu<T>(in, out, M, N, in_stride, out_stride, add_to, alpha); break;
    case ActivationType::Sigmoid:   kernel::sigmoid<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case ActivationType::Tanh:      kernel::tanh<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case ActivationType::Gelu:      kernel::gelu<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case ActivationType::Silu:      kernel::silu<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case ActivationType::HardSwish: kernel::hard_swish<T>(in, out, M, N, in_stride, out_stride, add_to, beta); break;
    case ActivationType::Elu:       kernel::elu<T>(in, out, M, N, in_stride, out_stride, add_to, alpha); break;
    }
}

}  // anonymous namespace

// ============================================================
// Templated implementation (f32 and f16)
// ============================================================

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

// ============================================================
// Fused quantized activation — dequantize → activation → quantize
// ============================================================

void activation_quant_impl(const ActivationAttributes& attrs,
                           TensorView& output,
                           std::span<const TensorView> inputs,
                           const ComputeContext& ctx)
{
    // add_to is only meaningful on the float (non-quantized) path.
    NNOPS_ASSERT(!attrs.add_to);

    quantized_unary_transform(inputs[0], output, ctx,
        [&](float* scratch, int h, int w) {
            apply_activation_flt<float>(attrs.type, scratch, scratch, h, w, w, w, false,
                                        attrs.alpha, attrs.beta);
        });
}

// ============================================================
// Main entry point with dtype dispatch
// ============================================================

void activation_cpu(const ActivationAttributes& attrs,
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
            activation_impl<T>(attrs, output, inputs, ctx);
        });
        return;
    }

    // Quantized path: both input and output must be s8/u8 (no float↔int mixing).
    NNOPS_ASSERT(in_is_int && out_is_int);

    activation_quant_impl(attrs, output, inputs, ctx);
}

}  // namespace nnops::backend::cpu
