/// @file unary.cpp
/// @brief SIMD-optimized CPU implementation of element-wise unary operations.
///
/// Supports both f32 and f16 via a single templated implementation.
/// Processing is tiled in groups of TILE_M rows and dispatched via
/// ComputeContext::cpu.run when available.
///
/// Design:
///   1. Rows are grouped into tiles of TILE_M (32) for SIMD-friendly blocking
///   2. Each tile calls into tiled_unary kernel library (lane=8, v_f32x8/v_f16x8)
///   3. Tile iteration is parallelized via ctx.cpu.run; falls back to
///      sequential when no parallel hook is provided
///   4. Op-type dispatch lives in the shared apply_unary_flt helper
///
/// Quantized input/output (s8/u8 → s8/u8) is supported through a fused path: the
/// input is dequantized (int → f32), the unary op is applied in f32, and the
/// result is re-quantized (f32 → int). Both input and output must be quantized
/// (no float↔int mixing). Only PerTensor / PerToken granularity is supported
/// (activation quantization); the quant/dequant work is delegated to the
/// raw-intrinsic arch kernels in `x86_64/quant.hpp` / `aarch64/quant.hpp` (NOT
/// `simd_kernel/simd_quant.hpp`). Float-meaning ops (Round/Ceil/Floor) do not
/// support quantized data types.

#include "nnops/ops/unary.hpp"
#include "nnops/detail/assert.hpp"
#include "simd_kernel/simd_unary.hpp"
#include "common/elementwise.hpp"
#include "common/dtype_dispatch.hpp"

#include <cstdint>

namespace nnops::backend::cpu {

// ============================================================
// Op-type dispatch helper
// ============================================================

namespace {

/// Apply a floating-point unary op to an M×N tile, dispatching on the op type.
/// Shared by the f32/f16 paths (in/out strides may differ) and the quantized
/// path (in == out, stride == N, add_to == false). Round/Ceil/Floor are only
/// reachable from the float path.
template <typename T>
void apply_unary_flt(UnaryType type, const T* in, T* out,
                     int64_t M, int64_t N, int64_t in_stride, int64_t out_stride,
                     bool add_to) {
    switch (type) {
    case UnaryType::Exp:   kernel::exp<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Log:   kernel::log<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Sin:   kernel::sin<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Cos:   kernel::cos<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Tan:   kernel::tan<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Tanh:  kernel::tanh<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Abs:   kernel::abs<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Neg:   kernel::neg<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Sqrt:  kernel::sqrt<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Erf:   kernel::erf<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Round: kernel::round<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Ceil:  kernel::ceil<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Floor: kernel::floor<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Recip: kernel::recip<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Sign:  kernel::sign<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    }
}

}  // anonymous namespace

// ============================================================
// Templated implementation (f32 and f16)
// ============================================================

template <typename T>
void unary_impl(const UnaryAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx)
{
    tiled_float_transform<T>(inputs[0], output, ctx,
        [&](const T* in, T* out, int64_t m, int64_t n, int64_t in_stride, int64_t out_stride) {
            apply_unary_flt<T>(attrs.type, in, out, m, n, in_stride, out_stride, attrs.add_to);
        });
}

// ============================================================
// Fused quantized unary — dequantize → unary → quantize
// ============================================================

void unary_quant_impl(const UnaryAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx)
{
    // add_to is only meaningful on the float (non-quantized) path.
    NNOPS_ASSERT(!attrs.add_to);

    quantized_unary_transform(inputs[0], output, ctx,
        [&](float* scratch, int h, int w) {
            apply_unary_flt<float>(attrs.type, scratch, scratch, h, w, w, w, false);
        });
}

// ============================================================
// Entry point with dtype dispatch
// ============================================================

void unary_cpu(const UnaryAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    const bool in_is_int  = is_quantized_dtype(dtype);
    const bool out_is_int = is_quantized_dtype(output.data_type());

    if (!in_is_int && !out_is_int) {
        dispatch_f32_f16(dtype, "unary_cpu", [&](auto tag) {
            using T = typename decltype(tag)::type;
            unary_impl<T>(attrs, output, inputs, ctx);
        });
        return;
    }

    // Quantized path: both input and output must be s8/u8 (no float↔int mixing).
    NNOPS_ASSERT(in_is_int && out_is_int);

    // Round/Ceil/Floor are float-meaning ops — no quantized-dtype support.
    if (attrs.type == UnaryType::Round ||
        attrs.type == UnaryType::Ceil  ||
        attrs.type == UnaryType::Floor) {
        NNOPS_ASSERT(!"unary_cpu: Round/Ceil/Floor do not support quantized data types");
    }

    unary_quant_impl(attrs, output, inputs, ctx);
}

}  // namespace nnops::backend::cpu
