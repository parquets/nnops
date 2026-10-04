/// @file eltwise.cpp
/// @brief SIMD-optimized CPU implementation of element-wise binary operations.
///
/// Supports both f32 and f16 via a single templated implementation.
/// Processing is tiled in groups of TILE_M rows and dispatched via
/// ComputeContext::cpu.run.
///
/// Design:
///   1. Rows are grouped into tiles of TILE_M (32) for SIMD-friendly blocking
///   2. Each tile calls into tiled_eltwise kernel library (lane=8, v_f32x8/v_f16x8)
///   3. Tile iteration is parallelized via ctx.cpu.run; falls back to
///      sequential when no parallel hook is provided
///   4. Op-type dispatch lives in the shared apply_eltwise_flt helper
///
/// Quantized input/output (s8/u8 → s8/u8) is supported through a fused path: both
/// inputs are dequantized (int → f32), the binary op is applied in f32, and the
/// result is re-quantized (f32 → int). Both inputs and the output must be
/// quantized (no float↔int mixing). Only PerTensor / PerToken granularity is
/// supported; the quant/dequant work is delegated to the raw-intrinsic arch
/// kernels in `x86_64/quant.hpp` / `aarch64/quant.hpp` (NOT `simd_quant.hpp`).

#include "nnops/ops/eltwise.hpp"
#include "nnops/detail/assert.hpp"
#include "simd_kernel/simd_eltwise.hpp"
#include "common/elementwise.hpp"
#include "common/dtype_dispatch.hpp"

#include <cstdint>

namespace nnops::backend::cpu {

namespace {

/// Apply a floating-point binary op to an M×N tile, dispatching on the op type.
/// Shared by the f32/f16 paths (strides may differ) and the quantized path
/// (a/b/out are f32 scratch, out may alias a, add_to == false).
template <typename T>
void apply_eltwise_flt(EltwiseType type, const T* a, const T* b, T* out,
                       int64_t M, int64_t N, int64_t a_stride, int64_t b_stride,
                       int64_t out_stride, bool add_to) {
    switch (type) {
    case EltwiseType::Add: kernel::add<T>(a, b, out, M, N, a_stride, b_stride, out_stride, add_to); break;
    case EltwiseType::Sub: kernel::sub<T>(a, b, out, M, N, a_stride, b_stride, out_stride, add_to); break;
    case EltwiseType::Mul: kernel::mul<T>(a, b, out, M, N, a_stride, b_stride, out_stride, add_to); break;
    case EltwiseType::Div: kernel::div<T>(a, b, out, M, N, a_stride, b_stride, out_stride, add_to); break;
    case EltwiseType::Min: kernel::min<T>(a, b, out, M, N, a_stride, b_stride, out_stride, add_to); break;
    case EltwiseType::Max: kernel::max<T>(a, b, out, M, N, a_stride, b_stride, out_stride, add_to); break;
    case EltwiseType::Pow: kernel::pow<T>(a, b, out, M, N, a_stride, b_stride, out_stride, add_to); break;
    }
}

}  // anonymous namespace

// ============================================================
// Templated implementation (f32 and f16)
// ============================================================

template <typename T>
void eltwise_impl(const EltwiseAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx)
{
    tiled_float_transform2<T>(inputs[0], inputs[1], output, ctx,
        [&](const T* a, const T* b, T* out, int64_t m, int64_t n,
            int64_t a_stride, int64_t b_stride, int64_t out_stride) {
            apply_eltwise_flt<T>(attrs.type, a, b, out, m, n, a_stride, b_stride,
                                 out_stride, attrs.add_to);
        });
}

// ============================================================
// Fused quantized eltwise — dequantize → op → quantize
// ============================================================

void eltwise_quant_impl(const EltwiseAttributes& attrs,
                        TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx)
{
    // add_to is only meaningful on the float (non-quantized) path.
    NNOPS_ASSERT(!attrs.add_to);

    quantized_binary_transform(inputs[0], inputs[1], output, ctx,
        [&](float* a, float* b, int h, int w) {
            apply_eltwise_flt<float>(attrs.type, a, b, a, h, w, w, w, w, false);
        });
}

// ============================================================
// Entry point with dtype dispatch
// ============================================================

void eltwise_cpu(const EltwiseAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx,
                   void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    const bool in_is_int  = is_quantized_dtype(dtype);
    const bool out_is_int = is_quantized_dtype(output.data_type());

    if (!in_is_int && !out_is_int) {
        dispatch_f32_f16(dtype, "eltwise_cpu", [&](auto tag) {
            using T = typename decltype(tag)::type;
            eltwise_impl<T>(attrs, output, inputs, ctx);
        });
        return;
    }

    // Quantized path: both inputs and the output must be s8/u8 (no float↔int mixing).
    NNOPS_ASSERT(in_is_int && out_is_int);

    eltwise_quant_impl(attrs, output, inputs, ctx);
}

}  // namespace nnops::backend::cpu
