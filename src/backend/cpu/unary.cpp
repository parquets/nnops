/// @file unary.cpp
/// @brief SIMD-optimized CPU implementation of element-wise unary operations.
///
/// One templated implementation covers f32 and f16. The pitch-aware row/tile
/// loop and the ComputeContext::cpu.run dispatch live in common/elementwise.hpp,
/// so this file is just the op switch and the dtype/quant routing.
///
/// Quantized input/output (s8/u8 → s8/u8) is a fused path: dequantize to f32,
/// apply the op, requantize. Input and output must both be quantized (no
/// float↔int mixing) and only PerTensor / PerToken granularity is supported;
/// the dequant/requant work goes to the raw-intrinsic arch kernels in
/// `x86_64/quant.hpp` / `aarch64/quant.hpp` (NOT `simd_kernel/simd_quant.hpp`).
/// Float-meaning ops (Round/Ceil/Floor) reject quantized data types.

#include "nnops/ops/unary.hpp"
#include "nnops/detail/assert.hpp"
#include "simd_kernel/simd_unary.hpp"
#include "common/elementwise.hpp"
#include "common/dtype_dispatch.hpp"

#include <cstdint>

namespace nnops::backend::cpu {

namespace {

/// Dispatch a float unary op over an M×N tile; the quantized path reuses it with in == out and add_to == false.
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

Status unary_quant_impl(const UnaryAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx)
{
    // add_to is only meaningful on the float (non-quantized) path.
    if (attrs.add_to) { return Status::unsupported; }

    quantized_unary_transform(inputs[0], output, ctx,
        [&](float* scratch, int h, int w) {
            apply_unary_flt<float>(attrs.type, scratch, scratch, h, w, w, w, false);
        });
    return Status::ok;
}

Status unary_cpu(const UnaryAttributes& attrs,
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
        return Status::ok;
    }

    // Quantized path: both input and output must be s8/u8 (no float↔int mixing).
    NNOPS_ASSERT(in_is_int && out_is_int);

    // Round/Ceil/Floor have no quantized meaning; refuse rather than fall through.
    if (attrs.type == UnaryType::Round || attrs.type == UnaryType::Ceil ||
        attrs.type == UnaryType::Floor) {
        return Status::unsupported;
    }

    return unary_quant_impl(attrs, output, inputs, ctx);
}

}  // namespace nnops::backend::cpu
