/// @file quant_linear.cpp
/// @brief SIMD-accelerated CPU implementation of QuantizeLinear / DequantizeLinear.
///
/// Strategy:
///   - The per-row f32→s8/u8 and s8/u8→f32 arithmetic is delegated to the raw
///     intrinsic arch kernels in `x86_64/quant.hpp` (AVX2+FMA) and
///     `aarch64/quant.hpp` (NEON) — `quantization<T>` / `dequantization<T>`,
///     which round half-to-even (ONNX default).
///   - Every tensor is treated as a flat row-major `numel()` array; packed and
///     planar layouts are NOT distinguished.
///   - Per-channel scale/zp maps the `axis` dimension to the kernel's per-row
///     `M`. The dimensions before `axis` are parallelized over (the `outer`
///     groups); the dimensions after `axis` form the contiguous per-row `N`.
///   - f16 input/output is converted to/from f32 at the boundary (the arch
///     kernels operate on f32 only).

#include "nnops/ops/quant_linear.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/cpu_features.hpp"
#include "nnops/detail/half.hpp"

#if defined(NNOPS_ARCH_X86_64)
#include "x86_64/quant.hpp"
#elif defined(NNOPS_ARCH_AARCH64)
#include "aarch64/quant.hpp"
#else
#error "quant_linear: unsupported architecture"
#endif

#include <cstdint>
#include <type_traits>
#include <vector>

namespace nnops::backend::cpu {

using nnops::backend::cpu::half;
using nnops::backend::cpu::half_to_float;
using nnops::backend::cpu::float_to_half;
using nnops::backend::cpu::convert_half_to_float;
using nnops::backend::cpu::convert_float_to_half;

#if defined(NNOPS_ARCH_X86_64)
namespace quant_kernel = nnops::backend::cpu::x86_64;
#elif defined(NNOPS_ARCH_AARCH64)
namespace quant_kernel = nnops::backend::cpu::aarch64;
#endif

namespace {

/// Decompose the tensor along `axis` into outer × D × inner element counts.
/// Dimensions before `axis` collapse into `outer`; after `axis` into `inner`.
struct AxisDecomp {
    int64_t outer = 1;
    int64_t D = 1;
    int64_t inner = 1;
};

inline AxisDecomp decompose_axis(const TensorView& x, int64_t axis) {
    AxisDecomp r;
    const int64_t rank = x.rank();
    r.D = x.shape(axis);
    for (int64_t d = 0; d < axis; ++d) {
        r.outer *= x.shape(d);
    }
    for (int64_t d = axis + 1; d < rank; ++d) {
        r.inner *= x.shape(d);
    }
    return r;
}

}  // anonymous namespace

// ============================================================
// QuantizeLinear: float → integer
// ============================================================

template <typename T>  // T = float or half (input type)
void quantize_linear_impl(const QuantLinearAttributes& attrs,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx)
{
    const auto& X     = inputs[0];
    const auto& scale = inputs[1];
    const auto& zp    = inputs[2];

    const int64_t rank = X.rank();
    int64_t axis = attrs.axis;
    if (axis < 0) {
        axis += rank;
    }

    const AxisDecomp dec = decompose_axis(X, axis);
    const int64_t D = dec.D;
    const bool is_per_channel = (scale.numel() > 1);
    const DataType out_dtype = output.data_type();
    const int64_t numel = X.numel();

    // Pre-load scale/zp as float (per-tensor: single value; per-channel: D values).
    std::vector<float> s_f32(static_cast<size_t>(D));
    std::vector<float> z_f32(static_cast<size_t>(D));
    for (int64_t k = 0; k < D; ++k) {
        s_f32[static_cast<size_t>(k)] = scale.ptr<float>()[is_per_channel ? k : 0];
        int32_t z = (zp.data_type() == DataType::s8)
            ? static_cast<int32_t>(zp.ptr<int8_t>()[is_per_channel ? k : 0])
            : static_cast<int32_t>(zp.ptr<uint8_t>()[is_per_channel ? k : 0]);
        z_f32[static_cast<size_t>(k)] = static_cast<float>(z);
    }

    // f16 input → f32 temp buffer.
    std::vector<float> f32_buf;
    const float* src_ptr;
    if constexpr (std::is_same_v<T, half>) {
        f32_buf.resize(static_cast<size_t>(numel));
        convert_half_to_float(f32_buf.data(), X.ptr<half>(), static_cast<int>(numel));
        src_ptr = f32_buf.data();
    } else {
        src_ptr = X.ptr<float>();
    }

    // Per-group kernel driver, templated on the output integer type U.
    const auto run = [&](auto u_tag) {
        using U = typename decltype(u_tag)::type;
        U* dst = static_cast<U*>(output.ptr<void>());

        if (!is_per_channel) {
            quant_kernel::quantization<U>(1, static_cast<int>(numel),
                                          dst, static_cast<int>(numel),
                                          src_ptr, static_cast<int>(numel),
                                          s_f32.data(), z_f32.data());
            return;
        }

        const auto process = [&](int64_t o) {
            quant_kernel::quantization<U>(static_cast<int>(D), static_cast<int>(dec.inner),
                                          dst + o * D * dec.inner, static_cast<int>(dec.inner),
                                          src_ptr + o * D * dec.inner, static_cast<int>(dec.inner),
                                          s_f32.data(), z_f32.data());
        };
        if (ctx.cpu_parallel_for) {
            ctx.cpu_parallel_for(0, dec.outer, process);
        } else {
            for (int64_t o = 0; o < dec.outer; ++o) {
                process(o);
            }
        }
    };

    if (out_dtype == DataType::s8) {
        run(std::type_identity<int8_t>{});
    } else {
        run(std::type_identity<uint8_t>{});
    }
}

// ============================================================
// DequantizeLinear: integer → float
// ============================================================

template <typename T>  // T = float or half (output type)
void dequantize_linear_impl(const QuantLinearAttributes& attrs,
                            TensorView& output,
                            std::span<const TensorView> inputs,
                            const ComputeContext& ctx)
{
    const auto& X     = inputs[0];  // s8 or u8
    const auto& scale = inputs[1];  // f32
    const auto& zp    = inputs[2];  // s8 or u8

    const int64_t rank = X.rank();
    int64_t axis = attrs.axis;
    if (axis < 0) {
        axis += rank;
    }

    const AxisDecomp dec = decompose_axis(X, axis);
    const int64_t D = dec.D;
    const bool is_per_channel = (scale.numel() > 1);
    const DataType in_dtype = X.data_type();
    const int64_t numel = X.numel();

    std::vector<float> s_f32(static_cast<size_t>(D));
    std::vector<float> z_f32(static_cast<size_t>(D));
    for (int64_t k = 0; k < D; ++k) {
        s_f32[static_cast<size_t>(k)] = scale.ptr<float>()[is_per_channel ? k : 0];
        int32_t z = (zp.data_type() == DataType::s8)
            ? static_cast<int32_t>(zp.ptr<int8_t>()[is_per_channel ? k : 0])
            : static_cast<int32_t>(zp.ptr<uint8_t>()[is_per_channel ? k : 0]);
        z_f32[static_cast<size_t>(k)] = static_cast<float>(z);
    }

    // f16 output → dequant into a f32 temp buffer, then convert.
    std::vector<float> f32_buf;
    float* dst_f32;
    if constexpr (std::is_same_v<T, half>) {
        f32_buf.resize(static_cast<size_t>(numel));
        dst_f32 = f32_buf.data();
    } else {
        dst_f32 = output.ptr<float>();
    }

    // Per-group kernel driver, templated on the input integer type U.
    const auto run = [&](auto u_tag) {
        using U = typename decltype(u_tag)::type;
        const U* src = static_cast<const U*>(X.ptr<void>());

        if (!is_per_channel) {
            quant_kernel::dequantization<U>(1, static_cast<int>(numel),
                                            dst_f32, static_cast<int>(numel),
                                            src, static_cast<int>(numel),
                                            s_f32.data(), z_f32.data());
            return;
        }

        const auto process = [&](int64_t o) {
            quant_kernel::dequantization<U>(static_cast<int>(D), static_cast<int>(dec.inner),
                                            dst_f32 + o * D * dec.inner, static_cast<int>(dec.inner),
                                            src + o * D * dec.inner, static_cast<int>(dec.inner),
                                            s_f32.data(), z_f32.data());
        };
        if (ctx.cpu_parallel_for) {
            ctx.cpu_parallel_for(0, dec.outer, process);
        } else {
            for (int64_t o = 0; o < dec.outer; ++o) {
                process(o);
            }
        }
    };

    if (in_dtype == DataType::s8) {
        run(std::type_identity<int8_t>{});
    } else {
        run(std::type_identity<uint8_t>{});
    }

    if constexpr (std::is_same_v<T, half>) {
        convert_float_to_half(output.ptr<half>(), f32_buf.data(), static_cast<int>(numel));
    }
}

// ============================================================
// Entry points
// ============================================================

void quantize_linear_cpu(const QuantLinearAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         void* /*workspace*/)
{
    switch (inputs[0].data_type()) {
    case DataType::f32:
        quantize_linear_impl<float>(attrs, output, inputs, ctx);
        break;
    case DataType::f16:
        quantize_linear_impl<half>(attrs, output, inputs, ctx);
        break;
    default:
        NNOPS_ASSERT(!"quantize_linear_cpu: unsupported input dtype");
    }
}

void dequantize_linear_cpu(const QuantLinearAttributes& attrs,
                           TensorView& output,
                           std::span<const TensorView> inputs,
                           const ComputeContext& ctx,
                           void* /*workspace*/)
{
    switch (output.data_type()) {
    case DataType::f32:
        dequantize_linear_impl<float>(attrs, output, inputs, ctx);
        break;
    case DataType::f16:
        dequantize_linear_impl<half>(attrs, output, inputs, ctx);
        break;
    default:
        NNOPS_ASSERT(!"dequantize_linear_cpu: unsupported output dtype");
    }
}

}  // namespace nnops::backend::cpu
