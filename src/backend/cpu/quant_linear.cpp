/// @file quant_linear.cpp
/// @brief SIMD-accelerated CPU implementation of QuantizeLinear / DequantizeLinear.
///
/// The per-row f32->s8/u8 and s8/u8->f32 arithmetic is delegated to the raw
/// intrinsic arch kernels in `x86_64/quant.hpp` (AVX2+FMA) and
/// `aarch64/quant.hpp` (NEON) — `quantization<T>` / `dequantization<T>`;
/// quantization rounds half-to-even (ONNX default).
///
/// Every tensor is treated as a flat row-major `numel()` array; packed and
/// No layout tag is interpreted. Granularity maps onto the kernel's
/// per-row M x N shape:
///   PerTensor  -> M=1,           N=numel
///   PerToken   -> M=numel/last,  N=last (one scale per row)
///   PerChannel -> M=shape[axis], N=inner, with the dims before `axis` as the
///                 `outer` groups parallelized over.
///
/// f16 input/output is converted to/from f32 at the boundary, so this file
/// always calls the f32 overloads.

#include "nnops/ops/quant_linear.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/cpu_features.hpp"
#include "nnops/detail/half.hpp"
#include "common/dtype_dispatch.hpp"
#include "common/memory_pool.hpp"

#if defined(NNOPS_ARCH_X86_64)
#include "x86_64/quant.hpp"
#elif defined(NNOPS_ARCH_AARCH64)
#include "aarch64/quant.hpp"
#else
#error "quant_linear: unsupported architecture"
#endif

#include <algorithm>
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

struct AxisDecomp {
    int64_t outer = 1;
    int64_t D = 1;
    int64_t inner = 1;
};

inline AxisDecomp decompose_axis(const TensorView& x, int64_t axis)
{
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

} // anonymous namespace

template <typename T> // T = float or half (input type)
void quantize_linear_impl(const QuantLinearAttributes& attrs, TensorView& output,
                          std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& X = inputs[0];
    const auto& scale = inputs[1];
    const auto& zp = inputs[2];

    const int64_t rank = X.rank();
    int64_t axis = attrs.axis;
    if (axis < 0) {
        axis += rank;
    }

    const QuantGranularity g = resolve_quant_granularity(scale, attrs.axis);
    const AxisDecomp dec = decompose_axis(X, axis);
    const DataType out_dtype = output.data_type();
    const int64_t numel = X.numel();

    int64_t M;
    int64_t N;
    switch (g) {
    case QuantGranularity::PerTensor:
        M = 1;
        N = numel;
        break;
    case QuantGranularity::PerToken: {
        const int64_t last_dim = X.shape(rank - 1);
        M = numel / last_dim;
        N = last_dim;
        break;
    }
    case QuantGranularity::PerChannel:
        M = dec.D;
        N = dec.inner;
        break;
    default:
        NNOPS_ASSERT(!"quantize_linear: unsupported granularity");
        M = 1;
        N = numel;
    }

    NNOPS_ASSERT(scale.numel() == M);
    PoolArray<float> s_f32(static_cast<size_t>(M));
    PoolArray<float> z_f32(static_cast<size_t>(M));
    for (int64_t k = 0; k < M; ++k) {
        s_f32[static_cast<size_t>(k)] = scale.ptr<float>()[k];
        int32_t z = (zp.data_type() == DataType::s8) ? static_cast<int32_t>(zp.ptr<int8_t>()[k]) :
                                                       static_cast<int32_t>(zp.ptr<uint8_t>()[k]);
        z_f32[static_cast<size_t>(k)] = static_cast<float>(z);
    }

    PoolArray<float> f32_buf;
    const float* src_ptr;
    if constexpr (std::is_same_v<T, half>) {
        f32_buf.reset(static_cast<size_t>(numel));
        convert_half_to_float(f32_buf.data(), X.ptr<half>(), static_cast<int>(numel));
        src_ptr = f32_buf.data();
    } else {
        src_ptr = X.ptr<float>();
    }

    // The arch kernels are single-call; the pool splits the work by elements
    // (per-tensor: one long row), by rows (per-token) or by outer group
    // (per-channel), so a boundary QDQ pass is no longer one core's job.
    constexpr int64_t kElemsPerTask = int64_t{1} << 16;

    const auto run = [&](auto u_tag) {
        using U = typename decltype(u_tag)::type;
        U* dst = static_cast<U*>(output.ptr<void>());

        if (g == QuantGranularity::PerTensor) {
            const int64_t chunks = (numel + kElemsPerTask - 1) / kElemsPerTask;
            ctx.cpu.run(0, chunks, [&](int64_t c) {
                const int64_t off = c * kElemsPerTask;
                const int n = static_cast<int>(std::min<int64_t>(kElemsPerTask, numel - off));
                quant_kernel::quantization<U>(1, n, dst + off, n, src_ptr + off, n, s_f32.data(),
                                              z_f32.data());
            });
            return;
        }

        if (g == QuantGranularity::PerToken) {
            const int64_t rows_per_task =
                std::max<int64_t>(1, std::min<int64_t>(64, kElemsPerTask / std::max<int64_t>(1, N)));
            const int64_t tasks = (M + rows_per_task - 1) / rows_per_task;
            ctx.cpu.run(0, tasks, [&](int64_t t) {
                const int64_t r0 = t * rows_per_task;
                const int64_t rn = std::min(rows_per_task, M - r0);
                quant_kernel::quantization<U>(static_cast<int>(rn), static_cast<int>(N),
                                              dst + r0 * N, static_cast<int>(N), src_ptr + r0 * N,
                                              static_cast<int>(N), s_f32.data() + r0,
                                              z_f32.data() + r0);
            });
            return;
        }

        const auto process = [&](int64_t o) {
            quant_kernel::quantization<U>(static_cast<int>(dec.D), static_cast<int>(dec.inner),
                                          dst + o * dec.D * dec.inner, static_cast<int>(dec.inner),
                                          src_ptr + o * dec.D * dec.inner,
                                          static_cast<int>(dec.inner), s_f32.data(), z_f32.data());
        };
        ctx.cpu.run(0, dec.outer, process);
    };

    if (out_dtype == DataType::s8) {
        run(std::type_identity<int8_t>{});
    } else {
        run(std::type_identity<uint8_t>{});
    }
}

template <typename T> // T = float or half (output type)
void dequantize_linear_impl(const QuantLinearAttributes& attrs, TensorView& output,
                            std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& X = inputs[0];     // s8 or u8
    const auto& scale = inputs[1]; // f32
    const auto& zp = inputs[2];    // s8 or u8

    const int64_t rank = X.rank();
    int64_t axis = attrs.axis;
    if (axis < 0) {
        axis += rank;
    }

    const QuantGranularity g = resolve_quant_granularity(scale, attrs.axis);
    const AxisDecomp dec = decompose_axis(X, axis);
    const DataType in_dtype = X.data_type();
    const int64_t numel = X.numel();

    int64_t M;
    int64_t N;
    switch (g) {
    case QuantGranularity::PerTensor:
        M = 1;
        N = numel;
        break;
    case QuantGranularity::PerToken: {
        const int64_t last_dim = X.shape(rank - 1);
        M = numel / last_dim;
        N = last_dim;
        break;
    }
    case QuantGranularity::PerChannel:
        M = dec.D;
        N = dec.inner;
        break;
    default:
        NNOPS_ASSERT(!"dequantize_linear: unsupported granularity");
        M = 1;
        N = numel;
    }

    NNOPS_ASSERT(scale.numel() == M);
    PoolArray<float> s_f32(static_cast<size_t>(M));
    PoolArray<float> z_f32(static_cast<size_t>(M));
    for (int64_t k = 0; k < M; ++k) {
        s_f32[static_cast<size_t>(k)] = scale.ptr<float>()[k];
        int32_t z = (zp.data_type() == DataType::s8) ? static_cast<int32_t>(zp.ptr<int8_t>()[k]) :
                                                       static_cast<int32_t>(zp.ptr<uint8_t>()[k]);
        z_f32[static_cast<size_t>(k)] = static_cast<float>(z);
    }

    PoolArray<float> f32_buf;
    float* dst_f32;
    if constexpr (std::is_same_v<T, half>) {
        f32_buf.reset(static_cast<size_t>(numel));
        dst_f32 = f32_buf.data();
    } else {
        dst_f32 = output.ptr<float>();
    }

    constexpr int64_t kElemsPerTask = int64_t{1} << 16;

    const auto run = [&](auto u_tag) {
        using U = typename decltype(u_tag)::type;
        const U* src = static_cast<const U*>(X.ptr<void>());

        if (g == QuantGranularity::PerTensor) {
            const int64_t chunks = (numel + kElemsPerTask - 1) / kElemsPerTask;
            ctx.cpu.run(0, chunks, [&](int64_t c) {
                const int64_t off = c * kElemsPerTask;
                const int n = static_cast<int>(std::min<int64_t>(kElemsPerTask, numel - off));
                quant_kernel::dequantization<U>(1, n, dst_f32 + off, n, src + off, n,
                                                s_f32.data(), z_f32.data());
            });
            return;
        }

        if (g == QuantGranularity::PerToken) {
            const int64_t rows_per_task =
                std::max<int64_t>(1, std::min<int64_t>(64, kElemsPerTask / std::max<int64_t>(1, N)));
            const int64_t tasks = (M + rows_per_task - 1) / rows_per_task;
            ctx.cpu.run(0, tasks, [&](int64_t t) {
                const int64_t r0 = t * rows_per_task;
                const int64_t rn = std::min(rows_per_task, M - r0);
                quant_kernel::dequantization<U>(static_cast<int>(rn), static_cast<int>(N),
                                                dst_f32 + r0 * N, static_cast<int>(N),
                                                src + r0 * N, static_cast<int>(N),
                                                s_f32.data() + r0, z_f32.data() + r0);
            });
            return;
        }

        const auto process = [&](int64_t o) {
            quant_kernel::dequantization<U>(
                static_cast<int>(dec.D), static_cast<int>(dec.inner),
                dst_f32 + o * dec.D * dec.inner, static_cast<int>(dec.inner),
                src + o * dec.D * dec.inner, static_cast<int>(dec.inner), s_f32.data(),
                z_f32.data());
        };
        ctx.cpu.run(0, dec.outer, process);
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

Status quantize_linear_cpu(const QuantLinearAttributes& attrs, TensorView& output,
                           std::span<const TensorView> inputs, const ComputeContext& ctx,
                           void* /*workspace*/)
{
    if (!dispatch_f32_f16(inputs[0].data_type(), "quantize_linear_cpu", [&](auto tag) {
            using T = typename decltype(tag)::type;
            quantize_linear_impl<T>(attrs, output, inputs, ctx);
        })) {
        return Status::unsupported;
    }
    return Status::ok;
}

Status dequantize_linear_cpu(const QuantLinearAttributes& attrs, TensorView& output,
                             std::span<const TensorView> inputs, const ComputeContext& ctx,
                             void* /*workspace*/)
{
    if (!dispatch_f32_f16(output.data_type(), "dequantize_linear_cpu", [&](auto tag) {
            using T = typename decltype(tag)::type;
            dequantize_linear_impl<T>(attrs, output, inputs, ctx);
        })) {
        return Status::unsupported;
    }
    return Status::ok;
}

} // namespace nnops::backend::cpu
