/// @file embed.cpp
/// @brief SIMD-optimized CPU implementation of embedding table lookup.
///
/// Supports direct lookup (f32/f16 weight) and int8 quantized weight with
/// per-row dequantization: output[j] = (weight[idx,j] - zp[idx]) * scale[idx].
///
/// The core operation is copying (f32/f16) or dequantizing (int8) embedding
/// rows, parallelized over indices via cpu_parallel_for.
///
/// For int8/uint8 dequantization, the per-row arithmetic is delegated to the
/// raw arch kernels in `x86_64/quant.hpp` / `aarch64/quant.hpp`
/// (quant_kernel::dequantization<T>), which output f32. For f16 output the
/// gathered rows are dequantized into a f32 workspace scratch, then converted
/// f32→f16.

#include "nnops/ops/embed.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd/cpu_features.hpp"

#if defined(NNOPS_ARCH_X86_64)
#include "x86_64/quant.hpp"
#elif defined(NNOPS_ARCH_AARCH64)
#include "aarch64/quant.hpp"
#else
#error "embed: unsupported architecture for quantization kernels"
#endif

#include <algorithm>
#include <cstring>
#include <type_traits>

namespace nnops::backend::cpu {

#if defined(NNOPS_ARCH_X86_64)
namespace quant_kernel = nnops::backend::cpu::x86_64;
#elif defined(NNOPS_ARCH_AARCH64)
namespace quant_kernel = nnops::backend::cpu::aarch64;
#endif

namespace {

// ============================================================
// Direct lookup (f32/f16 weight)
// ============================================================

template <typename T>
void embed_direct_impl(const EmbedAttributes& /*attrs*/,
                        TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx)
{
    const auto& weight  = inputs[0];
    const auto& indices = inputs[1];

    const int64_t V   = weight.shape(0);
    const int64_t dim = weight.shape(1);
    const int64_t num_indices = indices.numel();

    const T* weight_ptr = weight.ptr<T>();
    const int64_t w_row_stride = (weight.rank() == 2)
        ? dim
        : weight.row_stride_elems();

    T* out_ptr = output.ptr<T>();
    const int64_t out_row_stride = output.row_stride_elems();
    const bool is_i64 = (indices.data_type() == DataType::s64);
    const size_t row_bytes = static_cast<size_t>(dim) * sizeof(T);

    auto body = [&](int64_t n) {
        int64_t idx;
        if (is_i64) {
            idx = static_cast<const int64_t*>(indices.ptr<void>())[n];
        } else {
            idx = static_cast<const int32_t*>(indices.ptr<void>())[n];
        }

        // Clamp to valid range
        if (idx < 0) {
            idx = 0;
        }
        if (idx >= V) {
            idx = V - 1;
        }

        // Copy weight[idx, :] to output[n, :]
        std::memcpy(out_ptr + n * out_row_stride,
                    weight_ptr + idx * w_row_stride,
                    row_bytes);
    };

    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, num_indices, body);
    }
    else {
        for (int64_t n = 0; n < num_indices; ++n) {
            body(n);
        }
    }
}

// ============================================================
// Int8/uint8 lookup with per-row dequantization → f32 or f16
// ============================================================
//
// The per-row dequant arithmetic is delegated to the raw arch kernels in
// `x86_64/quant.hpp` (AVX2+FMA) / `aarch64/quant.hpp` (NEON):
// `quant_kernel::dequantization<T>`, which outputs f32. For f16 output the
// gathered rows are first dequantized into a f32 workspace scratch, then
// converted f32→f16.

template <typename T>  // T = weight storage type (int8_t / uint8_t)
void embed_int8_dequant_impl(const EmbedAttributes& /*attrs*/,
                              TensorView& output,
                              std::span<const TensorView> inputs,
                              const ComputeContext& ctx,
                              void* workspace)
{
    const auto& weight  = inputs[0];
    const auto& indices = inputs[1];

    const auto& qp = weight.quant_params();
    // For the 2D embed weight [vocab_size, dim], PerChannel quantizes along
    // axis 0 (the output channel = vocab entry), i.e. per-row — the same as
    // PerToken. Both use one (scale, zero_point) per weight row.
    const bool per_row = (qp.granularity == QuantGranularity::PerToken)
                      || (qp.granularity == QuantGranularity::PerChannel);
    const bool has_zp   = (qp.zero_point_data != nullptr);

    const float per_tensor_scale = qp.scale;
    const float per_tensor_zp    = static_cast<float>(qp.zero_point);

    const int64_t V = weight.shape(0);
    const int64_t dim = weight.shape(1);
    const int64_t num_indices = indices.numel();

    const T* weight_ptr = weight.ptr<T>();
    const int64_t w_row_elems = dim;  // int8/uint8: 1 byte per element

    const bool out_f16 = (output.data_type() == DataType::f16);
    const bool is_i64 = (indices.data_type() == DataType::s64);
    const int64_t out_row_stride = output.row_stride_elems();

    const int dim_i = static_cast<int>(dim);

    // f16 output is dequantized into this f32 scratch (caller-provided workspace).
    float* scratch = out_f16 ? static_cast<float*>(workspace) : nullptr;
    NNOPS_ASSERT(!out_f16 || scratch != nullptr);

    const auto resolve_idx = [&](int64_t n) -> int64_t {
        int64_t idx = is_i64
            ? static_cast<const int64_t*>(indices.ptr<void>())[n]
            : static_cast<const int32_t*>(indices.ptr<void>())[n];
        if (idx < 0) { idx = 0; }
        if (idx >= V) { idx = V - 1; }
        return idx;
    };

    const auto body = [&](int64_t n) {
        const int64_t idx = resolve_idx(n);
        const T* w_row = weight_ptr + idx * w_row_elems;

        const float s_val = per_row ? qp.scale_data[idx] : per_tensor_scale;
        const float zp_val = has_zp
            ? static_cast<float>(per_row ? qp.zero_point_data[idx] : qp.zero_point)
            : 0.0f;

        // Dequantize one (M=1) row into f32 — directly into the output, or into
        // the f16 scratch (single-element per-row scale/zero_point arrays).
        float* dst = out_f16
            ? scratch + n * dim
            : output.ptr<float>() + n * out_row_stride;
        quant_kernel::dequantization<T>(1, dim_i, dst, dim_i,
                                        w_row, dim_i, &s_val, &zp_val);
    };

    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, num_indices, body);
    }
    else {
        for (int64_t n = 0; n < num_indices; ++n) {
            body(n);
        }
    }

    if (out_f16) {
        // Convert the f32 scratch → f16 output (parallel over rows).
        half* out_ptr = output.ptr<half>();
        const auto convert = [&](int64_t n) {
            convert_float_to_half(out_ptr + n * out_row_stride,
                                  scratch + n * dim, dim_i);
        };
        if (ctx.cpu_parallel_for) {
            ctx.cpu_parallel_for(0, num_indices, convert);
        }
        else {
            for (int64_t n = 0; n < num_indices; ++n) {
                convert(n);
            }
        }
    }
}

}  // anonymous namespace

// ============================================================
// Entry point with dtype dispatch
// ============================================================

void embed_cpu(const EmbedAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* workspace)
{
    const auto w_dtype = inputs[0].data_type();
    const auto o_dtype = output.data_type();

    if (w_dtype == DataType::f32) {
        embed_direct_impl<float>(attrs, output, inputs, ctx);
        return;
    }
    if (w_dtype == DataType::f16) {
        embed_direct_impl<half>(attrs, output, inputs, ctx);
        return;
    }

    // Int8/uint8 weight: dequantize to f32 or f16 output.
    NNOPS_ASSERT(o_dtype == DataType::f32 || o_dtype == DataType::f16);
    if (w_dtype == DataType::s8) {
        embed_int8_dequant_impl<int8_t>(attrs, output, inputs, ctx, workspace);
    } else if (w_dtype == DataType::u8) {
        embed_int8_dequant_impl<uint8_t>(attrs, output, inputs, ctx, workspace);
    } else {
        NNOPS_ASSERT(!"embed_cpu: unsupported weight dtype");
    }
}

}  // namespace nnops::backend::cpu
