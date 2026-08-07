/// @file embed.cpp
/// @brief SIMD-optimized CPU implementation of embedding table lookup.
///
/// Supports direct lookup (f32/f16 weight) and int8 quantized weight with
/// per-row dequantization: output[j] = (weight[idx,j] - zp[idx]) * scale[idx].
///
/// The core operation is copying (f32/f16) or dequantizing (int8) embedding
/// rows, parallelized over indices via cpu_parallel_for.
///
/// For int8 dequantization, SIMD vector types are fully encapsulated in
/// simd_quant.hpp helpers (dequant_i8_store / dequant_i8_scalar). The
/// embed kernel itself is type-agnostic — pointer-type overloads handle
/// the f32 vs f16 dispatch at compile time.

#include "nnops/ops/embed.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"
#include "simd_kernel/simd_quant.hpp"

#include <algorithm>
#include <cstring>
#include <type_traits>

namespace nnops::backend::cpu {

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
    const bool is_i64 = (indices.data_type() == DataType::i64);
    const size_t row_bytes = static_cast<size_t>(dim) * sizeof(T);

    auto body = [&](int64_t n) {
        int64_t idx;
        if (is_i64) {
            idx = static_cast<const int64_t*>(indices.ptr<void>())[n];
        } else {
            idx = static_cast<const int32_t*>(indices.ptr<void>())[n];
        }

        // Clamp to valid range
        if (idx < 0) idx = 0;
        if (idx >= V) idx = V - 1;

        // Copy weight[idx, :] to output[n, :]
        std::memcpy(out_ptr + n * out_row_stride,
                    weight_ptr + idx * w_row_stride,
                    row_bytes);
    };

    if (ctx.cpu_parallel_for)
        ctx.cpu_parallel_for(0, num_indices, body);
    else
        for (int64_t n = 0; n < num_indices; ++n) body(n);
}

// ============================================================
// Int8 lookup with SIMD per-row dequantization → f32 or f16
// ============================================================
//
// The inner loop uses kernel::dequant_i8_store / dequant_i8_scalar
// which are overloaded on output pointer type (float* / half*).
// Template parameter Tout is only needed for typed pointer access
// from TensorView; the loop body is otherwise type-agnostic.

template <typename Tout>
void embed_int8_dequant_impl(const EmbedAttributes& /*attrs*/,
                              TensorView& output,
                              std::span<const TensorView> inputs,
                              const ComputeContext& ctx)
{
    const auto& weight  = inputs[0];
    const auto& indices = inputs[1];

    const auto& qp = weight.quant_params();
    const bool in_is_i8 = (weight.data_type() == DataType::i8);
    const bool per_token = (qp.granularity == QuantGranularity::PerToken);
    const bool has_zp   = (qp.zero_point_data != nullptr);

    const float per_tensor_scale = qp.scale;
    const float per_tensor_zp    = static_cast<float>(qp.zero_point);

    const int64_t V = weight.shape(0);
    const int64_t dim = weight.shape(1);
    const int64_t num_indices = indices.numel();

    const void* weight_ptr = weight.ptr<void>();
    const int64_t w_row_bytes = dim;  // int8: 1 byte per element

    Tout* out_ptr = output.ptr<Tout>();
    const int64_t out_row_stride = output.row_stride_elems();
    const bool is_i64 = (indices.data_type() == DataType::i64);

    constexpr int L = kernel::kQuantLane;  // 8

    auto body = [&](int64_t n) {
        int64_t idx;
        if (is_i64) {
            idx = static_cast<const int64_t*>(indices.ptr<void>())[n];
        } else {
            idx = static_cast<const int32_t*>(indices.ptr<void>())[n];
        }

        // Clamp to valid range
        if (idx < 0) idx = 0;
        if (idx >= V) idx = V - 1;

        const void* w_row = static_cast<const uint8_t*>(weight_ptr) + idx * w_row_bytes;
        Tout* out_row = out_ptr + n * out_row_stride;

        // Resolve scale/zp for this index
        const float s_val = per_token ? qp.scale_data[idx] : per_tensor_scale;
        const float zp_val = has_zp
            ? static_cast<float>(per_token ? qp.zero_point_data[idx] : qp.zero_point)
            : 0.0f;

        int64_t j = 0;

        for (; j + L <= dim; j += L) {
            kernel::dequant_i8_store(
                static_cast<const uint8_t*>(w_row) + j, in_is_i8,
                s_val, zp_val, out_row + j);
        }

        // Scalar tail
        for (; j < dim; ++j) {
            kernel::dequant_i8_scalar(
                static_cast<const uint8_t*>(w_row) + j, in_is_i8,
                s_val, zp_val, out_row + j);
        }
    };

    if (ctx.cpu_parallel_for)
        ctx.cpu_parallel_for(0, num_indices, body);
    else
        for (int64_t n = 0; n < num_indices; ++n) body(n);
}

}  // anonymous namespace

// ============================================================
// Entry point with dtype dispatch
// ============================================================

void embed_cpu(const EmbedAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* /*workspace*/)
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

    // Int8/uint8 weight: dequantize to f32 or f16 output
    if (o_dtype == DataType::f32) {
        embed_int8_dequant_impl<float>(attrs, output, inputs, ctx);
    } else if (o_dtype == DataType::f16) {
        embed_int8_dequant_impl<half>(attrs, output, inputs, ctx);
    } else {
        NNOPS_ASSERT(!"embed_cpu: unsupported output dtype for int8 weight (f32/f16 only)");
    }
}

}  // namespace nnops::backend::cpu
