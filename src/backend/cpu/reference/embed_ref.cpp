/// @file embed_ref.cpp
/// @brief Scalar CPU reference implementation of embedding table lookup.
///
/// For each index in the indices tensor, copies the corresponding row
/// from the weight table to the output. Out-of-range indices are clamped
/// to [0, vocab_size - 1].
///
/// Supports direct lookup (f32/f16 weight) and int8 quantized weight with
/// per-row dequantization: output[j] = (weight[idx,j] - zp[idx]) * scale[idx].

#include "nnops/ops/embed.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <cstring>

namespace nnops::backend::cpu::reference {

// ============================================================
// Direct lookup (f32/f16 weight)
// ============================================================

template <typename T>
void embed_direct_ref_impl(const EmbedAttributes& /*attrs*/,
                            TensorView& output,
                            std::span<const TensorView> inputs)
{
    const auto& weight  = inputs[0];
    const auto& indices = inputs[1];

    const int64_t V   = weight.shape(0);   // vocab size
    const int64_t dim = weight.shape(1);   // embedding dim
    const int64_t num_indices = indices.numel();

    const T* weight_ptr = weight.ptr<T>();
    const int64_t w_row_stride = (weight.rank() == 2)
        ? dim
        : weight.row_stride_elems();

    T* out_ptr = output.ptr<T>();
    const int64_t out_row_stride = output.row_stride_elems();

    // Handle both int64 and int32 indices
    const bool is_i64 = (indices.data_type() == DataType::s64);

    for (int64_t n = 0; n < num_indices; ++n) {
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
                    static_cast<size_t>(dim) * sizeof(T));
    }
}

// ============================================================
// Int8 lookup with scalar per-row dequantization
// ============================================================

template <typename Tout>
void embed_int8_dequant_ref_impl(const EmbedAttributes& /*attrs*/,
                                  TensorView& output,
                                  std::span<const TensorView> inputs)
{
    const auto& weight  = inputs[0];
    const auto& indices = inputs[1];

    const auto& qp = weight.quant_params();
    const bool in_is_i8 = (weight.data_type() == DataType::s8);
    // Embed weight is quantized per-row: PerChannel on a 2D weight quantizes
    // along axis 0 (the vocab entry) — identical to PerToken. Per-tensor and
    // block quantization are not supported: reject.
    const bool per_row = (qp.granularity == QuantGranularity::PerToken)
                      || (qp.granularity == QuantGranularity::PerChannel);
    NNOPS_ASSERT(per_row);
    const bool has_zp = (qp.zero_point_data != nullptr);

    const int64_t V = weight.shape(0);
    const int64_t dim = weight.shape(1);
    const int64_t num_indices = indices.numel();

    const void* weight_ptr = weight.ptr<void>();
    const int64_t w_row_bytes = dim;

    Tout* out_ptr = output.ptr<Tout>();
    const int64_t out_row_stride = output.row_stride_elems();
    const bool is_i64 = (indices.data_type() == DataType::s64);

    for (int64_t n = 0; n < num_indices; ++n) {
        int64_t idx;
        if (is_i64) {
            idx = static_cast<const int64_t*>(indices.ptr<void>())[n];
        } else {
            idx = static_cast<const int32_t*>(indices.ptr<void>())[n];
        }

        if (idx < 0) {
            idx = 0;
        }
        if (idx >= V) {
            idx = V - 1;
        }

        const float s_val = qp.scale_data[idx];
        const float zp_val = has_zp ? static_cast<float>(qp.zero_point_data[idx]) : 0.0f;

        Tout* out_row = out_ptr + n * out_row_stride;

        for (int64_t j = 0; j < dim; ++j) {
            float w = in_is_i8
                ? static_cast<float>(static_cast<const int8_t*>(weight_ptr)[idx * w_row_bytes + j])
                : static_cast<float>(static_cast<const uint8_t*>(weight_ptr)[idx * w_row_bytes + j]);
            float r = (w - zp_val) * s_val;
            simd::s_store(&out_row[j], r);
        }
    }
}

void embed_ref(const EmbedAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& /*ctx*/,
                 void* /*workspace*/)
{
    const auto w_dtype = inputs[0].data_type();
    const auto o_dtype = output.data_type();

    if (w_dtype == DataType::f32) {
        embed_direct_ref_impl<float>(attrs, output, inputs);
        return;
    }
    if (w_dtype == DataType::f16) {
        embed_direct_ref_impl<half>(attrs, output, inputs);
        return;
    }

    // Int8/uint8 weight: dequantize to f32 or f16
    if (o_dtype == DataType::f32) {
        embed_int8_dequant_ref_impl<float>(attrs, output, inputs);
    } else if (o_dtype == DataType::f16) {
        embed_int8_dequant_ref_impl<half>(attrs, output, inputs);
    }
}

}  // namespace nnops::backend::cpu::reference
