/// @file eltwise_ref.cpp
/// @brief Naive CPU reference implementation of element-wise binary operations.
///
/// Correctness baseline. Uses a straightforward single loop over all elements.
/// Matches the SIMD kernel's output exactly (same dtype dispatch, same switch).
///
/// Quantized input/output (s8/u8 → s8/u8) is handled by a scalar fused path:
/// dequantize both inputs (int → f32), apply the binary op, then re-quantize
/// (f32 → int). Granularity is PerTensor / PerToken only.

#include "nnops/ops/eltwise.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <cmath>
#include <cstdint>

namespace nnops::backend::cpu::reference {

using nnops::simd::s_load;
using nnops::simd::s_store;

namespace {

/// Scalar binary op (f32). Used by the quantized path.
inline float eltwise_scalar(EltwiseType type, float a, float b) {
    switch (type) {
    case EltwiseType::Add: return a + b;
    case EltwiseType::Sub: return a - b;
    case EltwiseType::Mul: return a * b;
    case EltwiseType::Div: return a / b;
    case EltwiseType::Min: return a < b ? a : b;
    case EltwiseType::Max: return a > b ? a : b;
    case EltwiseType::Pow: return std::pow(a, b);
    }
    return a;
}

}  // anonymous namespace

template <typename T>
void eltwise_impl_ref(const EltwiseAttributes& attrs,
                       TensorView& output,
                       std::span<const TensorView> inputs)
{
    const auto& A = inputs[0];
    const auto& B = inputs[1];
    const int64_t total = A.numel();
    if (total == 0) { return; }
    NNOPS_ASSERT(A.numel() == B.numel());
    NNOPS_ASSERT(output.numel() == total);

    const auto* a_ptr = A.ptr<T>();
    const auto* b_ptr = B.ptr<T>();
    auto*       o_ptr = output.ptr<T>();
    const bool add_to = attrs.add_to;

    // Row-by-row dimensions (for rank >= 2, respects pitch / packed layouts)
    const int64_t rank = A.rank();
    const bool use_flat = (rank < 2);
    const int64_t num_rows = use_flat ? total : A.total_rows();
    const int64_t last_dim = use_flat ? total : A.shape(rank - 1) * A.channel_pack_size();
    const int64_t a_rs = use_flat ? 1 : A.row_stride_elems();
    const int64_t b_rs = use_flat ? 1 : B.row_stride_elems();
    const int64_t o_rs = use_flat ? 1 : output.row_stride_elems();

    switch (attrs.type) {
    case EltwiseType::Add: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* a_row = a_ptr + r * a_rs;
            const T* b_row = b_ptr + r * b_rs;
            T* o_row = o_ptr + r * o_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float rv = s_load(&a_row[i]) + s_load(&b_row[i]);
                if (add_to) {
                    s_store(&o_row[i], s_load(&o_row[i]) + rv);
                } else {
                    s_store(&o_row[i], rv);
                }
            }
        }
        break;
    }
    case EltwiseType::Sub: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* a_row = a_ptr + r * a_rs;
            const T* b_row = b_ptr + r * b_rs;
            T* o_row = o_ptr + r * o_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float rv = s_load(&a_row[i]) - s_load(&b_row[i]);
                if (add_to) {
                    s_store(&o_row[i], s_load(&o_row[i]) + rv);
                } else {
                    s_store(&o_row[i], rv);
                }
            }
        }
        break;
    }
    case EltwiseType::Mul: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* a_row = a_ptr + r * a_rs;
            const T* b_row = b_ptr + r * b_rs;
            T* o_row = o_ptr + r * o_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float rv = s_load(&a_row[i]) * s_load(&b_row[i]);
                if (add_to) {
                    s_store(&o_row[i], s_load(&o_row[i]) + rv);
                } else {
                    s_store(&o_row[i], rv);
                }
            }
        }
        break;
    }
    case EltwiseType::Div: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* a_row = a_ptr + r * a_rs;
            const T* b_row = b_ptr + r * b_rs;
            T* o_row = o_ptr + r * o_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float rv = s_load(&a_row[i]) / s_load(&b_row[i]);
                if (add_to) {
                    s_store(&o_row[i], s_load(&o_row[i]) + rv);
                } else {
                    s_store(&o_row[i], rv);
                }
            }
        }
        break;
    }
    case EltwiseType::Min: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* a_row = a_ptr + r * a_rs;
            const T* b_row = b_ptr + r * b_rs;
            T* o_row = o_ptr + r * o_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float av = s_load(&a_row[i]);
                float bv = s_load(&b_row[i]);
                float rv = av < bv ? av : bv;
                if (add_to) {
                    s_store(&o_row[i], s_load(&o_row[i]) + rv);
                } else {
                    s_store(&o_row[i], rv);
                }
            }
        }
        break;
    }
    case EltwiseType::Max: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* a_row = a_ptr + r * a_rs;
            const T* b_row = b_ptr + r * b_rs;
            T* o_row = o_ptr + r * o_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float av = s_load(&a_row[i]);
                float bv = s_load(&b_row[i]);
                float rv = av > bv ? av : bv;
                if (add_to) {
                    s_store(&o_row[i], s_load(&o_row[i]) + rv);
                } else {
                    s_store(&o_row[i], rv);
                }
            }
        }
        break;
    }
    case EltwiseType::Pow: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* a_row = a_ptr + r * a_rs;
            const T* b_row = b_ptr + r * b_rs;
            T* o_row = o_ptr + r * o_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float rv = std::pow(s_load(&a_row[i]), s_load(&b_row[i]));
                if (add_to) {
                    s_store(&o_row[i], s_load(&o_row[i]) + rv);
                } else {
                    s_store(&o_row[i], rv);
                }
            }
        }
        break;
    }
    }
}

void eltwise_quant_ref(const EltwiseAttributes& attrs,
                       TensorView& output,
                       std::span<const TensorView> inputs)
{
    const auto& A = inputs[0];
    const auto& B = inputs[1];
    const int64_t numel = A.numel();
    if (numel == 0) { return; }
    NNOPS_ASSERT(A.numel() == B.numel());
    NNOPS_ASSERT(!attrs.add_to);

    const DataType a_dtype = A.data_type();   // s8 or u8
    const DataType b_dtype = B.data_type();   // s8 or u8
    const DataType o_dtype = output.data_type();  // s8 or u8

    const auto& a_qp = A.quant_params();
    const auto& b_qp = B.quant_params();
    const auto& o_qp = output.quant_params();

    const bool a_per_token = a_qp.granularity == QuantGranularity::PerToken && a_qp.scale_data != nullptr;
    const bool b_per_token = b_qp.granularity == QuantGranularity::PerToken && b_qp.scale_data != nullptr;
    const bool o_per_token = o_qp.granularity == QuantGranularity::PerToken && o_qp.scale_data != nullptr;
    const int64_t last_dim = (A.rank() >= 1) ? A.shape(A.rank() - 1) : 1;

    for (int64_t flat = 0; flat < numel; ++flat) {
        const int64_t a_idx = a_per_token ? (flat / last_dim) : 0;
        const int64_t b_idx = b_per_token ? (flat / last_dim) : 0;
        const int64_t o_idx = o_per_token ? (flat / last_dim) : 0;

        const float a_s = a_per_token ? a_qp.scale_data[a_idx] : a_qp.scale;
        const float a_z = a_per_token
            ? (a_qp.zero_point_data != nullptr ? static_cast<float>(a_qp.zero_point_data[a_idx]) : 0.0f)
            : static_cast<float>(a_qp.zero_point);
        const float b_s = b_per_token ? b_qp.scale_data[b_idx] : b_qp.scale;
        const float b_z = b_per_token
            ? (b_qp.zero_point_data != nullptr ? static_cast<float>(b_qp.zero_point_data[b_idx]) : 0.0f)
            : static_cast<float>(b_qp.zero_point);
        const float x = (a_dtype == DataType::s8)
            ? (static_cast<float>(A.ptr<int8_t>()[flat]) - a_z) * a_s
            : (static_cast<float>(A.ptr<uint8_t>()[flat]) - a_z) * a_s;
        const float y = (b_dtype == DataType::s8)
            ? (static_cast<float>(B.ptr<int8_t>()[flat]) - b_z) * b_s
            : (static_cast<float>(B.ptr<uint8_t>()[flat]) - b_z) * b_s;

        const float r = eltwise_scalar(attrs.type, x, y);

        const float o_s = o_per_token ? o_qp.scale_data[o_idx] : o_qp.scale;
        const float o_z = o_per_token
            ? (o_qp.zero_point_data != nullptr ? static_cast<float>(o_qp.zero_point_data[o_idx]) : 0.0f)
            : static_cast<float>(o_qp.zero_point);
        const int32_t qi = static_cast<int32_t>(std::nearbyintf(r / o_s) + o_z);
        if (o_dtype == DataType::s8) {
            const int32_t q = std::max<int32_t>(-128, std::min<int32_t>(127, qi));
            output.ptr<int8_t>()[flat] = static_cast<int8_t>(q);
        } else {
            const int32_t q = std::max<int32_t>(0, std::min<int32_t>(255, qi));
            output.ptr<uint8_t>()[flat] = static_cast<uint8_t>(q);
        }
    }
}

void eltwise_ref(const EltwiseAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& /*ctx*/,
                   void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    const bool in_is_int  = is_quantized_dtype(dtype);
    const bool out_is_int = is_quantized_dtype(output.data_type());

    if (!in_is_int && !out_is_int) {
        switch (dtype) {
        case DataType::f32:
            eltwise_impl_ref<float>(attrs, output, inputs);
            return;
        case DataType::f16:
            eltwise_impl_ref<half>(attrs, output, inputs);
            return;
        default:
            NNOPS_ASSERT(!"eltwise_ref: unsupported data type (only f32 and f16)");
        }
    }

    // Quantized path: both inputs and the output must be s8/u8 (no float↔int mixing).
    NNOPS_ASSERT(in_is_int && out_is_int);

    eltwise_quant_ref(attrs, output, inputs);
}

}  // namespace nnops::backend::cpu::reference
