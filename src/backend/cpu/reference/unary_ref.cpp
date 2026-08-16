/// @file unary_ref.cpp
/// @brief Naive CPU reference implementation of element-wise unary operations.
///
/// Correctness baseline. Uses a straightforward single loop over all elements.
/// Matches the SIMD kernel's output exactly (same dtype dispatch, same switch).
///
/// Quantized input/output (s8/u8 → s8/u8) is handled by a scalar fused path:
/// dequantize (int → f32), apply the unary op, then re-quantize (f32 → int).
/// Granularity is PerTensor / PerToken only.

#include "nnops/ops/unary.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "nnops/detail/half.hpp"

#include <cmath>
#include <cstdint>

namespace nnops::backend::cpu::reference {

using nnops::simd::s_load;
using nnops::simd::s_store;

namespace {

/// Scalar unary op (f32). Round/Ceil/Floor are not reached from the quant path.
inline float unary_scalar(UnaryType type, float v) {
    switch (type) {
    case UnaryType::Exp:   return std::exp(v);
    case UnaryType::Log:   return std::log(v);
    case UnaryType::Sin:   return std::sin(v);
    case UnaryType::Cos:   return std::cos(v);
    case UnaryType::Tan:   return std::tan(v);
    case UnaryType::Tanh:  return std::tanh(v);
    case UnaryType::Abs:   return v < 0.0f ? -v : v;
    case UnaryType::Neg:   return -v;
    case UnaryType::Sqrt:  return std::sqrt(v);
    case UnaryType::Erf:   return std::erf(v);
    case UnaryType::Recip: return 1.0f / v;
    case UnaryType::Sign:  return (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f);
    default:               return v;
    }
}

}  // anonymous namespace

template <typename T>
void unary_impl_ref(const UnaryAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs)
{
    const auto& input = inputs[0];
    const int64_t total = input.numel();
    if (total == 0) { return; }
    NNOPS_ASSERT(output.numel() == total);

    const auto* in_ptr  = input.ptr<T>();
    auto*       out_ptr = output.ptr<T>();
    const bool add_to = attrs.add_to;

    // Row-by-row dimensions (for rank >= 2, respects pitch / packed layouts)
    const int64_t rank = input.rank();
    const bool use_flat = (rank < 2);
    const int64_t num_rows = use_flat ? total : input.total_rows();
    const int64_t last_dim = use_flat ? total : input.shape(rank - 1) * input.channel_pack_size();
    const int64_t in_rs = use_flat ? 1 : input.row_stride_elems();
    const int64_t out_rs = use_flat ? 1 : output.row_stride_elems();

    switch (attrs.type) {

    case UnaryType::Exp: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_rs;
            T* out_row = out_ptr + r * out_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float rv = std::exp(s_load(&in_row[i]));
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    case UnaryType::Log: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_rs;
            T* out_row = out_ptr + r * out_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float rv = std::log(s_load(&in_row[i]));
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    case UnaryType::Sin: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_rs;
            T* out_row = out_ptr + r * out_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float rv = std::sin(s_load(&in_row[i]));
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    case UnaryType::Cos: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_rs;
            T* out_row = out_ptr + r * out_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float rv = std::cos(s_load(&in_row[i]));
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    case UnaryType::Tan: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_rs;
            T* out_row = out_ptr + r * out_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float rv = std::tan(s_load(&in_row[i]));
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    case UnaryType::Tanh: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_rs;
            T* out_row = out_ptr + r * out_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float rv = std::tanh(s_load(&in_row[i]));
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    case UnaryType::Abs: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_rs;
            T* out_row = out_ptr + r * out_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float v = s_load(&in_row[i]);
                float rv = (v < 0.0f ? -v : v);
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    case UnaryType::Neg: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_rs;
            T* out_row = out_ptr + r * out_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float rv = -s_load(&in_row[i]);
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    case UnaryType::Sqrt: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_rs;
            T* out_row = out_ptr + r * out_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float rv = std::sqrt(s_load(&in_row[i]));
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    case UnaryType::Erf: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_rs;
            T* out_row = out_ptr + r * out_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float rv = std::erf(s_load(&in_row[i]));
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    case UnaryType::Round: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_rs;
            T* out_row = out_ptr + r * out_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float rv = std::round(s_load(&in_row[i]));
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    case UnaryType::Ceil: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_rs;
            T* out_row = out_ptr + r * out_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float rv = std::ceil(s_load(&in_row[i]));
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    case UnaryType::Floor: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_rs;
            T* out_row = out_ptr + r * out_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float rv = std::floor(s_load(&in_row[i]));
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    case UnaryType::Recip: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_rs;
            T* out_row = out_ptr + r * out_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float rv = 1.0f / s_load(&in_row[i]);
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    case UnaryType::Sign: {
        for (int64_t r = 0; r < num_rows; ++r) {
            const T* in_row = in_ptr + r * in_rs;
            T* out_row = out_ptr + r * out_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float v = s_load(&in_row[i]);
                float rv = (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f);
                if (add_to) {
                    s_store(&out_row[i], s_load(&out_row[i]) + rv);
                } else {
                    s_store(&out_row[i], rv);
                }
            }
        }
        break;
    }

    }  // switch
}

// ============================================================
// Fused quantized unary (scalar reference)
// ============================================================

void unary_quant_ref(const UnaryAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs)
{
    const auto& input = inputs[0];
    const int64_t numel = input.numel();
    if (numel == 0) { return; }
    NNOPS_ASSERT(!attrs.add_to);

    const DataType in_dtype  = input.data_type();   // s8 or u8
    const DataType out_dtype = output.data_type();  // s8 or u8

    const auto& in_qp  = input.quant_params();
    const auto& out_qp = output.quant_params();
    const bool in_per_token  = in_qp.granularity  == QuantGranularity::PerToken  && in_qp.scale_data != nullptr;
    const bool out_per_token = out_qp.granularity == QuantGranularity::PerToken && out_qp.scale_data != nullptr;
    const int64_t last_dim = (input.rank() >= 1) ? input.shape(input.rank() - 1) : 1;

    for (int64_t flat = 0; flat < numel; ++flat) {
        const int64_t in_idx  = in_per_token  ? (flat / last_dim) : 0;
        const int64_t out_idx = out_per_token ? (flat / last_dim) : 0;

        // Dequantize → f32.
        const float in_s = in_per_token ? in_qp.scale_data[in_idx] : in_qp.scale;
        const float in_z = in_per_token
            ? (in_qp.zero_point_data != nullptr ? static_cast<float>(in_qp.zero_point_data[in_idx]) : 0.0f)
            : static_cast<float>(in_qp.zero_point);
        const float x = (in_dtype == DataType::s8)
            ? (static_cast<float>(input.ptr<int8_t>()[flat]) - in_z) * in_s
            : (static_cast<float>(input.ptr<uint8_t>()[flat]) - in_z) * in_s;

        // Unary op.
        const float y = unary_scalar(attrs.type, x);

        // Quantize → store.
        const float out_s = out_per_token ? out_qp.scale_data[out_idx] : out_qp.scale;
        const float out_z = out_per_token
            ? (out_qp.zero_point_data != nullptr ? static_cast<float>(out_qp.zero_point_data[out_idx]) : 0.0f)
            : static_cast<float>(out_qp.zero_point);
        const int32_t qi = static_cast<int32_t>(std::nearbyintf(y / out_s) + out_z);
        if (out_dtype == DataType::s8) {
            const int32_t q = std::max<int32_t>(-128, std::min<int32_t>(127, qi));
            output.ptr<int8_t>()[flat] = static_cast<int8_t>(q);
        } else {
            const int32_t q = std::max<int32_t>(0, std::min<int32_t>(255, qi));
            output.ptr<uint8_t>()[flat] = static_cast<uint8_t>(q);
        }
    }
}

void unary_ref(const UnaryAttributes& attrs,
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
            unary_impl_ref<float>(attrs, output, inputs);
            return;
        case DataType::f16:
            unary_impl_ref<half>(attrs, output, inputs);
            return;
        default:
            NNOPS_ASSERT(!"unary_ref: unsupported data type (only f32 and f16)");
        }
    }

    // Quantized path: both input and output must be s8/u8 (no float↔int mixing).
    NNOPS_ASSERT(in_is_int && out_is_int);

    // Round/Ceil/Floor are float-meaning ops — no quantized-dtype support.
    if (attrs.type == UnaryType::Round ||
        attrs.type == UnaryType::Ceil  ||
        attrs.type == UnaryType::Floor) {
        NNOPS_ASSERT(!"unary_ref: Round/Ceil/Floor do not support quantized data types");
    }

    unary_quant_ref(attrs, output, inputs);
}

}  // namespace nnops::backend::cpu::reference
