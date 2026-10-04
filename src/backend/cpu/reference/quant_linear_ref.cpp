/// @file quant_linear_ref.cpp
/// @brief Reference (scalar) CPU implementation of QuantizeLinear / DequantizeLinear.
///
/// Handles:
///   - QuantizeLinear:  f32/f16 → s8/u8
///   - DequantizeLinear: s8/u8 → f32
///   - All layouts: NCHW, NCDHW, NCHWC8, NCDHWC8
///   - PerTensor, PerToken and PerChannel granularities
///
/// Serves as the correctness baseline for SIMD kernels.

#include "nnops/ops/quant_linear.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace nnops::backend::cpu::reference {

using nnops::backend::cpu::half;
using nnops::backend::cpu::half_to_float;
using nnops::backend::cpu::float_to_half;

// ============================================================
// Helpers
// ============================================================

namespace {

/// Load a float value from a scale/zp tensor at a flat index.
/// The scale tensor is always f32; zp tensor is s8/u8.
inline float load_scale(const TensorView& scale_tensor, int64_t idx) {
    return scale_tensor.ptr<float>()[idx];
}

inline int32_t load_zero_point(const TensorView& zp_tensor, int64_t idx) {
    DataType zdt = zp_tensor.data_type();
    if (zdt == DataType::s8) {
        return static_cast<int32_t>(zp_tensor.ptr<int8_t>()[idx]);
    } else {
        return static_cast<int32_t>(zp_tensor.ptr<uint8_t>()[idx]);
    }
}

inline float read_float_input(const TensorView& input, int64_t off) {
    if (input.data_type() == DataType::f16) {
        return half_to_float(input.ptr<half>()[off]);
    }
    return input.ptr<float>()[off];
}

inline int32_t read_int_input(const TensorView& input, int64_t off) {
    if (input.data_type() == DataType::s8) {
        return static_cast<int32_t>(input.ptr<int8_t>()[off]);
    } else {
        return static_cast<int32_t>(input.ptr<uint8_t>()[off]);
    }
}

inline void write_float_output(TensorView& output, int64_t off, float val) {
    output.ptr<float>()[off] = val;
}

inline void write_int_output(TensorView& output, int64_t off, int32_t val) {
    if (output.data_type() == DataType::s8) {
        val = std::max<int32_t>(-128, std::min<int32_t>(127, val));
        output.ptr<int8_t>()[off] = static_cast<int8_t>(val);
    } else {
        val = std::max<int32_t>(0, std::min<int32_t>(255, val));
        output.ptr<uint8_t>()[off] = static_cast<uint8_t>(val);
    }
}

inline int32_t int_min_for(DataType dt) {
    return (dt == DataType::s8) ? -128 : 0;
}
inline int32_t int_max_for(DataType dt) {
    return (dt == DataType::s8) ? 127 : 255;
}

/// Resolve the PerChannel quantization axis (negative → from end).
inline int64_t resolve_quant_axis(const TensorView& x, int64_t axis) {
    return axis < 0 ? axis + x.rank() : axis;
}

/// Compute the index into scale/zero_point for a flat element offset.
///
///   PerTensor  → 0
///   PerToken   → row index (flat / last_dim) — one parameter per row
///   PerChannel → axis coordinate ((flat / stride_after_axis) % shape[axis])
inline int64_t quant_param_index(const TensorView& x, int64_t flat,
                                 QuantGranularity g, int64_t axis) {
    switch (g) {
    case QuantGranularity::PerToken: {
        const int64_t last_dim = x.shape(x.rank() - 1);
        return flat / last_dim;
    }
    case QuantGranularity::PerChannel: {
        const int64_t D = x.shape(axis);
        int64_t stride_after = 1;
        for (int64_t d = axis + 1; d < x.rank(); ++d) {
            stride_after *= x.shape(d);
        }
        return (flat / stride_after) % D;
    }
    case QuantGranularity::PerTensor:
    default:
        return 0;
    }
}

}  // anonymous namespace

// ============================================================
// QuantizeLinear reference: float → integer
// ============================================================

void quantize_linear_ref(const QuantLinearAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         void* /*workspace*/)
{
    const auto& x     = inputs[0];  // f32 or f16
    const auto& scale = inputs[1];  // f32
    const auto& zp    = inputs[2];  // s8 or u8

    const QuantGranularity g = resolve_quant_granularity(scale, attrs.axis);
    const int64_t axis = resolve_quant_axis(x, attrs.axis);
    NNOPS_ASSERT(axis >= 0 && axis < x.rank());

    const int64_t n_total = x.numel();

    for (int64_t flat = 0; flat < n_total; ++flat) {
        const int64_t s_idx = quant_param_index(x, flat, g, axis);
        const float s = load_scale(scale, s_idx);
        const int32_t z = load_zero_point(zp, s_idx);

        const float x_val = read_float_input(x, flat);
        const float q = std::nearbyintf(x_val / s) + static_cast<float>(z);
        write_int_output(output, flat, static_cast<int32_t>(q));
    }
}

// ============================================================
// DequantizeLinear reference: integer → float
// ============================================================

void dequantize_linear_ref(const QuantLinearAttributes& attrs,
                           TensorView& output,
                           std::span<const TensorView> inputs,
                           const ComputeContext& ctx,
                           void* /*workspace*/)
{
    const auto& x     = inputs[0];  // s8 or u8
    const auto& scale = inputs[1];  // f32
    const auto& zp    = inputs[2];  // s8 or u8

    const QuantGranularity g = resolve_quant_granularity(scale, attrs.axis);
    const int64_t axis = resolve_quant_axis(x, attrs.axis);
    NNOPS_ASSERT(axis >= 0 && axis < x.rank());

    const int64_t n_total = x.numel();

    for (int64_t flat = 0; flat < n_total; ++flat) {
        const int64_t s_idx = quant_param_index(x, flat, g, axis);
        const float s = load_scale(scale, s_idx);
        const int32_t z = load_zero_point(zp, s_idx);

        const int32_t x_val = read_int_input(x, flat);
        const float y_val = (static_cast<float>(x_val) - static_cast<float>(z)) * s;
        write_float_output(output, flat, y_val);
    }
}

}  // namespace nnops::backend::cpu::reference
