/// @file quant_linear_ref.cpp
/// @brief Reference (scalar) CPU implementation of QuantizeLinear / DequantizeLinear.
///
/// Handles:
///   - QuantizeLinear:  f32/f16 → s8/u8
///   - DequantizeLinear: s8/u8 → f32
///   - All layouts: NCHW, NCDHW, NCHWC8, NCDHWC8
///   - PerTensor and PerChannel granularities
///
/// Serves as the correctness baseline for SIMD kernels.

#include "nnops/ops/quant_linear.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/half.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

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

/// Read a float value from input (f32 or f16).
inline float read_float_input(const TensorView& input, int64_t off) {
    if (input.data_type() == DataType::f16) {
        return half_to_float(input.ptr<half>()[off]);
    }
    return input.ptr<float>()[off];
}

/// Read an integer value from input (s8 or u8), returned as int32.
inline int32_t read_int_input(const TensorView& input, int64_t off) {
    if (input.data_type() == DataType::s8) {
        return static_cast<int32_t>(input.ptr<int8_t>()[off]);
    } else {
        return static_cast<int32_t>(input.ptr<uint8_t>()[off]);
    }
}

/// Write a float to output (f32).
inline void write_float_output(TensorView& output, int64_t off, float val) {
    output.ptr<float>()[off] = val;
}

/// Write an integer to output (s8 or u8), clamped to type range.
inline void write_int_output(TensorView& output, int64_t off, int32_t val) {
    if (output.data_type() == DataType::s8) {
        val = std::max<int32_t>(-128, std::min<int32_t>(127, val));
        output.ptr<int8_t>()[off] = static_cast<int8_t>(val);
    } else {
        val = std::max<int32_t>(0, std::min<int32_t>(255, val));
        output.ptr<uint8_t>()[off] = static_cast<uint8_t>(val);
    }
}

/// Get integer type range for clamping.
inline int32_t int_min_for(DataType dt) {
    return (dt == DataType::s8) ? -128 : 0;
}
inline int32_t int_max_for(DataType dt) {
    return (dt == DataType::s8) ? 127 : 255;
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

    const int64_t rank = x.rank();
    int64_t axis = attrs.axis;
    if (axis < 0) axis += rank;
    NNOPS_ASSERT(axis >= 0 && axis < rank);

    const int64_t D = x.shape(axis);
    const bool is_per_channel = (scale.numel() > 1);
    const int64_t n_total = x.numel();

    // Flatten the tensor: for each element, compute its scale/zp index
    // and apply the quantization formula.
    for (int64_t flat = 0; flat < n_total; ++flat) {
        // Compute the axis index for this flat position
        int64_t ax_idx = 0;
        {
            int64_t rem = flat;
            for (int64_t d = 0; d <= axis; ++d) {
                int64_t dim = x.shape(d);
                int64_t coord = rem;
                if (d < axis) {
                    // Divide by inner dimensions
                    for (int64_t inner = d + 1; inner < rank; ++inner) {
                        coord /= x.shape(inner);
                    }
                } else {
                    coord = rem % dim;
                }
                if (d == axis) ax_idx = coord;
                rem = rem;
            }
            // Simpler approach: build coordinate and read axis
            int64_t tmp = flat;
            int64_t stride_after_axis = 1;
            for (int64_t d = axis + 1; d < rank; ++d) stride_after_axis *= x.shape(d);
            ax_idx = (flat / stride_after_axis) % D;
        }

        int64_t s_idx = is_per_channel ? ax_idx : 0;
        float s = load_scale(scale, s_idx);
        int32_t z = load_zero_point(zp, s_idx);

        float x_val = read_float_input(x, flat);
        float q = std::round(x_val / s) + static_cast<float>(z);
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

    const int64_t rank = x.rank();
    int64_t axis = attrs.axis;
    if (axis < 0) axis += rank;
    NNOPS_ASSERT(axis >= 0 && axis < rank);

    const int64_t D = x.shape(axis);
    const bool is_per_channel = (scale.numel() > 1);
    const int64_t n_total = x.numel();

    for (int64_t flat = 0; flat < n_total; ++flat) {
        int64_t stride_after_axis = 1;
        for (int64_t d = axis + 1; d < rank; ++d) stride_after_axis *= x.shape(d);
        int64_t ax_idx = (flat / stride_after_axis) % D;

        int64_t s_idx = is_per_channel ? ax_idx : 0;
        float s = load_scale(scale, s_idx);
        int32_t z = load_zero_point(zp, s_idx);

        int32_t x_val = read_int_input(x, flat);
        float y_val = (static_cast<float>(x_val) - static_cast<float>(z)) * s;
        write_float_output(output, flat, y_val);
    }
}

}  // namespace nnops::backend::cpu::reference
