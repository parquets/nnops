#pragma once
/// @file quant_params.hpp
/// @brief Quantization parameters for TensorView — scale, zero_point, granularity.
///
/// # Quantization Model
///
/// Integer quantization maps between integer storage and floating-point values.
/// The dequantization formula is:
///
///     float_val = (int_val - zero_point) * scale
///
/// Quantization (float → int):
///
///     int_val = clamp(round(float_val / scale) + zero_point, min, max)
///
/// ## Supported & planned quantized types
///
/// | Type   | Storage  | zero_point range | scale type | Status  |
/// |--------|----------|------------------|------------|---------|
/// | int8   | 1 byte   | [-128, 127]      | float      | done    |
/// | uint8  | 1 byte   | [0, 255]         | float      | done    |
/// | int4   | 0.5 byte | [-8, 7]          | float      | planned |
/// | uint4  | 0.5 byte | [0, 15]          | float      | planned |
/// | fp8    | 1 byte   | n/a (zp=0)       | float      | planned |
///
/// For fp8 types (f8_e4m3, f8_e5m2), the "quantization" is a data-type cast.
/// zero_point is always 0; only `scale` is meaningful (acts as a range-scaling
/// factor). The dequant formula simplifies to:
///
///     float_val = fp8_to_float(int_val) * scale
///
/// ## int4 / uint4 sub-byte layout
///
/// Each byte holds two 4-bit values. Within a byte, the lower 4 bits (mask 0x0F)
/// hold the first element and the upper 4 bits (mask 0xF0) hold the second.
/// For a row of W elements, the row occupies ceil(W/2) bytes. The pitch model
/// and stride_elems() semantics need adjustment — see data_type_size() which
/// returns 0 for sub-byte types to flag that packed-element helpers are required.
///
/// ## Three granularities (matching ONNX conventions)
///
///   PerTensor:  one (scale, zero_point) for the entire tensor.
///               Uses the scalar `scale` and `zero_point` fields directly.
///
///   PerToken:   one (scale, zero_point) per row (innermost grouping).
///               `scale_data` and `zero_point_data` point to arrays of
///               length `num_scales`, where num_scales = total_rows().
///
///   PerChannel: one (scale, zero_point) per channel (axis=1).
///               `scale_data` and `zero_point_data` point to arrays of
///               length `num_scales`, where num_scales = shape[1] (channels).
///
/// ## Design notes
///
/// - `zero_point` is `int32_t` — wide enough for any integer quant type
///   including future int4/uint4, avoiding the need to change the struct.
///
/// - `scale` is always `float` — sufficient for all types including fp8
///   range scaling.
///
/// - `scale_data` / `zero_point_data` are non-owning pointers matching
///   TensorView's lifetime model. The user manages the buffer memory.
///
/// - The storage DataType (s8/u8/future s4/u4) is held by
///   TensorView::data_type(), NOT by QuantParams. This separation lets
///   operators dispatch on dtype while QuantParams handles the mapping.

#include <cstdint>

namespace nnops {

/// Quantization granularity — determines how scale/zero_point are applied.
enum class QuantGranularity : uint8_t {
    PerTensor  = 0,  ///< single (scale, zero_point) for the whole tensor
    PerToken   = 1,  ///< one (scale, zero_point) per row (token-level)
    PerChannel = 2,  ///< one (scale, zero_point) per channel (axis=1)
    PerBlock   = 3,  ///< one (scale, zero_point) per physical block (used by KV-cache block quantization)
};

/// Lightweight quantization metadata (non-owning, like TensorView).
///
/// For PerTensor, the scalar `scale` and `zero_point` fields are used directly.
/// For PerToken / PerChannel, `scale_data` and `zero_point_data` point to
/// external buffers of length `num_scales` (user-managed lifetime).
///
/// The storage DataType (s8/u8; future s4/u4) is held by TensorView::data_type() —
/// this struct only holds the mapping parameters.
///
/// Extensibility:
///   - int4/uint4: zero_point in int32_t covers all 4-bit values ([-8,7] / [0,15]).
///     Scale/zero_point buffer pointers work unchanged.
///   - fp8 (f8_e4m3 / f8_e5m2): zero_point is always 0. Only `scale` is used
///     for optional range scaling. fp8 types are NOT in is_quantized_dtype() —
///     they use their own cast helpers, not the integer dequant formula.
struct QuantParams {
    QuantGranularity granularity = QuantGranularity::PerTensor;

    /// Per-tensor scale (always valid; also the fallback when per-channel/token
    /// buffers are not provided).
    float scale = 1.0f;

    /// Per-tensor zero_point (always valid).
    /// int32_t is wide enough for all integer quant types including int4/uint4.
    int32_t zero_point = 0;

    /// Per-token / per-channel scale buffer (non-owning pointer).
    /// nullptr for PerTensor; length = num_scales otherwise.
    const float* scale_data = nullptr;

    /// Per-token / per-channel zero_point buffer (non-owning pointer).
    /// nullptr for PerTensor; length = num_scales otherwise.
    const int32_t* zero_point_data = nullptr;

    /// Number of entries in scale_data / zero_point_data.
    /// For PerToken: equals the number of rows (total_rows()).
    /// For PerChannel: equals the number of channels (shape[1]).
    int64_t num_scales = 0;

    /// Whether these parameters represent an active quantization.
    /// Returns true when the tensor data should be dequantized before use.
    bool is_active() const noexcept {
        if (scale_data != nullptr || zero_point_data != nullptr) {
            return true;
        }
        return scale != 1.0f || zero_point != 0;
    }
};

}  // namespace nnops
