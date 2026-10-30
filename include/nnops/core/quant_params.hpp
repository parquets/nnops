#pragma once
/// @file quant_params.hpp
/// @brief Quantization parameters for TensorView — scale, zero_point, granularity.
///
/// Integer quantization maps between integer storage and float:
///
///     float_val = (int_val - zero_point) * scale
///     int_val   = clamp(round(float_val / scale) + zero_point, min, max)
///
/// fp8 (f8_e4m3 / f8_e5m2) is a data-type cast, not integer quantization:
/// zero_point is 0 and only `scale` (optional range scaling) is meaningful, so
/// fp8 is not in is_quantized_dtype(). Sub-byte s4/u4 pack two values per byte,
/// low nibble first, ceil(W/2) bytes per row — data_type_size() returns 0 for
/// them, so callers must use the packed helpers.
///
/// Granularity follows ONNX: PerTensor uses the scalar fields; PerToken points at
/// one pair per row (num_scales == total_rows()); PerChannel at one pair per
/// channel along the op's channel axis. The storage DataType lives on
/// TensorView::data_type(); QuantParams holds only the mapping, so operators
/// dispatch on dtype and the scale/zero_point buffers stay non-owning (the user
/// manages their lifetime). zero_point is int32_t — wide enough for every integer
/// quant type, including int4/uint4 — and scale is float.

#include <cstdint>

namespace nnops {

/// Quantization granularity — determines how scale/zero_point are applied.
enum class QuantGranularity : uint8_t {
    PerTensor  = 0,  ///< single (scale, zero_point) for the whole tensor
    PerToken   = 1,  ///< one (scale, zero_point) per row (token-level)
    PerChannel = 2,  ///< one (scale, zero_point) per channel (op-defined axis)
};

/// Lightweight, non-owning quantization metadata (lifetime like TensorView).
struct QuantParams {
    QuantGranularity granularity = QuantGranularity::PerTensor;

    /// Per-tensor scale; also the fallback when no per-channel/token buffer is set.
    float scale = 1.0f;

    /// Per-tensor zero_point.
    int32_t zero_point = 0;

    /// Per-token/channel scales (non-owning); nullptr for PerTensor.
    const float* scale_data = nullptr;

    /// Per-token/channel zero points (non-owning); nullptr for PerTensor.
    const int32_t* zero_point_data = nullptr;

    /// Entries in scale_data/zero_point_data: rows (PerToken) or channels (PerChannel).
    int64_t num_scales = 0;

    /// True when the tensor data should be dequantized before use.
    bool is_active() const noexcept {
        if (scale_data != nullptr || zero_point_data != nullptr) {
            return true;
        }
        return scale != 1.0f || zero_point != 0;
    }
};

}  // namespace nnops
