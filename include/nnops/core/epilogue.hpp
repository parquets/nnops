#pragma once
/// @file epilogue.hpp
/// @brief Epilogue type for operator fusion — apply activation functions
///        or other post-processing during output write-back.
///
/// This is intentionally separate from activation.hpp (ActivationType /
/// ActivationAttributes) so that:
///   - The standalone Activation operator stays clean (no "None" sentinel).
///   - Future epilogue types (quantize/dequantize, clamp, etc.) have a
///     natural home without polluting the activation concept.
///
/// The Epilogue struct is defined here so operator headers can embed it.
/// Implementations of apply_epilogue (scalar and SIMD) live in the CPU
/// backend at src/backend/cpu/simd_kernel/simd_epilogue.hpp.

#include <cstdint>

namespace nnops {

/// Types of activation post-processing that can be fused into an operator's
/// output write-back step.
enum class EpilogueActivateType : uint8_t {
    None       = 0,   ///< Identity — no epilogue (default)
    Relu       = 1,   ///< f(x) = max(0, x)
    Gelu       = 2,   ///< f(x) = x * Phi(x) (Gaussian error linear unit)
    Sigmoid    = 3,   ///< f(x) = 1 / (1 + exp(-x))
    Tanh       = 4,   ///< f(x) = tanh(x)
    LeakyRelu  = 5,   ///< f(x) = x > 0 ? x : alpha * x
    Silu       = 6,   ///< f(x) = x * sigmoid(x) (Swish)
    HardSwish  = 7,   ///< f(x) = x * relu6(x + 3) / 6
    Elu        = 8,   ///< f(x) = x > 0 ? x : alpha * (exp(x) - 1)
    Relu6      = 9,   ///< f(x) = min(max(0, x), 6)

    // ---- Quantize / Dequantize (future) ----
    // Dequantize  = 9,   ///< y = (x - zp[c]) * scale[c]  (per-channel or per-tensor)
    // Requantize  = 10,  ///< y = round(x / scale[c]) + zp[c]
};

/// Epilogue descriptor — bundled into operator attributes.
///
/// Covers two categories of post-processing:
///   1. Activation functions (Relu, Gelu, …) — stateless, applied per-element.
///   2. Quantize / Dequantize — per-channel scales and zero-points.
///
/// When type == EpilogueActivateType::None, the epilogue is identity and
/// has zero runtime cost beyond a predictable branch.
struct Epilogue {
    EpilogueActivateType type  = EpilogueActivateType::None;
    float                alpha = 0.0f;   ///< Slope for LeakyRelu, alpha for Elu
    float                beta  = 1.0f;   ///< Parameter for HardSwish

    float min_clip = -std::numeric_limits<float>::infinity();  ///< Optional clamp min (default: none)
    float max_clip =  std::numeric_limits<float>::infinity();  ///< Optional clamp max
    // ---- Per-channel quantization parameters (for Dequantize / Requantize) ----
    /// Per-channel quantization scales.
    /// Size: quant_param_count. Indexed by the channel axis.
    const float* quant_scales = nullptr;

    /// Per-channel zero points (optional — may be nullptr for symmetric quantization).
    /// Size: quant_param_count (same as scales). Type depends on quant_dtype.
    const int32_t* quant_zero_points = nullptr;

    /// Number of quantization parameters (== number of channels along quant_axis).
    /// 0 means per-tensor quantization (use quant_scales[0] / quant_zero_points[0]).
    /// > 0 means per-channel quantization with quant_param_count channels.
    int64_t quant_param_count = 0;

    /// Axis along which per-channel quantization is applied.
    /// For NCHW / NCDHW convolution output: axis 1 (output channels).
    /// For MatMul / Linear output [M, N]: axis 1 (N dimension).
    int64_t quant_axis = 1;
};

// ---- Scalar apply_epilogue (declarations only) ----
// Implementations live in src/backend/cpu/simd_kernel/simd_epilogue.hpp.

/// Apply an epilogue to a single scalar output value (activation types).
/// Returns the value unchanged when type == None (identity).
float apply_epilogue(const Epilogue& ep, float x);

/// Apply an epilogue to a single scalar output value at a given channel index.
///
/// For activation types: delegates to the scalar overload (channel is ignored).
/// For dequantize / requantize (future): uses per-channel or per-tensor
/// quantization parameters.
///
/// @param ep       Epilogue descriptor
/// @param x        Output value to transform
/// @param channel  Channel index along quant_axis (0-based). Only meaningful
///                 for quantization epilogues; ignored for activation types.
float apply_epilogue(const Epilogue& ep, float x, int64_t channel);

}  // namespace nnops
