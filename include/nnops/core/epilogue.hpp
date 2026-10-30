#pragma once
/// @file epilogue.hpp
/// @brief Epilogue type for operator fusion — apply activation functions or
///        other post-processing during output write-back.
///
/// Deliberately separate from activation.hpp (ActivationType /
/// ActivationAttributes): the standalone Activation operator stays free of a
/// "None" sentinel, and future epilogue types (quantize/dequantize, clamp) have
/// a home without polluting the activation concept. The Epilogue struct lives
/// here so operator headers can embed it; apply_epilogue() (scalar and SIMD) is
/// implemented in src/backend/cpu/simd_kernel/simd_epilogue.hpp.
///
/// Only the activation types are live: they are stateless and applied per
/// element. The per-channel quantization fields are reserved — no code path
/// reads them yet. apply_epilogue() ignores min_clip/max_clip; the MatMul kernels
/// apply them as a clamp, and MatMul/Conv fold Relu/Relu6 into their clamp.

#include <cstdint>

namespace nnops {

/// Types of activation post-processing that can be fused into an operator's
/// output write-back step.
enum class EpilogueActivateType : uint8_t {
    None       = 0,   ///< Identity — no epilogue (default)
    Relu       = 1,   ///< f(x) = max(0, x)
    Gelu       = 2,   ///< f(x) ≈ x * Phi(x), tanh approximation (matches activation)
    Sigmoid    = 3,   ///< f(x) = 1 / (1 + exp(-x))
    Tanh       = 4,   ///< f(x) = tanh(x)
    LeakyRelu  = 5,   ///< f(x) = x > 0 ? x : alpha * x
    Silu       = 6,   ///< f(x) = x * sigmoid(x) (Swish)
    HardSwish  = 7,   ///< f(x) = x * relu6(x + 3) * beta / 6
    Elu        = 8,   ///< f(x) = x > 0 ? x : alpha * (exp(x) - 1)
    Relu6      = 9,   ///< f(x) = min(max(0, x), 6)
};

/// Epilogue descriptor — bundled into operator attributes.
struct Epilogue {
    EpilogueActivateType type  = EpilogueActivateType::None;
    float                alpha = 0.0f;   ///< Slope for LeakyRelu, alpha for Elu
    float                beta  = 1.0f;   ///< Parameter for HardSwish

    float min_clip = -std::numeric_limits<float>::infinity();  ///< Clamp bounds; apply_epilogue() ignores them.
    float max_clip =  std::numeric_limits<float>::infinity();

    // ---- Per-channel quantization parameters (reserved: Dequantize / Requantize) ----
    const float* quant_scales = nullptr;  ///< Size quant_param_count, indexed by quant_axis.

    const int32_t* quant_zero_points = nullptr;  ///< Optional; nullptr for symmetric quant.

    int64_t quant_param_count = 0;  ///< Channels along quant_axis; 0 means per-tensor.

    int64_t quant_axis = 1;  ///< Axis the params apply to: 1 = conv out channels / MatMul N.
};

// ---- Scalar apply_epilogue ----
// Defined in src/backend/cpu/simd_kernel/simd_epilogue.hpp.

/// Apply an epilogue to a single scalar value; identity for type == None.
float apply_epilogue(const Epilogue& ep, float x);

/// Apply an epilogue at a channel index. Activation types ignore @p channel;
/// quantization epilogues (future) use per-channel or per-tensor parameters.
float apply_epilogue(const Epilogue& ep, float x, int64_t channel);

}  // namespace nnops
