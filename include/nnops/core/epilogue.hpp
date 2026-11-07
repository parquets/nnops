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
/// Every live epilogue is stateless and applied per element as
/// `activate(clamp(x, min_clip, max_clip))`: the clip bounds belong to the
/// epilogue contract, for every operator, not just MatMul. The quantization
/// fields below are the per-channel Dequantize/Requantize stage and are not
/// live yet.

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

/// A quantization stage applied after the activation — the fusion point for a
/// QuantizeLinear/DequantizeLinear that follows a GEMM.
///
/// The stage is per output *row* (the destination's leading axis): a MatMul row
/// is one token, a Conv2D row one output channel, so `quant_scales[k]` is the
/// scale of row k. The int8 GEMM's output pass reads it, which lets a QDQ graph
/// carry the output parameters in the attributes instead of on the tensor; a
/// per-row table on the output view works too. Quantize from a floating-point
/// producer (writing s8 directly) is not implemented.
enum class EpilogueQuantType : uint8_t {
    None = 0,       ///< identity
    Requantize = 1, ///< y = clamp(round(x / scale) + zero_point)
    Dequantize = 2, ///< y = (x − zero_point) · scale
};

/// Epilogue descriptor — bundled into operator attributes.
struct Epilogue {
    EpilogueActivateType type  = EpilogueActivateType::None;
    float                alpha = 0.0f;   ///< Slope for LeakyRelu, alpha for Elu
    float                beta  = 1.0f;   ///< Parameter for HardSwish

    float min_clip = -std::numeric_limits<float>::infinity();  ///< Clamp bounds, applied before the activation.
    float max_clip =  std::numeric_limits<float>::infinity();

    // ---- Quantization stage, applied after the activation ----
    EpilogueQuantType quant = EpilogueQuantType::None;

    /// One entry per output row when quant_param_count > 0; a single entry (or
    /// nullptr with quant_param_count == 0) means per-tensor.
    const float* quant_scales = nullptr;
    const int32_t* quant_zero_points = nullptr;  ///< Optional; nullptr means zero.
    int64_t quant_param_count = 0;

    /// Reserved: the row layout makes this the leading axis (0) everywhere.
    int64_t quant_axis = 0;
};

/// True when the epilogue does anything at all: an activation type, a finite
/// clip bound, or the quantization stage. Routes use this — not a type check —
/// to decide whether their write-back pass has work to do, so a pure Clip
/// written as `Epilogue{min_clip, max_clip}` is honoured everywhere.
inline bool epilogue_is_active(const Epilogue& ep) noexcept {
    return ep.type != EpilogueActivateType::None ||
           ep.quant != EpilogueQuantType::None ||
           ep.min_clip > -std::numeric_limits<float>::infinity() ||
           ep.max_clip <  std::numeric_limits<float>::infinity();
}

/// The per-row scale / zero point of the quant stage, with the per-tensor form
/// (count <= 1) shared by every row. A null table reads as scale 1 / zero 0.
inline float epilogue_quant_scale(const Epilogue& ep, int64_t row) noexcept {
    if (ep.quant_scales == nullptr) { return 1.0f; }
    return ep.quant_scales[ep.quant_param_count > 1 ? row : 0];
}

inline int32_t epilogue_quant_zero(const Epilogue& ep, int64_t row) noexcept {
    if (ep.quant_zero_points == nullptr) { return 0; }
    return ep.quant_zero_points[ep.quant_param_count > 1 ? row : 0];
}

// ---- Scalar apply_epilogue ----
// Defined in src/backend/cpu/simd_kernel/simd_epilogue.hpp.

/// Apply an epilogue to a single scalar value; identity for type == None.
float apply_epilogue(const Epilogue& ep, float x);

/// Apply an epilogue at a channel index. Activation types ignore @p channel;
/// quantization epilogues (future) use per-channel or per-tensor parameters.
float apply_epilogue(const Epilogue& ep, float x, int64_t channel);

}  // namespace nnops
