#pragma once
/// @file quant_linear.hpp
/// @brief QuantizeLinear / DequantizeLinear operators.
///
/// QuantizeLinear:  float (f32/f16) → integer (s8/u8)
///   y = clamp(round(x / scale) + zero_point, type_min, type_max)
///
/// DequantizeLinear: integer (s8/u8) → float (f32)
///   y = (x - zero_point) * scale
///
/// Three quantization granularities, selected by the `axis` attribute and the
/// scale/zero_point shape (matching ONNX conventions):
///
///   PerTensor:  scale and zero_point are scalars (numel == 1). One shared
///               parameter for the whole tensor, regardless of `axis`.
///
///   PerToken:   `axis == -1` (the default) with a per-row scale/zero_point.
///               The last (innermost) dimension is the quantization unit: every
///               element in the same row shares one parameter; different rows
///               do not. For an input of shape [..., H], scale/zero_point have
///               numel(x)/H entries. This is the activation-quantization mode
///               (per-token activation quantization in LLM inference).
///
///   PerChannel: `axis >= 0` with a per-channel scale/zero_point
///               (shape[axis] entries), broadcast across all other dims.
///               WEIGHTS ONLY — reserved for weight tensors; activations must
///               use PerTensor / PerToken.
///
/// Supported layouts: NCHW, NCDHW, NCHWC8, NCDHWC8
/// Supported float types: f32, f16
/// Supported integer types: s8, u8

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/quant_params.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes shared by QuantizeLinear and DequantizeLinear.
struct QuantLinearAttributes {
    /// Quantization axis — selects the granularity of scale/zero_point:
    ///
    ///   -1 (default): PerToken — one (scale, zero_point) per ROW. The last
    ///                 (innermost) dimension is the quantization unit: every
    ///                 element in the same row shares one parameter, different
    ///                 rows do not. For x of shape [..., H], scale/zero_point
    ///                 have numel(x)/H entries. Activation-quantization mode.
    ///
    ///   >= 0:         PerChannel — one (scale, zero_point) per channel along
    ///                 `axis` (shape[axis] entries), broadcast across all other
    ///                 dims. WEIGHTS ONLY: reserved for weight tensors.
    ///
    ///   When scale/zero_point are scalars (numel == 1), the result is
    ///   PerTensor regardless of `axis`.
    int64_t axis = -1;

    /// For QuantizeLinear: output integer type (s8 or u8).
    /// For DequantizeLinear: output float type (f32 or f16).
    DataType output_dtype = DataType::s8;
};

/// Resolve the effective quantization granularity from the scale tensor and
/// the `axis` attribute:
///   - scalar scale (numel <= 1)  → PerTensor
///   - axis < 0                   → PerToken  (per-row, last dim)
///   - axis >= 0                  → PerChannel (per-axis, weights only)
inline QuantGranularity resolve_quant_granularity(const TensorView& scale,
                                                  int64_t axis) {
    if (scale.numel() <= 1) {
        return QuantGranularity::PerTensor;
    }
    return (axis < 0) ? QuantGranularity::PerToken : QuantGranularity::PerChannel;
}

// ============================================================
// QuantizeLinear: float → integer
// ============================================================

/// QuantizeLinear operator.
///
/// Inputs:
///   [0] x:           input tensor (f32 or f16)
///   [1] y_scale:     scale tensor (f32, scalar or 1D per-channel)
///   [2] y_zero_point: zero_point tensor (s8 or u8, same shape as y_scale)
///
/// Output:
///   y: quantized tensor (s8 or u8, same shape as x)
class QuantizeLinear : public OpBase {
public:
    static std::unique_ptr<QuantizeLinear> create(
        const QuantLinearAttributes& attrs = {},
        Backend backend = Backend::CPU);

    static std::unique_ptr<QuantizeLinear> create(Backend backend) {
        return create(QuantLinearAttributes{}, backend);
    }

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    /// inputs[0] = x, inputs[1] = scale, inputs[2] = zero_point
    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::QuantizeLinear; }
    Backend getBackend() const override { return backend_; }

    const QuantLinearAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;

private:
    QuantizeLinear(const QuantLinearAttributes& attrs, Backend backend);
    std::unique_ptr<Impl> impl_;
    QuantLinearAttributes attrs_;
    Backend backend_;
};

// ============================================================
// DequantizeLinear: integer → float
// ============================================================

/// DequantizeLinear operator.
///
/// Inputs:
///   [0] x:            input tensor (s8 or u8)
///   [1] x_scale:      scale tensor (f32, scalar or 1D per-channel)
///   [2] x_zero_point:  zero_point tensor (s8 or u8, same shape as x_scale)
///
/// Output:
///   y: dequantized tensor (f32 or f16, same shape as x)
class DequantizeLinear : public OpBase {
public:
    static std::unique_ptr<DequantizeLinear> create(
        const QuantLinearAttributes& attrs = {},
        Backend backend = Backend::CPU);

    static std::unique_ptr<DequantizeLinear> create(Backend backend) {
        return create(QuantLinearAttributes{}, backend);
    }

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    /// inputs[0] = x, inputs[1] = scale, inputs[2] = zero_point
    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::DequantizeLinear; }
    Backend getBackend() const override { return backend_; }

    const QuantLinearAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;

private:
    DequantizeLinear(const QuantLinearAttributes& attrs, Backend backend);
    std::unique_ptr<Impl> impl_;
    QuantLinearAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
