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
/// Both support three quantization granularities via scale/zero_point shape:
///   PerTensor:  scale and zero_point are scalars (rank-0 or single-element)
///   PerChannel: scale and zero_point are 1D along `axis` (shape[axis] elements)
///   PerToken:   scale and zero_point are 1D along `axis` (generalized)
///
/// Supported layouts: NCHW, NCDHW, NCHWC8, NCDHWC8
/// Supported float types: f32, f16
/// Supported integer types: s8, u8

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes shared by QuantizeLinear and DequantizeLinear.
struct QuantLinearAttributes {
    /// Axis along which per-channel/per-token quantization is applied.
    /// -1 = last axis (default). Set to 1 for per-channel weight quantization.
    int64_t axis = -1;

    /// For QuantizeLinear: output integer type (s8 or u8).
    /// For DequantizeLinear: output float type (f32 or f16).
    DataType output_dtype = DataType::s8;
};

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

// ============================================================
// Functional API
// ============================================================

/// Functional quantize: float → integer.
void quantize_linear(const TensorView& input,
                     const TensorView& scale,
                     const TensorView& zero_point,
                     TensorView& output,
                     const QuantLinearAttributes& attrs = {},
                     const ComputeContext& ctx = {});

/// Functional dequantize: integer → float.
void dequantize_linear(const TensorView& input,
                       const TensorView& scale,
                       const TensorView& zero_point,
                       TensorView& output,
                       const QuantLinearAttributes& attrs = {},
                       const ComputeContext& ctx = {});

}  // namespace nnops
