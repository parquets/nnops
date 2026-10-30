#pragma once
/// @file quant_linear.hpp
/// @brief QuantizeLinear / DequantizeLinear operators.
///
/// QuantizeLinear:   y = clamp(round(x / scale) + zero_point, type_min, type_max)
/// DequantizeLinear: y = (x - zero_point) * scale
///
/// Three granularities — PerTensor, PerToken, PerChannel — picked from the
/// scale shape and `axis` (ONNX conventions); see resolve_quant_granularity.
/// Layout-agnostic: the kernels treat the tensor as a flat row-major numel()
/// array, so packed and planar are not distinguished. Floats f32/f16;
/// integers s8/u8.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/quant_params.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

struct QuantLinearAttributes {
    /// -1: PerToken — one param per innermost row (for [..., H], numel/H
    ///     entries). >= 0: PerChannel — one param per index along `axis`,
    ///     broadcast over the rest; intended for weights. A scalar scale
    ///     (numel == 1) is PerTensor regardless of `axis`.
    int64_t axis = -1;

    /// QuantizeLinear: output integer type (s8/u8). DequantizeLinear: output
    /// float type (f32/f16; anything else resolves to f32 at construction).
    DataType output_dtype = DataType::s8;
};

/// Scalar scale → PerTensor, axis < 0 → PerToken, else PerChannel.
inline QuantGranularity resolve_quant_granularity(const TensorView& scale,
                                                  int64_t axis) {
    if (scale.numel() <= 1) {
        return QuantGranularity::PerTensor;
    }
    return (axis < 0) ? QuantGranularity::PerToken : QuantGranularity::PerChannel;
}

/// x (f32/f16), y_scale (f32), y_zero_point (s8/u8) → y (s8/u8).
class QuantizeLinear : public OpBase {
public:
    static std::unique_ptr<QuantizeLinear> create(
        const QuantLinearAttributes& attrs = {},
        Backend backend = Backend::CPU);

    static std::unique_ptr<QuantizeLinear> create(Backend backend) {
        return create(QuantLinearAttributes{}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    Status validate(std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs) const override;

    using OpBase::compute;

    Status compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::QuantizeLinear; }
    Backend getBackend() const override { return backend_; }

    const QuantLinearAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~QuantizeLinear();

private:
    QuantizeLinear(const QuantLinearAttributes& attrs, Backend backend);
    std::unique_ptr<Impl> impl_;
    QuantLinearAttributes attrs_;
    Backend backend_;
};

/// x (s8/u8), x_scale (f32), x_zero_point (s8/u8) → y (f32/f16).
class DequantizeLinear : public OpBase {
public:
    static std::unique_ptr<DequantizeLinear> create(
        const QuantLinearAttributes& attrs = {},
        Backend backend = Backend::CPU);

    static std::unique_ptr<DequantizeLinear> create(Backend backend) {
        return create(QuantLinearAttributes{}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    Status validate(std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs) const override;

    using OpBase::compute;

    Status compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::DequantizeLinear; }
    Backend getBackend() const override { return backend_; }

    const QuantLinearAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~DequantizeLinear();

private:
    DequantizeLinear(const QuantLinearAttributes& attrs, Backend backend);
    std::unique_ptr<Impl> impl_;
    QuantLinearAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
