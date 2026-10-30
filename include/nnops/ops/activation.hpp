#pragma once
/// @file activation.hpp
/// @brief Element-wise Activation operator.
///
/// output[i] = f(input[i]); input and output have the same shape and dtype.
/// Quantized s8/u8 → s8/u8 is a fused dequantize → f32 activation → requantize
/// path; both operands must be quantized, and only PerTensor / PerToken
/// granularity is supported.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Supported activation types.
enum class ActivationType : uint8_t {
    Relu,
    LeakyRelu,  ///< x > 0 ? x : alpha * x
    Sigmoid,
    Tanh,
    Gelu,       ///< tanh approximation
    Silu,       ///< x * sigmoid(x) (Swish)
    HardSwish,  ///< x * relu6(x + 3) * beta / 6
    Elu,        ///< x > 0 ? x : alpha * (exp(x) - 1)
};

/// Attributes for the Activation operator.
struct ActivationAttributes {
    ActivationType type = ActivationType::Relu;
    float alpha = 0.0f;  ///< Slope for LeakyRelu, alpha for Elu
    float beta  = 1.0f;  ///< Parameter for HardSwish

    /// If true, add result to existing output buffer instead of overwriting.
    bool add_to = false;
};

class Activation : public OpBase {
public:
    static std::unique_ptr<Activation> create(const ActivationAttributes& attrs,
                                              Backend backend = Backend::CPU);

    static std::unique_ptr<Activation> create(Backend backend = Backend::CPU) {
        return create(ActivationAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Activation; }
    Backend getBackend() const override { return backend_; }

    const ActivationAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in activation.cpp (Pimpl pattern)
    ~Activation();

private:
    Activation(const ActivationAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    ActivationAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
