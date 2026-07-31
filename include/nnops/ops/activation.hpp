#pragma once
/// @file activation.hpp
/// @brief Activation operator — element-wise activation functions.

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
    Relu,       ///< f(x) = max(0, x)
    LeakyRelu,  ///< f(x) = x > 0 ? x : alpha * x
    Sigmoid,    ///< f(x) = 1 / (1 + exp(-x))
    Tanh,       ///< f(x) = tanh(x)
    Gelu,       ///< f(x) = x * Phi(x) (Gaussian error linear unit)
    Silu,       ///< f(x) = x * sigmoid(x) (Swish)
    HardSwish,  ///< f(x) = x * relu6(x + 3) / 6
    Elu,        ///< f(x) = x > 0 ? x : alpha * (exp(x) - 1)
};

/// Attributes for the Activation operator.
struct ActivationAttributes {
    ActivationType type = ActivationType::Relu;
    float alpha = 0.0f;  ///< Slope for LeakyRelu, alpha for Elu
    float beta  = 1.0f;  ///< Parameter for HardSwish

    /// If true, add result to existing output buffer instead of overwriting.
    bool add_to = false;
};

/// Activation operator (class-based API).
///
/// Element-wise operation: output[i] = f(input[i]).
/// Input and output must have the same shape and data type.
class Activation : public OpBase {
public:
    /// Create an Activation operator for the specified backend.
    static std::unique_ptr<Activation> create(const ActivationAttributes& attrs,
                                              Backend backend = Backend::CPU);

    /// Create with defaults.
    static std::unique_ptr<Activation> create(Backend backend = Backend::CPU) {
        return create(ActivationAttributes{}, backend);
    }

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Activation; }
    Backend getBackend() const override { return backend_; }

    const ActivationAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in activation.cpp (Pimpl pattern)

private:
    Activation(const ActivationAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    ActivationAttributes attrs_;
    Backend backend_;
};

/// Functional activation.
void activation(const TensorView& input,
                TensorView& output,
                const ActivationAttributes& attrs,
                const ComputeContext& ctx = {});

}  // namespace nnops
