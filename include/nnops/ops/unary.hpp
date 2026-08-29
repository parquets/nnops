#pragma once
/// @file unary.hpp
/// @brief Unary operator — element-wise single-tensor math operations.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Supported element-wise unary operations.
enum class UnaryType : uint8_t {
    Exp,    ///< f(x) = exp(x)
    Log,    ///< f(x) = ln(x)
    Sin,    ///< f(x) = sin(x)
    Cos,    ///< f(x) = cos(x)
    Tan,    ///< f(x) = tan(x)
    Tanh,   ///< f(x) = tanh(x)
    Abs,    ///< f(x) = |x|
    Neg,    ///< f(x) = -x
    Sqrt,   ///< f(x) = sqrt(x)
    Erf,    ///< f(x) = erf(x)  (Gauss error function)
    Round,  ///< f(x) = round(x) (nearest integer, ties to even)
    Ceil,   ///< f(x) = ceil(x)
    Floor,  ///< f(x) = floor(x)
    Recip,  ///< f(x) = 1/x
    Sign,   ///< f(x) = sign(x)  (-1, 0, or 1)
};

/// Attributes for the Unary operator.
struct UnaryAttributes {
    UnaryType type = UnaryType::Exp;

    /// If true, add result to existing output buffer instead of overwriting.
    bool add_to = false;
};

/// Unary operator (class-based API).
///
/// Element-wise math operations on a single tensor.
///
///   Exp:  output[i] = exp(input[i])
///   Log:  output[i] = ln(input[i])
///   Sin:  output[i] = sin(input[i])
///   Cos:  output[i] = cos(input[i])
///   Tan:  output[i] = tan(input[i])
///   Tanh: output[i] = tanh(input[i])
///   Abs:  output[i] = |input[i]|
///   Neg:  output[i] = -input[i]
///   Sqrt: output[i] = sqrt(input[i])
///
/// Input:  X [*]
/// Output: Y [*]  (same shape)
///
/// Quantized input/output (s8/u8 → s8/u8) is supported through a fused path: the
/// quantized input is dequantized, the op is applied in f32, and the result is
/// re-quantized (f32 → int). Both input and output must be quantized. Only
/// PerTensor / PerToken activation granularity is used; Round/Ceil/Floor are
/// float-meaning ops and do not support quantized data types.
class Unary : public OpBase {
public:
    /// Create a Unary operator for the specified backend.
    static std::unique_ptr<Unary> create(const UnaryAttributes& attrs = {},
                                          Backend backend = Backend::CPU);

    /// Create with defaults.
    static std::unique_ptr<Unary> create(Backend backend) {
        return create(UnaryAttributes{}, backend);
    }

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    /// inputs[0] = X tensor [*]
    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Unary; }
    Backend getBackend() const override { return backend_; }

    const UnaryAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in unary.cpp (Pimpl pattern)

private:
    Unary(const UnaryAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    UnaryAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
