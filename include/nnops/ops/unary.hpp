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
/// Output: Y [*]  (same shape and dtype)
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
    size_t getWorkspaceSize(std::span<const TensorDesc>,
                            std::span<const TensorDesc>) const override { return 0; }

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

// ============================================================
// Functional API
// ============================================================

/// Functional element-wise unary operation.
void unary(const TensorView& input,
           TensorView& output,
           const UnaryAttributes& attrs = {},
           const ComputeContext& ctx = {});

}  // namespace nnops
