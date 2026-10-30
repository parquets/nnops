#pragma once
/// @file unary.hpp
/// @brief Unary operator — element-wise single-tensor math operations.
///
/// output[i] = f(input[i]) with f from UnaryType; shape, rank and layout are
/// preserved. Quantized s8/u8 → s8/u8 runs through a fused
/// dequantize/apply/requantize path (both sides quantized, PerTensor/PerToken
/// only); Round/Ceil/Floor are float-meaning ops and reject quantized data.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

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

struct UnaryAttributes {
    UnaryType type = UnaryType::Exp;

    /// If true, add into the existing output buffer instead of overwriting.
    bool add_to = false;
};

class Unary : public OpBase {
public:
    static std::unique_ptr<Unary> create(const UnaryAttributes& attrs = {},
                                          Backend backend = Backend::CPU);

    static std::unique_ptr<Unary> create(Backend backend) {
        return create(UnaryAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Unary; }
    Backend getBackend() const override { return backend_; }

    const UnaryAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in unary.cpp
    ~Unary();

private:
    Unary(const UnaryAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    UnaryAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
