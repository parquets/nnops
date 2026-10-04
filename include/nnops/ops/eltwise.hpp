#pragma once
/// @file eltwise.hpp
/// @brief Eltwise operator — element-wise binary operations on two tensors.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Supported element-wise binary operations.
enum class EltwiseType : uint8_t {
    Add,   ///< C = A + B
    Sub,   ///< C = A - B
    Mul,   ///< C = A * B
    Div,   ///< C = A / B
    Min,   ///< C = element-wise min(A, B)
    Max,   ///< C = element-wise max(A, B)
    Pow,   ///< C = A ^ B  (pow(A, B))
};

/// Attributes for the Eltwise operator.
struct EltwiseAttributes {
    EltwiseType type = EltwiseType::Add;

    /// If true, add result to existing output buffer instead of overwriting.
    bool add_to = false;
};

/// Eltwise operator (class-based API).
///
/// Element-wise binary operation on two tensors of the same shape and data type,
/// chosen by EltwiseAttributes::type.
///
/// Quantized input/output (s8/u8 → s8/u8) is supported through a fused path: both
/// inputs are dequantized, the op is applied in f32, and the result is
/// re-quantized (f32 → int). Both inputs and the output must be quantized. Only
/// PerTensor / PerToken quantization granularity is supported.
class Eltwise : public OpBase {
public:
    static std::unique_ptr<Eltwise> create(const EltwiseAttributes& attrs = {},
                                            Backend backend = Backend::CPU);

    static std::unique_ptr<Eltwise> create(Backend backend) {
        return create(EltwiseAttributes{}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Eltwise; }
    Backend getBackend() const override { return backend_; }

    const EltwiseAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in eltwise.cpp (Pimpl pattern)
    ~Eltwise();

private:
    Eltwise(const EltwiseAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    EltwiseAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
