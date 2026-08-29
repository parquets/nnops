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
/// Element-wise binary operations on two tensors of the same shape and data type.
///
///   Add: output[i] = A[i] + B[i]
///   Sub: output[i] = A[i] - B[i]
///   Mul: output[i] = A[i] * B[i]
///   Div: output[i] = A[i] / B[i]
///   Min: output[i] = min(A[i], B[i])
///   Max: output[i] = max(A[i], B[i])
///
/// Input:  A [*], B [*]  (same shape and dtype)
/// Output: C [*]          (same shape and dtype)
///
/// Quantized input/output (s8/u8 → s8/u8) is supported through a fused path: both
/// inputs are dequantized, the op is applied in f32, and the result is
/// re-quantized (f32 → int). Both inputs and the output must be quantized. Only
/// PerTensor / PerToken activation granularity is used.
class Eltwise : public OpBase {
public:
    /// Create an Eltwise operator for the specified backend.
    static std::unique_ptr<Eltwise> create(const EltwiseAttributes& attrs = {},
                                            Backend backend = Backend::CPU);

    /// Create with defaults.
    static std::unique_ptr<Eltwise> create(Backend backend) {
        return create(EltwiseAttributes{}, backend);
    }

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    /// inputs[0] = A tensor [*]
    /// inputs[1] = B tensor [*]  (same shape and dtype as A)
    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Eltwise; }
    Backend getBackend() const override { return backend_; }

    const EltwiseAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in eltwise.cpp (Pimpl pattern)

private:
    Eltwise(const EltwiseAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    EltwiseAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
