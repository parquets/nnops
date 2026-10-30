#pragma once
/// @file eltwise.hpp
/// @brief Eltwise operator — element-wise binary operations on two tensors.
///
/// Both inputs and the output share a shape and a dtype. Quantized (s8/u8) in
/// and out is a fused path: dequantize both sides, apply the op in f32,
/// requantize. All three must be quantized — no float/int mixing — and share
/// one PerTensor or PerToken granularity.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

enum class EltwiseType : uint8_t {
    Add,   ///< C = A + B
    Sub,   ///< C = A - B
    Mul,   ///< C = A * B
    Div,   ///< C = A / B
    Min,   ///< C = element-wise min(A, B)
    Max,   ///< C = element-wise max(A, B)
    Pow,   ///< C = A ^ B  (pow(A, B))
};

struct EltwiseAttributes {
    EltwiseType type = EltwiseType::Add;

    /// Accumulate into the existing output instead of overwriting (residual);
    /// rejected on the quantized path.
    bool add_to = false;
};

class Eltwise : public OpBase {
public:
    static std::unique_ptr<Eltwise> create(const EltwiseAttributes& attrs = {},
                                            Backend backend = Backend::CPU);

    static std::unique_ptr<Eltwise> create(Backend backend) {
        return create(EltwiseAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Eltwise; }
    Backend getBackend() const override { return backend_; }

    const EltwiseAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~Eltwise();

private:
    Eltwise(const EltwiseAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    EltwiseAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
