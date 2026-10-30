#pragma once
/// @file concat.hpp
/// @brief Concatenate N tensors along one axis.
///
/// N >= 2. All inputs share rank, dtype and layout; every non-axis dimension
/// must match, and the output axis dimension is their sum. All layouts
/// (NCHW / NCDHW / NCHWC8 / NCDHWC8); f32 and f16.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes for the Concat operator.
struct ConcatAttributes {
    /// Axis; negative values wrap from the end.
    int64_t axis = -1;
};

class Concat : public OpBase {
public:
    static std::unique_ptr<Concat> create(const ConcatAttributes& attrs = {},
                                           Backend backend = Backend::CPU);

    static std::unique_ptr<Concat> create(Backend backend) {
        return create(ConcatAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Concat; }
    Backend getBackend() const override { return backend_; }

    const ConcatAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~Concat();

private:
    Concat(const ConcatAttributes& attrs, Backend backend);
    std::unique_ptr<Impl> impl_;
    ConcatAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
