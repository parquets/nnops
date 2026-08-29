#pragma once
/// @file concat.hpp
/// @brief Concat operator — concatenate multiple tensors along an axis.
///
/// Supports NCHW, NCDHW, NCHWC8, NCDHWC8 layouts.
/// All inputs must have the same rank, data type, and layout.
///
/// F32 and F16 data types are supported.

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
    /// Axis along which to concatenate (negative values wrap from the end).
    int64_t axis = -1;
};

/// Concat operator (class-based API).
///
/// Concatenates N input tensors (N >= 2) along the specified axis.
/// All inputs must share the same rank, dtype, and layout.
/// All non-axis dimensions must be identical across inputs.
///
/// Input:  A [*], B [*], C [*], ...  (same rank/dtype/layout)
/// Output: O [*]                     (axis dim = sum of all input axis dims)
class Concat : public OpBase {
public:
    /// Create a Concat operator for the specified backend.
    static std::unique_ptr<Concat> create(const ConcatAttributes& attrs = {},
                                           Backend backend = Backend::CPU);

    /// Create with defaults.
    static std::unique_ptr<Concat> create(Backend backend) {
        return create(ConcatAttributes{}, backend);
    }

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    /// inputs[0..N-1] = tensors to concatenate
    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Concat; }
    Backend getBackend() const override { return backend_; }

    const ConcatAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;

private:
    Concat(const ConcatAttributes& attrs, Backend backend);
    std::unique_ptr<Impl> impl_;
    ConcatAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
