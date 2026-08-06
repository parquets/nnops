#pragma once
/// @file flatten.hpp
/// @brief Flatten operator — convert any layout to dense contiguous planar data.
///
/// Takes any input tensor (any layout: NCHW, NCHWC8, NCDHW, NCDHWC8, with
/// or without pitch padding) and produces a dense contiguous output with
/// the same logical shape and data type.
///
/// This is the equivalent of `np.ascontiguousarray()` in NumPy or
/// `.contiguous()` in PyTorch.  All channel packing is unrolled and
/// all pitch padding is removed.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <memory>
#include <span>

namespace nnops {

/// Attributes for the Flatten operator (currently no attributes — always
/// produces dense planar output with the same logical shape).
struct FlattenAttributes {
    // Reserved for future options (e.g. flatten_to_1d, target_dtype).
};

/// Flatten operator — make tensor data dense and contiguous.
///
/// Input:  any tensor [N, C, (D,) H, W] in any layout
/// Output: dense planar tensor with the same logical shape and dtype
///
/// Key use cases:
///   - Export a packed tensor (NCHWC8 → NCHW dense)
///   - Remove pitch padding after aligned allocations
///   - Prepare data for serialization or external consumers
class Flatten : public OpBase {
public:
    /// Create a Flatten operator for the specified backend.
    static std::unique_ptr<Flatten> create(const FlattenAttributes& attrs = {},
                                            Backend backend = Backend::CPU);

    /// Create with defaults.
    static std::unique_ptr<Flatten> create(Backend backend) {
        return create(FlattenAttributes{}, backend);
    }

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Flatten; }
    Backend getBackend() const override { return backend_; }

    const FlattenAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;

private:
    Flatten(const FlattenAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    FlattenAttributes attrs_;
    Backend backend_;
};

/// Functional flatten.
void flatten(const TensorView& input,
             TensorView& output,
             const ComputeContext& ctx = {});

}  // namespace nnops
