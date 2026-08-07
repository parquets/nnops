#pragma once
/// @file slice.hpp
/// @brief Slice operator — extract a sub-tensor along specified axes.
///
/// Equivalent to `x[start:end:step, ...]` in numpy / torch.
/// Supports Planar layout only.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/detail/small_vector.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes for the Slice operator.
///
/// Axes not listed in @p axes are kept in full (same as identity).
/// For each axis in @p axes, output_dim = ceil((end - start) / step).
struct SliceAttributes {
    /// Start indices for each sliced axis (inclusive).
    detail::SmallVector<int64_t, TensorDesc::kMaxRank> starts;

    /// End indices for each sliced axis (exclusive).
    detail::SmallVector<int64_t, TensorDesc::kMaxRank> ends;

    /// Which axes to slice. Default: empty (all syntactic sugar handled by caller).
    detail::SmallVector<int64_t, TensorDesc::kMaxRank> axes;

    /// Step per sliced axis (must be >= 1). Default: all 1.
    detail::SmallVector<int64_t, TensorDesc::kMaxRank> steps;
};

/// Sub-tensor extraction operator.
///
/// Computes: output = input[starts[0]:ends[0]:steps[0], ...] along specified axes.
///
/// Input:  X [*]  (any rank, planar only, f32/f16)
/// Output: Y [*]  (same dtype, sliced shape, planar)
class Slice : public OpBase {
public:
    /// Create a Slice operator for the specified backend.
    static std::unique_ptr<Slice> create(const SliceAttributes& attrs,
                                          Backend backend = Backend::CPU);

    /// Create with defaults.
    static std::unique_ptr<Slice> create(Backend backend) {
        return create(SliceAttributes{}, backend);
    }

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Slice; }
    Backend getBackend() const override { return backend_; }

    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::PlanarOnly;
    }

    const SliceAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in slice.cpp (Pimpl pattern)

private:
    Slice(const SliceAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    SliceAttributes attrs_;
    Backend backend_;
};

// ============================================================
// Functional API
// ============================================================

/// Functional slice.
void slice(const TensorView& input,
           TensorView& output,
           const SliceAttributes& attrs,
           const ComputeContext& ctx = {});

}  // namespace nnops
