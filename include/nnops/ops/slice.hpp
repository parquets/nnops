#pragma once
/// @file slice.hpp
/// @brief Slice operator — extract a sub-tensor along specified axes.
///
/// Equivalent to numpy/torch `x[start:end:step, ...]` on the listed axes;
/// unlisted axes are kept in full. For each listed axis the output dim is
/// ceil((end - start) / step) after clamping to the input extent. Planar only.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/detail/small_vector.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

struct SliceAttributes {
    /// Start index per listed axis (inclusive).
    detail::SmallVector<int64_t, TensorDesc::kMaxRank> starts;

    /// End index per listed axis (exclusive).
    detail::SmallVector<int64_t, TensorDesc::kMaxRank> ends;

    /// Axes to slice; unlisted axes are kept whole.
    detail::SmallVector<int64_t, TensorDesc::kMaxRank> axes;

    /// Step per listed axis (>= 1). Default: all 1.
    detail::SmallVector<int64_t, TensorDesc::kMaxRank> steps;
};

/// output = input[starts[0]:ends[0]:steps[0], ...] along the listed axes.
class Slice : public OpBase {
public:
    static std::unique_ptr<Slice> create(const SliceAttributes& attrs,
                                          Backend backend = Backend::CPU);

    static std::unique_ptr<Slice> create(Backend backend) {
        return create(SliceAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Slice; }
    Backend getBackend() const override { return backend_; }

    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::PlanarOnly;
    }

    const SliceAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in slice.cpp
    ~Slice();

private:
    Slice(const SliceAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    SliceAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
