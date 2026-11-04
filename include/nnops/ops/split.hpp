#pragma once
/// @file split.hpp
/// @brief Split — one tensor into N slices along an axis.
///
/// `split` gives the slice lengths; when empty, `num_outputs` equal slices are
/// taken and the axis must divide evenly. The slices cover the axis in order, so
/// their lengths sum to it. Each output is a plain copy of a contiguous axis
/// region. Planar.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace nnops {

struct SplitAttributes {
    int64_t axis = 0;  ///< negative values wrap

    /// Slice lengths; empty means equal slices sized by num_outputs.
    std::vector<int64_t> split;

    /// Output count when `split` is empty (ONNX's num_outputs).
    int64_t num_outputs = 0;

    /// The number of outputs the attributes describe: split.size(), else
    /// num_outputs.
    int64_t output_count() const noexcept {
        return split.empty() ? num_outputs : static_cast<int64_t>(split.size());
    }
};

class Split : public OpBase {
public:
    static std::unique_ptr<Split> create(const SplitAttributes& attrs,
                                         Backend backend = Backend::CPU);
    static std::unique_ptr<Split> create(Backend backend = Backend::CPU) {
        return create(SplitAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Split; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::PlanarOnly;
    }

    struct Impl;
    ~Split();

private:
    Split(const SplitAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    SplitAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
