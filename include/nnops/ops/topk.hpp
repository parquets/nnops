#pragma once
/// @file topk.hpp
/// @brief TopK operator — top-k (or bottom-k) values and indices along an axis.
///
/// Planar (NCHW/NCDHW) layouts, f32/f16 input. Two outputs: Values (same dtype,
/// axis dim replaced by k) and Indices (int64, same shape as Values).

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Whether to select the largest or smallest values.
enum class TopKType : uint8_t {
    Max,
    Min,
};

struct TopKAttributes {
    TopKType type = TopKType::Max;

    /// Axis along which to select (negative values wrap; default last).
    int64_t axis = -1;

    /// Number of top elements to select (k >= 1).
    int64_t k = 1;

    /// If true, Values are sorted (descending for Max, ascending for Min).
    bool sorted = true;
};

class TopK : public OpBase {
public:
    static std::unique_ptr<TopK> create(const TopKAttributes& attrs = {},
                                         Backend backend = Backend::CPU);

    static std::unique_ptr<TopK> create(Backend backend) {
        return create(TopKAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::TopK; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PlanarOnly; }

    const TopKAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in topk.cpp
    ~TopK();

private:
    TopK(const TopKAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    TopKAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
