#pragma once
/// @file reduce.hpp
/// @brief Reduce operator — reduce a tensor over one or more axes.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/detail/small_vector.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

enum class ReduceType : uint8_t {
    Sum,
    Min,
    Max,
    Mean,  ///< arithmetic mean (sum / count)
};

struct ReduceAttributes {
    ReduceType type = ReduceType::Sum;

    /// Axes to reduce over; negative values wrap from the end. An empty set
    /// means every axis (the ONNX Reduce* default), unless
    /// `noop_with_empty_axes` turns that into a copy.
    detail::SmallVector<int64_t, TensorDesc::kMaxRank> axes;

    /// If true, each reduced axis is kept as a size-1 dimension.
    bool keepdims = false;

    /// ONNX opset-18 semantics: with an empty `axes`, pass the input through
    /// unchanged instead of reducing every axis.
    bool noop_with_empty_axes = false;
};

/// Reduces the axes in `axes`; the output drops them, or keeps them as size 1
/// if `keepdims`.
class Reduce : public OpBase {
public:
    static std::unique_ptr<Reduce> create(const ReduceAttributes& attrs = {},
                                          Backend backend = Backend::CPU);

    static std::unique_ptr<Reduce> create(Backend backend) {
        return create(ReduceAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::Reduce; }
    Backend getBackend() const override { return backend_; }

    const ReduceAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in reduce.cpp
    ~Reduce();

private:
    Reduce(const ReduceAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    ReduceAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
