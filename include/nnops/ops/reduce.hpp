#pragma once
/// @file reduce.hpp
/// @brief Reduce operator — reduce a tensor along a single axis.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
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

    /// Axis to reduce over (negative values wrap from the end).
    int64_t axis = 0;

    /// If true, the reduced axis is kept as a size-1 dimension.
    bool keepdims = false;
};

/// Reduces one axis; the output drops it, or keeps it as size 1 if `keepdims`.
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
