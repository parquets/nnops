#pragma once
/// @file reduce.hpp
/// @brief Reduce operator — reduce a tensor along a single axis (sum, min, max, mean).

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Supported reduction operations.
enum class ReduceType : uint8_t {
    Sum,    ///< sum of elements
    Min,    ///< minimum value
    Max,    ///< maximum value
    Mean,   ///< arithmetic mean (sum / count)
};

/// Attributes for the Reduce operator.
struct ReduceAttributes {
    /// Reduction operation type.
    ReduceType type = ReduceType::Sum;

    /// Axis to reduce over (negative values wrap from the end).
    int64_t axis = 0;

    /// If true, the reduced axis is kept as a size-1 dimension.
    bool keepdims = false;
};

/// Reduce operator (class-based API).
///
/// Reduces input tensor along a single axis using one of four reduction
/// operations: Sum, Min, Max, or Mean.
///
///   Sum:  output = Σ input over axis
///   Min:  output = min(input) over axis
///   Max:  output = max(input) over axis
///   Mean: output = mean(input) over axis
///
/// Input:  X [*]
/// Output: Y [*]  (reduced shape: axis removed or kept as 1 if keepdims)
class Reduce : public OpBase {
public:
    /// Create a Reduce operator for the specified backend.
    static std::unique_ptr<Reduce> create(const ReduceAttributes& attrs = {},
                                          Backend backend = Backend::CPU);

    /// Create with defaults.
    static std::unique_ptr<Reduce> create(Backend backend) {
        return create(ReduceAttributes{}, backend);
    }

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputShapes(
        std::span<const TensorDesc> inputs) const override;

    /// inputs[0] = X tensor [*]
    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Reduce; }
    Backend getBackend() const override { return backend_; }

    const ReduceAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in reduce.cpp (Pimpl pattern)

private:
    Reduce(const ReduceAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    ReduceAttributes attrs_;
    Backend backend_;
};

// ============================================================
// Functional API
// ============================================================

/// Functional tensor reduction.
void reduce(const TensorView& input,
            TensorView& output,
            const ReduceAttributes& attrs = {},
            const ComputeContext& ctx = {});

}  // namespace nnops
