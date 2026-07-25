#pragma once
/// @file reduce.hpp
/// @brief Reduce operator — reduce a tensor along specified axes (sum, min, max, mean).

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/detail/small_vector.hpp"

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

    /// Axes to reduce over. Empty = reduce all axes to a scalar.
    detail::SmallVector<int64_t, 4> axes;

    /// If true, reduced axes are kept as size-1 dimensions.
    bool keepdims = false;
};

/// Reduce operator (class-based API).
///
/// Reduces input tensor along specified axes using one of four reduction
/// operations: Sum, Min, Max, or Mean.
///
///   Sum:  output = Σ input over axes
///   Min:  output = min(input) over axes
///   Max:  output = max(input) over axes
///   Mean: output = mean(input) over axes
///
/// Input:  X [*]
/// Output: Y [*]  (reduced shape: axes removed or kept as 1 if keepdims)
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
    size_t getWorkspaceSize(std::span<const TensorDesc>,
                            std::span<const TensorDesc>) const override { return 0; }

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
