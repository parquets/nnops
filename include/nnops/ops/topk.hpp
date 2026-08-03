#pragma once
/// @file topk.hpp
/// @brief TopK operator — selects the top-k (or bottom-k) largest values and
/// their indices along an axis.
///
/// Supports NCHW, NCDHW (planar) layouts. F32 and F16 data types.
///
/// Input:  X [*]         (f32 or f16)
/// Output: Values  [*]   (same dtype as input, axis dim replaced by k)
///         Indices [*]   (int64, same shape as Values)

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
    Max,  ///< Top-K largest values (default)
    Min,  ///< Top-K smallest values
};

/// Attributes for the TopK operator.
struct TopKAttributes {
    /// Operation type: top-k largest or smallest.
    TopKType type = TopKType::Max;

    /// Axis along which to select top-k elements.
    /// Negative values wrap from the end. Default: -1 (last axis).
    int64_t axis = -1;

    /// Number of top elements to select (k >= 1).
    int64_t k = 1;

    /// If true, output values are sorted (descending for Max, ascending for Min).
    bool sorted = true;
};

/// TopK operator (class-based API).
///
/// Selects the k largest (or smallest) values and their indices along the
/// specified axis. Returns two outputs: a values tensor (same dtype as input)
/// and an indices tensor (int64).
///
/// Input:  X [*]         (one tensor, f32 or f16)
/// Output: Values  [*]   (axis dim replaced by k, same dtype)
///         Indices [*]   (same shape as Values, int64)
class TopK : public OpBase {
public:
    /// Create a TopK operator for the specified backend.
    static std::unique_ptr<TopK> create(const TopKAttributes& attrs = {},
                                         Backend backend = Backend::CPU);

    /// Create with defaults.
    static std::unique_ptr<TopK> create(Backend backend) {
        return create(TopKAttributes{}, backend);
    }

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    /// inputs[0] = X tensor [*]
    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::TopK; }
    Backend getBackend() const override { return backend_; }

    const TopKAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in topk.cpp (Pimpl pattern)

private:
    TopK(const TopKAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    TopKAttributes attrs_;
    Backend backend_;
};

// ============================================================
// Functional API
// ============================================================

/// Functional TopK operation.
/// @param input   Input tensor.
/// @param values  Output values tensor (axis dim replaced by k).
/// @param indices Output indices tensor (int64, same shape as values).
/// @param attrs   TopK attributes (k, axis, type, sorted).
/// @param ctx     Compute context.
void topk(const TensorView& input,
          TensorView& values,
          TensorView& indices,
          const TopKAttributes& attrs = {},
          const ComputeContext& ctx = {});

}  // namespace nnops
