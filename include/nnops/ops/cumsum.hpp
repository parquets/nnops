#pragma once
/// @file cumsum.hpp
/// @brief CumSum operator — cumulative sum along an axis.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"

#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes for the CumSum operator.
struct CumSumAttributes {
    /// If true, the i-th element excludes input[i] (output[0] = 0).
    bool exclusive = false;

    /// If true, cumulative sum is performed in the reverse direction.
    bool reverse = false;

    /// Axis along which to compute the cumulative sum.
    int64_t axis = 0;

};

/// CumSum operator (class-based API).
///
/// Computes the cumulative sum along a given axis.
///   inclusive:  output[i] = sum(input[0..i])
///   exclusive:  output[0] = 0, output[i] = sum(input[0..i-1])
///
/// With reverse: sum from the last element backward.
///
/// Input: [*, D_axis, *]  (rank >= 1)
/// Output: same shape as input
class CumSum : public OpBase {
public:
    /// Create a CumSum operator for the specified backend.
    static std::unique_ptr<CumSum> create(const CumSumAttributes& attrs = {},
                                          Backend backend = Backend::CPU);

    /// Create with defaults.
    static std::unique_ptr<CumSum> create(Backend backend) {
        return create(CumSumAttributes{}, backend);
    }

    // ---- OpBase interface ----
    size_t getWorkspaceSize(std::span<const TensorDesc>,
                            std::span<const TensorDesc>) const override { return 0; }

    /// inputs[0] = input tensor (rank >= 1)
    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::CumSum; }
    Backend getBackend() const override { return backend_; }

    const CumSumAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in cumsum.cpp (Pimpl pattern)

private:
    CumSum(const CumSumAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    CumSumAttributes attrs_;
    Backend backend_;
};

// ============================================================
// Functional API
// ============================================================

/// Functional cumsum.
void cumsum(const TensorView& input,
            TensorView& output,
            const CumSumAttributes& attrs = {},
            const ComputeContext& ctx = {});

}  // namespace nnops
