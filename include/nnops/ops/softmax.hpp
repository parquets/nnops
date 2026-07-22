#pragma once
/// @file softmax.hpp
/// @brief Softmax operator — softmax / log-softmax along an axis.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"

#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes for the Softmax operator.
struct SoftmaxAttributes {
    /// Axis along which to compute softmax. Negative values count from the end.
    /// Default -1 normalizes over the last dimension.
    int64_t axis = -1;

    /// If true, compute log-softmax instead of softmax.
    bool log_softmax = false;

    /// If true, add result to existing output buffer instead of overwriting.
    bool add_to = false;
};

/// Softmax operator (class-based API).
///
/// Computes softmax along a given axis:
///   softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
///
/// When log_softmax is true:
///   log_softmax(x_i) = (x_i - max) - log(sum(exp(x_j - max)))
///
/// Input: [*, D_axis, *]
/// Output: same shape as input
class Softmax : public OpBase {
public:
    /// Create a Softmax operator for the specified backend.
    static std::unique_ptr<Softmax> create(const SoftmaxAttributes& attrs = {},
                                           Backend backend = Backend::CPU);

    /// Create with defaults.
    static std::unique_ptr<Softmax> create(Backend backend) {
        return create(SoftmaxAttributes{}, backend);
    }

    // ---- OpBase interface ----
    size_t getWorkspace() const override { return 0; }

    /// inputs[0] = input tensor (any rank >= 1)
    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Softmax; }
    Backend getBackend() const override { return backend_; }

    const SoftmaxAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in softmax.cpp (Pimpl pattern)

private:
    Softmax(const SoftmaxAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    SoftmaxAttributes attrs_;
    Backend backend_;
};

// ============================================================
// Functional API
// ============================================================

/// Functional softmax.
void softmax(const TensorView& input,
             TensorView& output,
             const SoftmaxAttributes& attrs = {},
             const ComputeContext& ctx = {});

}  // namespace nnops
