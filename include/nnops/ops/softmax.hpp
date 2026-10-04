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

    /// Temperature scaling factor (T > 0).
    ///
    /// softmax(x_i, T) = exp(x_i / T) / sum(exp(x_j / T))
    ///
    /// T = 1.0  → standard softmax
    /// T > 1.0  → softer / more uniform distribution
    /// T < 1.0  → sharper / more peaked distribution
    ///
    /// The max-subtraction trick still applies:
    ///   softmax(x_i, T) = exp((x_i - max) / T) / sum(exp((x_j - max) / T))
    ///
    /// Must be > 0. Default 1.0 (standard softmax).
    float temperature = 1.0f;
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
/// Output: same shape as input.
///
/// Quantized input (s8/u8) is supported: the integer input is dequantized to
/// f32, softmax runs in f32, and the output is float (f32 by default, or f16).
/// Softmax probabilities are never re-quantized to integer. Only PerTensor /
/// PerToken granularity and planar (NCHW) layouts are supported for quantized
/// input; packed channel layouts are not defined there.
class Softmax : public OpBase {
public:
    static std::unique_ptr<Softmax> create(const SoftmaxAttributes& attrs = {},
                                           Backend backend = Backend::CPU);

    static std::unique_ptr<Softmax> create(Backend backend) {
        return create(SoftmaxAttributes{}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    /// Always 0: the CPU kernel pools its own scratch.
    size_t getWorkspaceSize(std::span<const TensorDesc> inputs,
                            std::span<const TensorDesc> outputs) const override;

    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Softmax; }
    Backend getBackend() const override { return backend_; }

    const SoftmaxAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in softmax.cpp (Pimpl pattern)
    ~Softmax();

private:
    Softmax(const SoftmaxAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    SoftmaxAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
