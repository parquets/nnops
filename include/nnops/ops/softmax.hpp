#pragma once
/// @file softmax.hpp
/// @brief Softmax operator — softmax / log-softmax along an axis.
///
/// softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max)); log-softmax subtracts
/// log(sum) instead. A temperature T divides the max-shifted logits by T
/// (equivalently softmax(x/T)). f32/f16 input only; NHWC/NDHWC.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

struct SoftmaxAttributes {
    /// Axis to normalize over (negative values count from the end; default last).
    int64_t axis = -1;

    /// If true, compute log-softmax instead of softmax.
    bool log_softmax = false;

    /// Temperature T > 0: logits are divided by T (T=1 standard; T>1 flatter,
    /// T<1 sharper). Must be > 0. Default 1.0.
    float temperature = 1.0f;
};

/// Softmax over `axis`; input and output share shape.
class Softmax : public OpBase {
public:
    static std::unique_ptr<Softmax> create(const SoftmaxAttributes& attrs = {},
                                           Backend backend = Backend::CPU);

    static std::unique_ptr<Softmax> create(Backend backend) {
        return create(SoftmaxAttributes{}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    Status validate(std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs) const override;

    /// In-place: outputs[0] may reuse an input buffer of the same geometry.
    bool supports_inplace(std::span<const TensorDesc> inputs,
                          std::span<const TensorDesc> outputs) const override;

    using OpBase::compute;

    Status compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Softmax; }
    Backend getBackend() const override { return backend_; }

    /// NHWC/NDHWC only; packed layouts are refused.
    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::ChannelLastOnly;
    }

    const SoftmaxAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in softmax.cpp
    ~Softmax();

private:
    Softmax(const SoftmaxAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    SoftmaxAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
