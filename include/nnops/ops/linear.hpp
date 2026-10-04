#pragma once
/// @file linear.hpp
/// @brief Linear (fully-connected) operator: output = input × weight^T + bias.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/epilogue.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes for the Linear (fully-connected) operator.
struct LinearAttributes {
    /// Post-processing applied during output write-back (default: identity).
    Epilogue epilogue{};

    /// If true, add result to existing output buffer instead of overwriting.
    bool add_to = false;
};

/// Linear / fully-connected operator (class-based API).
///
/// Computes: output = input × weight^T + bias
///   input:  [..., K]
///   weight: [N, K]
///   bias:   [N] (optional)
///   output: [..., N]   (leading dims preserved; only the last becomes N)
///
/// Transposed weight is implicit — the weight is stored as [N, K] and
/// the multiplication is input × weight^T.
class Linear : public OpBase {
public:
    /// Create a Linear operator for the specified backend.
    static std::unique_ptr<Linear> create(const LinearAttributes& attrs,
                                          Backend backend = Backend::CPU);

    /// Create with defaults (convenience).
    static std::unique_ptr<Linear> create(Backend backend = Backend::CPU) {
        return create(LinearAttributes{}, backend);
    }

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    /// inputs[0] = input tensor [M, K] (or broadcastable to 2D)
    /// inputs[1] = weight tensor [N, K]
    /// inputs[2] = bias tensor [N] (optional)
    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Linear; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PlanarOnly; }

    /// Access the linear attributes.
    const LinearAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in linear.cpp (Pimpl pattern)
    ~Linear();

private:
    Linear(const LinearAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    LinearAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
