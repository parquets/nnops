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
};

/// Linear / fully-connected operator (class-based API).
///
/// Computes: output = input × weight^T + bias
///   input:  [M, K] (2D, or [*, K] which broadcasts to 2D)
///   weight: [N, K]
///   bias:   [N] (optional)
///   output: [M, N]
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
    size_t getWorkspace() const override { return 0; }

    /// inputs[0] = input tensor [M, K] (or broadcastable to 2D)
    /// inputs[1] = weight tensor [N, K]
    /// inputs[2] = bias tensor [N] (optional)
    void compute(const TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Linear; }
    Backend getBackend() const override { return backend_; }

    /// Access the linear attributes.
    const LinearAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in linear.cpp (Pimpl pattern)

private:
    Linear(const LinearAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    LinearAttributes attrs_;
    Backend backend_;
};

// ---- Functional API ----

/// Functional linear without bias (default attributes).
void linear(const TensorView& input,
            const TensorView& weight,
            const TensorView& output,
            const ComputeContext& ctx = {});

/// Functional linear with bias (default attributes).
void linear(const TensorView& input,
            const TensorView& weight,
            const TensorView& bias,
            const TensorView& output,
            const ComputeContext& ctx = {});

/// Functional linear without bias, with epilogue support.
void linear(const TensorView& input,
            const TensorView& weight,
            const TensorView& output,
            const LinearAttributes& attrs,
            const ComputeContext& ctx = {});

/// Functional linear with bias, with epilogue support.
void linear(const TensorView& input,
            const TensorView& weight,
            const TensorView& bias,
            const TensorView& output,
            const LinearAttributes& attrs,
            const ComputeContext& ctx = {});

}  // namespace nnops
