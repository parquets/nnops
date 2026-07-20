#pragma once
/// @file linear.hpp
/// @brief Linear (fully-connected) operator: output = input × weight^T + bias.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"

#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

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
    static std::unique_ptr<Linear> create(Backend backend = Backend::CPU);

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

    struct Impl;  // defined in linear.cpp (Pimpl pattern)

private:
    explicit Linear(Backend backend);

    std::unique_ptr<Impl> impl_;
    Backend backend_;
};

// ---- Functional API ----

void linear(const TensorView& input,
            const TensorView& weight,
            const TensorView& output,
            const ComputeContext& ctx = {});

void linear(const TensorView& input,
            const TensorView& weight,
            const TensorView& bias,
            const TensorView& output,
            const ComputeContext& ctx = {});

}  // namespace nnops
