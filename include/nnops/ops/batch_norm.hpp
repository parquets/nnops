#pragma once
/// @file batch_norm.hpp
/// @brief BatchNorm operator — per-channel normalization (inference only).

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"

#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes for the BatchNorm operator (inference mode only).
struct BatchNormAttributes {
    /// Small constant for numerical stability.
    float epsilon = 1e-5f;

    /// If true, compute per-channel statistics (spatial=1 in ONNX).
    /// The scale/bias/mean/var have shape [C].
    /// If false, statistics are per-element (spatial=0).
    bool spatial = true;

    /// If true, add result to existing output buffer instead of overwriting.
    bool add_to = false;
};

/// BatchNorm operator (class-based API, inference only).
///
/// Computes:
///   inv_std = 1 / sqrt(var + epsilon)
///   y = (x - mean) * inv_std * scale + bias
///   fused as: y = x * (inv_std * scale) + (bias - mean * inv_std * scale)
///
/// Input:  X [N, C, D1, D2...], scale [C], bias [C], mean [C], var [C]
/// Output: Y [N, C, D1, D2...]
///
/// For 1-D input [N], C is assumed to be 1.
class BatchNorm : public OpBase {
public:
    /// Create a BatchNorm operator for the specified backend.
    static std::unique_ptr<BatchNorm> create(const BatchNormAttributes& attrs = {},
                                             Backend backend = Backend::CPU);

    /// Create with defaults.
    static std::unique_ptr<BatchNorm> create(Backend backend) {
        return create(BatchNormAttributes{}, backend);
    }

    // ---- OpBase interface ----
    size_t getWorkspace() const override { return 0; }

    /// inputs[0] = X [N, C, D1...]
    /// inputs[1] = scale [C] (or [C, D1...] if !spatial)
    /// inputs[2] = bias [C] (or [C, D1...] if !spatial)
    /// inputs[3] = mean [C] (or [C, D1...] if !spatial)
    /// inputs[4] = var [C] (or [C, D1...] if !spatial)
    void compute(const TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::BatchNorm; }
    Backend getBackend() const override { return backend_; }

    const BatchNormAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in batch_norm.cpp (Pimpl pattern)

private:
    BatchNorm(const BatchNormAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    BatchNormAttributes attrs_;
    Backend backend_;
};

// ============================================================
// Functional API
// ============================================================

/// Functional batchnorm (5 inputs).
void batch_norm(const TensorView& x,
                const TensorView& scale,
                const TensorView& bias,
                const TensorView& mean,
                const TensorView& var,
                const TensorView& output,
                const BatchNormAttributes& attrs = {},
                const ComputeContext& ctx = {});

}  // namespace nnops
