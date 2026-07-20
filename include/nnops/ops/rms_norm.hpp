#pragma once
/// @file rms_norm.hpp
/// @brief RMSNorm operator — normalize by root mean square (no mean subtraction).

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"

#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes for the RMSNorm operator.
struct RMSNormAttributes {
    /// The first axis to normalize over. Normalization is computed over
    /// dimensions [axis, rank). Default -1 normalizes the last dim only.
    int64_t axis = -1;

    /// Small constant for numerical stability.
    float epsilon = 1e-5f;

    /// If true, add result to existing output buffer instead of overwriting.
    bool add_to = false;
};

/// RMSNorm operator (class-based API).
///
/// Computes:
///   rms = sqrt(mean(x_i^2) + epsilon)
///   y_i = x_i / rms * scale_i
///
/// Unlike LayerNorm, RMSNorm does NOT subtract the mean and has no bias.
///
/// Input:  X [*], scale [norm_shape]
/// Output: Y [*]
class RMSNorm : public OpBase {
public:
    /// Create a RMSNorm operator for the specified backend.
    static std::unique_ptr<RMSNorm> create(const RMSNormAttributes& attrs = {},
                                           Backend backend = Backend::CPU);

    /// Create with defaults.
    static std::unique_ptr<RMSNorm> create(Backend backend) {
        return create(RMSNormAttributes{}, backend);
    }

    // ---- OpBase interface ----
    size_t getWorkspace() const override { return 0; }

    /// inputs[0] = X [*]
    /// inputs[1] = scale, broadcastable to X.shape[axis:]
    void compute(const TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::RMSNorm; }
    Backend getBackend() const override { return backend_; }

    const RMSNormAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in rms_norm.cpp (Pimpl pattern)

private:
    RMSNorm(const RMSNormAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    RMSNormAttributes attrs_;
    Backend backend_;
};

// ============================================================
// Functional API
// ============================================================

/// Functional rms_norm.
void rms_norm(const TensorView& x,
              const TensorView& scale,
              const TensorView& output,
              const RMSNormAttributes& attrs = {},
              const ComputeContext& ctx = {});

}  // namespace nnops
