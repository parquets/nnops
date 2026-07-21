#pragma once
/// @file layer_norm.hpp
/// @brief LayerNorm operator — normalize over the last dimensions.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"

#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes for the LayerNorm operator.
struct LayerNormAttributes {
    /// The first axis to normalize over. Normalization is computed over
    /// dimensions [axis, rank). Default -1 normalizes the last dim only.
    int64_t axis = -1;

    /// Small constant for numerical stability.
    float epsilon = 1e-5f;

    /// If true, add result to existing output buffer instead of overwriting.
    bool add_to = false;
};

/// LayerNorm operator (class-based API).
///
/// Computes:
///   mean = sum(x_i) / norm_size
///   var  = sum((x_i - mean)^2) / norm_size
///   y_i  = (x_i - mean) / sqrt(var + epsilon) * scale_i + bias_i
///
/// Normalization is over X.shape[axis:].
///
/// Input:  X [*], scale [norm_shape], bias [norm_shape] (optional)
/// Output: Y [*]
///
/// Uses Welford's online algorithm for numerical stability.
class LayerNorm : public OpBase {
public:
    /// Create a LayerNorm operator for the specified backend.
    static std::unique_ptr<LayerNorm> create(const LayerNormAttributes& attrs = {},
                                             Backend backend = Backend::CPU);

    /// Create with defaults.
    static std::unique_ptr<LayerNorm> create(Backend backend) {
        return create(LayerNormAttributes{}, backend);
    }

    // ---- OpBase interface ----
    size_t getWorkspace() const override { return 0; }

    /// inputs[0] = X [*]
    /// inputs[1] = scale, broadcastable to X.shape[axis:]
    /// inputs[2] = bias (optional), same broadcast rules as scale
    using OpBase::compute;

    void compute(std::span<const TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::LayerNorm; }
    Backend getBackend() const override { return backend_; }

    const LayerNormAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in layer_norm.cpp (Pimpl pattern)

private:
    LayerNorm(const LayerNormAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    LayerNormAttributes attrs_;
    Backend backend_;
};

// ============================================================
// Functional API
// ============================================================

/// Functional layernorm (2 inputs: x + scale, no bias).
void layer_norm(const TensorView& x,
                const TensorView& scale,
                const TensorView& output,
                const LayerNormAttributes& attrs = {},
                const ComputeContext& ctx = {});

/// Functional layernorm with bias (3 inputs).
void layer_norm(const TensorView& x,
                const TensorView& scale,
                const TensorView& bias,
                const TensorView& output,
                const LayerNormAttributes& attrs = {},
                const ComputeContext& ctx = {});

}  // namespace nnops
