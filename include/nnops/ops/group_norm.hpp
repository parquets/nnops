#pragma once
/// @file group_norm.hpp
/// @brief GroupNorm operator — normalize over groups of channels + spatial dims.
///
/// Divides C channels into G groups, then normalizes each group independently
/// over (C/G) channels and all spatial dimensions.
///
///   mean_g = sum(x_i) / norm_size
///   var_g  = sum((x_i - mean_g)^2) / norm_size
///   y_i    = (x_i - mean_g) / sqrt(var_g + epsilon) * scale_c + bias_c
///
/// Normalization is over channels [g*(C/G) : (g+1)*(C/G)) + all spatial dims.
///
/// References:
///   - Wu & He, "Group Normalization", ECCV 2018
///   - ONNX GroupNormalization (since opset 21)
///
/// Supported layouts: NCHW, NCDHW (PlanarOnly).
/// Supported data types: f32, f16.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes for the GroupNorm operator.
struct GroupNormAttributes {
    /// Number of channel groups. C must be divisible by num_groups.
    /// Default 1 = LayerNorm over [C, *spatial].
    /// num_groups = C = InstanceNorm over [*spatial].
    int64_t num_groups = 1;

    /// Small constant for numerical stability.
    float epsilon = 1e-5f;

    /// If true, add result to existing output buffer instead of overwriting.
    bool add_to = false;
};

/// GroupNorm operator (class-based API).
///
/// Computes per-group normalization with per-channel scale and bias.
///
/// Input:  X [N, C, *spatial], scale [C], bias [C] (optional)
/// Output: Y [N, C, *spatial]
class GroupNorm : public OpBase {
public:
    /// Create a GroupNorm operator for the specified backend.
    static std::unique_ptr<GroupNorm> create(const GroupNormAttributes& attrs = {},
                                              Backend backend = Backend::CPU);

    /// Create with defaults.
    static std::unique_ptr<GroupNorm> create(Backend backend) {
        return create(GroupNormAttributes{}, backend);
    }

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::PlanarOnly;
    }

    /// inputs[0] = X [N, C, *spatial]
    /// inputs[1] = scale [C]
    /// inputs[2] = bias [C] (optional)
    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::GroupNorm; }
    Backend getBackend() const override { return backend_; }

    const GroupNormAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in group_norm.cpp (Pimpl pattern)

private:
    GroupNorm(const GroupNormAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    GroupNormAttributes attrs_;
    Backend backend_;
};

// ============================================================
// Functional API
// ============================================================

/// Functional group_norm (2 inputs: x + scale, no bias).
void group_norm(const TensorView& x,
                const TensorView& scale,
                TensorView& output,
                const GroupNormAttributes& attrs = {},
                const ComputeContext& ctx = {});

/// Functional group_norm with bias (3 inputs).
void group_norm(const TensorView& x,
                const TensorView& scale,
                const TensorView& bias,
                TensorView& output,
                const GroupNormAttributes& attrs = {},
                const ComputeContext& ctx = {});

}  // namespace nnops
