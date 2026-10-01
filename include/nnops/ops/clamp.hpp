#pragma once
/// @file clamp.hpp
/// @brief Clamp operator — element-wise value clamping to [min, max].
///
/// Used for action bounds enforcement in VLA policies, numerical stability
/// clipping, and general-purpose value range limiting.
///
/// Supports arbitrary layouts (Planar + Packed) via pitch-aware row processing.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes for the Clamp operator.
struct ClampAttributes {
    /// Lower bound (values below this are set to min_val).
    float min_val = -std::numeric_limits<float>::infinity();

    /// Upper bound (values above this are set to max_val).
    float max_val = std::numeric_limits<float>::infinity();

    /// If true, add result to existing output buffer instead of overwriting.
    bool add_to = false;
};

/// Element-wise value clamping operator.
///
/// Computes: output[i] = clamp(input[i], min_val, max_val)
///   = max(min_val, min(max_val, input[i]))
///
/// Input:  X [*]          (any shape, rank, layout)
/// Output: Y [*]          (same shape, rank, layout, dtype)
class Clamp : public OpBase {
public:
    /// Create a Clamp operator for the specified backend.
    static std::unique_ptr<Clamp> create(const ClampAttributes& attrs = {},
                                          Backend backend = Backend::CPU);

    /// Create with defaults.
    static std::unique_ptr<Clamp> create(Backend backend) {
        return create(ClampAttributes{}, backend);
    }

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    /// inputs[0] = X tensor [*]
    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Clamp; }
    Backend getBackend() const override { return backend_; }

    const ClampAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in clamp.cpp (Pimpl pattern)
    ~Clamp();

private:
    Clamp(const ClampAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    ClampAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
