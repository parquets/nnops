#pragma once
/// @file conv3d.hpp
/// @brief Conv3D operator — 3D convolution with NCDHW layout.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/epilogue.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes describing a 3D convolution.
struct Conv3DAttributes {
    /// Kernel size along depth, height, and width.
    std::array<int64_t, 3> kernel_size = {1, 1, 1};

    /// Stride along depth, height, and width.
    std::array<int64_t, 3> stride = {1, 1, 1};

    /// Dilation along depth, height, and width.
    std::array<int64_t, 3> dilation = {1, 1, 1};

    /// Symmetric zero-padding along depth, height, and width.
    std::array<int64_t, 3> padding = {0, 0, 0};

    /// Number of groups for grouped / depthwise convolution.
    int64_t groups = 1;

    /// Convenience enum for automatic padding computation.
    enum class AutoPad : uint8_t {
        NOTSET = 0,
        SAME_UPPER,
        SAME_LOWER,
        VALID,
    };
    AutoPad auto_pad = AutoPad::NOTSET;

    /// Post-processing applied during output write-back (default: identity).
    Epilogue epilogue{};

    /// If true, add result to existing output buffer instead of overwriting.
    bool add_to = false;
};

/// Conv3D operator (class-based API).
///
/// Supports NCDHW layout with grouped and dilated convolution.
/// Input: [N, IC, ID, IH, IW], Weight: [OC, IC/G, KD, KH, KW].
/// Output: [N, OC, OD, OH, OW].
/// Optional bias: [OC].
class Conv3D : public OpBase {
public:
    /// Create a Conv3D operator for the specified backend.
    static std::unique_ptr<Conv3D> create(const Conv3DAttributes& attrs,
                                          Backend backend = Backend::CPU);

    /// Create a Conv3D operator (same as above, convenience).
    static std::unique_ptr<Conv3D> create(Backend backend = Backend::CPU) {
        return create(Conv3DAttributes{}, backend);
    }

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    /// inputs[0] = input tensor (NCDHW)
    /// inputs[1] = weight tensor (OIDHW or GOIDHW for grouped)
    /// inputs[2] = bias tensor [OC] (optional)
    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Conv3D; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PlanarOnly; }

    /// Access the convolution attributes.
    const Conv3DAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in conv3d.cpp (Pimpl pattern)

private:
    Conv3D(const Conv3DAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    Conv3DAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
