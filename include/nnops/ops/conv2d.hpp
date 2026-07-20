#pragma once
/// @file conv2d.hpp
/// @brief Conv2D operator — 2D convolution with NCHW layout.

#include "nnops/core/op_base.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/core/backend.hpp"
#include "nnops/core/compute_context.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>

namespace nnops {

/// Attributes describing a 2D convolution.
struct Conv2DAttributes {
    /// Stride along height and width.
    std::array<int64_t, 2> stride = {1, 1};

    /// Dilation along height and width.
    std::array<int64_t, 2> dilation = {1, 1};

    /// Symmetric zero-padding along height and width.
    std::array<int64_t, 2> padding = {0, 0};

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
};

/// Conv2D operator (class-based API).
///
/// Supports NCHW layout with grouped and dilated convolution.
/// Input: [N, IC, IH, IW], Weight: [OC, IC/G, KH, KW] or [OC, IC/G, KH, KW] for grouped.
/// Output: [N, OC, OH, OW].
/// Optional bias: [OC].
class Conv2D : public OpBase {
public:
    /// Create a Conv2D operator for the specified backend.
    static std::unique_ptr<Conv2D> create(const Conv2DAttributes& attrs,
                                          Backend backend = Backend::CPU);

    /// Create a Conv2D operator (same as above, convenience).
    static std::unique_ptr<Conv2D> create(Backend backend = Backend::CPU) {
        return create(Conv2DAttributes{}, backend);
    }

    // ---- OpBase interface ----
    size_t getWorkspace() const override;

    /// inputs[0] = input tensor (NCHW)
    /// inputs[1] = weight tensor (OIHW or GOIHW for grouped)
    /// inputs[2] = bias tensor [OC] (optional)
    void compute(const TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Conv2D; }
    Backend getBackend() const override { return backend_; }

    /// Access the convolution attributes.
    const Conv2DAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in conv2d.cpp (Pimpl pattern)

private:
    Conv2D(const Conv2DAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    Conv2DAttributes attrs_;
    Backend backend_;
};

// ============================================================
// Functional API (convenience wrappers)
// ============================================================

/// Functional conv2d without bias.
void conv2d(const TensorView& input,
            const TensorView& weight,
            const TensorView& output,
            const Conv2DAttributes& attrs,
            const ComputeContext& ctx = {},
            void* workspace = nullptr);

/// Functional conv2d with bias.
void conv2d(const TensorView& input,
            const TensorView& weight,
            const TensorView& bias,
            const TensorView& output,
            const Conv2DAttributes& attrs,
            const ComputeContext& ctx = {},
            void* workspace = nullptr);

}  // namespace nnops
