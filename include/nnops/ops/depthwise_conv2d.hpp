#pragma once
/// @file depthwise_conv2d.hpp
/// @brief DepthwiseConv2D operator — depthwise separable 2D convolution (NCHW layout).
///
/// Depthwise convolution applies a separate filter to each input channel independently.
/// There is no cross-channel mixing. This is the first step of a depthwise-separable
/// convolution (the second step being a 1x1 pointwise conv).
///
/// Input:  [N, C, IH, IW]
/// Weight: [C, 1, KH, KW]  (one KHxKW filter per channel)
/// Output: [N, C, OH, OW]
/// Bias:   [C] (optional, per-channel)

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

/// Attributes describing a 2D depthwise convolution.
struct DepthwiseConv2DAttributes {
    /// Kernel size along height and width.
    std::array<int64_t, 2> kernel_size = {1, 1};

    /// Stride along height and width.
    std::array<int64_t, 2> stride = {1, 1};

    /// Dilation along height and width.
    std::array<int64_t, 2> dilation = {1, 1};

    /// Symmetric zero-padding along height and width.
    std::array<int64_t, 2> padding = {0, 0};

    /// Post-processing applied during output write-back (default: identity).
    Epilogue epilogue{};

    /// If true, add result to existing output buffer instead of overwriting.
    /// Enables residual connections without a separate add kernel:
    ///   output += DepthwiseConv(input)  rather than  output = DepthwiseConv(input)
    bool add_to = false;
};

/// DepthwiseConv2D operator (class-based API).
///
/// Supports NCHW layout with depthwise (channel-wise) convolution.
/// Each input channel is convolved with its own independent filter.
/// Input: [N, C, IH, IW], Weight: [C, 1, KH, KW].
/// Output: [N, C, OH, OW].
/// Optional bias: [C].
class DepthwiseConv2D : public OpBase {
public:
    /// Create a DepthwiseConv2D operator for the specified backend.
    static std::unique_ptr<DepthwiseConv2D> create(
        const DepthwiseConv2DAttributes& attrs,
        Backend backend = Backend::CPU);

    /// Create a DepthwiseConv2D operator with default attributes.
    static std::unique_ptr<DepthwiseConv2D> create(Backend backend = Backend::CPU) {
        return create(DepthwiseConv2DAttributes{}, backend);
    }

    // ---- OpBase interface ----
    size_t getWorkspace() const override;

    /// inputs[0] = input tensor  [N, C, IH, IW]
    /// inputs[1] = weight tensor [C, 1, KH, KW]
    /// inputs[2] = bias tensor   [C] (optional)
    void compute(const TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::DepthwiseConv2D; }
    Backend getBackend() const override { return backend_; }

    /// Access the depthwise convolution attributes.
    const DepthwiseConv2DAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in depthwise_conv2d.cpp (Pimpl pattern)

private:
    DepthwiseConv2D(const DepthwiseConv2DAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    DepthwiseConv2DAttributes attrs_;
    Backend backend_;
};

// ============================================================
// Functional API (convenience wrappers)
// ============================================================

/// Functional depthwise_conv2d without bias.
void depthwise_conv2d(const TensorView& input,
                       const TensorView& weight,
                       const TensorView& output,
                       const DepthwiseConv2DAttributes& attrs,
                       const ComputeContext& ctx = {},
                       void* workspace = nullptr);

/// Functional depthwise_conv2d with bias.
void depthwise_conv2d(const TensorView& input,
                       const TensorView& weight,
                       const TensorView& bias,
                       const TensorView& output,
                       const DepthwiseConv2DAttributes& attrs,
                       const ComputeContext& ctx = {},
                       void* workspace = nullptr);

}  // namespace nnops
