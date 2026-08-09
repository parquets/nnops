#pragma once
/// @file transpose_conv2d.hpp
/// @brief TransposeConv2D operator — transposed 2D convolution (deconvolution).
///
/// Transpose convolution (also called deconvolution or fractionally-strided
/// convolution) is the backward pass of regular convolution. Instead of
/// mapping an input patch to a single output pixel (many-to-one), each input
/// pixel is scaled by the kernel and scattered/accumulated across a region
/// of the output (one-to-many).
///
/// Input:  [N, IC, IH, IW]   — NCHWC8 (packed, channels in groups of 8)
/// Weight: [IC, OC/G, KH, KW] — planar NCHW
/// Bias:   [OC]               — planar (optional)
/// Output: [N, OC, OH, OW]   — NCHWC8 (packed)
///
/// Reference: ncnn Deconvolution (scatter-add approach).

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

/// Attributes describing a 2D transposed convolution.
struct TransposeConv2DAttributes {
    /// Kernel size along height and width.
    std::array<int64_t, 2> kernel_size = {1, 1};

    /// Stride along height and width.
    std::array<int64_t, 2> stride = {1, 1};

    /// Dilation along height and width.
    std::array<int64_t, 2> dilation = {1, 1};

    /// Symmetric zero-padding along height and width.
    /// Padding is subtracted from the output (the opposite of regular conv).
    std::array<int64_t, 2> padding = {0, 0};

    /// Additional size added to the trailing edge of each spatial dimension.
    /// Used to disambiguate output size when stride > 1.
    std::array<int64_t, 2> output_padding = {0, 0};

    /// Number of groups for grouped / depthwise transpose convolution.
    /// Input channels and output channels are each divided into groups,
    /// and the convolution is performed independently within each group.
    int64_t groups = 1;

    /// Post-processing applied during output write-back (default: identity).
    Epilogue epilogue{};

    /// If true, add result to existing output buffer instead of overwriting.
    bool add_to = false;
};

/// TransposeConv2D operator (class-based API).
///
/// Supports NCHWC8 layout with grouped and dilated transpose convolution.
/// Weight is accepted in planar NCHW format and accessed directly.
class TransposeConv2D : public OpBase {
public:
    /// Create a TransposeConv2D operator for the specified backend.
    static std::unique_ptr<TransposeConv2D> create(
        const TransposeConv2DAttributes& attrs,
        Backend backend = Backend::CPU);

    /// Create a TransposeConv2D operator with default attributes.
    static std::unique_ptr<TransposeConv2D> create(Backend backend = Backend::CPU) {
        return create(TransposeConv2DAttributes{}, backend);
    }

    // ---- OpBase interface ----
    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    /// inputs[0] = input tensor  [N, IC, IH, IW] (NCHWC8)
    /// inputs[1] = weight tensor [IC, OC/G, KH, KW] (planar NCHW)
    /// inputs[2] = bias tensor   [OC] (optional, planar)
    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::TransposeConv2D; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PackedOnly; }

    /// Access the transpose convolution attributes.
    const TransposeConv2DAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in transpose_conv2d.cpp (Pimpl pattern)

private:
    TransposeConv2D(const TransposeConv2DAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    TransposeConv2DAttributes attrs_;
    Backend backend_;
};

// ============================================================
// Functional API (convenience wrappers)
// ============================================================

/// Functional transpose_conv2d without bias.
void transpose_conv2d(const TensorView& input,
                      const TensorView& weight,
                      TensorView& output,
                      const TransposeConv2DAttributes& attrs,
                      const ComputeContext& ctx = {},
                      void* workspace = nullptr);

/// Functional transpose_conv2d with bias.
void transpose_conv2d(const TensorView& input,
                      const TensorView& weight,
                      const TensorView& bias,
                      TensorView& output,
                      const TransposeConv2DAttributes& attrs,
                      const ComputeContext& ctx = {},
                      void* workspace = nullptr);

}  // namespace nnops
