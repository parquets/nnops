#pragma once
/// @file conv2d.hpp
/// @brief Conv2D operator — 2D convolution with NCHW layout.

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

/// Attributes describing a 2D convolution.
struct Conv2DAttributes {
    /// Kernel size along height and width.
    std::array<int64_t, 2> kernel_size = {1, 1};

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

    /// Post-processing applied during output write-back (default: identity).
    Epilogue epilogue{};

    /// If true, add result to existing output buffer instead of overwriting.
    /// Enables residual connections without a separate add kernel:
    ///   output += Conv(input)  rather than  output = Conv(input)
    bool add_to = false;
};

/// Conv2D operator (class-based API).
///
/// Supports NCHW layout with grouped and dilated convolution.
/// Input: [N, IC, IH, IW], Weight: [OC, IC/G, KH, KW], Output: [N, OC, OH, OW].
/// Optional bias: [OC].
class Conv2D : public OpBase {
public:
    static std::unique_ptr<Conv2D> create(const Conv2DAttributes& attrs,
                                          Backend backend = Backend::CPU);

    static std::unique_ptr<Conv2D> create(Backend backend = Backend::CPU) {
        return create(Conv2DAttributes{}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    /// Always 0: the CPU tiled im2col + GEMM kernel pools its scratch
    /// internally.
    size_t getWorkspaceSize(std::span<const TensorDesc> inputs,
                            std::span<const TensorDesc> outputs) const override;

    /// inputs[0] = input tensor (NCHW)
    /// inputs[1] = weight tensor [OC, IC/G, KH, KW] — ONNX-style, also when grouped
    /// inputs[2] = bias tensor [OC] (optional)
    using OpBase::compute;

    void compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::Conv2D; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override { return LayoutSupport::PlanarOnly; }

    const Conv2DAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in conv2d.cpp (Pimpl pattern)
    ~Conv2D();

private:
    Conv2D(const Conv2DAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    Conv2DAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
