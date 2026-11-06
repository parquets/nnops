#pragma once
/// @file depthwise_transpose_conv.hpp
/// @brief DepthwiseTransposeConv2D — per-channel transposed 2D convolution, the
///        backward of DepthwiseConv, on a channel-first or a channel-last
///        tensor.
///
/// One filter per channel and no cross-channel mixing, so the output has the
/// input's channel count. The input is [N, C, IH, IW]; the weight follows the
/// activation's family:
///
///   channel-first   [C, 1, KH, KW] — one tap block per channel, the same form
///                   DepthwiseConv reads, a scalar broadcast per tap;
///   channel-last    [KH][KW][C] (TensorLayout::HWCM) — TensorFlow's depthwise
///                   order with the multiplier dropped, so every tap is a run
///                   of C contiguous channels and the channel lane reads it as
///                   one vector.
///
/// Bias [C] is optional. The output is [N, C, OH, OW] with
/// OH = (IH - 1)·SH - 2·PH + DH·(KH - 1) + OP_h + 1, and likewise for W.
///
/// Data types: f32/f16 on every operand. A weight of s8 selects the quantized
/// route: per-channel weight QuantParams, per-tensor activation and output
/// parameters, an int32 accumulate, and either a dequantized f32/f16 or a
/// requantized s8/u8 destination (see output_dtype).
///
/// The gather form is used, so every output pixel is accumulated once and
/// add_to needs no save/restore pass: output (oh, ow) reads input (ih, iw)
/// where oh + PH - kh·DH == ih·SH, which pins each tap to one residue of the
/// output coordinate. The CPU driver hoists those residue facts out of the
/// pixel loop — the (kh, ih) row pairs for one output row, the (kw, shift)
/// column pairs for one residue class — and then runs a boundary-free vector
/// loop over the class steps, with the edges left to the scalar tap loop.

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

/// True when the weight dtype selects the quantized route; the CPU backend
/// dispatches on this predicate.
inline constexpr bool is_quantized_depthwise_transpose_conv2d(
    DataType weight_dtype) noexcept {
    return weight_dtype == DataType::s8;
}

/// Attributes describing a depthwise transposed 2D convolution.
struct DepthwiseTransposeConv2DAttributes {
    /// Kernel size [KH, KW].
    std::array<int64_t, 2> kernel_size = {1, 1};

    /// Stride [SH, SW].
    std::array<int64_t, 2> stride = {1, 1};

    /// Dilation [DH, DW].
    std::array<int64_t, 2> dilation = {1, 1};

    /// Symmetric zero-padding [PH, PW]; subtracted from the output, the
    /// opposite of a regular convolution.
    std::array<int64_t, 2> padding = {0, 0};

    /// Extra size on each spatial dimension's trailing edge, to disambiguate
    /// the output size when stride > 1.
    std::array<int64_t, 2> output_padding = {0, 0};

    /// Post-processing applied during output write-back (default: identity).
    Epilogue epilogue{};

    /// Accumulate into the existing output instead of overwriting (residual).
    bool add_to = false;

    // ---- quantized route (s8 weight, u8/s8/f16/f32 activation) --------------
    // Mirror Conv2D's two attributes: same names, same meaning; ignored on the
    // floating-point route.

    /// Quantized-route output dtype.
    ///   f32 (default) / f16: dequantize the int32 accumulate. With an fp
    ///                        activation the output follows the activation's
    ///                        dtype and this value is ignored; with a
    ///                        pre-quantized (u8/s8) activation it is honoured.
    ///   s8 / u8:             requantize with the output's QuantParams (scale
    ///                        s_w·s_x/out_scale, round half away from zero,
    ///                        clamped to the storage range; the product is f32).
    ///                        Honoured for every activation dtype; the bias is
    ///                        then int32 rather than the output dtype.
    DataType output_dtype = DataType::f32;

    /// Integer storage an fp activation is quantized to inside the kernel: s8
    /// (default) or u8, selecting the clamp range and zero-point convention.
    /// Only read when the activation is f32/f16.
    DataType act_quant_dtype = DataType::s8;
};

class DepthwiseTransposeConv2D : public OpBase {
public:
    static std::unique_ptr<DepthwiseTransposeConv2D> create(
        const DepthwiseTransposeConv2DAttributes& attrs,
        Backend backend = Backend::CPU);

    static std::unique_ptr<DepthwiseTransposeConv2D> create(Backend backend = Backend::CPU) {
        return create(DepthwiseTransposeConv2DAttributes{}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    Status validate(std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs) const override;

    using OpBase::compute;

    Status compute(std::span<TensorView> outputs,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx = {},
                   void* workspace = nullptr) override;

    OpType  getOpType()  const override { return OpType::DepthwiseTransposeConv2D; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::ChannelFirstOrChannelLast;
    }

    const DepthwiseTransposeConv2DAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;
    ~DepthwiseTransposeConv2D();

private:
    DepthwiseTransposeConv2D(const DepthwiseTransposeConv2DAttributes& attrs,
                             Backend backend);

    std::unique_ptr<Impl> impl_;
    DepthwiseTransposeConv2DAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
