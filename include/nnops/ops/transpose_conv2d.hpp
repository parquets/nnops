#pragma once
/// @file transpose_conv2d.hpp
/// @brief TransposeConv2D — transposed 2D convolution (deconvolution).
///
/// The backward of a regular convolution. The image tensor is channel-first or
/// channel-last
/// (the output carries its layout); the weight is channel-first [IC, OC/G, KH, KW] in
/// both cases; bias [OC] optional. Reference: ncnn Deconvolution.
///
/// Data types: f32/f16 on every input and output (the floating-point route), or
/// the quantized route — an s8 weight with per-channel QuantParams against a
/// u8/s8/f16/f32 activation with per-tensor QuantParams, producing fp or s8/u8
/// output (see output_dtype and is_quantized_transpose_conv2d()). The bias dtype
/// follows the output: [OC] in the output's fp dtype on an fp destination, [OC]
/// int32 (the raw accumulate, added before the activation, never scaled) on
/// s8/u8. add_to is refused on the quantized route, and with it the
/// pre-activation order (residual_pre_activation).

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
inline constexpr bool is_quantized_transpose_conv2d(DataType weight_dtype) noexcept {
    return weight_dtype == DataType::s8;
}

/// Attributes describing a 2D transposed convolution.
struct TransposeConv2DAttributes {
    /// Kernel size [KH, KW].
    std::array<int64_t, 2> kernel_size = {1, 1};

    /// Stride [SH, SW].
    std::array<int64_t, 2> stride = {1, 1};

    /// Dilation [DH, DW].
    std::array<int64_t, 2> dilation = {1, 1};

    /// Symmetric zero-padding [PH, PW]; subtracted from the output (the
    /// opposite of regular conv).
    std::array<int64_t, 2> padding = {0, 0};

    /// Extra size on each spatial dimension's trailing edge, to disambiguate
    /// the output size when stride > 1.
    std::array<int64_t, 2> output_padding = {0, 0};

    /// Groups for grouped / depthwise transpose convolution.
    int64_t groups = 1;

    /// Post-processing applied during output write-back (default: identity).
    Epilogue epilogue{};

    /// If true, add into the existing output buffer instead of overwriting.
    bool add_to = false;

    /// Order of the add_to term relative to the epilogue:
    ///   false: y = epilogue(conv) + residual
    ///   true:  y = epilogue(conv + residual)
    /// Only meaningful with add_to. The fp CPU routes — every layout and every
    /// sub-route — and the reference implement it; the quantized route refuses
    /// add_to altogether, so it has no pre-activation form either.
    bool residual_pre_activation = false;

    // ---- quantized route (is_quantized_transpose_conv2d(weight): s8 weights) --
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
    /// Any other value is refused by the kernel.
    DataType output_dtype = DataType::f32;

    /// Integer storage an fp activation is quantized to inside the kernel: s8
    /// (default) or u8, selecting the clamp range and zero-point convention.
    /// Only read when the activation is f32/f16.
    DataType act_quant_dtype = DataType::s8;
};

class TransposeConv2D : public OpBase {
public:
    static std::unique_ptr<TransposeConv2D> create(
        const TransposeConv2DAttributes& attrs,
        Backend backend = Backend::CPU);

    static std::unique_ptr<TransposeConv2D> create(Backend backend = Backend::CPU) {
        return create(TransposeConv2DAttributes{}, backend);
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

    OpType  getOpType()  const override { return OpType::TransposeConv2D; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override {
        return LayoutSupport::ChannelFirstOrChannelLast;
    }

    const TransposeConv2DAttributes& attributes() const noexcept { return attrs_; }

    struct Impl;  // defined in transpose_conv2d.cpp
    ~TransposeConv2D();

private:
    TransposeConv2D(const TransposeConv2DAttributes& attrs, Backend backend);

    std::unique_ptr<Impl> impl_;
    TransposeConv2DAttributes attrs_;
    Backend backend_;
};

}  // namespace nnops
