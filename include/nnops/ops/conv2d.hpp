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

/// Convolution algorithm selection for the CPU backend.
///
/// ImplicitGemm is the general tiled im2col + GEMM path and handles any
/// kernel size / stride / dilation / groups. Winograd is the F(2×2, 3×3)
/// minimal-filtering fast path: it is only valid for k3s1 (dilation 1) and
/// requires the caller to run prepackWeights first (compute then takes the
/// transformed weight U as inputs[1], not the raw weight).
///
/// Auto follows the weight it is handed — U means Winograd, a raw filter means
/// im2col — so a caller that predates the enum, or that simply prepacks when
/// winograd_applicable() says so, needs no other attribute. What Auto does not
/// do is decide the prepack: that stays the caller's call, because it is the
/// one that allocates U.
enum class ConvAlgorithm : uint8_t {
    ImplicitGemm = 0,  ///< Tiled im2col + GEMM (supports any geometry)
    Winograd,          ///< F(2×2, 3×3) Winograd, k3s1 only, requires prepacked U
    Auto,              ///< Winograd for a prepacked U, im2col otherwise (default)
};

/// True for the quantized Conv2D route: an s8 weight (per-channel QuantParams)
/// against a u8/s8/f16/f32 activation (per-tensor QuantParams), with an fp or
/// integer (s8/u8) output.
///
/// Analogous to `is_int8_matmul` / `is_fp_quant_matmul`, this lives here so the
/// operator and the CPU backend agree on which dtype combination is the
/// quantized one (the backend dispatches on exactly this predicate).
inline constexpr bool is_quantized_conv2d(DataType weight_dtype) noexcept {
    return weight_dtype == DataType::s8;
}

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
    /// On an s8/u8 destination (the quantized route) only Relu is applied — on
    /// the int32 accumulate, before the requant; the clip range and the other
    /// activations belong to the fp epilogue and are ignored there.
    Epilogue epilogue{};

    /// If true, add result to existing output buffer instead of overwriting.
    /// Enables residual connections without a separate add kernel:
    ///   output += Conv(input)  rather than  output = Conv(input)
    bool add_to = false;

    // ---- quantized route (is_quantized_conv2d(weight): s8 weights) ----------
    //
    // The two attributes mirror MatMul's: same names, same meaning. They are
    // ignored on the floating-point route.

    /// Output dtype of the quantized route.
    ///
    ///   f32 (default) / f16: dequantize the int32 accumulate. With an fp
    ///                        activation (f32/f16) the output follows the
    ///                        activation's dtype and the value is ignored; with
    ///                        a pre-quantized one (u8/s8) it is honoured.
    ///   s8 / u8:             requantize the accumulate with the output's
    ///                        QuantParams (scale s_w·s_x/out_scale, round half
    ///                        away from zero, clamped to the storage range; the
    ///                        product is f32, as in ONNX Runtime's requantize).
    ///                        Honoured for every activation dtype; the bias is
    ///                        then int32 rather than the output dtype.
    ///
    /// Any other value is refused by the kernel plan.
    DataType output_dtype = DataType::f32;

    /// Integer storage the fp activation (f32/f16) is quantized to inside the
    /// kernel, per im2col tile: s8 (default) or u8. Selects the clamp range and
    /// the zero-point convention. Only read when the activation is f32/f16;
    /// pre-quantized activations keep their storage type.
    DataType act_quant_dtype = DataType::s8;

    /// Kernel-selection hint for the CPU backend.
    ///
    /// Winograd is only honoured for k3s1 (KH == KW == 3, stride 1, dilation 1)
    /// with a raw f32/f16 weight, and it requires the transformed weight U —
    /// produced by prepackWeights — as inputs[1] of compute(). Shape inference
    /// is unaffected: getOutputTensorDesc still reads the *raw* weight desc, so
    /// the caller passes the raw weight for shape inference and U for compute.
    ConvAlgorithm algorithm = ConvAlgorithm::Auto;
};

/// The Winograd selection policy — what a caller checks before prepackWeights,
/// which is what actually turns the route on (compute() only follows the weight
/// it is given, since im2col cannot read U).
///
/// All four have to hold:
///
///   * The caller allows it: attrs.algorithm is Auto or Winograd. ImplicitGemm
///     is the off switch — it keeps the prepack from happening at all.
///   * k3s1 — the kernel implements F(2×2, 3×3) only (dilation 1 included).
///   * At least 32 input channels per group. The input/output transforms cost a
///     fixed amount per tile per channel, so the 2.25x multiply saving needs a
///     wide K to pay for them: a wash at 8 channels, ~1.2x at 32, 1.5-1.6x at
///     128-256.
///   * At least 14x14 of input. Below that the kernel's 8-lane tile packing
///     inflates the GEMM's N more than the multiply saving is worth — a 7x7
///     output packs its 4 real tile columns to 8 and runs 0.65x the speed of
///     im2col, while 14x14 packs 7 to 8 (14% waste) and keeps a 1.15x lead.
inline bool winograd_applicable(const Conv2DAttributes& attrs,
                                int64_t ic_per_group, int64_t ih, int64_t iw)
{
    return attrs.algorithm != ConvAlgorithm::ImplicitGemm &&
           attrs.kernel_size[0] == 3 && attrs.kernel_size[1] == 3 &&
           attrs.stride[0] == 1 && attrs.stride[1] == 1 &&
           attrs.dilation[0] == 1 && attrs.dilation[1] == 1 &&
           ic_per_group >= 32 && ih >= 14 && iw >= 14;
}

/// Conv2D operator (class-based API).
///
/// Supports NCHW layout with grouped and dilated convolution.
/// Input: [N, IC, IH, IW], Weight: [OC, IC/G, KH, KW], Output: [N, OC, OH, OW].
/// Optional bias: [OC].
///
/// Data types: f32/f16 on every input and output (the floating-point route), or
/// the quantized route — an s8 weight with per-channel QuantParams against a
/// u8/s8/f16/f32 activation with per-tensor QuantParams, producing fp or s8/u8
/// output (see Conv2DAttributes::output_dtype). See is_quantized_conv2d().
///
/// The bias dtype follows the output: [OC] in the output's fp dtype on an fp
/// destination, [OC] int32 on an s8/u8 one (the raw int32 accumulate, added
/// before the activation and never scaled by anything).
class Conv2D : public OpBase {
public:
    static std::unique_ptr<Conv2D> create(const Conv2DAttributes& attrs,
                                          Backend backend = Backend::CPU);

    static std::unique_ptr<Conv2D> create(Backend backend = Backend::CPU) {
        return create(Conv2DAttributes{}, backend);
    }

    std::vector<TensorDesc> getOutputTensorDesc(
        std::span<const TensorDesc> inputs) const override;

    Status validate(std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs) const override;

    /// Always 0: the CPU tiled im2col + GEMM kernel pools its scratch
    /// internally.
    size_t getWorkspaceSize(std::span<const TensorDesc> inputs,
                            std::span<const TensorDesc> outputs) const override;

    /// inputs[0] = input tensor (NCHW, or NHWC for a channel-last activation)
    /// inputs[1] = weight tensor [OC, IC/G, KH, KW] — ONNX-style, also when
    ///             grouped. Stored OIHW (planar) for a planar activation and
    ///             HWIO (TensorFlow's [KH, KW, IC/G, OC]) for an NHWC one — in
    ///             that order the tensor's *shape* is the filter itself; the two
    ///             must pair, see validate(). When attrs.algorithm == Winograd this is instead
    ///             the prepacked transform-domain weight U [16, OC, IC/G]
    ///             returned by prepackWeights.
    /// inputs[2] = bias tensor [OC] (optional)
    using OpBase::compute;

    Status compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    /// Winograd weight transform (U = G·g·Gᵀ) — two-step contract like
    /// DepthwiseConv:
    ///   Step 1 — query: outputs[0] empty. Fills shape [16, OC, IC/G], the
    ///            weight's dtype, layout PackedWinoWeight, and an explicit
    ///            pitch of IC/G × elem_size.
    ///   Step 2 — pack: outputs[0] pre-allocated to the queried size. Writes U.
    ///
    /// Requires attrs.algorithm == Winograd and k3s1 (KH == KW == 3, stride 1,
    /// dilation 1); asserts otherwise, since a later compute() receives U and
    /// cannot fall back to the raw-weight im2col path.
    void prepackWeights(std::span<const TensorView> inputs,
                        std::span<TensorView> outputs,
                        const ComputeContext& ctx = {}) override;

    OpType  getOpType()  const override { return OpType::Conv2D; }
    Backend getBackend() const override { return backend_; }
    LayoutSupport getLayoutSupport() const noexcept override
    {
        return LayoutSupport::PlanarOrChannelLast;
    }

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
