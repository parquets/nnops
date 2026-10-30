#pragma once
/// @file conv2d.hpp
/// @brief 2D convolution — planar NCHW or channel-last NHWC.
///
/// Planar: input [N, IC, IH, IW], weight [OC, IC/G, KH, KW] (OIHW), output
/// [N, OC, OH, OW], bias [OC]. Channel-last: the same activation with the weight
/// as HWIO [KH, KW, IC/G, OC] — there the tensor's *shape* is the filter itself;
/// the two must pair (see validate()). Both routes support groups and dilation.
///
/// Data types: f32/f16 on every input and output (the floating-point route), or
/// the quantized route — an s8 weight with per-channel QuantParams against a
/// u8/s8/f16/f32 activation with per-tensor QuantParams, producing fp or s8/u8
/// output (see output_dtype and is_quantized_conv2d()). The bias dtype follows
/// the output: [OC] in the output's fp dtype on an fp destination, [OC] int32
/// (the raw accumulate, added before the activation, never scaled) on s8/u8.
///
/// Algorithms: ImplicitGemm is the general tiled im2col + GEMM path and handles
/// any geometry. Winograd is the F(2x2, 3x3) fast path, valid only for k3s1
/// (dilation 1) and only when compute() is handed the transform-domain U from
/// prepackWeights, not the raw filter. Auto follows the weight it is handed
/// (U → Winograd, raw filter → im2col), so a caller that predates the enum needs
/// no attribute; the prepack stays the caller's decision because it allocates U.
/// Shape inference is unaffected: getOutputTensorDesc reads the raw filter, so
/// the caller passes the raw weight there and U only to compute().
///
/// winograd_applicable() is what a caller checks before prepacking. It requires
/// algorithm != ImplicitGemm (the off switch), k3s1, at least 32 input channels
/// per group, and at least 14x14 input. The transforms cost a fixed amount per
/// tile per channel, so the 2.25x multiply saving needs wide K to pay for them
/// (a wash at 8 channels, ~1.2x at 32, 1.5-1.6x at 128-256). Below 14x14 the
/// planar kernel's 8-lane tile packing inflates the GEMM's N more than the
/// saving is worth: a 7x7 output packs its 4 real tile columns to 8 and runs
/// 0.65x im2col, while 14x14 packs 7 to 8 (14% waste) and keeps a 1.15x lead.
/// The channel-last route has none of those costs — no deinterleaving load, no
/// strided column store, no 8-lane packing (its tiles are flattened, so a 7x7
/// plane's 16 tiles are exactly 16) — and beats the channel-last implicit GEMM
/// at every IC >= 32 shape from 7x7 to 224x224. The shared floor stays 14x14:
/// one signature can only be as permissive as the route that constrains it, and
/// this costs the channel-last route nothing.

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

/// Convolution algorithm selection; Auto follows the weight handed to compute().
enum class ConvAlgorithm : uint8_t {
    ImplicitGemm = 0,  ///< Tiled im2col + GEMM (any geometry)
    Winograd,          ///< F(2×2, 3×3), k3s1 only, requires prepacked U
    Auto,              ///< Winograd for a prepacked U, im2col otherwise (default)
};

/// True when the weight dtype selects the quantized route; the CPU backend
/// dispatches on this predicate.
inline constexpr bool is_quantized_conv2d(DataType weight_dtype) noexcept {
    return weight_dtype == DataType::s8;
}

/// Attributes describing a 2D convolution.
struct Conv2DAttributes {
    std::array<int64_t, 2> kernel_size = {1, 1};
    std::array<int64_t, 2> stride = {1, 1};
    std::array<int64_t, 2> dilation = {1, 1};
    std::array<int64_t, 2> padding = {0, 0};
    int64_t groups = 1;

    enum class AutoPad : uint8_t {
        NOTSET = 0,
        SAME_UPPER,
        SAME_LOWER,
        VALID,
    };
    AutoPad auto_pad = AutoPad::NOTSET;

    /// Write-back post-processing. On an s8/u8 destination only Relu applies,
    /// to the int32 accumulate before the requant; the clip range and the other
    /// activations are fp-epilogue only and are ignored there.
    Epilogue epilogue{};

    /// Accumulate into the existing output instead of overwriting (residual
    /// connections without a separate add kernel).
    bool add_to = false;

    // ---- quantized route (is_quantized_conv2d(weight): s8 weights) ----------
    // Mirror MatMul's two attributes: same names, same meaning; ignored on the
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
    /// Any other value is refused by the kernel plan.
    DataType output_dtype = DataType::f32;

    /// Integer storage an fp activation is quantized to inside the kernel, per
    /// im2col tile: s8 (default) or u8, selecting the clamp range and zero-point
    /// convention. Only read when the activation is f32/f16.
    DataType act_quant_dtype = DataType::s8;

    ConvAlgorithm algorithm = ConvAlgorithm::Auto;
};

/// Policy a caller checks before prepackWeights(); see the file header.
inline bool winograd_applicable(const Conv2DAttributes& attrs,
                                int64_t ic_per_group, int64_t ih, int64_t iw)
{
    return attrs.algorithm != ConvAlgorithm::ImplicitGemm &&
           attrs.kernel_size[0] == 3 && attrs.kernel_size[1] == 3 &&
           attrs.stride[0] == 1 && attrs.stride[1] == 1 &&
           attrs.dilation[0] == 1 && attrs.dilation[1] == 1 &&
           ic_per_group >= 32 && ih >= 14 && iw >= 14;
}

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

    /// Always 0 — the CPU kernels pool their scratch internally.
    size_t getWorkspaceSize(std::span<const TensorDesc> inputs,
                            std::span<const TensorDesc> outputs) const override;

    /// inputs: [0] input, [1] weight (raw filter, prepacked OIHW/HWIO, or Winograd U), [2] optional bias [OC].
    using OpBase::compute;

    Status compute(std::span<TensorView> outputs,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx = {},
                 void* workspace = nullptr) override;

    /// Winograd U = G·g·Gᵀ, two-step: an empty outputs[0] queries the packed
    /// descriptor — planar [16, OC, IC/G]/PackedWinoWeight (pitch IC/G × elem_size),
    /// channel-last [16, nb, G·jb, ldd]/PackedWinoWeightLast (pitch ldd × elem_size)
    /// — and a preallocated one receives U. Requires algorithm == Winograd and
    /// k3s1, asserted: a later compute() gets U and cannot fall back to im2col.
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
