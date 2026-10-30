/// @file conv2d.cpp
/// @brief Conv2D operator dispatch — connects class API to backend implementations.

#include "nnops/ops/conv2d.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "backend/cpu/conv/conv2d_implicit_gemm.h"
#include "backend/cpu/conv/conv2d_winograd.h"
#include "backend/cpu/simd_kernel/simd_winograd.hpp"

#include <cstring>

namespace nnops {

struct Conv2D::Impl {
    using KernelFn = Status (*)(const Conv2DAttributes&,
                                TensorView&,
                                std::span<const TensorView>,
                                const ComputeContext&,
                                void*);
    KernelFn kernel_fn = nullptr;
};
Conv2D::~Conv2D() = default;


namespace {
auto resolve_conv2d_kernel(Backend backend, ConvAlgorithm algorithm)
    -> Conv2D::Impl::KernelFn
{
    if (algorithm == ConvAlgorithm::Winograd && backend == Backend::CPU) {
        return backend::cpu::conv2d_winograd_kernel;
    }
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::conv2d_implicit_gemm_kernel;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        return nullptr;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return nullptr;
#endif
    default:
        return nullptr;
    }
}

/// Pack the raw [OC, IC/G, 3, 3] weight into U[16][OC][IC/G] = G g Gᵀ.
/// @p ld_u / @p ld_c are the row / column strides of one 4×4 transform block
/// inside the U buffer (p = i·4 + j maps to p·OC·ICG).
template <class T>
void wino_pack_weight(const TensorView& weight, TensorView& u_out,
                      int64_t OC, int64_t ICG, int64_t ld_u, int64_t ld_c)
{
    const T* g = weight.ptr<T>();
    T* u = u_out.ptr<T>();
    for (int64_t oc = 0; oc < OC; ++oc) {
        for (int64_t ic = 0; ic < ICG; ++ic) {
            nnops::kernel::winograd_weight_transform(
                g + (oc * ICG + ic) * 9, 3, u + oc * ICG + ic,
                static_cast<int>(ld_u), static_cast<int>(ld_c));
        }
    }
}

/// Pack an HWIO weight ([KH][KW][IC/G][OC], OC contiguous) into the channel-last
/// route's B panels — the transposed product's operand, not the planar U.
///
/// The transform is the same G g Gᵀ: a 3×3 is gathered per (ic, oc) and written
/// into [16][kb][g·jb + j][ldd], each panel [kc][nr] k-major, so the MMA can read
/// it in place and no pack is ever done per call. The oc axis is padded to whole
/// panels, so the buffer must arrive zeroed — the caller does that.
///
/// Scalar on purpose: prepack runs once per (ic, oc) at load time, and a
/// vectorised gather over a transposed filter would only pay for itself there.
template <class T>
void wino_pack_weight_hwio(const TensorView& weight, TensorView& u_out,
                           const backend::cpu::WinoChannelLastU& ug,
                           int64_t ICG, int64_t OCG, int64_t G)
{
    const T* g = weight.ptr<T>();
    T* u = u_out.ptr<T>();
    // Strides come from the dims, not the pitch: a weight is a load-time
    // constant, so a padded one is not a case this route has to absorb.
    const int64_t OC  = OCG * G;
    const int64_t tap = ICG * OC;   // one filter position ([kh][kw]) of all groups
    for (int64_t gg = 0; gg < G; ++gg) {
        for (int64_t n = 0; n < OCG; ++n) {
            const int64_t j   = gg * ug.jb + n / ug.nr;
            const int64_t col = n - (n / ug.nr) * ug.nr;
            for (int64_t ic = 0; ic < ICG; ++ic) {
                const int64_t kb = ic / ug.kc;
                const int64_t k  = ic - kb * ug.kc;
                const T* src = g + ic * OC + gg * OCG + n;
                T gt[9];
                for (int t = 0; t < 9; ++t) { gt[t] = src[t * tap]; }
                T uu[16];
                nnops::kernel::winograd_weight_transform(gt, 3, uu, 4, 1);
                T* dst = u + (kb * ug.jbg + j) * ug.ldd + k * ug.nr + col;
                for (int p = 0; p < 16; ++p) { dst[p * ug.plane] = uu[p]; }
            }
        }
    }
}
}  // anonymous namespace

NNOPS_OP_CREATE(Conv2D, Conv2DAttributes)

Conv2D::Conv2D(const Conv2DAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_conv2d_kernel(backend, attrs_.algorithm);
}

std::vector<TensorDesc> Conv2D::getOutputTensorDesc(
    std::span<const TensorDesc> inputs) const
{
    // The channel-last weight is TensorFlow's [KH, KW, IC/G, OC], so the output
    // channel count is its last axis rather than its first.
    const bool wt_hwio = inputs.size() > 1 &&
        is_channel_last_weight(inputs[1].layout);
    std::vector<TensorDesc> outs = conv2d_output_shape(
        attrs_.kernel_size, attrs_.stride, attrs_.dilation, attrs_.padding,
        static_cast<int>(attrs_.auto_pad), attrs_.groups, inputs, wt_hwio);

    // The quantized route writes either an integer destination (s8/u8, taken
    // from attributes.output_dtype) or an fp one — the fp activation's own
    // dtype, or output_dtype for a pre-quantized (u8/s8) activation. An
    // unsupported request is passed through untouched, so the kernel plan can
    // refuse it instead of silently running another route.
    if (inputs.size() > 1 && is_quantized_conv2d(inputs[1].dtype)) {
        const DataType act = inputs[0].dtype;
        const bool fp_act = (act == DataType::f32 || act == DataType::f16);
        const DataType odt = attrs_.output_dtype;
        const bool int_out = odt == DataType::s8 || odt == DataType::u8;
        outs[0].dtype = (int_out || !fp_act) ? odt : act;
    }
    return outs;
}

size_t Conv2D::getWorkspaceSize(std::span<const TensorDesc> inputs,
                                std::span<const TensorDesc> outputs) const
{
    (void)inputs;
    (void)outputs;
    // im2col scratch is pooled internally by the CPU kernel.
    return 0;
}

Status Conv2D::validate(std::span<const TensorDesc> inputs,
                        std::span<const TensorDesc> outputs) const
{
    // input, weight [, bias]
    NNOPS_TRY(detail::check_counts(inputs.size(), outputs.size(), 2, 3, 1));
    NNOPS_TRY(detail::require(
        is_layout_supported(inputs[0].layout, LayoutSupport::PlanarOrChannelLast)));
    NNOPS_TRY(detail::require(
        is_layout_supported(inputs[1].layout, LayoutSupport::PlanarOrChannelLast)));
    // The weight must be stored in the form that pairs with the activation: the
    // channel-last kernel reads HWIO (the output channel is the run) or, once
    // prepacked, U as B panels; the planar one OIHW or the planar U. Converting
    // is the caller's job, once per model, at load time — a weight reshaped per
    // call would pay that copy on every inference.
    //
    // The im2col route takes a raw filter *or* its prepacked form on either
    // side (PackedHWIO pairs with a channel-last activation, PackedOIHW with a
    // planar one), so a graph that prepacks unconditionally still validates.
    const TensorLayout wl = inputs[1].layout;
    const bool weight_last = is_channel_last_weight(wl) ||
                             wl == TensorLayout::PackedWinoWeightLast ||
                             wl == TensorLayout::PackedHWIO;
    return detail::require(is_channel_last(inputs[0].layout) == weight_last);
}

Status Conv2D::compute(std::span<TensorView> outputs,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace)
{
    NNOPS_TRY(detail::check_views(outputs, inputs, 2, 3, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    auto& output = outputs[0];

    // auto_pad lives in the descriptor; kernels only see explicit leading padding.
    Conv2DAttributes attrs = attrs_;
    if (attrs.auto_pad != Conv2DAttributes::AutoPad::NOTSET) {
        const int ap = static_cast<int>(attrs.auto_pad);
        attrs.padding[0] = resolve_leading_padding(
            inputs[0].shape(2), attrs.kernel_size[0], attrs.stride[0],
            attrs.dilation[0], attrs.padding[0], ap);
        attrs.padding[1] = resolve_leading_padding(
            inputs[0].shape(3), attrs.kernel_size[1], attrs.stride[1],
            attrs.dilation[1], attrs.padding[1], ap);
        attrs.auto_pad = Conv2DAttributes::AutoPad::NOTSET;
    }

    // Three cases always run the implicit-GEMM kernel, whatever the algorithm
    // says: a raw channel-last filter (the kernel then keeps the pixel axis on M
    // and reads HWIO), an s8 weight (the quantized driver), and a prepacked
    // im2col filter (PackedOIHW/PackedHWIO holds that kernel's own panels — U is
    // the only other prepacked form, and each U form pairs with its own
    // activation layout).
    const bool wino_u = is_wino_weight(inputs[1].layout());
    if (backend_ == Backend::CPU &&
        (is_quantized_conv2d(inputs[1].data_type()) ||
         is_packed_conv_weight(inputs[1].layout()) ||
         (is_channel_last(inputs[0].layout()) && !wino_u))) {
        return backend::cpu::conv2d_implicit_gemm_kernel(attrs, output, inputs, ctx,
                                                         workspace);
    }

    // Auto reads the weight it was handed: U can only be multiplied in the
    // transform domain, and a raw filter has no U to use.
    auto kernel_fn = impl_->kernel_fn;
    if (attrs.algorithm == ConvAlgorithm::Auto) {
        attrs.algorithm = wino_u ? ConvAlgorithm::Winograd
                                 : ConvAlgorithm::ImplicitGemm;
        kernel_fn = resolve_conv2d_kernel(backend_, attrs.algorithm);
    }

    if (!kernel_fn) { return Status::unsupported; }
    return kernel_fn(attrs, output, inputs, ctx, workspace);
}

void Conv2D::prepackWeights(std::span<const TensorView> inputs,
                            std::span<TensorView> outputs,
                            const ComputeContext& ctx)
{
    NNOPS_ASSERT(inputs.size() >= 1);
    NNOPS_ASSERT(outputs.size() >= 1);
    const auto& weight = inputs[0];
    NNOPS_ASSERT(weight.rank() == 4);
    NNOPS_ASSERT(is_layout_supported(weight.layout(), LayoutSupport::PlanarOrChannelLast));

    // Two products live behind this one call, and the algorithm picks: the
    // Winograd U, or the im2col route's packed filter (which both its fp and
    // int8 drivers read in place, instead of repacking per call or per task).
    //
    // Auto keeps producing U exactly where the Winograd transform exists — the
    // same k3s1 fp geometry its asserts demand — because that is what a caller
    // asking for Auto is prepared to feed back into compute(); every other
    // geometry, and every quantized filter, gets the im2col panels.
    const bool wino_product =
        attrs_.algorithm == ConvAlgorithm::Winograd ||
        (attrs_.algorithm == ConvAlgorithm::Auto &&
         !is_quantized_conv2d(weight.data_type()) &&
         attrs_.kernel_size[0] == 3 && attrs_.kernel_size[1] == 3 &&
         attrs_.stride[0] == 1 && attrs_.stride[1] == 1 &&
         attrs_.dilation[0] == 1 && attrs_.dilation[1] == 1);

    if (!wino_product) {
        // ---- the im2col route's packed filter: PackedOIHW / PackedHWIO ----
        // The descriptor is resolved from the attributes and the filter's own
        // dims, so the query needs no activation; the kernel later resolves the
        // identical geometry from the same function.
        const TensorDesc d =
            backend::cpu::conv_packed_weight_desc(attrs_, weight.desc());
        NNOPS_ASSERT(d.rank > 0 &&
                     "Conv2D::prepackWeights: this filter has no packed form");
        std::vector<int64_t> shape(d.dims.begin(), d.dims.begin() + d.rank);
        const int64_t esz = static_cast<int64_t>(data_type_size(d.dtype));

        if (outputs[0].is_empty()) {
            // The pitch is the row length in bytes: a packed filter is not
            // channel-packed, so the default row alignment would not describe it.
            outputs[0] = TensorView(shape, d.dtype, nullptr,
                                    d.row_stride_elems * esz, d.layout,
                                    weight.quant_params());
            return;
        }

        NNOPS_ASSERT(outputs[0].layout() == d.layout &&
                     outputs[0].rank() == d.rank &&
                     outputs[0].shape(0) == shape[0] &&
                     outputs[0].shape(1) == shape[1] &&
                     "Conv2D::prepackWeights: the packed buffer does not match "
                     "the geometry");
        backend::cpu::conv_packed_weight_pack(attrs_, weight, outputs[0], ctx);
        return;
    }

    // compute() receives U and cannot fall back to the raw-weight im2col path,
    // so every unsupported configuration has to fail here, at prepack time.
    NNOPS_ASSERT((attrs_.algorithm == ConvAlgorithm::Winograd ||
                  attrs_.algorithm == ConvAlgorithm::Auto) &&
                 "Conv2D::prepackWeights: attrs.algorithm must be Winograd or Auto");
    NNOPS_ASSERT(attrs_.kernel_size[0] == 3 && attrs_.kernel_size[1] == 3 &&
                 attrs_.stride[0] == 1 && attrs_.stride[1] == 1 &&
                 attrs_.dilation[0] == 1 && attrs_.dilation[1] == 1 &&
                 "Conv2D::prepackWeights: Winograd requires k3s1");

    const DataType dtype = weight.data_type();
    const int64_t elem_size = static_cast<int64_t>(data_type_size(dtype));

    // Two filter orders, two U forms: OIHW transforms into [16][OC][IC/G] for a
    // planar activation, HWIO (TensorFlow's order, OC contiguous) into the
    // channel-last route's B panels. Both hold the same G g Gᵀ — only the packing
    // differs, because the two routes run the same GEMM with its operands
    // swapped.
    if (is_channel_last_weight(weight.layout())) {
        // Weight: [KH, KW, IC/G, OC]
        const int64_t ICG = weight.shape(2);
        const int64_t OC  = weight.shape(3);
        const int64_t G   = attrs_.groups;
        NNOPS_ASSERT(G >= 1 && OC % G == 0 && ICG > 0);
        const int64_t OCG = OC / G;

        const backend::cpu::WinoChannelLastU ug =
            backend::cpu::wino_channel_last_u(ICG, OCG, G, dtype);
        NNOPS_ASSERT(ug.total > 0 && "Conv2D::prepackWeights: unsupported dtype");

        if (outputs[0].is_empty()) {
            // ---- Query: the panels [16][nb][G·jb][ldd]. The pitch is the
            // panel's own length — PackedWinoWeightLast is not channel-packed,
            // so the default 32 B row alignment would not be ldd × elem_size.
            const int64_t u_shape[] = {16, ug.nb, ug.jbg, ug.ldd};
            outputs[0] = TensorView(std::span<const int64_t>(u_shape, 4), dtype,
                                    nullptr, ug.ldd * elem_size,
                                    TensorLayout::PackedWinoWeightLast);
            return;
        }

        NNOPS_ASSERT(outputs[0].shape(0) == 16 && outputs[0].shape(1) == ug.nb &&
                     outputs[0].shape(2) == ug.jbg && outputs[0].shape(3) == ug.ldd &&
                     outputs[0].pitch() == ug.ldd * elem_size &&
                     "Conv2D::prepackWeights: U buffer does not match the plan");
        // The oc axis is padded to whole panels; those columns stay zero.
        std::memset(outputs[0].ptr<char>(), 0,
                    static_cast<size_t>(ug.total) * static_cast<size_t>(elem_size));
        if (dtype == DataType::f32) {
            wino_pack_weight_hwio<float>(weight, outputs[0], ug, ICG, OCG, G);
        } else {
            NNOPS_ASSERT(dtype == DataType::f16);
            wino_pack_weight_hwio<backend::cpu::half>(weight, outputs[0], ug, ICG, OCG, G);
        }
        return;
    }

    // Weight: [OC, IC/G, 3, 3]
    const int64_t OC  = weight.shape(0);
    const int64_t ICG = weight.shape(1);

    if (outputs[0].is_empty()) {
        // ---- Query: U is [16, OC, IC/G] — one 4×4 transform-domain kernel per
        // position. The pitch is explicit: PackedWinoWeight is not
        // channel-packed, so TensorDesc::row_pitch() would align the row to
        // 32 B, which is not ICG × elem_size in general.
        const int64_t u_shape[] = {16, OC, ICG};
        const int64_t u_pitch = ICG * elem_size;
        outputs[0] = TensorView(std::span<const int64_t>(u_shape, 3), dtype,
                                nullptr, u_pitch, TensorLayout::PackedWinoWeight);
        return;
    }

    // ---- Pack: U[p][oc][ic] = (G g Gᵀ)[p], row stride between p planes.
    NNOPS_ASSERT(outputs[0].shape(0) == 16 && outputs[0].shape(1) == OC &&
                 outputs[0].shape(2) == ICG);
    const int64_t plane = OC * ICG;  // elements between transform positions
    const int64_t ld_c  = plane;     // column (j) step of the 4×4 block
    const int64_t ld_u  = 4 * plane; // row (i) step: p = i·4 + j

    if (dtype == DataType::f32) {
        wino_pack_weight<float>(weight, outputs[0], OC, ICG, ld_u, ld_c);
    } else {
        NNOPS_ASSERT(dtype == DataType::f16);
        wino_pack_weight<backend::cpu::half>(weight, outputs[0], OC, ICG, ld_u, ld_c);
    }
}

}  // namespace nnops
