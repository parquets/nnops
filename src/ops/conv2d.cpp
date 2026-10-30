/// @file conv2d.cpp
/// @brief Conv2D operator dispatch — descriptor checks, kernel selection, and
///        the two prepacked weight products.
///
/// validate() makes the filter order pair with the activation: a channel-last
/// activation takes HWIO, PackedHWIO or PackedWinoWeightLast, a planar one any
/// other tag (a raw filter, PackedOIHW, or PackedWinoWeight). compute() forces
/// the implicit-GEMM kernel for three cases whatever the algorithm says — a
/// quantized (s8) filter, a prepacked im2col filter, and a raw channel-last
/// filter — and under Auto picks Winograd only when handed U, since a raw filter
/// has no U to multiply in the transform domain.
///
/// prepackWeights() produces one of two things: the Winograd U — [16][OC][IC/G]
/// for a planar activation, [16][nb][G·jb][ldd] B panels for a channel-last one —
/// when asked for Winograd (under Auto, only for a non-quantized k3s1 filter),
/// and the im2col route's PackedOIHW/PackedHWIO otherwise. Both are built once at
/// load time so compute() reads them in place instead of repacking per call.

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

/// Pack [OC, IC/G, 3, 3] into U[16][OC][IC/G] = G g Gᵀ; p = i·4 + j, with ld_u
/// and ld_c the block's row and column strides.
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

/// Pack HWIO ([KH][KW][IC/G][OC], OC contiguous) into the channel-last B panels:
/// the same G g Gᵀ, one 3×3 gather per (ic, oc), written [16][kb][g·jb + j][ldd]
/// k-major. The oc padding must arrive zeroed. Scalar because it runs once.
template <class T>
void wino_pack_weight_hwio(const TensorView& weight, TensorView& u_out,
                           const backend::cpu::WinoChannelLastU& ug,
                           int64_t ICG, int64_t OCG, int64_t G)
{
    const T* g = weight.ptr<T>();
    T* u = u_out.ptr<T>();
    // Strides come from the dims, not the pitch: a weight is a load-time constant.
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
    // HWIO's last axis is OC, so the channel count is the weight's last dim.
    const bool wt_hwio = inputs.size() > 1 &&
        is_channel_last_weight(inputs[1].layout);
    std::vector<TensorDesc> outs = conv2d_output_shape(
        attrs_.kernel_size, attrs_.stride, attrs_.dilation, attrs_.padding,
        static_cast<int>(attrs_.auto_pad), attrs_.groups, inputs, wt_hwio);

    // The quantized route writes s8/u8 when output_dtype says so, else the fp
    // activation's dtype (output_dtype for a pre-quantized one); an unsupported
    // request passes through for the kernel plan to refuse.
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
    // A raw filter or its prepacked form works on either side (PackedHWIO with a
    // channel-last activation, PackedOIHW with a planar one), so a graph that
    // prepacks unconditionally still validates.
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

    // A raw channel-last filter keeps the pixel axis on M; U is the only other
    // prepacked form, each pairing with its own activation layout.
    const bool wino_u = is_wino_weight(inputs[1].layout());
    if (backend_ == Backend::CPU &&
        (is_quantized_conv2d(inputs[1].data_type()) ||
         is_packed_conv_weight(inputs[1].layout()) ||
         (is_channel_last(inputs[0].layout()) && !wino_u))) {
        return backend::cpu::conv2d_implicit_gemm_kernel(attrs, output, inputs, ctx,
                                                         workspace);
    }

    // Auto reads the weight it was handed: a raw filter has no U to use.
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

    // Two products live behind this one call. Auto keeps producing U exactly
    // where the transform exists — the k3s1 fp geometry its asserts demand —
    // because that is what an Auto caller will feed back into compute().
    const bool wino_product =
        attrs_.algorithm == ConvAlgorithm::Winograd ||
        (attrs_.algorithm == ConvAlgorithm::Auto &&
         !is_quantized_conv2d(weight.data_type()) &&
         attrs_.kernel_size[0] == 3 && attrs_.kernel_size[1] == 3 &&
         attrs_.stride[0] == 1 && attrs_.stride[1] == 1 &&
         attrs_.dilation[0] == 1 && attrs_.dilation[1] == 1);

    if (!wino_product) {
        // The descriptor comes from the attributes and the filter's own dims, so
        // the query needs no activation.
        const TensorDesc d =
            backend::cpu::conv_packed_weight_desc(attrs_, weight.desc());
        NNOPS_ASSERT(d.rank > 0 &&
                     "Conv2D::prepackWeights: this filter has no packed form");
        std::vector<int64_t> shape(d.dims.begin(), d.dims.begin() + d.rank);
        const int64_t esz = static_cast<int64_t>(data_type_size(d.dtype));

        if (outputs[0].is_empty()) {
            // The pitch is the row length in bytes: a packed filter is not
            // channel-packed, so the default alignment would not describe it.
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
    // so every unsupported configuration must fail here, at prepack time.
    NNOPS_ASSERT((attrs_.algorithm == ConvAlgorithm::Winograd ||
                  attrs_.algorithm == ConvAlgorithm::Auto) &&
                 "Conv2D::prepackWeights: attrs.algorithm must be Winograd or Auto");
    NNOPS_ASSERT(attrs_.kernel_size[0] == 3 && attrs_.kernel_size[1] == 3 &&
                 attrs_.stride[0] == 1 && attrs_.stride[1] == 1 &&
                 attrs_.dilation[0] == 1 && attrs_.dilation[1] == 1 &&
                 "Conv2D::prepackWeights: Winograd requires k3s1");

    const DataType dtype = weight.data_type();
    const int64_t elem_size = static_cast<int64_t>(data_type_size(dtype));

    // Both U forms hold the same G g Gᵀ — only the packing differs, because the
    // two routes run the same GEMM with its operands swapped.
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
            // Query: the panels [16][nb][G·jb][ldd], pitched at ldd — not
            // channel-packed, so the default 32 B row alignment would not fit.
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
        // Query: U is [16, OC, IC/G], pitched at ICG × elem_size — not
        // channel-packed, so row_pitch() would 32 B-align it instead.
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
