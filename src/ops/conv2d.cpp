/// @file conv2d.cpp
/// @brief Conv2D operator dispatch — descriptor checks, kernel selection, and
///        the two prepacked weight products.
///
/// validate() makes the filter order pair with the activation: an NHWC
/// activation takes HWIO, PackedHWIO, PackedHWIOSme or PackedWinoWeightLast.
/// compute() forces
/// the implicit-GEMM kernel for three cases whatever the algorithm says — a
/// quantized (s8) filter, a prepacked im2col filter, and a raw filter — and
/// under Auto picks Winograd only when handed U, since a raw filter has no U to
/// multiply in the transform domain.
///
/// One grouped geometry is not a GEMM at all: a group of one input and one
/// output channel is a depthwise convolution, and compute() hands it to the
/// depthwise kernels. The filter is already in the order they read —
/// [KH, KW, 1, OC=G] — and indexes as tap·C + c, so nothing is converted. A
/// prepacked filter, or ConvAlgorithm::ImplicitGemm, keeps the grouped GEMM;
/// prepackWeights answers the same geometry with no packed form, so the two
/// cannot disagree.
///
/// prepackWeights() produces one of three things: the Winograd U —
/// [pos][nb][G·jb][ldd] B panels, pos being 16 (F(2,3)) or 36 (F(4,3)) — when
/// asked for Winograd (under Auto, only for a non-quantized k3s1 filter), the
/// im2col route's PackedHWIO (or PackedHWIOSme on an SME machine), and on CUDA
/// the int8 kernel's own PackedHWIOCuda. All are built once at load time so
/// compute() reads them in place instead of repacking per call. A blob belongs
/// to exactly one route: each backend refuses the others' forms.

#include "nnops/ops/conv2d.hpp"
#include "nnops/ops/depthwise_conv.hpp"
#include "nnops/detail/assert.hpp"
#include "dispatch.hpp"
#include "nnops/detail/shape_inference.hpp"
#include "backend/cpu/conv/conv2d_implicit_gemm.h"
#include "backend/cpu/conv/conv2d_winograd.h"
#include "backend/cpu/simd_kernel/simd_winograd.hpp"
#if defined(NNOPS_HAS_CUDA)
#include "backend/cuda/conv/conv2d_pack.h"
#endif

#include <algorithm>
#include <cstring>

namespace nnops {

NNOPS_DECLARE_CUDA(Conv2DAttributes, conv2d_cuda)
NNOPS_DECLARE_CPU(DepthwiseConvAttributes, depthwise_conv_cpu)
NNOPS_DECLARE_CUDA(DepthwiseConvAttributes, depthwise_conv_cuda)

struct Conv2D::Impl {
    using KernelFn = Status (*)(const Conv2DAttributes&, TensorView&, std::span<const TensorView>,
                                const ComputeContext&, void*);
    KernelFn kernel_fn = nullptr;
};
Conv2D::~Conv2D() = default;

namespace {

/// A group that is one input channel wide and one output channel wide makes the
/// whole convolution depthwise: groups == OC and IC/G == 1. The filter
/// [KH, KW, 1, OC] lays out as tap·C + c — the order the depthwise
/// kernels read, unwrapped. A packed blob is excluded on purpose: it is a
/// caller asking for the im2col route.
bool is_depthwise_geometry(const Conv2DAttributes& attrs, const TensorView& weight)
{
    if (attrs.groups <= 1 || !is_channel_last_weight(weight.layout())) {
        return false;
    }
    return weight.shape(2) == 1 && weight.shape(3) == attrs.groups;
}

bool is_fp(DataType dtype) noexcept
{
    return dtype == DataType::f32 || dtype == DataType::f16 || dtype == DataType::bf16;
}

/// The depthwise kernels walk the filter as a dense [KH][KW][C] run, so a padded
/// row is the one thing the hand-over cannot carry — it would read the pad as
/// channels. A pitched filter stays on the grouped route.
bool is_dense_run(const TensorView& w) noexcept
{
    const int64_t cols = w.shape(w.rank() - 1);
    return w.pitch() == static_cast<int64_t>(data_type_row_bytes(w.data_type(), cols));
}

auto resolve_conv2d_kernel(Backend backend, ConvAlgorithm algorithm) -> Conv2D::Impl::KernelFn
{
    if (algorithm == ConvAlgorithm::Winograd && backend == Backend::CPU) {
        return backend::cpu::conv2d_winograd_kernel;
    }
    switch (backend) {
    case Backend::CPU:
        return backend::cpu::conv2d_implicit_gemm_kernel;
#ifdef NNOPS_HAS_CUDA
    case Backend::CUDA:
        // The CUDA route is implicit GEMM only: a Winograd weight handed in
        // (U) has no CUDA consumer, so it is refused rather than mis-read.
        return algorithm == ConvAlgorithm::Winograd ? nullptr : backend::cuda::conv2d_cuda;
#endif
#ifdef NNOPS_HAS_VULKAN
    case Backend::Vulkan:
        return nullptr;
#endif
    default:
        return nullptr;
    }
}

/// Pack HWIO ([KH][KW][IC/G][OC], OC contiguous) into the B panels:
/// the same G g Gᵀ, one 3×3 gather per (ic, oc), written [pos][kb][g·jb + j][ldd]
/// k-major. The oc padding must arrive zeroed. Scalar per element because the
/// gather is a stride-tap walk, but one output channel is still one task: it
/// owns the (k, col) lane of every position, so the split needs no scratch.
template <class T>
void wino_pack_weight_hwio(const TensorView& weight, TensorView& u_out,
                           const backend::cpu::WinoU& ug, int64_t ICG, int64_t OCG,
                           int64_t G, const ComputeContext& ctx)
{
    const T* g = weight.ptr<T>();
    T* u = u_out.ptr<T>();
    // Strides come from the dims, not the pitch: a weight is a load-time constant.
    const int64_t OC = OCG * G;
    const int64_t tap = ICG * OC; // one filter position ([kh][kw]) of all groups
    const bool f23 = ug.order == backend::cpu::WinoOrder::F23;
    // The group and the in-group channel flatten to one index, so the task axis
    // is the filter's whole output-channel extent.
    ctx.cpu.run(0, OC, [&](int64_t o) {
        const int64_t gg = o / OCG;
        const int64_t n = o % OCG;
        const int64_t j = gg * ug.jb + n / ug.nr;
        const int64_t col = n - (n / ug.nr) * ug.nr;
        for (int64_t ic = 0; ic < ICG; ++ic) {
            const int64_t kb = ic / ug.kc;
            const int64_t k = ic - kb * ug.kc;
            const T* src = g + ic * OC + o;
            T gt[9];
            for (int t = 0; t < 9; ++t) {
                gt[t] = src[t * tap];
            }
            T uu[36];
            if (f23) {
                nnops::kernel::winograd_weight_transform(gt, 3, uu, 4, 1);
            } else {
                nnops::kernel::winograd43_weight_transform(gt, 3, uu, 6, 1);
            }
            // The ZA f16 panel interleaves a column's K pair: k and k+1 of one
            // column sit at col*2 and col*2+1 of the same K-pair row.
            int64_t lane = k * ug.nr + col;
            if (ug.za && std::is_same_v<T, backend::cpu::half>) {
                lane = (k / 2) * (2 * ug.nr) + col * 2 + (k & 1);
            }
            T* dst = u + (kb * ug.jbg + j) * ug.ldd + lane;
            for (int64_t p = 0; p < ug.pos; ++p) {
                dst[p * ug.plane] = uu[p];
            }
        }
    });
}
} // anonymous namespace

NNOPS_OP_CREATE(Conv2D, Conv2DAttributes)

Conv2D::Conv2D(const Conv2DAttributes& attrs, Backend backend)
    : impl_(std::make_unique<Impl>()), attrs_(attrs), backend_(backend)
{
    impl_->kernel_fn = resolve_conv2d_kernel(backend, attrs_.algorithm);
}

std::vector<TensorDesc> Conv2D::getOutputTensorDesc(std::span<const TensorDesc> inputs) const
{
    // HWIO's last axis is OC, so the channel count is the weight's last dim.
    const bool wt_hwio = inputs.size() > 1 && is_channel_last_weight(inputs[1].layout);
    std::vector<TensorDesc> outs =
        conv2d_output_shape(attrs_.kernel_size, attrs_.stride, attrs_.dilation, attrs_.padding,
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
    NNOPS_TRY(detail::require(is_channel_last(inputs[0].layout)));
    // The filter order must pair with the activation: HWIO, its prepacked form,
    // or the Winograd U.
    const TensorLayout wl = inputs[1].layout;
    NNOPS_TRY(detail::require(is_channel_last_weight(wl) ||
                              wl == TensorLayout::PackedWinoWeightLast ||
                              wl == TensorLayout::PackedHWIO ||
                              wl == TensorLayout::PackedHWIOCuda ||
                              wl == TensorLayout::PackedHWIOSme));

    // A quantized filter quantizes its activation per call and requantizes into
    // the same float, so a floating-point activation and output are one type:
    // there is no f16-in/f32-out (or f32-in/f16-out) route to fall back on.
    // getOutputTensorDesc already reports the activation's dtype for such a
    // call, so this refuses the view that asks for the other one instead of
    // converting it silently.
    if (is_quantized_conv2d(inputs[1].dtype) && !outputs.empty()) {
        const DataType act = inputs[0].dtype;
        const DataType odt = outputs[0].dtype;
        const bool fp_act = act == DataType::f32 || act == DataType::f16;
        const bool fp_out = odt == DataType::f32 || odt == DataType::f16;
        if (fp_act && fp_out && odt != act) {
            return Status::unsupported;
        }
    }

    // A pre-activation residual is implemented by every fp CPU route (both
    // layouts, the depthwise reroute and Winograd) and the reference. Refusing
    // the other routes beats silently applying the residual on the wrong side
    // of the activation.
    if (attrs_.add_to && attrs_.residual_pre_activation) {
        if (is_quantized_conv2d(inputs[1].dtype)) {
            return Status::unsupported;
        }
        if (backend_ == Backend::CUDA) {
            return Status::unsupported;
        }
    }
    return Status::ok;
}

Status Conv2D::compute(std::span<TensorView> outputs, std::span<const TensorView> inputs,
                       const ComputeContext& ctx, void* workspace)
{
    NNOPS_TRY(detail::check_views(backend_, outputs, inputs, 2, 3, 1));
    NNOPS_TRY(validate_views(outputs, inputs));
    auto& output = outputs[0];

    // auto_pad lives in the descriptor; kernels only see explicit leading padding.
    // A channel-last activation is [N, H, W, C]: H is axis 1, W axis 2.
    const auto& in = inputs[0];
    Conv2DAttributes attrs = attrs_;
    if (attrs.auto_pad != Conv2DAttributes::AutoPad::NOTSET) {
        const int ap = static_cast<int>(attrs.auto_pad);
        attrs.padding[0] =
            resolve_leading_padding(in.shape(1), attrs.kernel_size[0], attrs.stride[0],
                                    attrs.dilation[0], attrs.padding[0], ap);
        attrs.padding[1] =
            resolve_leading_padding(in.shape(2), attrs.kernel_size[1], attrs.stride[1],
                                    attrs.dilation[1], attrs.padding[1], ap);
        attrs.auto_pad = Conv2DAttributes::AutoPad::NOTSET;
    }

    // A depthwise geometry reads the filter Conv2D was handed, so it costs no
    // conversion in either layout. The grouped GEMM degenerates there — one
    // channel per group is a single-row product, re-dispatched per group — and
    // loses by roughly 3.8x on the MobileNetV2 shapes on the CPU, an order of
    // magnitude more on the GPU, where every group also pays its own block.
    if ((backend_ == Backend::CPU || backend_ == Backend::CUDA) &&
        attrs.algorithm != ConvAlgorithm::ImplicitGemm && inputs.size() >= 2 &&
        is_fp(inputs[0].data_type()) && is_fp(inputs[1].data_type()) &&
        is_depthwise_geometry(attrs, inputs[1]) && is_dense_run(inputs[1])) {
        DepthwiseConvAttributes dw;
        dw.kernel_size = {1, attrs.kernel_size[0], attrs.kernel_size[1]};
        dw.stride = {1, attrs.stride[0], attrs.stride[1]};
        dw.dilation = {1, attrs.dilation[0], attrs.dilation[1]};
        dw.padding = {0, attrs.padding[0], attrs.padding[1]};
        dw.epilogue = attrs.epilogue;
        dw.add_to = attrs.add_to;
        dw.residual_pre_activation = attrs.residual_pre_activation;
        if (backend_ == Backend::CPU) {
            return backend::cpu::depthwise_conv_cpu(dw, output, inputs, ctx, workspace);
        }
#if defined(NNOPS_HAS_CUDA)
        return backend::cuda::depthwise_conv_cuda(dw, output, inputs, ctx, workspace);
#else
        return Status::unsupported;
#endif
    }

    // A raw filter keeps the pixel axis on M; U is the only other prepacked
    // form, and only it selects Winograd.
    const bool wino_u = is_wino_weight(inputs[1].layout());
    if (backend_ == Backend::CPU &&
        (is_quantized_conv2d(inputs[1].data_type()) || is_packed_conv_weight(inputs[1].layout()) ||
         !wino_u)) {
        return backend::cpu::conv2d_implicit_gemm_kernel(attrs, output, inputs, ctx, workspace);
    }

    // Auto reads the weight it was handed: a raw filter has no U to use.
    auto kernel_fn = impl_->kernel_fn;
    if (attrs.algorithm == ConvAlgorithm::Auto) {
        attrs.algorithm = wino_u ? ConvAlgorithm::Winograd : ConvAlgorithm::ImplicitGemm;
        kernel_fn = resolve_conv2d_kernel(backend_, attrs.algorithm);
    }

    if (!kernel_fn) {
        return Status::unsupported;
    }
    return kernel_fn(attrs, output, inputs, ctx, workspace);
}

void Conv2D::prepackWeights(std::span<const TensorView> inputs, std::span<TensorView> outputs,
                            const ComputeContext& ctx)
{
    // The fp CUDA kernels read the raw filter in place (implicit GEMM only), so
    // the query stays empty for them — the same "no packed form" answer MatMul
    // gives a dtype it cannot pack. The quantized CUDA route does have one: its
    // fill would otherwise gather the HWIO filter per block per k tile.
    if (backend_ != Backend::CPU) {
#if defined(NNOPS_HAS_CUDA)
        if (backend_ == Backend::CUDA && inputs.size() >= 1 && outputs.size() >= 1 &&
            inputs[0].data_type() == DataType::s8 && inputs[0].rank() == 4 &&
            is_channel_last_weight(inputs[0].layout()) &&
            attrs_.algorithm != ConvAlgorithm::Winograd) {
            const TensorView& weight = inputs[0];
            const TensorDesc d = backend::cuda::conv_i8_packed_weight_desc(attrs_, weight.desc());
            if (d.rank == 0) {
                return;
            }
            std::vector<int64_t> shape(d.dims.begin(), d.dims.begin() + d.rank);
            if (outputs[0].is_empty()) {
                outputs[0] = TensorView(shape, d.dtype, nullptr, d.row_stride_elems,
                                        d.layout, weight.quant_params());
                return;
            }
            NNOPS_ASSERT(outputs[0].layout() == d.layout && outputs[0].rank() == d.rank &&
                         outputs[0].shape(0) == shape[0] && outputs[0].shape(1) == shape[1] &&
                         "Conv2D::prepackWeights: the packed buffer does not match the "
                         "geometry");
            const Status packed =
                backend::cuda::conv_i8_packed_weight_pack(attrs_, weight, outputs[0], ctx);
            NNOPS_ASSERT(packed == Status::ok &&
                         "Conv2D::prepackWeights: the CUDA pack pass failed");
        }
#endif
        return;
    }
    NNOPS_ASSERT(inputs.size() >= 1);
    NNOPS_ASSERT(outputs.size() >= 1);
    const auto& weight = inputs[0];
    NNOPS_ASSERT(weight.rank() == 4);
    NNOPS_ASSERT(is_channel_last_weight(weight.layout()));

    // A depthwise geometry runs on the raw filter, so there is nothing to pack.
    // Answering with a prepacked blob would hand compute() a form that steers it
    // back to the grouped GEMM — the caller would pay for a pack it cannot use.
    if (attrs_.algorithm != ConvAlgorithm::ImplicitGemm && is_fp(weight.data_type()) &&
        is_depthwise_geometry(attrs_, weight)) {
        return;
    }

    // Two products live behind this one call. Auto keeps producing U exactly
    // where the transform exists — the k3s1 fp geometry its asserts demand —
    // because that is what an Auto caller will feed back into compute().
    // The U exists for f32 and f16 only: the Winograd kernel refuses every
    // other dtype (its U panels are the dense fp pack), so neither an Auto nor
    // an explicit Winograd pack may hand back a U the same call would decline.
    const bool wino_dtype =
        weight.data_type() == DataType::f32 || weight.data_type() == DataType::f16;
    const bool wino_product =
        wino_dtype &&
        (attrs_.algorithm == ConvAlgorithm::Winograd ||
         (attrs_.algorithm == ConvAlgorithm::Auto && !is_quantized_conv2d(weight.data_type()) &&
          attrs_.kernel_size[0] == 3 && attrs_.kernel_size[1] == 3 && attrs_.stride[0] == 1 &&
          attrs_.stride[1] == 1 && attrs_.dilation[0] == 1 && attrs_.dilation[1] == 1));

    if (!wino_product) {
        // A square 3x3/5x5/7x7 over at most four channels has no packed form:
        // the direct route convolves the raw filter in place, and one call per
        // inference leaves nothing to amortize. The
        // conditions mirror that route's, so a shape it will not take still
        // packs (a declined pack is a per-call pack at compute time).
        const int64_t ks = attrs_.kernel_size[0];
        const bool epi_pass =
            attrs_.epilogue.type != EpilogueActivateType::None &&
            !(!attrs_.add_to && (attrs_.epilogue.type == EpilogueActivateType::Relu ||
                                 attrs_.epilogue.type == EpilogueActivateType::Relu6));
        if (is_channel_last_weight(weight.layout()) && attrs_.groups == 1 &&
            (weight.data_type() == DataType::f32 || weight.data_type() == DataType::f16) &&
            weight.shape(2) <= 4 && weight.shape(3) % 16 == 0 && ks == attrs_.kernel_size[1] &&
            ks >= 3 && ks <= 7 && (ks & 1) && !attrs_.add_to && !epi_pass) {
            return;
        }
        // The descriptor comes from the attributes and the filter's own dims, so
        // the query needs no activation.
        const TensorDesc d = backend::cpu::conv_packed_weight_desc(attrs_, weight.desc());
        NNOPS_ASSERT(d.rank > 0 && "Conv2D::prepackWeights: this filter has no packed form");
        std::vector<int64_t> shape(d.dims.begin(), d.dims.begin() + d.rank);
        const int64_t esz = static_cast<int64_t>(data_type_size(d.dtype));

        if (outputs[0].is_empty()) {
            // The pitch is the row length in bytes: a packed filter's rows are
            // not aligned to the default, so the default alignment would not
            // describe it.
            outputs[0] = TensorView(shape, d.dtype, nullptr, d.row_stride_elems * esz, d.layout,
                                    weight.quant_params());
            return;
        }

        NNOPS_ASSERT(outputs[0].layout() == d.layout && outputs[0].rank() == d.rank &&
                     outputs[0].shape(0) == shape[0] && outputs[0].shape(1) == shape[1] &&
                     "Conv2D::prepackWeights: the packed buffer does not match "
                     "the geometry");
        backend::cpu::conv_packed_weight_pack(attrs_, weight, outputs[0], ctx);
        return;
    }

    // compute() receives U and cannot fall back to the raw-weight im2col path,
    // so every unsupported configuration must fail here, at prepack time.
    NNOPS_ASSERT(
        (attrs_.algorithm == ConvAlgorithm::Winograd || attrs_.algorithm == ConvAlgorithm::Auto) &&
        "Conv2D::prepackWeights: attrs.algorithm must be Winograd or Auto");
    NNOPS_ASSERT(attrs_.kernel_size[0] == 3 && attrs_.kernel_size[1] == 3 &&
                 attrs_.stride[0] == 1 && attrs_.stride[1] == 1 && attrs_.dilation[0] == 1 &&
                 attrs_.dilation[1] == 1 && "Conv2D::prepackWeights: Winograd requires k3s1");

    const DataType dtype = weight.data_type();
    const int64_t elem_size = static_cast<int64_t>(data_type_size(dtype));

    // Optional inputs[1]: the activation descriptor. The order is chosen once,
    // here, and baked into U's leading dimension — compute() reads it back — so
    // this is the only chance to weigh the plane the transform will run on.
    int64_t plane_h = 0;
    int64_t plane_w = 0;
    if (inputs.size() > 1 && inputs[1].rank() == 4) {
        plane_h = inputs[1].shape(1);
        plane_w = inputs[1].shape(2);
    }
    NNOPS_ASSERT(
        (attrs_.winograd_order == 0 || attrs_.winograd_order == 2 || attrs_.winograd_order == 4) &&
        "Conv2D::prepackWeights: winograd_order is 0, 2 or 4");
    // The worker count, not the item grid: an over-subscribed pool hands a
    // worker several items, and what the transform's cost per task amortizes
    // against is the cores it runs on.
    const int team = std::max(ctx.cpu.thread_slot_count(), 1);
    const auto pick_order = [&](int64_t icg, int64_t ocg) {
        return attrs_.winograd_order == 2 ? backend::cpu::WinoOrder::F23 :
               attrs_.winograd_order == 4 ? backend::cpu::WinoOrder::F43 :
                                            backend::cpu::wino_pick_order(dtype, icg, ocg, plane_h,
                                                                          plane_w, team);
    };
    // Weight: [KH, KW, IC/G, OC]
    const int64_t ICG = weight.shape(2);
    const int64_t OC = weight.shape(3);
    const int64_t G = attrs_.groups;
    NNOPS_ASSERT(G >= 1 && OC % G == 0 && ICG > 0);
    const int64_t OCG = OC / G;

    const backend::cpu::WinoOrder order = pick_order(ICG, OCG);
    const backend::cpu::WinoU ug =
        backend::cpu::wino_u(ICG, OCG, G, dtype, order);
    NNOPS_ASSERT(ug.total > 0 && "Conv2D::prepackWeights: unsupported dtype");

    if (outputs[0].is_empty()) {
        // Query: the panels [pos][nb][G·jb][ldd], pitched at ldd — not aligned
        // to the default 32 B row, so the default alignment would not fit.
        const int64_t u_shape[] = {ug.pos, ug.nb, ug.jbg, ug.ldd};
        outputs[0] = TensorView(std::span<const int64_t>(u_shape, 4), dtype, nullptr,
                                ug.ldd * elem_size, TensorLayout::PackedWinoWeightLast);
        return;
    }

    NNOPS_ASSERT(outputs[0].shape(0) == ug.pos && outputs[0].shape(1) == ug.nb &&
                 outputs[0].shape(2) == ug.jbg && outputs[0].shape(3) == ug.ldd &&
                 outputs[0].pitch() == ug.ldd * elem_size &&
                 "Conv2D::prepackWeights: U buffer does not match the plan");
    // The oc axis is padded to whole panels; those columns stay zero.
    std::memset(outputs[0].ptr<char>(), 0,
                static_cast<size_t>(ug.total) * static_cast<size_t>(elem_size));
    if (dtype == DataType::f32) {
        wino_pack_weight_hwio<float>(weight, outputs[0], ug, ICG, OCG, G, ctx);
    } else {
        NNOPS_ASSERT(dtype == DataType::f16);
        wino_pack_weight_hwio<backend::cpu::half>(weight, outputs[0], ug, ICG, OCG, G, ctx);
    }
}

} // namespace nnops
