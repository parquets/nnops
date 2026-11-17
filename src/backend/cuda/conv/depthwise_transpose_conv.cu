/// @file depthwise_transpose_conv.cu
/// @brief CUDA depthwise transposed convolution — the class view of
///        transpose_conv_common.cuh, walked as a plain correlation.
///
/// A group of one input and one output channel has no channel axis to tile and
/// no reduction left inside a group, so the GEMM of the main TransposeConv2D
/// route has nothing to work with: it would run one column wide. The class view
/// is what that geometry wants instead. Cutting the outputs by residue class
/// removes the stride from the kernel — a class is a stride-1 correlation with a
/// flipped, cropped filter (see the header) — and nothing is left to do but
/// multiply-accumulate.
///
/// The walk lanes the channel inside a 16 B vector, so a tap is
/// one activation vector against one strided weight vector. The weight is the
/// scalar side — the filter is [C][1][KH][KW], so a channel step is w_ic
/// elements — and one output per thread is what measured fastest here: a window
/// over rows hoists `Vec` weights into a register array that ptxas then keeps in
/// local memory (a 32-128 byte stack frame on every windowed instantiation),
/// which costs more than the loads it saves.
///
/// The other lever the CPU route has is not here: the filter is read in place.
/// Its [C][1][KH][KW] order costs a strided load per tap, but a per-call
/// [KH][KW][C] repack plus its launch does not pay for itself on kernels this
/// short.
///
/// A geometry the 16 B lane cannot take — a channel count the
/// vector does not divide, or a row it cannot align to — falls to a second
/// script, one output element per thread with the channel innermost (which is
/// the coalesced order on that layout) and the tap search written out the way
/// the CPU's is. It is a fallback, not a route: nothing picks it on a shape the
/// lane can already do.
///
/// f32, f16 and bf16. The operator entry at the bottom of the file is what wires all
/// of this to DepthwiseTransposeConv2D.

#include "transpose_conv_common.cuh"
#include "nnops/ops/depthwise_transpose_conv.hpp"

namespace nnops::backend::cuda::tconv {
namespace {

/// The vector store of one Vec-wide lane: bias, clamp, activation, as a single
/// 16 B write. add_to takes the scalar path instead, which re-reads the pixel.
template <class Elem, int Vec>
__device__ inline void dw_store(const ConvParams& p, Elem* dst, const float* acc, int c0)
{
    using V = std::conditional_t<sizeof(Elem) == 2, uint4, float4>;
    const Elem* bias = static_cast<const Elem*>(p.bias);
    V out;
    Elem* os = reinterpret_cast<Elem*>(&out);
#pragma unroll
    for (int v = 0; v < Vec; ++v) {
        float x = acc[v];
        if (bias != nullptr) {
            x += float(bias[c0 + v]);
        }
        x = fminf(fmaxf(x, p.min_clip), p.max_clip);
        x = gemm_apply_epilogue(p.epilogue, p.ep_alpha, p.ep_beta, x);
        os[v] = Elem(x);
    }
    *reinterpret_cast<V*>(dst) = out;
}

/// Depthwise on one class: one output per thread on a 16 B channel lane.
template <class Elem, int Vec>
__global__ void dw_kernel(ConvParams pin, int th_max, int col_blocks)
{
    using V = std::conditional_t<sizeof(Elem) == 2, uint4, float4>;
    const int cls = int(blockIdx.z) / col_blocks;
    const TClass tc = tconv_class(pin, cls);
    if (tc.empty) {
        return;
    }

    const ConvParams p = tc.q;
    // Copied out: ConvParams is too wide to stay in registers as one object, and
    // a spilled field costs a local-memory load inside the tap loop.
    const int KH = p.kh, KW = p.kw, DH = p.dh, DW = p.dw, PH = p.ph, PW = p.pw;
    const int IH = p.ih, IW = p.iw, IN_H = p.in_h, IN_W = p.in_w;
    const int W_KH = p.w_kh, W_KW = p.w_kw, W_IC = p.w_ic;
    const int ow = (int(blockIdx.z) - cls * col_blocks) * int(blockDim.y) + int(threadIdx.y);
    const int img = int(blockIdx.x) / th_max;
    const int th = int(blockIdx.x) - img * th_max;
    const int c0 = (int(blockIdx.y) * int(blockDim.x) + int(threadIdx.x)) * Vec;
    if (th >= p.oh || ow >= p.ow || c0 >= p.oc) {
        return;
    }

    const Elem* in = static_cast<const Elem*>(p.input) + int64_t(img) * p.in_batch + c0;
    // The class starts at tap k0 of the original kernel, on both axes.
    const Elem* w = static_cast<const Elem*>(p.weight) + int64_t(c0) * p.w_ic + tc.w_off;

    // The taps walk by pointer: DW and W_KW are constants of the class, so each
    // step is one add where an index would be a 64-bit multiply.
    const int64_t in_step = int64_t(DW) * IN_W;

    float acc[Vec] = {};
    const int ih0 = th - PH;
    const int ow_in = ow - PW;
    for (int kh = 0; kh < KH; ++kh) {
        const int ih = ih0 + kh * DH;
        if (unsigned(ih) >= unsigned(IH)) {
            continue;
        }
        const Elem* xp = in + int64_t(ih) * IN_H + int64_t(ow_in) * IN_W;
        const Elem* wp = w + int64_t(kh) * W_KH;
        int iw = ow_in;
        for (int kw = 0; kw < KW; ++kw) {
            if (unsigned(iw) < unsigned(IW)) {
                const V x = *reinterpret_cast<const V*>(xp);
                const Elem* xs = reinterpret_cast<const Elem*>(&x);
                // Group == output channel here, so lane v's channel is c0 + v.
                const Elem* wv = wp;
#pragma unroll
                for (int v = 0; v < Vec; ++v) {
                    acc[v] = fmaf(float(wv[0]), float(xs[v]), acc[v]);
                    wv += W_IC;
                }
            }
            xp += in_step;
            wp += W_KW;
            iw += DW;
        }
    }

    Elem* dst = static_cast<Elem*>(p.output) + int64_t(img) * p.out_batch + tc.out_off +
                int64_t(th) * p.out_h + int64_t(ow) * p.out_w + c0;
    if (p.add_to) {
        const Elem* bias = static_cast<const Elem*>(p.bias);
        for (int v = 0; v < Vec; ++v) {
            tconv_store<Elem>(p, dst + v, acc[v], c0 + v, bias);
        }
    } else {
        dw_store<Elem, Vec>(p, dst, acc, c0);
    }
}

// ============================================================
// Launch
// ============================================================

template <class Elem>
void launch_impl(const ConvParams& p, int th_max, int col_blocks, int by, cudaStream_t stream)
{
    constexpr int Vec = 16 / int(sizeof(Elem));
    const int chunks = p.oc / Vec;
    const int bx = chunks < 32 ? chunks : 32;
    dim3 grid(unsigned(p.batch * th_max), unsigned((chunks + bx - 1) / bx),
              unsigned(p.classes * col_blocks));
    dw_kernel<Elem, Vec><<<grid, dim3(bx, by), 0, stream>>>(p, th_max, col_blocks);
}

template <class Elem>
bool run_impl(const ConvParams& p, int th_max, int tw_max, cudaStream_t stream)
{
    constexpr int Vec = 16 / int(sizeof(Elem));
    constexpr int64_t esz = int64_t(sizeof(Elem));
    if (p.in_c != 1 || p.out_c != 1) {
        return false;
    }
    if (p.ic % Vec != 0 || p.oc % Vec != 0) {
        return false;
    }
    if ((p.in_w * esz) % 16 != 0 || (p.out_w * esz) % 16 != 0) {
        return false;
    }
    if (reinterpret_cast<uintptr_t>(p.input) % 16 != 0) {
        return false;
    }
    if (reinterpret_cast<uintptr_t>(p.output) % 16 != 0) {
        return false;
    }
    if (int64_t(p.batch) * th_max > 0x7fffffff || p.oc > 65535) {
        return false;
    }

    const int chunks = p.oc / Vec;
    const int bx = chunks < 32 ? chunks : 32;
    const int max_by = kCudaBlockSize / bx;
    const int by = tw_max < max_by ? tw_max : max_by;
    const int col_blocks = (tw_max + by - 1) / by;
    if (int64_t(p.classes) * col_blocks > 65535) {
        return false;
    }

    launch_impl<Elem>(p, th_max, col_blocks, by, stream);
    return true;
}

// ============================================================
// Scalar fallback
// ============================================================

/// One output element per thread with the channel innermost, for a geometry the
/// 16 B lane declines. The tap search is the CPU's: a tap counts
/// when (o + P − k·D) divides by S and the pixel it names exists.
template <class Elem> __global__ void dw_scalar_kernel(ConvParams p)
{
    const int64_t total = int64_t(p.batch) * p.oc * p.oh * p.ow;
    const int64_t step = int64_t(blockDim.x) * gridDim.x;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += step) {
        int64_t r = i;
        const int c = int(r % p.oc);
        r /= p.oc;
        const int ow = int(r % p.ow);
        r /= p.ow;
        const int oh = int(r % p.oh);
        r /= p.oh;
        const int n = int(r);

        // The channel step is one element on this layout, as in the lane kernel.
        const Elem* in = static_cast<const Elem*>(p.input) + int64_t(n) * p.in_batch + c;
        const Elem* w = static_cast<const Elem*>(p.weight) + int64_t(c) * p.w_ic;

        float acc = 0.0f;
        for (int kh = 0; kh < p.kh; ++kh) {
            const int nh = oh + p.ph - kh * p.dh;
            if (nh < 0 || nh % p.sh != 0) {
                continue;
            }
            const int ih = nh / p.sh;
            if (ih >= p.ih) {
                continue;
            }
            const Elem* row = in + int64_t(ih) * p.in_h;
            const Elem* tap = w + int64_t(kh) * p.w_kh;
            for (int kw = 0; kw < p.kw; ++kw) {
                const int nw = ow + p.pw - kw * p.dw;
                if (nw < 0 || nw % p.sw != 0) {
                    continue;
                }
                const int iw = nw / p.sw;
                if (iw >= p.iw) {
                    continue;
                }
                acc = fmaf(float(row[int64_t(iw) * p.in_w]), float(tap[int64_t(kw) * p.w_kw]), acc);
            }
        }

        Elem* dst = static_cast<Elem*>(p.output) + int64_t(n) * p.out_batch + c +
                    int64_t(oh) * p.out_h + int64_t(ow) * p.out_w;
        tconv_store<Elem>(p, dst, acc, c, static_cast<const Elem*>(p.bias));
    }
}

template <class Elem> void launch_scalar(const ConvParams& p, cudaStream_t stream)
{
    const int64_t total = int64_t(p.batch) * p.oc * p.oh * p.ow;
    const int64_t blocks = (total + kCudaBlockSize - 1) / kCudaBlockSize;
    const unsigned grid = unsigned(blocks < 65535 ? blocks : 65535);
    dw_scalar_kernel<Elem><<<grid, kCudaBlockSize, 0, stream>>>(p);
}

} // namespace

bool transpose_conv2d_depthwise_cuda(const ConvParams& p, int th_max, int tw_max,
                                     DataType dtype, cudaStream_t stream)
{
    if (p.icg != 1 || p.ocg != 1) {
        return false;
    }
    if (p.batch <= 0 || th_max <= 0 || tw_max <= 0) {
        return false;
    }
    switch (dtype) {
    case DataType::f16:
        return run_impl<__half>(p, th_max, tw_max, stream);
    case DataType::bf16:
        return run_impl<__nv_bfloat16>(p, th_max, tw_max, stream);
    default:
        return run_impl<float>(p, th_max, tw_max, stream);
    }
}

} // namespace nnops::backend::cuda::tconv

namespace nnops::backend::cuda {

Status depthwise_transpose_conv_cuda(const DepthwiseTransposeConv2DAttributes& attrs,
                                     TensorView& output, std::span<const TensorView> inputs,
                                     const ComputeContext& ctx, void* /*workspace*/)
{
    const TensorView& act = inputs[0];
    const TensorView& wgt = inputs[1];
    const TensorView* bias = inputs.size() > 2 ? &inputs[2] : nullptr;

    const DataType dt = act.data_type();
    if ((dt != DataType::f32 && dt != DataType::f16 && dt != DataType::bf16) ||
        wgt.data_type() != dt || output.data_type() != dt) {
        return Status::unsupported; // the quantized route is CPU-only
    }
    if (output.numel() == 0) {
        return Status::ok;
    }

    if (!is_channel_last(act.layout()) || !is_channel_last(output.layout())) {
        return Status::unsupported;
    }
    const int64_t ch = act.shape(1);
    if (ch <= 0 || output.shape(1) != ch) {
        return Status::unsupported;
    }

    tconv::ConvParams p;
    p.batch = int(act.shape(0));
    p.ic = int(ch);
    p.ih = int(act.shape(2));
    p.iw = int(act.shape(3));
    p.oc = p.ic;
    p.oh = int(output.shape(2));
    p.ow = int(output.shape(3));
    p.groups = p.ic;
    p.icg = 1;
    p.ocg = 1;

    p.kh = int(attrs.kernel_size[0]);
    p.kw = int(attrs.kernel_size[1]);
    p.sh = int(attrs.stride[0]);
    p.sw = int(attrs.stride[1]);
    p.dh = int(attrs.dilation[0]);
    p.dw = int(attrs.dilation[1]);
    p.ph = int(attrs.padding[0]);
    p.pw = int(attrs.padding[1]);

    p.in_batch = act.stride_elems(0);
    p.in_c = act.stride_elems(1);
    p.in_h = act.stride_elems(2);
    p.in_w = act.stride_elems(3);
    p.out_batch = output.stride_elems(0);
    p.out_c = output.stride_elems(1);
    p.out_h = output.stride_elems(2);
    p.out_w = output.stride_elems(3);

    p.input = act.ptr<void>();
    p.weight = wgt.ptr<void>();
    p.output = output.ptr<void>();
    p.bias = bias != nullptr ? bias->ptr<void>() : nullptr;

    // The kernels address the filter by these three strides alone, so both
    // accepted forms are read in place — [C][1][KH][KW], and the per-tap
    // [KH][KW][C] an NHWC activation may carry.
    if (wgt.rank() == 3) {
        p.w_kh = wgt.stride_elems(0);
        p.w_kw = wgt.stride_elems(1);
        p.w_ic = wgt.stride_elems(2);
    } else {
        p.w_ic = wgt.stride_elems(0);
        p.w_kh = wgt.stride_elems(2);
        p.w_kw = wgt.stride_elems(3);
    }

    p.epilogue = attrs.epilogue.type;
    p.ep_alpha = attrs.epilogue.alpha;
    p.ep_beta = attrs.epilogue.beta;
    p.min_clip = attrs.epilogue.min_clip;
    p.max_clip = attrs.epilogue.max_clip;
    p.add_to = attrs.add_to;
    p.pre_act = attrs.add_to && attrs.residual_pre_activation;
    p.classes = p.sh * p.sw;

    const int th_max = (p.oh + p.sh - 1) / p.sh;
    const int tw_max = (p.ow + p.sw - 1) / p.sw;
    const bool half_elem = (dt != DataType::f32);
    const cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    if (!tconv::transpose_conv2d_depthwise_cuda(p, th_max, tw_max, dt, stream)) {
        // The class view declines a geometry or a grid it cannot hold; the
        // scalar script still covers it.
        if (dt == DataType::bf16) {
            tconv::launch_scalar<__nv_bfloat16>(p, stream);
        } else if (half_elem) {
            tconv::launch_scalar<__half>(p, stream);
        } else {
            tconv::launch_scalar<float>(p, stream);
        }
    }
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

} // namespace nnops::backend::cuda
