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
/// Two layouts, two walks of the class's compressed grid:
///
///   * channel-last lanes the channel inside a 16 B vector, so a tap is one
///     activation vector against one strided weight vector. The weight is the
///     scalar side — the filter is [C][1][KH][KW], so a channel step is w_ic
///     elements — and one output per thread is what measured fastest here: a
///     window over rows hoists `Vec` weights into a register array that ptxas
///     then keeps in local memory (a 32-128 byte stack frame on every windowed
///     instantiation), which costs more than the loads it saves.
///   * channel-first lanes the class's columns, so a tap is one broadcast weight
///     against a coalesced run of activations. There the weight is a single
///     scalar, VW columns per thread amortize it, and the block's column span is
///     fitted to ceil(tw / VW) so a narrow class grid does not launch idle lanes.
///
/// The other lever the CPU route has is not here: the filter is read in place.
/// Its [C][1][KH][KW] order costs a strided load per tap, but a per-call
/// [KH][KW][C] repack plus its launch does not pay for itself on kernels this
/// short.
///
/// f32 and f16.

#include "transpose_conv_common.cuh"

namespace nnops::backend::cuda::tconv {
namespace {

/// Channel-last store of one Vec-wide lane: bias, clamp, activation, as a single
/// 16 B write. add_to takes the scalar path instead, which re-reads the pixel.
template <class Elem, int Vec>
__device__ inline void dw_last_store(const ConvParams& p, Elem* dst, const float* acc,
                                     int c0)
{
    using V = std::conditional_t<sizeof(Elem) == 2, uint4, float4>;
    const Elem* bias = static_cast<const Elem*>(p.bias);
    V out;
    Elem* os = reinterpret_cast<Elem*>(&out);
    CUTE_UNROLL
    for (int v = 0; v < Vec; ++v) {
        float x = acc[v];
        if (bias != nullptr) { x += float(bias[c0 + v]); }
        x = fminf(fmaxf(x, p.min_clip), p.max_clip);
        x = gemm_apply_epilogue(p.epilogue, p.ep_alpha, p.ep_beta, x);
        os[v] = Elem(x);
    }
    *reinterpret_cast<V*>(dst) = out;
}

/// Depthwise on one class, channel-last: one output per thread on a 16 B channel
/// lane.
template <class Elem, int Vec>
__global__ void dw_last_kernel(ConvParams pin, int th_max, int col_blocks)
{
    using V = std::conditional_t<sizeof(Elem) == 2, uint4, float4>;
    const int cls = int(blockIdx.z) / col_blocks;
    const TClass tc = tconv_class(pin, cls);
    if (tc.empty) { return; }

    const ConvParams p = tc.q;
    // Copied out: ConvParams is too wide to stay in registers as one object, and
    // a spilled field costs a local-memory load inside the tap loop.
    const int KH = p.kh, KW = p.kw, DH = p.dh, DW = p.dw, PH = p.ph, PW = p.pw;
    const int IH = p.ih, IW = p.iw, IN_H = p.in_h, IN_W = p.in_w;
    const int W_KH = p.w_kh, W_KW = p.w_kw, W_IC = p.w_ic;
    const int ow = (int(blockIdx.z) - cls * col_blocks) * int(blockDim.y) +
                   int(threadIdx.y);
    const int img = int(blockIdx.x) / th_max;
    const int th = int(blockIdx.x) - img * th_max;
    const int c0 = (int(blockIdx.y) * int(blockDim.x) + int(threadIdx.x)) * Vec;
    if (th >= p.oh || ow >= p.ow || c0 >= p.oc) { return; }

    const Elem* in = static_cast<const Elem*>(p.input) + int64_t(img) * p.in_batch + c0;
    // The class starts at tap k0 of the original kernel, on both axes.
    const Elem* w = static_cast<const Elem*>(p.weight) + int64_t(c0) * p.w_ic +
                    tc.w_off;

    // The taps walk by pointer: DW and W_KW are constants of the class, so each
    // step is one add where an index would be a 64-bit multiply.
    const int64_t in_step = int64_t(DW) * IN_W;

    float acc[Vec] = {};
    const int ih0 = th - PH;
    const int ow_in = ow - PW;
    for (int kh = 0; kh < KH; ++kh) {
        const int ih = ih0 + kh * DH;
        if (unsigned(ih) >= unsigned(IH)) { continue; }
        const Elem* xp = in + int64_t(ih) * IN_H + int64_t(ow_in) * IN_W;
        const Elem* wp = w + int64_t(kh) * W_KH;
        int iw = ow_in;
        for (int kw = 0; kw < KW; ++kw) {
            if (unsigned(iw) < unsigned(IW)) {
                const V x = *reinterpret_cast<const V*>(xp);
                const Elem* xs = reinterpret_cast<const Elem*>(&x);
                // Group == output channel here, so lane v's channel is c0 + v.
                const Elem* wv = wp;
                CUTE_UNROLL
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
        dw_last_store<Elem, Vec>(p, dst, acc, c0);
    }
}

/// Depthwise on one class, channel-first: `VW` class columns per thread, lanes on
/// the columns, so consecutive threads read consecutive activations and a tap is
/// one broadcast weight against a coalesced run.
template <class Elem, int VW>
__global__ void dw_first_kernel(ConvParams pin, int th_max, int col_blocks)
{
    const int cls = int(blockIdx.z) / col_blocks;
    const TClass tc = tconv_class(pin, cls);
    if (tc.empty) { return; }

    const ConvParams p = tc.q;
    const int KH = p.kh, KW = p.kw, DH = p.dh, DW = p.dw, PH = p.ph, PW = p.pw;
    const int IH = p.ih, IW = p.iw, IN_H = p.in_h, IN_W = p.in_w;
    const int W_KH = p.w_kh, W_KW = p.w_kw;
    const int ow0 = (int(blockIdx.z) - cls * col_blocks) * int(blockDim.x) * VW +
                    int(threadIdx.x) * VW;
    const int row_i = int(blockIdx.y) * int(blockDim.y) + int(threadIdx.y);
    const int img = row_i / th_max;
    const int th = row_i - img * th_max;
    const int c = int(blockIdx.x);
    if (img >= p.batch || th >= p.oh || ow0 >= p.ow) { return; }
    const int cols = (p.ow - ow0 < VW) ? p.ow - ow0 : VW;

    const Elem* in = static_cast<const Elem*>(p.input) + int64_t(img) * p.in_batch +
                     int64_t(c) * p.in_c;
    // The class starts at tap k0 of the original kernel, on both axes.
    const Elem* w = static_cast<const Elem*>(p.weight) + int64_t(c) * p.w_ic +
                    tc.w_off;

    float acc[VW] = {};
    for (int kh = 0; kh < KH; ++kh) {
        const int ih = th - PH + kh * DH;
        if (unsigned(ih) >= unsigned(IH)) { continue; }
        const Elem* wp = w + int64_t(kh) * W_KH;
        int iw = ow0 - PW;
        for (int kw = 0; kw < KW; ++kw) {
            const float wv = float(*wp);
            const Elem* xp = in + int64_t(ih) * IN_H + int64_t(iw) * IN_W;
            CUTE_UNROLL
            for (int v = 0; v < VW; ++v) {
                if (unsigned(iw + v) < unsigned(IW)) {
                    acc[v] = fmaf(wv, float(*xp), acc[v]);
                }
                xp += IN_W;
            }
            wp += W_KW;
            iw += DW;
        }
    }

    Elem* dst = static_cast<Elem*>(p.output) + int64_t(img) * p.out_batch +
                int64_t(c) * p.out_c + tc.out_off + int64_t(th) * p.out_h +
                int64_t(ow0) * p.out_w;
    const Elem* bias = static_cast<const Elem*>(p.bias);
    for (int v = 0; v < cols; ++v) {
        tconv_store<Elem>(p, dst, acc[v], c, bias);
        dst += p.out_w;
    }
}

// ============================================================
// Launch
// ============================================================

/// The channel-first column window: the weight is one scalar per tap, so `VW`
/// columns per thread amortize it. Below a few thousand lanes the launch itself
/// dominates and the narrowest window is fine; the class grid also has to be
/// wide enough that the block keeps at least two lanes on the columns, or the
/// warp degenerates into one row per lane and the tap loads stop coalescing.
int col_window_first(int64_t lanes, int tw_max)
{
    if (tw_max < 2 || lanes < 4096) { return 1; }
    if (tw_max < 4 || lanes < 16384) { return 2; }
    return 4;
}

template <class Elem>
void launch_last(const ConvParams& p, int th_max, int col_blocks, int by,
                 cudaStream_t stream)
{
    constexpr int Vec = 16 / int(sizeof(Elem));
    const int chunks = p.oc / Vec;
    const int bx = chunks < 32 ? chunks : 32;
    dim3 grid(unsigned(p.batch * th_max), unsigned((chunks + bx - 1) / bx),
              unsigned(p.classes * col_blocks));
    dw_last_kernel<Elem, Vec><<<grid, dim3(bx, by), 0, stream>>>(p, th_max, col_blocks);
}

template <class Elem, int VW>
void launch_first(const ConvParams& p, int th_max, int col_blocks, int bx,
                  cudaStream_t stream)
{
    const int by = kCudaBlockSize / bx;
    const int rows = p.batch * th_max;
    dim3 grid(unsigned(p.oc), unsigned((rows + by - 1) / by),
              unsigned(p.classes * col_blocks));
    dw_first_kernel<Elem, VW><<<grid, dim3(bx, by), 0, stream>>>(p, th_max, col_blocks);
}

template <class Elem>
void dispatch_first(const ConvParams& p, int th_max, int col_blocks, int bx, int vw,
                    cudaStream_t stream)
{
    switch (vw) {
    case 4:  launch_first<Elem, 4>(p, th_max, col_blocks, bx, stream); break;
    case 2:  launch_first<Elem, 2>(p, th_max, col_blocks, bx, stream); break;
    default: launch_first<Elem, 1>(p, th_max, col_blocks, bx, stream); break;
    }
}

template <class Elem>
bool run_first(const ConvParams& p, int th_max, int tw_max, cudaStream_t stream)
{
    if (p.oc > 65535) { return false; }
    const int64_t rows = int64_t(p.batch) * th_max;
    if (rows > 0x7fffffff) { return false; }

    const int64_t lanes = rows * tw_max * p.oc;
    int vw = col_window_first(lanes, tw_max);
    // A window wider than the class row leaves one lane per row, and a warp then
    // walks 32 separate rows instead of one contiguous run.
    while (vw > 1 && (tw_max + vw - 1) / vw < 2) { vw >>= 1; }
    const int want = (tw_max + vw - 1) / vw;
    const int bx = want < 32 ? want : 32;
    const int by = kCudaBlockSize / bx;
    if ((rows + by - 1) / by > 65535) { return false; }
    const int col_blocks = (tw_max + bx * vw - 1) / (bx * vw);
    if (int64_t(p.classes) * col_blocks > 65535) { return false; }

    dispatch_first<Elem>(p, th_max, col_blocks, bx, vw, stream);
    return true;
}

template <class Elem>
bool run_last(const ConvParams& p, int th_max, int tw_max, cudaStream_t stream)
{
    constexpr int Vec = 16 / int(sizeof(Elem));
    constexpr int64_t esz = int64_t(sizeof(Elem));
    if (p.in_c != 1 || p.out_c != 1) { return false; }
    if (p.ic % Vec != 0 || p.oc % Vec != 0) { return false; }
    if ((p.in_w * esz) % 16 != 0 || (p.out_w * esz) % 16 != 0) { return false; }
    if (reinterpret_cast<uintptr_t>(p.input) % 16 != 0) { return false; }
    if (reinterpret_cast<uintptr_t>(p.output) % 16 != 0) { return false; }
    if (int64_t(p.batch) * th_max > 0x7fffffff || p.oc > 65535) { return false; }

    const int chunks = p.oc / Vec;
    const int bx = chunks < 32 ? chunks : 32;
    const int max_by = kCudaBlockSize / bx;
    const int by = tw_max < max_by ? tw_max : max_by;
    const int col_blocks = (tw_max + by - 1) / by;
    if (int64_t(p.classes) * col_blocks > 65535) { return false; }

    launch_last<Elem>(p, th_max, col_blocks, by, stream);
    return true;
}

}  // namespace

bool transpose_conv2d_depthwise_cuda(const ConvParams& p, int th_max, int tw_max,
                                     bool channel_last, bool half_elem,
                                     cudaStream_t stream)
{
    if (p.icg != 1 || p.ocg != 1) { return false; }
    if (p.batch <= 0 || th_max <= 0 || tw_max <= 0) { return false; }
    if (half_elem) {
        return channel_last ? run_last<cute::half_t>(p, th_max, tw_max, stream)
                            : run_first<cute::half_t>(p, th_max, tw_max, stream);
    }
    return channel_last ? run_last<float>(p, th_max, tw_max, stream)
                        : run_first<float>(p, th_max, tw_max, stream);
}

}  // namespace nnops::backend::cuda::tconv
