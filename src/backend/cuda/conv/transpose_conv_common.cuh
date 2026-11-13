/// @file transpose_conv_common.cuh
/// @brief The residue-class view of a transposed convolution, shared by the
///        TransposeConv2D CUDA routes.
///
/// A transposed convolution scatters: every input pixel contributes its whole
/// filter to a neighbourhood of outputs, so posed head-on it is not a GEMM. The
/// obvious repair — walk the output pixels and treat the KH·KW taps as the
/// reduction, zeroing the K positions a stride skips — costs SH·SW times the
/// MACs (4x on the stride-2 upsamplers that are this op's main use).
///
/// The outputs instead sort into residue classes. With
/// c = (oh + PH) mod SH, every output of one class is hit by the same subset of
/// kh, and that subset is an arithmetic progression: kh = k0 + t·step with
/// step = SH/gcd(DH, SH) (for DH = 1 that is just kh ≡ c (mod SH)). Writing
/// oh = c − PH + SH·m turns the class into an ordinary *correlation* with a
/// flipped, cropped kernel — ih = m − j0 − t·(DH/gcd) — which is exactly
/// Conv2D's gather with a negative dilation, a shifted origin and a strided
/// weight tap.
///
/// Everything below is that view: the parameter block both routes read, the
/// class decomposition, and the scalar store the two depthwise kernels share.

#pragma once

#include "../cuda_common.cuh"
#include "../gemm/cute_gemm.cuh"

#include <limits>

namespace nnops::backend::cuda::tconv {

struct ConvParams {
    const void* input = nullptr;
    const void* weight = nullptr;
    const void* bias = nullptr;
    void* output = nullptr;

    int batch = 0;
    int ic = 0, ih = 0, iw = 0;
    int oc = 0, oh = 0, ow = 0;
    int kh = 0, kw = 0;
    int groups = 0, icg = 0, ocg = 0;
    int sh = 0, sw = 0, dh = 0, dw = 0, ph = 0, pw = 0;

    int64_t in_batch = 0, in_c = 0, in_h = 0, in_w = 0;
    int64_t out_batch = 0, out_c = 0, out_h = 0, out_w = 0;
    // Filter strides in elements for [IC][OC/G][KH][KW]: the output channel, the
    // input channel within the group, and the two taps. A class re-points the
    // tap strides and adds a base offset; neither makes k contiguous.
    int64_t w_n = 0, w_ic = 0, w_kh = 0, w_kw = 0;

    EpilogueActivateType epilogue = EpilogueActivateType::None;
    float ep_alpha = 0.0f;
    float ep_beta = 1.0f;
    float min_clip = -std::numeric_limits<float>::infinity();
    float max_clip = std::numeric_limits<float>::infinity();
    bool add_to = false;
    bool pre_act = false;
    bool channels_last = false;
    bool fill_vec = false;

    /// SH·SW residue classes, the class axis blockIdx.z walks fastest.
    int classes = 1;
};

__device__ inline int tdiv_floor(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }
__device__ inline int tdiv_ceil(int a, int b) { return -tdiv_floor(-a, b); }
__device__ inline int tgcd(int a, int b)
{
    while (b != 0) {
        const int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

/// One residue class: the per-class geometry as a ConvParams the shared fill and
/// store helpers already understand, plus the two base offsets the class moves.
struct TClass {
    ConvParams q;
    int64_t out_off = 0;   ///< oh0·out_h + ow0·out_w of the class's first output
    int64_t w_off = 0;     ///< k0h·w_kh + k0w·w_kw, the class's first tap
    bool empty = true;
};

/// Decompose one (ch, cw) class.
///
/// The taps of a class are kh = k0h + t·steph with steph = SH/gcd(DH, SH), which
/// holds for any dilation because the congruence kh·DH ≡ ch (mod SH) has either
/// no solution or the solutions of a single residue class modulo SH/gcd. A class
/// with no solution still owns outputs (their sum is empty and only the bias
/// survives), so it is launched with a zero-length reduction rather than skipped.
__device__ inline TClass tconv_class(const ConvParams& p, int cls)
{
    TClass c;
    c.q = p;

    const int ch = cls / p.sw;
    const int cw = cls - ch * p.sw;

    const int gh = tgcd(p.dh, p.sh);
    const int gw = tgcd(p.dw, p.sw);
    int k0h = -1, k0w = -1;
    if (ch % gh == 0) {
        for (int k = 0; k < p.sh; ++k) {
            if ((k * p.dh) % p.sh == ch) { k0h = k; break; }
        }
    }
    if (cw % gw == 0) {
        for (int k = 0; k < p.sw; ++k) {
            if ((k * p.dw) % p.sw == cw) { k0w = k; break; }
        }
    }
    const int steph = p.sh / gh;
    const int stepw = p.sw / gw;
    const int jch = k0h >= 0 && k0h < p.kh ? (p.kh - k0h + steph - 1) / steph : 0;
    const int jcw = k0w >= 0 && k0w < p.kw ? (p.kw - k0w + stepw - 1) / stepw : 0;
    const int j0h = k0h >= 0 ? (k0h * p.dh - ch) / p.sh : 0;
    const int j0w = k0w >= 0 ? (k0w * p.dw - cw) / p.sw : 0;

    // oh = ch − PH + SH·m over the outputs that exist.
    const int m_lo_h = tdiv_ceil(p.ph - ch, p.sh);
    const int m_hi_h = tdiv_floor(p.oh - 1 + p.ph - ch, p.sh);
    const int m_lo_w = tdiv_ceil(p.pw - cw, p.sw);
    const int m_hi_w = tdiv_floor(p.ow - 1 + p.pw - cw, p.sw);
    const int th = m_hi_h - m_lo_h + 1;
    const int tw = m_hi_w - m_lo_w + 1;

    c.empty = th <= 0 || tw <= 0;
    c.q.oh = th > 0 ? th : 0;
    c.q.ow = tw > 0 ? tw : 0;
    c.q.kh = jch;
    c.q.kw = jcw;
    c.q.sh = 1;
    c.q.sw = 1;
    // The class's input origin. Its row index is t_h = m − m_lo, so
    // ih = t_h + m_lo − j0 − t·(DH/gcd); the fill's oh·sh − ph form carries
    // (j0 − m_lo) in ph.
    c.q.ph = j0h - m_lo_h;
    c.q.pw = j0w - m_lo_w;
    c.q.dh = -(p.dh / gh);
    c.q.dw = -(p.dw / gw);
    c.q.w_kh = p.w_kh * steph;
    c.q.w_kw = p.w_kw * stepw;
    // The class's m axis is the output axis with SH/SW folded into its strides.
    c.q.out_h = p.out_h * p.sh;
    c.q.out_w = p.out_w * p.sw;

    c.out_off = int64_t(ch - p.ph + p.sh * m_lo_h) * p.out_h +
                int64_t(cw - p.pw + p.sw * m_lo_w) * p.out_w;
    c.w_off = int64_t(k0h > 0 ? k0h : 0) * p.w_kh + int64_t(k0w > 0 ? k0w : 0) * p.w_kw;
    return c;
}

/// bias, residual, clamp, activation, residual — the CPU's ordering exactly:
/// the pre-activation form folds the residual into the sum, the default one
/// lands after the activation.
template <class Elem>
__device__ inline void tconv_store(const ConvParams& p, Elem* dst, float v, int n,
                                   const Elem* bias)
{
    if (bias != nullptr) { v += gemm_to_float(bias[n]); }
    if (p.add_to && p.pre_act) { v += float(*dst); }
    v = fminf(fmaxf(v, p.min_clip), p.max_clip);
    v = gemm_apply_epilogue(p.epilogue, p.ep_alpha, p.ep_beta, v);
    if (p.add_to && !p.pre_act) { v += float(*dst); }
    *dst = Elem(v);
}

/// Depthwise route, in depthwise_transpose_conv.cu. Returns false when the
/// geometry or the grid does not fit that route, and the caller falls back.
bool transpose_conv2d_depthwise_cuda(const ConvParams& p, int th_max, int tw_max,
                                     bool channel_last, bool half_elem,
                                     cudaStream_t stream);

}  // namespace nnops::backend::cuda::tconv
