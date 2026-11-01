#pragma once
/// @file simd_stem_conv.hpp
/// @brief The tiny filter: a square 3x3/5x5/7x7 over at most four channels,
/// convoluted in place.
///
/// A wide filter over at most four input channels cannot pay for a gather: the
/// GEMM route copies `karea·IC` elements per pixel into an im2col tile only to
/// read them back, and at the 3-channel stem that copy is most of the layer's
/// time. The patch is already in place on the channel-last plane, so convolve it
/// where it lies — one weight vector per (tap, input channel), one broadcast per
/// pixel tap, four pixels against two vectors of output channels. That is the
/// MMA's own FMA-to-load ratio with nothing copied on either side, and the
/// filter needs no packed form at all.
///
/// The lane is the output channel, so the output pitch is the channel count and
/// a pixel's taps are `SW·IC` apart. Four pixels is what keeps the eight
/// accumulators (four pixels × two vectors) in registers with room for the two
/// weight vectors and a broadcast.
///
/// f16 widens the pixel block's *whole tap span* into a stack buffer first —
/// `((3·SW) + (K-1)·DW + 1)·IC` elements, zeros where the padding pushes a tap
/// off the plane — so the inner loop is the f32 one. Converting per element
/// instead runs 11x more converts than the span needs (one per pixel-tap, however
/// cheap): the 7x7 stem came out at 1.3x its GEMM route's time at one thread and
/// 0.85x at twenty. A span wider than `kStemSpanCols` keeps the per-element path.
///
/// The gate is `IC ≤ 4 && KH == KW ∈ {3,5,7}` (see conv2d_implicit_gemm.cpp):
/// the gather copies karea elements per pixel *on top of* the FMAs, so the wider
/// the filter the more there is to skip — a 7x7 over three channels is the
/// ResNet stem, the widest filter any of the fixtures has.
///
/// A grouped, residual or multi-pass-epilogue conv takes the GEMM route, since
/// the accumulator seed and the clamp only exist here because they are free.

#include "backend/cpu/common/common_utils.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "nnops/detail/half.hpp"

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace nnops::kernel {

/// The widest tap span, in columns, the f16 route stages: four output pixels at
/// a stride of 2 under a 7x7 reach 13 columns, and a stride or dilation of 4
/// still fits. Past it the per-element path takes over.
constexpr int kStemSpanCols = 24;

/// Eight output channels as f32 lanes, whatever the plane's element type is.
template <typename T>
inline simd::v_f32x8 stem_load_oc(const T* p)
{
    if constexpr (std::is_same_v<T, half>) { return simd::v_cvt_f16_to_f32(p); }
    else { return simd::v_load(p); }
}

/// One input element, widened. Only the fallback path takes a tap scalar: x86's
/// `half_to_float` is the MLAS software routine, and even F16C's single
/// instruction cannot pay for one convert per (pixel, tap).
template <typename T>
inline float stem_scalar(const T* p)
{
    if constexpr (std::is_same_v<T, half>) {
#if defined(__F16C__)
        return _mm_cvtss_f32(_mm_cvtph_ps(
            _mm_cvtsi32_si128(static_cast<int>(half_to_bits(*p)))));
#else
        return ::nnops::backend::cpu::half_to_float(*p);
#endif
    } else {
        return *p;
    }
}

/// Store f32 accumulators into an element of either type.
template <typename T>
inline void stem_store(T* p, const simd::v_f32x8& v)
{
    if constexpr (std::is_same_v<T, half>) {
        simd::v_store(p, simd::v_cvt_f32_to_f16(v));
    } else {
        simd::v_store(p, v);
    }
}

/// One pixel's tap from the plane: broadcast the input, two vectors of output
/// channels. `c` is the tap *column*, so the element is `c·IC + ic`
/// (channel-last).
template <typename T, int IC, bool Fast>
inline void stem_px(simd::v_f32x8& a0, simd::v_f32x8& a1,
                    const T* NNOPS_RESTRICT r, int64_t c, int64_t iw, int ic,
                    const simd::v_f32x8& w0, const simd::v_f32x8& w1) noexcept
{
    const float xv = Fast ? stem_scalar(r + c * IC + ic)
                          : ((c >= 0 && c < iw) ? stem_scalar(r + c * IC + ic) : 0.0f);
    const simd::v_f32x8 xb = simd::v_set1_f32x8(xv);
    a0 = simd::v_fmadd(xb, w0, a0);
    a1 = simd::v_fmadd(xb, w1, a1);
}

/// The same tap out of the staged span: already widened, already zeroed for the
/// padding, so no test and no convert.
template <int IC>
inline void stem_px_s(simd::v_f32x8& a0, simd::v_f32x8& a1,
                      const float* NNOPS_RESTRICT s, int ic,
                      const simd::v_f32x8& w0, const simd::v_f32x8& w1) noexcept
{
    const simd::v_f32x8 xb = simd::v_set1_f32x8(s[ic]);
    a0 = simd::v_fmadd(xb, w0, a0);
    a1 = simd::v_fmadd(xb, w1, a1);
}

/// Four pixels over one vector pair of output channels. Rows of the filter that
/// the padding pushes off the plane arrive as a null row and are skipped; a block
/// whose tap columns are all in bounds (`Fast`) drops the per-pixel test. The
/// weight of one tap is `OC` past the last (HWIO is tap-major), so the walk needs
/// no index arithmetic, and the four pixels are written out so their accumulators
/// are indexed by a constant and stay in registers.
template <typename T, int IC, int K, bool Fast>
inline void stem_taps(simd::v_f32x8 (&acc)[4][2],
                      const T* const (&rows)[K], const T* NNOPS_RESTRICT w,
                      int64_t OC, int64_t oc0, int64_t ow0,
                      int SW, int PW, int DW, int64_t IW) noexcept
{
    const T* wp = w + oc0;
    for (int kh = 0; kh < K; ++kh) {
        const T* const r = rows[kh];
        if (r == nullptr) { wp += K * IC * OC; continue; }
        for (int kw = 0; kw < K; ++kw) {
            const int64_t c = ow0 * SW - PW + kw * DW;
            for (int ic = 0; ic < IC; ++ic) {
                const simd::v_f32x8 w0 = stem_load_oc(wp);
                const simd::v_f32x8 w1 = stem_load_oc(wp + 8);
                stem_px<T, IC, Fast>(acc[0][0], acc[0][1], r, c, IW, ic, w0, w1);
                stem_px<T, IC, Fast>(acc[1][0], acc[1][1], r, c + SW, IW, ic, w0, w1);
                stem_px<T, IC, Fast>(acc[2][0], acc[2][1], r, c + 2 * SW, IW, ic, w0, w1);
                stem_px<T, IC, Fast>(acc[3][0], acc[3][1], r, c + 3 * SW, IW, ic, w0, w1);
                wp += OC;
            }
        }
    }
}

/// The staged twin. `span[kh]` starts at the block's leftmost tap column, so the
/// k-th filter column is `kw·DW·IC` in and the j-th pixel `j·SW·IC` further.
template <typename T, int IC, int K>
inline void stem_taps_s(simd::v_f32x8 (&acc)[4][2],
                        const float* const (&span)[K], const T* NNOPS_RESTRICT w,
                        int64_t OC, int64_t oc0, int SW, int DW) noexcept
{
    const T* wp = w + oc0;
    for (int kh = 0; kh < K; ++kh) {
        const float* const s = span[kh];
        if (s == nullptr) { wp += K * IC * OC; continue; }
        for (int kw = 0; kw < K; ++kw) {
            const float* const sc = s + kw * DW * IC;
            for (int ic = 0; ic < IC; ++ic) {
                const simd::v_f32x8 w0 = stem_load_oc(wp);
                const simd::v_f32x8 w1 = stem_load_oc(wp + 8);
                stem_px_s<IC>(acc[0][0], acc[0][1], sc, ic, w0, w1);
                stem_px_s<IC>(acc[1][0], acc[1][1], sc + SW * IC, ic, w0, w1);
                stem_px_s<IC>(acc[2][0], acc[2][1], sc + 2 * SW * IC, ic, w0, w1);
                stem_px_s<IC>(acc[3][0], acc[3][1], sc + 3 * SW * IC, ic, w0, w1);
                wp += OC;
            }
        }
    }
}

/// One output row. `bias` may be null; the bias seeds the accumulator, and a
/// pure Relu/Relu6 is the store's clamp — the same two savings the MMA route
/// makes, here for free.
template <typename T, int IC, int K>
void stem_conv_row(T* NNOPS_RESTRICT out, const T* NNOPS_RESTRICT in,
                   const T* NNOPS_RESTRICT w, const T* NNOPS_RESTRICT bias,
                   int64_t IH, int64_t IW, int64_t OW, int64_t OC,
                   int64_t in_rs, int64_t out_rs, int64_t oh,
                   int SH, int SW, int PH, int PW, int DH, int DW,
                   bool clamp, float lo, float hi) noexcept
{
    constexpr int64_t L = simd::simd_default_lane_f32;
    constexpr bool kStage = std::is_same_v<T, half>;
    constexpr int kSpan = kStemSpanCols * IC;

    const simd::v_f32x8 vlo = simd::v_set1_f32x8(lo);
    const simd::v_f32x8 vhi = simd::v_set1_f32x8(hi);

    T* const orow = out + oh * out_rs;
    const int64_t top = oh * SH - PH;

    const T* rows[K];
    for (int kh = 0; kh < K; ++kh) {
        const int64_t ih = top + kh * DH;
        rows[kh] = (ih >= 0 && ih < IH) ? in + ih * in_rs : nullptr;
    }

    float sx[kStage ? K * kSpan : 1];
    const float* span[kStage ? K : 1];

    for (int64_t ow0 = 0; ow0 < OW; ow0 += 4) {
        const int64_t px = std::min<int64_t>(4, OW - ow0);
        // Every tap column of a full block in bounds (a sufficient test: the
        // block spans `c0` to `c1`, and stride and dilation only reach further).
        const int64_t c0 = ow0 * SW - PW;
        const int64_t c1 = (ow0 + px - 1) * SW - PW + (K - 1) * DW;
        const bool fast = px == 4 && c0 >= 0 && c1 < IW;

        bool staged = false;
        if constexpr (kStage) {
            staged = c1 - c0 < kStemSpanCols;
            if (staged) {
                const int64_t cols = c1 - c0 + 1;
                const int64_t lo = std::max<int64_t>(c0, 0);
                const int64_t hi = std::min<int64_t>(c1 + 1, IW);
                for (int kh = 0; kh < K; ++kh) {
                    float* const d = sx + kh * kSpan;
                    span[kh] = rows[kh] ? d : nullptr;
                    if (rows[kh] == nullptr) { continue; }
                    std::fill(d, d + cols * IC, 0.0f);
                    const T* const src = rows[kh] + lo * IC;
                    float* const dst = d + (lo - c0) * IC;
                    for (int64_t i = 0, e = (hi - lo) * IC; i < e; ++i) {
                        dst[i] = stem_scalar(src + i);
                    }
                }
            }
        }

        for (int64_t oc0 = 0; oc0 < OC; oc0 += 2 * L) {
            const simd::v_f32x8 b0 = bias ? stem_load_oc(bias + oc0)
                                          : simd::v_zero_f32x8();
            const simd::v_f32x8 b1 = bias ? stem_load_oc(bias + oc0 + L)
                                          : simd::v_zero_f32x8();
            simd::v_f32x8 acc[4][2] = {{b0, b1}, {b0, b1}, {b0, b1}, {b0, b1}};

            if constexpr (kStage) {
                if (staged) {
                    stem_taps_s<T, IC, K>(acc, span, w, OC, oc0, SW, DW);
                } else if (fast) {
                    stem_taps<T, IC, K, true>(acc, rows, w, OC, oc0, ow0,
                                              SW, PW, DW, IW);
                } else {
                    stem_taps<T, IC, K, false>(acc, rows, w, OC, oc0, ow0,
                                               SW, PW, DW, IW);
                }
            } else if (fast) {
                stem_taps<T, IC, K, true>(acc, rows, w, OC, oc0, ow0, SW, PW, DW, IW);
            } else {
                stem_taps<T, IC, K, false>(acc, rows, w, OC, oc0, ow0, SW, PW, DW, IW);
            }

            for (int64_t j = 0; j < px; ++j) {
                T* const d = orow + (ow0 + j) * OC + oc0;
                simd::v_f32x8 a0 = acc[j][0];
                simd::v_f32x8 a1 = acc[j][1];
                if (clamp) {
                    a0 = simd::v_max(simd::v_min(a0, vhi), vlo);
                    a1 = simd::v_max(simd::v_min(a1, vhi), vlo);
                }
                stem_store(d, a0);
                stem_store(d + L, a1);
            }
        }
    }
}

}  // namespace nnops::kernel
