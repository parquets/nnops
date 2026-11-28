#pragma once
/// @file simd_stem_conv.hpp
/// @brief The tiny filter: a square 3x3/5x5/7x7 over at most four channels,
/// convoluted in place.
///
/// A wide filter over at most four input channels cannot pay for a gather: the
/// GEMM route copies `karea·IC` elements per pixel into an im2col tile only to
/// read them back, and at the 3-channel stem that copy is most of the layer's
/// time. The patch is already in place on the plane, so convolve it
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
/// f16 has two arms. Where the backend has a native eight-lane f16 FMA
/// (AArch64), the lane is still the output channel and the accumulator, the
/// weight vector and the broadcast are all the plane's own type: a weight
/// vector is one 16-byte load instead of two widened ones, a tap is one
/// broadcast, and eight channels per register means the same FMA does twice the
/// work — the f16 stem then caps at the f16 instead of the f32 FMA rate. The
/// accumulator rounds to f16 exactly like this library's f16 GEMM route does;
/// its error budget is `kF16AccumTol`, not the f32 route's.
///
/// Elsewhere (x86 has no f16 FMA on F16C) f16 widens the pixel block's *whole
/// tap span* into a stack buffer first — `((3·SW) + (K-1)·DW + 1)·IC`
/// elements, zeros where the padding pushes a tap off the plane — so the inner
/// loop is the f32 one. Converting per element instead runs 11x more converts
/// than the span needs (one per pixel-tap, however cheap): the 7x7 stem came out
/// at 1.3x its GEMM route's time at one thread and 0.85x at twenty. A span wider
/// than `kStemSpanCols` keeps the per-element path.
///
/// The gate is `IC ≤ 4 && KH == KW ∈ {3,5,7}` (see conv2d_implicit_gemm.cpp):
/// the gather copies karea elements per pixel *on top of* the FMAs, so the wider
/// the filter the more there is to skip — a 7x7 over three channels is the
/// ResNet stem, the widest filter any of the fixtures has.
///
/// A grouped, residual or multi-pass-epilogue conv takes the GEMM route, since
/// the accumulator seed and the clamp only exist here because they are free.
///
/// The quantized twin at the bottom of this file is the same shape of answer
/// for an s8/u8 plane: the tap update becomes a four-lane dot product, the
/// filter is packed once per call, and the requantizing store is fused. It is
/// AArch64-only (the dot is `vdotq_s32`) and defers to the SME GEMM where one
/// exists, like the float stem.

#include "backend/cpu/common/common_utils.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "nnops/detail/half.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <type_traits>

#if defined(NNOPS_ARCH_AARCH64)
#include <arm_neon.h>
#endif

namespace nnops::kernel {

/// The widest tap span, in columns, the f16 route stages: four output pixels at
/// a stride of 2 under a 7x7 reach 13 columns, and a stride or dilation of 4
/// still fits. Past it the per-element path takes over.
constexpr int kStemSpanCols = 24;

/// Whether this backend has a native eight-lane f16 FMA. NEON does (`fmla.8h`
/// at the baseline AArch64 target), so f16 lanes are the plane's own type
/// there; x86's F16C only converts, so its f16 arm widens to f32 and stays.
#if defined(NNOPS_ARCH_AARCH64)
inline constexpr bool kStemF16Native = true;
#else
inline constexpr bool kStemF16Native = false;
#endif

/// Eight output channels as f32 lanes, whatever the plane's element type is.
template <typename T> inline simd::v_f32x8 stem_load_oc(const T* p)
{
    if constexpr (std::is_same_v<T, half>) {
        return simd::v_cvt_f16_to_f32(p);
    } else {
        return simd::v_load(p);
    }
}

/// One input element, widened. Only the fallback path takes a tap scalar: x86's
/// `half_to_float` is the MLAS software routine, and even F16C's single
/// instruction cannot pay for one convert per (pixel, tap).
template <typename T> inline float stem_scalar(const T* p)
{
    if constexpr (std::is_same_v<T, half>) {
#if defined(__F16C__)
        return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(half_to_bits(*p)))));
#else
        return ::nnops::backend::cpu::half_to_float(*p);
#endif
    } else {
        return *p;
    }
}

/// Store f32 accumulators into an element of either type.
template <typename T> inline void stem_store(T* p, const simd::v_f32x8& v)
{
    if constexpr (std::is_same_v<T, half>) {
        simd::v_store(p, simd::v_cvt_f32_to_f16(v));
    } else {
        simd::v_store(p, v);
    }
}

/// One pixel's tap from the plane: broadcast the input, two vectors of output
/// channels. `c` is the tap *column*, so the element is `c·IC + ic`
/// (one channel vector at a time).
template <typename T, int IC, bool Fast>
inline void stem_px(simd::v_f32x8& a0, simd::v_f32x8& a1, const T* NNOPS_RESTRICT r, int64_t c,
                    int64_t iw, int ic, const simd::v_f32x8& w0, const simd::v_f32x8& w1) noexcept
{
    const float xv = Fast ? stem_scalar(r + c * IC + ic) :
                            ((c >= 0 && c < iw) ? stem_scalar(r + c * IC + ic) : 0.0f);
    const simd::v_f32x8 xb = simd::v_set1_f32x8(xv);
    a0 = simd::v_fmadd(xb, w0, a0);
    a1 = simd::v_fmadd(xb, w1, a1);
}

/// The same tap out of the staged span: already widened, already zeroed for the
/// padding, so no test and no convert.
template <int IC>
inline void stem_px_s(simd::v_f32x8& a0, simd::v_f32x8& a1, const float* NNOPS_RESTRICT s, int ic,
                      const simd::v_f32x8& w0, const simd::v_f32x8& w1) noexcept
{
    const simd::v_f32x8 xb = simd::v_set1_f32x8(s[ic]);
    a0 = simd::v_fmadd(xb, w0, a0);
    a1 = simd::v_fmadd(xb, w1, a1);
}

/// One plane element as eight f16 lanes. `s_load` widens and `v_set1_f16x8`
/// narrows back; the round trip is exact (every f16 is an f32) and clang folds
/// it into the single LD1R a broadcast should cost, which is what lets the
/// native arm read its taps straight off the plane instead of staging a span.
inline simd::v_f16x8 stem_bcast_h(const half* p) noexcept
{
    return simd::v_set1_f16x8(simd::s_load(p));
}

/// One pixel's tap in f16 lanes: broadcast the input, NV vectors of output
/// channels (eight lanes each).
template <typename T, int IC, int NV, bool Fast>
NNOPS_FORCEINLINE void stem_px_h(simd::v_f16x8 (&a)[NV], const T* NNOPS_RESTRICT r, int64_t c,
                                 int64_t iw, int ic, const simd::v_f16x8 (&w)[NV]) noexcept
{
    const simd::v_f16x8 xb = Fast ? stem_bcast_h(r + c * IC + ic)
                                  : ((c >= 0 && c < iw) ? stem_bcast_h(r + c * IC + ic)
                                                        : simd::v_zero_f16x8());
    for (int v = 0; v < NV; ++v) {
        a[v] = simd::v_fmadd(xb, w[v], a[v]);
    }
}

/// The f16 twin of `stem_taps`, over NV vectors of output channels and PB
/// pixels. NV is 4 (32 channels) where the register budget allows it, and 2 at
/// IC 4: the tap loop's `ic` is unrolled, so a wide pass keeps IC·NV weight
/// vectors live alongside the PB·NV accumulators and the allocator spills them —
/// IC 4 at NV 4 measured 0.47x of ACL against NV 2's 1.11x, while IC 3 keeps
/// NV 4 profitable.
///
/// PB is the second lever: one weight load and one tap walk serve PB pixels, so
/// the load-to-FMA ratio falls as the block grows. IC 4 cannot afford NV 4's
/// registers at four pixels, but eight pixels at NV 2 does fit (16 accumulators
/// and 8 weights), which is what the row uses there.
template <typename T, int IC, int K, int NV, bool Fast, int PB>
NNOPS_FORCEINLINE void stem_taps_h(simd::v_f16x8 (&acc)[PB][NV], const T* const (&rows)[K],
                                   const T* NNOPS_RESTRICT w, int64_t OC, int64_t oc0,
                                   int64_t ow0, int SW, int PW, int DW, int64_t IW) noexcept
{
    const T* wp = w + oc0;
    for (int kh = 0; kh < K; ++kh) {
        const T* const r = rows[kh];
        if (r == nullptr) {
            wp += K * IC * OC;
            continue;
        }
        for (int kw = 0; kw < K; ++kw) {
            const int64_t c = ow0 * SW - PW + kw * DW;
            for (int ic = 0; ic < IC; ++ic) {
                simd::v_f16x8 wv[NV];
                for (int v = 0; v < NV; ++v) {
                    wv[v] = simd::v_load(wp + 8 * v);
                }
#pragma unroll
                for (int j = 0; j < PB; ++j) {
                    stem_px_h<T, IC, NV, Fast>(acc[j], r, c + j * SW, IW, ic, wv);
                }
                wp += OC;
            }
        }
    }
}

#if defined(NNOPS_ARCH_AARCH64)

/// Four contiguous f16 channels as one eight-byte load, and the lane-indexed
/// FMA that consumes it. IC 4's four taps per pixel are then one load and four
/// lane FMAs instead of four broadcasts and four register FMAs — the walk is
/// issue-bound, so the three saved instructions per (pixel, tap) are the win.
inline float16x4_t stem_load_h4(const half* p) noexcept
{
    return vld1_f16(reinterpret_cast<const __fp16*>(p));
}

template <int L>
NNOPS_FORCEINLINE simd::v_f16x8 stem_fma_lane_h(const simd::v_f16x8& w, float16x4_t x,
                                                const simd::v_f16x8& acc) noexcept
{
    return simd::v_f16x8(vfmaq_lane_f16(acc.val, w.val, x, L));
}

/// The IC-4 tap walk: load the four input channels of a pixel once, then fold
/// them into the accumulators by lane. The k order per accumulator is the same
/// (kh, kw, ic 0..3), so the result is bit-identical to `stem_taps_h`.
template <typename T, int K, int NV, bool Fast, int PB>
NNOPS_FORCEINLINE void stem_taps4_h(simd::v_f16x8 (&acc)[PB][NV],
                                    const T* const (&rows)[K], const T* NNOPS_RESTRICT w,
                                    int64_t OC, int64_t oc0, int64_t ow0, int SW, int PW,
                                    int DW, int64_t IW) noexcept
{
    const T* wp = w + oc0;
    for (int kh = 0; kh < K; ++kh) {
        const T* const r = rows[kh];
        if (r == nullptr) {
            wp += K * 4 * OC;
            continue;
        }
        for (int kw = 0; kw < K; ++kw) {
            const int64_t c = ow0 * SW - PW + kw * DW;
            simd::v_f16x8 wv[4][NV];
            for (int ic = 0; ic < 4; ++ic) {
                for (int v = 0; v < NV; ++v) {
                    wv[ic][v] = simd::v_load(wp + 8 * v);
                }
                wp += OC;
            }
#pragma unroll
            for (int j = 0; j < PB; ++j) {
                const int64_t cj = c + j * SW;
                const bool ok = Fast || (cj >= 0 && cj < IW);
                const float16x4_t x =
                    ok ? stem_load_h4(r + cj * 4) : vdup_n_f16(0.0f);
                for (int v = 0; v < NV; ++v) {
                    acc[j][v] = stem_fma_lane_h<0>(wv[0][v], x, acc[j][v]);
                    acc[j][v] = stem_fma_lane_h<1>(wv[1][v], x, acc[j][v]);
                    acc[j][v] = stem_fma_lane_h<2>(wv[2][v], x, acc[j][v]);
                    acc[j][v] = stem_fma_lane_h<3>(wv[3][v], x, acc[j][v]);
                }
            }
        }
    }
}

#endif // NNOPS_ARCH_AARCH64

/// One pass of the native-f16 row over 8·NV channels: seed from the bias, walk
/// the taps, clamp and store — every step in the plane's own half lanes, so the
/// epilogue costs no conversions either.
template <typename T, int IC, int K, int NV, int PB>
NNOPS_FORCEINLINE void stem_pass_h(T* NNOPS_RESTRICT orow, const T* const (&rows)[K],
                                   const T* NNOPS_RESTRICT w, const T* NNOPS_RESTRICT bias,
                                   int64_t OC, int64_t oc0, int64_t ow0, int64_t px, int SW,
                                   int PW, int DW, int64_t IW, bool fast, bool clamp,
                                   const simd::v_f16x8& vlo, const simd::v_f16x8& vhi) noexcept
{
    simd::v_f16x8 acc[PB][NV];
    for (int v = 0; v < NV; ++v) {
        const simd::v_f16x8 b = bias ? simd::v_load(bias + oc0 + 8 * v) : simd::v_zero_f16x8();
        for (int j = 0; j < PB; ++j) {
            acc[j][v] = b;
        }
    }
    if constexpr (IC == 4) {
#if defined(NNOPS_ARCH_AARCH64)
        if (fast) {
            stem_taps4_h<T, K, NV, true, PB>(acc, rows, w, OC, oc0, ow0, SW, PW, DW, IW);
        } else {
            stem_taps4_h<T, K, NV, false, PB>(acc, rows, w, OC, oc0, ow0, SW, PW, DW, IW);
        }
#else
        if (fast) {
            stem_taps_h<T, IC, K, NV, true, PB>(acc, rows, w, OC, oc0, ow0, SW, PW, DW, IW);
        } else {
            stem_taps_h<T, IC, K, NV, false, PB>(acc, rows, w, OC, oc0, ow0, SW, PW, DW, IW);
        }
#endif
    } else {
        if (fast) {
            stem_taps_h<T, IC, K, NV, true, PB>(acc, rows, w, OC, oc0, ow0, SW, PW, DW, IW);
        } else {
            stem_taps_h<T, IC, K, NV, false, PB>(acc, rows, w, OC, oc0, ow0, SW, PW, DW, IW);
        }
    }
    for (int64_t j = 0; j < px; ++j) {
        T* const d = orow + (ow0 + j) * OC + oc0;
        for (int v = 0; v < NV; ++v) {
            simd::v_f16x8 a = acc[j][v];
            if (clamp) {
                a = simd::v_max(simd::v_min(a, vhi), vlo);
            }
            simd::v_store(d + 8 * v, a);
        }
    }
}

/// Four pixels over one vector pair of output channels. Rows of the filter that
/// the padding pushes off the plane arrive as a null row and are skipped; a block
/// whose tap columns are all in bounds (`Fast`) drops the per-pixel test. The
/// weight of one tap is `OC` past the last (HWIO is tap-major), so the walk needs
/// no index arithmetic, and the four pixels are written out so their accumulators
/// are indexed by a constant and stay in registers.
template <typename T, int IC, int K, bool Fast>
inline void stem_taps(simd::v_f32x8 (&acc)[4][2], const T* const (&rows)[K],
                      const T* NNOPS_RESTRICT w, int64_t OC, int64_t oc0, int64_t ow0, int SW,
                      int PW, int DW, int64_t IW) noexcept
{
    const T* wp = w + oc0;
    for (int kh = 0; kh < K; ++kh) {
        const T* const r = rows[kh];
        if (r == nullptr) {
            wp += K * IC * OC;
            continue;
        }
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
inline void stem_taps_s(simd::v_f32x8 (&acc)[4][2], const float* const (&span)[K],
                        const T* NNOPS_RESTRICT w, int64_t OC, int64_t oc0, int SW, int DW) noexcept
{
    const T* wp = w + oc0;
    for (int kh = 0; kh < K; ++kh) {
        const float* const s = span[kh];
        if (s == nullptr) {
            wp += K * IC * OC;
            continue;
        }
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
void stem_conv_row(T* NNOPS_RESTRICT out, const T* NNOPS_RESTRICT in, const T* NNOPS_RESTRICT w,
                   const T* NNOPS_RESTRICT bias, int64_t IH, int64_t IW, int64_t OW, int64_t OC,
                   int64_t in_rs, int64_t out_rs, int64_t oh, int SH, int SW, int PH, int PW,
                   int DH, int DW, bool clamp, float lo, float hi) noexcept
{
    constexpr int64_t L = simd::simd_default_lane_f32;
    constexpr bool kF16Native = std::is_same_v<T, half> && kStemF16Native;
    constexpr bool kStage = std::is_same_v<T, half> && !kF16Native;
    constexpr int kSpan = kStemSpanCols * IC;

    [[maybe_unused]] const simd::v_f32x8 vlo = simd::v_set1_f32x8(lo);
    [[maybe_unused]] const simd::v_f32x8 vhi = simd::v_set1_f32x8(hi);

    T* const orow = out + oh * out_rs;
    const int64_t top = oh * SH - PH;

    const T* rows[K];
    for (int kh = 0; kh < K; ++kh) {
        const int64_t ih = top + kh * DH;
        rows[kh] = (ih >= 0 && ih < IH) ? in + ih * in_rs : nullptr;
    }

    float sx[kStage ? K * kSpan : 1];
    const float* span[kStage ? K : 1];

    // Eight pixels per block where the native f16 arm at IC 4 can hold them
    // (16 accumulators, 8 weights); every other arm keeps four.
    constexpr int kPB = (kF16Native && IC == 4) ? 8 : 4;
    for (int64_t ow0 = 0; ow0 < OW; ow0 += kPB) {
        const int64_t px = std::min<int64_t>(kPB, OW - ow0);
        // Every tap column of a full block in bounds (a sufficient test: the
        // block spans `c0` to `c1`, and stride and dilation only reach further).
        const int64_t c0 = ow0 * SW - PW;
        const int64_t c1 = (ow0 + px - 1) * SW - PW + (K - 1) * DW;
        const bool fast = px == kPB && c0 >= 0 && c1 < IW;

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
                    if (rows[kh] == nullptr) {
                        continue;
                    }
                    std::fill(d, d + cols * IC, 0.0f);
                    const T* const src = rows[kh] + lo * IC;
                    float* const dst = d + (lo - c0) * IC;
                    for (int64_t i = 0, e = (hi - lo) * IC; i < e; ++i) {
                        dst[i] = stem_scalar(src + i);
                    }
                }
            }
        }

        if constexpr (kF16Native) {
            const simd::v_f16x8 hlo = simd::v_set1_f16x8(lo);
            const simd::v_f16x8 hhi = simd::v_set1_f16x8(hi);
            int64_t oc0 = 0;
            // 32 channels per pass where the registers hold them (see stem_taps_h);
            // the 16-wide remainder, and IC 4, take two vectors. OC is a multiple
            // of 16, so 16 is the only remainder there is.
            if constexpr (IC <= 3) {
                for (; oc0 + 32 <= OC; oc0 += 32) {
                    stem_pass_h<T, IC, K, 4, 4>(orow, rows, w, bias, OC, oc0, ow0, px, SW, PW,
                                                DW, IW, fast, clamp, hlo, hhi);
                }
            }
            for (; oc0 < OC; oc0 += 16) {
                if constexpr (IC == 4) {
                    stem_pass_h<T, IC, K, 2, 8>(orow, rows, w, bias, OC, oc0, ow0, px, SW, PW,
                                                DW, IW, fast, clamp, hlo, hhi);
                } else {
                    stem_pass_h<T, IC, K, 2, 4>(orow, rows, w, bias, OC, oc0, ow0, px, SW, PW,
                                                DW, IW, fast, clamp, hlo, hhi);
                }
            }
        } else {
            for (int64_t oc0 = 0; oc0 < OC; oc0 += 2 * L) {
                const simd::v_f32x8 b0 = bias ? stem_load_oc(bias + oc0) : simd::v_zero_f32x8();
                const simd::v_f32x8 b1 = bias ? stem_load_oc(bias + oc0 + L) : simd::v_zero_f32x8();
                simd::v_f32x8 acc[4][2] = {{b0, b1}, {b0, b1}, {b0, b1}, {b0, b1}};

                if constexpr (kStage) {
                    if (staged) {
                        stem_taps_s<T, IC, K>(acc, span, w, OC, oc0, SW, DW);
                    } else if (fast) {
                        stem_taps<T, IC, K, true>(acc, rows, w, OC, oc0, ow0, SW, PW, DW, IW);
                    } else {
                        stem_taps<T, IC, K, false>(acc, rows, w, OC, oc0, ow0, SW, PW, DW, IW);
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
}

#if defined(NNOPS_ARCH_AARCH64)

// =========================================================================
//  The quantized twin: an s8/u8 tiny filter, requantized on the way out
// =========================================================================
//
// Same shape of answer as the float kernel above — the filter is tiny, the
// patch is already in place, so read it in place — but the arithmetic is a
// dot product: four taps at a time against four output channels
// (`vdotq_s32`), which is what SDOT makes of a rank-1 tap update. The
// gather the GEMM route pays for (karea·IC bytes per pixel, then an A pack)
// is the layer's whole cost at three channels, so this route skips both.
//
// The filter is packed once per call into the layout the dot wants:
//     wpack[(kg·OC + oc)·4 + j]  for k = kg·4 + j, taps in (kh, kw, ic) order
// i.e. four k values contiguous per output channel, four output channels a
// 16-byte operand. `r_b[oc] = Σ_k w[oc,k]` comes out of the same pass — the
// zero-point withdrawal is then one multiply per output channel.
//
// The activation bytes are the ones the GEMM gather would feed (a u8 input
// XOR-shifted to signed), so `coef = d + zp_x` is the epilogue's `coef_rb`
// and a padding tap reads `coef` — contributing exactly zero to the
// compensated accumulator, as the gather's padded fill does.

/// k groups a packed stem filter occupies: k = K·K·IC rounded up to four.
template <int IC, int K> inline constexpr int stem_i8_groups() noexcept
{
    return (K * K * IC + 3) / 4;
}

/// Pack the tiny s8 filter into the dot layout and sum each output channel's
/// k column into @p r_b. @p w is HWIO: one (tap, input channel) is one
/// OC-length run.
template <int IC, int K>
inline void stem_pack_w_i8(int8_t* NNOPS_RESTRICT dst, const int8_t* NNOPS_RESTRICT w,
                           int64_t w_tap_stride, int64_t w_ic_stride, int64_t OC,
                           int32_t* NNOPS_RESTRICT r_b) noexcept
{
    constexpr int kK = K * K * IC;
    constexpr int kG = stem_i8_groups<IC, K>();
    std::fill(r_b, r_b + OC, 0);
    for (int g = 0; g < kG; ++g) {
        for (int j = 0; j < 4; ++j) {
            const int kk = g * 4 + j;
            int8_t* const col = dst + (static_cast<int64_t>(g) * OC) * 4 + j;
            if (kk >= kK) {
                for (int64_t oc = 0; oc < OC; ++oc) {
                    col[oc * 4] = 0;
                }
                continue;
            }
            const int t = kk / IC;
            const int i = kk - t * IC;
            const int8_t* const src = w + t * w_tap_stride + i * w_ic_stride;
            for (int64_t oc = 0; oc < OC; ++oc) {
                const int8_t v = src[oc];
                col[oc * 4] = v;
                r_b[oc] += v;
            }
        }
    }
}

/// One output row of the quantized tiny filter. The accumulator is the raw
/// Σ a·w in int32; the epilogue adds the precombined `bias - coef·r_b`,
/// applies the Relu, requantizes and saturates — the same chain, in the same
/// order, as int8_epilogue_row_quant, so the two routes agree bit for bit.
///
/// @p bias_adj and @p scale are folded once per call (`try_stem_i8`): the
/// integer add is associative and stays inside int32, and `scale` is the same
/// `row_k·s_b` product the row epilogue forms per vector, so folding changes
/// no byte. Both are indexed by output channel and never null.
template <class TAct, class TOut, int IC, int K>
void stem_conv_row_i8(TOut* NNOPS_RESTRICT orow, const TAct* NNOPS_RESTRICT in, int64_t in_rs,
                      const int8_t* NNOPS_RESTRICT wpack,
                      const int32_t* NNOPS_RESTRICT bias_adj,
                      const float* NNOPS_RESTRICT scale, int64_t IH, int64_t IW, int64_t OW,
                      int64_t OC, int64_t oh, int SH, int SW, int PH, int PW, int DH, int DW,
                      int8_t pad, int32_t zp_out, bool relu) noexcept
{
    constexpr bool kU8In = std::is_same_v<TAct, uint8_t>;
    constexpr bool kU8Out = std::is_same_v<TOut, uint8_t>;
    constexpr int kK = K * K * IC;
    constexpr int kG = stem_i8_groups<IC, K>();
    constexpr float kMin = kU8Out ? 0.0f : -128.0f;
    constexpr float kMax = kU8Out ? 255.0f : 127.0f;

    const float32x4_t vmin = vdupq_n_f32(kMin - static_cast<float>(zp_out));
    const float32x4_t vmax = vdupq_n_f32(kMax - static_cast<float>(zp_out));
    const int32x4_t vzp = vdupq_n_s32(zp_out);
    const int32x4_t vzero = vdupq_n_s32(0);

    // One block of NG four-channel groups: accumulate every k group, then the
    // epilogue. The A word is built once per k group and reused by all NG.
    const auto block = [&](int64_t ow, int64_t oc0, const std::array<uint32_t, kG>& word,
                           auto ng) {
        constexpr int NG = decltype(ng)::value;
        int32x4_t acc[NG];
        for (int q = 0; q < NG; ++q) {
            acc[q] = vzero;
        }
        for (int g = 0; g < kG; ++g) {
            const int8x16_t va =
                vreinterpretq_s8_s32(vdupq_n_s32(static_cast<int32_t>(word[static_cast<size_t>(g)])));
            for (int q = 0; q < NG; ++q) {
                const int8x16_t vw =
                    vld1q_s8(wpack + (static_cast<int64_t>(g) * OC + oc0 + 4 * q) * 4);
                acc[q] = vdotq_s32(acc[q], va, vw);
            }
        }
        // Quantize each four-channel group to its final s8/u8 lanes, then
        // store two groups at a time: one eight-byte store where the first
        // cut did one four-byte store per group.
        int16x4_t lane[NG];
        for (int q = 0; q < NG; ++q) {
            const int64_t oc = oc0 + 4 * q;
            int32x4_t v = vaddq_s32(acc[q], vld1q_s32(bias_adj + oc));
            if (relu) {
                v = vmaxq_s32(v, vzero);
            }
            const float32x4_t f0 = vcvtq_f32_s32(v);
            const float32x4_t f = vmulq_f32(f0, vld1q_f32(scale + oc));
            // FCVTAS: nearest, ties away from zero. With a zero output zero
            // point the saturating narrow is the clamp, exactly as in the row
            // epilogue.
            const int32x4_t qv =
                zp_out != 0 ? vaddq_s32(vcvtaq_s32_f32(vminq_f32(vmaxq_f32(f, vmin), vmax)), vzp) :
                              vcvtaq_s32_f32(f);
            lane[q] = vqmovn_s32(qv);
        }
        TOut* const d = orow + ow * OC + oc0;
        int q = 0;
        for (; q + 2 <= NG; q += 2) {
            const int16x8_t i16 = vcombine_s16(lane[q], lane[q + 1]);
            if constexpr (kU8Out) {
                vst1_u8(reinterpret_cast<uint8_t*>(d + 4 * q), vqmovun_s16(i16));
            } else {
                vst1_s8(reinterpret_cast<int8_t*>(d + 4 * q), vqmovn_s16(i16));
            }
        }
        if constexpr ((NG & 1) != 0) {
            const int16x4_t i16 = lane[NG - 1];
            TOut* const t = d + 4 * (NG - 1);
            if constexpr (kU8Out) {
                const uint8x8_t v8 = vqmovun_s16(vcombine_s16(i16, i16));
                vst1_lane_u32(reinterpret_cast<uint32_t*>(t), vreinterpret_u32_u8(v8), 0);
            } else {
                const int8x8_t v8 = vqmovn_s16(vcombine_s16(i16, i16));
                vst1_lane_s32(reinterpret_cast<int32_t*>(t), vreinterpret_s32_s8(v8), 0);
            }
        }
    };

    const int64_t top = oh * SH - PH;
    // A whole filter's rows and its leftmost column are inside the plane: the
    // common case for a pixel, and the one that can read without a test. The
    // first and last column of the row (and every row when the filter's top
    // or bottom leaves the plane) take the checked path.
    const bool rows_in = top >= 0 && top + (K - 1) * DH < IH;
    for (int64_t ow = 0; ow < OW; ++ow) {
        std::array<uint32_t, kG> word{};
        const int64_t col = ow * SW - PW;
        const bool fast = rows_in && col >= 0 && col + (K - 1) * DW < IW;
        const auto build = [&](auto fast_tag) {
            int kk = 0;
            for (int kh = 0; kh < K; ++kh) {
                const int64_t ih = top + kh * DH;
                if constexpr (decltype(fast_tag)::value) {
                    const TAct* const row = in + ih * in_rs;
                    for (int kw = 0; kw < K; ++kw) {
                        const TAct* const p = row + (col + kw * DW) * IC;
                        for (int ic = 0; ic < IC; ++ic, ++kk) {
                            const uint8_t q = static_cast<uint8_t>(p[ic]);
                            const uint32_t b =
                                kU8In ? static_cast<uint32_t>(static_cast<uint8_t>(q ^ 0x80u)) :
                                        static_cast<uint32_t>(q);
                            word[static_cast<size_t>(kk >> 2)] |= b << (8 * (kk & 3));
                        }
                    }
                } else {
                    const uint32_t pb = static_cast<uint32_t>(static_cast<uint8_t>(pad));
                    const TAct* const row = (ih >= 0 && ih < IH) ? in + ih * in_rs : nullptr;
                    if (row == nullptr) {
                        // A whole filter row of padding: no per-tap test left.
                        for (int t = 0; t < K * IC; ++t, ++kk) {
                            word[static_cast<size_t>(kk >> 2)] |= pb << (8 * (kk & 3));
                        }
                        continue;
                    }
                    for (int kw = 0; kw < K; ++kw) {
                        const int64_t iw = col + kw * DW;
                        if (iw < 0 || iw >= IW) {
                            for (int ic = 0; ic < IC; ++ic, ++kk) {
                                word[static_cast<size_t>(kk >> 2)] |= pb << (8 * (kk & 3));
                            }
                            continue;
                        }
                        const TAct* const p = row + iw * IC;
                        for (int ic = 0; ic < IC; ++ic, ++kk) {
                            const uint8_t q = static_cast<uint8_t>(p[ic]);
                            const uint32_t b =
                                kU8In ? static_cast<uint32_t>(static_cast<uint8_t>(q ^ 0x80u)) :
                                        static_cast<uint32_t>(q);
                            word[static_cast<size_t>(kk >> 2)] |= b << (8 * (kk & 3));
                        }
                    }
                }
            }
        };
        if (fast) {
            build(std::true_type{});
        } else {
            build(std::false_type{});
        }
        int64_t oc0 = 0;
        for (; oc0 + 16 <= OC; oc0 += 16) {
            block(ow, oc0, word, std::integral_constant<int, 4>{});
        }
        const int groups = static_cast<int>((OC - oc0) / 4);
        switch (groups) {
        case 3:
            block(ow, oc0, word, std::integral_constant<int, 3>{});
            break;
        case 2:
            block(ow, oc0, word, std::integral_constant<int, 2>{});
            break;
        case 1:
            block(ow, oc0, word, std::integral_constant<int, 1>{});
            break;
        default:
            break; // OC is a multiple of four
        }
    }
}

#endif // NNOPS_ARCH_AARCH64

} // namespace nnops::kernel
