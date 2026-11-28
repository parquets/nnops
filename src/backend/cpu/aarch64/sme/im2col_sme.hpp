#pragma once
/// @file im2col_sme.hpp
/// @brief The conv routes' fused im2col + A pack, written straight into ZA.
///
/// The gathers in simd_kernel/simd_im2col.hpp are portable: they write a
/// row-major A buffer and know nothing about the engine. These two are the SME
/// counterparts of that step, and they are not gathers but *packs*: one ZA row
/// is one pixel's channel run, the K columns come out of the tile vertically,
/// and the destination is the GEMM's own A panel — so the gather, the transpose
/// and the pack are one pass. They belong next to mma_sme.hpp and
/// pack_i8_sme.hpp rather than with the portable kernels, and they exist only
/// when the build enabled SME (`NNOPS_ARCH_SME`); every entry point is called
/// behind available_sme().
///
/// Both share the tile-coordinate layer below. What differs is where validity
/// is resolved — per pixel for the float form, per row segment for the
/// quantized one — and that difference is deliberate (see each function).

#include "backend/cpu/aarch64/sme/mma_sme.hpp" // sme_mr
#include "backend/cpu/common/common_utils.hpp"

#include <algorithm>
#include <cstdint>
#include <type_traits>

#if defined(NNOPS_ARCH_SME)
#include <arm_sme.h>
#endif

namespace nnops::backend::cpu::aarch64 {

/// A flat pixel tile's first output coordinate. A tile walks the output in flat
/// order and may run past a row end, so every later pixel is this pair plus the
/// row length — no division per pixel.
inline void im2col_tile_origin(int64_t p0, int OW, int64_t& oh, int& ow) noexcept
{
    oh = p0 / OW;
    ow = static_cast<int>(p0 - oh * OW);
}

/// The per-pixel half of the tap arithmetic: pixel i's window origin in the
/// input (oy, ox) and its flat source offset at channel 0. A tap then costs only
/// `piv[i] + dy * row_stride + dx * ic_stride`, so the pixel-resolution work
/// never sits inside the tap loop.
template <int N>
inline void im2col_pix_origins(int64_t p0, int OW, int mc, int SH, int SW, int PH, int PW,
                               int64_t row_stride, int64_t ic_stride, int64_t (&oy)[N],
                               int64_t (&ox)[N], int64_t (&piv)[N]) noexcept
{
    int64_t oh;
    int ow;
    im2col_tile_origin(p0, OW, oh, ow);
    for (int i = 0; i < mc; ++i) {
        oy[i] = oh * SH - PH;
        ox[i] = ow * SW - PW;
        piv[i] = oy[i] * row_stride + ox[i] * ic_stride;
        if (++ow == OW) {
            ow = 0;
            ++oh;
        }
    }
}

/// The output columns of one tile row whose input column for this tap is inside
/// the image: iw = ow·SW + col_off lies in [0, IW) exactly for ow in [vlo, vhi).
inline void im2col_tap_cols(int col_off, int SW, int IW, int OW, int& vlo, int& vhi) noexcept
{
    vlo = col_off < 0 ? std::min(OW, (-col_off + SW - 1) / SW) : 0;
    vhi = IW - col_off > 0 ? std::min(OW, (IW - col_off + SW - 1) / SW) : 0;
}

#if defined(NNOPS_ARCH_SME)

/// The im2col gather written straight into the ZA A layout. A ZA row is one
/// pixel's channel run (NHWC, so it is contiguous) and the K columns come out
/// of the tile vertically, so the gather and the transpose are one pass
/// instead of a dense write, a ZA re-read and a second write.
///
/// f32 holds one K per lane. f16 packs a channel's K pair into one word, which
/// needs an even channel count so a pair never straddles a tap — the caller
/// checks that and falls back to the dense gather plus pack_a_f16_sme.
template <class T>
__arm_locally_streaming __arm_new("za") __attribute__((target("+sme2")))
inline void tile_im2col_pack_sme(T* dst, const T* in, int64_t in_row_stride, int64_t ic_stride,
                                 int64_t p0, int OW, int mc, int IH, int IW, int KH, int KW,
                                 int SH, int SW, int DH, int DW, int PH, int PW, int icnc,
                                 int karea) noexcept
{
    constexpr bool f32 = std::is_same_v<T, float>;
    constexpr int kRows = 2 * sme_mr; // 32 pixels: one A block
    constexpr int kChanStep = f32 ? 16 : 32;   // channels one ZA tile row spans
    const svfloat32_t vzero = svdup_n_f32(0.0f);
    int64_t base[kRows];
    bool live[kRows];
    // The pixel's output coordinate does not depend on the tap: resolve it once
    // per block instead of once per (pixel, tap). That resolution is what a
    // 3-channel 7x7 is made of -- 49 taps over 32 pixels is 1568 of them for one
    // A block, more instructions than the block's 1176 mopas -- so it costs the
    // narrow-IC shapes this path exists for.
    int64_t oy[kRows];
    int64_t ox[kRows];
    int64_t piv[kRows];
    im2col_pix_origins(p0, OW, mc, SH, SW, PH, PW, in_row_stride, ic_stride, oy, ox, piv);

    // A tap row is contiguous in memory at unit dilation, so a run of taps can
    // come in through one horizontal load instead of one per tap. That is what
    // the narrow-channel shapes need: below the row width every per-tap load
    // carries icnc of the 16 lanes, and it is the gather's instruction count,
    // not the mopas, that bounds the tile.
    const int taps_per_load = std::max(1, kChanStep / icnc);
    for (int t = 0; t < karea;) {
        const int kh = t / KW;
        const int kw = t - kh * KW;
        const int nchunk = std::min(taps_per_load, KW - kw);
        const int64_t dy = static_cast<int64_t>(kh) * DH;
        // The chunk lies wholly inside every pixel's window: no tap needs a test
        // and no lane is a zero, so the whole run is one load. Consecutive taps
        // are `ic_stride` apart, so the run is contiguous only when this block
        // covers the whole channel vector — a split channel count leaves the
        // rest of the pixel's row between two taps.
        bool wide = nchunk > 1 && DW == 1 && icnc == ic_stride;
        for (int i = 0; i < mc && wide; ++i) {
            const int64_t ih = oy[i] + dy;
            const int64_t iw = ox[i] + static_cast<int64_t>(kw) * DW;
            wide = ih >= 0 && ih < IH && iw >= 0 && iw + (nchunk - 1) * DW < IW;
        }
        if (wide) {
            const int lanes = f32 ? nchunk * icnc : (nchunk * icnc) / 2;
            const svbool_t pc = svwhilelt_b32(0, lanes);
            const int k0 = (kh * KW + kw) * icnc;
            for (int pstart = 0; pstart < mc; pstart += sme_mr) {
                const int cnt = std::min(sme_mr, mc - pstart);
                for (int i = 0; i < cnt; ++i) {
                    const int64_t off =
                        piv[pstart + i] + dy * in_row_stride + kw * ic_stride;
                    if constexpr (f32) {
                        svld1_hor_za32(0, static_cast<uint32_t>(i), pc, in + off);
                    } else {
                        svld1_hor_za32(0, static_cast<uint32_t>(i), pc,
                                       reinterpret_cast<const uint32_t*>(in) + off / 2);
                    }
                }
                const svbool_t pr = svwhilelt_b32(0, cnt);
                if constexpr (f32) {
                    for (int j = 0; j < lanes; ++j) {
                        svst1_ver_za32(0, static_cast<uint32_t>(j), pr,
                                       dst + static_cast<int64_t>(k0 + j) * kRows + pstart);
                    }
                } else {
                    auto* const dw = reinterpret_cast<uint32_t*>(dst);
                    for (int j = 0; j < lanes; ++j) {
                        svst1_ver_za32(0, static_cast<uint32_t>(j), pr,
                                       dw + static_cast<int64_t>(k0 / 2 + j) * kRows +
                                           pstart);
                    }
                }
            }
            t += nchunk;
            continue;
        }
        const int64_t dx = static_cast<int64_t>(kw) * DW;
        const int64_t toff = dy * in_row_stride + dx * ic_stride;
        for (int i = 0; i < mc; ++i) {
            const int64_t ih = oy[i] + dy;
            const int64_t iw = ox[i] + dx;
            live[i] = ih >= 0 && ih < IH && iw >= 0 && iw < IW;
            base[i] = live[i] ? piv[i] + toff : 0;
        }
        for (int c0 = 0; c0 < icnc; c0 += kChanStep) {
            const int lanes = f32 ? std::min(16, icnc - c0) :
                                    std::min(16, (icnc - c0) / 2);
            const svbool_t pc = svwhilelt_b32(0, lanes);
            const int k0 = t * icnc + c0;
            for (int pstart = 0; pstart < mc; pstart += sme_mr) {
                const int n = std::min(sme_mr, mc - pstart);
                for (int i = 0; i < n; ++i) {
                    if (live[pstart + i]) {
                        if constexpr (f32) {
                            svld1_hor_za32(0, static_cast<uint32_t>(i), pc,
                                           in + base[pstart + i] + c0);
                        } else {
                            svld1_hor_za32(0, static_cast<uint32_t>(i), pc,
                                           reinterpret_cast<const uint32_t*>(in) +
                                               (base[pstart + i] + c0) / 2);
                        }
                    } else {
                        svwrite_hor_za32_f32_m(0, static_cast<uint32_t>(i), pc, vzero);
                    }
                }
                const svbool_t pr = svwhilelt_b32(0, n);
                if constexpr (f32) {
                    for (int j = 0; j < lanes; ++j) {
                        svst1_ver_za32(0, static_cast<uint32_t>(j), pr,
                                       dst + static_cast<int64_t>(k0 + j) * kRows + pstart);
                    }
                } else {
                    auto* const dw = reinterpret_cast<uint32_t*>(dst);
                    for (int j = 0; j < lanes; ++j) {
                        svst1_ver_za32(0, static_cast<uint32_t>(j), pr,
                                       dw + static_cast<int64_t>(k0 / 2 + j) * kRows +
                                           pstart);
                    }
                }
            }
        }
        t += 1;
    }
}

/// The quantized twin of the float gather: a ZA row is one pixel's channel run (four channels a
/// word), so a tap's 64-byte run per pixel is one `ld1_hor_za32` into that
/// pixel's row, and the K groups come out of the tile vertically into
/// [m/16][k/4][64]. The gather and the A pack are one pass, as in the float
/// form — the source must already be signed (@p xr applied upstream) and four
/// bytes wide per k group.
///
/// A flat 256-pixel tile spans whole output rows, so it contains a row's first
/// and last column and is rarely fully interior — a tile-wide "interior" flag
/// made the branch-free path dead for the ResNet18 shapes. Validity is
/// therefore resolved per *row segment* into a per-pixel source pointer: every
/// invalid pixel points at @p fill (one input-channel run wide, so `+ c0` stays
/// in bounds), and the load loop is a plain pointer load with no range tests.
__arm_locally_streaming __arm_new("za") __attribute__((target("+sme2")))
inline void tile_im2col_pack_i8_sme(int8_t* dst, const int8_t* in, int64_t in_row_stride,
                                    int64_t ic_stride, const int8_t* fill, int64_t p0, int OW,
                                    int mc, int IH, int IW, int KH, int KW, int SH, int SW,
                                    int DH, int DW, int PH, int PW, int icnc, int karea) noexcept
{
    constexpr int kRows = sme_mr;      // 16 pixels: one A panel
    constexpr int kChanMax = 64;                // bytes one ZA tile row spans
    constexpr int kMaxPix = 16 * sme_mr; // one tile's pixels
    const int64_t groups = static_cast<int64_t>(karea) * icnc / 4;
    // Per pixel: this tap's run start, or `fill`.
    const int8_t* src[kMaxPix];
    int64_t oh0;
    int ow0;
    im2col_tile_origin(p0, OW, oh0, ow0);

    for (int t = 0; t < karea; ++t) {
        const int kh = t / KW;
        const int kw = t - kh * KW;
        const int trow = kh * DH;
        const int tcol = kw * DW;
        // iw = ow*SW + col_off is inside the image for one contiguous column
        // range: [vlo, vhi).
        const int col_off = tcol - PW;
        int vlo, vhi;
        im2col_tap_cols(col_off, SW, IW, OW, vlo, vhi);
        const int64_t col_step = static_cast<int64_t>(SW) * ic_stride;

        int i = 0;
        int64_t oh = oh0;
        int ow = ow0;
        while (i < mc) {
            const int run = std::min(OW - ow, mc - i);
            const int ih = static_cast<int>(oh * SH - PH) + trow;
            if (ih < 0 || ih >= IH) {
                for (int j = i; j < i + run; ++j) {
                    src[j] = fill;
                }
            } else {
                const int a = i + std::min(run, std::max(0, vlo - ow));
                const int b = i + std::min(run, std::max(0, vhi - ow));
                int j = i;
                for (; j < a; ++j) {
                    src[j] = fill;
                }
                if (j < b) {
                    const int64_t base = static_cast<int64_t>(ih) * in_row_stride +
                                         static_cast<int64_t>(ow + (a - i)) * SW * ic_stride +
                                         static_cast<int64_t>(col_off) * ic_stride;
                    const int8_t* p = in + base;
                    for (; j < b; ++j, p += col_step) {
                        src[j] = p;
                    }
                }
                for (; j < i + run; ++j) {
                    src[j] = fill;
                }
            }
            i += run;
            ow = 0;
            ++oh;
        }

        for (int c0 = 0; c0 < icnc; c0 += kChanMax) {
            const int words = std::min(kChanMax, icnc - c0) / 4;
            const svbool_t pc = svwhilelt_b32(0, words);
            for (int pstart = 0; pstart < mc; pstart += kRows) {
                const int n = std::min(kRows, mc - pstart);
                if (n == kRows) {
                    // Constant row indices: the ZA slice folds into the
                    // instruction instead of a register, and the whole panel
                    // pipelines.
#pragma unroll
                    for (int r = 0; r < kRows; ++r) {
                        svld1_hor_za32(
                            0, static_cast<uint32_t>(r), pc,
                            reinterpret_cast<const uint32_t*>(src[pstart + r] + c0));
                    }
                } else {
                    for (int r = 0; r < n; ++r) {
                        svld1_hor_za32(
                            0, static_cast<uint32_t>(r), pc,
                            reinterpret_cast<const uint32_t*>(src[pstart + r] + c0));
                    }
                }
                const svbool_t pr = svwhilelt_b32(0, n);
                auto* const dw = reinterpret_cast<int32_t*>(dst);
                const int64_t blk = static_cast<int64_t>(pstart / kRows) * groups;
                const int64_t bjat = blk + static_cast<int64_t>(t) * icnc / 4 + c0 / 4;
                // A vertical ZA store costs about five cycles; reading four
                // columns at once and storing them as two 128-byte memory
                // writes is twice as fast over the whole pack. Four columns are
                // 256 contiguous bytes in the A panel, so the pair stores land
                // back to back.
                int j = 0;
                if (n == kRows) {
                    for (; j + 4 <= words; j += 4) {
                        const svint32x4_t g = svread_ver_za32_s32_vg4(0, static_cast<uint32_t>(j));
                        svst1_s32_x4(svptrue_c32(), dw + (bjat + j) * sme_mr, g);
                    }
                }
                for (; j < words; ++j) {
                    svst1_ver_za32(0, static_cast<uint32_t>(j), pr,
                                   dw + (bjat + j) * sme_mr);
                }
            }
        }
    }
}

#endif // NNOPS_ARCH_SME

} // namespace nnops::backend::cpu::aarch64
