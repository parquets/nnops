#pragma once
/// @file simd_im2col.hpp
/// @brief SIMD im2col gathers — the gathered operand of a conv-as-GEMM.
///
/// Two gathers, both pixel-major and tap-major (k = tap*icnc + ic):
///  - Conv2D: one row per output pixel. A tap is a contiguous channel run, so a
///    row is karea straight copies — the same access the 1x1 route reads for
///    free — and that order is the weight's own (HWIO), so the ordinary panel
///    packer reads B. Adjacent taps of one filter row merge into a single
///    KW*icnc span when DW == 1 and ic_stride == icnc. tiled_im2col_i8 is the
///    same gather into s8 for the quantized route: bytes in, XORed by `xr`, the
///    zero point as `pad`, and the k tail zeroed to kdim4 for the direct-A MMA.
///  - Conv3D: one row per output voxel, [roi_volume][icnc*karea3d]. Each tap is
///    again one contiguous channel run; a padded tap fills its run with `pad`.
///    tiled_im2col_3d_i8 is the byte twin the quantized route runs.
///
/// Address arguments are element strides, not bytes. The 2D gather takes the
/// input row stride and the channel step; the 3D gather takes the pixel, row
/// and depth strides. `pad` / `pad_value` is 0 on the fp routes and the
/// activation zero point on the pre-quantized one.

#include "backend/cpu/common/common_utils.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <type_traits>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>

namespace nnops::kernel {

using namespace simd;

// ---- gathers ------------------------------------------

/// 3D gather into a pixel-major, tap-major col:
/// a[pixel][tap·icnc + ic]. A pixel is a whole channel run, so one tap is one
/// contiguous copy of `icnc` values; a padded tap writes the pad value.
template <class T>
inline void tiled_im2col_3d(
    T* NNOPS_RESTRICT a, int lda, const T* NNOPS_RESTRICT in, int64_t in_pix_stride,
    int64_t in_d_stride, int64_t in_row_stride, int icnc, int od0, int oh0, int rd, int rh,
    int rw, int in_d, int in_h, int in_w, int kernel_d, int kernel_h, int kernel_w, int stride_d,
    int stride_h, int stride_w, int dilate_d, int dilate_h, int dilate_w, int pad_d, int pad_h,
    int pad_w, T pad = T{}) noexcept
{
    using namespace nnops::simd;
    constexpr int L = simd_lane_for<T>;
    const float pad_f = [&] {
        if constexpr (std::is_same_v<T, float>) {
            return pad;
        } else {
            return ::nnops::backend::cpu::half_to_float(pad);
        }
    }();

    const auto fill = [&](T* dst) noexcept {
        int j = 0;
        for (; j + L <= icnc; j += L) {
            v_store(dst + j, v_set1(dst, pad_f));
        }
        for (; j < icnc; ++j) {
            dst[j] = pad;
        }
    };
    const auto copy = [&](T* dst, const T* src) noexcept {
        int j = 0;
        for (; j + L <= icnc; j += L) {
            v_store(dst + j, v_load(src + j));
        }
        for (; j < icnc; ++j) {
            dst[j] = src[j];
        }
    };

    for (int od = 0; od < rd; ++od) {
        const int id0 = (od0 + od) * stride_d - pad_d;
        for (int oh = 0; oh < rh; ++oh) {
            const int ih0 = (oh0 + oh) * stride_h - pad_h;
            T* pix = a + static_cast<int64_t>(od * rh + oh) * rw * lda;
            for (int ow = 0; ow < rw; ++ow) {
                const int iw0 = ow * stride_w - pad_w;
                T* dst = pix + static_cast<int64_t>(ow) * lda;
                for (int kd = 0; kd < kernel_d; ++kd) {
                    const int id = id0 + kd * dilate_d;
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        const int ih = ih0 + kh * dilate_h;
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            T* k = dst + ((kd * kernel_h + kh) * kernel_w + kw) * icnc;
                            const int iw = iw0 + kw * dilate_w;
                            if (id < 0 || id >= in_d || ih < 0 || ih >= in_h || iw < 0 ||
                                iw >= in_w) {
                                fill(k);
                                continue;
                            }
                            const T* src = in + id * in_d_stride + ih * in_row_stride +
                                           iw * in_pix_stride;
                            copy(k, src);
                        }
                    }
                }
            }
        }
    }
}

/// The quantized 3-D gather: bytes in, bytes out, the zero point as `pad`.
/// One tap is still one contiguous channel run, so a live tap is one byte copy
/// and a padded tap one byte fill. The fp gather above cannot serve int8 — its
/// vector fill converts through f32 — and the 2-D route keeps its own byte
/// gather for the same reason (tiled_im2col_i8).
template <class T>
inline void tiled_im2col_3d_i8(T* NNOPS_RESTRICT a, int lda, const T* NNOPS_RESTRICT in,
                               int64_t in_pix_stride, int64_t in_d_stride, int64_t in_row_stride,
                               int icnc, int od0, int oh0, int rd, int rh, int rw, int in_d,
                               int in_h, int in_w, int kernel_d, int kernel_h, int kernel_w,
                               int stride_d, int stride_h, int stride_w, int dilate_d,
                               int dilate_h, int dilate_w, int pad_d, int pad_h, int pad_w,
                               T pad) noexcept
{
    static_assert(sizeof(T) == 1, "quantized storage is one byte");
    const unsigned char pv = static_cast<unsigned char>(pad);
    for (int od = 0; od < rd; ++od) {
        const int id0 = (od0 + od) * stride_d - pad_d;
        for (int oh = 0; oh < rh; ++oh) {
            const int ih0 = (oh0 + oh) * stride_h - pad_h;
            T* pix = a + static_cast<int64_t>(od * rh + oh) * rw * lda;
            for (int ow = 0; ow < rw; ++ow) {
                const int iw0 = ow * stride_w - pad_w;
                T* dst = pix + static_cast<int64_t>(ow) * lda;
                for (int kd = 0; kd < kernel_d; ++kd) {
                    const int id = id0 + kd * dilate_d;
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        const int ih = ih0 + kh * dilate_h;
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            T* k = dst + ((kd * kernel_h + kh) * kernel_w + kw) * icnc;
                            const int iw = iw0 + kw * dilate_w;
                            if (id < 0 || id >= in_d || ih < 0 || ih >= in_h || iw < 0 ||
                                iw >= in_w) {
                                std::memset(k, pv, static_cast<size_t>(icnc));
                                continue;
                            }
                            std::memcpy(k,
                                        in + id * in_d_stride + ih * in_row_stride +
                                            iw * in_pix_stride,
                                        static_cast<size_t>(icnc));
                        }
                    }
                }
            }
        }
    }
}

/// Gather one pixel tile of one ic block into A: row m is the patch of output
/// pixel (oh, ow0 + m). @p p0 is that pixel's flat index (oh × OW + ow0);
/// @p flat walks the tile past a row end into the next, legal only because the
/// rows are dense.
template <class T>
inline void tiled_im2col(T* NNOPS_RESTRICT a, int lda, const T* NNOPS_RESTRICT in,
                                      int64_t in_row_stride, int64_t ic_stride, T pad, int64_t p0,
                                      int OW, bool flat, int mc, int IH, int IW, int KH, int KW,
                                      int SH, int SW, int DH, int DW, int PH, int PW, int icnc,
                                      int karea) noexcept
{
    using namespace nnops::simd;
    constexpr int L = simd_lane_for<T>;
    // A whole vector per step when the channel run is a lane multiple — every
    // ladder block but a tail. The element loop this replaces cost as much as
    // the GEMM it feeds.
    const bool vec = (icnc % L) == 0;
    const float pad_f = [&] {
        if constexpr (std::is_same_v<T, float>) {
            return pad;
        } else {
            return ::nnops::backend::cpu::half_to_float(pad);
        }
    }();
    const auto copy_run = [=](T* dst, const T* src) noexcept {
        if (!vec) {
            if (src) {
                for (int j = 0; j < icnc; ++j) {
                    dst[j] = src[j];
                }
            } else {
                for (int j = 0; j < icnc; ++j) {
                    dst[j] = pad;
                }
            }
            return;
        }
        if (src) {
            for (int j = 0; j < icnc; j += L) {
                v_store(dst + j, v_load(src + j));
            }
        } else {
            const auto vp = v_set1(dst, pad_f);
            for (int j = 0; j < icnc; j += L) {
                v_store(dst + j, vp);
            }
        }
    };

    int64_t oh = p0 / OW;
    int ow = static_cast<int>(p0 - oh * OW);
    // With the whole channel block in hand and no dilation, the taps of one
    // filter row are a single KW × icnc run in both the input and the A row —
    // k is tap-major, the input's own order. One copy per row, not per tap.
    const bool tap_run = (DW == 1) && (ic_stride == icnc);
    const auto copy_span = [=](T* dst, const T* src, int n) noexcept {
        if (!src) {
            const auto vp = v_set1(dst, pad_f);
            int j = 0;
            if (vec) {
                for (; j + L <= n; j += L) {
                    v_store(dst + j, vp);
                }
            }
            for (; j < n; ++j) {
                dst[j] = pad;
            }
            return;
        }
        if (n >= L) {
            int j = 0;
            for (; j + L <= n; j += L) {
                v_store(dst + j, v_load(src + j));
            }
            // The tail rides on one overlapping vector rather than going scalar.
            if (j < n) {
                v_store(dst + n - L, v_load(src + n - L));
            }
        } else {
            for (int j = 0; j < n; ++j) {
                dst[j] = src[j];
            }
        }
    };
    for (int m = 0; m < mc; ++m, ++ow) {
        if (flat && ow == OW) {
            ow = 0;
            ++oh;
        }
        const int64_t ih0 = oh * SH - PH;
        const int64_t iw0 = static_cast<int64_t>(ow) * SW - PW;
        T* row = a + static_cast<int64_t>(m) * lda;
        if (tap_run) {
            for (int kh = 0; kh < KH; ++kh) {
                const int64_t ih = ih0 + static_cast<int64_t>(kh) * DH;
                T* dst = row + static_cast<int64_t>(kh) * KW * icnc;
                if (ih < 0 || ih >= IH) {
                    copy_span(dst, nullptr, KW * icnc);
                    continue;
                }
                const T* hrow = in + ih * in_row_stride;
                if (iw0 >= 0 && iw0 + KW <= IW) {
                    copy_span(dst, hrow + iw0 * ic_stride, KW * icnc);
                } else {
                    for (int kw = 0; kw < KW; ++kw) {
                        const int64_t iw = iw0 + kw;
                        copy_run(dst + kw * icnc,
                                 (iw >= 0 && iw < IW) ? hrow + iw * ic_stride : nullptr);
                    }
                }
            }
            continue;
        }
        int t = 0;
        for (int kh = 0; kh < KH; ++kh) {
            const int64_t ih = ih0 + static_cast<int64_t>(kh) * DH;
            const bool h_ok = (ih >= 0 && ih < IH);
            const T* hrow = in + ih * in_row_stride;
            for (int kw = 0; kw < KW; ++kw, ++t) {
                const int64_t iw = iw0 + static_cast<int64_t>(kw) * DW;
                T* dst = row + static_cast<int64_t>(t) * icnc;
                copy_run(dst, (h_ok && iw >= 0 && iw < IW) ? hrow + iw * ic_stride : nullptr);
            }
        }
    }
}

/// The quantized twin: s8 A over a byte source with the shift applied on the
/// way through — @p xr is the XOR (+0x80 for u8, 0 otherwise) that puts a byte
/// in signed range, @p pad the zero point outside the input. The k tail to
/// @p kdim4 is zeroed so the direct-A kernel, which reads whole 4-byte groups,
/// sees a zero product there (the B panel is zero-padded over the same range).
/// @p p0 / @p flat mean what they do in the fp gather.
///
/// @p safe16 is the largest element offset from @p in at which a 16-byte load
/// still reads inside the source; 0 disables the wide copy. It exists for the
/// tiny-channel stems, where a filter row's whole tap run is at most a few
/// dozen bytes and the run is one 16-byte copy instead of two overlapping
/// 8-byte ones (whose store-then-partial-load on the same address costs a
/// store-forwarding stall). The bytes a short run's copy spills fall in the
/// same pixel's *later* filter rows, which this walk rewrites afterwards; the
/// last row spills past the k tail instead, so a short last run stays exact. A
/// run of 16 bytes or more reads only from within itself, so the bound only
/// guards the shorter ones.
inline void tiled_im2col_i8(int8_t* NNOPS_RESTRICT a, int lda,
                                         const uint8_t* NNOPS_RESTRICT in, int64_t in_row_stride,
                                         int64_t ic_stride, uint8_t xr, uint8_t pad, int64_t p0,
                                         int OW, bool flat, int mc, int IH, int IW, int KH, int KW,
                                         int SH, int SW, int DH, int DW, int PH, int PW, int icnc,
                                         int karea, int kdim, int kdim4,
                                         int64_t safe16 = 0) noexcept
{
    // The channel run of one tap is the whole copy; the 8- and 16-wide blocks
    // take one 8-byte (two for 16) load/xor/store instead of a byte loop, which
    // is what used to make the gather as expensive as the GEMM it feeds.
    const uint64_t xr64 = static_cast<uint64_t>(xr) * 0x0101010101010101ULL;
    const uint8_t pv = static_cast<uint8_t>(pad ^ xr);
    // src == nullptr fills the run with the padded value instead.
    const auto copy_run = [xr, xr64, pv, icnc](int8_t* dst, const uint8_t* src) noexcept {
        if (icnc == 16) {
            uint64_t a0 = 0, a1 = 0;
            if (src) {
                std::memcpy(&a0, src, 8);
                std::memcpy(&a1, src + 8, 8);
                a0 ^= xr64;
                a1 ^= xr64;
            } else {
                a0 = a1 = static_cast<uint64_t>(pv) * 0x0101010101010101ULL;
            }
            std::memcpy(dst, &a0, 8);
            std::memcpy(dst + 8, &a1, 8);
        } else if (icnc == 8) {
            uint64_t a0 = src ? 0 : static_cast<uint64_t>(pv) * 0x0101010101010101ULL;
            if (src) {
                std::memcpy(&a0, src, 8);
                a0 ^= xr64;
            }
            std::memcpy(dst, &a0, 8);
        } else if (src) {
            for (int j = 0; j < icnc; ++j) {
                dst[j] = static_cast<int8_t>(src[j] ^ xr);
            }
        } else {
            std::memset(dst, pv, static_cast<size_t>(icnc));
        }
    };

    // One 16-byte XOR copy: two 8-byte halves, so the shift is two XORs and a
    // 16-lane pair of loads/stores wherever the target has them.
    const auto copy16 = [xr64](int8_t* dst, const uint8_t* src) noexcept {
        uint64_t v0 = 0, v1 = 0;
        std::memcpy(&v0, src, 8);
        std::memcpy(&v1, src + 8, 8);
        v0 ^= xr64;
        v1 ^= xr64;
        std::memcpy(dst, &v0, 8);
        std::memcpy(dst + 8, &v1, 8);
    };
    // A run copied without overrun: an 8-byte body and a scalar tail.
    const auto copy_run_exact = [xr, xr64](int8_t* dst, const uint8_t* src, int n) noexcept {
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            uint64_t v = 0;
            std::memcpy(&v, src + j, 8);
            v ^= xr64;
            std::memcpy(dst + j, &v, 8);
        }
        for (; j < n; ++j) {
            dst[j] = static_cast<int8_t>(src[j] ^ xr);
        }
    };

    const bool tap_run = (DW == 1) && (ic_stride == icnc);
    if (tap_run) {
        // One filter row's taps are a fixed stride apart in the input and a
        // fixed stride apart in the A row, so walk the block in *row segments*
        // and, inside one, one filter row at a time: the interior columns take
        // a branch-free copy loop and only the segment's edges fall back to a
        // per-tap gather. At the three-channel stems the per-pixel tests this
        // removes cost more than the copies do.
        //
        // A pass writes one filter row of every pixel, `lda` apart, so it
        // cannot be longer than an L1 window: a 576-byte A row (64 channels,
        // 3x3) would stream 147 KB three times over. `pmax` bounds it, which
        // keeps the win on the narrow stems where the walk exists and costs
        // nothing where the block is wider than the cache.
        const int run = KW * icnc;
        const int64_t step = static_cast<int64_t>(SW) * ic_stride;
        const int pmax = std::max(8, 32768 / lda);
        const int lo_abs = PW == 0 ? 0 : (PW + SW - 1) / SW;
        const int64_t hi_last = IW - 1 + PW - static_cast<int64_t>(KW - 1) * DW;
        const int hi_abs = hi_last < 0 ? 0 : static_cast<int>(hi_last / SW) + 1;
        int m = 0;
        int64_t oh = p0 / OW;
        int ow = static_cast<int>(p0 - oh * OW);
        while (m < mc) {
            const int n = flat ? std::min(OW - ow, mc - m) : mc - m;
            const int64_t ih0 = oh * SH - PH;
            for (int c0 = 0; c0 < n; c0 += pmax) {
                const int cn = std::min(pmax, n - c0);
                // Columns of the output row whose whole tap run is in bounds:
                // c·SW ≥ PW and c·SW ≤ IW-1+PW-(KW-1)·DW. [e0, e1) is that
                // range relative to this pass's pixel window.
                const int e0 = std::clamp(lo_abs - (ow + c0), 0, cn);
                const int e1 = std::clamp(hi_abs - (ow + c0), 0, cn);
                for (int kh = 0; kh < KH; ++kh) {
                    const int64_t ih = ih0 + static_cast<int64_t>(kh) * DH;
                    int8_t* const dbase =
                        a + static_cast<int64_t>(m + c0) * lda + static_cast<int64_t>(kh) * run;
                    if (ih < 0 || ih >= IH) {
                        int8_t* d = dbase;
                        for (int j = 0; j < cn; ++j, d += lda) {
                            std::memset(d, pv, static_cast<size_t>(run));
                        }
                        continue;
                    }
                    const uint8_t* const hbase =
                        in + ih * in_row_stride +
                        (static_cast<int64_t>(ow + c0) * SW - PW) * ic_stride;
                    // Two 16-byte copies cover [0, run) for run in [8, 32]. Every
                    // filter row but the last spills into its own pixel's later
                    // rows, which the following passes rewrite; a short run in
                    // the last row would spill past the k tail instead, so it
                    // takes exact copies.
                    const bool wide =
                        safe16 > 0 && run >= 8 && run <= 32 && (kh + 1 < KH || run >= 16);
                    bool src_ok = wide;
                    if (wide && run < 16) {
                        // The window's last interior column decides: its 16-byte
                        // read must stay inside the source.
                        src_ok = e1 <= e0 ||
                                 (hbase - in) + static_cast<int64_t>(e1 - 1) * step <= safe16;
                    }
                    const auto tap = [&](int8_t* d, int j, int kw) {
                        const int64_t iw = static_cast<int64_t>(ow + c0 + j) * SW - PW +
                                           static_cast<int64_t>(kw) * DW;
                        if (iw < 0 || iw >= IW) {
                            std::memset(d, pv, static_cast<size_t>(icnc));
                            return;
                        }
                        const uint8_t* s = in + ih * in_row_stride + iw * ic_stride;
                        for (int c = 0; c < icnc; ++c) {
                            d[c] = static_cast<int8_t>(s[c] ^ xr);
                        }
                    };
                    // Edges: per pixel, per tap.
                    for (int j = 0; j < e0; ++j) {
                        int8_t* d = dbase + static_cast<int64_t>(j) * lda;
                        for (int kw = 0; kw < KW; ++kw, d += icnc) {
                            tap(d, j, kw);
                        }
                    }
                    for (int j = e1; j < cn; ++j) {
                        int8_t* d = dbase + static_cast<int64_t>(j) * lda;
                        for (int kw = 0; kw < KW; ++kw, d += icnc) {
                            tap(d, j, kw);
                        }
                    }
                    // Interior: the whole run per pixel, no tests.
                    int8_t* d = dbase + static_cast<int64_t>(e0) * lda;
                    const uint8_t* s = hbase + static_cast<int64_t>(e0) * step;
                    if (src_ok) {
                        for (int j = e0; j < e1; ++j, d += lda, s += step) {
                            copy16(d, s);
                            if (run > 16) {
                                copy16(d + run - 16, s + run - 16);
                            }
                        }
                    } else {
                        for (int j = e0; j < e1; ++j, d += lda, s += step) {
                            copy_run_exact(d, s, run);
                        }
                    }
                }
            }
            if (kdim4 > kdim) {
                for (int j = 0; j < n; ++j) {
                    int8_t* const d = a + (static_cast<int64_t>(m) + j) * lda + kdim;
                    std::memset(d, 0, static_cast<size_t>(kdim4 - kdim));
                }
            }
            m += n;
            ow = 0;
            ++oh;
        }
        return;
    }

    int64_t oh = p0 / OW;
    int ow = static_cast<int>(p0 - oh * OW);
    for (int m = 0; m < mc; ++m, ++ow) {
        if (flat && ow == OW) {
            ow = 0;
            ++oh;
        }
        const int64_t ih0 = oh * SH - PH;
        const int64_t iw0 = static_cast<int64_t>(ow) * SW - PW;
        int8_t* row = a + static_cast<int64_t>(m) * lda;
        int t = 0;
        for (int kh = 0; kh < KH; ++kh) {
            const int64_t ih = ih0 + static_cast<int64_t>(kh) * DH;
            const bool h_ok = (ih >= 0 && ih < IH);
            const uint8_t* hrow = in + ih * in_row_stride;
            for (int kw = 0; kw < KW; ++kw, ++t) {
                const int64_t iw = iw0 + static_cast<int64_t>(kw) * DW;
                int8_t* dst = row + static_cast<int64_t>(t) * icnc;
                copy_run(dst, (h_ok && iw >= 0 && iw < IW) ? hrow + iw * ic_stride : nullptr);
            }
        }
        if (kdim4 > kdim) {
            std::memset(row + kdim, 0, static_cast<size_t>(kdim4 - kdim));
        }
    }
}

} // namespace nnops::kernel
