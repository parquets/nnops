#pragma once
/// @file simd_im2col.hpp
/// @brief SIMD im2col gathers — the gathered operand of a conv-as-GEMM.
///
/// Two families:
///  - channel-first Conv3D: one row per (ic, tap), the row spanning the current
///    output block (the roi). tiled_im2col_3d writes a row-major col matrix
///    [ic*karea3d][roi_volume]; out-of-input positions store zero.
///  - channel-last Conv2D: one row per output pixel, k tap-major
///    (k = tap*icnc + ic). A tap is a contiguous channel run, so a row is
///    karea straight copies — the same access the 1x1 route reads for free —
///    and that order is the weight's own (HWIO), so the ordinary panel packer
///    reads B. Adjacent taps of one filter row merge into a single KW*icnc span
///    when DW == 1 and ic_stride == icnc. tiled_im2col_channel_last_i8 is the same
///    gather into s8 for the quantized route: bytes in, XORed by `xr`, the zero
///    point as `pad`, and the k tail zeroed to kdim4 for the direct-A MMA.
///
/// Address arguments are element strides, not bytes: icn_step between input
/// channels, ih_step between input rows (the padded row pitch), plus id_step
/// between depth slices on the 3D route. `pad` / `pad_value` is 0 on the fp
/// routes and the activation zero point on the pre-quantized one.
///
/// Padding on the 3D route: the *local* pad (the part inside this block) says
/// where the valid region starts; the *global* pad (the operator's PD/PH/PW)
/// says where it ends and what the input address is. Mixing them yields
/// off-by-padding input rows on every block after the first.
///
/// The fused pack zero-fills the columns past roi_area in the last panel,
/// because the MMA reads all NR lanes of every panel row. A panel row is the
/// unit of work, so it takes a compile-time store sequence: a dynamic-length
/// loop cannot be unrolled, and it measured ~3x per row. Width copies use an
/// L-wide SIMD loop at stride_w == 1 (the dominant case), a scalar stride_w
/// loop otherwise.

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

/// 3D Conv3D gather into
/// col_data[ic*karea3d + (kd*kernel_h + kh)*kernel_w + kw][roi_d*roi_h*roi_w];
/// out-of-input positions are zero. This one zeroes each row and copies the
/// valid interior — simpler for the 6-face/12-edge padding of a 3D block.
template <class T>
inline void tiled_im2col_3d(T* col_data, const T* input, int roi_od, int roi_oh,
                            int roi_ow,                      // roi origin (global output coords)
                            int roi_d, int roi_h, int roi_w, // roi extent (block sizes)
                            int pad_d_local, int pad_h_local, int pad_w_local, int pad_d_global,
                            int pad_h_global, int pad_w_global, int in_channel, int in_d, int in_h,
                            int in_w, int icn_step, int id_step, int ih_step, int kernel_d,
                            int kernel_h, int kernel_w, int stride_d, int stride_h, int stride_w,
                            int dilate_d, int dilate_h, int dilate_w) noexcept
{
    const int karea = kernel_d * kernel_h * kernel_w;
    const int roi_volume = roi_d * roi_h * roi_w;
    constexpr int L = simd_lane_for<T>;

    for (int ic = 0; ic < in_channel; ++ic) {
        const T* input_ptr = input + ic * icn_step;
        for (int kd = 0; kd < kernel_d; ++kd) {
            const int ikd = kd * dilate_d;
            const int od_beg =
                std::max(0, (int)std::ceil((pad_d_local - ikd) / (float)stride_d) - roi_od);
            const int od_end = std::max(
                od_beg,
                std::min(roi_d,
                         (int)std::ceil((pad_d_global + in_d - ikd) / (float)stride_d) - roi_od));

            for (int kh = 0; kh < kernel_h; ++kh) {
                const int ikh = kh * dilate_h;
                const int oh_beg =
                    std::max(0, (int)std::ceil((pad_h_local - ikh) / (float)stride_h) - roi_oh);
                const int oh_end = std::max(
                    oh_beg,
                    std::min(roi_h, (int)std::ceil((pad_h_global + in_h - ikh) / (float)stride_h) -
                                        roi_oh));

                for (int kw = 0; kw < kernel_w; ++kw) {
                    const int ikw = kw * dilate_w;
                    const int ow_beg =
                        std::max(0, (int)std::ceil((pad_w_local - ikw) / (float)stride_w) - roi_ow);
                    const int ow_end = std::max(
                        ow_beg, std::min(roi_w, (int)std::ceil((pad_w_global + in_w - ikw) /
                                                               (float)stride_w) -
                                                    roi_ow));

                    T* col_ptr =
                        col_data + (ic * karea + (kd * kernel_h + kh) * kernel_w + kw) * roi_volume;

                    std::memset(col_ptr, 0, sizeof(T) * static_cast<size_t>(roi_volume));

                    int id_paded = (od_beg + roi_od) * stride_d - pad_d_global + ikd;
                    for (int od = od_beg; od < od_end; ++od, id_paded += stride_d) {
                        const T* d_ptr = input_ptr + id_paded * id_step;
                        int ih_paded = (oh_beg + roi_oh) * stride_h - pad_h_global + ikh;

                        for (int oh = oh_beg; oh < oh_end; ++oh, ih_paded += stride_h) {
                            const T* in_row = d_ptr + ih_paded * ih_step;
                            int iw_paded = (ow_beg + roi_ow) * stride_w - pad_w_global + ikw;
                            T* out_row = col_ptr + (od * roi_h + oh) * roi_w;

                            int w = ow_beg;
                            int in_id = iw_paded;
                            if (stride_w == 1) {
                                for (; w + L <= ow_end; w += L, in_id += L) {
                                    v_store(out_row + w, v_load(in_row + in_id));
                                }
                            }
                            for (; w < ow_end; ++w, in_id += stride_w) {
                                out_row[w] = in_row[in_id];
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---- channel-last gathers ------------------------------------------

/// 3D channel-last gather into a pixel-major, tap-major col:
/// a[pixel][tap·icnc + ic]. A pixel is a whole channel run, so one tap is one
/// contiguous copy of `icnc` values; a padded tap writes the pad value.
template <class T>
inline void tiled_im2col_3d_channel_last(
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

/// Gather one pixel tile of one ic block into A: row m is the patch of output
/// pixel (oh, ow0 + m). @p p0 is that pixel's flat index (oh × OW + ow0);
/// @p flat walks the tile past a row end into the next, legal only because the
/// rows are dense.
template <class T>
inline void tiled_im2col_channel_last(T* NNOPS_RESTRICT a, int lda, const T* NNOPS_RESTRICT in,
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
inline void tiled_im2col_channel_last_i8(int8_t* NNOPS_RESTRICT a, int lda,
                                         const uint8_t* NNOPS_RESTRICT in, int64_t in_row_stride,
                                         int64_t ic_stride, uint8_t xr, uint8_t pad, int64_t p0,
                                         int OW, bool flat, int mc, int IH, int IW, int KH, int KW,
                                         int SH, int SW, int DH, int DW, int PH, int PW, int icnc,
                                         int karea, int kdim, int kdim4) noexcept
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

    // Same filter-row merge as the fp gather: one XOR copy per row instead of
    // one per tap, which matters most when icnc is small and each per-tap run
    // is only a few bytes.
    const bool tap_run = (DW == 1) && (ic_stride == icnc);
    const auto xor_span = [xr, xr64, pv](int8_t* dst, const uint8_t* src, int n) noexcept {
        int j = 0;
        if (!src) {
            for (; j + 8 <= n; j += 8) {
                std::memset(dst + j, pv, 8);
            }
            for (; j < n; ++j) {
                dst[j] = static_cast<int8_t>(pv);
            }
            return;
        }
        for (; j + 8 <= n; j += 8) {
            uint64_t v = 0;
            std::memcpy(&v, src + j, 8);
            v ^= xr64;
            std::memcpy(dst + j, &v, 8);
        }
        if (j < n) {
            if (n >= 8) {
                // The tail rides on one overlapping 8-byte copy.
                j = n - 8;
                uint64_t v = 0;
                std::memcpy(&v, src + j, 8);
                v ^= xr64;
                std::memcpy(dst + j, &v, 8);
            } else {
                for (; j < n; ++j) {
                    dst[j] = static_cast<int8_t>(src[j] ^ xr);
                }
            }
        }
    };

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
        if (tap_run) {
            for (int kh = 0; kh < KH; ++kh) {
                const int64_t ih = ih0 + static_cast<int64_t>(kh) * DH;
                int8_t* dst = row + static_cast<int64_t>(kh) * KW * icnc;
                if (ih < 0 || ih >= IH) {
                    xor_span(dst, nullptr, KW * icnc);
                    continue;
                }
                const uint8_t* hrow = in + ih * in_row_stride;
                if (iw0 >= 0 && iw0 + KW <= IW) {
                    xor_span(dst, hrow + iw0 * ic_stride, KW * icnc);
                } else {
                    for (int kw = 0; kw < KW; ++kw) {
                        const int64_t iw = iw0 + kw;
                        copy_run(dst + kw * icnc,
                                 (iw >= 0 && iw < IW) ? hrow + iw * ic_stride : nullptr);
                    }
                }
            }
            if (kdim4 > kdim) {
                std::memset(row + kdim, 0, static_cast<size_t>(kdim4 - kdim));
            }
            continue;
        }
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
