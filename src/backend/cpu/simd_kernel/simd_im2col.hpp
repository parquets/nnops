#pragma once
/// @file simd_im2col.hpp
/// @brief SIMD im2col kernels — the "unrolled input" step of im2col+GEMM conv.
///
/// Five kernels build the gathered operand of a conv-as-GEMM — three for the
/// planar routes (NCHW Conv2D, NCDHW Conv3D) and two for the channel-last one:
///   tiled_im2col_2d         — NCHW  Conv2D: col_data[ic*karea2d][roi_h*roi_w]
///   tiled_im2col_2d_packed  — the same gather, written straight into the
///                             uniform packed-B panels (fused im2col + pack);
///                             the panel width is a template parameter, so a
///                             full panel row is an unrolled store sequence
///   tiled_im2col_3d         — NCDHW Conv3D: col_data[ic*karea3d][roi_d*roi_h*roi_w]
///   tiled_im2col_nhwc       — NHWC  Conv2D: a_tile[pixel][tap*icnc + ic]
///   tiled_im2col_nhwc_i8    — the same gather, into s8 (or a dst of the
///                             caller's int8 type) for the quantized route
///
/// On the planar routes each row of the gathered operand is one
/// (ic, kd, kh, kw) and holds the input spatial positions across the current
/// output block (the roi). Positions that fall outside the input (padding) are
/// zero-filled.
///
/// Padding convention (mirrors nn_compute tiled_im2col_2d):
///   * *local* padding — the padding remaining inside this block (roi_pad_*)
///     — determines where the valid region *starts* (the `*_beg` bounds).
///   * *global* padding — the operator's padding (PD/PH/PW) — determines where
///     the valid region *ends* (`*_end` bounds) and the actual input addressing
///     (`id_paded`/`ih_paded`/`iw_paded`).
///
/// The channel-last pair flips those roles: one output pixel is one row and the
/// k axis runs tap-major (k = tap × icnc + local_ic) within an input-channel
/// block. A tap of a channel-last input is a contiguous channel run, so a gather
/// is `karea` straight copies per row — the same copy the 1x1 route reads for
/// free.
///
/// Mixing these up (using the local padding for the input addressing) produces
/// off-by-padding input rows on every block after the first.
///
/// The innermost (width) copy uses an L-wide SIMD loop when stride_w == 1 (the
/// dominant case); otherwise a scalar loop advancing by stride_w.
///
/// `tiled_im2col_2d` writes the input values through unchanged, with a
/// configurable pad value. That is all the quantized Conv2D route needs: it
/// quantizes an fp activation into a plane up front (conv2d_implicit_gemm.cpp,
/// quantize_plane) and gathers bytes from then on, so no store policy other
/// than "copy" is required.

#include "backend/cpu/common/common_utils.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>

namespace nnops::kernel {

using namespace simd;

namespace detail {

/// Write @p n copies of @p pad at @p d. The zero case — every fp route — takes
/// the memset path; the pre-quantized int8 route can pass a zero point.
template <class T>
inline void im2col_fill_pad(T* d, int n, T pad) noexcept {
    if constexpr (std::is_integral_v<T>) {
        if (pad == T(0)) {
            std::memset(d, 0, sizeof(T) * static_cast<size_t>(n));
        } else {
            std::fill(d, d + n, pad);
        }
    } else {
        // +0 and -0 are the same value and both memset to zero bits.
        if (s_load(&pad) == 0.0f) {
            std::memset(d, 0, sizeof(T) * static_cast<size_t>(n));
        } else {
            std::fill(d, d + n, pad);
        }
    }
}

/// Copy policy: write the gathered values (and `pad` outside the input)
/// unchanged into a row-major [K][roi_area] col matrix. `pad` is the value
/// standing in for a zero input sample, which is the activation zero point on
/// the pre-quantized route.
template <class T>
struct Im2colCopy {
    using src_t = T;
    using dst_t = T;

    T* col;        ///< row 0 of the col matrix
    int roi_area;  ///< columns per k row
    T pad;

    /// Columns [n0, n0+n) of k row.
    void fill(int k, int n0, int n) const noexcept {
        fill_run(col + static_cast<int64_t>(k) * roi_area + n0, n);
    }

    /// Columns [n0, n0+n) of k row, gathered from @p s with @p step.
    void emit(int k, int n0, const T* s, int n, int step) const noexcept {
        emit_run(col + static_cast<int64_t>(k) * roi_area + n0, s, n, step);
    }

private:
    void fill_run(T* d, int n) const noexcept {
        im2col_fill_pad(d, n, pad);
    }

    void emit_run(T* d, const T* s, int n, int step) const noexcept {
        if constexpr (std::is_integral_v<T>) {
            if (step == 1) {
                std::memcpy(d, s, sizeof(T) * static_cast<size_t>(n));
            } else {
                for (int i = 0; i < n; ++i) { d[i] = s[i * step]; }
            }
        } else if (step == 1) {
            constexpr int L = simd_lane_for<T>;
            int i = 0;
            for (; i + L <= n; i += L) { v_store(d + i, v_load(s + i)); }
            for (; i < n; ++i) { d[i] = s[i]; }
        } else {
            for (int i = 0; i < n; ++i) { d[i] = s[i * step]; }
        }
    }
};

/// Store one full panel row (NR elements) with a compile-time store count.
/// A dynamic-length loop (`for (; i + L <= n; ...)` with n <= NR) cannot be
/// unrolled and issues a scalar tail; measured on the fused gather it costs
/// ~3x per panel row, and one panel row is the unit of work here.
///
/// NR is the arch's nr_max for T: 12 (f32, aarch64) does not divide the
/// 8-lane vector, so it takes three 4-lane stores; 16 (f32 x86, f16) takes
/// two 8-lane stores.
template <class T, int NR>
inline void pack_panel_zero(T* d) noexcept {
    constexpr int L = simd_lane_for<T>;
    if constexpr (std::is_same_v<T, float> && (NR % L) != 0) {
        static_assert(NR % 4 == 0, "panel width must be a multiple of 4");
        const auto vz = v_zero_f32x4();
        for (int j = 0; j < NR; j += 4) { v_store(d + j, vz); }
    } else {
        static_assert(NR % L == 0, "panel width must be a multiple of the SIMD lanes");
        const auto vz = v_zero(d);
        for (int j = 0; j < NR; j += L) { v_store(d + j, vz); }
    }
}

/// Copy one full panel row from @p s (unit stride), same store policy.
template <class T, int NR>
inline void pack_panel_copy(T* d, const T* s) noexcept {
    constexpr int L = simd_lane_for<T>;
    if constexpr (std::is_same_v<T, float> && (NR % L) != 0) {
        static_assert(NR % 4 == 0, "panel width must be a multiple of 4");
        for (int j = 0; j < NR; j += 4) { v_store(d + j, v_load_f32x4(s + j)); }
    } else {
        static_assert(NR % L == 0, "panel width must be a multiple of the SIMD lanes");
        for (int j = 0; j < NR; j += L) { v_store(d + j, v_load(s + j)); }
    }
}

/// Scatter policy: write the gathered values straight into the uniform
/// NR-wide B panels the MMA consumes (the layout tile_pack_rhs produces:
/// column n of k row lives at dst[(n / NR) * ldd + k * NR + (n % NR)]), so
/// the row-major col matrix is never materialized. A run of columns is cut at
/// the panel boundaries; a piece that fills a whole panel row goes through the
/// compile-time store, the two partial pieces at a run's ends stay dynamic.
template <class T, int NR>
struct Im2colPack {
    using src_t = T;
    using dst_t = T;

    T* dst;   ///< panel 0
    int ldd;  ///< panel stride (elements)
    T pad;

    /// Columns [n0, n0+n) of k row.
    void fill(int k, int n0, int n) const noexcept {
        if (n <= 0) { return; }
        int p   = n0 / NR;
        int off = n0 - p * NR;
        T* d = dst + static_cast<int64_t>(p) * ldd + static_cast<int64_t>(k) * NR + off;
        for (;;) {
            if (off == 0 && n >= NR) {
                pack_panel_zero<T, NR>(d);
                n -= NR;
                if (n == 0) { return; }
                d += ldd;
                continue;
            }
            const int len = std::min(n, NR - off);
            fill_run(d, len);
            n -= len;
            if (n == 0) { return; }
            d += ldd - off;
            off = 0;
        }
    }

    /// Columns [n0, n0+n) of k row, gathered from @p s with @p step.
    void emit(int k, int n0, const T* s, int n, int step) const noexcept {
        if (n <= 0) { return; }
        int p   = n0 / NR;
        int off = n0 - p * NR;
        T* d = dst + static_cast<int64_t>(p) * ldd + static_cast<int64_t>(k) * NR + off;
        for (;;) {
            if (step == 1 && off == 0 && n >= NR) {
                pack_panel_copy<T, NR>(d, s);
                s += NR;
                n -= NR;
                if (n == 0) { return; }
                d += ldd;
                continue;
            }
            const int len = std::min(n, NR - off);
            emit_run(d, s, len, step);
            s += static_cast<int64_t>(len) * step;
            n -= len;
            if (n == 0) { return; }
            d += ldd - off;
            off = 0;
        }
    }

private:
    void fill_run(T* d, int n) const noexcept {
        im2col_fill_pad(d, n, pad);
    }

    void emit_run(T* d, const T* s, int n, int step) const noexcept {
        if (step == 1) {
            constexpr int L = simd_lane_for<T>;
            int i = 0;
            for (; i + L <= n; i += L) { v_store(d + i, v_load(s + i)); }
            for (; i < n; ++i) { d[i] = s[i]; }
        } else {
            for (int i = 0; i < n; ++i) { d[i] = s[i * step]; }
        }
    }
};

/// The shared index/bounds walk: for every (channel, tap, roi row) it fills the
/// padded faces and hands the valid run to the store policy, which decides
/// where k row / column n lands.
template <class Store>
inline void im2col_2d_walk(
    const typename Store::src_t* input,
    int roi_x, int roi_y, int roi_w, int roi_h,
    int pad_top_local, int pad_left_local,
    int pad_top_global, int pad_left_global,
    int in_channel, int in_h, int in_w,
    int icn_step, int ih_step,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int dilate_h, int dilate_w,
    const Store& store) noexcept
{
    using TSrc = typename Store::src_t;

    const int karea = kernel_h * kernel_w;

    for (int ic = 0; ic < in_channel; ++ic) {
        const TSrc* input_ptr = input + ic * icn_step;
        for (int k = 0; k < karea; ++k) {
            const int kh = k / kernel_w;
            const int kw = k % kernel_w;
            const int row_k = ic * karea + k;

            const int ikh = kh * dilate_h;
            const int ikw = kw * dilate_w;

            // Valid output range inside this roi for this tap.
            const int oh_beg = std::max(0, (int)std::ceil((pad_top_local - ikh) / (float)stride_h) - roi_y);
            const int oh_end = std::max(oh_beg, std::min(roi_h, (int)std::ceil((pad_top_global + in_h - ikh) / (float)stride_h) - roi_y));
            const int ow_beg = std::max(0, (int)std::ceil((pad_left_local - ikw) / (float)stride_w) - roi_x);
            const int ow_end = std::max(ow_beg, std::min(roi_w, (int)std::ceil((pad_left_global + in_w - ikw) / (float)stride_w) - roi_x));

            if (oh_beg > 0) {
                store.fill(row_k, 0, oh_beg * roi_w);
            }

            int ih_paded = (oh_beg + roi_y) * stride_h - pad_top_global + ikh;
            int iw_paded = (ow_beg + roi_x) * stride_w - pad_left_global + ikw;

            for (int h = oh_beg; h < oh_end; ++h, ih_paded += stride_h) {
                const int n0 = h * roi_w;
                store.fill(row_k, n0, ow_beg);
                const int in_id = ih_paded * ih_step + iw_paded;
                store.emit(row_k, n0 + ow_beg, input_ptr + in_id, ow_end - ow_beg, stride_w);
                store.fill(row_k, n0 + ow_end, roi_w - ow_end);
            }

            if (oh_end < roi_h) {
                store.fill(row_k, oh_end * roi_w, (roi_h - oh_end) * roi_w);
            }
        }
    }
}

}  // namespace detail

/// 2D im2col for NCHW convolution.
///
/// Builds col_data[ic*karea][roi_w*roi_h] where karea = kernel_h*kernel_w and
/// the roi is an output window at (roi_x, roi_y) of size roi_w×roi_h. Row
/// (ic*karea + kh*kernel_w + kw) holds the spatial positions for that
/// (channel, tap); positions outside the input hold @p pad_value.
///
/// Strides are in elements: icn_step between input channels, ih_step between
/// input rows (the padded row pitch).
///
/// @p pad_value is what an out-of-input sample stores. It is 0 for the fp conv
/// route; the pre-quantized activation route passes its activation zero point,
/// since a quantized tensor represents the real value 0 as `zero_point`.
template <class T>
inline void tiled_im2col_2d(
    T* col_data, const T* input,
    int roi_x, int roi_y, int roi_w, int roi_h,
    int pad_top_local, int pad_left_local,
    int pad_top_global, int pad_left_global,
    int in_channel, int in_h, int in_w,
    int icn_step, int ih_step,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int dilate_h, int dilate_w,
    T pad_value = T(0)) noexcept
{
    detail::im2col_2d_walk(input, roi_x, roi_y, roi_w, roi_h,
                           pad_top_local, pad_left_local,
                           pad_top_global, pad_left_global,
                           in_channel, in_h, in_w, icn_step, ih_step,
                           kernel_h, kernel_w, stride_h, stride_w,
                           dilate_h, dilate_w,
                           detail::Im2colCopy<T>{col_data, roi_w * roi_h, pad_value});
}

/// 2D im2col that writes the uniform packed-B panels directly, fusing the
/// gather with the pack step (tile_pack_rhs) the GEMM route would otherwise
/// run: column n of k row lands at packed[(n/nr)*ldd + k*nr + (n%nr)].
///
/// The gather and the pack move the same elements, so materializing the
/// row-major col matrix in between only adds a write and a re-read of the whole
/// block — exactly the two passes the fused kernel drops. Columns past
/// roi_area in the last panel are zero-filled, because the MMA loads all nr
/// lanes of every panel row (tile_pack_rhs zero-fills them the same way).
///
/// @p ldd is the panel stride in elements (mma_pack_align(NR*kdim*esz)/esz);
/// @p NR is the panel width the MMA consumes for this dtype (nr_max_flt<T>),
/// taken as a template parameter so a full panel row is a fixed, unrolled
/// store sequence. The arguments from @p input on are identical to
/// tiled_im2col_2d.
template <class T, int NR>
inline void tiled_im2col_2d_packed(
    T* packed, int ldd, const T* input,
    int roi_x, int roi_y, int roi_w, int roi_h,
    int pad_top_local, int pad_left_local,
    int pad_top_global, int pad_left_global,
    int in_channel, int in_h, int in_w,
    int icn_step, int ih_step,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int dilate_h, int dilate_w,
    T pad_value = T(0)) noexcept
{
    const detail::Im2colPack<T, NR> store{packed, ldd, pad_value};
    detail::im2col_2d_walk(input, roi_x, roi_y, roi_w, roi_h,
                           pad_top_local, pad_left_local,
                           pad_top_global, pad_left_global,
                           in_channel, in_h, in_w, icn_step, ih_step,
                           kernel_h, kernel_w, stride_h, stride_w,
                           dilate_h, dilate_w, store);

    // Tail of the last panel: the columns outside the ROI, which fill() above
    // never reaches (it only walks [0, roi_area)). One panel's worth at most,
    // NR - (roi_area % NR) per k row.
    const int roi_area = roi_w * roi_h;
    const int tail = roi_area % NR;
    if (tail != 0) {
        const int kdim = in_channel * kernel_h * kernel_w;
        for (int k = 0; k < kdim; ++k) {
            store.fill(k, roi_area, NR - tail);
        }
    }
}

/// 3D im2col for NCDHW convolution.
///
/// Builds col_data[ic*karea3d][roi_d*roi_h*roi_w] where
/// karea3d = kernel_d*kernel_h*kernel_w and the roi is an output window at
/// (roi_od, roi_oh, roi_ow) of size roi_d×roi_h×roi_w. Row
/// (ic*karea3d + (kd*kernel_h + kh)*kernel_w + kw) holds the spatial positions
/// for that (channel, tap); positions outside the input are zero.
///
/// Unlike the 2D kernel (which zero-fills only the padded faces), this kernel
/// zeroes the whole row then copies the valid interior — simpler and clearly
/// correct for the 6-face/12-edge padding structure of a 3D block.
///
/// Strides are in elements: icn_step between input channels, id_step between
/// input depth slices, ih_step between input rows.
template <class T>
inline void tiled_im2col_3d(
    T* col_data, const T* input,
    int roi_od, int roi_oh, int roi_ow,   // roi origin (global output coords)
    int roi_d, int roi_h, int roi_w,      // roi extent (block sizes)
    int pad_d_local, int pad_h_local, int pad_w_local,
    int pad_d_global, int pad_h_global, int pad_w_global,
    int in_channel, int in_d, int in_h, int in_w,
    int icn_step, int id_step, int ih_step,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int dilate_d, int dilate_h, int dilate_w) noexcept
{
    const int karea      = kernel_d * kernel_h * kernel_w;
    const int roi_volume = roi_d * roi_h * roi_w;
    constexpr int L      = simd_lane_for<T>;

    for (int ic = 0; ic < in_channel; ++ic) {
        const T* input_ptr = input + ic * icn_step;
        for (int kd = 0; kd < kernel_d; ++kd) {
            const int ikd = kd * dilate_d;
            const int od_beg = std::max(0, (int)std::ceil((pad_d_local - ikd) / (float)stride_d) - roi_od);
            const int od_end = std::max(od_beg, std::min(roi_d, (int)std::ceil((pad_d_global + in_d - ikd) / (float)stride_d) - roi_od));

            for (int kh = 0; kh < kernel_h; ++kh) {
                const int ikh = kh * dilate_h;
                const int oh_beg = std::max(0, (int)std::ceil((pad_h_local - ikh) / (float)stride_h) - roi_oh);
                const int oh_end = std::max(oh_beg, std::min(roi_h, (int)std::ceil((pad_h_global + in_h - ikh) / (float)stride_h) - roi_oh));

                for (int kw = 0; kw < kernel_w; ++kw) {
                    const int ikw = kw * dilate_w;
                    const int ow_beg = std::max(0, (int)std::ceil((pad_w_local - ikw) / (float)stride_w) - roi_ow);
                    const int ow_end = std::max(ow_beg, std::min(roi_w, (int)std::ceil((pad_w_global + in_w - ikw) / (float)stride_w) - roi_ow));

                    T* col_ptr = col_data + (ic * karea + (kd * kernel_h + kh) * kernel_w + kw) * roi_volume;

                    // Zero the whole row, then copy the valid interior.
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

// ---- channel-last (NHWC) gathers ------------------------------------------
//
// One output pixel becomes one A row; the k axis is tap-major within the ic
// block: k = tap × icnc + (ic − icn). A tap of the channel-last input is a
// contiguous channel run, so every tile row is written as karea straight copies
// of `icnc` elements — the same access shape the 1x1 route gets for free.
//
// The order matters beyond the copy itself: it is also the weight's own order
// (HWIO: [kh][kw][ic][oc]), which is what lets the B packer read the weight
// through the ordinary panel packer instead of a transposing one.

/// Gather one pixel tile of one ic block into A: row m is the patch of output
/// pixel (oh, ow0 + m), k = tap × icnc + local_ic. @p in points at the block's
/// first channel of pixel (0, 0); @p pad fills taps that fall outside the input.
/// @p p0 is that pixel as a flat index (oh × OW + ow0) and @p flat lets the tile
/// walk past the end of a row into the next one — legal only when the rows are
/// dense, which is where a short plane stops paying MR granularity for tiles the
/// width cannot fill.
template <class T>
inline void tiled_im2col_nhwc(T* NNOPS_RESTRICT a, int lda,
                              const T* NNOPS_RESTRICT in,
                              int64_t in_row_stride, int64_t ic_stride, T pad,
                              int64_t p0, int OW, bool flat, int mc,
                              int IH, int IW, int KH, int KW, int SH, int SW,
                              int DH, int DW, int PH, int PW, int icnc,
                              int karea) noexcept
{
    using namespace nnops::simd;
    constexpr int L = simd_lane_for<T>;
    // A whole vector per step whenever the channel run is a multiple of the
    // lane count — which every block of the ladder except a tail is. The
    // element loop this replaces was as expensive as the GEMM it feeds.
    const bool vec = (icnc % L) == 0;
    const float pad_f = [&] {
        if constexpr (std::is_same_v<T, float>) { return pad; }
        else { return half_to_float(pad); }
    }();
    const auto copy_run = [=](T* dst, const T* src) noexcept {
        if (!vec) {
            // A short channel run (a tail block, the border taps of a narrow-IC
            // layer) still copies as one block — the element loop this replaces
            // was as expensive as the GEMM it feeds.
            if (src) { for (int j = 0; j < icnc; ++j) { dst[j] = src[j]; } }
            else     { for (int j = 0; j < icnc; ++j) { dst[j] = pad; } }
            return;
        }
        if (src) {
            for (int j = 0; j < icnc; j += L) { v_store(dst + j, v_load(src + j)); }
        } else {
            const auto vp = v_set1(dst, pad_f);
            for (int j = 0; j < icnc; j += L) { v_store(dst + j, vp); }
        }
    };

    int64_t oh = p0 / OW;
    int ow = static_cast<int>(p0 - oh * OW);
    // When every channel of a pixel is in this block and the taps of a filter row
    // are adjacent, that row of taps is one run of `KW × icnc` elements in both
    // the input and the A row — the k order is tap-major, which is exactly the
    // input's own order. One copy per filter row instead of one per tap.
    const bool tap_run = (DW == 1) && (ic_stride == icnc);
    const auto copy_span = [=](T* dst, const T* src, int n) noexcept {
        if (!src) {
            const auto vp = v_set1(dst, pad_f);
            int j = 0;
            if (vec) { for (; j + L <= n; j += L) { v_store(dst + j, vp); } }
            for (; j < n; ++j) { dst[j] = pad; }
            return;
        }
        if (n >= L) {
            int j = 0;
            for (; j + L <= n; j += L) { v_store(dst + j, v_load(src + j)); }
            // The tail rides on one overlapping vector rather than going scalar.
            if (j < n) { v_store(dst + n - L, v_load(src + n - L)); }
        } else {
            for (int j = 0; j < n; ++j) { dst[j] = src[j]; }
        }
    };
    for (int m = 0; m < mc; ++m, ++ow) {
        if (flat && ow == OW) { ow = 0; ++oh; }
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
                copy_run(dst, (h_ok && iw >= 0 && iw < IW) ? hrow + iw * ic_stride
                                                           : nullptr);
            }
        }
    }
}

/// The same gather for the quantized route: A is s8 bytes and the source is
/// always *bytes* (a materialized plane is already s8, a pre-quantized
/// activation is u8 or s8), with the A shift applied on the way through. @p xr
/// is the XOR (+0x80 for a u8 source, 0 otherwise) that puts the byte in signed
/// range, and @p pad is the zero point that fills taps outside the input. The k
/// tail up to @p kdim4 is zeroed so the direct-A kernel — which reads whole
/// 4-byte groups and must be handed a multiple of 4 — sees a zero product there
/// (the B panel is zero-padded over the same range).
///
/// @p p0 / @p flat mean exactly what they do in the fp gather.
inline void tiled_im2col_nhwc_i8(int8_t* NNOPS_RESTRICT a, int lda,
                                 const uint8_t* NNOPS_RESTRICT in,
                                 int64_t in_row_stride, int64_t ic_stride,
                                 uint8_t xr, uint8_t pad,
                                 int64_t p0, int OW, bool flat, int mc,
                                 int IH, int IW, int KH, int KW, int SH, int SW,
                                 int DH, int DW, int PH, int PW,
                                 int icnc, int karea, int kdim, int kdim4) noexcept
{
    // The channel run of one tap is the whole copy, and the block ladder hands
    // it to us in whole 8-byte chunks (16 for every karea <= 9 filter): one
    // 8-byte load/xor/store per chunk instead of a byte loop, which is what
    // used to make the gather as expensive as the GEMM it feeds.
    const uint64_t xr64 = static_cast<uint64_t>(xr) * 0x0101010101010101ULL;
    const uint8_t pv   = static_cast<uint8_t>(pad ^ xr);
    // @p src == nullptr fills the run with the padded value instead.
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
            for (int j = 0; j < icnc; ++j) { dst[j] = static_cast<int8_t>(src[j] ^ xr); }
        } else {
            std::memset(dst, pv, static_cast<size_t>(icnc));
        }
    };

    // When every channel of a pixel is in this block and the taps of a filter row
    // are adjacent in the input, that row of taps is one run in both the input
    // and the A row (k is tap-major, the input's own order): one XOR copy per
    // filter row instead of one per tap. The 3-channel stem is the case this
    // exists for — 21 bytes a row instead of seven 3-byte runs.
    const bool tap_run = (DW == 1) && (ic_stride == icnc);
    const auto xor_span = [xr, xr64, pv](int8_t* dst, const uint8_t* src, int n) noexcept {
        int j = 0;
        if (!src) {
            for (; j + 8 <= n; j += 8) { std::memset(dst + j, pv, 8); }
            for (; j < n; ++j) { dst[j] = static_cast<int8_t>(pv); }
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
                for (; j < n; ++j) { dst[j] = static_cast<int8_t>(src[j] ^ xr); }
            }
        }
    };

    int64_t oh = p0 / OW;
    int ow = static_cast<int>(p0 - oh * OW);
    for (int m = 0; m < mc; ++m, ++ow) {
        if (flat && ow == OW) { ow = 0; ++oh; }
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
                copy_run(dst, (h_ok && iw >= 0 && iw < IW)
                                  ? hrow + iw * ic_stride : nullptr);
            }
        }
        if (kdim4 > kdim) {
            std::memset(row + kdim, 0, static_cast<size_t>(kdim4 - kdim));
        }
    }
}

}  // namespace nnops::kernel
