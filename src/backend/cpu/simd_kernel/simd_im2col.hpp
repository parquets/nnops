#pragma once
/// @file simd_im2col.hpp
/// @brief SIMD im2col kernels — the "unrolled input" step of im2col+GEMM conv.
///
/// Two kernels build the dense input matrix consumed by the GEMM:
///   tiled_im2col_2d  — NCHW  Conv2D: col_data[ic*karea2d][roi_h*roi_w]
///   tiled_im2col_3d  — NCDHW Conv3D: col_data[ic*karea3d][roi_d*roi_h*roi_w]
///
/// Each row (ic, kd, kh, kw) holds the input spatial positions gathered for one
/// input channel and one kernel tap, across the current output block (the roi).
/// Positions that fall outside the input (padding) are zero-filled.
///
/// Padding convention (mirrors nn_compute tiled_im2col_2d):
///   * *local* padding — the padding remaining inside this block (roi_pad_*)
///     — determines where the valid region *starts* (the `*_beg` bounds).
///   * *global* padding — the operator's padding (PD/PH/PW) — determines where
///     the valid region *ends* (`*_end` bounds) and the actual input addressing
///     (`id_paded`/`ih_paded`/`iw_paded`).
///
/// Mixing these up (using the local padding for the input addressing) produces
/// off-by-padding input rows on every block after the first.
///
/// The innermost (width) copy uses an L-wide SIMD loop when stride_w == 1 (the
/// dominant case); otherwise a scalar loop advancing by stride_w.
///
/// `tiled_im2col_2d` writes the input values through unchanged, with a
/// configurable pad value. That is all the quantized Conv2D route needs: it
/// quantizes an fp activation into a plane up front (conv2d_im2col_int8.cpp,
/// quantize_plane) and gathers bytes from then on, so no store policy other
/// than "copy" is required.

#include "nnops/detail/simd/simd.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>

namespace nnops::kernel {

using namespace simd;

namespace detail {

/// Copy policy: write the gathered values (and `pad` outside the input)
/// unchanged. `pad` is the value standing in for a zero input sample, which is
/// the activation zero point on the pre-quantized route.
template <class T>
struct Im2colCopy {
    using src_t = T;
    using dst_t = T;

    T pad;

    void fill(T* d, int n) const noexcept {
        if constexpr (std::is_integral_v<T>) {
            // The int8 activation path: only +0 exists, and no SIMD helper
            // handles the byte types.
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

    void emit(T* d, const T* s, int n, int step) const noexcept {
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

/// The shared index/bounds walk: for every (channel, tap, roi row) it fills the
/// padded faces and hands the valid run to the store policy.
template <class Store>
inline void im2col_2d_walk(
    typename Store::dst_t* col_data, const typename Store::src_t* input,
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
    using TDst = typename Store::dst_t;
    using TSrc = typename Store::src_t;

    const int karea    = kernel_h * kernel_w;
    const int roi_area = roi_w * roi_h;

    for (int ic = 0; ic < in_channel; ++ic) {
        const TSrc* input_ptr = input + ic * icn_step;
        for (int k = 0; k < karea; ++k) {
            const int kh = k / kernel_w;
            const int kw = k % kernel_w;
            TDst* col_ptr = col_data + (ic * karea + k) * roi_area;

            const int ikh = kh * dilate_h;
            const int ikw = kw * dilate_w;

            // Valid output range inside this roi for this tap.
            const int oh_beg = std::max(0, (int)std::ceil((pad_top_local - ikh) / (float)stride_h) - roi_y);
            const int oh_end = std::max(oh_beg, std::min(roi_h, (int)std::ceil((pad_top_global + in_h - ikh) / (float)stride_h) - roi_y));
            const int ow_beg = std::max(0, (int)std::ceil((pad_left_local - ikw) / (float)stride_w) - roi_x);
            const int ow_end = std::max(ow_beg, std::min(roi_w, (int)std::ceil((pad_left_global + in_w - ikw) / (float)stride_w) - roi_x));

            if (oh_beg > 0) {
                store.fill(col_ptr, oh_beg * roi_w);
            }

            int ih_paded = (oh_beg + roi_y) * stride_h - pad_top_global + ikh;
            int iw_paded = (ow_beg + roi_x) * stride_w - pad_left_global + ikw;

            for (int h = oh_beg; h < oh_end; ++h, ih_paded += stride_h) {
                TDst* row = col_ptr + h * roi_w;
                store.fill(row, ow_beg);
                const int in_id = ih_paded * ih_step + iw_paded;
                store.emit(row + ow_beg, input_ptr + in_id, ow_end - ow_beg, stride_w);
                store.fill(row + ow_end, roi_w - ow_end);
            }

            if (oh_end < roi_h) {
                store.fill(col_ptr + oh_end * roi_w, (roi_h - oh_end) * roi_w);
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
    detail::im2col_2d_walk(col_data, input, roi_x, roi_y, roi_w, roi_h,
                           pad_top_local, pad_left_local,
                           pad_top_global, pad_left_global,
                           in_channel, in_h, in_w, icn_step, ih_step,
                           kernel_h, kernel_w, stride_h, stride_w,
                           dilate_h, dilate_w,
                           detail::Im2colCopy<T>{pad_value});
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

}  // namespace nnops::kernel
