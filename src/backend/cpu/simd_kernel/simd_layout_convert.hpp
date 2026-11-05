#pragma once
/// @file simd_layout_convert.hpp
/// @brief CPU kernels for channel-first ↔ channel-last conversion.
///
/// A spatial row is W pixels × C channels; channel-first stores it as C channel rows
/// of W, channel-last as W pixels of C. Converting one 8-channel block against
/// 8 pixels is an 8×8 transpose, so f32/f16 move 64 elements per iteration with
/// 8 vector loads, one transpose, and 8 vector stores. Strided stores are fine:
/// a transposed vector lands on one pixel's 8 channels (stride 1) at a C-pixel
/// pitch, or on one channel row's 8 W positions. s8/u8 use scalar moves because
/// their `v_load` is 16 lanes.

#include "nnops/detail/simd/simd.hpp"

#include <algorithm>

namespace nnops::kernel {

using namespace simd;

template <typename T>
inline constexpr bool layout_convert_simd8 =
    std::is_same_v<T, float> || std::is_same_v<T, half>;

/// Copy one 8-channel block of a channel-first row into channel-last rows:
/// `out[w*C + c_base + lane] = in[(c_base + lane)*ch_stride + w]`.
template <typename T>
inline void layout_channel_first_to_cl_block(const T* in_row, T* out_row,
                                      int64_t W, int64_t C, int64_t ch_stride,
                                      int64_t c_base, int64_t valid_lanes)
{
    if constexpr (layout_convert_simd8<T>) {
        if (valid_lanes == 8) {
            int64_t w = 0;
            for (; w + 8 <= W; w += 8) {
                auto c0 = v_load(&in_row[(c_base + 0) * ch_stride + w]);
                auto c1 = v_load(&in_row[(c_base + 1) * ch_stride + w]);
                auto c2 = v_load(&in_row[(c_base + 2) * ch_stride + w]);
                auto c3 = v_load(&in_row[(c_base + 3) * ch_stride + w]);
                auto c4 = v_load(&in_row[(c_base + 4) * ch_stride + w]);
                auto c5 = v_load(&in_row[(c_base + 5) * ch_stride + w]);
                auto c6 = v_load(&in_row[(c_base + 6) * ch_stride + w]);
                auto c7 = v_load(&in_row[(c_base + 7) * ch_stride + w]);

                v_transpose_8x8(c0, c1, c2, c3, c4, c5, c6, c7);

                v_store(&out_row[(w + 0) * C + c_base], c0);
                v_store(&out_row[(w + 1) * C + c_base], c1);
                v_store(&out_row[(w + 2) * C + c_base], c2);
                v_store(&out_row[(w + 3) * C + c_base], c3);
                v_store(&out_row[(w + 4) * C + c_base], c4);
                v_store(&out_row[(w + 5) * C + c_base], c5);
                v_store(&out_row[(w + 6) * C + c_base], c6);
                v_store(&out_row[(w + 7) * C + c_base], c7);
            }
            for (; w < W; ++w) {
                for (int64_t lane = 0; lane < 8; ++lane) {
                    out_row[w * C + c_base + lane] =
                        in_row[(c_base + lane) * ch_stride + w];
                }
            }
            return;
        }
    }

    for (int64_t w = 0; w < W; ++w) {
        for (int64_t lane = 0; lane < valid_lanes; ++lane) {
            out_row[w * C + c_base + lane] =
                in_row[(c_base + lane) * ch_stride + w];
        }
    }
}

/// The reverse: gather one 8-channel block of a channel-last row back into
/// Channel-first channel rows.
template <typename T>
inline void layout_cl_to_channel_first_block(const T* in_row, T* out_row,
                                      int64_t W, int64_t C, int64_t ch_stride,
                                      int64_t c_base, int64_t valid_lanes)
{
    if constexpr (layout_convert_simd8<T>) {
        if (valid_lanes == 8) {
            int64_t w = 0;
            for (; w + 8 <= W; w += 8) {
                auto r0 = v_load(&in_row[(w + 0) * C + c_base]);
                auto r1 = v_load(&in_row[(w + 1) * C + c_base]);
                auto r2 = v_load(&in_row[(w + 2) * C + c_base]);
                auto r3 = v_load(&in_row[(w + 3) * C + c_base]);
                auto r4 = v_load(&in_row[(w + 4) * C + c_base]);
                auto r5 = v_load(&in_row[(w + 5) * C + c_base]);
                auto r6 = v_load(&in_row[(w + 6) * C + c_base]);
                auto r7 = v_load(&in_row[(w + 7) * C + c_base]);

                v_transpose_8x8(r0, r1, r2, r3, r4, r5, r6, r7);

                v_store(&out_row[(c_base + 0) * ch_stride + w], r0);
                v_store(&out_row[(c_base + 1) * ch_stride + w], r1);
                v_store(&out_row[(c_base + 2) * ch_stride + w], r2);
                v_store(&out_row[(c_base + 3) * ch_stride + w], r3);
                v_store(&out_row[(c_base + 4) * ch_stride + w], r4);
                v_store(&out_row[(c_base + 5) * ch_stride + w], r5);
                v_store(&out_row[(c_base + 6) * ch_stride + w], r6);
                v_store(&out_row[(c_base + 7) * ch_stride + w], r7);
            }
            for (; w < W; ++w) {
                for (int64_t lane = 0; lane < 8; ++lane) {
                    out_row[(c_base + lane) * ch_stride + w] =
                        in_row[w * C + c_base + lane];
                }
            }
            return;
        }
    }

    for (int64_t w = 0; w < W; ++w) {
        for (int64_t lane = 0; lane < valid_lanes; ++lane) {
            out_row[(c_base + lane) * ch_stride + w] =
                in_row[w * C + c_base + lane];
        }
    }
}

}  // namespace nnops::kernel
