#pragma once
/// @file simd_layout_convert.hpp
/// @brief SIMD kernel functions for NCHW ↔ NCHWC8 layout conversion.
///
/// Uses an 8×8 SIMD transpose to vectorize both loads AND stores —
/// 8 vector loads + transpose + 8 vector stores processes 8 W-positions
/// at once (64 elements per iteration). Works for both f32 and f16.

#include "nnops/detail/simd/simd.hpp"

#include <algorithm>

namespace nnops::kernel {

using namespace simd;

// ============================================================
// Per-element helpers — used for partial/remainder
// ============================================================

template <typename T>
inline void layout_pack_one_w(const T* in_row, T* out_row,
                               int64_t w, int64_t c_base, int64_t ch_stride,
                               int64_t valid_lanes) {
    T tmp[8] = {};
    for (int64_t lane = 0; lane < valid_lanes; ++lane) {
        tmp[lane] = in_row[(c_base + lane) * ch_stride + w];
    }
    v_store(&out_row[w * 8], v_load(tmp));
}

template <typename T>
inline void layout_unpack_one_w(const T* in_row, T* out_row,
                                 int64_t w, int64_t c_base, int64_t ch_stride,
                                 int64_t valid_lanes) {
    T tmp[8] = {};
    v_store(tmp, v_load(&in_row[w * 8]));
    for (int64_t lane = 0; lane < valid_lanes; ++lane) {
        out_row[(c_base + lane) * ch_stride + w] = tmp[lane];
    }
}

// ============================================================
// Pack: NCHW → NCHWC8 (one physical row at a time)
// ============================================================

template <typename T>
inline void layout_pack_row(
    const T* in_row, T* out_row,
    int64_t W, int64_t c_base, int64_t ch_stride,
    int64_t valid_lanes, int64_t C)
{
    auto vzero = v_set1(in_row, 0);
    int64_t w = 0;
    for (; w + 8 <= W; w += 8) {
        auto c0 = c_base + 0 < C ? v_load(&in_row[(c_base + 0) * ch_stride + w]) : vzero;
        auto c1 = c_base + 1 < C ? v_load(&in_row[(c_base + 1) * ch_stride + w]) : vzero;
        auto c2 = c_base + 2 < C ? v_load(&in_row[(c_base + 2) * ch_stride + w]) : vzero;
        auto c3 = c_base + 3 < C ? v_load(&in_row[(c_base + 3) * ch_stride + w]) : vzero;
        auto c4 = c_base + 4 < C ? v_load(&in_row[(c_base + 4) * ch_stride + w]) : vzero;
        auto c5 = c_base + 5 < C ? v_load(&in_row[(c_base + 5) * ch_stride + w]) : vzero;
        auto c6 = c_base + 6 < C ? v_load(&in_row[(c_base + 6) * ch_stride + w]) : vzero;
        auto c7 = c_base + 7 < C ? v_load(&in_row[(c_base + 7) * ch_stride + w]) : vzero;

        v_transpose_8x8(c0, c1, c2, c3, c4, c5, c6, c7);

        v_store(&out_row[(w + 0) * 8], c0);
        v_store(&out_row[(w + 1) * 8], c1);
        v_store(&out_row[(w + 2) * 8], c2);
        v_store(&out_row[(w + 3) * 8], c3);
        v_store(&out_row[(w + 4) * 8], c4);
        v_store(&out_row[(w + 5) * 8], c5);
        v_store(&out_row[(w + 6) * 8], c6);
        v_store(&out_row[(w + 7) * 8], c7);
    }
    for (; w < W; ++w) {
        layout_pack_one_w<T>(in_row, out_row, w, c_base, ch_stride, valid_lanes);
    }
}

// ============================================================
// Unpack: NCHWC8 → NCHW (one physical row at a time)
// ============================================================

template <typename T>
inline void layout_unpack_row(
    const T* in_row, T* out_row,
    int64_t W, int64_t c_base, int64_t ch_stride,
    int64_t valid_lanes, int64_t C)
{
    int64_t w = 0;
    for (; w + 8 <= W; w += 8) {
        auto r0 = v_load(&in_row[(w + 0) * 8]);
        auto r1 = v_load(&in_row[(w + 1) * 8]);
        auto r2 = v_load(&in_row[(w + 2) * 8]);
        auto r3 = v_load(&in_row[(w + 3) * 8]);
        auto r4 = v_load(&in_row[(w + 4) * 8]);
        auto r5 = v_load(&in_row[(w + 5) * 8]);
        auto r6 = v_load(&in_row[(w + 6) * 8]);
        auto r7 = v_load(&in_row[(w + 7) * 8]);

        v_transpose_8x8(r0, r1, r2, r3, r4, r5, r6, r7);

        if (c_base + 0 < C) {
            v_store(&out_row[(c_base + 0) * ch_stride + w], r0);
        }
        if (c_base + 1 < C) {
            v_store(&out_row[(c_base + 1) * ch_stride + w], r1);
        }
        if (c_base + 2 < C) {
            v_store(&out_row[(c_base + 2) * ch_stride + w], r2);
        }
        if (c_base + 3 < C) {
            v_store(&out_row[(c_base + 3) * ch_stride + w], r3);
        }
        if (c_base + 4 < C) {
            v_store(&out_row[(c_base + 4) * ch_stride + w], r4);
        }
        if (c_base + 5 < C) {
            v_store(&out_row[(c_base + 5) * ch_stride + w], r5);
        }
        if (c_base + 6 < C) {
            v_store(&out_row[(c_base + 6) * ch_stride + w], r6);
        }
        if (c_base + 7 < C) {
            v_store(&out_row[(c_base + 7) * ch_stride + w], r7);
        }
    }
    for (; w < W; ++w) {
        layout_unpack_one_w<T>(in_row, out_row, w, c_base, ch_stride, valid_lanes);
    }
}

// ============================================================
// Byte-copy pack/unpack for s8/u8 (8 channels = 8 bytes)
// ============================================================
//
// The 8×8 transpose above operates on 8-lane floating vectors (f32/f16) and is
// not valid for int8/uint8 (whose `v_load` returns 16 lanes). These byte paths
// copy 8 channel bytes per W position directly. Pack fills pad lanes (c >= C)
// with `pad_val` — the activation's zero_point — so that padded channels in a
// quantized NCHWC8 tensor dequantize to zero contribution.

template <typename T>
inline void layout_pack_row_byte(
    const T* in_row, T* out_row,
    int64_t W, int64_t c_base, int64_t ch_stride,
    int64_t /*valid_lanes*/, int64_t C, T pad_val)
{
    for (int64_t w = 0; w < W; ++w) {
        for (int64_t lane = 0; lane < 8; ++lane) {
            const int64_t c = c_base + lane;
            out_row[w * 8 + lane] = (c < C) ? in_row[c * ch_stride + w] : pad_val;
        }
    }
}

template <typename T>
inline void layout_unpack_row_byte(
    const T* in_row, T* out_row,
    int64_t W, int64_t c_base, int64_t ch_stride,
    int64_t valid_lanes, int64_t /*C*/)
{
    for (int64_t w = 0; w < W; ++w) {
        for (int64_t lane = 0; lane < valid_lanes; ++lane) {
            out_row[(c_base + lane) * ch_stride + w] = in_row[w * 8 + lane];
        }
    }
}

}  // namespace nnops::kernel
