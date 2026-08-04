#pragma once
/// @file simd_resize.hpp
/// @brief SIMD kernel functions for 2D/3D spatial resize (NCHWC8/NCDHWC8 only).
///
/// Input/output are always channel-packed (NCHWC8 or NCDHWC8). All C8 blocks
/// use SIMD — partial C8 blocks are zero-padded by the pack step.
///
/// Key design:
///   1. Nearest: compute source coordinate once per output position, then
///      v_load 8 channels at once from the source NCHWC8 position.
///   2. Bilinear/Trilinear: compute source coordinates once, load 4/8 corner
///      vectors (8 channels each), interpolate all 8 lanes in parallel via
///      v_fmadd (a*b + c).

#include "nnops/detail/simd/simd.hpp"
#include "nnops/ops/resize.hpp"

#include <algorithm>
#include <cmath>

namespace nnops::kernel {

using namespace simd;

// ============================================================
// Coordinate helpers
// ============================================================

inline int64_t resize_clamp_idx(int64_t idx, int64_t src_size) {
    return std::max<int64_t>(0, std::min(idx, src_size - 1));
}

inline float resize_compute_src_coord(int64_t dst_idx, int64_t src_size, int64_t dst_size,
                                      CoordinateTransformMode mode) {
    if (dst_size == src_size) {
        return static_cast<float>(dst_idx);
    }
    switch (mode) {
    case CoordinateTransformMode::HalfPixel: {
        float scale = static_cast<float>(src_size) / static_cast<float>(dst_size);
        return (static_cast<float>(dst_idx) + 0.5f) * scale - 0.5f;
    }
    case CoordinateTransformMode::AlignCorners: {
        if (dst_size <= 1) { return 0.0f; }
        return static_cast<float>(dst_idx) *
               static_cast<float>(src_size - 1) /
               static_cast<float>(dst_size - 1);
    }
    case CoordinateTransformMode::Asymmetric:
    default: {
        float scale = static_cast<float>(src_size) / static_cast<float>(dst_size);
        return static_cast<float>(dst_idx) * scale;
    }
    }
}

// ============================================================
// 2D Nearest-neighbor NCHWC8 SIMD kernel
// ============================================================

template <typename T>
void resize_nearest_2d(
    T* output, const T* input,
    int64_t IH, int64_t IW, int64_t OH, int64_t OW,
    int64_t in_row_stride, int64_t out_row_stride,
    CoordinateTransformMode coord_mode, bool add_to)
{
    for (int64_t oh = 0; oh < OH; ++oh) {
        float src_h = resize_compute_src_coord(oh, IH, OH, coord_mode);
        int64_t ih = resize_clamp_idx(static_cast<int64_t>(std::round(src_h)), IH);
        const T* in_row = input + ih * in_row_stride;

        for (int64_t ow = 0; ow < OW; ++ow) {
            float src_w = resize_compute_src_coord(ow, IW, OW, coord_mode);
            int64_t iw = resize_clamp_idx(static_cast<int64_t>(std::round(src_w)), IW);

            auto val = v_load(&in_row[iw * 8]);
            v_store_add(output + ow * 8, val, add_to);
        }
        output += out_row_stride;
    }
}

// ============================================================
// 2D Bilinear NCHWC8 SIMD kernel
// ============================================================

template <typename T>
void resize_bilinear_2d(
    T* output, const T* input,
    int64_t IH, int64_t IW, int64_t OH, int64_t OW,
    int64_t in_row_stride, int64_t out_row_stride,
    CoordinateTransformMode coord_mode, bool add_to)
{
    const T* type_tag = output;

    for (int64_t oh = 0; oh < OH; ++oh) {
        float src_h = resize_compute_src_coord(oh, IH, OH, coord_mode);
        int64_t y0 = resize_clamp_idx(static_cast<int64_t>(std::floor(src_h)), IH);
        int64_t y1 = resize_clamp_idx(y0 + 1, IH);
        float wy = src_h - std::floor(src_h);
        float wy0 = 1.0f - wy;
        auto v_wy  = v_set1(type_tag, wy);
        auto v_wy0 = v_set1(type_tag, wy0);

        const T* in_row0 = input + y0 * in_row_stride;
        const T* in_row1 = input + y1 * in_row_stride;

        for (int64_t ow = 0; ow < OW; ++ow) {
            float src_w = resize_compute_src_coord(ow, IW, OW, coord_mode);
            int64_t x0 = resize_clamp_idx(static_cast<int64_t>(std::floor(src_w)), IW);
            int64_t x1 = resize_clamp_idx(x0 + 1, IW);
            float wx = src_w - std::floor(src_w);
            float wx0 = 1.0f - wx;
            auto v_wx  = v_set1(type_tag, wx);
            auto v_wx0 = v_set1(type_tag, wx0);

            // Load 4 corner vectors (8 channels each)
            auto v00 = v_load(&in_row0[x0 * 8]);
            auto v10 = v_load(&in_row0[x1 * 8]);
            auto v01 = v_load(&in_row1[x0 * 8]);
            auto v11 = v_load(&in_row1[x1 * 8]);

            // result = wy0 * (wx0 * v00 + wx * v10) + wy * (wx0 * v01 + wx * v11)
            auto top0 = v_fmadd(v_wx0, v00, v_mul(v_wx, v10));
            auto top1 = v_fmadd(v_wx0, v01, v_mul(v_wx, v11));
            auto result = v_fmadd(v_wy0, top0, v_mul(v_wy, top1));

            v_store_add(output + ow * 8, result, add_to);
        }
        output += out_row_stride;
    }
}

// ============================================================
// 3D Nearest-neighbor NCDHWC8 SIMD kernel
// ============================================================

template <typename T>
void resize_nearest_3d(
    T* output, const T* input,
    int64_t ID, int64_t IH, int64_t IW,
    int64_t OD, int64_t OH, int64_t OW,
    int64_t in_d_stride, int64_t in_row_stride,
    int64_t out_d_stride, int64_t out_row_stride,
    CoordinateTransformMode coord_mode, bool add_to)
{
    for (int64_t od = 0; od < OD; ++od) {
        float src_d = resize_compute_src_coord(od, ID, OD, coord_mode);
        int64_t id = resize_clamp_idx(static_cast<int64_t>(std::round(src_d)), ID);
        const T* in_d = input + id * in_d_stride;
        T* out_d = output + od * out_d_stride;

        for (int64_t oh = 0; oh < OH; ++oh) {
            float src_h = resize_compute_src_coord(oh, IH, OH, coord_mode);
            int64_t ih = resize_clamp_idx(static_cast<int64_t>(std::round(src_h)), IH);
            const T* in_row = in_d + ih * in_row_stride;

            for (int64_t ow = 0; ow < OW; ++ow) {
                float src_w = resize_compute_src_coord(ow, IW, OW, coord_mode);
                int64_t iw = resize_clamp_idx(static_cast<int64_t>(std::round(src_w)), IW);

                auto val = v_load(&in_row[iw * 8]);
                v_store_add(out_d + oh * out_row_stride + ow * 8, val, add_to);
            }
        }
    }
}

// ============================================================
// 3D Trilinear NCDHWC8 SIMD kernel
// ============================================================

template <typename T>
void resize_trilinear_3d(
    T* output, const T* input,
    int64_t ID, int64_t IH, int64_t IW,
    int64_t OD, int64_t OH, int64_t OW,
    int64_t in_d_stride, int64_t in_row_stride,
    int64_t out_d_stride, int64_t out_row_stride,
    CoordinateTransformMode coord_mode, bool add_to)
{
    const T* type_tag = output;

    for (int64_t od = 0; od < OD; ++od) {
        float src_d = resize_compute_src_coord(od, ID, OD, coord_mode);
        int64_t z0 = resize_clamp_idx(static_cast<int64_t>(std::floor(src_d)), ID);
        int64_t z1 = resize_clamp_idx(z0 + 1, ID);
        float wz   = src_d - std::floor(src_d);
        float wz0  = 1.0f - wz;
        auto v_wz  = v_set1(type_tag, wz);
        auto v_wz0 = v_set1(type_tag, wz0);

        const T* in_d0 = input + z0 * in_d_stride;
        const T* in_d1 = input + z1 * in_d_stride;
        T* out_d = output + od * out_d_stride;

        for (int64_t oh = 0; oh < OH; ++oh) {
            float src_h = resize_compute_src_coord(oh, IH, OH, coord_mode);
            int64_t y0 = resize_clamp_idx(static_cast<int64_t>(std::floor(src_h)), IH);
            int64_t y1 = resize_clamp_idx(y0 + 1, IH);
            float wy   = src_h - std::floor(src_h);
            float wy0  = 1.0f - wy;
            auto v_wy  = v_set1(type_tag, wy);
            auto v_wy0 = v_set1(type_tag, wy0);

            const T* in_d0_r0 = in_d0 + y0 * in_row_stride;
            const T* in_d0_r1 = in_d0 + y1 * in_row_stride;
            const T* in_d1_r0 = in_d1 + y0 * in_row_stride;
            const T* in_d1_r1 = in_d1 + y1 * in_row_stride;

            for (int64_t ow = 0; ow < OW; ++ow) {
                float src_w = resize_compute_src_coord(ow, IW, OW, coord_mode);
                int64_t x0 = resize_clamp_idx(static_cast<int64_t>(std::floor(src_w)), IW);
                int64_t x1 = resize_clamp_idx(x0 + 1, IW);
                float wx   = src_w - std::floor(src_w);
                float wx0  = 1.0f - wx;
                auto v_wx  = v_set1(type_tag, wx);
                auto v_wx0 = v_set1(type_tag, wx0);

                // Load 8 corner vectors (8 channels each)
                auto v000 = v_load(&in_d0_r0[x0 * 8]);
                auto v100 = v_load(&in_d0_r0[x1 * 8]);
                auto v010 = v_load(&in_d0_r1[x0 * 8]);
                auto v110 = v_load(&in_d0_r1[x1 * 8]);
                auto v001 = v_load(&in_d1_r0[x0 * 8]);
                auto v101 = v_load(&in_d1_r0[x1 * 8]);
                auto v011 = v_load(&in_d1_r1[x0 * 8]);
                auto v111 = v_load(&in_d1_r1[x1 * 8]);

                // 2D interpolation in each z-plane, then interp across z
                auto r00 = v_fmadd(v_wx0, v000, v_mul(v_wx, v100));  // z0
                auto r10 = v_fmadd(v_wx0, v010, v_mul(v_wx, v110));
                auto z0_term = v_fmadd(v_wy0, r00, v_mul(v_wy, r10));

                auto r01 = v_fmadd(v_wx0, v001, v_mul(v_wx, v101));  // z1
                auto r11 = v_fmadd(v_wx0, v011, v_mul(v_wx, v111));
                auto z1_term = v_fmadd(v_wy0, r01, v_mul(v_wy, r11));

                auto result = v_fmadd(v_wz0, z0_term, v_mul(v_wz, z1_term));

                v_store_add(out_d + oh * out_row_stride + ow * 8, result, add_to);
            }
        }
    }
}

}  // namespace nnops::kernel
