#pragma once
/// @file simd_grid_sample.hpp
/// @brief SIMD kernel functions for 2D/3D grid sampling (NCHWC8/NCDHWC8 only).
///
/// Input (image to sample) is always NCHWC8 (or NCDHWC8 for 3D). Grid is always
/// plain NCHW/NCDHW float32. All C8 blocks use SIMD — grid coordinates are
/// per-output-position, so validity is all-or-nothing per C8 block.
///
/// Key design:
///   1. Nearest: compute pixel coordinate once per output position, v_load 8
///      channels from source NCHWC8 position, v_store.
///   2. Bilinear/Trilinear: compute pixel coords once, v_load 4/8 corner vectors
///      (8 channels each), interpolate all 8 lanes with v_fmadd (a*b + c).
///   3. Zeros padding: all-or-nothing per C8 block (grid coords are per-position).
///   4. Border/Reflection padding: always valid after padding → SIMD always works.

#include "nnops/detail/simd/simd.hpp"
#include "nnops/ops/grid_sample.hpp"

#include <algorithm>
#include <cmath>

namespace nnops::kernel {

using namespace simd;

// ============================================================
// Coordinate helpers
// ============================================================

inline float gridsample_grid_to_pixel(float coord, int64_t size, bool align_corners) {
    if (align_corners) {
        return (coord + 1.0f) * static_cast<float>(size - 1) * 0.5f;
    } else {
        return ((coord + 1.0f) * static_cast<float>(size) - 1.0f) * 0.5f;
    }
}

inline float gridsample_reflect_coord(float x, int64_t size) {
    if (size <= 1) {
        return 0.0f;
    }
    float max_val = static_cast<float>(size - 1);
    float r = std::fmod(std::abs(x), 2.0f * max_val);
    if (r > max_val) {
        r = 2.0f * max_val - r;
    }
    return r;
}

inline float gridsample_apply_padding(float pixel, int64_t size, GridSamplePaddingMode padding) {
    switch (padding) {
    case GridSamplePaddingMode::Border:
        return std::max(0.0f, std::min(pixel, static_cast<float>(size - 1)));
    case GridSamplePaddingMode::Reflection:
        return gridsample_reflect_coord(pixel, size);
    case GridSamplePaddingMode::Zeros:
    default:
        return pixel;
    }
}

inline bool gridsample_in_bounds(float pixel, int64_t size) {
    return pixel >= 0.0f && pixel <= static_cast<float>(size - 1);
}

inline int64_t gridsample_clamp_idx(int64_t idx, int64_t size) {
    return std::max<int64_t>(0, std::min(idx, size - 1));
}

/// Check if all 4 corners of a bilinear region are in bounds.
inline bool gridsample_bilinear_bounds_ok(float py, float pz, int64_t IH, int64_t IW) {
    return gridsample_in_bounds(std::floor(py), IH) &&
           gridsample_in_bounds(std::ceil(py),  IH) &&
           gridsample_in_bounds(std::floor(pz), IW) &&
           gridsample_in_bounds(std::ceil(pz),  IW);
}

/// Check if all 8 corners of a trilinear region are in bounds.
inline bool gridsample_trilinear_bounds_ok(float px, float py, float pz,
                                            int64_t ID, int64_t IH, int64_t IW) {
    return gridsample_in_bounds(std::floor(px), ID) &&
           gridsample_in_bounds(std::ceil(px),  ID) &&
           gridsample_in_bounds(std::floor(py), IH) &&
           gridsample_in_bounds(std::ceil(py),  IH) &&
           gridsample_in_bounds(std::floor(pz), IW) &&
           gridsample_in_bounds(std::ceil(pz),  IW);
}

// ============================================================
// 2D Bilinear NCHWC8 SIMD kernel
// ============================================================

template <typename T>
void gridsample_bilinear_2d(
    T* output, const T* input, const float* grid_n,
    int64_t IH, int64_t IW, int64_t OH, int64_t OW,
    int64_t in_row_stride, int64_t out_row_stride,
    int64_t grid_row_stride,
    bool align_corners, GridSamplePaddingMode padding, bool add_to)
{
    const T* type_tag = output;
    const bool zeros_pad = (padding == GridSamplePaddingMode::Zeros);

    for (int64_t oh = 0; oh < OH; ++oh) {
        const float* g_row = grid_n + oh * grid_row_stride;

        for (int64_t ow = 0; ow < OW; ++ow) {
            const float* g_pos = g_row + ow * 2;  // 2D: (y, x)
            float py = gridsample_grid_to_pixel(g_pos[0], IH, align_corners);
            float pz = gridsample_grid_to_pixel(g_pos[1], IW, align_corners);

            if (zeros_pad && !gridsample_bilinear_bounds_ok(py, pz, IH, IW)) {
                if (!add_to) {
                    v_store(output + ow * 8, v_zero(type_tag));
                }
                continue;
            }

            py = gridsample_apply_padding(py, IH, padding);
            pz = gridsample_apply_padding(pz, IW, padding);

            int64_t y0 = static_cast<int64_t>(std::floor(py));
            int64_t z0 = static_cast<int64_t>(std::floor(pz));
            int64_t y1 = gridsample_clamp_idx(y0 + 1, IH);
            int64_t z1 = gridsample_clamp_idx(z0 + 1, IW);

            float wy = py - static_cast<float>(y0);
            float wz = pz - static_cast<float>(z0);
            float wy0 = 1.0f - wy;
            float wz0 = 1.0f - wz;

            auto v_wy  = v_set1(type_tag, wy);
            auto v_wy0 = v_set1(type_tag, wy0);
            auto v_wz  = v_set1(type_tag, wz);
            auto v_wz0 = v_set1(type_tag, wz0);

            // Load 4 corner vectors (8 channels each)
            auto v00 = v_load(&input[y0 * in_row_stride + z0 * 8]);
            auto v01 = v_load(&input[y0 * in_row_stride + z1 * 8]);
            auto v10 = v_load(&input[y1 * in_row_stride + z0 * 8]);
            auto v11 = v_load(&input[y1 * in_row_stride + z1 * 8]);

            // result = wy0*(wz0*v00 + wz*v01) + wy*(wz0*v10 + wz*v11)
            auto row0 = v_fmadd(v_wz0, v00, v_mul(v_wz, v01));
            auto row1 = v_fmadd(v_wz0, v10, v_mul(v_wz, v11));
            auto result = v_fmadd(v_wy0, row0, v_mul(v_wy, row1));

            v_store_add(output + ow * 8, result, add_to);
        }
        output += out_row_stride;
    }
}

// ============================================================
// 2D Nearest-neighbor NCHWC8 SIMD kernel
// ============================================================

template <typename T>
void gridsample_nearest_2d(
    T* output, const T* input, const float* grid_n,
    int64_t IH, int64_t IW, int64_t OH, int64_t OW,
    int64_t in_row_stride, int64_t out_row_stride,
    int64_t grid_row_stride,
    bool align_corners, GridSamplePaddingMode padding, bool add_to)
{
    const T* type_tag = output;
    const bool zeros_pad = (padding == GridSamplePaddingMode::Zeros);

    for (int64_t oh = 0; oh < OH; ++oh) {
        const float* g_row = grid_n + oh * grid_row_stride;

        for (int64_t ow = 0; ow < OW; ++ow) {
            const float* g_pos = g_row + ow * 2;
            float py = gridsample_grid_to_pixel(g_pos[0], IH, align_corners);
            float pz = gridsample_grid_to_pixel(g_pos[1], IW, align_corners);

            float py_r = std::round(py);
            float pz_r = std::round(pz);

            if (zeros_pad && (!gridsample_in_bounds(py_r, IH) || !gridsample_in_bounds(pz_r, IW))) {
                if (!add_to) {
                    v_store(output + ow * 8, v_zero(type_tag));
                }
                continue;
            }

            py = gridsample_apply_padding(py, IH, padding);
            pz = gridsample_apply_padding(pz, IW, padding);
            int64_t iy = static_cast<int64_t>(std::round(py));
            int64_t iz = static_cast<int64_t>(std::round(pz));

            auto val = v_load(&input[iy * in_row_stride + iz * 8]);
            v_store_add(output + ow * 8, val, add_to);
        }
        output += out_row_stride;
    }
}

// ============================================================
// 3D Trilinear NCDHWC8 SIMD kernel
// ============================================================

template <typename T>
void gridsample_trilinear_3d(
    T* output, const T* input, const float* grid_n,
    int64_t ID, int64_t IH, int64_t IW,
    int64_t OD, int64_t OH, int64_t OW,
    int64_t in_d_stride, int64_t in_row_stride,
    int64_t out_d_stride, int64_t out_row_stride,
    int64_t grid_row_stride, int64_t /*grid_d_elems*/,
    bool align_corners, GridSamplePaddingMode padding, bool add_to)
{
    const T* type_tag = output;
    const bool zeros_pad = (padding == GridSamplePaddingMode::Zeros);

    for (int64_t od = 0; od < OD; ++od) {
        T* out_d = output + od * out_d_stride;

        for (int64_t oh = 0; oh < OH; ++oh) {
            const float* g_pos = grid_n + (od * OH + oh) * grid_row_stride;

            for (int64_t ow = 0; ow < OW; ++ow) {
                float px = gridsample_grid_to_pixel(g_pos[0], ID, align_corners);
                float py = gridsample_grid_to_pixel(g_pos[1], IH, align_corners);
                float pz = gridsample_grid_to_pixel(g_pos[2], IW, align_corners);
                g_pos += 3;  // coord_dim = 3

                if (zeros_pad && !gridsample_trilinear_bounds_ok(px, py, pz, ID, IH, IW)) {
                    if (!add_to) {
                        v_store(out_d + oh * out_row_stride + ow * 8, v_zero(type_tag));
                    }
                    continue;
                }

                px = gridsample_apply_padding(px, ID, padding);
                py = gridsample_apply_padding(py, IH, padding);
                pz = gridsample_apply_padding(pz, IW, padding);

                int64_t x0 = static_cast<int64_t>(std::floor(px));
                int64_t y0 = static_cast<int64_t>(std::floor(py));
                int64_t z0 = static_cast<int64_t>(std::floor(pz));
                int64_t x1 = gridsample_clamp_idx(x0 + 1, ID);
                int64_t y1 = gridsample_clamp_idx(y0 + 1, IH);
                int64_t z1 = gridsample_clamp_idx(z0 + 1, IW);

                float wx = px - static_cast<float>(x0);
                float wy = py - static_cast<float>(y0);
                float wz = pz - static_cast<float>(z0);
                float wx0 = 1.0f - wx;
                float wy0 = 1.0f - wy;
                float wz0 = 1.0f - wz;

                auto v_wx  = v_set1(type_tag, wx);
                auto v_wx0 = v_set1(type_tag, wx0);
                auto v_wy  = v_set1(type_tag, wy);
                auto v_wy0 = v_set1(type_tag, wy0);
                auto v_wz  = v_set1(type_tag, wz);
                auto v_wz0 = v_set1(type_tag, wz0);

                // Load 8 corner vectors
                auto v000 = v_load(&input[x0 * in_d_stride + y0 * in_row_stride + z0 * 8]);
                auto v001 = v_load(&input[x0 * in_d_stride + y0 * in_row_stride + z1 * 8]);
                auto v010 = v_load(&input[x0 * in_d_stride + y1 * in_row_stride + z0 * 8]);
                auto v011 = v_load(&input[x0 * in_d_stride + y1 * in_row_stride + z1 * 8]);
                auto v100 = v_load(&input[x1 * in_d_stride + y0 * in_row_stride + z0 * 8]);
                auto v101 = v_load(&input[x1 * in_d_stride + y0 * in_row_stride + z1 * 8]);
                auto v110 = v_load(&input[x1 * in_d_stride + y1 * in_row_stride + z0 * 8]);
                auto v111 = v_load(&input[x1 * in_d_stride + y1 * in_row_stride + z1 * 8]);

                // Interpolate across z, then y, then x
                auto r00 = v_fmadd(v_wz0, v000, v_mul(v_wz, v001));
                auto r01 = v_fmadd(v_wz0, v010, v_mul(v_wz, v011));
                auto r10 = v_fmadd(v_wz0, v100, v_mul(v_wz, v101));
                auto r11 = v_fmadd(v_wz0, v110, v_mul(v_wz, v111));

                auto s0 = v_fmadd(v_wy0, r00, v_mul(v_wy, r01));
                auto s1 = v_fmadd(v_wy0, r10, v_mul(v_wy, r11));

                auto result = v_fmadd(v_wx0, s0, v_mul(v_wx, s1));

                v_store_add(out_d + oh * out_row_stride + ow * 8, result, add_to);
            }
        }
    }
}

// ============================================================
// 3D Nearest-neighbor NCDHWC8 SIMD kernel
// ============================================================

template <typename T>
void gridsample_nearest_3d(
    T* output, const T* input, const float* grid_n,
    int64_t ID, int64_t IH, int64_t IW,
    int64_t OD, int64_t OH, int64_t OW,
    int64_t in_d_stride, int64_t in_row_stride,
    int64_t out_d_stride, int64_t out_row_stride,
    int64_t grid_row_stride, int64_t /*grid_d_elems*/,
    bool align_corners, GridSamplePaddingMode padding, bool add_to)
{
    const T* type_tag = output;
    const bool zeros_pad = (padding == GridSamplePaddingMode::Zeros);

    for (int64_t od = 0; od < OD; ++od) {
        T* out_d = output + od * out_d_stride;

        for (int64_t oh = 0; oh < OH; ++oh) {
            const float* g_pos = grid_n + (od * OH + oh) * grid_row_stride;

            for (int64_t ow = 0; ow < OW; ++ow) {
                float px = gridsample_grid_to_pixel(g_pos[0], ID, align_corners);
                float py = gridsample_grid_to_pixel(g_pos[1], IH, align_corners);
                float pz = gridsample_grid_to_pixel(g_pos[2], IW, align_corners);
                g_pos += 3;

                float px_r = std::round(px);
                float py_r = std::round(py);
                float pz_r = std::round(pz);

                if (zeros_pad &&
                    (!gridsample_in_bounds(px_r, ID) || !gridsample_in_bounds(py_r, IH) || !gridsample_in_bounds(pz_r, IW))) {
                    if (!add_to) {
                        v_store(out_d + oh * out_row_stride + ow * 8, v_zero(type_tag));
                    }
                    continue;
                }

                px = gridsample_apply_padding(px, ID, padding);
                py = gridsample_apply_padding(py, IH, padding);
                pz = gridsample_apply_padding(pz, IW, padding);
                int64_t ix = static_cast<int64_t>(std::round(px));
                int64_t iy = static_cast<int64_t>(std::round(py));
                int64_t iz = static_cast<int64_t>(std::round(pz));

                auto val = v_load(&input[ix * in_d_stride + iy * in_row_stride + iz * 8]);
                v_store_add(out_d + oh * out_row_stride + ow * 8, val, add_to);
            }
        }
    }
}

}  // namespace nnops::kernel
