#pragma once
/// @file simd_grid_sample.hpp
/// @brief CPU kernels for 2D/3D grid sampling, channel-first and channel-last.
///
/// The grid itself is always dense f32 channel-first ([N, (OD,) OH, OW, 2|3]).
/// Channel-first samples one channel plane at a time, scalar; channel-last computes
/// the source position once per output pixel and runs the interpolation across
/// the whole C vector with SIMD. Zeros padding is all-or-nothing per pixel, so
/// the channel vector is either interpolated or zeroed.
///
/// Same coordinate/padding semantics as ONNX GridSample: normalized grid in
/// [-1, 1], align_corners switches between the pixel-edge and pixel-center
/// mapping, and border/reflection wrap the source coefficient before sampling.

#include "nnops/detail/simd/simd.hpp"
#include "nnops/ops/grid_sample.hpp"

#include <algorithm>
#include <cmath>

namespace nnops::kernel {

using namespace simd;

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
// Shared pixel helpers (channel-last)
// ============================================================

/// Copy one pixel's C channels, or leave `add_to` destinations untouched.
template <typename T>
inline void gridsample_copy_pixel(T* dst, const T* src, int64_t C, bool add_to) {
    constexpr int L = simd_lane_for<T>;
    int64_t c = 0;
    for (; c + L <= C; c += L) {
        v_store_add(dst + c, v_load(src + c), add_to);
    }
    for (; c < C; ++c) {
        s_store_add(&dst[c], s_load(&src[c]), add_to);
    }
}

/// Zero one pixel's C channels (zeros padding); add_to leaves it alone.
template <typename T>
inline void gridsample_zero_pixel(T* dst, int64_t C, bool add_to) {
    if (add_to) { return; }
    constexpr int L = simd_lane_for<T>;
    int64_t c = 0;
    for (; c + L <= C; c += L) {
        v_store(dst + c, v_zero(dst));
    }
    for (; c < C; ++c) {
        s_store(&dst[c], 0.0f);
    }
}

// ============================================================
// Channel-first: scalar within one channel plane
// ============================================================

template <typename T>
void gridsample_channel_first_2d(
    T* output, const T* input, const float* grid_n,
    int64_t IH, int64_t IW, int64_t OH, int64_t OW,
    int64_t in_row_stride, int64_t out_row_stride,
    int64_t grid_row_stride,
    bool align_corners, GridSamplePaddingMode padding, bool add_to)
{
    const bool zeros_pad = (padding == GridSamplePaddingMode::Zeros);

    for (int64_t oh = 0; oh < OH; ++oh) {
        const float* g_row = grid_n + oh * grid_row_stride;

        for (int64_t ow = 0; ow < OW; ++ow) {
            const float* g_pos = g_row + ow * 2;  // 2D: (y, x)
            float py = gridsample_grid_to_pixel(g_pos[0], IH, align_corners);
            float pz = gridsample_grid_to_pixel(g_pos[1], IW, align_corners);

            T* dst = output + ow;
            if (zeros_pad && !gridsample_bilinear_bounds_ok(py, pz, IH, IW)) {
                if (!add_to) { s_store(dst, 0.0f); }
                continue;
            }

            py = gridsample_apply_padding(py, IH, padding);
            pz = gridsample_apply_padding(pz, IW, padding);

            const int64_t y0 = static_cast<int64_t>(std::floor(py));
            const int64_t z0 = static_cast<int64_t>(std::floor(pz));
            const int64_t y1 = gridsample_clamp_idx(y0 + 1, IH);
            const int64_t z1 = gridsample_clamp_idx(z0 + 1, IW);

            const float wy  = py - static_cast<float>(y0);
            const float wz  = pz - static_cast<float>(z0);
            const float wy0 = 1.0f - wy;
            const float wz0 = 1.0f - wz;

            const float v00 = s_load(&input[y0 * in_row_stride + z0]);
            const float v01 = s_load(&input[y0 * in_row_stride + z1]);
            const float v10 = s_load(&input[y1 * in_row_stride + z0]);
            const float v11 = s_load(&input[y1 * in_row_stride + z1]);

            const float row0 = wz0 * v00 + wz * v01;
            const float row1 = wz0 * v10 + wz * v11;
            s_store_add(dst, wy0 * row0 + wy * row1, add_to);
        }
        output += out_row_stride;
    }
}

template <typename T>
void gridsample_nearest_channel_first_2d(
    T* output, const T* input, const float* grid_n,
    int64_t IH, int64_t IW, int64_t OH, int64_t OW,
    int64_t in_row_stride, int64_t out_row_stride,
    int64_t grid_row_stride,
    bool align_corners, GridSamplePaddingMode padding, bool add_to)
{
    const bool zeros_pad = (padding == GridSamplePaddingMode::Zeros);

    for (int64_t oh = 0; oh < OH; ++oh) {
        const float* g_row = grid_n + oh * grid_row_stride;

        for (int64_t ow = 0; ow < OW; ++ow) {
            const float* g_pos = g_row + ow * 2;
            float py = gridsample_grid_to_pixel(g_pos[0], IH, align_corners);
            float pz = gridsample_grid_to_pixel(g_pos[1], IW, align_corners);
            const float py_r = std::round(py);
            const float pz_r = std::round(pz);

            T* dst = output + ow;
            if (zeros_pad &&
                (!gridsample_in_bounds(py_r, IH) || !gridsample_in_bounds(pz_r, IW))) {
                if (!add_to) { s_store(dst, 0.0f); }
                continue;
            }

            py = gridsample_apply_padding(py, IH, padding);
            pz = gridsample_apply_padding(pz, IW, padding);
            const int64_t iy = static_cast<int64_t>(std::round(py));
            const int64_t iz = static_cast<int64_t>(std::round(pz));

            s_store_add(dst, s_load(&input[iy * in_row_stride + iz]), add_to);
        }
        output += out_row_stride;
    }
}

template <typename T>
void gridsample_channel_first_3d(
    T* output, const T* input, const float* grid_n,
    int64_t ID, int64_t IH, int64_t IW,
    int64_t OD, int64_t OH, int64_t OW,
    int64_t in_d_stride, int64_t in_row_stride,
    int64_t out_d_stride, int64_t out_row_stride,
    int64_t grid_row_stride,
    bool align_corners, GridSamplePaddingMode padding, bool add_to)
{
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

                T* dst = out_d + oh * out_row_stride + ow;
                if (zeros_pad && !gridsample_trilinear_bounds_ok(px, py, pz, ID, IH, IW)) {
                    if (!add_to) { s_store(dst, 0.0f); }
                    continue;
                }

                px = gridsample_apply_padding(px, ID, padding);
                py = gridsample_apply_padding(py, IH, padding);
                pz = gridsample_apply_padding(pz, IW, padding);

                const int64_t x0 = static_cast<int64_t>(std::floor(px));
                const int64_t y0 = static_cast<int64_t>(std::floor(py));
                const int64_t z0 = static_cast<int64_t>(std::floor(pz));
                const int64_t x1 = gridsample_clamp_idx(x0 + 1, ID);
                const int64_t y1 = gridsample_clamp_idx(y0 + 1, IH);
                const int64_t z1 = gridsample_clamp_idx(z0 + 1, IW);

                const float wx = px - static_cast<float>(x0);
                const float wy = py - static_cast<float>(y0);
                const float wz = pz - static_cast<float>(z0);
                const float wx0 = 1.0f - wx;
                const float wy0 = 1.0f - wy;
                const float wz0 = 1.0f - wz;

                const T* p00 = input + x0 * in_d_stride + y0 * in_row_stride;
                const T* p01 = input + x0 * in_d_stride + y1 * in_row_stride;
                const T* p10 = input + x1 * in_d_stride + y0 * in_row_stride;
                const T* p11 = input + x1 * in_d_stride + y1 * in_row_stride;

                const float r00 = wz0 * s_load(&p00[z0]) + wz * s_load(&p00[z1]);
                const float r01 = wz0 * s_load(&p01[z0]) + wz * s_load(&p01[z1]);
                const float r10 = wz0 * s_load(&p10[z0]) + wz * s_load(&p10[z1]);
                const float r11 = wz0 * s_load(&p11[z0]) + wz * s_load(&p11[z1]);
                const float s0 = wy0 * r00 + wy * r01;
                const float s1 = wy0 * r10 + wy * r11;
                s_store_add(dst, wx0 * s0 + wx * s1, add_to);
            }
        }
    }
}

template <typename T>
void gridsample_nearest_channel_first_3d(
    T* output, const T* input, const float* grid_n,
    int64_t ID, int64_t IH, int64_t IW,
    int64_t OD, int64_t OH, int64_t OW,
    int64_t in_d_stride, int64_t in_row_stride,
    int64_t out_d_stride, int64_t out_row_stride,
    int64_t grid_row_stride,
    bool align_corners, GridSamplePaddingMode padding, bool add_to)
{
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

                const float px_r = std::round(px);
                const float py_r = std::round(py);
                const float pz_r = std::round(pz);

                T* dst = out_d + oh * out_row_stride + ow;
                if (zeros_pad &&
                    (!gridsample_in_bounds(px_r, ID) || !gridsample_in_bounds(py_r, IH) ||
                     !gridsample_in_bounds(pz_r, IW))) {
                    if (!add_to) { s_store(dst, 0.0f); }
                    continue;
                }

                px = gridsample_apply_padding(px, ID, padding);
                py = gridsample_apply_padding(py, IH, padding);
                pz = gridsample_apply_padding(pz, IW, padding);
                const int64_t ix = static_cast<int64_t>(std::round(px));
                const int64_t iy = static_cast<int64_t>(std::round(py));
                const int64_t iz = static_cast<int64_t>(std::round(pz));

                s_store_add(dst, s_load(&input[ix * in_d_stride + iy * in_row_stride + iz]),
                            add_to);
            }
        }
    }
}

// ============================================================
// Channel-last: the SIMD lane spans C
// ============================================================

template <typename T>
void gridsample_last_2d(
    T* output, const T* input, const float* grid_n, int64_t C,
    int64_t IH, int64_t IW, int64_t OH, int64_t OW,
    int64_t in_row_stride, int64_t out_row_stride,
    int64_t grid_row_stride,
    bool align_corners, GridSamplePaddingMode padding, bool add_to)
{
    constexpr int L = simd_lane_for<T>;
    const bool zeros_pad = (padding == GridSamplePaddingMode::Zeros);

    for (int64_t oh = 0; oh < OH; ++oh) {
        const float* g_row = grid_n + oh * grid_row_stride;
        T* orow = output + oh * out_row_stride;

        for (int64_t ow = 0; ow < OW; ++ow) {
            const float* g_pos = g_row + ow * 2;
            float py = gridsample_grid_to_pixel(g_pos[0], IH, align_corners);
            float pz = gridsample_grid_to_pixel(g_pos[1], IW, align_corners);
            T* dst = orow + ow * C;

            if (zeros_pad && !gridsample_bilinear_bounds_ok(py, pz, IH, IW)) {
                gridsample_zero_pixel<T>(dst, C, add_to);
                continue;
            }

            py = gridsample_apply_padding(py, IH, padding);
            pz = gridsample_apply_padding(pz, IW, padding);

            const int64_t y0 = static_cast<int64_t>(std::floor(py));
            const int64_t z0 = static_cast<int64_t>(std::floor(pz));
            const int64_t y1 = gridsample_clamp_idx(y0 + 1, IH);
            const int64_t z1 = gridsample_clamp_idx(z0 + 1, IW);

            const float wy  = py - static_cast<float>(y0);
            const float wz  = pz - static_cast<float>(z0);
            const float wy0 = 1.0f - wy;
            const float wz0 = 1.0f - wz;

            const T* p00 = input + y0 * in_row_stride + z0 * C;
            const T* p01 = input + y0 * in_row_stride + z1 * C;
            const T* p10 = input + y1 * in_row_stride + z0 * C;
            const T* p11 = input + y1 * in_row_stride + z1 * C;

            const auto vwz  = v_set1(dst, wz);
            const auto vwz0 = v_set1(dst, wz0);
            const auto vwy  = v_set1(dst, wy);
            const auto vwy0 = v_set1(dst, wy0);

            int64_t c = 0;
            for (; c + L <= C; c += L) {
                const auto row0 = v_fmadd(vwz0, v_load(p00 + c), v_mul(vwz, v_load(p01 + c)));
                const auto row1 = v_fmadd(vwz0, v_load(p10 + c), v_mul(vwz, v_load(p11 + c)));
                v_store_add(dst + c, v_fmadd(vwy0, row0, v_mul(vwy, row1)), add_to);
            }
            for (; c < C; ++c) {
                const float row0 = wz0 * s_load(&p00[c]) + wz * s_load(&p01[c]);
                const float row1 = wz0 * s_load(&p10[c]) + wz * s_load(&p11[c]);
                s_store_add(&dst[c], wy0 * row0 + wy * row1, add_to);
            }
        }
    }
}

template <typename T>
void gridsample_nearest_last_2d(
    T* output, const T* input, const float* grid_n, int64_t C,
    int64_t IH, int64_t IW, int64_t OH, int64_t OW,
    int64_t in_row_stride, int64_t out_row_stride,
    int64_t grid_row_stride,
    bool align_corners, GridSamplePaddingMode padding, bool add_to)
{
    const bool zeros_pad = (padding == GridSamplePaddingMode::Zeros);

    for (int64_t oh = 0; oh < OH; ++oh) {
        const float* g_row = grid_n + oh * grid_row_stride;
        T* orow = output + oh * out_row_stride;

        for (int64_t ow = 0; ow < OW; ++ow) {
            const float* g_pos = g_row + ow * 2;
            float py = gridsample_grid_to_pixel(g_pos[0], IH, align_corners);
            float pz = gridsample_grid_to_pixel(g_pos[1], IW, align_corners);
            const float py_r = std::round(py);
            const float pz_r = std::round(pz);
            T* dst = orow + ow * C;

            if (zeros_pad &&
                (!gridsample_in_bounds(py_r, IH) || !gridsample_in_bounds(pz_r, IW))) {
                gridsample_zero_pixel<T>(dst, C, add_to);
                continue;
            }

            py = gridsample_apply_padding(py, IH, padding);
            pz = gridsample_apply_padding(pz, IW, padding);
            const int64_t iy = static_cast<int64_t>(std::round(py));
            const int64_t iz = static_cast<int64_t>(std::round(pz));

            gridsample_copy_pixel<T>(dst, input + iy * in_row_stride + iz * C, C, add_to);
        }
    }
}

template <typename T>
void gridsample_last_3d(
    T* output, const T* input, const float* grid_n, int64_t C,
    int64_t ID, int64_t IH, int64_t IW,
    int64_t OD, int64_t OH, int64_t OW,
    int64_t in_d_stride, int64_t in_row_stride,
    int64_t out_d_stride, int64_t out_row_stride,
    int64_t grid_row_stride,
    bool align_corners, GridSamplePaddingMode padding, bool add_to)
{
    constexpr int L = simd_lane_for<T>;
    const bool zeros_pad = (padding == GridSamplePaddingMode::Zeros);

    for (int64_t od = 0; od < OD; ++od) {
        T* out_d = output + od * out_d_stride;

        for (int64_t oh = 0; oh < OH; ++oh) {
            const float* g_pos = grid_n + (od * OH + oh) * grid_row_stride;
            T* orow = out_d + oh * out_row_stride;

            for (int64_t ow = 0; ow < OW; ++ow) {
                float px = gridsample_grid_to_pixel(g_pos[0], ID, align_corners);
                float py = gridsample_grid_to_pixel(g_pos[1], IH, align_corners);
                float pz = gridsample_grid_to_pixel(g_pos[2], IW, align_corners);
                g_pos += 3;
                T* dst = orow + ow * C;

                if (zeros_pad &&
                    !gridsample_trilinear_bounds_ok(px, py, pz, ID, IH, IW)) {
                    gridsample_zero_pixel<T>(dst, C, add_to);
                    continue;
                }

                px = gridsample_apply_padding(px, ID, padding);
                py = gridsample_apply_padding(py, IH, padding);
                pz = gridsample_apply_padding(pz, IW, padding);

                const int64_t x0 = static_cast<int64_t>(std::floor(px));
                const int64_t y0 = static_cast<int64_t>(std::floor(py));
                const int64_t z0 = static_cast<int64_t>(std::floor(pz));
                const int64_t x1 = gridsample_clamp_idx(x0 + 1, ID);
                const int64_t y1 = gridsample_clamp_idx(y0 + 1, IH);
                const int64_t z1 = gridsample_clamp_idx(z0 + 1, IW);

                const float wx  = px - static_cast<float>(x0);
                const float wy  = py - static_cast<float>(y0);
                const float wz  = pz - static_cast<float>(z0);
                const float wx0 = 1.0f - wx;
                const float wy0 = 1.0f - wy;
                const float wz0 = 1.0f - wz;

                const T* base00 = input + x0 * in_d_stride + y0 * in_row_stride;
                const T* base01 = input + x0 * in_d_stride + y1 * in_row_stride;
                const T* base10 = input + x1 * in_d_stride + y0 * in_row_stride;
                const T* base11 = input + x1 * in_d_stride + y1 * in_row_stride;

                const auto vwx  = v_set1(dst, wx);
                const auto vwx0 = v_set1(dst, wx0);
                const auto vwy  = v_set1(dst, wy);
                const auto vwy0 = v_set1(dst, wy0);
                const auto vwz  = v_set1(dst, wz);
                const auto vwz0 = v_set1(dst, wz0);

                int64_t c = 0;
                for (; c + L <= C; c += L) {
                    const auto r00 = v_fmadd(vwz0, v_load(base00 + z0 * C + c),
                                             v_mul(vwz, v_load(base00 + z1 * C + c)));
                    const auto r01 = v_fmadd(vwz0, v_load(base01 + z0 * C + c),
                                             v_mul(vwz, v_load(base01 + z1 * C + c)));
                    const auto r10 = v_fmadd(vwz0, v_load(base10 + z0 * C + c),
                                             v_mul(vwz, v_load(base10 + z1 * C + c)));
                    const auto r11 = v_fmadd(vwz0, v_load(base11 + z0 * C + c),
                                             v_mul(vwz, v_load(base11 + z1 * C + c)));
                    const auto s0 = v_fmadd(vwy0, r00, v_mul(vwy, r01));
                    const auto s1 = v_fmadd(vwy0, r10, v_mul(vwy, r11));
                    v_store_add(dst + c, v_fmadd(vwx0, s0, v_mul(vwx, s1)), add_to);
                }
                for (; c < C; ++c) {
                    const float r00 = wz0 * s_load(&base00[z0 * C + c]) +
                                      wz * s_load(&base00[z1 * C + c]);
                    const float r01 = wz0 * s_load(&base01[z0 * C + c]) +
                                      wz * s_load(&base01[z1 * C + c]);
                    const float r10 = wz0 * s_load(&base10[z0 * C + c]) +
                                      wz * s_load(&base10[z1 * C + c]);
                    const float r11 = wz0 * s_load(&base11[z0 * C + c]) +
                                      wz * s_load(&base11[z1 * C + c]);
                    const float s0 = wy0 * r00 + wy * r01;
                    const float s1 = wy0 * r10 + wy * r11;
                    s_store_add(&dst[c], wx0 * s0 + wx * s1, add_to);
                }
            }
        }
    }
}

template <typename T>
void gridsample_nearest_last_3d(
    T* output, const T* input, const float* grid_n, int64_t C,
    int64_t ID, int64_t IH, int64_t IW,
    int64_t OD, int64_t OH, int64_t OW,
    int64_t in_d_stride, int64_t in_row_stride,
    int64_t out_d_stride, int64_t out_row_stride,
    int64_t grid_row_stride,
    bool align_corners, GridSamplePaddingMode padding, bool add_to)
{
    const bool zeros_pad = (padding == GridSamplePaddingMode::Zeros);

    for (int64_t od = 0; od < OD; ++od) {
        T* out_d = output + od * out_d_stride;

        for (int64_t oh = 0; oh < OH; ++oh) {
            const float* g_pos = grid_n + (od * OH + oh) * grid_row_stride;
            T* orow = out_d + oh * out_row_stride;

            for (int64_t ow = 0; ow < OW; ++ow) {
                float px = gridsample_grid_to_pixel(g_pos[0], ID, align_corners);
                float py = gridsample_grid_to_pixel(g_pos[1], IH, align_corners);
                float pz = gridsample_grid_to_pixel(g_pos[2], IW, align_corners);
                g_pos += 3;
                T* dst = orow + ow * C;

                const float px_r = std::round(px);
                const float py_r = std::round(py);
                const float pz_r = std::round(pz);

                if (zeros_pad &&
                    (!gridsample_in_bounds(px_r, ID) || !gridsample_in_bounds(py_r, IH) ||
                     !gridsample_in_bounds(pz_r, IW))) {
                    gridsample_zero_pixel<T>(dst, C, add_to);
                    continue;
                }

                px = gridsample_apply_padding(px, ID, padding);
                py = gridsample_apply_padding(py, IH, padding);
                pz = gridsample_apply_padding(pz, IW, padding);
                const int64_t ix = static_cast<int64_t>(std::round(px));
                const int64_t iy = static_cast<int64_t>(std::round(py));
                const int64_t iz = static_cast<int64_t>(std::round(pz));

                gridsample_copy_pixel<T>(
                    dst, input + ix * in_d_stride + iy * in_row_stride + iz * C,
                    C, add_to);
            }
        }
    }
}

}  // namespace nnops::kernel
