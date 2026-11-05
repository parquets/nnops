#pragma once
/// @file simd_resize.hpp
/// @brief CPU kernels for 2D/3D spatial resize, channel-first and channel-last.
///
/// Channel-first walks one channel plane and evaluates scalar output pixels; the
/// channel-last kernels put the SIMD lane on C, so a pixel's whole channel
/// vector interpolates with v_fmadd and only the C tail runs scalar.
///
/// The coordinate tables (source indices and the weight of the upper/right
/// neighbour) are built once per call by the driver; nearest encodes itself as
/// i1 == i0, w == 0.

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
// Channel-first: scalar over W inside one channel plane
// ============================================================

template <typename T>
void resize_channel_first_2d(
    T* out, const T* in,
    int64_t OH, int64_t OW,
    const int64_t* y0, const int64_t* y1, const float* wy,
    const int64_t* x0, const int64_t* x1, const float* wx,
    int64_t in_row_stride, int64_t out_row_stride,
    bool linear, bool add_to)
{
    for (int64_t oh = 0; oh < OH; ++oh) {
        const T* r0 = in + y0[oh] * in_row_stride;
        const T* r1 = in + y1[oh] * in_row_stride;
        T* orow = out + oh * out_row_stride;

        if (!linear) {
            for (int64_t ow = 0; ow < OW; ++ow) {
                s_store_add(&orow[ow], s_load(&r0[x0[ow]]), add_to);
            }
            continue;
        }

        const float w_y  = wy[oh];
        const float w_y0 = 1.0f - w_y;
        for (int64_t ow = 0; ow < OW; ++ow) {
            const float w_x  = wx[ow];
            const float w_x0 = 1.0f - w_x;
            const float v00 = s_load(&r0[x0[ow]]);
            const float v10 = s_load(&r0[x1[ow]]);
            const float v01 = s_load(&r1[x0[ow]]);
            const float v11 = s_load(&r1[x1[ow]]);
            const float top = w_x0 * v00 + w_x * v10;
            const float bot = w_x0 * v01 + w_x * v11;
            s_store_add(&orow[ow], w_y0 * top + w_y * bot, add_to);
        }
    }
}

template <typename T>
void resize_channel_first_3d(
    T* out, const T* in,
    int64_t OD, int64_t OH, int64_t OW,
    const int64_t* z0, const int64_t* z1, const float* wz,
    const int64_t* y0, const int64_t* y1, const float* wy,
    const int64_t* x0, const int64_t* x1, const float* wx,
    int64_t in_d_stride, int64_t in_row_stride,
    int64_t out_d_stride, int64_t out_row_stride,
    bool linear, bool add_to)
{
    for (int64_t od = 0; od < OD; ++od) {
        const T* d0 = in + z0[od] * in_d_stride;
        const T* d1 = in + z1[od] * in_d_stride;
        T* odst = out + od * out_d_stride;

        for (int64_t oh = 0; oh < OH; ++oh) {
            const T* r00 = d0 + y0[oh] * in_row_stride;
            const T* r01 = d0 + y1[oh] * in_row_stride;
            const T* r10 = d1 + y0[oh] * in_row_stride;
            const T* r11 = d1 + y1[oh] * in_row_stride;
            T* orow = odst + oh * out_row_stride;

            if (!linear) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    s_store_add(&orow[ow], s_load(&r00[x0[ow]]), add_to);
                }
                continue;
            }

            const float w_z  = wz[od];
            const float w_z0 = 1.0f - w_z;
            const float w_y  = wy[oh];
            const float w_y0 = 1.0f - w_y;

            for (int64_t ow = 0; ow < OW; ++ow) {
                const float w_x  = wx[ow];
                const float w_x0 = 1.0f - w_x;

                const float v000 = s_load(&r00[x0[ow]]);
                const float v100 = s_load(&r00[x1[ow]]);
                const float v010 = s_load(&r01[x0[ow]]);
                const float v110 = s_load(&r01[x1[ow]]);
                const float v001 = s_load(&r10[x0[ow]]);
                const float v101 = s_load(&r10[x1[ow]]);
                const float v011 = s_load(&r11[x0[ow]]);
                const float v111 = s_load(&r11[x1[ow]]);

                const float z0_top = w_x0 * v000 + w_x * v100;
                const float z0_bot = w_x0 * v010 + w_x * v110;
                const float z0_val = w_y0 * z0_top + w_y * z0_bot;

                const float z1_top = w_x0 * v001 + w_x * v101;
                const float z1_bot = w_x0 * v011 + w_x * v111;
                const float z1_val = w_y0 * z1_top + w_y * z1_bot;

                s_store_add(&orow[ow], w_z0 * z0_val + w_z * z1_val, add_to);
            }
        }
    }
}

// ============================================================
// Channel-last: the SIMD lane spans C
// ============================================================

/// Interpolate one output pixel: `dst[c] = w_y0*(w_x0*a0[c] + w_x*a1[c]) +
/// w_y*(w_x0*b0[c] + w_x*b1[c])`. With linear == false this is a whole-pixel
/// copy from a0.
template <typename T>
inline void resize_pixel_2d(T* dst, const T* a0, const T* a1,
                            const T* b0, const T* b1, int64_t C,
                            float w_x, float w_y, bool linear, bool add_to)
{
    constexpr int L = simd_lane_for<T>;
    if (!linear) {
        int64_t c = 0;
        for (; c + L <= C; c += L) {
            v_store_add(dst + c, v_load(a0 + c), add_to);
        }
        for (; c < C; ++c) {
            s_store_add(&dst[c], s_load(&a0[c]), add_to);
        }
        return;
    }

    const auto vw_x  = v_set1(dst, w_x);
    const auto vw_x0 = v_set1(dst, 1.0f - w_x);
    const auto vw_y  = v_set1(dst, w_y);
    const auto vw_y0 = v_set1(dst, 1.0f - w_y);

    int64_t c = 0;
    for (; c + L <= C; c += L) {
        const auto top = v_fmadd(vw_x0, v_load(a0 + c), v_mul(vw_x, v_load(a1 + c)));
        const auto bot = v_fmadd(vw_x0, v_load(b0 + c), v_mul(vw_x, v_load(b1 + c)));
        v_store_add(dst + c, v_fmadd(vw_y0, top, v_mul(vw_y, bot)), add_to);
    }
    const float w_x0 = 1.0f - w_x;
    const float w_y0 = 1.0f - w_y;
    for (; c < C; ++c) {
        const float top = w_x0 * s_load(&a0[c]) + w_x * s_load(&a1[c]);
        const float bot = w_x0 * s_load(&b0[c]) + w_x * s_load(&b1[c]);
        s_store_add(&dst[c], w_y0 * top + w_y * bot, add_to);
    }
}

template <typename T>
void resize_last_2d(
    T* out, const T* in, int64_t C, int64_t OH, int64_t OW,
    const int64_t* y0, const int64_t* y1, const float* wy,
    const int64_t* x0, const int64_t* x1, const float* wx,
    int64_t in_row_stride, int64_t out_row_stride,
    bool linear, bool add_to)
{
    for (int64_t oh = 0; oh < OH; ++oh) {
        const T* r0 = in + y0[oh] * in_row_stride;
        const T* r1 = in + y1[oh] * in_row_stride;
        T* orow = out + oh * out_row_stride;

        for (int64_t ow = 0; ow < OW; ++ow) {
            const T* a0 = r0 + x0[ow] * C;
            const T* a1 = r0 + x1[ow] * C;
            const T* b0 = r1 + x0[ow] * C;
            const T* b1 = r1 + x1[ow] * C;
            resize_pixel_2d<T>(orow + ow * C, a0, a1, b0, b1, C,
                               wx[ow], wy[oh], linear, add_to);
        }
    }
}

template <typename T>
void resize_last_3d(
    T* out, const T* in, int64_t C, int64_t OD, int64_t OH, int64_t OW,
    const int64_t* z0, const int64_t* z1, const float* wz,
    const int64_t* y0, const int64_t* y1, const float* wy,
    const int64_t* x0, const int64_t* x1, const float* wx,
    int64_t in_d_stride, int64_t in_row_stride,
    int64_t out_d_stride, int64_t out_row_stride,
    bool linear, bool add_to)
{
    constexpr int L = simd_lane_for<T>;

    for (int64_t od = 0; od < OD; ++od) {
        const T* d0 = in + z0[od] * in_d_stride;
        const T* d1 = in + z1[od] * in_d_stride;
        T* odst = out + od * out_d_stride;

        for (int64_t oh = 0; oh < OH; ++oh) {
            const T* plane00 = d0 + y0[oh] * in_row_stride;
            const T* plane01 = d0 + y1[oh] * in_row_stride;
            const T* plane10 = d1 + y0[oh] * in_row_stride;
            const T* plane11 = d1 + y1[oh] * in_row_stride;
            T* orow = odst + oh * out_row_stride;

            const float w_z  = wz[od];
            const float w_z0 = 1.0f - w_z;
            const float w_y  = wy[oh];
            const float w_y0 = 1.0f - w_y;

            for (int64_t ow = 0; ow < OW; ++ow) {
                const float w_x  = wx[ow];
                const float w_x0 = 1.0f - w_x;
                T* dst = orow + ow * C;

                const T* a000 = plane00 + x0[ow] * C;
                const T* a100 = plane00 + x1[ow] * C;
                const T* a010 = plane01 + x0[ow] * C;
                const T* a110 = plane01 + x1[ow] * C;
                const T* a001 = plane10 + x0[ow] * C;
                const T* a101 = plane10 + x1[ow] * C;
                const T* a011 = plane11 + x0[ow] * C;
                const T* a111 = plane11 + x1[ow] * C;

                if (!linear) {
                    resize_pixel_2d<T>(dst, a000, a000, a000, a000, C,
                                       0.0f, 0.0f, false, add_to);
                    continue;
                }

                const auto vw_x  = v_set1(dst, w_x);
                const auto vw_x0 = v_set1(dst, 1.0f - w_x);
                const auto vw_y  = v_set1(dst, w_y);
                const auto vw_y0 = v_set1(dst, 1.0f - w_y);
                const auto vw_z  = v_set1(dst, w_z);
                const auto vw_z0 = v_set1(dst, w_z0);

                int64_t c = 0;
                for (; c + L <= C; c += L) {
                    const auto z0_top = v_fmadd(vw_x0, v_load(a000 + c),
                                                v_mul(vw_x, v_load(a100 + c)));
                    const auto z0_bot = v_fmadd(vw_x0, v_load(a010 + c),
                                                v_mul(vw_x, v_load(a110 + c)));
                    const auto z0_val = v_fmadd(vw_y0, z0_top, v_mul(vw_y, z0_bot));

                    const auto z1_top = v_fmadd(vw_x0, v_load(a001 + c),
                                                v_mul(vw_x, v_load(a101 + c)));
                    const auto z1_bot = v_fmadd(vw_x0, v_load(a011 + c),
                                                v_mul(vw_x, v_load(a111 + c)));
                    const auto z1_val = v_fmadd(vw_y0, z1_top, v_mul(vw_y, z1_bot));

                    v_store_add(dst + c, v_fmadd(vw_z0, z0_val, v_mul(vw_z, z1_val)),
                                add_to);
                }
                for (; c < C; ++c) {
                    const float z0_top = w_x0 * s_load(&a000[c]) + w_x * s_load(&a100[c]);
                    const float z0_bot = w_x0 * s_load(&a010[c]) + w_x * s_load(&a110[c]);
                    const float z0_val = w_y0 * z0_top + w_y * z0_bot;

                    const float z1_top = w_x0 * s_load(&a001[c]) + w_x * s_load(&a101[c]);
                    const float z1_bot = w_x0 * s_load(&a011[c]) + w_x * s_load(&a111[c]);
                    const float z1_val = w_y0 * z1_top + w_y * z1_bot;

                    s_store_add(&dst[c], w_z0 * z0_val + w_z * z1_val, add_to);
                }
            }
        }
    }
}

}  // namespace nnops::kernel
