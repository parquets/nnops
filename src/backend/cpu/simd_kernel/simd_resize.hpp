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
/// i1 == i0, w == 0. Cubic gets four taps per axis instead, built from the Keys
/// kernel and clamped at the image edge; it is 2D only.

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

/// Keys cubic kernel at |x|: the parameterisation ONNX (cubic_coeff_a),
/// PyTorch and PIL all share.
inline float resize_cubic_weight(float x, float a) {
    const float ax = std::fabs(x);
    if (ax <= 1.0f) {
        return ((a + 2.0f) * ax - (a + 3.0f)) * ax * ax + 1.0f;
    }
    if (ax < 2.0f) {
        return ((a * ax - 5.0f * a) * ax + 8.0f * a) * ax - 4.0f * a;
    }
    return 0.0f;
}

/// Four taps base-1 .. base+2 around the source coordinate. The base index is
/// the raw floor, not clamped, so a source coordinate outside the image keeps
/// its fractional part and only the tap indices clamp; the weight of an
/// outside tap folds into the border pixel.
struct ResizeCubicTap {
    int64_t idx[4];
    float   w[4];
};

inline ResizeCubicTap resize_make_cubic_tap(float src, int64_t size, float a) {
    const int64_t base = static_cast<int64_t>(std::floor(src));
    const float   frac = src - static_cast<float>(base);

    ResizeCubicTap tap{};
    for (int j = 0; j < 4; ++j) {
        tap.idx[j] = resize_clamp_idx(base + j - 1, size);
    }
    tap.w[0] = resize_cubic_weight(frac + 1.0f, a);
    tap.w[1] = resize_cubic_weight(frac, a);
    tap.w[2] = resize_cubic_weight(1.0f - frac, a);
    tap.w[3] = resize_cubic_weight(2.0f - frac, a);
    return tap;
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

template <typename T>
void resize_channel_first_2d_cubic(
    T* out, const T* in,
    int64_t OH, int64_t OW,
    const ResizeCubicTap* yt, const ResizeCubicTap* xt,
    int64_t in_row_stride, int64_t out_row_stride,
    bool add_to)
{
    for (int64_t oh = 0; oh < OH; ++oh) {
        const T* rows[4];
        for (int j = 0; j < 4; ++j) {
            rows[j] = in + yt[oh].idx[j] * in_row_stride;
        }
        T* orow = out + oh * out_row_stride;

        for (int64_t ow = 0; ow < OW; ++ow) {
            const ResizeCubicTap& x = xt[ow];
            float v = 0.0f;
            for (int j = 0; j < 4; ++j) {
                const T* r = rows[j];
                float h = x.w[0] * s_load(&r[x.idx[0]]);
                h += x.w[1] * s_load(&r[x.idx[1]]);
                h += x.w[2] * s_load(&r[x.idx[2]]);
                h += x.w[3] * s_load(&r[x.idx[3]]);
                v += yt[oh].w[j] * h;
            }
            s_store_add(&orow[ow], v, add_to);
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

/// Bicubic for one output pixel: each of the four rows is the four W taps
/// combined, then the four row values blend with the H weights.
template <typename T>
inline void resize_pixel_2d_cubic(T* dst, const T* const* rows,
                                  const ResizeCubicTap& xt, const float* wy,
                                  int64_t C, bool add_to)
{
    constexpr int L = simd_lane_for<T>;
    const auto vx0 = v_set1(dst, xt.w[0]);
    const auto vx1 = v_set1(dst, xt.w[1]);
    const auto vx2 = v_set1(dst, xt.w[2]);
    const auto vx3 = v_set1(dst, xt.w[3]);
    const auto vy0 = v_set1(dst, wy[0]);
    const auto vy1 = v_set1(dst, wy[1]);
    const auto vy2 = v_set1(dst, wy[2]);
    const auto vy3 = v_set1(dst, wy[3]);

    int64_t c = 0;
    for (; c + L <= C; c += L) {
        auto acc = v_mul(vy0, v_fmadd(vx3, v_load(rows[0] + xt.idx[3] * C + c),
                        v_fmadd(vx2, v_load(rows[0] + xt.idx[2] * C + c),
                        v_fmadd(vx1, v_load(rows[0] + xt.idx[1] * C + c),
                                v_mul(vx0, v_load(rows[0] + xt.idx[0] * C + c))))));
        auto h1 = v_fmadd(vx3, v_load(rows[1] + xt.idx[3] * C + c),
                  v_fmadd(vx2, v_load(rows[1] + xt.idx[2] * C + c),
                  v_fmadd(vx1, v_load(rows[1] + xt.idx[1] * C + c),
                          v_mul(vx0, v_load(rows[1] + xt.idx[0] * C + c)))));
        auto h2 = v_fmadd(vx3, v_load(rows[2] + xt.idx[3] * C + c),
                  v_fmadd(vx2, v_load(rows[2] + xt.idx[2] * C + c),
                  v_fmadd(vx1, v_load(rows[2] + xt.idx[1] * C + c),
                          v_mul(vx0, v_load(rows[2] + xt.idx[0] * C + c)))));
        auto h3 = v_fmadd(vx3, v_load(rows[3] + xt.idx[3] * C + c),
                  v_fmadd(vx2, v_load(rows[3] + xt.idx[2] * C + c),
                  v_fmadd(vx1, v_load(rows[3] + xt.idx[1] * C + c),
                          v_mul(vx0, v_load(rows[3] + xt.idx[0] * C + c)))));
        acc = v_fmadd(vy1, h1, acc);
        acc = v_fmadd(vy2, h2, acc);
        acc = v_fmadd(vy3, h3, acc);
        v_store_add(dst + c, acc, add_to);
    }
    for (; c < C; ++c) {
        float v = 0.0f;
        for (int j = 0; j < 4; ++j) {
            const T* r = rows[j];
            float h = xt.w[0] * s_load(&r[xt.idx[0] * C + c]);
            h += xt.w[1] * s_load(&r[xt.idx[1] * C + c]);
            h += xt.w[2] * s_load(&r[xt.idx[2] * C + c]);
            h += xt.w[3] * s_load(&r[xt.idx[3] * C + c]);
            v += wy[j] * h;
        }
        s_store_add(&dst[c], v, add_to);
    }
}

template <typename T>
void resize_last_2d_cubic(
    T* out, const T* in, int64_t C, int64_t OH, int64_t OW,
    const ResizeCubicTap* yt, const ResizeCubicTap* xt,
    int64_t in_row_stride, int64_t out_row_stride,
    bool add_to)
{
    for (int64_t oh = 0; oh < OH; ++oh) {
        const T* rows[4];
        for (int j = 0; j < 4; ++j) {
            rows[j] = in + yt[oh].idx[j] * in_row_stride;
        }
        T* orow = out + oh * out_row_stride;

        for (int64_t ow = 0; ow < OW; ++ow) {
            resize_pixel_2d_cubic<T>(orow + ow * C, rows, xt[ow], yt[oh].w,
                                     C, add_to);
        }
    }
}

}  // namespace nnops::kernel
