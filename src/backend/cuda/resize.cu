/// @file resize.cu
/// @brief CUDA Resize — nearest / linear / bicubic, channel-first and
///        channel-last.
///
/// The CPU route builds per-axis coordinate tables once per call and walks them.
/// A launch cannot: the tables live in device memory behind their own allocation
/// and their own kernel, and on the plane counts a resize actually covers that
/// costs more than the arithmetic it saves. So the same four expressions are
/// evaluated per thread and the taps stay in registers. They are the CPU's
/// expressions down to the rounding — the multiply and the subtract that place a
/// source coordinate are kept apart (`__fmul_rn`/`__fsub_rn`) rather than fused,
/// so a half-pixel mapping lands on the same side of a .5 boundary as the CPU's,
/// and nearest rounds with `roundf`, which is the half-away-from-zero the CPU
/// uses.
///
/// Channel-first keeps the lane on output width, the contiguous axis there. Each
/// thread owns VW consecutive columns, so the index math around the tap — the
/// plane and row decode, the four strides, the bounds test — is paid once per VW
/// outputs instead of once per output. A row only has ow/VW lanes, which for the
/// 32- and 64-column planes of a decoder is a fraction of a block, so the block's
/// thread.y axis takes consecutive rows and the block coordinates carry the row
/// block and the plane (the depth plane too for a rank-5 input). The window
/// widens to eight columns while neighbouring output columns still share their
/// source taps — always going up, and always for a nearest pass, which reads one
/// tap a column; a bilinear downsample puts each column on its own pair, and
/// there the window stops at four so the block keeps its lanes.
///
/// Channel-last puts the lane on the channel: one thread owns one 16 B vector of
/// one output pixel (four f32 or eight f16 taps through a single 128-bit load),
/// adjacent threads take adjacent channel vectors, and the block's thread.y axis
/// takes consecutive pixels, so a warp reads and writes one contiguous run. A
/// channel count that is not a vector multiple, or a tensor whose base is not
/// 16 B aligned, falls to a scalar-per-pixel script; so does a channel count too
/// narrow to fill a warp's worth of vectors.
///
/// Every kernel walks the row and plane axes with a grid stride, so no dimension
/// of the work has to fit under 65535. Cubic is the 2D bicubic only, as on the
/// CPU and in PyTorch.
///
/// f32, f16 and bf16. Workspace: not used.

#include "cuda_common.cuh"
#include "nnops/ops/resize.hpp"

#include <algorithm>

namespace nnops::backend::cuda {
namespace {

// ============================================================
// Geometry
// ============================================================

/// One resize call's extents, effective (cropped) input sizes, element strides
/// and mapping mode. Scalars only, passed by value.
struct ResizeGeom {
    int64_t ow, oh, od;
    int64_t iw, ih, id;   // effective input extents, after the crop
    int64_t n, c;
    int64_t crop_off;     // crop origin, in elements

    int64_t in_n, in_c, in_d, in_row, in_pix;
    int64_t out_n, out_c, out_d, out_row, out_pix;

    int coord_mode;       // 0 half pixel, 1 align corners, 2 asymmetric
    float cubic_a;
    bool add_to;
};

// ============================================================
// Coordinate mapping — the CPU's, element for element
// ============================================================

/// mode: 0 half pixel, 1 align corners, 2 asymmetric.
__device__ inline float rsz_src(int64_t o, int64_t in_size, int64_t out_size, int mode)
{
    if (in_size == out_size) { return float(o); }
    if (mode == 1) {
        return out_size <= 1 ? 0.0f
                             : float(o) * float(in_size - 1) / float(out_size - 1);
    }
    const float scale = float(in_size) / float(out_size);
    return mode == 0 ? __fsub_rn(__fmul_rn(float(o) + 0.5f, scale), 0.5f)
                     : __fmul_rn(float(o), scale);
}

__device__ inline int64_t rsz_clamp(int64_t i, int64_t size)
{
    return i < 0 ? 0 : (i >= size ? size - 1 : i);
}

__device__ inline int64_t rsz_nearest(int64_t o, int64_t in_size, int64_t out_size,
                                      int mode)
{
    return rsz_clamp(int64_t(roundf(rsz_src(o, in_size, out_size, mode))), in_size);
}

/// The two neighbours and the weight of the upper/right one. Clamping the near
/// index *before* the step is what folds an outside tap's weight into the border
/// pixel; doing it the other way round moves the edge pixel.
struct LinearTap {
    int64_t i0, i1;
    float w;
};

__device__ inline LinearTap rsz_linear_tap(int64_t o, int64_t in_size, int64_t out_size,
                                           int mode)
{
    const float src = rsz_src(o, in_size, out_size, mode);
    const float floored = floorf(src);
    LinearTap t;
    t.i0 = rsz_clamp(int64_t(floored), in_size);
    t.i1 = rsz_clamp(t.i0 + 1, in_size);
    t.w = src - floored;
    return t;
}

/// Keys kernel at |x|: the parameterisation ONNX (cubic_coeff_a), PyTorch and
/// PIL share.
__device__ inline float rsz_cubic_weight(float x, float a)
{
    const float ax = fabsf(x);
    if (ax <= 1.0f) { return ((a + 2.0f) * ax - (a + 3.0f)) * ax * ax + 1.0f; }
    if (ax < 2.0f) { return ((a * ax - 5.0f * a) * ax + 8.0f * a) * ax - 4.0f * a; }
    return 0.0f;
}

struct CubicTap {
    int64_t idx[4];
    float w[4];
};

__device__ inline CubicTap rsz_cubic_tap(int64_t o, int64_t in_size, int64_t out_size,
                                         int mode, float a)
{
    const float src = rsz_src(o, in_size, out_size, mode);
    const int64_t base = int64_t(floorf(src));
    const float frac = src - float(base);
    CubicTap t;
    for (int j = 0; j < 4; ++j) { t.idx[j] = rsz_clamp(base + j - 1, in_size); }
    t.w[0] = rsz_cubic_weight(frac + 1.0f, a);
    t.w[1] = rsz_cubic_weight(frac, a);
    t.w[2] = rsz_cubic_weight(1.0f - frac, a);
    t.w[3] = rsz_cubic_weight(2.0f - frac, a);
    return t;
}

template <class T>
__device__ inline void rsz_store(T* dst, float v, bool add_to)
{
    if (add_to) { v += s_load(dst); }
    s_store(dst, v);
}

// ============================================================
// Channel-first — lanes on output width
// ============================================================

/// One output run of VW columns per thread, threadIdx.y carrying consecutive
/// output rows so a narrow plane still fills its block. The row block and the
/// (n, c) plane walk with a grid stride, so the launch never has to fit a
/// dimension under 65535 and a thread pays the decode once per row iteration
/// rather than once per element.
template <class T, int Mode, int VW, bool ThreeD>
__global__ void resize_first_kernel(const T* __restrict__ in, T* __restrict__ out,
                                    ResizeGeom g)
{
    const int64_t ow0 = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * VW;
    if (ow0 >= g.ow) { return; }
    const int cols = int(g.ow - ow0 < int64_t(VW) ? g.ow - ow0 : int64_t(VW));

    for (int64_t nc = blockIdx.z; nc < g.n * g.c; nc += gridDim.z) {
        const int64_t bn = nc / g.c;
        const int64_t bc = nc - bn * g.c;
        const int64_t in_plane = bn * g.in_n + bc * g.in_c + g.crop_off;
        const int64_t out_plane = bn * g.out_n + bc * g.out_c;

        const int64_t r0 = int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
        const int64_t rstep = int64_t(gridDim.y) * blockDim.y;
        for (int64_t r = r0; r < g.od * g.oh; r += rstep) {
            const int64_t od = ThreeD ? r / g.oh : 0;
            const int64_t oh = ThreeD ? r - od * g.oh : r;
            T* orow = out + out_plane + od * g.out_d + oh * g.out_row + ow0;

            if constexpr (Mode == 0) {
                int64_t iz = 0;
                if constexpr (ThreeD) {
                    iz = rsz_nearest(od, g.id, g.od, g.coord_mode);
                }
                const int64_t iy = rsz_nearest(oh, g.ih, g.oh, g.coord_mode);
                const T* row = in + in_plane + iz * g.in_d + iy * g.in_row;
                for (int v = 0; v < cols; ++v) {
                    const int64_t ix = rsz_nearest(ow0 + v, g.iw, g.ow, g.coord_mode);
                    rsz_store(&orow[v], s_load(&row[ix]), g.add_to);
                }
            } else if constexpr (Mode == 1) {
                const LinearTap ty = rsz_linear_tap(oh, g.ih, g.oh, g.coord_mode);
                if constexpr (!ThreeD) {
                    const T* r0 = in + in_plane + ty.i0 * g.in_row;
                    const T* r1 = in + in_plane + ty.i1 * g.in_row;
                    for (int v = 0; v < cols; ++v) {
                        const LinearTap tx =
                            rsz_linear_tap(ow0 + v, g.iw, g.ow, g.coord_mode);
                        const float wx0 = 1.0f - tx.w;
                        const float top = wx0 * s_load(&r0[tx.i0]) +
                                          tx.w * s_load(&r0[tx.i1]);
                        const float bot = wx0 * s_load(&r1[tx.i0]) +
                                          tx.w * s_load(&r1[tx.i1]);
                        rsz_store(&orow[v], (1.0f - ty.w) * top + ty.w * bot,
                                  g.add_to);
                    }
                } else {
                    const LinearTap tz = rsz_linear_tap(od, g.id, g.od, g.coord_mode);
                    const T* d0 = in + in_plane + tz.i0 * g.in_d;
                    const T* d1 = in + in_plane + tz.i1 * g.in_d;
                    const float wz0 = 1.0f - tz.w;
                    for (int v = 0; v < cols; ++v) {
                        const LinearTap tx =
                            rsz_linear_tap(ow0 + v, g.iw, g.ow, g.coord_mode);
                        const float wx0 = 1.0f - tx.w;
                        const float y0t = wx0 * s_load(&d0[ty.i0 * g.in_row + tx.i0]) +
                                          tx.w * s_load(&d0[ty.i0 * g.in_row + tx.i1]);
                        const float y0b = wx0 * s_load(&d0[ty.i1 * g.in_row + tx.i0]) +
                                          tx.w * s_load(&d0[ty.i1 * g.in_row + tx.i1]);
                        const float y1t = wx0 * s_load(&d1[ty.i0 * g.in_row + tx.i0]) +
                                          tx.w * s_load(&d1[ty.i0 * g.in_row + tx.i1]);
                        const float y1b = wx0 * s_load(&d1[ty.i1 * g.in_row + tx.i0]) +
                                          tx.w * s_load(&d1[ty.i1 * g.in_row + tx.i1]);
                        const float y0 = (1.0f - ty.w) * y0t + ty.w * y0b;
                        const float y1 = (1.0f - ty.w) * y1t + ty.w * y1b;
                        rsz_store(&orow[v], wz0 * y0 + tz.w * y1, g.add_to);
                    }
                }
            } else {
                const CubicTap ty =
                    rsz_cubic_tap(oh, g.ih, g.oh, g.coord_mode, g.cubic_a);
                const T* rows[4];
                for (int j = 0; j < 4; ++j) {
                    rows[j] = in + in_plane + ty.idx[j] * g.in_row;
                }
                for (int v = 0; v < cols; ++v) {
                    const CubicTap tx =
                        rsz_cubic_tap(ow0 + v, g.iw, g.ow, g.coord_mode, g.cubic_a);
                    float acc = 0.0f;
                    for (int j = 0; j < 4; ++j) {
                        const T* r = rows[j];
                        float h = tx.w[0] * s_load(&r[tx.idx[0]]);
                        h += tx.w[1] * s_load(&r[tx.idx[1]]);
                        h += tx.w[2] * s_load(&r[tx.idx[2]]);
                        h += tx.w[3] * s_load(&r[tx.idx[3]]);
                        acc += ty.w[j] * h;
                    }
                    rsz_store(&orow[v], acc, g.add_to);
                }
            }
        }
    }
}

// ============================================================
// Channel-last — lanes on the channel vector
// ============================================================

/// One 16 B channel lane of one output pixel per thread. threadIdx.x walks the
/// block's channel chunk and threadIdx.y consecutive output pixels, so a warp
/// touches one contiguous run of the pixel-major image.
template <class T, int Mode, int Vec, bool ThreeD>
__global__ void resize_last_vec_kernel(const T* __restrict__ in, T* __restrict__ out,
                                       ResizeGeom g, int c_blocks)
{
    using V = std::conditional_t<sizeof(T) == 2, pack8<T>, float4>;
    static_assert(Vec * sizeof(T) == 16, "the channel lane is one 128-bit vector");

    const int cb = int(blockIdx.x) % c_blocks;
    const int wb = int(blockIdx.x) / c_blocks;
    const int64_t c0 = (int64_t(cb) * blockDim.x + threadIdx.x) * Vec;
    const int64_t ow = int64_t(wb) * blockDim.y + threadIdx.y;
    if (c0 >= g.c || ow >= g.ow) { return; }

    // Only the channel offset is fixed here: every source tap below adds its own
    // column * in_pix, so carrying a column term here would count it twice.
    const int64_t in_x = c0;
    const int64_t out_x = ow * g.out_pix + c0;

    for (int64_t bn = blockIdx.z; bn < g.n; bn += gridDim.z) {
        const int64_t in_img = bn * g.in_n + g.crop_off + in_x;
        const int64_t out_img = bn * g.out_n + out_x;

        for (int64_t r = blockIdx.y; r < g.od * g.oh; r += gridDim.y) {
            const int64_t od = ThreeD ? r / g.oh : 0;
            const int64_t oh = ThreeD ? r - od * g.oh : r;
            T* dst = out + out_img + od * g.out_d + oh * g.out_row;
            const T* out_c = dst;

            if constexpr (Mode == 0) {
                int64_t iz = 0;
                if constexpr (ThreeD) {
                    iz = rsz_nearest(od, g.id, g.od, g.coord_mode);
                }
                const int64_t iy = rsz_nearest(oh, g.ih, g.oh, g.coord_mode);
                const int64_t ix = rsz_nearest(ow, g.iw, g.ow, g.coord_mode);
                const T* src = in + in_img + iz * g.in_d + iy * g.in_row + ix * g.in_pix;
                const V x = *reinterpret_cast<const V*>(src);
                const T* xs = reinterpret_cast<const T*>(&x);
                V res;
                T* rs = reinterpret_cast<T*>(&res);
#pragma unroll
                for (int i = 0; i < Vec; ++i) {
                    float f = s_load(&xs[i]);
                    if (g.add_to) { f += s_load(&out_c[i]); }
                    s_store(&rs[i], f);
                }
                *reinterpret_cast<V*>(dst) = res;
            } else if constexpr (Mode == 1) {
                const LinearTap ty = rsz_linear_tap(oh, g.ih, g.oh, g.coord_mode);
                const LinearTap tx = rsz_linear_tap(ow, g.iw, g.ow, g.coord_mode);
                const float wx0 = 1.0f - tx.w;
                const float wy0 = 1.0f - ty.w;
                const int64_t xa = tx.i0 * g.in_pix;
                const int64_t xb = tx.i1 * g.in_pix;

                if constexpr (!ThreeD) {
                    const T* a0 = in + in_img + ty.i0 * g.in_row + xa;
                    const T* a1 = in + in_img + ty.i0 * g.in_row + xb;
                    const T* b0 = in + in_img + ty.i1 * g.in_row + xa;
                    const T* b1 = in + in_img + ty.i1 * g.in_row + xb;
                    const V va0 = *reinterpret_cast<const V*>(a0);
                    const V va1 = *reinterpret_cast<const V*>(a1);
                    const V vb0 = *reinterpret_cast<const V*>(b0);
                    const V vb1 = *reinterpret_cast<const V*>(b1);
                    const T* p0 = reinterpret_cast<const T*>(&va0);
                    const T* p1 = reinterpret_cast<const T*>(&va1);
                    const T* p2 = reinterpret_cast<const T*>(&vb0);
                    const T* p3 = reinterpret_cast<const T*>(&vb1);
                    V res;
                    T* rs = reinterpret_cast<T*>(&res);
#pragma unroll
                    for (int i = 0; i < Vec; ++i) {
                        const float top = wx0 * s_load(&p0[i]) + tx.w * s_load(&p1[i]);
                        const float bot = wx0 * s_load(&p2[i]) + tx.w * s_load(&p3[i]);
                        float f = wy0 * top + ty.w * bot;
                        if (g.add_to) { f += s_load(&out_c[i]); }
                        s_store(&rs[i], f);
                    }
                    *reinterpret_cast<V*>(dst) = res;
                } else {
                    const LinearTap tz = rsz_linear_tap(od, g.id, g.od, g.coord_mode);
                    const float wz0 = 1.0f - tz.w;
                    const T* d0 = in + in_img + tz.i0 * g.in_d;
                    const T* d1 = in + in_img + tz.i1 * g.in_d;
                    const V v00 = *reinterpret_cast<const V*>(d0 + ty.i0 * g.in_row + xa);
                    const V v10 = *reinterpret_cast<const V*>(d0 + ty.i0 * g.in_row + xb);
                    const V v01 = *reinterpret_cast<const V*>(d0 + ty.i1 * g.in_row + xa);
                    const V v11 = *reinterpret_cast<const V*>(d0 + ty.i1 * g.in_row + xb);
                    const V u00 = *reinterpret_cast<const V*>(d1 + ty.i0 * g.in_row + xa);
                    const V u10 = *reinterpret_cast<const V*>(d1 + ty.i0 * g.in_row + xb);
                    const V u01 = *reinterpret_cast<const V*>(d1 + ty.i1 * g.in_row + xa);
                    const V u11 = *reinterpret_cast<const V*>(d1 + ty.i1 * g.in_row + xb);
                    const T* q0 = reinterpret_cast<const T*>(&v00);
                    const T* q1 = reinterpret_cast<const T*>(&v10);
                    const T* q2 = reinterpret_cast<const T*>(&v01);
                    const T* q3 = reinterpret_cast<const T*>(&v11);
                    const T* q4 = reinterpret_cast<const T*>(&u00);
                    const T* q5 = reinterpret_cast<const T*>(&u10);
                    const T* q6 = reinterpret_cast<const T*>(&u01);
                    const T* q7 = reinterpret_cast<const T*>(&u11);
                    V res;
                    T* rs = reinterpret_cast<T*>(&res);
#pragma unroll
                    for (int i = 0; i < Vec; ++i) {
                        const float a = wx0 * s_load(&q0[i]) + tx.w * s_load(&q1[i]);
                        const float b = wx0 * s_load(&q2[i]) + tx.w * s_load(&q3[i]);
                        const float c = wx0 * s_load(&q4[i]) + tx.w * s_load(&q5[i]);
                        const float d = wx0 * s_load(&q6[i]) + tx.w * s_load(&q7[i]);
                        const float y0 = wy0 * a + ty.w * b;
                        const float y1 = wy0 * c + ty.w * d;
                        float f = wz0 * y0 + tz.w * y1;
                        if (g.add_to) { f += s_load(&out_c[i]); }
                        s_store(&rs[i], f);
                    }
                    *reinterpret_cast<V*>(dst) = res;
                }
            } else {
                const CubicTap ty =
                    rsz_cubic_tap(oh, g.ih, g.oh, g.coord_mode, g.cubic_a);
                const CubicTap tx =
                    rsz_cubic_tap(ow, g.iw, g.ow, g.coord_mode, g.cubic_a);
                const T* rows[4];
                for (int j = 0; j < 4; ++j) {
                    rows[j] = in + in_img + ty.idx[j] * g.in_row;
                }
                V res;
                T* rs = reinterpret_cast<T*>(&res);
                float acc[Vec];
#pragma unroll
                for (int i = 0; i < Vec; ++i) { acc[i] = 0.0f; }
                for (int j = 0; j < 4; ++j) {
                    const T* r = rows[j];
                    const V v0 = *reinterpret_cast<const V*>(r + tx.idx[0] * g.in_pix);
                    const V v1 = *reinterpret_cast<const V*>(r + tx.idx[1] * g.in_pix);
                    const V v2 = *reinterpret_cast<const V*>(r + tx.idx[2] * g.in_pix);
                    const V v3 = *reinterpret_cast<const V*>(r + tx.idx[3] * g.in_pix);
                    const T* p0 = reinterpret_cast<const T*>(&v0);
                    const T* p1 = reinterpret_cast<const T*>(&v1);
                    const T* p2 = reinterpret_cast<const T*>(&v2);
                    const T* p3 = reinterpret_cast<const T*>(&v3);
#pragma unroll
                    for (int i = 0; i < Vec; ++i) {
                        float h = tx.w[0] * s_load(&p0[i]);
                        h += tx.w[1] * s_load(&p1[i]);
                        h += tx.w[2] * s_load(&p2[i]);
                        h += tx.w[3] * s_load(&p3[i]);
                        acc[i] += ty.w[j] * h;
                    }
                }
#pragma unroll
                for (int i = 0; i < Vec; ++i) {
                    float f = acc[i];
                    if (g.add_to) { f += s_load(&out_c[i]); }
                    s_store(&rs[i], f);
                }
                *reinterpret_cast<V*>(dst) = res;
            }
        }
    }
}

/// Channel-last with the channels walked inside one thread: the fallback for a
/// channel count that is not a 16 B multiple (an RGB image) or too narrow to fill
/// a warp's worth of vectors.
template <class T, int Mode, bool ThreeD>
__global__ void resize_last_scalar_kernel(const T* __restrict__ in, T* __restrict__ out,
                                          ResizeGeom g)
{
    const int64_t ow = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ow >= g.ow) { return; }

    for (int64_t bn = blockIdx.z; bn < g.n; bn += gridDim.z) {
        const int64_t in_img = bn * g.in_n + g.crop_off;
        const int64_t out_img = bn * g.out_n;

        for (int64_t r = blockIdx.y; r < g.od * g.oh; r += gridDim.y) {
            const int64_t od = ThreeD ? r / g.oh : 0;
            const int64_t oh = ThreeD ? r - od * g.oh : r;
            T* dst = out + out_img + od * g.out_d + oh * g.out_row + ow * g.out_pix;

            if constexpr (Mode == 0) {
                int64_t iz = 0;
                if constexpr (ThreeD) {
                    iz = rsz_nearest(od, g.id, g.od, g.coord_mode);
                }
                const int64_t iy = rsz_nearest(oh, g.ih, g.oh, g.coord_mode);
                const int64_t ix = rsz_nearest(ow, g.iw, g.ow, g.coord_mode);
                const T* src = in + in_img + iz * g.in_d + iy * g.in_row + ix * g.in_pix;
                for (int64_t c = 0; c < g.c; ++c) {
                    rsz_store(&dst[c], s_load(&src[c]), g.add_to);
                }
            } else if constexpr (Mode == 1) {
                const LinearTap ty = rsz_linear_tap(oh, g.ih, g.oh, g.coord_mode);
                const LinearTap tx = rsz_linear_tap(ow, g.iw, g.ow, g.coord_mode);
                const float wx0 = 1.0f - tx.w;
                const float wy0 = 1.0f - ty.w;
                const int64_t xa = tx.i0 * g.in_pix;
                const int64_t xb = tx.i1 * g.in_pix;
                if constexpr (!ThreeD) {
                    const T* a0 = in + in_img + ty.i0 * g.in_row + xa;
                    const T* a1 = in + in_img + ty.i0 * g.in_row + xb;
                    const T* b0 = in + in_img + ty.i1 * g.in_row + xa;
                    const T* b1 = in + in_img + ty.i1 * g.in_row + xb;
                    for (int64_t c = 0; c < g.c; ++c) {
                        const float top = wx0 * s_load(&a0[c]) + tx.w * s_load(&a1[c]);
                        const float bot = wx0 * s_load(&b0[c]) + tx.w * s_load(&b1[c]);
                        rsz_store(&dst[c], wy0 * top + ty.w * bot, g.add_to);
                    }
                } else {
                    const LinearTap tz = rsz_linear_tap(od, g.id, g.od, g.coord_mode);
                    const float wz0 = 1.0f - tz.w;
                    const T* d0 = in + in_img + tz.i0 * g.in_d;
                    const T* d1 = in + in_img + tz.i1 * g.in_d;
                    for (int64_t c = 0; c < g.c; ++c) {
                        const float a = wx0 * s_load(&d0[ty.i0 * g.in_row + xa + c]) +
                                        tx.w * s_load(&d0[ty.i0 * g.in_row + xb + c]);
                        const float b = wx0 * s_load(&d0[ty.i1 * g.in_row + xa + c]) +
                                        tx.w * s_load(&d0[ty.i1 * g.in_row + xb + c]);
                        const float e = wx0 * s_load(&d1[ty.i0 * g.in_row + xa + c]) +
                                        tx.w * s_load(&d1[ty.i0 * g.in_row + xb + c]);
                        const float f = wx0 * s_load(&d1[ty.i1 * g.in_row + xa + c]) +
                                        tx.w * s_load(&d1[ty.i1 * g.in_row + xb + c]);
                        const float y0 = wy0 * a + ty.w * b;
                        const float y1 = wy0 * e + ty.w * f;
                        rsz_store(&dst[c], wz0 * y0 + tz.w * y1, g.add_to);
                    }
                }
            } else {
                const CubicTap ty =
                    rsz_cubic_tap(oh, g.ih, g.oh, g.coord_mode, g.cubic_a);
                const CubicTap tx =
                    rsz_cubic_tap(ow, g.iw, g.ow, g.coord_mode, g.cubic_a);
                const T* rows[4];
                for (int j = 0; j < 4; ++j) {
                    rows[j] = in + in_img + ty.idx[j] * g.in_row;
                }
                for (int64_t c = 0; c < g.c; ++c) {
                    float acc = 0.0f;
                    for (int j = 0; j < 4; ++j) {
                        const T* r = rows[j];
                        float h = tx.w[0] * s_load(&r[tx.idx[0] * g.in_pix + c]);
                        h += tx.w[1] * s_load(&r[tx.idx[1] * g.in_pix + c]);
                        h += tx.w[2] * s_load(&r[tx.idx[2] * g.in_pix + c]);
                        h += tx.w[3] * s_load(&r[tx.idx[3] * g.in_pix + c]);
                        acc += ty.w[j] * h;
                    }
                    rsz_store(&dst[c], acc, g.add_to);
                }
            }
        }
    }
}

// ============================================================
// Launch
// ============================================================

template <class T, int Mode, int VW, bool ThreeD>
void launch_first(const ResizeGeom& g, const T* in, T* out, cudaStream_t stream)
{
    // The column lanes of a row are fixed by VW, so the block is filled with
    // rows instead: a 64-column plane leaves 16 live lanes out of 128, and at
    // one row per block the plane's 512 images cost 32k mostly-idle blocks.
    constexpr int kThreads = 256;
    const int lanes = static_cast<int>(std::min<int64_t>(ceil_div(g.ow, VW), kThreads));
    const int by = std::max(1, kThreads / lanes);
    const dim3 block(static_cast<unsigned>(lanes), static_cast<unsigned>(by), 1);
    const dim3 grid(static_cast<unsigned>(ceil_div(g.ow, int64_t(lanes) * VW)),
                    static_cast<unsigned>(
                        std::min<int64_t>(ceil_div(g.od * g.oh, by), 65535)),
                    static_cast<unsigned>(std::min<int64_t>(g.n * g.c, 65535)));
    resize_first_kernel<T, Mode, VW, ThreeD>
        <<<grid, block, 0, stream>>>(in, out, g);
}

template <class T, int Mode, int Vec, bool ThreeD>
void launch_last_vec(const ResizeGeom& g, const T* in, T* out, int bx, int by,
                     cudaStream_t stream)
{
    const int c_blocks =
        static_cast<int>(ceil_div(g.c / Vec, int64_t(bx)));
    const int w_blocks = static_cast<int>(ceil_div(g.ow, int64_t(by)));
    const dim3 block(static_cast<unsigned>(bx), static_cast<unsigned>(by), 1);
    const dim3 grid(static_cast<unsigned>(int64_t(c_blocks) * w_blocks),
                    static_cast<unsigned>(std::min<int64_t>(g.od * g.oh, 65535)),
                    static_cast<unsigned>(std::min<int64_t>(g.n, 65535)));
    resize_last_vec_kernel<T, Mode, Vec, ThreeD>
        <<<grid, block, 0, stream>>>(in, out, g, c_blocks);
}

template <class T, int Mode, bool ThreeD>
void launch_last_scalar(const ResizeGeom& g, const T* in, T* out, cudaStream_t stream)
{
    constexpr int kThreads = 128;
    const dim3 grid(static_cast<unsigned>(ceil_div(g.ow, int64_t(kThreads))),
                    static_cast<unsigned>(std::min<int64_t>(g.od * g.oh, 65535)),
                    static_cast<unsigned>(std::min<int64_t>(g.n, 65535)));
    resize_last_scalar_kernel<T, Mode, ThreeD>
        <<<grid, dim3(kThreads, 1, 1), 0, stream>>>(in, out, g);
}

/// The channel-vector lane needs a channel count that fills it, a pixel stride
/// that keeps every lane 16 B aligned, and 16 B-aligned buffers — the pitch model
/// supplies the row alignment, so only the base pointer is checked here.
template <class T>
bool lane_ok(const ResizeGeom& g, const T* in, T* out)
{
    constexpr int Vec = 16 / int(sizeof(T));
    if (g.c % Vec != 0 || g.c / Vec < 4) { return false; }
    if (g.in_pix != g.c || g.out_pix != g.c) { return false; }
    if (g.in_row % Vec != 0 || g.out_row % Vec != 0) { return false; }
    if (reinterpret_cast<uintptr_t>(in) % 16 != 0) { return false; }
    if (reinterpret_cast<uintptr_t>(out) % 16 != 0) { return false; }
    return true;
}

template <class T, int Mode, bool ThreeD>
void resize_launch(const ResizeGeom& g, bool last, const T* in, T* out,
                   cudaStream_t stream)
{
    if (!last) {
        // A wider window buys columns at the cost of row lanes, and it only pays
        // while neighbouring output columns still share their source taps: that
        // is always true going up and for a nearest pass (one load a column),
        // but on a bilinear downsample each column lands on its own pair, so the
        // window stops at four and the block keeps its lanes.
        const bool share = (Mode == 0) || g.ow >= g.iw;
        if (g.ow >= 32 && share) {
            launch_first<T, Mode, 8, ThreeD>(g, in, out, stream);
        } else if (g.ow >= 64) {
            launch_first<T, Mode, 4, ThreeD>(g, in, out, stream);
        } else {
            launch_first<T, Mode, 1, ThreeD>(g, in, out, stream);
        }
        return;
    }

    constexpr int Vec = 16 / int(sizeof(T));
    if (lane_ok<T>(g, in, out)) {
        const int64_t cvecs = g.c / Vec;
        const int bx = static_cast<int>(std::min<int64_t>(cvecs, 64));
        const int by = std::max(1, 256 / bx);
        launch_last_vec<T, Mode, Vec, ThreeD>(g, in, out, bx, by, stream);
        return;
    }
    launch_last_scalar<T, Mode, ThreeD>(g, in, out, stream);
}

// ============================================================
// Driver
// ============================================================

template <class T>
void resize_impl(const ResizeAttributes& attrs, TensorView& output,
                 std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    const int64_t srank = ResizeAttributes::spatial_rank(rank);
    const bool three_d = (srank == 3);
    const bool last = is_channel_last(input.layout());

    ResizeGeom g{};
    g.n = input.shape(0);
    g.c = input.shape(1);
    g.id = three_d ? input.shape(2) : 1;
    g.ih = input.shape(srank);
    g.iw = input.shape(srank + 1);
    g.od = three_d ? output.shape(2) : 1;
    g.oh = output.shape(srank);
    g.ow = output.shape(srank + 1);
    g.coord_mode = int(attrs.coord_mode);
    g.cubic_a = attrs.cubic_coeff_a;
    g.add_to = attrs.add_to;

    const int64_t raw_ih = g.ih;
    if (attrs.has_crop()) {
        if (three_d && attrs.crop_end[0] > 0) {
            g.id = attrs.crop_end[0] - attrs.crop_start[0];
            g.crop_off += attrs.crop_start[0] * input.stride_elems(2);
        }
        if (attrs.crop_end[1] > 0) {
            g.ih = attrs.crop_end[1] - attrs.crop_start[1];
            g.crop_off += attrs.crop_start[1] * input.stride_elems(rank - 2);
        }
        if (attrs.crop_end[2] > 0) {
            g.iw = attrs.crop_end[2] - attrs.crop_start[2];
            g.crop_off += attrs.crop_start[2] * input.stride_elems(rank - 1);
        }
    }

    if (last) {
        g.in_n = input.stride_elems(0);
        g.in_row = input.row_stride_elems();
        g.in_d = raw_ih * g.in_row;
        g.in_pix = g.c;
        g.out_n = output.stride_elems(0);
        g.out_row = output.row_stride_elems();
        g.out_d = g.oh * g.out_row;
        g.out_pix = g.c;
    } else {
        g.in_n = input.stride_elems(0);
        g.in_c = input.stride_elems(1);
        g.in_row = input.row_stride_elems();
        g.in_d = raw_ih * g.in_row;
        g.out_n = output.stride_elems(0);
        g.out_c = output.stride_elems(1);
        g.out_row = output.row_stride_elems();
        g.out_d = g.oh * g.out_row;
    }

    if (g.n == 0 || g.c == 0 || g.ow == 0 || g.oh == 0 || g.od == 0) { return; }
    if (g.iw == 0 || g.ih == 0 || g.id == 0) { return; }

    const T* in = input.ptr<T>();
    T* out = output.ptr<T>();
    const cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    switch (attrs.mode) {
    case ResizeMode::Nearest:
        if (three_d) {
            resize_launch<T, 0, true>(g, last, in, out, stream);
        } else {
            resize_launch<T, 0, false>(g, last, in, out, stream);
        }
        break;
    case ResizeMode::Linear:
        if (three_d) {
            resize_launch<T, 1, true>(g, last, in, out, stream);
        } else {
            resize_launch<T, 1, false>(g, last, in, out, stream);
        }
        break;
    default:
        resize_launch<T, 2, false>(g, last, in, out, stream);
        break;
    }
}

}  // anonymous namespace

Status resize_cuda(const ResizeAttributes& attrs, TensorView& output,
                   std::span<const TensorView> inputs, const ComputeContext& ctx,
                   void* /*workspace*/)
{
    // Cubic is the 2D bicubic; PyTorch and ONNX Runtime stop there too.
    if (attrs.mode == ResizeMode::Cubic &&
        ResizeAttributes::spatial_rank(inputs[0].rank()) != 2) {
        return Status::unsupported;
    }
    switch (inputs[0].data_type()) {
    case DataType::f32:
        resize_impl<float>(attrs, output, inputs, ctx);
        return Status::ok;
    case DataType::f16:
        resize_impl<__half>(attrs, output, inputs, ctx);
        return Status::ok;
    case DataType::bf16:
        resize_impl<__nv_bfloat16>(attrs, output, inputs, ctx);
        return Status::ok;
    default:
        return Status::unsupported;
    }
}

}  // namespace nnops::backend::cuda
