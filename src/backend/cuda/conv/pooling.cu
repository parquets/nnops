/// @file pooling.cu
/// @brief CUDA implementation of 2D/3D spatial pooling — Max and Average — on a
///        Channel-first or a channel-last tensor.
///
/// Channel-first keeps the lane on output width, because that is the contiguous
/// axis there. The block coordinates carry (ow, oh) and the channel index rides
/// in blockIdx.z, so no thread decodes a flat spatial index by dividing. A
/// square kernel takes one more step: a thread owns VW output columns and loads
/// its row window once per tap row — (VW-1)·SW + K elements instead of VW·KW
/// loads — and, because the column set does not depend on the tap row, the
/// per-output valid-column count is computed once and reused for every row.
/// Strided, dilated and 3D windows fall to the plain grid kernel; a geometry
/// whose (n, c, od) axis outgrows blockIdx.z keeps the flat original.
///
/// Channel-last puts the lane on the channel instead, for the same reason — a
/// thread owns one 16 B vector of a pixel (four f32 or eight f16) and the window
/// walk becomes one 128-bit load a tap. Adjacent threads take adjacent channel
/// vectors of one pixel, which is what keeps the loads coalesced. A channel
/// count that is not a vector multiple is covered by a second, scalar launch
/// over the remainder, and an unaligned pitch or pixel stride falls back to the
/// scalar script entirely.
///
/// A square unit-stride channel-last kernel walks ROWS output rows per thread:
/// consecutive rows share K-1 of their input rows, so ROWS+K-1 loads feed ROWS
/// output rows, and each load contributes to up to K of them.
///
/// Global pooling is a separate pair. A channel-first plane reduces across a
/// block. A channel-last image reduces across pixels with one thread per channel
/// vector, and the pixel range is split across blockDim.y so one image can fill
/// the machine: the lanes of a warp still hold one pixel's channel vectors,
/// which keeps the loads coalesced while the rows are walked in parallel.
///
/// Supports f32 and f16. Workspace: not used.

#include "../cuda_common.cuh"
#include "nnops/ops/pooling.hpp"
#include "nnops/detail/assert.hpp"

#include <algorithm>

namespace nnops::backend::cuda {

// ============================================================
// The geometry every route shares
// ============================================================

/// One pooling call's window, both operands' strides (in elements) and its
/// attributes. Scalars only, passed by value.
struct PoolGeom {
    int64_t N, C;
    int64_t ID, IH, IW;
    int64_t OD, OH, OW;
    int64_t KD, KH, KW;
    int64_t SD, SH, SW;
    int64_t DD, DH, DW;
    int64_t PD, PH, PW;
    // Channel-first planes, or the channel-last image/pixel/row strides
    int64_t in_ch_stride, in_d_stride, in_row_stride;
    int64_t out_ch_stride, out_d_stride, out_row_stride;
    int64_t in_pix_stride, out_pix_stride;
    PoolingType type;
    bool exclude_pad;
    bool add_to;
};

// ============================================================
// Channel-first — the width axis is contiguous, so lanes walk ow
// ============================================================

/// One output per thread. Both block coordinates are output positions and the
/// depth plane comes from blockIdx.z, so the kernel never divides a thread's
/// index into (od, oh, ow) the way the flat script does.
template <typename T, bool IsMax>
__global__ void pooling_first_grid_kernel(const T* __restrict__ input,
                                          T* __restrict__ output, PoolGeom g)
{
    const int64_t od = int64_t(blockIdx.z) % g.OD;
    const int64_t nc = int64_t(blockIdx.z) / g.OD;
    const int64_t ow = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t oh = int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
    if (ow >= g.OW || oh >= g.OH) { return; }

    const T* in_ch  = input  + nc * g.in_ch_stride;
    T*       out_ch = output + nc * g.out_ch_stride;

    float result = 0.0f;
    int64_t valid = 0;
    float acc = IsMax ? -1e30f : 0.0f;

    for (int64_t kd = 0; kd < g.KD; ++kd) {
        const int64_t id = od * g.SD + kd * g.DD - g.PD;
        if (id < 0 || id >= g.ID) { continue; }
        const T* in_d = in_ch + id * g.in_d_stride;
        for (int64_t kh = 0; kh < g.KH; ++kh) {
            const int64_t ih = oh * g.SH + kh * g.DH - g.PH;
            if (ih < 0 || ih >= g.IH) { continue; }
            const T* row = in_d + ih * g.in_row_stride;
            for (int64_t kw = 0; kw < g.KW; ++kw) {
                const int64_t iw = ow * g.SW + kw * g.DW - g.PW;
                if (iw < 0 || iw >= g.IW) { continue; }
                const float v = s_load(&row[iw]);
                acc = IsMax ? fmaxf(acc, v) : acc + v;
                ++valid;
            }
        }
    }

    if (IsMax) {
        result = valid > 0 ? acc : 0.0f;
    } else {
        const int64_t denom = g.exclude_pad ? valid : g.KD * g.KH * g.KW;
        result = denom > 0 ? acc / static_cast<float>(denom) : 0.0f;
    }

    T* out = out_ch + od * g.out_d_stride + oh * g.out_row_stride + ow;
    if (g.add_to) {
        s_store(out, s_load(out) + result);
    } else {
        s_store(out, result);
    }
}

/// VW columns per thread with the row window held in registers. The window is
/// contiguous because SW and the tap step are both known at compile time, and
/// the valid-column count is a property of the column set alone, so it is
/// computed once rather than per tap row.
template <typename T, int K, int SW_, int VW, bool IsMax>
__global__ void pooling_first_window_kernel(const T* __restrict__ input,
                                            T* __restrict__ output, PoolGeom g)
{
    const int64_t od = int64_t(blockIdx.z) % g.OD;
    const int64_t nc = int64_t(blockIdx.z) / g.OD;
    const int64_t ow0 = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * VW;
    const int64_t oh  = int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
    if (ow0 >= g.OW || oh >= g.OH) { return; }

    constexpr int kSpan = (VW - 1) * SW_ + K;
    const int64_t cols = (g.OW - ow0 < int64_t(VW)) ? (g.OW - ow0) : int64_t(VW);
    const int64_t base = ow0 * SW_ - g.PW;

    bool ok[kSpan];
    #pragma unroll
    for (int j = 0; j < kSpan; ++j) {
        const int64_t iw = base + j;
        ok[j] = (iw >= 0 && iw < g.IW);
    }
    int colcnt[VW];
    #pragma unroll
    for (int v = 0; v < VW; ++v) {
        int n = 0;
        if (int64_t(v) < cols) {
            #pragma unroll
            for (int kw = 0; kw < K; ++kw) {
                if (ok[v * SW_ + kw]) { ++n; }
            }
        }
        colcnt[v] = n;
    }

    const T* in_ch = input + nc * g.in_ch_stride;
    float acc[VW];
    #pragma unroll
    for (int v = 0; v < VW; ++v) { acc[v] = IsMax ? -1e30f : 0.0f; }

    int64_t rows = 0;
    for (int64_t kd = 0; kd < g.KD; ++kd) {
        const int64_t id = od * g.SD + kd * g.DD - g.PD;
        if (id < 0 || id >= g.ID) { continue; }
        const T* in_d = in_ch + id * g.in_d_stride;
        for (int64_t kh = 0; kh < g.KH; ++kh) {
            const int64_t ih = oh * g.SH + kh * g.DH - g.PH;
            if (ih < 0 || ih >= g.IH) { continue; }
            ++rows;
            const T* row = in_d + ih * g.in_row_stride;
            float x[kSpan];
            #pragma unroll
            for (int j = 0; j < kSpan; ++j) {
                const int64_t iw = base + j;
                x[j] = ok[j] ? s_load(&row[iw]) : (IsMax ? -1e30f : 0.0f);
            }
            #pragma unroll
            for (int v = 0; v < VW; ++v) {
                #pragma unroll
                for (int kw = 0; kw < K; ++kw) {
                    const float t = x[v * SW_ + kw];
                    acc[v] = IsMax ? fmaxf(acc[v], t) : acc[v] + t;
                }
            }
        }
    }

    T* out = output + nc * g.out_ch_stride + od * g.out_d_stride + oh * g.out_row_stride + ow0;
    #pragma unroll
    for (int v = 0; v < VW; ++v) {
        if (int64_t(v) >= cols) { continue; }
        float r = acc[v];
        const int64_t taps = int64_t(colcnt[v]) * rows;
        if (IsMax) {
            r = taps > 0 ? r : 0.0f;
        } else {
            const int64_t denom = g.exclude_pad ? taps : g.KD * int64_t(K) * K;
            r = denom > 0 ? r / static_cast<float>(denom) : 0.0f;
        }
        T* o = out + v;
        if (g.add_to) {
            s_store(o, s_load(o) + r);
        } else {
            s_store(o, r);
        }
    }
}

/// The generic script: a block per (n, c) plane, a device-stride loop over the
/// flat spatial index, three divisions per output. Every geometry reaches it —
/// only the ones the grid axes cannot carry stay here.
template <typename T>
__global__ void pooling_first_flat_kernel(const T* __restrict__ input,
                                          T* __restrict__ output, PoolGeom g)
{
    const int64_t nc = blockIdx.y;
    const int64_t total_spatial = g.OD * g.OH * g.OW;
    const bool is_max = g.type == PoolingType::Max;

    const T* in_ch  = input  + nc * g.in_ch_stride;
    T*       out_ch = output + nc * g.out_ch_stride;

    for (int64_t sp = threadIdx.x; sp < total_spatial; sp += blockDim.x) {
        const int64_t od = sp / (g.OH * g.OW);
        const int64_t oh = (sp / g.OW) % g.OH;
        const int64_t ow = sp % g.OW;

        float acc = is_max ? -1e30f : 0.0f;
        int64_t valid = 0;
        for (int64_t kd = 0; kd < g.KD; ++kd) {
            const int64_t id = od * g.SD + kd * g.DD - g.PD;
            if (id < 0 || id >= g.ID) { continue; }
            const T* in_d = in_ch + id * g.in_d_stride;
            for (int64_t kh = 0; kh < g.KH; ++kh) {
                const int64_t ih = oh * g.SH + kh * g.DH - g.PH;
                if (ih < 0 || ih >= g.IH) { continue; }
                const T* row = in_d + ih * g.in_row_stride;
                for (int64_t kw = 0; kw < g.KW; ++kw) {
                    const int64_t iw = ow * g.SW + kw * g.DW - g.PW;
                    if (iw < 0 || iw >= g.IW) { continue; }
                    const float v = s_load(&row[iw]);
                    acc = is_max ? fmaxf(acc, v) : acc + v;
                    ++valid;
                }
            }
        }

        float result;
        if (is_max) {
            result = valid > 0 ? acc : 0.0f;
        } else {
            const int64_t denom = g.exclude_pad ? valid : g.KD * g.KH * g.KW;
            result = denom > 0 ? acc / static_cast<float>(denom) : 0.0f;
        }

        T* out = out_ch + od * g.out_d_stride + oh * g.out_row_stride + ow;
        if (g.add_to) {
            s_store(out, s_load(out) + result);
        } else {
            s_store(out, result);
        }
    }
}

// ============================================================
// Channel-last — the lane spans channels
// ============================================================

/// One output vector per thread, its position carried by the block axes: x is
/// the channel chunk, y the output width, z the (image, depth plane, row).
template <typename T, int Vec, bool IsMax>
__global__ void pooling_last_flat_kernel(const T* __restrict__ input,
                                         T* __restrict__ output, PoolGeom g,
                                         int64_t c_begin, int64_t chunks)
{
    const int64_t oh = int64_t(blockIdx.z) % g.OH;
    const int64_t p  = int64_t(blockIdx.z) / g.OH;
    const int64_t od = p % g.OD;
    const int64_t img = p / g.OD;
    const int64_t chunk = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t ow    = int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
    if (chunk >= chunks || ow >= g.OW) { return; }

    const T* in_img  = input  + img * (g.ID * g.in_d_stride) + c_begin + chunk * Vec;
    T*       out_img = output + img * (g.OD * g.out_d_stride) + c_begin + chunk * Vec;

    float acc[Vec];
    #pragma unroll
    for (int i = 0; i < Vec; ++i) { acc[i] = IsMax ? -1e30f : 0.0f; }

    int64_t valid = 0;
    for (int64_t kd = 0; kd < g.KD; ++kd) {
        const int64_t id = od * g.SD + kd * g.DD - g.PD;
        if (id < 0 || id >= g.ID) { continue; }
        const T* in_d = in_img + id * g.in_d_stride;
        for (int64_t kh = 0; kh < g.KH; ++kh) {
            const int64_t ih = oh * g.SH + kh * g.DH - g.PH;
            if (ih < 0 || ih >= g.IH) { continue; }
            const T* row = in_d + ih * g.in_row_stride;
            for (int64_t kw = 0; kw < g.KW; ++kw) {
                const int64_t iw = ow * g.SW + kw * g.DW - g.PW;
                if (iw < 0 || iw >= g.IW) { continue; }
                float v[Vec];
                load_channels<T, Vec>(row + iw * g.in_pix_stride, v);
                #pragma unroll
                for (int i = 0; i < Vec; ++i) {
                    acc[i] = IsMax ? fmaxf(acc[i], v[i]) : acc[i] + v[i];
                }
                ++valid;
            }
        }
    }

    if (IsMax) {
        #pragma unroll
        for (int i = 0; i < Vec; ++i) {
            acc[i] = (valid > 0) ? acc[i] : 0.0f;
        }
    } else {
        const int64_t denom = g.exclude_pad ? valid : g.KD * g.KH * g.KW;
        const float scale = denom > 0 ? 1.0f / static_cast<float>(denom) : 0.0f;
        #pragma unroll
        for (int i = 0; i < Vec; ++i) { acc[i] *= scale; }
    }

    T* out = out_img + od * g.out_d_stride + oh * g.out_row_stride + ow * g.out_pix_stride;
    if (g.add_to) {
        float prev[Vec];
        load_channels<T, Vec>(out, prev);
        #pragma unroll
        for (int i = 0; i < Vec; ++i) { acc[i] += prev[i]; }
    }
    store_channels<T, Vec>(out, acc);
}

/// ROWS output rows per thread. Consecutive rows share K-1 of their input rows,
/// so a tap column costs ROWS+K-1 loads for ROWS outputs; the column validity is
/// row-independent, so the per-row divisor folds into one count multiplied by
/// the valid row and depth taps.
template <typename T, int Vec, int K, int ROWS, bool IsMax>
__global__ void pooling_last_rows_kernel(const T* __restrict__ input,
                                         T* __restrict__ output, PoolGeom g,
                                         int64_t c_begin, int64_t chunks)
{
    const int64_t row_blocks = (g.OH + ROWS - 1) / ROWS;
    const int64_t rb = int64_t(blockIdx.z) % row_blocks;
    const int64_t p  = int64_t(blockIdx.z) / row_blocks;
    const int64_t od = p % g.OD;
    const int64_t img = p / g.OD;
    const int64_t chunk = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t ow    = int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
    if (chunk >= chunks || ow >= g.OW) { return; }

    const T* in_img  = input  + img * (g.ID * g.in_d_stride) + c_begin + chunk * Vec;
    T*       out_img = output + img * (g.OD * g.out_d_stride) + c_begin + chunk * Vec;

    const int64_t oh0  = rb * ROWS;
    const int64_t base = ow * g.SW - g.PW;

    float acc[ROWS][Vec];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        #pragma unroll
        for (int i = 0; i < Vec; ++i) { acc[r][i] = IsMax ? -1e30f : 0.0f; }
    }

    int colcnt = 0;
    #pragma unroll
    for (int kw = 0; kw < K; ++kw) {
        const int64_t iw = base + kw;
        if (iw >= 0 && iw < g.IW) { ++colcnt; }
    }
    int rowcnt[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        int n = 0;
        if (oh0 + r < g.OH) {
            #pragma unroll
            for (int kh = 0; kh < K; ++kh) {
                const int64_t iy = oh0 + r + kh - g.PH;
                if (iy >= 0 && iy < g.IH) { ++n; }
            }
        }
        rowcnt[r] = n;
    }

    int64_t dcount = 0;
    for (int64_t kd = 0; kd < g.KD; ++kd) {
        const int64_t id = od * g.SD + kd * g.DD - g.PD;
        if (id < 0 || id >= g.ID) { continue; }
        ++dcount;
        const T* in_d = in_img + id * g.in_d_stride;
        #pragma unroll
        for (int kw = 0; kw < K; ++kw) {
            const int64_t iw = base + kw;
            if (iw < 0 || iw >= g.IW) { continue; }
            const T* col = in_d + iw * g.in_pix_stride;
            #pragma unroll
            for (int j = 0; j < ROWS + K - 1; ++j) {
                const int64_t iy = oh0 + j - g.PH;
                if (iy < 0 || iy >= g.IH) { continue; }
                float v[Vec];
                load_channels<T, Vec>(col + iy * g.in_row_stride, v);
                #pragma unroll
                for (int kh = 0; kh < K; ++kh) {
                    const int r = j - kh;
                    if (r < 0 || r >= ROWS || oh0 + r >= g.OH) { continue; }
                    #pragma unroll
                    for (int i = 0; i < Vec; ++i) {
                        acc[r][i] = IsMax ? fmaxf(acc[r][i], v[i]) : acc[r][i] + v[i];
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        if (oh0 + r >= g.OH) { continue; }
        const int64_t taps = int64_t(colcnt) * rowcnt[r] * dcount;
        if (IsMax) {
            if (taps == 0) {
                #pragma unroll
                for (int i = 0; i < Vec; ++i) { acc[r][i] = 0.0f; }
            }
        } else {
            const int64_t denom = g.exclude_pad ? taps : g.KD * int64_t(K) * K;
            const float scale = denom > 0 ? 1.0f / static_cast<float>(denom) : 0.0f;
            #pragma unroll
            for (int i = 0; i < Vec; ++i) { acc[r][i] *= scale; }
        }
        T* out = out_img + od * g.out_d_stride + (oh0 + r) * g.out_row_stride
               + ow * g.out_pix_stride;
        if (g.add_to) {
            float prev[Vec];
            load_channels<T, Vec>(out, prev);
            #pragma unroll
            for (int i = 0; i < Vec; ++i) { acc[r][i] += prev[i]; }
        }
        store_channels<T, Vec>(out, acc[r]);
    }
}

/// The flat script the older launchers used: adjacent threads take adjacent
/// channel vectors, the spatial index is decoded by division. Reached only by a
/// geometry whose block axes cannot hold its task count.
template <typename T, int Vec>
__global__ void pooling_last_linear_kernel(const T* __restrict__ input,
                                           T* __restrict__ output, PoolGeom g,
                                           int64_t c_begin, int64_t chunks)
{
    const int64_t pix_per_image = g.OD * g.OH * g.OW;
    const int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= g.N * pix_per_image * chunks) { return; }

    const int64_t chunk = t % chunks;
    const int64_t pix   = t / chunks;
    const int64_t c0    = c_begin + chunk * Vec;

    const int64_t img = pix / pix_per_image;
    const int64_t sp  = pix % pix_per_image;
    const int64_t od = sp / (g.OH * g.OW);
    const int64_t oh = (sp / g.OW) % g.OH;
    const int64_t ow = sp % g.OW;

    const T* in_img  = input  + img * (g.ID * g.in_d_stride) + c0;
    T*       out_img = output + img * (g.OD * g.out_d_stride) + c0;

    const bool is_max = g.type == PoolingType::Max;
    float acc[Vec];
    #pragma unroll
    for (int i = 0; i < Vec; ++i) { acc[i] = is_max ? -1e30f : 0.0f; }

    int64_t valid = 0;
    for (int64_t kd = 0; kd < g.KD; ++kd) {
        const int64_t id = od * g.SD + kd * g.DD - g.PD;
        if (id < 0 || id >= g.ID) { continue; }
        const T* in_d = in_img + id * g.in_d_stride;
        for (int64_t kh = 0; kh < g.KH; ++kh) {
            const int64_t ih = oh * g.SH + kh * g.DH - g.PH;
            if (ih < 0 || ih >= g.IH) { continue; }
            const T* row = in_d + ih * g.in_row_stride;
            for (int64_t kw = 0; kw < g.KW; ++kw) {
                const int64_t iw = ow * g.SW + kw * g.DW - g.PW;
                if (iw < 0 || iw >= g.IW) { continue; }
                float v[Vec];
                load_channels<T, Vec>(row + iw * g.in_pix_stride, v);
                #pragma unroll
                for (int i = 0; i < Vec; ++i) {
                    acc[i] = is_max ? fmaxf(acc[i], v[i]) : acc[i] + v[i];
                }
                ++valid;
            }
        }
    }

    if (is_max) {
        #pragma unroll
        for (int i = 0; i < Vec; ++i) {
            acc[i] = (valid > 0) ? acc[i] : 0.0f;
        }
    } else {
        const int64_t denom = g.exclude_pad ? valid : g.KD * g.KH * g.KW;
        const float scale = denom > 0 ? 1.0f / static_cast<float>(denom) : 0.0f;
        #pragma unroll
        for (int i = 0; i < Vec; ++i) { acc[i] *= scale; }
    }

    T* out = out_img + od * g.out_d_stride + oh * g.out_row_stride + ow * g.out_pix_stride;
    if (g.add_to) {
        float prev[Vec];
        load_channels<T, Vec>(out, prev);
        #pragma unroll
        for (int i = 0; i < Vec; ++i) { acc[i] += prev[i]; }
    }
    store_channels<T, Vec>(out, acc);
}

// ============================================================
// Global pooling — every output spatial axis is one
// ============================================================

/// A channel-first plane is walked per row rather than as one run: a padded
/// pitch puts dead bytes between rows, and reading through them would fold them
/// into the reduction.
template <typename T, bool IsMax>
__global__ void pooling_global_first_kernel(const T* __restrict__ input,
                                            T* __restrict__ output,
                                            int64_t rows, int64_t width,
                                            int64_t row_stride,
                                            int64_t in_ch_stride,
                                            int64_t out_ch_stride, bool add_to)
{
    __shared__ float smem[kCudaBlockSize / kCudaWarpSize];

    const int64_t nc = blockIdx.x;
    const T* in = input + nc * in_ch_stride;

    const int64_t plane = rows * width;
    float acc = IsMax ? -1e30f : 0.0f;
    for (int64_t i = threadIdx.x; i < plane; i += blockDim.x) {
        const int64_t r = i / width;
        const int64_t w = i - r * width;
        const float v = s_load(in + r * row_stride + w);
        acc = IsMax ? fmaxf(acc, v) : acc + v;
    }
    acc = IsMax ? block_reduce_max(acc, smem) : block_reduce_sum(acc, smem);

    if (threadIdx.x == 0) {
        float result = acc;
        if (!IsMax) {
            result = plane > 0 ? result / static_cast<float>(plane) : 0.0f;
        }
        T* out = output + nc * out_ch_stride;
        s_store(out, add_to ? s_load(out) + result : result);
    }
}

/// One image per blockIdx.y, its pixel rows split across blockDim.y: a warp
/// still holds one pixel's channel vectors (so the loads stay coalesced), but
/// one image can fill several blocks' worth of lanes instead of being one
/// thread's serial walk.
template <typename T, int Vec, bool IsMax>
__global__ void pooling_global_last_kernel(const T* __restrict__ input,
                                           T* __restrict__ output,
                                           int64_t c_begin, int64_t chunks,
                                           int64_t rows, int64_t width,
                                           int64_t in_img_stride,
                                           int64_t in_row_stride,
                                           int64_t in_pix_stride,
                                           int64_t out_img_stride,
                                           int64_t out_pix_stride,
                                           bool add_to)
{
    const int64_t chunk = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t img   = blockIdx.y;
    const bool active = chunk < chunks;

    float acc[Vec];
    #pragma unroll
    for (int i = 0; i < Vec; ++i) { acc[i] = IsMax ? -1e30f : 0.0f; }

    if (active) {
        const T* in_i = input + img * in_img_stride + c_begin + chunk * Vec;
        for (int64_t r = threadIdx.y; r < rows; r += blockDim.y) {
            const T* row = in_i + r * in_row_stride;
            for (int64_t w = 0; w < width; ++w) {
                float v[Vec];
                load_channels<T, Vec>(row + w * in_pix_stride, v);
                #pragma unroll
                for (int i = 0; i < Vec; ++i) {
                    acc[i] = IsMax ? fmaxf(acc[i], v[i]) : acc[i] + v[i];
                }
            }
        }
    }

    __shared__ float red[kCudaBlockSize * Vec];
    const int tid = int(threadIdx.y * blockDim.x + threadIdx.x);
    #pragma unroll
    for (int i = 0; i < Vec; ++i) { red[tid * Vec + i] = acc[i]; }
    __syncthreads();

    if (threadIdx.y != 0 || !active) { return; }
    const int bx = int(blockDim.x);
    #pragma unroll
    for (int i = 0; i < Vec; ++i) { acc[i] = red[threadIdx.x * Vec + i]; }
    for (int t = 1; t < int(blockDim.y); ++t) {
        const int base = (t * bx + int(threadIdx.x)) * Vec;
        #pragma unroll
        for (int i = 0; i < Vec; ++i) {
            acc[i] = IsMax ? fmaxf(acc[i], red[base + i]) : acc[i] + red[base + i];
        }
    }

    const int64_t pixels = rows * width;
    if (IsMax) {
        #pragma unroll
        for (int i = 0; i < Vec; ++i) {
            acc[i] = (acc[i] <= -1e29f) ? 0.0f : acc[i];
        }
    } else {
        const float scale = pixels > 0 ? 1.0f / static_cast<float>(pixels) : 0.0f;
        #pragma unroll
        for (int i = 0; i < Vec; ++i) { acc[i] *= scale; }
    }

    T* out = output + img * out_img_stride + c_begin + chunk * Vec;
    if (add_to) {
        float prev[Vec];
        load_channels<T, Vec>(out, prev);
        #pragma unroll
        for (int i = 0; i < Vec; ++i) { acc[i] += prev[i]; }
    }
    store_channels<T, Vec>(out, acc);
}

// ============================================================
// Host-side launchers
// ============================================================

inline PoolGeom pooling_geom(const PoolingAttributes& attrs,
                             const TensorView& input,
                             const TensorView& output)
{
    const int64_t srank = PoolingAttributes::spatial_rank(input.rank());
    PoolGeom g{};
    g.N = input.shape(0);
    g.C = input.shape(1);
    g.ID = (srank == 3) ? input.shape(2) : 1;
    g.IH = input.shape(srank);
    g.IW = input.shape(srank + 1);
    g.OD = (srank == 3) ? output.shape(2) : 1;
    g.OH = output.shape(srank);
    g.OW = output.shape(srank + 1);
    g.KD = (srank == 3) ? attrs.kernel_shape[0] : 1;
    g.KH = attrs.kernel_shape[1];
    g.KW = attrs.kernel_shape[2];
    g.SD = (srank == 3) ? attrs.stride[0] : 1;
    g.SH = attrs.stride[1];
    g.SW = attrs.stride[2];
    g.DD = (srank == 3) ? attrs.dilation[0] : 1;
    g.DH = attrs.dilation[1];
    g.DW = attrs.dilation[2];
    g.PD = (srank == 3) ? attrs.padding[0] : 0;
    g.PH = attrs.padding[1];
    g.PW = attrs.padding[2];
    g.type = attrs.type;
    g.exclude_pad = attrs.exclude_pad;
    g.add_to = attrs.add_to;

    if (is_channel_last(input.layout())) {
        g.in_pix_stride  = input.shape(1);
        g.in_row_stride  = input.row_stride_elems();
        g.in_d_stride    = g.IH * g.in_row_stride;
        g.out_pix_stride = output.shape(1);
        g.out_row_stride = output.row_stride_elems();
        g.out_d_stride   = g.OH * g.out_row_stride;
    } else {
        g.in_row_stride  = input.row_stride_elems();
        g.in_d_stride    = g.IH * g.in_row_stride;
        g.in_ch_stride   = g.ID * g.in_d_stride;
        g.out_row_stride = output.row_stride_elems();
        g.out_d_stride   = g.OH * g.out_row_stride;
        g.out_ch_stride  = g.OD * g.out_d_stride;
    }
    return g;
}

template <typename T, int K, int SW_, int VW>
void pooling_first_window_launch(const PoolGeom& g, const T* in, T* out,
                                 cudaStream_t stream)
{
    const dim3 block(16, 16, 1);
    const dim3 grid(static_cast<unsigned>((g.OW + 16 * VW - 1) / (16 * VW)),
                    static_cast<unsigned>((g.OH + 15) / 16),
                    static_cast<unsigned>(g.N * g.C * g.OD));
    if (g.type == PoolingType::Max) {
        pooling_first_window_kernel<T, K, SW_, VW, true>
            <<<grid, block, 0, stream>>>(in, out, g);
    } else {
        pooling_first_window_kernel<T, K, SW_, VW, false>
            <<<grid, block, 0, stream>>>(in, out, g);
    }
}

/// Dispatch the (K, SW) menu the window kernel is instantiated for. Returns
/// false for a geometry outside the menu, which then takes the grid kernel.
template <typename T, int VW>
bool pooling_first_window_dispatch(const PoolGeom& g, const T* in, T* out,
                                   cudaStream_t stream)
{
    switch (g.SW * 8 + g.KH) {
    case 1 * 8 + 2: pooling_first_window_launch<T, 2, 1, VW>(g, in, out, stream); return true;
    case 1 * 8 + 3: pooling_first_window_launch<T, 3, 1, VW>(g, in, out, stream); return true;
    case 1 * 8 + 5: pooling_first_window_launch<T, 5, 1, VW>(g, in, out, stream); return true;
    case 2 * 8 + 2: pooling_first_window_launch<T, 2, 2, VW>(g, in, out, stream); return true;
    case 2 * 8 + 3: pooling_first_window_launch<T, 3, 2, VW>(g, in, out, stream); return true;
    case 2 * 8 + 5: pooling_first_window_launch<T, 5, 2, VW>(g, in, out, stream); return true;
    default: return false;
    }
}

template <typename T>
void pooling_channel_first_impl(
    const PoolingAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const PoolGeom g = pooling_geom(attrs, input, output);
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    const T* in = input.ptr<T>();
    T* out = output.ptr<T>();

    constexpr int VW = 4;
    const int64_t nc_od = g.N * g.C * g.OD;
    const bool square = g.KH == g.KW && (g.KH == 2 || g.KH == 3 || g.KH == 5);
    const bool window = square && g.DH == 1 && g.DW == 1 && (g.SW == 1 || g.SW == 2)
                        && g.OW >= 2 * VW && nc_od <= 65535 && g.OH <= 65535;

    if (window && pooling_first_window_dispatch<T, VW>(g, in, out, stream)) { return; }

    if (nc_od <= 65535 && g.OH <= 65535) {
        // Narrow planes take a narrower block: at 7 and 14 columns a 32-lane
        // row leaves most of the block parked on the bounds check.
        const int bx = g.OW >= 32 ? 32 : (g.OW >= 16 ? 16 : 8);
        const int cap = kCudaBlockSize / bx;
        const int rows = int((g.OH + 3) / 4) * 4;
        const int by = rows < cap ? rows : cap;
        const dim3 block(static_cast<unsigned>(bx), static_cast<unsigned>(by), 1);
        const dim3 grid(static_cast<unsigned>((g.OW + bx - 1) / bx),
                        static_cast<unsigned>((g.OH + by - 1) / by),
                        static_cast<unsigned>(nc_od));
        if (g.type == PoolingType::Max) {
            pooling_first_grid_kernel<T, true><<<grid, block, 0, stream>>>(in, out, g);
        } else {
            pooling_first_grid_kernel<T, false><<<grid, block, 0, stream>>>(in, out, g);
        }
        return;
    }

    const int64_t total_spatial = g.OD * g.OH * g.OW;
    const int block_size = static_cast<int>(std::min<int64_t>(total_spatial, kCudaBlockSize));
    const dim3 grid(1, static_cast<unsigned>(g.N * g.C), 1);
    pooling_first_flat_kernel<T><<<grid, block_size, 0, stream>>>(in, out, g);
}

template <typename T, int Vec>
void pooling_last_launch(const PoolGeom& g,
                         TensorView& output,
                         const TensorView& input,
                         int64_t c_begin,
                         int64_t c_count,
                         cudaStream_t stream)
{
    if (c_count <= 0) { return; }

    constexpr int kRows = 2;
    const int64_t chunks = (c_count + Vec - 1) / Vec;
    const int bx = int(std::min<int64_t>(chunks, kCudaWarpSize));
    const int by = kCudaBlockSize / bx;
    const dim3 block(static_cast<unsigned>(bx), static_cast<unsigned>(by), 1);
    const dim3 gxy(static_cast<unsigned>((chunks + bx - 1) / bx),
                   static_cast<unsigned>((g.OW + by - 1) / by));
    const bool grid_ok = g.OW <= 65535;

    const bool square = g.KH == g.KW && (g.KH == 2 || g.KH == 3 || g.KH == 5);
    const bool rows = square && g.SH == 1 && g.DH == 1 && g.DW == 1
                      && g.OH >= 2 * kRows;
    const int64_t row_blocks = rows ? (g.OH + kRows - 1) / kRows : g.OH;
    const int64_t gz = g.N * g.OD * row_blocks;

    if (grid_ok && gz <= 65535) {
        const dim3 grid(gxy.x, gxy.y, static_cast<unsigned>(gz));
        const T* in = input.ptr<T>();
        T* out = output.ptr<T>();
        if (rows) {
            if (g.type == PoolingType::Max) {
                switch (int(g.KH)) {
                case 2: pooling_last_rows_kernel<T, Vec, 2, kRows, true>
                            <<<grid, block, 0, stream>>>(in, out, g, c_begin, chunks); return;
                case 3: pooling_last_rows_kernel<T, Vec, 3, kRows, true>
                            <<<grid, block, 0, stream>>>(in, out, g, c_begin, chunks); return;
                default: pooling_last_rows_kernel<T, Vec, 5, kRows, true>
                            <<<grid, block, 0, stream>>>(in, out, g, c_begin, chunks); return;
                }
            }
            switch (int(g.KH)) {
            case 2: pooling_last_rows_kernel<T, Vec, 2, kRows, false>
                        <<<grid, block, 0, stream>>>(in, out, g, c_begin, chunks); return;
            case 3: pooling_last_rows_kernel<T, Vec, 3, kRows, false>
                        <<<grid, block, 0, stream>>>(in, out, g, c_begin, chunks); return;
            default: pooling_last_rows_kernel<T, Vec, 5, kRows, false>
                        <<<grid, block, 0, stream>>>(in, out, g, c_begin, chunks); return;
            }
        }
        if (g.type == PoolingType::Max) {
            pooling_last_flat_kernel<T, Vec, true>
                <<<grid, block, 0, stream>>>(in, out, g, c_begin, chunks);
        } else {
            pooling_last_flat_kernel<T, Vec, false>
                <<<grid, block, 0, stream>>>(in, out, g, c_begin, chunks);
        }
        return;
    }

    const int64_t tasks = g.N * g.OD * g.OH * g.OW * chunks;
    const int64_t blocks = (tasks + kCudaBlockSize - 1) / kCudaBlockSize;
    pooling_last_linear_kernel<T, Vec>
        <<<static_cast<unsigned>(blocks), kCudaBlockSize, 0, stream>>>(
            input.ptr<T>(), output.ptr<T>(), g, c_begin, chunks);
}

template <typename T>
void pooling_last_impl(
    const PoolingAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const PoolGeom g = pooling_geom(attrs, input, output);
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    constexpr int kVec = static_cast<int>(16 / sizeof(T));
    const bool vec_ok = g.C % kVec == 0
                        && g.in_pix_stride % kVec == 0 && g.in_row_stride % kVec == 0
                        && g.out_pix_stride % kVec == 0 && g.out_row_stride % kVec == 0;

    const int64_t full = vec_ok ? (g.C / kVec) * kVec : 0;
    pooling_last_launch<T, kVec>(g, output, input, 0, full, stream);
    pooling_last_launch<T, 1>(g, output, input, full, g.C - full, stream);
}

template <typename T, int Vec>
void pooling_global_last_launch(const PoolingAttributes& attrs,
                                TensorView& output,
                                const TensorView& input,
                                int64_t c_begin,
                                int64_t c_count,
                                const PoolGeom& g,
                                int64_t in_img_stride,
                                int64_t out_img_stride,
                                cudaStream_t stream)
{
    if (c_count <= 0) { return; }
    const int64_t chunks = (c_count + Vec - 1) / Vec;
    const int bx = int(std::min<int64_t>(chunks, kCudaWarpSize));
    const int by = kCudaBlockSize / bx;
    const dim3 block(static_cast<unsigned>(bx), static_cast<unsigned>(by), 1);
    const dim3 grid(static_cast<unsigned>((chunks + bx - 1) / bx),
                    static_cast<unsigned>(g.N), 1);

    const T* in = input.ptr<T>();
    T* out = output.ptr<T>();
    if (attrs.type == PoolingType::Max) {
        pooling_global_last_kernel<T, Vec, true><<<grid, block, 0, stream>>>(
            in, out, c_begin, chunks, g.ID * g.IH, g.IW,
            in_img_stride, g.in_row_stride, g.in_pix_stride,
            out_img_stride, g.out_pix_stride, attrs.add_to);
    } else {
        pooling_global_last_kernel<T, Vec, false><<<grid, block, 0, stream>>>(
            in, out, c_begin, chunks, g.ID * g.IH, g.IW,
            in_img_stride, g.in_row_stride, g.in_pix_stride,
            out_img_stride, g.out_pix_stride, attrs.add_to);
    }
}

template <typename T>
void pooling_global_impl(
    const PoolingAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const PoolGeom g = pooling_geom(attrs, input, output);
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    if (!is_channel_last(input.layout())) {
        const int64_t in_ch_stride  = g.ID * g.IH * g.in_row_stride;
        const int64_t out_ch_stride = g.out_row_stride;
        const int64_t NC = g.N * g.C;
        if (g.type == PoolingType::Max) {
            pooling_global_first_kernel<T, true>
                <<<static_cast<unsigned>(NC), kCudaBlockSize, 0, stream>>>(
                    input.ptr<T>(), output.ptr<T>(), g.ID * g.IH, g.IW,
                    g.in_row_stride, in_ch_stride, out_ch_stride, attrs.add_to);
        } else {
            pooling_global_first_kernel<T, false>
                <<<static_cast<unsigned>(NC), kCudaBlockSize, 0, stream>>>(
                    input.ptr<T>(), output.ptr<T>(), g.ID * g.IH, g.IW,
                    g.in_row_stride, in_ch_stride, out_ch_stride, attrs.add_to);
        }
        return;
    }

    const int64_t in_img_stride  = g.ID * g.IH * g.in_row_stride;
    const int64_t out_img_stride = g.out_row_stride;

    constexpr int kVec = static_cast<int>(16 / sizeof(T));
    const bool vec_ok = g.C % kVec == 0 && g.in_pix_stride % kVec == 0
                        && g.out_pix_stride % kVec == 0
                        && g.in_row_stride % kVec == 0
                        && in_img_stride % kVec == 0 && out_img_stride % kVec == 0;

    const int64_t full = vec_ok ? (g.C / kVec) * kVec : 0;
    pooling_global_last_launch<T, kVec>(attrs, output, input, 0, full, g,
                                        in_img_stride, out_img_stride, stream);
    pooling_global_last_launch<T, 1>(attrs, output, input, full, g.C - full, g,
                                     in_img_stride, out_img_stride, stream);
}

// ============================================================
// Entry point
// ============================================================

template <typename T>
void pooling_dispatch(const PoolingAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx)
{
    if (attrs.global) {
        pooling_global_impl<T>(attrs, output, inputs, ctx);
    } else if (is_channel_last(inputs[0].layout())) {
        pooling_last_impl<T>(attrs, output, inputs, ctx);
    } else {
        pooling_channel_first_impl<T>(attrs, output, inputs, ctx);
    }
}

Status pooling_cuda(
    const PoolingAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx,
    void* /*workspace*/)
{
    switch (inputs[0].data_type()) {
    case DataType::f32:
        pooling_dispatch<float>(attrs, output, inputs, ctx);
        return Status::ok;
    case DataType::f16:
        pooling_dispatch<__half>(attrs, output, inputs, ctx);
        return Status::ok;
    default:
        return Status::unsupported;
    }
}

}  // namespace nnops::backend::cuda
