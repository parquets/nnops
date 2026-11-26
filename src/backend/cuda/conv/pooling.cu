/// @file pooling.cu
/// @brief CUDA implementation of 2D/3D spatial pooling — Max and Average — on a
///        NHWC/NDHWC tensor.
///
/// The lane spans channels: a thread owns one 16 B vector of a pixel (four f32
/// or eight f16) and the window walk becomes one 128-bit load a tap. Adjacent
/// threads take adjacent channel vectors of one pixel, which is what keeps the
/// loads coalesced. A channel count that is not a vector multiple is covered by
/// a second, scalar launch over the remainder, and an unaligned pitch or pixel
/// stride falls back to the scalar script entirely.
///
/// A square unit-stride kernel walks ROWS output rows per thread: consecutive
/// rows share K-1 of their input rows, so ROWS+K-1 loads feed ROWS output rows,
/// and each load contributes to up to K of them.
///
/// Global pooling is a separate pair. An NHWC image reduces across
/// pixels with one thread per channel vector, and the pixel range is split
/// across blockDim.y so one image can fill the machine: the lanes of a warp
/// still hold one pixel's channel vectors, which keeps the loads coalesced
/// while the rows are walked in parallel.
///
/// Pool1D is the width-only lift of this kernel: a rank-3 [N, L, C] signal is
/// the rank-4 [N, 1, L, C] activation with a one-row window, so pool1d_cuda at
/// the bottom only translates the attributes and lifts the views — no kernel of
/// its own, exactly as the CPU route does. The lifted view keeps the source
/// pitch, which for a channel-last rank-3 tensor is the whole spatial line
/// (L·C), so its rank-4 strides come out right.
///
/// Supports f32, f16 and bf16. Workspace: not used.

#include "../cuda_common.cuh"
#include "nnops/ops/pool1d.hpp"
#include "nnops/ops/pooling.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/shape_inference.hpp"

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
    // NHWC image/pixel/row strides
    int64_t in_d_stride, in_row_stride;
    int64_t out_d_stride, out_row_stride;
    int64_t in_pix_stride, out_pix_stride;
    PoolingType type;
    bool exclude_pad;
    bool add_to;
    /// Adaptive: the window comes from the output index, not the fixed fields.
    bool adaptive;
};

// ============================================================
// The vector kernel — the lane spans channels
// ============================================================

/// One output vector per thread, its position carried by the block axes: x is
/// the channel chunk, y the output width, z the (image, depth plane, row).
template <typename T, int Vec, bool IsMax>
__global__ void pooling_flat_kernel(const T* __restrict__ input, T* __restrict__ output,
                                         PoolGeom g, int64_t c_begin, int64_t chunks)
{
    const int64_t oh = int64_t(blockIdx.z) % g.OH;
    const int64_t p = int64_t(blockIdx.z) / g.OH;
    const int64_t od = p % g.OD;
    const int64_t img = p / g.OD;
    const int64_t chunk = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t ow = int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
    if (chunk >= chunks || ow >= g.OW) {
        return;
    }

    const T* in_img = input + img * (g.ID * g.in_d_stride) + c_begin + chunk * Vec;
    T* out_img = output + img * (g.OD * g.out_d_stride) + c_begin + chunk * Vec;

    float acc[Vec];
#pragma unroll
    for (int i = 0; i < Vec; ++i) {
        acc[i] = IsMax ? -1e30f : 0.0f;
    }

    int64_t valid = 0;
    for (int64_t kd = 0; kd < g.KD; ++kd) {
        const int64_t id = od * g.SD + kd * g.DD - g.PD;
        if (id < 0 || id >= g.ID) {
            continue;
        }
        const T* in_d = in_img + id * g.in_d_stride;
        for (int64_t kh = 0; kh < g.KH; ++kh) {
            const int64_t ih = oh * g.SH + kh * g.DH - g.PH;
            if (ih < 0 || ih >= g.IH) {
                continue;
            }
            const T* row = in_d + ih * g.in_row_stride;
            for (int64_t kw = 0; kw < g.KW; ++kw) {
                const int64_t iw = ow * g.SW + kw * g.DW - g.PW;
                if (iw < 0 || iw >= g.IW) {
                    continue;
                }
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
        for (int i = 0; i < Vec; ++i) {
            acc[i] *= scale;
        }
    }

    T* out = out_img + od * g.out_d_stride + oh * g.out_row_stride + ow * g.out_pix_stride;
    if (g.add_to) {
        float prev[Vec];
        load_channels<T, Vec>(out, prev);
#pragma unroll
        for (int i = 0; i < Vec; ++i) {
            acc[i] += prev[i];
        }
    }
    store_channels<T, Vec>(out, acc);
}

/// Adaptive pooling: output index `o` covers input `[floor(o*I/O),
/// ceil((o+1)*I/O))` -- PyTorch's rule, which is also the only one that
/// expresses an `I % O != 0` size. The lane still spans channels, and the two
/// loops let one launch cover any output grid.
__device__ inline void adaptive_span(int64_t o, int64_t in, int64_t out, int64_t& lo, int64_t& hi)
{
    lo = (o * in) / out;
    hi = ((o + 1) * in + out - 1) / out;
}

template <typename T, int Vec, bool IsMax>
__global__ void pooling_adaptive_kernel(const T* __restrict__ input, T* __restrict__ output,
                                        PoolGeom g, int64_t c_begin, int64_t chunks)
{
    const int64_t chunk = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (chunk >= chunks) {
        return;
    }
    const T* in_base = input + c_begin + chunk * Vec;
    T* out_base = output + c_begin + chunk * Vec;
    const int64_t z_end = g.N * g.OD * g.OH;

    for (int64_t z = int64_t(blockIdx.z); z < z_end; z += int64_t(gridDim.z)) {
        const int64_t oh = z % g.OH;
        const int64_t p = z / g.OH;
        const int64_t od = p % g.OD;
        const int64_t img = p / g.OD;

        int64_t d0, d1, h0, h1;
        adaptive_span(od, g.ID, g.OD, d0, d1);
        adaptive_span(oh, g.IH, g.OH, h0, h1);

        const T* in_img = in_base + img * (g.ID * g.in_d_stride);
        T* out_img = out_base + img * (g.OD * g.out_d_stride);

        for (int64_t ow = int64_t(blockIdx.y) * blockDim.y + threadIdx.y; ow < g.OW;
             ow += int64_t(gridDim.y) * blockDim.y) {
            int64_t w0, w1;
            adaptive_span(ow, g.IW, g.OW, w0, w1);

            float acc[Vec];
#pragma unroll
            for (int i = 0; i < Vec; ++i) {
                acc[i] = IsMax ? -std::numeric_limits<float>::infinity() : 0.0f;
            }
            for (int64_t id = d0; id < d1; ++id) {
                const T* in_d = in_img + id * g.in_d_stride;
                for (int64_t ih = h0; ih < h1; ++ih) {
                    const T* row = in_d + ih * g.in_row_stride;
                    for (int64_t iw = w0; iw < w1; ++iw) {
                        float v[Vec];
                        load_channels<T, Vec>(row + iw * g.in_pix_stride, v);
#pragma unroll
                        for (int i = 0; i < Vec; ++i) {
                            acc[i] = IsMax ? fmaxf(acc[i], v[i]) : acc[i] + v[i];
                        }
                    }
                }
            }
            if (!IsMax) {
                const int64_t vol = (d1 - d0) * (h1 - h0) * (w1 - w0);
                const float scale = vol > 0 ? 1.0f / static_cast<float>(vol) : 0.0f;
#pragma unroll
                for (int i = 0; i < Vec; ++i) {
                    acc[i] *= scale;
                }
            }

            T* out = out_img + od * g.out_d_stride + oh * g.out_row_stride + ow * g.out_pix_stride;
            if (g.add_to) {
                float prev[Vec];
                load_channels<T, Vec>(out, prev);
#pragma unroll
                for (int i = 0; i < Vec; ++i) {
                    acc[i] += prev[i];
                }
            }
            store_channels<T, Vec>(out, acc);
        }
    }
}

/// ROWS output rows per thread. Consecutive rows share K-1 of their input rows,
/// so a tap column costs ROWS+K-1 loads for ROWS outputs; the column validity is
/// row-independent, so the per-row divisor folds into one count multiplied by
/// the valid row and depth taps.
template <typename T, int Vec, int K, int ROWS, bool IsMax>
__global__ void pooling_rows_kernel(const T* __restrict__ input, T* __restrict__ output,
                                         PoolGeom g, int64_t c_begin, int64_t chunks)
{
    const int64_t row_blocks = (g.OH + ROWS - 1) / ROWS;
    const int64_t rb = int64_t(blockIdx.z) % row_blocks;
    const int64_t p = int64_t(blockIdx.z) / row_blocks;
    const int64_t od = p % g.OD;
    const int64_t img = p / g.OD;
    const int64_t chunk = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t ow = int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
    if (chunk >= chunks || ow >= g.OW) {
        return;
    }

    const T* in_img = input + img * (g.ID * g.in_d_stride) + c_begin + chunk * Vec;
    T* out_img = output + img * (g.OD * g.out_d_stride) + c_begin + chunk * Vec;

    const int64_t oh0 = rb * ROWS;
    const int64_t base = ow * g.SW - g.PW;

    float acc[ROWS][Vec];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int i = 0; i < Vec; ++i) {
            acc[r][i] = IsMax ? -1e30f : 0.0f;
        }
    }

    int colcnt = 0;
#pragma unroll
    for (int kw = 0; kw < K; ++kw) {
        const int64_t iw = base + kw;
        if (iw >= 0 && iw < g.IW) {
            ++colcnt;
        }
    }
    int rowcnt[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        int n = 0;
        if (oh0 + r < g.OH) {
#pragma unroll
            for (int kh = 0; kh < K; ++kh) {
                const int64_t iy = oh0 + r + kh - g.PH;
                if (iy >= 0 && iy < g.IH) {
                    ++n;
                }
            }
        }
        rowcnt[r] = n;
    }

    int64_t dcount = 0;
    for (int64_t kd = 0; kd < g.KD; ++kd) {
        const int64_t id = od * g.SD + kd * g.DD - g.PD;
        if (id < 0 || id >= g.ID) {
            continue;
        }
        ++dcount;
        const T* in_d = in_img + id * g.in_d_stride;
#pragma unroll
        for (int kw = 0; kw < K; ++kw) {
            const int64_t iw = base + kw;
            if (iw < 0 || iw >= g.IW) {
                continue;
            }
            const T* col = in_d + iw * g.in_pix_stride;
#pragma unroll
            for (int j = 0; j < ROWS + K - 1; ++j) {
                const int64_t iy = oh0 + j - g.PH;
                if (iy < 0 || iy >= g.IH) {
                    continue;
                }
                float v[Vec];
                load_channels<T, Vec>(col + iy * g.in_row_stride, v);
#pragma unroll
                for (int kh = 0; kh < K; ++kh) {
                    const int r = j - kh;
                    if (r < 0 || r >= ROWS || oh0 + r >= g.OH) {
                        continue;
                    }
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
        if (oh0 + r >= g.OH) {
            continue;
        }
        const int64_t taps = int64_t(colcnt) * rowcnt[r] * dcount;
        if (IsMax) {
            if (taps == 0) {
#pragma unroll
                for (int i = 0; i < Vec; ++i) {
                    acc[r][i] = 0.0f;
                }
            }
        } else {
            const int64_t denom = g.exclude_pad ? taps : g.KD * int64_t(K) * K;
            const float scale = denom > 0 ? 1.0f / static_cast<float>(denom) : 0.0f;
#pragma unroll
            for (int i = 0; i < Vec; ++i) {
                acc[r][i] *= scale;
            }
        }
        T* out =
            out_img + od * g.out_d_stride + (oh0 + r) * g.out_row_stride + ow * g.out_pix_stride;
        if (g.add_to) {
            float prev[Vec];
            load_channels<T, Vec>(out, prev);
#pragma unroll
            for (int i = 0; i < Vec; ++i) {
                acc[r][i] += prev[i];
            }
        }
        store_channels<T, Vec>(out, acc[r]);
    }
}

/// The flat script the older launchers used: adjacent threads take adjacent
/// channel vectors, the spatial index is decoded by division. Reached only by a
/// geometry whose block axes cannot hold its task count.
template <typename T, int Vec>
__global__ void pooling_linear_kernel(const T* __restrict__ input, T* __restrict__ output,
                                           PoolGeom g, int64_t c_begin, int64_t chunks)
{
    const int64_t pix_per_image = g.OD * g.OH * g.OW;
    const int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= g.N * pix_per_image * chunks) {
        return;
    }

    const int64_t chunk = t % chunks;
    const int64_t pix = t / chunks;
    const int64_t c0 = c_begin + chunk * Vec;

    const int64_t img = pix / pix_per_image;
    const int64_t sp = pix % pix_per_image;
    const int64_t od = sp / (g.OH * g.OW);
    const int64_t oh = (sp / g.OW) % g.OH;
    const int64_t ow = sp % g.OW;

    const T* in_img = input + img * (g.ID * g.in_d_stride) + c0;
    T* out_img = output + img * (g.OD * g.out_d_stride) + c0;

    const bool is_max = g.type == PoolingType::Max;
    float acc[Vec];
#pragma unroll
    for (int i = 0; i < Vec; ++i) {
        acc[i] = is_max ? -1e30f : 0.0f;
    }

    int64_t valid = 0;
    for (int64_t kd = 0; kd < g.KD; ++kd) {
        const int64_t id = od * g.SD + kd * g.DD - g.PD;
        if (id < 0 || id >= g.ID) {
            continue;
        }
        const T* in_d = in_img + id * g.in_d_stride;
        for (int64_t kh = 0; kh < g.KH; ++kh) {
            const int64_t ih = oh * g.SH + kh * g.DH - g.PH;
            if (ih < 0 || ih >= g.IH) {
                continue;
            }
            const T* row = in_d + ih * g.in_row_stride;
            for (int64_t kw = 0; kw < g.KW; ++kw) {
                const int64_t iw = ow * g.SW + kw * g.DW - g.PW;
                if (iw < 0 || iw >= g.IW) {
                    continue;
                }
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
        for (int i = 0; i < Vec; ++i) {
            acc[i] *= scale;
        }
    }

    T* out = out_img + od * g.out_d_stride + oh * g.out_row_stride + ow * g.out_pix_stride;
    if (g.add_to) {
        float prev[Vec];
        load_channels<T, Vec>(out, prev);
#pragma unroll
        for (int i = 0; i < Vec; ++i) {
            acc[i] += prev[i];
        }
    }
    store_channels<T, Vec>(out, acc);
}

// ============================================================
// Global pooling — every output spatial axis is one
// ============================================================

/// One image per blockIdx.y, its pixel rows split across blockDim.y: a warp
/// still holds one pixel's channel vectors (so the loads stay coalesced), but
/// one image can fill several blocks' worth of lanes instead of being one
/// thread's serial walk.
template <typename T, int Vec, bool IsMax>
__global__ void
pooling_global_kernel(const T* __restrict__ input, T* __restrict__ output, int64_t c_begin,
                           int64_t chunks, int64_t rows, int64_t width, int64_t in_img_stride,
                           int64_t in_row_stride, int64_t in_pix_stride, int64_t out_img_stride,
                           int64_t out_pix_stride, bool add_to)
{
    const int64_t chunk = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t img = blockIdx.y;
    const bool active = chunk < chunks;

    float acc[Vec];
#pragma unroll
    for (int i = 0; i < Vec; ++i) {
        acc[i] = IsMax ? -1e30f : 0.0f;
    }

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
    for (int i = 0; i < Vec; ++i) {
        red[tid * Vec + i] = acc[i];
    }
    __syncthreads();

    if (threadIdx.y != 0 || !active) {
        return;
    }
    const int bx = int(blockDim.x);
#pragma unroll
    for (int i = 0; i < Vec; ++i) {
        acc[i] = red[threadIdx.x * Vec + i];
    }
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
        for (int i = 0; i < Vec; ++i) {
            acc[i] *= scale;
        }
    }

    T* out = output + img * out_img_stride + c_begin + chunk * Vec;
    if (add_to) {
        float prev[Vec];
        load_channels<T, Vec>(out, prev);
#pragma unroll
        for (int i = 0; i < Vec; ++i) {
            acc[i] += prev[i];
        }
    }
    store_channels<T, Vec>(out, acc);
}

// ============================================================
// Host-side launchers
// ============================================================

inline PoolGeom pooling_geom(const PoolingAttributes& attrs, const TensorView& input,
                             const TensorView& output)
{
    const int64_t rank = input.rank();
    const int64_t srank = PoolingAttributes::spatial_rank(rank);
    PoolGeom g{};
    g.N = input.shape(0);
    g.C = input.shape(rank - 1);
    g.ID = (srank == 3) ? input.shape(rank - 4) : 1;
    g.IH = input.shape(rank - 3);
    g.IW = input.shape(rank - 2);
    g.OD = (srank == 3) ? output.shape(rank - 4) : 1;
    g.OH = output.shape(rank - 3);
    g.OW = output.shape(rank - 2);
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
    g.adaptive = attrs.adaptive();

    g.in_pix_stride = input.shape(rank - 1);
    g.in_row_stride = input.row_stride_elems();
    g.in_d_stride = g.IH * g.in_row_stride;
    g.out_pix_stride = output.shape(rank - 1);
    g.out_row_stride = output.row_stride_elems();
    g.out_d_stride = g.OH * g.out_row_stride;
    return g;
}

template <typename T, int Vec>
void pooling_launch(const PoolGeom& g, TensorView& output, const TensorView& input,
                         int64_t c_begin, int64_t c_count, cudaStream_t stream)
{
    if (c_count <= 0) {
        return;
    }

    constexpr int kRows = 2;
    const int64_t chunks = (c_count + Vec - 1) / Vec;
    const int bx = int(std::min<int64_t>(chunks, kCudaWarpSize));
    const int by = kCudaBlockSize / bx;
    const dim3 block(static_cast<unsigned>(bx), static_cast<unsigned>(by), 1);
    const dim3 gxy(static_cast<unsigned>((chunks + bx - 1) / bx),
                   static_cast<unsigned>((g.OW + by - 1) / by));

    // Adaptive first: its window comes from the output index, so none of the
    // fixed-window geometry below (the row kernel's square window, the grid's
    // 65 535 limit) applies -- the kernel strides its own z and ow instead.
    if (g.adaptive) {
        const int64_t z_end = g.N * g.OD * g.OH;
        const dim3 agrid(gxy.x,
                         static_cast<unsigned>(std::min<int64_t>(gxy.y, 65535)),
                         static_cast<unsigned>(std::min<int64_t>(z_end, 65535)));
        if (g.type == PoolingType::Max) {
            pooling_adaptive_kernel<T, Vec, true><<<agrid, block, 0, stream>>>(
                input.ptr<T>(), output.ptr<T>(), g, c_begin, chunks);
        } else {
            pooling_adaptive_kernel<T, Vec, false><<<agrid, block, 0, stream>>>(
                input.ptr<T>(), output.ptr<T>(), g, c_begin, chunks);
        }
        return;
    }

    const bool grid_ok = g.OW <= 65535;

    const bool square = g.KH == g.KW && (g.KH == 2 || g.KH == 3 || g.KH == 5);
    const bool rows = square && g.SH == 1 && g.DH == 1 && g.DW == 1 && g.OH >= 2 * kRows;
    const int64_t row_blocks = rows ? (g.OH + kRows - 1) / kRows : g.OH;
    const int64_t gz = g.N * g.OD * row_blocks;

    if (grid_ok && gz <= 65535) {
        const dim3 grid(gxy.x, gxy.y, static_cast<unsigned>(gz));
        const T* in = input.ptr<T>();
        T* out = output.ptr<T>();
        if (rows) {
            if (g.type == PoolingType::Max) {
                switch (int(g.KH)) {
                case 2:
                    pooling_rows_kernel<T, Vec, 2, kRows, true>
                        <<<grid, block, 0, stream>>>(in, out, g, c_begin, chunks);
                    return;
                case 3:
                    pooling_rows_kernel<T, Vec, 3, kRows, true>
                        <<<grid, block, 0, stream>>>(in, out, g, c_begin, chunks);
                    return;
                default:
                    pooling_rows_kernel<T, Vec, 5, kRows, true>
                        <<<grid, block, 0, stream>>>(in, out, g, c_begin, chunks);
                    return;
                }
            }
            switch (int(g.KH)) {
            case 2:
                pooling_rows_kernel<T, Vec, 2, kRows, false>
                    <<<grid, block, 0, stream>>>(in, out, g, c_begin, chunks);
                return;
            case 3:
                pooling_rows_kernel<T, Vec, 3, kRows, false>
                    <<<grid, block, 0, stream>>>(in, out, g, c_begin, chunks);
                return;
            default:
                pooling_rows_kernel<T, Vec, 5, kRows, false>
                    <<<grid, block, 0, stream>>>(in, out, g, c_begin, chunks);
                return;
            }
        }
        if (g.type == PoolingType::Max) {
            pooling_flat_kernel<T, Vec, true>
                <<<grid, block, 0, stream>>>(in, out, g, c_begin, chunks);
        } else {
            pooling_flat_kernel<T, Vec, false>
                <<<grid, block, 0, stream>>>(in, out, g, c_begin, chunks);
        }
        return;
    }

    const int64_t tasks = g.N * g.OD * g.OH * g.OW * chunks;
    const int64_t blocks = (tasks + kCudaBlockSize - 1) / kCudaBlockSize;
    pooling_linear_kernel<T, Vec>
        <<<static_cast<unsigned>(blocks), kCudaBlockSize, 0, stream>>>(
            input.ptr<T>(), output.ptr<T>(), g, c_begin, chunks);
}

template <typename T>
void pooling_impl(const PoolingAttributes& attrs, TensorView& output,
                       std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const PoolGeom g = pooling_geom(attrs, input, output);
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    constexpr int kVec = static_cast<int>(16 / sizeof(T));
    const bool vec_ok = g.C % kVec == 0 && g.in_pix_stride % kVec == 0 &&
                        g.in_row_stride % kVec == 0 && g.out_pix_stride % kVec == 0 &&
                        g.out_row_stride % kVec == 0;

    const int64_t full = vec_ok ? (g.C / kVec) * kVec : 0;
    pooling_launch<T, kVec>(g, output, input, 0, full, stream);
    pooling_launch<T, 1>(g, output, input, full, g.C - full, stream);
}

template <typename T, int Vec>
void pooling_global_last_launch(const PoolingAttributes& attrs, TensorView& output,
                                const TensorView& input, int64_t c_begin, int64_t c_count,
                                const PoolGeom& g, int64_t in_img_stride, int64_t out_img_stride,
                                cudaStream_t stream)
{
    if (c_count <= 0) {
        return;
    }
    const int64_t chunks = (c_count + Vec - 1) / Vec;
    const int bx = int(std::min<int64_t>(chunks, kCudaWarpSize));
    const int by = kCudaBlockSize / bx;
    const dim3 block(static_cast<unsigned>(bx), static_cast<unsigned>(by), 1);
    const dim3 grid(static_cast<unsigned>((chunks + bx - 1) / bx), static_cast<unsigned>(g.N), 1);

    const T* in = input.ptr<T>();
    T* out = output.ptr<T>();
    if (attrs.type == PoolingType::Max) {
        pooling_global_kernel<T, Vec, true><<<grid, block, 0, stream>>>(
            in, out, c_begin, chunks, g.ID * g.IH, g.IW, in_img_stride, g.in_row_stride,
            g.in_pix_stride, out_img_stride, g.out_pix_stride, attrs.add_to);
    } else {
        pooling_global_kernel<T, Vec, false><<<grid, block, 0, stream>>>(
            in, out, c_begin, chunks, g.ID * g.IH, g.IW, in_img_stride, g.in_row_stride,
            g.in_pix_stride, out_img_stride, g.out_pix_stride, attrs.add_to);
    }
}

template <typename T>
void pooling_global_impl(const PoolingAttributes& attrs, TensorView& output,
                         std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const PoolGeom g = pooling_geom(attrs, input, output);
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    const int64_t in_img_stride = g.ID * g.IH * g.in_row_stride;
    const int64_t out_img_stride = g.out_row_stride;

    constexpr int kVec = static_cast<int>(16 / sizeof(T));
    const bool vec_ok = g.C % kVec == 0 && g.in_pix_stride % kVec == 0 &&
                        g.out_pix_stride % kVec == 0 && g.in_row_stride % kVec == 0 &&
                        in_img_stride % kVec == 0 && out_img_stride % kVec == 0;

    const int64_t full = vec_ok ? (g.C / kVec) * kVec : 0;
    pooling_global_last_launch<T, kVec>(attrs, output, input, 0, full, g, in_img_stride,
                                        out_img_stride, stream);
    pooling_global_last_launch<T, 1>(attrs, output, input, full, g.C - full, g, in_img_stride,
                                     out_img_stride, stream);
}

// ============================================================
// Entry point
// ============================================================

template <typename T>
void pooling_dispatch(const PoolingAttributes& attrs, TensorView& output,
                      std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    if (attrs.global) {
        pooling_global_impl<T>(attrs, output, inputs, ctx);
    } else {
        pooling_impl<T>(attrs, output, inputs, ctx);
    }
}

Status pooling_cuda(const PoolingAttributes& attrs, TensorView& output,
                    std::span<const TensorView> inputs, const ComputeContext& ctx,
                    void* /*workspace*/)
{
    switch (inputs[0].data_type()) {
    case DataType::f32:
        pooling_dispatch<float>(attrs, output, inputs, ctx);
        return Status::ok;
    case DataType::f16:
        pooling_dispatch<__half>(attrs, output, inputs, ctx);
        return Status::ok;
    case DataType::bf16:
        pooling_dispatch<__nv_bfloat16>(attrs, output, inputs, ctx);
        return Status::ok;
    default:
        return Status::unsupported;
    }
}

namespace {

/// [N, L, C] as [N, 1, L, C] — the same view, re-described.
TensorView lift_wide(const TensorView& v)
{
    const int64_t s[4] = {v.shape(0), 1, v.shape(1), v.shape(2)};
    return TensorView(std::span<const int64_t>(s, 4), v.data_type(),
                      const_cast<void*>(v.ptr<void>()), v.pitch(), v.layout(), v.quant_params());
}

} // anonymous namespace

/// Pool1D: a rank-3 signal does not fit a kernel that takes its spatial rank
/// from the activation, so [N, L, C] lifts to [N, 1, L, C] with a {1, K} window
/// and the translate is a slot shift. auto_pad is resolved here because the
/// kernel only sees an explicit pad.
Status pool1d_cuda(const Pool1DAttributes& attrs, TensorView& output,
                   std::span<const TensorView> inputs, const ComputeContext& ctx, void* workspace)
{
    const TensorView& input = inputs[0];
    PoolingAttributes a{};
    a.type = attrs.type == Pool1DType::Max ? PoolingType::Max : PoolingType::Average;
    a.kernel_shape = {1, 1, attrs.kernel_shape};
    a.stride = {1, 1, attrs.stride};
    a.padding = {0, 0, attrs.padding};
    a.dilation = {1, 1, attrs.dilation};
    a.exclude_pad = attrs.exclude_pad;
    a.add_to = attrs.add_to;
    a.global = attrs.global;
    a.output_size = {0, 1, attrs.output_size};

    if (!a.global && !a.adaptive() && attrs.auto_pad != Pool1DAttributes::AutoPad::NOTSET) {
        a.padding[2] = resolve_leading_padding(
            input.shape(2), attrs.kernel_shape, attrs.stride, attrs.dilation, attrs.padding,
            static_cast<int>(attrs.auto_pad));
    }
    a.auto_pad = PoolingAttributes::AutoPad::NOTSET;

    TensorView out4 = lift_wide(output);
    TensorView in4 = lift_wide(input);
    const TensorView ins[1] = {in4};
    return pooling_cuda(a, out4, std::span<const TensorView>(ins, 1), ctx, workspace);
}

} // namespace nnops::backend::cuda
