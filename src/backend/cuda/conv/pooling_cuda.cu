/// @file pooling_cuda.cu
/// @brief CUDA implementation of 2D/3D spatial pooling — Max and Average — on a
///        planar (NCHW/NCDHW) or a channel-last (NHWC/NDHWC) tensor.
///
/// Planar keeps the lane on output width, because that is the contiguous axis
/// there: one thread per output position of one (n, c) plane, a block per plane.
/// Channel-last puts the lane on the channel instead, for the same reason — a
/// thread owns one 16 B vector of a pixel (four f32 or eight f16) and the window
/// walk becomes one 128-bit load a tap. A channel count that is not a vector
/// multiple is covered by a second, scalar launch over the remainder rather than
/// by masking inside the vector kernel, and an unaligned pitch or pixel stride
/// falls back to the scalar script entirely.
///
/// Global pooling is a separate pair: a planar plane reduces across a block, a
/// channel-last image reduces across pixels with one thread per channel vector,
/// which keeps the loads coalesced because adjacent threads hold adjacent
/// channels of the same pixel.
///
/// Supports f32 and f16. Workspace: not used.

#include "../cuda_common.cuh"
#include "nnops/ops/pooling.hpp"
#include "nnops/detail/assert.hpp"

#include <algorithm>

namespace nnops::backend::cuda {

// ============================================================
// Planar — one thread per output position, a block per (n, c) plane
// ============================================================

template <typename T>
__global__ void pooling_planar_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    // Spatial dimensions
    int64_t N, int64_t C,
    int64_t ID, int64_t IH, int64_t IW,
    int64_t OD, int64_t OH, int64_t OW,
    // Kernel / stride / padding / dilation
    int64_t KD, int64_t KH, int64_t KW,
    int64_t SD, int64_t SH, int64_t SW,
    int64_t DD, int64_t DH, int64_t DW,
    int64_t PD, int64_t PH, int64_t PW,
    // Strides (in elements)
    int64_t in_ch_stride,
    int64_t in_d_stride,
    int64_t in_row_stride,
    int64_t out_ch_stride,
    int64_t out_d_stride,
    int64_t out_row_stride,
    // Attributes
    PoolingType pool_type,
    bool exclude_pad,
    bool add_to)
{
    // Each block handles one (n, c) pair; threads map to output spatial positions
    const int n = blockIdx.y / C;
    const int c = blockIdx.y % C;

    // Thread maps to spatial output within this (n, c)
    const int tid = threadIdx.x;

    // Total spatial output positions in this block's (n, c)
    const int64_t total_spatial = OD * OH * OW;

    // Stride over spatial positions with blockDim.x threads
    for (int64_t sp = tid; sp < total_spatial; sp += blockDim.x) {
        // Decode (od, oh, ow) from flat spatial index
        const int64_t od = sp / (OH * OW);
        const int64_t oh = (sp / OW) % OH;
        const int64_t ow = sp % OW;

        // Input and output channel base pointers
        const T* in_ch  = input  + n * C * in_ch_stride  + c * in_ch_stride;
        T*       out_ch = output + n * C * out_ch_stride + c * out_ch_stride;

        float result = 0.0f;

        switch (pool_type) {
        // ---- MaxPooling ----
        case PoolingType::Max: {
            float max_val = -1e30f;
            bool any = false;
            for (int64_t kd = 0; kd < KD; ++kd) {
                const int64_t id = od * SD + kd * DD - PD;
                if (id < 0 || id >= ID) continue;
                for (int64_t kh = 0; kh < KH; ++kh) {
                    const int64_t ih = oh * SH + kh * DH - PH;
                    if (ih < 0 || ih >= IH) continue;
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        const int64_t iw = ow * SW + kw * DW - PW;
                        if (iw < 0 || iw >= IW) continue;
                        float val = s_load(&in_ch[id * in_d_stride + ih * in_row_stride + iw]);
                        if (val > max_val) max_val = val;
                        any = true;
                    }
                }
            }
            result = any ? max_val : 0.0f;
            break;
        }

        // ---- Average ----
        case PoolingType::Average: {
            float sum = 0.0f;
            int64_t pad_count = 0;
            const int64_t K_total = KD * KH * KW;
            for (int64_t kd = 0; kd < KD; ++kd) {
                const int64_t id = od * SD + kd * DD - PD;
                for (int64_t kh = 0; kh < KH; ++kh) {
                    const int64_t ih = oh * SH + kh * DH - PH;
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        const int64_t iw = ow * SW + kw * DW - PW;
                        if (id < 0 || id >= ID || ih < 0 || ih >= IH || iw < 0 || iw >= IW) {
                            ++pad_count;
                            continue;
                        }
                        sum += s_load(&in_ch[id * in_d_stride + ih * in_row_stride + iw]);
                    }
                }
            }
            if (exclude_pad) {
                const int64_t valid = K_total - pad_count;
                result = (valid > 0) ? sum / static_cast<float>(valid) : 0.0f;
            } else {
                result = sum / static_cast<float>(K_total);
            }
            break;
        }
        }

        // Write back
        const int64_t out_idx = od * out_d_stride + oh * out_row_stride + ow;
        if (add_to) {
            s_store(&out_ch[out_idx], s_load(&out_ch[out_idx]) + result);
        } else {
            s_store(&out_ch[out_idx], result);
        }
    }
}

// ============================================================
// Channel-last — the lane spans channels
// ============================================================

template <typename T, int Vec>
__global__ void pooling_last_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    // The channel range this launch covers, and the vector width it walks it in
    int64_t c_begin,
    int64_t chunks,
    // Spatial dimensions
    int64_t N,
    int64_t ID, int64_t IH, int64_t IW,
    int64_t OD, int64_t OH, int64_t OW,
    // Kernel / stride / padding / dilation
    int64_t KD, int64_t KH, int64_t KW,
    int64_t SD, int64_t SH, int64_t SW,
    int64_t DD, int64_t DH, int64_t DW,
    int64_t PD, int64_t PH, int64_t PW,
    // Strides (in elements): a pixel is C channels wide
    int64_t in_d_stride,
    int64_t in_row_stride,
    int64_t in_pix_stride,
    int64_t out_d_stride,
    int64_t out_row_stride,
    int64_t out_pix_stride,
    // Attributes
    PoolingType pool_type,
    bool exclude_pad,
    bool add_to)
{
    const int64_t pix_per_image = OD * OH * OW;
    const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= N * pix_per_image * chunks) { return; }

    // Adjacent threads take adjacent channel vectors of the same pixel, which is
    // what keeps the loads coalesced.
    const int64_t chunk = t % chunks;
    const int64_t pix   = t / chunks;
    const int64_t c0    = c_begin + chunk * Vec;

    const int64_t img = pix / pix_per_image;
    const int64_t sp  = pix % pix_per_image;
    const int64_t od = sp / (OH * OW);
    const int64_t oh = (sp / OW) % OH;
    const int64_t ow = sp % OW;

    const T* in_img  = input  + img * ID * in_d_stride  + c0;
    T*       out_img = output + img * OD * out_d_stride + c0;

    const bool is_max = pool_type == PoolingType::Max;
    float acc[Vec];
    #pragma unroll
    for (int i = 0; i < Vec; ++i) { acc[i] = is_max ? -1e30f : 0.0f; }

    int64_t valid = 0;
    for (int64_t kd = 0; kd < KD; ++kd) {
        const int64_t id = od * SD + kd * DD - PD;
        if (id < 0 || id >= ID) continue;
        for (int64_t kh = 0; kh < KH; ++kh) {
            const int64_t ih = oh * SH + kh * DH - PH;
            if (ih < 0 || ih >= IH) continue;
            const T* row = in_img + id * in_d_stride + ih * in_row_stride;
            for (int64_t kw = 0; kw < KW; ++kw) {
                const int64_t iw = ow * SW + kw * DW - PW;
                if (iw < 0 || iw >= IW) continue;
                float v[Vec];
                load_channels<T, Vec>(row + iw * in_pix_stride, v);
                #pragma unroll
                for (int i = 0; i < Vec; ++i) {
                    acc[i] = is_max ? fmaxf(acc[i], v[i]) : acc[i] + v[i];
                }
                ++valid;
            }
        }
    }

    if (is_max) {
        if (valid == 0) {
            #pragma unroll
            for (int i = 0; i < Vec; ++i) { acc[i] = 0.0f; }
        }
    } else {
        const float denom = static_cast<float>(exclude_pad ? valid : KD * KH * KW);
        const float scale = denom > 0.0f ? 1.0f / denom : 0.0f;
        #pragma unroll
        for (int i = 0; i < Vec; ++i) { acc[i] *= scale; }
    }

    T* out = out_img + od * out_d_stride + oh * out_row_stride + ow * out_pix_stride;
    if (add_to) {
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

/// A planar plane is walked per row rather than as one run: a padded pitch puts
/// dead bytes between rows, and reading through them would fold them into the
/// reduction.
template <typename T>
__global__ void pooling_global_planar_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t rows,           // ID * IH
    int64_t width,          // IW
    int64_t row_stride,
    int64_t in_ch_stride,
    int64_t out_ch_stride,
    PoolingType pool_type,
    bool add_to)
{
    __shared__ float smem[kCudaBlockSize / kCudaWarpSize];

    const int64_t nc = blockIdx.x;
    const T* in = input + nc * in_ch_stride;

    const int64_t plane = rows * width;
    const bool is_max = pool_type == PoolingType::Max;
    float acc = is_max ? -1e30f : 0.0f;
    for (int64_t i = threadIdx.x; i < plane; i += blockDim.x) {
        const int64_t r = i / width;
        const int64_t w = i - r * width;
        const float v = s_load(in + r * row_stride + w);
        acc = is_max ? fmaxf(acc, v) : acc + v;
    }
    acc = is_max ? block_reduce_max(acc, smem) : block_reduce_sum(acc, smem);

    if (threadIdx.x == 0) {
        float result = acc;
        if (!is_max) {
            result = plane > 0 ? result / static_cast<float>(plane) : 0.0f;
        }
        T* out = output + nc * out_ch_stride;
        s_store(out, add_to ? s_load(out) + result : result);
    }
}

/// Per row for the same reason as the planar twin: a padded pitch leaves dead
/// bytes after the last pixel of a row.
template <typename T, int Vec>
__global__ void pooling_global_last_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t c_begin,
    int64_t chunks,
    int64_t images,
    int64_t rows,           // ID * IH
    int64_t width,          // IW
    int64_t in_img_stride,
    int64_t in_row_stride,
    int64_t in_pix_stride,
    int64_t out_img_stride,
    int64_t out_pix_stride,
    PoolingType pool_type,
    bool add_to)
{
    const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= images * chunks) { return; }

    const int64_t chunk = t % chunks;
    const int64_t img   = t / chunks;
    const int64_t c0    = c_begin + chunk * Vec;

    const T* in  = input  + img * in_img_stride  + c0;
    T*       out = output + img * out_img_stride + c0;

    const int64_t pixels = rows * width;
    const bool is_max = pool_type == PoolingType::Max;
    float acc[Vec];
    #pragma unroll
    for (int i = 0; i < Vec; ++i) { acc[i] = is_max ? -1e30f : 0.0f; }

    for (int64_t p = 0; p < pixels; ++p) {
        const int64_t r = p / width;
        const int64_t w = p - r * width;
        float v[Vec];
        load_channels<T, Vec>(in + r * in_row_stride + w * in_pix_stride, v);
        #pragma unroll
        for (int i = 0; i < Vec; ++i) {
            acc[i] = is_max ? fmaxf(acc[i], v[i]) : acc[i] + v[i];
        }
    }

    if (!is_max) {
        const float scale = pixels > 0 ? 1.0f / static_cast<float>(pixels) : 0.0f;
        #pragma unroll
        for (int i = 0; i < Vec; ++i) { acc[i] *= scale; }
    }
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

/// The dimensions both routes spell out, planar and channel-last alike.
struct PoolingDims {
    int64_t N, C;
    int64_t ID, IH, IW;
    int64_t OD, OH, OW;
    int64_t KD, KH, KW;
    int64_t SD, SH, SW;
    int64_t DD, DH, DW;
    int64_t PD, PH, PW;
};

inline PoolingDims pooling_dims(const PoolingAttributes& attrs,
                                const TensorView& input,
                                const TensorView& output)
{
    const int64_t srank = PoolingAttributes::spatial_rank(input.rank());
    PoolingDims d{};
    d.N = input.shape(0);
    d.C = input.shape(1);
    d.ID = (srank == 3) ? input.shape(2) : 1;
    d.IH = input.shape(srank);
    d.IW = input.shape(srank + 1);
    d.OD = (srank == 3) ? output.shape(2) : 1;
    d.OH = output.shape(srank);
    d.OW = output.shape(srank + 1);
    d.KD = (srank == 3) ? attrs.kernel_shape[0] : 1;
    d.KH = attrs.kernel_shape[1];
    d.KW = attrs.kernel_shape[2];
    d.SD = (srank == 3) ? attrs.stride[0] : 1;
    d.SH = attrs.stride[1];
    d.SW = attrs.stride[2];
    d.DD = (srank == 3) ? attrs.dilation[0] : 1;
    d.DH = attrs.dilation[1];
    d.DW = attrs.dilation[2];
    d.PD = (srank == 3) ? attrs.padding[0] : 0;
    d.PH = attrs.padding[1];
    d.PW = attrs.padding[2];
    return d;
}

template <typename T>
void pooling_planar_impl(
    const PoolingAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const PoolingDims d = pooling_dims(attrs, input, output);

    // Strides from pitch
    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t in_d_stride    = d.IH * in_row_stride;
    const int64_t in_ch_stride   = d.ID * in_d_stride;
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_d_stride   = d.OH * out_row_stride;
    const int64_t out_ch_stride  = d.OD * out_d_stride;

    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    const int64_t total_spatial = d.OD * d.OH * d.OW;
    const int block_size = std::min(static_cast<int>(total_spatial), kCudaBlockSize);
    const int64_t NC = d.N * d.C;
    const dim3 grid(1, static_cast<unsigned int>(NC), 1);

    pooling_planar_kernel<<<grid, block_size, 0, stream>>>(
        input.ptr<T>(), output.ptr<T>(),
        d.N, d.C,
        d.ID, d.IH, d.IW,
        d.OD, d.OH, d.OW,
        d.KD, d.KH, d.KW,
        d.SD, d.SH, d.SW,
        d.DD, d.DH, d.DW,
        d.PD, d.PH, d.PW,
        in_ch_stride, in_d_stride, in_row_stride,
        out_ch_stride, out_d_stride, out_row_stride,
        attrs.type, attrs.exclude_pad, attrs.add_to);
}

template <typename T, int Vec>
void pooling_last_launch(
    const PoolingDims& d,
    const PoolingAttributes& attrs,
    TensorView& output,
    const TensorView& input,
    int64_t c_begin,
    int64_t c_count,
    int64_t in_d_stride,
    int64_t in_row_stride,
    int64_t in_pix_stride,
    int64_t out_d_stride,
    int64_t out_row_stride,
    int64_t out_pix_stride,
    cudaStream_t stream)
{
    if (c_count <= 0) { return; }
    const int64_t chunks = (c_count + Vec - 1) / Vec;
    const int64_t tasks = d.N * d.OD * d.OH * d.OW * chunks;
    const int block_size = kCudaBlockSize;
    const int64_t blocks = (tasks + block_size - 1) / block_size;

    pooling_last_kernel<T, Vec><<<static_cast<unsigned int>(blocks), block_size, 0, stream>>>(
        input.ptr<T>(), output.ptr<T>(),
        c_begin, chunks,
        d.N,
        d.ID, d.IH, d.IW,
        d.OD, d.OH, d.OW,
        d.KD, d.KH, d.KW,
        d.SD, d.SH, d.SW,
        d.DD, d.DH, d.DW,
        d.PD, d.PH, d.PW,
        in_d_stride, in_row_stride, in_pix_stride,
        out_d_stride, out_row_stride, out_pix_stride,
        attrs.type, attrs.exclude_pad, attrs.add_to);
}

template <typename T>
void pooling_last_impl(
    const PoolingAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const PoolingDims d = pooling_dims(attrs, input, output);

    // Channel-last: a pixel is C channels wide and a "row" is IW of them.
    const int64_t in_pix_stride  = input.shape(1);
    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t in_d_stride    = d.IH * in_row_stride;
    const int64_t out_pix_stride = output.shape(1);
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_d_stride   = d.OH * out_row_stride;

    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    constexpr int kVec = static_cast<int>(16 / sizeof(T));
    const bool vec_ok = d.C % kVec == 0
                        && in_pix_stride % kVec == 0 && in_row_stride % kVec == 0
                        && out_pix_stride % kVec == 0 && out_row_stride % kVec == 0;

    const int64_t full = vec_ok ? (d.C / kVec) * kVec : 0;
    pooling_last_launch<T, kVec>(d, attrs, output, input, 0, full,
                                 in_d_stride, in_row_stride, in_pix_stride,
                                 out_d_stride, out_row_stride, out_pix_stride,
                                 stream);
    pooling_last_launch<T, 1>(d, attrs, output, input, full, d.C - full,
                              in_d_stride, in_row_stride, in_pix_stride,
                              out_d_stride, out_row_stride, out_pix_stride,
                              stream);
}

template <typename T, int Vec>
void pooling_global_last_launch(
    const PoolingAttributes& attrs,
    TensorView& output,
    const TensorView& input,
    int64_t c_begin,
    int64_t c_count,
    const PoolingDims& d,
    int64_t in_img_stride,
    int64_t in_row_stride,
    int64_t in_pix_stride,
    int64_t out_img_stride,
    int64_t out_pix_stride,
    cudaStream_t stream)
{
    if (c_count <= 0) { return; }
    const int64_t chunks = (c_count + Vec - 1) / Vec;
    const int64_t tasks = d.N * chunks;
    const int64_t blocks = (tasks + kCudaBlockSize - 1) / kCudaBlockSize;

    pooling_global_last_kernel<T, Vec><<<static_cast<unsigned int>(blocks), kCudaBlockSize, 0, stream>>>(
        input.ptr<T>(), output.ptr<T>(), c_begin, chunks, d.N,
        d.ID * d.IH, d.IW,
        in_img_stride, in_row_stride, in_pix_stride, out_img_stride, out_pix_stride,
        attrs.type, attrs.add_to);
}

template <typename T>
void pooling_global_impl(
    const PoolingAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const PoolingDims d = pooling_dims(attrs, input, output);
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    if (!is_channel_last(input.layout())) {
        const int64_t in_row_stride = input.row_stride_elems();
        const int64_t in_ch_stride  = d.ID * d.IH * in_row_stride;
        const int64_t out_ch_stride = output.row_stride_elems();
        const int64_t NC = d.N * d.C;
        pooling_global_planar_kernel<T><<<static_cast<unsigned int>(NC), kCudaBlockSize, 0, stream>>>(
            input.ptr<T>(), output.ptr<T>(), d.ID * d.IH, d.IW, in_row_stride,
            in_ch_stride, out_ch_stride, attrs.type, attrs.add_to);
        return;
    }

    const int64_t in_pix_stride  = input.shape(1);
    const int64_t out_pix_stride = output.shape(1);
    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t in_img_stride  = d.ID * d.IH * in_row_stride;
    const int64_t out_img_stride = output.row_stride_elems();

    constexpr int kVec = static_cast<int>(16 / sizeof(T));
    const bool vec_ok = d.C % kVec == 0 && in_pix_stride % kVec == 0
                        && out_pix_stride % kVec == 0
                        && in_row_stride % kVec == 0
                        && in_img_stride % kVec == 0 && out_img_stride % kVec == 0;

    const int64_t full = vec_ok ? (d.C / kVec) * kVec : 0;
    pooling_global_last_launch<T, kVec>(attrs, output, input, 0, full, d,
                                        in_img_stride, in_row_stride, in_pix_stride,
                                        out_img_stride, out_pix_stride, stream);
    pooling_global_last_launch<T, 1>(attrs, output, input, full, d.C - full, d,
                                     in_img_stride, in_row_stride, in_pix_stride,
                                     out_img_stride, out_pix_stride, stream);
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
        pooling_planar_impl<T>(attrs, output, inputs, ctx);
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
