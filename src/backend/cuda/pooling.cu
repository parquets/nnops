/// @file pooling.cu
/// @brief CUDA implementation of 2D/3D spatial pooling (NCHW/NCDHW layout).
///
/// Pooling types: Max, Average, AverageExcludePad, Lp.
/// Supports 2D (rank=4, NCHW) and 3D (rank=5, NCDHW) via unified code path.
///
/// Algorithm:
///   One CUDA thread per output spatial position. Grid covers (N * C) blocks,
///   each block with enough threads for the spatial output elements.
///   Each thread iterates the kernel window to compute the result.
///
/// Supports f32 and f16. Workspace: not used.

#include "cuda_common.cuh"
#include "nnops/ops/pooling.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops::backend::cuda {

// ============================================================
// Pooling CUDA kernel — 2D/3D unified, one thread per output position
// ============================================================

template <typename T>
__global__ void pooling_kernel(
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
    int64_t p_norm,
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

        // ---- Average / AverageExcludePad ----
        case PoolingType::Average:
        case PoolingType::AverageExcludePad: {
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
            if (pool_type == PoolingType::AverageExcludePad) {
                const int64_t valid = K_total - pad_count;
                result = (valid > 0) ? sum / static_cast<float>(valid) : 0.0f;
            } else {
                result = sum / static_cast<float>(K_total);
            }
            break;
        }

        // ---- Lp Pooling ----
        case PoolingType::Lp: {
            float sum = 0.0f;
            const float fp = static_cast<float>(p_norm);
            for (int64_t kd = 0; kd < KD; ++kd) {
                const int64_t id = od * SD + kd * DD - PD;
                if (id < 0 || id >= ID) continue;
                for (int64_t kh = 0; kh < KH; ++kh) {
                    const int64_t ih = oh * SH + kh * DH - PH;
                    if (ih < 0 || ih >= IH) continue;
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        const int64_t iw = ow * SW + kw * DW - PW;
                        if (iw < 0 || iw >= IW) continue;
                        sum += powf(fabsf(s_load(&in_ch[id * in_d_stride + ih * in_row_stride + iw])), fp);
                    }
                }
            }
            result = powf(sum, 1.0f / fp);
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
// Host-side launcher
// ============================================================

template <typename T>
void pooling_cuda_impl(
    const PoolingAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    const int64_t srank = PoolingAttributes::spatial_rank(rank);  // 2 or 3

    // Common dimensions
    const int64_t N = input.shape(0);
    const int64_t C = input.shape(1);

    // Spatial input dimensions
    const int64_t ID = (srank == 3) ? input.shape(2) : 1;
    const int64_t IH = input.shape(srank);
    const int64_t IW = input.shape(srank + 1);

    // Spatial output dimensions
    const int64_t OD = (srank == 3) ? output.shape(2) : 1;
    const int64_t OH = output.shape(srank);
    const int64_t OW = output.shape(srank + 1);

    // Kernel / stride / padding / dilation
    const int64_t KD = (srank == 3) ? attrs.kernel_shape[0] : 1;
    const int64_t KH = attrs.kernel_shape[1];
    const int64_t KW = attrs.kernel_shape[2];

    const int64_t SD = (srank == 3) ? attrs.stride[0] : 1;
    const int64_t SH = attrs.stride[1];
    const int64_t SW = attrs.stride[2];

    const int64_t DD_ = (srank == 3) ? attrs.dilation[0] : 1;
    const int64_t DH = attrs.dilation[1];
    const int64_t DW = attrs.dilation[2];

    const int64_t PD = (srank == 3) ? attrs.padding[0] : 0;
    const int64_t PH = attrs.padding[1];
    const int64_t PW = attrs.padding[2];

    // Strides from pitch
    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t in_d_stride    = IH * in_row_stride;
    const int64_t in_ch_stride   = ID * in_d_stride;
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_d_stride   = OH * out_row_stride;
    const int64_t out_ch_stride  = OD * out_d_stride;

    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    const int64_t total_spatial = OD * OH * OW;
    const int block_size = std::min(static_cast<int>(total_spatial), kCudaBlockSize);
    const int64_t NC = N * C;
    const dim3 grid(1, static_cast<unsigned int>(NC), 1);

    pooling_kernel<<<grid, block_size, 0, stream>>>(
        input.ptr<T>(), output.ptr<T>(),
        N, C,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        DD_, DH, DW,
        PD, PH, PW,
        in_ch_stride, in_d_stride, in_row_stride,
        out_ch_stride, out_d_stride, out_row_stride,
        attrs.type, attrs.p_norm, attrs.add_to);
}

// ============================================================
// Entry point
// ============================================================

void pooling_cuda(
    const PoolingAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx,
    void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        pooling_cuda_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        pooling_cuda_impl<__half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"pooling_cuda: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cuda
