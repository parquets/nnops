/// @file depthwise_conv.cu
/// @brief CUDA implementation of 2D/3D depthwise convolution (NCHW/NCDHW layout).
///
/// Spatial rank is auto-detected from input tensor rank (4→2D, 5→3D).
///
/// Algorithm:
///   Grid: (N * C) blocks. Each block handles one (sample, channel) pair.
///   Threads within a block stride over output spatial positions.
///   Each thread computes the full kernel convolution for one output position.
///
/// Epilogue: activation fusion (Relu, Gelu, etc.) applied during output write-back.
///
/// Supports f32 and f16. Workspace: not used.

#include "cuda_common.cuh"
#include "nnops/ops/depthwise_conv.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops::backend::cuda {

// ============================================================
// Device-side epilogue apply (mirrors host apply_epilogue)
// ============================================================

__device__ inline float device_apply_epilogue(
    EpilogueActivateType ep_type, float alpha, float beta, float x)
{
    switch (ep_type) {
    case EpilogueActivateType::None:
        return x;
    case EpilogueActivateType::Relu:
        return fmaxf(x, 0.0f);
    case EpilogueActivateType::LeakyRelu:
        return (x > 0.0f) ? x : alpha * x;
    case EpilogueActivateType::Sigmoid:
        return 1.0f / (1.0f + expf(-x));
    case EpilogueActivateType::Tanh:
        return tanhf(x);
    case EpilogueActivateType::Gelu: {
        constexpr float c = 0.7978845608028654f;
        return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
    }
    case EpilogueActivateType::Silu:
        return x / (1.0f + expf(-x));
    case EpilogueActivateType::HardSwish: {
        float relu6 = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
        return x * relu6 * (beta / 6.0f);
    }
    case EpilogueActivateType::Elu:
        return (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
    }
    return x;
}

// ============================================================
// 2D DepthwiseConv CUDA kernel
// ============================================================

template <typename T>
__global__ void depthwise_conv_2d_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,     // may be nullptr
    T* __restrict__ output,
    int64_t N, int64_t C,
    int64_t IH, int64_t IW,
    int64_t OH, int64_t OW,
    int64_t KH, int64_t KW,
    int64_t SH, int64_t SW,
    int64_t DH, int64_t DW,
    int64_t PH, int64_t PW,
    int64_t in_ch_stride,
    int64_t in_row_stride,
    int64_t out_ch_stride,
    int64_t out_row_stride,
    int64_t w_ch_stride,
    EpilogueActivateType ep_type,
    float ep_alpha,
    float ep_beta,
    bool has_bias,
    bool add_to)
{
    const int n = blockIdx.y / C;
    const int c = blockIdx.y % C;
    const int tid = threadIdx.x;
    const int64_t total_spatial = OH * OW;

    const T* in_ch  = input  + n * C * in_ch_stride  + c * in_ch_stride;
    const T* w_ch   = weight + c * w_ch_stride;
    T*       out_ch = output + n * C * out_ch_stride + c * out_ch_stride;

    const float bias_val = has_bias ? s_load(&bias[c]) : 0.0f;

    for (int64_t sp = tid; sp < total_spatial; sp += blockDim.x) {
        const int64_t oh = sp / OW;
        const int64_t ow = sp % OW;

        float sum = bias_val;

        for (int64_t kh = 0; kh < KH; ++kh) {
            const int64_t ih = oh * SH + kh * DH - PH;
            if (ih < 0 || ih >= IH) continue;
            for (int64_t kw = 0; kw < KW; ++kw) {
                const int64_t iw = ow * SW + kw * DW - PW;
                if (iw < 0 || iw >= IW) continue;
                sum += s_load(&in_ch[ih * in_row_stride + iw]) * s_load(&w_ch[kh * KW + kw]);
            }
        }

        float val = device_apply_epilogue(ep_type, ep_alpha, ep_beta, sum);

        const int64_t out_idx = oh * out_row_stride + ow;
        if (add_to) {
            s_store(&out_ch[out_idx], s_load(&out_ch[out_idx]) + val);
        } else {
            s_store(&out_ch[out_idx], val);
        }
    }
}

// ============================================================
// 3D DepthwiseConv CUDA kernel
// ============================================================

template <typename T>
__global__ void depthwise_conv_3d_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int64_t N, int64_t C,
    int64_t ID, int64_t IH, int64_t IW,
    int64_t OD, int64_t OH, int64_t OW,
    int64_t KD, int64_t KH, int64_t KW,
    int64_t SD, int64_t SH, int64_t SW,
    int64_t DD, int64_t DH, int64_t DW,
    int64_t PD, int64_t PH, int64_t PW,
    int64_t in_ch_stride,
    int64_t in_row_stride,
    int64_t in_d_stride,
    int64_t out_ch_stride,
    int64_t out_row_stride,
    int64_t out_d_stride,
    int64_t w_ch_stride,
    EpilogueActivateType ep_type,
    float ep_alpha,
    float ep_beta,
    bool has_bias,
    bool add_to)
{
    const int n = blockIdx.y / C;
    const int c = blockIdx.y % C;
    const int tid = threadIdx.x;
    const int64_t total_spatial = OD * OH * OW;

    const T* in_ch  = input  + n * C * in_ch_stride  + c * in_ch_stride;
    const T* w_ch   = weight + c * w_ch_stride;
    T*       out_ch = output + n * C * out_ch_stride + c * out_ch_stride;

    const float bias_val = has_bias ? s_load(&bias[c]) : 0.0f;

    for (int64_t sp = tid; sp < total_spatial; sp += blockDim.x) {
        const int64_t od = sp / (OH * OW);
        const int64_t oh_ow = sp % (OH * OW);
        const int64_t oh = oh_ow / OW;
        const int64_t ow = oh_ow % OW;

        float sum = bias_val;

        for (int64_t kd = 0; kd < KD; ++kd) {
            const int64_t id = od * SD + kd * DD - PD;
            if (id < 0 || id >= ID) continue;
            for (int64_t kh = 0; kh < KH; ++kh) {
                const int64_t ih = oh * SH + kh * DH - PH;
                if (ih < 0 || ih >= IH) continue;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    const int64_t iw = ow * SW + kw * DW - PW;
                    if (iw < 0 || iw >= IW) continue;
                    sum += s_load(&in_ch[id * in_d_stride + ih * in_row_stride + iw])
                         * s_load(&w_ch[kd * KH * KW + kh * KW + kw]);
                }
            }
        }

        float val = device_apply_epilogue(ep_type, ep_alpha, ep_beta, sum);

        const int64_t out_idx = od * out_d_stride + oh * out_row_stride + ow;
        if (add_to) {
            s_store(&out_ch[out_idx], s_load(&out_ch[out_idx]) + val);
        } else {
            s_store(&out_ch[out_idx], val);
        }
    }
}

// ============================================================
// Host-side launcher — 2D
// ============================================================

template <typename T>
void depthwise_conv_cuda_2d_impl(
    const DepthwiseConvAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t N  = input.shape(0);
    const int64_t C  = input.shape(1);
    const int64_t IH = input.shape(2);
    const int64_t IW = input.shape(3);

    const int64_t KH = attrs.kernel_size[1];
    const int64_t KW = attrs.kernel_size[2];

    const int64_t OH = output.shape(2);
    const int64_t OW = output.shape(3);

    const int64_t SH = attrs.stride[1];
    const int64_t SW = attrs.stride[2];
    const int64_t DH = attrs.dilation[1];
    const int64_t DW = attrs.dilation[2];
    const int64_t PH = attrs.padding[1];
    const int64_t PW = attrs.padding[2];

    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t in_ch_stride   = IH * in_row_stride;
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_ch_stride  = OH * out_row_stride;
    const int64_t w_ch_stride    = KH * KW;

    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    const int64_t total_spatial = OH * OW;
    const int block_size = std::min(static_cast<int>(total_spatial), kCudaBlockSize);
    const int64_t NC = N * C;
    const dim3 grid(1, static_cast<unsigned int>(NC), 1);

    depthwise_conv_2d_kernel<<<grid, block_size, 0, stream>>>(
        input.ptr<T>(), weight.ptr<T>(),
        has_bias ? inputs[2].ptr<T>() : nullptr,
        output.ptr<T>(),
        N, C, IH, IW, OH, OW, KH, KW,
        SH, SW, DH, DW, PH, PW,
        in_ch_stride, in_row_stride,
        out_ch_stride, out_row_stride,
        w_ch_stride,
        attrs.epilogue.type, attrs.epilogue.alpha, attrs.epilogue.beta,
        has_bias, attrs.add_to);
}

// ============================================================
// Host-side launcher — 3D
// ============================================================

template <typename T>
void depthwise_conv_cuda_3d_impl(
    const DepthwiseConvAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t N  = input.shape(0);
    const int64_t C  = input.shape(1);
    const int64_t ID = input.shape(2);
    const int64_t IH = input.shape(3);
    const int64_t IW = input.shape(4);

    const int64_t KD = attrs.kernel_size[0];
    const int64_t KH = attrs.kernel_size[1];
    const int64_t KW = attrs.kernel_size[2];

    const int64_t OD = output.shape(2);
    const int64_t OH = output.shape(3);
    const int64_t OW = output.shape(4);

    const int64_t SD = attrs.stride[0];
    const int64_t SH = attrs.stride[1];
    const int64_t SW = attrs.stride[2];
    const int64_t DD = attrs.dilation[0];
    const int64_t DH = attrs.dilation[1];
    const int64_t DW = attrs.dilation[2];
    const int64_t PD = attrs.padding[0];
    const int64_t PH = attrs.padding[1];
    const int64_t PW = attrs.padding[2];

    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t in_d_stride    = IH * in_row_stride;
    const int64_t in_ch_stride   = ID * in_d_stride;
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_d_stride   = OH * out_row_stride;
    const int64_t out_ch_stride  = OD * out_d_stride;
    const int64_t w_ch_stride    = KD * KH * KW;

    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    const int64_t total_spatial = OD * OH * OW;
    const int block_size = std::min(static_cast<int>(total_spatial), kCudaBlockSize);
    const int64_t NC = N * C;
    const dim3 grid(1, static_cast<unsigned int>(NC), 1);

    depthwise_conv_3d_kernel<<<grid, block_size, 0, stream>>>(
        input.ptr<T>(), weight.ptr<T>(),
        has_bias ? inputs[2].ptr<T>() : nullptr,
        output.ptr<T>(),
        N, C, ID, IH, IW, OD, OH, OW, KD, KH, KW,
        SD, SH, SW, DD, DH, DW, PD, PH, PW,
        in_ch_stride, in_row_stride, in_d_stride,
        out_ch_stride, out_row_stride, out_d_stride,
        w_ch_stride,
        attrs.epilogue.type, attrs.epilogue.alpha, attrs.epilogue.beta,
        has_bias, attrs.add_to);
}

// ============================================================
// Unified entry point
// ============================================================

void depthwise_conv_cuda(
    const DepthwiseConvAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx,
    void* /*workspace*/)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    const int64_t srank = DepthwiseConvAttributes::spatial_rank(rank);

    const auto dtype = input.data_type();
    switch (dtype) {
    case DataType::f32:
        if (srank == 3)
            depthwise_conv_cuda_3d_impl<float>(attrs, output, inputs, ctx);
        else
            depthwise_conv_cuda_2d_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        if (srank == 3)
            depthwise_conv_cuda_3d_impl<__half>(attrs, output, inputs, ctx);
        else
            depthwise_conv_cuda_2d_impl<__half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"depthwise_conv_cuda: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cuda
