/// @file batch_norm.cu
/// @brief CUDA implementation of batch normalization (inference only).
///
/// Fused formula (mirrors CPU):
///   new_scale = inv_std * scale
///   new_bias  = bias - mean * new_scale
///   inv_std = 1 / sqrt(var + epsilon)
///   y = x * new_scale + new_bias
///
/// Two modes:
///   spatial (default): per-channel statistics, scale/bias/mean/var all shape [C].
///   non-spatial: per-element statistics — flat element-wise kernel.
///
/// Supports f32 and f16. Workspace: not used.

#include "cuda_common.cuh"
#include "nnops/ops/batch_norm.hpp"
#include "nnops/detail/assert.hpp"

#include <vector>

namespace nnops::backend::cuda {

// ============================================================
// Non-spatial batch_norm kernel — per-element statistics
// ============================================================

template <typename T>
__global__ void batch_norm_nonspatial_kernel(
    const T* __restrict__ input,
    const T* __restrict__ scale,
    const T* __restrict__ bias,
    const T* __restrict__ mean,
    const T* __restrict__ var,
    T* __restrict__ output,
    int64_t total,
    float epsilon,
    bool add_to)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < total; i += stride) {
        float inv_std = 1.0f / sqrtf(s_load(&var[i]) + epsilon);
        float ns = inv_std * s_load(&scale[i]);
        float nb = s_load(&bias[i]) - s_load(&mean[i]) * ns;
        float rv = s_load(&input[i]) * ns + nb;
        if (add_to) {
            s_store(&output[i], s_load(&output[i]) + rv);
        } else {
            s_store(&output[i], rv);
        }
    }
}

template <typename T>
__global__ void batch_norm_spatial_per_sample_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ new_scale,
    const float* __restrict__ new_bias,
    int64_t C,
    int64_t sample_size,
    int64_t in_n_stride,
    int64_t in_c_stride,
    int64_t out_n_stride,
    int64_t out_c_stride,
    bool add_to)
{
    // Grid: N blocks, C threads per block (or similar)
    // One block per sample, iterate over channels and spatial
    const int n = blockIdx.x;
    const int tid = threadIdx.x;

    // Each thread works on one or more spatial positions for each channel
    for (int c = 0; c < C; ++c) {
        float ns = new_scale[c];
        float nb = new_bias[c];
        const T* in_ch = input + n * in_n_stride + c * in_c_stride;
        T* out_ch = output + n * out_n_stride + c * out_c_stride;

        for (int64_t s = tid; s < sample_size; s += blockDim.x) {
            float rv = s_load(&in_ch[s]) * ns + nb;
            if (add_to) {
                s_store(&out_ch[s], s_load(&out_ch[s]) + rv);
            } else {
                s_store(&out_ch[s], rv);
            }
        }
    }
}

// ============================================================
// Host-side launcher
// ============================================================

template <typename T>
void batch_norm_cuda_impl(
    const BatchNormAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    const auto& X     = inputs[0];
    const auto& scale = inputs[1];
    const auto& bias  = inputs[2];
    const auto& mean  = inputs[3];
    const auto& var   = inputs[4];

    const int64_t rank = X.rank();
    const float epsilon = attrs.epsilon;

    int64_t N, C, sample_size;
    if (rank == 1) {
        N = X.shape(0);
        C = 1;
        sample_size = 1;
    } else {
        N = X.shape(0);
        C = X.shape(1);
        sample_size = 1;
        for (int64_t i = 2; i < rank; ++i) {
            sample_size *= X.shape(i);
        }
    }

    const auto* x_ptr  = X.ptr<T>();
    const auto* s_ptr  = scale.ptr<T>();
    const auto* b_ptr  = bias.ptr<T>();
    const auto* m_ptr  = mean.ptr<T>();
    const auto* v_ptr  = var.ptr<T>();
    auto* y_ptr = output.ptr<T>();

    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    if (attrs.spatial) {
        // Precompute new_scale and new_bias per channel (in float for precision)
        std::vector<float> new_scale(static_cast<size_t>(C));
        std::vector<float> new_bias(static_cast<size_t>(C));
        for (int64_t c = 0; c < C; ++c) {
            float inv_std = 1.0f / sqrtf(s_load(&v_ptr[c]) + epsilon);
            new_scale[static_cast<size_t>(c)] = inv_std * s_load(&s_ptr[c]);
            new_bias[static_cast<size_t>(c)]  = s_load(&b_ptr[c]) - s_load(&m_ptr[c]) * new_scale[static_cast<size_t>(c)];
        }

        // Copy precomputed values to device
        float* d_new_scale = nullptr;
        float* d_new_bias  = nullptr;
        CUDA_CHECK(cudaMalloc(&d_new_scale, C * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_new_bias,  C * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(d_new_scale, new_scale.data(), C * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_new_bias,  new_bias.data(),  C * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));

        const int64_t in_n_stride  = X.stride_elems(0);
        const int64_t in_c_stride  = (rank >= 2) ? X.stride_elems(1) : 1;
        const int64_t out_n_stride = output.stride_elems(0);
        const int64_t out_c_stride = (rank >= 2) ? output.stride_elems(1) : 1;

        const int block_size = std::min(static_cast<int>(sample_size), kCudaBlockSize);

        batch_norm_spatial_per_sample_kernel<<<N, block_size, 0, stream>>>(
            x_ptr, y_ptr,
            d_new_scale, d_new_bias,
            C, sample_size,
            in_n_stride, in_c_stride,
            out_n_stride, out_c_stride,
            attrs.add_to);

        CUDA_CHECK(cudaFree(d_new_scale));
        CUDA_CHECK(cudaFree(d_new_bias));
    } else {
        // Non-spatial: per-element statistics, flat element-wise
        const int64_t total = X.numel();
        const int block_size = kCudaBlockSize;
        const int grid_size = std::min(static_cast<int>(ceil_div(total, static_cast<int64_t>(block_size))),
                                       65535);

        batch_norm_nonspatial_kernel<<<grid_size, block_size, 0, stream>>>(
            x_ptr, s_ptr, b_ptr, m_ptr, v_ptr, y_ptr,
            total, epsilon, attrs.add_to);
    }
}

// ============================================================
// Entry point
// ============================================================

void batch_norm_cuda(
    const BatchNormAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx,
    void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        batch_norm_cuda_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        batch_norm_cuda_impl<__half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"batch_norm_cuda: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cuda
