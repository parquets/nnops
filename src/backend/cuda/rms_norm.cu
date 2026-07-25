/// @file rms_norm.cu
/// @brief CUDA implementation of RMS normalization (root mean square).
///
/// Formula:
///   rms = sqrt(mean(x_i^2) + epsilon)
///   y_i = x_i / rms * scale_i
///
/// Unlike LayerNorm, no mean subtraction and no bias. Simple sum_sq
/// reduction is numerically adequate.
///
/// Algorithm:
///   Pass 1 — block-level reduction of sum_sq.
///   Pass 2 — normalize with scale.
///
/// Supports f32 and f16. Workspace: not used.

#include "cuda_common.cuh"
#include "nnops/ops/rms_norm.hpp"
#include "nnops/detail/assert.hpp"

#include <vector>

namespace nnops::backend::cuda {

// ============================================================
// RMSNorm CUDA kernel — contiguous tail fast path
// ============================================================

template <typename T>
__global__ void rms_norm_kernel(
    const T* __restrict__ input,
    const T* __restrict__ scale,
    T* __restrict__ output,
    int64_t num_rows,
    int64_t norm_size,
    int64_t row_stride,
    bool scale_is_scalar,
    float epsilon,
    bool add_to)
{
    extern __shared__ float shared_buf[];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    if (row >= num_rows) return;

    const T* x_row = input + row * row_stride;
    T* y_row = output + row * row_stride;

    // ---- Pass 1: Block reduce sum_sq ----
    float sum_sq = 0.0f;

    for (int i = tid; i < norm_size; i += blockDim.x) {
        float x = s_load(&x_row[i]);
        sum_sq += x * x;
    }

    // Block-level sum reduction via shared memory tree
    shared_buf[tid] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_buf[tid] += shared_buf[tid + s];
        }
        __syncthreads();
    }

    const float rms = sqrtf(shared_buf[0] / static_cast<float>(norm_size) + epsilon);
    const float inv_rms = 1.0f / rms;

    __syncthreads();

    // ---- Pass 2: Normalize ----
    for (int i = tid; i < norm_size; i += blockDim.x) {
        int64_t s_idx = scale_is_scalar ? 0 : i;
        float x = s_load(&x_row[i]);
        float s = s_load(&scale[s_idx]);
        float rv = x * inv_rms * s;
        if (add_to) {
            s_store(&y_row[i], s_load(&y_row[i]) + rv);
        } else {
            s_store(&y_row[i], rv);
        }
    }
}

// ============================================================
// RMSNorm general axis kernel
// ============================================================

template <typename T>
__global__ void rms_norm_general_kernel(
    const T* __restrict__ input,
    const T* __restrict__ scale,
    T* __restrict__ output,
    int64_t num_rows,
    int64_t norm_size,
    const int64_t* __restrict__ inner_offsets,
    int64_t outer_stride,
    bool scale_is_scalar,
    float epsilon,
    bool add_to)
{
    extern __shared__ float shared_buf[];

    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    if (row >= num_rows) return;

    const int64_t row_base = row * outer_stride;

    // ---- Pass 1: sum_sq ----
    float sum_sq = 0.0f;
    for (int i = tid; i < norm_size; i += blockDim.x) {
        int64_t off = row_base + inner_offsets[i];
        float x = s_load(&input[off]);
        sum_sq += x * x;
    }

    shared_buf[tid] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared_buf[tid] += shared_buf[tid + s];
        __syncthreads();
    }

    const float rms = sqrtf(shared_buf[0] / static_cast<float>(norm_size) + epsilon);
    const float inv_rms = 1.0f / rms;
    __syncthreads();

    // ---- Pass 2: normalize ----
    for (int i = tid; i < norm_size; i += blockDim.x) {
        int64_t off = row_base + inner_offsets[i];
        int64_t s_idx = scale_is_scalar ? 0 : i;
        float x = s_load(&input[off]);
        float s = s_load(&scale[s_idx]);
        float rv = x * inv_rms * s;
        if (add_to) {
            s_store(&output[off], s_load(&output[off]) + rv);
        } else {
            s_store(&output[off], rv);
        }
    }
}

// ============================================================
// Host-side launcher
// ============================================================

template <typename T>
void rms_norm_cuda_impl(
    const RMSNormAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    const auto& X     = inputs[0];
    const auto& scale = inputs[1];

    const int64_t rank = X.rank();
    const float epsilon = attrs.epsilon;

    int64_t axis = attrs.axis;
    if (axis < 0) axis += rank;
    NNOPS_ASSERT(axis >= 0 && axis < rank);

    int64_t num_rows = 1;
    for (int64_t i = 0; i < axis; ++i) {
        num_rows *= X.shape(i);
    }
    int64_t norm_size = 1;
    for (int64_t i = axis; i < rank; ++i) {
        norm_size *= X.shape(i);
    }

    const auto* x_ptr = X.ptr<T>();
    const auto* s_ptr = scale.ptr<T>();
    auto* y_ptr = output.ptr<T>();

    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    const bool is_contiguous_tail = (axis == rank - 1);
    const bool scale_is_scalar = (scale.numel() == 1);

    int block_size = std::min(static_cast<int>(norm_size), kCudaBlockSize);
    int shared_bytes = block_size * sizeof(float);

    if (is_contiguous_tail) {
        const int64_t x_row_stride = X.row_stride_elems();

        rms_norm_kernel<<<num_rows, block_size, shared_bytes, stream>>>(
            x_ptr, s_ptr, y_ptr,
            num_rows, norm_size, x_row_stride,
            scale_is_scalar, epsilon, attrs.add_to);
    } else {
        std::vector<int64_t> inner_offsets(static_cast<size_t>(norm_size));
        for (int64_t flat = 0; flat < norm_size; ++flat) {
            int64_t off = 0;
            int64_t rem = flat;
            for (int64_t d = rank - 1; d >= axis; --d) {
                int64_t dim = X.shape(d);
                off += (rem % dim) * X.stride_elems(d);
                rem /= dim;
            }
            inner_offsets[static_cast<size_t>(flat)] = off;
        }

        const int64_t outer_stride = (axis > 0) ? X.stride_elems(axis - 1) : 0;

        int64_t* d_offsets = nullptr;
        CUDA_CHECK(cudaMalloc(&d_offsets, norm_size * sizeof(int64_t)));
        CUDA_CHECK(cudaMemcpyAsync(d_offsets, inner_offsets.data(),
                                     norm_size * sizeof(int64_t),
                                     cudaMemcpyHostToDevice, stream));

        rms_norm_general_kernel<<<num_rows, block_size, shared_bytes, stream>>>(
            x_ptr, s_ptr, y_ptr,
            num_rows, norm_size, d_offsets, outer_stride,
            scale_is_scalar, epsilon, attrs.add_to);

        CUDA_CHECK(cudaFree(d_offsets));
    }
}

// ============================================================
// Entry point
// ============================================================

void rms_norm_cuda(
    const RMSNormAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx,
    void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        rms_norm_cuda_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        rms_norm_cuda_impl<__half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"rms_norm_cuda: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cuda
