/// @file softmax.cu
/// @brief CUDA implementation of softmax / log-softmax.
///
/// Algorithm (per row, mirrors CPU path exactly):
///   1. ReduceMax — block-level max reduction via shared memory → max_val.
///   2. ComputeSumExp — exp(x - max) reduction + sum; for non-log non-add_to
///      exp values are stored to output buffer to avoid recomputation.
///   3. Normalize — softmax: stored exp * inv_sum or recompute;
///      log_softmax: (x - max) - log(sum).
///
/// Supports f32 and f16 via separate template instantiations.
/// Workspace: not used.

#include "cuda_common.cuh"
#include "nnops/ops/softmax.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops::backend::cuda {

// ============================================================
// Softmax CUDA kernel — single block per row, contiguous tail
// ============================================================

template <typename T>
__global__ void softmax_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t num_rows,
    int64_t norm_size,
    int64_t row_stride,       // in elements
    bool log_softmax)
{
    extern __shared__ float shared_buf[];  // shared memory for reductions: float[norm_size] or float[blockDim.x]
    float* smem = shared_buf;

    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    if (row >= num_rows) return;

    const T* x_row = input + row * row_stride;
    T* y_row = output + row * row_stride;

    // ============================================================
    // Pass 1 — Block-level max reduction
    // ============================================================

    float max_val = -1e30f;

    // Each thread reduces its stride over the row
    for (int i = tid; i < norm_size; i += blockDim.x) {
        float x = s_load(&x_row[i]);
        if (x > max_val) max_val = x;
    }

    // Block-level max reduction
    smem[tid] = max_val;
    __syncthreads();
    max_val = smem[0];
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] > max_val) {
            max_val = smem[tid + s];
        }
        smem[tid] = max_val;
        __syncthreads();
        max_val = smem[0];
    }

    const float neg_max = -max_val;
    __syncthreads();

    // ============================================================
    // Pass 2 — Sum of exp(x - max)
    // ============================================================

    const bool store_exp_to_output = !log_softmax;

    float sum_exp = 0.0f;

    for (int i = tid; i < norm_size; i += blockDim.x) {
        float x = s_load(&x_row[i]);
        float val = expf(x - max_val);
        if (store_exp_to_output) {
            s_store(&y_row[i], val);
        }
        sum_exp += val;
    }

    // Block-level sum reduction (same pattern as max)
    smem[tid] = sum_exp;
    __syncthreads();
    sum_exp = smem[0];
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_exp += smem[tid + s];
        }
        smem[tid] = sum_exp;
        __syncthreads();
        sum_exp = smem[0];
    }

    __syncthreads();

    // ============================================================
    // Pass 3 — Normalize
    // ============================================================

    if (log_softmax) {
        // log_softmax = (x - max) - log(sum)
        const float log_sum = logf(sum_exp);
        const float bias = neg_max - log_sum;  // -max - log(sum)

        for (int i = tid; i < norm_size; i += blockDim.x) {
            float rv = s_load(&x_row[i]) + bias;
            s_store(&y_row[i], rv);
        }
    } else {
        // softmax = exp(x - max) / sum_exp
        // exp already in output — just multiply by inv_sum
        const float inv_sum = 1.0f / sum_exp;
        for (int i = tid; i < norm_size; i += blockDim.x) {
            s_store(&y_row[i], s_load(&y_row[i]) * inv_sum);
        }
    }
}

// ============================================================
// General axis softmax kernel (non-contiguous tail)
// Each block processes one row, threads stride over norm_size
// ============================================================

template <typename T>
__global__ void softmax_general_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t num_rows,
    int64_t norm_size,
    const int64_t* __restrict__ inner_offsets,  // pre-computed offsets for each norm element
    int64_t outer_stride,    // element distance between rows
    bool log_softmax)
{
    extern __shared__ float shared_buf[];
    float* smem = shared_buf;

    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    if (row >= num_rows) return;

    const int64_t row_base = row * outer_stride;

    // ---- Pass 1: find max ----
    float max_val = -1e30f;
    for (int i = tid; i < norm_size; i += blockDim.x) {
        int64_t off = row_base + inner_offsets[i];
        float x = s_load(&input[off]);
        if (x > max_val) max_val = x;
    }

    smem[tid] = max_val;
    __syncthreads();
    max_val = smem[0];
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] > max_val) {
            max_val = smem[tid + s];
        }
        smem[tid] = max_val;
        __syncthreads();
        max_val = smem[0];
    }
    __syncthreads();

    // ---- Pass 2: sum of exp(x - max) ----
    float sum_exp = 0.0f;
    for (int i = tid; i < norm_size; i += blockDim.x) {
        int64_t off = row_base + inner_offsets[i];
        sum_exp += expf(s_load(&input[off]) - max_val);
    }

    smem[tid] = sum_exp;
    __syncthreads();
    sum_exp = smem[0];
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sum_exp += smem[tid + s];
        smem[tid] = sum_exp;
        __syncthreads();
        sum_exp = smem[0];
    }
    __syncthreads();

    // ---- Pass 3: normalize ----
    if (log_softmax) {
        float log_sum = logf(sum_exp);
        for (int i = tid; i < norm_size; i += blockDim.x) {
            int64_t off = row_base + inner_offsets[i];
            float val = s_load(&input[off]) - max_val - log_sum;
            s_store(&output[off], val);
        }
    } else {
        float inv_sum = 1.0f / sum_exp;
        for (int i = tid; i < norm_size; i += blockDim.x) {
            int64_t off = row_base + inner_offsets[i];
            float val = expf(s_load(&input[off]) - max_val) * inv_sum;
            s_store(&output[off], val);
        }
    }
}

// ============================================================
// Host-side kernel launcher (templated on T)
// ============================================================

template <typename T>
void softmax_cuda_impl(
    const SoftmaxAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    const auto& X = inputs[0];
    const int64_t rank = X.rank();
    NNOPS_ASSERT(rank >= 1);

    const bool log_softmax = attrs.log_softmax;

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
    auto* y_ptr = output.ptr<T>();

    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    const bool is_contiguous_tail = (axis == rank - 1);

    if (is_contiguous_tail) {
        // Fast path: contiguous tail, one block per row
        const int64_t x_row_stride = X.row_stride_elems();
        const int block_size = std::min(static_cast<int>(norm_size), kCudaBlockSize);
        const int shared_bytes = block_size * sizeof(float);

        softmax_kernel<<<num_rows, block_size, shared_bytes, stream>>>(
            x_ptr, y_ptr,
            num_rows, norm_size, x_row_stride,
            log_softmax);
    } else {
        // General axis: pre-compute inner offsets on host, copy to device
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

        const int block_size = std::min(static_cast<int>(norm_size), kCudaBlockSize);
        const int shared_bytes = block_size * sizeof(float);

        softmax_general_kernel<<<num_rows, block_size, shared_bytes, stream>>>(
            x_ptr, y_ptr,
            num_rows, norm_size, d_offsets, outer_stride,
            log_softmax);

        CUDA_CHECK(cudaFree(d_offsets));
    }
}

// ============================================================
// Entry point (dtype dispatch)
// ============================================================

void softmax_cuda(
    const SoftmaxAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx,
    void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        softmax_cuda_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        softmax_cuda_impl<__half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"softmax_cuda: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cuda
