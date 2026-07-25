/// @file layer_norm.cu
/// @brief CUDA implementation of layer normalization.
///
/// Algorithm (mirrors CPU path):
///   One block per row — block-level reduction for mean/variance (Pass 1),
///   then block-level normalize with scale/bias (Pass 2).
///
/// Numerical stability via Welford's online algorithm on per-thread partials,
/// then merged using Chan et al.'s parallel Welford formula.
///
/// Supports f32 and f16. Workspace: not used.

#include "cuda_common.cuh"
#include "nnops/ops/layer_norm.hpp"
#include "nnops/detail/assert.hpp"

#include <vector>

namespace nnops::backend::cuda {

// ============================================================
// Welford stats for per-thread partial aggregation
// ============================================================

struct WelfordStats {
    int64_t n;
    float mean;
    float M2;  // sum of squared differences from mean
};

// Chan et al. parallel merge formula
__device__ inline void welford_merge(WelfordStats& a, const WelfordStats& b) {
    if (b.n == 0) return;
    if (a.n == 0) { a = b; return; }
    const int64_t total = a.n + b.n;
    const float delta = b.mean - a.mean;
    const float na = static_cast<float>(a.n);
    const float nb = static_cast<float>(b.n);
    a.mean = (na * a.mean + nb * b.mean) / static_cast<float>(total);
    a.M2 = a.M2 + b.M2 + delta * delta * na * nb / static_cast<float>(total);
    a.n = total;
}

// ============================================================
// Block-level Welford merge via shared memory tree reduction
// ============================================================

__device__ inline WelfordStats block_reduce_welford(
    WelfordStats st, WelfordStats* shared, int tid, int block_size)
{
    shared[tid] = st;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            welford_merge(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }
    return shared[0];
}

// ============================================================
// LayerNorm CUDA kernel — contiguous tail fast path
// ============================================================

template <typename T>
__global__ void layer_norm_kernel(
    const T* __restrict__ input,
    const T* __restrict__ scale,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int64_t num_rows,
    int64_t norm_size,
    int64_t row_stride,
    bool scale_is_scalar,
    bool has_bias,
    float epsilon,
    bool add_to)
{
    extern __shared__ float shared_buf[];
    WelfordStats* welford_shared = reinterpret_cast<WelfordStats*>(shared_buf);

    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    if (row >= num_rows) return;

    const T* x_row = input + row * row_stride;
    T* y_row = output + row * row_stride;

    // ================================================================
    // Pass 1 — Per-thread Welford accumulation, then block-level merge
    // ================================================================

    WelfordStats st{0, 0.0f, 0.0f};

    for (int i = tid; i < norm_size; i += blockDim.x) {
        float x = s_load(&x_row[i]);
        int64_t n_new = st.n + 1;
        float delta = x - st.mean;
        st.mean += delta / static_cast<float>(n_new);
        float delta2 = x - st.mean;
        st.M2 += delta * delta2;
        st.n = n_new;
    }

    // Merge per-thread stats via shared memory tree
    st = block_reduce_welford(st, welford_shared, tid, blockDim.x);

    const float var_val = st.M2 / static_cast<float>(st.n);
    const float inv_std = rsqrtf(var_val + epsilon);
    const float mean_val = st.mean;

    __syncthreads();  // ensure block reduction complete before Pass 2 reads shared memory

    // ================================================================
    // Pass 2 — Normalize with scale/bias
    // ================================================================

    for (int i = tid; i < norm_size; i += blockDim.x) {
        int64_t s_idx = scale_is_scalar ? 0 : i;
        float x = s_load(&x_row[i]);
        float s = s_load(&scale[s_idx]);
        float b = has_bias ? s_load(&bias[s_idx]) : 0.0f;
        float rv = (x - mean_val) * inv_std * s + b;
        if (add_to) {
            s_store(&y_row[i], s_load(&y_row[i]) + rv);
        } else {
            s_store(&y_row[i], rv);
        }
    }
}

// ============================================================
// LayerNorm general axis kernel (non-contiguous tail)
// ============================================================

template <typename T>
__global__ void layer_norm_general_kernel(
    const T* __restrict__ input,
    const T* __restrict__ scale,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int64_t num_rows,
    int64_t norm_size,
    const int64_t* __restrict__ inner_offsets,
    int64_t outer_stride,
    bool scale_is_scalar,
    bool has_bias,
    float epsilon,
    bool add_to)
{
    extern __shared__ float shared_buf[];
    WelfordStats* welford_shared = reinterpret_cast<WelfordStats*>(shared_buf);

    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    if (row >= num_rows) return;

    const int64_t row_base = row * outer_stride;

    // ---- Pass 1: Welford accumulation ----
    WelfordStats st{0, 0.0f, 0.0f};

    for (int i = tid; i < norm_size; i += blockDim.x) {
        int64_t off = row_base + inner_offsets[i];
        float x = s_load(&input[off]);
        int64_t n_new = st.n + 1;
        float delta = x - st.mean;
        st.mean += delta / static_cast<float>(n_new);
        float delta2 = x - st.mean;
        st.M2 += delta * delta2;
        st.n = n_new;
    }

    st = block_reduce_welford(st, welford_shared, tid, blockDim.x);

    const float var_val = st.M2 / static_cast<float>(st.n);
    const float inv_std = rsqrtf(var_val + epsilon);
    const float mean_val = st.mean;

    __syncthreads();

    // ---- Pass 2: normalize ----
    for (int i = tid; i < norm_size; i += blockDim.x) {
        int64_t off = row_base + inner_offsets[i];
        int64_t s_idx = scale_is_scalar ? 0 : i;
        float x = s_load(&input[off]);
        float s = s_load(&scale[s_idx]);
        float b = has_bias ? s_load(&bias[s_idx]) : 0.0f;
        float rv = (x - mean_val) * inv_std * s + b;
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
void layer_norm_cuda_impl(
    const LayerNormAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    const auto& X     = inputs[0];
    const auto& scale = inputs[1];
    const bool has_bias = (inputs.size() >= 3 && !inputs[2].is_empty());

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

    const auto* x_ptr  = X.ptr<T>();
    const auto* s_ptr  = scale.ptr<T>();
    const auto* b_ptr  = (has_bias && !inputs[2].is_empty()) ? inputs[2].ptr<T>() : nullptr;
    auto* y_ptr = output.ptr<T>();

    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    const bool is_contiguous_tail = (axis == rank - 1);
    const bool scale_is_scalar = (scale.numel() == 1);

    int block_size = std::min(static_cast<int>(norm_size), kCudaBlockSize);
    int shared_bytes = block_size * sizeof(WelfordStats);

    if (is_contiguous_tail) {
        const int64_t x_row_stride = X.row_stride_elems();

        layer_norm_kernel<<<num_rows, block_size, shared_bytes, stream>>>(
            x_ptr, s_ptr, b_ptr, y_ptr,
            num_rows, norm_size, x_row_stride,
            scale_is_scalar, has_bias, epsilon, attrs.add_to);
    } else {
        // Pre-compute inner offsets
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

        layer_norm_general_kernel<<<num_rows, block_size, shared_bytes, stream>>>(
            x_ptr, s_ptr, b_ptr, y_ptr,
            num_rows, norm_size, d_offsets, outer_stride,
            scale_is_scalar, has_bias, epsilon, attrs.add_to);

        CUDA_CHECK(cudaFree(d_offsets));
    }
}

// ============================================================
// Entry point
// ============================================================

void layer_norm_cuda(
    const LayerNormAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx,
    void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        layer_norm_cuda_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        layer_norm_cuda_impl<__half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"layer_norm_cuda: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cuda
