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
/// Two paths, chosen by where the normalization axis sits in memory (mirrors
/// the CPU backend), over channel-first and channel-last
/// layouts:
///   - The axis is the innermost axis of the storage (channel-first's last dim,
///     channel-last's C): every group is D contiguous elements, one block per
///     group (softmax_kernel).
///   - Every other axis: reference-style outer/D/inner decomposition with
///     strided access (softmax_general_kernel).
///
/// Channel-packed layouts are refused by the operator, so no location is
/// `pack`-scaled: stride_elems() is the physical stride.
///
/// Supports f32, f16 and bf16. Workspace: not used.

#include "cuda_common.cuh"
#include "nnops/ops/softmax.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops::backend::cuda {

// ============================================================
// Block-wide reduction over the shared float array
// ============================================================

/// Folds the block's shared array down to slot 0. Each round merges the upper
/// half of the live region into the lower one and, when the live count is odd,
/// the last element into slot 0 too. Halving a *stride* instead (the usual s/2
/// loop) never touches that last element, so a block whose size is not a power
/// of two silently drops it — a 37-wide row gives a 37-thread block, and the
/// element at index 36 never joins the max or the sum. The trailing barrier
/// leaves the array free for the next pass.
template <bool IsMax>
__device__ inline float block_reduce(float* smem, float v)
{
    const int tid = threadIdx.x;
    const int n_threads = blockDim.x;

    smem[tid] = v;
    __syncthreads();

    for (int n = n_threads; n > 1; n >>= 1) {
        const int half = n >> 1;
        if (tid < half) {
            if (IsMax) {
                if (smem[tid + half] > smem[tid]) { smem[tid] = smem[tid + half]; }
            } else {
                smem[tid] += smem[tid + half];
            }
        }
        if (tid == 0 && (n & 1) != 0) {
            if (IsMax) {
                if (smem[n - 1] > smem[0]) { smem[0] = smem[n - 1]; }
            } else {
                smem[0] += smem[n - 1];
            }
        }
        __syncthreads();
    }

    const float result = smem[0];
    __syncthreads();
    return result;
}

// ============================================================
// Softmax CUDA kernel — single block per row, contiguous tail
// ============================================================

template <typename T>
__global__ void softmax_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t num_rows,
    int64_t norm_size,
    int64_t in_group_stride,   // elements between consecutive input groups
    int64_t out_group_stride,  // and between consecutive output groups
    bool log_softmax)
{
    extern __shared__ float shared_buf[];
    float* smem = shared_buf;

    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    if (row >= num_rows) return;

    const T* x_row = input + row * in_group_stride;
    T* y_row = output + row * out_group_stride;

    // ============================================================
    // Pass 1 — Block-level max reduction
    // ============================================================

    float max_val = -1e30f;

    for (int i = tid; i < norm_size; i += blockDim.x) {
        float x = s_load(&x_row[i]);
        if (x > max_val) max_val = x;
    }

    max_val = block_reduce<true>(smem, max_val);

    const float neg_max = -max_val;

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

    sum_exp = block_reduce<false>(smem, sum_exp);

    // ============================================================
    // Pass 3 — Normalize
    // ============================================================

    if (log_softmax) {
        const float log_sum = logf(sum_exp);
        const float bias = neg_max - log_sum;

        for (int i = tid; i < norm_size; i += blockDim.x) {
            float rv = s_load(&x_row[i]) + bias;
            s_store(&y_row[i], rv);
        }
    } else {
        const float inv_sum = 1.0f / sum_exp;
        for (int i = tid; i < norm_size; i += blockDim.x) {
            s_store(&y_row[i], s_load(&y_row[i]) * inv_sum);
        }
    }
}

// ============================================================
// General axis softmax kernel — reference-style decomposition
//
// Each block normalizes D elements at stride axis_stride.
// Launched with outer_size * inner_total blocks for full
// parallelism across independent normalization groups.
// outer_offsets/inner_offsets are the layout's own element offsets, walked once
// on the host from stride_elems().
// ============================================================

template <typename T>
__global__ void softmax_general_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t outer_size,
    int64_t D,
    int64_t in_axis_stride,
    int64_t out_axis_stride,
    int64_t inner_total,
    const int64_t* __restrict__ outer_offsets,
    const int64_t* __restrict__ inner_offsets,
    const int64_t* __restrict__ out_outer_offsets,
    const int64_t* __restrict__ out_inner_offsets,
    bool log_softmax)
{
    extern __shared__ float shared_buf[];
    float* smem = shared_buf;

    const int tid = threadIdx.x;
    const int outer = static_cast<int>(blockIdx.x / static_cast<unsigned int>(inner_total));
    const int inner_idx = static_cast<int>(blockIdx.x % static_cast<unsigned int>(inner_total));

    if (outer >= outer_size) return;

    const int64_t base = outer_offsets[outer];
    const int64_t inner_off = inner_offsets[inner_idx];
    const int64_t y_base = out_outer_offsets[outer];
    const int64_t y_inner_off = out_inner_offsets[inner_idx];

    // ---- Pass 1: find max ----
    float max_val = -1e30f;
    for (int k = tid; k < D; k += blockDim.x) {
        float x = s_load(&input[base + inner_off + k * in_axis_stride]);
        if (x > max_val) max_val = x;
    }

    max_val = block_reduce<true>(smem, max_val);

    // ---- Pass 2: sum of exp(x - max) ----
    float sum_exp = 0.0f;
    for (int k = tid; k < D; k += blockDim.x) {
        sum_exp += expf(s_load(&input[base + inner_off + k * in_axis_stride]) - max_val);
    }

    sum_exp = block_reduce<false>(smem, sum_exp);

    // ---- Pass 3: normalize ----
    if (log_softmax) {
        float log_sum = logf(sum_exp);
        for (int k = tid; k < D; k += blockDim.x) {
            int64_t off = base + inner_off + k * in_axis_stride;
            int64_t y_off = y_base + y_inner_off + k * out_axis_stride;
            float val = s_load(&input[off]) - max_val - log_sum;
            s_store(&output[y_off], val);
        }
    } else {
        float inv_sum = 1.0f / sum_exp;
        for (int k = tid; k < D; k += blockDim.x) {
            int64_t off = base + inner_off + k * in_axis_stride;
            int64_t y_off = y_base + y_inner_off + k * out_axis_stride;
            float val = expf(s_load(&input[off]) - max_val) * inv_sum;
            s_store(&output[y_off], val);
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

    if (X.numel() == 0 || X.shape(axis) == 0) return;  // nothing to launch

    const auto* x_ptr = X.ptr<T>();
    auto* y_ptr = output.ptr<T>();

    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    // ============================================================
    // Fast path: the axis is the innermost axis of the storage, so every group
    // is D contiguous elements and the groups are `group_stride` apart.
    // ============================================================

    const bool channel_last = is_channel_last(X.layout()) && rank > 1;
    int64_t in_group_stride = 0;
    int64_t out_group_stride = 0;
    bool contiguous_axis = false;
    if (rank == 1) {
        contiguous_axis = true;
        in_group_stride = out_group_stride = 1;
    } else if (channel_last) {
        // The groups are the layout's own rows, so both sides share the shape's
        // stride whatever their pitches are.
        if (axis == 1) {
            contiguous_axis = true;
            in_group_stride = out_group_stride = X.shape(1);
        } else if (axis == rank - 1 && X.shape(1) == 1) {
            contiguous_axis = true;
            in_group_stride = out_group_stride = X.shape(rank - 1);
        }
    } else if (axis == rank - 1) {
        // Channel-first rows: each side walks its own pitch, which a caller may
        // legitimately give the output differently from the input.
        contiguous_axis = true;
        in_group_stride = X.row_stride_elems();
        out_group_stride = output.row_stride_elems();
    }

    const int64_t norm_size = X.shape(axis);

    if (contiguous_axis && norm_size > 0) {
        const int64_t num_rows = X.numel() / norm_size;

        const int block_size = std::min(static_cast<int>(norm_size), kCudaBlockSize);
        const int shared_bytes = block_size * sizeof(float);

        softmax_kernel<<<num_rows, block_size, shared_bytes, stream>>>(
            x_ptr, y_ptr,
            num_rows, norm_size, in_group_stride, out_group_stride,
            log_softmax);
        return;
    }

    // ============================================================
    // General axis: reference-style outer/D/inner decomposition
    // ============================================================

    {
        int64_t outer_size = 1;
        for (int64_t i = 0; i < axis; ++i)
            outer_size *= X.shape(i);

        const int64_t D = norm_size;
        // No channel packing, so stride_elems() is the physical stride.
        const int64_t in_axis_stride = X.stride_elems(axis);
        const int64_t out_axis_stride = output.stride_elems(axis);

        int64_t inner_total = 1;
        for (int64_t i = axis + 1; i < rank; ++i)
            inner_total *= X.shape(i);

        // Precompute the position tables on host — one entry per outer position
        // (the axis index left at 0) and one per inner position, for each side,
        // since the output may carry a different pitch than the input.
        const size_t table = static_cast<size_t>(outer_size + inner_total);
        std::vector<int64_t> offsets(2 * table);
        int64_t* outer_offsets = offsets.data();
        int64_t* inner_offsets = offsets.data() + outer_size;
        int64_t* y_outer_offsets = offsets.data() + table;
        int64_t* y_inner_offsets = offsets.data() + table + outer_size;

        for (int64_t o = 0; o < outer_size; ++o) {
            int64_t off = 0, rem = o;
            int64_t y_off = 0;
            for (int64_t d = axis - 1; d >= 0; --d) {
                int64_t dim = X.shape(d);
                off += (rem % dim) * X.stride_elems(d);
                y_off += (rem % dim) * output.stride_elems(d);
                rem /= dim;
            }
            outer_offsets[o] = off;
            y_outer_offsets[o] = y_off;
        }
        for (int64_t s = 0; s < inner_total; ++s) {
            int64_t off = 0, rem = s;
            int64_t y_off = 0;
            for (int64_t d = rank - 1; d > axis; --d) {
                int64_t dim = X.shape(d);
                off += (rem % dim) * X.stride_elems(d);
                y_off += (rem % dim) * output.stride_elems(d);
                rem /= dim;
            }
            inner_offsets[static_cast<size_t>(s)] = off;
            y_inner_offsets[static_cast<size_t>(s)] = y_off;
        }

        int64_t* d_offsets = nullptr;
        CUDA_CHECK(cudaMalloc(&d_offsets, offsets.size() * sizeof(int64_t)));
        CUDA_CHECK(cudaMemcpyAsync(d_offsets, offsets.data(),
                                     offsets.size() * sizeof(int64_t),
                                     cudaMemcpyHostToDevice, stream));

        // Launch outer_size * inner_total blocks (one per normalization group).
        // Use gridDim.y to carry inner_total so the kernel can decompose blockIdx.x.
        const int block_size = std::min(static_cast<int>(D), kCudaBlockSize);
        const int shared_bytes = block_size * sizeof(float);

        const int total_blocks = static_cast<int>(outer_size * inner_total);
        dim3 grid(total_blocks);

        softmax_general_kernel<<<grid, block_size, shared_bytes, stream>>>(
            x_ptr, y_ptr,
            outer_size, D, in_axis_stride, out_axis_stride, inner_total,
            d_offsets, d_offsets + outer_size,
            d_offsets + table, d_offsets + table + outer_size,
            log_softmax);

        CUDA_CHECK(cudaFree(d_offsets));
    }
}

// ============================================================
// Entry point (dtype dispatch)
// ============================================================

Status softmax_cuda(
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
        return Status::ok;
    case DataType::f16:
        softmax_cuda_impl<__half>(attrs, output, inputs, ctx);
        return Status::ok;
    case DataType::bf16:
        softmax_cuda_impl<__nv_bfloat16>(attrs, output, inputs, ctx);
        return Status::ok;
    default:
        return Status::unsupported;
    }
    return Status::ok;
}

}  // namespace nnops::backend::cuda
