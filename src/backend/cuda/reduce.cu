/// @file reduce.cu
/// @brief CUDA implementation of the Reduce operator.
///
/// Single-axis reduction using shared-memory block-level reductions.
/// Supports Sum, Min, Max, Mean over f32 and f16 data types.
///
/// Two-kernel pattern (same as softmax/layer_norm/rms_norm):
///   - reduce_fast_kernel: contiguous tail (axis == rank-1), strided access.
///   - reduce_general_kernel: arbitrary axis with pre-computed inner offsets.
///
/// Grid = outer_dim_count (one block per independent reduction row),
/// Block = min(norm_size, kCudaBlockSize), Shared = block_size * sizeof(float).

#include "cuda_common.cuh"
#include "nnops/ops/reduce.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/detail/assert.hpp"

#include <vector>
#include <span>

namespace nnops::backend::cuda {

using namespace nnops;

// ============================================================
// Fast path kernel: axis == rank - 1 (contiguous tail)
// ============================================================

template <typename T>
__global__ void reduce_fast_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t norm_size,
    int64_t row_stride,
    int reduce_type)  // 0=Sum, 1=Min, 2=Max, 3=Mean
{
    extern __shared__ float smem[];

    const int tid = threadIdx.x;
    const int64_t row = blockIdx.x;

    const T* x_row = x + row * row_stride;

    // Each thread accumulates its partial over norm_size elements
    float result;
    switch (reduce_type) {
    case 0: // Sum
    case 3: { // Mean
        float sum = 0.0f;
        for (int i = tid; i < norm_size; i += blockDim.x) {
            sum += s_load(&x_row[i]);
        }
        result = block_reduce_sum(sum, smem);
        if (reduce_type == 3 && tid == 0) {
            result /= static_cast<float>(norm_size);
        }
        break;
    }
    case 1: { // Min
        float val = 1e30f;
        for (int i = tid; i < norm_size; i += blockDim.x) {
            float v = s_load(&x_row[i]);
            if (v < val) val = v;
        }
        result = block_reduce_min(val, smem);
        break;
    }
    case 2: { // Max
        float val = -1e30f;
        for (int i = tid; i < norm_size; i += blockDim.x) {
            float v = s_load(&x_row[i]);
            if (v > val) val = v;
        }
        result = block_reduce_max(val, smem);
        break;
    }
    default:
        result = 0.0f;
        break;
    }

    // Only thread 0 writes the result
    if (tid == 0) {
        s_store(&y[row], result);
    }
}

// ============================================================
// General path kernel: arbitrary axis (scattered access via offsets)
// ============================================================

template <typename T>
__global__ void reduce_general_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t norm_size,
    int64_t outer_stride,
    const int64_t* __restrict__ inner_offsets,
    int reduce_type)
{
    extern __shared__ float smem[];

    const int tid = threadIdx.x;
    const int64_t row = blockIdx.x;

    const T* x_base = x + row * outer_stride;

    float result;
    switch (reduce_type) {
    case 0: // Sum
    case 3: { // Mean
        float sum = 0.0f;
        for (int i = tid; i < norm_size; i += blockDim.x) {
            sum += s_load(&x_base[inner_offsets[i]]);
        }
        result = block_reduce_sum(sum, smem);
        if (reduce_type == 3 && tid == 0) {
            result /= static_cast<float>(norm_size);
        }
        break;
    }
    case 1: { // Min
        float val = 1e30f;
        for (int i = tid; i < norm_size; i += blockDim.x) {
            float v = s_load(&x_base[inner_offsets[i]]);
            if (v < val) val = v;
        }
        result = block_reduce_min(val, smem);
        break;
    }
    case 2: { // Max
        float val = -1e30f;
        for (int i = tid; i < norm_size; i += blockDim.x) {
            float v = s_load(&x_base[inner_offsets[i]]);
            if (v > val) val = v;
        }
        result = block_reduce_max(val, smem);
        break;
    }
    default:
        result = 0.0f;
        break;
    }

    if (tid == 0) {
        s_store(&y[row], result);
    }
}

// ============================================================
// Host-side launcher
// ============================================================

namespace {

int to_int_reduce_type(ReduceType t) {
    switch (t) {
    case ReduceType::Sum:  return 0;
    case ReduceType::Min:  return 1;
    case ReduceType::Max:  return 2;
    case ReduceType::Mean: return 3;
    }
    return 0;
}

template <typename T>
void reduce_cuda_impl(
    const ReduceAttributes& attrs,
    TensorView& output,
    const TensorView& input,
    const ComputeContext& ctx)
{
    const int64_t rank = input.rank();
    int64_t axis = attrs.axis;
    if (axis < 0) axis += rank;

    const int64_t pack = input.channel_pack_size();

    const int64_t outer_dim_count = [&]() {
        int64_t n = 1;
        for (int64_t d = 0; d < axis; ++d) n *= input.shape(d);
        return n;
    }();
    const int64_t norm_size = [&]() {
        int64_t n = 1;
        for (int64_t d = axis; d < rank; ++d) n *= input.shape(d);
        return n;
    }();

    const T* x_ptr = input.ptr<T>();
    T* y_ptr = output.ptr<T>();
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    const int reduce_type = to_int_reduce_type(attrs.type);

    const int block_size = std::min(static_cast<int>(norm_size), kCudaBlockSize);
    const int shared_bytes = block_size * sizeof(float);
    const int grid = std::min(static_cast<int>(outer_dim_count), 65535);

    if (axis == rank - 1 && pack == 1) {
        // Contiguous tail fast path (planar only; packed uses general path)
        const int64_t row_stride = (axis > 0) ? input.stride_elems(axis - 1) : input.numel();
        reduce_fast_kernel<T><<<grid, block_size, shared_bytes, stream>>>(
            x_ptr, y_ptr, norm_size, row_stride, reduce_type);
    } else {
        // General axis: pre-compute inner offsets, upload to device.
        // stride_elems() returns physical strides (pack-aware), so no extra
        // pack multiplication is needed.
        std::vector<int64_t> h_offsets(norm_size);
        for (int64_t flat = 0; flat < norm_size; ++flat) {
            int64_t off = 0;
            int64_t rem = flat;
            for (int64_t d = rank - 1; d >= axis; --d) {
                int64_t dim = input.shape(d);
                off += (rem % dim) * input.stride_elems(d);
                rem /= dim;
            }
            h_offsets[flat] = off;
        }

        int64_t* d_offsets = nullptr;
        cudaMalloc(&d_offsets, norm_size * sizeof(int64_t));
        cudaMemcpyAsync(d_offsets, h_offsets.data(),
                        norm_size * sizeof(int64_t),
                        cudaMemcpyHostToDevice, stream);

        const int64_t outer_stride = (axis > 0) ? input.stride_elems(axis - 1) : input.numel();
        reduce_general_kernel<T><<<grid, block_size, shared_bytes, stream>>>(
            x_ptr, y_ptr, norm_size, outer_stride, d_offsets, reduce_type);

        cudaFree(d_offsets);
    }
}

}  // anonymous namespace

// ============================================================
// Entry point
// ============================================================

void reduce_cuda(const ReduceAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* /*workspace*/)
{
    const auto& input = inputs[0];

    // CUDA supports single-axis reduction only
    switch (input.data_type()) {
    case DataType::f32:
        reduce_cuda_impl<float>(attrs, output, input, ctx);
        return;
    case DataType::f16:
        reduce_cuda_impl<__half>(attrs, output, input, ctx);
        return;
    default:
        NNOPS_ASSERT(false && "Reduce CUDA: unsupported data type");
        break;
    }
}

}  // namespace nnops::backend::cuda
