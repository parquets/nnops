/// @file unary.cu
/// @brief CUDA implementation of element-wise unary math operations.
///
/// Element-wise operations — flat grid-stride loop over all elements,
/// plus a row-by-row kernel for multi-dimensional pitch-aware tensors.
/// All 9 operations: Exp, Log, Sin, Cos, Tan, Tanh, Abs, Neg, Sqrt.
///
/// Supports f32 and f16. Workspace: not used.

#include "cuda_common.cuh"
#include "nnops/ops/unary.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops::backend::cuda {

// ============================================================
// Device unary functions (compute a single element)
// ============================================================

__device__ inline float unary_exp(float x)   { return expf(x); }
__device__ inline float unary_log(float x)   { return logf(x); }
__device__ inline float unary_sin(float x)   { return sinf(x); }
__device__ inline float unary_cos(float x)   { return cosf(x); }
__device__ inline float unary_tan(float x)   { return tanf(x); }
__device__ inline float unary_tanh(float x)  { return tanhf(x); }
__device__ inline float unary_abs(float x)   { return fabsf(x); }
__device__ inline float unary_neg(float x)   { return -x; }
__device__ inline float unary_sqrt(float x)  { return sqrtf(x); }

// ============================================================
// Function pointer type for unary
// ============================================================

using UnaryFn = float (*)(float);

__device__ UnaryFn get_unary_fn(UnaryType type) {
    switch (type) {
    case UnaryType::Exp:  return unary_exp;
    case UnaryType::Log:  return unary_log;
    case UnaryType::Sin:  return unary_sin;
    case UnaryType::Cos:  return unary_cos;
    case UnaryType::Tan:  return unary_tan;
    case UnaryType::Tanh: return unary_tanh;
    case UnaryType::Abs:  return unary_abs;
    case UnaryType::Neg:  return unary_neg;
    case UnaryType::Sqrt: return unary_sqrt;
    }
    return unary_exp;
}

// ============================================================
// Main unary CUDA kernel — row-by-row, pitch-aware
// ============================================================

template <typename T>
__global__ void unary_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t num_rows,
    int64_t last_dim,
    int64_t in_row_stride,
    int64_t out_row_stride,
    UnaryType type,
    bool add_to)
{
    const UnaryFn fn = get_unary_fn(type);

    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    if (row >= num_rows) return;

    const T* in_row  = input  + row * in_row_stride;
    T*       out_row = output + row * out_row_stride;

    for (int i = tid; i < last_dim; i += blockDim.x) {
        float x = s_load(&in_row[i]);
        float rv = fn(x);
        if (add_to) {
            s_store(&out_row[i], s_load(&out_row[i]) + rv);
        } else {
            s_store(&out_row[i], rv);
        }
    }
}

// ============================================================
// Flat kernel for 1D / scalar tensors
// ============================================================

template <typename T>
__global__ void unary_flat_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t total,
    UnaryType type,
    bool add_to)
{
    const UnaryFn fn = get_unary_fn(type);

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < total; i += stride) {
        float x = s_load(&input[i]);
        float rv = fn(x);
        if (add_to) {
            s_store(&output[i], s_load(&output[i]) + rv);
        } else {
            s_store(&output[i], rv);
        }
    }
}

// ============================================================
// Host-side launcher
// ============================================================

template <typename T>
void unary_cuda_impl(
    const UnaryAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t total = input.numel();
    if (total == 0) return;

    const int64_t rank = input.rank();
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    if (rank <= 1) {
        // Flat tensor — use flat kernel with grid-stride loop
        const int block_size = kCudaBlockSize;
        const int grid_size = std::min(
            static_cast<int>(ceil_div(total, static_cast<int64_t>(block_size))),
            65535);
        unary_flat_kernel<<<grid_size, block_size, 0, stream>>>(
            input.ptr<T>(), output.ptr<T>(), total,
            attrs.type, attrs.add_to);
    } else {
        // Row-by-row processing, respecting pitch.
        // For packed layouts, last_dim = W * pack covers all C8 lanes,
        // and num_rows = total_rows() accounts for channel block rounding.
        const int64_t last_dim = input.shape(rank - 1) * input.channel_pack_size();
        const int64_t num_rows = input.total_rows();
        const int64_t in_row_stride  = input.row_stride_elems();
        const int64_t out_row_stride = output.row_stride_elems();

        const int block_size = std::min(static_cast<int>(last_dim), kCudaBlockSize);

        unary_kernel<<<num_rows, block_size, 0, stream>>>(
            input.ptr<T>(), output.ptr<T>(),
            num_rows, last_dim, in_row_stride, out_row_stride,
            attrs.type, attrs.add_to);
    }
}

// ============================================================
// Entry point
// ============================================================

void unary_cuda(
    const UnaryAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx,
    void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        unary_cuda_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        unary_cuda_impl<__half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"unary_cuda: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cuda
