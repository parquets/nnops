/// @file cumsum.cu
/// @brief CUDA implementation of cumulative sum along an axis.
///
/// Algorithm:
///   Decompose tensor along axis into upper (before axis) and lower (after axis)
///   dimension groups. Each independent scan vector has length dim(axis) and
///   is associated with a specific upper-slice-index and lower-element-index.
///
/// Grid:
///   One block per upper slice (upper_dim_count blocks).
///   Within each block, one thread per lower-dim position.
///   Each thread does a sequential scan along the axis.
///
/// Modes: exclusive/inclusive, forward/reverse.
///
/// Supports f32 and f16. Workspace: not used.

#include "cuda_common.cuh"
#include "nnops/ops/cumsum.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops::backend::cuda {

// ============================================================
// CumSum CUDA kernel
// ============================================================

template <typename T>
__global__ void cumsum_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t dim,                 // axis size
    int64_t lower_dim_size,     // elements after axis (treated as vector tail)
    int64_t axis_stride,        // stride_elems(axis)
    bool exclusive,
    bool reverse,
    bool add_to)
{
    // Each block: one outer slice
    // Each thread: one lower-dim position
    const int outer = blockIdx.x;
    const int lower = threadIdx.x;

    if (lower >= lower_dim_size) return;

    const int64_t slice_start = outer * dim * lower_dim_size;

    if (!reverse) {
        // ---- Forward cumulative sum ----
        if (exclusive) {
            // output[0] = 0, output[i] += input[i-1], running sum tracks input
            float running = 0.0f;
            for (int64_t k = 0; k < dim; ++k) {
                int64_t idx = slice_start + k * axis_stride + lower;
                float val = running;  // exclusive: does not include current element
                if (add_to) {
                    s_store(&output[idx], s_load(&output[idx]) + val);
                } else {
                    s_store(&output[idx], val);
                }
                running += s_load(&input[idx]);  // add current to running sum
            }
        } else {
            // output[i] = sum(input[0..i])
            float running = 0.0f;
            for (int64_t k = 0; k < dim; ++k) {
                int64_t idx = slice_start + k * axis_stride + lower;
                running += s_load(&input[idx]);
                if (add_to) {
                    s_store(&output[idx], s_load(&output[idx]) + running);
                } else {
                    s_store(&output[idx], running);
                }
            }
        }
    } else {
        // ---- Reverse cumulative sum ----
        if (exclusive) {
            // output[dim-1] = 0, output[i] += input[i+1], running sum tracks input
            float running = 0.0f;
            for (int64_t k = dim - 1; k >= 0; --k) {
                int64_t idx = slice_start + k * axis_stride + lower;
                float val = running;  // exclusive: does not include current element
                if (add_to) {
                    s_store(&output[idx], s_load(&output[idx]) + val);
                } else {
                    s_store(&output[idx], val);
                }
                running += s_load(&input[idx]);  // add current to running sum
            }
        } else {
            // output[i] = sum(input[i..dim-1])
            float running = 0.0f;
            for (int64_t k = dim - 1; k >= 0; --k) {
                int64_t idx = slice_start + k * axis_stride + lower;
                running += s_load(&input[idx]);
                if (add_to) {
                    s_store(&output[idx], s_load(&output[idx]) + running);
                } else {
                    s_store(&output[idx], running);
                }
            }
        }
    }
}

// ============================================================
// Host-side launcher
// ============================================================

template <typename T>
void cumsum_cuda_impl(
    const CumSumAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    NNOPS_ASSERT(rank >= 1);

    int64_t axis = attrs.axis;
    if (axis < 0) axis += rank;
    NNOPS_ASSERT(axis >= 0 && axis < rank);

    const int64_t dim = input.shape(axis);

    // Number of slices before the axis
    int64_t upper_dim_count = 1;
    for (int64_t i = 0; i < axis; ++i) {
        upper_dim_count *= input.shape(i);
    }

    // Size of the tail after the axis (each treated as an independent scan)
    int64_t lower_dim_size = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        lower_dim_size *= input.shape(i);
    }

    const int64_t axis_stride = input.stride_elems(axis);

    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    const int block_size = std::min(static_cast<int>(lower_dim_size), kCudaBlockSize);
    const int grid_size = static_cast<int>(upper_dim_count);

    cumsum_kernel<<<grid_size, block_size, 0, stream>>>(
        input.ptr<T>(), output.ptr<T>(),
        dim, lower_dim_size, axis_stride,
        attrs.exclusive, attrs.reverse, attrs.add_to);
}

// ============================================================
// Entry point
// ============================================================

void cumsum_cuda(
    const CumSumAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx,
    void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        cumsum_cuda_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        cumsum_cuda_impl<__half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"cumsum_cuda: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cuda
