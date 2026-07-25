/// @file eltwise.cu
/// @brief CUDA implementation of element-wise binary operations.
///
/// Element-wise operations — flat grid-stride loop over all elements.
/// All 4 operations: Add, Sub, Mul, Div.
///
/// Supports f32 and f16. Workspace: not used.

#include "cuda_common.cuh"
#include "nnops/ops/eltwise.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops::backend::cuda {

// ============================================================
// CUDA kernel — flat grid-stride loop
// ============================================================

template <typename T>
__global__ void eltwise_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ output,
    int64_t total,
    EltwiseType op_type,
    bool add_to)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    switch (op_type) {
    case EltwiseType::Add: {
        for (int i = tid; i < total; i += stride) {
            float rv = s_load(&a[i]) + s_load(&b[i]);
            if (add_to) {
                s_store(&output[i], s_load(&output[i]) + rv);
            } else {
                s_store(&output[i], rv);
            }
        }
        break;
    }
    case EltwiseType::Sub: {
        for (int i = tid; i < total; i += stride) {
            float rv = s_load(&a[i]) - s_load(&b[i]);
            if (add_to) {
                s_store(&output[i], s_load(&output[i]) + rv);
            } else {
                s_store(&output[i], rv);
            }
        }
        break;
    }
    case EltwiseType::Mul: {
        for (int i = tid; i < total; i += stride) {
            float rv = s_load(&a[i]) * s_load(&b[i]);
            if (add_to) {
                s_store(&output[i], s_load(&output[i]) + rv);
            } else {
                s_store(&output[i], rv);
            }
        }
        break;
    }
    case EltwiseType::Div: {
        for (int i = tid; i < total; i += stride) {
            float rv = s_load(&a[i]) / s_load(&b[i]);
            if (add_to) {
                s_store(&output[i], s_load(&output[i]) + rv);
            } else {
                s_store(&output[i], rv);
            }
        }
        break;
    }
    }
}

// ============================================================
// Host-side launcher
// ============================================================

template <typename T>
void eltwise_cuda_impl(
    const EltwiseAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    const auto& A = inputs[0];
    const auto& B = inputs[1];
    const int64_t total = A.numel();
    if (total == 0) return;
    NNOPS_ASSERT(A.numel() == B.numel());
    NNOPS_ASSERT(output.numel() == total);

    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    const int block_size = kCudaBlockSize;
    const int grid_size = std::min(
        static_cast<int>(ceil_div(total, static_cast<int64_t>(block_size))),
        65535);

    eltwise_kernel<<<grid_size, block_size, 0, stream>>>(
        A.ptr<T>(), B.ptr<T>(), output.ptr<T>(),
        total, attrs.type, attrs.add_to);
}

// ============================================================
// Entry point
// ============================================================

void eltwise_cuda(
    const EltwiseAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx,
    void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        eltwise_cuda_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        eltwise_cuda_impl<__half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"eltwise_cuda: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cuda
