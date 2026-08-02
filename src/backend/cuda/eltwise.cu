/// @file eltwise.cu
/// @brief CUDA implementation of element-wise binary operations.
///
/// Element-wise operations — flat grid-stride loop for rank < 2,
/// row-by-row pitch-aware kernel for rank >= 2 (supports packed layouts).
/// All 4 operations: Add, Sub, Mul, Div.
///
/// Supports f32 and f16. Workspace: not used.

#include "cuda_common.cuh"
#include "nnops/ops/eltwise.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops::backend::cuda {

// ============================================================
// Row-by-row CUDA kernel — pitch-aware (rank >= 2)
// ============================================================

template <typename T>
__global__ void eltwise_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ output,
    int64_t num_rows,
    int64_t last_dim,
    int64_t a_row_stride,
    int64_t b_row_stride,
    int64_t o_row_stride,
    EltwiseType op_type,
    bool add_to)
{
    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    if (row >= num_rows) return;

    const T* a_row = a + row * a_row_stride;
    const T* b_row = b + row * b_row_stride;
    T* o_row = output + row * o_row_stride;

    switch (op_type) {
    case EltwiseType::Add: {
        for (int i = tid; i < last_dim; i += blockDim.x) {
            float rv = s_load(&a_row[i]) + s_load(&b_row[i]);
            if (add_to) {
                s_store(&o_row[i], s_load(&o_row[i]) + rv);
            } else {
                s_store(&o_row[i], rv);
            }
        }
        break;
    }
    case EltwiseType::Sub: {
        for (int i = tid; i < last_dim; i += blockDim.x) {
            float rv = s_load(&a_row[i]) - s_load(&b_row[i]);
            if (add_to) {
                s_store(&o_row[i], s_load(&o_row[i]) + rv);
            } else {
                s_store(&o_row[i], rv);
            }
        }
        break;
    }
    case EltwiseType::Mul: {
        for (int i = tid; i < last_dim; i += blockDim.x) {
            float rv = s_load(&a_row[i]) * s_load(&b_row[i]);
            if (add_to) {
                s_store(&o_row[i], s_load(&o_row[i]) + rv);
            } else {
                s_store(&o_row[i], rv);
            }
        }
        break;
    }
    case EltwiseType::Div: {
        for (int i = tid; i < last_dim; i += blockDim.x) {
            float rv = s_load(&a_row[i]) / s_load(&b_row[i]);
            if (add_to) {
                s_store(&o_row[i], s_load(&o_row[i]) + rv);
            } else {
                s_store(&o_row[i], rv);
            }
        }
        break;
    }
    case EltwiseType::Min: {
        for (int i = tid; i < last_dim; i += blockDim.x) {
            float av = s_load(&a_row[i]);
            float bv = s_load(&b_row[i]);
            float rv = av < bv ? av : bv;
            if (add_to) {
                s_store(&o_row[i], s_load(&o_row[i]) + rv);
            } else {
                s_store(&o_row[i], rv);
            }
        }
        break;
    }
    case EltwiseType::Max: {
        for (int i = tid; i < last_dim; i += blockDim.x) {
            float av = s_load(&a_row[i]);
            float bv = s_load(&b_row[i]);
            float rv = av > bv ? av : bv;
            if (add_to) {
                s_store(&o_row[i], s_load(&o_row[i]) + rv);
            } else {
                s_store(&o_row[i], rv);
            }
        }
        break;
    }
    }
}

// ============================================================
// Flat kernel for 1D / scalar tensors
// ============================================================

template <typename T>
__global__ void eltwise_flat_kernel(
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
    case EltwiseType::Min: {
        for (int i = tid; i < total; i += stride) {
            float av = s_load(&a[i]);
            float bv = s_load(&b[i]);
            float rv = av < bv ? av : bv;
            if (add_to) {
                s_store(&output[i], s_load(&output[i]) + rv);
            } else {
                s_store(&output[i], rv);
            }
        }
        break;
    }
    case EltwiseType::Max: {
        for (int i = tid; i < total; i += stride) {
            float av = s_load(&a[i]);
            float bv = s_load(&b[i]);
            float rv = av > bv ? av : bv;
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

    const int64_t rank = A.rank();
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    if (rank <= 1) {
        // Flat tensor — use flat kernel with grid-stride loop
        const int block_size = kCudaBlockSize;
        const int grid_size = std::min(
            static_cast<int>(ceil_div(total, static_cast<int64_t>(block_size))),
            65535);
        eltwise_flat_kernel<<<grid_size, block_size, 0, stream>>>(
            A.ptr<T>(), B.ptr<T>(), output.ptr<T>(),
            total, attrs.type, attrs.add_to);
    } else {
        // Row-by-row processing, respecting pitch.
        // For packed layouts, last_dim = W * pack covers all C8 lanes,
        // and num_rows = total_rows() accounts for channel block rounding.
        const int64_t last_dim = A.shape(rank - 1) * A.channel_pack_size();
        const int64_t num_rows = A.total_rows();
        const int64_t a_row_stride = A.row_stride_elems();
        const int64_t b_row_stride = B.row_stride_elems();
        const int64_t o_row_stride = output.row_stride_elems();

        const int block_size = std::min(static_cast<int>(last_dim), kCudaBlockSize);

        eltwise_kernel<<<num_rows, block_size, 0, stream>>>(
            A.ptr<T>(), B.ptr<T>(), output.ptr<T>(),
            num_rows, last_dim, a_row_stride, b_row_stride, o_row_stride,
            attrs.type, attrs.add_to);
    }
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
