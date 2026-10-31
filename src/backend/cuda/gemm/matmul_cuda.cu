/// @file matmul_cuda.cu
/// @brief CUDA MatMul — the dense CuTe GEMM behind the batched, transposable
///        operator front end.
///
/// The op resolves the logical (M, N, K) out of the two transpose flags and
/// hands the kernel the physical row strides, so a planar tensor's pitch simply
/// becomes the row stride of whichever physical axis is the row. Batch dims
/// broadcast numpy-style from the right and travel as (size, stride) pairs with
/// stride 0 for a broadcast axis; blockIdx.z walks the flattened product.
///
/// Elementwise contract (mirrors the CPU kernels): C = activate(clamp(beta·C_old
/// + bias + alpha·A·B)), bias indexed by the column and never scaled by alpha,
/// clamp and activation both on the finished sum. f32 and f16 only; the integer
/// pairs and the quantized-activation route return unsupported.

#include "gemm_op.cuh"
#include "nnops/ops/matmul.hpp"

namespace nnops::backend::cuda {

Status matmul_cuda(const MatMulAttributes& attrs, TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx, void*)
{
    const TensorView& a = inputs[0];
    const TensorView& b = inputs[1];
    const TensorView* bias = inputs.size() > 2 ? &inputs[2] : nullptr;

    const DataType dt = a.data_type();
    if ((dt != DataType::f32 && dt != DataType::f16) || b.data_type() != dt) {
        return Status::unsupported;
    }
    if (output.numel() == 0) { return Status::ok; }

    const int64_t M = attrs.transpose_a ? a.shape(a.rank() - 1) : a.shape(a.rank() - 2);
    const int64_t Ka = attrs.transpose_a ? a.shape(a.rank() - 2) : a.shape(a.rank() - 1);
    const int64_t Kb = attrs.transpose_b ? b.shape(b.rank() - 1) : b.shape(b.rank() - 2);
    const int64_t N = attrs.transpose_b ? b.shape(b.rank() - 2) : b.shape(b.rank() - 1);
    if (Ka != Kb) { return Status::invalid_argument; }

    GemmParams p;
    p.a = a.ptr<void>();
    p.b = b.ptr<void>();
    p.c = output.ptr<void>();
    p.bias = bias != nullptr ? bias->ptr<void>() : nullptr;
    p.m = static_cast<int>(M);
    p.n = static_cast<int>(N);
    p.k = static_cast<int>(Ka);
    p.lda = a.row_stride_elems();
    p.ldb = b.row_stride_elems();
    p.ldc = output.row_stride_elems();
    p.alpha = attrs.alpha;
    p.beta = attrs.beta;
    p.epilogue = attrs.epilogue.type;
    p.ep_alpha = attrs.epilogue.alpha;
    p.ep_beta = attrs.epilogue.beta;
    p.min_clip = attrs.epilogue.min_clip;
    p.max_clip = attrs.epilogue.max_clip;

    if (!gemm_batch_layout(a, b, output, nullptr, p)) {
        return Status::unsupported;
    }
    return dense_gemm_status(p, dt, attrs.transpose_a, attrs.transpose_b,
                             static_cast<cudaStream_t>(ctx.cuda_stream));
}

}  // namespace nnops::backend::cuda
