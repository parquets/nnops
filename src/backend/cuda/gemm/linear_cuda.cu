/// @file linear_cuda.cu
/// @brief CUDA Linear — the same dense GEMM as MatMul, with a [N, K] weight, a
///        [N] bias and the residual applied after the activation.
///
/// Linear's add_to adds the old output *after* the epilogue, so the kernel takes
/// it as the post-activation residual rather than as beta·C_old: the residual is
/// in place in the output buffer and each thread reads its element before
/// overwriting it. There is no alpha or beta on this operator — the activation
/// is everything the caller asked for. f32 and f16 only.

#include "gemm_op.cuh"
#include "nnops/ops/linear.hpp"

namespace nnops::backend::cuda {

Status linear_cuda(const LinearAttributes& attrs, TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx, void*)
{
    const TensorView& x = inputs[0];
    const TensorView& w = inputs[1];
    const TensorView* bias = inputs.size() > 2 ? &inputs[2] : nullptr;

    const DataType dt = x.data_type();
    if ((dt != DataType::f32 && dt != DataType::f16) || w.data_type() != dt) {
        return Status::unsupported;
    }
    if (output.numel() == 0) { return Status::ok; }

    const int64_t M = x.shape(x.rank() - 2);
    const int64_t K = x.shape(x.rank() - 1);
    const int64_t N = w.shape(0);
    if (w.shape(1) != K) { return Status::invalid_argument; }

    GemmParams p;
    p.a = x.ptr<void>();
    p.b = w.ptr<void>();
    p.c = output.ptr<void>();
    p.bias = bias != nullptr ? bias->ptr<void>() : nullptr;
    p.m = static_cast<int>(M);
    p.n = static_cast<int>(N);
    p.k = static_cast<int>(K);
    p.lda = x.row_stride_elems();
    p.ldb = w.row_stride_elems();
    p.ldc = output.row_stride_elems();
    p.alpha = 1.0f;
    p.beta = 0.0f;
    p.epilogue = attrs.epilogue.type;
    p.ep_alpha = attrs.epilogue.alpha;
    p.ep_beta = attrs.epilogue.beta;
    p.min_clip = attrs.epilogue.min_clip;
    p.max_clip = attrs.epilogue.max_clip;

    if (attrs.add_to) {
        p.residual = output.ptr<void>();
        p.ldr = p.ldc;
    }

    if (!gemm_batch_layout(x, w, output, attrs.add_to ? &output : nullptr, p)) {
        return Status::unsupported;
    }
    return dense_gemm_status(p, dt, /*transpose_a=*/false, /*transpose_b=*/true,
                             static_cast<cudaStream_t>(ctx.cuda_stream));
}

}  // namespace nnops::backend::cuda
