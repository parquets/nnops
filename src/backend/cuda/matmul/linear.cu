/// @file linear.cu
/// @brief CUDA Linear — the same dense GEMM as MatMul, with a [N, K] weight, a
///        [N] bias and the residual applied after the activation.
///
/// Linear's add_to adds the old output *after* the epilogue, so the kernel takes
/// it as the post-activation residual rather than as beta·C_old: the residual is
/// in place in the output buffer and each thread reads its element before
/// overwriting it. There is no alpha or beta on this operator — the activation
/// is everything the caller asked for. f32, f16 and bf16; an integer pair takes
/// matmul_i8.cuh's MatMulInteger route (s32, s8/u8, f32/f16 destinations), but
/// not add_to — an integer residual is not folded there yet.

#include "matmul_i8.cuh"
#include "matmul_f32_host.hpp"
#include "matmul_f16_host.hpp"
#include "detail/bench_knobs.hpp"
#include "matmul_op.cuh"
#include "nnops/ops/linear.hpp"
#include "nnops/ops/matmul.hpp" // is_int8_matmul

namespace nnops::backend::cuda {

namespace {

/// The integer route: x [.., M, K] × w [N, K]ᵀ, MatMulInteger semantics. The
/// residual (add_to) is not folded into an integer destination yet.
Status linear_i8_cuda(const LinearAttributes& attrs, TensorView& output,
                      std::span<const TensorView> inputs, const ComputeContext& ctx, I8OutKind kind)
{
    const TensorView& x = inputs[0];
    const TensorView& w = inputs[1];
    const TensorView* bias = inputs.size() > 2 ? &inputs[2] : nullptr;

    if (attrs.add_to) {
        return Status::unsupported;
    }

    const int64_t M = x.shape(x.rank() - 2);
    const int64_t K = x.shape(x.rank() - 1);
    const int64_t N = w.shape(0);
    if (w.shape(1) != K) {
        return Status::invalid_argument;
    }

    GemmI8Params q;
    q.g.a = x.ptr<void>();
    q.g.b = w.ptr<void>();
    q.g.c = output.ptr<void>();
    q.g.m = static_cast<int>(M);
    q.g.n = static_cast<int>(N);
    q.g.k = static_cast<int>(K);
    q.g.lda = x.row_stride_elems();
    q.g.ldb = w.row_stride_elems();
    q.g.ldc = output.row_stride_elems();
    q.g.epilogue = attrs.epilogue.type;
    q.g.ep_alpha = attrs.epilogue.alpha;
    q.g.ep_beta = attrs.epilogue.beta;
    q.g.min_clip = attrs.epilogue.min_clip;
    q.g.max_clip = attrs.epilogue.max_clip;

    if (!gemm_batch_layout(x, w, output, nullptr, q.g)) {
        return Status::unsupported;
    }

    const QuantParams& qx = x.quant_params();
    const QuantParams& qw = w.quant_params();
    q.zp_a = qx.zero_point_data;
    q.zp_a_scalar = qx.zero_point;
    q.sc_a = qx.scale_data;
    q.sc_a_scalar = qx.scale;
    q.zp_b = qw.zero_point_data;
    q.zp_b_scalar = qw.zero_point;
    q.sc_b = qw.scale_data;
    q.sc_b_scalar = qw.scale;

    q.out = kind;
    const bool int_out = kind == I8OutKind::S32 || kind == I8OutKind::S8 || kind == I8OutKind::U8;
    q.relu = int_out && attrs.epilogue.type == EpilogueActivateType::Relu;
    if (int_out) {
        const QuantParams& qc = output.quant_params();
        q.out_scale = qc.scale_data != nullptr ? qc.scale_data[0] : qc.scale;
        q.out_zp = qc.zero_point_data != nullptr ? qc.zero_point_data[0] : qc.zero_point;
    }
    if (bias != nullptr) {
        q.bias = int_out ? bias->ptr<int32_t>() : bias->ptr<void>();
    }

    return dense_gemm_i8_status(q, x.data_type(), /*transpose_a=*/false, /*transpose_b=*/true,
                                static_cast<cudaStream_t>(ctx.cuda_stream));
}

/// The w4a16 route: a 16-bit activation against a llama.cpp Q4_0/Q4_1/Q8_0
/// `[N, K]` weight. add_to is the post-activation residual, as on the dense
/// path, so it reaches the call as the epilogue's residual rather than beta.
Status linear_q4_cuda(const LinearAttributes& attrs, TensorView& output,
                      std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const TensorView& x = inputs[0];
    const TensorView& w = inputs[1];
    const TensorView* bias = inputs.size() > 2 ? &inputs[2] : nullptr;

    const int64_t M = x.shape(x.rank() - 2);
    const int64_t K = x.shape(x.rank() - 1);
    const int64_t N = w.shape(0);
    if (w.shape(1) != K) {
        return Status::invalid_argument;
    }

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

    const DataType dt = x.data_type();
    return q4_gemm_status(
        make_q4_call(p, /*trans_a=*/false, /*trans_b=*/true, dt == DataType::bf16,
                     w.data_type()),
        dt, static_cast<cudaStream_t>(ctx.cuda_stream));
}

} // namespace

Status linear_cuda(const LinearAttributes& attrs, TensorView& output,
                   std::span<const TensorView> inputs, const ComputeContext& ctx, void*)
{
    const TensorView& x = inputs[0];
    const TensorView& w = inputs[1];
    const TensorView* bias = inputs.size() > 2 ? &inputs[2] : nullptr;

    // The SwiGLU combine has no CUDA route yet; the op is CPU-only in that form.
    if (attrs.gated) {
        return Status::unsupported;
    }
    const DataType dt = x.data_type();
    if (is_int8_matmul(dt, w.data_type())) {
        if (output.numel() == 0) {
            return Status::ok;
        }
        I8OutKind kind;
        if (!i8_out_kind(output.data_type(), kind)) {
            return Status::unsupported;
        }
        return linear_i8_cuda(attrs, output, inputs, ctx, kind);
    }
    if (is_block_packed_dtype(w.data_type())) {
        if (output.numel() == 0) {
            return Status::ok;
        }
        return linear_q4_cuda(attrs, output, inputs, ctx);
    }
    if ((dt != DataType::f32 && dt != DataType::f16 && dt != DataType::bf16) ||
        w.data_type() != dt) {
        return Status::unsupported;
    }
    if (output.numel() == 0) {
        return Status::ok;
    }

    const int64_t M = x.shape(x.rank() - 2);
    const int64_t K = x.shape(x.rank() - 1);
    const int64_t N = w.shape(0);
    if (w.shape(1) != K) {
        return Status::invalid_argument;
    }

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

    // Same ladder as MatMul: Linear's weight is already [N, K], so the pose is
    // the n-major one throughout.
    return dense_gemm_status(p, dt, /*transpose_a=*/false, /*transpose_b=*/true,
                             static_cast<cudaStream_t>(ctx.cuda_stream));
}

} // namespace nnops::backend::cuda
