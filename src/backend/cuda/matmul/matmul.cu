/// @file matmul.cu
/// @brief CUDA MatMul — the dense CuTe GEMM behind the batched, transposable
///        operator front end.
///
/// The op resolves the logical (M, N, K) out of the two transpose flags and
/// hands the kernel the physical row strides, so a pitched tensor's pitch simply
/// becomes the row stride of whichever physical axis is the row. Batch dims
/// broadcast numpy-style from the right and travel as (size, stride) pairs with
/// stride 0 for a broadcast axis; blockIdx.z walks the flattened product.
///
/// The ceiling is rank six (four batch axes), and the flattened batch has to fit
/// the 16-bit blockIdx.z — the front end refuses both rather than truncating, so
/// such a call reports unsupported instead of failing at launch.
///
/// Elementwise contract (mirrors the CPU kernels): C = activate(clamp(beta·C_old
/// + bias + alpha·A·B)), bias indexed by the column and never scaled by alpha,
/// clamp and activation both on the finished sum. f32, f16 and bf16 take that
/// route; an integer pair (u8|s8 × s8) takes matmul_i8.cuh and writes the
/// MatMulInteger destinations (s32, s8/u8 requantized, f32/f16 dequantized).
/// The fp-activation-with-QuantParams form of the integer GEMM stays on the CPU.

#include "matmul_i8.cuh"
#include "matmul_f32_host.hpp"
#include "matmul_f16_host.hpp"
#include "nnops/detail/bench_knobs.hpp"
#include "matmul_op.cuh"
#include "nnops/ops/matmul.hpp"

namespace nnops::backend::cuda {

namespace {

/// The integer route's operand/quantization contract, shared by MatMul and
/// Linear (which differ only in transpose_b and where the bias comes from).
Status matmul_i8_cuda(const MatMulAttributes& attrs, TensorView& output,
                      std::span<const TensorView> inputs, const ComputeContext& ctx, I8OutKind kind)
{
    const TensorView& a = inputs[0];
    const TensorView& b = inputs[1];
    const TensorView* bias = inputs.size() > 2 ? &inputs[2] : nullptr;

    const int64_t M = attrs.transpose_a ? a.shape(a.rank() - 1) : a.shape(a.rank() - 2);
    const int64_t Ka = attrs.transpose_a ? a.shape(a.rank() - 2) : a.shape(a.rank() - 1);
    const int64_t Kb = attrs.transpose_b ? b.shape(b.rank() - 1) : b.shape(b.rank() - 2);
    const int64_t N = attrs.transpose_b ? b.shape(b.rank() - 2) : b.shape(b.rank() - 1);
    if (Ka != Kb) {
        return Status::invalid_argument;
    }

    GemmI8Params q;
    q.g.a = a.ptr<void>();
    q.g.b = b.ptr<void>();
    q.g.c = output.ptr<void>();
    q.g.m = static_cast<int>(M);
    q.g.n = static_cast<int>(N);
    q.g.k = static_cast<int>(Ka);
    q.g.lda = a.row_stride_elems();
    q.g.ldb = b.row_stride_elems();
    q.g.ldc = output.row_stride_elems();
    q.g.epilogue = attrs.epilogue.type;
    q.g.ep_alpha = attrs.epilogue.alpha;
    q.g.ep_beta = attrs.epilogue.beta;
    q.g.min_clip = attrs.epilogue.min_clip;
    q.g.max_clip = attrs.epilogue.max_clip;

    if (!gemm_batch_layout(a, b, output, nullptr, q.g)) {
        return Status::unsupported;
    }

    const QuantParams& qa = a.quant_params();
    const QuantParams& qb = b.quant_params();
    q.zp_a = qa.zero_point_data;
    q.zp_a_scalar = qa.zero_point;
    q.sc_a = qa.scale_data;
    q.sc_a_scalar = qa.scale;
    q.zp_b = qb.zero_point_data;
    q.zp_b_scalar = qb.zero_point;
    q.sc_b = qb.scale_data;
    q.sc_b_scalar = qb.scale;

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

    return dense_gemm_i8_status(q, a.data_type(), attrs.transpose_a, attrs.transpose_b,
                                static_cast<cudaStream_t>(ctx.cuda_stream));
}

} // namespace

Status matmul_cuda(const MatMulAttributes& attrs, TensorView& output,
                   std::span<const TensorView> inputs, const ComputeContext& ctx, void*)
{
    const TensorView& a = inputs[0];
    const TensorView& b = inputs[1];
    const TensorView* bias = inputs.size() > 2 ? &inputs[2] : nullptr;

    const DataType dt = a.data_type();
    if (is_int8_matmul(dt, b.data_type())) {
        if (output.numel() == 0) {
            return Status::ok;
        }
        I8OutKind kind;
        if (!i8_out_kind(output.data_type(), kind)) {
            return Status::unsupported;
        }
        return matmul_i8_cuda(attrs, output, inputs, ctx, kind);
    }
    if ((dt != DataType::f32 && dt != DataType::f16 && dt != DataType::bf16) ||
        b.data_type() != dt) {
        return Status::unsupported;
    }
    if (output.numel() == 0) {
        return Status::ok;
    }

    const int64_t M = attrs.transpose_a ? a.shape(a.rank() - 1) : a.shape(a.rank() - 2);
    const int64_t Ka = attrs.transpose_a ? a.shape(a.rank() - 2) : a.shape(a.rank() - 1);
    const int64_t Kb = attrs.transpose_b ? b.shape(b.rank() - 1) : b.shape(b.rank() - 2);
    const int64_t N = attrs.transpose_b ? b.shape(b.rank() - 2) : b.shape(b.rank() - 1);
    if (Ka != Kb) {
        return Status::invalid_argument;
    }

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

    // The hand-written PTX f16 kernel (64x32 per warp) owns the TN pose once the
    // grid can fill the machine; below that the CuTe family's split-K does. It
    // covers alpha 1, no bias, no epilogue, one batch element.
    const int ptx_h = ::nnops::detail::cuda_ptx_h.load(std::memory_order_relaxed);
    if ((dt == DataType::f16 || dt == DataType::bf16) && ptx_h >= 0) {
        {
            ManualHCall call;
            call.a = p.a;
            call.b = p.b;
            call.c = p.c;
            call.m = p.m;
            call.n = p.n;
            call.k = p.k;
            call.lda = p.lda;
            call.ldb = p.ldb;
            call.ldc = p.ldc;
            call.trans_a = attrs.transpose_a;
            call.trans_b = attrs.transpose_b;
            call.bf16 = dt == DataType::bf16;
            call.alpha = attrs.alpha;
            call.bias = p.bias;
            call.bias_row = p.bias_row;
            call.beta = attrs.beta;
            call.residual = p.residual;
            call.ldr = p.ldr;
            call.epilogue = int(attrs.epilogue.type);
            call.ep_alpha = attrs.epilogue.alpha;
            call.ep_beta = attrs.epilogue.beta;
            call.min_clip = attrs.epilogue.min_clip;
            call.max_clip = attrs.epilogue.max_clip;
            call.batch_rank = p.batch_rank;
            call.batch = detail::gemm_batch_count(p);
            for (int d = 0; d < 4; ++d) {
                call.bsize[d] = p.batch_size[d];
                call.a_bstr[d] = p.a_batch[d];
                call.b_bstr[d] = p.b_batch[d];
                call.c_bstr[d] = p.c_batch[d];
            }
            if (manual_dense_h_call(call, static_cast<cudaStream_t>(ctx.cuda_stream))) {
                return Status::ok;
            }
        }
    }

    // The hand-written f32 SIMT tile, on the same terms as the 16-bit one: the
    // dense unadorned product, once the grid can fill the machine.
    const int ptx_f = ::nnops::detail::cuda_ptx_f.load(std::memory_order_relaxed);
    if (dt == DataType::f32 && ptx_f >= 0) {
        ManualFCall call;
        call.a = p.a;
        call.b = p.b;
        call.c = p.c;
        call.m = p.m;
        call.n = p.n;
        call.k = p.k;
        call.lda = p.lda;
        call.ldb = p.ldb;
        call.ldc = p.ldc;
        call.trans_a = attrs.transpose_a;
        call.trans_b = attrs.transpose_b;
        call.alpha = attrs.alpha;
        call.bias = p.bias;
        call.bias_row = p.bias_row;
        call.beta = attrs.beta;
        call.residual = p.residual;
        call.ldr = p.ldr;
        call.epilogue = int(attrs.epilogue.type);
        call.ep_alpha = attrs.epilogue.alpha;
        call.ep_beta = attrs.epilogue.beta;
        call.min_clip = attrs.epilogue.min_clip;
        call.max_clip = attrs.epilogue.max_clip;
        call.batch_rank = p.batch_rank;
        call.batch = detail::gemm_batch_count(p);
        for (int d = 0; d < 4; ++d) {
            call.bsize[d] = p.batch_size[d];
            call.a_bstr[d] = p.a_batch[d];
            call.b_bstr[d] = p.b_batch[d];
            call.c_bstr[d] = p.c_batch[d];
        }
        if (manual_dense_f_call(call, static_cast<cudaStream_t>(ctx.cuda_stream))) {
            return Status::ok;
        }
    }

    return dense_gemm_status(p, dt, attrs.transpose_a, attrs.transpose_b,
                             static_cast<cudaStream_t>(ctx.cuda_stream));
}

} // namespace nnops::backend::cuda
