/// @file matmul_q4.cu
/// @brief The w4a16 GEMM's entry point: the block-packed weight, the 16-bit
///        activation, and the pose checks around them.
///
/// Both 16-bit element types are instantiated here — eight kernels in all
/// (four poses each), the same tile and the same decode, so neither type pays
/// for a translation unit of its own.

#include "matmul_q4_kernel.cuh"
#include "matmul_q4_host.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace nnops::backend::cuda {

namespace {

template <class Elem> bool q4_dispatch(const Q4Call& call, cudaStream_t stream)
{
    return q4_run<Q4Tile, Elem>(call, stream);
}

/// The dense struct the split-K reduce and the narrow-m gate read. The seven
/// epilogue fields, the batch axes and the strides are the same in both.
GemmParams q4_gemm_params(const Q4Call& call)
{
    const F16Call& c = call.gemm;
    GemmParams p;
    p.a = c.a;
    p.b = c.b;
    p.c = c.c;
    p.m = c.m;
    p.n = c.n;
    p.k = c.k;
    p.lda = c.lda;
    p.ldb = c.ldb;
    p.ldc = c.ldc;
    p.alpha = c.alpha;
    p.bias = c.bias;
    p.bias_row = c.bias_row;
    p.beta = c.beta;
    p.residual = c.residual;
    p.ldr = c.ldr;
    p.epilogue = static_cast<EpilogueActivateType>(c.epilogue);
    p.ep_alpha = c.ep_alpha;
    p.ep_beta = c.ep_beta;
    p.min_clip = c.min_clip;
    p.max_clip = c.max_clip;
    p.batch_rank = c.batch_rank;
    for (int d = 0; d < 4; ++d) {
        p.batch_size[d] = c.bsize[d];
        p.a_batch[d] = c.a_bstr[d];
        p.b_batch[d] = c.b_bstr[d];
        p.c_batch[d] = c.c_bstr[d];
        p.r_batch[d] = c.r_bstr[d];
    }
    return p;
}

/// The narrow-m GEMV: column-parallel, decode in registers, one f32 plane per
/// split and the dense reduce (which carries the whole epilogue).
template <class Elem> bool q4_launch_small_nt(const Q4Call& call, cudaStream_t stream)
{
    GemmParams p = q4_gemm_params(call);
    const int splits = detail::small_nt_splits(p);
    const int64_t bytes = int64_t(splits) * p.m * p.n * int64_t(sizeof(float));
    const detail::GemmWorkspace ws = detail::gemm_workspace_alloc(bytes, stream);
    if (ws.ptr == nullptr) {
        return false;
    }

    GemmParams q = p;
    q.k_splits = splits;
    q.partial = static_cast<float*>(ws.ptr);
    const int block_bytes = int(data_type_block_bytes(DataType(call.fmt)));
    const dim3 grid(static_cast<unsigned>((p.n + detail::kSmallNTCols - 1) / detail::kSmallNTCols),
                    1, static_cast<unsigned>(splits));
    if (p.m == 1) {
        q4_small_nt_kernel<Elem, 1><<<grid, detail::kSmallNTCols, 0, stream>>>(
            q, call.b_row_blocks, call.fmt, block_bytes);
    } else if (p.m <= 4) {
        q4_small_nt_kernel<Elem, 4><<<grid, detail::kSmallNTCols, 0, stream>>>(
            q, call.b_row_blocks, call.fmt, block_bytes);
    } else {
        q4_small_nt_kernel<Elem, detail::kSmallMRows><<<grid, detail::kSmallNTCols, 0, stream>>>(
            q, call.b_row_blocks, call.fmt, block_bytes);
    }
    detail::launch_splitk_reduce<Elem>(q, stream);
    detail::gemm_workspace_free(ws, stream);
    return cudaGetLastError() == cudaSuccess;
}

/// The pose the column-parallel loop covers: a `Linear` weight, at most
/// sixteen rows, one batch element, and a k deep enough for the loop to have
/// something to cut. Everything else is the tile.
///
/// Where this *beats the dense f16 route* is narrower, and measured at
/// 4096x4096 (ms): 1 row 0.115 against 0.136, 2 rows 0.117 against 0.135, 4
/// rows 0.118 against 0.136, then 8 rows 0.186 against 0.144 and 16 rows 0.178
/// against 0.149 — past four rows the decode's four instructions a value
/// outweigh the bytes it saves. The loop still runs to sixteen because there it
/// beats the q4 tile it would otherwise fall back to (0.178 against 0.237): the
/// choice here is between the block weight's own two engines, not against f16.
/// At N = 32000 even one row is only parity (0.92 against 0.84), so the
/// hidden-size GEMM is where the win lives.
inline bool q4_narrow_applies(const Q4Call& call)
{
    return !call.gemm.trans_a && call.gemm.trans_b && call.gemm.m >= 1 &&
           call.gemm.m <= detail::kSmallMRows && call.gemm.batch == 1 &&
           call.gemm.k >= 1024 && call.gemm.n >= 1;
}

} // namespace

bool q4_dense_call(const Q4Call& call_in, cudaStream_t stream)
{
    if (call_in.gemm.m <= 0 || call_in.gemm.n <= 0 || call_in.gemm.k <= 0) {
        return false;
    }
    if (call_in.fmt != int(DataType::q4_0) && call_in.fmt != int(DataType::q4_1) &&
        call_in.fmt != int(DataType::q8_0)) {
        return false;
    }

    // Only A rides cp.async: the weight's block stream is read with plain loads
    // at the block's own alignment, so it imposes no vector bound.
    Q4Call call = call_in;
    call.gemm.vec = ((reinterpret_cast<uintptr_t>(call.gemm.a) % 16) == 0 &&
                     (reinterpret_cast<uintptr_t>(call.gemm.c) % 16) == 0 &&
                     (call.gemm.lda % 8) == 0)
                        ? 1
                        : 0;

    if (q4_narrow_applies(call)) {
        if (call.gemm.bf16 ? q4_launch_small_nt<__nv_bfloat16>(call, stream)
                           : q4_launch_small_nt<__half>(call, stream)) {
            return true;
        }
        // A split plane that could not be had falls back to the tile, which
        // needs no workspace.
    }

    return call.gemm.bf16 ? q4_dispatch<__nv_bfloat16>(call, stream)
                          : q4_dispatch<__half>(call, stream);
}

} // namespace nnops::backend::cuda
