/// @file matmul_op.cuh
/// @brief Shared front end of the dense GEMM operators — batch broadcasting,
///        the 16 B vectorisable test, and the launch wrapper that turns a
///        configured GemmParams into a Status.
///
/// MatMul and Linear differ only in how they read their operands (Linear is
/// MatMul with transpose_b and a rank-2 weight), so both resolve their shapes
/// and strides here and nothing else.

#pragma once

#include "matmul_f16_host.hpp"
#include "matmul_f32_host.hpp"
#include "matmul_narrow.cuh"
#include "nnops/core/data_type.hpp"
#include "nnops/core/status.hpp"
#include "nnops/core/tensor_view.hpp"

#include <algorithm>

namespace nnops::backend::cuda {

/// Fill the broadcast batch shape and the per-axis batch strides of every
/// operand. `c` (and the optional residual `r`) supply their own strides;
/// a broadcast axis carries stride 0, so every batch slice reads the same data.
///
/// The call is refused rather than truncated when it has more than four batch
/// axes (rank above six) or when the flattened batch outgrows a 16-bit grid
/// axis: the batch rides blockIdx.z in the tile kernels, multiplied by the k
/// splits, and blockIdx.y in the split-K reduce pass.
inline bool gemm_batch_layout(const TensorView& a, const TensorView& b, const TensorView& c,
                              const TensorView* r, GemmParams& p)
{
    const int64_t al = a.rank() - 2;
    const int64_t bl = b.rank() - 2;
    const int64_t rank = std::max<int64_t>(al, bl);
    if (rank > kGemmMaxBatchRank) {
        return false;
    }

    p.batch_rank = static_cast<int>(rank);
    int64_t count = 1;
    for (int64_t d = 0; d < rank; ++d) {
        const int64_t ai = al - 1 - d;
        const int64_t bi = bl - 1 - d;
        const int64_t ad = ai >= 0 ? a.shape(ai) : 1;
        const int64_t bd = bi >= 0 ? b.shape(bi) : 1;
        if (ad != bd && ad != 1 && bd != 1) {
            return false;
        }

        const int64_t slot = rank - 1 - d;
        p.batch_size[slot] = std::max(ad, bd);
        p.a_batch[slot] = (ad == 1) ? 0 : a.stride_elems(ai);
        p.b_batch[slot] = (bd == 1) ? 0 : b.stride_elems(bi);

        const int64_t ci = c.rank() - 3 - d;
        p.c_batch[slot] = ci >= 0 ? c.stride_elems(ci) : 0;
        p.r_batch[slot] = (r != nullptr && ci >= 0) ? r->stride_elems(ci) : 0;

        // Capping each axis first keeps the running product inside int64 and
        // makes the two checks below the same bound.
        if (p.batch_size[slot] > kGemmMaxGridAxis) {
            return false;
        }
        count *= p.batch_size[slot];
        if (count > kGemmMaxGridAxis) {
            return false;
        }
    }
    return true;
}

/// A 16 B cp.async needs every vector address aligned: the two base pointers
/// and each row stride (the k and batch offsets are multiples of the row
/// stride). Anything else takes the scalar-fill kernel.
inline bool gemm_vectorizable(const GemmParams& p, int64_t elem_size)
{
    const int64_t vec = 16 / elem_size;
    return (reinterpret_cast<uintptr_t>(p.a) % 16 == 0) &&
           (reinterpret_cast<uintptr_t>(p.b) % 16 == 0) && (p.lda % vec == 0) && (p.ldb % vec == 0);
}

/// The hand-written 16-bit engine's view of a configured product.
inline F16Call make_f16_call(const GemmParams& p, bool trans_a, bool trans_b, bool bf16)
{
    F16Call call;
    call.a = p.a;
    call.b = p.b;
    call.c = p.c;
    call.m = p.m;
    call.n = p.n;
    call.k = p.k;
    call.lda = p.lda;
    call.ldb = p.ldb;
    call.ldc = p.ldc;
    call.alpha = p.alpha;
    call.trans_a = trans_a;
    call.trans_b = trans_b;
    call.bf16 = bf16;
    call.bias = p.bias;
    call.bias_row = p.bias_row;
    call.beta = p.beta;
    call.residual = p.residual;
    call.ldr = p.ldr;
    call.epilogue = int(p.epilogue);
    call.ep_alpha = p.ep_alpha;
    call.ep_beta = p.ep_beta;
    call.min_clip = p.min_clip;
    call.max_clip = p.max_clip;
    call.batch_rank = p.batch_rank;
    call.batch = detail::gemm_batch_count(p);
    for (int d = 0; d < 4; ++d) {
        call.bsize[d] = p.batch_size[d];
        call.a_bstr[d] = p.a_batch[d];
        call.b_bstr[d] = p.b_batch[d];
        call.c_bstr[d] = p.c_batch[d];
    }
    return call;
}

/// The hand-written f32 engine's view of the same product.
inline F32Call make_f32_call(const GemmParams& p, bool trans_a, bool trans_b)
{
    F32Call call;
    call.a = p.a;
    call.b = p.b;
    call.c = p.c;
    call.m = p.m;
    call.n = p.n;
    call.k = p.k;
    call.lda = p.lda;
    call.ldb = p.ldb;
    call.ldc = p.ldc;
    call.alpha = p.alpha;
    call.trans_a = trans_a;
    call.trans_b = trans_b;
    call.bias = p.bias;
    call.bias_row = p.bias_row;
    call.beta = p.beta;
    call.residual = p.residual;
    call.ldr = p.ldr;
    call.epilogue = int(p.epilogue);
    call.ep_alpha = p.ep_alpha;
    call.ep_beta = p.ep_beta;
    call.min_clip = p.min_clip;
    call.max_clip = p.max_clip;
    call.batch_rank = p.batch_rank;
    call.batch = detail::gemm_batch_count(p);
    for (int d = 0; d < 4; ++d) {
        call.bsize[d] = p.batch_size[d];
        call.a_bstr[d] = p.a_batch[d];
        call.b_bstr[d] = p.b_batch[d];
        call.c_bstr[d] = p.c_batch[d];
    }
    return call;
}

/// Launch a fully configured dense GEMM; f32, f16 and bf16 only.
///
/// Three engines, tried in the order that made them fastest: the tiled
/// hand-written kernel while its grid fills the machine, the narrow-m pair for
/// a product of at most sixteen rows, and then the tiled kernel again with its
/// fill gate bypassed. The last step is what makes every pose reachable: a
/// shape the gate declines still runs, just at the rate its grid allows.
///
/// `tile` names a row of the f16/f32 candidate table and skips the pose's own
/// pick; -1 (every operator caller) leaves that pick alone. A caller that knows
/// its shape better than the table does — Conv2D's 1x1 route, whose products are
/// short-K and narrow-N — is what the argument is for.
inline Status dense_gemm_status(const GemmParams& p, DataType dtype, bool transpose_a,
                                bool transpose_b, cudaStream_t stream, int tile = -1)
{
    const GemmElem elem = dtype == DataType::f32  ? GemmElem::F32 :
                          dtype == DataType::f16  ? GemmElem::F16 :
                          dtype == DataType::bf16 ? GemmElem::BF16 :
                                                    GemmElem::F32;
    if (dtype != DataType::f32 && dtype != DataType::f16 && dtype != DataType::bf16) {
        return Status::unsupported;
    }
    if (p.m <= 0 || p.n <= 0 || p.k <= 0) {
        return Status::unsupported;
    }
    // The f32 tile walks the n axis on blockIdx.y, which is 16-bit. 64 is the
    // narrowest n tile, so this bound covers both of them (and the f16 tile,
    // whose grid is flat along x, never hits it).
    if ((int64_t(p.n) + 63) / 64 > kGemmMaxGridAxis) {
        return Status::unsupported;
    }

    const int64_t elem_size = static_cast<int64_t>(data_type_size(dtype));
    const bool vectorized = gemm_vectorizable(p, elem_size);

    // The tiled kernel, while its grid fills the machine.
    if (dtype == DataType::f32) {
        F32Call call = make_f32_call(p, transpose_a, transpose_b);
        call.tile = tile;
        if (f32_dense_call(call, stream)) {
            return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
        }
    } else {
        F16Call call = make_f16_call(p, transpose_a, transpose_b, dtype == DataType::bf16);
        call.tile = tile;
        if (f16_dense_call(call, stream)) {
            return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
        }
    }

    // A product of at most sixteen rows is a bandwidth problem; the narrow-m
    // loop streams B once instead of computing a tile's worth of mma for it.
    if (elem == GemmElem::F16 || elem == GemmElem::F32) {
        const bool half = elem == GemmElem::F16;
        bool ran = false;
        if (detail::small_m_applies(p, elem, transpose_a, transpose_b, vectorized)) {
            ran = half ? detail::launch_small_m<__half>(p, stream) :
                         detail::launch_small_m<float>(p, stream);
        }
        if (!ran && detail::small_nt_applies(p, elem, transpose_a, transpose_b, vectorized)) {
            ran = half ? detail::launch_small_nt<__half>(p, stream) :
                         detail::launch_small_nt<float>(p, stream);
        }
        if (ran && cudaGetLastError() == cudaSuccess) {
            return Status::ok;
        }
    }

    // The tiled kernel again, with the fill gate bypassed.
    if (dtype == DataType::f32) {
        F32Call call = make_f32_call(p, transpose_a, transpose_b);
        call.tile = tile;
        if (f32_dense_call(call, stream, true)) {
            return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
        }
    } else {
        F16Call call = make_f16_call(p, transpose_a, transpose_b, dtype == DataType::bf16);
        call.tile = tile;
        if (f16_dense_call(call, stream, true)) {
            return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
        }
    }
    return Status::unsupported;
}

} // namespace nnops::backend::cuda
