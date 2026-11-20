/// @file matmul_i8.cuh
/// @brief The integer dense GEMM behind MatMul and Linear — MatMulInteger
///        semantics over the SM80 16x8x32 s8 tensor-core atom.
///
/// A is s8 or u8 and B is s8, matching the operator's `is_int8_matmul` pair; a
/// u8 A takes `SM80_16x8x32_S32U8S8S32_TN`, an s8 A its signed twin, so the raw
/// accumulate is the true integer dot product in A's own storage domain. No pack
/// shift is left in the accumulator, which is what keeps this route bit-for-bit
/// with the CPU's formula (`simd_epilogue.hpp`'s `matmul_int8_compensate` with
/// u8_offset 0):
///
///     v = raw − zp_a·R_b[n] − (R_a[m] − K·zp_a)·zp_b[n]
///
/// R_a[m] = Σ_k qa[m,k] and R_b[n] = Σ_k qb[k,n] are swept per batch element by
/// the prepass kernels below. The bias — int32 on an integer destination, the
/// output's fp dtype on a dequantizing one — is added after the compensation;
/// an integer destination then applies Relu (the only stage that applies there)
/// and either stores the int32 value or requantizes with the CPU's exact f32
/// round-away-from-zero. An fp destination dequantizes with s_a·s_b, clips and
/// activates, like every other fp route.
///
/// Batch broadcasting and the workspace pool are the shared front end's
/// (matmul_op.cuh / matmul_splitk.cuh); this engine never splits k.
///
/// The shared-memory panels are k-major for both operands — the byte-level
/// transpose `ldmatrix.trans` cannot do — padded 16 B per row so the LDSM reads
/// spread across banks. The 16 B cp.async fast path therefore needs the operand
/// to be k-contiguous: A `[.., M, K]` and a transposed B `[N, K]` qualify, while
/// `transpose_a` and a plain `[K, N]` B take the scalar gather.

#pragma once

#include "matmul_i8_common.cuh"
#include "matmul_i8_ptx_host.hpp"

#include "matmul_splitk.cuh" // the pooled workspace, plus the shared front end
#include "nnops/core/data_type.hpp"
#include "nnops/core/status.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace nnops::backend::cuda {

namespace detail {


// ---------------------------------------------------------------------------
// narrow m (decode shapes)
// ---------------------------------------------------------------------------

/// A product of at most sixteen rows is a bandwidth problem: B is streamed once
/// and the tensor-core tile computes four times the rows it needs. One thread
/// per output column walks k four bytes at a time (`__dp4a`) with the packed A
/// row group broadcast out of shared memory -- the fp engine's narrow-m shape,
/// with the int8 pair dot in place of the FMA sweep.
constexpr int kI8NarrowRows = 16;
constexpr int kI8NarrowThreads = 128;
constexpr int kI8NarrowCols = kI8NarrowThreads; ///< one output column per thread
constexpr int kI8NarrowKC = 32;                 ///< k staged per chunk
constexpr int kI8NarrowPad = 4;                 ///< 36 B rows: 4-byte reads, 32 banks
constexpr int kI8NarrowStages = 2;

/// Split-K: the column axis alone gives N/128 blocks (32 at N=4096), a third of
/// a wave. Mirrors the fp narrow heuristic: a half split per 512 k, capped at
/// sixteen, then capped so the f32 planes stay inside the cache.
inline int i8_narrow_splits(const GemmParams& p)
{
    const int forced = splitk_env();
    if (forced == 0) {
        return 1;
    }
    const int k_tiles = (p.k + kI8NarrowKC - 1) / kI8NarrowKC;
    if (forced > 1) {
        return std::max(1, std::min({forced, k_tiles, 16}));
    }
    const int64_t plane = int64_t(p.m) * p.n * int64_t(sizeof(int32_t));
    int splits = int(std::min<int64_t>(std::max<int64_t>(p.k / 512, 1), 16));
    const int64_t l2 = device_l2_bytes();
    if (l2 > 0) {
        splits = int(std::min<int64_t>(splits, std::max<int64_t>(l2 * 9 / 8 / plane, 1)));
    }
    splits = std::min(splits, k_tiles);
    while (splits > 1 && int64_t(splits) * plane > kSplitKMaxBytes) {
        splits /= 2;
    }
    return splits;
}

template <class AElem, bool TransB>
static __global__
__launch_bounds__(kI8NarrowThreads) void dense_gemm_i8_narrow_kernel(GemmI8Params q)
{
    const GemmParams& p = q.g;
    const int M = p.m;
    const int N = p.n;
    const int K = p.k;
    const int tid = int(threadIdx.x);
    const int n0 = int(blockIdx.x) * kI8NarrowCols;
    const int ncol = n0 + tid;
    const int split = gemm_block_split(p);
    const int batch = gemm_block_batch(p);

    int kt_begin = 0;
    int kt_count = 0;
    gemm_split_range(p, kI8NarrowKC, split, kt_begin, kt_count);
    if (kt_count <= 0) {
        return;
    }
    const int k_begin = kt_begin * kI8NarrowKC;
    const int k_end = min(K, (kt_begin + kt_count) * kI8NarrowKC);

    constexpr int kCols = kI8NarrowCols;
    constexpr int kKC = kI8NarrowKC;
    constexpr int kNG = kKC / 4; // four-byte k groups per chunk
    constexpr int kRow = kKC + kI8NarrowPad;
    constexpr int kStages = kI8NarrowStages;

    __shared__ __align__(16) int8_t smem_b[kStages * kCols * kRow];
    __shared__ __align__(16) uint32_t smem_a[kStages * kI8NarrowRows * kNG];

    const AElem* A = static_cast<const AElem*>(p.a) + gemm_batch_offset(p, p.a_batch, batch);
    const int8_t* B = static_cast<const int8_t*>(p.b) + gemm_batch_offset(p, p.b_batch, batch);
    // The u8 form cannot use an unsigned dp4a (there is none for a mixed pair),
    // so the A bytes are shifted by -128 and the epilogue moves the whole
    // compensation into that domain.
    const uint32_t a_xor = q.a_shift != 0 ? 0x80808080u : 0u;
    const int mlim = min(M, kI8NarrowRows);

    auto load_a = [&](int k0, int stage) {
        uint32_t* sa = smem_a + stage * kI8NarrowRows * kNG;
        const int m = tid / kNG;
        const int j = tid - m * kNG;
        uint32_t v = 0;
        if (m < mlim) {
            const AElem* src = A + int64_t(m) * p.lda + k0 + 4 * j;
            if (k0 + 4 * j + 3 < K) {
                v = *reinterpret_cast<const uint32_t*>(src);
            } else {
                uint8_t bb[4] = {0, 0, 0, 0};
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    if (k0 + 4 * j + t < K) {
                        bb[t] = reinterpret_cast<const uint8_t*>(src)[t];
                    }
                }
                v = *reinterpret_cast<uint32_t*>(bb);
            }
        }
        sa[m * kNG + j] = v ^ a_xor;
    };

    // B is staged k-major: [column][k] with a 4 B pad, so a thread's column is
    // four-byte aligned and the warp's reads walk the banks.
    uint32_t rb[kNG];
    auto fetch_b = [&](int k0) {
        if constexpr (TransB) { // B [N, K]: the column's k run is contiguous
            const int8_t* src = B + int64_t(ncol) * p.ldb + k0;
            if (ncol < N && k0 + kKC <= K) {
                #pragma unroll
                for (int g = 0; g < kNG; ++g) {
                    rb[g] = *reinterpret_cast<const uint32_t*>(src + 4 * g);
                }
            } else {
                #pragma unroll
                for (int g = 0; g < kNG; ++g) {
                    uint8_t bb[4] = {0, 0, 0, 0};
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        if (ncol < N && k0 + 4 * g + t < K) {
                            bb[t] = uint8_t(src[4 * g + t]);
                        }
                    }
                    rb[g] = *reinterpret_cast<uint32_t*>(bb);
                }
            }
        }
    };
    auto store_b = [&](int stage) {
        int8_t* dst = smem_b + stage * kCols * kRow + tid * kRow;
        #pragma unroll
        for (int g = 0; g < kNG; ++g) {
            *reinterpret_cast<uint32_t*>(dst + 4 * g) = rb[g];
        }
    };
    auto load_b_transposed = [&](int k0, int stage) {
        // B [K, N]: the coalesced run is along n, so the 4x4 byte transpose
        // fills the k-major rows (one word store per four k of one column).
        i8_transpose_fill<kCols, kKC>(smem_b + stage * kCols * kRow, kRow, B, n0, k0, N, K,
                                      int(p.ldb),
                                      (reinterpret_cast<uintptr_t>(B) % 4 == 0) &&
                                          (p.ldb % 4 == 0),
                                      tid, kI8NarrowThreads);
    };

    int32_t acc[kI8NarrowRows];
    #pragma unroll
    for (int m = 0; m < kI8NarrowRows; ++m) {
        acc[m] = 0;
    }

    int stage = 0;
    if constexpr (TransB) {
        fetch_b(k_begin);
    }
    for (int k0 = k_begin; k0 < k_end; k0 += kKC) {
        __syncthreads();
        load_a(k0, stage);
        if constexpr (TransB) {
            store_b(stage);
            if (k0 + kKC < k_end) {
                fetch_b(k0 + kKC);
            }
        } else {
            load_b_transposed(k0, stage);
        }
        __syncthreads();

        const int8_t* sb = smem_b + stage * kCols * kRow + tid * kRow;
        const uint32_t* sa = smem_a + stage * kI8NarrowRows * kNG;
        #pragma unroll
        for (int g = 0; g < kNG; ++g) {
            const uint32_t b = *reinterpret_cast<const uint32_t*>(sb + 4 * g);
            for (int m = 0; m < mlim; ++m) {
                acc[m] = __dp4a(int(sa[m * kNG + g]), int(b), acc[m]);
            }
        }
        stage ^= 1;
    }

    if (ncol >= N) {
        return;
    }
    if (p.k_splits > 1) {
        int32_t* part = reinterpret_cast<int32_t*>(p.partial) +
                        (int64_t(batch) * p.k_splits + split) * int64_t(M) * N;
        #pragma unroll
        for (int m = 0; m < mlim; ++m) {
            part[int64_t(m) * N + ncol] = acc[m];
        }
        return;
    }
    const int64_t c_off = gemm_batch_offset(p, p.c_batch, batch);
    const int32_t* r_a_b = q.r_a + int64_t(batch) * int64_t(p.m);
    const int32_t* r_b_b = q.r_b + int64_t(batch) * int64_t(p.n);
    #pragma unroll
    for (int m = 0; m < mlim; ++m) {
        i8_epilogue_element(q, c_off, r_a_b, r_b_b, acc[m], m, ncol);
    }
}

/// Sum the k-split planes and run the integer epilogue once, exactly as the
/// single-pass kernel would have.
static __global__ void dense_gemm_i8_reduce_kernel(GemmI8Params q)
{
    const GemmParams& p = q.g;
    const int64_t plane = int64_t(p.m) * p.n;
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= plane) {
        return;
    }
    const int m = int(idx / p.n);
    const int n = int(idx - int64_t(m) * p.n);
    const int batch = int(blockIdx.y);

    const int32_t* src =
        reinterpret_cast<const int32_t*>(p.partial) + (int64_t(batch) * p.k_splits) * plane + idx;
    int32_t sum = 0;
    for (int s = 0; s < p.k_splits; ++s) {
        sum += src[int64_t(s) * plane];
    }
    i8_epilogue_element(q, gemm_batch_offset(p, p.c_batch, batch),
                        q.r_a + int64_t(batch) * int64_t(p.m),
                        q.r_b + int64_t(batch) * int64_t(p.n), sum, m, n);
}

} // namespace detail

/// The int8 twin of `dense_gemm_status`: stage the per-token/per-channel
/// parameters, sweep the two reductions, run the tile, and return a Status.
inline Status dense_gemm_i8_status(GemmI8Params q, DataType a_dt, bool trans_a, bool trans_b,
                                   cudaStream_t stream)
{
    const GemmParams& p = q.g;
    if (a_dt != DataType::s8 && a_dt != DataType::u8) {
        return Status::unsupported;
    }
    if (p.m <= 0 || p.n <= 0 || p.k <= 0) {
        return Status::unsupported;
    }

    // A tile grid past a billion blocks is not a product anyone means to run;
    // the guard keeps the workspace arithmetic below inside int64 without a
    // second thought about the block count.
    const int64_t num_m = (p.m + 63) / 64;
    const int64_t num_n = (p.n + 63) / 64;
    if (num_m * num_n > (int64_t(1) << 30)) {
        return Status::unsupported;
    }

    q.a_row_major = !trans_a;
    q.b_row_major = trans_b;
    q.a_vec = !trans_a && i8_operand_vectorizable(p.a, p.lda);
    q.b_vec = trans_b && i8_operand_vectorizable(p.b, p.ldb);

    const int64_t batch = detail::gemm_batch_count(p);
    const bool need_scale = q.out != I8OutKind::S32;

    // A decode-shaped product (at most sixteen rows) takes the column-parallel
    // narrow kernel: the tile would compute four times the rows it needs, and
    // B is the only operand that matters.
    const bool narrow = !trans_a && p.m >= 1 && p.m <= detail::kI8NarrowRows && p.n >= 4 &&
                        p.k >= 256 && batch == 1 && (reinterpret_cast<uintptr_t>(p.a) % 4 == 0) &&
                        (p.lda % 4 == 0) && (reinterpret_cast<uintptr_t>(p.b) % 4 == 0) &&
                        (p.ldb % 4 == 0);
    const int splits = narrow ? detail::i8_narrow_splits(p) : 1;
    const int64_t partial_bytes =
        (splits > 1) ? int64_t(splits) * batch * int64_t(p.m) * p.n * int64_t(sizeof(int32_t)) : 0;

    // ---- workspace -------------------------------------------------------
    // The two reduction planes, then the staged parameter arrays. Each staged
    // array is a host pointer by contract (the same one the CPU route reads),
    // so the launcher owns its device copy.
    const int64_t r_a_off = 0;
    const int64_t r_b_off = r_a_off + batch * int64_t(p.m) * int64_t(sizeof(int32_t));
    int64_t cursor = r_b_off + batch * int64_t(p.n) * int64_t(sizeof(int32_t));

    const auto stage = [&](const void* src, int64_t bytes, int64_t& off) {
        if (src == nullptr) {
            off = -1;
            return;
        }
        off = cursor;
        cursor += (bytes + 15) & ~int64_t(15);
    };
    int64_t off_zpa = -1, off_zpb = -1, off_sca = -1, off_scb = -1;
    stage(q.zp_a, int64_t(p.m) * 4, off_zpa);
    stage(q.zp_b, int64_t(p.n) * 4, off_zpb);
    if (need_scale) {
        stage(q.sc_a, int64_t(p.m) * 4, off_sca);
        stage(q.sc_b, int64_t(p.n) * 4, off_scb);
    }
    const int64_t off_partial = cursor;
    cursor += partial_bytes;

    const detail::GemmWorkspace ws = detail::gemm_workspace_alloc(cursor, stream);
    if (ws.ptr == nullptr) {
        return Status::internal;
    }
    char* base = static_cast<char*>(ws.ptr);

    q.r_a = reinterpret_cast<int32_t*>(base + r_a_off);
    q.r_b = reinterpret_cast<int32_t*>(base + r_b_off);
    const auto copy_in = [&](const void* src, int64_t off, int64_t bytes) {
        if (off < 0) {
            return true;
        }
        return cudaMemcpyAsync(base + off, src, size_t(bytes), cudaMemcpyHostToDevice, stream) ==
               cudaSuccess;
    };
    const bool staged = copy_in(q.zp_a, off_zpa, int64_t(p.m) * 4) &&
                        copy_in(q.zp_b, off_zpb, int64_t(p.n) * 4) &&
                        (!need_scale || (copy_in(q.sc_a, off_sca, int64_t(p.m) * 4) &&
                                         copy_in(q.sc_b, off_scb, int64_t(p.n) * 4)));
    if (!staged) {
        detail::gemm_workspace_free(ws, stream);
        cudaGetLastError();
        return Status::internal;
    }
    q.zp_a = off_zpa >= 0 ? reinterpret_cast<const int32_t*>(base + off_zpa) : nullptr;
    q.zp_b = off_zpb >= 0 ? reinterpret_cast<const int32_t*>(base + off_zpb) : nullptr;
    q.sc_a = off_sca >= 0 ? reinterpret_cast<const float*>(base + off_sca) : nullptr;
    q.sc_b = off_scb >= 0 ? reinterpret_cast<const float*>(base + off_scb) : nullptr;

    // ---- reductions ------------------------------------------------------
    constexpr int kSumThreads = 128;
    const dim3 ra_grid(unsigned(p.m), unsigned(batch), 1);
    const dim3 rb_grid(unsigned(p.n), unsigned(batch), 1);
    if (a_dt == DataType::u8) {
        detail::i8_row_sums_kernel<uint8_t><<<ra_grid, kSumThreads, 0, stream>>>(q, q.r_a);
    } else {
        detail::i8_row_sums_kernel<int8_t><<<ra_grid, kSumThreads, 0, stream>>>(q, q.r_a);
    }
    if (q.b_row_major) {
        detail::i8_col_sums_kernel<<<rb_grid, kSumThreads, 0, stream>>>(q, q.r_b);
    } else {
        // The transposed walk accumulates with atomics; the plane is per call.
        cudaMemsetAsync(q.r_b, 0, size_t(batch) * size_t(p.n) * sizeof(int32_t), stream);
        const int64_t cols_per_block = int64_t(kSumThreads) * 4;
        const dim3 rb_grid2(unsigned((p.n + cols_per_block - 1) / cols_per_block), unsigned(batch),
                            8);
        detail::i8_col_sums_kernel<<<rb_grid2, kSumThreads, 0, stream>>>(q, q.r_b);
    }

    // ---- narrow m --------------------------------------------------------
    if (narrow) {
        q.a_shift = (a_dt == DataType::u8) ? -128 : 0;
        q.g.k_splits = splits;
        q.g.partial = splits > 1 ? reinterpret_cast<float*>(base + off_partial) : nullptr;
        const dim3 ngrid(unsigned((p.n + detail::kI8NarrowCols - 1) / detail::kI8NarrowCols), 1,
                         unsigned(splits));
        constexpr unsigned kNThreads = detail::kI8NarrowThreads;
        if (a_dt == DataType::u8) {
            if (trans_b) {
                detail::dense_gemm_i8_narrow_kernel<uint8_t, true>
                    <<<ngrid, kNThreads, 0, stream>>>(q);
            } else {
                detail::dense_gemm_i8_narrow_kernel<uint8_t, false>
                    <<<ngrid, kNThreads, 0, stream>>>(q);
            }
        } else {
            if (trans_b) {
                detail::dense_gemm_i8_narrow_kernel<int8_t, true>
                    <<<ngrid, kNThreads, 0, stream>>>(q);
            } else {
                detail::dense_gemm_i8_narrow_kernel<int8_t, false>
                    <<<ngrid, kNThreads, 0, stream>>>(q);
            }
        }
        if (splits > 1) {
            constexpr int kReduceThreads = 256;
            const int64_t plane = int64_t(p.m) * p.n;
            const dim3 rgrid(unsigned((plane + kReduceThreads - 1) / kReduceThreads),
                             unsigned(batch), 1);
            detail::dense_gemm_i8_reduce_kernel<<<rgrid, kReduceThreads, 0, stream>>>(q);
        }
        detail::gemm_workspace_free(ws, stream);
        return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
    }

    // ---- the tile --------------------------------------------------------
    // The hand-written PTX tile owns every pose: its gate is the destination
    // kind, the transposed-side word addressing and a non-empty reduction
    // scratch, all of which the launcher above has already established. A
    // refusal here means an operand whose row stride is not word-addressable,
    // which no engine below covers any more.
    const bool ran = i8_dense_tile(q, a_dt, trans_a, trans_b, stream, -1);
    detail::gemm_workspace_free(ws, stream);
    if (!ran) {
        return Status::unsupported;
    }
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}
} // namespace nnops::backend::cuda
