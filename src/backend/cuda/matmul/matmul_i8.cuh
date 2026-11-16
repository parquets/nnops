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

/// 64x64x64 on four warps: the s8 panels are half the fp16 footprint, so three
/// stages cost 30 KB and still leave three blocks a SM.
struct GemmTileI8 {
    static constexpr int threads = 128; ///< one four-warp group
    static constexpr int bM = 64;
    static constexpr int bN = 64;
    static constexpr int bK = 64;
    static constexpr int stages = 2;
    static constexpr int wm = 2;
    static constexpr int wn = 2;
    static constexpr int mmaM = 32;
    static constexpr int mmaN = 32;
};

/// A 16 B pad keeps every row start 16 B aligned for cp.async while walking the
/// eight rows an LDSM reads across eight different bank groups.
constexpr int kI8Pad = 16;

template <class AElem, bool TransA, bool TransB, int Groups = 1>
static __global__
__launch_bounds__(GemmTileI8::threads* Groups) void dense_gemm_i8_kernel(GemmI8Params q)
{
    using namespace cute;
    using BElem = int8_t;

    const GemmParams& p = q.g;
    constexpr int bM = GemmTileI8::bM;
    constexpr int bN = GemmTileI8::bN;
    constexpr int bK = GemmTileI8::bK;
    constexpr int kStages = GemmTileI8::stages;
    constexpr int kThreads = GemmTileI8::threads;
    constexpr int kCtaThreads = kThreads * Groups;

    // `Groups` copies of the four-warp tile side by side along n in one CTA: each
    // keeps its own accumulators and B panel and shares the A panel -- the
    // pairing the fp engine measured ahead of a taller warp grid, which the LDSM
    // atoms reject at a 4x2 warp tiling.
    const int wg = int(threadIdx.x) / kThreads;
    const int lane = int(threadIdx.x) % kThreads;

    const int M = p.m;
    const int N = p.n;
    const int K = p.k;
    // L2-grouped raster, the fp engine's scheme: blockIdx.x walks a band of
    // `kGroupM` m tiles by every n tile, so a wave shares both the band's k-slab
    // of A and one n tile's slab of B. Without it an int8 tile re-reads the
    // whole operand set from DRAM often enough to make it bandwidth-bound.
    const int num_m = (M + bM - 1) / bM;
    const int num_n = (N + bN - 1) / bN;
    const int num_nc = (num_n + Groups - 1) / Groups;
    constexpr int kGroupM = 4;
    const int group_span = kGroupM * num_nc;
    const int gid = int(blockIdx.x) / group_span;
    const int within = int(blockIdx.x) - gid * group_span;
    const int m_span = (num_m - gid * kGroupM) < kGroupM ? (num_m - gid * kGroupM) : kGroupM;
    const int tile_m = gid * kGroupM + within % m_span;
    const int tile_n = (within / m_span) * Groups + wg;
    const int m0 = tile_m * bM;
    const int n0 = tile_n * bN;

    const int batch = gemm_block_batch(p);
    AElem const* A = static_cast<AElem const*>(p.a) + gemm_batch_offset(p, p.a_batch, batch);
    BElem const* B = static_cast<BElem const*>(p.b) + gemm_batch_offset(p, p.b_batch, batch);

    Tensor mA = [&] {
        if constexpr (TransA) {
            return make_tensor(make_gmem_ptr(A), make_shape(M, K),
                               make_stride(Int<1>{}, int(p.lda)));
        } else {
            return make_tensor(make_gmem_ptr(A), make_shape(M, K),
                               make_stride(int(p.lda), Int<1>{}));
        }
    }();
    Tensor mB = [&] {
        if constexpr (TransB) {
            return make_tensor(make_gmem_ptr(B), make_shape(N, K),
                               make_stride(int(p.ldb), Int<1>{}));
        } else {
            return make_tensor(make_gmem_ptr(B), make_shape(N, K),
                               make_stride(Int<1>{}, int(p.ldb)));
        }
    }();

    auto cta_tiler = make_shape(Int<bM>{}, Int<bN>{}, Int<bK>{});
    auto cta_coord = make_coord(tile_m, tile_n, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});

    constexpr auto sA_layout =
        make_layout(make_shape(Int<bM>{}, Int<bK>{}, Int<kStages>{}),
                    make_stride(Int<bK + kI8Pad>{}, Int<1>{}, Int<(bK + kI8Pad) * bM>{}));
    constexpr auto sB_layout =
        make_layout(make_shape(Int<bN>{}, Int<bK>{}, Int<kStages>{}),
                    make_stride(Int<bK + kI8Pad>{}, Int<1>{}, Int<(bK + kI8Pad) * bN>{}));
    constexpr int kSmemA = int(cosize(sA_layout));

    extern __shared__ char gemm_smem[];
    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<AElem*>(gemm_smem)), sA_layout);
    constexpr int kSmemB = int(cosize(sB_layout));
    Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<BElem*>(
                                gemm_smem + (kSmemA + wg * kSmemB) * int(sizeof(AElem)))),
                            sB_layout);

    using Atom = std::conditional_t<std::is_same_v<AElem, uint8_t>, SM80_16x8x32_S32U8S8S32_TN,
                                    SM80_16x8x32_S32S8S8S32_TN>;
    TiledMMA mma =
        make_tiled_mma(Atom{}, Layout<Shape<Int<GemmTileI8::wm>, Int<GemmTileI8::wn>, _1>>{},
                       cute::Tile<Int<GemmTileI8::mmaM>, Int<GemmTileI8::mmaN>, Int<32>>{});

    // The vectorized copy is only legal for a k-contiguous operand.
    auto copy_a = gemm_copy_half<AElem, kCtaThreads, bM, bK, true>();
    auto copy_b = gemm_copy_half<BElem, kThreads, bN, bK, true>();
    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tAsA = thr_copy_a.partition_D(sA);
    ThrCopy thr_copy_b = copy_b.get_slice(lane);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    Tensor tBsB = thr_copy_b.partition_D(sB);

    const bool m_full = (m0 + bM <= M);
    const bool n_full = (n0 + bN <= N);

    // The fastest fill needs the operand k-contiguous (the cp.async copy). The
    // other orientation has the whole 16-wide row run in gmem along the *other*
    // axis: read it as one 16 B vector and scatter its bytes into the k-major
    // panel, which is coalesced where the element-wise walk is not. A 16 B
    // vector needs the row pitch and the base 16 B aligned.
    const bool a_rows = (reinterpret_cast<uintptr_t>(A) % 4 == 0) && (p.lda % 4 == 0);
    const bool b_rows = (reinterpret_cast<uintptr_t>(B) % 4 == 0) && (p.ldb % 4 == 0);

    auto fill_a = [&](int kt, int stage) {
        const int k0 = kt * bK;
        if constexpr (TransA) {
            if (a_rows) {
                i8_transpose_fill<bM, bK>(reinterpret_cast<int8_t*>(&sA(0, 0, stage)), bK + kI8Pad,
                                          reinterpret_cast<const int8_t*>(A), m0, k0, M, K,
                                          int(p.lda), int(threadIdx.x), int(blockDim.x));
                return;
            }
        }
        for (int i = int(threadIdx.x); i < bM * bK; i += int(blockDim.x)) {
            const int row = i / bK;
            const int col = i - row * bK;
            const int m = m0 + row;
            const int k = k0 + col;
            AElem v = AElem(0);
            if (m < M && k < K) {
                if constexpr (TransA) {
                    v = A[int64_t(m) + int64_t(k) * p.lda];
                } else {
                    v = A[int64_t(m) * p.lda + k];
                }
            }
            sA(row, col, stage) = v;
        }
    };
    auto fill_b = [&](int kt, int stage) {
        const int k0 = kt * bK;
        if constexpr (!TransB) {
            if (b_rows) {
                i8_transpose_fill<bN, bK>(reinterpret_cast<int8_t*>(&sB(0, 0, stage)), bK + kI8Pad,
                                          reinterpret_cast<const int8_t*>(B), n0, k0, N, K,
                                          int(p.ldb), lane, kThreads);
                return;
            }
        }
        for (int i = lane; i < bN * bK; i += kThreads) {
            const int row = i / bK;
            const int col = i - row * bK;
            const int n = n0 + row;
            const int k = k0 + col;
            BElem v = BElem(0);
            if (n < N && k < K) {
                if constexpr (TransB) {
                    v = B[int64_t(n) * p.ldb + k];
                } else {
                    v = B[int64_t(k) * p.ldb + n];
                }
            }
            sB(row, col, stage) = v;
        }
    };

    auto load_a = [&](int kt, int stage) {
        if constexpr (!TransA) {
            if (q.a_vec && m_full && (kt + 1) * bK <= K) {
                copy(copy_a, tAgA(_, _, _, kt), tAsA(_, _, _, stage));
                return;
            }
        }
        fill_a(kt, stage);
    };
    auto load_b = [&](int kt, int stage) {
        if constexpr (TransB) {
            if (q.b_vec && n_full && (kt + 1) * bK <= K) {
                copy(copy_b, tBgB(_, _, _, kt), tBsB(_, _, _, stage));
                return;
            }
        }
        fill_b(kt, stage);
    };

    const int k_tiles = (K + bK - 1) / bK;
    int k_tile_count = k_tiles;
    int k_tile_next = 0;

    CUTE_UNROLL
    for (int pipe = 0; pipe < kStages - 1; ++pipe) {
        load_a(k_tile_next, pipe);
        load_b(k_tile_next, pipe);
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) {
            ++k_tile_next;
        }
    }

    ThrMMA thr_mma = mma.get_slice(lane);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));
    Tensor tCrC = thr_mma.make_fragment_C(thr_mma.partition_C(local_tile(
        make_identity_tensor(make_shape(M, N)), cta_tiler, cta_coord, Step<_1, _1, X>{})));
    clear(tCrC);

    Copy_Atom<SM75_U32x4_LDSM_N, AElem> s2r_atom_a;
    Copy_Atom<SM75_U32x4_LDSM_N, BElem> s2r_atom_b;
    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
    ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(lane);
    Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
    Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);
    TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
    ThrCopy s2r_thr_copy_b = s2r_copy_b.get_slice(lane);
    Tensor tXsB = s2r_thr_copy_b.partition_S(sB);
    Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);

    constexpr int kBlocks = bK / 32;
    int smem_pipe_read = 0;
    int smem_pipe_write = kStages - 1;
    Tensor tXsA_p = tXsA(_, _, _, smem_pipe_read);
    Tensor tXsB_p = tXsB(_, _, _, smem_pipe_read);

    cp_async_wait<kStages - 2>();
    __syncthreads();
    copy(s2r_atom_a, tXsA_p(_, _, Int<0>{}), tXrA(_, _, Int<0>{}));
    copy(s2r_atom_b, tXsB_p(_, _, Int<0>{}), tXrB(_, _, Int<0>{}));

    CUTE_NO_UNROLL
    while (k_tile_count > -(kStages - 1)) {
        CUTE_UNROLL
        for (int k_block = 0; k_block < kBlocks; ++k_block) {
            if (k_block == kBlocks - 1) {
                tXsA_p = tXsA(_, _, _, smem_pipe_read);
                tXsB_p = tXsB(_, _, _, smem_pipe_read);
                cp_async_wait<kStages - 2>();
                __syncthreads();
            }
            const int k_block_next = (k_block + 1) % kBlocks;
            copy(s2r_atom_a, tXsA_p(_, _, k_block_next), tXrA(_, _, k_block_next));
            copy(s2r_atom_b, tXsB_p(_, _, k_block_next), tXrB(_, _, k_block_next));
            if (k_block == 0) {
                load_a(k_tile_next, smem_pipe_write);
                load_b(k_tile_next, smem_pipe_write);
                cp_async_fence();
                --k_tile_count;
                if (k_tile_count > 0) {
                    ++k_tile_next;
                }
                smem_pipe_write = smem_pipe_read;
                smem_pipe_read = (smem_pipe_read == kStages - 1) ? 0 : smem_pipe_read + 1;
            }
            gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }
    }

    // ---- output ---------------------------------------------------------
    Tensor cC = make_identity_tensor(make_shape(M, N));
    Tensor tCcC = thr_mma.partition_C(local_tile(cC, cta_tiler, cta_coord, Step<_1, _1, X>{}));

    // The epilogue is spelled out here rather than shared with the narrow and
    // reduce kernels: as a function call it costs ~35 registers per thread,
    // which drops this kernel from 3 CTAs a SM to 2 and measured ~15% of the
    // tile's throughput. Everything that can hoist out of the loop does.
    const int64_t c_off = gemm_batch_offset(p, p.c_batch, batch);
    const int32_t* r_a_b = q.r_a + int64_t(batch) * M;
    const int32_t* r_b_b = q.r_b + int64_t(batch) * N;
    const bool int_out =
        q.out == I8OutKind::S32 || q.out == I8OutKind::S8 || q.out == I8OutKind::U8;
    const int32_t* bias32 = int_out ? static_cast<const int32_t*>(q.bias) : nullptr;
    const float* biasf32 = q.out == I8OutKind::F32 ? static_cast<const float*>(q.bias) : nullptr;
    const __half* biasf16 = q.out == I8OutKind::F16 ? static_cast<const __half*>(q.bias) : nullptr;
    int32_t* c32 = q.out == I8OutKind::S32 ? static_cast<int32_t*>(p.c) + c_off : nullptr;
    int8_t* c8 = q.out == I8OutKind::S8 ? static_cast<int8_t*>(p.c) + c_off : nullptr;
    uint8_t* cu8 = q.out == I8OutKind::U8 ? static_cast<uint8_t*>(p.c) + c_off : nullptr;
    float* cf32 = q.out == I8OutKind::F32 ? static_cast<float*>(p.c) + c_off : nullptr;
    __half* cf16 = q.out == I8OutKind::F16 ? static_cast<__half*>(p.c) + c_off : nullptr;
    const int32_t a_shift_k = q.a_shift * p.k;

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        auto coord = tCcC(i);
        const int m = int(get<0>(coord));
        const int n = int(get<1>(coord));
        if (m >= M || n >= N) {
            continue;
        }

        const int32_t zpa = (q.zp_a != nullptr ? q.zp_a[m] : q.zp_a_scalar) + q.a_shift;
        const int32_t zpb = q.zp_b != nullptr ? q.zp_b[n] : q.zp_b_scalar;
        int32_t raw = int32_t(tCrC(i));
        raw -= zpa * r_b_b[n];
        raw -= (r_a_b[m] + a_shift_k - p.k * zpa) * zpb;
        const int64_t idx = int64_t(m) * p.ldc + n;

        if (int_out) {
            int32_t v = raw;
            if (bias32 != nullptr) {
                v += bias32[n];
            }
            if (q.relu) {
                v = v < 0 ? 0 : v;
            }
            if (c32 != nullptr) {
                c32[idx] = v;
                continue;
            }
            const float s_am = q.sc_a != nullptr ? q.sc_a[m] : q.sc_a_scalar;
            const float s_bn = q.sc_b != nullptr ? q.sc_b[n] : q.sc_b_scalar;
            const float row_k = s_am / q.out_scale;
            if (c8 != nullptr) {
                c8[idx] = i8_requant<int8_t>(v, row_k, s_bn, float(q.out_zp));
            } else {
                cu8[idx] = i8_requant<uint8_t>(v, row_k, s_bn, float(q.out_zp));
            }
            continue;
        }

        const float s_am = q.sc_a != nullptr ? q.sc_a[m] : q.sc_a_scalar;
        const float s_bn = q.sc_b != nullptr ? q.sc_b[n] : q.sc_b_scalar;
        float f = i8_to_float(raw) * (s_am * s_bn);
        float b = 0.0f;
        if (biasf32 != nullptr) {
            b = biasf32[n];
        } else if (biasf16 != nullptr) {
            b = __half2float(biasf16[n]);
        }
        if (b != 0.0f) {
            f += b;
        }
        f = fminf(fmaxf(f, p.min_clip), p.max_clip);
        f = gemm_apply_epilogue(p.epilogue, p.ep_alpha, p.ep_beta, f);
        if (cf32 != nullptr) {
            cf32[idx] = f;
        } else {
            cf16[idx] = __float2half(f);
        }
    }
}

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
    using namespace cute;
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
                CUTE_UNROLL
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
                CUTE_UNROLL
                for (int g = 0; g < kNG; ++g) {
                    rb[g] = *reinterpret_cast<const uint32_t*>(src + 4 * g);
                }
            } else {
                CUTE_UNROLL
                for (int g = 0; g < kNG; ++g) {
                    uint8_t bb[4] = {0, 0, 0, 0};
                    CUTE_UNROLL
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
        CUTE_UNROLL
        for (int g = 0; g < kNG; ++g) {
            *reinterpret_cast<uint32_t*>(dst + 4 * g) = rb[g];
        }
    };
    auto load_b_transposed = [&](int k0, int stage) {
        // B [K, N]: the coalesced run is along n, so the 4x4 byte transpose
        // fills the k-major rows (one word store per four k of one column).
        i8_transpose_fill<kCols, kKC>(smem_b + stage * kCols * kRow, kRow, B, n0, k0, N, K,
                                      int(p.ldb), tid, kI8NarrowThreads);
    };

    int32_t acc[kI8NarrowRows];
    CUTE_UNROLL
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
        CUTE_UNROLL
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
        CUTE_UNROLL
        for (int m = 0; m < mlim; ++m) {
            part[int64_t(m) * N + ncol] = acc[m];
        }
        return;
    }
    const int64_t c_off = gemm_batch_offset(p, p.c_batch, batch);
    const int32_t* r_a_b = q.r_a + int64_t(batch) * int64_t(p.m);
    const int32_t* r_b_b = q.r_b + int64_t(batch) * int64_t(p.n);
    CUTE_UNROLL
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

    constexpr int kGroups = 2;
    const int64_t num_m = (p.m + detail::GemmTileI8::bM - 1) / detail::GemmTileI8::bM;
    const int64_t num_n = (p.n + detail::GemmTileI8::bN - 1) / detail::GemmTileI8::bN;
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
    // The hand-written PTX tile takes the raw s32 destination; the requantizing
    // and dequantizing ones (and their scales) are still this file's engine. The
    // knob forces either side, and N >= 2 also names a menu row.
    const int ptx_i8 = ::nnops::detail::cuda_ptx_i8.load(std::memory_order_relaxed);
    if (ptx_i8 >= 0 &&
        manual_dense_i8_tile(q, a_dt, trans_a, trans_b, stream, ptx_i8 >= 2 ? ptx_i8 - 2 : -1)) {
        detail::gemm_workspace_free(ws, stream);
        return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
    }

    using Tile = detail::GemmTileI8;
    constexpr int smem = (Tile::bM + kGroups * Tile::bN) * (Tile::bK + detail::kI8Pad) *
                         Tile::stages * int(sizeof(int8_t));
    const int64_t num_nc = (num_n + kGroups - 1) / kGroups;
    const dim3 grid(unsigned(num_m * num_nc), 1, unsigned(batch));
    const unsigned threads = unsigned(Tile::threads * kGroups);

    if (a_dt == DataType::u8) {
        if (trans_a) {
            if (trans_b) {
                detail::dense_gemm_i8_kernel<uint8_t, true, true, kGroups>
                    <<<grid, threads, smem, stream>>>(q);
            } else {
                detail::dense_gemm_i8_kernel<uint8_t, true, false, kGroups>
                    <<<grid, threads, smem, stream>>>(q);
            }
        } else if (trans_b) {
            detail::dense_gemm_i8_kernel<uint8_t, false, true, kGroups>
                <<<grid, threads, smem, stream>>>(q);
        } else {
            detail::dense_gemm_i8_kernel<uint8_t, false, false, kGroups>
                <<<grid, threads, smem, stream>>>(q);
        }
    } else {
        if (trans_a) {
            if (trans_b) {
                detail::dense_gemm_i8_kernel<int8_t, true, true, kGroups>
                    <<<grid, threads, smem, stream>>>(q);
            } else {
                detail::dense_gemm_i8_kernel<int8_t, true, false, kGroups>
                    <<<grid, threads, smem, stream>>>(q);
            }
        } else if (trans_b) {
            detail::dense_gemm_i8_kernel<int8_t, false, true, kGroups>
                <<<grid, threads, smem, stream>>>(q);
        } else {
            detail::dense_gemm_i8_kernel<int8_t, false, false, kGroups>
                <<<grid, threads, smem, stream>>>(q);
        }
    }

    detail::gemm_workspace_free(ws, stream);
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

} // namespace nnops::backend::cuda
