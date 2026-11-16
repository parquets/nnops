/// @file matmul_f16.cuh
/// @brief A hand-written 16-bit GEMM: raw PTX (mma.sync / ldmatrix / cp.async)
///        instead of CuTe's TiledMMA/TiledCopy.
///
/// CuTe's ldmatrix copy atoms pin the per-warp value tile to 32x32 in this
/// checkout (a 64x32 tile trips "Copy_Traits: src failed to vectorize"), and
/// ncu puts the resulting kernel at ~66% of the shared-memory pipe with no pipe
/// above 90% -- shared-memory traffic and occupancy, not the tensor pipe, are
/// what the 32x32 tile costs. Hand-writing the shared-memory-to-register path
/// lifts that limit: the default tile runs a 128x128x32 CTA on four warps as a
/// 2x2 grid, so each warp owns a 64x64 value tile and the ldmatrix traffic per
/// MAC halves. The register mapping is spelled out below from the m16n8k16
/// layout.
///
/// The CTA shape is a template parameter (`ManualHShape`) rather than a set of
/// constants, so the search instantiates a handful and the host picks one: the
/// per-warp value tile, the warp grid, the k depth and the cp.async ring depth.
/// Shared memory is dynamic and one union covers both the cp.async panels and
/// the epilogue staging, so a 128x128x64 three-stage tile with a 128x128
/// write-back buffer fits in what the panels alone would need.
///
/// The element type is a template parameter too. f16 and bf16 share every
/// register path -- ldmatrix, the 16 B cp.async copies and the fragment map are
/// bit-agnostic -- and differ only in the mma opcode and the f32 conversion.
///
/// The epilogue is staged through that shared memory instead of writing half
/// lanes straight to global: the accumulator fragment hands each lane two
/// adjacent columns eight rows apart, which is a 4 B store per element pair and
/// a scattered pattern even then. Staging turns it into 16 B stores of eight
/// consecutive columns, one per thread per pass, with alpha applied on the way
/// in -- the same shape the CPU kernels' write-back helpers produce.
///
/// Covers A row-major with B either k-major `[N, K]` (Linear's weight, MatMul's
/// transposed B) or n-major `[K, N]`, alpha 1, no bias, no epilogue, one batch
/// element; the batch axis, the epilogue and the conv A-gather are the parts
/// still to port off CuTe.
///
/// Fragment map for `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`:
///   - A (16x16 per atom): lane l's ldmatrix address is row (l%16) at half
///     8*(l/16); one `ldmatrix.x4` yields the atom's four A registers in order
///     (rows 0-7, k 0-7), (8-15, 0-7), (0-7, 8-15), (8-15, 8-15) = a0..a3.
///   - B (8x16 per atom): the same address shape over the `[n][k]` panel, and
///     the same four matrices land as (atom j: b0, b1) = (m0, m2) and
///     (atom j+1: b0, b1) = (m1, m3). An `[k][n]` panel is read with
///     `ldmatrix.trans`, whose four matrices are already the two atoms' b0/b1 in
///     order -- the transposed 8x8 block is exactly the fragment's
///     (n = l/4, k = 2*(l%4)) layout.
///   - C (16x8 per atom): lane l holds (m = l/4, n = 2*(l%4)) plus the +1
///     column and the +8 row.

#pragma once

#include "matmul_epilogue.cuh"
#include "ptx.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

namespace nnops::backend::cuda {

/// One instantiation of the kernel. The per-warp value tile is the primary
/// knob -- it is what the ldmatrix traffic per MAC is drawn from -- and the CTA
/// tile is what the warp grid multiplies it into. Every derived value is a
/// compile-time constant, so a tile costs nothing at run time and the search is
/// a switch.
template <int WarpM_, int WarpN_, int WarpsM_, int WarpsN_, int BK_, int ST_, int Swz_ = 1,
          bool StagedEpi_ = true, int MinBlocks_ = 1>
struct ManualHShape {
    static constexpr int WM = WarpM_;  ///< rows per warp (16-row atoms)
    static constexpr int WN = WarpN_;  ///< columns per warp (8-col atoms)
    static constexpr int MA = WM / 16; ///< m atoms per warp
    static constexpr int NA = WN / 8;  ///< n atoms per warp
    static constexpr int wm = WarpsM_; ///< warps along m
    static constexpr int wn = WarpsN_; ///< warps along n
    static constexpr int threads = wm * wn * 32;
    static constexpr int BM = WM * wm; ///< rows per CTA
    static constexpr int BN = WN * wn; ///< columns per CTA
    static constexpr int BK = BK_;     ///< reduction depth per stage
    static constexpr int stages = ST_; ///< cp.async ring depth
    /// `__launch_bounds__` minimum blocks: it caps the register count, which is
    /// what stands between the small tiles and a fourth resident block.
    static constexpr int MinBlocks = MinBlocks_;
    /// Panel layout: 1 = the XOR swizzle (no padding bytes), 0 = a plain skewed
    /// stride. These are the two trades CuTe's Swizzle and its padded layouts
    /// make, and the search measures both.
    static constexpr int Swz = Swz_;
    /// Whether the epilogue stages through shared memory (16 B stores) or writes
    /// the accumulator fragment straight out (4 B stores).
    static constexpr bool StagedEpi = StagedEpi_;
    /// A skewed panel adds eight elements to the row so consecutive rows land on
    /// different bank quads -- the padding the XOR swizzle exists to avoid.
    static constexpr int kSkew = Swz_ == 0 ? 8 : 0;
    static constexpr int LDA = BK + kSkew; ///< k-major row halves
    static constexpr int LDT = BM + kSkew; ///< m-major row halves (A transposed)
    static constexpr int LDN = BN + kSkew; ///< n-major row halves
    static constexpr int kKSteps = BK / 16;
    static constexpr int kNChunks = BN / 8;
    /// One fill pass is one 16 B vector per thread: BK/8 vectors along a k row,
    /// so a pass covers threads/(BK/8) rows of either panel.
    static constexpr int kVecPerRow = BK / 8;
    static constexpr int kRowsPerPass = threads / kVecPerRow;
    /// The panels and the epilogue share one buffer; the union is the larger.
    /// Each panel is sized to the larger of its two arrival orders, so one
    /// buffer serves every pose.
    static constexpr int kAMax = BM * LDA > BK* LDT ? BM* LDA : BK* LDT;
    static constexpr int kBMax = BN * LDA > BK* LDN ? BN* LDA : BK* LDN;
    static constexpr int kPanelHalves = stages * (kAMax + kBMax);
    static constexpr int kEpiHalves = BM * LDN;
    static constexpr int kSmemHalves = kPanelHalves > kEpiHalves ? kPanelHalves : kEpiHalves;
    static constexpr int kSmemBytes = kSmemHalves * int(sizeof(__half));

    static_assert(WM % 16 == 0, "the warp tile must hold whole 16-row atoms");
    static_assert(WN % 8 == 0, "the warp tile must hold whole 8-column atoms");
    static_assert(BK % 16 == 0, "BK must hold whole m16n8k16 steps");
    static_assert(BM % kRowsPerPass == 0 && BN % kRowsPerPass == 0,
                  "a fill pass must divide both panels into whole row blocks");
    static_assert(BK >= 32 || BN >= 64, "the swizzle needs eight 16 B chunks");
    static_assert(kSmemBytes <= 227 * 1024, "the tile must fit one SM's memory");
};

struct ManualHParams {
    const void* a = nullptr;
    const void* b = nullptr;
    void* c = nullptr;
    int64_t m = 0, n = 0, k = 0;
    int64_t lda = 0, ldb = 0, ldc = 0;
    float alpha = 1.0f;
    /// The operator epilogue: alpha, bias, beta, clamp, activation, residual.
    ManualEpilogue epi;
    /// Batch: up to four leading axes as (size, stride) pairs. gridDim.z carries
    /// `batch * splits`, batch-major.
    int batch_rank = 0;
    int64_t batch = 1;
    int64_t bsize[4] = {1, 1, 1, 1};
    int64_t a_bstr[4] = {0, 0, 0, 0};
    int64_t b_bstr[4] = {0, 0, 0, 0};
    int64_t c_bstr[4] = {0, 0, 0, 0};
    /// Split-K scratch: [splits][batch][m][n] f32 planes, or null for one pass.
    float* partial = nullptr;
};

/// The k-major panel swizzle (LeetCUDA's scheme): a row is BK/8 16 B chunks, and
/// the chunk index is XORed with the low row bits, so the eight rows one
/// ldmatrix reads land on different bank quads. It is an involution, so the fill
/// and the read use the same function, and it costs no padding bytes -- which is
/// what lets the panels share the epilogue buffer.
template <class S> __device__ __forceinline__ int manual_swz(int row, int col)
{
    if constexpr (S::Swz == 0) {
        return row * S::LDA + col;
    } else {
        constexpr int kMask = S::LDA / 8 - 1;
        const int chunk = (col >> 3) ^ (row & kMask);
        return row * S::LDA + (chunk << 3) + (col & 7);
    }
}

/// The n-major B panel `[k][n]`: its row is a whole BN tile wide, a multiple of
/// the bank cycle, so only the chunk index can spread the rows one
/// ldmatrix.trans reads. The mask is the panel's chunk count, not eight: an x4
/// load reads sixteen k rows (0-7 and 8-15 for the two n halves), and a
/// seven-bit mask would give those two groups the same chunk -- a 2-way bank
/// conflict on every B load, which is what put this pose 20% behind the k-major
/// one.
/// The m-major A panel `[k][m]`: the n-major B scheme, one BM-wide row.
template <class S> __device__ __forceinline__ int manual_swz_t(int row, int col)
{
    if constexpr (S::Swz == 0) {
        return row * S::LDT + col;
    } else {
        constexpr int kMask = (S::BM / 8) >= 16 ? 15 : ((S::BM / 8) - 1);
        const int chunk = (col >> 3) ^ (row & kMask);
        return row * S::LDT + (chunk << 3) + (col & 7);
    }
}

template <class S> __device__ __forceinline__ int manual_swz_n(int row, int col)
{
    if constexpr (S::Swz == 0) {
        return row * S::LDN + col;
    } else {
        constexpr int kMask = (S::BN / 8) >= 16 ? 15 : ((S::BN / 8) - 1);
        const int chunk = (col >> 3) ^ (row & kMask);
        return row * S::BN + (chunk << 3) + (col & 7);
    }
}

/// The epilogue buffer `[bm][bn]`, swizzled the same way: the write is a lane's
/// element pair and the read is a thread's 16 B, and both want eight row groups
/// apart.
template <class S> __device__ __forceinline__ int manual_epi(int row, int col)
{
    if constexpr (S::Swz == 0) {
        return row * S::LDN + col;
    } else {
        constexpr int kMask = (S::BN / 8) >= 8 ? 7 : (S::BN / 8 - 1);
        const int chunk = (col >> 3) ^ (row & kMask);
        return row * S::BN + (chunk << 3) + (col & 7);
    }
}

/// bf16 and f16 differ in exactly two places: the mma opcode, and the f32
/// conversion the epilogue and the reduce pass need. Everything else is bits.
template <class Elem>
__device__ __forceinline__ void manual_mma(float* acc, const uint32_t* a, const uint32_t* b)
{
    if constexpr (std::is_same_v<Elem, __nv_bfloat16>) {
        mma_m16n8k16_bf16_f32(&acc[0], &acc[1], &acc[2], &acc[3], a[0], a[1], a[2], a[3], b[0],
                              b[1], acc[0], acc[1], acc[2], acc[3]);
    } else {
        mma_m16n8k16_f32(&acc[0], &acc[1], &acc[2], &acc[3], a[0], a[1], a[2], a[3], b[0], b[1],
                         acc[0], acc[1], acc[2], acc[3]);
    }
}

template <class Elem> __device__ __forceinline__ Elem manual_from_float(float v)
{
    if constexpr (std::is_same_v<Elem, __nv_bfloat16>) {
        return __float2bfloat16(v);
    } else {
        return __float2half(v);
    }
}

/// Two adjacent columns as one 4 B store; the accumulator fragment always pairs
/// them, so the epilogue never writes an element on its own.
template <class Elem> __device__ __forceinline__ uint32_t manual_pack2(float lo, float hi)
{
    if constexpr (std::is_same_v<Elem, __nv_bfloat16>) {
        const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
        return *reinterpret_cast<const uint32_t*>(&v);
    } else {
        const __half2 v = __floats2half2_rn(lo, hi);
        return *reinterpret_cast<const uint32_t*>(&v);
    }
}

template <class Elem> __device__ __forceinline__ Elem manual_zero()
{
    return manual_from_float<Elem>(0.0f);
}

/// One staged pair of panels through the warp's MMA grid: four ldmatrix reads per
/// m atom and two per n atom, then the m16n8k16 products. The panels are indexed
/// in `S`'s own layouts, so a kernel that fills them by any means -- a dense copy
/// or a convolution gather -- can call this for its mainloop.
template <class S, class Elem, bool TransA, bool TransB>
__device__ __forceinline__ void manual_gemm_h_tile(const Elem* sA, const Elem* sB, int warp_m,
                                                   int warp_n, int lane,
                                                   float (&acc)[S::MA][S::NA][4])
{
    constexpr int MA = S::MA, NA = S::NA;
    uint32_t frA[MA][4];
    uint32_t frB[NA][2];
    const int ld_row = lane % 16;
    const int ld_col = (lane / 16) * 8;
#pragma unroll
    for (int ks = 0; ks < S::kKSteps; ++ks) {
        const int kcol = ks * 16;
#pragma unroll
        for (int i = 0; i < MA; ++i) {
            if constexpr (!TransA) {
                const Elem* src = &sA[manual_swz<S>(warp_m * S::WM + i * 16 + ld_row, kcol + ld_col)];
                ldmatrix_x4(&frA[i][0], &frA[i][1], &frA[i][2], &frA[i][3], src);
            } else {
                // `[k][m]` transposed on load: (k0-7,m0-7) and (k8-15,m0-7) land
                // as a0/a2, the m+8 pair as a1/a3.
                const Elem* src =
                    &sA[manual_swz_t<S>(kcol + ld_row, warp_m * S::WM + i * 16 + ld_col)];
                uint32_t r0, r1, r2, r3;
                ldmatrix_x4_trans(&r0, &r1, &r2, &r3, src);
                frA[i][0] = r0;
                frA[i][1] = r2;
                frA[i][2] = r1;
                frA[i][3] = r3;
            }
        }
#pragma unroll
        for (int j = 0; j < NA; j += 2) {
            uint32_t r0, r1, r2, r3;
            if constexpr (TransB) {
                // `[n][k]`: the four 8x8 panels are (n0-7,k0-7), (n8-15,k0-7),
                // (n0-7,k8-15), (n8-15,k8-15).
                const Elem* src = &sB[manual_swz<S>(warp_n * S::WN + j * 8 + ld_row, kcol + ld_col)];
                ldmatrix_x4(&r0, &r1, &r2, &r3, src);
                frB[j][0] = r0;
                frB[j][1] = r2;
                frB[j + 1][0] = r1;
                frB[j + 1][1] = r3;
            } else {
                // `[k][n]` transposed on load: (k0-7,n0-7), (k8-15,n0-7),
                // (k0-7,n8-15), (k8-15,n8-15) are the two atoms' b0/b1.
                const Elem* src =
                    &sB[manual_swz_n<S>(kcol + ld_row, warp_n * S::WN + j * 8 + ld_col)];
                ldmatrix_x4_trans(&r0, &r1, &r2, &r3, src);
                frB[j][0] = r0;
                frB[j][1] = r1;
                frB[j + 1][0] = r2;
                frB[j + 1][1] = r3;
            }
        }
#pragma unroll
        for (int i = 0; i < MA; ++i) {
#pragma unroll
            for (int j = 0; j < NA; ++j) {
                manual_mma<Elem>(acc[i][j], frA[i], frB[j]);
            }
        }
    }
}

/// The 16-bit GEMM: per-warp value tile from `S`, cp.async ring, swizzled panels.
template <class S, class Elem, bool TransA, bool TransB>
__global__ __launch_bounds__(S::threads, S::MinBlocks) void manual_gemm_h_kernel(ManualHParams p)
{
    constexpr int BM = S::BM, BN = S::BN, BK = S::BK, ST = S::stages;
    constexpr int WM = S::WM, WN = S::WN, MA = S::MA, NA = S::NA;
    constexpr int LDA = S::LDA, LDN = S::LDN;

    // One buffer for both phases: the panels until the k loop ends, the staging
    // buffer after it. The loop's closing barrier is what makes the hand-off
    // legal without a second block of shared memory. Dynamic rather than static,
    // so one launch path covers every tile: the host does the opt-in once and
    // passes the size it computed from the very same constant the kernel
    // indexes with.
    extern __shared__ __align__(16) unsigned char manual_smem_raw[];
    Elem* const smem = reinterpret_cast<Elem*>(manual_smem_raw);
    Elem* const sAbase = smem;
    Elem* const sBbase = smem + ST * BM * LDA;

    const int tid = int(threadIdx.x);
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / S::wn;
    const int warp_n = warp - warp_m * S::wn;
    const int m_tile = int(blockIdx.x) * BM;
    const int n_tile = int(blockIdx.y) * BN;
    const int m0 = m_tile + warp_m * WM;
    const int n0 = n_tile + warp_n * WN;
    const int M = int(p.m), N = int(p.n), K = int(p.k);
    const int k_tiles = (K + BK - 1) / BK;
    // Split-K: blockIdx.z picks a proportional k-tile range. The tile grid alone
    // leaves a shallow product a fraction of the machine, and there is no third
    // axis left to cut.
    const int bsz = int(p.batch > 0 ? p.batch : 1);
    const int splits = int(gridDim.z) / bsz;
    const int batch = int(blockIdx.z) / splits;
    const int split = int(blockIdx.z) - batch * splits;
    const int kt_begin = int(int64_t(k_tiles) * split / splits);
    const int kt_end = int(int64_t(k_tiles) * (split + 1) / splits);
    if (kt_begin >= kt_end) {
        return;
    }

    // Batch offsets resolved once; a broadcast axis carries stride 0.
    const Elem* const A =
        static_cast<const Elem*>(p.a) + manual_batch_offset(p.bsize, p.a_bstr, p.batch_rank, batch);
    const Elem* const B =
        static_cast<const Elem*>(p.b) + manual_batch_offset(p.bsize, p.b_bstr, p.batch_rank, batch);
    Elem* const C =
        static_cast<Elem*>(p.c) + manual_batch_offset(p.bsize, p.c_bstr, p.batch_rank, batch);

    // gmem -> smem: one 16 B vector per thread per pass. A k row is BK/8 vectors
    // wide, so a pass covers threads/(BK/8) rows and the panel takes BM divided
    // by that many passes. B arrives transposed, so both panels are [rows][k].
    constexpr int kVecPerRow = S::kVecPerRow;
    constexpr int kRowsPerPass = S::kRowsPerPass;
    const int vrow = tid / kVecPerRow;
    const int vcol = (tid % kVecPerRow) * 8;
    // m-major A: one 8-wide vector of every `mRowStep`-th k row.
    constexpr int kMVecPerRow = BM / 8;
    constexpr int mRowStep = S::threads / kMVecPerRow;
    const int mCol = (tid % kMVecPerRow) * 8;
    const int mRow0 = tid / kMVecPerRow;

    constexpr int kNVec = BN / 8; // 16 B vectors along one B row of the NN panel
    // One call is one committed group, whether or not it copies anything: the
    // wait's group count has to stay uniform when a split runs dry near its end.
    auto fill = [&](int kt, int stage) {
        if (kt >= kt_end) {
            cp_async_commit_group();
            return;
        }
        const int k0 = kt * BK;
        Elem* const sA = sAbase + stage * S::kAMax;
        Elem* const sB = sBbase + stage * S::kBMax;
        // Interior tiles need no bounds test at all; the per-vector tests were
        // what took 13% off the f32 sibling until they were split out here.
        const bool kFull = k0 + BK <= K;
        const bool aFull = kFull && m_tile + BM <= M;
        const bool bFull = kFull && n_tile + BN <= N;
        if constexpr (!TransB) {
            // B arrives `[k][n]`: a vector is eight consecutive n of one k row.
            if (bFull) {
                for (int i = tid; i < BK * kNVec; i += S::threads) {
                    const int kr = i / kNVec;
                    const int nv = i - kr * kNVec;
                    cp_async_ca_16B(&sB[manual_swz_n<S>(kr, nv * 8)],
                                    B + int64_t(k0 + kr) * p.ldb + n_tile + nv * 8);
                }
            } else
                for (int i = tid; i < BK * kNVec; i += S::threads) {
                    const int kr = i / kNVec;
                    const int nv = i - kr * kNVec;
                    const int kk = k0 + kr;
                    const int nn = n_tile + nv * 8;
                    if (kk < K && nn + 8 <= N) {
                        cp_async_ca_16B(&sB[manual_swz_n<S>(kr, nv * 8)],
                                        B + int64_t(kk) * p.ldb + nn);
                    } else {
                        for (int t = 0; t < 8; ++t) {
                            sB[manual_swz_n<S>(kr, nv * 8 + t)] =
                                (kk < K && nn + t < N) ? B[int64_t(kk) * p.ldb + nn + t] :
                                                         manual_zero<Elem>();
                        }
                    }
                }
        }
        if constexpr (!TransA) {
            if (aFull) {
#pragma unroll
                for (int r = 0; r < BM / kRowsPerPass; ++r) {
                    const int row = vrow + r * kRowsPerPass;
                    cp_async_ca_16B(&sA[manual_swz<S>(row, vcol)],
                                    A + int64_t(m_tile + row) * p.lda + k0 + vcol);
                }
            } else {
#pragma unroll
                for (int r = 0; r < BM / kRowsPerPass; ++r) {
                    const int row = vrow + r * kRowsPerPass;
                    const int am = m_tile + row;
                    if (am < M && k0 + vcol + 8 <= K) {
                        cp_async_ca_16B(&sA[manual_swz<S>(row, vcol)],
                                        A + int64_t(am) * p.lda + k0 + vcol);
                    } else {
                        for (int t = 0; t < 8; ++t) {
                            sA[manual_swz<S>(row, vcol + t)] =
                                (am < M && k0 + vcol + t < K) ?
                                    A[int64_t(am) * p.lda + k0 + vcol + t] :
                                    manual_zero<Elem>();
                        }
                    }
                }
            }
        } else {
            // A arrives `[k][m]`: a vector is eight consecutive m of one k row,
            // and the panel keeps that order, so the copy stays async.
            for (int kr = mRow0; kr < BK; kr += mRowStep) {
                const int kk = k0 + kr;
                const int gm = m_tile + mCol;
                if (aFull || (kk < K && gm + 8 <= M)) {
                    cp_async_ca_16B(&sA[manual_swz_t<S>(kr, mCol)], A + int64_t(kk) * p.lda + gm);
                } else {
                    for (int t = 0; t < 8; ++t) {
                        sA[manual_swz_t<S>(kr, mCol + t)] = (kk < K && gm + t < M) ?
                                                                A[int64_t(kk) * p.lda + gm + t] :
                                                                manual_zero<Elem>();
                    }
                }
            }
        }
        // The two panels are not the same height: a CTA wider than it is tall
        // (128x256) has twice the B rows of A, so each panel walks its own.
        if constexpr (TransB) {
            if (bFull) {
#pragma unroll
                for (int r = 0; r < BN / kRowsPerPass; ++r) {
                    const int row = vrow + r * kRowsPerPass;
                    cp_async_ca_16B(&sB[manual_swz<S>(row, vcol)],
                                    B + int64_t(n_tile + row) * p.ldb + k0 + vcol);
                }
            } else
#pragma unroll
                for (int r = 0; r < BN / kRowsPerPass; ++r) {
                    const int row = vrow + r * kRowsPerPass;
                    const int an = n_tile + row;
                    if (an < N && k0 + vcol + 8 <= K) {
                        cp_async_ca_16B(&sB[manual_swz<S>(row, vcol)],
                                        B + int64_t(an) * p.ldb + k0 + vcol);
                    } else {
                        for (int t = 0; t < 8; ++t) {
                            sB[manual_swz<S>(row, vcol + t)] =
                                (an < N && k0 + vcol + t < K) ?
                                    B[int64_t(an) * p.ldb + k0 + vcol + t] :
                                    manual_zero<Elem>();
                        }
                    }
                }
        }
        cp_async_commit_group();
    };

    float acc[MA][NA][4];
#pragma unroll
    for (int i = 0; i < MA; ++i)
#pragma unroll
        for (int j = 0; j < NA; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                acc[i][j][t] = 0.0f;
            }

    // Prime every stage but one; the loop then keeps exactly ST groups in flight,
    // so `wait_group<ST-1>` retires precisely the tile about to be consumed.
#pragma unroll
    for (int t = 0; t < ST - 1; ++t) {
        fill(kt_begin + t, (kt_begin + t) % ST);
    }
    for (int kt = kt_begin; kt < kt_end; ++kt) {
        const int stage = kt % ST;
        // One barrier a tile: the wait retires this tile's group, the barrier
        // then publishes it and retires every warp's reads of the tile before
        // it -- whose stage is the one the fill now overwrites.
        cp_async_wait_group<ST - 2>();
        __syncthreads();
        fill(kt + ST - 1, (kt + ST - 1) % ST);

        manual_gemm_h_tile<S, Elem, TransA, TransB>(sAbase + stage * S::kAMax,
                                                    sBbase + stage * S::kBMax, warp_m, warp_n,
                                                    lane, acc);
    }

    // A split's block stops at its accumulator: the sum and the alpha belong to
    // the reduce pass, which sees every plane.
    if (p.partial != nullptr) {
        float* part = p.partial + (int64_t(batch) * splits + split) * M * N;
#pragma unroll
        for (int i = 0; i < MA; ++i) {
            const int gm = m0 + i * 16 + lane / 4;
#pragma unroll
            for (int j = 0; j < NA; ++j) {
                const int gn = n0 + j * 8 + 2 * (lane % 4);
                if (gm < M && gn + 1 < N) {
                    part[int64_t(gm) * N + gn] = acc[i][j][0];
                    part[int64_t(gm) * N + gn + 1] = acc[i][j][1];
                    if (gm + 8 < M) {
                        part[int64_t(gm + 8) * N + gn] = acc[i][j][2];
                        part[int64_t(gm + 8) * N + gn + 1] = acc[i][j][3];
                    }
                } else {
                    const int rows[2] = {gm, gm + 8};
                    for (int rr = 0; rr < 2; ++rr) {
                        for (int cc = 0; cc < 2; ++cc) {
                            if (rows[rr] < M && gn + cc < N) {
                                part[int64_t(rows[rr]) * N + gn + cc] = acc[i][j][rr * 2 + cc];
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    // The staging buffer is the panel buffer, and the loop no longer ends on a
    // barrier: this one retires every warp's last ldmatrix before the epilogue
    // writes over it.
    __syncthreads();

    // C: lane l holds (m = l/4, n = 2*(l%4)), plus the +1 column and +8 row.
    const int cm = lane / 4;
    const int cn = 2 * (lane % 4);
    if constexpr (!S::StagedEpi) {
        // Straight out: one 4 B store per atom row, masked at the tile edge.
        // Cheaper than staging (no barrier, no round trip) but the stores are
        // scattered, which is why this is the search's other arm, not the
        // default. A row pitch that is not a whole number of 4 B stores -- an
        // odd N on a dense destination -- walks the same two elements as 2 B
        // writes instead, which is what an unpadded N=4103 does.
        const bool pair4 = (p.ldc % 2) == 0;
#pragma unroll
        for (int i = 0; i < MA; ++i) {
            const int gm = m0 + i * 16 + cm;
#pragma unroll
            for (int j = 0; j < NA; ++j) {
                const int gn = n0 + j * 8 + cn;
                for (int rr = 0; rr < 2; ++rr) {
                    const int row = gm + rr * 8;
                    if (row >= M) {
                        continue;
                    }
                    Elem* const dst = C + int64_t(row) * p.ldc;
                    const float v0 = manual_epilogue<Elem>(p.epi, acc[i][j][rr * 2], row, gn, dst);
                    const float v1 =
                        gn + 1 < N ?
                            manual_epilogue<Elem>(p.epi, acc[i][j][rr * 2 + 1], row, gn + 1, dst) :
                            0.0f;
                    if (pair4 && gn + 2 <= N) {
                        *reinterpret_cast<uint32_t*>(dst + gn) = manual_pack2<Elem>(v0, v1);
                    } else {
                        for (int cc = 0; cc < 2 && gn + cc < N; ++cc) {
                            dst[gn + cc] = manual_from_float<Elem>(cc == 0 ? v0 : v1);
                        }
                    }
                }
            }
        }
        return;
    }

    // Every slot of the tile is written -- the tail is masked on the way out, not
    // on the way in -- so the staging pass needs no bounds test.
    Elem* const stage = smem;
#pragma unroll
    for (int i = 0; i < MA; ++i) {
        const int em = warp_m * WM + i * 16 + cm;
        const int gm = m0 + i * 16 + cm;
#pragma unroll
        for (int j = 0; j < NA; ++j) {
            const int en = warp_n * WN + j * 8 + cn;
            const int gn = n0 + j * 8 + cn;
            // Every slot is written and the tail is masked on the way out, so an
            // out-of-range slot stores zero: the epilogue's own loads (bias, the
            // beta term, the residual) must not run past the tensors.
            const bool in_hi = gm < M && gn < N;
            const bool in_lo = gm + 8 < M && gn < N;
            const Elem* const dhi = C + int64_t(gm) * p.ldc;
            const Elem* const dlo = C + int64_t(gm + 8) * p.ldc;
            const float v0 = in_hi ? manual_epilogue<Elem>(p.epi, acc[i][j][0], gm, gn, dhi) : 0.0f;
            const float v1 = in_hi && gn + 1 < N ?
                                 manual_epilogue<Elem>(p.epi, acc[i][j][1], gm, gn + 1, dhi) :
                                 0.0f;
            const float v2 =
                in_lo ? manual_epilogue<Elem>(p.epi, acc[i][j][2], gm + 8, gn, dlo) : 0.0f;
            const float v3 = in_lo && gn + 1 < N ?
                                 manual_epilogue<Elem>(p.epi, acc[i][j][3], gm + 8, gn + 1, dlo) :
                                 0.0f;
            *reinterpret_cast<uint32_t*>(&stage[manual_epi<S>(em, en)]) =
                manual_pack2<Elem>(v0, v1);
            *reinterpret_cast<uint32_t*>(&stage[manual_epi<S>(em + 8, en)]) =
                manual_pack2<Elem>(v2, v3);
        }
    }
    __syncthreads();

    // 16 B per thread per pass: eight consecutive columns of one row. The chunk
    // the swizzle moved them to is the chunk this thread reads, so the store
    // stays inside one 256 B span of the row and the banks stay clean.
    const bool vec = (p.ldc % 8) == 0;
    const int total = BM * S::kNChunks;
    for (int idx = tid; idx < total; idx += S::threads) {
        const int row = idx / S::kNChunks;
        const int c = idx - row * S::kNChunks;
        const int gm = m_tile + row;
        if (gm >= M) {
            continue;
        }
        const int gn = n_tile + c * 8;
        const Elem* src = &stage[manual_epi<S>(row, c * 8)];
        Elem* dst = C + int64_t(gm) * p.ldc + gn;
        if (vec && gn + 8 <= N) {
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
        } else {
            for (int t = 0; t < 8 && gn + t < N; ++t) {
                dst[t] = src[t];
            }
        }
    }
}

/// Sum the split-K planes and write the destination, one thread per element.
template <class Elem> __global__ void manual_gemm_h_reduce_kernel(ManualHParams p)
{
    const int64_t plane = int64_t(p.m) * p.n;
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= plane) {
        return;
    }
    const int splits = int(gridDim.y);
    const int m = int(idx / p.n);
    const int n = int(idx - int64_t(m) * p.n);
    for (int64_t b = 0; b < p.batch; ++b) {
        float sum = 0.0f;
        for (int s = 0; s < splits; ++s) {
            sum += p.partial[(b * splits + s) * plane + idx];
        }
        Elem* const dst =
            static_cast<Elem*>(p.c) + manual_batch_offset(p.bsize, p.c_bstr, p.batch_rank, b);
        dst[int64_t(m) * p.ldc + n] = manual_from_float<Elem>(p.alpha * sum);
    }
}

} // namespace nnops::backend::cuda
