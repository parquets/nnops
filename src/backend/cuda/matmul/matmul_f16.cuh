/// @file matmul_f16.cuh
/// @brief A hand-written 16-bit GEMM: raw PTX (mma.sync / ldmatrix / cp.async)
///        instead of a library tile. The per-warp value tile is the search's to
///        pick: a 64x32 tile is reachable here, and
///
///
/// ncu puts the resulting kernel at ~66% of the shared-memory pipe with no pipe
/// above 90% -- shared-memory traffic and occupancy, not the tensor pipe, are
/// what the 32x32 tile costs. Hand-writing the shared-memory-to-register path
/// lifts that limit: the default tile runs a 128x128x32 CTA on four warps as a
/// 2x2 grid, so each warp owns a 64x64 value tile and the ldmatrix traffic per
/// MAC halves. The register mapping is spelled out below from the m16n8k16
/// layout.
///
/// The CTA shape is a template parameter (`F16TileShape`) rather than a set of
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
/// Covers A row-major or transposed with B either k-major `[N, K]` (Linear's
/// weight, MatMul's transposed B) or n-major `[K, N]`, plus the batch axis and
/// the operator epilogue. The conv A-gather builds the same panels out of a
/// convolution instead of a dense operand.
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
          bool StagedEpi_ = true, int MinBlocks_ = 1, int Raster_ = 0>
struct F16TileShape {
    static constexpr int WM = WarpM_;  ///< rows per warp (16-row atoms)
    static constexpr int WN = WarpN_;  ///< columns per warp (8-col atoms)
    static constexpr int MA = WM / 16; ///< m atoms per warp
    static constexpr int NA = WN / 8;  ///< n atoms per warp
    static constexpr int wm = WarpsM_; ///< warps along m
    static constexpr int wn = WarpsN_; ///< warps along n
    static constexpr int threads = wm * wn * 32;
    /// L2 rasterization strip: the CTAs of one strip walk n before m, so the
    /// strip's B panels are the ones still in the cache. 0 keeps the plain
    /// (m, n) order. The f32 engine's tile knob, ported.
    static constexpr int Raster = Raster_;
    static constexpr int BM = WM * wm; ///< rows per CTA
    static constexpr int BN = WN * wn; ///< columns per CTA
    static constexpr int BK = BK_;     ///< reduction depth per stage
    static constexpr int stages = ST_; ///< cp.async ring depth
    /// `__launch_bounds__` minimum blocks: it caps the register count, which is
    /// what stands between the small tiles and a fourth resident block.
    static constexpr int MinBlocks = MinBlocks_;
    /// Panel layout: 1 = the XOR swizzle (no padding bytes), 0 = a plain skewed
    /// stride. These are the two standard trades, and the search measures both.
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

struct F16Params {
    const void* a = nullptr;
    const void* b = nullptr;
    void* c = nullptr;
    int64_t m = 0, n = 0, k = 0;
    int64_t lda = 0, ldb = 0, ldc = 0;
    float alpha = 1.0f;
    /// The operator epilogue: alpha, bias, beta, clamp, activation, residual.
    GemmEpilogue epi;
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
    /// 1 when both operands' bases and row pitches allow a 16 B cp.async. A
    /// call whose views start or step off the 16 B bound takes the fill's
    /// scalar path instead, which is the same code the tile's edges run.
    int vec = 1;
};

/// The k-major panel swizzle (LeetCUDA's scheme): a row is BK/8 16 B chunks, and
/// the chunk index is XORed with the low row bits, so the eight rows one
/// ldmatrix reads land on different bank quads. It is an involution, so the fill
/// and the read use the same function, and it costs no padding bytes -- which is
/// what lets the panels share the epilogue buffer.
template <class S> __device__ __forceinline__ int f16_swz(int row, int col)
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
template <class S> __device__ __forceinline__ int f16_swz_t(int row, int col)
{
    if constexpr (S::Swz == 0) {
        return row * S::LDT + col;
    } else {
        constexpr int kMask = (S::BM / 8) >= 16 ? 15 : ((S::BM / 8) - 1);
        const int chunk = (col >> 3) ^ (row & kMask);
        return row * S::LDT + (chunk << 3) + (col & 7);
    }
}

template <class S> __device__ __forceinline__ int f16_swz_n(int row, int col)
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
template <class S> __device__ __forceinline__ int f16_epi_offset(int row, int col)
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
__device__ __forceinline__ void f16_mma(float* acc, const uint32_t* a, const uint32_t* b)
{
    if constexpr (std::is_same_v<Elem, __nv_bfloat16>) {
        mma_m16n8k16_bf16_f32(&acc[0], &acc[1], &acc[2], &acc[3], a[0], a[1], a[2], a[3], b[0],
                              b[1], acc[0], acc[1], acc[2], acc[3]);
    } else {
        mma_m16n8k16_f32(&acc[0], &acc[1], &acc[2], &acc[3], a[0], a[1], a[2], a[3], b[0], b[1],
                         acc[0], acc[1], acc[2], acc[3]);
    }
}

template <class Elem> __device__ __forceinline__ Elem gemm_from_float(float v)
{
    if constexpr (std::is_same_v<Elem, __nv_bfloat16>) {
        return __float2bfloat16(v);
    } else {
        return __float2half(v);
    }
}

/// Two adjacent columns as one 4 B store; the accumulator fragment always pairs
/// them, so the epilogue never writes an element on its own.
template <class Elem> __device__ __forceinline__ uint32_t gemm_pack2(float lo, float hi)
{
    if constexpr (std::is_same_v<Elem, __nv_bfloat16>) {
        const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
        return *reinterpret_cast<const uint32_t*>(&v);
    } else {
        const __half2 v = __floats2half2_rn(lo, hi);
        return *reinterpret_cast<const uint32_t*>(&v);
    }
}

template <class Elem> __device__ __forceinline__ Elem gemm_zero()
{
    return gemm_from_float<Elem>(0.0f);
}

/// One staged pair of panels through the warp's MMA grid: four ldmatrix reads per
/// m atom and two per n atom, then the m16n8k16 products. The panels are indexed
/// in `S`'s own layouts, so a kernel that fills them by any means -- a dense copy
/// or a convolution gather -- can call this for its mainloop.
template <class S, class Elem, bool TransA, bool TransB>
__device__ __forceinline__ void f16_gemm_tile(const Elem* sA, const Elem* sB, int warp_m,
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
                const Elem* src = &sA[f16_swz<S>(warp_m * S::WM + i * 16 + ld_row, kcol + ld_col)];
                ldmatrix_x4(&frA[i][0], &frA[i][1], &frA[i][2], &frA[i][3], src);
            } else {
                // `[k][m]` transposed on load: (k0-7,m0-7) and (k8-15,m0-7) land
                // as a0/a2, the m+8 pair as a1/a3.
                const Elem* src =
                    &sA[f16_swz_t<S>(kcol + ld_row, warp_m * S::WM + i * 16 + ld_col)];
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
                const Elem* src = &sB[f16_swz<S>(warp_n * S::WN + j * 8 + ld_row, kcol + ld_col)];
                ldmatrix_x4(&r0, &r1, &r2, &r3, src);
                frB[j][0] = r0;
                frB[j][1] = r2;
                frB[j + 1][0] = r1;
                frB[j + 1][1] = r3;
            } else {
                // `[k][n]` transposed on load: (k0-7,n0-7), (k8-15,n0-7),
                // (k0-7,n8-15), (k8-15,n8-15) are the two atoms' b0/b1.
                const Elem* src =
                    &sB[f16_swz_n<S>(kcol + ld_row, warp_n * S::WN + j * 8 + ld_col)];
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
                f16_mma<Elem>(acc[i][j], frA[i], frB[j]);
            }
        }
    }
}


/// The kernels themselves live in the translation unit that launches them
/// (matmul_f16.cu), so a caller that only needs the tile
/// shapes and the device primitives does not re-parse them.

} // namespace nnops::backend::cuda
