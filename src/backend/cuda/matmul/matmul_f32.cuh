/// @file matmul_f32.cuh
/// @brief The hand-written f32 GEMM: a SIMT FFMA tile in the same skeleton as
///        the 16-bit PTX kernel.
///
/// f32 has no tensor-core path on this target (TF32 was measured at 15.3 TFLOPS
/// against 16.2 for plain FMA, and at a precision the operator contract does not
/// allow), so this is the classic register-blocked kernel: a cp.async panel
/// ring, a per-thread 8x8 (or 8x4) accumulator, and the k loop unrolled four
/// deep so each panel read is one 16 B vector.
///
/// The layout is the one the FFMA ratio wants. Per four k values a lane reads
/// `LaneRow` A vectors and `LaneCol` B vectors and issues
/// `LaneRow * LaneCol * 4` FMAs, so the smem traffic per FLOP is
/// 1/(LaneCol) + 1/(LaneRow) vectors' worth -- 8x8 is the cheapest square the
/// register file can hold twice.
///
/// Both B layouts share one code path: the n-major pose transposes into the
/// panel on the way in (four scalar stores per 16 B read), so the inner loop
/// only ever sees a k-major panel.
///
/// Batch and split-K share blockIdx.z: `z = b * splits + split`, and the batch
/// strides are 0 for a broadcast operand -- the same numpy-style contract the
/// `gemm_batch_layout` produces.

#pragma once

#include "matmul_epilogue.cuh"
#include "ptx.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace nnops::backend::cuda {

/// One instantiation of the f32 kernel.
template <int LaneRow_, int LaneCol_, int WarpsM_, int WarpsN_, int BK_, int ST_, int KU_ = 4,
          int Raster_ = 8, int Pad_ = 4, int MinBlocks_ = 1>
struct F32TileShape {
    static constexpr int LaneRow = LaneRow_; ///< rows one thread owns
    static constexpr int LaneCol = LaneCol_; ///< columns one thread owns
    /// A warp is 32 lanes as an 8x4 grid: 8 lanes down the rows, 4 across.
    static constexpr int warpRows = 8 * LaneRow;
    static constexpr int warpCols = 4 * LaneCol;
    static constexpr int wm = WarpsM_;
    static constexpr int wn = WarpsN_;
    static constexpr int threads = wm * wn * 32;
    static constexpr int BM = wm * warpRows;
    static constexpr int BN = wn * warpCols;
    static constexpr int BK = BK_;
    static constexpr int stages = ST_;
    static constexpr int KU = KU_; ///< k values per panel vector
    /// L2 rasterization strip: the grid is walked in groups of this many m tiles,
    /// stepping n inside a group. 0 disables it. Without it the block order is
    /// plain row-major, and at 4096^3 that alone cost 13% -- the tiles that
    /// follow a block along n re-read a B panel nothing has kept warm.
    static constexpr int Raster = Raster_;
    /// Panels are padded, not swizzled: sixteen bytes of row offset puts eight
    /// consecutive rows on eight bank quads, and f32 panels are small enough
    /// that the padding costs no residency.
    /// Every panel keeps its operand's arrival order, so the stride is whichever
    /// of the two shapes the pose makes it. A is `[m][k]` when it comes in
    /// row-major and `[k][m]` when it comes transposed; B likewise. `Pad` is the
    /// skew that spreads consecutive rows over the bank quads: four elements
    /// costs a fifth of the panel, and a tile that would rather have the
    /// residency than the conflict can ask for none.
    static constexpr int Pad = Pad_;
    /// `__launch_bounds__` minimum blocks: it caps the register count, which is
    /// what stands between the wide tiles and a fourth resident block.
    static constexpr int MinBlocks = MinBlocks_;
    static constexpr int LDA = BK + Pad;  ///< A as `[m][k]`
    static constexpr int LDAt = BM + Pad; ///< A as `[k][m]`
    static constexpr int LDN = BN + Pad;  ///< B as `[k][n]`
    static constexpr int LDk = BK + Pad;  ///< B as `[n][k]`
    static constexpr int kKSteps = BK / KU;
    static constexpr int kVecPerRow = BK / KU;
    static constexpr int kRowsPerPass = threads / kVecPerRow;
    /// A panel holds one operand in one arrival order, so a pose pays for the
    /// order it is handed and not the other: A is `[m][k]` -- BM rows of LDA --
    /// or `[k][m]`, BK rows of LDAt; B is `[n][k]` or `[k][n]` the same way.
    /// The two orders differ by the pad times the other dimension, which is
    /// 10-30% of a panel, so sizing both to the larger cost the kernel bytes it
    /// never touched.
    static constexpr int a_panel(bool trans_a) { return trans_a ? BK * LDAt : BM * LDA; }
    static constexpr int b_panel(bool trans_b) { return trans_b ? BN * LDk : BK * LDN; }
    static constexpr int panel_bytes(bool trans_a, bool trans_b)
    {
        return stages * (a_panel(trans_a) + b_panel(trans_b)) * int(sizeof(float));
    }

    static_assert(BK % KU == 0, "the k unroll must divide the stage depth");
    static_assert(LaneCol % 4 == 0, "a lane's columns fill whole 16 B vectors in the n-major pose");
    static_assert(BK % 4 == 0, "a panel row must hold whole 16 B vectors");
    static_assert(BM % kRowsPerPass == 0 && BN % kRowsPerPass == 0,
                  "a fill pass must divide both panels into whole row blocks");
};

struct F32Params {
    const void* a = nullptr;
    const void* b = nullptr;
    void* c = nullptr;
    int64_t m = 0, n = 0, k = 0;
    int64_t lda = 0, ldb = 0, ldc = 0;
    float alpha = 1.0f;
    /// Batch: up to four leading axes, each a (size, stride) pair with stride 0
    /// for a broadcast operand -- the same numpy-style contract
    /// `gemm_batch_layout` produces.
    int batch_rank = 0;
    int64_t batch = 1;
    int64_t bsize[4] = {1, 1, 1, 1};
    int64_t a_bstr[4] = {0, 0, 0, 0};
    int64_t b_bstr[4] = {0, 0, 0, 0};
    int64_t c_bstr[4] = {0, 0, 0, 0};
    /// The residual's batch strides: `add_to`'s old value is this slice's, so
    /// the epilogue adds the same batch offset C does -- stride 0 broadcasts,
    /// exactly as the operands' do.
    int64_t r_bstr[4] = {0, 0, 0, 0};
    /// The operator-level epilogue: alpha, bias, beta, clamp, activation,
    /// residual, in that order.
    GemmEpilogue epi;
    /// Split-K scratch: [splits][m][n] f32 planes, or null for the single pass.
    float* partial = nullptr;
    /// 1 when both operands' bases and row pitches allow a 16 B cp.async. A
    /// call whose views start or step off the 16 B bound takes the fill's
    /// scalar path instead, which is the same code the tile's edges run.
    int vec = 1;
};

/// One staged pair of panels through the register-blocked FFMA nest. The panels
/// are addressed in `S`'s own layouts, so a kernel that fills them by any means
/// -- a dense copy or a convolution gather -- can call this for its mainloop.
template <class S, bool TransA, bool TransB>
__device__ __forceinline__ void f32_gemm_tile(const float* sA, const float* sB, int warp_m,
                                                   int warp_n, int lane,
                                                   float (&acc)[S::LaneRow][S::LaneCol])
{
    constexpr int LR = S::LaneRow, LC = S::LaneCol, KU = S::KU;
    constexpr int G = LR / 4; ///< row groups a lane's four-row vector walks
    constexpr int LDA = TransA ? S::LDAt : S::LDA;
    constexpr int LDK = TransB ? S::LDk : S::LDN;

    const int lrow = lane % 8;
    const int lcol = lane / 8;
    const int row0 = warp_m * S::warpRows + (TransA ? 4 * lrow : lrow);
    const int col0 = warp_n * S::warpCols + lcol * LC;
    // Per-lane panel bases, hoisted out of the k groups. Measured honestly:
    // the SASS is byte-identical to spelling the full index per load, so
    // ptxas was already doing this -- the 544 IADD3 in this function are not
    // the load addressing. Kept because it reads as what the code means.
    const float* const aRow = sA + row0 * LDA;
    // The k-major B panel is indexed by row, so the column offset scales by
    // the row stride; the n-major one is indexed by column inside a row.
    const float* const bCol = sB + col0 * LDK;
    const float* const sA_m = sA + row0;
    const float* const sB_k = sB + col0;
    static_assert(KU == 4, "the unroll is one 16 B vector");

#pragma unroll
    for (int ks = 0; ks < S::kKSteps; ++ks) {
        const int kv = ks * KU;
        // The four poses differ only in where a lane's (row, k) and (k, column)
        // values sit inside the vector it loads. Unpacking both into these two
        // K-major views is register renaming, not instructions, so the
        // arithmetic is one nest whatever the pose -- and the two arrays are
        // the only fragment storage, which is what kept the transposed pose
        // off the register count that would have cost it a resident block.
        float ak[LR][KU];
        float bk[KU][LC];
        if constexpr (TransA) {
            // One vector per k of the unroll, four rows of it.
#pragma unroll
            for (int t = 0; t < KU; ++t) {
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    const float4 v =
                        *reinterpret_cast<const float4*>(&sA_m[(kv + t) * LDA + 32 * g]);
                    ak[4 * g + 0][t] = v.x;
                    ak[4 * g + 1][t] = v.y;
                    ak[4 * g + 2][t] = v.z;
                    ak[4 * g + 3][t] = v.w;
                }
            }
        } else {
            // One vector per row, four k values of it.
#pragma unroll
            for (int r = 0; r < LR; ++r) {
                const float4 v = *reinterpret_cast<const float4*>(&aRow[(r * 8) * LDA + kv]);
                ak[r][0] = v.x;
                ak[r][1] = v.y;
                ak[r][2] = v.z;
                ak[r][3] = v.w;
            }
        }
        if constexpr (TransB) {
            // `[n][k]`: one vector per column, holding four k values.
#pragma unroll
            for (int c = 0; c < LC; ++c) {
                const float4 v = *reinterpret_cast<const float4*>(&bCol[c * LDK + kv]);
                bk[0][c] = v.x;
                bk[1][c] = v.y;
                bk[2][c] = v.z;
                bk[3][c] = v.w;
            }
        } else {
            // `[k][n]`: one vector per k of the unroll, holding four adjacent
            // columns. Every lane in a lane-column reads the same vector, which
            // the hardware broadcasts.
#pragma unroll
            for (int t = 0; t < KU; ++t) {
#pragma unroll
                for (int g = 0; g < LC / 4; ++g) {
                    const float4 v =
                        *reinterpret_cast<const float4*>(&sB_k[(kv + t) * LDK + 4 * g]);
                    bk[t][4 * g + 0] = v.x;
                    bk[t][4 * g + 1] = v.y;
                    bk[t][4 * g + 2] = v.z;
                    bk[t][4 * g + 3] = v.w;
                }
            }
        }
        // acc[r][c] += sum_t ak[r][t] * bk[t][c], one k ascending per element:
        // the same rounding order the hand-expanded nests spelled out, and the
        // same row-then-column walk -- looping the columns outer instead cost
        // the 2048^3 grid 3.4% on the schedule alone.
#pragma unroll
        for (int r = 0; r < LR; ++r) {
#pragma unroll
            for (int c = 0; c < LC; ++c) {
#pragma unroll
                for (int t = 0; t < KU; ++t) {
                    acc[r][c] = fmaf(ak[r][t], bk[t][c], acc[r][c]);
                }
            }
        }
    }
}


/// The kernels themselves live in the translation unit that launches them
/// (matmul_f32.cu), so a caller that only needs the tile
/// shapes and the device primitives does not re-parse them.

} // namespace nnops::backend::cuda
