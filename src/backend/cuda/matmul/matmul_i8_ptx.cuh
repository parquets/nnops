/// @file matmul_i8_ptx.cuh
/// @brief A hand-written integer GEMM: raw PTX (mma.sync / cp.async) rather than
///        a library tile, for the raw s32 MatMulInteger destination.
///
/// The int8 atom differs from the 16-bit one in two ways that shape this kernel.
/// It carries four times the k per instruction (`m16n8k32`), and its operands are
/// four *bytes* per register, so `ldmatrix` — 16-bit granularity — cannot feed
/// it: every fragment is one `LDS.32` at `row * LD + k + 4*(lane%4)`. That is one
/// instruction where the 16-bit path needs a quarter of one, so the panel layout
/// exists to keep those reads bank-clean rather than to keep ldmatrix fed. A
/// 16 B pad on a 64 B row does exactly that: the eight rows a fragment read
/// touches start 20 words apart, which is 4 mod 8, so their four-word spans walk
/// the 32 banks without overlapping.
///
/// Both panels are k-major, in arrival order: A `[m][k]` is already that shape
/// whenever A is row-major, and a `[k][n]` B is turned over by the byte 4x4
/// `__byte_perm` transpose. `transpose_a` gives the
/// same trade on the other side. The only difference between the four poses is
/// which fill runs; the fragment map and the MMA are identical, because the
/// panel is k-major in every pose.
///
/// Fragment map for `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (lane l,
/// `g = l/4`, `t = l%4`):
///   - A (16x32) is four registers of four consecutive int8:
///     a0 = A[g][4t..4t+3],   a1 = A[g+8][4t..],
///     a2 = A[g][16+4t..],    a3 = A[g+8][16+4t..].
///   - B (8x32) is two: b0 = B[l/4][4t..], b1 = B[l/4][16+4t..] -- B's rows are
///     the atom's eight n columns.
///   - C (16x8) is the f16 atom's: lane l holds (m = l/4, n = 2*(l%4)) plus the
///     +1 column and the +8 row, which is why the int8 epilogue is the f16
///     one with the requant added.
///
/// Covers A row-major or transposed with B either k-major `[N, K]` or n-major
/// `[K, N]`, into every destination the operator resolves: s32 with the int32
/// bias and the Relu stage, s8/u8 with the CPU's exact requant, and f32/f16
/// dequantized by s_a*s_b with the bias, the clamp and the activation (bf16 too,
/// for the recurrent layer's compute dtype). The two families are distinct
/// template instantiations (`Dequant`), because the fp
/// epilogue's exp/tanh trees cost register slots that the integer destination
/// would otherwise spend on a fourth block per SM. Wave quantization and split-K
/// stay the shared front end's.

#pragma once

#include "matmul_i8_common.cuh"
#include "ptx.h"
#include "detail/bench_knobs.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

namespace nnops::backend::cuda {

/// One instantiation of the kernel. The per-warp value tile is the primary knob,
/// the warp grid multiplies it into the CTA tile, and the k depth and the ring
/// depth are the pipeline's. Every derived value is a compile-time constant, so
/// a tile costs nothing at run time and a search is a switch.
template <int WarpM_, int WarpN_, int WarpsM_, int WarpsN_, int BK_, int ST_, int MinBlocks_ = 2>
struct I8TileShape {
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
    static constexpr int MinBlocks = MinBlocks_;
    /// 16 B of pad per row: it keeps every row 16 B aligned for the cp.async
    /// vectors, and (LD/4 = 4 mod 8) it puts the eight lane rows of a fragment
    /// read on eight disjoint bank quads.
    static constexpr int kPad = 16;
    static constexpr int LD = BK + kPad; ///< row stride of either panel, bytes
    static constexpr int kABytes = BM * LD;
    static constexpr int kBBytes = BN * LD;
    static constexpr int kSmemBytes = stages * (kABytes + kBBytes);
    static constexpr int kKC = BK / 32;        ///< 32-byte k steps per stage
    static constexpr int kVecPerRow = BK / 16; ///< 16 B vectors along a row
    static constexpr int kRowsPerPass = threads / kVecPerRow;

    static_assert(WM % 16 == 0, "the warp tile must hold whole 16-row atoms");
    static_assert(WN % 8 == 0, "the warp tile must hold whole 8-column atoms");
    static_assert(BK % 32 == 0, "BK must hold whole m16n8k32 steps");
    static_assert(threads % kVecPerRow == 0, "the fill must divide the threads");
    static_assert(BM % kRowsPerPass == 0 && BN % kRowsPerPass == 0,
                  "a fill pass must divide both panels into whole row blocks");
    static_assert(kSmemBytes <= 227 * 1024, "the tile must fit one SM's memory");
};

/// s8 and u8 A differ only in the opcode: the raw accumulate is the true integer
/// dot product in whichever domain the bytes are stored in.
template <class AElem>
__device__ __forceinline__ void i8_mma(int32_t* acc, const uint32_t* a, const uint32_t* b)
{
    uint32_t* d = reinterpret_cast<uint32_t*>(acc);
    if constexpr (std::is_same_v<AElem, uint8_t>) {
        mma_m16n8k32_u8s8(&d[0], &d[1], &d[2], &d[3], a[0], a[1], a[2], a[3], b[0], b[1], d[0],
                          d[1], d[2], d[3]);
    } else {
        mma_m16n8k32_s8s8(&d[0], &d[1], &d[2], &d[3], a[0], a[1], a[2], a[3], b[0], b[1], d[0],
                          d[1], d[2], d[3]);
    }
}

/// Fill a k-major panel from a k-contiguous source, one 16 B segment per thread
/// per pass. `Full` is the caller's block-level test that the whole segment
/// block is inside both tensors: hoisting it here is what keeps the interior
/// tiles free of per-segment predicates.
template <bool Full, class S, int Rows>
__device__ __forceinline__ void i8_fill_kcontig(int8_t* sP, const int8_t* src, int64_t ld,
                                                int tile0, int limit, int k0, int K, int vrow,
                                                int vcol, bool vec, bool word)
{
    constexpr int LD = S::LD;
    constexpr int kRowsPerPass = S::kRowsPerPass;
#pragma unroll
    for (int r = 0; r < Rows / kRowsPerPass; ++r) {
        const int row = vrow + r * kRowsPerPass;
        const int gr = tile0 + row;
        const int kk = k0 + vcol;
        int8_t* const d = sP + row * LD + vcol;
        const int8_t* const s = src + int64_t(gr) * ld + kk;
        if constexpr (Full) {
            if (vec) {
                cp_async_ca_16B(d, s);
            } else if (word) {
                uint32_t* const dw = reinterpret_cast<uint32_t*>(d);
                const uint32_t* const sw = reinterpret_cast<const uint32_t*>(s);
                dw[0] = sw[0];
                dw[1] = sw[1];
                dw[2] = sw[2];
                dw[3] = sw[3];
            } else {
                for (int t = 0; t < 16; ++t) {
                    d[t] = s[t];
                }
            }
        } else if (gr < limit && kk + 16 <= K) {
            if (vec) {
                cp_async_ca_16B(d, s);
            } else if (word) {
                uint32_t* const dw = reinterpret_cast<uint32_t*>(d);
                const uint32_t* const sw = reinterpret_cast<const uint32_t*>(s);
                dw[0] = sw[0];
                dw[1] = sw[1];
                dw[2] = sw[2];
                dw[3] = sw[3];
            } else {
                for (int t = 0; t < 16; ++t) {
                    d[t] = s[t];
                }
            }
        } else {
            for (int t = 0; t < 16; ++t) {
                d[t] = (gr < limit && kk + t < K) ? s[t] : int8_t(0);
            }
        }
    }
}

/// One staged pair of k-major s8 panels through the warp's MMA grid. Both panels
/// carry the same pose whatever the operand's memory order -- the fills land
/// them k-major -- so the read has no transpose arm to choose.
template <class S, class AElem>
__device__ __forceinline__ void i8_gemm_tile(const int8_t* sA, const int8_t* sB,
                                                    int warp_m, int warp_n, int lane,
                                                    int32_t (&acc)[S::MA][S::NA][4])
{
    constexpr int MA = S::MA, NA = S::NA, LD = S::LD, WM = S::WM, WN = S::WN;
    uint32_t fa[MA][4];
    uint32_t fb[NA][2];
    // ldmatrix's per-lane row and k block. A's x4 names sixteen rows in two
    // 16-byte k halves (lanes 0-15 -> rows 0-15 at k+0, lanes 16-31 -> the same
    // rows at k+16); B's x2 names eight rows at k+0 and the same eight at k+16.
    const int a_row = lane % 16;
    const int a_koff = (lane / 16) * 16;
    const int b_row = lane % 8;
    const int b_koff = ((lane / 8) & 1) * 16;
    // ldmatrix is typed 16-bit but moves 16-byte rows, and one 32-bit register
    // is two halves -- four consecutive bytes -- which is exactly one int8
    // fragment register. Its m8n8 distribution hands lane l the four bytes at
    // 4*(l%4) of row l/4, the (g, 4t) layout a0 wants, so one x4 covers A's
    // four registers and one x2 covers B's pair: nine instructions where the
    // element-wise path needed twenty-four.
    //
    // The lane's base address is resolved once per stage into a 32-bit shared
    // offset, so every load below is `base + constant` with the constant known
    // at compile time -- no per-load cvta and no 64-bit pointer arithmetic. That
    // arithmetic was a large part of why this kernel spends 6.03 instructions
    // per MMA where cuBLAS spends 2.27.
    //
    // A register-level double buffer was tried here and measured flat (4096^3
    // NT 1.704 against 1.698 ms, and 12% worse on the 64x64 tile, which lost a
    // resident block to the extra fragments). That is what `short_scoreboard` at
    // 1.5% already said: the ldmatrix-to-MMA latency is hidden across warps, so
    // moving the loads earlier inside one warp buys nothing and costs registers.
    const uint32_t sAl = smem_addr_u32(sA + (warp_m * WM + a_row) * LD + a_koff);
    const uint32_t sBl = smem_addr_u32(sB + (warp_n * WN + b_row) * LD + b_koff);
#pragma unroll
    for (int ks = 0; ks < S::kKC; ++ks) {
        const int kc = ks * 32;
#pragma unroll
        for (int i = 0; i < MA; ++i) {
            ldmatrix_x4_at(&fa[i][0], &fa[i][1], &fa[i][2], &fa[i][3], sAl + i * 16 * LD + kc);
        }
#pragma unroll
        for (int j = 0; j < NA; ++j) {
            ldmatrix_x2_at(&fb[j][0], &fb[j][1], sBl + j * 8 * LD + kc);
        }
#pragma unroll
        for (int i = 0; i < MA; ++i) {
#pragma unroll
            for (int j = 0; j < NA; ++j) {
                i8_mma<AElem>(acc[i][j], fa[i], fb[j]);
            }
        }
    }
}


/// The kernels themselves live in the translation unit that launches them
/// (matmul_i8_ptx.cu), so a caller that only needs the tile
/// shapes and the device primitives does not re-parse them.

} // namespace nnops::backend::cuda
