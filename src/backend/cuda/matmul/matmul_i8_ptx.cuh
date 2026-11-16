/// @file matmul_i8_ptx.cuh
/// @brief A hand-written integer GEMM: raw PTX (mma.sync / cp.async) instead of
///        CuTe's TiledMMA/TiledCopy, for the raw s32 MatMulInteger destination.
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
/// `__byte_perm` transpose the CuTe engine already used. `transpose_a` gives the
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
///     +1 column and the +8 row. That is why the int8 epilogue carried over from
///     the CuTe engine unchanged.
///
/// Covers A row-major or transposed with B either k-major `[N, K]` or n-major
/// `[K, N]`, into every destination the operator resolves: s32 with the int32
/// bias and the Relu stage, s8/u8 with the CPU's exact requant, and f32/f16
/// dequantized by s_a*s_b with the bias, the clamp and the activation. The two
/// families are distinct template instantiations (`Dequant`), because the fp
/// epilogue's exp/tanh trees cost register slots that the integer destination
/// would otherwise spend on a fourth block per SM. Wave quantization and split-K
/// stay the shared front end's.

#pragma once

#include "matmul_i8_common.cuh"
#include "ptx.h"
#include "nnops/detail/bench_knobs.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

namespace nnops::backend::cuda {

/// One instantiation of the kernel. The per-warp value tile is the primary knob,
/// the warp grid multiplies it into the CTA tile, and the k depth and the ring
/// depth are the pipeline's. Every derived value is a compile-time constant, so
/// a tile costs nothing at run time and a search is a switch.
template <int WarpM_, int WarpN_, int WarpsM_, int WarpsN_, int BK_, int ST_, int MinBlocks_ = 2>
struct ManualI8Shape {
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
__device__ __forceinline__ void manual_mma_i8(int32_t* acc, const uint32_t* a, const uint32_t* b)
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
__device__ __forceinline__ void manual_gemm_i8_tile(const int8_t* sA, const int8_t* sB,
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
                manual_mma_i8<AElem>(acc[i][j], fa[i], fb[j]);
            }
        }
    }
}

/// The integer GEMM: per-warp value tile from `S`, cp.async ring, k-major panels.
template <class S, class AElem, bool TransA, bool TransB, bool Dequant>
__global__ __launch_bounds__(S::threads, S::MinBlocks) void manual_gemm_i8_kernel(GemmI8Params q)
{
    const GemmParams& gp = q.g;
    constexpr int BM = S::BM, BN = S::BN, BK = S::BK, ST = S::stages;
    constexpr int WM = S::WM, WN = S::WN, MA = S::MA, NA = S::NA;
    constexpr int LD = S::LD;

    extern __shared__ __align__(16) unsigned char manual_i8_smem[];
    int8_t* const sAbase = reinterpret_cast<int8_t*>(manual_i8_smem);
    int8_t* const sBbase = sAbase + ST * S::kABytes;

    const int M = int(gp.m), N = int(gp.n), K = int(gp.k);
    const int k_tiles = (K + BK - 1) / BK;
    const int tid = int(threadIdx.x);
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / S::wn;
    const int warp_n = warp - warp_m * S::wn;
    // An int8 tile re-reads its whole operand band from L2 on every pass, which
    // is enough traffic to make 4096^3 bandwidth-bound if consecutive blocks do
    // not share one. `blockIdx.x` therefore walks a band of `kGroupM` m tiles by
    // every n tile: a wave shares the band's k-slab of A and one n tile's slab
    // of B, and the tail band simply has fewer m tiles in it.
    constexpr int kGroupM = 4;
    const int num_m = (M + BM - 1) / BM;
    const int num_n = (N + BN - 1) / BN;
    const int group_span = kGroupM * num_n;
    const int gid = int(blockIdx.x) / group_span;
    const int within = int(blockIdx.x) - gid * group_span;
    const int m_span = (num_m - gid * kGroupM) < kGroupM ? (num_m - gid * kGroupM) : kGroupM;
    const int m_tile = (gid * kGroupM + within % m_span) * BM;
    const int n_tile = (within / m_span) * BN;
    const int batch = detail::gemm_block_batch(gp);

    const AElem* const A =
        static_cast<const AElem*>(gp.a) + detail::gemm_batch_offset(gp, gp.a_batch, batch);
    const int8_t* const B =
        static_cast<const int8_t*>(gp.b) + detail::gemm_batch_offset(gp, gp.b_batch, batch);

    /// One 16 B segment of a k-major panel per thread per pass.
    const int vrow = tid / S::kVecPerRow;
    const int vcol = (tid % S::kVecPerRow) * 16;
    // A 16 B cp.async needs the base and the pitch 16 B aligned (`q.a_vec` /
    // `q.b_vec`, decided by the launcher); a 4 B word otherwise, and a byte only
    // when even that fails.
    const bool aWord = (reinterpret_cast<uintptr_t>(A) % 4 == 0) && (gp.lda % 4 == 0);
    const bool bWord = (reinterpret_cast<uintptr_t>(B) % 4 == 0) && (gp.ldb % 4 == 0);

    // One call is one committed group, whether or not it copies anything: the
    // wait's group count has to stay uniform across the ring.
    auto fill = [&](int kt, int stage) {
        if (kt >= k_tiles) {
            cp_async_commit_group();
            return;
        }
        const int k0 = kt * BK;
        int8_t* const sA = sAbase + stage * S::kABytes;
        int8_t* const sB = sBbase + stage * S::kBBytes;
        const bool kFull = k0 + BK <= K;
        const bool aFull = kFull && m_tile + BM <= M;
        const bool bFull = kFull && n_tile + BN <= N;
        const int8_t* const a8 = reinterpret_cast<const int8_t*>(A);

        // The async half first, so the transpose's global reads overlap it.
        if constexpr (TransB) {
            if (bFull) {
                i8_fill_kcontig<true, S, BN>(sB, B, gp.ldb, n_tile, N, k0, K, vrow, vcol, q.b_vec,
                                             bWord);
            } else {
                i8_fill_kcontig<false, S, BN>(sB, B, gp.ldb, n_tile, N, k0, K, vrow, vcol, q.b_vec,
                                              bWord);
            }
        }
        if constexpr (!TransA) {
            if (aFull) {
                i8_fill_kcontig<true, S, BM>(sA, a8, gp.lda, m_tile, M, k0, K, vrow, vcol, q.a_vec,
                                             aWord);
            } else {
                i8_fill_kcontig<false, S, BM>(sA, a8, gp.lda, m_tile, M, k0, K, vrow, vcol, q.a_vec,
                                              aWord);
            }
        } else {
            // A arrives `[k][m]`: the transpose fill's 4x4 byte blocks are the
            // only way to a k-major panel without a scatter per byte.
            detail::i8_transpose_fill<BM, BK>(sA, LD, a8, m_tile, k0, M, K, int(gp.lda), tid,
                                              S::threads);
        }
        if constexpr (!TransB) {
            detail::i8_transpose_fill<BN, BK>(sB, LD, B, n_tile, k0, N, K, int(gp.ldb), tid,
                                              S::threads);
        }
        cp_async_commit_group();
    };

    int32_t acc[MA][NA][4];
#pragma unroll
    for (int i = 0; i < MA; ++i)
#pragma unroll
        for (int j = 0; j < NA; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                acc[i][j][t] = 0;
            }

    // Prime every stage but one; the loop then keeps exactly ST groups in
    // flight, so `wait_group<ST-2>` retires precisely the tile about to be read.
#pragma unroll
    for (int t = 0; t < ST - 1; ++t) {
        fill(t, t % ST);
    }
    for (int kt = 0; kt < k_tiles; ++kt) {
        const int stage = kt % ST;
        // One barrier a tile: the wait retires this tile's group, the barrier
        // then publishes it and retires every warp's reads of the tile before
        // it -- whose stage is the one the fill now overwrites.
        cp_async_wait_group<ST - 2>();
        __syncthreads();
        fill(kt + ST - 1, (kt + ST - 1) % ST);

        manual_gemm_i8_tile<S, AElem>(sAbase + stage * S::kABytes, sBbase + stage * S::kBBytes,
                                      warp_m, warp_n, lane, acc);
    }

    // ---- the destination --------------------------------------------------
    // C: lane l holds (m = l/4, n = 2*(l%4)), plus the +1 column and the +8 row,
    // so the two adjacent columns of an atom row leave as one store whenever the
    // pitch keeps that pair's alignment.
    const int64_t c_off = detail::gemm_batch_offset(gp, gp.c_batch, batch);
    const int32_t* const r_a_b = q.r_a + int64_t(batch) * int64_t(gp.m);
    const int32_t* const r_b_b = q.r_b + int64_t(batch) * int64_t(gp.n);
    const bool even_pitch = (gp.ldc % 2) == 0;
    const int cm = lane / 4;
    const int cn = 2 * (lane % 4);

    if constexpr (!Dequant) {
        // s32 keeps the raw integer; s8/u8 requantize with the CPU's exact
        // round-away-from-zero. One branch for the kind, then one loop nest.
        const bool wide = q.out == I8OutKind::S32;
        int32_t* const c32 = wide ? static_cast<int32_t*>(gp.c) + c_off : nullptr;
        int8_t* const c8 = wide ? nullptr : static_cast<int8_t*>(gp.c) + c_off;
        const bool pair32 = wide && even_pitch && (reinterpret_cast<uintptr_t>(c32) % 8) == 0;
        const bool pair2 = !wide && even_pitch && (reinterpret_cast<uintptr_t>(c8) % 2) == 0;
        // The compensation `raw - zp_a*R_b[n] - (R_a[m] - K*zp_a)*zp_b[n]`
        // factors into a row scalar and a column scalar,
        //   v = raw - P[m]*R[n] - Q[m]*S[n],
        // P = zp_a + a_shift, Q = R_a + a_shift*K - K*zp_a, R = R_b, S = zp_b.
        // A thread's warp tile spans only MA*2 rows and NA*2 columns, so the
        // four global loads the per-element form costs collapse to a couple of
        // dozen -- and the hundreds of IMADs fall to two per element. This is
        // where the profile said the instructions were: in an int8 GEMM there is
        // one MMA per output element, so every instruction the epilogue spends
        // is one the tensor pipe does not get.
        const auto vint = [](int32_t raw, int32_t P, int32_t Q, int32_t R, int32_t S, int32_t bias,
                             bool relu) {
            int32_t v = raw - P * R - Q * S + bias;
            if (relu) {
                v = v < 0 ? 0 : v;
            }
            return v;
        };
        const int32_t zp_b_scalar = q.zp_b_scalar;
        const int32_t* const zp_b = q.zp_b;
        const int32_t* const bias32 = static_cast<const int32_t*>(q.bias);
        // The requant's two scale factors are per row and per column as well.
        // A division, not a reciprocal multiply: the CPU oracle divides, and
        // the two are not bit-identical.
        const float rq_out = q.out_scale;
        const float rq_zp = float(q.out_zp);
        const bool s8out = q.out == I8OutKind::S8;
#pragma unroll
        for (int i = 0; i < MA; ++i) {
            const int gm = m_tile + warp_m * WM + i * 16 + cm;
            if (gm >= M) {
                continue;
            }
            int32_t* const r32 = wide ? c32 + int64_t(gm) * gp.ldc : nullptr;
            int8_t* const rb = wide ? nullptr : c8 + int64_t(gm) * gp.ldc;
            const bool lo8 = gm + 8 < M;
            // Rows gm and gm+8: two scalars each, and the requant's row factor.
            const int32_t zp0 = (q.zp_a != nullptr ? q.zp_a[gm] : q.zp_a_scalar) + q.a_shift;
            const int32_t P0 = zp0;
            const int32_t Q0 = r_a_b[gm] + q.a_shift * gp.k - gp.k * zp0;
            const float rk0 = (q.sc_a != nullptr ? q.sc_a[gm] : q.sc_a_scalar) / rq_out;
            const int32_t zp8 =
                lo8 ? (q.zp_a != nullptr ? q.zp_a[gm + 8] : q.zp_a_scalar) + q.a_shift : 0;
            const int32_t P8 = zp8;
            const int32_t Q8 = lo8 ? r_a_b[gm + 8] + q.a_shift * gp.k - gp.k * zp8 : 0;
            const float rk8 =
                lo8 ? (q.sc_a != nullptr ? q.sc_a[gm + 8] : q.sc_a_scalar) / rq_out : 0.0f;
#pragma unroll
            for (int j = 0; j < NA; ++j) {
                const int gn = n_tile + warp_n * WN + j * 8 + cn;
                if (gn >= N) {
                    continue;
                }
                const bool hi = gn + 1 < N;
                // Columns gn and gn+1.
                const int32_t R0 = r_b_b[gn];
                const int32_t S0 = zp_b != nullptr ? zp_b[gn] : zp_b_scalar;
                const int32_t B0 = bias32 != nullptr ? bias32[gn] : 0;
                const float sb0 = q.sc_b != nullptr ? q.sc_b[gn] : q.sc_b_scalar;
                const int32_t R1 = hi ? r_b_b[gn + 1] : 0;
                const int32_t S1 = hi ? (zp_b != nullptr ? zp_b[gn + 1] : zp_b_scalar) : 0;
                const int32_t B1 = (hi && bias32 != nullptr) ? bias32[gn + 1] : 0;
                const float sb1 = hi ? (q.sc_b != nullptr ? q.sc_b[gn + 1] : q.sc_b_scalar) : 0.0f;
                if (wide) {
                    const int32_t v0 = vint(acc[i][j][0], P0, Q0, R0, S0, B0, q.relu);
                    const int32_t v1 = hi ? vint(acc[i][j][1], P0, Q0, R1, S1, B1, q.relu) : 0;
                    const int32_t v2 = lo8 ? vint(acc[i][j][2], P8, Q8, R0, S0, B0, q.relu) : 0;
                    const int32_t v3 =
                        (lo8 && hi) ? vint(acc[i][j][3], P8, Q8, R1, S1, B1, q.relu) : 0;
                    int32_t* const row8 = r32 + 8 * gp.ldc;
                    if (pair32 && hi) {
                        *reinterpret_cast<int2*>(r32 + gn) = make_int2(v0, v1);
                        if (lo8) {
                            *reinterpret_cast<int2*>(row8 + gn) = make_int2(v2, v3);
                        }
                    } else {
                        r32[gn] = v0;
                        if (hi) {
                            r32[gn + 1] = v1;
                        }
                        if (lo8) {
                            row8[gn] = v2;
                            if (hi) {
                                row8[gn + 1] = v3;
                            }
                        }
                    }
                } else {
                    // Same factoring, and the requant's per-row and per-column
                    // scales are already resolved above.
                    const uint8_t b0 = uint8_t(
                        s8out ?
                            int32_t(detail::i8_requant<int8_t>(
                                vint(acc[i][j][0], P0, Q0, R0, S0, B0, q.relu), rk0, sb0, rq_zp)) :
                            int32_t(detail::i8_requant<uint8_t>(
                                vint(acc[i][j][0], P0, Q0, R0, S0, B0, q.relu), rk0, sb0, rq_zp)));
                    const uint8_t b1 =
                        hi ? uint8_t(s8out ? int32_t(detail::i8_requant<int8_t>(
                                                 vint(acc[i][j][1], P0, Q0, R1, S1, B1, q.relu),
                                                 rk0, sb1, rq_zp)) :
                                             int32_t(detail::i8_requant<uint8_t>(
                                                 vint(acc[i][j][1], P0, Q0, R1, S1, B1, q.relu),
                                                 rk0, sb1, rq_zp))) :
                             uint8_t(0);
                    const uint8_t b2 =
                        lo8 ? uint8_t(s8out ? int32_t(detail::i8_requant<int8_t>(
                                                  vint(acc[i][j][2], P8, Q8, R0, S0, B0, q.relu),
                                                  rk8, sb0, rq_zp)) :
                                              int32_t(detail::i8_requant<uint8_t>(
                                                  vint(acc[i][j][2], P8, Q8, R0, S0, B0, q.relu),
                                                  rk8, sb0, rq_zp))) :
                              uint8_t(0);
                    const uint8_t b3 =
                        (lo8 && hi) ?
                            uint8_t(s8out ? int32_t(detail::i8_requant<int8_t>(
                                                vint(acc[i][j][3], P8, Q8, R1, S1, B1, q.relu), rk8,
                                                sb1, rq_zp)) :
                                            int32_t(detail::i8_requant<uint8_t>(
                                                vint(acc[i][j][3], P8, Q8, R1, S1, B1, q.relu), rk8,
                                                sb1, rq_zp))) :
                            uint8_t(0);
                    int8_t* const row8 = rb + 8 * gp.ldc;
                    if (pair2 && hi) {
                        *reinterpret_cast<uint16_t*>(rb + gn) = uint16_t(b0 | (uint16_t(b1) << 8));
                        if (lo8) {
                            *reinterpret_cast<uint16_t*>(row8 + gn) =
                                uint16_t(b2 | (uint16_t(b3) << 8));
                        }
                    } else {
                        rb[gn] = int8_t(b0);
                        if (hi) {
                            rb[gn + 1] = int8_t(b1);
                        }
                        if (lo8) {
                            row8[gn] = int8_t(b2);
                            if (hi) {
                                row8[gn + 1] = int8_t(b3);
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    // A dequantizing destination: f32 or f16 under the operator's epilogue.
    const bool wide = q.out == I8OutKind::F32;
    float* const cf = wide ? static_cast<float*>(gp.c) + c_off : nullptr;
    __half* const ch = wide ? nullptr : static_cast<__half*>(gp.c) + c_off;
    const bool pair32 = wide && even_pitch && (reinterpret_cast<uintptr_t>(cf) % 8) == 0;
    const bool pair4 = !wide && even_pitch && (reinterpret_cast<uintptr_t>(ch) % 4) == 0;
#pragma unroll
    for (int i = 0; i < MA; ++i) {
        const int gm = m_tile + warp_m * WM + i * 16 + cm;
        if (gm >= M) {
            continue;
        }
        float* const rf = wide ? cf + int64_t(gm) * gp.ldc : nullptr;
        __half* const rh = wide ? nullptr : ch + int64_t(gm) * gp.ldc;
        const bool lo8 = gm + 8 < M;
#pragma unroll
        for (int j = 0; j < NA; ++j) {
            const int gn = n_tile + warp_n * WN + j * 8 + cn;
            if (gn >= N) {
                continue;
            }
            const bool hi = gn + 1 < N;
            const float f0 = detail::i8_fp_value(q, acc[i][j][0], gm, gn, r_a_b, r_b_b);
            const float f1 =
                hi ? detail::i8_fp_value(q, acc[i][j][1], gm, gn + 1, r_a_b, r_b_b) : 0.0f;
            const float f2 =
                lo8 ? detail::i8_fp_value(q, acc[i][j][2], gm + 8, gn, r_a_b, r_b_b) : 0.0f;
            const float f3 =
                (lo8 && hi) ? detail::i8_fp_value(q, acc[i][j][3], gm + 8, gn + 1, r_a_b, r_b_b) :
                              0.0f;
            if (wide) {
                float* const row8 = rf + 8 * gp.ldc;
                if (pair32 && hi) {
                    *reinterpret_cast<float2*>(rf + gn) = make_float2(f0, f1);
                    if (lo8) {
                        *reinterpret_cast<float2*>(row8 + gn) = make_float2(f2, f3);
                    }
                } else {
                    rf[gn] = f0;
                    if (hi) {
                        rf[gn + 1] = f1;
                    }
                    if (lo8) {
                        row8[gn] = f2;
                        if (hi) {
                            row8[gn + 1] = f3;
                        }
                    }
                }
            } else {
                __half* const row8 = rh + 8 * gp.ldc;
                if (pair4 && hi) {
                    const __half2 h0 = __floats2half2_rn(f0, f1);
                    *reinterpret_cast<uint32_t*>(rh + gn) = *reinterpret_cast<const uint32_t*>(&h0);
                    if (lo8) {
                        const __half2 h8 = __floats2half2_rn(f2, f3);
                        *reinterpret_cast<uint32_t*>(row8 + gn) =
                            *reinterpret_cast<const uint32_t*>(&h8);
                    }
                } else {
                    rh[gn] = __float2half(f0);
                    if (hi) {
                        rh[gn + 1] = __float2half(f1);
                    }
                    if (lo8) {
                        row8[gn] = __float2half(f2);
                        if (hi) {
                            row8[gn + 1] = __float2half(f3);
                        }
                    }
                }
            }
        }
    }
}

} // namespace nnops::backend::cuda
