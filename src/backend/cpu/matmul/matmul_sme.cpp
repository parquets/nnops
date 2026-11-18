/// @file matmul_sme.cpp
/// @brief Whole-product MatMul through the AArch64 SME kernels.
///
/// SME's outer-product accumulator (ZA) makes a 16x64 output block the natural
/// unit, and its throughput lives almost entirely in the K loop: the fixed cost
/// is the ZA read-back and store, one per output row-panel, which only
/// amortises over a long K. That is why this route sets its own kc (sme_kc,
/// far above the NEON plan's 128, as long as the B slab fits) instead of
/// borrowing the plan, and why it owns
/// its pack layouts — [m/16][k][16] for A, [n/64][k][64] for B, K-pair
/// interleaved for f16 (see mma_sme.hpp and pack_*_sme.hpp).
///
/// The route is deliberately separate from the plan-driven skeleton: the tiles,
/// the pack geometry and the thread split all differ, and the two share only
/// the batch walk, the epilogue and the accumulator-init conventions. It runs
/// when the CPU has SME, both operands are dense f32 or f16 of the same type,
/// alpha is 1, and M and N fill at least one 16-row / 16-column face. Everything
/// else falls through to the generic routes.
///
/// Threads split N; each worker packs its own B slab for the k-block it is on,
/// so the slabs are disjoint and need no synchronisation.

#include "matmul_internal.h"

#if defined(NNOPS_ARCH_SME)

#include "backend/cpu/aarch64/mma_f16_sme.hpp"
#include "backend/cpu/aarch64/mma_f32_sme.hpp"
#include "backend/cpu/aarch64/mma_sme.hpp"
#include "backend/cpu/aarch64/pack_f16_sme.hpp"
#include "backend/cpu/aarch64/pack_f32_sme.hpp"
#include "backend/cpu/common/memory_pool.hpp"
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"
#include "nnops/detail/half.hpp"

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace nnops::backend::cpu {

using namespace aarch64;

namespace {

constexpr int kSmeRows = 16;  // rows in one ZA tile
constexpr int kSmeMSmall = 4; // MatMul's GEMV gate: M at or below it is theirs
constexpr int kSmeMc = 64;   // rows per A slab: four ZA tiles
/// Widest column block one worker takes; also the bound that keeps a worker's
/// B slab at 4 MiB for the kc below.
constexpr int kSmeNMax = 1024;

/// K per block, chosen by measurement rather than by a model. Long is what the
/// kernel wants: the ZA read-back and the stores cost a fixed ~1360 cycles per
/// 16x64 output block, which only amortises over K, and every extra k-block
/// also re-reads C. Two regimes show up on an M4 (1T, per-worker column block
/// of 448-1024):
///   - too short is the dominant loss: kc 512 -> 1024 on 1024^3 f32 moved the
///     kernel from 1129 to 1380 GFLOPS and the route from 823 to 1121;
///   - past the point where a worker's B slab (nw*kc*sizeof(T)) leaves L2 it
///     turns back: f32 prefill 509 -> 433 GFLOPS going 1024 -> 2048, while f16
///     (half the bytes) keeps gaining, 624 -> 770.
/// So each dtype gets the length that measured best at its widest slab, ~4 MiB
/// per worker at nw = kSmeNMax.
template <typename T> constexpr int sme_kc() noexcept
{
    return sizeof(T) == 2 ? 2048 : 1024;
}

/// The dtype-named pack/kernel calls; the routing above is type-agnostic.
template <typename T> struct SmeOp;

template <> struct SmeOp<float> {
    static int64_t packed_a(int mc, int kc) { return packed_a_elems_f32_sme(mc, kc); }
    static int64_t packed_b(int nc, int kc) { return packed_b_elems_f32_sme(nc, kc); }
    static void pack_a(int mc, int kc, float* d, const float* s, int64_t ld, bool trans)
    {
        pack_a_f32_sme(mc, kc, d, s, ld, trans);
    }
    static void pack_b(int nc, int kc, float* d, const float* s, int64_t ld, bool trans)
    {
        pack_b_f32_sme(nc, kc, d, s, ld, trans);
    }
    static void mma(int mc, int nc, int kc, float* c, int ldc, const float* a, const float* b,
                     bool zero_mode, float lo, float hi, const float* col_bias,
                     int64_t b_block_stride)
    {
        mma_f32_sme(mc, nc, kc, c, ldc, a, b, zero_mode, lo, hi, col_bias, b_block_stride);
    }
};

template <> struct SmeOp<half> {
    static int64_t packed_a(int mc, int kc) { return packed_a_elems_f16_sme(mc, kc); }
    static int64_t packed_b(int nc, int kc) { return packed_b_elems_f16_sme(nc, kc); }
    static void pack_a(int mc, int kc, half* d, const half* s, int64_t ld, bool trans)
    {
        pack_a_f16_sme(mc, kc, d, s, ld, trans);
    }
    static void pack_b(int nc, int kc, half* d, const half* s, int64_t ld, bool trans)
    {
        pack_b_f16_sme(nc, kc, d, s, ld, trans);
    }
    static void mma(int mc, int nc, int kc, half* c, int ldc, const half* a, const half* b,
                     bool zero_mode, float lo, float hi, const half* col_bias,
                     int64_t b_block_stride)
    {
        mma_f16_sme(mc, nc, kc, c, ldc, a, b, zero_mode, lo, hi, col_bias, b_block_stride);
    }
};

/// One worker's column block over the whole product: pack B per k-block, pack A
/// per row slab, accumulate.
template <typename T>
void sme_block(const MatMulAttributes& attrs, T* c_base, int ldc, const T* a, int lda,
               const T* b, int ldb, int M, int n0, int nw, int K, const T* bias, float beta,
               const T* prepacked)
{
    using Op = SmeOp<T>;
    const int kc_block = sme_kc<T>();
    PoolArray<T> pack_b(prepacked != nullptr ? 0 : static_cast<size_t>(Op::packed_b(nw, kc_block)));
    PoolArray<T> pack_a(static_cast<size_t>(Op::packed_a(kSmeMc, kc_block)));
    // A prepacked blob is [ceil(N/64)][K][64]: one k's 64 columns contiguous and
    // blocks K*64 apart, so a k block is just an offset into it.
    const int64_t blob_stride = std::is_same_v<T, float> ?
                                    static_cast<int64_t>(K) * sme_nr_max :
                                    static_cast<int64_t>((K + 1) / 2) * (2 * sme_nr_max);
    const int64_t blob_block0 = static_cast<int64_t>(n0 / sme_nr_max) * blob_stride;

    for (int k = 0; k < K; k += kc_block) {
        const int kc = std::min(kc_block, K - k);
        const bool last_k = (k + kc >= K);
        // The pack reads from the first row it owns; the k and n offsets land
        // it on this block's corner.
        if (prepacked == nullptr) {
            Op::pack_b(nw, kc, pack_b.data(),
                       b + (attrs.transpose_b ? static_cast<int64_t>(n0) * ldb + k :
                                                static_cast<int64_t>(k) * ldb + n0),
                       ldb, attrs.transpose_b);
        }

        for (int m0 = 0; m0 < M; m0 += kSmeMc) {
            const int mc = std::min(kSmeMc, M - m0);
            T* c_tile = c_base + static_cast<int64_t>(m0) * ldc + n0;

            // Accumulator init, the same order the tiled route uses: a bias
            // with nothing to fold seeds the accumulator (and the first block
            // overwrites C), otherwise it primes C together with beta·C_old; a
            // bare beta scales C in place, and the block then accumulates.
            const bool seed = k == 0 && bias != nullptr && beta == 0.0f;
            if (k == 0) {
                if (bias != nullptr) {
                    if (!seed) {
                        tile_scale_add_bias(c_tile, ldc, beta, bias + n0, mc, nw);
                    }
                } else if (beta != 0.0f && beta != 1.0f) {
                    tile_scale(c_tile, ldc, beta, mc, nw);
                }
            }

            Op::pack_a(mc, kc, pack_a.data(),
                       a + (attrs.transpose_a ? static_cast<int64_t>(k) * lda + m0 :
                                                static_cast<int64_t>(m0) * lda + k),
                       lda, attrs.transpose_a);

            const auto [lo, hi] = kblock_clamp(attrs, last_k);
            Op::mma(mc, nw, kc, c_tile, ldc, pack_a.data(),
                     prepacked != nullptr ? prepacked + blob_block0 + static_cast<int64_t>(k) * sme_nr_max
                                          : pack_b.data(),
                     /*zero_mode=*/k == 0 && beta == 0.0f, lo, hi,
                     seed ? bias + n0 : nullptr, prepacked != nullptr ? blob_stride : 0);

            if (last_k && has_inplace_epilogue(attrs)) {
                epilogue_inplace(mc, nw, c_tile, ldc, static_cast<const T*>(nullptr),
                                 attrs.epilogue);
            }
        }
    }
}

template <typename T>
void matmul_sme_run(const MatMulAttributes& attrs, TensorView& output,
                    std::span<const TensorView> inputs, const ComputeContext& ctx, int M, int N,
                    int K, const T* prepacked)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    const int lda = static_cast<int>(a.row_stride_elems());
    const int ldb = static_cast<int>(b.row_stride_elems());
    const int ldc = static_cast<int>(output.row_stride_elems());

    const T* a_base = a.ptr<T>();
    const T* b_base = b.ptr<T>();
    T* c_base = output.ptr<T>();
    const T* bias = inputs.size() > 2 ? inputs[2].ptr<T>() : nullptr;
    const float beta = attrs.residual_pre_activation ? 1.0f : attrs.beta;

    // One worker per 64-column block at most, and never wider than the cap that
    // keeps a B slab small.
    const int64_t n_blocks_total = (N + sme_nr_max - 1) / sme_nr_max;
    const int workers = static_cast<int>(
        std::max<int64_t>(1, std::min<int64_t>(ctx.cpu.thread_count(), n_blocks_total)));
    int nw = std::min(kSmeNMax, (N + workers - 1) / workers);
    nw = ((nw + sme_nr_max - 1) / sme_nr_max) * sme_nr_max;
    const int blocks = (N + nw - 1) / nw;

    const BatchWalk batch(a, b);
    ctx.cpu.run(0, batch.total * blocks, [&](int64_t t) {
        const int64_t bi = t / blocks;
        const int n0 = static_cast<int>(t % blocks) * nw;
        int64_t a_off, b_off, c_off;
        batch.offsets(bi, output, a_off, b_off, c_off);
        sme_block<T>(attrs, c_base + c_off, ldc, a_base + a_off, lda, b_base + b_off, ldb, M, n0,
                     std::min(nw, N - n0), K, bias, beta, prepacked);
    });
}

} // namespace

bool matmul_sme(const MatMulAttributes& attrs, TensorView& output,
                std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    if (!available_sme() || inputs.size() < 2) {
        return false;
    }
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    const DataType dt = a.data_type();
    if ((dt != DataType::f32 && dt != DataType::f16) || b.data_type() != dt ||
        output.data_type() != dt) {
        return false;
    }
    // The NEON panel blob is unreadable here; the SME blob is ours and needs no
    // per-call pack. A block-quantised B is not this dtype at all.
    if ((is_packed_matmul_b(b.layout()) && b.layout() != TensorLayout::PackedBSme) ||
        b.rank() < 2) {
        return false;
    }
    // alpha rides the RHS pack on the generic routes; this one would have to
    // fold it into every element, so a scaled product stays there.
    if (attrs.alpha != 1.0f) {
        return false;
    }

    // M: the small routes above own M <= 4 (GEMV) and the tiny products, so the
    // SME route starts where they stop. It wins from there: measured 4.0x the
    // NEON tiled engine at M = 8 (1024x1024 face), 6.7x at M = 16. A short M is
    // padded onto the 16-row tile, which costs rows but not columns.
    // N: one full 16-column panel, or the pass would waste lanes.
    const auto [M64, N64, K64] = matmul_dims(attrs, a, b);
    if (M64 <= kSmeMSmall || N64 < kSmeRows || K64 <= 0) {
        return false;
    }
    const int M = static_cast<int>(M64);
    const int N = static_cast<int>(N64);
    const int K = static_cast<int>(K64);

    const bool has_blob = b.layout() == TensorLayout::PackedBSme;
    if (dt == DataType::f32) {
        matmul_sme_run<float>(attrs, output, inputs, ctx, M, N, K,
                              has_blob ? b.ptr<float>() : nullptr);
    } else {
        matmul_sme_run<half>(attrs, output, inputs, ctx, M, N, K,
                             has_blob ? b.ptr<half>() : nullptr);
    }
    return true;
}

} // namespace nnops::backend::cpu

#else // no SME kernels compiled in

namespace nnops::backend::cpu {

bool matmul_sme(const MatMulAttributes&, TensorView&, std::span<const TensorView>,
                const ComputeContext&)
{
    return false;
}

} // namespace nnops::backend::cpu

#endif
