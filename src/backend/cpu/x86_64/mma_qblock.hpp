#pragma once
/// @file mma_qblock.hpp
/// @brief Fused block-quant × f32 dot kernels for the small-M MatMul route.
///
/// The packed-B route decodes a Q5_1/Q4_1 block into a dense panel and then
/// reads that panel back in the MMA. At M=1 the panel is consumed by exactly
/// one row, so the decode + panel round trip is the whole call. These kernels
/// skip the panel: each block is decoded into registers and multiplied
/// straight into the accumulator, so a block is read once (24 B for Q5_1, 20 B
/// for Q4_1) and never written back.
///
/// The two formats differ only in the block stream: same 32-value blocks, same
/// fp16 `d`/`m` affine decode, and Q5_1's extra 5th-bit mask is the only work
/// Q4_1 does not do. So the kernels are written once and parameterized by a
/// block policy — `Q5Block` or `Q4Block` — that supplies the stride, the width
/// and the two decode entry points.
///
/// Two physical B layouts, each with its natural loop shape:
///
///  * trans ([N][K], the Linear-weight layout): one n row at a time, sweeping
///    its k blocks linearly. MR accumulator vectors (MR ≤ 8) hold one row each;
///    a block decode is amortized over all MR rows.
///  * copy ([K][N]): one 32-wide n block at a time. A block is exactly one
///    quant block per k row, and the outer-product form needs four
///    accumulators per row, which caps MR at 2.
///
/// Both compute the whole K in one call, so the clamp is applied once, on the
/// final value. Accumulation reorders the reduction (four partial sums reduced
/// at the end for trans), which is why callers must keep these on tiles that
/// are not shared with the packed route.
///
/// @p MR is the exact number of live rows. A partial M panel instantiates its
/// own width, so no row aliases a duplicate and no per-row guard is needed.

#include <immintrin.h>

#include <algorithm>
#include <cstdint>

#include "backend/cpu/common/common_utils.hpp"
#include "nnops/core/data_type.hpp"
#include "pack_q4_1.hpp"
#include "pack_q5_1.hpp"

namespace nnops::backend::cpu::x86_64 {

// =========================================================================
//  Block policies
// =========================================================================
//
/// Everything the fused kernels need to read one 32-value quant block. Both
/// formats encode `d * q + m`; only the quant width differs.

struct Q5Block {
    static constexpr int block_elems = q5_1_block_elems;
    static constexpr int block_bytes = q5_1_block_bytes;
    static constexpr DataType dtype = DataType::q5_1;

    /// Largest M the fused kernels still win, per layout, measured at
    /// K=N=4096 (the dot kernel reuses one decode across MR = 8 rows, the
    /// outer-product one across 2, and the packed route decodes B once).
    static constexpr int fused_max_m_trans = 16;
    static constexpr int fused_max_m_copy = 4;

    static inline void decode_ps(const uint8_t* p, __m256& v0, __m256& v1,
                                 __m256& v2, __m256& v3) noexcept {
        decode_q5_1_block_ps(p, v0, v1, v2, v3);
    }

    template <class T>
    static inline void decode(const uint8_t* p, T* out) noexcept {
        decode_q5_1_block<T>(p, out);
    }
};

struct Q4Block {
    static constexpr int block_elems = q4_1_block_elems;
    static constexpr int block_bytes = q4_1_block_bytes;
    static constexpr DataType dtype = DataType::q4_1;

    /// Same measurement as Q5Block's, except the copy limit: a Q4_1 decode does
    /// less work per block, so re-decoding B once per 2-row panel stays cheaper
    /// than the packed panel round trip for twice as many rows. At M=8 the
    /// fused route is still +34% at 4096² and +89% at 2048², and it only loses
    /// from M=16 at 4096² (ratio 0.85) / M=24.
    static constexpr int fused_max_m_trans = 16;
    static constexpr int fused_max_m_copy = 8;

    static inline void decode_ps(const uint8_t* p, __m256& v0, __m256& v1,
                                 __m256& v2, __m256& v3) noexcept {
        decode_q4_1_block_ps(p, v0, v1, v2, v3);
    }

    template <class T>
    static inline void decode(const uint8_t* p, T* out) noexcept {
        decode_q4_1_block<T>(p, out);
    }
};

static_assert(Q5Block::block_elems == Q4Block::block_elems,
              "the fused kernels assume one 32-value block per k step in the "
              "copy layout");

/// Sum the eight lanes of @p v into lane 0.
inline float qblock_hsum_ps(__m256 v) noexcept {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v),
                          _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

/// One row of the trans kernel's k body: the four decoded B vectors against
/// the row's four A vectors, reduced into that row's accumulator.
#define NNOPS_QBLOCK_DOT_ROW(ACC, ROW)                                    \
    do {                                                                  \
        const float* a = A + static_cast<int64_t>(ROW) * lda + k;         \
        __m256 t = _mm256_mul_ps(b0, _mm256_loadu_ps(a));                 \
        t = _mm256_fmadd_ps(b1, _mm256_loadu_ps(a + 8), t);               \
        t = _mm256_fmadd_ps(b2, _mm256_loadu_ps(a + 16), t);              \
        t = _mm256_fmadd_ps(b3, _mm256_loadu_ps(a + 24), t);              \
        ACC = _mm256_add_ps(ACC, t);                                      \
    } while (0)

// =========================================================================
//  trans: B physically [N][K]
// =========================================================================
//
/// Dot-product form. For each of the @p nc columns starting at @p n_start, walk
/// the column's k blocks once, decoding each block into four vectors and
/// multiplying it against the same four-vector slice of every live A row.
///
/// @param C   C[m0][n_start]; already offset by the caller.
/// @param A   A[m0][0], row stride @p lda.
/// @param B   first block of the batch element's [N][K] stream.
template <class BQ, int MR, bool ZERO>
void mma_qblock_f32_trans(int nc, int kc, float* NNOPS_RESTRICT C, int ldc,
                          const float* NNOPS_RESTRICT A, int lda,
                          const uint8_t* NNOPS_RESTRICT B, int64_t row_blocks,
                          int n_start, float alpha, float clamp_min,
                          float clamp_max) noexcept {
    static_assert(MR >= 1 && MR <= 8, "trans qblock kernel: MR in [1, 8]");

    const int64_t row_bytes = row_blocks * static_cast<int64_t>(BQ::block_bytes);
    const uint8_t* src = B + static_cast<int64_t>(n_start) * row_bytes;

    const __m256 vzero = _mm256_setzero_ps();
    const int nblk = kc / BQ::block_elems;

    for (int n = 0; n < nc; ++n, src += row_bytes) {
        // One accumulator chain per row is latency-bound on the
        // mul -> fma -> fma -> fma -> add sequence once the decode stops
        // covering it, which is the M <= 2 case. There, give each row four
        // independent chains (one per chunk) and reduce them once per column.
        // Two rows' worth is all that fits alongside the decoded block; wider
        // panels keep the single chain, where the decode is amortized anyway.
        __m256 q00 = vzero, q01 = vzero, q02 = vzero, q03 = vzero;
        __m256 q10 = vzero, q11 = vzero, q12 = vzero, q13 = vzero;
        __m256 acc0 = vzero, acc1 = vzero, acc2 = vzero, acc3 = vzero;
        __m256 acc4 = vzero, acc5 = vzero, acc6 = vzero, acc7 = vzero;

        int k = 0;
        for (int b = 0; b < nblk; ++b, k += BQ::block_elems) {
            __m256 b0, b1, b2, b3;
            BQ::decode_ps(src + static_cast<int64_t>(b) * BQ::block_bytes,
                          b0, b1, b2, b3);

            if constexpr (MR <= 2) {
#define NNOPS_QBLOCK_CHAINS(Q, ROW)                                           \
                do {                                                          \
                    const float* a = A + static_cast<int64_t>(ROW) * lda + k; \
                    Q##0 = _mm256_fmadd_ps(b0, _mm256_loadu_ps(a), Q##0);     \
                    Q##1 = _mm256_fmadd_ps(b1, _mm256_loadu_ps(a + 8), Q##1); \
                    Q##2 = _mm256_fmadd_ps(b2, _mm256_loadu_ps(a + 16), Q##2);\
                    Q##3 = _mm256_fmadd_ps(b3, _mm256_loadu_ps(a + 24), Q##3);\
                } while (0)
                NNOPS_QBLOCK_CHAINS(q0, 0);
                if constexpr (MR > 1) { NNOPS_QBLOCK_CHAINS(q1, 1); }
#undef NNOPS_QBLOCK_CHAINS
            } else {
                // One four-vector slice per row: 4 mul/fma + 1 add, and the
                // block decode is reused by every row in this panel.
                NNOPS_QBLOCK_DOT_ROW(acc0, 0);
                if constexpr (MR > 1) { NNOPS_QBLOCK_DOT_ROW(acc1, 1); }
                if constexpr (MR > 2) { NNOPS_QBLOCK_DOT_ROW(acc2, 2); }
                if constexpr (MR > 3) { NNOPS_QBLOCK_DOT_ROW(acc3, 3); }
                if constexpr (MR > 4) { NNOPS_QBLOCK_DOT_ROW(acc4, 4); }
                if constexpr (MR > 5) { NNOPS_QBLOCK_DOT_ROW(acc5, 5); }
                if constexpr (MR > 6) { NNOPS_QBLOCK_DOT_ROW(acc6, 6); }
                if constexpr (MR > 7) { NNOPS_QBLOCK_DOT_ROW(acc7, 7); }
            }
        }

        if constexpr (MR <= 2) {
            acc0 = _mm256_add_ps(_mm256_add_ps(q00, q01), _mm256_add_ps(q02, q03));
            if constexpr (MR > 1) {
                acc1 = _mm256_add_ps(_mm256_add_ps(q10, q11), _mm256_add_ps(q12, q13));
            }
        }

        // Trailing partial block: scalar, at most 31 values.
        float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
        float t4 = 0.0f, t5 = 0.0f, t6 = 0.0f, t7 = 0.0f;
        if (k < kc) {
            float vals[BQ::block_elems];
            BQ::template decode<float>(
                src + static_cast<int64_t>(nblk) * BQ::block_bytes, vals);
            const int rem = kc - k;
            for (int i = 0; i < rem; ++i) {
                const int kk = k + i;
                const float bv = vals[i];
                t0 += bv * A[kk];
                if constexpr (MR > 1) { t1 += bv * A[static_cast<int64_t>(lda) + kk]; }
                if constexpr (MR > 2) { t2 += bv * A[2 * static_cast<int64_t>(lda) + kk]; }
                if constexpr (MR > 3) { t3 += bv * A[3 * static_cast<int64_t>(lda) + kk]; }
                if constexpr (MR > 4) { t4 += bv * A[4 * static_cast<int64_t>(lda) + kk]; }
                if constexpr (MR > 5) { t5 += bv * A[5 * static_cast<int64_t>(lda) + kk]; }
                if constexpr (MR > 6) { t6 += bv * A[6 * static_cast<int64_t>(lda) + kk]; }
                if constexpr (MR > 7) { t7 += bv * A[7 * static_cast<int64_t>(lda) + kk]; }
            }
        }

        float* c = C + n;
        auto finish = [&](int row, __m256 acc, float tail) {
            float* cp = c + static_cast<int64_t>(row) * ldc;
            float v = qblock_hsum_ps(acc) + tail;
            if (alpha != 1.0f) { v *= alpha; }
            if constexpr (!ZERO) { v += *cp; }
            *cp = std::min(std::max(v, clamp_min), clamp_max);
        };

        finish(0, acc0, t0);
        if constexpr (MR > 1) { finish(1, acc1, t1); }
        if constexpr (MR > 2) { finish(2, acc2, t2); }
        if constexpr (MR > 3) { finish(3, acc3, t3); }
        if constexpr (MR > 4) { finish(4, acc4, t4); }
        if constexpr (MR > 5) { finish(5, acc5, t5); }
        if constexpr (MR > 6) { finish(6, acc6, t6); }
        if constexpr (MR > 7) { finish(7, acc7, t7); }
    }
}

#undef NNOPS_QBLOCK_DOT_ROW

// =========================================================================
//  copy: B physically [K][N]
// =========================================================================
//
/// Outer-product form. A 32-wide n block spans exactly one quant block per k
/// row, so each k step decodes one block into four vectors and broadcasts one
/// A scalar per live row. Four accumulators per row cap MR at 2.
///
/// @param n_start first column of the tile; must be a multiple of 32 so each
///                decoded block is one panel.
template <class BQ, int MR, bool ZERO>
void mma_qblock_f32_copy(int nc, int kc, float* NNOPS_RESTRICT C, int ldc,
                         const float* NNOPS_RESTRICT A, int lda,
                         const uint8_t* NNOPS_RESTRICT B, int64_t row_blocks,
                         int n_start, float alpha, float clamp_min,
                         float clamp_max) noexcept {
    static_assert(MR == 1 || MR == 2, "copy qblock kernel: MR in {1, 2}");

    const int64_t row_bytes = row_blocks * static_cast<int64_t>(BQ::block_bytes);
    const int64_t blk0 = n_start / BQ::block_elems;

    const __m256 vlo = _mm256_set1_ps(clamp_min);
    const __m256 vhi = _mm256_set1_ps(clamp_max);
    const __m256 vzero = _mm256_setzero_ps();
    const __m256 valpha = _mm256_set1_ps(alpha);
    const bool scaled = (alpha != 1.0f);

    for (int n = 0; n < nc; n += BQ::block_elems) {
        const int width = std::min(BQ::block_elems, nc - n);
        const int64_t blk = blk0 + (n / BQ::block_elems);

        __m256 r0a = vzero, r0b = vzero, r0c = vzero, r0d = vzero;
        __m256 r1a = vzero, r1b = vzero, r1c = vzero, r1d = vzero;

        for (int k = 0; k < kc; ++k) {
            const uint8_t* src = B + static_cast<int64_t>(k) * row_bytes
                               + blk * static_cast<int64_t>(BQ::block_bytes);
            __m256 b0, b1, b2, b3;
            BQ::decode_ps(src, b0, b1, b2, b3);

            const __m256 va = _mm256_broadcast_ss(A + k);
            r0a = _mm256_fmadd_ps(va, b0, r0a);
            r0b = _mm256_fmadd_ps(va, b1, r0b);
            r0c = _mm256_fmadd_ps(va, b2, r0c);
            r0d = _mm256_fmadd_ps(va, b3, r0d);

            if constexpr (MR > 1) {
                const __m256 vb =
                    _mm256_broadcast_ss(A + static_cast<int64_t>(lda) + k);
                r1a = _mm256_fmadd_ps(vb, b0, r1a);
                r1b = _mm256_fmadd_ps(vb, b1, r1b);
                r1c = _mm256_fmadd_ps(vb, b2, r1c);
                r1d = _mm256_fmadd_ps(vb, b3, r1d);
            }
        }

        // Clamp (and optionally accumulate into) one row's four n-chunks.
        auto emit = [&](int row, __m256 q0, __m256 q1, __m256 q2, __m256 q3) {
            float* cp = C + static_cast<int64_t>(row) * ldc + n;
            if (width == BQ::block_elems) {
                if (scaled) {
                    q0 = _mm256_mul_ps(q0, valpha);
                    q1 = _mm256_mul_ps(q1, valpha);
                    q2 = _mm256_mul_ps(q2, valpha);
                    q3 = _mm256_mul_ps(q3, valpha);
                }
                if constexpr (!ZERO) {
                    q0 = _mm256_add_ps(q0, _mm256_loadu_ps(cp + 0));
                    q1 = _mm256_add_ps(q1, _mm256_loadu_ps(cp + 8));
                    q2 = _mm256_add_ps(q2, _mm256_loadu_ps(cp + 16));
                    q3 = _mm256_add_ps(q3, _mm256_loadu_ps(cp + 24));
                }
                _mm256_storeu_ps(cp + 0, _mm256_min_ps(_mm256_max_ps(q0, vlo), vhi));
                _mm256_storeu_ps(cp + 8, _mm256_min_ps(_mm256_max_ps(q1, vlo), vhi));
                _mm256_storeu_ps(cp + 16, _mm256_min_ps(_mm256_max_ps(q2, vlo), vhi));
                _mm256_storeu_ps(cp + 24, _mm256_min_ps(_mm256_max_ps(q3, vlo), vhi));
            } else {
                float tmp[BQ::block_elems];
                _mm256_storeu_ps(tmp + 0, q0);
                _mm256_storeu_ps(tmp + 8, q1);
                _mm256_storeu_ps(tmp + 16, q2);
                _mm256_storeu_ps(tmp + 24, q3);
                for (int i = 0; i < width; ++i) {
                    float v = tmp[i];
                    if (scaled) { v *= alpha; }
                    if constexpr (!ZERO) { v += cp[i]; }
                    cp[i] = std::min(std::max(v, clamp_min), clamp_max);
                }
            }
        };

        emit(0, r0a, r0b, r0c, r0d);
        if constexpr (MR > 1) { emit(1, r1a, r1b, r1c, r1d); }
    }
}

}  // namespace nnops::backend::cpu::x86_64
