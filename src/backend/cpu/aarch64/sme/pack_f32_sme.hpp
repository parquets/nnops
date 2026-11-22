#pragma once
/// @file pack_f32_sme.hpp
/// @brief AArch64 SME float32 pack kernels for MMA LHS/RHS.
///
/// Repacks matrix rows/columns into the layouts mma_f32_sme.hpp reads. Both
/// transforms write k-major blocks whose 32 (A) / 64 (B) values per k are
/// contiguous, not the NEON pack's 12/16-column panels:
///   - A: [m/32][kc][32], the 32 rows of an M tile contiguous per k, so the
///     kernel's two A vectors are one SVE2 pair load. Padding rows past @p mc
///     are left undefined: the kernel loads them only under a predicate.
///   - B: [n/32][kc][32], the 32 columns of an n tile contiguous per k, so
///     its two B vectors are one pair load and the tile's stream is
///     sequential. Padding columns past @p nc are zeroed by the pack; the
///     kernel's predicates keep them out of the accumulator anyway.
///
/// `trans` reads the physical K×M (A) / N×K (B) layout, which is what MatMul's
/// transpose_a / transpose_b describe.

#if defined(NNOPS_ARCH_SME)

#include <algorithm>
#include <cstdint>

#include <arm_neon.h>

#include "backend/cpu/aarch64/sme/mma_sme.hpp"

namespace nnops::backend::cpu::aarch64 {

/// Elements one packed A slab occupies: [ceil(mc/32)][kc][32]. The 32-row
/// block is the kernel's 2x2 ZA tile, so its two A vectors are one pair load.
inline int64_t packed_a_elems_f32_sme(int mc, int kc) noexcept
{
    constexpr int kBlock = 2 * sme_mr;
    return static_cast<int64_t>((mc + kBlock - 1) / kBlock) * kc * kBlock;
}

/// Elements one packed B slab occupies: [ceil(nc/32)][kc][32].
inline int64_t packed_b_elems_f32_sme(int nc, int kc) noexcept
{
    return static_cast<int64_t>((nc + sme_fp_nr - 1) / sme_fp_nr) * kc * sme_fp_nr;
}

/// The A transpose with ZA as the buffer: rows go in horizontally, columns
/// come out vertically, so the 32x32 block costs 32 loads and 32 stores and no
/// shuffles. KleidiAI packs its LHS the same way; the NEON transpose below is
/// the fallback when the hardware has no ZA.
__arm_locally_streaming __arm_new("za") __attribute__((target("+sme2")))
inline void pack_a_f32_sme_za(int mc, int kc, float* dst, const float* src,
                              int64_t lds) noexcept
{
    constexpr int kBlock = 2 * sme_mr;
    for (int m0 = 0; m0 < mc; m0 += kBlock) {
        const int rows = std::min(kBlock, mc - m0);
        float* const blk = dst + static_cast<int64_t>(m0 / kBlock) * kc * kBlock;
        for (int k0 = 0; k0 < kc; k0 += sme_mr) {
            const int klive = std::min(sme_mr, kc - k0);
            const svbool_t pk = svwhilelt_b32(0, klive);
            for (int r0 = 0; r0 < kBlock; r0 += sme_mr) {
                const int rlive = std::min(sme_mr, rows - r0);
                if (rlive <= 0) {
                    break;
                }
                for (int i = 0; i < rlive; ++i) {
                    svld1_hor_za32(0, static_cast<uint32_t>(i), pk,
                                   src + static_cast<int64_t>(m0 + r0 + i) * lds + k0);
                }
                const svbool_t pr = svwhilelt_b32(0, rlive);
                for (int j = 0; j < klive; ++j) {
                    svst1_ver_za32(0, static_cast<uint32_t>(j), pr,
                                   blk + static_cast<int64_t>(k0 + j) * kBlock + r0);
                }
            }
        }
    }
}

/// Pack @p mc rows of A (row stride @p lds elements) into [m/32][kc][32].
inline void pack_a_f32_sme(int mc, int kc, float* dst, const float* src, int64_t lds,
                           bool trans = false) noexcept
{
    constexpr int kBlock = 2 * sme_mr;
#if defined(NNOPS_ARCH_SME)
    if (!trans && available_sme()) {
        pack_a_f32_sme_za(mc, kc, dst, src, lds);
        return;
    }
#endif
    if (trans) {
        // A's physical layout is K×M: row k, column m. Rows of this layout are
        // strided whatever the loop order, so keep the simple m-major walk.
        for (int m = 0; m < mc; ++m) {
            float* out = dst + (static_cast<int64_t>(m / kBlock) * kc) * kBlock + (m % kBlock);
            for (int k = 0; k < kc; ++k) {
                out[static_cast<int64_t>(k) * kBlock] = src[static_cast<int64_t>(k) * lds + m];
            }
        }
        return;
    }
    // k-major: for one k the block's 32 rows are contiguous, one full cache
    // line per output row. The m-major walk wrote 4 bytes per line instead.
    // The gathers are strided but each source row stays L1-resident for the
    // next k.
    for (int m0 = 0; m0 < mc; m0 += kBlock) {
        const int rows = std::min(kBlock, mc - m0);
        const float* const in0 = src + static_cast<int64_t>(m0) * lds;
        float* const blk = dst + static_cast<int64_t>(m0 / kBlock) * kc * kBlock;
        int k = 0;
#if defined(__ARM_NEON)
        // A full block transposes four k by four rows at a time: four vector
        // loads, one 4x4 transpose, four 16-byte stores that fill a line.
        if (rows == kBlock) {
            for (; k + 4 <= kc; k += 4) {
                float* const out = blk + static_cast<int64_t>(k) * kBlock;
                for (int mq = 0; mq < kBlock; mq += 4) {
                    const float* const p0 = in0 + static_cast<int64_t>(mq) * lds + k;
                    const float* const p1 = p0 + lds;
                    const float* const p2 = p1 + lds;
                    const float* const p3 = p2 + lds;
                    const float32x4_t r0 = vld1q_f32(p0);
                    const float32x4_t r1 = vld1q_f32(p1);
                    const float32x4_t r2 = vld1q_f32(p2);
                    const float32x4_t r3 = vld1q_f32(p3);
                    const float32x4x2_t t01 = vtrnq_f32(r0, r1);
                    const float32x4x2_t t23 = vtrnq_f32(r2, r3);
                    vst1q_f32(out + mq,
                              vcombine_f32(vget_low_f32(t01.val[0]),
                                           vget_low_f32(t23.val[0])));
                    vst1q_f32(out + kBlock + mq,
                              vcombine_f32(vget_low_f32(t01.val[1]),
                                           vget_low_f32(t23.val[1])));
                    vst1q_f32(out + 2 * kBlock + mq,
                              vcombine_f32(vget_high_f32(t01.val[0]),
                                           vget_high_f32(t23.val[0])));
                    vst1q_f32(out + 3 * kBlock + mq,
                              vcombine_f32(vget_high_f32(t01.val[1]),
                                           vget_high_f32(t23.val[1])));
                }
            }
        }
#endif
        for (; k < kc; ++k) {
            float* out = blk + static_cast<int64_t>(k) * kBlock;
            for (int m = 0; m < rows; ++m) {
                out[m] = in0[static_cast<int64_t>(m) * lds + k];
            }
        }
    }
}

/// Pack @p nc columns of B (row-major, stride @p ldb, @p kc rows) into
/// [n/32][kc][32]; padding columns are zeroed.
inline void pack_b_f32_sme(int nc, int kc, float* dst, const float* src, int64_t ldb,
                           bool trans = false) noexcept
{
    const int n_blocks = (nc + sme_fp_nr - 1) / sme_fp_nr;
    for (int nb = 0; nb < n_blocks; ++nb) {
        const int n0 = nb * sme_fp_nr;
        const int cols = std::min(sme_fp_nr, nc - n0);
        float* blk = dst + static_cast<int64_t>(nb) * kc * sme_fp_nr;
        if (trans) {
            // B's physical layout is N×K, so its 32 columns are 32 rows of
            // contiguous K — exactly the A pack's transpose. ZA turns the
            // block around with one load and one store per 16×16 half instead
            // of 32 strided scalar gathers per k.
#if defined(NNOPS_ARCH_SME)
            if (available_sme()) {
                pack_a_f32_sme_za(cols, kc, blk, src + static_cast<int64_t>(n0) * ldb, ldb);
                for (int k = 0; k < kc; ++k) {
                    float* out = blk + static_cast<int64_t>(k) * sme_fp_nr;
                    for (int j = cols; j < sme_fp_nr; ++j) {
                        out[j] = 0.0f;
                    }
                }
                continue;
            }
#endif
            // B's physical layout is N×K: row n, column k.
            for (int k = 0; k < kc; ++k) {
                float* out = blk + static_cast<int64_t>(k) * sme_fp_nr;
                for (int j = 0; j < cols; ++j) {
                    out[j] = src[static_cast<int64_t>(n0 + j) * ldb + k];
                }
                for (int j = cols; j < sme_fp_nr; ++j) {
                    out[j] = 0.0f;
                }
            }
            continue;
        }
        for (int k = 0; k < kc; ++k) {
            float* out = blk + static_cast<int64_t>(k) * sme_fp_nr;
            const float* in = src + static_cast<int64_t>(k) * ldb + n0;
            for (int j = 0; j < cols; ++j) {
                out[j] = in[j];
            }
            for (int j = cols; j < sme_fp_nr; ++j) {
                out[j] = 0.0f;
            }
        }
    }
}

} // namespace nnops::backend::cpu::aarch64

#endif // NNOPS_ARCH_SME
