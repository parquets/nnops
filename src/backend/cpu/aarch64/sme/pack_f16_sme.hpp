#pragma once
/// @file pack_f16_sme.hpp
/// @brief AArch64 SME float16 pack kernels for MMA LHS/RHS.
///
/// Mirrors pack_f32_sme.hpp on half, with the K-pair interleave the hardware
/// two-way dot needs: the MOPA reads lane pairs, so A stores [row][k,k+1] and B
/// [col][k,k+1] already zipped, and the kernel's K loop is a straight load.
/// Building the interleave in the loop instead costs five permutes per K pair
/// that contend with the MOPA issue slot.
///   - A: [m/32][ceil(kc/2)][64], lane 2i and 2i+1 holding row i's K pair.
///   - B: [n/32][ceil(kc/2)][64], column j's K pair at j*2 and j*2+1.
/// An odd kc pads the last K pair with zeros at pack time; padding columns past
/// @p nc are zeroed too.
///
/// `trans` reads the physical K×M (A) / N×K (B) layout, which is what MatMul's
/// transpose_a / transpose_b describe.

#if defined(NNOPS_ARCH_SME)

#include <algorithm>
#include <cstdint>

#include <arm_neon.h>

#include "nnops/detail/half.hpp"

#include "backend/cpu/aarch64/sme/mma_sme.hpp"

namespace nnops::backend::cpu::aarch64 {

/// Elements one packed A slab occupies: [ceil(mc/32)][ceil(kc/2)][64]. The
/// 32-row block is the kernel's 2x2 ZA tile, so its two A vectors are one pair
/// load.
inline int64_t packed_a_elems_f16_sme(int mc, int kc) noexcept
{
    constexpr int kBlock = 2 * sme_mr;
    return static_cast<int64_t>((mc + kBlock - 1) / kBlock) * ((kc + 1) / 2) *
           (2 * kBlock);
}

/// Elements one packed B slab occupies: [ceil(nc/32)][ceil(kc/2)][64].
inline int64_t packed_b_elems_f16_sme(int nc, int kc) noexcept
{
    return static_cast<int64_t>((nc + sme_fp_nr - 1) / sme_fp_nr) * ((kc + 1) / 2) *
           (2 * sme_fp_nr);
}

namespace sme_pack_detail {

/// Two 16-bit columns' pairs as 32-bit words, zipped: the 4x4 16-bit
/// transpose's second half (vzip1/2_s32 on the transposed pairs).
inline int16x4_t vzip1_s32_as16(int16x4_t a, int16x4_t b) noexcept
{
    return vreinterpret_s16_s32(vzip1_s32(vreinterpret_s32_s16(a), vreinterpret_s32_s16(b)));
}

inline int16x4_t vzip2_s32_as16(int16x4_t a, int16x4_t b) noexcept
{
    return vreinterpret_s16_s32(vzip2_s32(vreinterpret_s32_s16(a), vreinterpret_s32_s16(b)));
}

} // namespace sme_pack_detail

/// Pack @p mc rows of A (row stride @p lds elements) into
/// [m/32][ceil(kc/2)][64]; an odd K tail is zeroed.
///
/// One K pair of four rows is 16 bytes, so a 4x4 transpose (4 rows x 4 K)
/// plus two zips fills two of them per store pair. The scalar walk wrote two
/// halfs at a time and ran at a third of this.
#if defined(NNOPS_ARCH_SME)
/// A 16-bit row pair is one 32-bit word, so the K-pair layout is the same
/// 32-bit transpose as f32 — through ZA, as pack_a_f32_sme_za does.
__arm_locally_streaming __arm_new("za") __attribute__((target("+sme2")))
inline void pack_a_f16_sme_za(int mc, int kc, half* dst, const half* src,
                              int64_t lds) noexcept
{
    constexpr int kBlock = 2 * sme_mr;
    const int k_pairs = kc / 2;
    const auto* const sw = reinterpret_cast<const uint32_t*>(src);
    const int64_t lds_w = lds / 2;
    for (int m0 = 0; m0 < mc; m0 += kBlock) {
        const int rows = std::min(kBlock, mc - m0);
        auto* const blk = reinterpret_cast<uint32_t*>(
            dst + static_cast<int64_t>(m0 / kBlock) * k_pairs * (2 * kBlock));
        for (int k0 = 0; k0 < k_pairs; k0 += sme_mr) {
            const int klive = std::min(sme_mr, k_pairs - k0);
            const svbool_t pk = svwhilelt_b32(0, klive);
            for (int r0 = 0; r0 < kBlock; r0 += sme_mr) {
                const int rlive = std::min(sme_mr, rows - r0);
                if (rlive <= 0) {
                    break;
                }
                for (int i = 0; i < rlive; ++i) {
                    svld1_hor_za32(0, static_cast<uint32_t>(i), pk,
                                   sw + static_cast<int64_t>(m0 + r0 + i) * lds_w + k0);
                }
                const svbool_t pr = svwhilelt_b32(0, rlive);
                // 64 halfs, i.e. 32 words, per K pair.
                for (int j = 0; j < klive; ++j) {
                    svst1_ver_za32(0, static_cast<uint32_t>(j), pr,
                                   blk + static_cast<int64_t>(k0 + j) * kBlock + r0);
                }
            }
        }
    }
}
#endif

inline void pack_a_f16_sme(int mc, int kc, half* dst, const half* src, int64_t lds,
                           bool trans = false) noexcept
{
    constexpr int kBlock = 2 * sme_mr;
    const int k_pairs = (kc + 1) / 2;
    const int full_k = kc & ~1; // K covered by whole pairs

#if defined(NNOPS_ARCH_SME)
    // The word view needs even K and an even stride; an odd one falls through
    // to the scalar pair fill below.
    if (!trans && available_sme() && (kc & 1) == 0 && (lds & 1) == 0 &&
        (reinterpret_cast<uintptr_t>(src) & 3) == 0) {
        pack_a_f16_sme_za(mc, kc, dst, src, lds);
        return;
    }
#endif
    if (trans) {
        for (int m = 0; m < mc; ++m) {
            half* out = dst + static_cast<int64_t>(m / kBlock) * k_pairs * (2 * kBlock) +
                        (m % kBlock) * 2;
            for (int kp = 0; kp < k_pairs; ++kp) {
                const int k = 2 * kp;
                half* lane = out + static_cast<int64_t>(kp) * 2 * kBlock;
                lane[0] = src[static_cast<int64_t>(k) * lds + m];
                lane[1] = k + 1 < kc ?
                              src[static_cast<int64_t>(k + 1) * lds + m] :
                              half{};
            }
        }
        return;
    }

    for (int m0 = 0; m0 < mc; m0 += kBlock) {
        const int rows = std::min(kBlock, mc - m0);
        half* const blk = dst + static_cast<int64_t>(m0 / kBlock) * k_pairs * (2 * kBlock);
        const half* const in0 = src + static_cast<int64_t>(m0) * lds;
        int mq = 0;
        for (; mq + 4 <= rows; mq += 4) {
            const half* const p0 = in0 + static_cast<int64_t>(mq) * lds;
            const half* const p1 = p0 + lds;
            const half* const p2 = p1 + lds;
            const half* const p3 = p2 + lds;
            int kp = 0;
            for (; 2 * kp + 4 <= full_k; kp += 2) {
                const half* const q = p0 + 2 * kp;
                const int16x4_t r0 = vld1_s16(reinterpret_cast<const int16_t*>(q));
                const int16x4_t r1 = vld1_s16(reinterpret_cast<const int16_t*>(q + lds));
                const int16x4_t r2 = vld1_s16(reinterpret_cast<const int16_t*>(q + 2 * lds));
                const int16x4_t r3 = vld1_s16(reinterpret_cast<const int16_t*>(q + 3 * lds));
                const int16x4x2_t t01 = vtrn_s16(r0, r1);
                const int16x4x2_t t23 = vtrn_s16(r2, r3);
                // The four K columns, each four rows: a 4x4 transpose in
                // 16-bit lanes, then vst2 interleaves two K columns per row.
                const int16x4_t c0 = sme_pack_detail::vzip1_s32_as16(t01.val[0], t23.val[0]);
                const int16x4_t c2 = sme_pack_detail::vzip2_s32_as16(t01.val[0], t23.val[0]);
                const int16x4_t c1 = sme_pack_detail::vzip1_s32_as16(t01.val[1], t23.val[1]);
                const int16x4_t c3 = sme_pack_detail::vzip2_s32_as16(t01.val[1], t23.val[1]);
                half* const o0 = blk + static_cast<int64_t>(kp) * 2 * kBlock + mq * 2;
                const int16x4x2_t p0v = {c0, c1};
                const int16x4x2_t p1v = {c2, c3};
                vst2_s16(reinterpret_cast<int16_t*>(o0), p0v);
                vst2_s16(reinterpret_cast<int16_t*>(o0 + 2 * kBlock), p1v);
            }
            for (; kp < k_pairs; ++kp) {
                const int k = 2 * kp;
                half* const lane = blk + static_cast<int64_t>(kp) * 2 * kBlock + mq * 2;
                for (int i = 0; i < 4; ++i) {
                    lane[2 * i] = in0[static_cast<int64_t>(mq + i) * lds + k];
                    lane[2 * i + 1] = k + 1 < kc ?
                                          in0[static_cast<int64_t>(mq + i) * lds + k + 1] :
                                          half{};
                }
            }
        }
        for (; mq < rows; ++mq) {
            half* out = blk + mq * 2;
            const half* in = in0 + static_cast<int64_t>(mq) * lds;
            for (int kp = 0; kp < k_pairs; ++kp) {
                const int k = 2 * kp;
                half* lane = out + static_cast<int64_t>(kp) * 2 * kBlock;
                lane[0] = in[k];
                lane[1] = k + 1 < kc ? in[k + 1] : half{};
            }
        }
    }
}

/// Pack @p nc columns of B (row-major, stride @p ldb, @p kc rows) into
/// [n/32][ceil(kc/2)][64]; an odd K tail and padding columns are zeroed.
inline void pack_b_f16_sme(int nc, int kc, half* dst, const half* src, int64_t ldb,
                           bool trans = false) noexcept
{
    const int n_blocks = (nc + sme_fp_nr - 1) / sme_fp_nr;
    const int k_pairs = (kc + 1) / 2;
    for (int nb = 0; nb < n_blocks; ++nb) {
        const int n0 = nb * sme_fp_nr;
        const int cols = std::min(sme_fp_nr, nc - n0);
        half* blk = dst + static_cast<int64_t>(nb) * k_pairs * (2 * sme_fp_nr);
        for (int kp = 0; kp < k_pairs; ++kp) {
            const int k = 2 * kp;
            half* out = blk + static_cast<int64_t>(kp) * 2 * sme_fp_nr;
            if (trans) {
                for (int j = 0; j < cols; ++j) {
                    out[2 * j] = src[static_cast<int64_t>(n0 + j) * ldb + k];
                    out[2 * j + 1] =
                        k + 1 < kc ? src[static_cast<int64_t>(n0 + j) * ldb + k + 1] : half{};
                }
            } else {
                // Rows k and k+1 are contiguous, so the pair interleave is one
                // zip of 8 columns at a time. This pack is on the critical path
                // of a prefill (small M), where B's traffic is not amortised.
                const half* in0 = src + static_cast<int64_t>(k) * ldb + n0;
                const half* in1 = k + 1 < kc ? in0 + ldb : nullptr;
                int j = 0;
                for (; j + 8 <= cols; j += 8) {
                    const float16x8_t a =
                        vld1q_f16(reinterpret_cast<const float16_t*>(in0 + j));
                    const float16x8_t b =
                        in1 ? vld1q_f16(reinterpret_cast<const float16_t*>(in1 + j)) :
                              vdupq_n_f16(0.0f);
                    vst1q_f16(reinterpret_cast<float16_t*>(out + 2 * j), vzip1q_f16(a, b));
                    vst1q_f16(reinterpret_cast<float16_t*>(out + 2 * j + 8), vzip2q_f16(a, b));
                }
                for (; j < cols; ++j) {
                    out[2 * j] = in0[j];
                    out[2 * j + 1] = in1 ? in1[j] : half{};
                }
            }
            for (int j = cols; j < sme_fp_nr; ++j) {
                out[2 * j] = half{};
                out[2 * j + 1] = half{};
            }
        }
    }
}

} // namespace nnops::backend::cpu::aarch64

#endif // NNOPS_ARCH_SME
