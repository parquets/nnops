#pragma once
/// @file mma_i8.hpp
/// @brief AArch64 NEON int8 (SDOT) mrxncxkc MMA kernels: s8×s8 → int32.
///
/// One call covers the whole N side of one M panel: it computes
/// C[mr][nc] += A[mr][kc] × B[kc][nc] panel by panel and clamps every int32
/// result to [clamp_min, clamp_max]. This is the int8 sibling of mma_f32.hpp /
/// mma_f16.hpp and takes the same runtime panel height `mr` and the same
/// partial-M convention: rows past `mr` reuse the last live row's index and C
/// row, so a short panel needs no per-row guards.
///
/// A is packed as four-byte k groups, one group per row:
///   [ceil(kc/4)][MR][4] bytes (32 per group at MR=8), which is exactly what
///   the pack_trans_nN_dp4a_i8 kernels write -- a 16-byte load holds four rows'
///   groups, so one vector feeds four vdotq_laneq lanes.
/// The `mma_packed_b_*` kernels instead read A row-major through `lda` (int8
/// elements) and broadcast each row's group with vdupq_n_s32, so they need
/// kc % 4 == 0 -- a row-major row has no zero-filled final group.
///
/// B is always packed, in the layout the dot-product packers already write
/// (pack_trans_n12_dp4a_i8 / pack_copy_n12_dp4a_i8), and uniform across
/// panels: ceil(nc/12) panels of 12 columns each, [ceil(kc/4)][12][4] bytes --
/// 48 bytes per k group. A panel is 12*kbytes bytes and sits in a
/// mma_pack_align(12*kbytes)-byte slot.
/// Twelve is the widest panel an 8-row block can keep in 32 vector registers
/// (24 accumulators + 3 B + A), and it is mma_i8_nr_max on this arch.
///
/// Only a panel's live columns are stored, so the columns past `nc` never
/// reach C -- the packer may zero-fill them or alias the last live column
/// (which is what the transposing packer does when handed a short width).
/// The same goes for the packed-A panel's row slots past `mr`: the kernels
/// take their SDOT lanes straight from the panel, so a short panel's dead
/// slots must repeat the last live row (the pack_trans_* convention that
/// mma_f32.hpp relies on) — they are not masked off in here.
///
/// Arithmetic is signed × signed (vdotq_s32 / vdotq_laneq_s32), which is the
/// contract x86_64/mma_i8.hpp also presents: both take int8 A and int8 B and
/// return the exact s8×s8 product, so the MatMul driver is arch-blind. x86
/// reaches the same answer through u8×s8 plus a bias correction; SDOT has no
/// unsigned restriction, so there is nothing to correct here and
/// mma_i8_b_panel_sums below is a no-op.
///
/// A uint8 A is shifted into signed range by the A pack (XOR 0x80, i.e. a-128)
/// rather than by anything in here: SDOT then accumulates Σ(a-128)·b, and the
/// 128·Σb the shift introduced is removed once in the epilogue, which already
/// has the full-K column sums (`matmul_int8_compensate`'s u8_offset). That is
/// why this arch has no panel tail — see mma_i8_folds_a_bias.
///
/// `kc` is the number of int8 elements. On the packed route a final partial
/// group (1..3) is consumed as a whole zero-padded group, matching the packers.
///
/// Tile sizes (AArch64-optimised): M ∈ {8, 4, 1}    N: 12 plus an in-kernel tail.

#include <array>
#include <algorithm>
#include <arm_neon.h>
#include <cstdint>
#include "backend/cpu/common/common_utils.hpp"

namespace nnops::backend::cpu::aarch64 {

/// One row's k group as an SDOT operand: the four bytes repeated in every
/// 4-byte lane, so a plain vdotq_s32 against a 4-column B vector computes the
/// same 4-k dot for all four columns. Used by the raw-A kernels, whose rows are
/// not adjacent and so cannot share one 16-byte load.
inline int8x16_t mma_i8_bcast_group(const int8_t* p) noexcept {
    return vreinterpretq_s8_s32(vdupq_n_s32(*reinterpret_cast<const int32_t*>(p)));
}

/// Packed-B panel width: the kernels' uniform nr.
inline constexpr int mma_i8_nr_max = 12;

/// Bytes a packed-B panel carries past its nr*kbytes payload. x86 appends the
/// column sums its u8×s8 emulation needs; SDOT needs none, so the panel is
/// exactly its data and the panel slot is align(12*kbytes).
inline constexpr int mma_i8_b_panel_tail = 0;

/// False: the kernels do not fold an A bias, so a shifted A's correction stays
/// with the epilogue (which has the full-K sums). See the header comment.
inline constexpr bool mma_i8_folds_a_bias = false;

/// x86 counterpart writes the panel's column sums here (see the x86 header);
/// SDOT reads B unsigned-bit-exact already, so the packing step has nothing to
/// add. Same signature in both headers so matmul_helper is arch-blind.
inline void mma_i8_b_panel_sums(int8_t* /*panel*/, int /*kbytes*/, bool /*a_biased*/) noexcept {}

// =========================================================================
//  mr=8  kernels
// =========================================================================

template <bool zero_mode = false>
inline void mma_packed_ab_mr8_nr12_mrxncxkc_i8(
    int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
    const int8_t* NNOPS_RESTRICT A,
    const int8_t* NNOPS_RESTRICT B,
    int32_t clamp_min, int32_t clamp_max) noexcept {

    // Rows past `mr` alias the last live row in both A and C: they compute a
    // duplicate of it and store it back to the same place, so a partial-M tile
    // needs no per-row guards.
    int32_t* NNOPS_RESTRICT c0_ptr = C + 0 * ldc;
    int32_t* NNOPS_RESTRICT c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    int32_t* NNOPS_RESTRICT c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    int32_t* NNOPS_RESTRICT c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;
    int32_t* NNOPS_RESTRICT c4_ptr = (mr > 4) ? C + 4 * ldc : c3_ptr;
    int32_t* NNOPS_RESTRICT c5_ptr = (mr > 5) ? C + 5 * ldc : c4_ptr;
    int32_t* NNOPS_RESTRICT c6_ptr = (mr > 6) ? C + 6 * ldc : c5_ptr;
    int32_t* NNOPS_RESTRICT c7_ptr = (mr > 7) ? C + 7 * ldc : c6_ptr;

    const int kbytes = (kc + 3) & ~3;
    const int kg = kbytes >> 2;
    const int ldb = static_cast<int>(mma_pack_align(12 * kbytes));
    const int32x4_t vmin = vdupq_n_s32(clamp_min);
    const int32x4_t vmax = vdupq_n_s32(clamp_max);

    int n = 0;
    for (; n < nc - 11; n += 12) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;
        int32x4_t v_c10 = vdupq_n_s32(0), v_c11 = v_c10, v_c12 = v_c10;
        int32x4_t v_c20 = vdupq_n_s32(0), v_c21 = v_c20, v_c22 = v_c20;
        int32x4_t v_c30 = vdupq_n_s32(0), v_c31 = v_c30, v_c32 = v_c30;
        int32x4_t v_c40 = vdupq_n_s32(0), v_c41 = v_c40, v_c42 = v_c40;
        int32x4_t v_c50 = vdupq_n_s32(0), v_c51 = v_c50, v_c52 = v_c50;
        int32x4_t v_c60 = vdupq_n_s32(0), v_c61 = v_c60, v_c62 = v_c60;
        int32x4_t v_c70 = vdupq_n_s32(0), v_c71 = v_c70, v_c72 = v_c70;

        const int8_t* NNOPS_RESTRICT a_ptr = A;
        const int8_t* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kg; ++k) {
            const int8x16_t v_a0 = vld1q_s8(a_ptr + 0 * 16);  // rows 0..3
            const int8x16_t v_a1 = vld1q_s8(a_ptr + 1 * 16);  // rows 4..7
            a_ptr += 32;

            const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);

            v_c00 = vdotq_laneq_s32(v_c00, v_b0, v_a0, 0);
            v_c10 = vdotq_laneq_s32(v_c10, v_b0, v_a0, 1);
            v_c20 = vdotq_laneq_s32(v_c20, v_b0, v_a0, 2);
            v_c30 = vdotq_laneq_s32(v_c30, v_b0, v_a0, 3);

            v_c40 = vdotq_laneq_s32(v_c40, v_b0, v_a1, 0);
            v_c50 = vdotq_laneq_s32(v_c50, v_b0, v_a1, 1);
            v_c60 = vdotq_laneq_s32(v_c60, v_b0, v_a1, 2);
            v_c70 = vdotq_laneq_s32(v_c70, v_b0, v_a1, 3);

            const int8x16_t v_b1 = vld1q_s8(b_ptr + 1 * 16);

            v_c01 = vdotq_laneq_s32(v_c01, v_b1, v_a0, 0);
            v_c11 = vdotq_laneq_s32(v_c11, v_b1, v_a0, 1);
            v_c21 = vdotq_laneq_s32(v_c21, v_b1, v_a0, 2);
            v_c31 = vdotq_laneq_s32(v_c31, v_b1, v_a0, 3);

            v_c41 = vdotq_laneq_s32(v_c41, v_b1, v_a1, 0);
            v_c51 = vdotq_laneq_s32(v_c51, v_b1, v_a1, 1);
            v_c61 = vdotq_laneq_s32(v_c61, v_b1, v_a1, 2);
            v_c71 = vdotq_laneq_s32(v_c71, v_b1, v_a1, 3);

            const int8x16_t v_b2 = vld1q_s8(b_ptr + 2 * 16);

            v_c02 = vdotq_laneq_s32(v_c02, v_b2, v_a0, 0);
            v_c12 = vdotq_laneq_s32(v_c12, v_b2, v_a0, 1);
            v_c22 = vdotq_laneq_s32(v_c22, v_b2, v_a0, 2);
            v_c32 = vdotq_laneq_s32(v_c32, v_b2, v_a0, 3);

            v_c42 = vdotq_laneq_s32(v_c42, v_b2, v_a1, 0);
            v_c52 = vdotq_laneq_s32(v_c52, v_b2, v_a1, 1);
            v_c62 = vdotq_laneq_s32(v_c62, v_b2, v_a1, 2);
            v_c72 = vdotq_laneq_s32(v_c72, v_b2, v_a1, 3);

            b_ptr += 48;
        }

        if constexpr (!zero_mode) {
            v_c00 = vaddq_s32(v_c00, vld1q_s32(c0_ptr + 0 * 4));
            v_c01 = vaddq_s32(v_c01, vld1q_s32(c0_ptr + 1 * 4));
            v_c02 = vaddq_s32(v_c02, vld1q_s32(c0_ptr + 2 * 4));
            v_c10 = vaddq_s32(v_c10, vld1q_s32(c1_ptr + 0 * 4));
            v_c11 = vaddq_s32(v_c11, vld1q_s32(c1_ptr + 1 * 4));
            v_c12 = vaddq_s32(v_c12, vld1q_s32(c1_ptr + 2 * 4));
            v_c20 = vaddq_s32(v_c20, vld1q_s32(c2_ptr + 0 * 4));
            v_c21 = vaddq_s32(v_c21, vld1q_s32(c2_ptr + 1 * 4));
            v_c22 = vaddq_s32(v_c22, vld1q_s32(c2_ptr + 2 * 4));
            v_c30 = vaddq_s32(v_c30, vld1q_s32(c3_ptr + 0 * 4));
            v_c31 = vaddq_s32(v_c31, vld1q_s32(c3_ptr + 1 * 4));
            v_c32 = vaddq_s32(v_c32, vld1q_s32(c3_ptr + 2 * 4));
            v_c40 = vaddq_s32(v_c40, vld1q_s32(c4_ptr + 0 * 4));
            v_c41 = vaddq_s32(v_c41, vld1q_s32(c4_ptr + 1 * 4));
            v_c42 = vaddq_s32(v_c42, vld1q_s32(c4_ptr + 2 * 4));
            v_c50 = vaddq_s32(v_c50, vld1q_s32(c5_ptr + 0 * 4));
            v_c51 = vaddq_s32(v_c51, vld1q_s32(c5_ptr + 1 * 4));
            v_c52 = vaddq_s32(v_c52, vld1q_s32(c5_ptr + 2 * 4));
            v_c60 = vaddq_s32(v_c60, vld1q_s32(c6_ptr + 0 * 4));
            v_c61 = vaddq_s32(v_c61, vld1q_s32(c6_ptr + 1 * 4));
            v_c62 = vaddq_s32(v_c62, vld1q_s32(c6_ptr + 2 * 4));
            v_c70 = vaddq_s32(v_c70, vld1q_s32(c7_ptr + 0 * 4));
            v_c71 = vaddq_s32(v_c71, vld1q_s32(c7_ptr + 1 * 4));
            v_c72 = vaddq_s32(v_c72, vld1q_s32(c7_ptr + 2 * 4));
        }

        vst1q_s32(c0_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c00, vmin), vmax));
        vst1q_s32(c0_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c01, vmin), vmax));
        vst1q_s32(c0_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c02, vmin), vmax));
        vst1q_s32(c1_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c10, vmin), vmax));
        vst1q_s32(c1_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c11, vmin), vmax));
        vst1q_s32(c1_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c12, vmin), vmax));
        vst1q_s32(c2_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c20, vmin), vmax));
        vst1q_s32(c2_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c21, vmin), vmax));
        vst1q_s32(c2_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c22, vmin), vmax));
        vst1q_s32(c3_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c30, vmin), vmax));
        vst1q_s32(c3_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c31, vmin), vmax));
        vst1q_s32(c3_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c32, vmin), vmax));
        vst1q_s32(c4_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c40, vmin), vmax));
        vst1q_s32(c4_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c41, vmin), vmax));
        vst1q_s32(c4_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c42, vmin), vmax));
        vst1q_s32(c5_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c50, vmin), vmax));
        vst1q_s32(c5_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c51, vmin), vmax));
        vst1q_s32(c5_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c52, vmin), vmax));
        vst1q_s32(c6_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c60, vmin), vmax));
        vst1q_s32(c6_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c61, vmin), vmax));
        vst1q_s32(c6_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c62, vmin), vmax));
        vst1q_s32(c7_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c70, vmin), vmax));
        vst1q_s32(c7_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c71, vmin), vmax));
        vst1q_s32(c7_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c72, vmin), vmax));

        c0_ptr += 12;
        c1_ptr += 12;
        c2_ptr += 12;
        c3_ptr += 12;
        c4_ptr += 12;
        c5_ptr += 12;
        c6_ptr += 12;
        c7_ptr += 12;

        B += ldb;
    }

    const int rem = nc - n;
    if (rem > 0) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;
        int32x4_t v_c10 = vdupq_n_s32(0), v_c11 = v_c10, v_c12 = v_c10;
        int32x4_t v_c20 = vdupq_n_s32(0), v_c21 = v_c20, v_c22 = v_c20;
        int32x4_t v_c30 = vdupq_n_s32(0), v_c31 = v_c30, v_c32 = v_c30;
        int32x4_t v_c40 = vdupq_n_s32(0), v_c41 = v_c40, v_c42 = v_c40;
        int32x4_t v_c50 = vdupq_n_s32(0), v_c51 = v_c50, v_c52 = v_c50;
        int32x4_t v_c60 = vdupq_n_s32(0), v_c61 = v_c60, v_c62 = v_c60;
        int32x4_t v_c70 = vdupq_n_s32(0), v_c71 = v_c70, v_c72 = v_c70;

        const int8_t* NNOPS_RESTRICT a_ptr = A;
        const int8_t* NNOPS_RESTRICT b_ptr = B;
        if (rem > 4) {
            for (int k = 0; k < kg; ++k) {
                const int8x16_t v_a0 = vld1q_s8(a_ptr + 0 * 16);
                const int8x16_t v_a1 = vld1q_s8(a_ptr + 1 * 16);
                a_ptr += 32;

                const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);

                v_c00 = vdotq_laneq_s32(v_c00, v_b0, v_a0, 0);
                v_c10 = vdotq_laneq_s32(v_c10, v_b0, v_a0, 1);
                v_c20 = vdotq_laneq_s32(v_c20, v_b0, v_a0, 2);
                v_c30 = vdotq_laneq_s32(v_c30, v_b0, v_a0, 3);
                v_c40 = vdotq_laneq_s32(v_c40, v_b0, v_a1, 0);
                v_c50 = vdotq_laneq_s32(v_c50, v_b0, v_a1, 1);
                v_c60 = vdotq_laneq_s32(v_c60, v_b0, v_a1, 2);
                v_c70 = vdotq_laneq_s32(v_c70, v_b0, v_a1, 3);

                const int8x16_t v_b1 = vld1q_s8(b_ptr + 1 * 16);

                v_c01 = vdotq_laneq_s32(v_c01, v_b1, v_a0, 0);
                v_c11 = vdotq_laneq_s32(v_c11, v_b1, v_a0, 1);
                v_c21 = vdotq_laneq_s32(v_c21, v_b1, v_a0, 2);
                v_c31 = vdotq_laneq_s32(v_c31, v_b1, v_a0, 3);
                v_c41 = vdotq_laneq_s32(v_c41, v_b1, v_a1, 0);
                v_c51 = vdotq_laneq_s32(v_c51, v_b1, v_a1, 1);
                v_c61 = vdotq_laneq_s32(v_c61, v_b1, v_a1, 2);
                v_c71 = vdotq_laneq_s32(v_c71, v_b1, v_a1, 3);

                const int8x16_t v_b2 = vld1q_s8(b_ptr + 2 * 16);

                v_c02 = vdotq_laneq_s32(v_c02, v_b2, v_a0, 0);
                v_c12 = vdotq_laneq_s32(v_c12, v_b2, v_a0, 1);
                v_c22 = vdotq_laneq_s32(v_c22, v_b2, v_a0, 2);
                v_c32 = vdotq_laneq_s32(v_c32, v_b2, v_a0, 3);
                v_c42 = vdotq_laneq_s32(v_c42, v_b2, v_a1, 0);
                v_c52 = vdotq_laneq_s32(v_c52, v_b2, v_a1, 1);
                v_c62 = vdotq_laneq_s32(v_c62, v_b2, v_a1, 2);
                v_c72 = vdotq_laneq_s32(v_c72, v_b2, v_a1, 3);

                b_ptr += 48;
            }
        } else {
            for (int k = 0; k < kg; ++k) {
                const int8x16_t v_a0 = vld1q_s8(a_ptr + 0 * 16);
                const int8x16_t v_a1 = vld1q_s8(a_ptr + 1 * 16);
                a_ptr += 32;

                const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);

                v_c00 = vdotq_laneq_s32(v_c00, v_b0, v_a0, 0);
                v_c10 = vdotq_laneq_s32(v_c10, v_b0, v_a0, 1);
                v_c20 = vdotq_laneq_s32(v_c20, v_b0, v_a0, 2);
                v_c30 = vdotq_laneq_s32(v_c30, v_b0, v_a0, 3);
                v_c40 = vdotq_laneq_s32(v_c40, v_b0, v_a1, 0);
                v_c50 = vdotq_laneq_s32(v_c50, v_b0, v_a1, 1);
                v_c60 = vdotq_laneq_s32(v_c60, v_b0, v_a1, 2);
                v_c70 = vdotq_laneq_s32(v_c70, v_b0, v_a1, 3);

                b_ptr += 48;
            }
        }

        const int32x4_t acc[8][3] = {
            {v_c00, v_c01, v_c02}, {v_c10, v_c11, v_c12},
            {v_c20, v_c21, v_c22}, {v_c30, v_c31, v_c32},
            {v_c40, v_c41, v_c42}, {v_c50, v_c51, v_c52},
            {v_c60, v_c61, v_c62}, {v_c70, v_c71, v_c72},
        };
        int32_t* NNOPS_RESTRICT c_rows[8] = {c0_ptr, c1_ptr, c2_ptr, c3_ptr,
                                             c4_ptr, c5_ptr, c6_ptr, c7_ptr};
        // Only the live rows are stored; an alias row past `mr` would write a
        // duplicate of the last live row back over itself.
        for (int r = 0; r < mr; ++r) {
            int32_t* NNOPS_RESTRICT cp = c_rows[r];
            int rr = rem;
            int g = 0;
            for (; rr >= 4; rr -= 4, ++g) {
                int32x4_t v = acc[r][g];
                if constexpr (!zero_mode) {
                    v = vaddq_s32(v, vld1q_s32(cp));
                }
                vst1q_s32(cp, vminq_s32(vmaxq_s32(v, vmin), vmax));
                cp += 4;
            }
            if (rr > 0) {
                int32_t lane[4];
                vst1q_s32(lane, acc[r][g]);
                for (int i = 0; i < rr; ++i) {
                    int32_t v = lane[i];
                    if constexpr (!zero_mode) {
                        v += cp[i];
                    }
                    cp[i] = std::min(std::max(v, clamp_min), clamp_max);
                }
            }
        }
    }
}

// =========================================================================
//  mr=4  kernels
// =========================================================================

template <bool zero_mode = false>
inline void mma_packed_ab_mr4_nr12_mrxncxkc_i8(
    int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
    const int8_t* NNOPS_RESTRICT A,
    const int8_t* NNOPS_RESTRICT B,
    int32_t clamp_min, int32_t clamp_max) noexcept {

    int32_t* NNOPS_RESTRICT c0_ptr = C + 0 * ldc;
    int32_t* NNOPS_RESTRICT c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    int32_t* NNOPS_RESTRICT c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    int32_t* NNOPS_RESTRICT c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;

    const int kbytes = (kc + 3) & ~3;
    const int kg = kbytes >> 2;
    const int ldb = static_cast<int>(mma_pack_align(12 * kbytes));
    const int32x4_t vmin = vdupq_n_s32(clamp_min);
    const int32x4_t vmax = vdupq_n_s32(clamp_max);

    int n = 0;
    for (; n < nc - 11; n += 12) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;
        int32x4_t v_c10 = vdupq_n_s32(0), v_c11 = v_c10, v_c12 = v_c10;
        int32x4_t v_c20 = vdupq_n_s32(0), v_c21 = v_c20, v_c22 = v_c20;
        int32x4_t v_c30 = vdupq_n_s32(0), v_c31 = v_c30, v_c32 = v_c30;

        const int8_t* NNOPS_RESTRICT a_ptr = A;
        const int8_t* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kg; ++k) {
            const int8x16_t v_a = vld1q_s8(a_ptr);  // rows 0..3
            a_ptr += 16;

            const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);

            v_c00 = vdotq_laneq_s32(v_c00, v_b0, v_a, 0);
            v_c10 = vdotq_laneq_s32(v_c10, v_b0, v_a, 1);
            v_c20 = vdotq_laneq_s32(v_c20, v_b0, v_a, 2);
            v_c30 = vdotq_laneq_s32(v_c30, v_b0, v_a, 3);

            const int8x16_t v_b1 = vld1q_s8(b_ptr + 1 * 16);

            v_c01 = vdotq_laneq_s32(v_c01, v_b1, v_a, 0);
            v_c11 = vdotq_laneq_s32(v_c11, v_b1, v_a, 1);
            v_c21 = vdotq_laneq_s32(v_c21, v_b1, v_a, 2);
            v_c31 = vdotq_laneq_s32(v_c31, v_b1, v_a, 3);

            const int8x16_t v_b2 = vld1q_s8(b_ptr + 2 * 16);

            v_c02 = vdotq_laneq_s32(v_c02, v_b2, v_a, 0);
            v_c12 = vdotq_laneq_s32(v_c12, v_b2, v_a, 1);
            v_c22 = vdotq_laneq_s32(v_c22, v_b2, v_a, 2);
            v_c32 = vdotq_laneq_s32(v_c32, v_b2, v_a, 3);

            b_ptr += 48;
        }

        if constexpr (!zero_mode) {
            v_c00 = vaddq_s32(v_c00, vld1q_s32(c0_ptr + 0 * 4));
            v_c01 = vaddq_s32(v_c01, vld1q_s32(c0_ptr + 1 * 4));
            v_c02 = vaddq_s32(v_c02, vld1q_s32(c0_ptr + 2 * 4));
            v_c10 = vaddq_s32(v_c10, vld1q_s32(c1_ptr + 0 * 4));
            v_c11 = vaddq_s32(v_c11, vld1q_s32(c1_ptr + 1 * 4));
            v_c12 = vaddq_s32(v_c12, vld1q_s32(c1_ptr + 2 * 4));
            v_c20 = vaddq_s32(v_c20, vld1q_s32(c2_ptr + 0 * 4));
            v_c21 = vaddq_s32(v_c21, vld1q_s32(c2_ptr + 1 * 4));
            v_c22 = vaddq_s32(v_c22, vld1q_s32(c2_ptr + 2 * 4));
            v_c30 = vaddq_s32(v_c30, vld1q_s32(c3_ptr + 0 * 4));
            v_c31 = vaddq_s32(v_c31, vld1q_s32(c3_ptr + 1 * 4));
            v_c32 = vaddq_s32(v_c32, vld1q_s32(c3_ptr + 2 * 4));
        }

        vst1q_s32(c0_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c00, vmin), vmax));
        vst1q_s32(c0_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c01, vmin), vmax));
        vst1q_s32(c0_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c02, vmin), vmax));
        vst1q_s32(c1_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c10, vmin), vmax));
        vst1q_s32(c1_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c11, vmin), vmax));
        vst1q_s32(c1_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c12, vmin), vmax));
        vst1q_s32(c2_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c20, vmin), vmax));
        vst1q_s32(c2_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c21, vmin), vmax));
        vst1q_s32(c2_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c22, vmin), vmax));
        vst1q_s32(c3_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c30, vmin), vmax));
        vst1q_s32(c3_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c31, vmin), vmax));
        vst1q_s32(c3_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c32, vmin), vmax));

        c0_ptr += 12;
        c1_ptr += 12;
        c2_ptr += 12;
        c3_ptr += 12;

        B += ldb;
    }

    const int rem = nc - n;
    if (rem > 0) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;
        int32x4_t v_c10 = vdupq_n_s32(0), v_c11 = v_c10, v_c12 = v_c10;
        int32x4_t v_c20 = vdupq_n_s32(0), v_c21 = v_c20, v_c22 = v_c20;
        int32x4_t v_c30 = vdupq_n_s32(0), v_c31 = v_c30, v_c32 = v_c30;

        const int8_t* NNOPS_RESTRICT a_ptr = A;
        const int8_t* NNOPS_RESTRICT b_ptr = B;
        if (rem > 4) {
            for (int k = 0; k < kg; ++k) {
                const int8x16_t v_a = vld1q_s8(a_ptr);
                a_ptr += 16;

                const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);

                v_c00 = vdotq_laneq_s32(v_c00, v_b0, v_a, 0);
                v_c10 = vdotq_laneq_s32(v_c10, v_b0, v_a, 1);
                v_c20 = vdotq_laneq_s32(v_c20, v_b0, v_a, 2);
                v_c30 = vdotq_laneq_s32(v_c30, v_b0, v_a, 3);

                const int8x16_t v_b1 = vld1q_s8(b_ptr + 1 * 16);

                v_c01 = vdotq_laneq_s32(v_c01, v_b1, v_a, 0);
                v_c11 = vdotq_laneq_s32(v_c11, v_b1, v_a, 1);
                v_c21 = vdotq_laneq_s32(v_c21, v_b1, v_a, 2);
                v_c31 = vdotq_laneq_s32(v_c31, v_b1, v_a, 3);

                const int8x16_t v_b2 = vld1q_s8(b_ptr + 2 * 16);

                v_c02 = vdotq_laneq_s32(v_c02, v_b2, v_a, 0);
                v_c12 = vdotq_laneq_s32(v_c12, v_b2, v_a, 1);
                v_c22 = vdotq_laneq_s32(v_c22, v_b2, v_a, 2);
                v_c32 = vdotq_laneq_s32(v_c32, v_b2, v_a, 3);

                b_ptr += 48;
            }
        } else {
            for (int k = 0; k < kg; ++k) {
                const int8x16_t v_a = vld1q_s8(a_ptr);
                a_ptr += 16;

                const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);

                v_c00 = vdotq_laneq_s32(v_c00, v_b0, v_a, 0);
                v_c10 = vdotq_laneq_s32(v_c10, v_b0, v_a, 1);
                v_c20 = vdotq_laneq_s32(v_c20, v_b0, v_a, 2);
                v_c30 = vdotq_laneq_s32(v_c30, v_b0, v_a, 3);

                b_ptr += 48;
            }
        }

        const int32x4_t acc[4][3] = {
            {v_c00, v_c01, v_c02}, {v_c10, v_c11, v_c12},
            {v_c20, v_c21, v_c22}, {v_c30, v_c31, v_c32},
        };
        int32_t* NNOPS_RESTRICT c_rows[4] = {c0_ptr, c1_ptr, c2_ptr, c3_ptr};
        for (int r = 0; r < mr; ++r) {
            int32_t* NNOPS_RESTRICT cp = c_rows[r];
            int rr = rem;
            int g = 0;
            for (; rr >= 4; rr -= 4, ++g) {
                int32x4_t v = acc[r][g];
                if constexpr (!zero_mode) {
                    v = vaddq_s32(v, vld1q_s32(cp));
                }
                vst1q_s32(cp, vminq_s32(vmaxq_s32(v, vmin), vmax));
                cp += 4;
            }
            if (rr > 0) {
                int32_t lane[4];
                vst1q_s32(lane, acc[r][g]);
                for (int i = 0; i < rr; ++i) {
                    int32_t v = lane[i];
                    if constexpr (!zero_mode) {
                        v += cp[i];
                    }
                    cp[i] = std::min(std::max(v, clamp_min), clamp_max);
                }
            }
        }
    }
}

// =========================================================================
//  mr=1  kernels
// =========================================================================

template <bool zero_mode = false>
inline void mma_packed_ab_mr1_nr12_mrxncxkc_i8(
    int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
    const int8_t* NNOPS_RESTRICT A,
    const int8_t* NNOPS_RESTRICT B,
    int32_t clamp_min, int32_t clamp_max) noexcept {
    (void)mr;   // one row, so the row count never varies
    (void)ldc;  // one row, so the C row stride never applies

    int32_t* NNOPS_RESTRICT c0_ptr = C;

    const int kbytes = (kc + 3) & ~3;
    const int kg = kbytes >> 2;
    const int ldb = static_cast<int>(mma_pack_align(12 * kbytes));
    const int32x4_t vmin = vdupq_n_s32(clamp_min);
    const int32x4_t vmax = vdupq_n_s32(clamp_max);

    int n = 0;
    for (; n < nc - 11; n += 12) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;

        const int8_t* NNOPS_RESTRICT a_ptr = A;
        const int8_t* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kg; ++k) {
            const int8x16_t v_a = mma_i8_bcast_group(a_ptr);
            a_ptr += 4;

            const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);
            const int8x16_t v_b1 = vld1q_s8(b_ptr + 1 * 16);
            const int8x16_t v_b2 = vld1q_s8(b_ptr + 2 * 16);

            v_c00 = vdotq_s32(v_c00, v_b0, v_a);
            v_c01 = vdotq_s32(v_c01, v_b1, v_a);
            v_c02 = vdotq_s32(v_c02, v_b2, v_a);

            b_ptr += 48;
        }

        if constexpr (!zero_mode) {
            v_c00 = vaddq_s32(v_c00, vld1q_s32(c0_ptr + 0 * 4));
            v_c01 = vaddq_s32(v_c01, vld1q_s32(c0_ptr + 1 * 4));
            v_c02 = vaddq_s32(v_c02, vld1q_s32(c0_ptr + 2 * 4));
        }

        vst1q_s32(c0_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c00, vmin), vmax));
        vst1q_s32(c0_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c01, vmin), vmax));
        vst1q_s32(c0_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c02, vmin), vmax));

        c0_ptr += 12;
        B += ldb;
    }

    const int rem = nc - n;
    if (rem > 0) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;

        const int8_t* NNOPS_RESTRICT a_ptr = A;
        const int8_t* NNOPS_RESTRICT b_ptr = B;
        if (rem > 4) {
            for (int k = 0; k < kg; ++k) {
                const int8x16_t v_a = mma_i8_bcast_group(a_ptr);
                a_ptr += 4;

                const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);
                const int8x16_t v_b1 = vld1q_s8(b_ptr + 1 * 16);
                const int8x16_t v_b2 = vld1q_s8(b_ptr + 2 * 16);

                v_c00 = vdotq_s32(v_c00, v_b0, v_a);
                v_c01 = vdotq_s32(v_c01, v_b1, v_a);
                v_c02 = vdotq_s32(v_c02, v_b2, v_a);

                b_ptr += 48;
            }
        } else {
            for (int k = 0; k < kg; ++k) {
                const int8x16_t v_a = mma_i8_bcast_group(a_ptr);
                a_ptr += 4;

                const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);

                v_c00 = vdotq_s32(v_c00, v_b0, v_a);

                b_ptr += 48;
            }
        }

        const int32x4_t acc[3] = {v_c00, v_c01, v_c02};
        int32_t* NNOPS_RESTRICT cp = c0_ptr;
        int rr = rem;
        int g = 0;
        for (; rr >= 4; rr -= 4, ++g) {
            int32x4_t v = acc[g];
            if constexpr (!zero_mode) {
                v = vaddq_s32(v, vld1q_s32(cp));
            }
            vst1q_s32(cp, vminq_s32(vmaxq_s32(v, vmin), vmax));
            cp += 4;
        }
        if (rr > 0) {
            int32_t lane[4];
            vst1q_s32(lane, acc[g]);
            for (int i = 0; i < rr; ++i) {
                int32_t v = lane[i];
                if constexpr (!zero_mode) {
                    v += cp[i];
                }
                cp[i] = std::min(std::max(v, clamp_min), clamp_max);
            }
        }
    }
}

// =========================================================================
//  mr=8/4/1  kernels, raw A — only B packed
// =========================================================================
//
// `lda` is the A row stride in int8 elements. A row's four k bytes are read
// straight from memory and broadcast, so kc must be a multiple of 4: a
// row-major row has no zero-filled final group.

template <bool zero_mode = false>
inline void mma_packed_b_mr8_nr12_mrxncxkc_i8(
    int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
    const int8_t* NNOPS_RESTRICT A, int lda,
    const int8_t* NNOPS_RESTRICT B,
    int32_t clamp_min, int32_t clamp_max) noexcept {

    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;
    const int r4 = (mr > 4) ? 4 : r3;
    const int r5 = (mr > 5) ? 5 : r4;
    const int r6 = (mr > 6) ? 6 : r5;
    const int r7 = (mr > 7) ? 7 : r6;

    int32_t* NNOPS_RESTRICT c0_ptr = C + 0 * ldc;
    int32_t* NNOPS_RESTRICT c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    int32_t* NNOPS_RESTRICT c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    int32_t* NNOPS_RESTRICT c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;
    int32_t* NNOPS_RESTRICT c4_ptr = (mr > 4) ? C + 4 * ldc : c3_ptr;
    int32_t* NNOPS_RESTRICT c5_ptr = (mr > 5) ? C + 5 * ldc : c4_ptr;
    int32_t* NNOPS_RESTRICT c6_ptr = (mr > 6) ? C + 6 * ldc : c5_ptr;
    int32_t* NNOPS_RESTRICT c7_ptr = (mr > 7) ? C + 7 * ldc : c6_ptr;

    const int kg = kc >> 2;
    const int ldb = static_cast<int>(mma_pack_align(12 * kc));
    const int32x4_t vmin = vdupq_n_s32(clamp_min);
    const int32x4_t vmax = vdupq_n_s32(clamp_max);

    int n = 0;
    for (; n < nc - 11; n += 12) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;
        int32x4_t v_c10 = vdupq_n_s32(0), v_c11 = v_c10, v_c12 = v_c10;
        int32x4_t v_c20 = vdupq_n_s32(0), v_c21 = v_c20, v_c22 = v_c20;
        int32x4_t v_c30 = vdupq_n_s32(0), v_c31 = v_c30, v_c32 = v_c30;
        int32x4_t v_c40 = vdupq_n_s32(0), v_c41 = v_c40, v_c42 = v_c40;
        int32x4_t v_c50 = vdupq_n_s32(0), v_c51 = v_c50, v_c52 = v_c50;
        int32x4_t v_c60 = vdupq_n_s32(0), v_c61 = v_c60, v_c62 = v_c60;
        int32x4_t v_c70 = vdupq_n_s32(0), v_c71 = v_c70, v_c72 = v_c70;

        const int8_t* NNOPS_RESTRICT a0 = A + 0 * lda;
        const int8_t* NNOPS_RESTRICT a1 = A + r1 * lda;
        const int8_t* NNOPS_RESTRICT a2 = A + r2 * lda;
        const int8_t* NNOPS_RESTRICT a3 = A + r3 * lda;
        const int8_t* NNOPS_RESTRICT a4 = A + r4 * lda;
        const int8_t* NNOPS_RESTRICT a5 = A + r5 * lda;
        const int8_t* NNOPS_RESTRICT a6 = A + r6 * lda;
        const int8_t* NNOPS_RESTRICT a7 = A + r7 * lda;
        const int8_t* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kg; ++k) {
            // Three B vectors and one A broadcast stay live: 28 registers for
            // the 24 accumulators, so nothing spills.
            const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);
            const int8x16_t v_b1 = vld1q_s8(b_ptr + 1 * 16);
            const int8x16_t v_b2 = vld1q_s8(b_ptr + 2 * 16);
            b_ptr += 48;

            int8x16_t v_a = mma_i8_bcast_group(a0);
            v_c00 = vdotq_s32(v_c00, v_b0, v_a);
            v_c01 = vdotq_s32(v_c01, v_b1, v_a);
            v_c02 = vdotq_s32(v_c02, v_b2, v_a);
            a0 += 4;

            v_a = mma_i8_bcast_group(a1);
            v_c10 = vdotq_s32(v_c10, v_b0, v_a);
            v_c11 = vdotq_s32(v_c11, v_b1, v_a);
            v_c12 = vdotq_s32(v_c12, v_b2, v_a);
            a1 += 4;

            v_a = mma_i8_bcast_group(a2);
            v_c20 = vdotq_s32(v_c20, v_b0, v_a);
            v_c21 = vdotq_s32(v_c21, v_b1, v_a);
            v_c22 = vdotq_s32(v_c22, v_b2, v_a);
            a2 += 4;

            v_a = mma_i8_bcast_group(a3);
            v_c30 = vdotq_s32(v_c30, v_b0, v_a);
            v_c31 = vdotq_s32(v_c31, v_b1, v_a);
            v_c32 = vdotq_s32(v_c32, v_b2, v_a);
            a3 += 4;

            v_a = mma_i8_bcast_group(a4);
            v_c40 = vdotq_s32(v_c40, v_b0, v_a);
            v_c41 = vdotq_s32(v_c41, v_b1, v_a);
            v_c42 = vdotq_s32(v_c42, v_b2, v_a);
            a4 += 4;

            v_a = mma_i8_bcast_group(a5);
            v_c50 = vdotq_s32(v_c50, v_b0, v_a);
            v_c51 = vdotq_s32(v_c51, v_b1, v_a);
            v_c52 = vdotq_s32(v_c52, v_b2, v_a);
            a5 += 4;

            v_a = mma_i8_bcast_group(a6);
            v_c60 = vdotq_s32(v_c60, v_b0, v_a);
            v_c61 = vdotq_s32(v_c61, v_b1, v_a);
            v_c62 = vdotq_s32(v_c62, v_b2, v_a);
            a6 += 4;

            v_a = mma_i8_bcast_group(a7);
            v_c70 = vdotq_s32(v_c70, v_b0, v_a);
            v_c71 = vdotq_s32(v_c71, v_b1, v_a);
            v_c72 = vdotq_s32(v_c72, v_b2, v_a);
            a7 += 4;
        }

        if constexpr (!zero_mode) {
            v_c00 = vaddq_s32(v_c00, vld1q_s32(c0_ptr + 0 * 4));
            v_c01 = vaddq_s32(v_c01, vld1q_s32(c0_ptr + 1 * 4));
            v_c02 = vaddq_s32(v_c02, vld1q_s32(c0_ptr + 2 * 4));
            v_c10 = vaddq_s32(v_c10, vld1q_s32(c1_ptr + 0 * 4));
            v_c11 = vaddq_s32(v_c11, vld1q_s32(c1_ptr + 1 * 4));
            v_c12 = vaddq_s32(v_c12, vld1q_s32(c1_ptr + 2 * 4));
            v_c20 = vaddq_s32(v_c20, vld1q_s32(c2_ptr + 0 * 4));
            v_c21 = vaddq_s32(v_c21, vld1q_s32(c2_ptr + 1 * 4));
            v_c22 = vaddq_s32(v_c22, vld1q_s32(c2_ptr + 2 * 4));
            v_c30 = vaddq_s32(v_c30, vld1q_s32(c3_ptr + 0 * 4));
            v_c31 = vaddq_s32(v_c31, vld1q_s32(c3_ptr + 1 * 4));
            v_c32 = vaddq_s32(v_c32, vld1q_s32(c3_ptr + 2 * 4));
            v_c40 = vaddq_s32(v_c40, vld1q_s32(c4_ptr + 0 * 4));
            v_c41 = vaddq_s32(v_c41, vld1q_s32(c4_ptr + 1 * 4));
            v_c42 = vaddq_s32(v_c42, vld1q_s32(c4_ptr + 2 * 4));
            v_c50 = vaddq_s32(v_c50, vld1q_s32(c5_ptr + 0 * 4));
            v_c51 = vaddq_s32(v_c51, vld1q_s32(c5_ptr + 1 * 4));
            v_c52 = vaddq_s32(v_c52, vld1q_s32(c5_ptr + 2 * 4));
            v_c60 = vaddq_s32(v_c60, vld1q_s32(c6_ptr + 0 * 4));
            v_c61 = vaddq_s32(v_c61, vld1q_s32(c6_ptr + 1 * 4));
            v_c62 = vaddq_s32(v_c62, vld1q_s32(c6_ptr + 2 * 4));
            v_c70 = vaddq_s32(v_c70, vld1q_s32(c7_ptr + 0 * 4));
            v_c71 = vaddq_s32(v_c71, vld1q_s32(c7_ptr + 1 * 4));
            v_c72 = vaddq_s32(v_c72, vld1q_s32(c7_ptr + 2 * 4));
        }

        vst1q_s32(c0_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c00, vmin), vmax));
        vst1q_s32(c0_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c01, vmin), vmax));
        vst1q_s32(c0_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c02, vmin), vmax));
        vst1q_s32(c1_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c10, vmin), vmax));
        vst1q_s32(c1_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c11, vmin), vmax));
        vst1q_s32(c1_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c12, vmin), vmax));
        vst1q_s32(c2_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c20, vmin), vmax));
        vst1q_s32(c2_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c21, vmin), vmax));
        vst1q_s32(c2_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c22, vmin), vmax));
        vst1q_s32(c3_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c30, vmin), vmax));
        vst1q_s32(c3_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c31, vmin), vmax));
        vst1q_s32(c3_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c32, vmin), vmax));
        vst1q_s32(c4_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c40, vmin), vmax));
        vst1q_s32(c4_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c41, vmin), vmax));
        vst1q_s32(c4_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c42, vmin), vmax));
        vst1q_s32(c5_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c50, vmin), vmax));
        vst1q_s32(c5_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c51, vmin), vmax));
        vst1q_s32(c5_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c52, vmin), vmax));
        vst1q_s32(c6_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c60, vmin), vmax));
        vst1q_s32(c6_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c61, vmin), vmax));
        vst1q_s32(c6_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c62, vmin), vmax));
        vst1q_s32(c7_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c70, vmin), vmax));
        vst1q_s32(c7_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c71, vmin), vmax));
        vst1q_s32(c7_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c72, vmin), vmax));

        c0_ptr += 12;
        c1_ptr += 12;
        c2_ptr += 12;
        c3_ptr += 12;
        c4_ptr += 12;
        c5_ptr += 12;
        c6_ptr += 12;
        c7_ptr += 12;

        B += ldb;
    }

    const int rem = nc - n;
    if (rem > 0) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;
        int32x4_t v_c10 = vdupq_n_s32(0), v_c11 = v_c10, v_c12 = v_c10;
        int32x4_t v_c20 = vdupq_n_s32(0), v_c21 = v_c20, v_c22 = v_c20;
        int32x4_t v_c30 = vdupq_n_s32(0), v_c31 = v_c30, v_c32 = v_c30;
        int32x4_t v_c40 = vdupq_n_s32(0), v_c41 = v_c40, v_c42 = v_c40;
        int32x4_t v_c50 = vdupq_n_s32(0), v_c51 = v_c50, v_c52 = v_c50;
        int32x4_t v_c60 = vdupq_n_s32(0), v_c61 = v_c60, v_c62 = v_c60;
        int32x4_t v_c70 = vdupq_n_s32(0), v_c71 = v_c70, v_c72 = v_c70;

        const int8_t* NNOPS_RESTRICT a0 = A + 0 * lda;
        const int8_t* NNOPS_RESTRICT a1 = A + r1 * lda;
        const int8_t* NNOPS_RESTRICT a2 = A + r2 * lda;
        const int8_t* NNOPS_RESTRICT a3 = A + r3 * lda;
        const int8_t* NNOPS_RESTRICT a4 = A + r4 * lda;
        const int8_t* NNOPS_RESTRICT a5 = A + r5 * lda;
        const int8_t* NNOPS_RESTRICT a6 = A + r6 * lda;
        const int8_t* NNOPS_RESTRICT a7 = A + r7 * lda;
        const int8_t* NNOPS_RESTRICT b_ptr = B;
        if (rem > 4) {
            for (int k = 0; k < kg; ++k) {
                const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);
                const int8x16_t v_b1 = vld1q_s8(b_ptr + 1 * 16);
                const int8x16_t v_b2 = vld1q_s8(b_ptr + 2 * 16);
                b_ptr += 48;

                int8x16_t v_a = mma_i8_bcast_group(a0);
                v_c00 = vdotq_s32(v_c00, v_b0, v_a);
                v_c01 = vdotq_s32(v_c01, v_b1, v_a);
                v_c02 = vdotq_s32(v_c02, v_b2, v_a);
                a0 += 4;

                v_a = mma_i8_bcast_group(a1);
                v_c10 = vdotq_s32(v_c10, v_b0, v_a);
                v_c11 = vdotq_s32(v_c11, v_b1, v_a);
                v_c12 = vdotq_s32(v_c12, v_b2, v_a);
                a1 += 4;

                v_a = mma_i8_bcast_group(a2);
                v_c20 = vdotq_s32(v_c20, v_b0, v_a);
                v_c21 = vdotq_s32(v_c21, v_b1, v_a);
                v_c22 = vdotq_s32(v_c22, v_b2, v_a);
                a2 += 4;

                v_a = mma_i8_bcast_group(a3);
                v_c30 = vdotq_s32(v_c30, v_b0, v_a);
                v_c31 = vdotq_s32(v_c31, v_b1, v_a);
                v_c32 = vdotq_s32(v_c32, v_b2, v_a);
                a3 += 4;

                v_a = mma_i8_bcast_group(a4);
                v_c40 = vdotq_s32(v_c40, v_b0, v_a);
                v_c41 = vdotq_s32(v_c41, v_b1, v_a);
                v_c42 = vdotq_s32(v_c42, v_b2, v_a);
                a4 += 4;

                v_a = mma_i8_bcast_group(a5);
                v_c50 = vdotq_s32(v_c50, v_b0, v_a);
                v_c51 = vdotq_s32(v_c51, v_b1, v_a);
                v_c52 = vdotq_s32(v_c52, v_b2, v_a);
                a5 += 4;

                v_a = mma_i8_bcast_group(a6);
                v_c60 = vdotq_s32(v_c60, v_b0, v_a);
                v_c61 = vdotq_s32(v_c61, v_b1, v_a);
                v_c62 = vdotq_s32(v_c62, v_b2, v_a);
                a6 += 4;

                v_a = mma_i8_bcast_group(a7);
                v_c70 = vdotq_s32(v_c70, v_b0, v_a);
                v_c71 = vdotq_s32(v_c71, v_b1, v_a);
                v_c72 = vdotq_s32(v_c72, v_b2, v_a);
                a7 += 4;
            }
        } else {
            for (int k = 0; k < kg; ++k) {
                const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);
                b_ptr += 48;

                v_c00 = vdotq_s32(v_c00, v_b0, mma_i8_bcast_group(a0));
                v_c10 = vdotq_s32(v_c10, v_b0, mma_i8_bcast_group(a1));
                v_c20 = vdotq_s32(v_c20, v_b0, mma_i8_bcast_group(a2));
                v_c30 = vdotq_s32(v_c30, v_b0, mma_i8_bcast_group(a3));
                v_c40 = vdotq_s32(v_c40, v_b0, mma_i8_bcast_group(a4));
                v_c50 = vdotq_s32(v_c50, v_b0, mma_i8_bcast_group(a5));
                v_c60 = vdotq_s32(v_c60, v_b0, mma_i8_bcast_group(a6));
                v_c70 = vdotq_s32(v_c70, v_b0, mma_i8_bcast_group(a7));

                a0 += 4;
                a1 += 4;
                a2 += 4;
                a3 += 4;
                a4 += 4;
                a5 += 4;
                a6 += 4;
                a7 += 4;
            }
        }

        const int32x4_t acc[8][3] = {
            {v_c00, v_c01, v_c02}, {v_c10, v_c11, v_c12},
            {v_c20, v_c21, v_c22}, {v_c30, v_c31, v_c32},
            {v_c40, v_c41, v_c42}, {v_c50, v_c51, v_c52},
            {v_c60, v_c61, v_c62}, {v_c70, v_c71, v_c72},
        };
        int32_t* NNOPS_RESTRICT c_rows[8] = {c0_ptr, c1_ptr, c2_ptr, c3_ptr,
                                             c4_ptr, c5_ptr, c6_ptr, c7_ptr};
        for (int r = 0; r < mr; ++r) {
            int32_t* NNOPS_RESTRICT cp = c_rows[r];
            int rr = rem;
            int g = 0;
            for (; rr >= 4; rr -= 4, ++g) {
                int32x4_t v = acc[r][g];
                if constexpr (!zero_mode) {
                    v = vaddq_s32(v, vld1q_s32(cp));
                }
                vst1q_s32(cp, vminq_s32(vmaxq_s32(v, vmin), vmax));
                cp += 4;
            }
            if (rr > 0) {
                int32_t lane[4];
                vst1q_s32(lane, acc[r][g]);
                for (int i = 0; i < rr; ++i) {
                    int32_t v = lane[i];
                    if constexpr (!zero_mode) {
                        v += cp[i];
                    }
                    cp[i] = std::min(std::max(v, clamp_min), clamp_max);
                }
            }
        }
    }
}

template <bool zero_mode = false>
inline void mma_packed_b_mr4_nr12_mrxncxkc_i8(
    int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
    const int8_t* NNOPS_RESTRICT A, int lda,
    const int8_t* NNOPS_RESTRICT B,
    int32_t clamp_min, int32_t clamp_max) noexcept {

    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;

    int32_t* NNOPS_RESTRICT c0_ptr = C + 0 * ldc;
    int32_t* NNOPS_RESTRICT c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    int32_t* NNOPS_RESTRICT c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    int32_t* NNOPS_RESTRICT c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;

    const int kg = kc >> 2;
    const int ldb = static_cast<int>(mma_pack_align(12 * kc));
    const int32x4_t vmin = vdupq_n_s32(clamp_min);
    const int32x4_t vmax = vdupq_n_s32(clamp_max);

    int n = 0;
    for (; n < nc - 11; n += 12) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;
        int32x4_t v_c10 = vdupq_n_s32(0), v_c11 = v_c10, v_c12 = v_c10;
        int32x4_t v_c20 = vdupq_n_s32(0), v_c21 = v_c20, v_c22 = v_c20;
        int32x4_t v_c30 = vdupq_n_s32(0), v_c31 = v_c30, v_c32 = v_c30;

        const int8_t* NNOPS_RESTRICT a0 = A + 0 * lda;
        const int8_t* NNOPS_RESTRICT a1 = A + r1 * lda;
        const int8_t* NNOPS_RESTRICT a2 = A + r2 * lda;
        const int8_t* NNOPS_RESTRICT a3 = A + r3 * lda;
        const int8_t* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kg; ++k) {
            const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);
            const int8x16_t v_b1 = vld1q_s8(b_ptr + 1 * 16);
            const int8x16_t v_b2 = vld1q_s8(b_ptr + 2 * 16);
            b_ptr += 48;

            int8x16_t v_a = mma_i8_bcast_group(a0);
            v_c00 = vdotq_s32(v_c00, v_b0, v_a);
            v_c01 = vdotq_s32(v_c01, v_b1, v_a);
            v_c02 = vdotq_s32(v_c02, v_b2, v_a);
            a0 += 4;

            v_a = mma_i8_bcast_group(a1);
            v_c10 = vdotq_s32(v_c10, v_b0, v_a);
            v_c11 = vdotq_s32(v_c11, v_b1, v_a);
            v_c12 = vdotq_s32(v_c12, v_b2, v_a);
            a1 += 4;

            v_a = mma_i8_bcast_group(a2);
            v_c20 = vdotq_s32(v_c20, v_b0, v_a);
            v_c21 = vdotq_s32(v_c21, v_b1, v_a);
            v_c22 = vdotq_s32(v_c22, v_b2, v_a);
            a2 += 4;

            v_a = mma_i8_bcast_group(a3);
            v_c30 = vdotq_s32(v_c30, v_b0, v_a);
            v_c31 = vdotq_s32(v_c31, v_b1, v_a);
            v_c32 = vdotq_s32(v_c32, v_b2, v_a);
            a3 += 4;
        }

        if constexpr (!zero_mode) {
            v_c00 = vaddq_s32(v_c00, vld1q_s32(c0_ptr + 0 * 4));
            v_c01 = vaddq_s32(v_c01, vld1q_s32(c0_ptr + 1 * 4));
            v_c02 = vaddq_s32(v_c02, vld1q_s32(c0_ptr + 2 * 4));
            v_c10 = vaddq_s32(v_c10, vld1q_s32(c1_ptr + 0 * 4));
            v_c11 = vaddq_s32(v_c11, vld1q_s32(c1_ptr + 1 * 4));
            v_c12 = vaddq_s32(v_c12, vld1q_s32(c1_ptr + 2 * 4));
            v_c20 = vaddq_s32(v_c20, vld1q_s32(c2_ptr + 0 * 4));
            v_c21 = vaddq_s32(v_c21, vld1q_s32(c2_ptr + 1 * 4));
            v_c22 = vaddq_s32(v_c22, vld1q_s32(c2_ptr + 2 * 4));
            v_c30 = vaddq_s32(v_c30, vld1q_s32(c3_ptr + 0 * 4));
            v_c31 = vaddq_s32(v_c31, vld1q_s32(c3_ptr + 1 * 4));
            v_c32 = vaddq_s32(v_c32, vld1q_s32(c3_ptr + 2 * 4));
        }

        vst1q_s32(c0_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c00, vmin), vmax));
        vst1q_s32(c0_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c01, vmin), vmax));
        vst1q_s32(c0_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c02, vmin), vmax));
        vst1q_s32(c1_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c10, vmin), vmax));
        vst1q_s32(c1_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c11, vmin), vmax));
        vst1q_s32(c1_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c12, vmin), vmax));
        vst1q_s32(c2_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c20, vmin), vmax));
        vst1q_s32(c2_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c21, vmin), vmax));
        vst1q_s32(c2_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c22, vmin), vmax));
        vst1q_s32(c3_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c30, vmin), vmax));
        vst1q_s32(c3_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c31, vmin), vmax));
        vst1q_s32(c3_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c32, vmin), vmax));

        c0_ptr += 12;
        c1_ptr += 12;
        c2_ptr += 12;
        c3_ptr += 12;

        B += ldb;
    }

    const int rem = nc - n;
    if (rem > 0) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;
        int32x4_t v_c10 = vdupq_n_s32(0), v_c11 = v_c10, v_c12 = v_c10;
        int32x4_t v_c20 = vdupq_n_s32(0), v_c21 = v_c20, v_c22 = v_c20;
        int32x4_t v_c30 = vdupq_n_s32(0), v_c31 = v_c30, v_c32 = v_c30;

        const int8_t* NNOPS_RESTRICT a0 = A + 0 * lda;
        const int8_t* NNOPS_RESTRICT a1 = A + r1 * lda;
        const int8_t* NNOPS_RESTRICT a2 = A + r2 * lda;
        const int8_t* NNOPS_RESTRICT a3 = A + r3 * lda;
        const int8_t* NNOPS_RESTRICT b_ptr = B;
        if (rem > 4) {
            for (int k = 0; k < kg; ++k) {
                const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);
                const int8x16_t v_b1 = vld1q_s8(b_ptr + 1 * 16);
                const int8x16_t v_b2 = vld1q_s8(b_ptr + 2 * 16);
                b_ptr += 48;

                int8x16_t v_a = mma_i8_bcast_group(a0);
                v_c00 = vdotq_s32(v_c00, v_b0, v_a);
                v_c01 = vdotq_s32(v_c01, v_b1, v_a);
                v_c02 = vdotq_s32(v_c02, v_b2, v_a);
                a0 += 4;

                v_a = mma_i8_bcast_group(a1);
                v_c10 = vdotq_s32(v_c10, v_b0, v_a);
                v_c11 = vdotq_s32(v_c11, v_b1, v_a);
                v_c12 = vdotq_s32(v_c12, v_b2, v_a);
                a1 += 4;

                v_a = mma_i8_bcast_group(a2);
                v_c20 = vdotq_s32(v_c20, v_b0, v_a);
                v_c21 = vdotq_s32(v_c21, v_b1, v_a);
                v_c22 = vdotq_s32(v_c22, v_b2, v_a);
                a2 += 4;

                v_a = mma_i8_bcast_group(a3);
                v_c30 = vdotq_s32(v_c30, v_b0, v_a);
                v_c31 = vdotq_s32(v_c31, v_b1, v_a);
                v_c32 = vdotq_s32(v_c32, v_b2, v_a);
                a3 += 4;
            }
        } else {
            for (int k = 0; k < kg; ++k) {
                const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);
                b_ptr += 48;

                v_c00 = vdotq_s32(v_c00, v_b0, mma_i8_bcast_group(a0));
                v_c10 = vdotq_s32(v_c10, v_b0, mma_i8_bcast_group(a1));
                v_c20 = vdotq_s32(v_c20, v_b0, mma_i8_bcast_group(a2));
                v_c30 = vdotq_s32(v_c30, v_b0, mma_i8_bcast_group(a3));

                a0 += 4;
                a1 += 4;
                a2 += 4;
                a3 += 4;
            }
        }

        const int32x4_t acc[4][3] = {
            {v_c00, v_c01, v_c02}, {v_c10, v_c11, v_c12},
            {v_c20, v_c21, v_c22}, {v_c30, v_c31, v_c32},
        };
        int32_t* NNOPS_RESTRICT c_rows[4] = {c0_ptr, c1_ptr, c2_ptr, c3_ptr};
        for (int r = 0; r < mr; ++r) {
            int32_t* NNOPS_RESTRICT cp = c_rows[r];
            int rr = rem;
            int g = 0;
            for (; rr >= 4; rr -= 4, ++g) {
                int32x4_t v = acc[r][g];
                if constexpr (!zero_mode) {
                    v = vaddq_s32(v, vld1q_s32(cp));
                }
                vst1q_s32(cp, vminq_s32(vmaxq_s32(v, vmin), vmax));
                cp += 4;
            }
            if (rr > 0) {
                int32_t lane[4];
                vst1q_s32(lane, acc[r][g]);
                for (int i = 0; i < rr; ++i) {
                    int32_t v = lane[i];
                    if constexpr (!zero_mode) {
                        v += cp[i];
                    }
                    cp[i] = std::min(std::max(v, clamp_min), clamp_max);
                }
            }
        }
    }
}

template <bool zero_mode = false>
inline void mma_packed_b_mr1_nr12_mrxncxkc_i8(
    int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
    const int8_t* NNOPS_RESTRICT A, int lda,
    const int8_t* NNOPS_RESTRICT B,
    int32_t clamp_min, int32_t clamp_max) noexcept {
    (void)mr;   // one row, so the row count never varies
    (void)lda;  // A is read consecutively along its single row
    (void)ldc;  // one row, so the C row stride never applies

    int32_t* NNOPS_RESTRICT c0_ptr = C;

    const int kg = kc >> 2;
    const int ldb = static_cast<int>(mma_pack_align(12 * kc));
    const int32x4_t vmin = vdupq_n_s32(clamp_min);
    const int32x4_t vmax = vdupq_n_s32(clamp_max);

    int n = 0;
    for (; n < nc - 11; n += 12) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;

        const int8_t* NNOPS_RESTRICT a_ptr = A;
        const int8_t* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kg; ++k) {
            const int8x16_t v_a = mma_i8_bcast_group(a_ptr);
            a_ptr += 4;

            const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);
            const int8x16_t v_b1 = vld1q_s8(b_ptr + 1 * 16);
            const int8x16_t v_b2 = vld1q_s8(b_ptr + 2 * 16);

            v_c00 = vdotq_s32(v_c00, v_b0, v_a);
            v_c01 = vdotq_s32(v_c01, v_b1, v_a);
            v_c02 = vdotq_s32(v_c02, v_b2, v_a);

            b_ptr += 48;
        }

        if constexpr (!zero_mode) {
            v_c00 = vaddq_s32(v_c00, vld1q_s32(c0_ptr + 0 * 4));
            v_c01 = vaddq_s32(v_c01, vld1q_s32(c0_ptr + 1 * 4));
            v_c02 = vaddq_s32(v_c02, vld1q_s32(c0_ptr + 2 * 4));
        }

        vst1q_s32(c0_ptr + 0 * 4, vminq_s32(vmaxq_s32(v_c00, vmin), vmax));
        vst1q_s32(c0_ptr + 1 * 4, vminq_s32(vmaxq_s32(v_c01, vmin), vmax));
        vst1q_s32(c0_ptr + 2 * 4, vminq_s32(vmaxq_s32(v_c02, vmin), vmax));

        c0_ptr += 12;
        B += ldb;
    }

    const int rem = nc - n;
    if (rem > 0) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;

        const int8_t* NNOPS_RESTRICT a_ptr = A;
        const int8_t* NNOPS_RESTRICT b_ptr = B;
        if (rem > 4) {
            for (int k = 0; k < kg; ++k) {
                const int8x16_t v_a = mma_i8_bcast_group(a_ptr);
                a_ptr += 4;

                const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);
                const int8x16_t v_b1 = vld1q_s8(b_ptr + 1 * 16);
                const int8x16_t v_b2 = vld1q_s8(b_ptr + 2 * 16);

                v_c00 = vdotq_s32(v_c00, v_b0, v_a);
                v_c01 = vdotq_s32(v_c01, v_b1, v_a);
                v_c02 = vdotq_s32(v_c02, v_b2, v_a);

                b_ptr += 48;
            }
        } else {
            for (int k = 0; k < kg; ++k) {
                const int8x16_t v_a = mma_i8_bcast_group(a_ptr);
                a_ptr += 4;

                const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);

                v_c00 = vdotq_s32(v_c00, v_b0, v_a);

                b_ptr += 48;
            }
        }

        const int32x4_t acc[3] = {v_c00, v_c01, v_c02};
        int32_t* NNOPS_RESTRICT cp = c0_ptr;
        int rr = rem;
        int g = 0;
        for (; rr >= 4; rr -= 4, ++g) {
            int32x4_t v = acc[g];
            if constexpr (!zero_mode) {
                v = vaddq_s32(v, vld1q_s32(cp));
            }
            vst1q_s32(cp, vminq_s32(vmaxq_s32(v, vmin), vmax));
            cp += 4;
        }
        if (rr > 0) {
            int32_t lane[4];
            vst1q_s32(lane, acc[g]);
            for (int i = 0; i < rr; ++i) {
                int32_t v = lane[i];
                if constexpr (!zero_mode) {
                    v += cp[i];
                }
                cp[i] = std::min(std::max(v, clamp_min), clamp_max);
            }
        }
    }
}

// ---- dispatch tables, consumed by matmul_helper.cpp ---------------------
// Rows are the MR levels {8,4,1}; every kernel takes the runtime panel height,
// so a remainder panel below its own MR runs through the level above it.
template <bool Z>
inline constexpr auto mma_pack_fns_i8 = std::array{
    &mma_packed_ab_mr8_nr12_mrxncxkc_i8<Z>,
    &mma_packed_ab_mr4_nr12_mrxncxkc_i8<Z>,
    &mma_packed_ab_mr1_nr12_mrxncxkc_i8<Z>};

// Row-major A (`lda` in int8 elements, kc % 4 == 0), same MR levels.
template <bool Z>
inline constexpr auto mma_direct_fns_i8 = std::array{
    &mma_packed_b_mr8_nr12_mrxncxkc_i8<Z>,
    &mma_packed_b_mr4_nr12_mrxncxkc_i8<Z>,
    &mma_packed_b_mr1_nr12_mrxncxkc_i8<Z>};

}  // namespace nnops::backend::cpu::aarch64
