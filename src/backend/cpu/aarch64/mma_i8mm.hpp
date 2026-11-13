#pragma once
/// @file mma_i8mm.hpp
/// @brief AArch64 FEAT_I8MM (SMMLA) int8 MMA kernels: s8×s8 → int32.
///
/// One SMMLA multiplies a 2×8 A fragment by an 8×2 B fragment and accumulates
/// the 2×2 int32 result — four times the work of one SDOT lane group, at
/// exactly twice the instruction cost on the cores measured, so the two routes
/// have the same MAC rate. The operands are therefore k-grouped by eight, and a
/// kernel here covers a whole M panel times the N side of one 12-column B
/// panel — the same shape, tile geometry and MR ladder as mma_i8.hpp. The panel
/// width is 12 on purpose: every plan, panel count and blob above this file is
/// shared with the SDOT route, so only the panel *stride* (k rounded up to
/// eight) and the operand layout inside it differ.
///
/// Two A presentations share the same B (the pack_i8mm.hpp layouts):
///   - packed A (mma_pack_mr{8,4,1}_nr12_i8mm): [ceil(kc/8)][rows][8] bytes, so
///     a 2-row fragment — rows r, r+1, eight k values each — is one 16-byte
///     load of the panel.
///   - row-major A (mma_direct_mr{8,4,1}_nr12_i8mm): the same fragment is two
///     8-byte loads through `lda` and a combine. A short final k group is staged
///     through a zeroed buffer, because a row-major row holds exactly kc bytes.
///
/// The accumulators hold each 2×2 block row-major: [r0c0, r0c1, r1c0, r1c1], so
/// a row's two columns are the low (or high) 64 bits and vtrn1q/vtrn2q pair two
/// accumulators into one four-column vector.
///
/// Clamping and zero_mode are the SDOT kernels' contract; `mr` is the live row
/// count, rows past it alias the last live one in A and in C, and only the live
/// rows are stored. `ldb` (0 = the kernel's own slot) overrides the
/// panel-to-panel step for a prepacked B pitched for another k length.

#include <algorithm>
#include <arm_neon.h>
#include <cstdint>
#include <cstring>
#include "backend/cpu/common/common_utils.hpp"

namespace nnops::backend::cpu::aarch64 {

/// B panel width; shared with the SDOT route, so no plan forks on it.
inline constexpr int mma_i8mm_nr_max = 12;
/// Bytes of one 8-k group of a full B panel (12 columns x 8 k).
inline constexpr int mma_i8mm_group_bytes = mma_i8mm_nr_max * 8;

/// A panel slot's byte pitch: the caller's `ldb` when it gave one, else the
/// uniform slot `ceil(kbytes/8)` groups of 96 bytes take.
inline constexpr int mma_i8mm_panel_stride(int ldb, int kbytes) noexcept {
    return (ldb > 0) ? ldb : static_cast<int>(mma_pack_align(mma_i8mm_group_bytes *
                                                            ((kbytes + 7) >> 3)));
}

/// One row pair's twelve columns: pair the six 2×2 accumulators into the even
/// row's three vectors and the odd row's three, add C (unless zero_mode),
/// clamp, store. `co` may alias `ce`; it is only written when store_odd.
template <bool zero_mode>
__attribute__((target("i8mm"), always_inline))
inline void i8mm_store12(int32_t* NNOPS_RESTRICT ce, int32_t* NNOPS_RESTRICT co,
                         bool store_odd,
                         int32x4_t c0, int32x4_t c1, int32x4_t c2,
                         int32x4_t c3, int32x4_t c4, int32x4_t c5,
                         int32x4_t vmin, int32x4_t vmax) noexcept {
    int32x4_t v_e0 = vreinterpretq_s32_u64(
        vtrn1q_u64(vreinterpretq_u64_s32(c0), vreinterpretq_u64_s32(c1)));
    int32x4_t v_e1 = vreinterpretq_s32_u64(
        vtrn1q_u64(vreinterpretq_u64_s32(c2), vreinterpretq_u64_s32(c3)));
    int32x4_t v_e2 = vreinterpretq_s32_u64(
        vtrn1q_u64(vreinterpretq_u64_s32(c4), vreinterpretq_u64_s32(c5)));
    int32x4_t v_o0 = vreinterpretq_s32_u64(
        vtrn2q_u64(vreinterpretq_u64_s32(c0), vreinterpretq_u64_s32(c1)));
    int32x4_t v_o1 = vreinterpretq_s32_u64(
        vtrn2q_u64(vreinterpretq_u64_s32(c2), vreinterpretq_u64_s32(c3)));
    int32x4_t v_o2 = vreinterpretq_s32_u64(
        vtrn2q_u64(vreinterpretq_u64_s32(c4), vreinterpretq_u64_s32(c5)));

    if constexpr (!zero_mode) {
        v_e0 = vaddq_s32(v_e0, vld1q_s32(ce + 0));
        v_e1 = vaddq_s32(v_e1, vld1q_s32(ce + 4));
        v_e2 = vaddq_s32(v_e2, vld1q_s32(ce + 8));
    }
    vst1q_s32(ce + 0, vminq_s32(vmaxq_s32(v_e0, vmin), vmax));
    vst1q_s32(ce + 4, vminq_s32(vmaxq_s32(v_e1, vmin), vmax));
    vst1q_s32(ce + 8, vminq_s32(vmaxq_s32(v_e2, vmin), vmax));

    if (store_odd) {
        if constexpr (!zero_mode) {
            v_o0 = vaddq_s32(v_o0, vld1q_s32(co + 0));
            v_o1 = vaddq_s32(v_o1, vld1q_s32(co + 4));
            v_o2 = vaddq_s32(v_o2, vld1q_s32(co + 8));
        }
        vst1q_s32(co + 0, vminq_s32(vmaxq_s32(v_o0, vmin), vmax));
        vst1q_s32(co + 4, vminq_s32(vmaxq_s32(v_o1, vmin), vmax));
        vst1q_s32(co + 8, vminq_s32(vmaxq_s32(v_o2, vmin), vmax));
    }
}

/// The live width of a panel that is not a whole 12 columns: lane stores only.
template <bool zero_mode>
__attribute__((target("i8mm"), always_inline))
inline void i8mm_store12_tail(int32_t* NNOPS_RESTRICT ce,
                              int32_t* NNOPS_RESTRICT co, bool store_odd, int rem,
                              int32x4_t c0, int32x4_t c1, int32x4_t c2,
                              int32x4_t c3, int32x4_t c4, int32x4_t c5,
                              int32_t clamp_min, int32_t clamp_max) noexcept {
    const int32x4_t acc[6] = {c0, c1, c2, c3, c4, c5};
    for (int half = 0; half < 2; ++half) {
        if (half == 1 && !store_odd) { break; }
        int32_t* NNOPS_RESTRICT cp = (half == 0) ? ce : co;
        for (int j = 0; j < 6; ++j) {
            const int col0 = 2 * j;
            if (col0 >= rem) { break; }
            int32_t lane[4];
            vst1q_s32(lane, acc[j]);
            for (int h = 0; h < 2; ++h) {
                if (col0 + h >= rem) { break; }
                int32_t v = lane[half * 2 + h];
                if constexpr (!zero_mode) { v += cp[col0 + h]; }
                cp[col0 + h] = std::min(std::max(v, clamp_min), clamp_max);
            }
        }
    }
}

// =========================================================================
//  packed A + packed B  (mr = 8, 4, 1)
// =========================================================================

template <bool zero_mode = false>
__attribute__((target("i8mm")))
inline void mma_pack_mr8_nr12_i8mm(
    int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
    const int8_t* NNOPS_RESTRICT A,
    const int8_t* NNOPS_RESTRICT B, int ldb,
    int32_t clamp_min, int32_t clamp_max) noexcept {

    int32_t* NNOPS_RESTRICT c0_ptr = C + 0 * ldc;
    int32_t* NNOPS_RESTRICT c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    int32_t* NNOPS_RESTRICT c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    int32_t* NNOPS_RESTRICT c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;
    int32_t* NNOPS_RESTRICT c4_ptr = (mr > 4) ? C + 4 * ldc : c3_ptr;
    int32_t* NNOPS_RESTRICT c5_ptr = (mr > 5) ? C + 5 * ldc : c4_ptr;
    int32_t* NNOPS_RESTRICT c6_ptr = (mr > 6) ? C + 6 * ldc : c5_ptr;
    int32_t* NNOPS_RESTRICT c7_ptr = (mr > 7) ? C + 7 * ldc : c6_ptr;

    const int kbytes = (kc + 7) & ~7;
    const int kg = kbytes >> 3;
    const int ldb_panel = mma_i8mm_panel_stride(ldb, kbytes);
    const int32x4_t vmin = vdupq_n_s32(clamp_min);
    const int32x4_t vmax = vdupq_n_s32(clamp_max);

    for (int n = 0; n < nc; n += mma_i8mm_nr_max) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;
        int32x4_t v_c03 = v_c00, v_c04 = v_c00, v_c05 = v_c00;
        int32x4_t v_c10 = vdupq_n_s32(0), v_c11 = v_c10, v_c12 = v_c10;
        int32x4_t v_c13 = v_c10, v_c14 = v_c10, v_c15 = v_c10;
        int32x4_t v_c20 = vdupq_n_s32(0), v_c21 = v_c20, v_c22 = v_c20;
        int32x4_t v_c23 = v_c20, v_c24 = v_c20, v_c25 = v_c20;
        int32x4_t v_c30 = vdupq_n_s32(0), v_c31 = v_c30, v_c32 = v_c30;
        int32x4_t v_c33 = v_c30, v_c34 = v_c30, v_c35 = v_c30;

        const int8_t* NNOPS_RESTRICT a_ptr = A;
        const int8_t* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kg; ++k) {
            const int8x16_t v_a0 = vld1q_s8(a_ptr + 0 * 16);  // rows 0,1
            const int8x16_t v_a1 = vld1q_s8(a_ptr + 1 * 16);  // rows 2,3
            const int8x16_t v_a2 = vld1q_s8(a_ptr + 2 * 16);  // rows 4,5
            const int8x16_t v_a3 = vld1q_s8(a_ptr + 3 * 16);  // rows 6,7
            a_ptr += 4 * 16;

            const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);  // columns 0,1
            v_c00 = vmmlaq_s32(v_c00, v_a0, v_b0);
            v_c10 = vmmlaq_s32(v_c10, v_a1, v_b0);
            v_c20 = vmmlaq_s32(v_c20, v_a2, v_b0);
            v_c30 = vmmlaq_s32(v_c30, v_a3, v_b0);

            const int8x16_t v_b1 = vld1q_s8(b_ptr + 1 * 16);  // columns 2,3
            v_c01 = vmmlaq_s32(v_c01, v_a0, v_b1);
            v_c11 = vmmlaq_s32(v_c11, v_a1, v_b1);
            v_c21 = vmmlaq_s32(v_c21, v_a2, v_b1);
            v_c31 = vmmlaq_s32(v_c31, v_a3, v_b1);

            const int8x16_t v_b2 = vld1q_s8(b_ptr + 2 * 16);  // columns 4,5
            v_c02 = vmmlaq_s32(v_c02, v_a0, v_b2);
            v_c12 = vmmlaq_s32(v_c12, v_a1, v_b2);
            v_c22 = vmmlaq_s32(v_c22, v_a2, v_b2);
            v_c32 = vmmlaq_s32(v_c32, v_a3, v_b2);

            const int8x16_t v_b3 = vld1q_s8(b_ptr + 3 * 16);  // columns 6,7
            v_c03 = vmmlaq_s32(v_c03, v_a0, v_b3);
            v_c13 = vmmlaq_s32(v_c13, v_a1, v_b3);
            v_c23 = vmmlaq_s32(v_c23, v_a2, v_b3);
            v_c33 = vmmlaq_s32(v_c33, v_a3, v_b3);

            const int8x16_t v_b4 = vld1q_s8(b_ptr + 4 * 16);  // columns 8,9
            v_c04 = vmmlaq_s32(v_c04, v_a0, v_b4);
            v_c14 = vmmlaq_s32(v_c14, v_a1, v_b4);
            v_c24 = vmmlaq_s32(v_c24, v_a2, v_b4);
            v_c34 = vmmlaq_s32(v_c34, v_a3, v_b4);

            const int8x16_t v_b5 = vld1q_s8(b_ptr + 5 * 16);  // columns 10,11
            v_c05 = vmmlaq_s32(v_c05, v_a0, v_b5);
            v_c15 = vmmlaq_s32(v_c15, v_a1, v_b5);
            v_c25 = vmmlaq_s32(v_c25, v_a2, v_b5);
            v_c35 = vmmlaq_s32(v_c35, v_a3, v_b5);

            b_ptr += mma_i8mm_group_bytes;
        }

        if (n + mma_i8mm_nr_max <= nc) {
            i8mm_store12<zero_mode>(c0_ptr, c1_ptr, mr > 1, v_c00, v_c01, v_c02,
                                    v_c03, v_c04, v_c05, vmin, vmax);
            i8mm_store12<zero_mode>(c2_ptr, c3_ptr, mr > 3, v_c10, v_c11, v_c12,
                                    v_c13, v_c14, v_c15, vmin, vmax);
            i8mm_store12<zero_mode>(c4_ptr, c5_ptr, mr > 5, v_c20, v_c21, v_c22,
                                    v_c23, v_c24, v_c25, vmin, vmax);
            i8mm_store12<zero_mode>(c6_ptr, c7_ptr, mr > 7, v_c30, v_c31, v_c32,
                                    v_c33, v_c34, v_c35, vmin, vmax);
        } else {
            const int rem = nc - n;
            i8mm_store12_tail<zero_mode>(c0_ptr, c1_ptr, mr > 1, rem, v_c00, v_c01,
                                         v_c02, v_c03, v_c04, v_c05,
                                         clamp_min, clamp_max);
            i8mm_store12_tail<zero_mode>(c2_ptr, c3_ptr, mr > 3, rem, v_c10, v_c11,
                                         v_c12, v_c13, v_c14, v_c15,
                                         clamp_min, clamp_max);
            i8mm_store12_tail<zero_mode>(c4_ptr, c5_ptr, mr > 5, rem, v_c20, v_c21,
                                         v_c22, v_c23, v_c24, v_c25,
                                         clamp_min, clamp_max);
            i8mm_store12_tail<zero_mode>(c6_ptr, c7_ptr, mr > 7, rem, v_c30, v_c31,
                                         v_c32, v_c33, v_c34, v_c35,
                                         clamp_min, clamp_max);
        }

        c0_ptr += 12; c1_ptr += 12; c2_ptr += 12; c3_ptr += 12;
        c4_ptr += 12; c5_ptr += 12; c6_ptr += 12; c7_ptr += 12;
        B += ldb_panel;
    }
}

template <bool zero_mode = false>
__attribute__((target("i8mm")))
inline void mma_pack_mr4_nr12_i8mm(
    int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
    const int8_t* NNOPS_RESTRICT A,
    const int8_t* NNOPS_RESTRICT B, int ldb,
    int32_t clamp_min, int32_t clamp_max) noexcept {

    int32_t* NNOPS_RESTRICT c0_ptr = C + 0 * ldc;
    int32_t* NNOPS_RESTRICT c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    int32_t* NNOPS_RESTRICT c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    int32_t* NNOPS_RESTRICT c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;

    const int kbytes = (kc + 7) & ~7;
    const int kg = kbytes >> 3;
    const int ldb_panel = mma_i8mm_panel_stride(ldb, kbytes);
    const int32x4_t vmin = vdupq_n_s32(clamp_min);
    const int32x4_t vmax = vdupq_n_s32(clamp_max);

    for (int n = 0; n < nc; n += mma_i8mm_nr_max) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;
        int32x4_t v_c03 = v_c00, v_c04 = v_c00, v_c05 = v_c00;
        int32x4_t v_c10 = vdupq_n_s32(0), v_c11 = v_c10, v_c12 = v_c10;
        int32x4_t v_c13 = v_c10, v_c14 = v_c10, v_c15 = v_c10;

        const int8_t* NNOPS_RESTRICT a_ptr = A;
        const int8_t* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kg; ++k) {
            const int8x16_t v_a0 = vld1q_s8(a_ptr + 0 * 16);  // rows 0,1
            const int8x16_t v_a1 = vld1q_s8(a_ptr + 1 * 16);  // rows 2,3
            a_ptr += 2 * 16;

            const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);
            v_c00 = vmmlaq_s32(v_c00, v_a0, v_b0);
            v_c10 = vmmlaq_s32(v_c10, v_a1, v_b0);

            const int8x16_t v_b1 = vld1q_s8(b_ptr + 1 * 16);
            v_c01 = vmmlaq_s32(v_c01, v_a0, v_b1);
            v_c11 = vmmlaq_s32(v_c11, v_a1, v_b1);

            const int8x16_t v_b2 = vld1q_s8(b_ptr + 2 * 16);
            v_c02 = vmmlaq_s32(v_c02, v_a0, v_b2);
            v_c12 = vmmlaq_s32(v_c12, v_a1, v_b2);

            const int8x16_t v_b3 = vld1q_s8(b_ptr + 3 * 16);
            v_c03 = vmmlaq_s32(v_c03, v_a0, v_b3);
            v_c13 = vmmlaq_s32(v_c13, v_a1, v_b3);

            const int8x16_t v_b4 = vld1q_s8(b_ptr + 4 * 16);
            v_c04 = vmmlaq_s32(v_c04, v_a0, v_b4);
            v_c14 = vmmlaq_s32(v_c14, v_a1, v_b4);

            const int8x16_t v_b5 = vld1q_s8(b_ptr + 5 * 16);
            v_c05 = vmmlaq_s32(v_c05, v_a0, v_b5);
            v_c15 = vmmlaq_s32(v_c15, v_a1, v_b5);

            b_ptr += mma_i8mm_group_bytes;
        }

        if (n + mma_i8mm_nr_max <= nc) {
            i8mm_store12<zero_mode>(c0_ptr, c1_ptr, mr > 1, v_c00, v_c01, v_c02,
                                    v_c03, v_c04, v_c05, vmin, vmax);
            i8mm_store12<zero_mode>(c2_ptr, c3_ptr, mr > 3, v_c10, v_c11, v_c12,
                                    v_c13, v_c14, v_c15, vmin, vmax);
        } else {
            const int rem = nc - n;
            i8mm_store12_tail<zero_mode>(c0_ptr, c1_ptr, mr > 1, rem, v_c00, v_c01,
                                         v_c02, v_c03, v_c04, v_c05,
                                         clamp_min, clamp_max);
            i8mm_store12_tail<zero_mode>(c2_ptr, c3_ptr, mr > 3, rem, v_c10, v_c11,
                                         v_c12, v_c13, v_c14, v_c15,
                                         clamp_min, clamp_max);
        }

        c0_ptr += 12; c1_ptr += 12; c2_ptr += 12; c3_ptr += 12;
        B += ldb_panel;
    }
}

template <bool zero_mode = false>
__attribute__((target("i8mm")))
inline void mma_pack_mr1_nr12_i8mm(
    int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
    const int8_t* NNOPS_RESTRICT A,
    const int8_t* NNOPS_RESTRICT B, int ldb,
    int32_t clamp_min, int32_t clamp_max) noexcept {
    (void)mr;   // one row, so the row count never varies
    (void)ldc;  // one row, so the C row stride never applies

    int32_t* NNOPS_RESTRICT c0_ptr = C;

    const int kbytes = (kc + 7) & ~7;
    const int kg = kbytes >> 3;
    const int ldb_panel = mma_i8mm_panel_stride(ldb, kbytes);
    const int32x4_t vmin = vdupq_n_s32(clamp_min);
    const int32x4_t vmax = vdupq_n_s32(clamp_max);
    const int8x8_t vz = vdup_n_s8(0);

    for (int n = 0; n < nc; n += mma_i8mm_nr_max) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;
        int32x4_t v_c03 = v_c00, v_c04 = v_c00, v_c05 = v_c00;

        const int8_t* NNOPS_RESTRICT a_ptr = A;
        const int8_t* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kg; ++k) {
            // The second row of the fragment is zero: the odd lanes of every
            // accumulator are dead and only the low 64 bits are stored.
            const int8x16_t v_a = vcombine_s8(vld1_s8(a_ptr), vz);
            a_ptr += 8;

            v_c00 = vmmlaq_s32(v_c00, v_a, vld1q_s8(b_ptr + 0 * 16));
            v_c01 = vmmlaq_s32(v_c01, v_a, vld1q_s8(b_ptr + 1 * 16));
            v_c02 = vmmlaq_s32(v_c02, v_a, vld1q_s8(b_ptr + 2 * 16));
            v_c03 = vmmlaq_s32(v_c03, v_a, vld1q_s8(b_ptr + 3 * 16));
            v_c04 = vmmlaq_s32(v_c04, v_a, vld1q_s8(b_ptr + 4 * 16));
            v_c05 = vmmlaq_s32(v_c05, v_a, vld1q_s8(b_ptr + 5 * 16));

            b_ptr += mma_i8mm_group_bytes;
        }

        const int rem = nc - n;
        if (rem >= mma_i8mm_nr_max) {
            int32x4_t v = vcombine_s32(vget_low_s32(v_c00), vget_low_s32(v_c01));
            if constexpr (!zero_mode) { v = vaddq_s32(v, vld1q_s32(c0_ptr + 0)); }
            vst1q_s32(c0_ptr + 0, vminq_s32(vmaxq_s32(v, vmin), vmax));
            v = vcombine_s32(vget_low_s32(v_c02), vget_low_s32(v_c03));
            if constexpr (!zero_mode) { v = vaddq_s32(v, vld1q_s32(c0_ptr + 4)); }
            vst1q_s32(c0_ptr + 4, vminq_s32(vmaxq_s32(v, vmin), vmax));
            v = vcombine_s32(vget_low_s32(v_c04), vget_low_s32(v_c05));
            if constexpr (!zero_mode) { v = vaddq_s32(v, vld1q_s32(c0_ptr + 8)); }
            vst1q_s32(c0_ptr + 8, vminq_s32(vmaxq_s32(v, vmin), vmax));
        } else {
            const int32x4_t acc[6] = {v_c00, v_c01, v_c02, v_c03, v_c04, v_c05};
            for (int j = 0; j < 6; ++j) {
                const int col0 = 2 * j;
                if (col0 >= rem) { break; }
                int32_t lane[4];
                vst1q_s32(lane, acc[j]);
                for (int h = 0; h < 2; ++h) {
                    if (col0 + h >= rem) { break; }
                    int32_t v = lane[h];
                    if constexpr (!zero_mode) { v += c0_ptr[col0 + h]; }
                    c0_ptr[col0 + h] = std::min(std::max(v, clamp_min), clamp_max);
                }
            }
        }

        c0_ptr += 12;
        B += ldb_panel;
    }
}

// =========================================================================
//  row-major A + packed B  (mr = 8, 4, 1)
// =========================================================================
//
// The fragment is two 8-byte loads and a combine; a short final k group is
// staged through a zeroed 8-byte row per A row, since a row holds kc bytes.

template <bool zero_mode = false>
__attribute__((target("i8mm")))
inline void mma_direct_mr8_nr12_i8mm(
    int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
    const int8_t* NNOPS_RESTRICT A, int lda,
    const int8_t* NNOPS_RESTRICT B, int ldb,
    int32_t clamp_min, int32_t clamp_max) noexcept {

    int32_t* NNOPS_RESTRICT c0_ptr = C + 0 * ldc;
    int32_t* NNOPS_RESTRICT c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    int32_t* NNOPS_RESTRICT c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    int32_t* NNOPS_RESTRICT c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;
    int32_t* NNOPS_RESTRICT c4_ptr = (mr > 4) ? C + 4 * ldc : c3_ptr;
    int32_t* NNOPS_RESTRICT c5_ptr = (mr > 5) ? C + 5 * ldc : c4_ptr;
    int32_t* NNOPS_RESTRICT c6_ptr = (mr > 6) ? C + 6 * ldc : c5_ptr;
    int32_t* NNOPS_RESTRICT c7_ptr = (mr > 7) ? C + 7 * ldc : c6_ptr;

    // Rows past the live count read the last live row's bytes.
    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;
    const int r4 = (mr > 4) ? 4 : r3;
    const int r5 = (mr > 5) ? 5 : r4;
    const int r6 = (mr > 6) ? 6 : r5;
    const int r7 = (mr > 7) ? 7 : r6;

    const int kbytes = (kc + 7) & ~7;
    const int kg = kbytes >> 3;
    const int ldb_panel = mma_i8mm_panel_stride(ldb, kbytes);
    const int32x4_t vmin = vdupq_n_s32(clamp_min);
    const int32x4_t vmax = vdupq_n_s32(clamp_max);

    for (int n = 0; n < nc; n += mma_i8mm_nr_max) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;
        int32x4_t v_c03 = v_c00, v_c04 = v_c00, v_c05 = v_c00;
        int32x4_t v_c10 = vdupq_n_s32(0), v_c11 = v_c10, v_c12 = v_c10;
        int32x4_t v_c13 = v_c10, v_c14 = v_c10, v_c15 = v_c10;
        int32x4_t v_c20 = vdupq_n_s32(0), v_c21 = v_c20, v_c22 = v_c20;
        int32x4_t v_c23 = v_c20, v_c24 = v_c20, v_c25 = v_c20;
        int32x4_t v_c30 = vdupq_n_s32(0), v_c31 = v_c30, v_c32 = v_c30;
        int32x4_t v_c33 = v_c30, v_c34 = v_c30, v_c35 = v_c30;

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
            int8x16_t v_a0, v_a1, v_a2, v_a3;
            if (k + 1 < kg || (kc & 7) == 0) {
                v_a0 = vcombine_s8(vld1_s8(a0), vld1_s8(a1));
                v_a1 = vcombine_s8(vld1_s8(a2), vld1_s8(a3));
                v_a2 = vcombine_s8(vld1_s8(a4), vld1_s8(a5));
                v_a3 = vcombine_s8(vld1_s8(a6), vld1_s8(a7));
                a0 += 8; a1 += 8; a2 += 8; a3 += 8;
                a4 += 8; a5 += 8; a6 += 8; a7 += 8;
            } else {
                const int live = kc - 8 * k;
                int8_t t[8][8] = {};
                std::memcpy(t[0], a0, static_cast<size_t>(live));
                std::memcpy(t[1], a1, static_cast<size_t>(live));
                std::memcpy(t[2], a2, static_cast<size_t>(live));
                std::memcpy(t[3], a3, static_cast<size_t>(live));
                std::memcpy(t[4], a4, static_cast<size_t>(live));
                std::memcpy(t[5], a5, static_cast<size_t>(live));
                std::memcpy(t[6], a6, static_cast<size_t>(live));
                std::memcpy(t[7], a7, static_cast<size_t>(live));
                v_a0 = vcombine_s8(vld1_s8(t[0]), vld1_s8(t[1]));
                v_a1 = vcombine_s8(vld1_s8(t[2]), vld1_s8(t[3]));
                v_a2 = vcombine_s8(vld1_s8(t[4]), vld1_s8(t[5]));
                v_a3 = vcombine_s8(vld1_s8(t[6]), vld1_s8(t[7]));
            }

            const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);
            v_c00 = vmmlaq_s32(v_c00, v_a0, v_b0);
            v_c10 = vmmlaq_s32(v_c10, v_a1, v_b0);
            v_c20 = vmmlaq_s32(v_c20, v_a2, v_b0);
            v_c30 = vmmlaq_s32(v_c30, v_a3, v_b0);

            const int8x16_t v_b1 = vld1q_s8(b_ptr + 1 * 16);
            v_c01 = vmmlaq_s32(v_c01, v_a0, v_b1);
            v_c11 = vmmlaq_s32(v_c11, v_a1, v_b1);
            v_c21 = vmmlaq_s32(v_c21, v_a2, v_b1);
            v_c31 = vmmlaq_s32(v_c31, v_a3, v_b1);

            const int8x16_t v_b2 = vld1q_s8(b_ptr + 2 * 16);
            v_c02 = vmmlaq_s32(v_c02, v_a0, v_b2);
            v_c12 = vmmlaq_s32(v_c12, v_a1, v_b2);
            v_c22 = vmmlaq_s32(v_c22, v_a2, v_b2);
            v_c32 = vmmlaq_s32(v_c32, v_a3, v_b2);

            const int8x16_t v_b3 = vld1q_s8(b_ptr + 3 * 16);
            v_c03 = vmmlaq_s32(v_c03, v_a0, v_b3);
            v_c13 = vmmlaq_s32(v_c13, v_a1, v_b3);
            v_c23 = vmmlaq_s32(v_c23, v_a2, v_b3);
            v_c33 = vmmlaq_s32(v_c33, v_a3, v_b3);

            const int8x16_t v_b4 = vld1q_s8(b_ptr + 4 * 16);
            v_c04 = vmmlaq_s32(v_c04, v_a0, v_b4);
            v_c14 = vmmlaq_s32(v_c14, v_a1, v_b4);
            v_c24 = vmmlaq_s32(v_c24, v_a2, v_b4);
            v_c34 = vmmlaq_s32(v_c34, v_a3, v_b4);

            const int8x16_t v_b5 = vld1q_s8(b_ptr + 5 * 16);
            v_c05 = vmmlaq_s32(v_c05, v_a0, v_b5);
            v_c15 = vmmlaq_s32(v_c15, v_a1, v_b5);
            v_c25 = vmmlaq_s32(v_c25, v_a2, v_b5);
            v_c35 = vmmlaq_s32(v_c35, v_a3, v_b5);

            b_ptr += mma_i8mm_group_bytes;
        }

        if (n + mma_i8mm_nr_max <= nc) {
            i8mm_store12<zero_mode>(c0_ptr, c1_ptr, mr > 1, v_c00, v_c01, v_c02,
                                    v_c03, v_c04, v_c05, vmin, vmax);
            i8mm_store12<zero_mode>(c2_ptr, c3_ptr, mr > 3, v_c10, v_c11, v_c12,
                                    v_c13, v_c14, v_c15, vmin, vmax);
            i8mm_store12<zero_mode>(c4_ptr, c5_ptr, mr > 5, v_c20, v_c21, v_c22,
                                    v_c23, v_c24, v_c25, vmin, vmax);
            i8mm_store12<zero_mode>(c6_ptr, c7_ptr, mr > 7, v_c30, v_c31, v_c32,
                                    v_c33, v_c34, v_c35, vmin, vmax);
        } else {
            const int rem = nc - n;
            i8mm_store12_tail<zero_mode>(c0_ptr, c1_ptr, mr > 1, rem, v_c00, v_c01,
                                         v_c02, v_c03, v_c04, v_c05,
                                         clamp_min, clamp_max);
            i8mm_store12_tail<zero_mode>(c2_ptr, c3_ptr, mr > 3, rem, v_c10, v_c11,
                                         v_c12, v_c13, v_c14, v_c15,
                                         clamp_min, clamp_max);
            i8mm_store12_tail<zero_mode>(c4_ptr, c5_ptr, mr > 5, rem, v_c20, v_c21,
                                         v_c22, v_c23, v_c24, v_c25,
                                         clamp_min, clamp_max);
            i8mm_store12_tail<zero_mode>(c6_ptr, c7_ptr, mr > 7, rem, v_c30, v_c31,
                                         v_c32, v_c33, v_c34, v_c35,
                                         clamp_min, clamp_max);
        }

        c0_ptr += 12; c1_ptr += 12; c2_ptr += 12; c3_ptr += 12;
        c4_ptr += 12; c5_ptr += 12; c6_ptr += 12; c7_ptr += 12;
        B += ldb_panel;
    }
}

template <bool zero_mode = false>
__attribute__((target("i8mm")))
inline void mma_direct_mr4_nr12_i8mm(
    int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
    const int8_t* NNOPS_RESTRICT A, int lda,
    const int8_t* NNOPS_RESTRICT B, int ldb,
    int32_t clamp_min, int32_t clamp_max) noexcept {

    int32_t* NNOPS_RESTRICT c0_ptr = C + 0 * ldc;
    int32_t* NNOPS_RESTRICT c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    int32_t* NNOPS_RESTRICT c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    int32_t* NNOPS_RESTRICT c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;

    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;

    const int kbytes = (kc + 7) & ~7;
    const int kg = kbytes >> 3;
    const int ldb_panel = mma_i8mm_panel_stride(ldb, kbytes);
    const int32x4_t vmin = vdupq_n_s32(clamp_min);
    const int32x4_t vmax = vdupq_n_s32(clamp_max);

    for (int n = 0; n < nc; n += mma_i8mm_nr_max) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;
        int32x4_t v_c03 = v_c00, v_c04 = v_c00, v_c05 = v_c00;
        int32x4_t v_c10 = vdupq_n_s32(0), v_c11 = v_c10, v_c12 = v_c10;
        int32x4_t v_c13 = v_c10, v_c14 = v_c10, v_c15 = v_c10;

        const int8_t* NNOPS_RESTRICT a0 = A + 0 * lda;
        const int8_t* NNOPS_RESTRICT a1 = A + r1 * lda;
        const int8_t* NNOPS_RESTRICT a2 = A + r2 * lda;
        const int8_t* NNOPS_RESTRICT a3 = A + r3 * lda;
        const int8_t* NNOPS_RESTRICT b_ptr = B;

        for (int k = 0; k < kg; ++k) {
            int8x16_t v_a0, v_a1;
            if (k + 1 < kg || (kc & 7) == 0) {
                v_a0 = vcombine_s8(vld1_s8(a0), vld1_s8(a1));
                v_a1 = vcombine_s8(vld1_s8(a2), vld1_s8(a3));
                a0 += 8; a1 += 8; a2 += 8; a3 += 8;
            } else {
                const int live = kc - 8 * k;
                int8_t t[4][8] = {};
                std::memcpy(t[0], a0, static_cast<size_t>(live));
                std::memcpy(t[1], a1, static_cast<size_t>(live));
                std::memcpy(t[2], a2, static_cast<size_t>(live));
                std::memcpy(t[3], a3, static_cast<size_t>(live));
                v_a0 = vcombine_s8(vld1_s8(t[0]), vld1_s8(t[1]));
                v_a1 = vcombine_s8(vld1_s8(t[2]), vld1_s8(t[3]));
            }

            const int8x16_t v_b0 = vld1q_s8(b_ptr + 0 * 16);
            v_c00 = vmmlaq_s32(v_c00, v_a0, v_b0);
            v_c10 = vmmlaq_s32(v_c10, v_a1, v_b0);

            const int8x16_t v_b1 = vld1q_s8(b_ptr + 1 * 16);
            v_c01 = vmmlaq_s32(v_c01, v_a0, v_b1);
            v_c11 = vmmlaq_s32(v_c11, v_a1, v_b1);

            const int8x16_t v_b2 = vld1q_s8(b_ptr + 2 * 16);
            v_c02 = vmmlaq_s32(v_c02, v_a0, v_b2);
            v_c12 = vmmlaq_s32(v_c12, v_a1, v_b2);

            const int8x16_t v_b3 = vld1q_s8(b_ptr + 3 * 16);
            v_c03 = vmmlaq_s32(v_c03, v_a0, v_b3);
            v_c13 = vmmlaq_s32(v_c13, v_a1, v_b3);

            const int8x16_t v_b4 = vld1q_s8(b_ptr + 4 * 16);
            v_c04 = vmmlaq_s32(v_c04, v_a0, v_b4);
            v_c14 = vmmlaq_s32(v_c14, v_a1, v_b4);

            const int8x16_t v_b5 = vld1q_s8(b_ptr + 5 * 16);
            v_c05 = vmmlaq_s32(v_c05, v_a0, v_b5);
            v_c15 = vmmlaq_s32(v_c15, v_a1, v_b5);

            b_ptr += mma_i8mm_group_bytes;
        }

        if (n + mma_i8mm_nr_max <= nc) {
            i8mm_store12<zero_mode>(c0_ptr, c1_ptr, mr > 1, v_c00, v_c01, v_c02,
                                    v_c03, v_c04, v_c05, vmin, vmax);
            i8mm_store12<zero_mode>(c2_ptr, c3_ptr, mr > 3, v_c10, v_c11, v_c12,
                                    v_c13, v_c14, v_c15, vmin, vmax);
        } else {
            const int rem = nc - n;
            i8mm_store12_tail<zero_mode>(c0_ptr, c1_ptr, mr > 1, rem, v_c00, v_c01,
                                         v_c02, v_c03, v_c04, v_c05,
                                         clamp_min, clamp_max);
            i8mm_store12_tail<zero_mode>(c2_ptr, c3_ptr, mr > 3, rem, v_c10, v_c11,
                                         v_c12, v_c13, v_c14, v_c15,
                                         clamp_min, clamp_max);
        }

        c0_ptr += 12; c1_ptr += 12; c2_ptr += 12; c3_ptr += 12;
        B += ldb_panel;
    }
}

template <bool zero_mode = false>
__attribute__((target("i8mm")))
inline void mma_direct_mr1_nr12_i8mm(
    int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
    const int8_t* NNOPS_RESTRICT A, int lda,
    const int8_t* NNOPS_RESTRICT B, int ldb,
    int32_t clamp_min, int32_t clamp_max) noexcept {
    (void)mr;
    (void)ldc;
    (void)lda;  // one row: A is read consecutively

    int32_t* NNOPS_RESTRICT c0_ptr = C;

    const int kbytes = (kc + 7) & ~7;
    const int kg = kbytes >> 3;
    const int ldb_panel = mma_i8mm_panel_stride(ldb, kbytes);
    const int32x4_t vmin = vdupq_n_s32(clamp_min);
    const int32x4_t vmax = vdupq_n_s32(clamp_max);
    const int8x8_t vz = vdup_n_s8(0);

    for (int n = 0; n < nc; n += mma_i8mm_nr_max) {
        int32x4_t v_c00 = vdupq_n_s32(0), v_c01 = v_c00, v_c02 = v_c00;
        int32x4_t v_c03 = v_c00, v_c04 = v_c00, v_c05 = v_c00;

        const int8_t* NNOPS_RESTRICT a_ptr = A;
        const int8_t* NNOPS_RESTRICT b_ptr = B;
        for (int k = 0; k < kg; ++k) {
            int8x8_t v_row;
            if (k + 1 < kg || (kc & 7) == 0) {
                v_row = vld1_s8(a_ptr);
                a_ptr += 8;
            } else {
                int8_t t[8] = {};
                std::memcpy(t, a_ptr, static_cast<size_t>(kc - 8 * k));
                v_row = vld1_s8(t);
            }
            const int8x16_t v_a = vcombine_s8(v_row, vz);

            v_c00 = vmmlaq_s32(v_c00, v_a, vld1q_s8(b_ptr + 0 * 16));
            v_c01 = vmmlaq_s32(v_c01, v_a, vld1q_s8(b_ptr + 1 * 16));
            v_c02 = vmmlaq_s32(v_c02, v_a, vld1q_s8(b_ptr + 2 * 16));
            v_c03 = vmmlaq_s32(v_c03, v_a, vld1q_s8(b_ptr + 3 * 16));
            v_c04 = vmmlaq_s32(v_c04, v_a, vld1q_s8(b_ptr + 4 * 16));
            v_c05 = vmmlaq_s32(v_c05, v_a, vld1q_s8(b_ptr + 5 * 16));

            b_ptr += mma_i8mm_group_bytes;
        }

        const int rem = nc - n;
        if (rem >= mma_i8mm_nr_max) {
            int32x4_t v = vcombine_s32(vget_low_s32(v_c00), vget_low_s32(v_c01));
            if constexpr (!zero_mode) { v = vaddq_s32(v, vld1q_s32(c0_ptr + 0)); }
            vst1q_s32(c0_ptr + 0, vminq_s32(vmaxq_s32(v, vmin), vmax));
            v = vcombine_s32(vget_low_s32(v_c02), vget_low_s32(v_c03));
            if constexpr (!zero_mode) { v = vaddq_s32(v, vld1q_s32(c0_ptr + 4)); }
            vst1q_s32(c0_ptr + 4, vminq_s32(vmaxq_s32(v, vmin), vmax));
            v = vcombine_s32(vget_low_s32(v_c04), vget_low_s32(v_c05));
            if constexpr (!zero_mode) { v = vaddq_s32(v, vld1q_s32(c0_ptr + 8)); }
            vst1q_s32(c0_ptr + 8, vminq_s32(vmaxq_s32(v, vmin), vmax));
        } else {
            const int32x4_t acc[6] = {v_c00, v_c01, v_c02, v_c03, v_c04, v_c05};
            for (int j = 0; j < 6; ++j) {
                const int col0 = 2 * j;
                if (col0 >= rem) { break; }
                int32_t lane[4];
                vst1q_s32(lane, acc[j]);
                for (int h = 0; h < 2; ++h) {
                    if (col0 + h >= rem) { break; }
                    int32_t v = lane[h];
                    if constexpr (!zero_mode) { v += c0_ptr[col0 + h]; }
                    c0_ptr[col0 + h] = std::min(std::max(v, clamp_min), clamp_max);
                }
            }
        }

        c0_ptr += 12;
        B += ldb_panel;
    }
}

// Dispatch tables, consumed by matmul_helper.cpp; rows are the MR levels {8,4,1}.
template <bool Z>
inline constexpr auto mma_pack_fns_i8mm = std::array{
    &mma_pack_mr8_nr12_i8mm<Z>,
    &mma_pack_mr4_nr12_i8mm<Z>,
    &mma_pack_mr1_nr12_i8mm<Z>};

template <bool Z>
inline constexpr auto mma_direct_fns_i8mm = std::array{
    &mma_direct_mr8_nr12_i8mm<Z>,
    &mma_direct_mr4_nr12_i8mm<Z>,
    &mma_direct_mr1_nr12_i8mm<Z>};

}  // namespace nnops::backend::cpu::aarch64
