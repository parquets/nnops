#pragma once
/// @file pack_i8mm.hpp
/// @brief AArch64 NEON int8 pack kernels for the FEAT_I8MM (SMMLA) route.
///
/// SMMLA multiplies a 2x8 A fragment by an 8x2 B fragment, so both operands
/// want eight k-values per group — twice the SDOT pack's four, and arranged so
/// that each fragment is one contiguous 16-byte run:
///
///   - A: [ceil(K/8)][rows][8] bytes — a group holds the rows' eight k values,
///     row after row, so a two-row fragment (rows r, r+1) is one load.
///   - B: [ceil(K/8)][12 columns][8 k] bytes — inside a group each column's
///     eight k values are contiguous, which is the 8x2 operand (two
///     neighbouring columns per 16-byte load).
///
/// Twelve columns is the same panel width the SDOT kernels use, so every plan,
/// panel count and blob geometry above this file is unchanged; only the panel
/// *stride* rounds k up to eight instead of four (int8_panel_stride in
/// matmul_helper.h), for both operands.
///
/// Two families, mirroring pack_dp4a_i8.hpp:
///   - pack_trans_nN (LHS/A): reads N rows (stride ir_step) of a row-major
///     source and writes each row's eight k values per group.
///   - pack_copy_nN (RHS/B): rows are contiguous — every k row's N values are
///     gathered into the group's per-row (per-column for B) run.
///
/// Columns past `nr` and k past `K` are zero-filled, the same contract the
/// SDOT packers present, so a short panel or a short final k group is read as a
/// whole and the kernel's tail paths never mask.
///
/// Pure data movement: `scale` exists for signature parity with the other
/// packs. Two k groups are handled per iteration wherever a 16-byte load feeds
/// two 16-byte groups.

#include <algorithm>
#include <arm_neon.h>
#include <cstdint>
#include <cstring>
#include "backend/cpu/common/common_utils.hpp"

namespace nnops::backend::cpu::aarch64 {

/// k-values per SMMLA group.
inline constexpr int i8mm_k_group = 8;
/// B panel width, matched to the SDOT panel so no plan above changes.
inline constexpr int i8mm_nr = 12;
/// Bytes one k group of a full panel carries.
inline constexpr int i8mm_group_bytes = i8mm_nr * i8mm_k_group;

/// Two rows' low eight bytes into one 16-byte group fragment: [a][b].
inline int8x16_t i8mm_pair_lo(int8x16_t a, int8x16_t b) noexcept {
    return vcombine_s8(vget_low_s8(a), vget_low_s8(b));
}

/// The same for the high halves: the next k group's eight bytes.
inline int8x16_t i8mm_pair_hi(int8x16_t a, int8x16_t b) noexcept {
    return vcombine_s8(vget_high_s8(a), vget_high_s8(b));
}

inline void i8mm_copy8(int8_t* dst, const int8_t* src) noexcept {
    std::memcpy(dst, src, 8);
}

inline void i8mm_row8(int8_t* dst, const int8_t* src, int live) noexcept {
    if (live >= i8mm_k_group) {
        i8mm_copy8(dst, src);
    } else {
        std::memcpy(dst, src, static_cast<size_t>(live));
        std::memset(dst + live, 0, static_cast<size_t>(i8mm_k_group - live));
    }
}

// =========================================================================
//  LHS packs (A)  —  [ceil(K/8)][rows][8] bytes
// =========================================================================

inline void pack_trans_n1_i8mm(void* NNOPS_RESTRICT output,
                               const void* NNOPS_RESTRICT input,
                               int /*ir_step*/, int K, float scale) noexcept {
    (void)scale;
    int8_t* NNOPS_RESTRICT out = static_cast<int8_t*>(output);
    // One row: the packed groups are the source row, padded to whole groups.
    std::memcpy(out, input, static_cast<size_t>(K));
    std::memset(out + K, 0,
                static_cast<size_t>((((K + 7) >> 3) * i8mm_k_group) - K));
}

inline void pack_trans_n4_i8mm(void* NNOPS_RESTRICT output,
                               const void* NNOPS_RESTRICT input,
                               int ir_step, int K, float scale) noexcept {
    (void)scale;
    int8_t* NNOPS_RESTRICT out = static_cast<int8_t*>(output);
    const int8_t* NNOPS_RESTRICT input_i8 = static_cast<const int8_t*>(input);

    const int8_t* NNOPS_RESTRICT p0 = input_i8 + 0 * ir_step;
    const int8_t* NNOPS_RESTRICT p1 = input_i8 + 1 * ir_step;
    const int8_t* NNOPS_RESTRICT p2 = input_i8 + 2 * ir_step;
    const int8_t* NNOPS_RESTRICT p3 = input_i8 + 3 * ir_step;

    const int kgroups = (K + 7) >> 3;
    int g = 0;
    for (; g + 2 <= kgroups && (g + 2) * i8mm_k_group <= K; g += 2) {
        const int8x16_t r0 = vld1q_s8(p0);
        const int8x16_t r1 = vld1q_s8(p1);
        const int8x16_t r2 = vld1q_s8(p2);
        const int8x16_t r3 = vld1q_s8(p3);

        vst1q_s8(out + 0 * 16, i8mm_pair_lo(r0, r1));
        vst1q_s8(out + 1 * 16, i8mm_pair_lo(r2, r3));
        vst1q_s8(out + 2 * 16, i8mm_pair_hi(r0, r1));
        vst1q_s8(out + 3 * 16, i8mm_pair_hi(r2, r3));

        p0 += 16; p1 += 16; p2 += 16; p3 += 16;
        out += 4 * 16;
    }
    for (; g < kgroups; ++g) {
        const int live = std::min(K - g * i8mm_k_group, i8mm_k_group);
        i8mm_row8(out + 0 * 8, p0, live);
        i8mm_row8(out + 1 * 8, p1, live);
        i8mm_row8(out + 2 * 8, p2, live);
        i8mm_row8(out + 3 * 8, p3, live);
        p0 += live; p1 += live; p2 += live; p3 += live;
        out += 4 * 8;
    }
}

inline void pack_trans_n8_i8mm(void* NNOPS_RESTRICT output,
                               const void* NNOPS_RESTRICT input,
                               int ir_step, int K, float scale) noexcept {
    (void)scale;
    int8_t* NNOPS_RESTRICT out = static_cast<int8_t*>(output);
    const int8_t* NNOPS_RESTRICT input_i8 = static_cast<const int8_t*>(input);

    const int8_t* NNOPS_RESTRICT p0 = input_i8 + 0 * ir_step;
    const int8_t* NNOPS_RESTRICT p1 = input_i8 + 1 * ir_step;
    const int8_t* NNOPS_RESTRICT p2 = input_i8 + 2 * ir_step;
    const int8_t* NNOPS_RESTRICT p3 = input_i8 + 3 * ir_step;
    const int8_t* NNOPS_RESTRICT p4 = input_i8 + 4 * ir_step;
    const int8_t* NNOPS_RESTRICT p5 = input_i8 + 5 * ir_step;
    const int8_t* NNOPS_RESTRICT p6 = input_i8 + 6 * ir_step;
    const int8_t* NNOPS_RESTRICT p7 = input_i8 + 7 * ir_step;

    const int kgroups = (K + 7) >> 3;
    int g = 0;
    for (; g + 2 <= kgroups && (g + 2) * i8mm_k_group <= K; g += 2) {
        const int8x16_t r0 = vld1q_s8(p0);
        const int8x16_t r1 = vld1q_s8(p1);
        const int8x16_t r2 = vld1q_s8(p2);
        const int8x16_t r3 = vld1q_s8(p3);
        const int8x16_t r4 = vld1q_s8(p4);
        const int8x16_t r5 = vld1q_s8(p5);
        const int8x16_t r6 = vld1q_s8(p6);
        const int8x16_t r7 = vld1q_s8(p7);

        vst1q_s8(out + 0 * 16, i8mm_pair_lo(r0, r1));
        vst1q_s8(out + 1 * 16, i8mm_pair_lo(r2, r3));
        vst1q_s8(out + 2 * 16, i8mm_pair_lo(r4, r5));
        vst1q_s8(out + 3 * 16, i8mm_pair_lo(r6, r7));
        vst1q_s8(out + 4 * 16, i8mm_pair_hi(r0, r1));
        vst1q_s8(out + 5 * 16, i8mm_pair_hi(r2, r3));
        vst1q_s8(out + 6 * 16, i8mm_pair_hi(r4, r5));
        vst1q_s8(out + 7 * 16, i8mm_pair_hi(r6, r7));

        p0 += 16; p1 += 16; p2 += 16; p3 += 16;
        p4 += 16; p5 += 16; p6 += 16; p7 += 16;
        out += 8 * 16;
    }
    for (; g < kgroups; ++g) {
        const int live = std::min(K - g * i8mm_k_group, i8mm_k_group);
        i8mm_row8(out + 0 * 8, p0, live);
        i8mm_row8(out + 1 * 8, p1, live);
        i8mm_row8(out + 2 * 8, p2, live);
        i8mm_row8(out + 3 * 8, p3, live);
        i8mm_row8(out + 4 * 8, p4, live);
        i8mm_row8(out + 5 * 8, p5, live);
        i8mm_row8(out + 6 * 8, p6, live);
        i8mm_row8(out + 7 * 8, p7, live);
        p0 += live; p1 += live; p2 += live; p3 += live;
        p4 += live; p5 += live; p6 += live; p7 += live;
        out += 8 * 8;
    }
}

/// The transposed-source LHS variants: A is [K][rows].
template <int ROWS>
inline void pack_copy_nN_i8mm(void* NNOPS_RESTRICT output,
                              const void* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale) noexcept {
    (void)scale;
    int8_t* NNOPS_RESTRICT out = static_cast<int8_t*>(output);
    const int8_t* NNOPS_RESTRICT in = static_cast<const int8_t*>(input);
    const int kgroups = (K + 7) >> 3;

    for (int g = 0; g < kgroups; ++g) {
        const int k0 = g * i8mm_k_group;
        const int live = K - k0;
        int8_t* NNOPS_RESTRICT dst = out + g * (ROWS * i8mm_k_group);
        for (int r = 0; r < ROWS; ++r, dst += i8mm_k_group) {
            dst[0] = (0 < live) ? in[(k0 + 0) * ir_step + r] : 0;
            dst[1] = (1 < live) ? in[(k0 + 1) * ir_step + r] : 0;
            dst[2] = (2 < live) ? in[(k0 + 2) * ir_step + r] : 0;
            dst[3] = (3 < live) ? in[(k0 + 3) * ir_step + r] : 0;
            dst[4] = (4 < live) ? in[(k0 + 4) * ir_step + r] : 0;
            dst[5] = (5 < live) ? in[(k0 + 5) * ir_step + r] : 0;
            dst[6] = (6 < live) ? in[(k0 + 6) * ir_step + r] : 0;
            dst[7] = (7 < live) ? in[(k0 + 7) * ir_step + r] : 0;
        }
    }
}

inline void pack_copy_n8_i8mm(void* o, const void* i, int s, int K, float sc) noexcept {
    pack_copy_nN_i8mm<8>(o, i, s, K, sc);
}
inline void pack_copy_n4_i8mm(void* o, const void* i, int s, int K, float sc) noexcept {
    pack_copy_nN_i8mm<4>(o, i, s, K, sc);
}
inline void pack_copy_n1_i8mm(void* o, const void* i, int s, int K, float sc) noexcept {
    pack_copy_nN_i8mm<1>(o, i, s, K, sc);
}

// =========================================================================
//  RHS transpose pack (B stored [N][K])  —  8-int8 run per column per group
// =========================================================================

inline void pack_trans_n12_i8mm(void* NNOPS_RESTRICT output,
                                const void* NNOPS_RESTRICT input,
                                int ir_step, int K, float scale,
                                int nr = 12) noexcept {
    (void)scale;
    int8_t* NNOPS_RESTRICT out = static_cast<int8_t*>(output);
    const int8_t* NNOPS_RESTRICT input_i8 = static_cast<const int8_t*>(input);

    const int8_t* NNOPS_RESTRICT p0 = input_i8 + 0 * ir_step;
    const int8_t* NNOPS_RESTRICT p1 = input_i8 + 1 * ir_step;
    const int8_t* NNOPS_RESTRICT p2 = input_i8 + 2 * ir_step;
    const int8_t* NNOPS_RESTRICT p3 = input_i8 + 3 * ir_step;
    const int8_t* NNOPS_RESTRICT p4 = input_i8 + 4 * ir_step;
    const int8_t* NNOPS_RESTRICT p5 = input_i8 + 5 * ir_step;
    const int8_t* NNOPS_RESTRICT p6 = input_i8 + 6 * ir_step;
    const int8_t* NNOPS_RESTRICT p7 = input_i8 + 7 * ir_step;
    const int8_t* NNOPS_RESTRICT p8 = input_i8 + 8 * ir_step;
    const int8_t* NNOPS_RESTRICT p9 = input_i8 + 9 * ir_step;
    const int8_t* NNOPS_RESTRICT pa = input_i8 + 10 * ir_step;
    const int8_t* NNOPS_RESTRICT pb = input_i8 + 11 * ir_step;

    const int kgroups = (K + 7) >> 3;
    int g = 0;
    if (nr == i8mm_nr) {
        // Column pairs, two groups at a time.
        for (; g + 2 <= kgroups && (g + 2) * i8mm_k_group <= K; g += 2) {
            const int8x16_t r0 = vld1q_s8(p0);
            const int8x16_t r1 = vld1q_s8(p1);
            const int8x16_t r2 = vld1q_s8(p2);
            const int8x16_t r3 = vld1q_s8(p3);
            const int8x16_t r4 = vld1q_s8(p4);
            const int8x16_t r5 = vld1q_s8(p5);
            const int8x16_t r6 = vld1q_s8(p6);
            const int8x16_t r7 = vld1q_s8(p7);
            const int8x16_t r8 = vld1q_s8(p8);
            const int8x16_t r9 = vld1q_s8(p9);
            const int8x16_t ra = vld1q_s8(pa);
            const int8x16_t rb = vld1q_s8(pb);

            vst1q_s8(out + 0 * 16, i8mm_pair_lo(r0, r1));
            vst1q_s8(out + 1 * 16, i8mm_pair_lo(r2, r3));
            vst1q_s8(out + 2 * 16, i8mm_pair_lo(r4, r5));
            vst1q_s8(out + 3 * 16, i8mm_pair_lo(r6, r7));
            vst1q_s8(out + 4 * 16, i8mm_pair_lo(r8, r9));
            vst1q_s8(out + 5 * 16, i8mm_pair_lo(ra, rb));

            vst1q_s8(out + 6 * 16, i8mm_pair_hi(r0, r1));
            vst1q_s8(out + 7 * 16, i8mm_pair_hi(r2, r3));
            vst1q_s8(out + 8 * 16, i8mm_pair_hi(r4, r5));
            vst1q_s8(out + 9 * 16, i8mm_pair_hi(r6, r7));
            vst1q_s8(out + 10 * 16, i8mm_pair_hi(r8, r9));
            vst1q_s8(out + 11 * 16, i8mm_pair_hi(ra, rb));

            p0 += 16; p1 += 16; p2 += 16; p3 += 16;
            p4 += 16; p5 += 16; p6 += 16; p7 += 16;
            p8 += 16; p9 += 16; pa += 16; pb += 16;
            out += 2 * i8mm_group_bytes;
        }
    }
    for (; g < kgroups; ++g) {
        const int live = std::min(K - g * i8mm_k_group, i8mm_k_group);
        int8_t* NNOPS_RESTRICT dst = out;
        if (nr == i8mm_nr) {
            i8mm_row8(dst + 0 * 8, p0, live);
            i8mm_row8(dst + 1 * 8, p1, live);
            i8mm_row8(dst + 2 * 8, p2, live);
            i8mm_row8(dst + 3 * 8, p3, live);
            i8mm_row8(dst + 4 * 8, p4, live);
            i8mm_row8(dst + 5 * 8, p5, live);
            i8mm_row8(dst + 6 * 8, p6, live);
            i8mm_row8(dst + 7 * 8, p7, live);
            i8mm_row8(dst + 8 * 8, p8, live);
            i8mm_row8(dst + 9 * 8, p9, live);
            i8mm_row8(dst + 10 * 8, pa, live);
            i8mm_row8(dst + 11 * 8, pb, live);
        } else {
            // Short panel: the columns past `nr` were never loaded.
            i8mm_row8(dst + 0 * 8, p0, nr > 0 ? live : 0);
            if (nr > 1)  { i8mm_row8(dst + 1 * 8, p1, live); }
            if (nr > 2)  { i8mm_row8(dst + 2 * 8, p2, live); }
            if (nr > 3)  { i8mm_row8(dst + 3 * 8, p3, live); }
            if (nr > 4)  { i8mm_row8(dst + 4 * 8, p4, live); }
            if (nr > 5)  { i8mm_row8(dst + 5 * 8, p5, live); }
            if (nr > 6)  { i8mm_row8(dst + 6 * 8, p6, live); }
            if (nr > 7)  { i8mm_row8(dst + 7 * 8, p7, live); }
            if (nr > 8)  { i8mm_row8(dst + 8 * 8, p8, live); }
            if (nr > 9)  { i8mm_row8(dst + 9 * 8, p9, live); }
            if (nr > 10) { i8mm_row8(dst + 10 * 8, pa, live); }
            if (nr > 11) { i8mm_row8(dst + 11 * 8, pb, live); }
            for (int c = nr; c < i8mm_nr; ++c) {
                std::memset(dst + c * 8, 0, 8);
            }
        }
        p0 += live; p1 += live; p2 += live; p3 += live;
        p4 += live; p5 += live; p6 += live; p7 += live;
        p8 += live; p9 += live; pa += live; pb += live;
        out += i8mm_group_bytes;
    }
}

// =========================================================================
//  RHS copy pack (B stored [K][N])  —  two 4-row interleaves per group
// =========================================================================

inline void pack_copy_n12_i8mm(void* NNOPS_RESTRICT output,
                               const void* NNOPS_RESTRICT input,
                               int ir_step, int K, float scale,
                               int nr = 12) noexcept {
    (void)scale;
    int8_t* NNOPS_RESTRICT out = static_cast<int8_t*>(output);
    const int8_t* NNOPS_RESTRICT input_i8 = static_cast<const int8_t*>(input);
    const int kgroups = (K + 7) >> 3;

    int g = 0;
    if (nr == i8mm_nr) {
        // Two 4-row passes per group, each an 8-column vst4 interleave plus
        // four scalar columns; then the halves are zipped into 8 bytes per
        // column.
        for (; g + 1 <= kgroups && (g + 1) * i8mm_k_group <= K; ++g) {
            const int8_t* NNOPS_RESTRICT r = input_i8 + g * i8mm_k_group * ir_step;
            alignas(16) int8_t half[2][i8mm_group_bytes / 2];

            const int8x8x4_t lo0 = {vld1_s8(r + 0 * ir_step),
                                    vld1_s8(r + 1 * ir_step),
                                    vld1_s8(r + 2 * ir_step),
                                    vld1_s8(r + 3 * ir_step)};
            const int8x8x4_t lo1 = {vld1_s8(r + 4 * ir_step),
                                    vld1_s8(r + 5 * ir_step),
                                    vld1_s8(r + 6 * ir_step),
                                    vld1_s8(r + 7 * ir_step)};
            vst4_s8(half[0], lo0);
            vst4_s8(half[1], lo1);

            // Columns 8..11, both 4-k halves, straight from the source rows.
            int8_t* NNOPS_RESTRICT t0 = half[0] + 32;
            int8_t* NNOPS_RESTRICT t1 = half[1] + 32;
            for (int x = 0; x < 4; ++x, t0 += 4, t1 += 4) {
                t0[0] = r[0 * ir_step + 8 + x];
                t0[1] = r[1 * ir_step + 8 + x];
                t0[2] = r[2 * ir_step + 8 + x];
                t0[3] = r[3 * ir_step + 8 + x];
                t1[0] = r[4 * ir_step + 8 + x];
                t1[1] = r[5 * ir_step + 8 + x];
                t1[2] = r[6 * ir_step + 8 + x];
                t1[3] = r[7 * ir_step + 8 + x];
            }

            const int32x4_t a0 = vld1q_s32(reinterpret_cast<const int32_t*>(half[0] + 0 * 4));
            const int32x4_t b0 = vld1q_s32(reinterpret_cast<const int32_t*>(half[1] + 0 * 4));
            const int32x4_t a1 = vld1q_s32(reinterpret_cast<const int32_t*>(half[0] + 4 * 4));
            const int32x4_t b1 = vld1q_s32(reinterpret_cast<const int32_t*>(half[1] + 4 * 4));
            const int32x4_t a2 = vld1q_s32(reinterpret_cast<const int32_t*>(half[0] + 8 * 4));
            const int32x4_t b2 = vld1q_s32(reinterpret_cast<const int32_t*>(half[1] + 8 * 4));

            vst1q_s32(reinterpret_cast<int32_t*>(out + 0 * 16), vzip1q_s32(a0, b0));
            vst1q_s32(reinterpret_cast<int32_t*>(out + 1 * 16), vzip2q_s32(a0, b0));
            vst1q_s32(reinterpret_cast<int32_t*>(out + 2 * 16), vzip1q_s32(a1, b1));
            vst1q_s32(reinterpret_cast<int32_t*>(out + 3 * 16), vzip2q_s32(a1, b1));
            vst1q_s32(reinterpret_cast<int32_t*>(out + 4 * 16), vzip1q_s32(a2, b2));
            vst1q_s32(reinterpret_cast<int32_t*>(out + 5 * 16), vzip2q_s32(a2, b2));

            out += i8mm_group_bytes;
        }
    }

    for (; g < kgroups; ++g) {
        const int8_t* NNOPS_RESTRICT r = input_i8 + g * i8mm_k_group * ir_step;
        const int live = std::min(K - g * i8mm_k_group, i8mm_k_group);
        int8_t* NNOPS_RESTRICT dst = out;
        for (int c = 0; c < i8mm_nr; ++c, dst += i8mm_k_group) {
            if (c >= nr) {
                std::memset(dst, 0, 8);
                continue;
            }
            if (live >= i8mm_k_group) {
                dst[0] = r[0 * ir_step + c];
                dst[1] = r[1 * ir_step + c];
                dst[2] = r[2 * ir_step + c];
                dst[3] = r[3 * ir_step + c];
                dst[4] = r[4 * ir_step + c];
                dst[5] = r[5 * ir_step + c];
                dst[6] = r[6 * ir_step + c];
                dst[7] = r[7 * ir_step + c];
            } else {
                dst[0] = (0 < live) ? r[0 * ir_step + c] : 0;
                dst[1] = (1 < live) ? r[1 * ir_step + c] : 0;
                dst[2] = (2 < live) ? r[2 * ir_step + c] : 0;
                dst[3] = (3 < live) ? r[3 * ir_step + c] : 0;
                dst[4] = (4 < live) ? r[4 * ir_step + c] : 0;
                dst[5] = (5 < live) ? r[5 * ir_step + c] : 0;
                dst[6] = (6 < live) ? r[6 * ir_step + c] : 0;
                dst[7] = (7 < live) ? r[7 * ir_step + c] : 0;
            }
        }
        out += i8mm_group_bytes;
    }
}

}  // namespace nnops::backend::cpu::aarch64
