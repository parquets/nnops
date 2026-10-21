#pragma once
/// @file mma_i8.hpp
/// @brief x86_64 VNNI int8×int8 → int32 mrxncxkc MMA (matrix micro-accumulate)
///        kernels.
///
/// One call covers the whole N side of one M panel: it computes
/// C[mr][nc] += A[mr][kc] × B[kc][nc] panel by panel and clamps every int32
/// result to [clamp_min, clamp_max]. This is the int8 sibling of
/// mma_f32.hpp / mma_f16.hpp, and takes the same runtime panel height `mr` and
/// the same partial-M convention: rows past `mr` reuse the last live row's
/// index and C row (they compute a duplicate and store it back to the same
/// place), so a short panel needs no per-row guards.
///
/// Both operand routes exist, as in the float files:
///   - `mma_packed_ab_*` takes a packed A: four consecutive k per int32, with
///     the kernel's MR row slots interleaved inside each k group --
///     [ceil(kc/4)][MR] int32s, in a mma_pack_align(MR*kbytes)-byte slot; only
///     the first `mr` slots of a short panel are read.
///   - `mma_packed_b_*` reads A row-major through `lda` (in int8 elements). The
///     four consecutive k of one row are already the dword dpbusd wants, so no
///     packing is needed -- but a row-major row has no zero-filled final group,
///     so these require kc % 4 == 0.
///   B is always packed, and always in VNNI order:
///   - ceil(nc/16) uniform 16-column panels, each byte-group
///     [ceil(kc/4)][16][4] -- one int32 per column per k group -- followed by
///     16 int32 of column sums. The data half is 16*ceil(kc/4)*4 =
///     64*ceil(kc/4) bytes, itself a multiple of the 64-byte mma_pack_align
///     slot, so the sums block starts at panel + 64*ceil(kc/4) and the panel's
///     slot is mma_pack_align(64*ceil(kc/4) + 64) (see mma_i8_b_panel_tail).
///
/// Panels past `nc` in the last one are zero-filled by the packer, so the tail
/// can always read all 16 columns at the uniform stride and discard the dead
/// ones. `kc` is the number of int8 elements; on the packed route a final
/// partial group (1..3) is consumed as a whole zero-padded group, matching what
/// the packers write.
///
/// Operand contract: B is int8; A is int8 or uint8, and the kernels return the
/// exact integer dot product either way (what MatMulInteger wants, before
/// zero-point compensation). vpdpbusd is u8×s8, so:
///   - s8 A: the A pack stores every byte biased by +128 (XOR 0x80) and the
///     kernel folds -128·Σ_k B[k][n] back out.
///   - u8 A: the A pack stores the bytes as they are; nothing to fold, so the
///     B panel's correction block is zeroed and the accumulators start at zero.
/// That sum is a property of the B panel, not of A, so the B packer writes it
/// once per panel as 16 int32 right after the panel's data (see
/// mma_i8_b_panel_sums). The kernel starts its accumulators at -128·sum instead
/// of zero, which costs nothing in the k loop. The direct-A route has no packer
/// to bias it, so each broadcast group is XORed in the kernel -- i.e. that route
/// is s8 A only.
///
/// Tile sizes (x86_64-optimised): M ∈ {6, 4, 1}    N: 16 plus an in-kernel tail.

#include <array>
#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include "backend/cpu/common/common_utils.hpp"

namespace nnops::backend::cpu::x86_64 {

// vpdpbusd: acc + sum of 4 u8×s8 products per 32-bit lane. The build always
// defines __AVXVNNI__ on x86_64 (MSVC has no /arch switch for it, so CMake adds
// the macro by hand); a build without VNNI cannot run these kernels at all, so
// fail loudly instead of quietly accumulating nothing.
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
inline __m256i mma_i8_dp256(__m256i acc, __m256i a, __m256i b) noexcept {
    return _mm256_dpbusd_epi32(acc, a, b);
}
#elif defined(__AVXVNNI__)
inline __m256i mma_i8_dp256(__m256i acc, __m256i a, __m256i b) noexcept {
    return _mm256_dpbusd_avx_epi32(acc, a, b);
}
#else
#error "mma_i8.hpp requires AVX-VNNI (vpdpbusd)"
#endif

/// One k group of a row-major A row as a VNNI operand: the four int8 values
/// plus the +128 bias that turns the signed row into vpdpbusd's unsigned
/// operand. The 4-byte load is unaligned whenever `lda` is not a multiple of 4;
/// x86 has no penalty for that, and this is the same int32 read the dp4a pack
/// kernels use.
inline __m256i mma_i8_bcast_group(const int8_t* p) noexcept {
    return _mm256_set1_epi32(*reinterpret_cast<const int32_t*>(p) ^ 0x80808080);
}

/// Packed-B panel width: the kernels' uniform nr.
inline constexpr int mma_i8_nr_max = 16;

/// Bytes a packed-B panel carries past its nr*kbytes payload: one int32 of
/// column sums per column.
inline constexpr int mma_i8_b_panel_tail = mma_i8_nr_max * 4;

/// The kernels seed their accumulators from the panel's trailing block, i.e.
/// they fold the A bias per panel. matmul_helper uses this to decide how much
/// of a u8 A's -128 shift the epilogue still has to correct (nothing here).
inline constexpr bool mma_i8_folds_a_bias = true;

/// Column sums of one packed-B panel, written by the pack step. Every 32-bit
/// lane of the panel holds four consecutive k of one column, so accumulating
/// it against an all-ones u8 vector gives exactly that column's k sum.
///
/// @p a_biased is the A pack's +128 shift for this GEMM: when it is set the
/// block holds the sums (the kernel turns them into -128·Σb), otherwise it is
/// zeroed and the accumulators start at zero, which is what a u8 A wants.
inline void mma_i8_b_panel_sums(int8_t* panel, int kbytes, bool a_biased) noexcept {
    __m256i s0 = _mm256_setzero_si256();
    __m256i s1 = _mm256_setzero_si256();
    if (a_biased) {
        const __m256i ones = _mm256_set1_epi8(1);
        const int8_t* p = panel;
        for (int g = kbytes >> 2; g > 0; --g, p += 64) {
            s0 = mma_i8_dp256(s0, ones, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
            s1 = mma_i8_dp256(s1, ones, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32)));
        }
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(panel + 16 * kbytes), s0);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(panel + 16 * kbytes + 32), s1);
}

/// An accumulator's starting value for one packed-B panel: -128 times the
/// panel's column sums, i.e. the correction that cancels the +128 A bias over
/// the k loop. For a u8 A the block is zeroed and this is the plain zero the
/// accumulators used to start at. `off` picks the low (0) or high (32) half of
/// the 16 columns.
inline __m256i mma_i8_bias_fix(const int8_t* panel, int kbytes, int off) noexcept {
    const __m256i sums = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(panel + 16 * kbytes + off));
    return _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_slli_epi32(sums, 7));
}

// =========================================================================
//  mr = 6
// =========================================================================

template <bool zero_mode = false>
void mma_packed_ab_mr6_nr16_mrxncxkc_i8(
    int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
    const int8_t* NNOPS_RESTRICT A,
    const int8_t* NNOPS_RESTRICT B,
    int32_t clamp_min, int32_t clamp_max) noexcept {

    // Rows past `mr` alias the last live row in both A and C: they compute a
    // duplicate of it and store it back to the same place, so a partial-M tile
    // needs no per-row guards (the XNNPACK 6x16 broadcast convention).
    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;
    const int r4 = (mr > 4) ? 4 : r3;
    const int r5 = (mr > 5) ? 5 : r4;

    int32_t* c0_ptr = C + 0 * ldc;
    int32_t* c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    int32_t* c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    int32_t* c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;
    int32_t* c4_ptr = (mr > 4) ? C + 4 * ldc : c3_ptr;
    int32_t* c5_ptr = (mr > 5) ? C + 5 * ldc : c4_ptr;

    const int kbytes = (kc + 3) & ~3;
    const int kg = kbytes >> 2;
    // 16 columns x kbytes bytes per panel, i.e. 64 bytes per k group -- already
    // a whole mma_pack_align slot, so no padding between panels.
    // Panel slot: 16 columns x kbytes of data, then the 16 int32 of column
    // sums the accumulator is seeded from.
    const int ldb = static_cast<int>(mma_pack_align(16 * kbytes + mma_i8_b_panel_tail));

    int n = 0;
    for(; n < nc - 15; n += 16) {
        __m256i v_acc0x01234567 = mma_i8_bias_fix(B, kbytes, 0);
        __m256i v_acc0x89abcdef = mma_i8_bias_fix(B, kbytes, 32);
        __m256i v_acc1x01234567 = v_acc0x01234567;
        __m256i v_acc1x89abcdef = v_acc0x89abcdef;
        __m256i v_acc2x01234567 = v_acc0x01234567;
        __m256i v_acc2x89abcdef = v_acc0x89abcdef;
        __m256i v_acc3x01234567 = v_acc0x01234567;
        __m256i v_acc3x89abcdef = v_acc0x89abcdef;
        __m256i v_acc4x01234567 = v_acc0x01234567;
        __m256i v_acc4x89abcdef = v_acc0x89abcdef;
        __m256i v_acc5x01234567 = v_acc0x01234567;
        __m256i v_acc5x89abcdef = v_acc0x89abcdef;

        const int32_t* a_i32 = reinterpret_cast<const int32_t*>(A);
        const int8_t* b_ptr = B;
        for(int k = 0; k < kg; ++k) {
            const __m256i v_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
            const __m256i v_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 32));
            b_ptr += 64;

            __m256i v_a0 = _mm256_set1_epi32(a_i32[0]);
            __m256i v_a1 = _mm256_set1_epi32(a_i32[r1]);
            v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567, v_a0, v_b0);
            v_acc0x89abcdef = mma_i8_dp256(v_acc0x89abcdef, v_a0, v_b1);
            v_acc1x01234567 = mma_i8_dp256(v_acc1x01234567, v_a1, v_b0);
            v_acc1x89abcdef = mma_i8_dp256(v_acc1x89abcdef, v_a1, v_b1);

            v_a0 = _mm256_set1_epi32(a_i32[r2]);
            v_a1 = _mm256_set1_epi32(a_i32[r3]);
            v_acc2x01234567 = mma_i8_dp256(v_acc2x01234567, v_a0, v_b0);
            v_acc2x89abcdef = mma_i8_dp256(v_acc2x89abcdef, v_a0, v_b1);
            v_acc3x01234567 = mma_i8_dp256(v_acc3x01234567, v_a1, v_b0);
            v_acc3x89abcdef = mma_i8_dp256(v_acc3x89abcdef, v_a1, v_b1);

            v_a0 = _mm256_set1_epi32(a_i32[r4]);
            v_a1 = _mm256_set1_epi32(a_i32[r5]);
            v_acc4x01234567 = mma_i8_dp256(v_acc4x01234567, v_a0, v_b0);
            v_acc4x89abcdef = mma_i8_dp256(v_acc4x89abcdef, v_a0, v_b1);
            v_acc5x01234567 = mma_i8_dp256(v_acc5x01234567, v_a1, v_b0);
            v_acc5x89abcdef = mma_i8_dp256(v_acc5x89abcdef, v_a1, v_b1);

            a_i32 += 6;
        }
        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_epi32(v_acc0x01234567, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c0_ptr + 0)));
            v_acc0x89abcdef = _mm256_add_epi32(v_acc0x89abcdef, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c0_ptr + 8)));
            v_acc1x01234567 = _mm256_add_epi32(v_acc1x01234567, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c1_ptr + 0)));
            v_acc1x89abcdef = _mm256_add_epi32(v_acc1x89abcdef, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c1_ptr + 8)));
            v_acc2x01234567 = _mm256_add_epi32(v_acc2x01234567, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c2_ptr + 0)));
            v_acc2x89abcdef = _mm256_add_epi32(v_acc2x89abcdef, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c2_ptr + 8)));
            v_acc3x01234567 = _mm256_add_epi32(v_acc3x01234567, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c3_ptr + 0)));
            v_acc3x89abcdef = _mm256_add_epi32(v_acc3x89abcdef, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c3_ptr + 8)));
            v_acc4x01234567 = _mm256_add_epi32(v_acc4x01234567, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c4_ptr + 0)));
            v_acc4x89abcdef = _mm256_add_epi32(v_acc4x89abcdef, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c4_ptr + 8)));
            v_acc5x01234567 = _mm256_add_epi32(v_acc5x01234567, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c5_ptr + 0)));
            v_acc5x89abcdef = _mm256_add_epi32(v_acc5x89abcdef, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c5_ptr + 8)));
        }
        const __m256i v_min = _mm256_set1_epi32(clamp_min);
        const __m256i v_max = _mm256_set1_epi32(clamp_max);

        v_acc0x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc0x01234567, v_min), v_max);
        v_acc0x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc0x89abcdef, v_min), v_max);
        v_acc1x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc1x01234567, v_min), v_max);
        v_acc1x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc1x89abcdef, v_min), v_max);
        v_acc2x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc2x01234567, v_min), v_max);
        v_acc2x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc2x89abcdef, v_min), v_max);
        v_acc3x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc3x01234567, v_min), v_max);
        v_acc3x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc3x89abcdef, v_min), v_max);
        v_acc4x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc4x01234567, v_min), v_max);
        v_acc4x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc4x89abcdef, v_min), v_max);
        v_acc5x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc5x01234567, v_min), v_max);
        v_acc5x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc5x89abcdef, v_min), v_max);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c0_ptr + 0), v_acc0x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c0_ptr + 8), v_acc0x89abcdef);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c1_ptr + 0), v_acc1x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c1_ptr + 8), v_acc1x89abcdef);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c2_ptr + 0), v_acc2x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c2_ptr + 8), v_acc2x89abcdef);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c3_ptr + 0), v_acc3x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c3_ptr + 8), v_acc3x89abcdef);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c4_ptr + 0), v_acc4x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c4_ptr + 8), v_acc4x89abcdef);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c5_ptr + 0), v_acc5x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c5_ptr + 8), v_acc5x89abcdef);

        c0_ptr += 16;
        c1_ptr += 16;
        c2_ptr += 16;
        c3_ptr += 16;
        c4_ptr += 16;
        c5_ptr += 16;

        B += ldb;
    }

    int rem = nc - n;
    const bool tail_hi = rem > 8;  // the tail k loop filled both halves

    if (rem > 0) {
        // The tail is one full 16-wide panel in the packed layout -- the packer
        // zero-fills the columns past `nc`, so both halves can always be read at
        // the same 64-byte stride. At most 8 columns fit one 8-lane read; a
        // wider tail reads both halves and reuses the full-panel body, so it
        // costs one k loop instead of two.
        __m256i v_acc0x01234567 = mma_i8_bias_fix(B, kbytes, 0);
        __m256i v_acc0x89abcdef = mma_i8_bias_fix(B, kbytes, 32);
        __m256i v_acc1x01234567 = v_acc0x01234567;
        __m256i v_acc1x89abcdef = v_acc0x89abcdef;
        __m256i v_acc2x01234567 = v_acc0x01234567;
        __m256i v_acc2x89abcdef = v_acc0x89abcdef;
        __m256i v_acc3x01234567 = v_acc0x01234567;
        __m256i v_acc3x89abcdef = v_acc0x89abcdef;
        __m256i v_acc4x01234567 = v_acc0x01234567;
        __m256i v_acc4x89abcdef = v_acc0x89abcdef;
        __m256i v_acc5x01234567 = v_acc0x01234567;
        __m256i v_acc5x89abcdef = v_acc0x89abcdef;

        const int32_t* a_i32 = reinterpret_cast<const int32_t*>(A);
        const int8_t* b_ptr = B;
        if (rem > 8) {
            for(int k = 0; k < kg; ++k) {
                const __m256i v_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
                const __m256i v_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 32));
                b_ptr += 64;

                __m256i v_a0 = _mm256_set1_epi32(a_i32[0]);
                __m256i v_a1 = _mm256_set1_epi32(a_i32[r1]);
                v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567, v_a0, v_b0);
                v_acc0x89abcdef = mma_i8_dp256(v_acc0x89abcdef, v_a0, v_b1);
                v_acc1x01234567 = mma_i8_dp256(v_acc1x01234567, v_a1, v_b0);
                v_acc1x89abcdef = mma_i8_dp256(v_acc1x89abcdef, v_a1, v_b1);

                v_a0 = _mm256_set1_epi32(a_i32[r2]);
                v_a1 = _mm256_set1_epi32(a_i32[r3]);
                v_acc2x01234567 = mma_i8_dp256(v_acc2x01234567, v_a0, v_b0);
                v_acc2x89abcdef = mma_i8_dp256(v_acc2x89abcdef, v_a0, v_b1);
                v_acc3x01234567 = mma_i8_dp256(v_acc3x01234567, v_a1, v_b0);
                v_acc3x89abcdef = mma_i8_dp256(v_acc3x89abcdef, v_a1, v_b1);

                v_a0 = _mm256_set1_epi32(a_i32[r4]);
                v_a1 = _mm256_set1_epi32(a_i32[r5]);
                v_acc4x01234567 = mma_i8_dp256(v_acc4x01234567, v_a0, v_b0);
                v_acc4x89abcdef = mma_i8_dp256(v_acc4x89abcdef, v_a0, v_b1);
                v_acc5x01234567 = mma_i8_dp256(v_acc5x01234567, v_a1, v_b0);
                v_acc5x89abcdef = mma_i8_dp256(v_acc5x89abcdef, v_a1, v_b1);

                a_i32 += 6;
            }
        } else {
            for(int k = 0; k < kg; ++k) {
                const __m256i v_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
                b_ptr += 64;

                v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567, _mm256_set1_epi32(a_i32[0]), v_b0);
                v_acc1x01234567 = mma_i8_dp256(v_acc1x01234567, _mm256_set1_epi32(a_i32[r1]), v_b0);
                v_acc2x01234567 = mma_i8_dp256(v_acc2x01234567, _mm256_set1_epi32(a_i32[r2]), v_b0);
                v_acc3x01234567 = mma_i8_dp256(v_acc3x01234567, _mm256_set1_epi32(a_i32[r3]), v_b0);
                v_acc4x01234567 = mma_i8_dp256(v_acc4x01234567, _mm256_set1_epi32(a_i32[r4]), v_b0);
                v_acc5x01234567 = mma_i8_dp256(v_acc5x01234567, _mm256_set1_epi32(a_i32[r5]), v_b0);

                a_i32 += 6;
            }
        }

        if (rem >= 8) {

            if constexpr (!zero_mode) {
                v_acc0x01234567 = _mm256_add_epi32(v_acc0x01234567, _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(c0_ptr)));
                v_acc1x01234567 = _mm256_add_epi32(v_acc1x01234567, _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(c1_ptr)));
                v_acc2x01234567 = _mm256_add_epi32(v_acc2x01234567, _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(c2_ptr)));
                v_acc3x01234567 = _mm256_add_epi32(v_acc3x01234567, _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(c3_ptr)));
                v_acc4x01234567 = _mm256_add_epi32(v_acc4x01234567, _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(c4_ptr)));
                v_acc5x01234567 = _mm256_add_epi32(v_acc5x01234567, _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(c5_ptr)));
            }

            const __m256i v_min = _mm256_set1_epi32(clamp_min);
            const __m256i v_max = _mm256_set1_epi32(clamp_max);

            v_acc0x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc0x01234567, v_min), v_max);
            v_acc1x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc1x01234567, v_min), v_max);
            v_acc2x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc2x01234567, v_min), v_max);
            v_acc3x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc3x01234567, v_min), v_max);
            v_acc4x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc4x01234567, v_min), v_max);
            v_acc5x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc5x01234567, v_min), v_max);

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c0_ptr), v_acc0x01234567);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c1_ptr), v_acc1x01234567);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c2_ptr), v_acc2x01234567);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c3_ptr), v_acc3x01234567);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c4_ptr), v_acc4x01234567);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c5_ptr), v_acc5x01234567);

            c0_ptr += 8;
            c1_ptr += 8;
            c2_ptr += 8;
            c3_ptr += 8;
            c4_ptr += 8;
            c5_ptr += 8;

            rem -= 8;
        }

        if (rem > 0) {
            // Lane j holds column j, so the 4/2/1 ladder is just a lo/hi split of
            // the same accumulator: after the 4-wide step the leftover 1..3
            // columns sit at lane 0 of the high half. A tail wider than 8 kept
            // accumulating into the high half, so that is the one it starts from.
            const __m256i v_acc0 = tail_hi ? v_acc0x89abcdef : v_acc0x01234567;
            const __m256i v_acc1 = tail_hi ? v_acc1x89abcdef : v_acc1x01234567;
            const __m256i v_acc2 = tail_hi ? v_acc2x89abcdef : v_acc2x01234567;
            const __m256i v_acc3 = tail_hi ? v_acc3x89abcdef : v_acc3x01234567;
            const __m256i v_acc4 = tail_hi ? v_acc4x89abcdef : v_acc4x01234567;
            const __m256i v_acc5 = tail_hi ? v_acc5x89abcdef : v_acc5x01234567;

            __m128i v_acc0x0123 = _mm256_castsi256_si128(v_acc0);
            __m128i v_acc1x0123 = _mm256_castsi256_si128(v_acc1);
            __m128i v_acc2x0123 = _mm256_castsi256_si128(v_acc2);
            __m128i v_acc3x0123 = _mm256_castsi256_si128(v_acc3);
            __m128i v_acc4x0123 = _mm256_castsi256_si128(v_acc4);
            __m128i v_acc5x0123 = _mm256_castsi256_si128(v_acc5);

            const __m128i v_min = _mm_set1_epi32(clamp_min);
            const __m128i v_max = _mm_set1_epi32(clamp_max);

            if (rem & 4) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_epi32(v_acc0x0123, _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(c0_ptr)));
                    v_acc1x0123 = _mm_add_epi32(v_acc1x0123, _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(c1_ptr)));
                    v_acc2x0123 = _mm_add_epi32(v_acc2x0123, _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(c2_ptr)));
                    v_acc3x0123 = _mm_add_epi32(v_acc3x0123, _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(c3_ptr)));
                    v_acc4x0123 = _mm_add_epi32(v_acc4x0123, _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(c4_ptr)));
                    v_acc5x0123 = _mm_add_epi32(v_acc5x0123, _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(c5_ptr)));
                }

                v_acc0x0123 = _mm_min_epi32(_mm_max_epi32(v_acc0x0123, v_min), v_max);
                v_acc1x0123 = _mm_min_epi32(_mm_max_epi32(v_acc1x0123, v_min), v_max);
                v_acc2x0123 = _mm_min_epi32(_mm_max_epi32(v_acc2x0123, v_min), v_max);
                v_acc3x0123 = _mm_min_epi32(_mm_max_epi32(v_acc3x0123, v_min), v_max);
                v_acc4x0123 = _mm_min_epi32(_mm_max_epi32(v_acc4x0123, v_min), v_max);
                v_acc5x0123 = _mm_min_epi32(_mm_max_epi32(v_acc5x0123, v_min), v_max);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr), v_acc0x0123);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(c1_ptr), v_acc1x0123);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(c2_ptr), v_acc2x0123);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(c3_ptr), v_acc3x0123);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(c4_ptr), v_acc4x0123);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(c5_ptr), v_acc5x0123);

                c0_ptr += 4;
                c1_ptr += 4;
                c2_ptr += 4;
                c3_ptr += 4;
                c4_ptr += 4;
                c5_ptr += 4;

                v_acc0x0123 = _mm256_extracti128_si256(v_acc0, 1);
                v_acc1x0123 = _mm256_extracti128_si256(v_acc1, 1);
                v_acc2x0123 = _mm256_extracti128_si256(v_acc2, 1);
                v_acc3x0123 = _mm256_extracti128_si256(v_acc3, 1);
                v_acc4x0123 = _mm256_extracti128_si256(v_acc4, 1);
                v_acc5x0123 = _mm256_extracti128_si256(v_acc5, 1);
            }

            if (rem & 2) {
                // The clamp below is 4-wide, so keep the untouched leftovers for
                // the 1-wide step -- lanes 2,3 are what it will store.
                const __m128i v_rest0 = v_acc0x0123;
                const __m128i v_rest1 = v_acc1x0123;
                const __m128i v_rest2 = v_acc2x0123;
                const __m128i v_rest3 = v_acc3x0123;
                const __m128i v_rest4 = v_acc4x0123;
                const __m128i v_rest5 = v_acc5x0123;

                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_epi32(v_acc0x0123, _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(c0_ptr)));
                    v_acc1x0123 = _mm_add_epi32(v_acc1x0123, _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(c1_ptr)));
                    v_acc2x0123 = _mm_add_epi32(v_acc2x0123, _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(c2_ptr)));
                    v_acc3x0123 = _mm_add_epi32(v_acc3x0123, _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(c3_ptr)));
                    v_acc4x0123 = _mm_add_epi32(v_acc4x0123, _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(c4_ptr)));
                    v_acc5x0123 = _mm_add_epi32(v_acc5x0123, _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(c5_ptr)));
                }

                v_acc0x0123 = _mm_min_epi32(_mm_max_epi32(v_acc0x0123, v_min), v_max);
                v_acc1x0123 = _mm_min_epi32(_mm_max_epi32(v_acc1x0123, v_min), v_max);
                v_acc2x0123 = _mm_min_epi32(_mm_max_epi32(v_acc2x0123, v_min), v_max);
                v_acc3x0123 = _mm_min_epi32(_mm_max_epi32(v_acc3x0123, v_min), v_max);
                v_acc4x0123 = _mm_min_epi32(_mm_max_epi32(v_acc4x0123, v_min), v_max);
                v_acc5x0123 = _mm_min_epi32(_mm_max_epi32(v_acc5x0123, v_min), v_max);

                _mm_storel_epi64(reinterpret_cast<__m128i*>(c0_ptr), v_acc0x0123);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c1_ptr), v_acc1x0123);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c2_ptr), v_acc2x0123);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c3_ptr), v_acc3x0123);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c4_ptr), v_acc4x0123);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c5_ptr), v_acc5x0123);

                c0_ptr += 2;
                c1_ptr += 2;
                c2_ptr += 2;
                c3_ptr += 2;
                c4_ptr += 2;
                c5_ptr += 2;

                v_acc0x0123 = _mm_srli_si128(v_rest0, 8);
                v_acc1x0123 = _mm_srli_si128(v_rest1, 8);
                v_acc2x0123 = _mm_srli_si128(v_rest2, 8);
                v_acc3x0123 = _mm_srli_si128(v_rest3, 8);
                v_acc4x0123 = _mm_srli_si128(v_rest4, 8);
                v_acc5x0123 = _mm_srli_si128(v_rest5, 8);
            }

            if (rem & 1) {
                int c0 = _mm_cvtsi128_si32(v_acc0x0123);
                int c1 = _mm_cvtsi128_si32(v_acc1x0123);
                int c2 = _mm_cvtsi128_si32(v_acc2x0123);
                int c3 = _mm_cvtsi128_si32(v_acc3x0123);
                int c4 = _mm_cvtsi128_si32(v_acc4x0123);
                int c5 = _mm_cvtsi128_si32(v_acc5x0123);
                if constexpr (!zero_mode) {
                    c0 += c0_ptr[0];
                    c1 += c1_ptr[0];
                    c2 += c2_ptr[0];
                    c3 += c3_ptr[0];
                    c4 += c4_ptr[0];
                    c5 += c5_ptr[0];
                }
                c0_ptr[0] = std::min(std::max(c0, clamp_min), clamp_max);
                c1_ptr[0] = std::min(std::max(c1, clamp_min), clamp_max);
                c2_ptr[0] = std::min(std::max(c2, clamp_min), clamp_max);
                c3_ptr[0] = std::min(std::max(c3, clamp_min), clamp_max);
                c4_ptr[0] = std::min(std::max(c4, clamp_min), clamp_max);
                c5_ptr[0] = std::min(std::max(c5, clamp_min), clamp_max);
            }
        }
    }
}

// =========================================================================
//  mr = 4
// =========================================================================

template <bool zero_mode = false>
void mma_packed_ab_mr4_nr16_mrxncxkc_i8(
    int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
    const int8_t* NNOPS_RESTRICT A,
    const int8_t* NNOPS_RESTRICT B,
    int32_t clamp_min, int32_t clamp_max) noexcept {

    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;

    int32_t* c0_ptr = C + 0 * ldc;
    int32_t* c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    int32_t* c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    int32_t* c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;

    const int kbytes = (kc + 3) & ~3;
    const int kg = kbytes >> 2;
    // Panel slot: 16 columns x kbytes of data, then the 16 int32 of column
    // sums the accumulator is seeded from.
    const int ldb = static_cast<int>(mma_pack_align(16 * kbytes + mma_i8_b_panel_tail));

    int n = 0;
    for(; n < nc - 15; n += 16) {
        __m256i v_acc0x01234567 = mma_i8_bias_fix(B, kbytes, 0);
        __m256i v_acc0x89abcdef = mma_i8_bias_fix(B, kbytes, 32);
        __m256i v_acc1x01234567 = v_acc0x01234567;
        __m256i v_acc1x89abcdef = v_acc0x89abcdef;
        __m256i v_acc2x01234567 = v_acc0x01234567;
        __m256i v_acc2x89abcdef = v_acc0x89abcdef;
        __m256i v_acc3x01234567 = v_acc0x01234567;
        __m256i v_acc3x89abcdef = v_acc0x89abcdef;

        const int32_t* a_i32 = reinterpret_cast<const int32_t*>(A);
        const int8_t* b_ptr = B;
        for(int k = 0; k < kg; ++k) {
            const __m256i v_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
            const __m256i v_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 32));
            b_ptr += 64;

            __m256i v_a0 = _mm256_set1_epi32(a_i32[0]);
            __m256i v_a1 = _mm256_set1_epi32(a_i32[r1]);
            v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567, v_a0, v_b0);
            v_acc0x89abcdef = mma_i8_dp256(v_acc0x89abcdef, v_a0, v_b1);
            v_acc1x01234567 = mma_i8_dp256(v_acc1x01234567, v_a1, v_b0);
            v_acc1x89abcdef = mma_i8_dp256(v_acc1x89abcdef, v_a1, v_b1);

            v_a0 = _mm256_set1_epi32(a_i32[r2]);
            v_a1 = _mm256_set1_epi32(a_i32[r3]);
            v_acc2x01234567 = mma_i8_dp256(v_acc2x01234567, v_a0, v_b0);
            v_acc2x89abcdef = mma_i8_dp256(v_acc2x89abcdef, v_a0, v_b1);
            v_acc3x01234567 = mma_i8_dp256(v_acc3x01234567, v_a1, v_b0);
            v_acc3x89abcdef = mma_i8_dp256(v_acc3x89abcdef, v_a1, v_b1);

            a_i32 += 4;
        }
        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_epi32(v_acc0x01234567, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c0_ptr + 0)));
            v_acc0x89abcdef = _mm256_add_epi32(v_acc0x89abcdef, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c0_ptr + 8)));
            v_acc1x01234567 = _mm256_add_epi32(v_acc1x01234567, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c1_ptr + 0)));
            v_acc1x89abcdef = _mm256_add_epi32(v_acc1x89abcdef, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c1_ptr + 8)));
            v_acc2x01234567 = _mm256_add_epi32(v_acc2x01234567, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c2_ptr + 0)));
            v_acc2x89abcdef = _mm256_add_epi32(v_acc2x89abcdef, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c2_ptr + 8)));
            v_acc3x01234567 = _mm256_add_epi32(v_acc3x01234567, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c3_ptr + 0)));
            v_acc3x89abcdef = _mm256_add_epi32(v_acc3x89abcdef, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c3_ptr + 8)));
        }
        const __m256i v_min = _mm256_set1_epi32(clamp_min);
        const __m256i v_max = _mm256_set1_epi32(clamp_max);

        v_acc0x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc0x01234567, v_min), v_max);
        v_acc0x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc0x89abcdef, v_min), v_max);
        v_acc1x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc1x01234567, v_min), v_max);
        v_acc1x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc1x89abcdef, v_min), v_max);
        v_acc2x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc2x01234567, v_min), v_max);
        v_acc2x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc2x89abcdef, v_min), v_max);
        v_acc3x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc3x01234567, v_min), v_max);
        v_acc3x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc3x89abcdef, v_min), v_max);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c0_ptr + 0), v_acc0x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c0_ptr + 8), v_acc0x89abcdef);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c1_ptr + 0), v_acc1x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c1_ptr + 8), v_acc1x89abcdef);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c2_ptr + 0), v_acc2x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c2_ptr + 8), v_acc2x89abcdef);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c3_ptr + 0), v_acc3x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c3_ptr + 8), v_acc3x89abcdef);

        c0_ptr += 16;
        c1_ptr += 16;
        c2_ptr += 16;
        c3_ptr += 16;

        B += ldb;
    }

    int rem = nc - n;
    const bool tail_hi = rem > 8;  // the tail k loop filled both halves

    if (rem > 0) {
        __m256i v_acc0x01234567 = mma_i8_bias_fix(B, kbytes, 0);
        __m256i v_acc0x89abcdef = mma_i8_bias_fix(B, kbytes, 32);
        __m256i v_acc1x01234567 = v_acc0x01234567;
        __m256i v_acc1x89abcdef = v_acc0x89abcdef;
        __m256i v_acc2x01234567 = v_acc0x01234567;
        __m256i v_acc2x89abcdef = v_acc0x89abcdef;
        __m256i v_acc3x01234567 = v_acc0x01234567;
        __m256i v_acc3x89abcdef = v_acc0x89abcdef;

        const int32_t* a_i32 = reinterpret_cast<const int32_t*>(A);
        const int8_t* b_ptr = B;
        if (rem > 8) {
            for(int k = 0; k < kg; ++k) {
                const __m256i v_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
                const __m256i v_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 32));
                b_ptr += 64;

                __m256i v_a0 = _mm256_set1_epi32(a_i32[0]);
                __m256i v_a1 = _mm256_set1_epi32(a_i32[r1]);
                v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567, v_a0, v_b0);
                v_acc0x89abcdef = mma_i8_dp256(v_acc0x89abcdef, v_a0, v_b1);
                v_acc1x01234567 = mma_i8_dp256(v_acc1x01234567, v_a1, v_b0);
                v_acc1x89abcdef = mma_i8_dp256(v_acc1x89abcdef, v_a1, v_b1);

                v_a0 = _mm256_set1_epi32(a_i32[r2]);
                v_a1 = _mm256_set1_epi32(a_i32[r3]);
                v_acc2x01234567 = mma_i8_dp256(v_acc2x01234567, v_a0, v_b0);
                v_acc2x89abcdef = mma_i8_dp256(v_acc2x89abcdef, v_a0, v_b1);
                v_acc3x01234567 = mma_i8_dp256(v_acc3x01234567, v_a1, v_b0);
                v_acc3x89abcdef = mma_i8_dp256(v_acc3x89abcdef, v_a1, v_b1);

                a_i32 += 4;
            }
        } else {
            for(int k = 0; k < kg; ++k) {
                const __m256i v_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
                b_ptr += 64;

                v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567, _mm256_set1_epi32(a_i32[0]), v_b0);
                v_acc1x01234567 = mma_i8_dp256(v_acc1x01234567, _mm256_set1_epi32(a_i32[r1]), v_b0);
                v_acc2x01234567 = mma_i8_dp256(v_acc2x01234567, _mm256_set1_epi32(a_i32[r2]), v_b0);
                v_acc3x01234567 = mma_i8_dp256(v_acc3x01234567, _mm256_set1_epi32(a_i32[r3]), v_b0);

                a_i32 += 4;
            }
        }

        if (rem >= 8) {

            if constexpr (!zero_mode) {
                v_acc0x01234567 = _mm256_add_epi32(v_acc0x01234567, _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(c0_ptr)));
                v_acc1x01234567 = _mm256_add_epi32(v_acc1x01234567, _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(c1_ptr)));
                v_acc2x01234567 = _mm256_add_epi32(v_acc2x01234567, _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(c2_ptr)));
                v_acc3x01234567 = _mm256_add_epi32(v_acc3x01234567, _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(c3_ptr)));
            }

            const __m256i v_min = _mm256_set1_epi32(clamp_min);
            const __m256i v_max = _mm256_set1_epi32(clamp_max);

            v_acc0x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc0x01234567, v_min), v_max);
            v_acc1x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc1x01234567, v_min), v_max);
            v_acc2x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc2x01234567, v_min), v_max);
            v_acc3x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc3x01234567, v_min), v_max);

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c0_ptr), v_acc0x01234567);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c1_ptr), v_acc1x01234567);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c2_ptr), v_acc2x01234567);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c3_ptr), v_acc3x01234567);

            c0_ptr += 8;
            c1_ptr += 8;
            c2_ptr += 8;
            c3_ptr += 8;

            rem -= 8;
        }

        if (rem > 0) {
            const __m256i v_acc0 = tail_hi ? v_acc0x89abcdef : v_acc0x01234567;
            const __m256i v_acc1 = tail_hi ? v_acc1x89abcdef : v_acc1x01234567;
            const __m256i v_acc2 = tail_hi ? v_acc2x89abcdef : v_acc2x01234567;
            const __m256i v_acc3 = tail_hi ? v_acc3x89abcdef : v_acc3x01234567;

            __m128i v_acc0x0123 = _mm256_castsi256_si128(v_acc0);
            __m128i v_acc1x0123 = _mm256_castsi256_si128(v_acc1);
            __m128i v_acc2x0123 = _mm256_castsi256_si128(v_acc2);
            __m128i v_acc3x0123 = _mm256_castsi256_si128(v_acc3);

            const __m128i v_min = _mm_set1_epi32(clamp_min);
            const __m128i v_max = _mm_set1_epi32(clamp_max);

            if (rem & 4) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_epi32(v_acc0x0123, _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(c0_ptr)));
                    v_acc1x0123 = _mm_add_epi32(v_acc1x0123, _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(c1_ptr)));
                    v_acc2x0123 = _mm_add_epi32(v_acc2x0123, _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(c2_ptr)));
                    v_acc3x0123 = _mm_add_epi32(v_acc3x0123, _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(c3_ptr)));
                }

                v_acc0x0123 = _mm_min_epi32(_mm_max_epi32(v_acc0x0123, v_min), v_max);
                v_acc1x0123 = _mm_min_epi32(_mm_max_epi32(v_acc1x0123, v_min), v_max);
                v_acc2x0123 = _mm_min_epi32(_mm_max_epi32(v_acc2x0123, v_min), v_max);
                v_acc3x0123 = _mm_min_epi32(_mm_max_epi32(v_acc3x0123, v_min), v_max);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr), v_acc0x0123);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(c1_ptr), v_acc1x0123);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(c2_ptr), v_acc2x0123);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(c3_ptr), v_acc3x0123);

                c0_ptr += 4;
                c1_ptr += 4;
                c2_ptr += 4;
                c3_ptr += 4;

                v_acc0x0123 = _mm256_extracti128_si256(v_acc0, 1);
                v_acc1x0123 = _mm256_extracti128_si256(v_acc1, 1);
                v_acc2x0123 = _mm256_extracti128_si256(v_acc2, 1);
                v_acc3x0123 = _mm256_extracti128_si256(v_acc3, 1);
            }

            if (rem & 2) {
                const __m128i v_rest0 = v_acc0x0123;
                const __m128i v_rest1 = v_acc1x0123;
                const __m128i v_rest2 = v_acc2x0123;
                const __m128i v_rest3 = v_acc3x0123;

                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_epi32(v_acc0x0123, _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(c0_ptr)));
                    v_acc1x0123 = _mm_add_epi32(v_acc1x0123, _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(c1_ptr)));
                    v_acc2x0123 = _mm_add_epi32(v_acc2x0123, _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(c2_ptr)));
                    v_acc3x0123 = _mm_add_epi32(v_acc3x0123, _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(c3_ptr)));
                }

                v_acc0x0123 = _mm_min_epi32(_mm_max_epi32(v_acc0x0123, v_min), v_max);
                v_acc1x0123 = _mm_min_epi32(_mm_max_epi32(v_acc1x0123, v_min), v_max);
                v_acc2x0123 = _mm_min_epi32(_mm_max_epi32(v_acc2x0123, v_min), v_max);
                v_acc3x0123 = _mm_min_epi32(_mm_max_epi32(v_acc3x0123, v_min), v_max);

                _mm_storel_epi64(reinterpret_cast<__m128i*>(c0_ptr), v_acc0x0123);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c1_ptr), v_acc1x0123);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c2_ptr), v_acc2x0123);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c3_ptr), v_acc3x0123);

                c0_ptr += 2;
                c1_ptr += 2;
                c2_ptr += 2;
                c3_ptr += 2;

                v_acc0x0123 = _mm_srli_si128(v_rest0, 8);
                v_acc1x0123 = _mm_srli_si128(v_rest1, 8);
                v_acc2x0123 = _mm_srli_si128(v_rest2, 8);
                v_acc3x0123 = _mm_srli_si128(v_rest3, 8);
            }

            if (rem & 1) {
                int c0 = _mm_cvtsi128_si32(v_acc0x0123);
                int c1 = _mm_cvtsi128_si32(v_acc1x0123);
                int c2 = _mm_cvtsi128_si32(v_acc2x0123);
                int c3 = _mm_cvtsi128_si32(v_acc3x0123);
                if constexpr (!zero_mode) {
                    c0 += c0_ptr[0];
                    c1 += c1_ptr[0];
                    c2 += c2_ptr[0];
                    c3 += c3_ptr[0];
                }
                c0_ptr[0] = std::min(std::max(c0, clamp_min), clamp_max);
                c1_ptr[0] = std::min(std::max(c1, clamp_min), clamp_max);
                c2_ptr[0] = std::min(std::max(c2, clamp_min), clamp_max);
                c3_ptr[0] = std::min(std::max(c3, clamp_min), clamp_max);
            }
        }
    }
}

// =========================================================================
//  mr = 1
// =========================================================================

template <bool zero_mode = false>
void mma_packed_ab_mr1_nr16_mrxncxkc_i8(
    int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
    const int8_t* NNOPS_RESTRICT A,
    const int8_t* NNOPS_RESTRICT B,
    int32_t clamp_min, int32_t clamp_max) noexcept {
    (void)mr;

    int32_t* c0_ptr = C;

    const int kbytes = (kc + 3) & ~3;
    const int kg = kbytes >> 2;
    // Panel slot: 16 columns x kbytes of data, then the 16 int32 of column
    // sums the accumulator is seeded from.
    const int ldb = static_cast<int>(mma_pack_align(16 * kbytes + mma_i8_b_panel_tail));

    int n = 0;
    for(; n < nc - 15; n += 16) {
        __m256i v_acc0x01234567 = mma_i8_bias_fix(B, kbytes, 0);
        __m256i v_acc0x89abcdef = mma_i8_bias_fix(B, kbytes, 32);

        const int32_t* a_i32 = reinterpret_cast<const int32_t*>(A);
        const int8_t* b_ptr = B;
        for(int k = 0; k < kg; ++k) {
            const __m256i v_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
            const __m256i v_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 32));
            b_ptr += 64;

            const __m256i v_a0 = _mm256_set1_epi32(a_i32[0]);
            v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567, v_a0, v_b0);
            v_acc0x89abcdef = mma_i8_dp256(v_acc0x89abcdef, v_a0, v_b1);

            a_i32 += 1;
        }
        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_epi32(v_acc0x01234567, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c0_ptr + 0)));
            v_acc0x89abcdef = _mm256_add_epi32(v_acc0x89abcdef, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c0_ptr + 8)));
        }
        const __m256i v_min = _mm256_set1_epi32(clamp_min);
        const __m256i v_max = _mm256_set1_epi32(clamp_max);

        v_acc0x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc0x01234567, v_min), v_max);
        v_acc0x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc0x89abcdef, v_min), v_max);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c0_ptr + 0), v_acc0x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c0_ptr + 8), v_acc0x89abcdef);

        c0_ptr += 16;

        B += ldb;
    }

    int rem = nc - n;
    const bool tail_hi = rem > 8;  // the tail k loop filled both halves

    if (rem > 0) {
        __m256i v_acc0x01234567 = mma_i8_bias_fix(B, kbytes, 0);
        __m256i v_acc0x89abcdef = mma_i8_bias_fix(B, kbytes, 32);

        const int32_t* a_i32 = reinterpret_cast<const int32_t*>(A);
        const int8_t* b_ptr = B;
        if (rem > 8) {
            for(int k = 0; k < kg; ++k) {
                const __m256i v_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
                const __m256i v_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 32));
                b_ptr += 64;

                const __m256i v_a0 = _mm256_set1_epi32(a_i32[0]);
                v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567, v_a0, v_b0);
                v_acc0x89abcdef = mma_i8_dp256(v_acc0x89abcdef, v_a0, v_b1);

                a_i32 += 1;
            }
        } else {
            for(int k = 0; k < kg; ++k) {
                const __m256i v_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
                b_ptr += 64;

                v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567,
                    _mm256_set1_epi32(a_i32[0]), v_b0);

                a_i32 += 1;
            }
        }

        if (rem >= 8) {

            if constexpr (!zero_mode) {
                v_acc0x01234567 = _mm256_add_epi32(v_acc0x01234567, _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(c0_ptr)));
            }

            const __m256i v_min = _mm256_set1_epi32(clamp_min);
            const __m256i v_max = _mm256_set1_epi32(clamp_max);

            v_acc0x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc0x01234567, v_min), v_max);

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c0_ptr), v_acc0x01234567);

            c0_ptr += 8;

            rem -= 8;
        }

        if (rem > 0) {
            const __m256i v_acc0 = tail_hi ? v_acc0x89abcdef : v_acc0x01234567;

            __m128i v_acc0x0123 = _mm256_castsi256_si128(v_acc0);

            const __m128i v_min = _mm_set1_epi32(clamp_min);
            const __m128i v_max = _mm_set1_epi32(clamp_max);

            if (rem & 4) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_epi32(v_acc0x0123, _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(c0_ptr)));
                }

                v_acc0x0123 = _mm_min_epi32(_mm_max_epi32(v_acc0x0123, v_min), v_max);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr), v_acc0x0123);

                c0_ptr += 4;

                v_acc0x0123 = _mm256_extracti128_si256(v_acc0, 1);
            }

            if (rem & 2) {
                const __m128i v_rest0 = v_acc0x0123;

                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_epi32(v_acc0x0123, _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(c0_ptr)));
                }

                v_acc0x0123 = _mm_min_epi32(_mm_max_epi32(v_acc0x0123, v_min), v_max);

                _mm_storel_epi64(reinterpret_cast<__m128i*>(c0_ptr), v_acc0x0123);

                c0_ptr += 2;

                v_acc0x0123 = _mm_srli_si128(v_rest0, 8);
            }

            if (rem & 1) {
                int c0 = _mm_cvtsi128_si32(v_acc0x0123);
                if constexpr (!zero_mode) {
                    c0 += c0_ptr[0];
                }
                c0_ptr[0] = std::min(std::max(c0, clamp_min), clamp_max);
            }
        }
    }
}

// =========================================================================
//  Direct-A (row-major A) variants
// =========================================================================
//
// `lda` is the A row stride in int8 elements. These read each row's 4-k groups
// straight from memory instead of from a packed panel, so kc must be a
// multiple of 4 -- a row-major row has no zero-filled final group to fall back
// on.

// =========================================================================
//  mr = 6, direct A
// =========================================================================

template <bool zero_mode = false>
void mma_packed_b_mr6_nr16_mrxncxkc_i8(
    int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
    const int8_t* NNOPS_RESTRICT A, int lda,
    const int8_t* NNOPS_RESTRICT B,
    int32_t clamp_min, int32_t clamp_max) noexcept {

    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;
    const int r4 = (mr > 4) ? 4 : r3;
    const int r5 = (mr > 5) ? 5 : r4;

    int32_t* c0_ptr = C + 0 * ldc;
    int32_t* c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    int32_t* c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    int32_t* c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;
    int32_t* c4_ptr = (mr > 4) ? C + 4 * ldc : c3_ptr;
    int32_t* c5_ptr = (mr > 5) ? C + 5 * ldc : c4_ptr;

    const int kg = kc >> 2;
    // Panel slot: 16 columns x kc of data, then the 16 int32 of column sums.
    const int ldb = static_cast<int>(mma_pack_align(16 * kc + mma_i8_b_panel_tail));

    int n = 0;
    for(; n < nc - 15; n += 16) {
        const int8_t* a0_ptr = A + 0 * lda;
        const int8_t* a1_ptr = A + r1 * lda;
        const int8_t* a2_ptr = A + r2 * lda;
        const int8_t* a3_ptr = A + r3 * lda;
        const int8_t* a4_ptr = A + r4 * lda;
        const int8_t* a5_ptr = A + r5 * lda;

        __m256i v_acc0x01234567 = mma_i8_bias_fix(B, kc, 0);
        __m256i v_acc0x89abcdef = mma_i8_bias_fix(B, kc, 32);
        __m256i v_acc1x01234567 = v_acc0x01234567;
        __m256i v_acc1x89abcdef = v_acc0x89abcdef;
        __m256i v_acc2x01234567 = v_acc0x01234567;
        __m256i v_acc2x89abcdef = v_acc0x89abcdef;
        __m256i v_acc3x01234567 = v_acc0x01234567;
        __m256i v_acc3x89abcdef = v_acc0x89abcdef;
        __m256i v_acc4x01234567 = v_acc0x01234567;
        __m256i v_acc4x89abcdef = v_acc0x89abcdef;
        __m256i v_acc5x01234567 = v_acc0x01234567;
        __m256i v_acc5x89abcdef = v_acc0x89abcdef;

        const int8_t* b_ptr = B;
        for(int k = 0; k < kg; ++k) {
            const __m256i v_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
            const __m256i v_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 32));
            b_ptr += 64;

            __m256i v_a0 = mma_i8_bcast_group(a0_ptr);
            __m256i v_a1 = mma_i8_bcast_group(a1_ptr);
            v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567, v_a0, v_b0);
            v_acc0x89abcdef = mma_i8_dp256(v_acc0x89abcdef, v_a0, v_b1);
            v_acc1x01234567 = mma_i8_dp256(v_acc1x01234567, v_a1, v_b0);
            v_acc1x89abcdef = mma_i8_dp256(v_acc1x89abcdef, v_a1, v_b1);

            v_a0 = mma_i8_bcast_group(a2_ptr);
            v_a1 = mma_i8_bcast_group(a3_ptr);
            v_acc2x01234567 = mma_i8_dp256(v_acc2x01234567, v_a0, v_b0);
            v_acc2x89abcdef = mma_i8_dp256(v_acc2x89abcdef, v_a0, v_b1);
            v_acc3x01234567 = mma_i8_dp256(v_acc3x01234567, v_a1, v_b0);
            v_acc3x89abcdef = mma_i8_dp256(v_acc3x89abcdef, v_a1, v_b1);

            v_a0 = mma_i8_bcast_group(a4_ptr);
            v_a1 = mma_i8_bcast_group(a5_ptr);
            v_acc4x01234567 = mma_i8_dp256(v_acc4x01234567, v_a0, v_b0);
            v_acc4x89abcdef = mma_i8_dp256(v_acc4x89abcdef, v_a0, v_b1);
            v_acc5x01234567 = mma_i8_dp256(v_acc5x01234567, v_a1, v_b0);
            v_acc5x89abcdef = mma_i8_dp256(v_acc5x89abcdef, v_a1, v_b1);

            a0_ptr += 4;
            a1_ptr += 4;
            a2_ptr += 4;
            a3_ptr += 4;
            a4_ptr += 4;
            a5_ptr += 4;
        }
        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_epi32(v_acc0x01234567, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c0_ptr + 0)));
            v_acc0x89abcdef = _mm256_add_epi32(v_acc0x89abcdef, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c0_ptr + 8)));
            v_acc1x01234567 = _mm256_add_epi32(v_acc1x01234567, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c1_ptr + 0)));
            v_acc1x89abcdef = _mm256_add_epi32(v_acc1x89abcdef, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c1_ptr + 8)));
            v_acc2x01234567 = _mm256_add_epi32(v_acc2x01234567, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c2_ptr + 0)));
            v_acc2x89abcdef = _mm256_add_epi32(v_acc2x89abcdef, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c2_ptr + 8)));
            v_acc3x01234567 = _mm256_add_epi32(v_acc3x01234567, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c3_ptr + 0)));
            v_acc3x89abcdef = _mm256_add_epi32(v_acc3x89abcdef, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c3_ptr + 8)));
            v_acc4x01234567 = _mm256_add_epi32(v_acc4x01234567, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c4_ptr + 0)));
            v_acc4x89abcdef = _mm256_add_epi32(v_acc4x89abcdef, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c4_ptr + 8)));
            v_acc5x01234567 = _mm256_add_epi32(v_acc5x01234567, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c5_ptr + 0)));
            v_acc5x89abcdef = _mm256_add_epi32(v_acc5x89abcdef, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c5_ptr + 8)));
        }
        const __m256i v_min = _mm256_set1_epi32(clamp_min);
        const __m256i v_max = _mm256_set1_epi32(clamp_max);

        v_acc0x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc0x01234567, v_min), v_max);
        v_acc0x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc0x89abcdef, v_min), v_max);
        v_acc1x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc1x01234567, v_min), v_max);
        v_acc1x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc1x89abcdef, v_min), v_max);
        v_acc2x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc2x01234567, v_min), v_max);
        v_acc2x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc2x89abcdef, v_min), v_max);
        v_acc3x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc3x01234567, v_min), v_max);
        v_acc3x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc3x89abcdef, v_min), v_max);
        v_acc4x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc4x01234567, v_min), v_max);
        v_acc4x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc4x89abcdef, v_min), v_max);
        v_acc5x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc5x01234567, v_min), v_max);
        v_acc5x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc5x89abcdef, v_min), v_max);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c0_ptr + 0), v_acc0x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c0_ptr + 8), v_acc0x89abcdef);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c1_ptr + 0), v_acc1x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c1_ptr + 8), v_acc1x89abcdef);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c2_ptr + 0), v_acc2x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c2_ptr + 8), v_acc2x89abcdef);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c3_ptr + 0), v_acc3x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c3_ptr + 8), v_acc3x89abcdef);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c4_ptr + 0), v_acc4x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c4_ptr + 8), v_acc4x89abcdef);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c5_ptr + 0), v_acc5x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c5_ptr + 8), v_acc5x89abcdef);

        c0_ptr += 16;
        c1_ptr += 16;
        c2_ptr += 16;
        c3_ptr += 16;
        c4_ptr += 16;
        c5_ptr += 16;

        B += ldb;
    }

    int rem = nc - n;
    const bool tail_hi = rem > 8;

    if (rem > 0) {
        const int8_t* a0_ptr = A + 0 * lda;
        const int8_t* a1_ptr = A + r1 * lda;
        const int8_t* a2_ptr = A + r2 * lda;
        const int8_t* a3_ptr = A + r3 * lda;
        const int8_t* a4_ptr = A + r4 * lda;
        const int8_t* a5_ptr = A + r5 * lda;

        __m256i v_acc0x01234567 = mma_i8_bias_fix(B, kc, 0);
        __m256i v_acc0x89abcdef = mma_i8_bias_fix(B, kc, 32);
        __m256i v_acc1x01234567 = v_acc0x01234567;
        __m256i v_acc1x89abcdef = v_acc0x89abcdef;
        __m256i v_acc2x01234567 = v_acc0x01234567;
        __m256i v_acc2x89abcdef = v_acc0x89abcdef;
        __m256i v_acc3x01234567 = v_acc0x01234567;
        __m256i v_acc3x89abcdef = v_acc0x89abcdef;
        __m256i v_acc4x01234567 = v_acc0x01234567;
        __m256i v_acc4x89abcdef = v_acc0x89abcdef;
        __m256i v_acc5x01234567 = v_acc0x01234567;
        __m256i v_acc5x89abcdef = v_acc0x89abcdef;

        const int8_t* b_ptr = B;
        if (rem > 8) {
            for(int k = 0; k < kg; ++k) {
                const __m256i v_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
                const __m256i v_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 32));
                b_ptr += 64;

                __m256i v_a0 = mma_i8_bcast_group(a0_ptr);
                __m256i v_a1 = mma_i8_bcast_group(a1_ptr);
                v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567, v_a0, v_b0);
                v_acc0x89abcdef = mma_i8_dp256(v_acc0x89abcdef, v_a0, v_b1);
                v_acc1x01234567 = mma_i8_dp256(v_acc1x01234567, v_a1, v_b0);
                v_acc1x89abcdef = mma_i8_dp256(v_acc1x89abcdef, v_a1, v_b1);

                v_a0 = mma_i8_bcast_group(a2_ptr);
                v_a1 = mma_i8_bcast_group(a3_ptr);
                v_acc2x01234567 = mma_i8_dp256(v_acc2x01234567, v_a0, v_b0);
                v_acc2x89abcdef = mma_i8_dp256(v_acc2x89abcdef, v_a0, v_b1);
                v_acc3x01234567 = mma_i8_dp256(v_acc3x01234567, v_a1, v_b0);
                v_acc3x89abcdef = mma_i8_dp256(v_acc3x89abcdef, v_a1, v_b1);

                v_a0 = mma_i8_bcast_group(a4_ptr);
                v_a1 = mma_i8_bcast_group(a5_ptr);
                v_acc4x01234567 = mma_i8_dp256(v_acc4x01234567, v_a0, v_b0);
                v_acc4x89abcdef = mma_i8_dp256(v_acc4x89abcdef, v_a0, v_b1);
                v_acc5x01234567 = mma_i8_dp256(v_acc5x01234567, v_a1, v_b0);
                v_acc5x89abcdef = mma_i8_dp256(v_acc5x89abcdef, v_a1, v_b1);

                a0_ptr += 4;
                a1_ptr += 4;
                a2_ptr += 4;
                a3_ptr += 4;
                a4_ptr += 4;
                a5_ptr += 4;
            }
        } else {
            for(int k = 0; k < kg; ++k) {
                const __m256i v_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
                b_ptr += 64;

                v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567, mma_i8_bcast_group(a0_ptr), v_b0);
                v_acc1x01234567 = mma_i8_dp256(v_acc1x01234567, mma_i8_bcast_group(a1_ptr), v_b0);
                v_acc2x01234567 = mma_i8_dp256(v_acc2x01234567, mma_i8_bcast_group(a2_ptr), v_b0);
                v_acc3x01234567 = mma_i8_dp256(v_acc3x01234567, mma_i8_bcast_group(a3_ptr), v_b0);
                v_acc4x01234567 = mma_i8_dp256(v_acc4x01234567, mma_i8_bcast_group(a4_ptr), v_b0);
                v_acc5x01234567 = mma_i8_dp256(v_acc5x01234567, mma_i8_bcast_group(a5_ptr), v_b0);

                a0_ptr += 4;
                a1_ptr += 4;
                a2_ptr += 4;
                a3_ptr += 4;
                a4_ptr += 4;
                a5_ptr += 4;
            }
        }

        if (rem >= 8) {

            if constexpr (!zero_mode) {
                v_acc0x01234567 = _mm256_add_epi32(v_acc0x01234567, _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(c0_ptr)));
                v_acc1x01234567 = _mm256_add_epi32(v_acc1x01234567, _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(c1_ptr)));
                v_acc2x01234567 = _mm256_add_epi32(v_acc2x01234567, _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(c2_ptr)));
                v_acc3x01234567 = _mm256_add_epi32(v_acc3x01234567, _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(c3_ptr)));
                v_acc4x01234567 = _mm256_add_epi32(v_acc4x01234567, _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(c4_ptr)));
                v_acc5x01234567 = _mm256_add_epi32(v_acc5x01234567, _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(c5_ptr)));
            }

            const __m256i v_min = _mm256_set1_epi32(clamp_min);
            const __m256i v_max = _mm256_set1_epi32(clamp_max);

            v_acc0x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc0x01234567, v_min), v_max);
            v_acc1x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc1x01234567, v_min), v_max);
            v_acc2x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc2x01234567, v_min), v_max);
            v_acc3x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc3x01234567, v_min), v_max);
            v_acc4x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc4x01234567, v_min), v_max);
            v_acc5x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc5x01234567, v_min), v_max);

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c0_ptr), v_acc0x01234567);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c1_ptr), v_acc1x01234567);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c2_ptr), v_acc2x01234567);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c3_ptr), v_acc3x01234567);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c4_ptr), v_acc4x01234567);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c5_ptr), v_acc5x01234567);

            c0_ptr += 8;
            c1_ptr += 8;
            c2_ptr += 8;
            c3_ptr += 8;
            c4_ptr += 8;
            c5_ptr += 8;

            rem -= 8;
        }

        if (rem > 0) {
            const __m256i v_acc0 = tail_hi ? v_acc0x89abcdef : v_acc0x01234567;
            const __m256i v_acc1 = tail_hi ? v_acc1x89abcdef : v_acc1x01234567;
            const __m256i v_acc2 = tail_hi ? v_acc2x89abcdef : v_acc2x01234567;
            const __m256i v_acc3 = tail_hi ? v_acc3x89abcdef : v_acc3x01234567;
            const __m256i v_acc4 = tail_hi ? v_acc4x89abcdef : v_acc4x01234567;
            const __m256i v_acc5 = tail_hi ? v_acc5x89abcdef : v_acc5x01234567;

            __m128i v_acc0x0123 = _mm256_castsi256_si128(v_acc0);
            __m128i v_acc1x0123 = _mm256_castsi256_si128(v_acc1);
            __m128i v_acc2x0123 = _mm256_castsi256_si128(v_acc2);
            __m128i v_acc3x0123 = _mm256_castsi256_si128(v_acc3);
            __m128i v_acc4x0123 = _mm256_castsi256_si128(v_acc4);
            __m128i v_acc5x0123 = _mm256_castsi256_si128(v_acc5);

            const __m128i v_min = _mm_set1_epi32(clamp_min);
            const __m128i v_max = _mm_set1_epi32(clamp_max);

            if (rem & 4) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_epi32(v_acc0x0123, _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(c0_ptr)));
                    v_acc1x0123 = _mm_add_epi32(v_acc1x0123, _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(c1_ptr)));
                    v_acc2x0123 = _mm_add_epi32(v_acc2x0123, _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(c2_ptr)));
                    v_acc3x0123 = _mm_add_epi32(v_acc3x0123, _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(c3_ptr)));
                    v_acc4x0123 = _mm_add_epi32(v_acc4x0123, _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(c4_ptr)));
                    v_acc5x0123 = _mm_add_epi32(v_acc5x0123, _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(c5_ptr)));
                }

                v_acc0x0123 = _mm_min_epi32(_mm_max_epi32(v_acc0x0123, v_min), v_max);
                v_acc1x0123 = _mm_min_epi32(_mm_max_epi32(v_acc1x0123, v_min), v_max);
                v_acc2x0123 = _mm_min_epi32(_mm_max_epi32(v_acc2x0123, v_min), v_max);
                v_acc3x0123 = _mm_min_epi32(_mm_max_epi32(v_acc3x0123, v_min), v_max);
                v_acc4x0123 = _mm_min_epi32(_mm_max_epi32(v_acc4x0123, v_min), v_max);
                v_acc5x0123 = _mm_min_epi32(_mm_max_epi32(v_acc5x0123, v_min), v_max);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr), v_acc0x0123);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(c1_ptr), v_acc1x0123);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(c2_ptr), v_acc2x0123);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(c3_ptr), v_acc3x0123);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(c4_ptr), v_acc4x0123);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(c5_ptr), v_acc5x0123);

                c0_ptr += 4;
                c1_ptr += 4;
                c2_ptr += 4;
                c3_ptr += 4;
                c4_ptr += 4;
                c5_ptr += 4;

                v_acc0x0123 = _mm256_extracti128_si256(v_acc0, 1);
                v_acc1x0123 = _mm256_extracti128_si256(v_acc1, 1);
                v_acc2x0123 = _mm256_extracti128_si256(v_acc2, 1);
                v_acc3x0123 = _mm256_extracti128_si256(v_acc3, 1);
                v_acc4x0123 = _mm256_extracti128_si256(v_acc4, 1);
                v_acc5x0123 = _mm256_extracti128_si256(v_acc5, 1);
            }

            if (rem & 2) {
                const __m128i v_rest0 = v_acc0x0123;
                const __m128i v_rest1 = v_acc1x0123;
                const __m128i v_rest2 = v_acc2x0123;
                const __m128i v_rest3 = v_acc3x0123;
                const __m128i v_rest4 = v_acc4x0123;
                const __m128i v_rest5 = v_acc5x0123;

                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_epi32(v_acc0x0123, _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(c0_ptr)));
                    v_acc1x0123 = _mm_add_epi32(v_acc1x0123, _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(c1_ptr)));
                    v_acc2x0123 = _mm_add_epi32(v_acc2x0123, _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(c2_ptr)));
                    v_acc3x0123 = _mm_add_epi32(v_acc3x0123, _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(c3_ptr)));
                    v_acc4x0123 = _mm_add_epi32(v_acc4x0123, _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(c4_ptr)));
                    v_acc5x0123 = _mm_add_epi32(v_acc5x0123, _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(c5_ptr)));
                }

                v_acc0x0123 = _mm_min_epi32(_mm_max_epi32(v_acc0x0123, v_min), v_max);
                v_acc1x0123 = _mm_min_epi32(_mm_max_epi32(v_acc1x0123, v_min), v_max);
                v_acc2x0123 = _mm_min_epi32(_mm_max_epi32(v_acc2x0123, v_min), v_max);
                v_acc3x0123 = _mm_min_epi32(_mm_max_epi32(v_acc3x0123, v_min), v_max);
                v_acc4x0123 = _mm_min_epi32(_mm_max_epi32(v_acc4x0123, v_min), v_max);
                v_acc5x0123 = _mm_min_epi32(_mm_max_epi32(v_acc5x0123, v_min), v_max);

                _mm_storel_epi64(reinterpret_cast<__m128i*>(c0_ptr), v_acc0x0123);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c1_ptr), v_acc1x0123);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c2_ptr), v_acc2x0123);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c3_ptr), v_acc3x0123);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c4_ptr), v_acc4x0123);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c5_ptr), v_acc5x0123);

                c0_ptr += 2;
                c1_ptr += 2;
                c2_ptr += 2;
                c3_ptr += 2;
                c4_ptr += 2;
                c5_ptr += 2;

                v_acc0x0123 = _mm_srli_si128(v_rest0, 8);
                v_acc1x0123 = _mm_srli_si128(v_rest1, 8);
                v_acc2x0123 = _mm_srli_si128(v_rest2, 8);
                v_acc3x0123 = _mm_srli_si128(v_rest3, 8);
                v_acc4x0123 = _mm_srli_si128(v_rest4, 8);
                v_acc5x0123 = _mm_srli_si128(v_rest5, 8);
            }

            if (rem & 1) {
                int c0 = _mm_cvtsi128_si32(v_acc0x0123);
                int c1 = _mm_cvtsi128_si32(v_acc1x0123);
                int c2 = _mm_cvtsi128_si32(v_acc2x0123);
                int c3 = _mm_cvtsi128_si32(v_acc3x0123);
                int c4 = _mm_cvtsi128_si32(v_acc4x0123);
                int c5 = _mm_cvtsi128_si32(v_acc5x0123);
                if constexpr (!zero_mode) {
                    c0 += c0_ptr[0];
                    c1 += c1_ptr[0];
                    c2 += c2_ptr[0];
                    c3 += c3_ptr[0];
                    c4 += c4_ptr[0];
                    c5 += c5_ptr[0];
                }
                c0_ptr[0] = std::min(std::max(c0, clamp_min), clamp_max);
                c1_ptr[0] = std::min(std::max(c1, clamp_min), clamp_max);
                c2_ptr[0] = std::min(std::max(c2, clamp_min), clamp_max);
                c3_ptr[0] = std::min(std::max(c3, clamp_min), clamp_max);
                c4_ptr[0] = std::min(std::max(c4, clamp_min), clamp_max);
                c5_ptr[0] = std::min(std::max(c5, clamp_min), clamp_max);
            }
        }
    }
}

// =========================================================================
//  mr = 4, direct A
// =========================================================================

template <bool zero_mode = false>
void mma_packed_b_mr4_nr16_mrxncxkc_i8(
    int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
    const int8_t* NNOPS_RESTRICT A, int lda,
    const int8_t* NNOPS_RESTRICT B,
    int32_t clamp_min, int32_t clamp_max) noexcept {

    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;

    int32_t* c0_ptr = C + 0 * ldc;
    int32_t* c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    int32_t* c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    int32_t* c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;

    const int kg = kc >> 2;
    // Panel slot: 16 columns x kc of data, then the 16 int32 of column sums.
    const int ldb = static_cast<int>(mma_pack_align(16 * kc + mma_i8_b_panel_tail));

    int n = 0;
    for(; n < nc - 15; n += 16) {
        const int8_t* a0_ptr = A + 0 * lda;
        const int8_t* a1_ptr = A + r1 * lda;
        const int8_t* a2_ptr = A + r2 * lda;
        const int8_t* a3_ptr = A + r3 * lda;

        __m256i v_acc0x01234567 = mma_i8_bias_fix(B, kc, 0);
        __m256i v_acc0x89abcdef = mma_i8_bias_fix(B, kc, 32);
        __m256i v_acc1x01234567 = v_acc0x01234567;
        __m256i v_acc1x89abcdef = v_acc0x89abcdef;
        __m256i v_acc2x01234567 = v_acc0x01234567;
        __m256i v_acc2x89abcdef = v_acc0x89abcdef;
        __m256i v_acc3x01234567 = v_acc0x01234567;
        __m256i v_acc3x89abcdef = v_acc0x89abcdef;

        const int8_t* b_ptr = B;
        for(int k = 0; k < kg; ++k) {
            const __m256i v_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
            const __m256i v_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 32));
            b_ptr += 64;

            __m256i v_a0 = mma_i8_bcast_group(a0_ptr);
            __m256i v_a1 = mma_i8_bcast_group(a1_ptr);
            v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567, v_a0, v_b0);
            v_acc0x89abcdef = mma_i8_dp256(v_acc0x89abcdef, v_a0, v_b1);
            v_acc1x01234567 = mma_i8_dp256(v_acc1x01234567, v_a1, v_b0);
            v_acc1x89abcdef = mma_i8_dp256(v_acc1x89abcdef, v_a1, v_b1);

            v_a0 = mma_i8_bcast_group(a2_ptr);
            v_a1 = mma_i8_bcast_group(a3_ptr);
            v_acc2x01234567 = mma_i8_dp256(v_acc2x01234567, v_a0, v_b0);
            v_acc2x89abcdef = mma_i8_dp256(v_acc2x89abcdef, v_a0, v_b1);
            v_acc3x01234567 = mma_i8_dp256(v_acc3x01234567, v_a1, v_b0);
            v_acc3x89abcdef = mma_i8_dp256(v_acc3x89abcdef, v_a1, v_b1);

            a0_ptr += 4;
            a1_ptr += 4;
            a2_ptr += 4;
            a3_ptr += 4;
        }
        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_epi32(v_acc0x01234567, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c0_ptr + 0)));
            v_acc0x89abcdef = _mm256_add_epi32(v_acc0x89abcdef, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c0_ptr + 8)));
            v_acc1x01234567 = _mm256_add_epi32(v_acc1x01234567, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c1_ptr + 0)));
            v_acc1x89abcdef = _mm256_add_epi32(v_acc1x89abcdef, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c1_ptr + 8)));
            v_acc2x01234567 = _mm256_add_epi32(v_acc2x01234567, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c2_ptr + 0)));
            v_acc2x89abcdef = _mm256_add_epi32(v_acc2x89abcdef, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c2_ptr + 8)));
            v_acc3x01234567 = _mm256_add_epi32(v_acc3x01234567, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c3_ptr + 0)));
            v_acc3x89abcdef = _mm256_add_epi32(v_acc3x89abcdef, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c3_ptr + 8)));
        }
        const __m256i v_min = _mm256_set1_epi32(clamp_min);
        const __m256i v_max = _mm256_set1_epi32(clamp_max);

        v_acc0x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc0x01234567, v_min), v_max);
        v_acc0x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc0x89abcdef, v_min), v_max);
        v_acc1x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc1x01234567, v_min), v_max);
        v_acc1x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc1x89abcdef, v_min), v_max);
        v_acc2x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc2x01234567, v_min), v_max);
        v_acc2x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc2x89abcdef, v_min), v_max);
        v_acc3x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc3x01234567, v_min), v_max);
        v_acc3x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc3x89abcdef, v_min), v_max);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c0_ptr + 0), v_acc0x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c0_ptr + 8), v_acc0x89abcdef);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c1_ptr + 0), v_acc1x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c1_ptr + 8), v_acc1x89abcdef);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c2_ptr + 0), v_acc2x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c2_ptr + 8), v_acc2x89abcdef);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c3_ptr + 0), v_acc3x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c3_ptr + 8), v_acc3x89abcdef);

        c0_ptr += 16;
        c1_ptr += 16;
        c2_ptr += 16;
        c3_ptr += 16;

        B += ldb;
    }

    int rem = nc - n;
    const bool tail_hi = rem > 8;

    if (rem > 0) {
        const int8_t* a0_ptr = A + 0 * lda;
        const int8_t* a1_ptr = A + r1 * lda;
        const int8_t* a2_ptr = A + r2 * lda;
        const int8_t* a3_ptr = A + r3 * lda;

        __m256i v_acc0x01234567 = mma_i8_bias_fix(B, kc, 0);
        __m256i v_acc0x89abcdef = mma_i8_bias_fix(B, kc, 32);
        __m256i v_acc1x01234567 = v_acc0x01234567;
        __m256i v_acc1x89abcdef = v_acc0x89abcdef;
        __m256i v_acc2x01234567 = v_acc0x01234567;
        __m256i v_acc2x89abcdef = v_acc0x89abcdef;
        __m256i v_acc3x01234567 = v_acc0x01234567;
        __m256i v_acc3x89abcdef = v_acc0x89abcdef;

        const int8_t* b_ptr = B;
        if (rem > 8) {
            for(int k = 0; k < kg; ++k) {
                const __m256i v_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
                const __m256i v_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 32));
                b_ptr += 64;

                __m256i v_a0 = mma_i8_bcast_group(a0_ptr);
                __m256i v_a1 = mma_i8_bcast_group(a1_ptr);
                v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567, v_a0, v_b0);
                v_acc0x89abcdef = mma_i8_dp256(v_acc0x89abcdef, v_a0, v_b1);
                v_acc1x01234567 = mma_i8_dp256(v_acc1x01234567, v_a1, v_b0);
                v_acc1x89abcdef = mma_i8_dp256(v_acc1x89abcdef, v_a1, v_b1);

                v_a0 = mma_i8_bcast_group(a2_ptr);
                v_a1 = mma_i8_bcast_group(a3_ptr);
                v_acc2x01234567 = mma_i8_dp256(v_acc2x01234567, v_a0, v_b0);
                v_acc2x89abcdef = mma_i8_dp256(v_acc2x89abcdef, v_a0, v_b1);
                v_acc3x01234567 = mma_i8_dp256(v_acc3x01234567, v_a1, v_b0);
                v_acc3x89abcdef = mma_i8_dp256(v_acc3x89abcdef, v_a1, v_b1);

                a0_ptr += 4;
                a1_ptr += 4;
                a2_ptr += 4;
                a3_ptr += 4;
            }
        } else {
            for(int k = 0; k < kg; ++k) {
                const __m256i v_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
                b_ptr += 64;

                v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567, mma_i8_bcast_group(a0_ptr), v_b0);
                v_acc1x01234567 = mma_i8_dp256(v_acc1x01234567, mma_i8_bcast_group(a1_ptr), v_b0);
                v_acc2x01234567 = mma_i8_dp256(v_acc2x01234567, mma_i8_bcast_group(a2_ptr), v_b0);
                v_acc3x01234567 = mma_i8_dp256(v_acc3x01234567, mma_i8_bcast_group(a3_ptr), v_b0);

                a0_ptr += 4;
                a1_ptr += 4;
                a2_ptr += 4;
                a3_ptr += 4;
            }
        }

        if (rem >= 8) {

            if constexpr (!zero_mode) {
                v_acc0x01234567 = _mm256_add_epi32(v_acc0x01234567, _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(c0_ptr)));
                v_acc1x01234567 = _mm256_add_epi32(v_acc1x01234567, _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(c1_ptr)));
                v_acc2x01234567 = _mm256_add_epi32(v_acc2x01234567, _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(c2_ptr)));
                v_acc3x01234567 = _mm256_add_epi32(v_acc3x01234567, _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(c3_ptr)));
            }

            const __m256i v_min = _mm256_set1_epi32(clamp_min);
            const __m256i v_max = _mm256_set1_epi32(clamp_max);

            v_acc0x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc0x01234567, v_min), v_max);
            v_acc1x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc1x01234567, v_min), v_max);
            v_acc2x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc2x01234567, v_min), v_max);
            v_acc3x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc3x01234567, v_min), v_max);

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c0_ptr), v_acc0x01234567);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c1_ptr), v_acc1x01234567);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c2_ptr), v_acc2x01234567);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c3_ptr), v_acc3x01234567);

            c0_ptr += 8;
            c1_ptr += 8;
            c2_ptr += 8;
            c3_ptr += 8;

            rem -= 8;
        }

        if (rem > 0) {
            const __m256i v_acc0 = tail_hi ? v_acc0x89abcdef : v_acc0x01234567;
            const __m256i v_acc1 = tail_hi ? v_acc1x89abcdef : v_acc1x01234567;
            const __m256i v_acc2 = tail_hi ? v_acc2x89abcdef : v_acc2x01234567;
            const __m256i v_acc3 = tail_hi ? v_acc3x89abcdef : v_acc3x01234567;

            __m128i v_acc0x0123 = _mm256_castsi256_si128(v_acc0);
            __m128i v_acc1x0123 = _mm256_castsi256_si128(v_acc1);
            __m128i v_acc2x0123 = _mm256_castsi256_si128(v_acc2);
            __m128i v_acc3x0123 = _mm256_castsi256_si128(v_acc3);

            const __m128i v_min = _mm_set1_epi32(clamp_min);
            const __m128i v_max = _mm_set1_epi32(clamp_max);

            if (rem & 4) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_epi32(v_acc0x0123, _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(c0_ptr)));
                    v_acc1x0123 = _mm_add_epi32(v_acc1x0123, _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(c1_ptr)));
                    v_acc2x0123 = _mm_add_epi32(v_acc2x0123, _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(c2_ptr)));
                    v_acc3x0123 = _mm_add_epi32(v_acc3x0123, _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(c3_ptr)));
                }

                v_acc0x0123 = _mm_min_epi32(_mm_max_epi32(v_acc0x0123, v_min), v_max);
                v_acc1x0123 = _mm_min_epi32(_mm_max_epi32(v_acc1x0123, v_min), v_max);
                v_acc2x0123 = _mm_min_epi32(_mm_max_epi32(v_acc2x0123, v_min), v_max);
                v_acc3x0123 = _mm_min_epi32(_mm_max_epi32(v_acc3x0123, v_min), v_max);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr), v_acc0x0123);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(c1_ptr), v_acc1x0123);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(c2_ptr), v_acc2x0123);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(c3_ptr), v_acc3x0123);

                c0_ptr += 4;
                c1_ptr += 4;
                c2_ptr += 4;
                c3_ptr += 4;

                v_acc0x0123 = _mm256_extracti128_si256(v_acc0, 1);
                v_acc1x0123 = _mm256_extracti128_si256(v_acc1, 1);
                v_acc2x0123 = _mm256_extracti128_si256(v_acc2, 1);
                v_acc3x0123 = _mm256_extracti128_si256(v_acc3, 1);
            }

            if (rem & 2) {
                const __m128i v_rest0 = v_acc0x0123;
                const __m128i v_rest1 = v_acc1x0123;
                const __m128i v_rest2 = v_acc2x0123;
                const __m128i v_rest3 = v_acc3x0123;

                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_epi32(v_acc0x0123, _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(c0_ptr)));
                    v_acc1x0123 = _mm_add_epi32(v_acc1x0123, _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(c1_ptr)));
                    v_acc2x0123 = _mm_add_epi32(v_acc2x0123, _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(c2_ptr)));
                    v_acc3x0123 = _mm_add_epi32(v_acc3x0123, _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(c3_ptr)));
                }

                v_acc0x0123 = _mm_min_epi32(_mm_max_epi32(v_acc0x0123, v_min), v_max);
                v_acc1x0123 = _mm_min_epi32(_mm_max_epi32(v_acc1x0123, v_min), v_max);
                v_acc2x0123 = _mm_min_epi32(_mm_max_epi32(v_acc2x0123, v_min), v_max);
                v_acc3x0123 = _mm_min_epi32(_mm_max_epi32(v_acc3x0123, v_min), v_max);

                _mm_storel_epi64(reinterpret_cast<__m128i*>(c0_ptr), v_acc0x0123);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c1_ptr), v_acc1x0123);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c2_ptr), v_acc2x0123);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c3_ptr), v_acc3x0123);

                c0_ptr += 2;
                c1_ptr += 2;
                c2_ptr += 2;
                c3_ptr += 2;

                v_acc0x0123 = _mm_srli_si128(v_rest0, 8);
                v_acc1x0123 = _mm_srli_si128(v_rest1, 8);
                v_acc2x0123 = _mm_srli_si128(v_rest2, 8);
                v_acc3x0123 = _mm_srli_si128(v_rest3, 8);
            }

            if (rem & 1) {
                int c0 = _mm_cvtsi128_si32(v_acc0x0123);
                int c1 = _mm_cvtsi128_si32(v_acc1x0123);
                int c2 = _mm_cvtsi128_si32(v_acc2x0123);
                int c3 = _mm_cvtsi128_si32(v_acc3x0123);
                if constexpr (!zero_mode) {
                    c0 += c0_ptr[0];
                    c1 += c1_ptr[0];
                    c2 += c2_ptr[0];
                    c3 += c3_ptr[0];
                }
                c0_ptr[0] = std::min(std::max(c0, clamp_min), clamp_max);
                c1_ptr[0] = std::min(std::max(c1, clamp_min), clamp_max);
                c2_ptr[0] = std::min(std::max(c2, clamp_min), clamp_max);
                c3_ptr[0] = std::min(std::max(c3, clamp_min), clamp_max);
            }
        }
    }
}

// =========================================================================
//  mr = 1, direct A
// =========================================================================

template <bool zero_mode = false>
void mma_packed_b_mr1_nr16_mrxncxkc_i8(
    int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
    const int8_t* NNOPS_RESTRICT A, int lda,
    const int8_t* NNOPS_RESTRICT B,
    int32_t clamp_min, int32_t clamp_max) noexcept {
    (void)mr;

    int32_t* c0_ptr = C;

    const int kg = kc >> 2;
    // Panel slot: 16 columns x kc of data, then the 16 int32 of column sums.
    const int ldb = static_cast<int>(mma_pack_align(16 * kc + mma_i8_b_panel_tail));

    int n = 0;
    for(; n < nc - 15; n += 16) {
        const int8_t* a0_ptr = A;

        __m256i v_acc0x01234567 = mma_i8_bias_fix(B, kc, 0);
        __m256i v_acc0x89abcdef = mma_i8_bias_fix(B, kc, 32);

        const int8_t* b_ptr = B;
        for(int k = 0; k < kg; ++k) {
            const __m256i v_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
            const __m256i v_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 32));
            b_ptr += 64;

            const __m256i v_a0 = mma_i8_bcast_group(a0_ptr);
            v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567, v_a0, v_b0);
            v_acc0x89abcdef = mma_i8_dp256(v_acc0x89abcdef, v_a0, v_b1);

            a0_ptr += 4;
        }
        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_epi32(v_acc0x01234567, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c0_ptr + 0)));
            v_acc0x89abcdef = _mm256_add_epi32(v_acc0x89abcdef, _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(c0_ptr + 8)));
        }
        const __m256i v_min = _mm256_set1_epi32(clamp_min);
        const __m256i v_max = _mm256_set1_epi32(clamp_max);

        v_acc0x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc0x01234567, v_min), v_max);
        v_acc0x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc0x89abcdef, v_min), v_max);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c0_ptr + 0), v_acc0x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c0_ptr + 8), v_acc0x89abcdef);

        c0_ptr += 16;

        B += ldb;
    }

    int rem = nc - n;
    const bool tail_hi = rem > 8;

    if (rem > 0) {
        const int8_t* a0_ptr = A;

        __m256i v_acc0x01234567 = mma_i8_bias_fix(B, kc, 0);
        __m256i v_acc0x89abcdef = mma_i8_bias_fix(B, kc, 32);

        const int8_t* b_ptr = B;
        if (rem > 8) {
            for(int k = 0; k < kg; ++k) {
                const __m256i v_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
                const __m256i v_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 32));
                b_ptr += 64;

                const __m256i v_a0 = mma_i8_bcast_group(a0_ptr);
                v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567, v_a0, v_b0);
                v_acc0x89abcdef = mma_i8_dp256(v_acc0x89abcdef, v_a0, v_b1);

                a0_ptr += 4;
            }
        } else {
            for(int k = 0; k < kg; ++k) {
                const __m256i v_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
                b_ptr += 64;

                v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567,
                    mma_i8_bcast_group(a0_ptr), v_b0);

                a0_ptr += 4;
            }
        }

        if (rem >= 8) {

            if constexpr (!zero_mode) {
                v_acc0x01234567 = _mm256_add_epi32(v_acc0x01234567, _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(c0_ptr)));
            }

            const __m256i v_min = _mm256_set1_epi32(clamp_min);
            const __m256i v_max = _mm256_set1_epi32(clamp_max);

            v_acc0x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc0x01234567, v_min), v_max);

            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c0_ptr), v_acc0x01234567);

            c0_ptr += 8;

            rem -= 8;
        }

        if (rem > 0) {
            const __m256i v_acc0 = tail_hi ? v_acc0x89abcdef : v_acc0x01234567;

            __m128i v_acc0x0123 = _mm256_castsi256_si128(v_acc0);

            const __m128i v_min = _mm_set1_epi32(clamp_min);
            const __m128i v_max = _mm_set1_epi32(clamp_max);

            if (rem & 4) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_epi32(v_acc0x0123, _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(c0_ptr)));
                }

                v_acc0x0123 = _mm_min_epi32(_mm_max_epi32(v_acc0x0123, v_min), v_max);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr), v_acc0x0123);

                c0_ptr += 4;

                v_acc0x0123 = _mm256_extracti128_si256(v_acc0, 1);
            }

            if (rem & 2) {
                const __m128i v_rest0 = v_acc0x0123;

                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_epi32(v_acc0x0123, _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(c0_ptr)));
                }

                v_acc0x0123 = _mm_min_epi32(_mm_max_epi32(v_acc0x0123, v_min), v_max);

                _mm_storel_epi64(reinterpret_cast<__m128i*>(c0_ptr), v_acc0x0123);

                c0_ptr += 2;

                v_acc0x0123 = _mm_srli_si128(v_rest0, 8);
            }

            if (rem & 1) {
                int c0 = _mm_cvtsi128_si32(v_acc0x0123);
                if constexpr (!zero_mode) {
                    c0 += c0_ptr[0];
                }
                c0_ptr[0] = std::min(std::max(c0, clamp_min), clamp_max);
            }
        }
    }
}

// ---- dispatch table ------------------------------------------------------
// Rows are the MR levels {6,4,1} (MR_I8 on x86_64); every kernel takes the
// runtime panel height, so a remainder panel below its own MR runs through the
// level above it. The packed table is what matmul_helper's int8 route walks.
template <bool Z>
inline constexpr auto mma_pack_fns_i8 = std::array{
    &mma_packed_ab_mr6_nr16_mrxncxkc_i8<Z>,
    &mma_packed_ab_mr4_nr16_mrxncxkc_i8<Z>,
    &mma_packed_ab_mr1_nr16_mrxncxkc_i8<Z>};

// Row-major A (`lda` in int8 elements, kc % 4 == 0), same MR levels.
template <bool Z>
inline constexpr auto mma_direct_fns_i8 = std::array{
    &mma_packed_b_mr6_nr16_mrxncxkc_i8<Z>,
    &mma_packed_b_mr4_nr16_mrxncxkc_i8<Z>,
    &mma_packed_b_mr1_nr16_mrxncxkc_i8<Z>};

}  // namespace nnops::backend::cpu::x86_64
