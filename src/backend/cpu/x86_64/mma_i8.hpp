#pragma once
/// @file mma_i8.hpp
/// @brief x86_64 VNNI int8×int8 → int32 mrxncxkc MMA (matrix micro-accumulate)
///        kernels.
///
/// One call covers the whole N side of one M panel: C[mr][nc] += A[mr][kc] ×
/// B[kc][nc] panel by panel, every int32 result clamped to [clamp_min, clamp_max]
/// once the sum is finished. The int8 sibling of mma_f32.hpp / mma_f16.hpp,
/// taking the same runtime panel height `mr` and the same partial-M convention:
/// rows past `mr` reuse the last live row's index and C row (they compute a
/// duplicate and store it back to the same place; the XNNPACK broadcast
/// convention), so a short panel needs no per-row guards. `zero_mode` false
/// accumulates into C, true overwrites it; there is no bias seed argument -- the
/// panel's column sums are the only seed.
///
/// Two A routes, as in the float files; B is always packed, always VNNI order.
///   - `mma_pack_*` takes a packed A: four consecutive k per int32, with
///     the kernel's MR row slots interleaved inside each k group --
///     [ceil(kc/4)][MR] int32s, in a mma_pack_align(MR*kbytes)-byte slot; only
///     the first `mr` slots of a short panel are read.
///   - `mma_direct_*` reads A row-major through `lda` (in int8 elements): the
///     four consecutive k of one row are already the dword dpbusd wants, so the
///     broadcast is one unaligned 4-byte load (no penalty on x86) rather than a
///     packed panel -- but a row-major row has no zero-filled final group, so
///     these require kc % 4 == 0.
///   - B is ceil(nc/16) uniform 16-column panels, each byte-group
///     [ceil(kc/4)][16][4] -- one int32 per column per k group -- followed by
///     16 int32 of column sums. The data half is 16*ceil(kc/4)*4 =
///     64*ceil(kc/4) bytes, itself a multiple of the 64-byte mma_pack_align
///     slot, so the sums block starts at panel + 64*ceil(kc/4) and the panel's
///     slot is mma_pack_align(64*ceil(kc/4) + 64) (see mma_i8_b_panel_tail).
///
/// `ldb` (0 = the natural slot above), the B-side argument that follows the
/// operand it strides, overrides the panel-to-panel step in bytes: a prepacked B
/// need not be one uniform pitch, because its panels can be cut into k-block
/// segments -- [block 0 data][block 0 sums][block 1 data][block 1 sums]... --
/// whose total is the caller's, not the kernel's. What stays fixed is the sums'
/// position *within* a call: each panel's sums are read at that panel's own
/// start + 16*ceil(kc/4), so the segment's sums land where a per-call pack would
/// have put them.
///
/// Panels past `nc` in the last one are zero-filled by the packer, so the tail
/// can always read all 16 columns at the uniform stride and discard the dead
/// ones. `kc` is the number of int8 elements; on the packed route a final
/// partial group (1..3) is consumed as a whole zero-padded group, matching what
/// the packers write. After the whole panels the N tail splits 8/4/2/1 lanes
/// out of the same accumulators -- lane j is column j, so it is a lo/hi split;
/// a tail wider than 8 runs both 8-lane halves in one k loop instead of two,
/// and the 2-wide step keeps an unclamped copy of lanes 2,3 because the 1-wide
/// step still stores them.
///
/// Operand contract: B is int8; A is int8 or uint8, and the kernels return the
/// exact integer dot product either way (what MatMulInteger wants, before
/// zero-point compensation). vpdpbusd is u8×s8, so:
///   - s8 A: the A pack stores every byte biased by +128 (XOR 0x80) and the
///     kernel folds -128·Σ_k B[k][n] back out. The direct-A route has no packer
///     to bias it, so it XORs each broadcast group in the kernel and is s8 only.
///   - u8 A: the A pack stores the bytes as they are; nothing to fold, so the
///     B panel's correction block is zeroed and the accumulators start at zero.
/// That sum is a property of the B panel, not of A, so the B packer writes it
/// once per panel as 16 int32 right after the panel's data (see
/// mma_i8_b_panel_sums), and the kernel starts its accumulators at -128·sum
/// instead of zero, which costs nothing in the k loop.
///
/// Build: the kernels need VNNI -- AVX-VNNI, or AVX512-VNNI with AVX512VL for the
/// 256-bit form. A build with neither cannot run them at all, so mma_i8_dp256 is a
/// hard #error rather than a silent no-op; MSVC has no /arch switch for VNNI, so
/// CMake defines __AVXVNNI__ by hand.
///
/// Tile sizes (x86_64-optimised): M ∈ {6, 4, 1}    N: 16 plus an in-kernel tail.

#include <array>
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <immintrin.h>
#include "backend/cpu/common/common_utils.hpp"

namespace nnops::backend::cpu::x86_64 {

#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
inline __m256i mma_i8_dp256(__m256i acc, __m256i a, __m256i b) noexcept
{
    return _mm256_dpbusd_epi32(acc, a, b);
}
#elif defined(__AVXVNNI__)
inline __m256i mma_i8_dp256(__m256i acc, __m256i a, __m256i b) noexcept
{
    return _mm256_dpbusd_avx_epi32(acc, a, b);
}
#else
#error "mma_i8.hpp requires AVX-VNNI (vpdpbusd)"
#endif

/// A row-major A row's 4-k group, biased into vpdpbusd's unsigned operand.
inline __m256i mma_i8_bcast_group(const int8_t* p) noexcept
{
    return _mm256_set1_epi32(*reinterpret_cast<const int32_t*>(p) ^ 0x80808080);
}

/// Packed-B panel width: the kernels' uniform nr.
inline constexpr int mma_i8_nr_max = 16;

/// Bytes a packed-B panel carries past its payload: one int32 of column sums per column.
inline constexpr int mma_i8_b_panel_tail = mma_i8_nr_max * 4;

/// A packed-B panel's byte pitch: the caller's `ldb` when it gave one (a
/// prepacked blob's own), else the uniform slot `k` int8 k-values take.
inline constexpr int mma_i8_panel_stride(int ldb, int k) noexcept
{
    return (ldb > 0) ? ldb : static_cast<int>(mma_pack_align(16 * k + mma_i8_b_panel_tail));
}

/// The kernels fold the A bias per panel, so matmul_helper's epilogue corrects none of it.
inline constexpr bool mma_i8_folds_a_bias = true;

/// Column sums of one packed-B panel; @p a_biased selects the s8-A sums or a zeroed block.
inline void mma_i8_b_panel_sums(int8_t* panel, int kbytes, bool a_biased) noexcept
{
    __m256i s0 = _mm256_setzero_si256();
    __m256i s1 = _mm256_setzero_si256();
    if (a_biased) {
        const __m256i ones = _mm256_set1_epi8(1);
        const int8_t* p = panel;
        for (int g = kbytes >> 2; g > 0; --g, p += 64) {
            s0 = mma_i8_dp256(s0, ones, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
            s1 = mma_i8_dp256(s1, ones,
                              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32)));
        }
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(panel + 16 * kbytes), s0);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(panel + 16 * kbytes + 32), s1);
}

/// An accumulator's seed for one packed-B panel: -128 × its column sums, or zero; `off` picks the low (0) or high (32) half.
inline __m256i mma_i8_bias_fix(const int8_t* panel, int kbytes, int off) noexcept
{
    const __m256i sums =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(panel + 16 * kbytes + off));
    return _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_slli_epi32(sums, 7));
}

template <bool zero_mode = false>
void mma_pack_mr6_nr16_mrxncxkc_i8(int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
                                   const int8_t* NNOPS_RESTRICT A, const int8_t* NNOPS_RESTRICT B,
                                   int ldb, int32_t clamp_min, int32_t clamp_max) noexcept
{
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
    const int ldb_panel = mma_i8_panel_stride(ldb, kbytes);

    int n = 0;
    for (; n < nc - 15; n += 16) {
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
        for (int k = 0; k < kg; ++k) {
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
            v_acc0x01234567 = _mm256_add_epi32(
                v_acc0x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c0_ptr + 0)));
            v_acc0x89abcdef = _mm256_add_epi32(
                v_acc0x89abcdef, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c0_ptr + 8)));
            v_acc1x01234567 = _mm256_add_epi32(
                v_acc1x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c1_ptr + 0)));
            v_acc1x89abcdef = _mm256_add_epi32(
                v_acc1x89abcdef, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c1_ptr + 8)));
            v_acc2x01234567 = _mm256_add_epi32(
                v_acc2x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c2_ptr + 0)));
            v_acc2x89abcdef = _mm256_add_epi32(
                v_acc2x89abcdef, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c2_ptr + 8)));
            v_acc3x01234567 = _mm256_add_epi32(
                v_acc3x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c3_ptr + 0)));
            v_acc3x89abcdef = _mm256_add_epi32(
                v_acc3x89abcdef, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c3_ptr + 8)));
            v_acc4x01234567 = _mm256_add_epi32(
                v_acc4x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c4_ptr + 0)));
            v_acc4x89abcdef = _mm256_add_epi32(
                v_acc4x89abcdef, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c4_ptr + 8)));
            v_acc5x01234567 = _mm256_add_epi32(
                v_acc5x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c5_ptr + 0)));
            v_acc5x89abcdef = _mm256_add_epi32(
                v_acc5x89abcdef, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c5_ptr + 8)));
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

        B += ldb_panel;
    }

    int rem = nc - n;
    const bool tail_hi = rem > 8;

    if (rem > 0) {
        // A tail wider than 8 reads both halves in one k loop instead of two.
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
            for (int k = 0; k < kg; ++k) {
                const __m256i v_b0 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
                const __m256i v_b1 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 32));
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
            for (int k = 0; k < kg; ++k) {
                const __m256i v_b0 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
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
                v_acc0x01234567 = _mm256_add_epi32(
                    v_acc0x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c0_ptr)));
                v_acc1x01234567 = _mm256_add_epi32(
                    v_acc1x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c1_ptr)));
                v_acc2x01234567 = _mm256_add_epi32(
                    v_acc2x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c2_ptr)));
                v_acc3x01234567 = _mm256_add_epi32(
                    v_acc3x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c3_ptr)));
                v_acc4x01234567 = _mm256_add_epi32(
                    v_acc4x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c4_ptr)));
                v_acc5x01234567 = _mm256_add_epi32(
                    v_acc5x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c5_ptr)));
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
            // Lane j is column j, so the 4/2/1 ladder is a lo/hi split of one
            // accumulator; a tail wider than 8 accumulated in the high half.
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
                    v_acc0x0123 = _mm_add_epi32(
                        v_acc0x0123, _mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr)));
                    v_acc1x0123 = _mm_add_epi32(
                        v_acc1x0123, _mm_loadu_si128(reinterpret_cast<const __m128i*>(c1_ptr)));
                    v_acc2x0123 = _mm_add_epi32(
                        v_acc2x0123, _mm_loadu_si128(reinterpret_cast<const __m128i*>(c2_ptr)));
                    v_acc3x0123 = _mm_add_epi32(
                        v_acc3x0123, _mm_loadu_si128(reinterpret_cast<const __m128i*>(c3_ptr)));
                    v_acc4x0123 = _mm_add_epi32(
                        v_acc4x0123, _mm_loadu_si128(reinterpret_cast<const __m128i*>(c4_ptr)));
                    v_acc5x0123 = _mm_add_epi32(
                        v_acc5x0123, _mm_loadu_si128(reinterpret_cast<const __m128i*>(c5_ptr)));
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
                // The 4-wide clamp below would corrupt lanes 2,3 that the 1-wide step stores.
                const __m128i v_rest0 = v_acc0x0123;
                const __m128i v_rest1 = v_acc1x0123;
                const __m128i v_rest2 = v_acc2x0123;
                const __m128i v_rest3 = v_acc3x0123;
                const __m128i v_rest4 = v_acc4x0123;
                const __m128i v_rest5 = v_acc5x0123;

                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_epi32(
                        v_acc0x0123, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0_ptr)));
                    v_acc1x0123 = _mm_add_epi32(
                        v_acc1x0123, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c1_ptr)));
                    v_acc2x0123 = _mm_add_epi32(
                        v_acc2x0123, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c2_ptr)));
                    v_acc3x0123 = _mm_add_epi32(
                        v_acc3x0123, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c3_ptr)));
                    v_acc4x0123 = _mm_add_epi32(
                        v_acc4x0123, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c4_ptr)));
                    v_acc5x0123 = _mm_add_epi32(
                        v_acc5x0123, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c5_ptr)));
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

template <bool zero_mode = false>
void mma_pack_mr4_nr16_mrxncxkc_i8(int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
                                   const int8_t* NNOPS_RESTRICT A, const int8_t* NNOPS_RESTRICT B,
                                   int ldb, int32_t clamp_min, int32_t clamp_max) noexcept
{
    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;

    int32_t* c0_ptr = C + 0 * ldc;
    int32_t* c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    int32_t* c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    int32_t* c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;

    const int kbytes = (kc + 3) & ~3;
    const int kg = kbytes >> 2;
    const int ldb_panel = mma_i8_panel_stride(ldb, kbytes);

    int n = 0;
    for (; n < nc - 15; n += 16) {
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
        for (int k = 0; k < kg; ++k) {
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
            v_acc0x01234567 = _mm256_add_epi32(
                v_acc0x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c0_ptr + 0)));
            v_acc0x89abcdef = _mm256_add_epi32(
                v_acc0x89abcdef, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c0_ptr + 8)));
            v_acc1x01234567 = _mm256_add_epi32(
                v_acc1x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c1_ptr + 0)));
            v_acc1x89abcdef = _mm256_add_epi32(
                v_acc1x89abcdef, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c1_ptr + 8)));
            v_acc2x01234567 = _mm256_add_epi32(
                v_acc2x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c2_ptr + 0)));
            v_acc2x89abcdef = _mm256_add_epi32(
                v_acc2x89abcdef, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c2_ptr + 8)));
            v_acc3x01234567 = _mm256_add_epi32(
                v_acc3x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c3_ptr + 0)));
            v_acc3x89abcdef = _mm256_add_epi32(
                v_acc3x89abcdef, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c3_ptr + 8)));
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

        B += ldb_panel;
    }

    int rem = nc - n;
    const bool tail_hi = rem > 8;

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
            for (int k = 0; k < kg; ++k) {
                const __m256i v_b0 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
                const __m256i v_b1 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 32));
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
            for (int k = 0; k < kg; ++k) {
                const __m256i v_b0 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
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
                v_acc0x01234567 = _mm256_add_epi32(
                    v_acc0x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c0_ptr)));
                v_acc1x01234567 = _mm256_add_epi32(
                    v_acc1x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c1_ptr)));
                v_acc2x01234567 = _mm256_add_epi32(
                    v_acc2x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c2_ptr)));
                v_acc3x01234567 = _mm256_add_epi32(
                    v_acc3x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c3_ptr)));
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
                    v_acc0x0123 = _mm_add_epi32(
                        v_acc0x0123, _mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr)));
                    v_acc1x0123 = _mm_add_epi32(
                        v_acc1x0123, _mm_loadu_si128(reinterpret_cast<const __m128i*>(c1_ptr)));
                    v_acc2x0123 = _mm_add_epi32(
                        v_acc2x0123, _mm_loadu_si128(reinterpret_cast<const __m128i*>(c2_ptr)));
                    v_acc3x0123 = _mm_add_epi32(
                        v_acc3x0123, _mm_loadu_si128(reinterpret_cast<const __m128i*>(c3_ptr)));
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
                    v_acc0x0123 = _mm_add_epi32(
                        v_acc0x0123, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0_ptr)));
                    v_acc1x0123 = _mm_add_epi32(
                        v_acc1x0123, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c1_ptr)));
                    v_acc2x0123 = _mm_add_epi32(
                        v_acc2x0123, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c2_ptr)));
                    v_acc3x0123 = _mm_add_epi32(
                        v_acc3x0123, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c3_ptr)));
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

template <bool zero_mode = false>
void mma_pack_mr1_nr16_mrxncxkc_i8(int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
                                   const int8_t* NNOPS_RESTRICT A, const int8_t* NNOPS_RESTRICT B,
                                   int ldb, int32_t clamp_min, int32_t clamp_max) noexcept
{
    (void)mr;

    int32_t* c0_ptr = C;

    const int kbytes = (kc + 3) & ~3;
    const int kg = kbytes >> 2;
    const int ldb_panel = mma_i8_panel_stride(ldb, kbytes);

    int n = 0;
    for (; n < nc - 15; n += 16) {
        __m256i v_acc0x01234567 = mma_i8_bias_fix(B, kbytes, 0);
        __m256i v_acc0x89abcdef = mma_i8_bias_fix(B, kbytes, 32);

        const int32_t* a_i32 = reinterpret_cast<const int32_t*>(A);
        const int8_t* b_ptr = B;
        for (int k = 0; k < kg; ++k) {
            const __m256i v_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
            const __m256i v_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 32));
            b_ptr += 64;

            const __m256i v_a0 = _mm256_set1_epi32(a_i32[0]);
            v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567, v_a0, v_b0);
            v_acc0x89abcdef = mma_i8_dp256(v_acc0x89abcdef, v_a0, v_b1);

            a_i32 += 1;
        }
        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_epi32(
                v_acc0x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c0_ptr + 0)));
            v_acc0x89abcdef = _mm256_add_epi32(
                v_acc0x89abcdef, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c0_ptr + 8)));
        }
        const __m256i v_min = _mm256_set1_epi32(clamp_min);
        const __m256i v_max = _mm256_set1_epi32(clamp_max);

        v_acc0x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc0x01234567, v_min), v_max);
        v_acc0x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc0x89abcdef, v_min), v_max);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c0_ptr + 0), v_acc0x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c0_ptr + 8), v_acc0x89abcdef);

        c0_ptr += 16;

        B += ldb_panel;
    }

    int rem = nc - n;
    const bool tail_hi = rem > 8;

    if (rem > 0) {
        __m256i v_acc0x01234567 = mma_i8_bias_fix(B, kbytes, 0);
        __m256i v_acc0x89abcdef = mma_i8_bias_fix(B, kbytes, 32);

        const int32_t* a_i32 = reinterpret_cast<const int32_t*>(A);
        const int8_t* b_ptr = B;
        if (rem > 8) {
            for (int k = 0; k < kg; ++k) {
                const __m256i v_b0 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
                const __m256i v_b1 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 32));
                b_ptr += 64;

                const __m256i v_a0 = _mm256_set1_epi32(a_i32[0]);
                v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567, v_a0, v_b0);
                v_acc0x89abcdef = mma_i8_dp256(v_acc0x89abcdef, v_a0, v_b1);

                a_i32 += 1;
            }
        } else {
            for (int k = 0; k < kg; ++k) {
                const __m256i v_b0 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
                b_ptr += 64;

                v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567, _mm256_set1_epi32(a_i32[0]), v_b0);

                a_i32 += 1;
            }
        }

        if (rem >= 8) {
            if constexpr (!zero_mode) {
                v_acc0x01234567 = _mm256_add_epi32(
                    v_acc0x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c0_ptr)));
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
                    v_acc0x0123 = _mm_add_epi32(
                        v_acc0x0123, _mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr)));
                }

                v_acc0x0123 = _mm_min_epi32(_mm_max_epi32(v_acc0x0123, v_min), v_max);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr), v_acc0x0123);

                c0_ptr += 4;

                v_acc0x0123 = _mm256_extracti128_si256(v_acc0, 1);
            }

            if (rem & 2) {
                const __m128i v_rest0 = v_acc0x0123;

                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_epi32(
                        v_acc0x0123, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0_ptr)));
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

template <bool zero_mode = false>
void mma_direct_mr6_nr16_mrxncxkc_i8(int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
                                     const int8_t* NNOPS_RESTRICT A, int lda,
                                     const int8_t* NNOPS_RESTRICT B, int ldb, int32_t clamp_min,
                                     int32_t clamp_max) noexcept
{
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
    const int ldb_panel = mma_i8_panel_stride(ldb, kc);

    int n = 0;
    for (; n < nc - 15; n += 16) {
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
        for (int k = 0; k < kg; ++k) {
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
            v_acc0x01234567 = _mm256_add_epi32(
                v_acc0x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c0_ptr + 0)));
            v_acc0x89abcdef = _mm256_add_epi32(
                v_acc0x89abcdef, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c0_ptr + 8)));
            v_acc1x01234567 = _mm256_add_epi32(
                v_acc1x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c1_ptr + 0)));
            v_acc1x89abcdef = _mm256_add_epi32(
                v_acc1x89abcdef, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c1_ptr + 8)));
            v_acc2x01234567 = _mm256_add_epi32(
                v_acc2x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c2_ptr + 0)));
            v_acc2x89abcdef = _mm256_add_epi32(
                v_acc2x89abcdef, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c2_ptr + 8)));
            v_acc3x01234567 = _mm256_add_epi32(
                v_acc3x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c3_ptr + 0)));
            v_acc3x89abcdef = _mm256_add_epi32(
                v_acc3x89abcdef, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c3_ptr + 8)));
            v_acc4x01234567 = _mm256_add_epi32(
                v_acc4x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c4_ptr + 0)));
            v_acc4x89abcdef = _mm256_add_epi32(
                v_acc4x89abcdef, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c4_ptr + 8)));
            v_acc5x01234567 = _mm256_add_epi32(
                v_acc5x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c5_ptr + 0)));
            v_acc5x89abcdef = _mm256_add_epi32(
                v_acc5x89abcdef, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c5_ptr + 8)));
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

        B += ldb_panel;
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
            for (int k = 0; k < kg; ++k) {
                const __m256i v_b0 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
                const __m256i v_b1 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 32));
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
            for (int k = 0; k < kg; ++k) {
                const __m256i v_b0 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
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
                v_acc0x01234567 = _mm256_add_epi32(
                    v_acc0x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c0_ptr)));
                v_acc1x01234567 = _mm256_add_epi32(
                    v_acc1x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c1_ptr)));
                v_acc2x01234567 = _mm256_add_epi32(
                    v_acc2x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c2_ptr)));
                v_acc3x01234567 = _mm256_add_epi32(
                    v_acc3x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c3_ptr)));
                v_acc4x01234567 = _mm256_add_epi32(
                    v_acc4x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c4_ptr)));
                v_acc5x01234567 = _mm256_add_epi32(
                    v_acc5x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c5_ptr)));
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
                    v_acc0x0123 = _mm_add_epi32(
                        v_acc0x0123, _mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr)));
                    v_acc1x0123 = _mm_add_epi32(
                        v_acc1x0123, _mm_loadu_si128(reinterpret_cast<const __m128i*>(c1_ptr)));
                    v_acc2x0123 = _mm_add_epi32(
                        v_acc2x0123, _mm_loadu_si128(reinterpret_cast<const __m128i*>(c2_ptr)));
                    v_acc3x0123 = _mm_add_epi32(
                        v_acc3x0123, _mm_loadu_si128(reinterpret_cast<const __m128i*>(c3_ptr)));
                    v_acc4x0123 = _mm_add_epi32(
                        v_acc4x0123, _mm_loadu_si128(reinterpret_cast<const __m128i*>(c4_ptr)));
                    v_acc5x0123 = _mm_add_epi32(
                        v_acc5x0123, _mm_loadu_si128(reinterpret_cast<const __m128i*>(c5_ptr)));
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
                    v_acc0x0123 = _mm_add_epi32(
                        v_acc0x0123, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0_ptr)));
                    v_acc1x0123 = _mm_add_epi32(
                        v_acc1x0123, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c1_ptr)));
                    v_acc2x0123 = _mm_add_epi32(
                        v_acc2x0123, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c2_ptr)));
                    v_acc3x0123 = _mm_add_epi32(
                        v_acc3x0123, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c3_ptr)));
                    v_acc4x0123 = _mm_add_epi32(
                        v_acc4x0123, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c4_ptr)));
                    v_acc5x0123 = _mm_add_epi32(
                        v_acc5x0123, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c5_ptr)));
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

template <bool zero_mode = false>
void mma_direct_mr4_nr16_mrxncxkc_i8(int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
                                     const int8_t* NNOPS_RESTRICT A, int lda,
                                     const int8_t* NNOPS_RESTRICT B, int ldb, int32_t clamp_min,
                                     int32_t clamp_max) noexcept
{
    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;

    int32_t* c0_ptr = C + 0 * ldc;
    int32_t* c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    int32_t* c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    int32_t* c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;

    const int kg = kc >> 2;
    const int ldb_panel = mma_i8_panel_stride(ldb, kc);

    int n = 0;
    for (; n < nc - 15; n += 16) {
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
        for (int k = 0; k < kg; ++k) {
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
            v_acc0x01234567 = _mm256_add_epi32(
                v_acc0x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c0_ptr + 0)));
            v_acc0x89abcdef = _mm256_add_epi32(
                v_acc0x89abcdef, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c0_ptr + 8)));
            v_acc1x01234567 = _mm256_add_epi32(
                v_acc1x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c1_ptr + 0)));
            v_acc1x89abcdef = _mm256_add_epi32(
                v_acc1x89abcdef, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c1_ptr + 8)));
            v_acc2x01234567 = _mm256_add_epi32(
                v_acc2x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c2_ptr + 0)));
            v_acc2x89abcdef = _mm256_add_epi32(
                v_acc2x89abcdef, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c2_ptr + 8)));
            v_acc3x01234567 = _mm256_add_epi32(
                v_acc3x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c3_ptr + 0)));
            v_acc3x89abcdef = _mm256_add_epi32(
                v_acc3x89abcdef, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c3_ptr + 8)));
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

        B += ldb_panel;
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
            for (int k = 0; k < kg; ++k) {
                const __m256i v_b0 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
                const __m256i v_b1 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 32));
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
            for (int k = 0; k < kg; ++k) {
                const __m256i v_b0 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
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
                v_acc0x01234567 = _mm256_add_epi32(
                    v_acc0x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c0_ptr)));
                v_acc1x01234567 = _mm256_add_epi32(
                    v_acc1x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c1_ptr)));
                v_acc2x01234567 = _mm256_add_epi32(
                    v_acc2x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c2_ptr)));
                v_acc3x01234567 = _mm256_add_epi32(
                    v_acc3x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c3_ptr)));
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
                    v_acc0x0123 = _mm_add_epi32(
                        v_acc0x0123, _mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr)));
                    v_acc1x0123 = _mm_add_epi32(
                        v_acc1x0123, _mm_loadu_si128(reinterpret_cast<const __m128i*>(c1_ptr)));
                    v_acc2x0123 = _mm_add_epi32(
                        v_acc2x0123, _mm_loadu_si128(reinterpret_cast<const __m128i*>(c2_ptr)));
                    v_acc3x0123 = _mm_add_epi32(
                        v_acc3x0123, _mm_loadu_si128(reinterpret_cast<const __m128i*>(c3_ptr)));
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
                    v_acc0x0123 = _mm_add_epi32(
                        v_acc0x0123, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0_ptr)));
                    v_acc1x0123 = _mm_add_epi32(
                        v_acc1x0123, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c1_ptr)));
                    v_acc2x0123 = _mm_add_epi32(
                        v_acc2x0123, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c2_ptr)));
                    v_acc3x0123 = _mm_add_epi32(
                        v_acc3x0123, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c3_ptr)));
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

template <bool zero_mode = false>
void mma_direct_mr1_nr16_mrxncxkc_i8(int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C, int ldc,
                                     const int8_t* NNOPS_RESTRICT A, int lda,
                                     const int8_t* NNOPS_RESTRICT B, int ldb, int32_t clamp_min,
                                     int32_t clamp_max) noexcept
{
    (void)mr;

    int32_t* c0_ptr = C;

    const int kg = kc >> 2;
    const int ldb_panel = mma_i8_panel_stride(ldb, kc);

    int n = 0;
    for (; n < nc - 15; n += 16) {
        const int8_t* a0_ptr = A;

        __m256i v_acc0x01234567 = mma_i8_bias_fix(B, kc, 0);
        __m256i v_acc0x89abcdef = mma_i8_bias_fix(B, kc, 32);

        const int8_t* b_ptr = B;
        for (int k = 0; k < kg; ++k) {
            const __m256i v_b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
            const __m256i v_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 32));
            b_ptr += 64;

            const __m256i v_a0 = mma_i8_bcast_group(a0_ptr);
            v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567, v_a0, v_b0);
            v_acc0x89abcdef = mma_i8_dp256(v_acc0x89abcdef, v_a0, v_b1);

            a0_ptr += 4;
        }
        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_epi32(
                v_acc0x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c0_ptr + 0)));
            v_acc0x89abcdef = _mm256_add_epi32(
                v_acc0x89abcdef, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c0_ptr + 8)));
        }
        const __m256i v_min = _mm256_set1_epi32(clamp_min);
        const __m256i v_max = _mm256_set1_epi32(clamp_max);

        v_acc0x01234567 = _mm256_min_epi32(_mm256_max_epi32(v_acc0x01234567, v_min), v_max);
        v_acc0x89abcdef = _mm256_min_epi32(_mm256_max_epi32(v_acc0x89abcdef, v_min), v_max);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c0_ptr + 0), v_acc0x01234567);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c0_ptr + 8), v_acc0x89abcdef);

        c0_ptr += 16;

        B += ldb_panel;
    }

    int rem = nc - n;
    const bool tail_hi = rem > 8;

    if (rem > 0) {
        const int8_t* a0_ptr = A;

        __m256i v_acc0x01234567 = mma_i8_bias_fix(B, kc, 0);
        __m256i v_acc0x89abcdef = mma_i8_bias_fix(B, kc, 32);

        const int8_t* b_ptr = B;
        if (rem > 8) {
            for (int k = 0; k < kg; ++k) {
                const __m256i v_b0 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
                const __m256i v_b1 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 32));
                b_ptr += 64;

                const __m256i v_a0 = mma_i8_bcast_group(a0_ptr);
                v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567, v_a0, v_b0);
                v_acc0x89abcdef = mma_i8_dp256(v_acc0x89abcdef, v_a0, v_b1);

                a0_ptr += 4;
            }
        } else {
            for (int k = 0; k < kg; ++k) {
                const __m256i v_b0 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
                b_ptr += 64;

                v_acc0x01234567 = mma_i8_dp256(v_acc0x01234567, mma_i8_bcast_group(a0_ptr), v_b0);

                a0_ptr += 4;
            }
        }

        if (rem >= 8) {
            if constexpr (!zero_mode) {
                v_acc0x01234567 = _mm256_add_epi32(
                    v_acc0x01234567, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c0_ptr)));
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
                    v_acc0x0123 = _mm_add_epi32(
                        v_acc0x0123, _mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr)));
                }

                v_acc0x0123 = _mm_min_epi32(_mm_max_epi32(v_acc0x0123, v_min), v_max);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr), v_acc0x0123);

                c0_ptr += 4;

                v_acc0x0123 = _mm256_extracti128_si256(v_acc0, 1);
            }

            if (rem & 2) {
                const __m128i v_rest0 = v_acc0x0123;

                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_epi32(
                        v_acc0x0123, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0_ptr)));
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
//  the requantizing variants (packed A + packed B)
// =========================================================================
//
// The same accumulators written as the destination's bytes instead of int32:
//
//   out = sat(round_away((acc + bias[n]) * (s_a/s_out * s_b[n])) + zp_out)
//
// One call covers the whole K, so no int32 tile reaches memory. The math is
// int8_epilogue_row_quant's x86 path verbatim — clamp the float to the
// destination's range first, round away from zero by adding ±0.5 before a
// truncating convert, then the two saturating packs — so a fused store and a
// separate row pass agree bit for bit. `col_bias` is the per-column int32 bias
// with the caller's shift folded in; on this arch the panel's `-128·Σb` seed is
// already the whole A-side correction, so nothing else joins it here.

/// std::round semantics in f32 — nearest, ties away from zero (AVX2 only has
/// ties-to-even).
inline __m256 i8_round_away_ps(__m256 x) noexcept
{
    const __m256 half = _mm256_or_ps(_mm256_and_ps(x, _mm256_set1_ps(-0.0f)),
                                     _mm256_set1_ps(0.5f));
    return _mm256_add_ps(x, half);
}

/// The per-column scale and bias of one eight-column group, zeroing the dead
/// lanes of a ragged last group instead of reading past the table.
template <int NG>
__attribute__((always_inline)) inline void
i8_col_q_x86(__m256 (&sc)[NG], __m256i (&bi)[NG], __m256 vk, __m256 vreq, const float* s_b,
             const int32_t* col_bias, int live) noexcept
{
    for (int g = 0; g < NG; ++g) {
        const int col = 8 * g;
        if (s_b == nullptr) {
            sc[g] = vreq;
        } else if (col + 8 <= live) {
            sc[g] = _mm256_mul_ps(vk, _mm256_loadu_ps(s_b + col));
        } else {
            alignas(32) float sv[8] = {};
            for (int c = 0; c < 8 && col + c < live; ++c) {
                sv[c] = s_b[col + c];
            }
            sc[g] = _mm256_mul_ps(vk, _mm256_loadu_ps(sv));
        }
        if (col_bias == nullptr) {
            bi[g] = _mm256_setzero_si256();
        } else if (col + 8 <= live) {
            bi[g] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(col_bias + col));
        } else {
            alignas(32) int32_t bv[8] = {};
            for (int c = 0; c < 8 && col + c < live; ++c) {
                bv[c] = col_bias[col + c];
            }
            bi[g] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bv));
        }
    }
}

/// Eight columns of one row: bias, Relu, scale, clamp, round, offset, narrow,
/// store — or the `live` leading bytes of it.
template <class TOut>
__attribute__((always_inline)) inline void
i8_storeq_x86(TOut* dst, int live, __m256i v, __m256 vscale, __m256i vbias, __m256i vlo,
              __m256i vzp, __m256 vfmin, __m256 vfmax) noexcept
{
    v = _mm256_add_epi32(v, vbias);
    v = _mm256_max_epi32(v, vlo);
    __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(v), vscale);
    f = _mm256_min_ps(_mm256_max_ps(f, vfmin), vfmax);
    const __m256i q = _mm256_add_epi32(_mm256_cvttps_epi32(i8_round_away_ps(f)), vzp);
    const __m128i i16 =
        _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
    const __m128i v8 = std::is_same_v<TOut, uint8_t> ? _mm_packus_epi16(i16, i16)
                                                     : _mm_packs_epi16(i16, i16);
    if (live >= 8) {
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dst), v8);
    } else {
        alignas(16) TOut buf[16];
        _mm_store_si128(reinterpret_cast<__m128i*>(buf), v8);
        for (int c = 0; c < live; ++c) {
            dst[c] = buf[c];
        }
    }
}

/// The panel's whole K into `acc`, seeded from the B panel's column sums.
template <int MR, int NH>
__attribute__((always_inline)) inline void
i8_x86_mma(const int8_t* A, const int8_t* B, int kbytes, int kg,
           __m256i (&acc)[MR][NH]) noexcept
{
    const __m256i seed0 = mma_i8_bias_fix(B, kbytes, 0);
    __m256i seed1 = seed0;
    if constexpr (NH == 2) {
        seed1 = mma_i8_bias_fix(B, kbytes, 32);
    }
    for (int r = 0; r < MR; ++r) {
        acc[r][0] = seed0;
        if constexpr (NH == 2) {
            acc[r][1] = seed1;
        }
    }
    const int32_t* a_i32 = reinterpret_cast<const int32_t*>(A);
    const int8_t* b_ptr = B;
    for (int k = 0; k < kg; ++k) {
        const __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
        __m256i vb1 = vb0;
        if constexpr (NH == 2) {
            vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 32));
        }
        b_ptr += 64;
        for (int r = 0; r < MR; ++r) {
            const __m256i va = _mm256_set1_epi32(a_i32[r]);
            acc[r][0] = mma_i8_dp256(acc[r][0], va, vb0);
            if constexpr (NH == 2) {
                acc[r][1] = mma_i8_dp256(acc[r][1], va, vb1);
            }
        }
        a_i32 += MR;
    }
}

template <int MR, class TOut>
__attribute__((always_inline)) inline void
mma_pack_mr_nr16_epi_i8(int mr, int nc, int kc, TOut* NNOPS_RESTRICT C, int64_t ldm,
                        const int8_t* NNOPS_RESTRICT A, const int8_t* NNOPS_RESTRICT B, int ldb,
                        const float* s_b, float s_b_scalar, float row_k,
                        const int32_t* col_bias, int32_t zp_out, bool relu) noexcept
{
    TOut* row[MR];
    for (int r = 0; r < MR; ++r) {
        // Rows past the live count alias the last live one, as the int32 store does.
        row[r] = C + std::min(r, mr - 1) * ldm;
    }
    const int kbytes = (kc + 3) & ~3;
    const int kg = kbytes >> 2;
    const int ldb_panel = mma_i8_panel_stride(ldb, kbytes);
    constexpr float kLo = std::is_same_v<TOut, uint8_t> ? 0.0f : -128.0f;
    constexpr float kHi = std::is_same_v<TOut, uint8_t> ? 255.0f : 127.0f;
    const __m256 vk = _mm256_set1_ps(row_k);
    const __m256 vreq = _mm256_set1_ps(row_k * s_b_scalar);
    const float zpf = static_cast<float>(zp_out);
    const __m256 vfmin = _mm256_set1_ps(kLo - zpf);
    const __m256 vfmax = _mm256_set1_ps(kHi - zpf);
    const __m256i vlo = _mm256_set1_epi32(relu ? 0 : INT32_MIN);
    const __m256i vzp = _mm256_set1_epi32(zp_out);

    int n = 0;
    for (; n + mma_i8_nr_max <= nc; n += mma_i8_nr_max) {
        __m256i acc[MR][2];
        i8_x86_mma<MR, 2>(A, B, kbytes, kg, acc);
        __m256 sc[2];
        __m256i bi[2];
        i8_col_q_x86<2>(sc, bi, vk, vreq, s_b ? s_b + n : nullptr,
                        col_bias ? col_bias + n : nullptr, 16);
        for (int r = 0; r < MR; ++r) {
            i8_storeq_x86<TOut>(row[r] + n, 8, acc[r][0], sc[0], bi[0], vlo, vzp, vfmin, vfmax);
            i8_storeq_x86<TOut>(row[r] + n + 8, 8, acc[r][1], sc[1], bi[1], vlo, vzp, vfmin,
                                vfmax);
        }
        B += ldb_panel;
    }

    const int live = nc - n;
    if (live > 8) {
        __m256i acc[MR][2];
        i8_x86_mma<MR, 2>(A, B, kbytes, kg, acc);
        __m256 sc[2];
        __m256i bi[2];
        i8_col_q_x86<2>(sc, bi, vk, vreq, s_b ? s_b + n : nullptr,
                        col_bias ? col_bias + n : nullptr, live);
        for (int r = 0; r < MR; ++r) {
            i8_storeq_x86<TOut>(row[r] + n, 8, acc[r][0], sc[0], bi[0], vlo, vzp, vfmin, vfmax);
            i8_storeq_x86<TOut>(row[r] + n + 8, live - 8, acc[r][1], sc[1], bi[1], vlo, vzp,
                                vfmin, vfmax);
        }
    } else if (live > 0) {
        // One eight-column half is all the remainder can use.
        __m256i acc[MR][1];
        i8_x86_mma<MR, 1>(A, B, kbytes, kg, acc);
        __m256 sc[1];
        __m256i bi[1];
        i8_col_q_x86<1>(sc, bi, vk, vreq, s_b ? s_b + n : nullptr,
                        col_bias ? col_bias + n : nullptr, live);
        for (int r = 0; r < MR; ++r) {
            i8_storeq_x86<TOut>(row[r] + n, live, acc[r][0], sc[0], bi[0], vlo, vzp, vfmin,
                                vfmax);
        }
    }
}

template <class TOut>
__attribute__((always_inline)) inline void
mma_pack_mr6_nr16_epi_i8(int mr, int nc, int kc, TOut* NNOPS_RESTRICT C, int64_t ldm,
                         const int8_t* NNOPS_RESTRICT A, const int8_t* NNOPS_RESTRICT B,
                         int ldb, const float* s_b, float s_b_scalar, float row_k,
                         const int32_t* col_bias, int32_t zp_out, bool relu) noexcept
{
    mma_pack_mr_nr16_epi_i8<6, TOut>(mr, nc, kc, C, ldm, A, B, ldb, s_b, s_b_scalar, row_k,
                                     col_bias, zp_out, relu);
}

template <class TOut>
__attribute__((always_inline)) inline void
mma_pack_mr4_nr16_epi_i8(int mr, int nc, int kc, TOut* NNOPS_RESTRICT C, int64_t ldm,
                         const int8_t* NNOPS_RESTRICT A, const int8_t* NNOPS_RESTRICT B,
                         int ldb, const float* s_b, float s_b_scalar, float row_k,
                         const int32_t* col_bias, int32_t zp_out, bool relu) noexcept
{
    mma_pack_mr_nr16_epi_i8<4, TOut>(mr, nc, kc, C, ldm, A, B, ldb, s_b, s_b_scalar, row_k,
                                     col_bias, zp_out, relu);
}

template <class TOut>
__attribute__((always_inline)) inline void
mma_pack_mr1_nr16_epi_i8(int mr, int nc, int kc, TOut* NNOPS_RESTRICT C, int64_t ldm,
                         const int8_t* NNOPS_RESTRICT A, const int8_t* NNOPS_RESTRICT B,
                         int ldb, const float* s_b, float s_b_scalar, float row_k,
                         const int32_t* col_bias, int32_t zp_out, bool relu) noexcept
{
    mma_pack_mr_nr16_epi_i8<1, TOut>(mr, nc, kc, C, ldm, A, B, ldb, s_b, s_b_scalar, row_k,
                                     col_bias, zp_out, relu);
}

/// The kernel signature the requantizing rungs are dispatched through.
template <class TOut>
using MmaI8EpiFn = void (*)(int, int, int, TOut*, int64_t, const int8_t*, const int8_t*, int,
                            const float*, float, float, const int32_t*, int32_t, bool);

/// The requantizing rungs, consumed by matmul_helper.cpp; rows are {6,4,1}.
template <class TOut>
inline constexpr std::array<MmaI8EpiFn<TOut>, 3> mma_pack_epi_fns_i8 = {
    &mma_pack_mr6_nr16_epi_i8<TOut>,
    &mma_pack_mr4_nr16_epi_i8<TOut>,
    &mma_pack_mr1_nr16_epi_i8<TOut>,
};

/// MR levels {6,4,1} (MR_I8); a remainder panel runs through the level above it.
template <bool Z>
inline constexpr auto mma_pack_fns_i8 =
    std::array{&mma_pack_mr6_nr16_mrxncxkc_i8<Z>, &mma_pack_mr4_nr16_mrxncxkc_i8<Z>,
               &mma_pack_mr1_nr16_mrxncxkc_i8<Z>};

/// Row-major A (`lda` in int8 elements, kc % 4 == 0), same MR levels.
template <bool Z>
inline constexpr auto mma_direct_fns_i8 =
    std::array{&mma_direct_mr6_nr16_mrxncxkc_i8<Z>, &mma_direct_mr4_nr16_mrxncxkc_i8<Z>,
               &mma_direct_mr1_nr16_mrxncxkc_i8<Z>};

} // namespace nnops::backend::cpu::x86_64
