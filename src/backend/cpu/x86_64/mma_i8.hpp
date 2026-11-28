#pragma once
/// @file mma_i8.hpp
/// @brief x86_64 VNNI int8×int8 → int32 mrxncxkc MMA kernels over uniform
///        16-wide B.
///
/// One templated rung per A presentation, with the height a template parameter:
/// `MR` ∈ {6, 4, 1} is the panel's row count, while the calls' `mr` (1..MR) is
/// the live height the M decomposition emits, so each level serves every panel
/// the largest-first walk produces instead of needing a kernel per height.
/// A and B may both be packed (`mma_pack_*`) or only B packed while A is read
/// row-major through `lda` (`mma_direct_*`, kc % 4 == 0). `zero_mode` drops the
/// C read-add. A packed-A panel is [ceil(kbytes/4)][MR] int32s — four
/// consecutive k per element, one MR-wide slot per k group.
///
/// B is ceil(nc/16) uniform 16-column panels, each byte-group
/// [ceil(kbytes/4)][16][4] followed by 16 int32 of column sums, in a
/// mma_i8_panel_stride(ldb, kbytes)-byte slot. Each k group is two 8-column
/// halves, which is why the accumulators are [NG][MR] with NG ∈ {1, 2}: a tail
/// wider than 8 runs both halves in one k loop, a tail of 8 or less the single
/// half, and only the lanes it stores are written.
///
/// After the whole panels the N tail splits 8/4/2/1 lanes out of the same
/// accumulators -- lane j is column j, so it is a lo/hi split; the 2-wide step
/// keeps an unclamped copy of lanes 2,3 because the 1-wide step still stores
/// them.
///
/// Rows past `mr` alias the last live row: the A panel repeats the last live
/// row's slot (the packer's convention, and the kernel indexes A by the alias
/// index for it), and the C row pointers repeat here, so a partial-M tile needs
/// no per-row guards.
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
///
/// Every index into the accumulator block is a literal: the group walk and the
/// store are written out at the panel's two widths instead of looping to a
/// template bound. A loop costs nothing where the compiler unrolls it and
/// everything where it does not -- the array is then indexed at run time and
/// the whole block moves to the stack.

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
NNOPS_FORCEINLINE __m256i mma_i8_bcast_group(const int8_t* p) noexcept
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

// =========================================================================
//  one walk: the seed, the K loop, the stores
// =========================================================================
//
// Every rung -- packed A or row-major, any height -- is the same walk: one k
// group of MR rows against NG eight-column halves, then a store. The
// accumulators are laid out [NG][MR] so a half's rows are contiguous, and every
// index into them is a literal (see the header).

/// The alias index of each row: rows past `mr` reuse the last live one.
template <int MR>
NNOPS_FORCEINLINE void i8_row_alias(int mr, int (&rr)[MR]) noexcept
{
    rr[0] = 0;
    if constexpr (MR > 1) {
        rr[1] = std::min(1, mr - 1);
    }
    if constexpr (MR > 2) {
        rr[2] = std::min(2, mr - 1);
    }
    if constexpr (MR > 3) {
        rr[3] = std::min(3, mr - 1);
    }
    if constexpr (MR > 4) {
        rr[4] = std::min(4, mr - 1);
    }
    if constexpr (MR > 5) {
        rr[5] = std::min(5, mr - 1);
    }
}

/// The C rows, aliasing the last live one.
template <int MR>
NNOPS_FORCEINLINE void i8_row_ptrs(int32_t* C, int ldc, const int (&rr)[MR],
                                   int32_t* (&row)[MR]) noexcept
{
    row[0] = C;
    if constexpr (MR > 1) {
        row[1] = C + rr[1] * ldc;
    }
    if constexpr (MR > 2) {
        row[2] = C + rr[2] * ldc;
    }
    if constexpr (MR > 3) {
        row[3] = C + rr[3] * ldc;
    }
    if constexpr (MR > 4) {
        row[4] = C + rr[4] * ldc;
    }
    if constexpr (MR > 5) {
        row[5] = C + rr[5] * ldc;
    }
}

/// The row-major A rows, aliasing the last live one.
template <int MR>
NNOPS_FORCEINLINE void i8_a_ptrs(const int8_t* A, int lda, const int (&rr)[MR],
                                 const int8_t* (&a)[MR]) noexcept
{
    a[0] = A;
    if constexpr (MR > 1) {
        a[1] = A + rr[1] * lda;
    }
    if constexpr (MR > 2) {
        a[2] = A + rr[2] * lda;
    }
    if constexpr (MR > 3) {
        a[3] = A + rr[3] * lda;
    }
    if constexpr (MR > 4) {
        a[4] = A + rr[4] * lda;
    }
    if constexpr (MR > 5) {
        a[5] = A + rr[5] * lda;
    }
}

/// Seed NG halves from the B panel's column sums (the A-side bias fold).
template <int MR, int NG>
NNOPS_FORCEINLINE void i8_col_init(__m256i (&acc)[NG][MR], const int8_t* B,
                                   int klen) noexcept
{
    const __m256i s0 = mma_i8_bias_fix(B, klen, 0);
    acc[0][0] = s0;
    if constexpr (MR > 1) {
        acc[0][1] = s0;
    }
    if constexpr (MR > 2) {
        acc[0][2] = s0;
    }
    if constexpr (MR > 3) {
        acc[0][3] = s0;
    }
    if constexpr (MR > 4) {
        acc[0][4] = s0;
    }
    if constexpr (MR > 5) {
        acc[0][5] = s0;
    }
    if constexpr (NG > 1) {
        const __m256i s1 = mma_i8_bias_fix(B, klen, 32);
        acc[1][0] = s1;
        if constexpr (MR > 1) {
            acc[1][1] = s1;
        }
        if constexpr (MR > 2) {
            acc[1][2] = s1;
        }
        if constexpr (MR > 3) {
            acc[1][3] = s1;
        }
        if constexpr (MR > 4) {
            acc[1][4] = s1;
        }
        if constexpr (MR > 5) {
            acc[1][5] = s1;
        }
    }
}

/// One k group of a packed-A panel: MR rows against NG eight-column halves.
template <int MR, int NG>
NNOPS_FORCEINLINE void i8_group(const int32_t*& a_i32, const int8_t* b_ptr, const int (&rr)[MR],
                                __m256i (&acc)[NG][MR]) noexcept
{
    // The advance is hoisted above the reads on purpose: MSVC then emits the
    // pointer add at the top of the k loop (see mma_f32.hpp).
    const int32_t* const a_cur = a_i32;
    a_i32 += MR;
    const __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
    __m256i vb1 = vb0;
    if constexpr (NG > 1) {
        vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 32));
    }

    const __m256i va0 = _mm256_set1_epi32(a_cur[rr[0]]);
    acc[0][0] = mma_i8_dp256(acc[0][0], va0, vb0);
    if constexpr (NG > 1) {
        acc[1][0] = mma_i8_dp256(acc[1][0], va0, vb1);
    }
    if constexpr (MR > 1) {
        const __m256i va1 = _mm256_set1_epi32(a_cur[rr[1]]);
        acc[0][1] = mma_i8_dp256(acc[0][1], va1, vb0);
        if constexpr (NG > 1) {
            acc[1][1] = mma_i8_dp256(acc[1][1], va1, vb1);
        }
    }
    if constexpr (MR > 2) {
        const __m256i va2 = _mm256_set1_epi32(a_cur[rr[2]]);
        acc[0][2] = mma_i8_dp256(acc[0][2], va2, vb0);
        if constexpr (NG > 1) {
            acc[1][2] = mma_i8_dp256(acc[1][2], va2, vb1);
        }
    }
    if constexpr (MR > 3) {
        const __m256i va3 = _mm256_set1_epi32(a_cur[rr[3]]);
        acc[0][3] = mma_i8_dp256(acc[0][3], va3, vb0);
        if constexpr (NG > 1) {
            acc[1][3] = mma_i8_dp256(acc[1][3], va3, vb1);
        }
    }
    if constexpr (MR > 4) {
        const __m256i va4 = _mm256_set1_epi32(a_cur[rr[4]]);
        acc[0][4] = mma_i8_dp256(acc[0][4], va4, vb0);
        if constexpr (NG > 1) {
            acc[1][4] = mma_i8_dp256(acc[1][4], va4, vb1);
        }
    }
    if constexpr (MR > 5) {
        const __m256i va5 = _mm256_set1_epi32(a_cur[rr[5]]);
        acc[0][5] = mma_i8_dp256(acc[0][5], va5, vb0);
        if constexpr (NG > 1) {
            acc[1][5] = mma_i8_dp256(acc[1][5], va5, vb1);
        }
    }
}

/// One k group of a row-major panel: MR rows against NG eight-column halves.
/// Each row's four bytes are splatted with the s8→u8 bias, and the row pointers
/// walk with k.
template <int MR, int NG>
NNOPS_FORCEINLINE void i8_step_rows(const int8_t* (&a)[MR], const int8_t* b_ptr,
                                    __m256i (&acc)[NG][MR]) noexcept
{
    const __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 0));
    __m256i vb1 = vb0;
    if constexpr (NG > 1) {
        vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_ptr + 32));
    }

    const int8_t* const p0 = a[0];
    a[0] = p0 + 4;
    const __m256i va0 = mma_i8_bcast_group(p0);
    acc[0][0] = mma_i8_dp256(acc[0][0], va0, vb0);
    if constexpr (NG > 1) {
        acc[1][0] = mma_i8_dp256(acc[1][0], va0, vb1);
    }
    if constexpr (MR > 1) {
        const int8_t* const p1 = a[1];
        a[1] = p1 + 4;
        const __m256i va1 = mma_i8_bcast_group(p1);
        acc[0][1] = mma_i8_dp256(acc[0][1], va1, vb0);
        if constexpr (NG > 1) {
            acc[1][1] = mma_i8_dp256(acc[1][1], va1, vb1);
        }
    }
    if constexpr (MR > 2) {
        const int8_t* const p2 = a[2];
        a[2] = p2 + 4;
        const __m256i va2 = mma_i8_bcast_group(p2);
        acc[0][2] = mma_i8_dp256(acc[0][2], va2, vb0);
        if constexpr (NG > 1) {
            acc[1][2] = mma_i8_dp256(acc[1][2], va2, vb1);
        }
    }
    if constexpr (MR > 3) {
        const int8_t* const p3 = a[3];
        a[3] = p3 + 4;
        const __m256i va3 = mma_i8_bcast_group(p3);
        acc[0][3] = mma_i8_dp256(acc[0][3], va3, vb0);
        if constexpr (NG > 1) {
            acc[1][3] = mma_i8_dp256(acc[1][3], va3, vb1);
        }
    }
    if constexpr (MR > 4) {
        const int8_t* const p4 = a[4];
        a[4] = p4 + 4;
        const __m256i va4 = mma_i8_bcast_group(p4);
        acc[0][4] = mma_i8_dp256(acc[0][4], va4, vb0);
        if constexpr (NG > 1) {
            acc[1][4] = mma_i8_dp256(acc[1][4], va4, vb1);
        }
    }
    if constexpr (MR > 5) {
        const int8_t* const p5 = a[5];
        a[5] = p5 + 4;
        const __m256i va5 = mma_i8_bcast_group(p5);
        acc[0][5] = mma_i8_dp256(acc[0][5], va5, vb0);
        if constexpr (NG > 1) {
            acc[1][5] = mma_i8_dp256(acc[1][5], va5, vb1);
        }
    }
}

/// The panel's whole K from packed A into seeded `acc`.
template <int MR, int NG>
NNOPS_FORCEINLINE void i8_mma(const int8_t* A, const int8_t* B, int klen, int kg,
                              const int (&rr)[MR], __m256i (&acc)[NG][MR]) noexcept
{
    i8_col_init<MR, NG>(acc, B, klen);
    const int32_t* a_i32 = reinterpret_cast<const int32_t*>(A);
    const int8_t* b_ptr = B;
    for (int k = 0; k < kg; ++k) {
        i8_group<MR, NG>(a_i32, b_ptr, rr, acc);
        b_ptr += 64;
    }
}

/// The panel's whole K from row-major rows into seeded `acc`.
template <int MR, int NG>
NNOPS_FORCEINLINE void i8_mma_rows(const int8_t* (&a)[MR], const int8_t* B, int klen, int kg,
                                   __m256i (&acc)[NG][MR]) noexcept
{
    i8_col_init<MR, NG>(acc, B, klen);
    const int8_t* b_ptr = B;
    for (int k = 0; k < kg; ++k) {
        i8_step_rows<MR, NG>(a, b_ptr, acc);
        b_ptr += 64;
    }
}

// ---- the store ------------------------------------------------------------
//
// A half is written in two passes: every row's C is folded into its accumulator
// first, then the rows are stored. Aliased rows share an address, so a
// row-at-a-time add would have the second duplicate reading C with the first
// one's result already in it.

/// Fold C into one half's accumulators, every row before any store.
template <int MR>
NNOPS_FORCEINLINE void i8_add_c(int32_t* const (&row)[MR], int off, __m256i (&col)[MR]) noexcept
{
    col[0] = _mm256_add_epi32(
        col[0], _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row[0] + off)));
    if constexpr (MR > 1) {
        col[1] = _mm256_add_epi32(
            col[1], _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row[1] + off)));
    }
    if constexpr (MR > 2) {
        col[2] = _mm256_add_epi32(
            col[2], _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row[2] + off)));
    }
    if constexpr (MR > 3) {
        col[3] = _mm256_add_epi32(
            col[3], _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row[3] + off)));
    }
    if constexpr (MR > 4) {
        col[4] = _mm256_add_epi32(
            col[4], _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row[4] + off)));
    }
    if constexpr (MR > 5) {
        col[5] = _mm256_add_epi32(
            col[5], _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row[5] + off)));
    }
}

/// Clamp and store one full half, no C add.
template <int MR>
NNOPS_FORCEINLINE void i8_store_c(int32_t* const (&row)[MR], int off, __m256i (&col)[MR],
                                  __m256i vmin, __m256i vmax) noexcept
{
    col[0] = _mm256_min_epi32(_mm256_max_epi32(col[0], vmin), vmax);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(row[0] + off), col[0]);
    if constexpr (MR > 1) {
        col[1] = _mm256_min_epi32(_mm256_max_epi32(col[1], vmin), vmax);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(row[1] + off), col[1]);
    }
    if constexpr (MR > 2) {
        col[2] = _mm256_min_epi32(_mm256_max_epi32(col[2], vmin), vmax);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(row[2] + off), col[2]);
    }
    if constexpr (MR > 3) {
        col[3] = _mm256_min_epi32(_mm256_max_epi32(col[3], vmin), vmax);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(row[3] + off), col[3]);
    }
    if constexpr (MR > 4) {
        col[4] = _mm256_min_epi32(_mm256_max_epi32(col[4], vmin), vmax);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(row[4] + off), col[4]);
    }
    if constexpr (MR > 5) {
        col[5] = _mm256_min_epi32(_mm256_max_epi32(col[5], vmin), vmax);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(row[5] + off), col[5]);
    }
}

/// The live lanes of a half: one stage (four int32, then two, then one) for
/// every row before that stage's stores, so duplicate (aliased) rows all read
/// the same C.
template <int MR, bool zero_mode>
NNOPS_FORCEINLINE void i8_store_lanes(int32_t* const (&row)[MR], int off, int lanes,
                                      __m256i (&col)[MR], int32_t clamp_min,
                                      int32_t clamp_max) noexcept
{
    const __m128i vmin = _mm_set1_epi32(clamp_min);
    const __m128i vmax = _mm_set1_epi32(clamp_max);
    __m128i lo[MR];
    lo[0] = _mm256_castsi256_si128(col[0]);
    if constexpr (MR > 1) {
        lo[1] = _mm256_castsi256_si128(col[1]);
    }
    if constexpr (MR > 2) {
        lo[2] = _mm256_castsi256_si128(col[2]);
    }
    if constexpr (MR > 3) {
        lo[3] = _mm256_castsi256_si128(col[3]);
    }
    if constexpr (MR > 4) {
        lo[4] = _mm256_castsi256_si128(col[4]);
    }
    if constexpr (MR > 5) {
        lo[5] = _mm256_castsi256_si128(col[5]);
    }

    if (lanes & 4) {
        if constexpr (!zero_mode) {
            lo[0] = _mm_add_epi32(
                lo[0], _mm_loadu_si128(reinterpret_cast<const __m128i*>(row[0] + off)));
            if constexpr (MR > 1) {
                lo[1] = _mm_add_epi32(
                    lo[1], _mm_loadu_si128(reinterpret_cast<const __m128i*>(row[1] + off)));
            }
            if constexpr (MR > 2) {
                lo[2] = _mm_add_epi32(
                    lo[2], _mm_loadu_si128(reinterpret_cast<const __m128i*>(row[2] + off)));
            }
            if constexpr (MR > 3) {
                lo[3] = _mm_add_epi32(
                    lo[3], _mm_loadu_si128(reinterpret_cast<const __m128i*>(row[3] + off)));
            }
            if constexpr (MR > 4) {
                lo[4] = _mm_add_epi32(
                    lo[4], _mm_loadu_si128(reinterpret_cast<const __m128i*>(row[4] + off)));
            }
            if constexpr (MR > 5) {
                lo[5] = _mm_add_epi32(
                    lo[5], _mm_loadu_si128(reinterpret_cast<const __m128i*>(row[5] + off)));
            }
        }
        lo[0] = _mm_min_epi32(_mm_max_epi32(lo[0], vmin), vmax);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(row[0] + off), lo[0]);
        if constexpr (MR > 1) {
            lo[1] = _mm_min_epi32(_mm_max_epi32(lo[1], vmin), vmax);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(row[1] + off), lo[1]);
        }
        if constexpr (MR > 2) {
            lo[2] = _mm_min_epi32(_mm_max_epi32(lo[2], vmin), vmax);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(row[2] + off), lo[2]);
        }
        if constexpr (MR > 3) {
            lo[3] = _mm_min_epi32(_mm_max_epi32(lo[3], vmin), vmax);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(row[3] + off), lo[3]);
        }
        if constexpr (MR > 4) {
            lo[4] = _mm_min_epi32(_mm_max_epi32(lo[4], vmin), vmax);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(row[4] + off), lo[4]);
        }
        if constexpr (MR > 5) {
            lo[5] = _mm_min_epi32(_mm_max_epi32(lo[5], vmin), vmax);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(row[5] + off), lo[5]);
        }
        off += 4;
        lo[0] = _mm256_extracti128_si256(col[0], 1);
        if constexpr (MR > 1) {
            lo[1] = _mm256_extracti128_si256(col[1], 1);
        }
        if constexpr (MR > 2) {
            lo[2] = _mm256_extracti128_si256(col[2], 1);
        }
        if constexpr (MR > 3) {
            lo[3] = _mm256_extracti128_si256(col[3], 1);
        }
        if constexpr (MR > 4) {
            lo[4] = _mm256_extracti128_si256(col[4], 1);
        }
        if constexpr (MR > 5) {
            lo[5] = _mm256_extracti128_si256(col[5], 1);
        }
    }

    if (lanes & 2) {
        // The 4-wide clamp above would corrupt lanes 2,3 that the 1-wide step
        // stores, so the shifted-down copy comes from the pre-clamp vector.
        __m128i rest[MR];
        rest[0] = lo[0];
        if constexpr (MR > 1) {
            rest[1] = lo[1];
        }
        if constexpr (MR > 2) {
            rest[2] = lo[2];
        }
        if constexpr (MR > 3) {
            rest[3] = lo[3];
        }
        if constexpr (MR > 4) {
            rest[4] = lo[4];
        }
        if constexpr (MR > 5) {
            rest[5] = lo[5];
        }
        if constexpr (!zero_mode) {
            lo[0] = _mm_add_epi32(lo[0],
                                  _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row[0] + off)));
            if constexpr (MR > 1) {
                lo[1] = _mm_add_epi32(
                    lo[1], _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row[1] + off)));
            }
            if constexpr (MR > 2) {
                lo[2] = _mm_add_epi32(
                    lo[2], _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row[2] + off)));
            }
            if constexpr (MR > 3) {
                lo[3] = _mm_add_epi32(
                    lo[3], _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row[3] + off)));
            }
            if constexpr (MR > 4) {
                lo[4] = _mm_add_epi32(
                    lo[4], _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row[4] + off)));
            }
            if constexpr (MR > 5) {
                lo[5] = _mm_add_epi32(
                    lo[5], _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row[5] + off)));
            }
        }
        lo[0] = _mm_min_epi32(_mm_max_epi32(lo[0], vmin), vmax);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(row[0] + off), lo[0]);
        if constexpr (MR > 1) {
            lo[1] = _mm_min_epi32(_mm_max_epi32(lo[1], vmin), vmax);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(row[1] + off), lo[1]);
        }
        if constexpr (MR > 2) {
            lo[2] = _mm_min_epi32(_mm_max_epi32(lo[2], vmin), vmax);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(row[2] + off), lo[2]);
        }
        if constexpr (MR > 3) {
            lo[3] = _mm_min_epi32(_mm_max_epi32(lo[3], vmin), vmax);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(row[3] + off), lo[3]);
        }
        if constexpr (MR > 4) {
            lo[4] = _mm_min_epi32(_mm_max_epi32(lo[4], vmin), vmax);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(row[4] + off), lo[4]);
        }
        if constexpr (MR > 5) {
            lo[5] = _mm_min_epi32(_mm_max_epi32(lo[5], vmin), vmax);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(row[5] + off), lo[5]);
        }
        off += 2;
        lo[0] = _mm_srli_si128(rest[0], 8);
        if constexpr (MR > 1) {
            lo[1] = _mm_srli_si128(rest[1], 8);
        }
        if constexpr (MR > 2) {
            lo[2] = _mm_srli_si128(rest[2], 8);
        }
        if constexpr (MR > 3) {
            lo[3] = _mm_srli_si128(rest[3], 8);
        }
        if constexpr (MR > 4) {
            lo[4] = _mm_srli_si128(rest[4], 8);
        }
        if constexpr (MR > 5) {
            lo[5] = _mm_srli_si128(rest[5], 8);
        }
    }

    if (lanes & 1) {
        int v[MR];
        v[0] = _mm_cvtsi128_si32(lo[0]);
        if constexpr (MR > 1) {
            v[1] = _mm_cvtsi128_si32(lo[1]);
        }
        if constexpr (MR > 2) {
            v[2] = _mm_cvtsi128_si32(lo[2]);
        }
        if constexpr (MR > 3) {
            v[3] = _mm_cvtsi128_si32(lo[3]);
        }
        if constexpr (MR > 4) {
            v[4] = _mm_cvtsi128_si32(lo[4]);
        }
        if constexpr (MR > 5) {
            v[5] = _mm_cvtsi128_si32(lo[5]);
        }
        if constexpr (!zero_mode) {
            v[0] += row[0][off];
            if constexpr (MR > 1) {
                v[1] += row[1][off];
            }
            if constexpr (MR > 2) {
                v[2] += row[2][off];
            }
            if constexpr (MR > 3) {
                v[3] += row[3][off];
            }
            if constexpr (MR > 4) {
                v[4] += row[4][off];
            }
            if constexpr (MR > 5) {
                v[5] += row[5][off];
            }
        }
        row[0][off] = std::min(std::max(v[0], clamp_min), clamp_max);
        if constexpr (MR > 1) {
            row[1][off] = std::min(std::max(v[1], clamp_min), clamp_max);
        }
        if constexpr (MR > 2) {
            row[2][off] = std::min(std::max(v[2], clamp_min), clamp_max);
        }
        if constexpr (MR > 3) {
            row[3][off] = std::min(std::max(v[3], clamp_min), clamp_max);
        }
        if constexpr (MR > 4) {
            row[4][off] = std::min(std::max(v[4], clamp_min), clamp_max);
        }
        if constexpr (MR > 5) {
            row[5][off] = std::min(std::max(v[5], clamp_min), clamp_max);
        }
    }
}

/// Write one half of a panel: MR rows, `live` columns left from this half's
/// first column.
template <int MR, bool zero_mode>
NNOPS_FORCEINLINE void i8_store_col(int32_t* const (&row)[MR], int col0, int col, int live,
                                    __m256i (&col_acc)[MR], __m256i vmin, __m256i vmax,
                                    int32_t clamp_min, int32_t clamp_max) noexcept
{
    const int l = live - col;
    if (l >= 8) {
        if constexpr (!zero_mode) {
            i8_add_c<MR>(row, col0 + col, col_acc);
        }
        i8_store_c<MR>(row, col0 + col, col_acc, vmin, vmax);
    } else if (l > 0) {
        i8_store_lanes<MR, zero_mode>(row, col0 + col, l, col_acc, clamp_min, clamp_max);
    }
}

/// Write the live columns of NG halves, starting at `col0`.
template <int MR, int NG, bool zero_mode>
NNOPS_FORCEINLINE void i8_store(int32_t* const (&row)[MR], int col0, int live,
                                __m256i (&acc)[NG][MR], __m256i vmin, __m256i vmax,
                                int32_t clamp_min, int32_t clamp_max) noexcept
{
    i8_store_col<MR, zero_mode>(row, col0, 0, live, acc[0], vmin, vmax, clamp_min, clamp_max);
    if constexpr (NG > 1) {
        i8_store_col<MR, zero_mode>(row, col0, 8, live, acc[1], vmin, vmax, clamp_min,
                                    clamp_max);
    }
}

// =========================================================================
//  packed A + packed B  (MR = 6, 4, 1)
// =========================================================================

/// One packed-A rung: the whole N side into int32 C.
template <int MR, bool zero_mode = false>
inline void mma_pack_mr_nr16_mrxncxkc_i8(int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C,
                                         int ldc, const int8_t* NNOPS_RESTRICT A,
                                         const int8_t* NNOPS_RESTRICT B, int ldb,
                                         int32_t clamp_min, int32_t clamp_max) noexcept
{
    int rr[MR];
    i8_row_alias<MR>(mr, rr);
    int32_t* row[MR];
    i8_row_ptrs<MR>(C, ldc, rr, row);
    const int kbytes = (kc + 3) & ~3;
    const int kg = kbytes >> 2;
    const int ldb_panel = mma_i8_panel_stride(ldb, kbytes);
    const __m256i vmin = _mm256_set1_epi32(clamp_min);
    const __m256i vmax = _mm256_set1_epi32(clamp_max);

    int n = 0;
    for (; n + mma_i8_nr_max <= nc; n += mma_i8_nr_max) {
        __m256i acc[2][MR];
        i8_mma<MR, 2>(A, B, kbytes, kg, rr, acc);
        i8_store<MR, 2, zero_mode>(row, n, 16, acc, vmin, vmax, clamp_min, clamp_max);
        B += ldb_panel;
    }

    // The live remainder narrows to the halves it needs: 8 or 16 columns.
    const int live = nc - n;
    if (live > 8) {
        __m256i acc[2][MR];
        i8_mma<MR, 2>(A, B, kbytes, kg, rr, acc);
        i8_store<MR, 2, zero_mode>(row, n, live, acc, vmin, vmax, clamp_min, clamp_max);
    } else if (live > 0) {
        __m256i acc[1][MR];
        i8_mma<MR, 1>(A, B, kbytes, kg, rr, acc);
        i8_store<MR, 1, zero_mode>(row, n, live, acc, vmin, vmax, clamp_min, clamp_max);
    }
}

// =========================================================================
//  raw A + packed B  (MR = 6, 4, 1) — kc % 4 == 0
// =========================================================================
//
// The same walk with each row's four bytes broadcast instead of a shared
// 16-byte panel slot, so the rung reads A through `lda`.

/// One row-major-A rung: the whole N side into int32 C.
template <int MR, bool zero_mode = false>
inline void mma_direct_mr_nr16_mrxncxkc_i8(int mr, int nc, int kc, int32_t* NNOPS_RESTRICT C,
                                           int ldc, const int8_t* NNOPS_RESTRICT A, int lda,
                                           const int8_t* NNOPS_RESTRICT B, int ldb,
                                           int32_t clamp_min, int32_t clamp_max) noexcept
{
    int rr[MR];
    i8_row_alias<MR>(mr, rr);
    int32_t* row[MR];
    i8_row_ptrs<MR>(C, ldc, rr, row);
    const int kg = kc >> 2;
    const int ldb_panel = mma_i8_panel_stride(ldb, kc);
    const __m256i vmin = _mm256_set1_epi32(clamp_min);
    const __m256i vmax = _mm256_set1_epi32(clamp_max);

    int n = 0;
    for (; n + mma_i8_nr_max <= nc; n += mma_i8_nr_max) {
        const int8_t* a0[MR];
        i8_a_ptrs<MR>(A, lda, rr, a0);
        __m256i acc[2][MR];
        i8_mma_rows<MR, 2>(a0, B, kc, kg, acc);
        i8_store<MR, 2, zero_mode>(row, n, 16, acc, vmin, vmax, clamp_min, clamp_max);
        B += ldb_panel;
    }

    const int live = nc - n;
    if (live > 8) {
        const int8_t* a0[MR];
        i8_a_ptrs<MR>(A, lda, rr, a0);
        __m256i acc[2][MR];
        i8_mma_rows<MR, 2>(a0, B, kc, kg, acc);
        i8_store<MR, 2, zero_mode>(row, n, live, acc, vmin, vmax, clamp_min, clamp_max);
    } else if (live > 0) {
        const int8_t* a0[MR];
        i8_a_ptrs<MR>(A, lda, rr, a0);
        __m256i acc[1][MR];
        i8_mma_rows<MR, 1>(a0, B, kc, kg, acc);
        i8_store<MR, 1, zero_mode>(row, n, live, acc, vmin, vmax, clamp_min, clamp_max);
    }
}

// ---- dispatch tables, consumed by matmul_helper.cpp ---------------------

/// MR levels {6,4,1} (MR_I8); a remainder panel runs through the level above it.
template <bool Z>
inline constexpr auto mma_pack_fns_i8 =
    std::array{&mma_pack_mr_nr16_mrxncxkc_i8<6, Z>, &mma_pack_mr_nr16_mrxncxkc_i8<4, Z>,
               &mma_pack_mr_nr16_mrxncxkc_i8<1, Z>};

/// Row-major A (`lda` in int8 elements, kc % 4 == 0), same MR levels.
template <bool Z>
inline constexpr auto mma_direct_fns_i8 =
    std::array{&mma_direct_mr_nr16_mrxncxkc_i8<6, Z>, &mma_direct_mr_nr16_mrxncxkc_i8<4, Z>,
               &mma_direct_mr_nr16_mrxncxkc_i8<1, Z>};

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
NNOPS_FORCEINLINE void
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
NNOPS_FORCEINLINE void
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
NNOPS_FORCEINLINE void
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
NNOPS_FORCEINLINE void
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

/// The kernel signature the requantizing rungs are dispatched through.
template <class TOut>
using MmaI8EpiFn = void (*)(int, int, int, TOut*, int64_t, const int8_t*, const int8_t*, int,
                            const float*, float, float, const int32_t*, int32_t, bool);

/// The requantizing rungs, consumed by matmul_helper.cpp; rows are {6,4,1}.
template <class TOut>
inline constexpr std::array<MmaI8EpiFn<TOut>, 3> mma_pack_epi_fns_i8 = {
    &mma_pack_mr_nr16_epi_i8<6, TOut>,
    &mma_pack_mr_nr16_epi_i8<4, TOut>,
    &mma_pack_mr_nr16_epi_i8<1, TOut>,
};

} // namespace nnops::backend::cpu::x86_64
