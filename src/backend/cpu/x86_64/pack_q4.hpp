#pragma once
/// @file pack_q4.hpp
/// @brief x86_64 AVX2/F16C Q4_0/Q4_1 decode and packed-B kernels.
///
/// Both are llama.cpp 4-bit block formats covering 32 values in 16 shared nibble
/// bytes (byte j's low nibble is value j, its high nibble value j + 16). Q4_0 is
/// 18 bytes — an fp16 delta `d`, and value j is `d * (q - 8)`; Q4_1 is 20 bytes —
/// `d`, an fp16 minimum `m`, and value j is `d * q + m`. The layout and the
/// portable scalar reference live in <nnops/detail/q4.hpp>; the dense `pack_q4`
/// here is its vectorized twin, and `fmt` selects the codec.
///
/// `pack_q4_rhs` is the GEMM-side entry point: it fuses the decode (and, for
/// f16, the narrowing) into the RHS panel pack, so a quantized weight never
/// passes through an intermediate dense buffer. It writes the same uniform
/// `nr_max`-wide (16 on x86_64), `[k][nr]`-major panels the f32/f16 packers
/// write, which is what lets MatMul treat a Q4 B as just another packed-B
/// source — the MMA route knows nothing about the format. `scale` (MatMul's
/// alpha) folds into each block's `d` (and `m`): two multiplies per 32 values,
/// not one per element.
///
/// The half fields decode through F16C rather than the portable half_to_float,
/// which on x86_64 is a software conversion with branches for the special cases;
/// here each field costs one `cvtph` convert plus a broadcast.
///
/// A tail panel's lanes past its live width are zero-filled, unlike the
/// transposing f32/f16 packers, which alias the last live row: a Q4 decode has
/// no row to alias, and the MMA kernels only store live columns anyway. Zeroing
/// them is what keeps the workspace deterministic.

#include <immintrin.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "backend/cpu/common/common_utils.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/q4.hpp"

namespace nnops::backend::cpu::x86_64 {

using nnops::backend::cpu::float_to_half;
using nnops::backend::cpu::half;
using nnops::backend::cpu::half_from_bits;
using nnops::backend::cpu::half_to_float;

/// Decode one block's 32 values into four 8-lane f32 vectors.
inline void decode_q4_block_ps(const uint8_t* p, __m256& v0, __m256& v1, __m256& v2, __m256& v3,
                               float scale, DataType fmt) noexcept
{
    const bool with_min = q4_has_min(fmt);
    uint16_t d_bits = 0, m_bits = 0;
    std::memcpy(&d_bits, p + 0, sizeof(d_bits));
    if (with_min) {
        std::memcpy(&m_bits, p + 2, sizeof(m_bits));
    }

    __m256 md = _mm256_broadcastss_ps(_mm_cvtph_ps(_mm_cvtsi32_si128(d_bits)));
    __m256 mm = with_min ? _mm256_broadcastss_ps(_mm_cvtph_ps(_mm_cvtsi32_si128(m_bits))) :
                           _mm256_setzero_ps();
    if (scale != 1.0f) {
        const __m256 vs = _mm256_set1_ps(scale);
        md = _mm256_mul_ps(md, vs);
        mm = _mm256_mul_ps(mm, vs);
    }

    const __m128i qs = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + (with_min ? 4 : 2)));
    const __m128i m0f = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_and_si128(qs, m0f);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(qs, 4), m0f);

    __m256i x0 = _mm256_cvtepu8_epi32(lo);
    __m256i x1 = _mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8));
    __m256i x2 = _mm256_cvtepu8_epi32(hi);
    __m256i x3 = _mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8));
    if (!with_min) {
        const __m256i eight = _mm256_set1_epi32(8);
        x0 = _mm256_sub_epi32(x0, eight);
        x1 = _mm256_sub_epi32(x1, eight);
        x2 = _mm256_sub_epi32(x2, eight);
        x3 = _mm256_sub_epi32(x3, eight);
    }

    v0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(x0), md, mm);
    v1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(x1), md, mm);
    v2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(x2), md, mm);
    v3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(x3), md, mm);
}

/// Decode one block into 32 f32 or f16 values (f16 rounds once per element).
template <class T>
inline void decode_q4_block(const uint8_t* p, T* out, float scale, DataType fmt) noexcept
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                  "decode_q4_block: T must be float or half");

    __m256 v0, v1, v2, v3;
    decode_q4_block_ps(p, v0, v1, v2, v3, scale, fmt);

    if constexpr (std::is_same_v<T, float>) {
        _mm256_storeu_ps(out + 0, v0);
        _mm256_storeu_ps(out + 8, v1);
        _mm256_storeu_ps(out + 16, v2);
        _mm256_storeu_ps(out + 24, v3);
    } else {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 0),
                         _mm256_cvtps_ph(v0, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 8),
                         _mm256_cvtps_ph(v1, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 16),
                         _mm256_cvtps_ph(v2, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 24),
                         _mm256_cvtps_ph(v3, _MM_FROUND_TO_NEAREST_INT));
    }
}

/// Decode @p count values from a stream of blocks into a dense array; a trailing
/// partial block goes through a block-sized scratch and only its live values are
/// copied.
template <class T>
void pack_q4(const void* blocks, int64_t count, T* out, DataType fmt = DataType::q4_1) noexcept
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                  "pack_q4: T must be float or half");
    if (count <= 0) {
        return;
    }

    const uint8_t* src = static_cast<const uint8_t*>(blocks);
    const int64_t block_bytes = static_cast<int64_t>(q4_block_bytes(fmt));
    int64_t b = 0;
    for (; (b + 1) * q4_block_elems <= count; ++b) {
        decode_q4_block<T>(src + b * block_bytes, out + b * q4_block_elems, 1.0f, fmt);
    }

    const int64_t base = b * q4_block_elems;
    const int n = static_cast<int>(count - base);
    if (n > 0) {
        T vals[q4_block_elems];
        decode_q4_block<T>(src + b * block_bytes, vals, 1.0f, fmt);
        for (int j = 0; j < n; ++j) {
            out[base + j] = vals[j];
        }
    }
}

// =========================================================================
//  Packed-B panels  —  the uniform `nr_max`-wide panels the mrxncxkc kernels
// =========================================================================

/// Panel width the MatMul pack route uses on x86_64 (nr_f32[0] == nr_f16[0]).
inline constexpr int q4_nr_max = 16;

/// Row-major B (phys [K][N]): panel columns [n_start, n_start + width) of rows
/// [k_start, k_start + kc); a panel straddles at most two 32-value blocks. Width
/// is ignored when NR > 0.
template <class T, int NR>
void pack_q4_copy_panel(T* out, const uint8_t* src, int64_t row_blocks, int k_start, int kc,
                        int n_start, int width, float scale, DataType fmt) noexcept
{
    const int nb = (NR > 0) ? NR : width;
    const int blk0 = n_start >> 5;
    const int off0 = n_start & 31;
    const int first = std::min(nb, 32 - off0);
    const bool two = (nb > first);
    const int64_t block_bytes = static_cast<int64_t>(q4_block_bytes(fmt));
    const int64_t row_bytes = row_blocks * block_bytes;

    T v0[q4_block_elems];
    T v1[q4_block_elems];
    for (int k = 0; k < kc; ++k) {
        const uint8_t* row = src + static_cast<int64_t>(k_start + k) * row_bytes;
        T* o = out + static_cast<int64_t>(k) * q4_nr_max;

        decode_q4_block<T>(row + static_cast<int64_t>(blk0) * block_bytes, v0, scale, fmt);
        for (int j = 0; j < first; ++j) {
            o[j] = v0[off0 + j];
        }
        if (two) {
            decode_q4_block<T>(row + static_cast<int64_t>(blk0 + 1) * block_bytes, v1, scale, fmt);
            for (int j = first; j < nb; ++j) {
                o[j] = v1[j - first];
            }
        }
        for (int j = nb; j < q4_nr_max; ++j) {
            o[j] = T{};
        }
    }
}

/// Transposed B (phys [N][K]): panel rows [n_start, n_start + width) of columns
/// [k_start, k_start + kc). Each source row is swept linearly, so one decode
/// yields up to 32 consecutive k values — the efficient direction for Linear's
/// weights.
template <class T, int NR>
void pack_q4_trans_panel(T* out, const uint8_t* src, int64_t row_blocks, int k_start, int kc,
                         int n_start, int width, float scale, DataType fmt) noexcept
{
    const int nb = (NR > 0) ? NR : width;
    const int blk0 = k_start >> 5;
    const int off0 = k_start & 31;
    const int nblk = (off0 + kc + 31) >> 5;
    const int64_t block_bytes = static_cast<int64_t>(q4_block_bytes(fmt));
    const int64_t row_bytes = row_blocks * block_bytes;

    T vals[q4_block_elems];
    for (int j = 0; j < nb; ++j) {
        const uint8_t* row = src + static_cast<int64_t>(n_start + j) * row_bytes +
                             static_cast<int64_t>(blk0) * block_bytes;

        int k = 0;
        for (int b = 0; b < nblk; ++b) {
            decode_q4_block<T>(row + static_cast<int64_t>(b) * block_bytes, vals, scale, fmt);
            const int start = (b == 0) ? off0 : 0;
            const int count = std::min(32 - start, kc - k);
            T* o = out + static_cast<int64_t>(k) * q4_nr_max + j;
            for (int i = 0; i < count; ++i) {
                o[static_cast<int64_t>(i) * q4_nr_max] = vals[start + i];
            }
            k += count;
        }
    }

    // Lanes past the live width never reach C, but zeroing them keeps the workspace deterministic.
    for (int k = 0; k < kc; ++k) {
        T* o = out + static_cast<int64_t>(k) * q4_nr_max;
        for (int j = nb; j < q4_nr_max; ++j) {
            o[j] = T{};
        }
    }
}

/// Pack one k-block of the Q4 RHS into uniform panels. @p trans means B is
/// physically [N][K]; @p dst steps by @p ldd (64-byte-aligned nr_max * kc) per
/// panel, and @p row_blocks is the B row pitch in blocks of @p fmt.
template <class T>
void pack_q4_rhs(bool trans, int n_start, int nc, int k_start, int kc, T* dst, int ldd,
                 const void* src, int64_t row_blocks, float scale, DataType fmt) noexcept
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                  "pack_q4_rhs: T must be float or half");

    const uint8_t* base = static_cast<const uint8_t*>(src);

    int n = 0;
    for (; n + q4_nr_max <= nc; n += q4_nr_max) {
        if (trans) {
            pack_q4_trans_panel<T, q4_nr_max>(dst, base, row_blocks, k_start, kc, n_start + n, 0,
                                              scale, fmt);
        } else {
            pack_q4_copy_panel<T, q4_nr_max>(dst, base, row_blocks, k_start, kc, n_start + n, 0,
                                             scale, fmt);
        }
        dst += ldd;
    }
    if (n < nc) {
        if (trans) {
            pack_q4_trans_panel<T, 0>(dst, base, row_blocks, k_start, kc, n_start + n, nc - n,
                                      scale, fmt);
        } else {
            pack_q4_copy_panel<T, 0>(dst, base, row_blocks, k_start, kc, n_start + n, nc - n, scale,
                                     fmt);
        }
    }
}

} // namespace nnops::backend::cpu::x86_64
