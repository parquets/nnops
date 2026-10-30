#pragma once
/// @file pack_q4_1.hpp
/// @brief x86_64 AVX2/F16C Q4_1 decode and packed-B kernels.
///
/// Q4_1 is llama.cpp's affine 4-bit block format: 32 values in 20 bytes — an
/// fp16 delta `d`, an fp16 minimum `m`, and 16 bytes of nibbles (low nibble of
/// byte j is value j, high nibble is value j + 16). Value j is `d * q + m` with
/// q in [0, 15]: two nibble vectors per block and an affine (d, m) decode, so
/// a block covers 32 values in 20 bytes.
///
/// The dense `pack_q4` here is the vectorized counterpart of the portable
/// scalar loop in <nnops/detail/q4_1.hpp>; the block layout and the scalar
/// reference live there, so anything needing only the format can include the
/// public header without pulling in this arch directory.
///
/// `pack_q4_1_rhs` is the GEMM-side entry point: it fuses unpack + dequantize
/// + f16 narrowing into the RHS panel pack, so a Q4_1 weight never passes
/// through an intermediate dense buffer. It writes the same uniform
/// `nr_max`-wide (16 on x86_64), `[k][nr]`-major panels the f32/f16 packers
/// write, which is what lets MatMul treat a Q4_1 B as just another packed-B
/// source — the MMA route knows nothing about the format.
///
/// The pad lanes of a tail panel (past its live width) are zero-filled here,
/// unlike the transposing f32/f16 packers, which alias the last live row: a
/// Q4_1 decode has no row to alias, and the MMA kernels only store live
/// columns anyway.

#include <immintrin.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "backend/cpu/common/common_utils.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/q4_1.hpp"

namespace nnops::backend::cpu::x86_64 {

using nnops::backend::cpu::float_to_half;
using nnops::backend::cpu::half;
using nnops::backend::cpu::half_from_bits;
using nnops::backend::cpu::half_to_float;

/// Decode one block's 32 values into four 8-lane f32 vectors.
///
/// The nibble bytes are widened once into two 16-byte halves (low nibbles and
/// high nibbles), then each half is widened to 8-lane integers: four
/// `cvtepu8_epi32` and four FMAs.
///
/// @param scale multiplies the whole block: the decode is `q·d + m`, so scaling
///              `d` and `m` once per block scales all 32 values — MatMul's alpha
///              without a per-element pass.
inline void decode_q4_1_block_ps(const uint8_t* p,
                                 __m256& v0, __m256& v1,
                                 __m256& v2, __m256& v3,
                                 float scale = 1.0f) noexcept {
    uint16_t d_bits, m_bits;
    std::memcpy(&d_bits, p + 0, sizeof(d_bits));
    std::memcpy(&m_bits, p + 2, sizeof(m_bits));

    // F16C on the two scale fields: the portable half_to_float is a scalar
    // branchy routine, and at two calls per 32 values it costs more than the
    // rest of the decode. `_mm_cvtph_ps` + a broadcast is one instruction pair.
    __m256 md = _mm256_broadcastss_ps(_mm_cvtph_ps(_mm_cvtsi32_si128(d_bits)));
    __m256 mm = _mm256_broadcastss_ps(_mm_cvtph_ps(_mm_cvtsi32_si128(m_bits)));
    if (scale != 1.0f) {
        const __m256 vs = _mm256_set1_ps(scale);
        md = _mm256_mul_ps(md, vs);
        mm = _mm256_mul_ps(mm, vs);
    }

    const __m128i qs = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 4));
    const __m128i m0f = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_and_si128(qs, m0f);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(qs, 4), m0f);

    const __m256i x0 = _mm256_cvtepu8_epi32(lo);
    const __m256i x1 = _mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8));
    const __m256i x2 = _mm256_cvtepu8_epi32(hi);
    const __m256i x3 = _mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8));

    v0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(x0), md, mm);
    v1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(x1), md, mm);
    v2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(x2), md, mm);
    v3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(x3), md, mm);
}

/// Decode one block into 32 f32 or f16 values (f16 rounds once per element).
template <class T>
inline void decode_q4_1_block(const uint8_t* p, T* out,
                              float scale = 1.0f) noexcept {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                  "decode_q4_1_block: T must be float or half");

    __m256 v0, v1, v2, v3;
    decode_q4_1_block_ps(p, v0, v1, v2, v3, scale);

    if constexpr (std::is_same_v<T, float>) {
        _mm256_storeu_ps(out + 0,  v0);
        _mm256_storeu_ps(out + 8,  v1);
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

/// Fused unpack + dequantize: decode @p count values from a stream of 20-byte
/// Q4_1 blocks into a dense f32 or f16 array.
///
/// Same contract as nnops::pack_q4 (the portable scalar reference): full
/// blocks decode straight into the destination, a trailing partial block is
/// decoded into a register-sized scratch and only the live values are copied.
template <class T>
void pack_q4(const void* blocks, int64_t count, T* out) noexcept {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                  "pack_q4: T must be float or half");

    if (count <= 0) {
        return;
    }

    const uint8_t* src = static_cast<const uint8_t*>(blocks);
    int64_t b = 0;
    for (; (b + 1) * q4_1_block_elems <= count; ++b) {
        decode_q4_1_block<T>(src + b * static_cast<int64_t>(q4_1_block_bytes),
                             out + b * q4_1_block_elems);
    }

    const int64_t base = b * q4_1_block_elems;
    const int n = static_cast<int>(count - base);
    if (n > 0) {
        T vals[q4_1_block_elems];
        decode_q4_1_block<T>(src + b * static_cast<int64_t>(q4_1_block_bytes), vals);
        for (int j = 0; j < n; ++j) {
            out[base + j] = vals[j];
        }
    }
}

// =========================================================================
//  Packed-B panels
// =========================================================================
//
// Panels are the uniform `nr_max`-wide kind the mrxncxkc MMA kernels read:
// ceil(nc/nr_max) panels of `kc` rows, each row `nr_max` live lanes, all at
// the caller's aligned panel stride `ldd` (elements). A tail panel's lanes
// past its live width are zero-filled.

/// Panel width the MatMul pack route uses on x86_64 (nr_f32[0] == nr_f16[0]).
inline constexpr int q4_1_nr_max = 16;

/// Row-major B (phys [K][N]): panel columns [n_start, n_start + width) of rows
/// [k_start, k_start + kc). A panel can straddle two 32-value blocks, so at
/// most two block decodes feed one row. @p width is ignored when NR > 0.
template <class T, int NR>
void pack_q4_1_copy_panel(T* out, const uint8_t* src, int64_t row_blocks,
                          int k_start, int kc, int n_start, int width,
                          float scale = 1.0f) noexcept {
    const int nb    = (NR > 0) ? NR : width;
    const int blk0  = n_start >> 5;
    const int off0  = n_start & 31;
    const int first = std::min(nb, 32 - off0);
    const bool two  = (nb > first);
    const int64_t row_bytes = row_blocks * static_cast<int64_t>(q4_1_block_bytes);

    T v0[q4_1_block_elems];
    T v1[q4_1_block_elems];
    for (int k = 0; k < kc; ++k) {
        const uint8_t* row = src + static_cast<int64_t>(k_start + k) * row_bytes;
        T* o = out + static_cast<int64_t>(k) * q4_1_nr_max;

        decode_q4_1_block<T>(row + static_cast<int64_t>(blk0) * q4_1_block_bytes, v0,
                             scale);
        for (int j = 0; j < first; ++j) {
            o[j] = v0[off0 + j];
        }
        if (two) {
            decode_q4_1_block<T>(
                row + static_cast<int64_t>(blk0 + 1) * q4_1_block_bytes, v1, scale);
            for (int j = first; j < nb; ++j) {
                o[j] = v1[j - first];
            }
        }
        for (int j = nb; j < q4_1_nr_max; ++j) {
            o[j] = T{};
        }
    }
}

/// Transposed B (phys [N][K]): panel rows [n_start, n_start + width) of columns
/// [k_start, k_start + kc). Each source row is swept linearly, so every block
/// decode yields up to 32 consecutive k values — the efficient direction for
/// Linear's [N][K] weights.
template <class T, int NR>
void pack_q4_1_trans_panel(T* out, const uint8_t* src, int64_t row_blocks,
                           int k_start, int kc, int n_start, int width,
                           float scale = 1.0f) noexcept {
    const int nb     = (NR > 0) ? NR : width;
    const int blk0   = k_start >> 5;
    const int off0   = k_start & 31;
    const int nblk   = (off0 + kc + 31) >> 5;
    const int64_t row_bytes = row_blocks * static_cast<int64_t>(q4_1_block_bytes);

    T vals[q4_1_block_elems];
    for (int j = 0; j < nb; ++j) {
        const uint8_t* row = src + static_cast<int64_t>(n_start + j) * row_bytes
                           + static_cast<int64_t>(blk0) * q4_1_block_bytes;

        int k = 0;
        for (int b = 0; b < nblk; ++b) {
            decode_q4_1_block<T>(row + static_cast<int64_t>(b) * q4_1_block_bytes,
                                 vals, scale);
            const int start = (b == 0) ? off0 : 0;
            const int count = std::min(32 - start, kc - k);
            T* o = out + static_cast<int64_t>(k) * q4_1_nr_max + j;
            for (int i = 0; i < count; ++i) {
                o[static_cast<int64_t>(i) * q4_1_nr_max] = vals[start + i];
            }
            k += count;
        }
    }

    // Lanes past the live width: never read, never stored to C, but keep the
    // panel fully defined so the workspace is deterministic.
    for (int k = 0; k < kc; ++k) {
        T* o = out + static_cast<int64_t>(k) * q4_1_nr_max;
        for (int j = nb; j < q4_1_nr_max; ++j) {
            o[j] = T{};
        }
    }
}

/// Pack the Q4_1 RHS into uniform panels for one k-block.
///
/// @param trans      true when B is physically [N][K], false for [K][N].
/// @param n_start    first logical column of this tile.
/// @param nc         live columns in the tile.
/// @param k_start    first logical k of the k-block.
/// @param kc         k length of the block.
/// @param dst        first panel; panels advance by @p ldd elements.
/// @param ldd        aligned panel stride in elements (nr_max * kc rounded).
/// @param src        first block of the batch element's B.
/// @param row_blocks 20-byte blocks per physical row (the B row pitch / 20).
/// @param scale      MatMul alpha. It folds into each block's `d` and `m`
///                   (two multiplies per 32 weights), so the pack itself does
///                   no extra work per element.
template <class T>
void pack_q4_1_rhs(bool trans, int n_start, int nc, int k_start, int kc,
                   T* dst, int ldd, const void* src, int64_t row_blocks,
                   float scale) noexcept {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                  "pack_q4_1_rhs: T must be float or half");

    const uint8_t* base = static_cast<const uint8_t*>(src);

    int n = 0;
    for (; n + q4_1_nr_max <= nc; n += q4_1_nr_max) {
        if (trans) {
            pack_q4_1_trans_panel<T, q4_1_nr_max>(
                dst, base, row_blocks, k_start, kc, n_start + n, 0, scale);
        } else {
            pack_q4_1_copy_panel<T, q4_1_nr_max>(
                dst, base, row_blocks, k_start, kc, n_start + n, 0, scale);
        }
        dst += ldd;
    }
    if (n < nc) {
        if (trans) {
            pack_q4_1_trans_panel<T, 0>(
                dst, base, row_blocks, k_start, kc, n_start + n, nc - n, scale);
        } else {
            pack_q4_1_copy_panel<T, 0>(
                dst, base, row_blocks, k_start, kc, n_start + n, nc - n, scale);
        }
    }
}

}  // namespace nnops::backend::cpu::x86_64
