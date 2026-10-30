#pragma once
/// @file mma_f16.hpp
/// @brief x86_64 F16C/FMA float16 MMA (matrix micro-accumulate) kernels.
///
/// Storage is fp16; accumulation, clamping and the C add are in fp32, and the
/// result is rounded back to fp16 on store. A and B may both be packed
/// (`mma_packed_ab_*`) or only B packed while A is read row-major
/// (`mma_packed_b_*`).
///
/// Tile sizes (x86_64-optimised):
///   M: {6, 4, 2, 1}    N: {16, 8, 1}
///
/// Packed B is uniform: ceil(nc / 16) panels, every one a full 16 columns
/// wide with the columns past `nc` zero-filled, each in a
/// mma_pack_align(16*kc*sizeof(half))-byte slot (16*kc halves). That costs the
/// packer nothing -- one 16-wide copy/transpose plus a zero pad -- and lets the
/// kernel read 16 halves per k at every column, so a tail wider than 8 costs
/// one k-loop instead of two.

#include <array>
#include <immintrin.h>
#include <cstdint>
#include "nnops/detail/half.hpp"
#include "backend/cpu/common/common_utils.hpp"

namespace nnops::backend::cpu::x86_64 {

using nnops::backend::cpu::half;
using nnops::backend::cpu::half_to_float;
using nnops::backend::cpu::float_to_half;

template <bool zero_mode = false>
void mma_packed_ab_mr6_nr16_mrxncxkc_f16(
    int mr, int nc, int kc, half* NNOPS_RESTRICT C, int ldc,
    const half* NNOPS_RESTRICT A,
    const half* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max) noexcept {

    // Rows past `mr` alias the last live row in both A and C: they compute a
    // duplicate of it and store it back to the same place, so a partial-M tile
    // needs no per-row guards (the XNNPACK broadcast convention).
    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;
    const int r4 = (mr > 4) ? 4 : r3;
    const int r5 = (mr > 5) ? 5 : r4;

    half* c0_ptr = C + 0 * ldc;
    half* c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    half* c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    half* c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;
    half* c4_ptr = (mr > 4) ? C + 4 * ldc : c3_ptr;
    half* c5_ptr = (mr > 5) ? C + 5 * ldc : c4_ptr;

    int ldb = static_cast<int>(mma_pack_align(16 * kc * sizeof(half)) / sizeof(half));

    int n = 0;
    for(; n < nc - 15; n += 16) {
        __m256 v_acc0x01234567 = _mm256_setzero_ps();
        __m256 v_acc0x89abcdef = _mm256_setzero_ps();
        __m256 v_acc1x01234567 = v_acc0x01234567;
        __m256 v_acc1x89abcdef = v_acc0x89abcdef;
        __m256 v_acc2x01234567 = v_acc0x01234567;
        __m256 v_acc2x89abcdef = v_acc0x89abcdef;
        __m256 v_acc3x01234567 = v_acc0x01234567;
        __m256 v_acc3x89abcdef = v_acc0x89abcdef;
        __m256 v_acc4x01234567 = v_acc0x01234567;
        __m256 v_acc4x89abcdef = v_acc0x89abcdef;
        __m256 v_acc5x01234567 = v_acc0x01234567;
        __m256 v_acc5x89abcdef = v_acc0x89abcdef;

        const half* a_ptr = A;
        const half* b_ptr = B;
        for(int k = 0; k < kc; ++k) {
            const __m256 v_b0 = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 0 * 8)));
            const __m256 v_b1 = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 1 * 8)));
            b_ptr += 16;

            __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[0].bits)));
            __m256 v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r1].bits)));
            v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
            v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
            v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
            v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);

            v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r2].bits)));
            v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r3].bits)));
            v_acc2x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc2x01234567);
            v_acc2x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc2x89abcdef);
            v_acc3x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc3x01234567);
            v_acc3x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc3x89abcdef);

            v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r4].bits)));
            v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r5].bits)));
            v_acc4x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc4x01234567);
            v_acc4x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc4x89abcdef);
            v_acc5x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc5x01234567);
            v_acc5x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc5x89abcdef);

            a_ptr += 6;
        }
        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr + 0))));
            v_acc0x89abcdef = _mm256_add_ps(v_acc0x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr + 8))));
            v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c1_ptr + 0))));
            v_acc1x89abcdef = _mm256_add_ps(v_acc1x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c1_ptr + 8))));
            v_acc2x01234567 = _mm256_add_ps(v_acc2x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c2_ptr + 0))));
            v_acc2x89abcdef = _mm256_add_ps(v_acc2x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c2_ptr + 8))));
            v_acc3x01234567 = _mm256_add_ps(v_acc3x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c3_ptr + 0))));
            v_acc3x89abcdef = _mm256_add_ps(v_acc3x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c3_ptr + 8))));
            v_acc4x01234567 = _mm256_add_ps(v_acc4x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c4_ptr + 0))));
            v_acc4x89abcdef = _mm256_add_ps(v_acc4x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c4_ptr + 8))));
            v_acc5x01234567 = _mm256_add_ps(v_acc5x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c5_ptr + 0))));
            v_acc5x89abcdef = _mm256_add_ps(v_acc5x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c5_ptr + 8))));
        }
        const __m256 v_min = _mm256_set1_ps(clamp_min);
        const __m256 v_max = _mm256_set1_ps(clamp_max);

        v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
        v_acc0x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc0x89abcdef, v_max), v_min);
        v_acc1x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc1x01234567, v_max), v_min);
        v_acc1x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc1x89abcdef, v_max), v_min);
        v_acc2x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc2x01234567, v_max), v_min);
        v_acc2x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc2x89abcdef, v_max), v_min);
        v_acc3x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc3x01234567, v_max), v_min);
        v_acc3x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc3x89abcdef, v_max), v_min);
        v_acc4x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc4x01234567, v_max), v_min);
        v_acc4x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc4x89abcdef, v_max), v_min);
        v_acc5x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc5x01234567, v_max), v_min);
        v_acc5x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc5x89abcdef, v_max), v_min);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr + 0),
                         _mm256_cvtps_ph(v_acc0x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr + 8),
                         _mm256_cvtps_ph(v_acc0x89abcdef, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c1_ptr + 0),
                         _mm256_cvtps_ph(v_acc1x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c1_ptr + 8),
                         _mm256_cvtps_ph(v_acc1x89abcdef, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c2_ptr + 0),
                         _mm256_cvtps_ph(v_acc2x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c2_ptr + 8),
                         _mm256_cvtps_ph(v_acc2x89abcdef, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c3_ptr + 0),
                         _mm256_cvtps_ph(v_acc3x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c3_ptr + 8),
                         _mm256_cvtps_ph(v_acc3x89abcdef, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c4_ptr + 0),
                         _mm256_cvtps_ph(v_acc4x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c4_ptr + 8),
                         _mm256_cvtps_ph(v_acc4x89abcdef, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c5_ptr + 0),
                         _mm256_cvtps_ph(v_acc5x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c5_ptr + 8),
                         _mm256_cvtps_ph(v_acc5x89abcdef, _MM_FROUND_TO_NEAREST_INT));

        c0_ptr += 16;
        c1_ptr += 16;
        c2_ptr += 16;
        c3_ptr += 16;
        c4_ptr += 16;
        c5_ptr += 16;

        B += ldb;
    }

    int rem = nc - n;

    if (rem > 0) {
        // The tail is one full 16-wide panel in the packed layout -- the
        // packer zero-fills the columns past `nc`, so both halves can always
        // be read at the same 16-element stride. At most 8 columns still fit
        // one 8-lane read; a wider tail reads both halves and reuses the
        // full-panel body, so it costs one k-loop instead of two.
        __m256 v_acc0x01234567 = _mm256_setzero_ps();
        __m256 v_acc0x89abcdef = v_acc0x01234567;
        __m256 v_acc1x01234567 = _mm256_setzero_ps();
        __m256 v_acc1x89abcdef = v_acc1x01234567;
        __m256 v_acc2x01234567 = _mm256_setzero_ps();
        __m256 v_acc2x89abcdef = v_acc2x01234567;
        __m256 v_acc3x01234567 = _mm256_setzero_ps();
        __m256 v_acc3x89abcdef = v_acc3x01234567;
        __m256 v_acc4x01234567 = _mm256_setzero_ps();
        __m256 v_acc4x89abcdef = v_acc4x01234567;
        __m256 v_acc5x01234567 = _mm256_setzero_ps();
        __m256 v_acc5x89abcdef = v_acc5x01234567;

        const half* a_ptr = A;
        const half* b_ptr = B;
        if (rem > 8) {
            for(int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 0 * 8)));
                const __m256 v_b1 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 1 * 8)));
                b_ptr += 16;

                __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[0].bits)));
                __m256 v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r1].bits)));
                v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
                v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
                v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
                v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);

                v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r2].bits)));
                v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r3].bits)));
                v_acc2x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc2x01234567);
                v_acc2x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc2x89abcdef);
                v_acc3x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc3x01234567);
                v_acc3x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc3x89abcdef);

                v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r4].bits)));
                v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r5].bits)));
                v_acc4x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc4x01234567);
                v_acc4x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc4x89abcdef);
                v_acc5x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc5x01234567);
                v_acc5x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc5x89abcdef);

                a_ptr += 6;
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr))); b_ptr += 16;
                v_acc0x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[0].bits))), v_b0, v_acc0x01234567);
                v_acc1x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r1].bits))), v_b0, v_acc1x01234567);
                v_acc2x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r2].bits))), v_b0, v_acc2x01234567);
                v_acc3x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r3].bits))), v_b0, v_acc3x01234567);
                v_acc4x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r4].bits))), v_b0, v_acc4x01234567);
                v_acc5x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r5].bits))), v_b0, v_acc5x01234567);
                a_ptr += 6;
            }
        }

        if (rem >= 8) {

            if constexpr (!zero_mode) {
                v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr))));
                v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c1_ptr))));
                v_acc2x01234567 = _mm256_add_ps(v_acc2x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c2_ptr))));
                v_acc3x01234567 = _mm256_add_ps(v_acc3x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c3_ptr))));
                v_acc4x01234567 = _mm256_add_ps(v_acc4x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c4_ptr))));
                v_acc5x01234567 = _mm256_add_ps(v_acc5x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c5_ptr))));
            }

            const __m256 v_min = _mm256_set1_ps(clamp_min);
            const __m256 v_max = _mm256_set1_ps(clamp_max);

            v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
            v_acc1x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc1x01234567, v_max), v_min);
            v_acc2x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc2x01234567, v_max), v_min);
            v_acc3x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc3x01234567, v_max), v_min);
            v_acc4x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc4x01234567, v_max), v_min);
            v_acc5x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc5x01234567, v_max), v_min);

            _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr),
                             _mm256_cvtps_ph(v_acc0x01234567, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c1_ptr),
                             _mm256_cvtps_ph(v_acc1x01234567, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c2_ptr),
                             _mm256_cvtps_ph(v_acc2x01234567, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c3_ptr),
                             _mm256_cvtps_ph(v_acc3x01234567, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c4_ptr),
                             _mm256_cvtps_ph(v_acc4x01234567, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c5_ptr),
                             _mm256_cvtps_ph(v_acc5x01234567, _MM_FROUND_TO_NEAREST_INT));

            c0_ptr += 8;
            c1_ptr += 8;
            c2_ptr += 8;
            c3_ptr += 8;
            c4_ptr += 8;
            c5_ptr += 8;

            if (rem > 8) {
                v_acc0x01234567 = v_acc0x89abcdef;
                v_acc1x01234567 = v_acc1x89abcdef;
                v_acc2x01234567 = v_acc2x89abcdef;
                v_acc3x01234567 = v_acc3x89abcdef;
                v_acc4x01234567 = v_acc4x89abcdef;
                v_acc5x01234567 = v_acc5x89abcdef;
            }
            rem -= 8;
        }

        if (rem > 0) {
            // Lane j holds column j, so the 4/2/1 ladder is just a lo/hi split of
            // the same accumulator: after the 4-wide step the leftover 1..3 columns
            // sit at lane 0 of the high half. Four fp32 lanes convert to four
            // halves, so the partial C stores are an 8/4/2-byte store of that.
            __m128 v_acc0x0123 = _mm256_castps256_ps128(v_acc0x01234567);
            __m128 v_acc1x0123 = _mm256_castps256_ps128(v_acc1x01234567);
            __m128 v_acc2x0123 = _mm256_castps256_ps128(v_acc2x01234567);
            __m128 v_acc3x0123 = _mm256_castps256_ps128(v_acc3x01234567);
            __m128 v_acc4x0123 = _mm256_castps256_ps128(v_acc4x01234567);
            __m128 v_acc5x0123 = _mm256_castps256_ps128(v_acc5x01234567);

            const __m128 v_min = _mm_set1_ps(clamp_min);
            const __m128 v_max = _mm_set1_ps(clamp_max);

            if (rem & 4) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0_ptr))));
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c1_ptr))));
                    v_acc2x0123 = _mm_add_ps(v_acc2x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c2_ptr))));
                    v_acc3x0123 = _mm_add_ps(v_acc3x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c3_ptr))));
                    v_acc4x0123 = _mm_add_ps(v_acc4x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c4_ptr))));
                    v_acc5x0123 = _mm_add_ps(v_acc5x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c5_ptr))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ps(_mm_min_ps(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ps(_mm_min_ps(v_acc3x0123, v_max), v_min);
                v_acc4x0123 = _mm_max_ps(_mm_min_ps(v_acc4x0123, v_max), v_min);
                v_acc5x0123 = _mm_max_ps(_mm_min_ps(v_acc5x0123, v_max), v_min);

                _mm_storel_epi64(reinterpret_cast<__m128i*>(c0_ptr),
                                 _mm_cvtps_ph(v_acc0x0123, _MM_FROUND_TO_NEAREST_INT));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c1_ptr),
                                 _mm_cvtps_ph(v_acc1x0123, _MM_FROUND_TO_NEAREST_INT));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c2_ptr),
                                 _mm_cvtps_ph(v_acc2x0123, _MM_FROUND_TO_NEAREST_INT));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c3_ptr),
                                 _mm_cvtps_ph(v_acc3x0123, _MM_FROUND_TO_NEAREST_INT));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c4_ptr),
                                 _mm_cvtps_ph(v_acc4x0123, _MM_FROUND_TO_NEAREST_INT));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c5_ptr),
                                 _mm_cvtps_ph(v_acc5x0123, _MM_FROUND_TO_NEAREST_INT));

                c0_ptr += 4;
                c1_ptr += 4;
                c2_ptr += 4;
                c3_ptr += 4;
                c4_ptr += 4;
                c5_ptr += 4;

                v_acc0x0123 = _mm256_extractf128_ps(v_acc0x01234567, 1);
                v_acc1x0123 = _mm256_extractf128_ps(v_acc1x01234567, 1);
                v_acc2x0123 = _mm256_extractf128_ps(v_acc2x01234567, 1);
                v_acc3x0123 = _mm256_extractf128_ps(v_acc3x01234567, 1);
                v_acc4x0123 = _mm256_extractf128_ps(v_acc4x01234567, 1);
                v_acc5x0123 = _mm256_extractf128_ps(v_acc5x01234567, 1);
            }

            if (rem & 2) {
                // The clamp below is 4-wide, so keep the untouched leftovers for
                // the 1-wide step -- lanes 2,3 are what it will store.
                const __m128 v_rest0 = v_acc0x0123;
                const __m128 v_rest1 = v_acc1x0123;
                const __m128 v_rest2 = v_acc2x0123;
                const __m128 v_rest3 = v_acc3x0123;
                const __m128 v_rest4 = v_acc4x0123;
                const __m128 v_rest5 = v_acc5x0123;

                if constexpr (!zero_mode) {
                    uint32_t c_bits;
                    c_bits = static_cast<uint32_t>(c0_ptr[0].bits) |
                             (static_cast<uint32_t>(c0_ptr[1].bits) << 16);
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                    c_bits = static_cast<uint32_t>(c1_ptr[0].bits) |
                             (static_cast<uint32_t>(c1_ptr[1].bits) << 16);
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                    c_bits = static_cast<uint32_t>(c2_ptr[0].bits) |
                             (static_cast<uint32_t>(c2_ptr[1].bits) << 16);
                    v_acc2x0123 = _mm_add_ps(v_acc2x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                    c_bits = static_cast<uint32_t>(c3_ptr[0].bits) |
                             (static_cast<uint32_t>(c3_ptr[1].bits) << 16);
                    v_acc3x0123 = _mm_add_ps(v_acc3x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                    c_bits = static_cast<uint32_t>(c4_ptr[0].bits) |
                             (static_cast<uint32_t>(c4_ptr[1].bits) << 16);
                    v_acc4x0123 = _mm_add_ps(v_acc4x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                    c_bits = static_cast<uint32_t>(c5_ptr[0].bits) |
                             (static_cast<uint32_t>(c5_ptr[1].bits) << 16);
                    v_acc5x0123 = _mm_add_ps(v_acc5x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ps(_mm_min_ps(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ps(_mm_min_ps(v_acc3x0123, v_max), v_min);
                v_acc4x0123 = _mm_max_ps(_mm_min_ps(v_acc4x0123, v_max), v_min);
                v_acc5x0123 = _mm_max_ps(_mm_min_ps(v_acc5x0123, v_max), v_min);

                {
                    const __m128i v_h0 = _mm_cvtps_ph(
                        v_acc0x0123, _MM_FROUND_TO_NEAREST_INT);
                    c0_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h0, 0));
                    c0_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h0, 1));
                    const __m128i v_h1 = _mm_cvtps_ph(
                        v_acc1x0123, _MM_FROUND_TO_NEAREST_INT);
                    c1_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h1, 0));
                    c1_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h1, 1));
                    const __m128i v_h2 = _mm_cvtps_ph(
                        v_acc2x0123, _MM_FROUND_TO_NEAREST_INT);
                    c2_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h2, 0));
                    c2_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h2, 1));
                    const __m128i v_h3 = _mm_cvtps_ph(
                        v_acc3x0123, _MM_FROUND_TO_NEAREST_INT);
                    c3_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h3, 0));
                    c3_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h3, 1));
                    const __m128i v_h4 = _mm_cvtps_ph(
                        v_acc4x0123, _MM_FROUND_TO_NEAREST_INT);
                    c4_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h4, 0));
                    c4_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h4, 1));
                    const __m128i v_h5 = _mm_cvtps_ph(
                        v_acc5x0123, _MM_FROUND_TO_NEAREST_INT);
                    c5_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h5, 0));
                    c5_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h5, 1));
                }

                c0_ptr += 2;
                c1_ptr += 2;
                c2_ptr += 2;
                c3_ptr += 2;
                c4_ptr += 2;
                c5_ptr += 2;

                v_acc0x0123 = _mm_movehl_ps(v_rest0, v_rest0);
                v_acc1x0123 = _mm_movehl_ps(v_rest1, v_rest1);
                v_acc2x0123 = _mm_movehl_ps(v_rest2, v_rest2);
                v_acc3x0123 = _mm_movehl_ps(v_rest3, v_rest3);
                v_acc4x0123 = _mm_movehl_ps(v_rest4, v_rest4);
                v_acc5x0123 = _mm_movehl_ps(v_rest5, v_rest5);
            }

            if (rem & 1) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ss(v_acc0x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c0_ptr->bits))));
                    v_acc1x0123 = _mm_add_ss(v_acc1x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c1_ptr->bits))));
                    v_acc2x0123 = _mm_add_ss(v_acc2x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c2_ptr->bits))));
                    v_acc3x0123 = _mm_add_ss(v_acc3x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c3_ptr->bits))));
                    v_acc4x0123 = _mm_add_ss(v_acc4x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c4_ptr->bits))));
                    v_acc5x0123 = _mm_add_ss(v_acc5x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c5_ptr->bits))));
                }

                v_acc0x0123 = _mm_max_ss(_mm_min_ss(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ss(_mm_min_ss(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ss(_mm_min_ss(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ss(_mm_min_ss(v_acc3x0123, v_max), v_min);
                v_acc4x0123 = _mm_max_ss(_mm_min_ss(v_acc4x0123, v_max), v_min);
                v_acc5x0123 = _mm_max_ss(_mm_min_ss(v_acc5x0123, v_max), v_min);

                c0_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc0x0123, _MM_FROUND_TO_NEAREST_INT), 0));
                c1_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc1x0123, _MM_FROUND_TO_NEAREST_INT), 0));
                c2_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc2x0123, _MM_FROUND_TO_NEAREST_INT), 0));
                c3_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc3x0123, _MM_FROUND_TO_NEAREST_INT), 0));
                c4_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc4x0123, _MM_FROUND_TO_NEAREST_INT), 0));
                c5_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc5x0123, _MM_FROUND_TO_NEAREST_INT), 0));
            }
        }
    }
}

template <bool zero_mode = false>
void mma_packed_ab_mr4_nr16_mrxncxkc_f16(
    int mr, int nc, int kc, half* NNOPS_RESTRICT C, int ldc,
    const half* NNOPS_RESTRICT A,
    const half* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max) noexcept {

    // Rows past `mr` alias the last live row in both A and C (see the mr6
    // kernel): a partial-M tile needs no per-row guards.
    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;

    half* c0_ptr = C + 0 * ldc;
    half* c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    half* c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    half* c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;

    int ldb = static_cast<int>(mma_pack_align(16 * kc * sizeof(half)) / sizeof(half));

    int n = 0;
    for(; n < nc - 15; n += 16) {
        __m256 v_acc0x01234567 = _mm256_setzero_ps();
        __m256 v_acc0x89abcdef = _mm256_setzero_ps();
        __m256 v_acc1x01234567 = v_acc0x01234567;
        __m256 v_acc1x89abcdef = v_acc0x89abcdef;
        __m256 v_acc2x01234567 = v_acc0x01234567;
        __m256 v_acc2x89abcdef = v_acc0x89abcdef;
        __m256 v_acc3x01234567 = v_acc0x01234567;
        __m256 v_acc3x89abcdef = v_acc0x89abcdef;

        const half* a_ptr = A;
        const half* b_ptr = B;
        for(int k = 0; k < kc; ++k) {
            const __m256 v_b0 = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 0 * 8)));
            const __m256 v_b1 = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 1 * 8)));
            b_ptr += 16;

            __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[0].bits)));
            __m256 v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r1].bits)));
            v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
            v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
            v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
            v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);

            v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r2].bits)));
            v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r3].bits)));
            v_acc2x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc2x01234567);
            v_acc2x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc2x89abcdef);
            v_acc3x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc3x01234567);
            v_acc3x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc3x89abcdef);

            a_ptr += 4;
        }
        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr + 0))));
            v_acc0x89abcdef = _mm256_add_ps(v_acc0x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr + 8))));
            v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c1_ptr + 0))));
            v_acc1x89abcdef = _mm256_add_ps(v_acc1x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c1_ptr + 8))));
            v_acc2x01234567 = _mm256_add_ps(v_acc2x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c2_ptr + 0))));
            v_acc2x89abcdef = _mm256_add_ps(v_acc2x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c2_ptr + 8))));
            v_acc3x01234567 = _mm256_add_ps(v_acc3x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c3_ptr + 0))));
            v_acc3x89abcdef = _mm256_add_ps(v_acc3x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c3_ptr + 8))));
        }
        const __m256 v_min = _mm256_set1_ps(clamp_min);
        const __m256 v_max = _mm256_set1_ps(clamp_max);

        v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
        v_acc0x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc0x89abcdef, v_max), v_min);
        v_acc1x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc1x01234567, v_max), v_min);
        v_acc1x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc1x89abcdef, v_max), v_min);
        v_acc2x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc2x01234567, v_max), v_min);
        v_acc2x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc2x89abcdef, v_max), v_min);
        v_acc3x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc3x01234567, v_max), v_min);
        v_acc3x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc3x89abcdef, v_max), v_min);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr + 0),
                         _mm256_cvtps_ph(v_acc0x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr + 8),
                         _mm256_cvtps_ph(v_acc0x89abcdef, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c1_ptr + 0),
                         _mm256_cvtps_ph(v_acc1x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c1_ptr + 8),
                         _mm256_cvtps_ph(v_acc1x89abcdef, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c2_ptr + 0),
                         _mm256_cvtps_ph(v_acc2x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c2_ptr + 8),
                         _mm256_cvtps_ph(v_acc2x89abcdef, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c3_ptr + 0),
                         _mm256_cvtps_ph(v_acc3x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c3_ptr + 8),
                         _mm256_cvtps_ph(v_acc3x89abcdef, _MM_FROUND_TO_NEAREST_INT));

        c0_ptr += 16;
        c1_ptr += 16;
        c2_ptr += 16;
        c3_ptr += 16;

        B += ldb;
    }

    int rem = nc - n;

    if (rem > 0) {
        // The tail is one full 16-wide panel in the packed layout -- the
        // packer zero-fills the columns past `nc`, so both halves can always
        // be read at the same 16-element stride. At most 8 columns still fit
        // one 8-lane read; a wider tail reads both halves and reuses the
        // full-panel body, so it costs one k-loop instead of two.
        __m256 v_acc0x01234567 = _mm256_setzero_ps();
        __m256 v_acc0x89abcdef = v_acc0x01234567;
        __m256 v_acc1x01234567 = _mm256_setzero_ps();
        __m256 v_acc1x89abcdef = v_acc1x01234567;
        __m256 v_acc2x01234567 = _mm256_setzero_ps();
        __m256 v_acc2x89abcdef = v_acc2x01234567;
        __m256 v_acc3x01234567 = _mm256_setzero_ps();
        __m256 v_acc3x89abcdef = v_acc3x01234567;

        const half* a_ptr = A;
        const half* b_ptr = B;
        if (rem > 8) {
            for(int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 0 * 8)));
                const __m256 v_b1 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 1 * 8)));
                b_ptr += 16;

                __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[0].bits)));
                __m256 v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r1].bits)));
                v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
                v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
                v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
                v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);

                v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r2].bits)));
                v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r3].bits)));
                v_acc2x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc2x01234567);
                v_acc2x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc2x89abcdef);
                v_acc3x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc3x01234567);
                v_acc3x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc3x89abcdef);

                a_ptr += 4;
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr))); b_ptr += 16;
                v_acc0x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[0].bits))), v_b0, v_acc0x01234567);
                v_acc1x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r1].bits))), v_b0, v_acc1x01234567);
                v_acc2x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r2].bits))), v_b0, v_acc2x01234567);
                v_acc3x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r3].bits))), v_b0, v_acc3x01234567);
                a_ptr += 4;
            }
        }

        if (rem >= 8) {

            if constexpr (!zero_mode) {
                v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr))));
                v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c1_ptr))));
                v_acc2x01234567 = _mm256_add_ps(v_acc2x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c2_ptr))));
                v_acc3x01234567 = _mm256_add_ps(v_acc3x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c3_ptr))));
            }

            const __m256 v_min = _mm256_set1_ps(clamp_min);
            const __m256 v_max = _mm256_set1_ps(clamp_max);

            v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
            v_acc1x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc1x01234567, v_max), v_min);
            v_acc2x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc2x01234567, v_max), v_min);
            v_acc3x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc3x01234567, v_max), v_min);

            _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr),
                             _mm256_cvtps_ph(v_acc0x01234567, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c1_ptr),
                             _mm256_cvtps_ph(v_acc1x01234567, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c2_ptr),
                             _mm256_cvtps_ph(v_acc2x01234567, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c3_ptr),
                             _mm256_cvtps_ph(v_acc3x01234567, _MM_FROUND_TO_NEAREST_INT));

            c0_ptr += 8;
            c1_ptr += 8;
            c2_ptr += 8;
            c3_ptr += 8;

            if (rem > 8) {
                v_acc0x01234567 = v_acc0x89abcdef;
                v_acc1x01234567 = v_acc1x89abcdef;
                v_acc2x01234567 = v_acc2x89abcdef;
                v_acc3x01234567 = v_acc3x89abcdef;
            }
            rem -= 8;
        }

        if (rem > 0) {
            // Lane j holds column j, so the 4/2/1 ladder is just a lo/hi split of
            // the same accumulator: after the 4-wide step the leftover 1..3 columns
            // sit at lane 0 of the high half. Four fp32 lanes convert to four
            // halves, so the partial C stores are an 8/4/2-byte store of that.
            __m128 v_acc0x0123 = _mm256_castps256_ps128(v_acc0x01234567);
            __m128 v_acc1x0123 = _mm256_castps256_ps128(v_acc1x01234567);
            __m128 v_acc2x0123 = _mm256_castps256_ps128(v_acc2x01234567);
            __m128 v_acc3x0123 = _mm256_castps256_ps128(v_acc3x01234567);

            const __m128 v_min = _mm_set1_ps(clamp_min);
            const __m128 v_max = _mm_set1_ps(clamp_max);

            if (rem & 4) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0_ptr))));
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c1_ptr))));
                    v_acc2x0123 = _mm_add_ps(v_acc2x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c2_ptr))));
                    v_acc3x0123 = _mm_add_ps(v_acc3x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c3_ptr))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ps(_mm_min_ps(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ps(_mm_min_ps(v_acc3x0123, v_max), v_min);

                _mm_storel_epi64(reinterpret_cast<__m128i*>(c0_ptr),
                                 _mm_cvtps_ph(v_acc0x0123, _MM_FROUND_TO_NEAREST_INT));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c1_ptr),
                                 _mm_cvtps_ph(v_acc1x0123, _MM_FROUND_TO_NEAREST_INT));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c2_ptr),
                                 _mm_cvtps_ph(v_acc2x0123, _MM_FROUND_TO_NEAREST_INT));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c3_ptr),
                                 _mm_cvtps_ph(v_acc3x0123, _MM_FROUND_TO_NEAREST_INT));

                c0_ptr += 4;
                c1_ptr += 4;
                c2_ptr += 4;
                c3_ptr += 4;

                v_acc0x0123 = _mm256_extractf128_ps(v_acc0x01234567, 1);
                v_acc1x0123 = _mm256_extractf128_ps(v_acc1x01234567, 1);
                v_acc2x0123 = _mm256_extractf128_ps(v_acc2x01234567, 1);
                v_acc3x0123 = _mm256_extractf128_ps(v_acc3x01234567, 1);
            }

            if (rem & 2) {
                // The clamp below is 4-wide, so keep the untouched leftovers for
                // the 1-wide step -- lanes 2,3 are what it will store.
                const __m128 v_rest0 = v_acc0x0123;
                const __m128 v_rest1 = v_acc1x0123;
                const __m128 v_rest2 = v_acc2x0123;
                const __m128 v_rest3 = v_acc3x0123;

                if constexpr (!zero_mode) {
                    uint32_t c_bits;
                    c_bits = static_cast<uint32_t>(c0_ptr[0].bits) |
                             (static_cast<uint32_t>(c0_ptr[1].bits) << 16);
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                    c_bits = static_cast<uint32_t>(c1_ptr[0].bits) |
                             (static_cast<uint32_t>(c1_ptr[1].bits) << 16);
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                    c_bits = static_cast<uint32_t>(c2_ptr[0].bits) |
                             (static_cast<uint32_t>(c2_ptr[1].bits) << 16);
                    v_acc2x0123 = _mm_add_ps(v_acc2x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                    c_bits = static_cast<uint32_t>(c3_ptr[0].bits) |
                             (static_cast<uint32_t>(c3_ptr[1].bits) << 16);
                    v_acc3x0123 = _mm_add_ps(v_acc3x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ps(_mm_min_ps(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ps(_mm_min_ps(v_acc3x0123, v_max), v_min);

                {
                    const __m128i v_h0 = _mm_cvtps_ph(
                        v_acc0x0123, _MM_FROUND_TO_NEAREST_INT);
                    c0_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h0, 0));
                    c0_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h0, 1));
                    const __m128i v_h1 = _mm_cvtps_ph(
                        v_acc1x0123, _MM_FROUND_TO_NEAREST_INT);
                    c1_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h1, 0));
                    c1_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h1, 1));
                    const __m128i v_h2 = _mm_cvtps_ph(
                        v_acc2x0123, _MM_FROUND_TO_NEAREST_INT);
                    c2_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h2, 0));
                    c2_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h2, 1));
                    const __m128i v_h3 = _mm_cvtps_ph(
                        v_acc3x0123, _MM_FROUND_TO_NEAREST_INT);
                    c3_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h3, 0));
                    c3_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h3, 1));
                }

                c0_ptr += 2;
                c1_ptr += 2;
                c2_ptr += 2;
                c3_ptr += 2;

                v_acc0x0123 = _mm_movehl_ps(v_rest0, v_rest0);
                v_acc1x0123 = _mm_movehl_ps(v_rest1, v_rest1);
                v_acc2x0123 = _mm_movehl_ps(v_rest2, v_rest2);
                v_acc3x0123 = _mm_movehl_ps(v_rest3, v_rest3);
            }

            if (rem & 1) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ss(v_acc0x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c0_ptr->bits))));
                    v_acc1x0123 = _mm_add_ss(v_acc1x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c1_ptr->bits))));
                    v_acc2x0123 = _mm_add_ss(v_acc2x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c2_ptr->bits))));
                    v_acc3x0123 = _mm_add_ss(v_acc3x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c3_ptr->bits))));
                }

                v_acc0x0123 = _mm_max_ss(_mm_min_ss(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ss(_mm_min_ss(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ss(_mm_min_ss(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ss(_mm_min_ss(v_acc3x0123, v_max), v_min);

                c0_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc0x0123, _MM_FROUND_TO_NEAREST_INT), 0));
                c1_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc1x0123, _MM_FROUND_TO_NEAREST_INT), 0));
                c2_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc2x0123, _MM_FROUND_TO_NEAREST_INT), 0));
                c3_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc3x0123, _MM_FROUND_TO_NEAREST_INT), 0));
            }
        }
    }
}

template <bool zero_mode = false>
void mma_packed_ab_mr2_nr16_mrxncxkc_f16(
    int mr, int nc, int kc, half* NNOPS_RESTRICT C, int ldc,
    const half* NNOPS_RESTRICT A,
    const half* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max) noexcept {

    // Rows past `mr` alias the last live row in both A and C: they compute a
    // duplicate of it and store it back to the same place, so a partial-M tile
    // needs no per-row guards (the XNNPACK broadcast convention).
    const int r1 = (mr > 1) ? 1 : 0;

    half* c0_ptr = C + 0 * ldc;
    half* c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;

    int ldb = static_cast<int>(mma_pack_align(16 * kc * sizeof(half)) / sizeof(half));

    int n = 0;
    for(; n < nc - 15; n += 16) {
        __m256 v_acc0x01234567 = _mm256_setzero_ps();
        __m256 v_acc0x89abcdef = _mm256_setzero_ps();
        __m256 v_acc1x01234567 = v_acc0x01234567;
        __m256 v_acc1x89abcdef = v_acc0x89abcdef;

        const half* a_ptr = A;
        const half* b_ptr = B;
        for(int k = 0; k < kc; ++k) {
            const __m256 v_b0 = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 0 * 8)));
            const __m256 v_b1 = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 1 * 8)));
            b_ptr += 16;

            const __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[0].bits)));
            const __m256 v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r1].bits)));
            v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
            v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
            v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
            v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);

            a_ptr += 2;
        }
        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr + 0))));
            v_acc0x89abcdef = _mm256_add_ps(v_acc0x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr + 8))));
            v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c1_ptr + 0))));
            v_acc1x89abcdef = _mm256_add_ps(v_acc1x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c1_ptr + 8))));
        }
        const __m256 v_min = _mm256_set1_ps(clamp_min);
        const __m256 v_max = _mm256_set1_ps(clamp_max);

        v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
        v_acc0x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc0x89abcdef, v_max), v_min);
        v_acc1x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc1x01234567, v_max), v_min);
        v_acc1x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc1x89abcdef, v_max), v_min);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr + 0),
                         _mm256_cvtps_ph(v_acc0x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr + 8),
                         _mm256_cvtps_ph(v_acc0x89abcdef, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c1_ptr + 0),
                         _mm256_cvtps_ph(v_acc1x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c1_ptr + 8),
                         _mm256_cvtps_ph(v_acc1x89abcdef, _MM_FROUND_TO_NEAREST_INT));

        c0_ptr += 16;
        c1_ptr += 16;

        B += ldb;
    }

    int rem = nc - n;

    if (rem > 0) {
        // The tail is one full 16-wide panel in the packed layout -- the
        // packer zero-fills the columns past `nc`, so both halves can always
        // be read at the same 16-element stride. At most 8 columns still fit
        // one 8-lane read; a wider tail reads both halves and reuses the
        // full-panel body, so it costs one k-loop instead of two.
        __m256 v_acc0x01234567 = _mm256_setzero_ps();
        __m256 v_acc0x89abcdef = v_acc0x01234567;
        __m256 v_acc1x01234567 = _mm256_setzero_ps();
        __m256 v_acc1x89abcdef = v_acc1x01234567;

        const half* a_ptr = A;
        const half* b_ptr = B;
        if (rem > 8) {
            for(int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 0 * 8)));
                const __m256 v_b1 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 1 * 8)));
                b_ptr += 16;

                const __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[0].bits)));
                const __m256 v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r1].bits)));
                v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
                v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
                v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
                v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);

                a_ptr += 2;
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr))); b_ptr += 16;
                v_acc0x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[0].bits))), v_b0, v_acc0x01234567);
                v_acc1x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[r1].bits))), v_b0, v_acc1x01234567);
                a_ptr += 2;
            }
        }

        if (rem >= 8) {

            if constexpr (!zero_mode) {
                v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr))));
                v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c1_ptr))));
            }

            const __m256 v_min = _mm256_set1_ps(clamp_min);
            const __m256 v_max = _mm256_set1_ps(clamp_max);

            v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
            v_acc1x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc1x01234567, v_max), v_min);

            _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr),
                             _mm256_cvtps_ph(v_acc0x01234567, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c1_ptr),
                             _mm256_cvtps_ph(v_acc1x01234567, _MM_FROUND_TO_NEAREST_INT));

            c0_ptr += 8;
            c1_ptr += 8;

            if (rem > 8) {
                v_acc0x01234567 = v_acc0x89abcdef;
                v_acc1x01234567 = v_acc1x89abcdef;
            }
            rem -= 8;
        }

        if (rem > 0) {
            // Lane j holds column j, so the 4/2/1 ladder is just a lo/hi split of
            // the same accumulator: after the 4-wide step the leftover 1..3 columns
            // sit at lane 0 of the high half. Four fp32 lanes convert to four
            // halves, so the partial C stores are an 8/4/2-byte store of that.
            __m128 v_acc0x0123 = _mm256_castps256_ps128(v_acc0x01234567);
            __m128 v_acc1x0123 = _mm256_castps256_ps128(v_acc1x01234567);

            const __m128 v_min = _mm_set1_ps(clamp_min);
            const __m128 v_max = _mm_set1_ps(clamp_max);

            if (rem & 4) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0_ptr))));
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c1_ptr))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);

                _mm_storel_epi64(reinterpret_cast<__m128i*>(c0_ptr),
                                 _mm_cvtps_ph(v_acc0x0123, _MM_FROUND_TO_NEAREST_INT));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c1_ptr),
                                 _mm_cvtps_ph(v_acc1x0123, _MM_FROUND_TO_NEAREST_INT));

                c0_ptr += 4;
                c1_ptr += 4;

                v_acc0x0123 = _mm256_extractf128_ps(v_acc0x01234567, 1);
                v_acc1x0123 = _mm256_extractf128_ps(v_acc1x01234567, 1);
            }

            if (rem & 2) {
                // The clamp below is 4-wide, so keep the untouched leftovers for
                // the 1-wide step -- lanes 2,3 are what it will store.
                const __m128 v_rest0 = v_acc0x0123;
                const __m128 v_rest1 = v_acc1x0123;

                if constexpr (!zero_mode) {
                    uint32_t c_bits;
                    c_bits = static_cast<uint32_t>(c0_ptr[0].bits) |
                             (static_cast<uint32_t>(c0_ptr[1].bits) << 16);
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                    c_bits = static_cast<uint32_t>(c1_ptr[0].bits) |
                             (static_cast<uint32_t>(c1_ptr[1].bits) << 16);
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);

                {
                    const __m128i v_h0 = _mm_cvtps_ph(
                        v_acc0x0123, _MM_FROUND_TO_NEAREST_INT);
                    c0_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h0, 0));
                    c0_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h0, 1));
                    const __m128i v_h1 = _mm_cvtps_ph(
                        v_acc1x0123, _MM_FROUND_TO_NEAREST_INT);
                    c1_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h1, 0));
                    c1_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h1, 1));
                }

                c0_ptr += 2;
                c1_ptr += 2;

                v_acc0x0123 = _mm_movehl_ps(v_rest0, v_rest0);
                v_acc1x0123 = _mm_movehl_ps(v_rest1, v_rest1);
            }

            if (rem & 1) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ss(v_acc0x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c0_ptr->bits))));
                    v_acc1x0123 = _mm_add_ss(v_acc1x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c1_ptr->bits))));
                }

                v_acc0x0123 = _mm_max_ss(_mm_min_ss(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ss(_mm_min_ss(v_acc1x0123, v_max), v_min);

                c0_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc0x0123, _MM_FROUND_TO_NEAREST_INT), 0));
                c1_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc1x0123, _MM_FROUND_TO_NEAREST_INT), 0));
            }
        }
    }
}

template <bool zero_mode = false>
void mma_packed_ab_mr1_nr16_mrxncxkc_f16(
    int mr, int nc, int kc, half* NNOPS_RESTRICT C, int ldc,
    const half* NNOPS_RESTRICT A,
    const half* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max) noexcept {

    (void)mr;   // one row, so the row count never varies
    (void)ldc;  // one row, so the C row stride never applies

    half* c0_ptr = C;

    int ldb = static_cast<int>(mma_pack_align(16 * kc * sizeof(half)) / sizeof(half));

    int n = 0;
    for(; n < nc - 15; n += 16) {
        __m256 v_acc0x01234567 = _mm256_setzero_ps();
        __m256 v_acc0x89abcdef = _mm256_setzero_ps();

        const half* a_ptr = A;
        const half* b_ptr = B;
        for(int k = 0; k < kc; ++k) {
            const __m256 v_b0 = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 0 * 8)));
            const __m256 v_b1 = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 1 * 8)));
            b_ptr += 16;

            const __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[0].bits)));
            v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
            v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);

            a_ptr += 1;
        }
        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr + 0))));
            v_acc0x89abcdef = _mm256_add_ps(v_acc0x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr + 8))));
        }
        const __m256 v_min = _mm256_set1_ps(clamp_min);
        const __m256 v_max = _mm256_set1_ps(clamp_max);

        v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
        v_acc0x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc0x89abcdef, v_max), v_min);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr + 0),
                         _mm256_cvtps_ph(v_acc0x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr + 8),
                         _mm256_cvtps_ph(v_acc0x89abcdef, _MM_FROUND_TO_NEAREST_INT));

        c0_ptr += 16;

        B += ldb;
    }

    int rem = nc - n;

    if (rem > 0) {
        // The tail is one full 16-wide panel in the packed layout -- the
        // packer zero-fills the columns past `nc`, so both halves can always
        // be read at the same 16-element stride. At most 8 columns still fit
        // one 8-lane read; a wider tail reads both halves and reuses the
        // full-panel body, so it costs one k-loop instead of two.
        __m256 v_acc0x01234567 = _mm256_setzero_ps();
        __m256 v_acc0x89abcdef = v_acc0x01234567;

        const half* a_ptr = A;
        const half* b_ptr = B;
        if (rem > 8) {
            for(int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 0 * 8)));
                const __m256 v_b1 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 1 * 8)));
                b_ptr += 16;

                const __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[0].bits)));
                v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
                v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);

                a_ptr += 1;
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr))); b_ptr += 16;
                v_acc0x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a_ptr[0].bits))), v_b0, v_acc0x01234567);
                a_ptr += 1;
            }
        }

        if (rem >= 8) {

            if constexpr (!zero_mode) {
                v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr))));
            }

            const __m256 v_min = _mm256_set1_ps(clamp_min);
            const __m256 v_max = _mm256_set1_ps(clamp_max);

            v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);

            _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr),
                             _mm256_cvtps_ph(v_acc0x01234567, _MM_FROUND_TO_NEAREST_INT));

            c0_ptr += 8;

            if (rem > 8) {
                v_acc0x01234567 = v_acc0x89abcdef;
            }
            rem -= 8;
        }

        if (rem > 0) {
            // Lane j holds column j, so the 4/2/1 ladder is just a lo/hi split of
            // the same accumulator: after the 4-wide step the leftover 1..3 columns
            // sit at lane 0 of the high half. Four fp32 lanes convert to four
            // halves, so the partial C stores are an 8/4/2-byte store of that.
            __m128 v_acc0x0123 = _mm256_castps256_ps128(v_acc0x01234567);

            const __m128 v_min = _mm_set1_ps(clamp_min);
            const __m128 v_max = _mm_set1_ps(clamp_max);

            if (rem & 4) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0_ptr))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);

                _mm_storel_epi64(reinterpret_cast<__m128i*>(c0_ptr),
                                 _mm_cvtps_ph(v_acc0x0123, _MM_FROUND_TO_NEAREST_INT));

                c0_ptr += 4;

                v_acc0x0123 = _mm256_extractf128_ps(v_acc0x01234567, 1);
            }

            if (rem & 2) {
                // The clamp below is 4-wide, so keep the untouched leftovers for
                // the 1-wide step -- lanes 2,3 are what it will store.
                const __m128 v_rest0 = v_acc0x0123;

                if constexpr (!zero_mode) {
                    uint32_t c_bits;
                    c_bits = static_cast<uint32_t>(c0_ptr[0].bits) |
                             (static_cast<uint32_t>(c0_ptr[1].bits) << 16);
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);

                {
                    const __m128i v_h0 = _mm_cvtps_ph(
                        v_acc0x0123, _MM_FROUND_TO_NEAREST_INT);
                    c0_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h0, 0));
                    c0_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h0, 1));
                }

                c0_ptr += 2;

                v_acc0x0123 = _mm_movehl_ps(v_rest0, v_rest0);
            }

            if (rem & 1) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ss(v_acc0x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c0_ptr->bits))));
                }

                v_acc0x0123 = _mm_max_ss(_mm_min_ss(v_acc0x0123, v_max), v_min);

                c0_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc0x0123, _MM_FROUND_TO_NEAREST_INT), 0));
            }
        }
    }
}

template <bool zero_mode = false, int seed = 0>
void mma_packed_b_mr6_nr16_mrxncxkc_f16(
    int mr, int nc, int kc, half* NNOPS_RESTRICT C, int ldc,
    const half* NNOPS_RESTRICT A, int lda,
    const half* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max,
    const half* NNOPS_RESTRICT bias = nullptr) noexcept {

    // Rows past `mr` alias the last live row in both A and C: they compute a
    // duplicate of it and store it back to the same place, so a partial-M tile
    // needs no per-row guards (the XNNPACK broadcast convention).
    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;
    const int r4 = (mr > 4) ? 4 : r3;
    const int r5 = (mr > 5) ? 5 : r4;

    half* c0_ptr = C + 0 * ldc;
    half* c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    half* c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    half* c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;
    half* c4_ptr = (mr > 4) ? C + 4 * ldc : c3_ptr;
    half* c5_ptr = (mr > 5) ? C + 5 * ldc : c4_ptr;

    int ldb = static_cast<int>(mma_pack_align(16 * kc * sizeof(half)) / sizeof(half));

    int n = 0;

    // Accumulator initial value. Seed 1 is a per-row bias — the planar route,
    // where a row is an output channel; seed 2 is a per-column one, which is the
    // channel-last route's shape: the rows are pixels and the bias lands on the
    // panel's 16 lanes, so it costs two loads per panel where seeding the tile
    // first costs a full write and read-back of the output. `off` is the lane
    // offset inside the panel. A row past `mr` aliases the last live row in A
    // and C, so the same alias index keeps the row seed in range.
    const auto init = [&](int r, int off) {
        if constexpr (seed == 1) { return _mm256_set1_ps(half_to_float(bias[r])); }
        else if constexpr (seed == 2) {
            return _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(bias + n + off)));
        } else { return _mm256_setzero_ps(); }
    };

    for(; n < nc - 15; n += 16) {
        const half* a0_ptr = A + 0 * lda;
        const half* a1_ptr = A + r1 * lda;
        const half* a2_ptr = A + r2 * lda;
        const half* a3_ptr = A + r3 * lda;
        const half* a4_ptr = A + r4 * lda;
        const half* a5_ptr = A + r5 * lda;

        __m256 v_acc0x01234567 = init(0, 0);
        __m256 v_acc0x89abcdef = init(0, 8);
        __m256 v_acc1x01234567 = init(r1, 0);
        __m256 v_acc1x89abcdef = init(r1, 8);
        __m256 v_acc2x01234567 = init(r2, 0);
        __m256 v_acc2x89abcdef = init(r2, 8);
        __m256 v_acc3x01234567 = init(r3, 0);
        __m256 v_acc3x89abcdef = init(r3, 8);
        __m256 v_acc4x01234567 = init(r4, 0);
        __m256 v_acc4x89abcdef = init(r4, 8);
        __m256 v_acc5x01234567 = init(r5, 0);
        __m256 v_acc5x89abcdef = init(r5, 8);

        const half* b_ptr = B;
        for(int k = 0; k < kc; ++k) {
            const __m256 v_b0 = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 0 * 8)));
            const __m256 v_b1 = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 1 * 8)));
            b_ptr += 16;

            __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a0_ptr[0].bits))); a0_ptr += 1;
            __m256 v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a1_ptr[0].bits))); a1_ptr += 1;
            v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
            v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
            v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
            v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);

            v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a2_ptr[0].bits))); a2_ptr += 1;
            v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a3_ptr[0].bits))); a3_ptr += 1;
            v_acc2x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc2x01234567);
            v_acc2x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc2x89abcdef);
            v_acc3x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc3x01234567);
            v_acc3x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc3x89abcdef);

            v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a4_ptr[0].bits))); a4_ptr += 1;
            v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a5_ptr[0].bits))); a5_ptr += 1;
            v_acc4x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc4x01234567);
            v_acc4x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc4x89abcdef);
            v_acc5x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc5x01234567);
            v_acc5x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc5x89abcdef);
        }

        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr + 0))));
            v_acc0x89abcdef = _mm256_add_ps(v_acc0x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr + 8))));
            v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c1_ptr + 0))));
            v_acc1x89abcdef = _mm256_add_ps(v_acc1x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c1_ptr + 8))));
            v_acc2x01234567 = _mm256_add_ps(v_acc2x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c2_ptr + 0))));
            v_acc2x89abcdef = _mm256_add_ps(v_acc2x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c2_ptr + 8))));
            v_acc3x01234567 = _mm256_add_ps(v_acc3x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c3_ptr + 0))));
            v_acc3x89abcdef = _mm256_add_ps(v_acc3x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c3_ptr + 8))));
            v_acc4x01234567 = _mm256_add_ps(v_acc4x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c4_ptr + 0))));
            v_acc4x89abcdef = _mm256_add_ps(v_acc4x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c4_ptr + 8))));
            v_acc5x01234567 = _mm256_add_ps(v_acc5x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c5_ptr + 0))));
            v_acc5x89abcdef = _mm256_add_ps(v_acc5x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c5_ptr + 8))));
        }
        const __m256 v_min = _mm256_set1_ps(clamp_min);
        const __m256 v_max = _mm256_set1_ps(clamp_max);

        v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
        v_acc0x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc0x89abcdef, v_max), v_min);
        v_acc1x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc1x01234567, v_max), v_min);
        v_acc1x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc1x89abcdef, v_max), v_min);
        v_acc2x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc2x01234567, v_max), v_min);
        v_acc2x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc2x89abcdef, v_max), v_min);
        v_acc3x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc3x01234567, v_max), v_min);
        v_acc3x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc3x89abcdef, v_max), v_min);
        v_acc4x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc4x01234567, v_max), v_min);
        v_acc4x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc4x89abcdef, v_max), v_min);
        v_acc5x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc5x01234567, v_max), v_min);
        v_acc5x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc5x89abcdef, v_max), v_min);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr + 0),
                         _mm256_cvtps_ph(v_acc0x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr + 8),
                         _mm256_cvtps_ph(v_acc0x89abcdef, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c1_ptr + 0),
                         _mm256_cvtps_ph(v_acc1x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c1_ptr + 8),
                         _mm256_cvtps_ph(v_acc1x89abcdef, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c2_ptr + 0),
                         _mm256_cvtps_ph(v_acc2x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c2_ptr + 8),
                         _mm256_cvtps_ph(v_acc2x89abcdef, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c3_ptr + 0),
                         _mm256_cvtps_ph(v_acc3x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c3_ptr + 8),
                         _mm256_cvtps_ph(v_acc3x89abcdef, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c4_ptr + 0),
                         _mm256_cvtps_ph(v_acc4x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c4_ptr + 8),
                         _mm256_cvtps_ph(v_acc4x89abcdef, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c5_ptr + 0),
                         _mm256_cvtps_ph(v_acc5x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c5_ptr + 8),
                         _mm256_cvtps_ph(v_acc5x89abcdef, _MM_FROUND_TO_NEAREST_INT));

        c0_ptr += 16;
        c1_ptr += 16;
        c2_ptr += 16;
        c3_ptr += 16;
        c4_ptr += 16;
        c5_ptr += 16;

        B += ldb;
    }

    int rem = nc - n;

    if (rem > 0) {
        // The tail is one full 16-wide panel in the packed layout -- the
        // packer zero-fills the columns past `nc`, so both halves can always
        // be read at the same 16-element stride. At most 8 columns still fit
        // one 8-lane read; a wider tail reads both halves and reuses the
        // full-panel body, so it costs one k-loop instead of two.
        const half* a0_ptr = A + 0 * lda;
        const half* a1_ptr = A + r1 * lda;
        const half* a2_ptr = A + r2 * lda;
        const half* a3_ptr = A + r3 * lda;
        const half* a4_ptr = A + r4 * lda;
        const half* a5_ptr = A + r5 * lda;
        __m256 v_acc0x01234567 = init(0, 0);
        __m256 v_acc0x89abcdef = init(0, 8);
        __m256 v_acc1x01234567 = init(r1, 0);
        __m256 v_acc1x89abcdef = init(r1, 8);
        __m256 v_acc2x01234567 = init(r2, 0);
        __m256 v_acc2x89abcdef = init(r2, 8);
        __m256 v_acc3x01234567 = init(r3, 0);
        __m256 v_acc3x89abcdef = init(r3, 8);
        __m256 v_acc4x01234567 = init(r4, 0);
        __m256 v_acc4x89abcdef = init(r4, 8);
        __m256 v_acc5x01234567 = init(r5, 0);
        __m256 v_acc5x89abcdef = init(r5, 8);

        const half* b_ptr = B;
        if (rem > 8) {
            for(int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 0 * 8)));
                const __m256 v_b1 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 1 * 8)));
                b_ptr += 16;

                __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a0_ptr[0].bits))); a0_ptr += 1;
                __m256 v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a1_ptr[0].bits))); a1_ptr += 1;
                v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
                v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
                v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
                v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);

                v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a2_ptr[0].bits))); a2_ptr += 1;
                v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a3_ptr[0].bits))); a3_ptr += 1;
                v_acc2x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc2x01234567);
                v_acc2x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc2x89abcdef);
                v_acc3x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc3x01234567);
                v_acc3x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc3x89abcdef);

                v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a4_ptr[0].bits))); a4_ptr += 1;
                v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a5_ptr[0].bits))); a5_ptr += 1;
                v_acc4x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc4x01234567);
                v_acc4x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc4x89abcdef);
                v_acc5x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc5x01234567);
                v_acc5x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc5x89abcdef);
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr))); b_ptr += 16;
                v_acc0x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a0_ptr[0].bits))), v_b0, v_acc0x01234567);
                v_acc1x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a1_ptr[0].bits))), v_b0, v_acc1x01234567);
                v_acc2x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a2_ptr[0].bits))), v_b0, v_acc2x01234567);
                v_acc3x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a3_ptr[0].bits))), v_b0, v_acc3x01234567);
                v_acc4x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a4_ptr[0].bits))), v_b0, v_acc4x01234567);
                v_acc5x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a5_ptr[0].bits))), v_b0, v_acc5x01234567);
                a0_ptr += 1;
                a1_ptr += 1;
                a2_ptr += 1;
                a3_ptr += 1;
                a4_ptr += 1;
                a5_ptr += 1;
            }
        }

        if (rem >= 8) {

            if constexpr (!zero_mode) {
                v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr))));
                v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c1_ptr))));
                v_acc2x01234567 = _mm256_add_ps(v_acc2x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c2_ptr))));
                v_acc3x01234567 = _mm256_add_ps(v_acc3x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c3_ptr))));
                v_acc4x01234567 = _mm256_add_ps(v_acc4x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c4_ptr))));
                v_acc5x01234567 = _mm256_add_ps(v_acc5x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c5_ptr))));
            }

            const __m256 v_min = _mm256_set1_ps(clamp_min);
            const __m256 v_max = _mm256_set1_ps(clamp_max);

            v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
            v_acc1x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc1x01234567, v_max), v_min);
            v_acc2x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc2x01234567, v_max), v_min);
            v_acc3x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc3x01234567, v_max), v_min);
            v_acc4x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc4x01234567, v_max), v_min);
            v_acc5x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc5x01234567, v_max), v_min);

            _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr),
                             _mm256_cvtps_ph(v_acc0x01234567, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c1_ptr),
                             _mm256_cvtps_ph(v_acc1x01234567, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c2_ptr),
                             _mm256_cvtps_ph(v_acc2x01234567, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c3_ptr),
                             _mm256_cvtps_ph(v_acc3x01234567, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c4_ptr),
                             _mm256_cvtps_ph(v_acc4x01234567, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c5_ptr),
                             _mm256_cvtps_ph(v_acc5x01234567, _MM_FROUND_TO_NEAREST_INT));

            c0_ptr += 8;
            c1_ptr += 8;
            c2_ptr += 8;
            c3_ptr += 8;
            c4_ptr += 8;
            c5_ptr += 8;

            if (rem > 8) {
                v_acc0x01234567 = v_acc0x89abcdef;
                v_acc1x01234567 = v_acc1x89abcdef;
                v_acc2x01234567 = v_acc2x89abcdef;
                v_acc3x01234567 = v_acc3x89abcdef;
                v_acc4x01234567 = v_acc4x89abcdef;
                v_acc5x01234567 = v_acc5x89abcdef;
            }
            rem -= 8;
        }

        if (rem > 0) {
            // Lane j holds column j, so the 4/2/1 ladder is just a lo/hi split of
            // the same accumulator: after the 4-wide step the leftover 1..3 columns
            // sit at lane 0 of the high half. Four fp32 lanes convert to four
            // halves, so the partial C stores are an 8/4/2-byte store of that.
            __m128 v_acc0x0123 = _mm256_castps256_ps128(v_acc0x01234567);
            __m128 v_acc1x0123 = _mm256_castps256_ps128(v_acc1x01234567);
            __m128 v_acc2x0123 = _mm256_castps256_ps128(v_acc2x01234567);
            __m128 v_acc3x0123 = _mm256_castps256_ps128(v_acc3x01234567);
            __m128 v_acc4x0123 = _mm256_castps256_ps128(v_acc4x01234567);
            __m128 v_acc5x0123 = _mm256_castps256_ps128(v_acc5x01234567);

            const __m128 v_min = _mm_set1_ps(clamp_min);
            const __m128 v_max = _mm_set1_ps(clamp_max);

            if (rem & 4) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0_ptr))));
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c1_ptr))));
                    v_acc2x0123 = _mm_add_ps(v_acc2x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c2_ptr))));
                    v_acc3x0123 = _mm_add_ps(v_acc3x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c3_ptr))));
                    v_acc4x0123 = _mm_add_ps(v_acc4x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c4_ptr))));
                    v_acc5x0123 = _mm_add_ps(v_acc5x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c5_ptr))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ps(_mm_min_ps(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ps(_mm_min_ps(v_acc3x0123, v_max), v_min);
                v_acc4x0123 = _mm_max_ps(_mm_min_ps(v_acc4x0123, v_max), v_min);
                v_acc5x0123 = _mm_max_ps(_mm_min_ps(v_acc5x0123, v_max), v_min);

                _mm_storel_epi64(reinterpret_cast<__m128i*>(c0_ptr),
                                 _mm_cvtps_ph(v_acc0x0123, _MM_FROUND_TO_NEAREST_INT));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c1_ptr),
                                 _mm_cvtps_ph(v_acc1x0123, _MM_FROUND_TO_NEAREST_INT));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c2_ptr),
                                 _mm_cvtps_ph(v_acc2x0123, _MM_FROUND_TO_NEAREST_INT));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c3_ptr),
                                 _mm_cvtps_ph(v_acc3x0123, _MM_FROUND_TO_NEAREST_INT));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c4_ptr),
                                 _mm_cvtps_ph(v_acc4x0123, _MM_FROUND_TO_NEAREST_INT));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c5_ptr),
                                 _mm_cvtps_ph(v_acc5x0123, _MM_FROUND_TO_NEAREST_INT));

                c0_ptr += 4;
                c1_ptr += 4;
                c2_ptr += 4;
                c3_ptr += 4;
                c4_ptr += 4;
                c5_ptr += 4;

                v_acc0x0123 = _mm256_extractf128_ps(v_acc0x01234567, 1);
                v_acc1x0123 = _mm256_extractf128_ps(v_acc1x01234567, 1);
                v_acc2x0123 = _mm256_extractf128_ps(v_acc2x01234567, 1);
                v_acc3x0123 = _mm256_extractf128_ps(v_acc3x01234567, 1);
                v_acc4x0123 = _mm256_extractf128_ps(v_acc4x01234567, 1);
                v_acc5x0123 = _mm256_extractf128_ps(v_acc5x01234567, 1);
            }

            if (rem & 2) {
                // The clamp below is 4-wide, so keep the untouched leftovers for
                // the 1-wide step -- lanes 2,3 are what it will store.
                const __m128 v_rest0 = v_acc0x0123;
                const __m128 v_rest1 = v_acc1x0123;
                const __m128 v_rest2 = v_acc2x0123;
                const __m128 v_rest3 = v_acc3x0123;
                const __m128 v_rest4 = v_acc4x0123;
                const __m128 v_rest5 = v_acc5x0123;

                if constexpr (!zero_mode) {
                    uint32_t c_bits;
                    c_bits = static_cast<uint32_t>(c0_ptr[0].bits) |
                             (static_cast<uint32_t>(c0_ptr[1].bits) << 16);
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                    c_bits = static_cast<uint32_t>(c1_ptr[0].bits) |
                             (static_cast<uint32_t>(c1_ptr[1].bits) << 16);
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                    c_bits = static_cast<uint32_t>(c2_ptr[0].bits) |
                             (static_cast<uint32_t>(c2_ptr[1].bits) << 16);
                    v_acc2x0123 = _mm_add_ps(v_acc2x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                    c_bits = static_cast<uint32_t>(c3_ptr[0].bits) |
                             (static_cast<uint32_t>(c3_ptr[1].bits) << 16);
                    v_acc3x0123 = _mm_add_ps(v_acc3x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                    c_bits = static_cast<uint32_t>(c4_ptr[0].bits) |
                             (static_cast<uint32_t>(c4_ptr[1].bits) << 16);
                    v_acc4x0123 = _mm_add_ps(v_acc4x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                    c_bits = static_cast<uint32_t>(c5_ptr[0].bits) |
                             (static_cast<uint32_t>(c5_ptr[1].bits) << 16);
                    v_acc5x0123 = _mm_add_ps(v_acc5x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ps(_mm_min_ps(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ps(_mm_min_ps(v_acc3x0123, v_max), v_min);
                v_acc4x0123 = _mm_max_ps(_mm_min_ps(v_acc4x0123, v_max), v_min);
                v_acc5x0123 = _mm_max_ps(_mm_min_ps(v_acc5x0123, v_max), v_min);

                {
                    const __m128i v_h0 = _mm_cvtps_ph(
                        v_acc0x0123, _MM_FROUND_TO_NEAREST_INT);
                    c0_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h0, 0));
                    c0_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h0, 1));
                    const __m128i v_h1 = _mm_cvtps_ph(
                        v_acc1x0123, _MM_FROUND_TO_NEAREST_INT);
                    c1_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h1, 0));
                    c1_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h1, 1));
                    const __m128i v_h2 = _mm_cvtps_ph(
                        v_acc2x0123, _MM_FROUND_TO_NEAREST_INT);
                    c2_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h2, 0));
                    c2_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h2, 1));
                    const __m128i v_h3 = _mm_cvtps_ph(
                        v_acc3x0123, _MM_FROUND_TO_NEAREST_INT);
                    c3_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h3, 0));
                    c3_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h3, 1));
                    const __m128i v_h4 = _mm_cvtps_ph(
                        v_acc4x0123, _MM_FROUND_TO_NEAREST_INT);
                    c4_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h4, 0));
                    c4_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h4, 1));
                    const __m128i v_h5 = _mm_cvtps_ph(
                        v_acc5x0123, _MM_FROUND_TO_NEAREST_INT);
                    c5_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h5, 0));
                    c5_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h5, 1));
                }

                c0_ptr += 2;
                c1_ptr += 2;
                c2_ptr += 2;
                c3_ptr += 2;
                c4_ptr += 2;
                c5_ptr += 2;

                v_acc0x0123 = _mm_movehl_ps(v_rest0, v_rest0);
                v_acc1x0123 = _mm_movehl_ps(v_rest1, v_rest1);
                v_acc2x0123 = _mm_movehl_ps(v_rest2, v_rest2);
                v_acc3x0123 = _mm_movehl_ps(v_rest3, v_rest3);
                v_acc4x0123 = _mm_movehl_ps(v_rest4, v_rest4);
                v_acc5x0123 = _mm_movehl_ps(v_rest5, v_rest5);
            }

            if (rem & 1) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ss(v_acc0x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c0_ptr->bits))));
                    v_acc1x0123 = _mm_add_ss(v_acc1x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c1_ptr->bits))));
                    v_acc2x0123 = _mm_add_ss(v_acc2x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c2_ptr->bits))));
                    v_acc3x0123 = _mm_add_ss(v_acc3x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c3_ptr->bits))));
                    v_acc4x0123 = _mm_add_ss(v_acc4x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c4_ptr->bits))));
                    v_acc5x0123 = _mm_add_ss(v_acc5x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c5_ptr->bits))));
                }

                v_acc0x0123 = _mm_max_ss(_mm_min_ss(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ss(_mm_min_ss(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ss(_mm_min_ss(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ss(_mm_min_ss(v_acc3x0123, v_max), v_min);
                v_acc4x0123 = _mm_max_ss(_mm_min_ss(v_acc4x0123, v_max), v_min);
                v_acc5x0123 = _mm_max_ss(_mm_min_ss(v_acc5x0123, v_max), v_min);

                c0_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc0x0123, _MM_FROUND_TO_NEAREST_INT), 0));
                c1_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc1x0123, _MM_FROUND_TO_NEAREST_INT), 0));
                c2_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc2x0123, _MM_FROUND_TO_NEAREST_INT), 0));
                c3_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc3x0123, _MM_FROUND_TO_NEAREST_INT), 0));
                c4_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc4x0123, _MM_FROUND_TO_NEAREST_INT), 0));
                c5_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc5x0123, _MM_FROUND_TO_NEAREST_INT), 0));
            }
        }
    }
}

template <bool zero_mode = false, int seed = 0>
void mma_packed_b_mr4_nr16_mrxncxkc_f16(
    int mr, int nc, int kc, half* NNOPS_RESTRICT C, int ldc,
    const half* NNOPS_RESTRICT A, int lda,
    const half* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max,
    const half* NNOPS_RESTRICT bias = nullptr) noexcept {

    // Rows past `mr` alias the last live row in both A and C (see the mr6
    // kernel): a partial-M tile needs no per-row guards.
    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;

    half* c0_ptr = C + 0 * ldc;
    half* c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    half* c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    half* c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;

    int ldb = static_cast<int>(mma_pack_align(16 * kc * sizeof(half)) / sizeof(half));

    int n = 0;

    // Accumulator initial value. Seed 1 is a per-row bias — the planar route,
    // where a row is an output channel; seed 2 is a per-column one, which is the
    // channel-last route's shape: the rows are pixels and the bias lands on the
    // panel's 16 lanes, so it costs two loads per panel where seeding the tile
    // first costs a full write and read-back of the output. `off` is the lane
    // offset inside the panel. A row past `mr` aliases the last live row in A
    // and C, so the same alias index keeps the row seed in range.
    const auto init = [&](int r, int off) {
        if constexpr (seed == 1) { return _mm256_set1_ps(half_to_float(bias[r])); }
        else if constexpr (seed == 2) {
            return _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(bias + n + off)));
        } else { return _mm256_setzero_ps(); }
    };

    for(; n < nc - 15; n += 16) {
        const half* a0_ptr = A + 0 * lda;
        const half* a1_ptr = A + r1 * lda;
        const half* a2_ptr = A + r2 * lda;
        const half* a3_ptr = A + r3 * lda;

        __m256 v_acc0x01234567 = init(0, 0);
        __m256 v_acc0x89abcdef = init(0, 8);
        __m256 v_acc1x01234567 = init(r1, 0);
        __m256 v_acc1x89abcdef = init(r1, 8);
        __m256 v_acc2x01234567 = init(r2, 0);
        __m256 v_acc2x89abcdef = init(r2, 8);
        __m256 v_acc3x01234567 = init(r3, 0);
        __m256 v_acc3x89abcdef = init(r3, 8);

        const half* b_ptr = B;
        for(int k = 0; k < kc; ++k) {
            const __m256 v_b0 = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 0 * 8)));
            const __m256 v_b1 = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 1 * 8)));
            b_ptr += 16;

            __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a0_ptr[0].bits))); a0_ptr += 1;
            __m256 v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a1_ptr[0].bits))); a1_ptr += 1;
            v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
            v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
            v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
            v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);

            v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a2_ptr[0].bits))); a2_ptr += 1;
            v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a3_ptr[0].bits))); a3_ptr += 1;
            v_acc2x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc2x01234567);
            v_acc2x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc2x89abcdef);
            v_acc3x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc3x01234567);
            v_acc3x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc3x89abcdef);
        }

        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr + 0))));
            v_acc0x89abcdef = _mm256_add_ps(v_acc0x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr + 8))));
            v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c1_ptr + 0))));
            v_acc1x89abcdef = _mm256_add_ps(v_acc1x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c1_ptr + 8))));
            v_acc2x01234567 = _mm256_add_ps(v_acc2x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c2_ptr + 0))));
            v_acc2x89abcdef = _mm256_add_ps(v_acc2x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c2_ptr + 8))));
            v_acc3x01234567 = _mm256_add_ps(v_acc3x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c3_ptr + 0))));
            v_acc3x89abcdef = _mm256_add_ps(v_acc3x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c3_ptr + 8))));
        }
        const __m256 v_min = _mm256_set1_ps(clamp_min);
        const __m256 v_max = _mm256_set1_ps(clamp_max);

        v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
        v_acc0x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc0x89abcdef, v_max), v_min);
        v_acc1x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc1x01234567, v_max), v_min);
        v_acc1x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc1x89abcdef, v_max), v_min);
        v_acc2x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc2x01234567, v_max), v_min);
        v_acc2x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc2x89abcdef, v_max), v_min);
        v_acc3x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc3x01234567, v_max), v_min);
        v_acc3x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc3x89abcdef, v_max), v_min);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr + 0),
                         _mm256_cvtps_ph(v_acc0x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr + 8),
                         _mm256_cvtps_ph(v_acc0x89abcdef, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c1_ptr + 0),
                         _mm256_cvtps_ph(v_acc1x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c1_ptr + 8),
                         _mm256_cvtps_ph(v_acc1x89abcdef, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c2_ptr + 0),
                         _mm256_cvtps_ph(v_acc2x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c2_ptr + 8),
                         _mm256_cvtps_ph(v_acc2x89abcdef, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c3_ptr + 0),
                         _mm256_cvtps_ph(v_acc3x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c3_ptr + 8),
                         _mm256_cvtps_ph(v_acc3x89abcdef, _MM_FROUND_TO_NEAREST_INT));

        c0_ptr += 16;
        c1_ptr += 16;
        c2_ptr += 16;
        c3_ptr += 16;

        B += ldb;
    }

    int rem = nc - n;

    if (rem > 0) {
        // The tail is one full 16-wide panel in the packed layout -- the
        // packer zero-fills the columns past `nc`, so both halves can always
        // be read at the same 16-element stride. At most 8 columns still fit
        // one 8-lane read; a wider tail reads both halves and reuses the
        // full-panel body, so it costs one k-loop instead of two.
        const half* a0_ptr = A + 0 * lda;
        const half* a1_ptr = A + r1 * lda;
        const half* a2_ptr = A + r2 * lda;
        const half* a3_ptr = A + r3 * lda;
        __m256 v_acc0x01234567 = init(0, 0);
        __m256 v_acc0x89abcdef = init(0, 8);
        __m256 v_acc1x01234567 = init(r1, 0);
        __m256 v_acc1x89abcdef = init(r1, 8);
        __m256 v_acc2x01234567 = init(r2, 0);
        __m256 v_acc2x89abcdef = init(r2, 8);
        __m256 v_acc3x01234567 = init(r3, 0);
        __m256 v_acc3x89abcdef = init(r3, 8);

        const half* b_ptr = B;
        if (rem > 8) {
            for(int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 0 * 8)));
                const __m256 v_b1 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 1 * 8)));
                b_ptr += 16;

                __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a0_ptr[0].bits))); a0_ptr += 1;
                __m256 v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a1_ptr[0].bits))); a1_ptr += 1;
                v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
                v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
                v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
                v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);

                v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a2_ptr[0].bits))); a2_ptr += 1;
                v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a3_ptr[0].bits))); a3_ptr += 1;
                v_acc2x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc2x01234567);
                v_acc2x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc2x89abcdef);
                v_acc3x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc3x01234567);
                v_acc3x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc3x89abcdef);
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr))); b_ptr += 16;
                v_acc0x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a0_ptr[0].bits))), v_b0, v_acc0x01234567);
                v_acc1x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a1_ptr[0].bits))), v_b0, v_acc1x01234567);
                v_acc2x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a2_ptr[0].bits))), v_b0, v_acc2x01234567);
                v_acc3x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a3_ptr[0].bits))), v_b0, v_acc3x01234567);
                a0_ptr += 1;
                a1_ptr += 1;
                a2_ptr += 1;
                a3_ptr += 1;
            }
        }

        if (rem >= 8) {

            if constexpr (!zero_mode) {
                v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr))));
                v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c1_ptr))));
                v_acc2x01234567 = _mm256_add_ps(v_acc2x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c2_ptr))));
                v_acc3x01234567 = _mm256_add_ps(v_acc3x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c3_ptr))));
            }

            const __m256 v_min = _mm256_set1_ps(clamp_min);
            const __m256 v_max = _mm256_set1_ps(clamp_max);

            v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
            v_acc1x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc1x01234567, v_max), v_min);
            v_acc2x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc2x01234567, v_max), v_min);
            v_acc3x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc3x01234567, v_max), v_min);

            _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr),
                             _mm256_cvtps_ph(v_acc0x01234567, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c1_ptr),
                             _mm256_cvtps_ph(v_acc1x01234567, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c2_ptr),
                             _mm256_cvtps_ph(v_acc2x01234567, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c3_ptr),
                             _mm256_cvtps_ph(v_acc3x01234567, _MM_FROUND_TO_NEAREST_INT));

            c0_ptr += 8;
            c1_ptr += 8;
            c2_ptr += 8;
            c3_ptr += 8;

            if (rem > 8) {
                v_acc0x01234567 = v_acc0x89abcdef;
                v_acc1x01234567 = v_acc1x89abcdef;
                v_acc2x01234567 = v_acc2x89abcdef;
                v_acc3x01234567 = v_acc3x89abcdef;
            }
            rem -= 8;
        }

        if (rem > 0) {
            // Lane j holds column j, so the 4/2/1 ladder is just a lo/hi split of
            // the same accumulator: after the 4-wide step the leftover 1..3 columns
            // sit at lane 0 of the high half. Four fp32 lanes convert to four
            // halves, so the partial C stores are an 8/4/2-byte store of that.
            __m128 v_acc0x0123 = _mm256_castps256_ps128(v_acc0x01234567);
            __m128 v_acc1x0123 = _mm256_castps256_ps128(v_acc1x01234567);
            __m128 v_acc2x0123 = _mm256_castps256_ps128(v_acc2x01234567);
            __m128 v_acc3x0123 = _mm256_castps256_ps128(v_acc3x01234567);

            const __m128 v_min = _mm_set1_ps(clamp_min);
            const __m128 v_max = _mm_set1_ps(clamp_max);

            if (rem & 4) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0_ptr))));
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c1_ptr))));
                    v_acc2x0123 = _mm_add_ps(v_acc2x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c2_ptr))));
                    v_acc3x0123 = _mm_add_ps(v_acc3x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c3_ptr))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ps(_mm_min_ps(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ps(_mm_min_ps(v_acc3x0123, v_max), v_min);

                _mm_storel_epi64(reinterpret_cast<__m128i*>(c0_ptr),
                                 _mm_cvtps_ph(v_acc0x0123, _MM_FROUND_TO_NEAREST_INT));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c1_ptr),
                                 _mm_cvtps_ph(v_acc1x0123, _MM_FROUND_TO_NEAREST_INT));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c2_ptr),
                                 _mm_cvtps_ph(v_acc2x0123, _MM_FROUND_TO_NEAREST_INT));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c3_ptr),
                                 _mm_cvtps_ph(v_acc3x0123, _MM_FROUND_TO_NEAREST_INT));

                c0_ptr += 4;
                c1_ptr += 4;
                c2_ptr += 4;
                c3_ptr += 4;

                v_acc0x0123 = _mm256_extractf128_ps(v_acc0x01234567, 1);
                v_acc1x0123 = _mm256_extractf128_ps(v_acc1x01234567, 1);
                v_acc2x0123 = _mm256_extractf128_ps(v_acc2x01234567, 1);
                v_acc3x0123 = _mm256_extractf128_ps(v_acc3x01234567, 1);
            }

            if (rem & 2) {
                // The clamp below is 4-wide, so keep the untouched leftovers for
                // the 1-wide step -- lanes 2,3 are what it will store.
                const __m128 v_rest0 = v_acc0x0123;
                const __m128 v_rest1 = v_acc1x0123;
                const __m128 v_rest2 = v_acc2x0123;
                const __m128 v_rest3 = v_acc3x0123;

                if constexpr (!zero_mode) {
                    uint32_t c_bits;
                    c_bits = static_cast<uint32_t>(c0_ptr[0].bits) |
                             (static_cast<uint32_t>(c0_ptr[1].bits) << 16);
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                    c_bits = static_cast<uint32_t>(c1_ptr[0].bits) |
                             (static_cast<uint32_t>(c1_ptr[1].bits) << 16);
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                    c_bits = static_cast<uint32_t>(c2_ptr[0].bits) |
                             (static_cast<uint32_t>(c2_ptr[1].bits) << 16);
                    v_acc2x0123 = _mm_add_ps(v_acc2x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                    c_bits = static_cast<uint32_t>(c3_ptr[0].bits) |
                             (static_cast<uint32_t>(c3_ptr[1].bits) << 16);
                    v_acc3x0123 = _mm_add_ps(v_acc3x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ps(_mm_min_ps(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ps(_mm_min_ps(v_acc3x0123, v_max), v_min);

                {
                    const __m128i v_h0 = _mm_cvtps_ph(
                        v_acc0x0123, _MM_FROUND_TO_NEAREST_INT);
                    c0_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h0, 0));
                    c0_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h0, 1));
                    const __m128i v_h1 = _mm_cvtps_ph(
                        v_acc1x0123, _MM_FROUND_TO_NEAREST_INT);
                    c1_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h1, 0));
                    c1_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h1, 1));
                    const __m128i v_h2 = _mm_cvtps_ph(
                        v_acc2x0123, _MM_FROUND_TO_NEAREST_INT);
                    c2_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h2, 0));
                    c2_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h2, 1));
                    const __m128i v_h3 = _mm_cvtps_ph(
                        v_acc3x0123, _MM_FROUND_TO_NEAREST_INT);
                    c3_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h3, 0));
                    c3_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h3, 1));
                }

                c0_ptr += 2;
                c1_ptr += 2;
                c2_ptr += 2;
                c3_ptr += 2;

                v_acc0x0123 = _mm_movehl_ps(v_rest0, v_rest0);
                v_acc1x0123 = _mm_movehl_ps(v_rest1, v_rest1);
                v_acc2x0123 = _mm_movehl_ps(v_rest2, v_rest2);
                v_acc3x0123 = _mm_movehl_ps(v_rest3, v_rest3);
            }

            if (rem & 1) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ss(v_acc0x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c0_ptr->bits))));
                    v_acc1x0123 = _mm_add_ss(v_acc1x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c1_ptr->bits))));
                    v_acc2x0123 = _mm_add_ss(v_acc2x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c2_ptr->bits))));
                    v_acc3x0123 = _mm_add_ss(v_acc3x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c3_ptr->bits))));
                }

                v_acc0x0123 = _mm_max_ss(_mm_min_ss(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ss(_mm_min_ss(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ss(_mm_min_ss(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ss(_mm_min_ss(v_acc3x0123, v_max), v_min);

                c0_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc0x0123, _MM_FROUND_TO_NEAREST_INT), 0));
                c1_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc1x0123, _MM_FROUND_TO_NEAREST_INT), 0));
                c2_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc2x0123, _MM_FROUND_TO_NEAREST_INT), 0));
                c3_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc3x0123, _MM_FROUND_TO_NEAREST_INT), 0));
            }
        }
    }
}

template <bool zero_mode = false, int seed = 0>
void mma_packed_b_mr2_nr16_mrxncxkc_f16(
    int mr, int nc, int kc, half* NNOPS_RESTRICT C, int ldc,
    const half* NNOPS_RESTRICT A, int lda,
    const half* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max,
    const half* NNOPS_RESTRICT bias = nullptr) noexcept {

    // Rows past `mr` alias the last live row in both A and C: they compute a
    // duplicate of it and store it back to the same place, so a partial-M tile
    // needs no per-row guards (the XNNPACK broadcast convention).
    const int r1 = (mr > 1) ? 1 : 0;

    half* c0_ptr = C + 0 * ldc;
    half* c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;

    int ldb = static_cast<int>(mma_pack_align(16 * kc * sizeof(half)) / sizeof(half));

    int n = 0;

    // Accumulator initial value. Seed 1 is a per-row bias — the planar route,
    // where a row is an output channel; seed 2 is a per-column one, which is the
    // channel-last route's shape: the rows are pixels and the bias lands on the
    // panel's 16 lanes, so it costs two loads per panel where seeding the tile
    // first costs a full write and read-back of the output. `off` is the lane
    // offset inside the panel. A row past `mr` aliases the last live row in A
    // and C, so the same alias index keeps the row seed in range.
    const auto init = [&](int r, int off) {
        if constexpr (seed == 1) { return _mm256_set1_ps(half_to_float(bias[r])); }
        else if constexpr (seed == 2) {
            return _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(bias + n + off)));
        } else { return _mm256_setzero_ps(); }
    };

    for(; n < nc - 15; n += 16) {
        const half* a0_ptr = A + 0 * lda;
        const half* a1_ptr = A + r1 * lda;

        __m256 v_acc0x01234567 = init(0, 0);
        __m256 v_acc0x89abcdef = init(0, 8);
        __m256 v_acc1x01234567 = init(r1, 0);
        __m256 v_acc1x89abcdef = init(r1, 8);

        const half* b_ptr = B;
        for(int k = 0; k < kc; ++k) {
            const __m256 v_b0 = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 0 * 8)));
            const __m256 v_b1 = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 1 * 8)));
            b_ptr += 16;

            const __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a0_ptr->bits))); a0_ptr += 1;
            const __m256 v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a1_ptr->bits))); a1_ptr += 1;
            v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
            v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
            v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
            v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);
        }

        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr + 0))));
            v_acc0x89abcdef = _mm256_add_ps(v_acc0x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr + 8))));
            v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c1_ptr + 0))));
            v_acc1x89abcdef = _mm256_add_ps(v_acc1x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c1_ptr + 8))));
        }
        const __m256 v_min = _mm256_set1_ps(clamp_min);
        const __m256 v_max = _mm256_set1_ps(clamp_max);

        v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
        v_acc0x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc0x89abcdef, v_max), v_min);
        v_acc1x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc1x01234567, v_max), v_min);
        v_acc1x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc1x89abcdef, v_max), v_min);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr + 0),
                         _mm256_cvtps_ph(v_acc0x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr + 8),
                         _mm256_cvtps_ph(v_acc0x89abcdef, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c1_ptr + 0),
                         _mm256_cvtps_ph(v_acc1x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c1_ptr + 8),
                         _mm256_cvtps_ph(v_acc1x89abcdef, _MM_FROUND_TO_NEAREST_INT));

        c0_ptr += 16;
        c1_ptr += 16;

        B += ldb;
    }

    int rem = nc - n;

    if (rem > 0) {
        // The tail is one full 16-wide panel in the packed layout -- the
        // packer zero-fills the columns past `nc`, so both halves can always
        // be read at the same 16-element stride. At most 8 columns still fit
        // one 8-lane read; a wider tail reads both halves and reuses the
        // full-panel body, so it costs one k-loop instead of two.
        const half* a0_ptr = A + 0 * lda;
        const half* a1_ptr = A + r1 * lda;
        __m256 v_acc0x01234567 = init(0, 0);
        __m256 v_acc0x89abcdef = init(0, 8);
        __m256 v_acc1x01234567 = init(r1, 0);
        __m256 v_acc1x89abcdef = init(r1, 8);

        const half* b_ptr = B;
        if (rem > 8) {
            for(int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 0 * 8)));
                const __m256 v_b1 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 1 * 8)));
                b_ptr += 16;

                const __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a0_ptr->bits))); a0_ptr += 1;
                const __m256 v_a1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a1_ptr->bits))); a1_ptr += 1;
                v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
                v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
                v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
                v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr))); b_ptr += 16;
                v_acc0x01234567 = _mm256_fmadd_ps(
                    _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a0_ptr->bits))), v_b0, v_acc0x01234567);
                v_acc1x01234567 = _mm256_fmadd_ps(
                    _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a1_ptr->bits))), v_b0, v_acc1x01234567);
                a0_ptr += 1;
                a1_ptr += 1;
            }
        }

        if (rem >= 8) {

            if constexpr (!zero_mode) {
                v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr))));
                v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c1_ptr))));
            }

            const __m256 v_min = _mm256_set1_ps(clamp_min);
            const __m256 v_max = _mm256_set1_ps(clamp_max);

            v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
            v_acc1x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc1x01234567, v_max), v_min);

            _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr),
                             _mm256_cvtps_ph(v_acc0x01234567, _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c1_ptr),
                             _mm256_cvtps_ph(v_acc1x01234567, _MM_FROUND_TO_NEAREST_INT));

            c0_ptr += 8;
            c1_ptr += 8;

            if (rem > 8) {
                v_acc0x01234567 = v_acc0x89abcdef;
                v_acc1x01234567 = v_acc1x89abcdef;
            }
            rem -= 8;
        }

        if (rem > 0) {
            // Lane j holds column j, so the 4/2/1 ladder is just a lo/hi split of
            // the same accumulator: after the 4-wide step the leftover 1..3 columns
            // sit at lane 0 of the high half. Four fp32 lanes convert to four
            // halves, so the partial C stores are an 8/4/2-byte store of that.
            __m128 v_acc0x0123 = _mm256_castps256_ps128(v_acc0x01234567);
            __m128 v_acc1x0123 = _mm256_castps256_ps128(v_acc1x01234567);

            const __m128 v_min = _mm_set1_ps(clamp_min);
            const __m128 v_max = _mm_set1_ps(clamp_max);

            if (rem & 4) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0_ptr))));
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c1_ptr))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);

                _mm_storel_epi64(reinterpret_cast<__m128i*>(c0_ptr),
                                 _mm_cvtps_ph(v_acc0x0123, _MM_FROUND_TO_NEAREST_INT));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c1_ptr),
                                 _mm_cvtps_ph(v_acc1x0123, _MM_FROUND_TO_NEAREST_INT));

                c0_ptr += 4;
                c1_ptr += 4;

                v_acc0x0123 = _mm256_extractf128_ps(v_acc0x01234567, 1);
                v_acc1x0123 = _mm256_extractf128_ps(v_acc1x01234567, 1);
            }

            if (rem & 2) {
                // The clamp below is 4-wide, so keep the untouched leftovers for
                // the 1-wide step -- lanes 2,3 are what it will store.
                const __m128 v_rest0 = v_acc0x0123;
                const __m128 v_rest1 = v_acc1x0123;

                if constexpr (!zero_mode) {
                    uint32_t c_bits;
                    c_bits = static_cast<uint32_t>(c0_ptr[0].bits) |
                             (static_cast<uint32_t>(c0_ptr[1].bits) << 16);
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                    c_bits = static_cast<uint32_t>(c1_ptr[0].bits) |
                             (static_cast<uint32_t>(c1_ptr[1].bits) << 16);
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);

                {
                    const __m128i v_h0 = _mm_cvtps_ph(
                        v_acc0x0123, _MM_FROUND_TO_NEAREST_INT);
                    c0_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h0, 0));
                    c0_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h0, 1));
                    const __m128i v_h1 = _mm_cvtps_ph(
                        v_acc1x0123, _MM_FROUND_TO_NEAREST_INT);
                    c1_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h1, 0));
                    c1_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h1, 1));
                }

                c0_ptr += 2;
                c1_ptr += 2;

                v_acc0x0123 = _mm_movehl_ps(v_rest0, v_rest0);
                v_acc1x0123 = _mm_movehl_ps(v_rest1, v_rest1);
            }

            if (rem & 1) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ss(v_acc0x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c0_ptr->bits))));
                    v_acc1x0123 = _mm_add_ss(v_acc1x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c1_ptr->bits))));
                }

                v_acc0x0123 = _mm_max_ss(_mm_min_ss(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ss(_mm_min_ss(v_acc1x0123, v_max), v_min);

                c0_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc0x0123, _MM_FROUND_TO_NEAREST_INT), 0));
                c1_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc1x0123, _MM_FROUND_TO_NEAREST_INT), 0));
            }
        }
    }
}

template <bool zero_mode = false, int seed = 0>
void mma_packed_b_mr1_nr16_mrxncxkc_f16(
    int mr, int nc, int kc, half* NNOPS_RESTRICT C, int ldc,
    const half* NNOPS_RESTRICT A, int lda,
    const half* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max,
    const half* NNOPS_RESTRICT bias = nullptr) noexcept {

    (void)mr;   // one row, so the row count never varies
    (void)ldc;  // one row, so the C row stride never applies
    (void)lda;  // one row, so the A row stride never applies

    half* c0_ptr = C;

    int ldb = static_cast<int>(mma_pack_align(16 * kc * sizeof(half)) / sizeof(half));

    int n = 0;

    // Accumulator initial value. Seed 1 is a per-row bias — the planar route,
    // where a row is an output channel; seed 2 is a per-column one, which is the
    // channel-last route's shape: the rows are pixels and the bias lands on the
    // panel's 16 lanes, so it costs two loads per panel where seeding the tile
    // first costs a full write and read-back of the output. `off` is the lane
    // offset inside the panel. A row past `mr` aliases the last live row in A
    // and C, so the same alias index keeps the row seed in range.
    const auto init = [&](int r, int off) {
        if constexpr (seed == 1) { return _mm256_set1_ps(half_to_float(bias[r])); }
        else if constexpr (seed == 2) {
            return _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(bias + n + off)));
        } else { return _mm256_setzero_ps(); }
    };
    for(; n < nc - 15; n += 16) {
        __m256 v_acc0x01234567 = init(0, 0);
        __m256 v_acc0x89abcdef = init(0, 8);

        const half* a0_ptr = A;
        const half* b_ptr = B;
        for(int k = 0; k < kc; ++k) {
            const __m256 v_b0 = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 0 * 8)));
            const __m256 v_b1 = _mm256_cvtph_ps(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 1 * 8)));
            b_ptr += 16;

            const __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a0_ptr[0].bits))); a0_ptr += 1;
            v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
            v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
        }
        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr + 0))));
            v_acc0x89abcdef = _mm256_add_ps(v_acc0x89abcdef,
                                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr + 8))));
        }
        const __m256 v_min = _mm256_set1_ps(clamp_min);
        const __m256 v_max = _mm256_set1_ps(clamp_max);

        v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
        v_acc0x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc0x89abcdef, v_max), v_min);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr + 0),
                         _mm256_cvtps_ph(v_acc0x01234567, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr + 8),
                         _mm256_cvtps_ph(v_acc0x89abcdef, _MM_FROUND_TO_NEAREST_INT));

        c0_ptr += 16;

        B += ldb;
    }

    int rem = nc - n;

    if (rem > 0) {
        // The tail is one full 16-wide panel in the packed layout -- the
        // packer zero-fills the columns past `nc`, so both halves can always
        // be read at the same 16-element stride. At most 8 columns still fit
        // one 8-lane read; a wider tail reads both halves and reuses the
        // full-panel body, so it costs one k-loop instead of two.
        const half* a0_ptr = A + 0 * lda;
        __m256 v_acc0x01234567 = init(0, 0);
        __m256 v_acc0x89abcdef = init(0, 8);

        const half* b_ptr = B;
        if (rem > 8) {
            for(int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 0 * 8)));
                const __m256 v_b1 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + 1 * 8)));
                b_ptr += 16;

                const __m256 v_a0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a0_ptr[0].bits))); a0_ptr += 1;
                v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
                v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_cvtph_ps(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr))); b_ptr += 16;
                v_acc0x01234567 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(a0_ptr[0].bits))), v_b0, v_acc0x01234567);
                a0_ptr += 1;
            }
        }

        if (rem >= 8) {

            if constexpr (!zero_mode) {
                v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567,
                                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c0_ptr))));
            }

            const __m256 v_min = _mm256_set1_ps(clamp_min);
            const __m256 v_max = _mm256_set1_ps(clamp_max);

            v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);

            _mm_storeu_si128(reinterpret_cast<__m128i*>(c0_ptr),
                             _mm256_cvtps_ph(v_acc0x01234567, _MM_FROUND_TO_NEAREST_INT));

            c0_ptr += 8;

            if (rem > 8) {
                v_acc0x01234567 = v_acc0x89abcdef;
            }
            rem -= 8;
        }

        if (rem > 0) {
            // Lane j holds column j, so the 4/2/1 ladder is just a lo/hi split of
            // the same accumulator: after the 4-wide step the leftover 1..3 columns
            // sit at lane 0 of the high half. Four fp32 lanes convert to four
            // halves, so the partial C stores are an 8/4/2-byte store of that.
            __m128 v_acc0x0123 = _mm256_castps256_ps128(v_acc0x01234567);

            const __m128 v_min = _mm_set1_ps(clamp_min);
            const __m128 v_max = _mm_set1_ps(clamp_max);

            if (rem & 4) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_cvtph_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0_ptr))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);

                _mm_storel_epi64(reinterpret_cast<__m128i*>(c0_ptr),
                                 _mm_cvtps_ph(v_acc0x0123, _MM_FROUND_TO_NEAREST_INT));

                c0_ptr += 4;

                v_acc0x0123 = _mm256_extractf128_ps(v_acc0x01234567, 1);
            }

            if (rem & 2) {
                // The clamp below is 4-wide, so keep the untouched leftovers for
                // the 1-wide step -- lanes 2,3 are what it will store.
                const __m128 v_rest0 = v_acc0x0123;

                if constexpr (!zero_mode) {
                    uint32_t c_bits;
                    c_bits = static_cast<uint32_t>(c0_ptr[0].bits) |
                             (static_cast<uint32_t>(c0_ptr[1].bits) << 16);
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_cvtph_ps(
                        _mm_cvtsi32_si128(static_cast<int>(c_bits))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);

                {
                    const __m128i v_h0 = _mm_cvtps_ph(
                        v_acc0x0123, _MM_FROUND_TO_NEAREST_INT);
                    c0_ptr[0].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h0, 0));
                    c0_ptr[1].bits = static_cast<uint16_t>(_mm_extract_epi16(v_h0, 1));
                }

                c0_ptr += 2;

                v_acc0x0123 = _mm_movehl_ps(v_rest0, v_rest0);
            }

            if (rem & 1) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ss(v_acc0x0123,
                        _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<short>(c0_ptr->bits))));
                }

                v_acc0x0123 = _mm_max_ss(_mm_min_ss(v_acc0x0123, v_max), v_min);

                c0_ptr->bits = static_cast<uint16_t>(_mm_extract_epi16(
                    _mm_cvtps_ph(v_acc0x0123, _MM_FROUND_TO_NEAREST_INT), 0));
            }
        }
    }
}

// ---- dispatch tables, consumed by matmul_helper.cpp ---------------------
template <bool Z>
inline constexpr auto mma_pack_fns_f16 = std::array{
    &mma_packed_ab_mr6_nr16_mrxncxkc_f16<Z>,
    &mma_packed_ab_mr4_nr16_mrxncxkc_f16<Z>,
    &mma_packed_ab_mr1_nr16_mrxncxkc_f16<Z>};

template <bool Z, int Seed = 0>
inline constexpr auto mma_direct_fns_f16 = std::array{
    &mma_packed_b_mr6_nr16_mrxncxkc_f16<Z, Seed>,
    &mma_packed_b_mr4_nr16_mrxncxkc_f16<Z, Seed>,
    &mma_packed_b_mr1_nr16_mrxncxkc_f16<Z, Seed>};

}  // namespace nnops::backend::cpu::x86_64
