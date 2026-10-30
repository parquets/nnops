#pragma once
/// @file mma_f32.hpp
/// @brief x86_64 AVX2/FMA float32 MMA (matrix micro-accumulate) kernels.
///
/// A and B may both be packed (`mma_packed_ab_*`) or only B packed while A is
/// read row-major (`mma_packed_b_*`). Tile sizes: M: {6, 4, 2, 1}, N: {16, 8, 1}.
///
/// Packed B is uniform: ceil(nc / 16) panels, every one a full 16 columns wide
/// with the columns past `nc` zero-filled, each in a
/// mma_pack_align(16*kc*sizeof(float))-byte slot (16*kc floats). That lets the
/// kernel read 16 floats per k at every column, so a tail wider than 8 costs
/// one k-loop instead of two.

#include <array>
#include <immintrin.h>
#include <algorithm>
#include "backend/cpu/common/common_utils.hpp"


namespace nnops::backend::cpu::x86_64 {

template <bool zero_mode = false>
void mma_packed_ab_mr6_nr16_mrxncxkc_f32(
    int mr, int nc, int kc, float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A,
    const float* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max) noexcept {

    // Rows past `mr` alias the last live row in both A and C: they compute a
    // duplicate of it and store it back to the same place, so a partial-M tile
    // needs no per-row guards (the XNNPACK 6x16 broadcast convention).
    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;
    const int r4 = (mr > 4) ? 4 : r3;
    const int r5 = (mr > 5) ? 5 : r4;

    float* c0_ptr = C + 0 * ldc;
    float* c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    float* c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    float* c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;
    float* c4_ptr = (mr > 4) ? C + 4 * ldc : c3_ptr;
    float* c5_ptr = (mr > 5) ? C + 5 * ldc : c4_ptr;

    int ldb = static_cast<int>(mma_pack_align(16 * kc * sizeof(float)) / sizeof(float));

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

        const float* a_ptr = A;
        const float* b_ptr = B;
        for(int k = 0; k < kc; ++k) {
            const __m256 v_b0 = _mm256_loadu_ps(b_ptr + 0 * 8);
            const __m256 v_b1 = _mm256_loadu_ps(b_ptr + 1 * 8);
            b_ptr += 16;

            __m256 v_a0 = _mm256_broadcast_ss(a_ptr + 0);
            __m256 v_a1 = _mm256_broadcast_ss(a_ptr + r1);
            v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
            v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
            v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
            v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);

            v_a0 = _mm256_broadcast_ss(a_ptr + r2);
            v_a1 = _mm256_broadcast_ss(a_ptr + r3);
            v_acc2x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc2x01234567);
            v_acc2x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc2x89abcdef);
            v_acc3x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc3x01234567);
            v_acc3x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc3x89abcdef);

            v_a0 = _mm256_broadcast_ss(a_ptr + r4);
            v_a1 = _mm256_broadcast_ss(a_ptr + r5);
            v_acc4x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc4x01234567);
            v_acc4x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc4x89abcdef);
            v_acc5x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc5x01234567);
            v_acc5x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc5x89abcdef);

            a_ptr += 6;
        }
        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567, _mm256_loadu_ps(c0_ptr + 0));
            v_acc0x89abcdef = _mm256_add_ps(v_acc0x89abcdef, _mm256_loadu_ps(c0_ptr + 8));
            v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567, _mm256_loadu_ps(c1_ptr + 0));
            v_acc1x89abcdef = _mm256_add_ps(v_acc1x89abcdef, _mm256_loadu_ps(c1_ptr + 8));
            v_acc2x01234567 = _mm256_add_ps(v_acc2x01234567, _mm256_loadu_ps(c2_ptr + 0));
            v_acc2x89abcdef = _mm256_add_ps(v_acc2x89abcdef, _mm256_loadu_ps(c2_ptr + 8));
            v_acc3x01234567 = _mm256_add_ps(v_acc3x01234567, _mm256_loadu_ps(c3_ptr + 0));
            v_acc3x89abcdef = _mm256_add_ps(v_acc3x89abcdef, _mm256_loadu_ps(c3_ptr + 8));
            v_acc4x01234567 = _mm256_add_ps(v_acc4x01234567, _mm256_loadu_ps(c4_ptr + 0));
            v_acc4x89abcdef = _mm256_add_ps(v_acc4x89abcdef, _mm256_loadu_ps(c4_ptr + 8));
            v_acc5x01234567 = _mm256_add_ps(v_acc5x01234567, _mm256_loadu_ps(c5_ptr + 0));
            v_acc5x89abcdef = _mm256_add_ps(v_acc5x89abcdef, _mm256_loadu_ps(c5_ptr + 8));
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

        _mm256_storeu_ps(c0_ptr + 0, v_acc0x01234567);
        _mm256_storeu_ps(c0_ptr + 8, v_acc0x89abcdef);
        _mm256_storeu_ps(c1_ptr + 0, v_acc1x01234567);
        _mm256_storeu_ps(c1_ptr + 8, v_acc1x89abcdef);
        _mm256_storeu_ps(c2_ptr + 0, v_acc2x01234567);
        _mm256_storeu_ps(c2_ptr + 8, v_acc2x89abcdef);
        _mm256_storeu_ps(c3_ptr + 0, v_acc3x01234567);
        _mm256_storeu_ps(c3_ptr + 8, v_acc3x89abcdef);
        _mm256_storeu_ps(c4_ptr + 0, v_acc4x01234567);
        _mm256_storeu_ps(c4_ptr + 8, v_acc4x89abcdef);
        _mm256_storeu_ps(c5_ptr + 0, v_acc5x01234567);
        _mm256_storeu_ps(c5_ptr + 8, v_acc5x89abcdef);

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

        const float* a_ptr = A;
        const float* b_ptr = B;
        if (rem > 8) {
            for(int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_loadu_ps(b_ptr + 0 * 8);
                const __m256 v_b1 = _mm256_loadu_ps(b_ptr + 1 * 8);
                b_ptr += 16;

                __m256 v_a0 = _mm256_broadcast_ss(a_ptr + 0);
                __m256 v_a1 = _mm256_broadcast_ss(a_ptr + r1);
                v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
                v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
                v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
                v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);

                v_a0 = _mm256_broadcast_ss(a_ptr + r2);
                v_a1 = _mm256_broadcast_ss(a_ptr + r3);
                v_acc2x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc2x01234567);
                v_acc2x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc2x89abcdef);
                v_acc3x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc3x01234567);
                v_acc3x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc3x89abcdef);

                v_a0 = _mm256_broadcast_ss(a_ptr + r4);
                v_a1 = _mm256_broadcast_ss(a_ptr + r5);
                v_acc4x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc4x01234567);
                v_acc4x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc4x89abcdef);
                v_acc5x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc5x01234567);
                v_acc5x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc5x89abcdef);

                a_ptr += 6;
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_loadu_ps(b_ptr); b_ptr += 16;
                v_acc0x01234567 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 0), v_b0, v_acc0x01234567);
                v_acc1x01234567 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + r1), v_b0, v_acc1x01234567);
                v_acc2x01234567 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + r2), v_b0, v_acc2x01234567);
                v_acc3x01234567 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + r3), v_b0, v_acc3x01234567);
                v_acc4x01234567 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + r4), v_b0, v_acc4x01234567);
                v_acc5x01234567 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + r5), v_b0, v_acc5x01234567);
                a_ptr += 6;
            }
        }

        if (rem >= 8) {

            if constexpr (!zero_mode) {
                v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567, _mm256_loadu_ps(c0_ptr));
                v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567, _mm256_loadu_ps(c1_ptr));
                v_acc2x01234567 = _mm256_add_ps(v_acc2x01234567, _mm256_loadu_ps(c2_ptr));
                v_acc3x01234567 = _mm256_add_ps(v_acc3x01234567, _mm256_loadu_ps(c3_ptr));
                v_acc4x01234567 = _mm256_add_ps(v_acc4x01234567, _mm256_loadu_ps(c4_ptr));
                v_acc5x01234567 = _mm256_add_ps(v_acc5x01234567, _mm256_loadu_ps(c5_ptr));
            }

            const __m256 v_min = _mm256_set1_ps(clamp_min);
            const __m256 v_max = _mm256_set1_ps(clamp_max);

            v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
            v_acc1x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc1x01234567, v_max), v_min);
            v_acc2x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc2x01234567, v_max), v_min);
            v_acc3x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc3x01234567, v_max), v_min);
            v_acc4x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc4x01234567, v_max), v_min);
            v_acc5x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc5x01234567, v_max), v_min);

            _mm256_storeu_ps(c0_ptr, v_acc0x01234567);
            _mm256_storeu_ps(c1_ptr, v_acc1x01234567);
            _mm256_storeu_ps(c2_ptr, v_acc2x01234567);
            _mm256_storeu_ps(c3_ptr, v_acc3x01234567);
            _mm256_storeu_ps(c4_ptr, v_acc4x01234567);
            _mm256_storeu_ps(c5_ptr, v_acc5x01234567);

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
            // sit at lane 0 of the high half.
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
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_loadu_ps(c0_ptr));
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_loadu_ps(c1_ptr));
                    v_acc2x0123 = _mm_add_ps(v_acc2x0123, _mm_loadu_ps(c2_ptr));
                    v_acc3x0123 = _mm_add_ps(v_acc3x0123, _mm_loadu_ps(c3_ptr));
                    v_acc4x0123 = _mm_add_ps(v_acc4x0123, _mm_loadu_ps(c4_ptr));
                    v_acc5x0123 = _mm_add_ps(v_acc5x0123, _mm_loadu_ps(c5_ptr));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ps(_mm_min_ps(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ps(_mm_min_ps(v_acc3x0123, v_max), v_min);
                v_acc4x0123 = _mm_max_ps(_mm_min_ps(v_acc4x0123, v_max), v_min);
                v_acc5x0123 = _mm_max_ps(_mm_min_ps(v_acc5x0123, v_max), v_min);

                _mm_storeu_ps(c0_ptr, v_acc0x0123);
                _mm_storeu_ps(c1_ptr, v_acc1x0123);
                _mm_storeu_ps(c2_ptr, v_acc2x0123);
                _mm_storeu_ps(c3_ptr, v_acc3x0123);
                _mm_storeu_ps(c4_ptr, v_acc4x0123);
                _mm_storeu_ps(c5_ptr, v_acc5x0123);

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
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0_ptr))));
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c1_ptr))));
                    v_acc2x0123 = _mm_add_ps(v_acc2x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c2_ptr))));
                    v_acc3x0123 = _mm_add_ps(v_acc3x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c3_ptr))));
                    v_acc4x0123 = _mm_add_ps(v_acc4x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c4_ptr))));
                    v_acc5x0123 = _mm_add_ps(v_acc5x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c5_ptr))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ps(_mm_min_ps(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ps(_mm_min_ps(v_acc3x0123, v_max), v_min);
                v_acc4x0123 = _mm_max_ps(_mm_min_ps(v_acc4x0123, v_max), v_min);
                v_acc5x0123 = _mm_max_ps(_mm_min_ps(v_acc5x0123, v_max), v_min);

                _mm_storel_epi64(reinterpret_cast<__m128i*>(c0_ptr), _mm_castps_si128(v_acc0x0123));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c1_ptr), _mm_castps_si128(v_acc1x0123));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c2_ptr), _mm_castps_si128(v_acc2x0123));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c3_ptr), _mm_castps_si128(v_acc3x0123));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c4_ptr), _mm_castps_si128(v_acc4x0123));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c5_ptr), _mm_castps_si128(v_acc5x0123));

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
                    v_acc0x0123 = _mm_add_ss(v_acc0x0123, _mm_load_ss(c0_ptr));
                    v_acc1x0123 = _mm_add_ss(v_acc1x0123, _mm_load_ss(c1_ptr));
                    v_acc2x0123 = _mm_add_ss(v_acc2x0123, _mm_load_ss(c2_ptr));
                    v_acc3x0123 = _mm_add_ss(v_acc3x0123, _mm_load_ss(c3_ptr));
                    v_acc4x0123 = _mm_add_ss(v_acc4x0123, _mm_load_ss(c4_ptr));
                    v_acc5x0123 = _mm_add_ss(v_acc5x0123, _mm_load_ss(c5_ptr));
                }

                v_acc0x0123 = _mm_max_ss(_mm_min_ss(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ss(_mm_min_ss(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ss(_mm_min_ss(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ss(_mm_min_ss(v_acc3x0123, v_max), v_min);
                v_acc4x0123 = _mm_max_ss(_mm_min_ss(v_acc4x0123, v_max), v_min);
                v_acc5x0123 = _mm_max_ss(_mm_min_ss(v_acc5x0123, v_max), v_min);

                _mm_store_ss(c0_ptr, v_acc0x0123);
                _mm_store_ss(c1_ptr, v_acc1x0123);
                _mm_store_ss(c2_ptr, v_acc2x0123);
                _mm_store_ss(c3_ptr, v_acc3x0123);
                _mm_store_ss(c4_ptr, v_acc4x0123);
                _mm_store_ss(c5_ptr, v_acc5x0123);
            }
        }
    }
}

template <bool zero_mode = false>
void mma_packed_ab_mr4_nr16_mrxncxkc_f32(
    int mr, int nc, int kc, float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A,
    const float* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max) noexcept {

    // Rows past `mr` alias the last live row in both A and C: they compute a
    // duplicate of it and store it back to the same place, so a partial-M tile
    // needs no per-row guards (the XNNPACK 6x16 broadcast convention).
    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;

    float* c0_ptr = C + 0 * ldc;
    float* c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    float* c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    float* c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;

    int ldb = static_cast<int>(mma_pack_align(16 * kc * sizeof(float)) / sizeof(float));

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

        const float* a_ptr = A;
        const float* b_ptr = B;
        for(int k = 0; k < kc; ++k) {
            const __m256 v_b0 = _mm256_loadu_ps(b_ptr + 0 * 8);
            const __m256 v_b1 = _mm256_loadu_ps(b_ptr + 1 * 8);
            b_ptr += 16;

            __m256 v_a0 = _mm256_broadcast_ss(a_ptr + 0);
            __m256 v_a1 = _mm256_broadcast_ss(a_ptr + r1);
            v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
            v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
            v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
            v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);

            v_a0 = _mm256_broadcast_ss(a_ptr + r2);
            v_a1 = _mm256_broadcast_ss(a_ptr + r3);
            v_acc2x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc2x01234567);
            v_acc2x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc2x89abcdef);
            v_acc3x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc3x01234567);
            v_acc3x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc3x89abcdef);

            a_ptr += 4;
        }
        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567, _mm256_loadu_ps(c0_ptr + 0));
            v_acc0x89abcdef = _mm256_add_ps(v_acc0x89abcdef, _mm256_loadu_ps(c0_ptr + 8));
            v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567, _mm256_loadu_ps(c1_ptr + 0));
            v_acc1x89abcdef = _mm256_add_ps(v_acc1x89abcdef, _mm256_loadu_ps(c1_ptr + 8));
            v_acc2x01234567 = _mm256_add_ps(v_acc2x01234567, _mm256_loadu_ps(c2_ptr + 0));
            v_acc2x89abcdef = _mm256_add_ps(v_acc2x89abcdef, _mm256_loadu_ps(c2_ptr + 8));
            v_acc3x01234567 = _mm256_add_ps(v_acc3x01234567, _mm256_loadu_ps(c3_ptr + 0));
            v_acc3x89abcdef = _mm256_add_ps(v_acc3x89abcdef, _mm256_loadu_ps(c3_ptr + 8));
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

        _mm256_storeu_ps(c0_ptr + 0, v_acc0x01234567);
        _mm256_storeu_ps(c0_ptr + 8, v_acc0x89abcdef);
        _mm256_storeu_ps(c1_ptr + 0, v_acc1x01234567);
        _mm256_storeu_ps(c1_ptr + 8, v_acc1x89abcdef);
        _mm256_storeu_ps(c2_ptr + 0, v_acc2x01234567);
        _mm256_storeu_ps(c2_ptr + 8, v_acc2x89abcdef);
        _mm256_storeu_ps(c3_ptr + 0, v_acc3x01234567);
        _mm256_storeu_ps(c3_ptr + 8, v_acc3x89abcdef);

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

        const float* a_ptr = A;
        const float* b_ptr = B;
        if (rem > 8) {
            for(int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_loadu_ps(b_ptr + 0 * 8);
                const __m256 v_b1 = _mm256_loadu_ps(b_ptr + 1 * 8);
                b_ptr += 16;

                __m256 v_a0 = _mm256_broadcast_ss(a_ptr + 0);
                __m256 v_a1 = _mm256_broadcast_ss(a_ptr + r1);
                v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
                v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
                v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
                v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);

                v_a0 = _mm256_broadcast_ss(a_ptr + r2);
                v_a1 = _mm256_broadcast_ss(a_ptr + r3);
                v_acc2x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc2x01234567);
                v_acc2x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc2x89abcdef);
                v_acc3x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc3x01234567);
                v_acc3x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc3x89abcdef);

                a_ptr += 4;
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_loadu_ps(b_ptr); b_ptr += 16;
                v_acc0x01234567 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 0), v_b0, v_acc0x01234567);
                v_acc1x01234567 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + r1), v_b0, v_acc1x01234567);
                v_acc2x01234567 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + r2), v_b0, v_acc2x01234567);
                v_acc3x01234567 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + r3), v_b0, v_acc3x01234567);
                a_ptr += 4;
            }
        }

        if (rem >= 8) {

            if constexpr (!zero_mode) {
                v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567, _mm256_loadu_ps(c0_ptr));
                v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567, _mm256_loadu_ps(c1_ptr));
                v_acc2x01234567 = _mm256_add_ps(v_acc2x01234567, _mm256_loadu_ps(c2_ptr));
                v_acc3x01234567 = _mm256_add_ps(v_acc3x01234567, _mm256_loadu_ps(c3_ptr));
            }

            const __m256 v_min = _mm256_set1_ps(clamp_min);
            const __m256 v_max = _mm256_set1_ps(clamp_max);

            v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
            v_acc1x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc1x01234567, v_max), v_min);
            v_acc2x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc2x01234567, v_max), v_min);
            v_acc3x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc3x01234567, v_max), v_min);

            _mm256_storeu_ps(c0_ptr, v_acc0x01234567);
            _mm256_storeu_ps(c1_ptr, v_acc1x01234567);
            _mm256_storeu_ps(c2_ptr, v_acc2x01234567);
            _mm256_storeu_ps(c3_ptr, v_acc3x01234567);

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
            // sit at lane 0 of the high half.
            __m128 v_acc0x0123 = _mm256_castps256_ps128(v_acc0x01234567);
            __m128 v_acc1x0123 = _mm256_castps256_ps128(v_acc1x01234567);
            __m128 v_acc2x0123 = _mm256_castps256_ps128(v_acc2x01234567);
            __m128 v_acc3x0123 = _mm256_castps256_ps128(v_acc3x01234567);

            const __m128 v_min = _mm_set1_ps(clamp_min);
            const __m128 v_max = _mm_set1_ps(clamp_max);

            if (rem & 4) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_loadu_ps(c0_ptr));
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_loadu_ps(c1_ptr));
                    v_acc2x0123 = _mm_add_ps(v_acc2x0123, _mm_loadu_ps(c2_ptr));
                    v_acc3x0123 = _mm_add_ps(v_acc3x0123, _mm_loadu_ps(c3_ptr));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ps(_mm_min_ps(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ps(_mm_min_ps(v_acc3x0123, v_max), v_min);

                _mm_storeu_ps(c0_ptr, v_acc0x0123);
                _mm_storeu_ps(c1_ptr, v_acc1x0123);
                _mm_storeu_ps(c2_ptr, v_acc2x0123);
                _mm_storeu_ps(c3_ptr, v_acc3x0123);

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
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0_ptr))));
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c1_ptr))));
                    v_acc2x0123 = _mm_add_ps(v_acc2x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c2_ptr))));
                    v_acc3x0123 = _mm_add_ps(v_acc3x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c3_ptr))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ps(_mm_min_ps(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ps(_mm_min_ps(v_acc3x0123, v_max), v_min);

                _mm_storel_epi64(reinterpret_cast<__m128i*>(c0_ptr), _mm_castps_si128(v_acc0x0123));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c1_ptr), _mm_castps_si128(v_acc1x0123));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c2_ptr), _mm_castps_si128(v_acc2x0123));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c3_ptr), _mm_castps_si128(v_acc3x0123));

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
                    v_acc0x0123 = _mm_add_ss(v_acc0x0123, _mm_load_ss(c0_ptr));
                    v_acc1x0123 = _mm_add_ss(v_acc1x0123, _mm_load_ss(c1_ptr));
                    v_acc2x0123 = _mm_add_ss(v_acc2x0123, _mm_load_ss(c2_ptr));
                    v_acc3x0123 = _mm_add_ss(v_acc3x0123, _mm_load_ss(c3_ptr));
                }

                v_acc0x0123 = _mm_max_ss(_mm_min_ss(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ss(_mm_min_ss(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ss(_mm_min_ss(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ss(_mm_min_ss(v_acc3x0123, v_max), v_min);

                _mm_store_ss(c0_ptr, v_acc0x0123);
                _mm_store_ss(c1_ptr, v_acc1x0123);
                _mm_store_ss(c2_ptr, v_acc2x0123);
                _mm_store_ss(c3_ptr, v_acc3x0123);
            }
        }
    }
}

template <bool zero_mode = false>
void mma_packed_ab_mr2_nr16_mrxncxkc_f32(
    int mr, int nc, int kc, float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A,
    const float* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max) noexcept {

    // Rows past `mr` alias the last live row in both A and C: they compute a
    // duplicate of it and store it back to the same place, so a partial-M tile
    // needs no per-row guards (the XNNPACK broadcast convention).
    const int r1 = (mr > 1) ? 1 : 0;

    float* c0_ptr = C;
    float* c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;

    int ldb = static_cast<int>(mma_pack_align(16 * kc * sizeof(float)) / sizeof(float));

    int n = 0;
    for(; n < nc - 15; n += 16) {
        __m256 v_acc0x01234567 = _mm256_setzero_ps();
        __m256 v_acc0x89abcdef = _mm256_setzero_ps();
        __m256 v_acc1x01234567 = v_acc0x01234567;
        __m256 v_acc1x89abcdef = v_acc0x89abcdef;

        const float* a_ptr = A;
        const float* b_ptr = B;
        for(int k = 0; k < kc; ++k) {
            const __m256 v_b0 = _mm256_loadu_ps(b_ptr + 0 * 8);
            const __m256 v_b1 = _mm256_loadu_ps(b_ptr + 1 * 8);
            b_ptr += 16;

            const __m256 v_a0 = _mm256_broadcast_ss(a_ptr + 0);
            const __m256 v_a1 = _mm256_broadcast_ss(a_ptr + r1);
            v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
            v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
            v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
            v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);

            a_ptr += 2;
        }
        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567, _mm256_loadu_ps(c0_ptr + 0));
            v_acc0x89abcdef = _mm256_add_ps(v_acc0x89abcdef, _mm256_loadu_ps(c0_ptr + 8));
            v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567, _mm256_loadu_ps(c1_ptr + 0));
            v_acc1x89abcdef = _mm256_add_ps(v_acc1x89abcdef, _mm256_loadu_ps(c1_ptr + 8));
        }
        const __m256 v_min = _mm256_set1_ps(clamp_min);
        const __m256 v_max = _mm256_set1_ps(clamp_max);

        v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
        v_acc0x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc0x89abcdef, v_max), v_min);
        v_acc1x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc1x01234567, v_max), v_min);
        v_acc1x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc1x89abcdef, v_max), v_min);

        _mm256_storeu_ps(c0_ptr + 0, v_acc0x01234567);
        _mm256_storeu_ps(c0_ptr + 8, v_acc0x89abcdef);
        _mm256_storeu_ps(c1_ptr + 0, v_acc1x01234567);
        _mm256_storeu_ps(c1_ptr + 8, v_acc1x89abcdef);

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

        const float* a_ptr = A;
        const float* b_ptr = B;
        if (rem > 8) {
            for(int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_loadu_ps(b_ptr + 0 * 8);
                const __m256 v_b1 = _mm256_loadu_ps(b_ptr + 1 * 8);
                b_ptr += 16;

                const __m256 v_a0 = _mm256_broadcast_ss(a_ptr + 0);
                const __m256 v_a1 = _mm256_broadcast_ss(a_ptr + r1);
                v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
                v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
                v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
                v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);

                a_ptr += 2;
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_loadu_ps(b_ptr); b_ptr += 16;
                const __m256 v_a0 = _mm256_broadcast_ss(a_ptr + 0);
                const __m256 v_a1 = _mm256_broadcast_ss(a_ptr + r1);
                v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
                v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
                a_ptr += 2;
            }
        }

        if (rem >= 8) {

            if constexpr (!zero_mode) {
                v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567, _mm256_loadu_ps(c0_ptr));
                v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567, _mm256_loadu_ps(c1_ptr));
            }

            const __m256 v_min = _mm256_set1_ps(clamp_min);
            const __m256 v_max = _mm256_set1_ps(clamp_max);

            v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
            v_acc1x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc1x01234567, v_max), v_min);

            _mm256_storeu_ps(c0_ptr, v_acc0x01234567);
            _mm256_storeu_ps(c1_ptr, v_acc1x01234567);

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
            // sit at lane 0 of the high half.
            __m128 v_acc0x0123 = _mm256_castps256_ps128(v_acc0x01234567);
            __m128 v_acc1x0123 = _mm256_castps256_ps128(v_acc1x01234567);

            const __m128 v_min = _mm_set1_ps(clamp_min);
            const __m128 v_max = _mm_set1_ps(clamp_max);

            if (rem & 4) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_loadu_ps(c0_ptr));
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_loadu_ps(c1_ptr));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);

                _mm_storeu_ps(c0_ptr, v_acc0x0123);
                _mm_storeu_ps(c1_ptr, v_acc1x0123);

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
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0_ptr))));
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c1_ptr))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);

                _mm_storel_epi64(reinterpret_cast<__m128i*>(c0_ptr), _mm_castps_si128(v_acc0x0123));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c1_ptr), _mm_castps_si128(v_acc1x0123));

                c0_ptr += 2;
                c1_ptr += 2;

                v_acc0x0123 = _mm_movehl_ps(v_rest0, v_rest0);
                v_acc1x0123 = _mm_movehl_ps(v_rest1, v_rest1);
            }

            if (rem & 1) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ss(v_acc0x0123, _mm_load_ss(c0_ptr));
                    v_acc1x0123 = _mm_add_ss(v_acc1x0123, _mm_load_ss(c1_ptr));
                }

                v_acc0x0123 = _mm_max_ss(_mm_min_ss(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ss(_mm_min_ss(v_acc1x0123, v_max), v_min);

                _mm_store_ss(c0_ptr, v_acc0x0123);
                _mm_store_ss(c1_ptr, v_acc1x0123);
            }
        }
    }
}

template <bool zero_mode = false>
void mma_packed_ab_mr1_nr16_mrxncxkc_f32(
    int mr, int nc, int kc, float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A,
    const float* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max) noexcept {

    (void)mr;   // one row, so the row count never varies
    (void)ldc;  // one row, so the C row stride never applies

    float* c0_ptr = C;

    int ldb = static_cast<int>(mma_pack_align(16 * kc * sizeof(float)) / sizeof(float));

    int n = 0;
    for(; n < nc - 15; n += 16) {
        __m256 v_acc0x01234567 = _mm256_setzero_ps();
        __m256 v_acc0x89abcdef = _mm256_setzero_ps();

        const float* a_ptr = A;
        const float* b_ptr = B;
        for(int k = 0; k < kc; ++k) {
            const __m256 v_b0 = _mm256_loadu_ps(b_ptr + 0 * 8);
            const __m256 v_b1 = _mm256_loadu_ps(b_ptr + 1 * 8);
            b_ptr += 16;

            const __m256 v_a0 = _mm256_broadcast_ss(a_ptr + 0);
            v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
            v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);

            a_ptr += 1;
        }
        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567, _mm256_loadu_ps(c0_ptr + 0));
            v_acc0x89abcdef = _mm256_add_ps(v_acc0x89abcdef, _mm256_loadu_ps(c0_ptr + 8));
        }
        const __m256 v_min = _mm256_set1_ps(clamp_min);
        const __m256 v_max = _mm256_set1_ps(clamp_max);

        v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
        v_acc0x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc0x89abcdef, v_max), v_min);

        _mm256_storeu_ps(c0_ptr + 0, v_acc0x01234567);
        _mm256_storeu_ps(c0_ptr + 8, v_acc0x89abcdef);

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

        const float* a_ptr = A;
        const float* b_ptr = B;
        if (rem > 8) {
            for(int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_loadu_ps(b_ptr + 0 * 8);
                const __m256 v_b1 = _mm256_loadu_ps(b_ptr + 1 * 8);
                b_ptr += 16;

                const __m256 v_a0 = _mm256_broadcast_ss(a_ptr + 0);
                v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
                v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);

                a_ptr += 1;
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_loadu_ps(b_ptr); b_ptr += 16;
                v_acc0x01234567 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 0), v_b0, v_acc0x01234567);
                a_ptr += 1;
            }
        }

        if (rem >= 8) {

            if constexpr (!zero_mode) {
                v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567, _mm256_loadu_ps(c0_ptr));
            }

            const __m256 v_min = _mm256_set1_ps(clamp_min);
            const __m256 v_max = _mm256_set1_ps(clamp_max);

            v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);

            _mm256_storeu_ps(c0_ptr, v_acc0x01234567);

            c0_ptr += 8;

            if (rem > 8) {
                v_acc0x01234567 = v_acc0x89abcdef;
            }
            rem -= 8;
        }

        if (rem > 0) {
            // Lane j holds column j, so the 4/2/1 ladder is just a lo/hi split of
            // the same accumulator: after the 4-wide step the leftover 1..3 columns
            // sit at lane 0 of the high half.
            __m128 v_acc0x0123 = _mm256_castps256_ps128(v_acc0x01234567);

            const __m128 v_min = _mm_set1_ps(clamp_min);
            const __m128 v_max = _mm_set1_ps(clamp_max);

            if (rem & 4) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_loadu_ps(c0_ptr));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);

                _mm_storeu_ps(c0_ptr, v_acc0x0123);

                c0_ptr += 4;

                v_acc0x0123 = _mm256_extractf128_ps(v_acc0x01234567, 1);
            }

            if (rem & 2) {
                // The clamp below is 4-wide, so keep the untouched leftovers for
                // the 1-wide step -- lanes 2,3 are what it will store.
                const __m128 v_rest0 = v_acc0x0123;

                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0_ptr))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);

                _mm_storel_epi64(reinterpret_cast<__m128i*>(c0_ptr), _mm_castps_si128(v_acc0x0123));

                c0_ptr += 2;

                v_acc0x0123 = _mm_movehl_ps(v_rest0, v_rest0);
            }

            if (rem & 1) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ss(v_acc0x0123, _mm_load_ss(c0_ptr));
                }

                v_acc0x0123 = _mm_max_ss(_mm_min_ss(v_acc0x0123, v_max), v_min);

                _mm_store_ss(c0_ptr, v_acc0x0123);
            }
        }
    }
}

template <bool zero_mode = false, int seed = 0>
void mma_packed_b_mr6_nr16_mrxncxkc_f32(
    int mr, int nc, int kc, float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A, int lda,
    const float* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max,
    const float* NNOPS_RESTRICT bias = nullptr) noexcept {

    // Rows past `mr` alias the last live row in both A and C: they compute a
    // duplicate of it and store it back to the same place, so a partial-M tile
    // needs no per-row guards (the XNNPACK 6x16 broadcast convention).
    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;
    const int r4 = (mr > 4) ? 4 : r3;
    const int r5 = (mr > 5) ? 5 : r4;

    float* c0_ptr = C + 0 * ldc;
    float* c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    float* c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    float* c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;
    float* c4_ptr = (mr > 4) ? C + 4 * ldc : c3_ptr;
    float* c5_ptr = (mr > 5) ? C + 5 * ldc : c4_ptr;

    int ldb = static_cast<int>(mma_pack_align(16 * kc * sizeof(float)) / sizeof(float));

    int n = 0;

    // Accumulator initial value. Seed 1 is a per-row bias — the planar route,
    // where a row is an output channel; seed 2 is a per-column one, which is the
    // channel-last route's shape: the rows are pixels and the bias lands on the
    // panel's 16 lanes, so it costs two loads per panel where seeding the tile
    // first costs a full write and read-back of the output. `off` is the lane
    // offset inside the panel. A row past `mr` aliases the last live row in A
    // and C, so the same alias index keeps the row seed in range.
    const auto init = [&](int r, int off) {
        if constexpr (seed == 1) { return _mm256_set1_ps(bias[r]); }
        else if constexpr (seed == 2) { return _mm256_loadu_ps(bias + n + off); }
        else { return _mm256_setzero_ps(); }
    };

    for(; n < nc - 15; n += 16) {
        const float* a0_ptr = A + 0 * lda;
        const float* a1_ptr = A + r1 * lda;
        const float* a2_ptr = A + r2 * lda;
        const float* a3_ptr = A + r3 * lda;
        const float* a4_ptr = A + r4 * lda;
        const float* a5_ptr = A + r5 * lda;

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

        const float* b_ptr = B;
        for(int k = 0; k < kc; ++k) {
            const __m256 v_b0 = _mm256_loadu_ps(b_ptr + 0 * 8);
            const __m256 v_b1 = _mm256_loadu_ps(b_ptr + 1 * 8);
            b_ptr += 16;

            __m256 v_a0 = _mm256_broadcast_ss(a0_ptr); a0_ptr += 1;
            __m256 v_a1 = _mm256_broadcast_ss(a1_ptr); a1_ptr += 1;
            v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
            v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
            v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
            v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);

            v_a0 = _mm256_broadcast_ss(a2_ptr); a2_ptr += 1;
            v_a1 = _mm256_broadcast_ss(a3_ptr); a3_ptr += 1;
            v_acc2x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc2x01234567);
            v_acc2x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc2x89abcdef);
            v_acc3x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc3x01234567);
            v_acc3x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc3x89abcdef);

            v_a0 = _mm256_broadcast_ss(a4_ptr); a4_ptr += 1;
            v_a1 = _mm256_broadcast_ss(a5_ptr); a5_ptr += 1;
            v_acc4x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc4x01234567);
            v_acc4x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc4x89abcdef);
            v_acc5x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc5x01234567);
            v_acc5x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc5x89abcdef);
        }

        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567, _mm256_loadu_ps(c0_ptr + 0));
            v_acc0x89abcdef = _mm256_add_ps(v_acc0x89abcdef, _mm256_loadu_ps(c0_ptr + 8));
            v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567, _mm256_loadu_ps(c1_ptr + 0));
            v_acc1x89abcdef = _mm256_add_ps(v_acc1x89abcdef, _mm256_loadu_ps(c1_ptr + 8));
            v_acc2x01234567 = _mm256_add_ps(v_acc2x01234567, _mm256_loadu_ps(c2_ptr + 0));
            v_acc2x89abcdef = _mm256_add_ps(v_acc2x89abcdef, _mm256_loadu_ps(c2_ptr + 8));
            v_acc3x01234567 = _mm256_add_ps(v_acc3x01234567, _mm256_loadu_ps(c3_ptr + 0));
            v_acc3x89abcdef = _mm256_add_ps(v_acc3x89abcdef, _mm256_loadu_ps(c3_ptr + 8));
            v_acc4x01234567 = _mm256_add_ps(v_acc4x01234567, _mm256_loadu_ps(c4_ptr + 0));
            v_acc4x89abcdef = _mm256_add_ps(v_acc4x89abcdef, _mm256_loadu_ps(c4_ptr + 8));
            v_acc5x01234567 = _mm256_add_ps(v_acc5x01234567, _mm256_loadu_ps(c5_ptr + 0));
            v_acc5x89abcdef = _mm256_add_ps(v_acc5x89abcdef, _mm256_loadu_ps(c5_ptr + 8));
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

        _mm256_storeu_ps(c0_ptr + 0, v_acc0x01234567);
        _mm256_storeu_ps(c0_ptr + 8, v_acc0x89abcdef);
        _mm256_storeu_ps(c1_ptr + 0, v_acc1x01234567);
        _mm256_storeu_ps(c1_ptr + 8, v_acc1x89abcdef);
        _mm256_storeu_ps(c2_ptr + 0, v_acc2x01234567);
        _mm256_storeu_ps(c2_ptr + 8, v_acc2x89abcdef);
        _mm256_storeu_ps(c3_ptr + 0, v_acc3x01234567);
        _mm256_storeu_ps(c3_ptr + 8, v_acc3x89abcdef);
        _mm256_storeu_ps(c4_ptr + 0, v_acc4x01234567);
        _mm256_storeu_ps(c4_ptr + 8, v_acc4x89abcdef);
        _mm256_storeu_ps(c5_ptr + 0, v_acc5x01234567);
        _mm256_storeu_ps(c5_ptr + 8, v_acc5x89abcdef);

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
        const float* a0_ptr = A + 0 * lda;
        const float* a1_ptr = A + r1 * lda;
        const float* a2_ptr = A + r2 * lda;
        const float* a3_ptr = A + r3 * lda;
        const float* a4_ptr = A + r4 * lda;
        const float* a5_ptr = A + r5 * lda;
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

        const float* b_ptr = B;
        if (rem > 8) {
            for(int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_loadu_ps(b_ptr + 0 * 8);
                const __m256 v_b1 = _mm256_loadu_ps(b_ptr + 1 * 8);
                b_ptr += 16;

                __m256 v_a0 = _mm256_broadcast_ss(a0_ptr); a0_ptr += 1;
                __m256 v_a1 = _mm256_broadcast_ss(a1_ptr); a1_ptr += 1;
                v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
                v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
                v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
                v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);

                v_a0 = _mm256_broadcast_ss(a2_ptr); a2_ptr += 1;
                v_a1 = _mm256_broadcast_ss(a3_ptr); a3_ptr += 1;
                v_acc2x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc2x01234567);
                v_acc2x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc2x89abcdef);
                v_acc3x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc3x01234567);
                v_acc3x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc3x89abcdef);

                v_a0 = _mm256_broadcast_ss(a4_ptr); a4_ptr += 1;
                v_a1 = _mm256_broadcast_ss(a5_ptr); a5_ptr += 1;
                v_acc4x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc4x01234567);
                v_acc4x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc4x89abcdef);
                v_acc5x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc5x01234567);
                v_acc5x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc5x89abcdef);
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_loadu_ps(b_ptr); b_ptr += 16;
                v_acc0x01234567 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0_ptr), v_b0, v_acc0x01234567);
                v_acc1x01234567 = _mm256_fmadd_ps(_mm256_broadcast_ss(a1_ptr), v_b0, v_acc1x01234567);
                v_acc2x01234567 = _mm256_fmadd_ps(_mm256_broadcast_ss(a2_ptr), v_b0, v_acc2x01234567);
                v_acc3x01234567 = _mm256_fmadd_ps(_mm256_broadcast_ss(a3_ptr), v_b0, v_acc3x01234567);
                v_acc4x01234567 = _mm256_fmadd_ps(_mm256_broadcast_ss(a4_ptr), v_b0, v_acc4x01234567);
                v_acc5x01234567 = _mm256_fmadd_ps(_mm256_broadcast_ss(a5_ptr), v_b0, v_acc5x01234567);
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
                v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567, _mm256_loadu_ps(c0_ptr));
                v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567, _mm256_loadu_ps(c1_ptr));
                v_acc2x01234567 = _mm256_add_ps(v_acc2x01234567, _mm256_loadu_ps(c2_ptr));
                v_acc3x01234567 = _mm256_add_ps(v_acc3x01234567, _mm256_loadu_ps(c3_ptr));
                v_acc4x01234567 = _mm256_add_ps(v_acc4x01234567, _mm256_loadu_ps(c4_ptr));
                v_acc5x01234567 = _mm256_add_ps(v_acc5x01234567, _mm256_loadu_ps(c5_ptr));
            }

            const __m256 v_min = _mm256_set1_ps(clamp_min);
            const __m256 v_max = _mm256_set1_ps(clamp_max);

            v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
            v_acc1x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc1x01234567, v_max), v_min);
            v_acc2x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc2x01234567, v_max), v_min);
            v_acc3x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc3x01234567, v_max), v_min);
            v_acc4x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc4x01234567, v_max), v_min);
            v_acc5x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc5x01234567, v_max), v_min);

            _mm256_storeu_ps(c0_ptr, v_acc0x01234567);
            _mm256_storeu_ps(c1_ptr, v_acc1x01234567);
            _mm256_storeu_ps(c2_ptr, v_acc2x01234567);
            _mm256_storeu_ps(c3_ptr, v_acc3x01234567);
            _mm256_storeu_ps(c4_ptr, v_acc4x01234567);
            _mm256_storeu_ps(c5_ptr, v_acc5x01234567);

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
            // sit at lane 0 of the high half.
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
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_loadu_ps(c0_ptr));
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_loadu_ps(c1_ptr));
                    v_acc2x0123 = _mm_add_ps(v_acc2x0123, _mm_loadu_ps(c2_ptr));
                    v_acc3x0123 = _mm_add_ps(v_acc3x0123, _mm_loadu_ps(c3_ptr));
                    v_acc4x0123 = _mm_add_ps(v_acc4x0123, _mm_loadu_ps(c4_ptr));
                    v_acc5x0123 = _mm_add_ps(v_acc5x0123, _mm_loadu_ps(c5_ptr));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ps(_mm_min_ps(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ps(_mm_min_ps(v_acc3x0123, v_max), v_min);
                v_acc4x0123 = _mm_max_ps(_mm_min_ps(v_acc4x0123, v_max), v_min);
                v_acc5x0123 = _mm_max_ps(_mm_min_ps(v_acc5x0123, v_max), v_min);

                _mm_storeu_ps(c0_ptr, v_acc0x0123);
                _mm_storeu_ps(c1_ptr, v_acc1x0123);
                _mm_storeu_ps(c2_ptr, v_acc2x0123);
                _mm_storeu_ps(c3_ptr, v_acc3x0123);
                _mm_storeu_ps(c4_ptr, v_acc4x0123);
                _mm_storeu_ps(c5_ptr, v_acc5x0123);

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
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0_ptr))));
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c1_ptr))));
                    v_acc2x0123 = _mm_add_ps(v_acc2x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c2_ptr))));
                    v_acc3x0123 = _mm_add_ps(v_acc3x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c3_ptr))));
                    v_acc4x0123 = _mm_add_ps(v_acc4x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c4_ptr))));
                    v_acc5x0123 = _mm_add_ps(v_acc5x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c5_ptr))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ps(_mm_min_ps(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ps(_mm_min_ps(v_acc3x0123, v_max), v_min);
                v_acc4x0123 = _mm_max_ps(_mm_min_ps(v_acc4x0123, v_max), v_min);
                v_acc5x0123 = _mm_max_ps(_mm_min_ps(v_acc5x0123, v_max), v_min);

                _mm_storel_epi64(reinterpret_cast<__m128i*>(c0_ptr), _mm_castps_si128(v_acc0x0123));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c1_ptr), _mm_castps_si128(v_acc1x0123));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c2_ptr), _mm_castps_si128(v_acc2x0123));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c3_ptr), _mm_castps_si128(v_acc3x0123));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c4_ptr), _mm_castps_si128(v_acc4x0123));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c5_ptr), _mm_castps_si128(v_acc5x0123));

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
                    v_acc0x0123 = _mm_add_ss(v_acc0x0123, _mm_load_ss(c0_ptr));
                    v_acc1x0123 = _mm_add_ss(v_acc1x0123, _mm_load_ss(c1_ptr));
                    v_acc2x0123 = _mm_add_ss(v_acc2x0123, _mm_load_ss(c2_ptr));
                    v_acc3x0123 = _mm_add_ss(v_acc3x0123, _mm_load_ss(c3_ptr));
                    v_acc4x0123 = _mm_add_ss(v_acc4x0123, _mm_load_ss(c4_ptr));
                    v_acc5x0123 = _mm_add_ss(v_acc5x0123, _mm_load_ss(c5_ptr));
                }

                v_acc0x0123 = _mm_max_ss(_mm_min_ss(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ss(_mm_min_ss(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ss(_mm_min_ss(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ss(_mm_min_ss(v_acc3x0123, v_max), v_min);
                v_acc4x0123 = _mm_max_ss(_mm_min_ss(v_acc4x0123, v_max), v_min);
                v_acc5x0123 = _mm_max_ss(_mm_min_ss(v_acc5x0123, v_max), v_min);

                _mm_store_ss(c0_ptr, v_acc0x0123);
                _mm_store_ss(c1_ptr, v_acc1x0123);
                _mm_store_ss(c2_ptr, v_acc2x0123);
                _mm_store_ss(c3_ptr, v_acc3x0123);
                _mm_store_ss(c4_ptr, v_acc4x0123);
                _mm_store_ss(c5_ptr, v_acc5x0123);
            }
        }
    }
}

template <bool zero_mode = false, int seed = 0>
void mma_packed_b_mr4_nr16_mrxncxkc_f32(
    int mr, int nc, int kc, float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A, int lda,
    const float* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max,
    const float* NNOPS_RESTRICT bias = nullptr) noexcept {

    // Rows past `mr` alias the last live row in both A and C: they compute a
    // duplicate of it and store it back to the same place, so a partial-M tile
    // needs no per-row guards (the XNNPACK 6x16 broadcast convention).
    const int r1 = (mr > 1) ? 1 : 0;
    const int r2 = (mr > 2) ? 2 : r1;
    const int r3 = (mr > 3) ? 3 : r2;

    float* c0_ptr = C + 0 * ldc;
    float* c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;
    float* c2_ptr = (mr > 2) ? C + 2 * ldc : c1_ptr;
    float* c3_ptr = (mr > 3) ? C + 3 * ldc : c2_ptr;

    int ldb = static_cast<int>(mma_pack_align(16 * kc * sizeof(float)) / sizeof(float));

    int n = 0;

    // Accumulator initial value. Seed 1 is a per-row bias — the planar route,
    // where a row is an output channel; seed 2 is a per-column one, which is the
    // channel-last route's shape: the rows are pixels and the bias lands on the
    // panel's 16 lanes, so it costs two loads per panel where seeding the tile
    // first costs a full write and read-back of the output. `off` is the lane
    // offset inside the panel. A row past `mr` aliases the last live row in A
    // and C, so the same alias index keeps the row seed in range.
    const auto init = [&](int r, int off) {
        if constexpr (seed == 1) { return _mm256_set1_ps(bias[r]); }
        else if constexpr (seed == 2) { return _mm256_loadu_ps(bias + n + off); }
        else { return _mm256_setzero_ps(); }
    };

    for(; n < nc - 15; n += 16) {
        const float* a0_ptr = A + 0 * lda;
        const float* a1_ptr = A + r1 * lda;
        const float* a2_ptr = A + r2 * lda;
        const float* a3_ptr = A + r3 * lda;

        __m256 v_acc0x01234567 = init(0, 0);
        __m256 v_acc0x89abcdef = init(0, 8);
        __m256 v_acc1x01234567 = init(r1, 0);
        __m256 v_acc1x89abcdef = init(r1, 8);
        __m256 v_acc2x01234567 = init(r2, 0);
        __m256 v_acc2x89abcdef = init(r2, 8);
        __m256 v_acc3x01234567 = init(r3, 0);
        __m256 v_acc3x89abcdef = init(r3, 8);

        const float* b_ptr = B;
        for(int k = 0; k < kc; ++k) {
            const __m256 v_b0 = _mm256_loadu_ps(b_ptr + 0 * 8);
            const __m256 v_b1 = _mm256_loadu_ps(b_ptr + 1 * 8);
            b_ptr += 16;

            __m256 v_a0 = _mm256_broadcast_ss(a0_ptr); a0_ptr += 1;
            __m256 v_a1 = _mm256_broadcast_ss(a1_ptr); a1_ptr += 1;
            v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
            v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
            v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
            v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);

            v_a0 = _mm256_broadcast_ss(a2_ptr); a2_ptr += 1;
            v_a1 = _mm256_broadcast_ss(a3_ptr); a3_ptr += 1;
            v_acc2x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc2x01234567);
            v_acc2x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc2x89abcdef);
            v_acc3x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc3x01234567);
            v_acc3x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc3x89abcdef);
        }

        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567, _mm256_loadu_ps(c0_ptr + 0));
            v_acc0x89abcdef = _mm256_add_ps(v_acc0x89abcdef, _mm256_loadu_ps(c0_ptr + 8));
            v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567, _mm256_loadu_ps(c1_ptr + 0));
            v_acc1x89abcdef = _mm256_add_ps(v_acc1x89abcdef, _mm256_loadu_ps(c1_ptr + 8));
            v_acc2x01234567 = _mm256_add_ps(v_acc2x01234567, _mm256_loadu_ps(c2_ptr + 0));
            v_acc2x89abcdef = _mm256_add_ps(v_acc2x89abcdef, _mm256_loadu_ps(c2_ptr + 8));
            v_acc3x01234567 = _mm256_add_ps(v_acc3x01234567, _mm256_loadu_ps(c3_ptr + 0));
            v_acc3x89abcdef = _mm256_add_ps(v_acc3x89abcdef, _mm256_loadu_ps(c3_ptr + 8));
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

        _mm256_storeu_ps(c0_ptr + 0, v_acc0x01234567);
        _mm256_storeu_ps(c0_ptr + 8, v_acc0x89abcdef);
        _mm256_storeu_ps(c1_ptr + 0, v_acc1x01234567);
        _mm256_storeu_ps(c1_ptr + 8, v_acc1x89abcdef);
        _mm256_storeu_ps(c2_ptr + 0, v_acc2x01234567);
        _mm256_storeu_ps(c2_ptr + 8, v_acc2x89abcdef);
        _mm256_storeu_ps(c3_ptr + 0, v_acc3x01234567);
        _mm256_storeu_ps(c3_ptr + 8, v_acc3x89abcdef);

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
        const float* a0_ptr = A + 0 * lda;
        const float* a1_ptr = A + r1 * lda;
        const float* a2_ptr = A + r2 * lda;
        const float* a3_ptr = A + r3 * lda;
        __m256 v_acc0x01234567 = init(0, 0);
        __m256 v_acc0x89abcdef = init(0, 8);
        __m256 v_acc1x01234567 = init(r1, 0);
        __m256 v_acc1x89abcdef = init(r1, 8);
        __m256 v_acc2x01234567 = init(r2, 0);
        __m256 v_acc2x89abcdef = init(r2, 8);
        __m256 v_acc3x01234567 = init(r3, 0);
        __m256 v_acc3x89abcdef = init(r3, 8);

        const float* b_ptr = B;
        if (rem > 8) {
            for(int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_loadu_ps(b_ptr + 0 * 8);
                const __m256 v_b1 = _mm256_loadu_ps(b_ptr + 1 * 8);
                b_ptr += 16;

                __m256 v_a0 = _mm256_broadcast_ss(a0_ptr); a0_ptr += 1;
                __m256 v_a1 = _mm256_broadcast_ss(a1_ptr); a1_ptr += 1;
                v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
                v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
                v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
                v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);

                v_a0 = _mm256_broadcast_ss(a2_ptr); a2_ptr += 1;
                v_a1 = _mm256_broadcast_ss(a3_ptr); a3_ptr += 1;
                v_acc2x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc2x01234567);
                v_acc2x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc2x89abcdef);
                v_acc3x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc3x01234567);
                v_acc3x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc3x89abcdef);
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_loadu_ps(b_ptr); b_ptr += 16;
                v_acc0x01234567 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0_ptr), v_b0, v_acc0x01234567);
                v_acc1x01234567 = _mm256_fmadd_ps(_mm256_broadcast_ss(a1_ptr), v_b0, v_acc1x01234567);
                v_acc2x01234567 = _mm256_fmadd_ps(_mm256_broadcast_ss(a2_ptr), v_b0, v_acc2x01234567);
                v_acc3x01234567 = _mm256_fmadd_ps(_mm256_broadcast_ss(a3_ptr), v_b0, v_acc3x01234567);
                a0_ptr += 1;
                a1_ptr += 1;
                a2_ptr += 1;
                a3_ptr += 1;
            }
        }

        if (rem >= 8) {

            if constexpr (!zero_mode) {
                v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567, _mm256_loadu_ps(c0_ptr));
                v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567, _mm256_loadu_ps(c1_ptr));
                v_acc2x01234567 = _mm256_add_ps(v_acc2x01234567, _mm256_loadu_ps(c2_ptr));
                v_acc3x01234567 = _mm256_add_ps(v_acc3x01234567, _mm256_loadu_ps(c3_ptr));
            }

            const __m256 v_min = _mm256_set1_ps(clamp_min);
            const __m256 v_max = _mm256_set1_ps(clamp_max);

            v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
            v_acc1x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc1x01234567, v_max), v_min);
            v_acc2x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc2x01234567, v_max), v_min);
            v_acc3x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc3x01234567, v_max), v_min);

            _mm256_storeu_ps(c0_ptr, v_acc0x01234567);
            _mm256_storeu_ps(c1_ptr, v_acc1x01234567);
            _mm256_storeu_ps(c2_ptr, v_acc2x01234567);
            _mm256_storeu_ps(c3_ptr, v_acc3x01234567);

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
            // sit at lane 0 of the high half.
            __m128 v_acc0x0123 = _mm256_castps256_ps128(v_acc0x01234567);
            __m128 v_acc1x0123 = _mm256_castps256_ps128(v_acc1x01234567);
            __m128 v_acc2x0123 = _mm256_castps256_ps128(v_acc2x01234567);
            __m128 v_acc3x0123 = _mm256_castps256_ps128(v_acc3x01234567);

            const __m128 v_min = _mm_set1_ps(clamp_min);
            const __m128 v_max = _mm_set1_ps(clamp_max);

            if (rem & 4) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_loadu_ps(c0_ptr));
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_loadu_ps(c1_ptr));
                    v_acc2x0123 = _mm_add_ps(v_acc2x0123, _mm_loadu_ps(c2_ptr));
                    v_acc3x0123 = _mm_add_ps(v_acc3x0123, _mm_loadu_ps(c3_ptr));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ps(_mm_min_ps(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ps(_mm_min_ps(v_acc3x0123, v_max), v_min);

                _mm_storeu_ps(c0_ptr, v_acc0x0123);
                _mm_storeu_ps(c1_ptr, v_acc1x0123);
                _mm_storeu_ps(c2_ptr, v_acc2x0123);
                _mm_storeu_ps(c3_ptr, v_acc3x0123);

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
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0_ptr))));
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c1_ptr))));
                    v_acc2x0123 = _mm_add_ps(v_acc2x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c2_ptr))));
                    v_acc3x0123 = _mm_add_ps(v_acc3x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c3_ptr))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ps(_mm_min_ps(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ps(_mm_min_ps(v_acc3x0123, v_max), v_min);

                _mm_storel_epi64(reinterpret_cast<__m128i*>(c0_ptr), _mm_castps_si128(v_acc0x0123));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c1_ptr), _mm_castps_si128(v_acc1x0123));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c2_ptr), _mm_castps_si128(v_acc2x0123));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c3_ptr), _mm_castps_si128(v_acc3x0123));

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
                    v_acc0x0123 = _mm_add_ss(v_acc0x0123, _mm_load_ss(c0_ptr));
                    v_acc1x0123 = _mm_add_ss(v_acc1x0123, _mm_load_ss(c1_ptr));
                    v_acc2x0123 = _mm_add_ss(v_acc2x0123, _mm_load_ss(c2_ptr));
                    v_acc3x0123 = _mm_add_ss(v_acc3x0123, _mm_load_ss(c3_ptr));
                }

                v_acc0x0123 = _mm_max_ss(_mm_min_ss(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ss(_mm_min_ss(v_acc1x0123, v_max), v_min);
                v_acc2x0123 = _mm_max_ss(_mm_min_ss(v_acc2x0123, v_max), v_min);
                v_acc3x0123 = _mm_max_ss(_mm_min_ss(v_acc3x0123, v_max), v_min);

                _mm_store_ss(c0_ptr, v_acc0x0123);
                _mm_store_ss(c1_ptr, v_acc1x0123);
                _mm_store_ss(c2_ptr, v_acc2x0123);
                _mm_store_ss(c3_ptr, v_acc3x0123);
            }
        }
    }
}

template <bool zero_mode = false, int seed = 0>
void mma_packed_b_mr2_nr16_mrxncxkc_f32(
    int mr, int nc, int kc, float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A, int lda,
    const float* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max,
    const float* NNOPS_RESTRICT bias = nullptr) noexcept {

    // Rows past `mr` alias the last live row in both A and C: they compute a
    // duplicate of it and store it back to the same place, so a partial-M tile
    // needs no per-row guards (the XNNPACK broadcast convention).
    const int r1 = (mr > 1) ? 1 : 0;

    float* c0_ptr = C;
    float* c1_ptr = (mr > 1) ? C + 1 * ldc : c0_ptr;

    int ldb = static_cast<int>(mma_pack_align(16 * kc * sizeof(float)) / sizeof(float));

    int n = 0;

    // Accumulator initial value. Seed 1 is a per-row bias — the planar route,
    // where a row is an output channel; seed 2 is a per-column one, which is the
    // channel-last route's shape: the rows are pixels and the bias lands on the
    // panel's 16 lanes, so it costs two loads per panel where seeding the tile
    // first costs a full write and read-back of the output. `off` is the lane
    // offset inside the panel. A row past `mr` aliases the last live row in A
    // and C, so the same alias index keeps the row seed in range.
    const auto init = [&](int r, int off) {
        if constexpr (seed == 1) { return _mm256_set1_ps(bias[r]); }
        else if constexpr (seed == 2) { return _mm256_loadu_ps(bias + n + off); }
        else { return _mm256_setzero_ps(); }
    };

    for(; n < nc - 15; n += 16) {
        const float* a0_ptr = A + 0 * lda;
        const float* a1_ptr = A + r1 * lda;

        __m256 v_acc0x01234567 = init(0, 0);
        __m256 v_acc0x89abcdef = init(0, 8);
        __m256 v_acc1x01234567 = init(r1, 0);
        __m256 v_acc1x89abcdef = init(r1, 8);

        const float* b_ptr = B;
        for(int k = 0; k < kc; ++k) {
            const __m256 v_b0 = _mm256_loadu_ps(b_ptr + 0 * 8);
            const __m256 v_b1 = _mm256_loadu_ps(b_ptr + 1 * 8);
            b_ptr += 16;

            const __m256 v_a0 = _mm256_broadcast_ss(a0_ptr); a0_ptr += 1;
            const __m256 v_a1 = _mm256_broadcast_ss(a1_ptr); a1_ptr += 1;
            v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
            v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
            v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
            v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);
        }

        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567, _mm256_loadu_ps(c0_ptr + 0));
            v_acc0x89abcdef = _mm256_add_ps(v_acc0x89abcdef, _mm256_loadu_ps(c0_ptr + 8));
            v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567, _mm256_loadu_ps(c1_ptr + 0));
            v_acc1x89abcdef = _mm256_add_ps(v_acc1x89abcdef, _mm256_loadu_ps(c1_ptr + 8));
        }
        const __m256 v_min = _mm256_set1_ps(clamp_min);
        const __m256 v_max = _mm256_set1_ps(clamp_max);

        v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
        v_acc0x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc0x89abcdef, v_max), v_min);
        v_acc1x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc1x01234567, v_max), v_min);
        v_acc1x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc1x89abcdef, v_max), v_min);

        _mm256_storeu_ps(c0_ptr + 0, v_acc0x01234567);
        _mm256_storeu_ps(c0_ptr + 8, v_acc0x89abcdef);
        _mm256_storeu_ps(c1_ptr + 0, v_acc1x01234567);
        _mm256_storeu_ps(c1_ptr + 8, v_acc1x89abcdef);

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
        const float* a0_ptr = A + 0 * lda;
        const float* a1_ptr = A + r1 * lda;
        __m256 v_acc0x01234567 = init(0, 0);
        __m256 v_acc0x89abcdef = init(0, 8);
        __m256 v_acc1x01234567 = init(r1, 0);
        __m256 v_acc1x89abcdef = init(r1, 8);

        const float* b_ptr = B;
        if (rem > 8) {
            for(int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_loadu_ps(b_ptr + 0 * 8);
                const __m256 v_b1 = _mm256_loadu_ps(b_ptr + 1 * 8);
                b_ptr += 16;

                const __m256 v_a0 = _mm256_broadcast_ss(a0_ptr); a0_ptr += 1;
                const __m256 v_a1 = _mm256_broadcast_ss(a1_ptr); a1_ptr += 1;
                v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
                v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
                v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
                v_acc1x89abcdef = _mm256_fmadd_ps(v_a1, v_b1, v_acc1x89abcdef);
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_loadu_ps(b_ptr); b_ptr += 16;
                const __m256 v_a0 = _mm256_broadcast_ss(a0_ptr); a0_ptr += 1;
                const __m256 v_a1 = _mm256_broadcast_ss(a1_ptr); a1_ptr += 1;
                v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
                v_acc1x01234567 = _mm256_fmadd_ps(v_a1, v_b0, v_acc1x01234567);
            }
        }

        if (rem >= 8) {

            if constexpr (!zero_mode) {
                v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567, _mm256_loadu_ps(c0_ptr));
                v_acc1x01234567 = _mm256_add_ps(v_acc1x01234567, _mm256_loadu_ps(c1_ptr));
            }

            const __m256 v_min = _mm256_set1_ps(clamp_min);
            const __m256 v_max = _mm256_set1_ps(clamp_max);

            v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
            v_acc1x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc1x01234567, v_max), v_min);

            _mm256_storeu_ps(c0_ptr, v_acc0x01234567);
            _mm256_storeu_ps(c1_ptr, v_acc1x01234567);

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
            // sit at lane 0 of the high half.
            __m128 v_acc0x0123 = _mm256_castps256_ps128(v_acc0x01234567);
            __m128 v_acc1x0123 = _mm256_castps256_ps128(v_acc1x01234567);

            const __m128 v_min = _mm_set1_ps(clamp_min);
            const __m128 v_max = _mm_set1_ps(clamp_max);

            if (rem & 4) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_loadu_ps(c0_ptr));
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_loadu_ps(c1_ptr));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);

                _mm_storeu_ps(c0_ptr, v_acc0x0123);
                _mm_storeu_ps(c1_ptr, v_acc1x0123);

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
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0_ptr))));
                    v_acc1x0123 = _mm_add_ps(v_acc1x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c1_ptr))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ps(_mm_min_ps(v_acc1x0123, v_max), v_min);

                _mm_storel_epi64(reinterpret_cast<__m128i*>(c0_ptr), _mm_castps_si128(v_acc0x0123));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(c1_ptr), _mm_castps_si128(v_acc1x0123));

                c0_ptr += 2;
                c1_ptr += 2;

                v_acc0x0123 = _mm_movehl_ps(v_rest0, v_rest0);
                v_acc1x0123 = _mm_movehl_ps(v_rest1, v_rest1);
            }

            if (rem & 1) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ss(v_acc0x0123, _mm_load_ss(c0_ptr));
                    v_acc1x0123 = _mm_add_ss(v_acc1x0123, _mm_load_ss(c1_ptr));
                }

                v_acc0x0123 = _mm_max_ss(_mm_min_ss(v_acc0x0123, v_max), v_min);
                v_acc1x0123 = _mm_max_ss(_mm_min_ss(v_acc1x0123, v_max), v_min);

                _mm_store_ss(c0_ptr, v_acc0x0123);
                _mm_store_ss(c1_ptr, v_acc1x0123);
            }
        }
    }
}

template <bool zero_mode = false, int seed = 0>
void mma_packed_b_mr1_nr16_mrxncxkc_f32(
    int mr, int nc, int kc, float* NNOPS_RESTRICT C, int ldc,
    const float* NNOPS_RESTRICT A, int lda,
    const float* NNOPS_RESTRICT B,
    float clamp_min, float clamp_max,
    const float* NNOPS_RESTRICT bias = nullptr) noexcept {

    (void)mr;   // one row, so the row count never varies
    (void)ldc;  // one row, so neither the A nor the C row stride applies
    (void)lda;

    float* c0_ptr = C;

    int ldb = static_cast<int>(mma_pack_align(16 * kc * sizeof(float)) / sizeof(float));

    int n = 0;

    // Accumulator initial value. Seed 1 is a per-row bias — the planar route,
    // where a row is an output channel; seed 2 is a per-column one, which is the
    // channel-last route's shape: the rows are pixels and the bias lands on the
    // panel's 16 lanes, so it costs two loads per panel where seeding the tile
    // first costs a full write and read-back of the output. `off` is the lane
    // offset inside the panel. A row past `mr` aliases the last live row in A
    // and C, so the same alias index keeps the row seed in range.
    const auto init = [&](int r, int off) {
        if constexpr (seed == 1) { return _mm256_set1_ps(bias[r]); }
        else if constexpr (seed == 2) { return _mm256_loadu_ps(bias + n + off); }
        else { return _mm256_setzero_ps(); }
    };

    for(; n < nc - 15; n += 16) {
        const float* a0_ptr = A;

        __m256 v_acc0x01234567 = init(0, 0);
        __m256 v_acc0x89abcdef = init(0, 8);

        const float* b_ptr = B;
        for(int k = 0; k < kc; ++k) {
            const __m256 v_b0 = _mm256_loadu_ps(b_ptr + 0 * 8);
            const __m256 v_b1 = _mm256_loadu_ps(b_ptr + 1 * 8);
            b_ptr += 16;

            const __m256 v_a0 = _mm256_broadcast_ss(a0_ptr); a0_ptr += 1;
            v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
            v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
        }

        if constexpr (!zero_mode) {
            v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567, _mm256_loadu_ps(c0_ptr + 0));
            v_acc0x89abcdef = _mm256_add_ps(v_acc0x89abcdef, _mm256_loadu_ps(c0_ptr + 8));
        }
        const __m256 v_min = _mm256_set1_ps(clamp_min);
        const __m256 v_max = _mm256_set1_ps(clamp_max);

        v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);
        v_acc0x89abcdef = _mm256_max_ps(_mm256_min_ps(v_acc0x89abcdef, v_max), v_min);

        _mm256_storeu_ps(c0_ptr + 0, v_acc0x01234567);
        _mm256_storeu_ps(c0_ptr + 8, v_acc0x89abcdef);

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
        const float* a0_ptr = A + 0 * lda;
        __m256 v_acc0x01234567 = init(0, 0);
        __m256 v_acc0x89abcdef = init(0, 8);

        const float* b_ptr = B;
        if (rem > 8) {
            for(int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_loadu_ps(b_ptr + 0 * 8);
                const __m256 v_b1 = _mm256_loadu_ps(b_ptr + 1 * 8);
                b_ptr += 16;

                const __m256 v_a0 = _mm256_broadcast_ss(a0_ptr); a0_ptr += 1;
                v_acc0x01234567 = _mm256_fmadd_ps(v_a0, v_b0, v_acc0x01234567);
                v_acc0x89abcdef = _mm256_fmadd_ps(v_a0, v_b1, v_acc0x89abcdef);
            }
        } else {
            for (int k = 0; k < kc; ++k) {
                const __m256 v_b0 = _mm256_loadu_ps(b_ptr); b_ptr += 16;
                v_acc0x01234567 = _mm256_fmadd_ps(_mm256_broadcast_ss(a0_ptr), v_b0, v_acc0x01234567);
                a0_ptr += 1;
            }
        }

        if (rem >= 8) {

            if constexpr (!zero_mode) {
                v_acc0x01234567 = _mm256_add_ps(v_acc0x01234567, _mm256_loadu_ps(c0_ptr));
            }

            const __m256 v_min = _mm256_set1_ps(clamp_min);
            const __m256 v_max = _mm256_set1_ps(clamp_max);

            v_acc0x01234567 = _mm256_max_ps(_mm256_min_ps(v_acc0x01234567, v_max), v_min);

            _mm256_storeu_ps(c0_ptr, v_acc0x01234567);

            c0_ptr += 8;

            if (rem > 8) {
                v_acc0x01234567 = v_acc0x89abcdef;
            }
            rem -= 8;
        }

        if (rem > 0) {
            // Lane j holds column j, so the 4/2/1 ladder is just a lo/hi split of
            // the same accumulator: after the 4-wide step the leftover 1..3 columns
            // sit at lane 0 of the high half.
            __m128 v_acc0x0123 = _mm256_castps256_ps128(v_acc0x01234567);

            const __m128 v_min = _mm_set1_ps(clamp_min);
            const __m128 v_max = _mm_set1_ps(clamp_max);

            if (rem & 4) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_loadu_ps(c0_ptr));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);

                _mm_storeu_ps(c0_ptr, v_acc0x0123);

                c0_ptr += 4;

                v_acc0x0123 = _mm256_extractf128_ps(v_acc0x01234567, 1);
            }

            if (rem & 2) {
                // The clamp below is 4-wide, so keep the untouched leftovers for
                // the 1-wide step -- lanes 2,3 are what it will store.
                const __m128 v_rest0 = v_acc0x0123;

                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ps(v_acc0x0123, _mm_castsi128_ps(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c0_ptr))));
                }

                v_acc0x0123 = _mm_max_ps(_mm_min_ps(v_acc0x0123, v_max), v_min);

                _mm_storel_epi64(reinterpret_cast<__m128i*>(c0_ptr), _mm_castps_si128(v_acc0x0123));

                c0_ptr += 2;

                v_acc0x0123 = _mm_movehl_ps(v_rest0, v_rest0);
            }

            if (rem & 1) {
                if constexpr (!zero_mode) {
                    v_acc0x0123 = _mm_add_ss(v_acc0x0123, _mm_load_ss(c0_ptr));
                }

                v_acc0x0123 = _mm_max_ss(_mm_min_ss(v_acc0x0123, v_max), v_min);

                _mm_store_ss(c0_ptr, v_acc0x0123);
            }
        }
    }
}

// ---- dispatch tables, consumed by matmul_helper.cpp ---------------------
// Rows are the MR levels {6,4,1}; every kernel takes the runtime panel height,
// so a remainder panel below its own MR runs through the level above it.
template <bool Z>
inline constexpr auto mma_pack_fns_f32 = std::array{
    &mma_packed_ab_mr6_nr16_mrxncxkc_f32<Z>,
    &mma_packed_ab_mr4_nr16_mrxncxkc_f32<Z>,
    &mma_packed_ab_mr1_nr16_mrxncxkc_f32<Z>};

template <bool Z, int Seed = 0>
inline constexpr auto mma_direct_fns_f32 = std::array{
    &mma_packed_b_mr6_nr16_mrxncxkc_f32<Z, Seed>,
    &mma_packed_b_mr4_nr16_mrxncxkc_f32<Z, Seed>,
    &mma_packed_b_mr1_nr16_mrxncxkc_f32<Z, Seed>};

}
