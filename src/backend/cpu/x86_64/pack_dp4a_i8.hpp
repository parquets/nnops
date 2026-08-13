#pragma once
/// @file pack_dp4a_i8.hpp
/// @brief x86_64 int8 (dp4a) pack kernels for GEMM LHS/RHS.
///
/// These kernels repack quantised int8 data into the layout consumed by the
/// int8 dot-product (dp4a / VNNI) micro-kernels. They perform pure data
/// movement — no dot product happens here. `scale` is accepted for signature
/// parity with the f32/f16 packs but ignored (quantisation is handled by the
/// caller). Four int8 elements are packed into each int32 to match the 4-wide
/// dot-product accumulator.
///
/// Two transform families, mirroring pack_f32/pack_f16:
///   - pack_trans_nN (LHS/A): reads N int8 rows (stride ir_step), transposes
///     N×K blocks, packs 4 int8 per int32, writes contiguously.
///   - pack_copy_nN (RHS/B): reads N-column rows, groups each 4-k block into
///     contiguous 4-int8 runs per column.
///
/// Unlike the reference (which uses the AVX512 `_mm*_epi8`/`_mm*_epi32` load and
/// store aliases), these kernels use the portable `_si256`/`_si128` intrinsics
/// so they compile under the baseline AVX2 ISA — the pack step needs no
/// dot-product instruction.
///
/// Reference: nn_compute/src/cpu/kernel/pack/x86_64/pack_dp4a_i8.hpp

#include <cstdint>
#include <immintrin.h>
#include "backend/cpu/common/restrict.hpp"
#include "transpose.hpp"

namespace nnops::backend::cpu::x86_64 {

// =========================================================================
//  LHS Transpose pack (int8)  —  4 int8 packed per int32 (K step = 32)
// =========================================================================

inline void pack_trans_n1_dp4a_i8(void* NNOPS_RESTRICT output,
                                  const void* NNOPS_RESTRICT input,
                                  int ir_step, int K, float scale) noexcept {
    (void)scale;
    int32_t* NNOPS_RESTRICT output_i32 = static_cast<int32_t*>(output);
    const int32_t* NNOPS_RESTRICT input_i32 = static_cast<const int32_t*>(input);

    int k = 0;
    for (; k < K - 3; k += 4) {
        *output_i32++ = input_i32[0];
        input_i32 += 1;
    }

    if (k < K) {
        int8_t* output_i8 = reinterpret_cast<int8_t*>(output_i32);
        const int8_t* input_i8 = reinterpret_cast<const int8_t*>(input_i32);
        output_i8[0] = k + 0 < K ? input_i8[0] : 0;
        output_i8[1] = k + 1 < K ? input_i8[1] : 0;
        output_i8[2] = k + 2 < K ? input_i8[2] : 0;
        output_i8[3] = k + 3 < K ? input_i8[3] : 0;
    }
}

inline void pack_trans_n4_dp4a_i8(void* NNOPS_RESTRICT output,
                                  const void* NNOPS_RESTRICT input,
                                  int ir_step, int K, float scale) noexcept {
    (void)scale;
    int32_t* NNOPS_RESTRICT output_i32 = static_cast<int32_t*>(output);
    const int8_t* NNOPS_RESTRICT input_i8 = static_cast<const int8_t*>(input);

    const int8_t* NNOPS_RESTRICT input_i8_ptr0 = input_i8 + 0 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr1 = input_i8 + 1 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr2 = input_i8 + 2 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr3 = input_i8 + 3 * ir_step;

    int k = 0;
    for (; k < K - 31; k += 32) {
        __m256i v_in0 = _mm256_loadu_si256((const __m256i*)input_i8_ptr0); input_i8_ptr0 += 32;
        __m256i v_in1 = _mm256_loadu_si256((const __m256i*)input_i8_ptr1); input_i8_ptr1 += 32;
        __m256i v_in2 = _mm256_loadu_si256((const __m256i*)input_i8_ptr2); input_i8_ptr2 += 32;
        __m256i v_in3 = _mm256_loadu_si256((const __m256i*)input_i8_ptr3); input_i8_ptr3 += 32;

        __m256 v_in0_f32 = _mm256_castsi256_ps(v_in0);
        __m256 v_in1_f32 = _mm256_castsi256_ps(v_in1);
        __m256 v_in2_f32 = _mm256_castsi256_ps(v_in2);
        __m256 v_in3_f32 = _mm256_castsi256_ps(v_in3);

        transpose_4x8_f32(v_in0_f32, v_in1_f32, v_in2_f32, v_in3_f32);

        v_in0 = _mm256_castps_si256(v_in0_f32);
        v_in1 = _mm256_castps_si256(v_in1_f32);
        v_in2 = _mm256_castps_si256(v_in2_f32);
        v_in3 = _mm256_castps_si256(v_in3_f32);

        _mm256_storeu_si256((__m256i*)(output_i32 + 0 * 8), v_in0);
        _mm256_storeu_si256((__m256i*)(output_i32 + 1 * 8), v_in1);
        _mm256_storeu_si256((__m256i*)(output_i32 + 2 * 8), v_in2);
        _mm256_storeu_si256((__m256i*)(output_i32 + 3 * 8), v_in3);

        output_i32 += (4 * 8);
    }

    for (; k < K - 3; k += 4) {
        output_i32[0] = *(const int32_t*)input_i8_ptr0; input_i8_ptr0 += 4;
        output_i32[1] = *(const int32_t*)input_i8_ptr1; input_i8_ptr1 += 4;
        output_i32[2] = *(const int32_t*)input_i8_ptr2; input_i8_ptr2 += 4;
        output_i32[3] = *(const int32_t*)input_i8_ptr3; input_i8_ptr3 += 4;

        output_i32 += 4;
    }

    if (k < K) {
        int8_t* output_i8 = reinterpret_cast<int8_t*>(output_i32);
        for (int i = 0; i < 4; ++i) {
            output_i8[4 * 0 + i] = k + i < K ? input_i8_ptr0[i] : 0;
            output_i8[4 * 1 + i] = k + i < K ? input_i8_ptr1[i] : 0;
            output_i8[4 * 2 + i] = k + i < K ? input_i8_ptr2[i] : 0;
            output_i8[4 * 3 + i] = k + i < K ? input_i8_ptr3[i] : 0;
        }
    }
}

inline void pack_trans_n6_dp4a_i8(void* NNOPS_RESTRICT output,
                                  const void* NNOPS_RESTRICT input,
                                  int ir_step, int K, float scale) noexcept {
    (void)scale;
    int32_t* NNOPS_RESTRICT output_i32 = static_cast<int32_t*>(output);
    const int8_t* NNOPS_RESTRICT input_i8 = static_cast<const int8_t*>(input);

    const int8_t* NNOPS_RESTRICT input_i8_ptr0 = input_i8 + 0 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr1 = input_i8 + 1 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr2 = input_i8 + 2 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr3 = input_i8 + 3 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr4 = input_i8 + 4 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr5 = input_i8 + 5 * ir_step;

    int k = 0;
    for (; k < K - 31; k += 32) {
        __m256i v_in0 = _mm256_loadu_si256((const __m256i*)input_i8_ptr0);
        __m256i v_in1 = _mm256_loadu_si256((const __m256i*)input_i8_ptr1);
        __m256i v_in2 = _mm256_loadu_si256((const __m256i*)input_i8_ptr2);
        __m256i v_in3 = _mm256_loadu_si256((const __m256i*)input_i8_ptr3);
        __m256i v_in4 = _mm256_loadu_si256((const __m256i*)input_i8_ptr4);
        __m256i v_in5 = _mm256_loadu_si256((const __m256i*)input_i8_ptr5);

        __m256 v_in0_f32 = _mm256_castsi256_ps(v_in0);
        __m256 v_in1_f32 = _mm256_castsi256_ps(v_in1);
        __m256 v_in2_f32 = _mm256_castsi256_ps(v_in2);
        __m256 v_in3_f32 = _mm256_castsi256_ps(v_in3);
        __m256 v_in4_f32 = _mm256_castsi256_ps(v_in4);
        __m256 v_in5_f32 = _mm256_castsi256_ps(v_in5);

        transpose_6x8_f32(v_in0_f32, v_in1_f32, v_in2_f32, v_in3_f32, v_in4_f32, v_in5_f32);

        v_in0 = _mm256_castps_si256(v_in0_f32);
        v_in1 = _mm256_castps_si256(v_in1_f32);
        v_in2 = _mm256_castps_si256(v_in2_f32);
        v_in3 = _mm256_castps_si256(v_in3_f32);
        v_in4 = _mm256_castps_si256(v_in4_f32);
        v_in5 = _mm256_castps_si256(v_in5_f32);

        _mm256_storeu_si256((__m256i*)(output_i32 + 0 * 8), v_in0);
        _mm256_storeu_si256((__m256i*)(output_i32 + 1 * 8), v_in1);
        _mm256_storeu_si256((__m256i*)(output_i32 + 2 * 8), v_in2);
        _mm256_storeu_si256((__m256i*)(output_i32 + 3 * 8), v_in3);
        _mm256_storeu_si256((__m256i*)(output_i32 + 4 * 8), v_in4);
        _mm256_storeu_si256((__m256i*)(output_i32 + 5 * 8), v_in5);

        input_i8_ptr0 += (4 * 8);
        input_i8_ptr1 += (4 * 8);
        input_i8_ptr2 += (4 * 8);
        input_i8_ptr3 += (4 * 8);
        input_i8_ptr4 += (4 * 8);
        input_i8_ptr5 += (4 * 8);
        output_i32 += (6 * 8);
    }

    for (; k < K - 3; k += 4) {
        output_i32[0] = *(const int32_t*)input_i8_ptr0;
        output_i32[1] = *(const int32_t*)input_i8_ptr1;
        output_i32[2] = *(const int32_t*)input_i8_ptr2;
        output_i32[3] = *(const int32_t*)input_i8_ptr3;
        output_i32[4] = *(const int32_t*)input_i8_ptr4;
        output_i32[5] = *(const int32_t*)input_i8_ptr5;

        input_i8_ptr0 += 4;
        input_i8_ptr1 += 4;
        input_i8_ptr2 += 4;
        input_i8_ptr3 += 4;
        input_i8_ptr4 += 4;
        input_i8_ptr5 += 4;
        output_i32 += 6;
    }

    if (k < K) {
        int8_t* output_i8 = reinterpret_cast<int8_t*>(output_i32);
        for (int i = 0; i < 4; ++i) {
            output_i8[4 * 0 + i] = k + i < K ? input_i8_ptr0[i] : 0;
            output_i8[4 * 1 + i] = k + i < K ? input_i8_ptr1[i] : 0;
            output_i8[4 * 2 + i] = k + i < K ? input_i8_ptr2[i] : 0;
            output_i8[4 * 3 + i] = k + i < K ? input_i8_ptr3[i] : 0;
            output_i8[4 * 4 + i] = k + i < K ? input_i8_ptr4[i] : 0;
            output_i8[4 * 5 + i] = k + i < K ? input_i8_ptr5[i] : 0;
        }
    }
}

inline void pack_trans_n8_dp4a_i8(void* NNOPS_RESTRICT output,
                                  const void* NNOPS_RESTRICT input,
                                  int ir_step, int K, float scale) noexcept {
    (void)scale;
    int32_t* NNOPS_RESTRICT output_i32 = static_cast<int32_t*>(output);
    const int8_t* NNOPS_RESTRICT input_i8 = static_cast<const int8_t*>(input);

    const int8_t* NNOPS_RESTRICT input_i8_ptr0 = input_i8 + 0 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr1 = input_i8 + 1 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr2 = input_i8 + 2 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr3 = input_i8 + 3 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr4 = input_i8 + 4 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr5 = input_i8 + 5 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr6 = input_i8 + 6 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr7 = input_i8 + 7 * ir_step;

    int k = 0;
    for (; k < K - 31; k += 32) {
        __m256i v_in0 = _mm256_loadu_si256((const __m256i*)input_i8_ptr0);
        __m256i v_in1 = _mm256_loadu_si256((const __m256i*)input_i8_ptr1);
        __m256i v_in2 = _mm256_loadu_si256((const __m256i*)input_i8_ptr2);
        __m256i v_in3 = _mm256_loadu_si256((const __m256i*)input_i8_ptr3);
        __m256i v_in4 = _mm256_loadu_si256((const __m256i*)input_i8_ptr4);
        __m256i v_in5 = _mm256_loadu_si256((const __m256i*)input_i8_ptr5);
        __m256i v_in6 = _mm256_loadu_si256((const __m256i*)input_i8_ptr6);
        __m256i v_in7 = _mm256_loadu_si256((const __m256i*)input_i8_ptr7);

        __m256 v_in0_f32 = _mm256_castsi256_ps(v_in0);
        __m256 v_in1_f32 = _mm256_castsi256_ps(v_in1);
        __m256 v_in2_f32 = _mm256_castsi256_ps(v_in2);
        __m256 v_in3_f32 = _mm256_castsi256_ps(v_in3);
        __m256 v_in4_f32 = _mm256_castsi256_ps(v_in4);
        __m256 v_in5_f32 = _mm256_castsi256_ps(v_in5);
        __m256 v_in6_f32 = _mm256_castsi256_ps(v_in6);
        __m256 v_in7_f32 = _mm256_castsi256_ps(v_in7);

        transpose_8x8_f32(v_in0_f32, v_in1_f32, v_in2_f32, v_in3_f32,
                          v_in4_f32, v_in5_f32, v_in6_f32, v_in7_f32);

        v_in0 = _mm256_castps_si256(v_in0_f32);
        v_in1 = _mm256_castps_si256(v_in1_f32);
        v_in2 = _mm256_castps_si256(v_in2_f32);
        v_in3 = _mm256_castps_si256(v_in3_f32);
        v_in4 = _mm256_castps_si256(v_in4_f32);
        v_in5 = _mm256_castps_si256(v_in5_f32);
        v_in6 = _mm256_castps_si256(v_in6_f32);
        v_in7 = _mm256_castps_si256(v_in7_f32);

        _mm256_storeu_si256((__m256i*)(output_i32 + 0 * 8), v_in0);
        _mm256_storeu_si256((__m256i*)(output_i32 + 1 * 8), v_in1);
        _mm256_storeu_si256((__m256i*)(output_i32 + 2 * 8), v_in2);
        _mm256_storeu_si256((__m256i*)(output_i32 + 3 * 8), v_in3);
        _mm256_storeu_si256((__m256i*)(output_i32 + 4 * 8), v_in4);
        _mm256_storeu_si256((__m256i*)(output_i32 + 5 * 8), v_in5);
        _mm256_storeu_si256((__m256i*)(output_i32 + 6 * 8), v_in6);
        _mm256_storeu_si256((__m256i*)(output_i32 + 7 * 8), v_in7);

        input_i8_ptr0 += (4 * 8);
        input_i8_ptr1 += (4 * 8);
        input_i8_ptr2 += (4 * 8);
        input_i8_ptr3 += (4 * 8);
        input_i8_ptr4 += (4 * 8);
        input_i8_ptr5 += (4 * 8);
        input_i8_ptr6 += (4 * 8);
        input_i8_ptr7 += (4 * 8);
        output_i32 += (8 * 8);
    }

    for (; k < K - 3; k += 4) {
        output_i32[0] = *(const int32_t*)input_i8_ptr0;
        output_i32[1] = *(const int32_t*)input_i8_ptr1;
        output_i32[2] = *(const int32_t*)input_i8_ptr2;
        output_i32[3] = *(const int32_t*)input_i8_ptr3;
        output_i32[4] = *(const int32_t*)input_i8_ptr4;
        output_i32[5] = *(const int32_t*)input_i8_ptr5;
        output_i32[6] = *(const int32_t*)input_i8_ptr6;
        output_i32[7] = *(const int32_t*)input_i8_ptr7;

        input_i8_ptr0 += 4;
        input_i8_ptr1 += 4;
        input_i8_ptr2 += 4;
        input_i8_ptr3 += 4;
        input_i8_ptr4 += 4;
        input_i8_ptr5 += 4;
        input_i8_ptr6 += 4;
        input_i8_ptr7 += 4;
        output_i32 += 8;
    }

    if (k < K) {
        int8_t* output_i8 = reinterpret_cast<int8_t*>(output_i32);
        for (int i = 0; i < 4; ++i) {
            output_i8[4 * 0 + i] = k + i < K ? input_i8_ptr0[i] : 0;
            output_i8[4 * 1 + i] = k + i < K ? input_i8_ptr1[i] : 0;
            output_i8[4 * 2 + i] = k + i < K ? input_i8_ptr2[i] : 0;
            output_i8[4 * 3 + i] = k + i < K ? input_i8_ptr3[i] : 0;
            output_i8[4 * 4 + i] = k + i < K ? input_i8_ptr4[i] : 0;
            output_i8[4 * 5 + i] = k + i < K ? input_i8_ptr5[i] : 0;
            output_i8[4 * 6 + i] = k + i < K ? input_i8_ptr6[i] : 0;
            output_i8[4 * 7 + i] = k + i < K ? input_i8_ptr7[i] : 0;
        }
    }
}

inline void pack_trans_n12_dp4a_i8(void* NNOPS_RESTRICT output,
                                   const void* NNOPS_RESTRICT input,
                                   int ir_step, int K, float scale) noexcept {
    (void)scale;
    int32_t* NNOPS_RESTRICT output_i32 = static_cast<int32_t*>(output);
    const int8_t* NNOPS_RESTRICT input_i8 = static_cast<const int8_t*>(input);

    const int8_t* NNOPS_RESTRICT input_i8_ptr0  = input_i8 + 0 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr1  = input_i8 + 1 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr2  = input_i8 + 2 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr3  = input_i8 + 3 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr4  = input_i8 + 4 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr5  = input_i8 + 5 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr6  = input_i8 + 6 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr7  = input_i8 + 7 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr8  = input_i8 + 8 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr9  = input_i8 + 9 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptra  = input_i8 + 10 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptrb  = input_i8 + 11 * ir_step;

    int k = 0;
    for (; k < K - 31; k += 32) {
        __m256i v_in0  = _mm256_loadu_si256((const __m256i*)input_i8_ptr0);
        __m256i v_in1  = _mm256_loadu_si256((const __m256i*)input_i8_ptr1);
        __m256i v_in2  = _mm256_loadu_si256((const __m256i*)input_i8_ptr2);
        __m256i v_in3  = _mm256_loadu_si256((const __m256i*)input_i8_ptr3);
        __m256i v_in4  = _mm256_loadu_si256((const __m256i*)input_i8_ptr4);
        __m256i v_in5  = _mm256_loadu_si256((const __m256i*)input_i8_ptr5);
        __m256i v_in6  = _mm256_loadu_si256((const __m256i*)input_i8_ptr6);
        __m256i v_in7  = _mm256_loadu_si256((const __m256i*)input_i8_ptr7);
        __m256i v_in8  = _mm256_loadu_si256((const __m256i*)input_i8_ptr8);
        __m256i v_in9  = _mm256_loadu_si256((const __m256i*)input_i8_ptr9);
        __m256i v_ina  = _mm256_loadu_si256((const __m256i*)input_i8_ptra);
        __m256i v_inb  = _mm256_loadu_si256((const __m256i*)input_i8_ptrb);

        __m256 v_in0_f32  = _mm256_castsi256_ps(v_in0);
        __m256 v_in1_f32  = _mm256_castsi256_ps(v_in1);
        __m256 v_in2_f32  = _mm256_castsi256_ps(v_in2);
        __m256 v_in3_f32  = _mm256_castsi256_ps(v_in3);
        __m256 v_in4_f32  = _mm256_castsi256_ps(v_in4);
        __m256 v_in5_f32  = _mm256_castsi256_ps(v_in5);
        __m256 v_in6_f32  = _mm256_castsi256_ps(v_in6);
        __m256 v_in7_f32  = _mm256_castsi256_ps(v_in7);
        __m256 v_in8_f32  = _mm256_castsi256_ps(v_in8);
        __m256 v_in9_f32  = _mm256_castsi256_ps(v_in9);
        __m256 v_ina_f32  = _mm256_castsi256_ps(v_ina);
        __m256 v_inb_f32  = _mm256_castsi256_ps(v_inb);

        transpose_12x8_f32(
            v_in0_f32, v_in1_f32, v_in2_f32, v_in3_f32,
            v_in4_f32, v_in5_f32, v_in6_f32, v_in7_f32,
            v_in8_f32, v_in9_f32, v_ina_f32, v_inb_f32);

        v_in0  = _mm256_castps_si256(v_in0_f32);
        v_in1  = _mm256_castps_si256(v_in1_f32);
        v_in2  = _mm256_castps_si256(v_in2_f32);
        v_in3  = _mm256_castps_si256(v_in3_f32);
        v_in4  = _mm256_castps_si256(v_in4_f32);
        v_in5  = _mm256_castps_si256(v_in5_f32);
        v_in6  = _mm256_castps_si256(v_in6_f32);
        v_in7  = _mm256_castps_si256(v_in7_f32);
        v_in8  = _mm256_castps_si256(v_in8_f32);
        v_in9  = _mm256_castps_si256(v_in9_f32);
        v_ina  = _mm256_castps_si256(v_ina_f32);
        v_inb  = _mm256_castps_si256(v_inb_f32);

        _mm256_storeu_si256((__m256i*)(output_i32 + 0 * 8),  v_in0);
        _mm256_storeu_si256((__m256i*)(output_i32 + 1 * 8),  v_in1);
        _mm256_storeu_si256((__m256i*)(output_i32 + 2 * 8),  v_in2);
        _mm256_storeu_si256((__m256i*)(output_i32 + 3 * 8),  v_in3);
        _mm256_storeu_si256((__m256i*)(output_i32 + 4 * 8),  v_in4);
        _mm256_storeu_si256((__m256i*)(output_i32 + 5 * 8),  v_in5);
        _mm256_storeu_si256((__m256i*)(output_i32 + 6 * 8),  v_in6);
        _mm256_storeu_si256((__m256i*)(output_i32 + 7 * 8),  v_in7);
        _mm256_storeu_si256((__m256i*)(output_i32 + 8 * 8),  v_in8);
        _mm256_storeu_si256((__m256i*)(output_i32 + 9 * 8),  v_in9);
        _mm256_storeu_si256((__m256i*)(output_i32 + 10 * 8), v_ina);
        _mm256_storeu_si256((__m256i*)(output_i32 + 11 * 8), v_inb);

        input_i8_ptr0  += (4 * 8);
        input_i8_ptr1  += (4 * 8);
        input_i8_ptr2  += (4 * 8);
        input_i8_ptr3  += (4 * 8);
        input_i8_ptr4  += (4 * 8);
        input_i8_ptr5  += (4 * 8);
        input_i8_ptr6  += (4 * 8);
        input_i8_ptr7  += (4 * 8);
        input_i8_ptr8  += (4 * 8);
        input_i8_ptr9  += (4 * 8);
        input_i8_ptra  += (4 * 8);
        input_i8_ptrb  += (4 * 8);
        output_i32 += (12 * 8);
    }

    for (; k < K - 3; k += 4) {
        output_i32[0]  = *(const int32_t*)input_i8_ptr0;
        output_i32[1]  = *(const int32_t*)input_i8_ptr1;
        output_i32[2]  = *(const int32_t*)input_i8_ptr2;
        output_i32[3]  = *(const int32_t*)input_i8_ptr3;
        output_i32[4]  = *(const int32_t*)input_i8_ptr4;
        output_i32[5]  = *(const int32_t*)input_i8_ptr5;
        output_i32[6]  = *(const int32_t*)input_i8_ptr6;
        output_i32[7]  = *(const int32_t*)input_i8_ptr7;
        output_i32[8]  = *(const int32_t*)input_i8_ptr8;
        output_i32[9]  = *(const int32_t*)input_i8_ptr9;
        output_i32[10] = *(const int32_t*)input_i8_ptra;
        output_i32[11] = *(const int32_t*)input_i8_ptrb;

        input_i8_ptr0  += 4;
        input_i8_ptr1  += 4;
        input_i8_ptr2  += 4;
        input_i8_ptr3  += 4;
        input_i8_ptr4  += 4;
        input_i8_ptr5  += 4;
        input_i8_ptr6  += 4;
        input_i8_ptr7  += 4;
        input_i8_ptr8  += 4;
        input_i8_ptr9  += 4;
        input_i8_ptra  += 4;
        input_i8_ptrb  += 4;
        output_i32 += 12;
    }

    if (k < K) {
        int8_t* output_i8 = reinterpret_cast<int8_t*>(output_i32);
        for (int i = 0; i < 4; ++i) {
            output_i8[4 * 0 + i]  = k + i < K ? input_i8_ptr0[i] : 0;
            output_i8[4 * 1 + i]  = k + i < K ? input_i8_ptr1[i] : 0;
            output_i8[4 * 2 + i]  = k + i < K ? input_i8_ptr2[i] : 0;
            output_i8[4 * 3 + i]  = k + i < K ? input_i8_ptr3[i] : 0;
            output_i8[4 * 4 + i]  = k + i < K ? input_i8_ptr4[i] : 0;
            output_i8[4 * 5 + i]  = k + i < K ? input_i8_ptr5[i] : 0;
            output_i8[4 * 6 + i]  = k + i < K ? input_i8_ptr6[i] : 0;
            output_i8[4 * 7 + i]  = k + i < K ? input_i8_ptr7[i] : 0;
            output_i8[4 * 8 + i]  = k + i < K ? input_i8_ptr8[i] : 0;
            output_i8[4 * 9 + i]  = k + i < K ? input_i8_ptr9[i] : 0;
            output_i8[4 * 10 + i] = k + i < K ? input_i8_ptra[i] : 0;
            output_i8[4 * 11 + i] = k + i < K ? input_i8_ptrb[i] : 0;
        }
    }
}

inline void pack_trans_n16_dp4a_i8(void* NNOPS_RESTRICT output,
                                   const void* NNOPS_RESTRICT input,
                                   int ir_step, int K, float scale) noexcept {
    (void)scale;
    int32_t* NNOPS_RESTRICT output_i32 = static_cast<int32_t*>(output);
    const int8_t* NNOPS_RESTRICT input_i8 = static_cast<const int8_t*>(input);

    const int8_t* NNOPS_RESTRICT input_i8_ptr0  = input_i8 + 0 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr1  = input_i8 + 1 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr2  = input_i8 + 2 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr3  = input_i8 + 3 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr4  = input_i8 + 4 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr5  = input_i8 + 5 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr6  = input_i8 + 6 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr7  = input_i8 + 7 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr8  = input_i8 + 8 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr9  = input_i8 + 9 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptra  = input_i8 + 10 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptrb  = input_i8 + 11 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptrc  = input_i8 + 12 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptrd  = input_i8 + 13 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptre  = input_i8 + 14 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptrf  = input_i8 + 15 * ir_step;

    int k = 0;
    for (; k < K - 31; k += 32) {
        __m256i v_in0  = _mm256_loadu_si256((const __m256i*)input_i8_ptr0);
        __m256i v_in1  = _mm256_loadu_si256((const __m256i*)input_i8_ptr1);
        __m256i v_in2  = _mm256_loadu_si256((const __m256i*)input_i8_ptr2);
        __m256i v_in3  = _mm256_loadu_si256((const __m256i*)input_i8_ptr3);
        __m256i v_in4  = _mm256_loadu_si256((const __m256i*)input_i8_ptr4);
        __m256i v_in5  = _mm256_loadu_si256((const __m256i*)input_i8_ptr5);
        __m256i v_in6  = _mm256_loadu_si256((const __m256i*)input_i8_ptr6);
        __m256i v_in7  = _mm256_loadu_si256((const __m256i*)input_i8_ptr7);
        __m256i v_in8  = _mm256_loadu_si256((const __m256i*)input_i8_ptr8);
        __m256i v_in9  = _mm256_loadu_si256((const __m256i*)input_i8_ptr9);
        __m256i v_ina  = _mm256_loadu_si256((const __m256i*)input_i8_ptra);
        __m256i v_inb  = _mm256_loadu_si256((const __m256i*)input_i8_ptrb);
        __m256i v_inc  = _mm256_loadu_si256((const __m256i*)input_i8_ptrc);
        __m256i v_ind  = _mm256_loadu_si256((const __m256i*)input_i8_ptrd);
        __m256i v_ine  = _mm256_loadu_si256((const __m256i*)input_i8_ptre);
        __m256i v_inf  = _mm256_loadu_si256((const __m256i*)input_i8_ptrf);

        __m256 v_in0_f32  = _mm256_castsi256_ps(v_in0);
        __m256 v_in1_f32  = _mm256_castsi256_ps(v_in1);
        __m256 v_in2_f32  = _mm256_castsi256_ps(v_in2);
        __m256 v_in3_f32  = _mm256_castsi256_ps(v_in3);
        __m256 v_in4_f32  = _mm256_castsi256_ps(v_in4);
        __m256 v_in5_f32  = _mm256_castsi256_ps(v_in5);
        __m256 v_in6_f32  = _mm256_castsi256_ps(v_in6);
        __m256 v_in7_f32  = _mm256_castsi256_ps(v_in7);
        __m256 v_in8_f32  = _mm256_castsi256_ps(v_in8);
        __m256 v_in9_f32  = _mm256_castsi256_ps(v_in9);
        __m256 v_ina_f32  = _mm256_castsi256_ps(v_ina);
        __m256 v_inb_f32  = _mm256_castsi256_ps(v_inb);
        __m256 v_inc_f32  = _mm256_castsi256_ps(v_inc);
        __m256 v_ind_f32  = _mm256_castsi256_ps(v_ind);
        __m256 v_ine_f32  = _mm256_castsi256_ps(v_ine);
        __m256 v_inf_f32  = _mm256_castsi256_ps(v_inf);

        transpose_16x8_f32(
            v_in0_f32, v_in1_f32, v_in2_f32, v_in3_f32, v_in4_f32, v_in5_f32, v_in6_f32, v_in7_f32,
            v_in8_f32, v_in9_f32, v_ina_f32, v_inb_f32, v_inc_f32, v_ind_f32, v_ine_f32, v_inf_f32);

        v_in0  = _mm256_castps_si256(v_in0_f32);
        v_in1  = _mm256_castps_si256(v_in1_f32);
        v_in2  = _mm256_castps_si256(v_in2_f32);
        v_in3  = _mm256_castps_si256(v_in3_f32);
        v_in4  = _mm256_castps_si256(v_in4_f32);
        v_in5  = _mm256_castps_si256(v_in5_f32);
        v_in6  = _mm256_castps_si256(v_in6_f32);
        v_in7  = _mm256_castps_si256(v_in7_f32);
        v_in8  = _mm256_castps_si256(v_in8_f32);
        v_in9  = _mm256_castps_si256(v_in9_f32);
        v_ina  = _mm256_castps_si256(v_ina_f32);
        v_inb  = _mm256_castps_si256(v_inb_f32);
        v_inc  = _mm256_castps_si256(v_inc_f32);
        v_ind  = _mm256_castps_si256(v_ind_f32);
        v_ine  = _mm256_castps_si256(v_ine_f32);
        v_inf  = _mm256_castps_si256(v_inf_f32);

        _mm256_storeu_si256((__m256i*)(output_i32 + 0 * 8),  v_in0);
        _mm256_storeu_si256((__m256i*)(output_i32 + 1 * 8),  v_in1);
        _mm256_storeu_si256((__m256i*)(output_i32 + 2 * 8),  v_in2);
        _mm256_storeu_si256((__m256i*)(output_i32 + 3 * 8),  v_in3);
        _mm256_storeu_si256((__m256i*)(output_i32 + 4 * 8),  v_in4);
        _mm256_storeu_si256((__m256i*)(output_i32 + 5 * 8),  v_in5);
        _mm256_storeu_si256((__m256i*)(output_i32 + 6 * 8),  v_in6);
        _mm256_storeu_si256((__m256i*)(output_i32 + 7 * 8),  v_in7);
        _mm256_storeu_si256((__m256i*)(output_i32 + 8 * 8),  v_in8);
        _mm256_storeu_si256((__m256i*)(output_i32 + 9 * 8),  v_in9);
        _mm256_storeu_si256((__m256i*)(output_i32 + 10 * 8), v_ina);
        _mm256_storeu_si256((__m256i*)(output_i32 + 11 * 8), v_inb);
        _mm256_storeu_si256((__m256i*)(output_i32 + 12 * 8), v_inc);
        _mm256_storeu_si256((__m256i*)(output_i32 + 13 * 8), v_ind);
        _mm256_storeu_si256((__m256i*)(output_i32 + 14 * 8), v_ine);
        _mm256_storeu_si256((__m256i*)(output_i32 + 15 * 8), v_inf);

        input_i8_ptr0  += (4 * 8);
        input_i8_ptr1  += (4 * 8);
        input_i8_ptr2  += (4 * 8);
        input_i8_ptr3  += (4 * 8);
        input_i8_ptr4  += (4 * 8);
        input_i8_ptr5  += (4 * 8);
        input_i8_ptr6  += (4 * 8);
        input_i8_ptr7  += (4 * 8);
        input_i8_ptr8  += (4 * 8);
        input_i8_ptr9  += (4 * 8);
        input_i8_ptra  += (4 * 8);
        input_i8_ptrb  += (4 * 8);
        input_i8_ptrc  += (4 * 8);
        input_i8_ptrd  += (4 * 8);
        input_i8_ptre  += (4 * 8);
        input_i8_ptrf  += (4 * 8);
        output_i32 += (16 * 8);
    }

    for (; k < K - 3; k += 4) {
        output_i32[0]  = *(const int32_t*)input_i8_ptr0;
        output_i32[1]  = *(const int32_t*)input_i8_ptr1;
        output_i32[2]  = *(const int32_t*)input_i8_ptr2;
        output_i32[3]  = *(const int32_t*)input_i8_ptr3;
        output_i32[4]  = *(const int32_t*)input_i8_ptr4;
        output_i32[5]  = *(const int32_t*)input_i8_ptr5;
        output_i32[6]  = *(const int32_t*)input_i8_ptr6;
        output_i32[7]  = *(const int32_t*)input_i8_ptr7;
        output_i32[8]  = *(const int32_t*)input_i8_ptr8;
        output_i32[9]  = *(const int32_t*)input_i8_ptr9;
        output_i32[10] = *(const int32_t*)input_i8_ptra;
        output_i32[11] = *(const int32_t*)input_i8_ptrb;
        output_i32[12] = *(const int32_t*)input_i8_ptrc;
        output_i32[13] = *(const int32_t*)input_i8_ptrd;
        output_i32[14] = *(const int32_t*)input_i8_ptre;
        output_i32[15] = *(const int32_t*)input_i8_ptrf;

        input_i8_ptr0  += 4;
        input_i8_ptr1  += 4;
        input_i8_ptr2  += 4;
        input_i8_ptr3  += 4;
        input_i8_ptr4  += 4;
        input_i8_ptr5  += 4;
        input_i8_ptr6  += 4;
        input_i8_ptr7  += 4;
        input_i8_ptr8  += 4;
        input_i8_ptr9  += 4;
        input_i8_ptra  += 4;
        input_i8_ptrb  += 4;
        input_i8_ptrc  += 4;
        input_i8_ptrd  += 4;
        input_i8_ptre  += 4;
        input_i8_ptrf  += 4;
        output_i32 += 16;
    }

    if (k < K) {
        int8_t* output_i8 = reinterpret_cast<int8_t*>(output_i32);
        for (int i = 0; i < 4; ++i) {
            output_i8[4 * 0 + i]  = k + i < K ? input_i8_ptr0[i] : 0;
            output_i8[4 * 1 + i]  = k + i < K ? input_i8_ptr1[i] : 0;
            output_i8[4 * 2 + i]  = k + i < K ? input_i8_ptr2[i] : 0;
            output_i8[4 * 3 + i]  = k + i < K ? input_i8_ptr3[i] : 0;
            output_i8[4 * 4 + i]  = k + i < K ? input_i8_ptr4[i] : 0;
            output_i8[4 * 5 + i]  = k + i < K ? input_i8_ptr5[i] : 0;
            output_i8[4 * 6 + i]  = k + i < K ? input_i8_ptr6[i] : 0;
            output_i8[4 * 7 + i]  = k + i < K ? input_i8_ptr7[i] : 0;
            output_i8[4 * 8 + i]  = k + i < K ? input_i8_ptr8[i] : 0;
            output_i8[4 * 9 + i]  = k + i < K ? input_i8_ptr9[i] : 0;
            output_i8[4 * 10 + i] = k + i < K ? input_i8_ptra[i] : 0;
            output_i8[4 * 11 + i] = k + i < K ? input_i8_ptrb[i] : 0;
            output_i8[4 * 12 + i] = k + i < K ? input_i8_ptrc[i] : 0;
            output_i8[4 * 13 + i] = k + i < K ? input_i8_ptrd[i] : 0;
            output_i8[4 * 14 + i] = k + i < K ? input_i8_ptre[i] : 0;
            output_i8[4 * 15 + i] = k + i < K ? input_i8_ptrf[i] : 0;
        }
    }
}

// =========================================================================
//  RHS Copy pack (int8)  —  contiguous-row copy, 4 int8 grouped per column
// =========================================================================

inline void pack_copy_n1_dp4a_i8(void* NNOPS_RESTRICT output,
                                 const void* NNOPS_RESTRICT input,
                                 int ir_step, int K, float scale) noexcept {
    (void)scale;
    int8_t* NNOPS_RESTRICT output_i8 = static_cast<int8_t*>(output);
    const int8_t* NNOPS_RESTRICT input_i8 = static_cast<const int8_t*>(input);

    int k = 0;
    for (; k < K - 3; k += 4) {
        output_i8[0] = input_i8[0 * ir_step];
        output_i8[1] = input_i8[1 * ir_step];
        output_i8[2] = input_i8[2 * ir_step];
        output_i8[3] = input_i8[3 * ir_step];
        output_i8 += 4;
        input_i8 += 4 * ir_step;
    }
    if (k < K) {
        output_i8[0] = (k + 0) < K ? input_i8[0 * ir_step] : 0;
        output_i8[1] = (k + 1) < K ? input_i8[1 * ir_step] : 0;
        output_i8[2] = (k + 2) < K ? input_i8[2 * ir_step] : 0;
        output_i8[3] = (k + 3) < K ? input_i8[3 * ir_step] : 0;
    }
}

inline void pack_copy_n4_dp4a_i8(void* NNOPS_RESTRICT output,
                                 const void* NNOPS_RESTRICT input,
                                 int ir_step, int K, float scale) noexcept {
    (void)scale;
    int8_t* NNOPS_RESTRICT output_i8 = static_cast<int8_t*>(output);
    const int8_t* NNOPS_RESTRICT input_i8 = static_cast<const int8_t*>(input);

    int k = 0;
    for (; k < K - 3; k += 4) {
        for (int x = 0; x < 4; ++x) {
            output_i8[0] = input_i8[0 * ir_step + x];
            output_i8[1] = input_i8[1 * ir_step + x];
            output_i8[2] = input_i8[2 * ir_step + x];
            output_i8[3] = input_i8[3 * ir_step + x];
            output_i8 += 4;
        }
        input_i8 += 4 * ir_step;
    }
    if (k < K) {
        for (int x = 0; x < 4; ++x) {
            output_i8[0] = k + 0 < K ? input_i8[0 * ir_step + x] : 0;
            output_i8[1] = k + 1 < K ? input_i8[1 * ir_step + x] : 0;
            output_i8[2] = k + 2 < K ? input_i8[2 * ir_step + x] : 0;
            output_i8[3] = k + 3 < K ? input_i8[3 * ir_step + x] : 0;
            output_i8 += 4;
        }
    }
}

inline void pack_copy_n6_dp4a_i8(void* NNOPS_RESTRICT output,
                                 const void* NNOPS_RESTRICT input,
                                 int ir_step, int K, float scale) noexcept {
    (void)scale;
    int8_t* NNOPS_RESTRICT output_i8 = static_cast<int8_t*>(output);
    const int8_t* NNOPS_RESTRICT input_i8 = static_cast<const int8_t*>(input);

    int k = 0;
    for (; k < K - 3; k += 4) {
        for (int x = 0; x < 6; ++x) {
            output_i8[0] = input_i8[0 * ir_step + x];
            output_i8[1] = input_i8[1 * ir_step + x];
            output_i8[2] = input_i8[2 * ir_step + x];
            output_i8[3] = input_i8[3 * ir_step + x];
            output_i8 += 4;
        }
        input_i8 += 4 * ir_step;
    }
    if (k < K) {
        for (int x = 0; x < 6; ++x) {
            output_i8[0] = k + 0 < K ? input_i8[0 * ir_step + x] : 0;
            output_i8[1] = k + 1 < K ? input_i8[1 * ir_step + x] : 0;
            output_i8[2] = k + 2 < K ? input_i8[2 * ir_step + x] : 0;
            output_i8[3] = k + 3 < K ? input_i8[3 * ir_step + x] : 0;
            output_i8 += 4;
        }
    }
}

inline void pack_copy_n8_dp4a_i8(void* NNOPS_RESTRICT output,
                                 const void* NNOPS_RESTRICT input,
                                 int ir_step, int K, float scale) noexcept {
    (void)scale;
    int8_t* NNOPS_RESTRICT output_i8 = static_cast<int8_t*>(output);
    const int8_t* NNOPS_RESTRICT input_i8 = static_cast<const int8_t*>(input);

    int k = 0;
    for (; k < K - 3; k += 4) {
        for (int x = 0; x < 8; ++x) {
            output_i8[0] = input_i8[0 * ir_step + x];
            output_i8[1] = input_i8[1 * ir_step + x];
            output_i8[2] = input_i8[2 * ir_step + x];
            output_i8[3] = input_i8[3 * ir_step + x];
            output_i8 += 4;
        }
        input_i8 += 4 * ir_step;
    }
    if (k < K) {
        for (int x = 0; x < 8; ++x) {
            output_i8[0] = k + 0 < K ? input_i8[0 * ir_step + x] : 0;
            output_i8[1] = k + 1 < K ? input_i8[1 * ir_step + x] : 0;
            output_i8[2] = k + 2 < K ? input_i8[2 * ir_step + x] : 0;
            output_i8[3] = k + 3 < K ? input_i8[3 * ir_step + x] : 0;
            output_i8 += 4;
        }
    }
}

inline void pack_copy_n12_dp4a_i8(void* NNOPS_RESTRICT output,
                                  const void* NNOPS_RESTRICT input,
                                  int ir_step, int K, float scale) noexcept {
    (void)scale;
    int8_t* NNOPS_RESTRICT output_i8 = static_cast<int8_t*>(output);
    const int8_t* NNOPS_RESTRICT input_i8 = static_cast<const int8_t*>(input);

    int k = 0;
    for (; k < K - 3; k += 4) {
        for (int x = 0; x < 12; ++x) {
            output_i8[0] = input_i8[0 * ir_step + x];
            output_i8[1] = input_i8[1 * ir_step + x];
            output_i8[2] = input_i8[2 * ir_step + x];
            output_i8[3] = input_i8[3 * ir_step + x];
            output_i8 += 4;
        }
        input_i8 += 4 * ir_step;
    }
    if (k < K) {
        for (int x = 0; x < 12; ++x) {
            output_i8[0] = k + 0 < K ? input_i8[0 * ir_step + x] : 0;
            output_i8[1] = k + 1 < K ? input_i8[1 * ir_step + x] : 0;
            output_i8[2] = k + 2 < K ? input_i8[2 * ir_step + x] : 0;
            output_i8[3] = k + 3 < K ? input_i8[3 * ir_step + x] : 0;
            output_i8 += 4;
        }
    }
}

inline void pack_copy_n16_dp4a_i8(void* NNOPS_RESTRICT output,
                                  const void* NNOPS_RESTRICT input,
                                  int ir_step, int K, float scale) noexcept {
    (void)scale;
    int8_t* NNOPS_RESTRICT output_i8 = static_cast<int8_t*>(output);
    const int8_t* NNOPS_RESTRICT input_i8 = static_cast<const int8_t*>(input);

    int k = 0;
    for (; k < K - 3; k += 4) {
        __m128i v_in0 = _mm_loadu_si128((const __m128i*)(input_i8 + 0 * ir_step));
        __m128i v_in1 = _mm_loadu_si128((const __m128i*)(input_i8 + 1 * ir_step));
        __m128i v_in2 = _mm_loadu_si128((const __m128i*)(input_i8 + 2 * ir_step));
        __m128i v_in3 = _mm_loadu_si128((const __m128i*)(input_i8 + 3 * ir_step));

        transpose_4x16_i8(v_in0, v_in1, v_in2, v_in3);

        _mm_storeu_si128((__m128i*)(output_i8 + 0 * 16), v_in0);
        _mm_storeu_si128((__m128i*)(output_i8 + 1 * 16), v_in1);
        _mm_storeu_si128((__m128i*)(output_i8 + 2 * 16), v_in2);
        _mm_storeu_si128((__m128i*)(output_i8 + 3 * 16), v_in3);

        input_i8 += 4 * ir_step;
        output_i8 += (4 * 16);
    }
    if (k < K) {
        for (int x = 0; x < 16; ++x) {
            output_i8[0] = k + 0 < K ? input_i8[0 * ir_step + x] : 0;
            output_i8[1] = k + 1 < K ? input_i8[1 * ir_step + x] : 0;
            output_i8[2] = k + 2 < K ? input_i8[2 * ir_step + x] : 0;
            output_i8[3] = k + 3 < K ? input_i8[3 * ir_step + x] : 0;
            output_i8 += 4;
        }
    }
}

}  // namespace nnops::backend::cpu::x86_64
