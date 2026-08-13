#pragma once
/// @file pack_dp4a_i8.hpp
/// @brief AArch64 NEON int8 (dp4a) pack kernels for GEMM LHS/RHS.
///
/// These kernels repack quantised int8 data into the layout consumed by the
/// int8 dot-product (SDOT/UDOT) micro-kernels. They perform pure data movement
/// — no dot product happens here. `scale` is accepted for signature parity with
/// the f32/f16 packs but ignored (quantisation is handled by the caller). Four
/// int8 elements are packed into each int32 to match the 4-wide dot-product
/// accumulator.
///
/// Two transform families, mirroring pack_f32/pack_f16:
///   - pack_trans_nN (LHS/A): reads N int8 rows (stride ir_step), transposes
///     N×K blocks, packs 4 int8 per int32, writes contiguously.
///   - pack_copy_nN (RHS/B): reads N-column rows, groups each 4-k block into
///     contiguous 4-int8 runs per column.
///
/// Reference: nn_compute/src/cpu/kernel/pack/aarch64/pack_dp4a_i8.hpp

#include <arm_neon.h>
#include <cstdint>
#include "backend/cpu/common/restrict.hpp"
#include "transpose.hpp"

namespace nnops::backend::cpu::aarch64 {

// =========================================================================
//  LHS Transpose pack (int8)  —  4 int8 packed per int32 (K step = 16)
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
    for (; k < K - 15; k += 16) {
        int32x4x4_t v_in;
        v_in.val[0] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptr0));
        v_in.val[1] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptr1));
        v_in.val[2] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptr2));
        v_in.val[3] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptr3));

        vst4q_s32(output_i32, v_in);

        input_i8_ptr0 += 16;
        input_i8_ptr1 += 16;
        input_i8_ptr2 += 16;
        input_i8_ptr3 += 16;
        output_i32 += 4 * 4;
    }
    for (; k < K - 3; k += 4) {
        output_i32[0] = *(const int32_t*)input_i8_ptr0;
        output_i32[1] = *(const int32_t*)input_i8_ptr1;
        output_i32[2] = *(const int32_t*)input_i8_ptr2;
        output_i32[3] = *(const int32_t*)input_i8_ptr3;

        input_i8_ptr0 += 4;
        input_i8_ptr1 += 4;
        input_i8_ptr2 += 4;
        input_i8_ptr3 += 4;
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
    for (; k < K - 15; k += 16) {
        int32x4x4_t v_in0, v_in1;
        v_in0.val[0] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptr0));
        v_in0.val[1] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptr1));
        v_in0.val[2] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptr2));
        v_in0.val[3] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptr3));
        v_in1.val[0] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptr4));
        v_in1.val[1] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptr5));
        v_in1.val[2] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptr6));
        v_in1.val[3] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptr7));

        transpose_8x4_i32(
            v_in0.val[0], v_in0.val[1], v_in0.val[2], v_in0.val[3],
            v_in1.val[0], v_in1.val[1], v_in1.val[2], v_in1.val[3]);

        vst1q_s32(output_i32 + 0 * 4, v_in0.val[0]);
        vst1q_s32(output_i32 + 1 * 4, v_in0.val[1]);
        vst1q_s32(output_i32 + 2 * 4, v_in0.val[2]);
        vst1q_s32(output_i32 + 3 * 4, v_in0.val[3]);
        vst1q_s32(output_i32 + 4 * 4, v_in1.val[0]);
        vst1q_s32(output_i32 + 5 * 4, v_in1.val[1]);
        vst1q_s32(output_i32 + 6 * 4, v_in1.val[2]);
        vst1q_s32(output_i32 + 7 * 4, v_in1.val[3]);

        input_i8_ptr0 += 16;
        input_i8_ptr1 += 16;
        input_i8_ptr2 += 16;
        input_i8_ptr3 += 16;
        input_i8_ptr4 += 16;
        input_i8_ptr5 += 16;
        input_i8_ptr6 += 16;
        input_i8_ptr7 += 16;

        output_i32 += 8 * 4;
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

    const int8_t* NNOPS_RESTRICT input_i8_ptr0 = input_i8 + 0 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr1 = input_i8 + 1 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr2 = input_i8 + 2 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr3 = input_i8 + 3 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr4 = input_i8 + 4 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr5 = input_i8 + 5 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr6 = input_i8 + 6 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr7 = input_i8 + 7 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr8 = input_i8 + 8 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptr9 = input_i8 + 9 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptra = input_i8 + 10 * ir_step;
    const int8_t* NNOPS_RESTRICT input_i8_ptrb = input_i8 + 11 * ir_step;

    int k = 0;
    for (; k < K - 15; k += 16) {
        int32x4x4_t v_in0, v_in1, v_in2;
        v_in0.val[0] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptr0));
        v_in0.val[1] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptr1));
        v_in0.val[2] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptr2));
        v_in0.val[3] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptr3));
        v_in1.val[0] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptr4));
        v_in1.val[1] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptr5));
        v_in1.val[2] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptr6));
        v_in1.val[3] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptr7));
        v_in2.val[0] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptr8));
        v_in2.val[1] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptr9));
        v_in2.val[2] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptra));
        v_in2.val[3] = vreinterpretq_s32_s8(vld1q_s8(input_i8_ptrb));

        transpose_12x4_i32(
            v_in0.val[0], v_in0.val[1], v_in0.val[2], v_in0.val[3],
            v_in1.val[0], v_in1.val[1], v_in1.val[2], v_in1.val[3],
            v_in2.val[0], v_in2.val[1], v_in2.val[2], v_in2.val[3]);

        vst1q_s32(output_i32 + 0 * 4,  v_in0.val[0]);
        vst1q_s32(output_i32 + 1 * 4,  v_in0.val[1]);
        vst1q_s32(output_i32 + 2 * 4,  v_in0.val[2]);
        vst1q_s32(output_i32 + 3 * 4,  v_in0.val[3]);
        vst1q_s32(output_i32 + 4 * 4,  v_in1.val[0]);
        vst1q_s32(output_i32 + 5 * 4,  v_in1.val[1]);
        vst1q_s32(output_i32 + 6 * 4,  v_in1.val[2]);
        vst1q_s32(output_i32 + 7 * 4,  v_in1.val[3]);
        vst1q_s32(output_i32 + 8 * 4,  v_in2.val[0]);
        vst1q_s32(output_i32 + 9 * 4,  v_in2.val[1]);
        vst1q_s32(output_i32 + 10 * 4, v_in2.val[2]);
        vst1q_s32(output_i32 + 11 * 4, v_in2.val[3]);

        input_i8_ptr0 += 16;
        input_i8_ptr1 += 16;
        input_i8_ptr2 += 16;
        input_i8_ptr3 += 16;
        input_i8_ptr4 += 16;
        input_i8_ptr5 += 16;
        input_i8_ptr6 += 16;
        input_i8_ptr7 += 16;
        input_i8_ptr8 += 16;
        input_i8_ptr9 += 16;
        input_i8_ptra += 16;
        input_i8_ptrb += 16;

        output_i32 += 12 * 4;
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
        output_i32[8] = *(const int32_t*)input_i8_ptr8;
        output_i32[9] = *(const int32_t*)input_i8_ptr9;
        output_i32[10] = *(const int32_t*)input_i8_ptra;
        output_i32[11] = *(const int32_t*)input_i8_ptrb;

        input_i8_ptr0 += 4;
        input_i8_ptr1 += 4;
        input_i8_ptr2 += 4;
        input_i8_ptr3 += 4;
        input_i8_ptr4 += 4;
        input_i8_ptr5 += 4;
        input_i8_ptr6 += 4;
        input_i8_ptr7 += 4;
        input_i8_ptr8 += 4;
        input_i8_ptr9 += 4;
        input_i8_ptra += 4;
        input_i8_ptrb += 4;
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

// =========================================================================
//  RHS Copy pack (int8)  —  contiguous-row copy, 4 int8 grouped per column
// =========================================================================

inline void pack_copy_n1_dp4a_i8(void* NNOPS_RESTRICT output,
                                 const void* NNOPS_RESTRICT input,
                                 int ir_step, int K, float scale) noexcept {
    (void)scale;
    int8_t* NNOPS_RESTRICT output_i8 = reinterpret_cast<int8_t*>(output);
    const int8_t* NNOPS_RESTRICT input_i8 = reinterpret_cast<const int8_t*>(input);

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
        output_i8[0] = k + 0 < K ? input_i8[0 * ir_step] : 0;
        output_i8[1] = k + 1 < K ? input_i8[1 * ir_step] : 0;
        output_i8[2] = k + 2 < K ? input_i8[2 * ir_step] : 0;
        output_i8[3] = k + 3 < K ? input_i8[3 * ir_step] : 0;
    }
}

inline void pack_copy_n4_dp4a_i8(void* NNOPS_RESTRICT output,
                                 const void* NNOPS_RESTRICT input,
                                 int ir_step, int K, float scale) noexcept {
    (void)scale;
    int8_t* NNOPS_RESTRICT output_i8 = reinterpret_cast<int8_t*>(output);
    const int8_t* NNOPS_RESTRICT input_i8 = reinterpret_cast<const int8_t*>(input);

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

inline void pack_copy_n8_dp4a_i8(void* NNOPS_RESTRICT output,
                                 const void* NNOPS_RESTRICT input,
                                 int ir_step, int K, float scale) noexcept {
    (void)scale;
    int8_t* NNOPS_RESTRICT output_i8 = reinterpret_cast<int8_t*>(output);
    const int8_t* NNOPS_RESTRICT input_i8 = reinterpret_cast<const int8_t*>(input);

    int k = 0;
    for (; k < K - 3; k += 4) {
        int8x8x4_t v_in;
        v_in.val[0] = vld1_s8(input_i8 + 0 * ir_step);
        v_in.val[1] = vld1_s8(input_i8 + 1 * ir_step);
        v_in.val[2] = vld1_s8(input_i8 + 2 * ir_step);
        v_in.val[3] = vld1_s8(input_i8 + 3 * ir_step);

        vst4_s8(output_i8, v_in);
        output_i8 += 4 * 8;
        input_i8 += 4 * ir_step;
    }

    if (k < K) {
        int8x8x4_t v_in;
        v_in.val[0] = k + 0 < K ? vld1_s8(input_i8 + 0 * ir_step) : vdup_n_s8(0);
        v_in.val[1] = k + 1 < K ? vld1_s8(input_i8 + 1 * ir_step) : vdup_n_s8(0);
        v_in.val[2] = k + 2 < K ? vld1_s8(input_i8 + 2 * ir_step) : vdup_n_s8(0);
        v_in.val[3] = k + 3 < K ? vld1_s8(input_i8 + 3 * ir_step) : vdup_n_s8(0);
        vst4_s8(output_i8, v_in);
    }
}

inline void pack_copy_n12_dp4a_i8(void* NNOPS_RESTRICT output,
                                  const void* NNOPS_RESTRICT input,
                                  int ir_step, int K, float scale) noexcept {
    (void)scale;
    int8_t* NNOPS_RESTRICT output_i8 = reinterpret_cast<int8_t*>(output);
    const int8_t* NNOPS_RESTRICT input_i8 = reinterpret_cast<const int8_t*>(input);

    int k = 0;
    for (; k < K - 3; k += 4) {
        int8x8x4_t v_in;
        v_in.val[0] = vld1_s8(input_i8 + 0 * ir_step);
        v_in.val[1] = vld1_s8(input_i8 + 1 * ir_step);
        v_in.val[2] = vld1_s8(input_i8 + 2 * ir_step);
        v_in.val[3] = vld1_s8(input_i8 + 3 * ir_step);

        vst4_s8(output_i8, v_in);
        output_i8 += 4 * 8;

        for (int x = 0; x < 4; ++x) {
            output_i8[0] = input_i8[0 * ir_step + x + 8];
            output_i8[1] = input_i8[1 * ir_step + x + 8];
            output_i8[2] = input_i8[2 * ir_step + x + 8];
            output_i8[3] = input_i8[3 * ir_step + x + 8];
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
    int8_t* NNOPS_RESTRICT output_i8 = reinterpret_cast<int8_t*>(output);
    const int8_t* NNOPS_RESTRICT input_i8 = reinterpret_cast<const int8_t*>(input);

    int k = 0;
    for (; k < K - 3; k += 4) {
        int8x16x4_t v_in;
        v_in.val[0] = vld1q_s8(input_i8 + 0 * ir_step);
        v_in.val[1] = vld1q_s8(input_i8 + 1 * ir_step);
        v_in.val[2] = vld1q_s8(input_i8 + 2 * ir_step);
        v_in.val[3] = vld1q_s8(input_i8 + 3 * ir_step);

        vst4q_s8(output_i8, v_in);

        output_i8 += 4 * 16;
        input_i8 += 4 * ir_step;
    }

    if (k < K) {
        int8x16x4_t v_in;
        v_in.val[0] = k + 0 < K ? vld1q_s8(input_i8 + 0 * ir_step) : vdupq_n_s8(0);
        v_in.val[1] = k + 1 < K ? vld1q_s8(input_i8 + 1 * ir_step) : vdupq_n_s8(0);
        v_in.val[2] = k + 2 < K ? vld1q_s8(input_i8 + 2 * ir_step) : vdupq_n_s8(0);
        v_in.val[3] = k + 3 < K ? vld1q_s8(input_i8 + 3 * ir_step) : vdupq_n_s8(0);
        vst4q_s8(output_i8, v_in);
    }
}

}  // namespace nnops::backend::cpu::aarch64
