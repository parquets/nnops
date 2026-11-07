#pragma once
/// @file mma_qblock.hpp
/// @brief Fused block-quant dot kernels for the small-M MatMul route (NEON).
///
/// The packed-B route decodes a Q4_1 block into a dense panel and then reads
/// that panel back in the MMA. At M=1 the panel is consumed by exactly one
/// row, so the round trip is the whole call. These kernels skip the panel: a
/// block is decoded into registers and multiplied straight into the
/// accumulator, read once (20 B) and never written back.
///
/// A block policy (`Q4Block`) supplies the stride, the width, the M ceilings
/// the driver admits on, the panel widths it walks, and the two decode entry
/// points — registers for the k body, a scratch array for the trailing partial
/// block — so a second block format costs one policy and no kernel.
///
/// Two physical B layouts, each with its natural loop shape:
///
///  * trans ([N][K], the Linear-weight layout): one n row at a time, its k
///    blocks swept linearly. Four accumulator vectors per A row, MR rows, so a
///    decode is amortized over all MR rows.
///  * copy ([K][N]): one 32-wide n block at a time. A block is exactly one
///    quant block per k row, and the outer-product form needs eight
///    accumulators per row, which caps MR at 2.
///
/// Both sweep the whole K in one call, so the clamp lands once, on the final
/// value. The sum order is the kernels' own (four chunk partials reduced at
/// the end for trans), so a tile must not be shared with the packed route.
///
/// NEON's four lanes per vector are why this is not the x86 shape: a 32-value
/// block is eight vectors, not four, so one A row costs four accumulators (two
/// interleaved chains each) and the dot kernel walks four rows at a time.
///
/// @p MR is the exact number of live rows: a partial M panel instantiates its
/// own width, so no row aliases a duplicate and no per-row guard is needed.
/// @p AC is the activation/output element type — f32 reads through, f16 converts
/// on the load and rounds once at the store, so a w4a16 call decodes a block
/// into f32 registers instead of packing an f16 panel first.

#include <arm_neon.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "backend/cpu/common/common_utils.hpp"
#include "nnops/core/data_type.hpp"
#include "nnops/detail/q4_1.hpp"

namespace nnops::backend::cpu::aarch64 {

// =========================================================================
//  Block policies
// =========================================================================
//
/// One 32-value quant block: its width and byte size, the dtype it names, the
/// panel widths, and the decode pair.

struct Q4Block {
    static constexpr int block_elems = q4_1_block_elems;
    static constexpr int block_bytes = q4_1_block_bytes;
    static constexpr DataType dtype = DataType::q4_1;

    /// Rows per kernel call. The dot kernel holds four accumulators and eight
    /// decoded vectors per row; the outer product holds eight per row.
    static constexpr int panel_trans = 4;
    static constexpr int panel_copy = 2;

    /// Per-layout M ceilings. There is no ARM hardware behind these numbers —
    /// they copy the x86 gate (where the dot kernel reads 2.3x the packed
    /// route at M=16 and the outer product 1.3x at M=8), because the A/B that
    /// set them was measured on x86 and the win is structural: at these M the
    /// packed route's panel write and read are the call. Kept at the f32 rather
    /// than the f16 width for both activation types until there is a device to
    /// measure on. The kernels themselves are checked by scalar emulation on
    /// x86 (lane mapping, tails, alpha/beta, clamp); nothing has run on ARM.
    static constexpr int fused_max_m_trans = 16;
    static constexpr int fused_max_m_trans_f16 = 16;
    static constexpr int fused_max_m_copy = 8;

    /// Decode one block into eight 4-lane vectors: values 4j..4j+3 land in
    /// v[j]. Same arithmetic as decode_q4_1_block, one rounding fewer.
    static inline void decode_v(const uint8_t* p, float32x4_t v[8]) noexcept {
        uint16_t d_bits, m_bits;
        std::memcpy(&d_bits, p + 0, sizeof(d_bits));
        std::memcpy(&m_bits, p + 2, sizeof(m_bits));

        const float32x4_t vd = vdupq_n_f32(half_to_float(half_from_bits(d_bits)));
        const float32x4_t vm = vdupq_n_f32(half_to_float(half_from_bits(m_bits)));

        const uint8x16_t qs = vld1q_u8(p + 4);
        const uint8x16_t lo = vandq_u8(qs, vdupq_n_u8(0x0F));
        const uint8x16_t hi = vshrq_n_u8(qs, 4);

        const auto conv = [&](uint8x8_t nib, float32x4_t& r0, float32x4_t& r1) {
            const uint16x8_t w = vmovl_u8(nib);
            r0 = vfmaq_f32(vm, vcvtq_f32_u32(vmovl_u16(vget_low_u16(w))), vd);
            r1 = vfmaq_f32(vm, vcvtq_f32_u32(vmovl_u16(vget_high_u16(w))), vd);
        };
        conv(vget_low_u8(lo),  v[0], v[1]);   // values  0..7
        conv(vget_high_u8(lo), v[2], v[3]);   // values  8..15
        conv(vget_low_u8(hi),  v[4], v[5]);   // values 16..23
        conv(vget_high_u8(hi), v[6], v[7]);   // values 24..31
    }

    /// The trailing partial block, through the portable decoder: at most 31
    /// values, so the vector path above never sees it.
    template <class T>
    static inline void decode(const uint8_t* p, T* out) noexcept {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                      "decode: T must be float or half");
        float vals[q4_1_block_elems];
        q4_1_unpack_block(p, vals);
        for (int i = 0; i < q4_1_block_elems; ++i) {
            if constexpr (std::is_same_v<T, float>) {
                out[i] = vals[i];
            } else {
                out[i] = float_to_half(vals[i]);
            }
        }
    }
};

static_assert(Q4Block::block_elems == 32,
              "the fused kernels assume one 32-value block per k step in the "
              "copy layout");

// =========================================================================
//  Activation access — f32 reads through, f16 converts on the fly
// =========================================================================
//
/// The product accumulates in f32 registers either way; only the operand load
/// and the final store differ, so an f16 activation costs one convert per
/// vector instead of a dense f32 copy of A.

inline float32x4_t qblock_load4(const float* p) noexcept { return vld1q_f32(p); }
inline float32x4_t qblock_load4(const half* p) noexcept {
    return vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t*>(p)));
}

inline void qblock_store4(float* p, float32x4_t v) noexcept { vst1q_f32(p, v); }
inline void qblock_store4(half* p, float32x4_t v) noexcept {
    vst1_f16(reinterpret_cast<float16_t*>(p), vcvt_f16_f32(v));
}

inline float qblock_load1(const float* p) noexcept { return *p; }
inline float qblock_load1(const half* p) noexcept { return half_to_float(*p); }

inline void qblock_store1(float* p, float v) noexcept { *p = v; }
inline void qblock_store1(half* p, float v) noexcept { *p = float_to_half(v); }

/// One trans row's k step: its eight A vectors against the eight decoded ones,
/// folded into four interleaved chains.
#define NNOPS_QBLOCK_DOT_ROW(A0, A1, A2, A3, ROW)                         \
    do {                                                                  \
        const AC* a = A + static_cast<int64_t>(ROW) * lda + k;            \
        A0 = vfmaq_f32(A0, b0, qblock_load4(a + 0));                      \
        A0 = vfmaq_f32(A0, b4, qblock_load4(a + 16));                     \
        A1 = vfmaq_f32(A1, b1, qblock_load4(a + 4));                      \
        A1 = vfmaq_f32(A1, b5, qblock_load4(a + 20));                     \
        A2 = vfmaq_f32(A2, b2, qblock_load4(a + 8));                      \
        A2 = vfmaq_f32(A2, b6, qblock_load4(a + 24));                     \
        A3 = vfmaq_f32(A3, b3, qblock_load4(a + 12));                     \
        A3 = vfmaq_f32(A3, b7, qblock_load4(a + 28));                     \
    } while (0)

// =========================================================================
//  trans: B physically [N][K]
// =========================================================================
//
/// Dot-product form: one n row at a time, applied to every live A row.
///
/// @param C C[m0][n_start]; already offset by the caller.
/// @param A A[m0][0], row stride @p lda.
/// @param B first block of the batch element's [N][K] stream.
template <class BQ, class AC, int MR, bool ZERO>
void mma_qblock_trans(int nc, int kc, AC* NNOPS_RESTRICT C, int ldc,
                      const AC* NNOPS_RESTRICT A, int lda,
                      const uint8_t* NNOPS_RESTRICT B, int64_t row_blocks,
                      int n_start, float alpha, float clamp_min,
                      float clamp_max) noexcept {
    static_assert(MR >= 1 && MR <= BQ::panel_trans,
                  "trans qblock kernel: MR in [1, panel_trans]");

    const int64_t row_bytes = row_blocks * static_cast<int64_t>(BQ::block_bytes);
    const uint8_t* src = B + static_cast<int64_t>(n_start) * row_bytes;

    const int nblk = kc / BQ::block_elems;

    for (int n = 0; n < nc; ++n, src += row_bytes) {
        float32x4_t q00 = vdupq_n_f32(0.0f), q01 = q00, q02 = q00, q03 = q00;
        float32x4_t q10 = q00, q11 = q00, q12 = q00, q13 = q00;
        float32x4_t q20 = q00, q21 = q00, q22 = q00, q23 = q00;
        float32x4_t q30 = q00, q31 = q00, q32 = q00, q33 = q00;

        int k = 0;
        for (int b = 0; b < nblk; ++b, k += BQ::block_elems) {
            float32x4_t bv[8];
            BQ::decode_v(src + static_cast<int64_t>(b) * BQ::block_bytes, bv);
            const float32x4_t b0 = bv[0], b1 = bv[1], b2 = bv[2], b3 = bv[3];
            const float32x4_t b4 = bv[4], b5 = bv[5], b6 = bv[6], b7 = bv[7];

            NNOPS_QBLOCK_DOT_ROW(q00, q01, q02, q03, 0);
            if constexpr (MR > 1) { NNOPS_QBLOCK_DOT_ROW(q10, q11, q12, q13, 1); }
            if constexpr (MR > 2) { NNOPS_QBLOCK_DOT_ROW(q20, q21, q22, q23, 2); }
            if constexpr (MR > 3) { NNOPS_QBLOCK_DOT_ROW(q30, q31, q32, q33, 3); }
        }

        // Trailing partial block: scalar, at most 31 values.
        float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
        if (k < kc) {
            float vals[BQ::block_elems];
            BQ::template decode<float>(
                src + static_cast<int64_t>(nblk) * BQ::block_bytes, vals);
            const int rem = kc - k;
            for (int i = 0; i < rem; ++i) {
                const int kk = k + i;
                const float bv = vals[i];
                t0 += bv * qblock_load1(A + kk);
                if constexpr (MR > 1) { t1 += bv * qblock_load1(A + static_cast<int64_t>(lda) + kk); }
                if constexpr (MR > 2) { t2 += bv * qblock_load1(A + 2 * static_cast<int64_t>(lda) + kk); }
                if constexpr (MR > 3) { t3 += bv * qblock_load1(A + 3 * static_cast<int64_t>(lda) + kk); }
            }
        }

        AC* c = C + n;
        auto finish = [&](int row, float32x4_t a0, float32x4_t a1,
                          float32x4_t a2, float32x4_t a3, float tail) {
            AC* cp = c + static_cast<int64_t>(row) * ldc;
            float v = vaddvq_f32(a0) + vaddvq_f32(a1) + vaddvq_f32(a2) +
                      vaddvq_f32(a3) + tail;
            if (alpha != 1.0f) { v *= alpha; }
            if constexpr (!ZERO) { v += qblock_load1(cp); }
            qblock_store1(cp, std::min(std::max(v, clamp_min), clamp_max));
        };

        finish(0, q00, q01, q02, q03, t0);
        if constexpr (MR > 1) { finish(1, q10, q11, q12, q13, t1); }
        if constexpr (MR > 2) { finish(2, q20, q21, q22, q23, t2); }
        if constexpr (MR > 3) { finish(3, q30, q31, q32, q33, t3); }
    }
}

#undef NNOPS_QBLOCK_DOT_ROW

// =========================================================================
//  copy: B physically [K][N]
// =========================================================================
//
/// Outer-product form: one 32-wide B block at a time, one A scalar per row.
///
/// @param n_start first column of the tile; must be a multiple of 32, so each
///                decoded block is one panel.
template <class BQ, class AC, int MR, bool ZERO>
void mma_qblock_copy(int nc, int kc, AC* NNOPS_RESTRICT C, int ldc,
                     const AC* NNOPS_RESTRICT A, int lda,
                     const uint8_t* NNOPS_RESTRICT B, int64_t row_blocks,
                     int n_start, float alpha, float clamp_min,
                     float clamp_max) noexcept {
    static_assert(MR == 1 || MR == 2, "copy qblock kernel: MR in {1, 2}");

    const int64_t row_bytes = row_blocks * static_cast<int64_t>(BQ::block_bytes);
    const int64_t blk0 = n_start / BQ::block_elems;

    const float32x4_t vlo = vdupq_n_f32(clamp_min);
    const float32x4_t vhi = vdupq_n_f32(clamp_max);
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    const float32x4_t valpha = vdupq_n_f32(alpha);
    const bool scaled = (alpha != 1.0f);

    for (int n = 0; n < nc; n += BQ::block_elems) {
        const int width = std::min(BQ::block_elems, nc - n);
        const int64_t blk = blk0 + (n / BQ::block_elems);

        float32x4_t r0[8] = {vzero, vzero, vzero, vzero,
                             vzero, vzero, vzero, vzero};
        float32x4_t r1[8] = {vzero, vzero, vzero, vzero,
                             vzero, vzero, vzero, vzero};

        for (int k = 0; k < kc; ++k) {
            const uint8_t* src = B + static_cast<int64_t>(k) * row_bytes
                               + blk * static_cast<int64_t>(BQ::block_bytes);
            float32x4_t bv[8];
            BQ::decode_v(src, bv);

            const float32x4_t va = vdupq_n_f32(qblock_load1(A + k));
            for (int j = 0; j < 8; ++j) { r0[j] = vfmaq_f32(r0[j], va, bv[j]); }

            if constexpr (MR > 1) {
                const float32x4_t vb = vdupq_n_f32(
                    qblock_load1(A + static_cast<int64_t>(lda) + k));
                for (int j = 0; j < 8; ++j) { r1[j] = vfmaq_f32(r1[j], vb, bv[j]); }
            }
        }

        auto emit = [&](int row, float32x4_t* q) {
            AC* cp = C + static_cast<int64_t>(row) * ldc + n;
            if (width == BQ::block_elems) {
                for (int j = 0; j < 8; ++j) {
                    float32x4_t v = scaled ? vmulq_f32(q[j], valpha) : q[j];
                    if constexpr (!ZERO) {
                        v = vaddq_f32(v, qblock_load4(cp + 4 * j));
                    }
                    v = vminq_f32(vmaxq_f32(v, vlo), vhi);
                    qblock_store4(cp + 4 * j, v);
                }
            } else {
                float tmp[BQ::block_elems];
                float* tp = tmp;
                vst1q_f32(tp + 0,  q[0]); vst1q_f32(tp + 4,  q[1]);
                vst1q_f32(tp + 8,  q[2]); vst1q_f32(tp + 12, q[3]);
                vst1q_f32(tp + 16, q[4]); vst1q_f32(tp + 20, q[5]);
                vst1q_f32(tp + 24, q[6]); vst1q_f32(tp + 28, q[7]);
                for (int i = 0; i < width; ++i) {
                    float v = tmp[i];
                    if (scaled) { v *= alpha; }
                    if constexpr (!ZERO) { v += qblock_load1(cp + i); }
                    qblock_store1(cp + i, std::min(std::max(v, clamp_min), clamp_max));
                }
            }
        };

        emit(0, r0);
        if constexpr (MR > 1) { emit(1, r1); }
    }
}

}  // namespace nnops::backend::cpu::aarch64
