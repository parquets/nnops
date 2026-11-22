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
#include "nnops/detail/bf16.hpp"
#include "nnops/detail/q4.hpp"

namespace nnops::backend::cpu::aarch64 {

// =========================================================================
//  Block policies
// =========================================================================
//
/// One 32-value quant block: its width and byte size, the dtype it names, the
/// panel widths, and the decode pair.

template <DataType FMT> struct Q4Block {
    static constexpr int block_elems = static_cast<int>(q4_block_elems);
    static constexpr int block_bytes = static_cast<int>(data_type_block_bytes(FMT));
    static constexpr DataType dtype = FMT;

    /// Rows per kernel call. The dot kernel holds four accumulators for MR <= 4
    /// and two for wider panels (the decoded block is 8 vectors, so eight rows
    /// at two chains is 24 live vectors — the register file's limit). The outer
    /// product holds eight accumulators per row.
    static constexpr int panel_trans = 8;
    static constexpr int panel_copy = 2;

    /// Per-layout M ceilings. The trans figure is measured on an M4: the eight-row
    /// panel amortizes one decode over 8 rows instead of 4, which roughly doubles
    /// the fused route's throughput and pushes its crossover with the packed plan
    /// out to here. The copy limit keeps the x86 figure (eight accumulators per
    /// row leaves no room for a wider panel).
    static constexpr int fused_max_m_trans = 32;
    static constexpr int fused_max_m_trans_f16 = 32;
    static constexpr int fused_max_m_copy = 8;

    /// Decode one block into eight 4-lane vectors: values 4j..4j+3 land in
    /// v[j]. Same arithmetic as decode_q4_1_block, one rounding fewer.
    static inline void decode_v(const uint8_t* p, float32x4_t v[8]) noexcept
    {
        constexpr bool with_min = FMT == DataType::q4_1;
        uint16_t d_bits = 0, m_bits = 0;
        std::memcpy(&d_bits, p + 0, sizeof(d_bits));
        if constexpr (with_min) {
            std::memcpy(&m_bits, p + 2, sizeof(m_bits));
        }

        const float32x4_t vd = vdupq_n_f32(half_to_float(half_from_bits(d_bits)));
        const float32x4_t vm = vdupq_n_f32(with_min ? half_to_float(half_from_bits(m_bits)) : 0.0f);
        const int16x8_t vbias = vdupq_n_s16(with_min ? 0 : -8);

        const uint8x16_t qs = vld1q_u8(p + (with_min ? 4 : 2));
        const uint8x16_t lo = vandq_u8(qs, vdupq_n_u8(0x0F));
        const uint8x16_t hi = vshrq_n_u8(qs, 4);

        const auto conv = [&](uint8x8_t nib, float32x4_t& r0, float32x4_t& r1) {
            const int16x8_t q = vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(nib)), vbias);
            r0 = vfmaq_f32(vm, vcvtq_f32_s32(vmovl_s16(vget_low_s16(q))), vd);
            r1 = vfmaq_f32(vm, vcvtq_f32_s32(vmovl_s16(vget_high_s16(q))), vd);
        };
        conv(vget_low_u8(lo), v[0], v[1]);  // values  0..7
        conv(vget_high_u8(lo), v[2], v[3]); // values  8..15
        conv(vget_low_u8(hi), v[4], v[5]);  // values 16..23
        conv(vget_high_u8(hi), v[6], v[7]); // values 24..31
    }

    /// The trailing partial block, through the portable decoder: at most 31
    /// values, so the vector path above never sees it.
    template <class T> static inline void decode(const uint8_t* p, T* out) noexcept
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, half> ||
                          std::is_same_v<T, bfloat16>,
                      "decode: T must be float, half or bfloat16");
        float vals[q4_block_elems];
        q4_unpack_block(p, vals, FMT);
        for (int i = 0; i < q4_block_elems; ++i) {
            if constexpr (std::is_same_v<T, float>) {
                out[i] = vals[i];
            } else if constexpr (std::is_same_v<T, half>) {
                out[i] = float_to_half(vals[i]);
            } else {
                out[i] = float_to_bfloat16(vals[i]);
            }
        }
    }
};

using Q4_1Block = Q4Block<DataType::q4_1>;
using Q4_0Block = Q4Block<DataType::q4_0>;

static_assert(Q4Block<DataType::q4_1>::block_elems == 32,
              "the fused kernels assume one 32-value block per k step in the "
              "copy layout");

// =========================================================================
//  Activation access — f32 reads through, f16/bf16 convert on the fly
// =========================================================================
//
/// The product accumulates in f32 registers either way; only the operand load
/// and the final store differ, so a narrow activation costs one convert per
/// vector instead of a dense f32 copy of A.

inline float32x4_t qblock_load4(const float* p) noexcept { return vld1q_f32(p); }
inline float32x4_t qblock_load4(const half* p) noexcept
{
    return vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t*>(p)));
}
/// bf16 is the top half of f32: widening is a left shift.
inline float32x4_t qblock_load4(const bfloat16* p) noexcept
{
    const uint16x4_t u = vld1_u16(reinterpret_cast<const uint16_t*>(p));
    return vreinterpretq_f32_u32(vshll_n_u16(u, 16));
}

inline void qblock_store4(float* p, float32x4_t v) noexcept { vst1q_f32(p, v); }
inline void qblock_store4(half* p, float32x4_t v) noexcept
{
    vst1_f16(reinterpret_cast<float16_t*>(p), vcvt_f16_f32(v));
}
/// Round-to-nearest-even, the same narrowing `pack_bf16.hpp` uses.
inline void qblock_store4(bfloat16* p, float32x4_t v) noexcept
{
    uint32x4_t u = vreinterpretq_u32_f32(v);
    u = vaddq_u32(u,
                  vaddq_u32(vdupq_n_u32(0x7FFFu), vandq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(1u))));
    vst1_u16(reinterpret_cast<uint16_t*>(p), vshrn_n_u32(u, 16));
}

inline float qblock_load1(const float* p) noexcept { return *p; }
inline float qblock_load1(const half* p) noexcept { return half_to_float(*p); }
inline float qblock_load1(const bfloat16* p) noexcept { return *p; }

inline void qblock_store1(float* p, float v) noexcept { *p = v; }
inline void qblock_store1(half* p, float v) noexcept { *p = float_to_half(v); }
inline void qblock_store1(bfloat16* p, float v) noexcept { *p = float_to_bfloat16(v); }

/// One trans row's k step: its eight A vectors against the eight decoded ones,
/// folded into four interleaved chains.
#define NNOPS_QBLOCK_DOT_ROW(A0, A1, A2, A3, ROW)              \
    do {                                                       \
        const AC* a = A + static_cast<int64_t>(ROW) * lda + k; \
        A0 = vfmaq_f32(A0, b0, qblock_load4(a + 0));           \
        A0 = vfmaq_f32(A0, b4, qblock_load4(a + 16));          \
        A1 = vfmaq_f32(A1, b1, qblock_load4(a + 4));           \
        A1 = vfmaq_f32(A1, b5, qblock_load4(a + 20));          \
        A2 = vfmaq_f32(A2, b2, qblock_load4(a + 8));           \
        A2 = vfmaq_f32(A2, b6, qblock_load4(a + 24));          \
        A3 = vfmaq_f32(A3, b3, qblock_load4(a + 12));          \
        A3 = vfmaq_f32(A3, b7, qblock_load4(a + 28));          \
    } while (0)

/// The same step in two chains: half the accumulators, so a panel can hold
/// eight rows without spilling. Sixteen independent chains across the panel
/// still hide the four-deep dependency.
#define NNOPS_QBLOCK_DOT2_ROW(A0, A1, ROW)                     \
    do {                                                       \
        const AC* a = A + static_cast<int64_t>(ROW) * lda + k; \
        A0 = vfmaq_f32(A0, b0, qblock_load4(a + 0));           \
        A0 = vfmaq_f32(A0, b2, qblock_load4(a + 8));           \
        A0 = vfmaq_f32(A0, b4, qblock_load4(a + 16));          \
        A0 = vfmaq_f32(A0, b6, qblock_load4(a + 24));          \
        A1 = vfmaq_f32(A1, b1, qblock_load4(a + 4));           \
        A1 = vfmaq_f32(A1, b3, qblock_load4(a + 12));          \
        A1 = vfmaq_f32(A1, b5, qblock_load4(a + 20));          \
        A1 = vfmaq_f32(A1, b7, qblock_load4(a + 28));          \
    } while (0)

// =========================================================================
//  trans: B physically [N][K]
// =========================================================================
//
/// f16 specialization of the trans dot. B decodes into four f16 vectors (8
/// lanes each), A loads as f16 with no widening, and the product accumulates in
/// f16 — one FMA per 8 lanes instead of two f32 FMAs per 8. Every 256 k the
/// partial sums fold into f32, so the whole-K value keeps the f32 route's
/// tolerance; the tail and the alpha/beta/clamp epilogue stay f32.
template <class BQ, int MR, bool ZERO>
void mma_qblock_trans_f16(int nc, int kc, half* NNOPS_RESTRICT C, int ldc,
                          const half* NNOPS_RESTRICT A, int lda,
                          const uint8_t* NNOPS_RESTRICT B, int64_t row_blocks, int n_start,
                          float alpha, float clamp_min, float clamp_max) noexcept
{
    constexpr int kFoldBlocks = 8; // 256 k values per f32 fold
    static_assert(MR >= 1 && MR <= BQ::panel_trans, "trans qblock kernel: MR in [1, panel_trans]");

    const int64_t row_bytes = row_blocks * static_cast<int64_t>(BQ::block_bytes);
    const uint8_t* src = B + static_cast<int64_t>(n_start) * row_bytes;
    const int nblk = kc / BQ::block_elems;

#define NNOPS_QBLOCK_F16_DOT4(A0, A1, A2, A3, ROW)                                       \
    do {                                                                                 \
        const half* a = A + static_cast<int64_t>(ROW) * lda + k;                         \
        A0 = vfmaq_f16(A0, b0, vld1q_f16(reinterpret_cast<const float16_t*>(a + 0)));    \
        A1 = vfmaq_f16(A1, b1, vld1q_f16(reinterpret_cast<const float16_t*>(a + 8)));    \
        A2 = vfmaq_f16(A2, b2, vld1q_f16(reinterpret_cast<const float16_t*>(a + 16)));   \
        A3 = vfmaq_f16(A3, b3, vld1q_f16(reinterpret_cast<const float16_t*>(a + 24)));   \
    } while (0)

#define NNOPS_QBLOCK_F16_DOT2(A0, A1, ROW)                                               \
    do {                                                                                 \
        const half* a = A + static_cast<int64_t>(ROW) * lda + k;                         \
        A0 = vfmaq_f16(A0, b0, vld1q_f16(reinterpret_cast<const float16_t*>(a + 0)));    \
        A0 = vfmaq_f16(A0, b2, vld1q_f16(reinterpret_cast<const float16_t*>(a + 16)));   \
        A1 = vfmaq_f16(A1, b1, vld1q_f16(reinterpret_cast<const float16_t*>(a + 8)));    \
        A1 = vfmaq_f16(A1, b3, vld1q_f16(reinterpret_cast<const float16_t*>(a + 24)));   \
    } while (0)

    for (int n = 0; n < nc; ++n, src += row_bytes) {
        const float16x8_t zero = vdupq_n_f16(0.0f);
        float16x8_t q00 = zero, q01 = zero, q02 = zero, q03 = zero;
        float16x8_t q10 = zero, q11 = zero, q12 = zero, q13 = zero;
        float16x8_t q20 = zero, q21 = zero, q22 = zero, q23 = zero;
        float16x8_t q30 = zero, q31 = zero, q32 = zero, q33 = zero;
        float16x8_t q40 = zero, q41 = zero, q42 = zero, q43 = zero;
        float16x8_t q50 = zero, q51 = zero, q52 = zero, q53 = zero;
        float16x8_t q60 = zero, q61 = zero, q62 = zero, q63 = zero;
        float16x8_t q70 = zero, q71 = zero, q72 = zero, q73 = zero;

        float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
        float t4 = 0.0f, t5 = 0.0f, t6 = 0.0f, t7 = 0.0f;

        auto fold_reset = [&](float& t, float16x8_t& a0, float16x8_t& a1, float16x8_t& a2,
                              float16x8_t& a3) {
            const float32x4_t s0 =
                vaddq_f32(vcvt_f32_f16(vget_low_f16(a0)), vcvt_high_f32_f16(a0));
            const float32x4_t s1 =
                vaddq_f32(vcvt_f32_f16(vget_low_f16(a1)), vcvt_high_f32_f16(a1));
            const float32x4_t s2 =
                vaddq_f32(vcvt_f32_f16(vget_low_f16(a2)), vcvt_high_f32_f16(a2));
            const float32x4_t s3 =
                vaddq_f32(vcvt_f32_f16(vget_low_f16(a3)), vcvt_high_f32_f16(a3));
            t += vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
            a0 = a1 = a2 = a3 = zero;
        };

        auto fold_all = [&] {
            fold_reset(t0, q00, q01, q02, q03);
            if constexpr (MR > 1) {
                fold_reset(t1, q10, q11, q12, q13);
            }
            if constexpr (MR > 2) {
                fold_reset(t2, q20, q21, q22, q23);
            }
            if constexpr (MR > 3) {
                fold_reset(t3, q30, q31, q32, q33);
            }
            if constexpr (MR > 4) {
                fold_reset(t4, q40, q41, q42, q43);
            }
            if constexpr (MR > 5) {
                fold_reset(t5, q50, q51, q52, q53);
            }
            if constexpr (MR > 6) {
                fold_reset(t6, q60, q61, q62, q63);
            }
            if constexpr (MR > 7) {
                fold_reset(t7, q70, q71, q72, q73);
            }
        };

        int k = 0;
        auto step = [&](int bi) {
            const uint8_t* p = src + static_cast<int64_t>(bi) * BQ::block_bytes;
            uint16_t d_bits = 0, m_bits = 0;
            std::memcpy(&d_bits, p + 0, sizeof(d_bits));
            if constexpr (BQ::dtype == DataType::q4_1) {
                std::memcpy(&m_bits, p + 2, sizeof(m_bits));
            }
            const float16x8_t vd =
                vdupq_n_f16(half_to_float(half_from_bits(d_bits)));
            const float16x8_t vm =
                BQ::dtype == DataType::q4_1 ?
                    vdupq_n_f16(half_to_float(half_from_bits(m_bits))) :
                    zero;
            const int16x8_t vbias = vdupq_n_s16(BQ::dtype == DataType::q4_1 ? 0 : -8);
            const uint8x16_t qs = vld1q_u8(p + (BQ::dtype == DataType::q4_1 ? 4 : 2));
            const uint8x16_t lo = vandq_u8(qs, vdupq_n_u8(0x0F));
            const uint8x16_t hi = vshrq_n_u8(qs, 4);

            auto vec = [&](uint8x8_t nib) {
                const int16x8_t q =
                    vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(nib)), vbias);
                const float16x8_t qf =
                    vcombine_f16(vcvt_f16_s16(vget_low_s16(q)),
                                 vcvt_f16_s16(vget_high_s16(q)));
                return vfmaq_f16(vm, qf, vd);
            };
            const float16x8_t b0 = vec(vget_low_u8(lo));   // values 0..7
            const float16x8_t b1 = vec(vget_high_u8(lo));  // values 8..15
            const float16x8_t b2 = vec(vget_low_u8(hi));   // values 16..23
            const float16x8_t b3 = vec(vget_high_u8(hi));  // values 24..31

            if constexpr (MR <= 4) {
                NNOPS_QBLOCK_F16_DOT4(q00, q01, q02, q03, 0);
                if constexpr (MR > 1) {
                    NNOPS_QBLOCK_F16_DOT4(q10, q11, q12, q13, 1);
                }
                if constexpr (MR > 2) {
                    NNOPS_QBLOCK_F16_DOT4(q20, q21, q22, q23, 2);
                }
                if constexpr (MR > 3) {
                    NNOPS_QBLOCK_F16_DOT4(q30, q31, q32, q33, 3);
                }
            } else {
                NNOPS_QBLOCK_F16_DOT2(q00, q01, 0);
                NNOPS_QBLOCK_F16_DOT2(q10, q11, 1);
                NNOPS_QBLOCK_F16_DOT2(q20, q21, 2);
                NNOPS_QBLOCK_F16_DOT2(q30, q31, 3);
                NNOPS_QBLOCK_F16_DOT2(q40, q41, 4);
                if constexpr (MR > 5) {
                    NNOPS_QBLOCK_F16_DOT2(q50, q51, 5);
                }
                if constexpr (MR > 6) {
                    NNOPS_QBLOCK_F16_DOT2(q60, q61, 6);
                }
                if constexpr (MR > 7) {
                    NNOPS_QBLOCK_F16_DOT2(q70, q71, 7);
                }
            }
            k += BQ::block_elems;
        };

        int b = 0;
        for (; b + kFoldBlocks <= nblk; b += kFoldBlocks) {
            for (int j = 0; j < kFoldBlocks; ++j) {
                step(b + j);
            }
            fold_all();
        }
        for (; b < nblk; ++b) {
            step(b);
        }
        fold_all();

        // Trailing partial block: f32 scalar, folded into the same sums.
        if (k < kc) {
            float vals[BQ::block_elems];
            BQ::template decode<float>(src + static_cast<int64_t>(nblk) * BQ::block_bytes, vals);
            const int rem = kc - k;
            for (int i = 0; i < rem; ++i) {
                const int kk = k + i;
                const float bv = vals[i];
                t0 += bv * qblock_load1(A + kk);
                if constexpr (MR > 1) {
                    t1 += bv * qblock_load1(A + static_cast<int64_t>(lda) + kk);
                }
                if constexpr (MR > 2) {
                    t2 += bv * qblock_load1(A + 2 * static_cast<int64_t>(lda) + kk);
                }
                if constexpr (MR > 3) {
                    t3 += bv * qblock_load1(A + 3 * static_cast<int64_t>(lda) + kk);
                }
                if constexpr (MR > 4) {
                    t4 += bv * qblock_load1(A + 4 * static_cast<int64_t>(lda) + kk);
                }
                if constexpr (MR > 5) {
                    t5 += bv * qblock_load1(A + 5 * static_cast<int64_t>(lda) + kk);
                }
                if constexpr (MR > 6) {
                    t6 += bv * qblock_load1(A + 6 * static_cast<int64_t>(lda) + kk);
                }
                if constexpr (MR > 7) {
                    t7 += bv * qblock_load1(A + 7 * static_cast<int64_t>(lda) + kk);
                }
            }
        }

        half* c = C + n;
        auto finish = [&](int row, float v) {
            half* cp = c + static_cast<int64_t>(row) * ldc;
            if (alpha != 1.0f) {
                v *= alpha;
            }
            if constexpr (!ZERO) {
                v += qblock_load1(cp);
            }
            qblock_store1(cp, std::min(std::max(v, clamp_min), clamp_max));
        };

        finish(0, t0);
        if constexpr (MR > 1) {
            finish(1, t1);
        }
        if constexpr (MR > 2) {
            finish(2, t2);
        }
        if constexpr (MR > 3) {
            finish(3, t3);
        }
        if constexpr (MR > 4) {
            finish(4, t4);
        }
        if constexpr (MR > 5) {
            finish(5, t5);
        }
        if constexpr (MR > 6) {
            finish(6, t6);
        }
        if constexpr (MR > 7) {
            finish(7, t7);
        }
    }

#undef NNOPS_QBLOCK_F16_DOT4
#undef NNOPS_QBLOCK_F16_DOT2
}

/// Dot-product form: one n row at a time, applied to every live A row.
///
/// @param C C[m0][n_start]; already offset by the caller.
/// @param A A[m0][0], row stride @p lda.
/// @param B first block of the batch element's [N][K] stream.
template <class BQ, class AC, int MR, bool ZERO>
void mma_qblock_trans(int nc, int kc, AC* NNOPS_RESTRICT C, int ldc, const AC* NNOPS_RESTRICT A,
                      int lda, const uint8_t* NNOPS_RESTRICT B, int64_t row_blocks, int n_start,
                      float alpha, float clamp_min, float clamp_max) noexcept
{
    if constexpr (std::is_same_v<AC, half>) {
        mma_qblock_trans_f16<BQ, MR, ZERO>(nc, kc, C, ldc, A, lda, B, row_blocks, n_start, alpha,
                                           clamp_min, clamp_max);
        return;
    }

    static_assert(MR >= 1 && MR <= BQ::panel_trans, "trans qblock kernel: MR in [1, panel_trans]");

    const int64_t row_bytes = row_blocks * static_cast<int64_t>(BQ::block_bytes);
    const uint8_t* src = B + static_cast<int64_t>(n_start) * row_bytes;

    const int nblk = kc / BQ::block_elems;

    for (int n = 0; n < nc; ++n, src += row_bytes) {
        float32x4_t q00 = vdupq_n_f32(0.0f), q01 = q00, q02 = q00, q03 = q00;
        float32x4_t q10 = q00, q11 = q00, q12 = q00, q13 = q00;
        float32x4_t q20 = q00, q21 = q00, q22 = q00, q23 = q00;
        float32x4_t q30 = q00, q31 = q00, q32 = q00, q33 = q00;
        float32x4_t q40 = q00, q41 = q00, q42 = q00, q43 = q00;
        float32x4_t q50 = q00, q51 = q00, q52 = q00, q53 = q00;
        float32x4_t q60 = q00, q61 = q00, q62 = q00, q63 = q00;
        float32x4_t q70 = q00, q71 = q00, q72 = q00, q73 = q00;

        int k = 0;
        for (int b = 0; b < nblk; ++b, k += BQ::block_elems) {
            float32x4_t bv[8];
            BQ::decode_v(src + static_cast<int64_t>(b) * BQ::block_bytes, bv);
            const float32x4_t b0 = bv[0], b1 = bv[1], b2 = bv[2], b3 = bv[3];
            const float32x4_t b4 = bv[4], b5 = bv[5], b6 = bv[6], b7 = bv[7];

            if constexpr (MR <= 4) {
                NNOPS_QBLOCK_DOT_ROW(q00, q01, q02, q03, 0);
                if constexpr (MR > 1) {
                    NNOPS_QBLOCK_DOT_ROW(q10, q11, q12, q13, 1);
                }
                if constexpr (MR > 2) {
                    NNOPS_QBLOCK_DOT_ROW(q20, q21, q22, q23, 2);
                }
                if constexpr (MR > 3) {
                    NNOPS_QBLOCK_DOT_ROW(q30, q31, q32, q33, 3);
                }
            } else {
                NNOPS_QBLOCK_DOT2_ROW(q00, q01, 0);
                NNOPS_QBLOCK_DOT2_ROW(q10, q11, 1);
                NNOPS_QBLOCK_DOT2_ROW(q20, q21, 2);
                NNOPS_QBLOCK_DOT2_ROW(q30, q31, 3);
                NNOPS_QBLOCK_DOT2_ROW(q40, q41, 4);
                if constexpr (MR > 5) {
                    NNOPS_QBLOCK_DOT2_ROW(q50, q51, 5);
                }
                if constexpr (MR > 6) {
                    NNOPS_QBLOCK_DOT2_ROW(q60, q61, 6);
                }
                if constexpr (MR > 7) {
                    NNOPS_QBLOCK_DOT2_ROW(q70, q71, 7);
                }
            }
        }

        // Trailing partial block: scalar, at most 31 values.
        float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
        float t4 = 0.0f, t5 = 0.0f, t6 = 0.0f, t7 = 0.0f;
        if (k < kc) {
            float vals[BQ::block_elems];
            BQ::template decode<float>(src + static_cast<int64_t>(nblk) * BQ::block_bytes, vals);
            const int rem = kc - k;
            for (int i = 0; i < rem; ++i) {
                const int kk = k + i;
                const float bv = vals[i];
                t0 += bv * qblock_load1(A + kk);
                if constexpr (MR > 1) {
                    t1 += bv * qblock_load1(A + static_cast<int64_t>(lda) + kk);
                }
                if constexpr (MR > 2) {
                    t2 += bv * qblock_load1(A + 2 * static_cast<int64_t>(lda) + kk);
                }
                if constexpr (MR > 3) {
                    t3 += bv * qblock_load1(A + 3 * static_cast<int64_t>(lda) + kk);
                }
                if constexpr (MR > 4) {
                    t4 += bv * qblock_load1(A + 4 * static_cast<int64_t>(lda) + kk);
                }
                if constexpr (MR > 5) {
                    t5 += bv * qblock_load1(A + 5 * static_cast<int64_t>(lda) + kk);
                }
                if constexpr (MR > 6) {
                    t6 += bv * qblock_load1(A + 6 * static_cast<int64_t>(lda) + kk);
                }
                if constexpr (MR > 7) {
                    t7 += bv * qblock_load1(A + 7 * static_cast<int64_t>(lda) + kk);
                }
            }
        }

        AC* c = C + n;
        auto finish = [&](int row, float32x4_t a0, float32x4_t a1, float32x4_t a2, float32x4_t a3,
                          float tail) {
            AC* cp = c + static_cast<int64_t>(row) * ldc;
            float v = vaddvq_f32(a0) + vaddvq_f32(a1) + vaddvq_f32(a2) + vaddvq_f32(a3) + tail;
            if (alpha != 1.0f) {
                v *= alpha;
            }
            if constexpr (!ZERO) {
                v += qblock_load1(cp);
            }
            qblock_store1(cp, std::min(std::max(v, clamp_min), clamp_max));
        };

        finish(0, q00, q01, q02, q03, t0);
        if constexpr (MR > 1) {
            finish(1, q10, q11, q12, q13, t1);
        }
        if constexpr (MR > 2) {
            finish(2, q20, q21, q22, q23, t2);
        }
        if constexpr (MR > 3) {
            finish(3, q30, q31, q32, q33, t3);
        }
        if constexpr (MR > 4) {
            finish(4, q40, q41, q42, q43, t4);
        }
        if constexpr (MR > 5) {
            finish(5, q50, q51, q52, q53, t5);
        }
        if constexpr (MR > 6) {
            finish(6, q60, q61, q62, q63, t6);
        }
        if constexpr (MR > 7) {
            finish(7, q70, q71, q72, q73, t7);
        }
    }
}

#undef NNOPS_QBLOCK_DOT2_ROW

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
void mma_qblock_copy(int nc, int kc, AC* NNOPS_RESTRICT C, int ldc, const AC* NNOPS_RESTRICT A,
                     int lda, const uint8_t* NNOPS_RESTRICT B, int64_t row_blocks, int n_start,
                     float alpha, float clamp_min, float clamp_max) noexcept
{
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

        float32x4_t r0[8] = {vzero, vzero, vzero, vzero, vzero, vzero, vzero, vzero};
        float32x4_t r1[8] = {vzero, vzero, vzero, vzero, vzero, vzero, vzero, vzero};

        for (int k = 0; k < kc; ++k) {
            const uint8_t* src = B + static_cast<int64_t>(k) * row_bytes +
                                 blk * static_cast<int64_t>(BQ::block_bytes);
            float32x4_t bv[8];
            BQ::decode_v(src, bv);

            const float32x4_t va = vdupq_n_f32(qblock_load1(A + k));
            for (int j = 0; j < 8; ++j) {
                r0[j] = vfmaq_f32(r0[j], va, bv[j]);
            }

            if constexpr (MR > 1) {
                const float32x4_t vb = vdupq_n_f32(qblock_load1(A + static_cast<int64_t>(lda) + k));
                for (int j = 0; j < 8; ++j) {
                    r1[j] = vfmaq_f32(r1[j], vb, bv[j]);
                }
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
                vst1q_f32(tp + 0, q[0]);
                vst1q_f32(tp + 4, q[1]);
                vst1q_f32(tp + 8, q[2]);
                vst1q_f32(tp + 12, q[3]);
                vst1q_f32(tp + 16, q[4]);
                vst1q_f32(tp + 20, q[5]);
                vst1q_f32(tp + 24, q[6]);
                vst1q_f32(tp + 28, q[7]);
                for (int i = 0; i < width; ++i) {
                    float v = tmp[i];
                    if (scaled) {
                        v *= alpha;
                    }
                    if constexpr (!ZERO) {
                        v += qblock_load1(cp + i);
                    }
                    qblock_store1(cp + i, std::min(std::max(v, clamp_min), clamp_max));
                }
            }
        };

        emit(0, r0);
        if constexpr (MR > 1) {
            emit(1, r1);
        }
    }
}

} // namespace nnops::backend::cpu::aarch64
