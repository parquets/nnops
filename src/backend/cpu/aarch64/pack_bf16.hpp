#pragma once
/// @file pack_bf16.hpp
/// @brief AArch64 bfloat16 pack kernels for GEMM LHS/RHS.
///
/// The two layouts are the ones mma_bf16.hpp reads:
///   - LHS (A): dense rows, `K` elements apart, written in place of the
///     k-major float layouts. A row past the live width repeats the last live
///     row, the MMA kernels' alias convention, so a partial-M panel needs no
///     padding. `pack_copy_*` takes the already-transposed source ([K][M]) and
///     writes the same rows through a transpose; `pack_trans_*` takes the
///     row-major [M][K] source, which is the rows layout already, so it is a
///     copy — plus the caller's scale, when there is one.
///   - RHS (B): the BFMMLA operand layout — ⌈K/4⌉ k-groups of [colpair][8],
///     where colpair c holds {col 2c k0..3, col 2c+1 k0..3}. A 16-column panel
///     is therefore 64 elements per k-group, and the columns past the live
///     width, like a short final k-group, are zero-filled.
///
/// NEON, like pack_f16.hpp. The scale paths widen and narrow eight lanes at a
/// time; the transposed LHS source runs 8×8 tiles through the f16 transpose —
/// a bit shuffle, so the bf16 payload does not care — and the RHS either joins
/// two 4-lane k loads into a colpair (source [N][K]) or transposes four k rows
/// at once (source [K][N]). A narrow panel (live width below the tile) and a
/// short final k-group take the scalar fill, which is also what zeroes the
/// padding.
///
/// Reference: onnxruntime MLAS SbgemmKernelNeon.S / SbgemmPackB — the same
/// operand layout and the same B, with A left row-major because the packer is
/// ours and a row pair loads as two 4-element halves either way.

#include <arm_neon.h>
#include <cstring>

#include "nnops/detail/bf16.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "nnops/detail/simd/cpu_features.hpp"
#include "backend/cpu/common/common_utils.hpp"
#include "transpose.hpp"

namespace nnops::backend::cpu::aarch64 {

using bfloat16 = ::nnops::backend::cpu::bfloat16;
namespace sims = ::nnops::simd;

inline constexpr int bf16_nr_max = 16;
/// B panel elements per k-group: 16 columns × 4 k.
inline constexpr int bf16_b_k4_elems = 64;

/// Elements of one packed-B panel at a k-block of @p kc.
inline constexpr int64_t bf16_b_panel_stride(int64_t kc) noexcept {
    return ((kc + 3) / 4) * bf16_b_k4_elems;
}

// =========================================================================
//  bf16 ⇄ f32, in registers
// =========================================================================
//
// bf16 *is* the top half of an f32: widening is a shift and narrowing is the
// same RNE add + shift the simd helpers use. Repeated here because the packers
// keep the result in a register rather than writing it out and reading back.

inline sims::v_f32x8 bf16_widen(uint16x8_t u) noexcept {
    return sims::v_f32x8(vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(u), 16)),
                         vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(u), 16)));
}

inline uint16x8_t bf16_narrow(const sims::v_f32x8& v) noexcept {
    const auto narrow = [](float32x4_t x) {
        uint32x4_t u = vreinterpretq_u32_f32(x);
        u = vaddq_u32(u, vaddq_u32(vdupq_n_u32(0x7FFFu),
                                   vandq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(1u))));
        return vshrn_n_u32(u, 16);
    };
    return vcombine_u16(narrow(v.lo), narrow(v.hi));
}

/// Eight source lanes times @p s, rounded once.
inline uint16x8_t bf16_scaled(uint16x8_t u, const sims::v_f32x8& s) noexcept {
    return bf16_narrow(sims::v_mul(bf16_widen(u), s));
}

// =========================================================================
//  LHS — dense rows
// =========================================================================

/// The [M][K] source is the rows layout already. Rows past @p w repeat the
/// last live row.
inline void pack_lhs_bf16_rows(bfloat16* NNOPS_RESTRICT output,
                               const bfloat16* NNOPS_RESTRICT input,
                               int ir_step, int K, float scale, int w,
                               int rows) noexcept {
    const bfloat16* NNOPS_RESTRICT last =
        input + static_cast<int64_t>(w > 0 ? w - 1 : 0) * ir_step;
    const bool unit = (scale == 1.0f);
    const sims::v_f32x8 s = sims::v_set1_f32x8(scale);
    for (int r = 0; r < rows; ++r) {
        const bfloat16* NNOPS_RESTRICT p =
            (r < w) ? input + static_cast<int64_t>(r) * ir_step : last;
        if (unit) {
            std::memcpy(output, p, static_cast<size_t>(K) * sizeof(bfloat16));
        } else {
            int k = 0;
            for (; k + 8 <= K; k += 8) {
                vst1q_u16(reinterpret_cast<uint16_t*>(output + k),
                          bf16_scaled(vld1q_u16(reinterpret_cast<const uint16_t*>(p) + k), s));
            }
            for (; k < K; ++k) {
                output[k] = bfloat16(static_cast<float>(p[k]) * scale);
            }
        }
        output += K;
    }
}

/// Eight k rows of the [K][M] source, transposed into eight dense rows.
/// @param w live source columns; the tile needs all 8. Returns the k reached,
/// so the caller's scalar pass can finish a short last tile.
inline int pack_lhs_bf16_trans8(bfloat16* NNOPS_RESTRICT output,
                                const bfloat16* NNOPS_RESTRICT input,
                                int ir_step, int K, float scale) noexcept {
    const bool unit = (scale == 1.0f);
    const sims::v_f32x8 s = sims::v_set1_f32x8(scale);
    const uint16_t* NNOPS_RESTRICT src = reinterpret_cast<const uint16_t*>(input);
    uint16_t* NNOPS_RESTRICT dst = reinterpret_cast<uint16_t*>(output);
    int k = 0;
    for (; k + 8 <= K; k += 8) {
        float16x8_t v0 = vreinterpretq_f16_u16(vld1q_u16(src + static_cast<int64_t>(k + 0) * ir_step));
        float16x8_t v1 = vreinterpretq_f16_u16(vld1q_u16(src + static_cast<int64_t>(k + 1) * ir_step));
        float16x8_t v2 = vreinterpretq_f16_u16(vld1q_u16(src + static_cast<int64_t>(k + 2) * ir_step));
        float16x8_t v3 = vreinterpretq_f16_u16(vld1q_u16(src + static_cast<int64_t>(k + 3) * ir_step));
        float16x8_t v4 = vreinterpretq_f16_u16(vld1q_u16(src + static_cast<int64_t>(k + 4) * ir_step));
        float16x8_t v5 = vreinterpretq_f16_u16(vld1q_u16(src + static_cast<int64_t>(k + 5) * ir_step));
        float16x8_t v6 = vreinterpretq_f16_u16(vld1q_u16(src + static_cast<int64_t>(k + 6) * ir_step));
        float16x8_t v7 = vreinterpretq_f16_u16(vld1q_u16(src + static_cast<int64_t>(k + 7) * ir_step));

        // Row i of the source is row i of k, so the transpose turns each row's
        // eight m values into one m row's eight k values.
        transpose_8x8_f16(v0, v1, v2, v3, v4, v5, v6, v7);

        uint16_t* o0 = dst + 0 * static_cast<int64_t>(K) + k;
        uint16_t* o1 = dst + 1 * static_cast<int64_t>(K) + k;
        uint16_t* o2 = dst + 2 * static_cast<int64_t>(K) + k;
        uint16_t* o3 = dst + 3 * static_cast<int64_t>(K) + k;
        uint16_t* o4 = dst + 4 * static_cast<int64_t>(K) + k;
        uint16_t* o5 = dst + 5 * static_cast<int64_t>(K) + k;
        uint16_t* o6 = dst + 6 * static_cast<int64_t>(K) + k;
        uint16_t* o7 = dst + 7 * static_cast<int64_t>(K) + k;
        if (unit) {
            vst1q_u16(o0, vreinterpretq_u16_f16(v0));
            vst1q_u16(o1, vreinterpretq_u16_f16(v1));
            vst1q_u16(o2, vreinterpretq_u16_f16(v2));
            vst1q_u16(o3, vreinterpretq_u16_f16(v3));
            vst1q_u16(o4, vreinterpretq_u16_f16(v4));
            vst1q_u16(o5, vreinterpretq_u16_f16(v5));
            vst1q_u16(o6, vreinterpretq_u16_f16(v6));
            vst1q_u16(o7, vreinterpretq_u16_f16(v7));
        } else {
            vst1q_u16(o0, bf16_scaled(vreinterpretq_u16_f16(v0), s));
            vst1q_u16(o1, bf16_scaled(vreinterpretq_u16_f16(v1), s));
            vst1q_u16(o2, bf16_scaled(vreinterpretq_u16_f16(v2), s));
            vst1q_u16(o3, bf16_scaled(vreinterpretq_u16_f16(v3), s));
            vst1q_u16(o4, bf16_scaled(vreinterpretq_u16_f16(v4), s));
            vst1q_u16(o5, bf16_scaled(vreinterpretq_u16_f16(v5), s));
            vst1q_u16(o6, bf16_scaled(vreinterpretq_u16_f16(v6), s));
            vst1q_u16(o7, bf16_scaled(vreinterpretq_u16_f16(v7), s));
        }
    }
    return k;
}

/// Four k rows at a time, the 4-row rung's version of the above.
inline int pack_lhs_bf16_trans4(bfloat16* NNOPS_RESTRICT output,
                                const bfloat16* NNOPS_RESTRICT input,
                                int ir_step, int K, float scale) noexcept {
    const bool unit = (scale == 1.0f);
    const sims::v_f32x8 s = sims::v_set1_f32x8(scale);
    const uint16_t* NNOPS_RESTRICT src = reinterpret_cast<const uint16_t*>(input);
    uint16_t* NNOPS_RESTRICT dst = reinterpret_cast<uint16_t*>(output);
    int k = 0;
    for (; k + 4 <= K; k += 4) {
        float16x4_t v0 = vreinterpret_f16_u16(vld1_u16(src + static_cast<int64_t>(k + 0) * ir_step));
        float16x4_t v1 = vreinterpret_f16_u16(vld1_u16(src + static_cast<int64_t>(k + 1) * ir_step));
        float16x4_t v2 = vreinterpret_f16_u16(vld1_u16(src + static_cast<int64_t>(k + 2) * ir_step));
        float16x4_t v3 = vreinterpret_f16_u16(vld1_u16(src + static_cast<int64_t>(k + 3) * ir_step));

        transpose_4x4_f16(v0, v1, v2, v3);

        uint16_t* o0 = dst + 0 * static_cast<int64_t>(K) + k;
        uint16_t* o1 = dst + 1 * static_cast<int64_t>(K) + k;
        uint16_t* o2 = dst + 2 * static_cast<int64_t>(K) + k;
        uint16_t* o3 = dst + 3 * static_cast<int64_t>(K) + k;
        if (unit) {
            vst1_u16(o0, vreinterpret_u16_f16(v0));
            vst1_u16(o1, vreinterpret_u16_f16(v1));
            vst1_u16(o2, vreinterpret_u16_f16(v2));
            vst1_u16(o3, vreinterpret_u16_f16(v3));
        } else {
            vst1_u16(o0, vget_low_u16(bf16_scaled(vcombine_u16(vreinterpret_u16_f16(v0),
                                                               vdup_n_u16(0)), s)));
            vst1_u16(o1, vget_low_u16(bf16_scaled(vcombine_u16(vreinterpret_u16_f16(v1),
                                                               vdup_n_u16(0)), s)));
            vst1_u16(o2, vget_low_u16(bf16_scaled(vcombine_u16(vreinterpret_u16_f16(v2),
                                                               vdup_n_u16(0)), s)));
            vst1_u16(o3, vget_low_u16(bf16_scaled(vcombine_u16(vreinterpret_u16_f16(v3),
                                                               vdup_n_u16(0)), s)));
        }
    }
    return k;
}

/// The [K][M] source: rows of length M at @p ir_step, written out as the same
/// rows of length K.
inline void pack_lhs_bf16_trans(bfloat16* NNOPS_RESTRICT output,
                                const bfloat16* NNOPS_RESTRICT input,
                                int ir_step, int K, float scale, int w,
                                int rows) noexcept {
    const int last = (w > 0) ? w - 1 : 0;
    int done = 0;
    if (rows == 8 && w >= 8) {
        done = pack_lhs_bf16_trans8(output, input, ir_step, K, scale);
    } else if (rows == 4 && w >= 4) {
        done = pack_lhs_bf16_trans4(output, input, ir_step, K, scale);
    }
    for (int r = 0; r < rows; ++r) {
        const int col = (r < w) ? r : last;
        bfloat16* NNOPS_RESTRICT out =
            output + static_cast<int64_t>(r) * K + done;
        const bfloat16* NNOPS_RESTRICT p =
            input + static_cast<int64_t>(done) * ir_step + col;
        for (int k = done; k < K; ++k) {
            *out++ = bfloat16(static_cast<float>(p[0]) * scale);
            p += ir_step;
        }
    }
}

inline void pack_trans_n8_bf16(bfloat16* NNOPS_RESTRICT output,
                               const bfloat16* NNOPS_RESTRICT input,
                               int ir_step, int K, float scale, int mr = 8) noexcept {
    pack_lhs_bf16_rows(output, input, ir_step, K, scale, mr, 8);
}
inline void pack_trans_n4_bf16(bfloat16* NNOPS_RESTRICT output,
                               const bfloat16* NNOPS_RESTRICT input,
                               int ir_step, int K, float scale, int mr = 4) noexcept {
    pack_lhs_bf16_rows(output, input, ir_step, K, scale, mr, 4);
}
inline void pack_trans_n1_bf16(bfloat16* NNOPS_RESTRICT output,
                               const bfloat16* NNOPS_RESTRICT input,
                               int ir_step, int K, float scale, int mr = 1) noexcept {
    pack_lhs_bf16_rows(output, input, ir_step, K, scale, mr, 1);
}

inline void pack_copy_n8_bf16(bfloat16* NNOPS_RESTRICT output,
                              const bfloat16* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int mr = 8) noexcept {
    pack_lhs_bf16_trans(output, input, ir_step, K, scale, mr, 8);
}
inline void pack_copy_n4_bf16(bfloat16* NNOPS_RESTRICT output,
                              const bfloat16* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int mr = 4) noexcept {
    pack_lhs_bf16_trans(output, input, ir_step, K, scale, mr, 4);
}
inline void pack_copy_n1_bf16(bfloat16* NNOPS_RESTRICT output,
                              const bfloat16* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale, int mr = 1) noexcept {
    pack_lhs_bf16_trans(output, input, ir_step, K, scale, mr, 1);
}

// =========================================================================
//  RHS — the BFMMLA operand layout
// =========================================================================

/// Shared scalar fill. @p at(c, k) yields the source element at column c, k
/// index k; columns past @p w and k past @p K write zero. @p gstart resumes
/// where the vector pass stopped.
template <class At>
inline void pack_rhs_bf16_core(bfloat16* NNOPS_RESTRICT output, int w, int K,
                               float scale, At&& at, int gstart = 0) noexcept {
    const int kgroups = (K + 3) / 4;
    for (int g = gstart; g < kgroups; ++g) {
        bfloat16* NNOPS_RESTRICT q =
            output + static_cast<int64_t>(g) * bf16_b_k4_elems;
        for (int cp = 0; cp < bf16_nr_max / 2; ++cp) {
            for (int lane = 0; lane < 8; ++lane) {
                const int c = 2 * cp + (lane >> 2);
                const int k = g * 4 + (lane & 3);
                float v = 0.0f;
                if (c < w && k < K) { v = static_cast<float>(at(c, k)) * scale; }
                q[lane] = bfloat16(v);
            }
            q += 8;
        }
    }
}

/// Source [N][K]: column c is a row, its k contiguous, so a colpair's eight
/// lanes are exactly two 4-lane k loads joined — and eight k at a time fill
/// two k-groups with one load pair.
inline void pack_rhs_bf16_trans(bfloat16* NNOPS_RESTRICT output,
                                const bfloat16* NNOPS_RESTRICT input,
                                int ir_step, int K, float scale, int w = 16) noexcept {
    int done = 0;
    if (w >= bf16_nr_max) {
        const bool unit = (scale == 1.0f);
        const sims::v_f32x8 s = sims::v_set1_f32x8(scale);
        const uint16_t* NNOPS_RESTRICT src = reinterpret_cast<const uint16_t*>(input);
        uint16_t* NNOPS_RESTRICT dst = reinterpret_cast<uint16_t*>(output);
        const int chunks = K / 8;
        for (int gg = 0; gg < chunks; ++gg) {
            const int64_t kb = 8LL * gg;
            for (int cp = 0; cp < bf16_nr_max / 2; ++cp) {
                const uint16x8_t a =
                    vld1q_u16(src + static_cast<int64_t>(2 * cp) * ir_step + kb);
                const uint16x8_t b =
                    vld1q_u16(src + static_cast<int64_t>(2 * cp + 1) * ir_step + kb);
                // Lanes 0..3 are the first column's four k, 4..7 the second's:
                // one k-group each, the low and high halves of the two loads.
                const uint16x8_t lo = vcombine_u16(vget_low_u16(a), vget_low_u16(b));
                const uint16x8_t hi = vcombine_u16(vget_high_u16(a), vget_high_u16(b));
                uint16_t* q = dst + (2 * gg) * bf16_b_k4_elems + cp * 8;
                if (unit) {
                    vst1q_u16(q, lo);
                    vst1q_u16(q + bf16_b_k4_elems, hi);
                } else {
                    vst1q_u16(q, bf16_scaled(lo, s));
                    vst1q_u16(q + bf16_b_k4_elems, bf16_scaled(hi, s));
                }
            }
        }
        done = 2 * chunks;
    }
    pack_rhs_bf16_core(output, w, K, scale,
                       [&](int c, int k) { return input[static_cast<int64_t>(c) * ir_step + k]; },
                       done);
}

/// Four k rows of eight columns → the four colpairs those columns make:
/// {col 2c k0..3, col 2c+1 k0..3} for c = 0..3. Two 16-bit and two 32-bit
/// interleaves, the same 4×4 transpose the f16 packers use, then the pairs
/// are concatenated.
inline void bf16_colpairs_4x8(uint16x8_t r0, uint16x8_t r1, uint16x8_t r2,
                              uint16x8_t r3, uint16x8_t& o0, uint16x8_t& o1,
                              uint16x8_t& o2, uint16x8_t& o3) noexcept {
    const uint16x8_t a0 = vtrn1q_u16(r0, r1);
    const uint16x8_t a1 = vtrn2q_u16(r0, r1);
    const uint16x8_t b0 = vtrn1q_u16(r2, r3);
    const uint16x8_t b1 = vtrn2q_u16(r2, r3);
    // x0 holds columns 0 and 4, x1 columns 2 and 6, x2 columns 1 and 5,
    // x3 columns 3 and 7 — four k values each in both halves.
    const uint16x8_t x0 = vreinterpretq_u16_u32(
        vtrn1q_u32(vreinterpretq_u32_u16(a0), vreinterpretq_u32_u16(b0)));
    const uint16x8_t x1 = vreinterpretq_u16_u32(
        vtrn2q_u32(vreinterpretq_u32_u16(a0), vreinterpretq_u32_u16(b0)));
    const uint16x8_t x2 = vreinterpretq_u16_u32(
        vtrn1q_u32(vreinterpretq_u32_u16(a1), vreinterpretq_u32_u16(b1)));
    const uint16x8_t x3 = vreinterpretq_u16_u32(
        vtrn2q_u32(vreinterpretq_u32_u16(a1), vreinterpretq_u32_u16(b1)));
    o0 = vcombine_u16(vget_low_u16(x0), vget_low_u16(x2));
    o1 = vcombine_u16(vget_low_u16(x1), vget_low_u16(x3));
    o2 = vcombine_u16(vget_high_u16(x0), vget_high_u16(x2));
    o3 = vcombine_u16(vget_high_u16(x1), vget_high_u16(x3));
}

/// Source [K][N]: row k holds the columns, so a k-group is four row loads and
/// a 4×8 transpose — twice, once per 8-column block of the panel.
inline void pack_rhs_bf16_copy(bfloat16* NNOPS_RESTRICT output,
                               const bfloat16* NNOPS_RESTRICT input,
                               int ir_step, int K, float scale, int w = 16) noexcept {
    int done = 0;
    if (w >= bf16_nr_max) {
        const bool unit = (scale == 1.0f);
        const sims::v_f32x8 s = sims::v_set1_f32x8(scale);
        const uint16_t* NNOPS_RESTRICT src = reinterpret_cast<const uint16_t*>(input);
        uint16_t* NNOPS_RESTRICT dst = reinterpret_cast<uint16_t*>(output);
        const int kgroups = K / 4;
        for (int g = 0; g < kgroups; ++g) {
            const uint16_t* NNOPS_RESTRICT r0 = src + static_cast<int64_t>(4 * g + 0) * ir_step;
            const uint16_t* NNOPS_RESTRICT r1 = src + static_cast<int64_t>(4 * g + 1) * ir_step;
            const uint16_t* NNOPS_RESTRICT r2 = src + static_cast<int64_t>(4 * g + 2) * ir_step;
            const uint16_t* NNOPS_RESTRICT r3 = src + static_cast<int64_t>(4 * g + 3) * ir_step;

            uint16_t* q = dst + static_cast<int64_t>(g) * bf16_b_k4_elems;
            uint16x8_t o0, o1, o2, o3;
            bf16_colpairs_4x8(vld1q_u16(r0), vld1q_u16(r1), vld1q_u16(r2),
                              vld1q_u16(r3), o0, o1, o2, o3);
            if (unit) {
                vst1q_u16(q + 0, o0);
                vst1q_u16(q + 8, o1);
                vst1q_u16(q + 16, o2);
                vst1q_u16(q + 24, o3);
            } else {
                vst1q_u16(q + 0, bf16_scaled(o0, s));
                vst1q_u16(q + 8, bf16_scaled(o1, s));
                vst1q_u16(q + 16, bf16_scaled(o2, s));
                vst1q_u16(q + 24, bf16_scaled(o3, s));
            }
            bf16_colpairs_4x8(vld1q_u16(r0 + 8), vld1q_u16(r1 + 8),
                              vld1q_u16(r2 + 8), vld1q_u16(r3 + 8),
                              o0, o1, o2, o3);
            if (unit) {
                vst1q_u16(q + 32, o0);
                vst1q_u16(q + 40, o1);
                vst1q_u16(q + 48, o2);
                vst1q_u16(q + 56, o3);
            } else {
                vst1q_u16(q + 32, bf16_scaled(o0, s));
                vst1q_u16(q + 40, bf16_scaled(o1, s));
                vst1q_u16(q + 48, bf16_scaled(o2, s));
                vst1q_u16(q + 56, bf16_scaled(o3, s));
            }
        }
        done = kgroups;
    }
    pack_rhs_bf16_core(output, w, K, scale,
                       [&](int c, int k) { return input[static_cast<int64_t>(k) * ir_step + c]; },
                       done);
}

inline void pack_trans_n16_bf16(bfloat16* NNOPS_RESTRICT output,
                                const bfloat16* NNOPS_RESTRICT input,
                                int ir_step, int K, float scale, int nr = 16) noexcept {
    pack_rhs_bf16_trans(output, input, ir_step, K, scale, nr);
}
inline void pack_copy_n16_bf16(bfloat16* NNOPS_RESTRICT output,
                               const bfloat16* NNOPS_RESTRICT input,
                               int ir_step, int K, float scale, int nr = 16) noexcept {
    pack_rhs_bf16_copy(output, input, ir_step, K, scale, nr);
}

// =========================================================================
//  Micro-panel size menu (bf16)  —  mr = LHS rows, nr = RHS columns.
// =========================================================================

inline constexpr int mr_bf16[4] = {8, 4, 1, 0};

inline constexpr int nr_bf16[3] = {16, 8, 1};

static_assert(bf16_nr_max == nr_bf16[0], "the B panel width is the pack's");

}  // namespace nnops::backend::cpu::aarch64
