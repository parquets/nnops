#pragma once
/// @file i8_requant_store.hpp
/// @brief The 8-bit requantizing store the int8 MMA kernels share: s8×s8
/// accumulators written as the destination's bytes.
///
///     out = sat(round_away((acc + bias[n]) * (s_a/s_out * s_b[n])) + zp_out)
///
/// Relu is a floor of zero before the scale — the compensated accumulate is the
/// true product, so its sign is the activation's. `vqmovn`/`vqmovun` saturate at
/// the destination's range, which is exactly the clamp the generic int8 row
/// pass applies, so no separate min/max is needed; the zero point is added with
/// a saturating add, so a product past int32 still lands on an end byte.
///
/// The kernels take the per-column int32 bias already folded with the A-side
/// shift ((a_bias_offset + zp_a) * r_b[n], see the callers), so one vector add
/// is the whole per-element correction. Everything here is baseline NEON: the
/// SDOT and SMMLA routes share it.
///
/// Every index into the column blocks is a literal — the three-group wider
/// forms are written out instead of looping to a template bound. A loop costs
/// nothing on a compiler that unrolls it and everything on one that does not:
/// the arrays would be indexed at run time and stay in memory.

#include <arm_neon.h>
#include <cstdint>
#include <type_traits>

namespace nnops::backend::cpu::aarch64 {

/// The kernel signature the requantizing rungs are dispatched through.
template <class TOut>
using MmaI8EpiFn = void (*)(int, int, int, TOut*, int64_t, const int8_t*, const int8_t*, int,
                            const float*, float, float, const int32_t*, int32_t, bool);

/// One four-column group's scale and bias. Hoisted out of the row pairs: without
/// it a panel reloads and re-multiplies them once per row.
template <int NG>
__attribute__((always_inline)) inline void
i8_col_q(float32x4_t (&sc)[NG], int32x4_t (&bi)[NG], float32x4_t vk, float32x4_t vreq,
         const float* s_b, const int32_t* col_bias) noexcept
{
    static_assert(NG >= 1 && NG <= 3, "a panel is one to three column groups");
    const int32x4_t vz = vdupq_n_s32(0);
    sc[0] = s_b ? vmulq_f32(vk, vld1q_f32(s_b + 0)) : vreq;
    bi[0] = col_bias ? vld1q_s32(col_bias + 0) : vz;
    if constexpr (NG > 1) {
        sc[1] = s_b ? vmulq_f32(vk, vld1q_f32(s_b + 4)) : vreq;
        bi[1] = col_bias ? vld1q_s32(col_bias + 4) : vz;
    }
    if constexpr (NG > 2) {
        sc[2] = s_b ? vmulq_f32(vk, vld1q_f32(s_b + 8)) : vreq;
        bi[2] = col_bias ? vld1q_s32(col_bias + 8) : vz;
    }
}

/// One group at `BASE` columns of a tail, where the last group can be narrower
/// than four columns: only the live lanes are read, and the dead ones stay zero
/// rather than reading past the table's end. Only live lanes are ever stored,
/// so their values do not matter. No local buffer: the lanes are assembled with
/// vector lane inserts, so nothing here is a stack object.
template <int BASE>
__attribute__((always_inline)) inline void
i8_col_group_live(float32x4_t& sc, int32x4_t& bi, float32x4_t vk, float32x4_t vreq,
                  const float* s_b, const int32_t* col_bias, int live) noexcept
{
    const int32x4_t vz = vdupq_n_s32(0);
    if (live >= BASE + 4) {
        sc = s_b ? vmulq_f32(vk, vld1q_f32(s_b + BASE)) : vreq;
        bi = col_bias ? vld1q_s32(col_bias + BASE) : vz;
        return;
    }
    const int lanes = live - BASE;
    float32x4_t s4 = vdupq_n_f32(0.0f);
    int32x4_t b4 = vz;
    if (lanes > 0) {
        if (s_b) {
            s4 = vsetq_lane_f32(s_b[BASE + 0], s4, 0);
        }
        if (col_bias) {
            b4 = vsetq_lane_s32(col_bias[BASE + 0], b4, 0);
        }
    }
    if (lanes > 1) {
        if (s_b) {
            s4 = vsetq_lane_f32(s_b[BASE + 1], s4, 1);
        }
        if (col_bias) {
            b4 = vsetq_lane_s32(col_bias[BASE + 1], b4, 1);
        }
    }
    if (lanes > 2) {
        if (s_b) {
            s4 = vsetq_lane_f32(s_b[BASE + 2], s4, 2);
        }
        if (col_bias) {
            b4 = vsetq_lane_s32(col_bias[BASE + 2], b4, 2);
        }
    }
    sc = s_b ? vmulq_f32(vk, s4) : vreq;
    bi = col_bias ? b4 : vz;
}

/// The tail's per-column scale and bias: one call per group, literals only.
template <int NG>
__attribute__((always_inline)) inline void
i8_col_q_live(float32x4_t (&sc)[NG], int32x4_t (&bi)[NG], float32x4_t vk, float32x4_t vreq,
              const float* s_b, const int32_t* col_bias, int live) noexcept
{
    static_assert(NG >= 1 && NG <= 3, "a panel is one to three column groups");
    i8_col_group_live<0>(sc[0], bi[0], vk, vreq, s_b, col_bias, live);
    if constexpr (NG > 1) {
        i8_col_group_live<4>(sc[1], bi[1], vk, vreq, s_b, col_bias, live);
    }
    if constexpr (NG > 2) {
        i8_col_group_live<8>(sc[2], bi[2], vk, vreq, s_b, col_bias, live);
    }
}

/// Four columns of one row: bias, Relu, scale, round, offset, narrow, store.
template <class TOut>
__attribute__((always_inline)) inline void
i8_store4q(TOut* dst, int32x4_t v, float32x4_t vscale, int32x4_t vbias, int32x4_t vlo,
           int32x4_t vzp) noexcept
{
    v = vaddq_s32(v, vbias);
    v = vmaxq_s32(v, vlo);
    const int32x4_t q = vqaddq_s32(vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(v), vscale)), vzp);
    const int16x4_t i16 = vqmovn_s32(q);
    if constexpr (std::is_same_v<TOut, uint8_t>) {
        const uint8x8_t v8 = vqmovun_s16(vcombine_s16(i16, i16));
        vst1_lane_u32(reinterpret_cast<uint32_t*>(dst), vreinterpret_u32_u8(v8), 0);
    } else {
        const int8x8_t v8 = vqmovn_s16(vcombine_s16(i16, i16));
        vst1_lane_s32(reinterpret_cast<int32_t*>(dst), vreinterpret_s32_s8(v8), 0);
    }
}

/// The same for a ragged group: `live` (1-3) bytes are written, one lane at a
/// time — no buffer and no run-time loop, so the tail's accumulators stay in
/// registers on a compiler that would otherwise leave this block in memory.
template <class TOut>
__attribute__((always_inline)) inline void
i8_store4q_lanes(TOut* dst, int live, int32x4_t v, float32x4_t vscale, int32x4_t vbias,
                 int32x4_t vlo, int32x4_t vzp) noexcept
{
    v = vaddq_s32(v, vbias);
    v = vmaxq_s32(v, vlo);
    const int32x4_t q = vqaddq_s32(vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(v), vscale)), vzp);
    const int16x4_t i16 = vqmovn_s32(q);
    uint8x8_t bytes;
    if constexpr (std::is_same_v<TOut, uint8_t>) {
        bytes = vqmovun_s16(vcombine_s16(i16, i16));
    } else {
        bytes = vreinterpret_u8_s8(vqmovn_s16(vcombine_s16(i16, i16)));
    }
    uint8_t* const p = reinterpret_cast<uint8_t*>(dst);
    vst1_lane_u8(p + 0, bytes, 0);
    if (live > 1) {
        vst1_lane_u8(p + 1, bytes, 1);
    }
    if (live > 2) {
        vst1_lane_u8(p + 2, bytes, 2);
    }
}

} // namespace nnops::backend::cpu::aarch64
