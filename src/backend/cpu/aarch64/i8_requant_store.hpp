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

#include <arm_neon.h>
#include <cstdint>
#include <type_traits>

namespace nnops::backend::cpu::aarch64 {

/// The kernel signature the requantizing rungs are dispatched through.
template <class TOut>
using MmaI8EpiFn = void (*)(int, int, int, TOut*, int64_t, const int8_t*, const int8_t*, int,
                            const float*, float, float, const int32_t*, int32_t, bool);

/// The per-column scale and bias of one four-column group. Hoisted out of the
/// row pairs: without it a panel reloads and re-multiplies them once per row.
template <int NG>
__attribute__((always_inline)) inline void
i8_col_q(float32x4_t (&sc)[NG], int32x4_t (&bi)[NG], float32x4_t vk, float32x4_t vreq,
         const float* s_b, const int32_t* col_bias) noexcept
{
    const int32x4_t vz = vdupq_n_s32(0);
    for (int g = 0; g < NG; ++g) {
        sc[g] = s_b ? vmulq_f32(vk, vld1q_f32(s_b + 4 * g)) : vreq;
        bi[g] = col_bias ? vld1q_s32(col_bias + 4 * g) : vz;
    }
}

/// The same for the tail, where the last group can be narrower than four
/// columns: the dead lanes are zeroed rather than read past the table's end.
/// Only the live lanes are ever stored, so their values do not matter.
template <int NG>
__attribute__((always_inline)) inline void
i8_col_q_live(float32x4_t (&sc)[NG], int32x4_t (&bi)[NG], float32x4_t vk, float32x4_t vreq,
              const float* s_b, const int32_t* col_bias, int live) noexcept
{
    const int32x4_t vz = vdupq_n_s32(0);
    for (int g = 0; g < NG; ++g) {
        const int base = 4 * g;
        if (base + 4 <= live) {
            sc[g] = s_b ? vmulq_f32(vk, vld1q_f32(s_b + base)) : vreq;
            bi[g] = col_bias ? vld1q_s32(col_bias + base) : vz;
            continue;
        }
        alignas(16) float sv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        alignas(16) int32_t bv[4] = {0, 0, 0, 0};
        for (int c = 0; c < 4 && base + c < live; ++c) {
            if (s_b) {
                sv[c] = s_b[base + c];
            }
            if (col_bias) {
                bv[c] = col_bias[base + c];
            }
        }
        sc[g] = s_b ? vmulq_f32(vk, vld1q_f32(sv)) : vreq;
        bi[g] = col_bias ? vld1q_s32(bv) : vz;
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

/// The same for a ragged group: `live` (1-3) bytes are written.
template <class TOut>
__attribute__((always_inline)) inline void
i8_store4q_lanes(TOut* dst, int live, int32x4_t v, float32x4_t vscale, int32x4_t vbias,
                 int32x4_t vlo, int32x4_t vzp) noexcept
{
    v = vaddq_s32(v, vbias);
    v = vmaxq_s32(v, vlo);
    const int32x4_t q = vqaddq_s32(vcvtaq_s32_f32(vmulq_f32(vcvtq_f32_s32(v), vscale)), vzp);
    const int16x4_t i16 = vqmovn_s32(q);
    if constexpr (std::is_same_v<TOut, uint8_t>) {
        uint8_t buf[8];
        vst1_u8(buf, vqmovun_s16(vcombine_s16(i16, i16)));
        for (int c = 0; c < live; ++c) {
            dst[c] = buf[c];
        }
    } else {
        int8_t buf[8];
        vst1_s8(buf, vqmovn_s16(vcombine_s16(i16, i16)));
        for (int c = 0; c < live; ++c) {
            dst[c] = buf[c];
        }
    }
}

} // namespace nnops::backend::cpu::aarch64
