#pragma once
/// @file transpose.hpp
/// @brief ARM NEON matrix transpose primitives for pack operations.
///
/// Provides vector register-level transposes used by the pack kernels
/// to reorder data between row-major (input) and packed (output) layouts.

#include <arm_neon.h>
#include "backend/cpu/common/restrict.hpp"

namespace nnops::backend::cpu::aarch64 {

// =========================================================================
// float32  —  2-way / 4-way / 6-way / 8-way / 12-way / 16-way
// =========================================================================

inline void transpose_2x4_f32(float32x4_t& v0, float32x4_t& v1) noexcept {
    const float32x4x2_t zip = vzipq_f32(v0, v1);
    v0 = zip.val[0];
    v1 = zip.val[1];
}

inline void transpose_4x4_f32(float32x4_t& a, float32x4_t& b,
                               float32x4_t& c, float32x4_t& d) noexcept {
    const float32x4x2_t ab = vzipq_f32(a, b);
    const float32x4x2_t cd = vzipq_f32(c, d);
    a = vcombine_f32(vget_low_f32(ab.val[0]),  vget_low_f32(cd.val[0]));
    b = vcombine_f32(vget_high_f32(ab.val[0]), vget_high_f32(cd.val[0]));
    c = vcombine_f32(vget_low_f32(ab.val[1]),  vget_low_f32(cd.val[1]));
    d = vcombine_f32(vget_high_f32(ab.val[1]), vget_high_f32(cd.val[1]));
}

inline void transpose_4x4_i32(int32x4_t& a, int32x4_t& b,
                               int32x4_t& c, int32x4_t& d) noexcept {
    const int32x4x2_t ab = vzipq_s32(a, b);
    const int32x4x2_t cd = vzipq_s32(c, d);
    a = vcombine_s32(vget_low_s32(ab.val[0]),  vget_low_s32(cd.val[0]));
    b = vcombine_s32(vget_high_s32(ab.val[0]), vget_high_s32(cd.val[0]));
    c = vcombine_s32(vget_low_s32(ab.val[1]),  vget_low_s32(cd.val[1]));
    d = vcombine_s32(vget_high_s32(ab.val[1]), vget_high_s32(cd.val[1]));
}

inline void transpose_6x4_f32(float32x4_t& a, float32x4_t& b, float32x4_t& c,
                               float32x4_t& d, float32x4_t& e, float32x4_t& f) noexcept {
    const float32x4x2_t ab = vzipq_f32(a, b);
    const float32x4x2_t cd = vzipq_f32(c, d);
    const float32x4x2_t ef = vzipq_f32(e, f);

    a = vcombine_f32(vget_low_f32(ab.val[0]),  vget_low_f32(cd.val[0]));
    b = vcombine_f32(vget_low_f32(ef.val[0]),  vget_high_f32(ab.val[0]));
    c = vcombine_f32(vget_high_f32(cd.val[0]), vget_high_f32(ef.val[0]));
    d = vcombine_f32(vget_low_f32(ab.val[1]),  vget_low_f32(cd.val[1]));
    e = vcombine_f32(vget_low_f32(ef.val[1]),  vget_high_f32(ab.val[1]));
    f = vcombine_f32(vget_high_f32(cd.val[1]), vget_high_f32(ef.val[1]));
}

inline void transpose_8x4_f32(float32x4_t& a, float32x4_t& b, float32x4_t& c, float32x4_t& d,
                               float32x4_t& e, float32x4_t& f, float32x4_t& g, float32x4_t& h) noexcept {
    const float32x4x2_t ab = vzipq_f32(a, b);
    const float32x4x2_t cd = vzipq_f32(c, d);
    const float32x4x2_t ef = vzipq_f32(e, f);
    const float32x4x2_t gh = vzipq_f32(g, h);

    a = vcombine_f32(vget_low_f32(ab.val[0]),  vget_low_f32(cd.val[0]));
    b = vcombine_f32(vget_low_f32(ef.val[0]),  vget_low_f32(gh.val[0]));
    c = vcombine_f32(vget_high_f32(ab.val[0]), vget_high_f32(cd.val[0]));
    d = vcombine_f32(vget_high_f32(ef.val[0]), vget_high_f32(gh.val[0]));
    e = vcombine_f32(vget_low_f32(ab.val[1]),  vget_low_f32(cd.val[1]));
    f = vcombine_f32(vget_low_f32(ef.val[1]),  vget_low_f32(gh.val[1]));
    g = vcombine_f32(vget_high_f32(ab.val[1]), vget_high_f32(cd.val[1]));
    h = vcombine_f32(vget_high_f32(ef.val[1]), vget_high_f32(gh.val[1]));
}

inline void transpose_8x4_i32(int32x4_t& a, int32x4_t& b, int32x4_t& c, int32x4_t& d,
                               int32x4_t& e, int32x4_t& f, int32x4_t& g, int32x4_t& h) noexcept {
    const int32x4x2_t ab = vzipq_s32(a, b);
    const int32x4x2_t cd = vzipq_s32(c, d);
    const int32x4x2_t ef = vzipq_s32(e, f);
    const int32x4x2_t gh = vzipq_s32(g, h);

    a = vcombine_s32(vget_low_s32(ab.val[0]),  vget_low_s32(cd.val[0]));
    b = vcombine_s32(vget_low_s32(ef.val[0]),  vget_low_s32(gh.val[0]));
    c = vcombine_s32(vget_high_s32(ab.val[0]), vget_high_s32(cd.val[0]));
    d = vcombine_s32(vget_high_s32(ef.val[0]), vget_high_s32(gh.val[0]));
    e = vcombine_s32(vget_low_s32(ab.val[1]),  vget_low_s32(cd.val[1]));
    f = vcombine_s32(vget_low_s32(ef.val[1]),  vget_low_s32(gh.val[1]));
    g = vcombine_s32(vget_high_s32(ab.val[1]), vget_high_s32(cd.val[1]));
    h = vcombine_s32(vget_high_s32(ef.val[1]), vget_high_s32(gh.val[1]));
}

inline void transpose_12x4_f32(float32x4_t& a, float32x4_t& b, float32x4_t& c, float32x4_t& d,
                                float32x4_t& e, float32x4_t& f, float32x4_t& g, float32x4_t& h,
                                float32x4_t& i, float32x4_t& j, float32x4_t& k, float32x4_t& l) noexcept {
    const float32x4x2_t ab = vzipq_f32(a, b);
    const float32x4x2_t cd = vzipq_f32(c, d);
    const float32x4x2_t ef = vzipq_f32(e, f);
    const float32x4x2_t gh = vzipq_f32(g, h);
    const float32x4x2_t ij = vzipq_f32(i, j);
    const float32x4x2_t kl = vzipq_f32(k, l);

    a = vcombine_f32(vget_low_f32(ab.val[0]),  vget_low_f32(cd.val[0]));
    b = vcombine_f32(vget_low_f32(ef.val[0]),  vget_low_f32(gh.val[0]));
    c = vcombine_f32(vget_low_f32(ij.val[0]),  vget_low_f32(kl.val[0]));
    d = vcombine_f32(vget_high_f32(ab.val[0]), vget_high_f32(cd.val[0]));
    e = vcombine_f32(vget_high_f32(ef.val[0]), vget_high_f32(gh.val[0]));
    f = vcombine_f32(vget_high_f32(ij.val[0]), vget_high_f32(kl.val[0]));

    g = vcombine_f32(vget_low_f32(ab.val[1]),  vget_low_f32(cd.val[1]));
    h = vcombine_f32(vget_low_f32(ef.val[1]),  vget_low_f32(gh.val[1]));
    i = vcombine_f32(vget_low_f32(ij.val[1]),  vget_low_f32(kl.val[1]));
    j = vcombine_f32(vget_high_f32(ab.val[1]), vget_high_f32(cd.val[1]));
    k = vcombine_f32(vget_high_f32(ef.val[1]), vget_high_f32(gh.val[1]));
    l = vcombine_f32(vget_high_f32(ij.val[1]), vget_high_f32(kl.val[1]));
}

inline void transpose_12x4_i32(int32x4_t& a, int32x4_t& b, int32x4_t& c, int32x4_t& d,
                                int32x4_t& e, int32x4_t& f, int32x4_t& g, int32x4_t& h,
                                int32x4_t& i, int32x4_t& j, int32x4_t& k, int32x4_t& l) noexcept {
    const int32x4x2_t ab = vzipq_s32(a, b);
    const int32x4x2_t cd = vzipq_s32(c, d);
    const int32x4x2_t ef = vzipq_s32(e, f);
    const int32x4x2_t gh = vzipq_s32(g, h);
    const int32x4x2_t ij = vzipq_s32(i, j);
    const int32x4x2_t kl = vzipq_s32(k, l);

    a = vcombine_s32(vget_low_s32(ab.val[0]),  vget_low_s32(cd.val[0]));
    b = vcombine_s32(vget_low_s32(ef.val[0]),  vget_low_s32(gh.val[0]));
    c = vcombine_s32(vget_low_s32(ij.val[0]),  vget_low_s32(kl.val[0]));
    d = vcombine_s32(vget_high_s32(ab.val[0]), vget_high_s32(cd.val[0]));
    e = vcombine_s32(vget_high_s32(ef.val[0]), vget_high_s32(gh.val[0]));
    f = vcombine_s32(vget_high_s32(ij.val[0]), vget_high_s32(kl.val[0]));

    g = vcombine_s32(vget_low_s32(ab.val[1]),  vget_low_s32(cd.val[1]));
    h = vcombine_s32(vget_low_s32(ef.val[1]),  vget_low_s32(gh.val[1]));
    i = vcombine_s32(vget_low_s32(ij.val[1]),  vget_low_s32(kl.val[1]));
    j = vcombine_s32(vget_high_s32(ab.val[1]), vget_high_s32(cd.val[1]));
    k = vcombine_s32(vget_high_s32(ef.val[1]), vget_high_s32(gh.val[1]));
    l = vcombine_s32(vget_high_s32(ij.val[1]), vget_high_s32(kl.val[1]));
}

inline void transpose_16x4_f32(float32x4_t& a, float32x4_t& b, float32x4_t& c, float32x4_t& d,
                                float32x4_t& e, float32x4_t& f, float32x4_t& g, float32x4_t& h,
                                float32x4_t& i, float32x4_t& j, float32x4_t& k, float32x4_t& l,
                                float32x4_t& m, float32x4_t& n, float32x4_t& o, float32x4_t& p) noexcept {
    const float32x4x2_t ab = vzipq_f32(a, b);
    const float32x4x2_t cd = vzipq_f32(c, d);
    const float32x4x2_t ef = vzipq_f32(e, f);
    const float32x4x2_t gh = vzipq_f32(g, h);
    const float32x4x2_t ij = vzipq_f32(i, j);
    const float32x4x2_t kl = vzipq_f32(k, l);
    const float32x4x2_t mn = vzipq_f32(m, n);
    const float32x4x2_t op = vzipq_f32(o, p);

    a = vcombine_f32(vget_low_f32(ab.val[0]),  vget_low_f32(cd.val[0]));
    b = vcombine_f32(vget_low_f32(ef.val[0]),  vget_low_f32(gh.val[0]));
    c = vcombine_f32(vget_low_f32(ij.val[0]),  vget_low_f32(kl.val[0]));
    d = vcombine_f32(vget_low_f32(mn.val[0]),  vget_low_f32(op.val[0]));
    e = vcombine_f32(vget_high_f32(ab.val[0]), vget_high_f32(cd.val[0]));
    f = vcombine_f32(vget_high_f32(ef.val[0]), vget_high_f32(gh.val[0]));
    g = vcombine_f32(vget_high_f32(ij.val[0]), vget_high_f32(kl.val[0]));
    h = vcombine_f32(vget_high_f32(mn.val[0]), vget_high_f32(op.val[0]));

    i = vcombine_f32(vget_low_f32(ab.val[1]),  vget_low_f32(cd.val[1]));
    j = vcombine_f32(vget_low_f32(ef.val[1]),  vget_low_f32(gh.val[1]));
    k = vcombine_f32(vget_low_f32(ij.val[1]),  vget_low_f32(kl.val[1]));
    l = vcombine_f32(vget_low_f32(mn.val[1]),  vget_low_f32(op.val[1]));
    m = vcombine_f32(vget_high_f32(ab.val[1]), vget_high_f32(cd.val[1]));
    n = vcombine_f32(vget_high_f32(ef.val[1]), vget_high_f32(gh.val[1]));
    o = vcombine_f32(vget_high_f32(ij.val[1]), vget_high_f32(kl.val[1]));
    p = vcombine_f32(vget_high_f32(mn.val[1]), vget_high_f32(op.val[1]));
}

// =========================================================================
// float16  —  2-way / 4-way / 8-way / 12-way / 16-way / 24-way
// =========================================================================

inline void transpose_2x4_f16(float16x4_t& v0, float16x4_t& v1) noexcept {
    const float16x4x2_t zip = vzip_f16(v0, v1);
    v0 = zip.val[0];
    v1 = zip.val[1];
}

inline void transpose_2x8_f16(float16x8_t& v0, float16x8_t& v1) noexcept {
    const float16x8x2_t zip = vzipq_f16(v0, v1);
    v0 = zip.val[0];
    v1 = zip.val[1];
}

inline void transpose_4x4_f16(float16x4_t& a, float16x4_t& b,
                               float16x4_t& c, float16x4_t& d) noexcept {
    const float16x4x2_t ab = vzip_f16(a, b);
    const float16x4x2_t cd = vzip_f16(c, d);

    const float32x2_t ab0 = vreinterpret_f32_f16(ab.val[0]);
    const float32x2_t cd0 = vreinterpret_f32_f16(cd.val[0]);
    const float32x2_t ab1 = vreinterpret_f32_f16(ab.val[1]);
    const float32x2_t cd1 = vreinterpret_f32_f16(cd.val[1]);

    const float32x2x2_t zip0 = vzip_f32(ab0, cd0);
    const float32x2x2_t zip1 = vzip_f32(ab1, cd1);

    a = vreinterpret_f16_f32(zip0.val[0]);
    b = vreinterpret_f16_f32(zip0.val[1]);
    c = vreinterpret_f16_f32(zip1.val[0]);
    d = vreinterpret_f16_f32(zip1.val[1]);
}

inline void transpose_4x8_f16(float16x8_t& a, float16x8_t& b,
                               float16x8_t& c, float16x8_t& d) noexcept {
    const float16x8x2_t ab = vzipq_f16(a, b);
    const float16x8x2_t cd = vzipq_f16(c, d);

    const float32x4_t ab0 = vreinterpretq_f32_f16(ab.val[0]);
    const float32x4_t cd0 = vreinterpretq_f32_f16(cd.val[0]);
    const float32x4x2_t zip0 = vzipq_f32(ab0, cd0);
    a = vreinterpretq_f16_f32(zip0.val[0]);
    b = vreinterpretq_f16_f32(zip0.val[1]);

    const float32x4_t ab1 = vreinterpretq_f32_f16(ab.val[1]);
    const float32x4_t cd1 = vreinterpretq_f32_f16(cd.val[1]);
    const float32x4x2_t zip1 = vzipq_f32(ab1, cd1);
    c = vreinterpretq_f16_f32(zip1.val[0]);
    d = vreinterpretq_f16_f32(zip1.val[1]);
}

inline void transpose_8x8_f16(float16x8_t& a, float16x8_t& b, float16x8_t& c, float16x8_t& d,
                               float16x8_t& e, float16x8_t& f, float16x8_t& g, float16x8_t& h) noexcept {
    const float16x8x2_t ab = vzipq_f16(a, b);
    const float16x8x2_t cd = vzipq_f16(c, d);
    const float16x8x2_t ef = vzipq_f16(e, f);
    const float16x8x2_t gh = vzipq_f16(g, h);

    const float32x4_t ab0 = vreinterpretq_f32_f16(ab.val[0]);
    const float32x4_t cd0 = vreinterpretq_f32_f16(cd.val[0]);
    const float32x4_t ef0 = vreinterpretq_f32_f16(ef.val[0]);
    const float32x4_t gh0 = vreinterpretq_f32_f16(gh.val[0]);

    const float32x4x2_t abcd0 = vzipq_f32(ab0, cd0);
    const float32x4x2_t efgh0 = vzipq_f32(ef0, gh0);

    a = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(abcd0.val[0]),  vget_low_f32(efgh0.val[0])));
    b = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(abcd0.val[0]), vget_high_f32(efgh0.val[0])));
    c = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(abcd0.val[1]),  vget_low_f32(efgh0.val[1])));
    d = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(abcd0.val[1]), vget_high_f32(efgh0.val[1])));

    const float32x4_t ab1 = vreinterpretq_f32_f16(ab.val[1]);
    const float32x4_t cd1 = vreinterpretq_f32_f16(cd.val[1]);
    const float32x4_t ef1 = vreinterpretq_f32_f16(ef.val[1]);
    const float32x4_t gh1 = vreinterpretq_f32_f16(gh.val[1]);

    const float32x4x2_t abcd1 = vzipq_f32(ab1, cd1);
    const float32x4x2_t efgh1 = vzipq_f32(ef1, gh1);

    e = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(abcd1.val[0]),  vget_low_f32(efgh1.val[0])));
    f = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(abcd1.val[0]), vget_high_f32(efgh1.val[0])));
    g = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(abcd1.val[1]),  vget_low_f32(efgh1.val[1])));
    h = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(abcd1.val[1]), vget_high_f32(efgh1.val[1])));
}

inline void transpose_12x8_f16(float16x8_t& a, float16x8_t& b, float16x8_t& c, float16x8_t& d,
                                float16x8_t& e, float16x8_t& f, float16x8_t& g, float16x8_t& h,
                                float16x8_t& i, float16x8_t& j, float16x8_t& k, float16x8_t& l) noexcept {
    const float16x8x2_t ab = vzipq_f16(a, b);
    const float16x8x2_t cd = vzipq_f16(c, d);
    const float16x8x2_t ef = vzipq_f16(e, f);
    const float16x8x2_t gh = vzipq_f16(g, h);
    const float16x8x2_t ij = vzipq_f16(i, j);
    const float16x8x2_t kl = vzipq_f16(k, l);

    float32x4x2_t ab_f32{ vreinterpretq_f32_f16(ab.val[0]), vreinterpretq_f32_f16(ab.val[1]) };
    float32x4x2_t cd_f32{ vreinterpretq_f32_f16(cd.val[0]), vreinterpretq_f32_f16(cd.val[1]) };
    float32x4x2_t ef_f32{ vreinterpretq_f32_f16(ef.val[0]), vreinterpretq_f32_f16(ef.val[1]) };
    float32x4x2_t gh_f32{ vreinterpretq_f32_f16(gh.val[0]), vreinterpretq_f32_f16(gh.val[1]) };
    float32x4x2_t ij_f32{ vreinterpretq_f32_f16(ij.val[0]), vreinterpretq_f32_f16(ij.val[1]) };
    float32x4x2_t kl_f32{ vreinterpretq_f32_f16(kl.val[0]), vreinterpretq_f32_f16(kl.val[1]) };

    float32x4x2_t abcd_f32 = vzipq_f32(ab_f32.val[0], cd_f32.val[0]);
    float32x4x2_t efgh_f32 = vzipq_f32(ef_f32.val[0], gh_f32.val[0]);
    float32x4x2_t ijkl_f32 = vzipq_f32(ij_f32.val[0], kl_f32.val[0]);

    a = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(abcd_f32.val[0]),  vget_low_f32(efgh_f32.val[0])));
    b = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(ijkl_f32.val[0]),  vget_high_f32(abcd_f32.val[0])));
    c = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(efgh_f32.val[0]), vget_high_f32(ijkl_f32.val[0])));
    d = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(abcd_f32.val[1]),  vget_low_f32(efgh_f32.val[1])));
    e = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(ijkl_f32.val[1]),  vget_high_f32(abcd_f32.val[1])));
    f = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(efgh_f32.val[1]), vget_high_f32(ijkl_f32.val[1])));

    abcd_f32 = vzipq_f32(ab_f32.val[1], cd_f32.val[1]);
    efgh_f32 = vzipq_f32(ef_f32.val[1], gh_f32.val[1]);
    ijkl_f32 = vzipq_f32(ij_f32.val[1], kl_f32.val[1]);

    g = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(abcd_f32.val[0]),  vget_low_f32(efgh_f32.val[0])));
    h = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(ijkl_f32.val[0]),  vget_high_f32(abcd_f32.val[0])));
    i = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(efgh_f32.val[0]), vget_high_f32(ijkl_f32.val[0])));
    j = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(abcd_f32.val[1]),  vget_low_f32(efgh_f32.val[1])));
    k = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(ijkl_f32.val[1]),  vget_high_f32(abcd_f32.val[1])));
    l = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(efgh_f32.val[1]), vget_high_f32(ijkl_f32.val[1])));
}

inline void transpose_16x8_f16(float16x8_t& a, float16x8_t& b, float16x8_t& c, float16x8_t& d,
                                float16x8_t& e, float16x8_t& f, float16x8_t& g, float16x8_t& h,
                                float16x8_t& i, float16x8_t& j, float16x8_t& k, float16x8_t& l,
                                float16x8_t& m, float16x8_t& n, float16x8_t& o, float16x8_t& p) noexcept {
    const float16x8x2_t ab = vzipq_f16(a, b);
    const float16x8x2_t cd = vzipq_f16(c, d);
    const float16x8x2_t ef = vzipq_f16(e, f);
    const float16x8x2_t gh = vzipq_f16(g, h);
    const float16x8x2_t ij = vzipq_f16(i, j);
    const float16x8x2_t kl = vzipq_f16(k, l);
    const float16x8x2_t mn = vzipq_f16(m, n);
    const float16x8x2_t op = vzipq_f16(o, p);

    float32x4x2_t ab_f32{ vreinterpretq_f32_f16(ab.val[0]), vreinterpretq_f32_f16(ab.val[1]) };
    float32x4x2_t cd_f32{ vreinterpretq_f32_f16(cd.val[0]), vreinterpretq_f32_f16(cd.val[1]) };
    float32x4x2_t ef_f32{ vreinterpretq_f32_f16(ef.val[0]), vreinterpretq_f32_f16(ef.val[1]) };
    float32x4x2_t gh_f32{ vreinterpretq_f32_f16(gh.val[0]), vreinterpretq_f32_f16(gh.val[1]) };
    float32x4x2_t ij_f32{ vreinterpretq_f32_f16(ij.val[0]), vreinterpretq_f32_f16(ij.val[1]) };
    float32x4x2_t kl_f32{ vreinterpretq_f32_f16(kl.val[0]), vreinterpretq_f32_f16(kl.val[1]) };
    float32x4x2_t mn_f32{ vreinterpretq_f32_f16(mn.val[0]), vreinterpretq_f32_f16(mn.val[1]) };
    float32x4x2_t op_f32{ vreinterpretq_f32_f16(op.val[0]), vreinterpretq_f32_f16(op.val[1]) };

    float32x4x2_t abcd_f32 = vzipq_f32(ab_f32.val[0], cd_f32.val[0]);
    float32x4x2_t efgh_f32 = vzipq_f32(ef_f32.val[0], gh_f32.val[0]);
    float32x4x2_t ijkl_f32 = vzipq_f32(ij_f32.val[0], kl_f32.val[0]);
    float32x4x2_t mnop_f32 = vzipq_f32(mn_f32.val[0], op_f32.val[0]);

    a = vreinterpretq_f16_f32(abcd_f32.val[0]);
    b = vreinterpretq_f16_f32(efgh_f32.val[0]);
    c = vreinterpretq_f16_f32(ijkl_f32.val[0]);
    d = vreinterpretq_f16_f32(mnop_f32.val[0]);
    e = vreinterpretq_f16_f32(abcd_f32.val[1]);
    f = vreinterpretq_f16_f32(efgh_f32.val[1]);
    g = vreinterpretq_f16_f32(ijkl_f32.val[1]);
    h = vreinterpretq_f16_f32(mnop_f32.val[1]);

    abcd_f32 = vzipq_f32(ab_f32.val[1], cd_f32.val[1]);
    efgh_f32 = vzipq_f32(ef_f32.val[1], gh_f32.val[1]);
    ijkl_f32 = vzipq_f32(ij_f32.val[1], kl_f32.val[1]);
    mnop_f32 = vzipq_f32(mn_f32.val[1], op_f32.val[1]);

    i = vreinterpretq_f16_f32(abcd_f32.val[0]);
    j = vreinterpretq_f16_f32(efgh_f32.val[0]);
    k = vreinterpretq_f16_f32(ijkl_f32.val[0]);
    l = vreinterpretq_f16_f32(mnop_f32.val[0]);
    m = vreinterpretq_f16_f32(abcd_f32.val[1]);
    n = vreinterpretq_f16_f32(efgh_f32.val[1]);
    o = vreinterpretq_f16_f32(ijkl_f32.val[1]);
    p = vreinterpretq_f16_f32(mnop_f32.val[1]);
}

inline void transpose_24x8_f16(float16x8_t& a, float16x8_t& b, float16x8_t& c, float16x8_t& d,
                                float16x8_t& e, float16x8_t& f, float16x8_t& g, float16x8_t& h,
                                float16x8_t& i, float16x8_t& j, float16x8_t& k, float16x8_t& l,
                                float16x8_t& m, float16x8_t& n, float16x8_t& o, float16x8_t& p,
                                float16x8_t& q, float16x8_t& r, float16x8_t& s, float16x8_t& t,
                                float16x8_t& u, float16x8_t& v, float16x8_t& w, float16x8_t& x) noexcept {
    const float16x8x2_t ab = vzipq_f16(a, b);
    const float16x8x2_t cd = vzipq_f16(c, d);
    const float16x8x2_t ef = vzipq_f16(e, f);
    const float16x8x2_t gh = vzipq_f16(g, h);
    const float16x8x2_t ij = vzipq_f16(i, j);
    const float16x8x2_t kl = vzipq_f16(k, l);
    const float16x8x2_t mn = vzipq_f16(m, n);
    const float16x8x2_t op = vzipq_f16(o, p);
    const float16x8x2_t qr = vzipq_f16(q, r);
    const float16x8x2_t st = vzipq_f16(s, t);
    const float16x8x2_t uv = vzipq_f16(u, v);
    const float16x8x2_t wx = vzipq_f16(w, x);

    float32x4x2_t ab_f32{ vreinterpretq_f32_f16(ab.val[0]), vreinterpretq_f32_f16(ab.val[1]) };
    float32x4x2_t cd_f32{ vreinterpretq_f32_f16(cd.val[0]), vreinterpretq_f32_f16(cd.val[1]) };
    float32x4x2_t ef_f32{ vreinterpretq_f32_f16(ef.val[0]), vreinterpretq_f32_f16(ef.val[1]) };
    float32x4x2_t gh_f32{ vreinterpretq_f32_f16(gh.val[0]), vreinterpretq_f32_f16(gh.val[1]) };
    float32x4x2_t ij_f32{ vreinterpretq_f32_f16(ij.val[0]), vreinterpretq_f32_f16(ij.val[1]) };
    float32x4x2_t kl_f32{ vreinterpretq_f32_f16(kl.val[0]), vreinterpretq_f32_f16(kl.val[1]) };
    float32x4x2_t mn_f32{ vreinterpretq_f32_f16(mn.val[0]), vreinterpretq_f32_f16(mn.val[1]) };
    float32x4x2_t op_f32{ vreinterpretq_f32_f16(op.val[0]), vreinterpretq_f32_f16(op.val[1]) };
    float32x4x2_t qr_f32{ vreinterpretq_f32_f16(qr.val[0]), vreinterpretq_f32_f16(qr.val[1]) };
    float32x4x2_t st_f32{ vreinterpretq_f32_f16(st.val[0]), vreinterpretq_f32_f16(st.val[1]) };
    float32x4x2_t uv_f32{ vreinterpretq_f32_f16(uv.val[0]), vreinterpretq_f32_f16(uv.val[1]) };
    float32x4x2_t wx_f32{ vreinterpretq_f32_f16(wx.val[0]), vreinterpretq_f32_f16(wx.val[1]) };

    float32x4x2_t z0 = vzipq_f32(ab_f32.val[0], cd_f32.val[0]);
    float32x4x2_t z1 = vzipq_f32(ef_f32.val[0], gh_f32.val[0]);
    float32x4x2_t z2 = vzipq_f32(ij_f32.val[0], kl_f32.val[0]);
    float32x4x2_t z3 = vzipq_f32(mn_f32.val[0], op_f32.val[0]);
    float32x4x2_t z4 = vzipq_f32(qr_f32.val[0], st_f32.val[0]);
    float32x4x2_t z5 = vzipq_f32(uv_f32.val[0], wx_f32.val[0]);

    a = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(z0.val[0]),  vget_low_f32(z1.val[0])));
    b = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(z2.val[0]),  vget_low_f32(z3.val[0])));
    c = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(z4.val[0]),  vget_low_f32(z5.val[0])));
    d = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(z0.val[0]), vget_high_f32(z1.val[0])));
    e = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(z2.val[0]), vget_high_f32(z3.val[0])));
    f = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(z4.val[0]), vget_high_f32(z5.val[0])));
    g = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(z0.val[1]),  vget_low_f32(z1.val[1])));
    h = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(z2.val[1]),  vget_low_f32(z3.val[1])));
    i = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(z4.val[1]),  vget_low_f32(z5.val[1])));
    j = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(z0.val[1]), vget_high_f32(z1.val[1])));
    k = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(z2.val[1]), vget_high_f32(z3.val[1])));
    l = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(z4.val[1]), vget_high_f32(z5.val[1])));

    z0 = vzipq_f32(ab_f32.val[1], cd_f32.val[1]);
    z1 = vzipq_f32(ef_f32.val[1], gh_f32.val[1]);
    z2 = vzipq_f32(ij_f32.val[1], kl_f32.val[1]);
    z3 = vzipq_f32(mn_f32.val[1], op_f32.val[1]);
    z4 = vzipq_f32(qr_f32.val[1], st_f32.val[1]);
    z5 = vzipq_f32(uv_f32.val[1], wx_f32.val[1]);

    m = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(z0.val[0]),  vget_low_f32(z1.val[0])));
    n = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(z2.val[0]),  vget_low_f32(z3.val[0])));
    o = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(z4.val[0]),  vget_low_f32(z5.val[0])));
    p = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(z0.val[0]), vget_high_f32(z1.val[0])));
    q = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(z2.val[0]), vget_high_f32(z3.val[0])));
    r = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(z4.val[0]), vget_high_f32(z5.val[0])));
    s = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(z0.val[1]),  vget_low_f32(z1.val[1])));
    t = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(z2.val[1]),  vget_low_f32(z3.val[1])));
    u = vreinterpretq_f16_f32(vcombine_f32(vget_low_f32(z4.val[1]),  vget_low_f32(z5.val[1])));
    v = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(z0.val[1]), vget_high_f32(z1.val[1])));
    w = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(z2.val[1]), vget_high_f32(z3.val[1])));
    x = vreinterpretq_f16_f32(vcombine_f32(vget_high_f32(z4.val[1]), vget_high_f32(z5.val[1])));
}

}  // namespace nnops::backend::cpu::aarch64
