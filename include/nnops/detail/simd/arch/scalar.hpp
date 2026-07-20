#pragma once
/// @file scalar.hpp
/// @brief Scalar C++ fallback for SIMD vector types.
///
/// Provides pure C++ implementations of all vector operations.
/// Used when no SIMD ISA is available at compile time, and as a
/// correctness reference for testing vectorized kernels.

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace nnops {
namespace simd {
namespace arch {
namespace scalar {

// ============================================================
// VecF32x4 — 128-bit float vector (4 floats), scalar emulation
// ============================================================
struct VecF32x4 {
    float v[4];

    VecF32x4() = default;
    explicit VecF32x4(float s) { v[0] = v[1] = v[2] = v[3] = s; }
    VecF32x4(float v0, float v1, float v2, float v3) {
        v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
    }
    float operator[](int i) const { return v[i]; }
    float& operator[](int i) { return v[i]; }
};

// ============================================================
// VecF32x8 — 256-bit float vector (8 floats), scalar emulation
// ============================================================
struct VecF32x8 {
    float v[8];

    VecF32x8() = default;
    explicit VecF32x8(float s) {
        v[0] = v[1] = v[2] = v[3] = v[4] = v[5] = v[6] = v[7] = s;
    }
    VecF32x8(float v0, float v1, float v2, float v3,
             float v4, float v5, float v6, float v7) {
        v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
        v[4] = v4; v[5] = v5; v[6] = v6; v[7] = v7;
    }
    float operator[](int i) const { return v[i]; }
    float& operator[](int i) { return v[i]; }
};

// ============================================================
// VecF32x4 operations
// ============================================================
inline VecF32x4 vec_load_f32x4(const float* p) {
    return VecF32x4(p[0], p[1], p[2], p[3]);
}
inline void vec_store_f32x4(float* p, const VecF32x4& a) {
    p[0] = a[0]; p[1] = a[1]; p[2] = a[2]; p[3] = a[3];
}
inline VecF32x4 vec_set1_f32x4(float s)  { return VecF32x4(s); }
inline VecF32x4 vec_zero_f32x4()         { return VecF32x4(0.0f); }

inline VecF32x4 vec_add_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]);
}
inline VecF32x4 vec_sub_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]);
}
inline VecF32x4 vec_mul_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(a[0]*b[0], a[1]*b[1], a[2]*b[2], a[3]*b[3]);
}
inline VecF32x4 vec_div_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(a[0]/b[0], a[1]/b[1], a[2]/b[2], a[3]/b[3]);
}
inline VecF32x4 vec_fmadd_f32x4(const VecF32x4& a, const VecF32x4& b, const VecF32x4& c) {
    return VecF32x4(a[0]*b[0]+c[0], a[1]*b[1]+c[1], a[2]*b[2]+c[2], a[3]*b[3]+c[3]);
}

inline VecF32x4 vec_min_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(std::min(a[0],b[0]), std::min(a[1],b[1]), std::min(a[2],b[2]), std::min(a[3],b[3]));
}
inline VecF32x4 vec_max_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(std::max(a[0],b[0]), std::max(a[1],b[1]), std::max(a[2],b[2]), std::max(a[3],b[3]));
}

inline VecF32x4 vec_abs_f32x4(const VecF32x4& a) {
    return VecF32x4(std::abs(a[0]), std::abs(a[1]), std::abs(a[2]), std::abs(a[3]));
}
inline VecF32x4 vec_neg_f32x4(const VecF32x4& a) {
    return VecF32x4(-a[0], -a[1], -a[2], -a[3]);
}

// Comparison: returns 0xFFFFFFFF (NaN as float) for true, 0x00000000 for false.
// This matches the SSE _mm_cmplt_ps / NEON vcltq_f32 convention so that
// results are bit-identical between scalar and SIMD backends.
namespace {
inline float _scalar_mask(bool cond) {
    uint32_t v = cond ? 0xFFFFFFFFu : 0u;
    float f;
    std::memcpy(&f, &v, sizeof(f));
    return f;
}
}
inline VecF32x4 vec_cmplt_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(_scalar_mask(a[0] < b[0]), _scalar_mask(a[1] < b[1]),
                    _scalar_mask(a[2] < b[2]), _scalar_mask(a[3] < b[3]));
}
inline VecF32x4 vec_cmpgt_f32x4(const VecF32x4& a, const VecF32x4& b) {
    return VecF32x4(_scalar_mask(a[0] > b[0]), _scalar_mask(a[1] > b[1]),
                    _scalar_mask(a[2] > b[2]), _scalar_mask(a[3] > b[3]));
}

inline VecF32x4 vec_and_f32x4(const VecF32x4& a, const VecF32x4& b) {
    uint32_t r[4];
    for (int i = 0; i < 4; ++i) {
        uint32_t ai, bi;
        std::memcpy(&ai, &a.v[i], 4);
        std::memcpy(&bi, &b.v[i], 4);
        ai &= bi;
        std::memcpy(&r[i], &ai, 4);
    }
    VecF32x4 result;
    std::memcpy(&result.v[0], r, 16);
    return result;
}

inline VecF32x4 vec_sqrt_f32x4(const VecF32x4& a) {
    return VecF32x4(std::sqrt(a[0]), std::sqrt(a[1]), std::sqrt(a[2]), std::sqrt(a[3]));
}
inline VecF32x4 vec_exp_f32x4(const VecF32x4& a) {
    return VecF32x4(std::exp(a[0]), std::exp(a[1]), std::exp(a[2]), std::exp(a[3]));
}

// Horizontal sum
inline float vec_reduce_sum_f32x4(const VecF32x4& a) {
    return a[0] + a[1] + a[2] + a[3];
}

// ============================================================
// VecF32x8 operations
// ============================================================
inline VecF32x8 vec_load_f32x8(const float* p) {
    return VecF32x8(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]);
}
inline void vec_store_f32x8(float* p, const VecF32x8& a) {
    for (int i = 0; i < 8; ++i) p[i] = a[i];
}
inline VecF32x8 vec_set1_f32x8(float s)  { return VecF32x8(s); }
inline VecF32x8 vec_zero_f32x8()         { return VecF32x8(0.0f); }

inline VecF32x8 vec_add_f32x8(const VecF32x8& a, const VecF32x8& b) {
    VecF32x8 r;
    for (int i = 0; i < 8; ++i) r[i] = a[i] + b[i];
    return r;
}
inline VecF32x8 vec_sub_f32x8(const VecF32x8& a, const VecF32x8& b) {
    VecF32x8 r;
    for (int i = 0; i < 8; ++i) r[i] = a[i] - b[i];
    return r;
}
inline VecF32x8 vec_mul_f32x8(const VecF32x8& a, const VecF32x8& b) {
    VecF32x8 r;
    for (int i = 0; i < 8; ++i) r[i] = a[i] * b[i];
    return r;
}
inline VecF32x8 vec_div_f32x8(const VecF32x8& a, const VecF32x8& b) {
    VecF32x8 r;
    for (int i = 0; i < 8; ++i) r[i] = a[i] / b[i];
    return r;
}
inline VecF32x8 vec_fmadd_f32x8(const VecF32x8& a, const VecF32x8& b, const VecF32x8& c) {
    VecF32x8 r;
    for (int i = 0; i < 8; ++i) r[i] = a[i] * b[i] + c[i];
    return r;
}

inline VecF32x8 vec_min_f32x8(const VecF32x8& a, const VecF32x8& b) {
    VecF32x8 r;
    for (int i = 0; i < 8; ++i) r[i] = std::min(a[i], b[i]);
    return r;
}
inline VecF32x8 vec_max_f32x8(const VecF32x8& a, const VecF32x8& b) {
    VecF32x8 r;
    for (int i = 0; i < 8; ++i) r[i] = std::max(a[i], b[i]);
    return r;
}
inline VecF32x8 vec_sqrt_f32x8(const VecF32x8& a) {
    VecF32x8 r;
    for (int i = 0; i < 8; ++i) r[i] = std::sqrt(a[i]);
    return r;
}
inline VecF32x8 vec_exp_f32x8(const VecF32x8& a) {
    VecF32x8 r;
    for (int i = 0; i < 8; ++i) r[i] = std::exp(a[i]);
    return r;
}

inline float vec_reduce_sum_f32x8(const VecF32x8& a) {
    float s = 0.0f;
    for (int i = 0; i < 8; ++i) s += a[i];
    return s;
}

} // namespace scalar
} // namespace arch
} // namespace simd
} // namespace nnops
