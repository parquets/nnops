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
// v_fp32x4 — 128-bit float vector (4 floats), scalar emulation
// ============================================================
struct v_fp32x4 {
    float v[4];

    v_fp32x4() = default;
    explicit v_fp32x4(float s) { v[0] = v[1] = v[2] = v[3] = s; }
    v_fp32x4(float v0, float v1, float v2, float v3) {
        v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
    }
    float operator[](int i) const { return v[i]; }
    float& operator[](int i) { return v[i]; }
};

// ============================================================
// v_fp32x8 — 256-bit float vector (8 floats), scalar emulation
// ============================================================
struct v_fp32x8 {
    float v[8];

    v_fp32x8() = default;
    explicit v_fp32x8(float s) {
        v[0] = v[1] = v[2] = v[3] = v[4] = v[5] = v[6] = v[7] = s;
    }
    v_fp32x8(float v0, float v1, float v2, float v3,
             float v4, float v5, float v6, float v7) {
        v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
        v[4] = v4; v[5] = v5; v[6] = v6; v[7] = v7;
    }
    float operator[](int i) const { return v[i]; }
    float& operator[](int i) { return v[i]; }
};

// ============================================================
// v_fp32x4 operations
// ============================================================
inline v_fp32x4 load_fp32x4(const float* p) {
    return v_fp32x4(p[0], p[1], p[2], p[3]);
}
inline void store(float* p, const v_fp32x4& a) {
    p[0] = a[0]; p[1] = a[1]; p[2] = a[2]; p[3] = a[3];
}
inline v_fp32x4 set1_fp32x4(float s)  { return v_fp32x4(s); }
inline v_fp32x4 zero_fp32x4()         { return v_fp32x4(0.0f); }

inline v_fp32x4 add(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]);
}
inline v_fp32x4 sub(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]);
}
inline v_fp32x4 mul(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(a[0]*b[0], a[1]*b[1], a[2]*b[2], a[3]*b[3]);
}
inline v_fp32x4 div(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(a[0]/b[0], a[1]/b[1], a[2]/b[2], a[3]/b[3]);
}
inline v_fp32x4 fmadd(const v_fp32x4& a, const v_fp32x4& b, const v_fp32x4& c) {
    return v_fp32x4(a[0]*b[0]+c[0], a[1]*b[1]+c[1], a[2]*b[2]+c[2], a[3]*b[3]+c[3]);
}

inline v_fp32x4 min(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(std::min(a[0],b[0]), std::min(a[1],b[1]), std::min(a[2],b[2]), std::min(a[3],b[3]));
}
inline v_fp32x4 max(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(std::max(a[0],b[0]), std::max(a[1],b[1]), std::max(a[2],b[2]), std::max(a[3],b[3]));
}

inline v_fp32x4 abs(const v_fp32x4& a) {
    return v_fp32x4(std::abs(a[0]), std::abs(a[1]), std::abs(a[2]), std::abs(a[3]));
}
inline v_fp32x4 neg(const v_fp32x4& a) {
    return v_fp32x4(-a[0], -a[1], -a[2], -a[3]);
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
inline v_fp32x4 cmplt(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(_scalar_mask(a[0] < b[0]), _scalar_mask(a[1] < b[1]),
                    _scalar_mask(a[2] < b[2]), _scalar_mask(a[3] < b[3]));
}
inline v_fp32x4 cmpgt(const v_fp32x4& a, const v_fp32x4& b) {
    return v_fp32x4(_scalar_mask(a[0] > b[0]), _scalar_mask(a[1] > b[1]),
                    _scalar_mask(a[2] > b[2]), _scalar_mask(a[3] > b[3]));
}

inline v_fp32x4 and_(const v_fp32x4& a, const v_fp32x4& b) {
    uint32_t r[4];
    for (int i = 0; i < 4; ++i) {
        uint32_t ai, bi;
        std::memcpy(&ai, &a.v[i], 4);
        std::memcpy(&bi, &b.v[i], 4);
        ai &= bi;
        std::memcpy(&r[i], &ai, 4);
    }
    v_fp32x4 result;
    std::memcpy(&result.v[0], r, 16);
    return result;
}

inline v_fp32x4 sqrt(const v_fp32x4& a) {
    return v_fp32x4(std::sqrt(a[0]), std::sqrt(a[1]), std::sqrt(a[2]), std::sqrt(a[3]));
}
inline v_fp32x4 exp(const v_fp32x4& a) {
    return v_fp32x4(std::exp(a[0]), std::exp(a[1]), std::exp(a[2]), std::exp(a[3]));
}

// Horizontal sum
inline float reduce_sum(const v_fp32x4& a) {
    return a[0] + a[1] + a[2] + a[3];
}

// ============================================================
// v_fp32x8 operations
// ============================================================
inline v_fp32x8 load_fp32x8(const float* p) {
    return v_fp32x8(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]);
}
inline void store(float* p, const v_fp32x8& a) {
    for (int i = 0; i < 8; ++i) p[i] = a[i];
}
inline v_fp32x8 set1_fp32x8(float s)  { return v_fp32x8(s); }
inline v_fp32x8 zero_fp32x8()         { return v_fp32x8(0.0f); }

inline v_fp32x8 add(const v_fp32x8& a, const v_fp32x8& b) {
    v_fp32x8 r;
    for (int i = 0; i < 8; ++i) r[i] = a[i] + b[i];
    return r;
}
inline v_fp32x8 sub(const v_fp32x8& a, const v_fp32x8& b) {
    v_fp32x8 r;
    for (int i = 0; i < 8; ++i) r[i] = a[i] - b[i];
    return r;
}
inline v_fp32x8 mul(const v_fp32x8& a, const v_fp32x8& b) {
    v_fp32x8 r;
    for (int i = 0; i < 8; ++i) r[i] = a[i] * b[i];
    return r;
}
inline v_fp32x8 div(const v_fp32x8& a, const v_fp32x8& b) {
    v_fp32x8 r;
    for (int i = 0; i < 8; ++i) r[i] = a[i] / b[i];
    return r;
}
inline v_fp32x8 fmadd(const v_fp32x8& a, const v_fp32x8& b, const v_fp32x8& c) {
    v_fp32x8 r;
    for (int i = 0; i < 8; ++i) r[i] = a[i] * b[i] + c[i];
    return r;
}

inline v_fp32x8 min(const v_fp32x8& a, const v_fp32x8& b) {
    v_fp32x8 r;
    for (int i = 0; i < 8; ++i) r[i] = std::min(a[i], b[i]);
    return r;
}
inline v_fp32x8 max(const v_fp32x8& a, const v_fp32x8& b) {
    v_fp32x8 r;
    for (int i = 0; i < 8; ++i) r[i] = std::max(a[i], b[i]);
    return r;
}
inline v_fp32x8 sqrt(const v_fp32x8& a) {
    v_fp32x8 r;
    for (int i = 0; i < 8; ++i) r[i] = std::sqrt(a[i]);
    return r;
}
inline v_fp32x8 exp(const v_fp32x8& a) {
    v_fp32x8 r;
    for (int i = 0; i < 8; ++i) r[i] = std::exp(a[i]);
    return r;
}

inline float reduce_sum(const v_fp32x8& a) {
    float s = 0.0f;
    for (int i = 0; i < 8; ++i) s += a[i];
    return s;
}

// ============================================================
// Float16 conversion utilities (IEEE 754 binary16 <-> binary32)
// Optimized using onnxruntime MLAS techniques:
//   - Denorm magic: float addition hardware normalizes subnormals
//   - Combined bias+rounding: single add for bias adjust + rounding
//   - Magic subtraction: FP subtraction renormalizes half->float subnormals
// ============================================================
namespace {

inline float f16_to_f32(uint16_t val) {
    // onnxruntime MLAS_Half2Float technique:
    // 1. Shift mantissa+exponent into f32 position (<< 13)
    // 2. Add bias adjustment (127-15) << 23
    // 3. For subnormals: promote to normal (+1<<23), then subtract magic to renormalize
    // 4. For Inf/NaN: extra exponent adjustment

    constexpr uint32_t magic_val = 113u << 23;     // magic for subnormal renormalization
    constexpr uint32_t shifted_exp_mask = 0x7C00u << 13;  // exponent field after shift

    uint32_t bits = (val & 0x7FFFu) << 13;          // mantissa + exponent
    uint32_t exp = shifted_exp_mask & bits;          // extract exponent
    bits += (127u - 15u) << 23;                      // adjust bias: f16→f32

    if (exp == shifted_exp_mask) {
        // Inf/NaN: need extra exponent adjustment
        bits += (128u - 16u) << 23;
    } else if (exp == 0) {
        // Zero or subnormal
        if (bits != 0) {
            // Subnormal: promote to normal and renormalize via FP subtraction
            bits += 1u << 23;
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            float magic;
            std::memcpy(&magic, &magic_val, sizeof(magic));
            f -= magic;
            std::memcpy(&bits, &f, sizeof(bits));
        }
    }

    bits |= (val & 0x8000u) << 16;  // sign bit

    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline uint16_t f32_to_f16(float ff) {
    // onnxruntime MLAS_Float2Half technique:
    // 1. For subnormals: add denorm_magic float, then subtract its integer form
    //    → FP addition hardware normalizes and rounds correctly
    // 2. For normals: single add of bias_adjust + 0xFFF + mant_odd
    //    → rounds and adjusts bias in one step, then shift >> 13
    // 3. For Inf/NaN: threshold check → preset values

    constexpr uint32_t f32infty_u   = 255u << 23;
    constexpr uint32_t f16max_u     = (127u + 16u) << 23;  // max f16 representable in f32
    constexpr uint32_t denorm_magic = ((127u - 15u) + (23u - 10u) + 1u) << 23;
    constexpr uint32_t sign_mask    = 0x80000000u;

    uint32_t bits;
    std::memcpy(&bits, &ff, sizeof(bits));

    uint32_t sign = bits & sign_mask;
    bits ^= sign;  // work with absolute value

    uint16_t val;
    if (bits >= f16max_u) {
        // Inf or NaN (all exponent bits set)
        val = (bits > f32infty_u) ? 0x7E00u  // NaN → quiet NaN
                                  : 0x7C00u; // Inf
    } else {
        if (bits < (113u << 23)) {
            // Subnormal or zero: use magic float trick
            // FP addition with round-to-nearest-even normalizes the mantissa
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            float magic_f;
            uint32_t magic_u = denorm_magic;
            std::memcpy(&magic_f, &magic_u, sizeof(magic_f));
            f += magic_f;
            std::memcpy(&bits, &f, sizeof(bits));
            val = static_cast<uint16_t>(bits - magic_u);
        } else {
            // Normal number: bias adjustment + rounding in one step
            uint32_t mant_odd = (bits >> 13) & 1u;  // resulting mantissa LSB for round-to-even
            bits += ((15u - 127u) << 23) + 0xFFFu;  // bias adjust + rounding constant
            bits += mant_odd;                         // round to nearest even
            val = static_cast<uint16_t>(bits >> 13);
        }
    }

    val |= static_cast<uint16_t>(sign >> 16);
    return val;
}

} // anonymous namespace

// ============================================================
// v_fp16x4 — 64-bit float16 vector (4 half floats), scalar emulation
// ============================================================
struct v_fp16x4 {
    uint16_t bits[4];

    v_fp16x4() = default;
    v_fp16x4(uint16_t v0, uint16_t v1, uint16_t v2, uint16_t v3) {
        bits[0] = v0; bits[1] = v1; bits[2] = v2; bits[3] = v3;
    }
};

// ============================================================
// v_fp16x8 — 128-bit float16 vector (8 half floats), scalar emulation
// ============================================================
struct v_fp16x8 {
    uint16_t bits[8];

    v_fp16x8() = default;
    v_fp16x8(uint16_t v0, uint16_t v1, uint16_t v2, uint16_t v3,
             uint16_t v4, uint16_t v5, uint16_t v6, uint16_t v7) {
        bits[0] = v0; bits[1] = v1; bits[2] = v2; bits[3] = v3;
        bits[4] = v4; bits[5] = v5; bits[6] = v6; bits[7] = v7;
    }
};

// ============================================================
// v_fp16x4 operations
// ============================================================
inline v_fp16x4 load_fp16x4(const uint16_t* p) {
    return v_fp16x4(p[0], p[1], p[2], p[3]);
}
inline void store(uint16_t* p, const v_fp16x4& a) {
    p[0] = a.bits[0]; p[1] = a.bits[1]; p[2] = a.bits[2]; p[3] = a.bits[3];
}
inline v_fp16x4 set1_fp16x4(float s) {
    uint16_t h = f32_to_f16(s);
    return v_fp16x4(h, h, h, h);
}
inline v_fp16x4 zero_fp16x4() {
    return v_fp16x4(0, 0, 0, 0);
}

// FP16 <-> FP32 conversion
inline v_fp32x4 cvt_f16_to_f32(const v_fp16x4& a) {
    return v_fp32x4(f16_to_f32(a.bits[0]), f16_to_f32(a.bits[1]),
                    f16_to_f32(a.bits[2]), f16_to_f32(a.bits[3]));
}
inline v_fp16x4 cvt_f32_to_f16(const v_fp32x4& a) {
    return v_fp16x4(f32_to_f16(a[0]), f32_to_f16(a[1]),
                    f32_to_f16(a[2]), f32_to_f16(a[3]));
}

// Arithmetic: convert to f32, operate, convert back
inline v_fp16x4 add(const v_fp16x4& a, const v_fp16x4& b) {
    return v_fp16x4(
        f32_to_f16(f16_to_f32(a.bits[0]) + f16_to_f32(b.bits[0])),
        f32_to_f16(f16_to_f32(a.bits[1]) + f16_to_f32(b.bits[1])),
        f32_to_f16(f16_to_f32(a.bits[2]) + f16_to_f32(b.bits[2])),
        f32_to_f16(f16_to_f32(a.bits[3]) + f16_to_f32(b.bits[3])));
}
inline v_fp16x4 sub(const v_fp16x4& a, const v_fp16x4& b) {
    return v_fp16x4(
        f32_to_f16(f16_to_f32(a.bits[0]) - f16_to_f32(b.bits[0])),
        f32_to_f16(f16_to_f32(a.bits[1]) - f16_to_f32(b.bits[1])),
        f32_to_f16(f16_to_f32(a.bits[2]) - f16_to_f32(b.bits[2])),
        f32_to_f16(f16_to_f32(a.bits[3]) - f16_to_f32(b.bits[3])));
}
inline v_fp16x4 mul(const v_fp16x4& a, const v_fp16x4& b) {
    return v_fp16x4(
        f32_to_f16(f16_to_f32(a.bits[0]) * f16_to_f32(b.bits[0])),
        f32_to_f16(f16_to_f32(a.bits[1]) * f16_to_f32(b.bits[1])),
        f32_to_f16(f16_to_f32(a.bits[2]) * f16_to_f32(b.bits[2])),
        f32_to_f16(f16_to_f32(a.bits[3]) * f16_to_f32(b.bits[3])));
}
inline v_fp16x4 div(const v_fp16x4& a, const v_fp16x4& b) {
    return v_fp16x4(
        f32_to_f16(f16_to_f32(a.bits[0]) / f16_to_f32(b.bits[0])),
        f32_to_f16(f16_to_f32(a.bits[1]) / f16_to_f32(b.bits[1])),
        f32_to_f16(f16_to_f32(a.bits[2]) / f16_to_f32(b.bits[2])),
        f32_to_f16(f16_to_f32(a.bits[3]) / f16_to_f32(b.bits[3])));
}
inline v_fp16x4 fmadd(const v_fp16x4& a, const v_fp16x4& b, const v_fp16x4& c) {
    return v_fp16x4(
        f32_to_f16(f16_to_f32(a.bits[0]) * f16_to_f32(b.bits[0]) + f16_to_f32(c.bits[0])),
        f32_to_f16(f16_to_f32(a.bits[1]) * f16_to_f32(b.bits[1]) + f16_to_f32(c.bits[1])),
        f32_to_f16(f16_to_f32(a.bits[2]) * f16_to_f32(b.bits[2]) + f16_to_f32(c.bits[2])),
        f32_to_f16(f16_to_f32(a.bits[3]) * f16_to_f32(b.bits[3]) + f16_to_f32(c.bits[3])));
}

inline v_fp16x4 min(const v_fp16x4& a, const v_fp16x4& b) {
    return v_fp16x4(
        f32_to_f16(std::min(f16_to_f32(a.bits[0]), f16_to_f32(b.bits[0]))),
        f32_to_f16(std::min(f16_to_f32(a.bits[1]), f16_to_f32(b.bits[1]))),
        f32_to_f16(std::min(f16_to_f32(a.bits[2]), f16_to_f32(b.bits[2]))),
        f32_to_f16(std::min(f16_to_f32(a.bits[3]), f16_to_f32(b.bits[3]))));
}
inline v_fp16x4 max(const v_fp16x4& a, const v_fp16x4& b) {
    return v_fp16x4(
        f32_to_f16(std::max(f16_to_f32(a.bits[0]), f16_to_f32(b.bits[0]))),
        f32_to_f16(std::max(f16_to_f32(a.bits[1]), f16_to_f32(b.bits[1]))),
        f32_to_f16(std::max(f16_to_f32(a.bits[2]), f16_to_f32(b.bits[2]))),
        f32_to_f16(std::max(f16_to_f32(a.bits[3]), f16_to_f32(b.bits[3]))));
}
inline v_fp16x4 abs(const v_fp16x4& a) {
    return v_fp16x4(
        static_cast<uint16_t>(a.bits[0] & 0x7FFFu),
        static_cast<uint16_t>(a.bits[1] & 0x7FFFu),
        static_cast<uint16_t>(a.bits[2] & 0x7FFFu),
        static_cast<uint16_t>(a.bits[3] & 0x7FFFu));
}
inline v_fp16x4 neg(const v_fp16x4& a) {
    return v_fp16x4(
        static_cast<uint16_t>(a.bits[0] ^ 0x8000u),
        static_cast<uint16_t>(a.bits[1] ^ 0x8000u),
        static_cast<uint16_t>(a.bits[2] ^ 0x8000u),
        static_cast<uint16_t>(a.bits[3] ^ 0x8000u));
}

// Comparison: returns FP16 bitmask (0xFFFF for true, 0x0000 for false)
inline v_fp16x4 cmplt(const v_fp16x4& a, const v_fp16x4& b) {
    return v_fp16x4(
        static_cast<uint16_t>(f16_to_f32(a.bits[0]) < f16_to_f32(b.bits[0]) ? 0xFFFFu : 0u),
        static_cast<uint16_t>(f16_to_f32(a.bits[1]) < f16_to_f32(b.bits[1]) ? 0xFFFFu : 0u),
        static_cast<uint16_t>(f16_to_f32(a.bits[2]) < f16_to_f32(b.bits[2]) ? 0xFFFFu : 0u),
        static_cast<uint16_t>(f16_to_f32(a.bits[3]) < f16_to_f32(b.bits[3]) ? 0xFFFFu : 0u));
}
inline v_fp16x4 cmpgt(const v_fp16x4& a, const v_fp16x4& b) {
    return v_fp16x4(
        static_cast<uint16_t>(f16_to_f32(a.bits[0]) > f16_to_f32(b.bits[0]) ? 0xFFFFu : 0u),
        static_cast<uint16_t>(f16_to_f32(a.bits[1]) > f16_to_f32(b.bits[1]) ? 0xFFFFu : 0u),
        static_cast<uint16_t>(f16_to_f32(a.bits[2]) > f16_to_f32(b.bits[2]) ? 0xFFFFu : 0u),
        static_cast<uint16_t>(f16_to_f32(a.bits[3]) > f16_to_f32(b.bits[3]) ? 0xFFFFu : 0u));
}

inline v_fp16x4 sqrt(const v_fp16x4& a) {
    return v_fp16x4(
        f32_to_f16(std::sqrt(f16_to_f32(a.bits[0]))),
        f32_to_f16(std::sqrt(f16_to_f32(a.bits[1]))),
        f32_to_f16(std::sqrt(f16_to_f32(a.bits[2]))),
        f32_to_f16(std::sqrt(f16_to_f32(a.bits[3]))));
}

// Reduce sum: accumulate as float32, then convert back to float16
inline float reduce_sum(const v_fp16x4& a) {
    return f16_to_f32(a.bits[0]) + f16_to_f32(a.bits[1])
         + f16_to_f32(a.bits[2]) + f16_to_f32(a.bits[3]);
}

// ============================================================
// v_fp16x8 operations
// ============================================================
inline v_fp16x8 load_fp16x8(const uint16_t* p) {
    return v_fp16x8(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
}
inline void store(uint16_t* p, const v_fp16x8& a) {
    for (int i = 0; i < 8; ++i) p[i] = a.bits[i];
}
inline v_fp16x8 set1_fp16x8(float s) {
    uint16_t h = f32_to_f16(s);
    return v_fp16x8(h, h, h, h, h, h, h, h);
}
inline v_fp16x8 zero_fp16x8() {
    return v_fp16x8(0, 0, 0, 0, 0, 0, 0, 0);
}

// FP16 <-> FP32 conversion
inline v_fp32x8 cvt_f16_to_f32(const v_fp16x8& a) {
    return v_fp32x8(
        f16_to_f32(a.bits[0]), f16_to_f32(a.bits[1]),
        f16_to_f32(a.bits[2]), f16_to_f32(a.bits[3]),
        f16_to_f32(a.bits[4]), f16_to_f32(a.bits[5]),
        f16_to_f32(a.bits[6]), f16_to_f32(a.bits[7]));
}
inline v_fp16x8 cvt_f32_to_f16(const v_fp32x8& a) {
    return v_fp16x8(
        f32_to_f16(a[0]), f32_to_f16(a[1]),
        f32_to_f16(a[2]), f32_to_f16(a[3]),
        f32_to_f16(a[4]), f32_to_f16(a[5]),
        f32_to_f16(a[6]), f32_to_f16(a[7]));
}

inline v_fp16x8 add(const v_fp16x8& a, const v_fp16x8& b) {
    v_fp16x8 r;
    for (int i = 0; i < 8; ++i)
        r.bits[i] = f32_to_f16(f16_to_f32(a.bits[i]) + f16_to_f32(b.bits[i]));
    return r;
}
inline v_fp16x8 sub(const v_fp16x8& a, const v_fp16x8& b) {
    v_fp16x8 r;
    for (int i = 0; i < 8; ++i)
        r.bits[i] = f32_to_f16(f16_to_f32(a.bits[i]) - f16_to_f32(b.bits[i]));
    return r;
}
inline v_fp16x8 mul(const v_fp16x8& a, const v_fp16x8& b) {
    v_fp16x8 r;
    for (int i = 0; i < 8; ++i)
        r.bits[i] = f32_to_f16(f16_to_f32(a.bits[i]) * f16_to_f32(b.bits[i]));
    return r;
}
inline v_fp16x8 div(const v_fp16x8& a, const v_fp16x8& b) {
    v_fp16x8 r;
    for (int i = 0; i < 8; ++i)
        r.bits[i] = f32_to_f16(f16_to_f32(a.bits[i]) / f16_to_f32(b.bits[i]));
    return r;
}
inline v_fp16x8 fmadd(const v_fp16x8& a, const v_fp16x8& b, const v_fp16x8& c) {
    v_fp16x8 r;
    for (int i = 0; i < 8; ++i)
        r.bits[i] = f32_to_f16(f16_to_f32(a.bits[i]) * f16_to_f32(b.bits[i]) + f16_to_f32(c.bits[i]));
    return r;
}
inline v_fp16x8 min(const v_fp16x8& a, const v_fp16x8& b) {
    v_fp16x8 r;
    for (int i = 0; i < 8; ++i)
        r.bits[i] = f32_to_f16(std::min(f16_to_f32(a.bits[i]), f16_to_f32(b.bits[i])));
    return r;
}
inline v_fp16x8 max(const v_fp16x8& a, const v_fp16x8& b) {
    v_fp16x8 r;
    for (int i = 0; i < 8; ++i)
        r.bits[i] = f32_to_f16(std::max(f16_to_f32(a.bits[i]), f16_to_f32(b.bits[i])));
    return r;
}
inline v_fp16x8 sqrt(const v_fp16x8& a) {
    v_fp16x8 r;
    for (int i = 0; i < 8; ++i)
        r.bits[i] = f32_to_f16(std::sqrt(f16_to_f32(a.bits[i])));
    return r;
}

inline float reduce_sum(const v_fp16x8& a) {
    float s = 0.0f;
    for (int i = 0; i < 8; ++i) s += f16_to_f32(a.bits[i]);
    return s;
}

} // namespace scalar
} // namespace arch
} // namespace simd
} // namespace nnops
