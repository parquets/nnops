/// @file test_simd.cpp
/// @brief Unit tests for the SIMD abstraction layer.
///
/// Tests cover:
///   1. CPU feature detection (at least doesn't crash, baseline features present)
///   2. Scalar fallback — all v_fp32x4 and v_fp32x8 operations with known values
///   3. Platform backend vs scalar reference — cross-validation (scalar is the oracle)
///   4. Edge cases: zeros, negatives, large values

#include "common/test_harness.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "nnops/detail/simd/arch/scalar.hpp"

#include <cmath>
#include <cstring>
#include <vector>
#include <cstdint>

using namespace nnops::simd;

// ============================================================
// Helper constants
// ============================================================
namespace {
constexpr float kTestVals4[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
constexpr float kTestVals8[8] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
constexpr float kNegVals4[4]  = { -1.0f, 0.0f, 3.5f, -4.5f };
} // anonymous namespace

// ============================================================
// 1. CPU feature detection
// ============================================================

NNOPS_TEST(simd_cpu_features_smoke) {
    const auto& feats = CpuFeatures::get();
    uint32_t raw = feats.raw();
    (void)raw;

    (void)cpu_has_sse4_1();
    (void)cpu_has_avx();
    (void)cpu_has_avx2();
    (void)cpu_has_fma();
    (void)cpu_has_avx512f();
    (void)cpu_has_neon();

    NNOPS_EXPECT_TRUE(true); // reached here without crashing
}

NNOPS_TEST(simd_cpu_features_x64_baseline) {
#if defined(NNOPS_ARCH_X86_64)
    NNOPS_EXPECT_TRUE(cpu_has_sse4_1());

    if (cpu_has_avx() && cpu_has_avx2()) {
        // Consistency: AVX2 → AVX must be present
        NNOPS_EXPECT_TRUE(cpu_has_avx());
    }
#elif defined(NNOPS_ARCH_AARCH64)
    NNOPS_EXPECT_TRUE(cpu_has_neon());
#else
    NNOPS_EXPECT_TRUE(true);
#endif
}

// ============================================================
// 2. Scalar backend — v_fp32x4 operations with known values
// ============================================================
namespace scal = nnops::simd::arch::scalar;

NNOPS_TEST(simd_scalar_f32x4_load_store) {
    float in[4]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4] = {0, 0, 0, 0};

    scal::v_fp32x4 v = scal::load_fp32x4(in);
    scal::store(out, v);

    for (int i = 0; i < 4; ++i)
        NNOPS_EXPECT_NEAR(out[i], in[i], 1e-6f);

    // Unaligned load/store (offset by 1 float from aligned boundary)
    float buf[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::memcpy(buf + 1, in, 4 * sizeof(float));
    scal::v_fp32x4 vu = scal::load_fp32x4(buf + 1);
    float out2[4] = {0};
    scal::store(out2, vu);
    for (int i = 0; i < 4; ++i)
        NNOPS_EXPECT_NEAR(out2[i], in[i], 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x4_set_constants) {
    scal::v_fp32x4 z = scal::zero_fp32x4();
    for (int i = 0; i < 4; ++i) NNOPS_EXPECT_NEAR(z[i], 0.0f, 1e-6f);

    scal::v_fp32x4 s = scal::set1_fp32x4(3.14f);
    for (int i = 0; i < 4; ++i) NNOPS_EXPECT_NEAR(s[i], 3.14f, 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x4_arithmetic) {
    scal::v_fp32x4 a = scal::load_fp32x4(kTestVals4);  // {1,2,3,4}
    scal::v_fp32x4 b = scal::set1_fp32x4(2.0f);        // {2,2,2,2}

    scal::v_fp32x4 add = scal::add(a, b);
    NNOPS_EXPECT_NEAR(add[0], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(add[3], 6.0f, 1e-6f);

    scal::v_fp32x4 sub = scal::sub(a, b);
    NNOPS_EXPECT_NEAR(sub[0], -1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(sub[3], 2.0f, 1e-6f);

    scal::v_fp32x4 mul = scal::mul(a, b);
    NNOPS_EXPECT_NEAR(mul[0], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mul[3], 8.0f, 1e-6f);

    scal::v_fp32x4 div = scal::div(a, b);
    NNOPS_EXPECT_NEAR(div[0], 0.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(div[3], 2.0f, 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x4_fma) {
    scal::v_fp32x4 a = scal::load_fp32x4(kTestVals4);  // {1,2,3,4}
    scal::v_fp32x4 b = scal::set1_fp32x4(2.0f);        // {2,2,2,2}
    scal::v_fp32x4 c = scal::set1_fp32x4(1.0f);        // {1,1,1,1}

    // fmadd: a*b + c = {2+1, 4+1, 6+1, 8+1}
    scal::v_fp32x4 r = scal::fmadd(a, b, c);
    NNOPS_EXPECT_NEAR(r[0], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(r[1], 5.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(r[2], 7.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(r[3], 9.0f, 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x4_minmax) {
    scal::v_fp32x4 a = scal::load_fp32x4(kTestVals4);  // {1, 2, 3, 4}
    scal::v_fp32x4 b = scal::load_fp32x4(kNegVals4);   // {-1, 0, 3.5, -4.5}

    scal::v_fp32x4 mn = scal::min(a, b);
    NNOPS_EXPECT_NEAR(mn[0], -1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mn[1], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mn[2], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mn[3], -4.5f, 1e-6f);

    scal::v_fp32x4 mx = scal::max(a, b);
    NNOPS_EXPECT_NEAR(mx[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mx[1], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mx[2], 3.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(mx[3], 4.0f, 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x4_abs_neg) {
    scal::v_fp32x4 a = scal::load_fp32x4(kNegVals4);  // {-1, 0, 3.5, -4.5}

    scal::v_fp32x4 absv = scal::abs(a);
    NNOPS_EXPECT_NEAR(absv[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(absv[1], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(absv[2], 3.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(absv[3], 4.5f, 1e-6f);

    scal::v_fp32x4 negv = scal::neg(a);
    NNOPS_EXPECT_NEAR(negv[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(negv[2], -3.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(negv[3], 4.5f, 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x4_comparison) {
    scal::v_fp32x4 a = scal::load_fp32x4(kTestVals4);  // {1, 2, 3, 4}
    scal::v_fp32x4 b = scal::set1_fp32x4(2.5f);        // {2.5, 2.5, 2.5, 2.5}

    scal::v_fp32x4 lt = scal::cmplt(a, b);
    // a < 2.5 → {T, T, F, F}
    NNOPS_EXPECT_TRUE(lt[0] != 0.0f);
    NNOPS_EXPECT_TRUE(lt[1] != 0.0f);
    NNOPS_EXPECT_FALSE(lt[2] != 0.0f);
    NNOPS_EXPECT_FALSE(lt[3] != 0.0f);

    scal::v_fp32x4 gt = scal::cmpgt(a, b);
    // a > 2.5 → {F, F, T, T}
    NNOPS_EXPECT_FALSE(gt[0] != 0.0f);
    NNOPS_EXPECT_FALSE(gt[1] != 0.0f);
    NNOPS_EXPECT_TRUE(gt[2] != 0.0f);
    NNOPS_EXPECT_TRUE(gt[3] != 0.0f);
}

NNOPS_TEST(simd_scalar_f32x4_sqrt) {
    scal::v_fp32x4 a = scal::load_fp32x4(kTestVals4);  // {1, 2, 3, 4}
    scal::v_fp32x4 r = scal::sqrt(a);
    NNOPS_EXPECT_NEAR(r[0], 1.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(r[1], std::sqrt(2.0f), 1e-4f);
    NNOPS_EXPECT_NEAR(r[2], std::sqrt(3.0f), 1e-4f);
    NNOPS_EXPECT_NEAR(r[3], 2.0f, 1e-4f);
}

NNOPS_TEST(simd_scalar_f32x4_reduce_sum) {
    scal::v_fp32x4 a = scal::load_fp32x4(kTestVals4);  // {1,2,3,4} → sum=10
    float sum = scal::reduce_sum(a);
    NNOPS_EXPECT_NEAR(sum, 10.0f, 1e-4f);

    scal::v_fp32x4 z = scal::zero_fp32x4();
    NNOPS_EXPECT_NEAR(scal::reduce_sum(z), 0.0f, 1e-6f);
}

// ============================================================
// 3. Scalar backend — v_fp32x8 operations
// ============================================================

NNOPS_TEST(simd_scalar_f32x8_load_store) {
    float in[8]  = {1,2,3,4,5,6,7,8};
    float out[8] = {0};
    scal::v_fp32x8 v = scal::load_fp32x8(in);
    scal::store(out, v);
    for (int i = 0; i < 8; ++i)
        NNOPS_EXPECT_NEAR(out[i], float(i+1), 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x8_arithmetic) {
    scal::v_fp32x8 a = scal::load_fp32x8(kTestVals8);  // {1..8}
    scal::v_fp32x8 b = scal::set1_fp32x8(2.0f);

    scal::v_fp32x8 add = scal::add(a, b);
    NNOPS_EXPECT_NEAR(add[0], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(add[7], 10.0f, 1e-6f);

    scal::v_fp32x8 mul = scal::mul(a, b);
    NNOPS_EXPECT_NEAR(mul[0], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mul[7], 16.0f, 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x8_fma) {
    scal::v_fp32x8 a = scal::load_fp32x8(kTestVals8);
    scal::v_fp32x8 b = scal::set1_fp32x8(2.0f);
    scal::v_fp32x8 c = scal::set1_fp32x8(1.0f);

    scal::v_fp32x8 r = scal::fmadd(a, b, c);
    NNOPS_EXPECT_NEAR(r[0], 3.0f, 1e-6f);   // 1*2+1
    NNOPS_EXPECT_NEAR(r[7], 17.0f, 1e-6f);  // 8*2+1
}

NNOPS_TEST(simd_scalar_f32x8_minmax) {
    scal::v_fp32x8 a = scal::load_fp32x8(kTestVals8);  // {1..8}
    scal::v_fp32x8 b = scal::set1_fp32x8(4.5f);

    scal::v_fp32x8 mn = scal::min(a, b);
    NNOPS_EXPECT_NEAR(mn[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mn[3], 4.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mn[4], 4.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(mn[7], 4.5f, 1e-6f);

    scal::v_fp32x8 mx = scal::max(a, b);
    NNOPS_EXPECT_NEAR(mx[0], 4.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(mx[4], 5.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mx[7], 8.0f, 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x8_reduce_sum) {
    scal::v_fp32x8 a = scal::load_fp32x8(kTestVals8);  // 1+2+...+8 = 36
    float sum = scal::reduce_sum(a);
    NNOPS_EXPECT_NEAR(sum, 36.0f, 1e-4f);

    scal::v_fp32x8 z = scal::zero_fp32x8();
    NNOPS_EXPECT_NEAR(scal::reduce_sum(z), 0.0f, 1e-6f);
}

// ============================================================
// 4. Platform backend vs scalar reference — cross-validation
//    Compares the active platform backend (SSE on x86_64,
//    NEON on AArch64) against the scalar fallback, which
//    serves as the correctness oracle.
// ============================================================

// Helper: compare platform v_fp32x4 result against scalar reference
static void check_f32x4_matches_scalar(
    float plat_out[4], const float scal_out[4], const char* tag)
{
    for (int i = 0; i < 4; ++i) {
        if (std::abs(plat_out[i] - scal_out[i]) > 1e-5f) {
            throw std::runtime_error(
                std::string("platform[") + std::to_string(i) + "] != scalar[" +
                std::to_string(i) + "] at " + tag + ": plat=" +
                std::to_string(plat_out[i]) + " scal=" + std::to_string(scal_out[i]));
        }
    }
}

NNOPS_TEST(simd_platform_f32x4_matches_scalar) {
    float a_buf[4] = {1.5f, -2.0f, 3.25f, -0.5f};
    float b_buf[4] = {0.5f, 3.0f, -1.0f, 2.0f};
    float c_buf[4] = {0.1f, 0.2f, 0.3f, 0.4f};

    // Platform vectors (SSE on x86_64, NEON on AArch64)
    v_fp32x4 pa = load_fp32x4(a_buf);
    v_fp32x4 pb = load_fp32x4(b_buf);
    v_fp32x4 pc = load_fp32x4(c_buf);

    // Scalar reference
    scal::v_fp32x4 sa = scal::load_fp32x4(a_buf);
    scal::v_fp32x4 sb = scal::load_fp32x4(b_buf);
    scal::v_fp32x4 sc = scal::load_fp32x4(c_buf);

    float plat[4], scal_out[4];

    // add
    store(plat, add(pa, pb));
    scal::store(scal_out, scal::add(sa, sb));
    check_f32x4_matches_scalar(plat, scal_out, "add");

    // mul
    store(plat, mul(pa, pb));
    scal::store(scal_out, scal::mul(sa, sb));
    check_f32x4_matches_scalar(plat, scal_out, "mul");

    // fmadd
    store(plat, fmadd(pa, pb, pc));
    scal::store(scal_out, scal::fmadd(sa, sb, sc));
    check_f32x4_matches_scalar(plat, scal_out, "fmadd");

    // min
    store(plat, min(pa, pb));
    scal::store(scal_out, scal::min(sa, sb));
    check_f32x4_matches_scalar(plat, scal_out, "min");

    // max
    store(plat, max(pa, pb));
    scal::store(scal_out, scal::max(sa, sb));
    check_f32x4_matches_scalar(plat, scal_out, "max");
}

NNOPS_TEST(simd_platform_f32x4_sqrt_matches_scalar) {
    float in[4] = {0.5f, 1.0f, 2.0f, 16.0f};

    v_fp32x4 pv = load_fp32x4(in);
    scal::v_fp32x4 sv = scal::load_fp32x4(in);

    float plat[4], scal_out[4];
    store(plat, sqrt(pv));
    scal::store(scal_out, scal::sqrt(sv));

    for (int i = 0; i < 4; ++i)
        NNOPS_EXPECT_NEAR(plat[i], scal_out[i], 1e-3f);  // SIMD sqrt may use approximations
}

NNOPS_TEST(simd_platform_f32x4_reduce_matches_scalar) {
    float in[4] = {1.0f, 10.0f, 100.0f, 1000.0f};

    v_fp32x4 pv = load_fp32x4(in);
    scal::v_fp32x4 sv = scal::load_fp32x4(in);

    float psum = reduce_sum(pv);
    float ssum = scal::reduce_sum(sv);

    NNOPS_EXPECT_NEAR(psum, ssum, 1e-4f);
    NNOPS_EXPECT_NEAR(psum, 1111.0f, 1e-4f);
}

NNOPS_TEST(simd_platform_f32x8_matches_scalar) {
    float a_buf[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b_buf[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    v_fp32x8 pa = load_fp32x8(a_buf);
    v_fp32x8 pb = load_fp32x8(b_buf);
    scal::v_fp32x8 sa = scal::load_fp32x8(a_buf);
    scal::v_fp32x8 sb = scal::load_fp32x8(b_buf);

    float plat[8], scal_out[8];

    // add
    store(plat, add(pa, pb));
    scal::store(scal_out, scal::add(sa, sb));
    for (int i = 0; i < 8; ++i)
        NNOPS_EXPECT_NEAR(plat[i], scal_out[i], 1e-5f);

    // mul
    store(plat, mul(pa, pb));
    scal::store(scal_out, scal::mul(sa, sb));
    for (int i = 0; i < 8; ++i)
        NNOPS_EXPECT_NEAR(plat[i], scal_out[i], 1e-5f);

    // fmadd
    store(plat, fmadd(pa, pb, pa));
    scal::store(scal_out, scal::fmadd(sa, sb, sa));
    for (int i = 0; i < 8; ++i)
        NNOPS_EXPECT_NEAR(plat[i], scal_out[i], 1e-5f);
}

// ============================================================
// 5. Edge cases
// ============================================================

NNOPS_TEST(simd_edge_zero_division) {
    float in[4]   = {1.0f, 2.0f, 0.0f, -3.0f};
    float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    scal::v_fp32x4 a = scal::load_fp32x4(in);
    scal::v_fp32x4 z = scal::load_fp32x4(zero);
    scal::v_fp32x4 r = scal::div(a, z);

    // 1/0 = +Inf, 0/0 = NaN, -3/0 = -Inf — just verify we didn't crash
    NNOPS_EXPECT_TRUE(std::isinf(r[0]));  // 1/0
    NNOPS_EXPECT_TRUE(std::isnan(r[2]));  // 0/0
}

NNOPS_TEST(simd_edge_large_values) {
    float big_val = 1.0e10f;
    float in[4] = {big_val, -big_val, big_val, -big_val};

    scal::v_fp32x4 a = scal::load_fp32x4(in);
    scal::v_fp32x4 b = scal::set1_fp32x4(2.0f);

    scal::v_fp32x4 add = scal::add(a, a);
    NNOPS_EXPECT_NEAR(add[0], 2.0e10f, 1e4f);
    NNOPS_EXPECT_NEAR(add[1], -2.0e10f, 1e4f);

    scal::v_fp32x4 mul = scal::mul(a, b);
    NNOPS_EXPECT_NEAR(mul[0], 2.0e10f, 1e4f);
    NNOPS_EXPECT_NEAR(mul[1], -2.0e10f, 1e4f);
}

NNOPS_TEST(simd_edge_subnormal) {
    float tiny = 1.0e-40f;
    float in[4]  = {tiny, tiny, tiny, tiny};

    scal::v_fp32x4 a = scal::load_fp32x4(in);

    scal::v_fp32x4 add = scal::add(a, a);
    NNOPS_EXPECT_NEAR(add[0], 2.0e-40f, 1e-45f);

    scal::v_fp32x4 mul = scal::mul(a, scal::set1_fp32x4(2.0f));
    NNOPS_EXPECT_NEAR(mul[0], 2.0e-40f, 1e-45f);

    scal::v_fp32x4 sub = scal::sub(a, a);
    for (int i = 0; i < 4; ++i)
        NNOPS_EXPECT_NEAR(sub[i], 0.0f, 1e-45f);
}

// ============================================================
// 6. FP16 — f16 <-> f32 conversion accuracy (onnxruntime-optimized)
// ============================================================

NNOPS_TEST(simd_f16_f32_conversion_roundtrip) {
    float test_vals[] = {0.0f, -0.0f, 1.0f, -1.0f, 2.0f, 0.5f, 3.14159f,
                         100.0f, -42.5f, 0.001f, 65504.0f, -65504.0f,
                         1.0f/3.0f, 42.0f, 0.1f};

    for (float f_in : test_vals) {
        uint16_t h = scal::f32_to_f16(f_in);
        float f_out = scal::f16_to_f32(h);
        float rel_err = (f_in == 0.0f) ? std::abs(f_out)
            : std::abs(f_out - f_in) / std::max(std::abs(f_in), 1e-10f);
        NNOPS_EXPECT_TRUE(rel_err < 0.002f);
    }
}

NNOPS_TEST(simd_f16_conversion_special_values) {
    // Zero
    uint16_t h0 = scal::f32_to_f16(0.0f);
    NNOPS_EXPECT_EQ(h0, 0x0000u);

    // Infinity
    uint16_t hinf = scal::f32_to_f16(INFINITY);
    NNOPS_EXPECT_EQ(hinf & 0x7C00u, 0x7C00u);
    NNOPS_EXPECT_EQ(hinf & 0x03FFu, 0u);
    NNOPS_EXPECT_TRUE(std::isinf(scal::f16_to_f32(hinf)));

    // NaN
    uint16_t hnan = scal::f32_to_f16(NAN);
    NNOPS_EXPECT_EQ(hnan & 0x7C00u, 0x7C00u);
    NNOPS_EXPECT_TRUE((hnan & 0x03FFu) != 0);
    NNOPS_EXPECT_TRUE(std::isnan(scal::f16_to_f32(hnan)));

    // Subnormal: very small float flushed to zero
    NNOPS_EXPECT_EQ(scal::f32_to_f16(1.0e-8f), 0u);
}

// ============================================================
// 7. FP16 — Scalar backend v_fp16x4 operations
// ============================================================

// Helper: make an f16 array from float values
static void make_f16_arr4(uint16_t (&out)[4], float v0, float v1, float v2, float v3) {
    out[0] = scal::f32_to_f16(v0); out[1] = scal::f32_to_f16(v1);
    out[2] = scal::f32_to_f16(v2); out[3] = scal::f32_to_f16(v3);
}

NNOPS_TEST(simd_scalar_f16x4_load_store) {
    uint16_t in[4];
    make_f16_arr4(in, 1.0f, 2.0f, 3.0f, 4.0f);
    uint16_t out[4] = {0, 0, 0, 0};

    scal::v_fp16x4 v = scal::load_fp16x4(in);
    scal::store(out, v);

    for (int i = 0; i < 4; ++i)
        NNOPS_EXPECT_EQ(out[i], in[i]);
}

NNOPS_TEST(simd_scalar_f16x4_arithmetic) {
    uint16_t raw[4];
    make_f16_arr4(raw, 1.0f, 2.0f, 3.0f, 4.0f);
    scal::v_fp16x4 a = scal::load_fp16x4(raw);
    scal::v_fp16x4 b = scal::set1_fp16x4(2.0f);

    // add: {3, 4, 5, 6}
    scal::v_fp32x4 add_f32 = scal::cvt_f16_to_f32(scal::add(a, b));
    NNOPS_EXPECT_NEAR(add_f32[0], 3.0f, 0.01f);
    NNOPS_EXPECT_NEAR(add_f32[3], 6.0f, 0.01f);

    // mul: {2, 4, 6, 8}
    scal::v_fp32x4 mul_f32 = scal::cvt_f16_to_f32(scal::mul(a, b));
    NNOPS_EXPECT_NEAR(mul_f32[0], 2.0f, 0.01f);
    NNOPS_EXPECT_NEAR(mul_f32[3], 8.0f, 0.01f);

    // div: {0.5, 1.0, 1.5, 2.0}
    scal::v_fp32x4 div_f32 = scal::cvt_f16_to_f32(scal::div(a, b));
    NNOPS_EXPECT_NEAR(div_f32[0], 0.5f, 0.005f);
    NNOPS_EXPECT_NEAR(div_f32[3], 2.0f, 0.01f);
}

NNOPS_TEST(simd_scalar_f16x4_fma) {
    scal::v_fp16x4 a = scal::set1_fp16x4(2.0f);
    scal::v_fp16x4 b = scal::set1_fp16x4(3.0f);
    scal::v_fp16x4 c = scal::set1_fp16x4(1.0f);
    // fmadd: 2*3 + 1 = 7
    scal::v_fp32x4 r_f32 = scal::cvt_f16_to_f32(scal::fmadd(a, b, c));
    NNOPS_EXPECT_NEAR(r_f32[0], 7.0f, 0.01f);
    NNOPS_EXPECT_NEAR(r_f32[3], 7.0f, 0.01f);
}

NNOPS_TEST(simd_scalar_f16x4_minmax) {
    uint16_t raw[4];
    make_f16_arr4(raw, 1.0f, 5.0f, 3.0f, -2.0f);
    scal::v_fp16x4 a = scal::load_fp16x4(raw);
    scal::v_fp16x4 b = scal::set1_fp16x4(2.5f);

    scal::v_fp32x4 mn_f32 = scal::cvt_f16_to_f32(scal::min(a, b));
    NNOPS_EXPECT_NEAR(mn_f32[0], 1.0f, 0.01f);
    NNOPS_EXPECT_NEAR(mn_f32[1], 2.5f, 0.01f);
    NNOPS_EXPECT_NEAR(mn_f32[3], -2.0f, 0.01f);

    scal::v_fp32x4 mx_f32 = scal::cvt_f16_to_f32(scal::max(a, b));
    NNOPS_EXPECT_NEAR(mx_f32[0], 2.5f, 0.01f);
    NNOPS_EXPECT_NEAR(mx_f32[1], 5.0f, 0.01f);
}

NNOPS_TEST(simd_scalar_f16x4_abs_neg) {
    uint16_t raw[4];
    make_f16_arr4(raw, -1.0f, 2.0f, -3.5f, 0.0f);
    scal::v_fp16x4 a = scal::load_fp16x4(raw);

    scal::v_fp32x4 abs_f32 = scal::cvt_f16_to_f32(scal::abs(a));
    NNOPS_EXPECT_NEAR(abs_f32[0], 1.0f, 0.01f);
    NNOPS_EXPECT_NEAR(abs_f32[2], 3.5f, 0.02f);
    NNOPS_EXPECT_NEAR(abs_f32[3], 0.0f, 0.01f);

    scal::v_fp32x4 neg_f32 = scal::cvt_f16_to_f32(scal::neg(a));
    NNOPS_EXPECT_NEAR(neg_f32[0], 1.0f, 0.01f);
    NNOPS_EXPECT_NEAR(neg_f32[1], -2.0f, 0.01f);
}

NNOPS_TEST(simd_scalar_f16x4_comparison) {
    uint16_t raw[4];
    make_f16_arr4(raw, 1.0f, 2.0f, 3.0f, 4.0f);
    scal::v_fp16x4 a = scal::load_fp16x4(raw);
    scal::v_fp16x4 b = scal::set1_fp16x4(2.5f);

    // a < 2.5 → {T, T, F, F}
    scal::v_fp16x4 lt = scal::cmplt(a, b);
    NNOPS_EXPECT_TRUE(lt.bits[0] != 0);
    NNOPS_EXPECT_TRUE(lt.bits[1] != 0);
    NNOPS_EXPECT_FALSE(lt.bits[2] != 0);
    NNOPS_EXPECT_FALSE(lt.bits[3] != 0);

    // a > 2.5 → {F, F, T, T}
    scal::v_fp16x4 gt = scal::cmpgt(a, b);
    NNOPS_EXPECT_FALSE(gt.bits[0] != 0);
    NNOPS_EXPECT_FALSE(gt.bits[1] != 0);
    NNOPS_EXPECT_TRUE(gt.bits[2] != 0);
    NNOPS_EXPECT_TRUE(gt.bits[3] != 0);
}

NNOPS_TEST(simd_scalar_f16x4_sqrt) {
    uint16_t raw[4];
    make_f16_arr4(raw, 1.0f, 4.0f, 9.0f, 16.0f);
    scal::v_fp16x4 a = scal::load_fp16x4(raw);
    scal::v_fp32x4 r_f32 = scal::cvt_f16_to_f32(scal::sqrt(a));
    NNOPS_EXPECT_NEAR(r_f32[0], 1.0f, 0.01f);
    NNOPS_EXPECT_NEAR(r_f32[1], 2.0f, 0.01f);
    NNOPS_EXPECT_NEAR(r_f32[2], 3.0f, 0.01f);
    NNOPS_EXPECT_NEAR(r_f32[3], 4.0f, 0.01f);
}

NNOPS_TEST(simd_scalar_f16x4_reduce_sum) {
    uint16_t raw[4];
    make_f16_arr4(raw, 1.0f, 2.0f, 3.0f, 4.0f);
    scal::v_fp16x4 a = scal::load_fp16x4(raw);
    float sum = scal::reduce_sum(a);
    NNOPS_EXPECT_NEAR(sum, 10.0f, 0.05f);

    scal::v_fp16x4 z = scal::zero_fp16x4();
    NNOPS_EXPECT_NEAR(scal::reduce_sum(z), 0.0f, 1e-6f);
}

NNOPS_TEST(simd_scalar_f16x4_cvt_roundtrip) {
    float f32_in[4] = {1.0f, -2.5f, 100.0f, 0.125f};
    scal::v_fp32x4 vec_f32 = scal::load_fp32x4(f32_in);

    scal::v_fp16x4 vec_f16 = scal::cvt_f32_to_f16(vec_f32);
    scal::v_fp32x4 vec_back = scal::cvt_f16_to_f32(vec_f16);

    for (int i = 0; i < 4; ++i) {
        float rel_err = std::abs(vec_back[i] - f32_in[i])
                      / std::max(std::abs(f32_in[i]), 1e-10f);
        NNOPS_EXPECT_TRUE(rel_err < 0.002f);
    }
}

// ============================================================
// 8. FP16 — Scalar backend v_fp16x8 operations
// ============================================================

NNOPS_TEST(simd_scalar_f16x8_load_store) {
    uint16_t in[8];
    for (int i = 0; i < 8; ++i) in[i] = scal::f32_to_f16(float(i + 1));
    uint16_t out[8] = {0};

    scal::v_fp16x8 v = scal::load_fp16x8(in);
    scal::store(out, v);

    for (int i = 0; i < 8; ++i)
        NNOPS_EXPECT_EQ(out[i], in[i]);
}

NNOPS_TEST(simd_scalar_f16x8_arithmetic) {
    uint16_t raw[8];
    for (int i = 0; i < 8; ++i) raw[i] = scal::f32_to_f16(float(i + 1));
    scal::v_fp16x8 a = scal::load_fp16x8(raw);
    scal::v_fp16x8 b = scal::set1_fp16x8(2.0f);

    scal::v_fp32x8 add_f32 = scal::cvt_f16_to_f32(scal::add(a, b));
    NNOPS_EXPECT_NEAR(add_f32[0], 3.0f, 0.01f);
    NNOPS_EXPECT_NEAR(add_f32[7], 10.0f, 0.01f);

    scal::v_fp32x8 mul_f32 = scal::cvt_f16_to_f32(scal::mul(a, b));
    NNOPS_EXPECT_NEAR(mul_f32[0], 2.0f, 0.01f);
    NNOPS_EXPECT_NEAR(mul_f32[7], 16.0f, 0.02f);
}

NNOPS_TEST(simd_scalar_f16x8_fma) {
    scal::v_fp16x8 a = scal::set1_fp16x8(2.0f);
    scal::v_fp16x8 b = scal::set1_fp16x8(3.0f);
    scal::v_fp16x8 c = scal::set1_fp16x8(1.0f);

    scal::v_fp32x8 r_f32 = scal::cvt_f16_to_f32(scal::fmadd(a, b, c));
    for (int i = 0; i < 8; ++i)
        NNOPS_EXPECT_NEAR(r_f32[i], 7.0f, 0.01f);
}

NNOPS_TEST(simd_scalar_f16x8_reduce_sum) {
    uint16_t raw[8];
    for (int i = 0; i < 8; ++i) raw[i] = scal::f32_to_f16(float(i + 1));
    scal::v_fp16x8 a = scal::load_fp16x8(raw);

    float sum = scal::reduce_sum(a);
    NNOPS_EXPECT_NEAR(sum, 36.0f, 0.1f);
}

// ============================================================
// 9. FP16 — Platform backend vs scalar reference cross-validation
// ============================================================

NNOPS_TEST(simd_platform_f16x4_matches_scalar) {
    float a_vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_vals[4] = {0.5f, 1.5f, 2.5f, 3.5f};

    uint16_t ah[4], bh[4];
    for (int i = 0; i < 4; ++i) {
        ah[i] = scal::f32_to_f16(a_vals[i]);
        bh[i] = scal::f32_to_f16(b_vals[i]);
    }

    v_fp16x4 pa = load_fp16x4(ah);
    v_fp16x4 pb = load_fp16x4(bh);
    scal::v_fp16x4 sa = scal::load_fp16x4(ah);
    scal::v_fp16x4 sb = scal::load_fp16x4(bh);

    uint16_t plat[4], sbuf[4];

    // add
    store(plat, add(pa, pb));
    scal::store(sbuf, scal::add(sa, sb));
    for (int i = 0; i < 4; ++i) NNOPS_EXPECT_EQ(plat[i], sbuf[i]);

    // mul
    store(plat, mul(pa, pb));
    scal::store(sbuf, scal::mul(sa, sb));
    for (int i = 0; i < 4; ++i) NNOPS_EXPECT_EQ(plat[i], sbuf[i]);

    // fmadd
    store(plat, fmadd(pa, pb, pa));
    scal::store(sbuf, scal::fmadd(sa, sb, sa));
    for (int i = 0; i < 4; ++i) NNOPS_EXPECT_EQ(plat[i], sbuf[i]);

    // min
    store(plat, min(pa, pb));
    scal::store(sbuf, scal::min(sa, sb));
    for (int i = 0; i < 4; ++i) NNOPS_EXPECT_EQ(plat[i], sbuf[i]);

    // max
    store(plat, max(pa, pb));
    scal::store(sbuf, scal::max(sa, sb));
    for (int i = 0; i < 4; ++i) NNOPS_EXPECT_EQ(plat[i], sbuf[i]);
}

NNOPS_TEST(simd_platform_f16x4_cvt_matches_scalar) {
    float a_vals[4] = {0.5f, 1.0f, 2.0f, 16.0f};
    uint16_t ah[4];
    for (int i = 0; i < 4; ++i) ah[i] = scal::f32_to_f16(a_vals[i]);

    v_fp16x4 pv = load_fp16x4(ah);
    scal::v_fp16x4 sv = scal::load_fp16x4(ah);

    // f16 → f32: store to float arrays for cross-type comparison
    float plat[4], scal_out[4];
    store(plat, cvt_f16_to_f32(pv));
    scal::store(scal_out, scal::cvt_f16_to_f32(sv));
    for (int i = 0; i < 4; ++i)
        NNOPS_EXPECT_NEAR(plat[i], scal_out[i], 1e-6f);

    // f32 → f16: compare bit-identical uint16_t output
    uint16_t plat_h[4], scal_h[4];
    {
        scal::v_fp32x4 sf32 = scal::load_fp32x4(a_vals);
        store(plat_h, cvt_f32_to_f16(sf32));
        scal::store(scal_h, scal::cvt_f32_to_f16(sf32));
        for (int i = 0; i < 4; ++i)
            NNOPS_EXPECT_EQ(plat_h[i], scal_h[i]);
    }
}

NNOPS_TEST(simd_platform_f16x8_matches_scalar) {
    uint16_t ah[8], bh[8];
    for (int i = 0; i < 8; ++i) {
        ah[i] = scal::f32_to_f16(float(i + 1));
        bh[i] = scal::f32_to_f16(float(8 - i));
    }

    v_fp16x8 pa = load_fp16x8(ah);
    v_fp16x8 pb = load_fp16x8(bh);
    scal::v_fp16x8 sa = scal::load_fp16x8(ah);
    scal::v_fp16x8 sb = scal::load_fp16x8(bh);

    uint16_t plat[8], sbuf[8];

    // add
    store(plat, add(pa, pb));
    scal::store(sbuf, scal::add(sa, sb));
    for (int i = 0; i < 8; ++i) NNOPS_EXPECT_EQ(plat[i], sbuf[i]);

    // mul
    store(plat, mul(pa, pb));
    scal::store(sbuf, scal::mul(sa, sb));
    for (int i = 0; i < 8; ++i) NNOPS_EXPECT_EQ(plat[i], sbuf[i]);

    // fmadd
    store(plat, fmadd(pa, pb, pa));
    scal::store(sbuf, scal::fmadd(sa, sb, sa));
    for (int i = 0; i < 8; ++i) NNOPS_EXPECT_EQ(plat[i], sbuf[i]);
}

NNOPS_TEST(simd_platform_f16x8_cvt_matches_scalar) {
    float a_vals[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    uint16_t ah[8];
    for (int i = 0; i < 8; ++i) ah[i] = scal::f32_to_f16(a_vals[i]);

    v_fp16x8 pv = load_fp16x8(ah);
    scal::v_fp16x8 sv = scal::load_fp16x8(ah);

    // f16 → f32
    float plat[8], scal_out[8];
    store(plat, cvt_f16_to_f32(pv));
    scal::store(scal_out, scal::cvt_f16_to_f32(sv));
    for (int i = 0; i < 8; ++i)
        NNOPS_EXPECT_NEAR(plat[i], scal_out[i], 1e-5f);
}

// ============================================================
// 10. FP16 — Edge cases
// ============================================================

NNOPS_TEST(simd_f16_edge_zero_set1) {
    scal::v_fp16x4 z = scal::zero_fp16x4();
    for (int i = 0; i < 4; ++i) NNOPS_EXPECT_EQ(z.bits[i], 0u);

    scal::v_fp16x8 z8 = scal::zero_fp16x8();
    for (int i = 0; i < 8; ++i) NNOPS_EXPECT_EQ(z8.bits[i], 0u);

    scal::v_fp16x4 s = scal::set1_fp16x4(1.0f);
    NNOPS_EXPECT_NEAR(scal::f16_to_f32(s.bits[0]), 1.0f, 0.01f);
}

NNOPS_TEST(simd_f16_edge_infinity) {
    uint16_t hinf = scal::f32_to_f16(INFINITY);
    uint16_t arr[4] = {hinf, hinf, hinf, hinf};
    scal::v_fp16x4 v = scal::load_fp16x4(arr);

    scal::v_fp16x4 add = scal::add(v, v);
    NNOPS_EXPECT_TRUE(std::isinf(scal::f16_to_f32(add.bits[0])));
}

NNOPS_TEST(simd_f16_edge_subnormal) {
    // Small value representable in f16: 1e-4 > f16 min positive normal (≈6.1e-5)
    float tiny = 1.0e-4f;
    uint16_t h = scal::f32_to_f16(tiny);
    float back = scal::f16_to_f32(h);
    NNOPS_EXPECT_TRUE(back > 0.0f);
    float rel_err = std::abs(back - tiny) / tiny;
    NNOPS_EXPECT_TRUE(rel_err < 0.2f);
}

NNOPS_TEST(simd_f16_lane_count_constants) {
    NNOPS_EXPECT_EQ(simd_len_fp16x4, 4);
    NNOPS_EXPECT_EQ(simd_len_fp16x8, 8);
}

NNOPS_TEST(simd_lane_count_constants) {
    NNOPS_EXPECT_EQ(simd_len_fp32x4, 4);
    NNOPS_EXPECT_EQ(simd_len_fp32x8, 8);
}
