/// @file test_simd.cpp
/// @brief Unit tests for the SIMD abstraction layer.
///
/// Tests cover:
///   1. CPU feature detection (at least doesn't crash, baseline features present)
///   2. Scalar fallback — all v_f32x4 and v_f32x8 operations with known values
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
// 2. Scalar backend — v_f32x4 operations with known values
// ============================================================
namespace scal = nnops::simd::arch::scalar;

NNOPS_TEST(simd_scalar_f32x4_load_store) {
    float in[4]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4] = {0, 0, 0, 0};

    scal::v_f32x4 v = scal::v_load_f32x4(in);
    scal::v_store(out, v);

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(out[i], in[i], 1e-6f);
    }

    // Unaligned load/v_store (offset by 1 float from aligned boundary)
    float buf[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::memcpy(buf + 1, in, 4 * sizeof(float));
    scal::v_f32x4 vu = scal::v_load_f32x4(buf + 1);
    float out2[4] = {0};
    scal::v_store(out2, vu);
    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(out2[i], in[i], 1e-6f);
    }
}

NNOPS_TEST(simd_scalar_f32x4_set_constants) {
    scal::v_f32x4 z = scal::v_zero_f32x4();
    for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_NEAR(z[i], 0.0f, 1e-6f); }

    scal::v_f32x4 s = scal::v_set1_f32x4(3.14f);
    for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_NEAR(s[i], 3.14f, 1e-6f); }
}

NNOPS_TEST(simd_scalar_f32x4_arithmetic) {
    scal::v_f32x4 a = scal::v_load_f32x4(kTestVals4);  // {1,2,3,4}
    scal::v_f32x4 b = scal::v_set1_f32x4(2.0f);        // {2,2,2,2}

    scal::v_f32x4 v_add = scal::v_add(a, b);
    NNOPS_EXPECT_NEAR(v_add[0], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(v_add[3], 6.0f, 1e-6f);

    scal::v_f32x4 v_sub = scal::v_sub(a, b);
    NNOPS_EXPECT_NEAR(v_sub[0], -1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(v_sub[3], 2.0f, 1e-6f);

    scal::v_f32x4 v_mul = scal::v_mul(a, b);
    NNOPS_EXPECT_NEAR(v_mul[0], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(v_mul[3], 8.0f, 1e-6f);

    scal::v_f32x4 v_div = scal::v_div(a, b);
    NNOPS_EXPECT_NEAR(v_div[0], 0.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(v_div[3], 2.0f, 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x4_fma) {
    scal::v_f32x4 a = scal::v_load_f32x4(kTestVals4);  // {1,2,3,4}
    scal::v_f32x4 b = scal::v_set1_f32x4(2.0f);        // {2,2,2,2}
    scal::v_f32x4 c = scal::v_set1_f32x4(1.0f);        // {1,1,1,1}

    // v_fmadd: a*b + c = {2+1, 4+1, 6+1, 8+1}
    scal::v_f32x4 r = scal::v_fmadd(a, b, c);
    NNOPS_EXPECT_NEAR(r[0], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(r[1], 5.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(r[2], 7.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(r[3], 9.0f, 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x4_minmax) {
    scal::v_f32x4 a = scal::v_load_f32x4(kTestVals4);  // {1, 2, 3, 4}
    scal::v_f32x4 b = scal::v_load_f32x4(kNegVals4);   // {-1, 0, 3.5, -4.5}

    scal::v_f32x4 mn = scal::v_min(a, b);
    NNOPS_EXPECT_NEAR(mn[0], -1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mn[1], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mn[2], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mn[3], -4.5f, 1e-6f);

    scal::v_f32x4 mx = scal::v_max(a, b);
    NNOPS_EXPECT_NEAR(mx[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mx[1], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mx[2], 3.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(mx[3], 4.0f, 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x4_abs_neg) {
    scal::v_f32x4 a = scal::v_load_f32x4(kNegVals4);  // {-1, 0, 3.5, -4.5}

    scal::v_f32x4 absv = scal::v_abs(a);
    NNOPS_EXPECT_NEAR(absv[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(absv[1], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(absv[2], 3.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(absv[3], 4.5f, 1e-6f);

    scal::v_f32x4 negv = scal::v_neg(a);
    NNOPS_EXPECT_NEAR(negv[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(negv[2], -3.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(negv[3], 4.5f, 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x4_comparison) {
    scal::v_f32x4 a = scal::v_load_f32x4(kTestVals4);  // {1, 2, 3, 4}
    scal::v_f32x4 b = scal::v_set1_f32x4(2.5f);        // {2.5, 2.5, 2.5, 2.5}

    scal::v_f32x4 lt = scal::v_cmplt(a, b);
    // a < 2.5 → {T, T, F, F}
    NNOPS_EXPECT_TRUE(lt[0] != 0.0f);
    NNOPS_EXPECT_TRUE(lt[1] != 0.0f);
    NNOPS_EXPECT_FALSE(lt[2] != 0.0f);
    NNOPS_EXPECT_FALSE(lt[3] != 0.0f);

    scal::v_f32x4 gt = scal::v_cmpgt(a, b);
    // a > 2.5 → {F, F, T, T}
    NNOPS_EXPECT_FALSE(gt[0] != 0.0f);
    NNOPS_EXPECT_FALSE(gt[1] != 0.0f);
    NNOPS_EXPECT_TRUE(gt[2] != 0.0f);
    NNOPS_EXPECT_TRUE(gt[3] != 0.0f);
}

NNOPS_TEST(simd_scalar_f32x4_sqrt) {
    scal::v_f32x4 a = scal::v_load_f32x4(kTestVals4);  // {1, 2, 3, 4}
    scal::v_f32x4 r = scal::v_sqrt(a);
    NNOPS_EXPECT_NEAR(r[0], 1.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(r[1], std::sqrt(2.0f), 1e-4f);
    NNOPS_EXPECT_NEAR(r[2], std::sqrt(3.0f), 1e-4f);
    NNOPS_EXPECT_NEAR(r[3], 2.0f, 1e-4f);
}

NNOPS_TEST(simd_scalar_f32x4_reduce_sum) {
    scal::v_f32x4 a = scal::v_load_f32x4(kTestVals4);  // {1,2,3,4} → sum=10
    float sum = scal::v_reduce_sum(a);
    NNOPS_EXPECT_NEAR(sum, 10.0f, 1e-4f);

    scal::v_f32x4 z = scal::v_zero_f32x4();
    NNOPS_EXPECT_NEAR(scal::v_reduce_sum(z), 0.0f, 1e-6f);
}

// ============================================================
// 3. Scalar backend — v_f32x8 operations
// ============================================================

NNOPS_TEST(simd_scalar_f32x8_load_store) {
    float in[8]  = {1,2,3,4,5,6,7,8};
    float out[8] = {0};
    scal::v_f32x8 v = scal::v_load_f32x8(in);
    scal::v_store(out, v);
    for (int i = 0; i < 8; ++i) {
        NNOPS_EXPECT_NEAR(out[i], float(i+1), 1e-6f);
    }
}

NNOPS_TEST(simd_scalar_f32x8_arithmetic) {
    scal::v_f32x8 a = scal::v_load_f32x8(kTestVals8);  // {1..8}
    scal::v_f32x8 b = scal::v_set1_f32x8(2.0f);

    scal::v_f32x8 v_add = scal::v_add(a, b);
    NNOPS_EXPECT_NEAR(v_add[0], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(v_add[7], 10.0f, 1e-6f);

    scal::v_f32x8 v_mul = scal::v_mul(a, b);
    NNOPS_EXPECT_NEAR(v_mul[0], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(v_mul[7], 16.0f, 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x8_fma) {
    scal::v_f32x8 a = scal::v_load_f32x8(kTestVals8);
    scal::v_f32x8 b = scal::v_set1_f32x8(2.0f);
    scal::v_f32x8 c = scal::v_set1_f32x8(1.0f);

    scal::v_f32x8 r = scal::v_fmadd(a, b, c);
    NNOPS_EXPECT_NEAR(r[0], 3.0f, 1e-6f);   // 1*2+1
    NNOPS_EXPECT_NEAR(r[7], 17.0f, 1e-6f);  // 8*2+1
}

NNOPS_TEST(simd_scalar_f32x8_minmax) {
    scal::v_f32x8 a = scal::v_load_f32x8(kTestVals8);  // {1..8}
    scal::v_f32x8 b = scal::v_set1_f32x8(4.5f);

    scal::v_f32x8 mn = scal::v_min(a, b);
    NNOPS_EXPECT_NEAR(mn[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mn[3], 4.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mn[4], 4.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(mn[7], 4.5f, 1e-6f);

    scal::v_f32x8 mx = scal::v_max(a, b);
    NNOPS_EXPECT_NEAR(mx[0], 4.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(mx[4], 5.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mx[7], 8.0f, 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x8_reduce_sum) {
    scal::v_f32x8 a = scal::v_load_f32x8(kTestVals8);  // 1+2+...+8 = 36
    float sum = scal::v_reduce_sum(a);
    NNOPS_EXPECT_NEAR(sum, 36.0f, 1e-4f);

    scal::v_f32x8 z = scal::v_zero_f32x8();
    NNOPS_EXPECT_NEAR(scal::v_reduce_sum(z), 0.0f, 1e-6f);
}

// ============================================================
// 4. Platform backend vs scalar reference — cross-validation
//    Compares the active platform backend (SSE on x86_64,
//    NEON on AArch64) against the scalar fallback, which
//    serves as the correctness oracle.
// ============================================================

// Helper: compare platform v_f32x4 result against scalar reference
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
    v_f32x4 pa = v_load_f32x4(a_buf);
    v_f32x4 pb = v_load_f32x4(b_buf);
    v_f32x4 pc = v_load_f32x4(c_buf);

    // Scalar reference
    scal::v_f32x4 sa = scal::v_load_f32x4(a_buf);
    scal::v_f32x4 sb = scal::v_load_f32x4(b_buf);
    scal::v_f32x4 sc = scal::v_load_f32x4(c_buf);

    float plat[4], scal_out[4];

    // v_add
    v_store(plat, v_add(pa, pb));
    scal::v_store(scal_out, scal::v_add(sa, sb));
    check_f32x4_matches_scalar(plat, scal_out, "v_add");

    // v_mul
    v_store(plat, v_mul(pa, pb));
    scal::v_store(scal_out, scal::v_mul(sa, sb));
    check_f32x4_matches_scalar(plat, scal_out, "v_mul");

    // v_fmadd
    v_store(plat, v_fmadd(pa, pb, pc));
    scal::v_store(scal_out, scal::v_fmadd(sa, sb, sc));
    check_f32x4_matches_scalar(plat, scal_out, "v_fmadd");

    // v_min
    v_store(plat, v_min(pa, pb));
    scal::v_store(scal_out, scal::v_min(sa, sb));
    check_f32x4_matches_scalar(plat, scal_out, "v_min");

    // v_max
    v_store(plat, v_max(pa, pb));
    scal::v_store(scal_out, scal::v_max(sa, sb));
    check_f32x4_matches_scalar(plat, scal_out, "v_max");
}

NNOPS_TEST(simd_platform_f32x4_sqrt_matches_scalar) {
    float in[4] = {0.5f, 1.0f, 2.0f, 16.0f};

    v_f32x4 pv = v_load_f32x4(in);
    scal::v_f32x4 sv = scal::v_load_f32x4(in);

    float plat[4], scal_out[4];
    v_store(plat, v_sqrt(pv));
    scal::v_store(scal_out, scal::v_sqrt(sv));

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(plat[i], scal_out[i], 1e-3f);  // SIMD v_sqrt may use approximations
    }
}

NNOPS_TEST(simd_platform_f32x4_reduce_matches_scalar) {
    float in[4] = {1.0f, 10.0f, 100.0f, 1000.0f};

    v_f32x4 pv = v_load_f32x4(in);
    scal::v_f32x4 sv = scal::v_load_f32x4(in);

    float psum = v_reduce_sum(pv);
    float ssum = scal::v_reduce_sum(sv);

    NNOPS_EXPECT_NEAR(psum, ssum, 1e-4f);
    NNOPS_EXPECT_NEAR(psum, 1111.0f, 1e-4f);
}

NNOPS_TEST(simd_platform_f32x8_matches_scalar) {
    float a_buf[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b_buf[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    v_f32x8 pa = v_load_f32x8(a_buf);
    v_f32x8 pb = v_load_f32x8(b_buf);
    scal::v_f32x8 sa = scal::v_load_f32x8(a_buf);
    scal::v_f32x8 sb = scal::v_load_f32x8(b_buf);

    float plat[8], scal_out[8];

    // v_add
    v_store(plat, v_add(pa, pb));
    scal::v_store(scal_out, scal::v_add(sa, sb));
    for (int i = 0; i < 8; ++i) {
        NNOPS_EXPECT_NEAR(plat[i], scal_out[i], 1e-5f);
    }

    // v_mul
    v_store(plat, v_mul(pa, pb));
    scal::v_store(scal_out, scal::v_mul(sa, sb));
    for (int i = 0; i < 8; ++i) {
        NNOPS_EXPECT_NEAR(plat[i], scal_out[i], 1e-5f);
    }

    // v_fmadd
    v_store(plat, v_fmadd(pa, pb, pa));
    scal::v_store(scal_out, scal::v_fmadd(sa, sb, sa));
    for (int i = 0; i < 8; ++i) {
        NNOPS_EXPECT_NEAR(plat[i], scal_out[i], 1e-5f);
    }
}

// ============================================================
// 5. Edge cases
// ============================================================

NNOPS_TEST(simd_edge_zero_division) {
    float in[4]   = {1.0f, 2.0f, 0.0f, -3.0f};
    float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    scal::v_f32x4 a = scal::v_load_f32x4(in);
    scal::v_f32x4 z = scal::v_load_f32x4(zero);
    scal::v_f32x4 r = scal::v_div(a, z);

    // 1/0 = +Inf, 0/0 = NaN, -3/0 = -Inf — just verify we didn't crash
    NNOPS_EXPECT_TRUE(std::isinf(r[0]));  // 1/0
    NNOPS_EXPECT_TRUE(std::isnan(r[2]));  // 0/0
}

NNOPS_TEST(simd_edge_large_values) {
    float big_val = 1.0e10f;
    float in[4] = {big_val, -big_val, big_val, -big_val};

    scal::v_f32x4 a = scal::v_load_f32x4(in);
    scal::v_f32x4 b = scal::v_set1_f32x4(2.0f);

    scal::v_f32x4 v_add = scal::v_add(a, a);
    NNOPS_EXPECT_NEAR(v_add[0], 2.0e10f, 1e4f);
    NNOPS_EXPECT_NEAR(v_add[1], -2.0e10f, 1e4f);

    scal::v_f32x4 v_mul = scal::v_mul(a, b);
    NNOPS_EXPECT_NEAR(v_mul[0], 2.0e10f, 1e4f);
    NNOPS_EXPECT_NEAR(v_mul[1], -2.0e10f, 1e4f);
}

NNOPS_TEST(simd_edge_subnormal) {
    float tiny = 1.0e-40f;
    float in[4]  = {tiny, tiny, tiny, tiny};

    scal::v_f32x4 a = scal::v_load_f32x4(in);

    scal::v_f32x4 v_add = scal::v_add(a, a);
    NNOPS_EXPECT_NEAR(v_add[0], 2.0e-40f, 1e-45f);

    scal::v_f32x4 v_mul = scal::v_mul(a, scal::v_set1_f32x4(2.0f));
    NNOPS_EXPECT_NEAR(v_mul[0], 2.0e-40f, 1e-45f);

    scal::v_f32x4 v_sub = scal::v_sub(a, a);
    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(v_sub[i], 0.0f, 1e-45f);
    }
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
// 7. FP16 — Scalar backend v_f16x8 operations
// ============================================================

NNOPS_TEST(simd_scalar_f16x8_load_store) {
    uint16_t in[8];
    for (int i = 0; i < 8; ++i) { in[i] = scal::f32_to_f16(float(i + 1)); }
    uint16_t out[8] = {0};

    scal::v_f16x8 v = scal::v_load_f16x8(in);
    scal::v_store(out, v);

    for (int i = 0; i < 8; ++i) {
        NNOPS_EXPECT_EQ(out[i], in[i]);
    }
}

NNOPS_TEST(simd_scalar_f16x8_arithmetic) {
    uint16_t raw[8];
    for (int i = 0; i < 8; ++i) { raw[i] = scal::f32_to_f16(float(i + 1)); }
    scal::v_f16x8 a = scal::v_load_f16x8(raw);
    scal::v_f16x8 b = scal::v_set1_f16x8(2.0f);

    scal::v_f32x8 add_f32 = scal::v_cvt_f16_to_f32(scal::v_add(a, b));
    NNOPS_EXPECT_NEAR(add_f32[0], 3.0f, 0.01f);
    NNOPS_EXPECT_NEAR(add_f32[7], 10.0f, 0.01f);

    scal::v_f32x8 mul_f32 = scal::v_cvt_f16_to_f32(scal::v_mul(a, b));
    NNOPS_EXPECT_NEAR(mul_f32[0], 2.0f, 0.01f);
    NNOPS_EXPECT_NEAR(mul_f32[7], 16.0f, 0.02f);
}

NNOPS_TEST(simd_scalar_f16x8_fma) {
    scal::v_f16x8 a = scal::v_set1_f16x8(2.0f);
    scal::v_f16x8 b = scal::v_set1_f16x8(3.0f);
    scal::v_f16x8 c = scal::v_set1_f16x8(1.0f);

    scal::v_f32x8 r_f32 = scal::v_cvt_f16_to_f32(scal::v_fmadd(a, b, c));
    for (int i = 0; i < 8; ++i) {
        NNOPS_EXPECT_NEAR(r_f32[i], 7.0f, 0.01f);
    }
}

NNOPS_TEST(simd_scalar_f16x8_reduce_sum) {
    uint16_t raw[8];
    for (int i = 0; i < 8; ++i) { raw[i] = scal::f32_to_f16(float(i + 1)); }
    scal::v_f16x8 a = scal::v_load_f16x8(raw);

    float sum = scal::v_reduce_sum(a);
    NNOPS_EXPECT_NEAR(sum, 36.0f, 0.1f);
}


NNOPS_TEST(simd_platform_f16x8_matches_scalar) {
    uint16_t ah[8], bh[8];
    for (int i = 0; i < 8; ++i) {
        ah[i] = scal::f32_to_f16(float(i + 1));
        bh[i] = scal::f32_to_f16(float(8 - i));
    }

    v_f16x8 pa = v_load_f16x8(ah);
    v_f16x8 pb = v_load_f16x8(bh);
    scal::v_f16x8 sa = scal::v_load_f16x8(ah);
    scal::v_f16x8 sb = scal::v_load_f16x8(bh);

    uint16_t plat[8], sbuf[8];

    // v_add
    v_store(plat, v_add(pa, pb));
    scal::v_store(sbuf, scal::v_add(sa, sb));
    for (int i = 0; i < 8; ++i) { NNOPS_EXPECT_EQ(plat[i], sbuf[i]); }

    // v_mul
    v_store(plat, v_mul(pa, pb));
    scal::v_store(sbuf, scal::v_mul(sa, sb));
    for (int i = 0; i < 8; ++i) { NNOPS_EXPECT_EQ(plat[i], sbuf[i]); }

    // v_fmadd
    v_store(plat, v_fmadd(pa, pb, pa));
    scal::v_store(sbuf, scal::v_fmadd(sa, sb, sa));
    for (int i = 0; i < 8; ++i) { NNOPS_EXPECT_EQ(plat[i], sbuf[i]); }
}

NNOPS_TEST(simd_platform_f16x8_cvt_matches_scalar) {
    float a_vals[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    uint16_t ah[8];
    for (int i = 0; i < 8; ++i) { ah[i] = scal::f32_to_f16(a_vals[i]); }

    v_f16x8 pv = v_load_f16x8(ah);
    scal::v_f16x8 sv = scal::v_load_f16x8(ah);

    // f16 → f32
    float plat[8], scal_out[8];
    v_store(plat, v_cvt_f16_to_f32(pv));
    scal::v_store(scal_out, scal::v_cvt_f16_to_f32(sv));
    for (int i = 0; i < 8; ++i) {
        NNOPS_EXPECT_NEAR(plat[i], scal_out[i], 1e-5f);
    }
}

// ============================================================
// 8. FP16 — Edge cases
// ============================================================


NNOPS_TEST(simd_f16_edge_subnormal) {
    // Small value representable in f16: 1e-4 > f16 v_min positive normal (≈6.1e-5)
    float tiny = 1.0e-4f;
    uint16_t h = scal::f32_to_f16(tiny);
    float back = scal::f16_to_f32(h);
    NNOPS_EXPECT_TRUE(back > 0.0f);
    float rel_err = std::abs(back - tiny) / tiny;
    NNOPS_EXPECT_TRUE(rel_err < 0.2f);
}

NNOPS_TEST(simd_f16_lane_count_constants) {
    NNOPS_EXPECT_EQ(simd_lane_f16x8, 8);
}

NNOPS_TEST(simd_lane_count_constants) {
    NNOPS_EXPECT_EQ(simd_lane_f32x4, 4);
    NNOPS_EXPECT_EQ(simd_lane_f32x8, 8);
}

NNOPS_TEST(simd_default_lane_constants) {
    NNOPS_EXPECT_EQ(simd_default_lane_f32, 8);
    NNOPS_EXPECT_EQ(simd_default_lane_f16, 8);
}

NNOPS_TEST(simd_lane_for_template) {
    NNOPS_EXPECT_EQ((simd_lane_for<float>), 8);
    NNOPS_EXPECT_EQ((simd_lane_for<half>), 8);
}
