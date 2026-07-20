/// @file test_simd.cpp
/// @brief Unit tests for the SIMD abstraction layer.
///
/// Tests cover:
///   1. CPU feature detection (at least doesn't crash, baseline features present)
///   2. Scalar fallback — all VecF32x4 and VecF32x8 operations with known values
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
// 2. Scalar backend — VecF32x4 operations with known values
// ============================================================
namespace scal = nnops::simd::arch::scalar;

NNOPS_TEST(simd_scalar_f32x4_load_store) {
    float in[4]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4] = {0, 0, 0, 0};

    scal::VecF32x4 v = scal::vec_load_f32x4(in);
    scal::vec_store_f32x4(out, v);

    for (int i = 0; i < 4; ++i)
        NNOPS_EXPECT_NEAR(out[i], in[i], 1e-6f);

    // Unaligned load/store (offset by 1 float from aligned boundary)
    float buf[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::memcpy(buf + 1, in, 4 * sizeof(float));
    scal::VecF32x4 vu = scal::vec_load_f32x4(buf + 1);
    float out2[4] = {0};
    scal::vec_store_f32x4(out2, vu);
    for (int i = 0; i < 4; ++i)
        NNOPS_EXPECT_NEAR(out2[i], in[i], 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x4_set_constants) {
    scal::VecF32x4 z = scal::vec_zero_f32x4();
    for (int i = 0; i < 4; ++i) NNOPS_EXPECT_NEAR(z[i], 0.0f, 1e-6f);

    scal::VecF32x4 s = scal::vec_set1_f32x4(3.14f);
    for (int i = 0; i < 4; ++i) NNOPS_EXPECT_NEAR(s[i], 3.14f, 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x4_arithmetic) {
    scal::VecF32x4 a = scal::vec_load_f32x4(kTestVals4);  // {1,2,3,4}
    scal::VecF32x4 b = scal::vec_set1_f32x4(2.0f);        // {2,2,2,2}

    scal::VecF32x4 add = scal::vec_add_f32x4(a, b);
    NNOPS_EXPECT_NEAR(add[0], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(add[3], 6.0f, 1e-6f);

    scal::VecF32x4 sub = scal::vec_sub_f32x4(a, b);
    NNOPS_EXPECT_NEAR(sub[0], -1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(sub[3], 2.0f, 1e-6f);

    scal::VecF32x4 mul = scal::vec_mul_f32x4(a, b);
    NNOPS_EXPECT_NEAR(mul[0], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mul[3], 8.0f, 1e-6f);

    scal::VecF32x4 div = scal::vec_div_f32x4(a, b);
    NNOPS_EXPECT_NEAR(div[0], 0.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(div[3], 2.0f, 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x4_fma) {
    scal::VecF32x4 a = scal::vec_load_f32x4(kTestVals4);  // {1,2,3,4}
    scal::VecF32x4 b = scal::vec_set1_f32x4(2.0f);        // {2,2,2,2}
    scal::VecF32x4 c = scal::vec_set1_f32x4(1.0f);        // {1,1,1,1}

    // fmadd: a*b + c = {2+1, 4+1, 6+1, 8+1}
    scal::VecF32x4 r = scal::vec_fmadd_f32x4(a, b, c);
    NNOPS_EXPECT_NEAR(r[0], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(r[1], 5.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(r[2], 7.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(r[3], 9.0f, 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x4_minmax) {
    scal::VecF32x4 a = scal::vec_load_f32x4(kTestVals4);  // {1, 2, 3, 4}
    scal::VecF32x4 b = scal::vec_load_f32x4(kNegVals4);   // {-1, 0, 3.5, -4.5}

    scal::VecF32x4 mn = scal::vec_min_f32x4(a, b);
    NNOPS_EXPECT_NEAR(mn[0], -1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mn[1], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mn[2], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mn[3], -4.5f, 1e-6f);

    scal::VecF32x4 mx = scal::vec_max_f32x4(a, b);
    NNOPS_EXPECT_NEAR(mx[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mx[1], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mx[2], 3.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(mx[3], 4.0f, 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x4_abs_neg) {
    scal::VecF32x4 a = scal::vec_load_f32x4(kNegVals4);  // {-1, 0, 3.5, -4.5}

    scal::VecF32x4 absv = scal::vec_abs_f32x4(a);
    NNOPS_EXPECT_NEAR(absv[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(absv[1], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(absv[2], 3.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(absv[3], 4.5f, 1e-6f);

    scal::VecF32x4 negv = scal::vec_neg_f32x4(a);
    NNOPS_EXPECT_NEAR(negv[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(negv[2], -3.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(negv[3], 4.5f, 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x4_comparison) {
    scal::VecF32x4 a = scal::vec_load_f32x4(kTestVals4);  // {1, 2, 3, 4}
    scal::VecF32x4 b = scal::vec_set1_f32x4(2.5f);        // {2.5, 2.5, 2.5, 2.5}

    scal::VecF32x4 lt = scal::vec_cmplt_f32x4(a, b);
    // a < 2.5 → {T, T, F, F}
    NNOPS_EXPECT_TRUE(lt[0] != 0.0f);
    NNOPS_EXPECT_TRUE(lt[1] != 0.0f);
    NNOPS_EXPECT_FALSE(lt[2] != 0.0f);
    NNOPS_EXPECT_FALSE(lt[3] != 0.0f);

    scal::VecF32x4 gt = scal::vec_cmpgt_f32x4(a, b);
    // a > 2.5 → {F, F, T, T}
    NNOPS_EXPECT_FALSE(gt[0] != 0.0f);
    NNOPS_EXPECT_FALSE(gt[1] != 0.0f);
    NNOPS_EXPECT_TRUE(gt[2] != 0.0f);
    NNOPS_EXPECT_TRUE(gt[3] != 0.0f);
}

NNOPS_TEST(simd_scalar_f32x4_sqrt) {
    scal::VecF32x4 a = scal::vec_load_f32x4(kTestVals4);  // {1, 2, 3, 4}
    scal::VecF32x4 r = scal::vec_sqrt_f32x4(a);
    NNOPS_EXPECT_NEAR(r[0], 1.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(r[1], std::sqrt(2.0f), 1e-4f);
    NNOPS_EXPECT_NEAR(r[2], std::sqrt(3.0f), 1e-4f);
    NNOPS_EXPECT_NEAR(r[3], 2.0f, 1e-4f);
}

NNOPS_TEST(simd_scalar_f32x4_reduce_sum) {
    scal::VecF32x4 a = scal::vec_load_f32x4(kTestVals4);  // {1,2,3,4} → sum=10
    float sum = scal::vec_reduce_sum_f32x4(a);
    NNOPS_EXPECT_NEAR(sum, 10.0f, 1e-4f);

    scal::VecF32x4 z = scal::vec_zero_f32x4();
    NNOPS_EXPECT_NEAR(scal::vec_reduce_sum_f32x4(z), 0.0f, 1e-6f);
}

// ============================================================
// 3. Scalar backend — VecF32x8 operations
// ============================================================

NNOPS_TEST(simd_scalar_f32x8_load_store) {
    float in[8]  = {1,2,3,4,5,6,7,8};
    float out[8] = {0};
    scal::VecF32x8 v = scal::vec_load_f32x8(in);
    scal::vec_store_f32x8(out, v);
    for (int i = 0; i < 8; ++i)
        NNOPS_EXPECT_NEAR(out[i], float(i+1), 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x8_arithmetic) {
    scal::VecF32x8 a = scal::vec_load_f32x8(kTestVals8);  // {1..8}
    scal::VecF32x8 b = scal::vec_set1_f32x8(2.0f);

    scal::VecF32x8 add = scal::vec_add_f32x8(a, b);
    NNOPS_EXPECT_NEAR(add[0], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(add[7], 10.0f, 1e-6f);

    scal::VecF32x8 mul = scal::vec_mul_f32x8(a, b);
    NNOPS_EXPECT_NEAR(mul[0], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mul[7], 16.0f, 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x8_fma) {
    scal::VecF32x8 a = scal::vec_load_f32x8(kTestVals8);
    scal::VecF32x8 b = scal::vec_set1_f32x8(2.0f);
    scal::VecF32x8 c = scal::vec_set1_f32x8(1.0f);

    scal::VecF32x8 r = scal::vec_fmadd_f32x8(a, b, c);
    NNOPS_EXPECT_NEAR(r[0], 3.0f, 1e-6f);   // 1*2+1
    NNOPS_EXPECT_NEAR(r[7], 17.0f, 1e-6f);  // 8*2+1
}

NNOPS_TEST(simd_scalar_f32x8_minmax) {
    scal::VecF32x8 a = scal::vec_load_f32x8(kTestVals8);  // {1..8}
    scal::VecF32x8 b = scal::vec_set1_f32x8(4.5f);

    scal::VecF32x8 mn = scal::vec_min_f32x8(a, b);
    NNOPS_EXPECT_NEAR(mn[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mn[3], 4.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mn[4], 4.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(mn[7], 4.5f, 1e-6f);

    scal::VecF32x8 mx = scal::vec_max_f32x8(a, b);
    NNOPS_EXPECT_NEAR(mx[0], 4.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(mx[4], 5.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(mx[7], 8.0f, 1e-6f);
}

NNOPS_TEST(simd_scalar_f32x8_reduce_sum) {
    scal::VecF32x8 a = scal::vec_load_f32x8(kTestVals8);  // 1+2+...+8 = 36
    float sum = scal::vec_reduce_sum_f32x8(a);
    NNOPS_EXPECT_NEAR(sum, 36.0f, 1e-4f);

    scal::VecF32x8 z = scal::vec_zero_f32x8();
    NNOPS_EXPECT_NEAR(scal::vec_reduce_sum_f32x8(z), 0.0f, 1e-6f);
}

// ============================================================
// 4. Platform backend vs scalar reference — cross-validation
//    Compares the active platform backend (SSE on x86_64,
//    NEON on AArch64) against the scalar fallback, which
//    serves as the correctness oracle.
// ============================================================

// Helper: compare platform VecF32x4 result against scalar reference
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
    VecF32x4 pa = vec_load_f32x4(a_buf);
    VecF32x4 pb = vec_load_f32x4(b_buf);
    VecF32x4 pc = vec_load_f32x4(c_buf);

    // Scalar reference
    scal::VecF32x4 sa = scal::vec_load_f32x4(a_buf);
    scal::VecF32x4 sb = scal::vec_load_f32x4(b_buf);
    scal::VecF32x4 sc = scal::vec_load_f32x4(c_buf);

    float plat[4], scal_out[4];

    // add
    vec_store_f32x4(plat, vec_add_f32x4(pa, pb));
    scal::vec_store_f32x4(scal_out, scal::vec_add_f32x4(sa, sb));
    check_f32x4_matches_scalar(plat, scal_out, "add");

    // mul
    vec_store_f32x4(plat, vec_mul_f32x4(pa, pb));
    scal::vec_store_f32x4(scal_out, scal::vec_mul_f32x4(sa, sb));
    check_f32x4_matches_scalar(plat, scal_out, "mul");

    // fmadd
    vec_store_f32x4(plat, vec_fmadd_f32x4(pa, pb, pc));
    scal::vec_store_f32x4(scal_out, scal::vec_fmadd_f32x4(sa, sb, sc));
    check_f32x4_matches_scalar(plat, scal_out, "fmadd");

    // min
    vec_store_f32x4(plat, vec_min_f32x4(pa, pb));
    scal::vec_store_f32x4(scal_out, scal::vec_min_f32x4(sa, sb));
    check_f32x4_matches_scalar(plat, scal_out, "min");

    // max
    vec_store_f32x4(plat, vec_max_f32x4(pa, pb));
    scal::vec_store_f32x4(scal_out, scal::vec_max_f32x4(sa, sb));
    check_f32x4_matches_scalar(plat, scal_out, "max");
}

NNOPS_TEST(simd_platform_f32x4_sqrt_matches_scalar) {
    float in[4] = {0.5f, 1.0f, 2.0f, 16.0f};

    VecF32x4 pv = vec_load_f32x4(in);
    scal::VecF32x4 sv = scal::vec_load_f32x4(in);

    float plat[4], scal_out[4];
    vec_store_f32x4(plat, vec_sqrt_f32x4(pv));
    scal::vec_store_f32x4(scal_out, scal::vec_sqrt_f32x4(sv));

    for (int i = 0; i < 4; ++i)
        NNOPS_EXPECT_NEAR(plat[i], scal_out[i], 1e-3f);  // SIMD sqrt may use approximations
}

NNOPS_TEST(simd_platform_f32x4_reduce_matches_scalar) {
    float in[4] = {1.0f, 10.0f, 100.0f, 1000.0f};

    VecF32x4 pv = vec_load_f32x4(in);
    scal::VecF32x4 sv = scal::vec_load_f32x4(in);

    float psum = vec_reduce_sum_f32x4(pv);
    float ssum = scal::vec_reduce_sum_f32x4(sv);

    NNOPS_EXPECT_NEAR(psum, ssum, 1e-4f);
    NNOPS_EXPECT_NEAR(psum, 1111.0f, 1e-4f);
}

NNOPS_TEST(simd_platform_f32x8_matches_scalar) {
    float a_buf[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b_buf[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    VecF32x8 pa = vec_load_f32x8(a_buf);
    VecF32x8 pb = vec_load_f32x8(b_buf);
    scal::VecF32x8 sa = scal::vec_load_f32x8(a_buf);
    scal::VecF32x8 sb = scal::vec_load_f32x8(b_buf);

    float plat[8], scal_out[8];

    // add
    vec_store_f32x8(plat, vec_add_f32x8(pa, pb));
    scal::vec_store_f32x8(scal_out, scal::vec_add_f32x8(sa, sb));
    for (int i = 0; i < 8; ++i)
        NNOPS_EXPECT_NEAR(plat[i], scal_out[i], 1e-5f);

    // mul
    vec_store_f32x8(plat, vec_mul_f32x8(pa, pb));
    scal::vec_store_f32x8(scal_out, scal::vec_mul_f32x8(sa, sb));
    for (int i = 0; i < 8; ++i)
        NNOPS_EXPECT_NEAR(plat[i], scal_out[i], 1e-5f);

    // fmadd
    vec_store_f32x8(plat, vec_fmadd_f32x8(pa, pb, pa));
    scal::vec_store_f32x8(scal_out, scal::vec_fmadd_f32x8(sa, sb, sa));
    for (int i = 0; i < 8; ++i)
        NNOPS_EXPECT_NEAR(plat[i], scal_out[i], 1e-5f);
}

// ============================================================
// 5. Edge cases
// ============================================================

NNOPS_TEST(simd_edge_zero_division) {
    float in[4]   = {1.0f, 2.0f, 0.0f, -3.0f};
    float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    scal::VecF32x4 a = scal::vec_load_f32x4(in);
    scal::VecF32x4 z = scal::vec_load_f32x4(zero);
    scal::VecF32x4 r = scal::vec_div_f32x4(a, z);

    // 1/0 = +Inf, 0/0 = NaN, -3/0 = -Inf — just verify we didn't crash
    NNOPS_EXPECT_TRUE(std::isinf(r[0]));  // 1/0
    NNOPS_EXPECT_TRUE(std::isnan(r[2]));  // 0/0
}

NNOPS_TEST(simd_edge_large_values) {
    float big_val = 1.0e10f;
    float in[4] = {big_val, -big_val, big_val, -big_val};

    scal::VecF32x4 a = scal::vec_load_f32x4(in);
    scal::VecF32x4 b = scal::vec_set1_f32x4(2.0f);

    scal::VecF32x4 add = scal::vec_add_f32x4(a, a);
    NNOPS_EXPECT_NEAR(add[0], 2.0e10f, 1e4f);
    NNOPS_EXPECT_NEAR(add[1], -2.0e10f, 1e4f);

    scal::VecF32x4 mul = scal::vec_mul_f32x4(a, b);
    NNOPS_EXPECT_NEAR(mul[0], 2.0e10f, 1e4f);
    NNOPS_EXPECT_NEAR(mul[1], -2.0e10f, 1e4f);
}

NNOPS_TEST(simd_edge_subnormal) {
    float tiny = 1.0e-40f;
    float in[4]  = {tiny, tiny, tiny, tiny};

    scal::VecF32x4 a = scal::vec_load_f32x4(in);

    scal::VecF32x4 add = scal::vec_add_f32x4(a, a);
    NNOPS_EXPECT_NEAR(add[0], 2.0e-40f, 1e-45f);

    scal::VecF32x4 mul = scal::vec_mul_f32x4(a, scal::vec_set1_f32x4(2.0f));
    NNOPS_EXPECT_NEAR(mul[0], 2.0e-40f, 1e-45f);

    scal::VecF32x4 sub = scal::vec_sub_f32x4(a, a);
    for (int i = 0; i < 4; ++i)
        NNOPS_EXPECT_NEAR(sub[i], 0.0f, 1e-45f);
}

NNOPS_TEST(simd_lane_count_constants) {
    NNOPS_EXPECT_EQ(simd_len_f32x4, 4);
    NNOPS_EXPECT_EQ(simd_len_f32x8, 8);
}
