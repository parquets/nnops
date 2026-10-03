/// @file test_simd_add_fuse_norm.cpp
/// @brief Unit tests for the fused multiply-add + sum-of-squares SIMD primitive
/// (simd_kernel/simd_add_fuse_norm.hpp).

#include "../src/backend/cpu/simd_kernel/simd_mul_add_reduce.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace nnops::kernel;

namespace {

std::mt19937 rng(7);

std::vector<float> random_f32(int64_t n, float lo = -5.0f, float hi = 5.0f) {
    std::vector<float> v(static_cast<size_t>(n));
    std::uniform_real_distribution<float> dist(lo, hi);
    for (int64_t i = 0; i < n; ++i) {
        v[static_cast<size_t>(i)] = dist(rng);
    }
    return v;
}

/// Reference: out[i] = a[i] + b[i] * scale; returns {sum(out), sum_sq(out)}.
std::pair<float, float> ref_mul_add_reduce_square_sum(
    const std::vector<float>& a, const std::vector<float>& b, float scale,
    std::vector<float>& out)
{
    float sum = 0.0f, sum_sq = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float s = a[i] + b[i] * scale;
        out[i] = s;
        sum += s;
        sum_sq += s * s;
    }
    return {sum, sum_sq};
}

void expect_close(float got, float ref, float rel_tol, float abs_tol) {
    const float tol = abs_tol + rel_tol * std::fabs(ref);
    NNOPS_EXPECT_NEAR(got, ref, tol);
}

// Exercise the float path over a range of row lengths (SIMD widths + tails).
void check_f32(int64_t n, float scale) {
    auto a = random_f32(n);
    auto b = random_f32(n);
    std::vector<float> out(static_cast<size_t>(n), 0.0f);

    auto [sum, sum_sq] =
        mul_add_reduce_square_sum<float>(a.data(), b.data(), scale, out.data(), n);

    std::vector<float> exp(static_cast<size_t>(n), 0.0f);
    auto [esum, esq] = ref_mul_add_reduce_square_sum(a, b, scale, exp);

    for (int64_t i = 0; i < n; ++i) {
        NNOPS_EXPECT_NEAR(out[static_cast<size_t>(i)],
                          exp[static_cast<size_t>(i)], 1e-6f);
    }
    expect_close(sum, esum, 1e-5f, 1e-4f);
    expect_close(sum_sq, esq, 1e-5f, 1e-3f);
}

// Exercise the half path (template instantiation + loose fp16 tolerance).
void check_f16(int64_t n, float scale) {
    auto a_f32 = random_f32(n, -2.0f, 2.0f);
    auto b_f32 = random_f32(n, -2.0f, 2.0f);

    auto a = nnops::test::f32_to_f16(a_f32);
    auto b = nnops::test::f32_to_f16(b_f32);
    std::vector<nnops::backend::cpu::half> out(static_cast<size_t>(n));

    auto [sum, sum_sq] = mul_add_reduce_square_sum<nnops::backend::cpu::half>(
        a.data(), b.data(), scale, out.data(), n);

    // Reference in float; the sum output is rounded to half like the SIMD does.
    std::vector<float> exp(static_cast<size_t>(n), 0.0f);
    auto [esum, esq] = ref_mul_add_reduce_square_sum(a_f32, b_f32, scale, exp);

    for (int64_t i = 0; i < n; ++i) {
        float got = nnops::simd::s_load(&out[static_cast<size_t>(i)]);
        NNOPS_EXPECT_NEAR(got, exp[static_cast<size_t>(i)], 0.02f);
    }
    expect_close(sum, esum, 0.05f, 0.05f);
    expect_close(sum_sq, esq, 0.05f, 0.05f);
}

}  // anonymous namespace

// ============================================================
// f32 tests
// ============================================================

NNOPS_TEST(mul_add_reduce_square_sum_f32_empty) {
    std::vector<float> a, b, out;
    auto [sum, sum_sq] =
        mul_add_reduce_square_sum<float>(a.data(), b.data(), 1.0f, out.data(), 0);
    NNOPS_EXPECT_EQ(sum, 0.0f);
    NNOPS_EXPECT_EQ(sum_sq, 0.0f);
}

NNOPS_TEST(mul_add_reduce_square_sum_f32_basic) {
    check_f32(1, 1.0f);
    check_f32(3, 0.5f);
    check_f32(8, -1.0f);
}

NNOPS_TEST(mul_add_reduce_square_sum_f32_tails) {
    for (int64_t n : {1, 7, 9, 15, 16, 17, 31, 32, 33}) {
        check_f32(n, 2.5f);
    }
}

NNOPS_TEST(mul_add_reduce_square_sum_f32_large) {
    check_f32(64, 0.0f);
    check_f32(257, 0.5f);
    check_f32(1000, -2.0f);
}

// ============================================================
// f16 tests
// ============================================================

NNOPS_TEST(mul_add_reduce_square_sum_f16_basic) {
    check_f16(1, 1.0f);
    check_f16(7, 0.5f);
    check_f16(8, -1.0f);
    check_f16(9, 0.0f);
}

NNOPS_TEST(mul_add_reduce_square_sum_f16_tails) {
    check_f16(16, 2.0f);
    check_f16(17, 0.25f);
    check_f16(33, -1.5f);
    check_f16(65, 1.0f);
}