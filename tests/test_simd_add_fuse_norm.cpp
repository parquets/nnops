/// @file test_simd_add_fuse_norm.cpp
/// @brief Unit tests for the fused add + sum-of-squares SIMD primitive
/// (simd_kernel/simd_add_fuse_norm.hpp).

#include "../src/backend/cpu/simd_kernel/simd_add_fuse_norm.hpp"
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

/// Reference: out[i] = a[i] + b[i]; returns {sum(out), sum_sq(out)}.
std::pair<float, float> ref_add_fuse_square_sum(
    const std::vector<float>& a, const std::vector<float>& b,
    std::vector<float>& out)
{
    float sum = 0.0f, sum_sq = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float s = a[i] + b[i];
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
void check_f32(int64_t n) {
    auto a = random_f32(n);
    auto b = random_f32(n);
    std::vector<float> out(static_cast<size_t>(n), 0.0f);

    auto [sum, sum_sq] = add_fuse_square_sum<float>(a.data(), b.data(), out.data(), n);

    std::vector<float> exp(static_cast<size_t>(n), 0.0f);
    auto [esum, esq] = ref_add_fuse_square_sum(a, b, exp);

    for (int64_t i = 0; i < n; ++i) {
        NNOPS_EXPECT_NEAR(out[static_cast<size_t>(i)],
                          exp[static_cast<size_t>(i)], 1e-6f);
    }
    expect_close(sum, esum, 1e-5f, 1e-4f);
    expect_close(sum_sq, esq, 1e-5f, 1e-3f);
}

// Exercise the half path (template instantiation + loose fp16 tolerance).
void check_f16(int64_t n) {
    auto a_f32 = random_f32(n, -2.0f, 2.0f);
    auto b_f32 = random_f32(n, -2.0f, 2.0f);

    auto a = nnops::test::f32_to_f16(a_f32);
    auto b = nnops::test::f32_to_f16(b_f32);
    std::vector<nnops::backend::cpu::half> out(static_cast<size_t>(n));

    auto [sum, sum_sq] = add_fuse_square_sum<nnops::backend::cpu::half>(
        a.data(), b.data(), out.data(), n);

    // Reference in float, then round the sum output to half like the SIMD does.
    std::vector<float> exp(static_cast<size_t>(n), 0.0f);
    auto [esum, esq] = ref_add_fuse_square_sum(a_f32, b_f32, exp);

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

NNOPS_TEST(add_fuse_square_sum_f32_empty) {
    std::vector<float> a, b, out;
    auto [sum, sum_sq] = add_fuse_square_sum<float>(a.data(), b.data(), out.data(), 0);
    NNOPS_EXPECT_EQ(sum, 0.0f);
    NNOPS_EXPECT_EQ(sum_sq, 0.0f);
}

NNOPS_TEST(add_fuse_square_sum_f32_basic) {
    check_f32(1);
    check_f32(3);
    check_f32(8);
}

NNOPS_TEST(add_fuse_square_sum_f32_tails) {
    check_f32(1);
    check_f32(7);
    check_f32(9);
    check_f32(15);
    check_f32(16);
    check_f32(17);
    check_f32(31);
    check_f32(32);
    check_f32(33);
}

NNOPS_TEST(add_fuse_square_sum_f32_large) {
    check_f32(64);
    check_f32(257);
    check_f32(1000);
}

// ============================================================
// f16 tests
// ============================================================

NNOPS_TEST(add_fuse_square_sum_f16_basic) {
    check_f16(1);
    check_f16(7);
    check_f16(8);
    check_f16(9);
}

NNOPS_TEST(add_fuse_square_sum_f16_tails) {
    check_f16(16);
    check_f16(17);
    check_f16(33);
    check_f16(65);
}