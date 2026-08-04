/// @file test_tiled_eltwise.cpp
/// @brief Unit tests for tiled eltwise SIMD kernels (simd_kernel/simd_eltwise.hpp).

#include "../src/backend/cpu/simd_kernel/simd_eltwise.hpp"
#include "common/test_harness.hpp"

#include <cmath>
#include <vector>
#include <random>

using namespace nnops::kernel;

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

static std::mt19937 rng(43);

static std::vector<float> random_f32(int64_t n, float lo = -5.0f, float hi = 5.0f) {
    std::vector<float> v(n);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (int64_t i = 0; i < n; ++i) v[i] = dist(rng);
    return v;
}

template <typename Kernel, typename Ref>
static void verify_kernel(const char* test_name,
                          const std::vector<float>& a_in,
                          const std::vector<float>& b_in,
                          int64_t m, int64_t n,
                          int64_t a_pitch, int64_t b_pitch, int64_t out_pitch,
                          bool add_to,
                          Kernel&& kernel, Ref&& ref_fn,
                          float tol = 1e-4f)
{
    int64_t out_size = (m - 1) * out_pitch + n;

    std::vector<float> out(out_size, add_to ? 0.1f : 0.0f);
    std::vector<float> expected(out_size);
    for (int64_t i = 0; i < out_size; ++i) expected[i] = out[i];

    // Reference: scalar row-by-row
    for (int64_t r = 0; r < m; ++r) {
        const float* a_row = a_in.data() + r * a_pitch;
        const float* b_row = b_in.data() + r * b_pitch;
        float*       o_row = expected.data() + r * out_pitch;
        for (int64_t j = 0; j < n; ++j) {
            float val = ref_fn(a_row[j], b_row[j]);
            if (add_to)
                o_row[j] += val;
            else
                o_row[j] = val;
        }
    }

    // Tiled kernel
    kernel(a_in.data(), b_in.data(), out.data(),
           m, n, a_pitch, b_pitch, out_pitch, add_to);

    // Compare
    for (int64_t r = 0; r < m; ++r) {
        float* out_row = out.data() + r * out_pitch;
        float* exp_row = expected.data() + r * out_pitch;
        for (int64_t j = 0; j < n; ++j) {
            NNOPS_EXPECT_NEAR(out_row[j], exp_row[j], tol);
        }
    }
}

// ============================================================
// SIMD kernel tests
// ============================================================

NNOPS_TEST(tiled_eltwise_add) {
    auto a = random_f32(128, -5.0f, 5.0f);
    auto b = random_f32(128, -5.0f, 5.0f);
    verify_kernel("add", a, b, 4, 8, 8, 8, 8, false,
        [](const float* pa, const float* pb, float* po,
           int64_t m, int64_t n, int64_t ap, int64_t bp, int64_t op, bool at) {
            add<float>(pa, pb, po, m, n, ap, bp, op, at);
        },
        [](float fa, float fb) { return fa + fb; });
}

NNOPS_TEST(tiled_eltwise_sub) {
    auto a = random_f32(128, -5.0f, 5.0f);
    auto b = random_f32(128, -5.0f, 5.0f);
    verify_kernel("sub", a, b, 4, 8, 8, 8, 8, false,
        [](const float* pa, const float* pb, float* po,
           int64_t m, int64_t n, int64_t ap, int64_t bp, int64_t op, bool at) {
            sub<float>(pa, pb, po, m, n, ap, bp, op, at);
        },
        [](float fa, float fb) { return fa - fb; });
}

NNOPS_TEST(tiled_eltwise_mul) {
    auto a = random_f32(128, -3.0f, 3.0f);
    auto b = random_f32(128, -3.0f, 3.0f);
    verify_kernel("mul", a, b, 4, 8, 8, 8, 8, false,
        [](const float* pa, const float* pb, float* po,
           int64_t m, int64_t n, int64_t ap, int64_t bp, int64_t op, bool at) {
            mul<float>(pa, pb, po, m, n, ap, bp, op, at);
        },
        [](float fa, float fb) { return fa * fb; });
}

NNOPS_TEST(tiled_eltwise_div) {
    auto a = random_f32(128, -5.0f, 5.0f);
    auto b = random_f32(128, 0.5f, 5.0f);  // avoid zero
    verify_kernel("div", a, b, 4, 8, 8, 8, 8, false,
        [](const float* pa, const float* pb, float* po,
           int64_t m, int64_t n, int64_t ap, int64_t bp, int64_t op, bool at) {
            div<float>(pa, pb, po, m, n, ap, bp, op, at);
        },
        [](float fa, float fb) { return fa / fb; });
}

NNOPS_TEST(tiled_eltwise_min) {
    auto a = random_f32(128, -5.0f, 5.0f);
    auto b = random_f32(128, -5.0f, 5.0f);
    verify_kernel("min", a, b, 4, 8, 8, 8, 8, false,
        [](const float* pa, const float* pb, float* po,
           int64_t m, int64_t n, int64_t ap, int64_t bp, int64_t op, bool at) {
            min<float>(pa, pb, po, m, n, ap, bp, op, at);
        },
        [](float fa, float fb) { return fa < fb ? fa : fb; },
        0.0f);  // exact
}

NNOPS_TEST(tiled_eltwise_max) {
    auto a = random_f32(128, -5.0f, 5.0f);
    auto b = random_f32(128, -5.0f, 5.0f);
    verify_kernel("max", a, b, 4, 8, 8, 8, 8, false,
        [](const float* pa, const float* pb, float* po,
           int64_t m, int64_t n, int64_t ap, int64_t bp, int64_t op, bool at) {
            max<float>(pa, pb, po, m, n, ap, bp, op, at);
        },
        [](float fa, float fb) { return fa > fb ? fa : fb; },
        0.0f);  // exact
}

// ============================================================
// Scalar-only kernel test
// ============================================================

NNOPS_TEST(tiled_eltwise_pow) {
    auto a = random_f32(128, 0.5f, 4.0f);
    auto b = random_f32(128, 0.1f, 2.0f);
    verify_kernel("pow", a, b, 4, 8, 8, 8, 8, false,
        [](const float* pa, const float* pb, float* po,
           int64_t m, int64_t n, int64_t ap, int64_t bp, int64_t op, bool at) {
            pow<float>(pa, pb, po, m, n, ap, bp, op, at);
        },
        [](float fa, float fb) { return std::pow(fa, fb); });
}

// ============================================================
// Pitch / stride tests
// ============================================================

NNOPS_TEST(tiled_eltwise_pitch_gt_n) {
    auto a = random_f32(3 * 16, -5.0f, 5.0f);
    auto b = random_f32(3 * 16, -5.0f, 5.0f);
    verify_kernel("add pitch>n", a, b, 3, 7, 16, 16, 16, false,
        [](const float* pa, const float* pb, float* po,
           int64_t m, int64_t n, int64_t ap, int64_t bp, int64_t op, bool at) {
            add<float>(pa, pb, po, m, n, ap, bp, op, at);
        },
        [](float fa, float fb) { return fa + fb; });
}

NNOPS_TEST(tiled_eltwise_mismatched_pitch) {
    auto a = random_f32(2 * 16, -5.0f, 5.0f);
    auto b = random_f32(2 * 32, -5.0f, 5.0f);
    verify_kernel("add mismatched pitch", a, b, 2, 6, 16, 32, 8, false,
        [](const float* pa, const float* pb, float* po,
           int64_t m, int64_t n, int64_t ap, int64_t bp, int64_t op, bool at) {
            add<float>(pa, pb, po, m, n, ap, bp, op, at);
        },
        [](float fa, float fb) { return fa + fb; });
}

NNOPS_TEST(tiled_eltwise_add_to) {
    auto a = random_f32(2 * 8, -3.0f, 3.0f);
    auto b = random_f32(2 * 8, -3.0f, 3.0f);
    verify_kernel("add_to", a, b, 2, 8, 8, 8, 8, true,
        [](const float* pa, const float* pb, float* po,
           int64_t m, int64_t n, int64_t ap, int64_t bp, int64_t op, bool at) {
            add<float>(pa, pb, po, m, n, ap, bp, op, at);
        },
        [](float fa, float fb) { return fa + fb; });
}

NNOPS_TEST(tiled_eltwise_large_tile) {
    auto a = random_f32(10 * 64, -5.0f, 5.0f);
    auto b = random_f32(10 * 64, -5.0f, 5.0f);
    verify_kernel("add large tile", a, b, 10, 31, 64, 64, 64, false,
        [](const float* pa, const float* pb, float* po,
           int64_t m, int64_t n, int64_t ap, int64_t bp, int64_t op, bool at) {
            add<float>(pa, pb, po, m, n, ap, bp, op, at);
        },
        [](float fa, float fb) { return fa + fb; });
}
