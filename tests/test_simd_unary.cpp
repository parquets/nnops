/// @file test_tiled_unary.cpp
/// @brief Unit tests for tiled unary SIMD kernels (simd_kernel/simd_unary.hpp).

#include "../src/backend/cpu/simd_kernel/simd_unary.hpp"
#include "common/test_harness.hpp"

#include <cmath>
#include <vector>
#include <random>

using namespace nnops::kernel;

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

static std::mt19937 rng(42);

static std::vector<float> random_f32(int64_t n, float lo = -5.0f, float hi = 5.0f) {
    std::vector<float> v(n);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (int64_t i = 0; i < n; ++i) {
        v[i] = dist(rng);
    }
    return v;
}

template <typename Kernel, typename Ref>
static void verify_kernel(const char* test_name,
                          const std::vector<float>& input,
                          int64_t m, int64_t n,
                          int64_t in_pitch, int64_t out_pitch,
                          bool add_to,
                          Kernel&& kernel, Ref&& ref_fn,
                          float tol = 1e-4f)
{
    int64_t out_size = (m - 1) * out_pitch + n;

    std::vector<float> out(out_size, add_to ? 0.1f : 0.0f);
    std::vector<float> expected(out_size);
    for (int64_t i = 0; i < out_size; ++i) {
        expected[i] = out[i];
    }

    // Reference: scalar row-by-row
    for (int64_t r = 0; r < m; ++r) {
        const float* in_row  = input.data()  + r * in_pitch;
        float*       out_row = expected.data() + r * out_pitch;
        for (int64_t j = 0; j < n; ++j) {
            float val = ref_fn(in_row[j]);
            if (add_to) {
                out_row[j] += val;
            }
            else {
                out_row[j] = val;
            }
        }
    }

    // Tiled kernel
    kernel(input.data(), out.data(), m, n, in_pitch, out_pitch, add_to);

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
// SIMD kernel tests (v_exp, v_log, v_sin, v_cos, v_tan, v_tanh, v_sqrt, v_abs, v_neg)
// ============================================================

NNOPS_TEST(tiled_unary_exp) {
    auto input = random_f32(128, -2.0f, 2.0f);
    verify_kernel("exp", input, 4, 8, 8, 8, false,
        [](const float* in, float* out, int64_t m, int64_t n,
           int64_t ip, int64_t op, bool at) { exp<float>(in, out, m, n, ip, op, at); },
        [](float v) { return std::exp(v); });
}

NNOPS_TEST(tiled_unary_log) {
    auto input = random_f32(128, 0.1f, 3.0f);
    verify_kernel("log", input, 4, 8, 8, 8, false,
        [](const float* in, float* out, int64_t m, int64_t n,
           int64_t ip, int64_t op, bool at) { log<float>(in, out, m, n, ip, op, at); },
        [](float v) { return std::log(v); });
}

NNOPS_TEST(tiled_unary_sin) {
    auto input = random_f32(128, -3.14159f, 3.14159f);
    verify_kernel("sin", input, 4, 8, 8, 8, false,
        [](const float* in, float* out, int64_t m, int64_t n,
           int64_t ip, int64_t op, bool at) { sin<float>(in, out, m, n, ip, op, at); },
        [](float v) { return std::sin(v); });
}

NNOPS_TEST(tiled_unary_cos) {
    auto input = random_f32(128, -3.14159f, 3.14159f);
    verify_kernel("cos", input, 4, 8, 8, 8, false,
        [](const float* in, float* out, int64_t m, int64_t n,
           int64_t ip, int64_t op, bool at) { cos<float>(in, out, m, n, ip, op, at); },
        [](float v) { return std::cos(v); });
}

NNOPS_TEST(tiled_unary_tan) {
    auto input = random_f32(128, -1.0f, 1.0f);
    verify_kernel("tan", input, 4, 8, 8, 8, false,
        [](const float* in, float* out, int64_t m, int64_t n,
           int64_t ip, int64_t op, bool at) { tan<float>(in, out, m, n, ip, op, at); },
        [](float v) { return std::tan(v); });
}

NNOPS_TEST(tiled_unary_tanh) {
    auto input = random_f32(128, -3.0f, 3.0f);
    verify_kernel("tanh", input, 4, 8, 8, 8, false,
        [](const float* in, float* out, int64_t m, int64_t n,
           int64_t ip, int64_t op, bool at) { tanh<float>(in, out, m, n, ip, op, at); },
        [](float v) { return std::tanh(v); });
}

NNOPS_TEST(tiled_unary_sqrt) {
    auto input = random_f32(128, 0.0f, 10.0f);
    verify_kernel("sqrt", input, 4, 8, 8, 8, false,
        [](const float* in, float* out, int64_t m, int64_t n,
           int64_t ip, int64_t op, bool at) { sqrt<float>(in, out, m, n, ip, op, at); },
        [](float v) { return std::sqrt(v); });
}

NNOPS_TEST(tiled_unary_abs) {
    auto input = random_f32(128, -10.0f, 10.0f);
    verify_kernel("abs", input, 4, 8, 8, 8, false,
        [](const float* in, float* out, int64_t m, int64_t n,
           int64_t ip, int64_t op, bool at) { abs<float>(in, out, m, n, ip, op, at); },
        [](float v) { return v < 0.0f ? -v : v; });
}

NNOPS_TEST(tiled_unary_neg) {
    auto input = random_f32(128, -10.0f, 10.0f);
    verify_kernel("neg", input, 4, 8, 8, 8, false,
        [](const float* in, float* out, int64_t m, int64_t n,
           int64_t ip, int64_t op, bool at) { neg<float>(in, out, m, n, ip, op, at); },
        [](float v) { return -v; });
}

// ============================================================
// Scalar-only kernel tests (erf, round, ceil, floor, recip, sign)
// ============================================================

NNOPS_TEST(tiled_unary_erf) {
    auto input = random_f32(128, -3.0f, 3.0f);
    verify_kernel("erf", input, 4, 8, 8, 8, false,
        [](const float* in, float* out, int64_t m, int64_t n,
           int64_t ip, int64_t op, bool at) { erf<float>(in, out, m, n, ip, op, at); },
        [](float v) { return std::erf(v); });
}

NNOPS_TEST(tiled_unary_round) {
    auto input = random_f32(128, -5.0f, 5.0f);
    verify_kernel("round", input, 4, 8, 8, 8, false,
        [](const float* in, float* out, int64_t m, int64_t n,
           int64_t ip, int64_t op, bool at) { round<float>(in, out, m, n, ip, op, at); },
        [](float v) { return std::round(v); },
        0.0f);  // exact
}

NNOPS_TEST(tiled_unary_ceil) {
    auto input = random_f32(128, -5.0f, 5.0f);
    verify_kernel("ceil", input, 4, 8, 8, 8, false,
        [](const float* in, float* out, int64_t m, int64_t n,
           int64_t ip, int64_t op, bool at) { ceil<float>(in, out, m, n, ip, op, at); },
        [](float v) { return std::ceil(v); },
        0.0f);  // exact
}

NNOPS_TEST(tiled_unary_floor) {
    auto input = random_f32(128, -5.0f, 5.0f);
    verify_kernel("floor", input, 4, 8, 8, 8, false,
        [](const float* in, float* out, int64_t m, int64_t n,
           int64_t ip, int64_t op, bool at) { floor<float>(in, out, m, n, ip, op, at); },
        [](float v) { return std::floor(v); },
        0.0f);  // exact
}

NNOPS_TEST(tiled_unary_recip) {
    auto input = random_f32(128, 0.5f, 5.0f);
    verify_kernel("recip", input, 4, 8, 8, 8, false,
        [](const float* in, float* out, int64_t m, int64_t n,
           int64_t ip, int64_t op, bool at) { recip<float>(in, out, m, n, ip, op, at); },
        [](float v) { return 1.0f / v; },
        2e-4f);  // v_rcp is approximate (~12-bit precision on x86)
}

NNOPS_TEST(tiled_unary_sign) {
    auto input = random_f32(128, -10.0f, 10.0f);
    verify_kernel("sign", input, 4, 8, 8, 8, false,
        [](const float* in, float* out, int64_t m, int64_t n,
           int64_t ip, int64_t op, bool at) { sign<float>(in, out, m, n, ip, op, at); },
        [](float v) { return (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f); },
        0.0f);  // exact
}

// ============================================================
// Pitch / stride tests
// ============================================================

NNOPS_TEST(tiled_unary_pitch_gt_n) {
    // 3 rows, 7 cols, pitch=16 (padded)
    auto input = random_f32(3 * 16, -2.0f, 2.0f);
    verify_kernel("exp pitch>n", input, 3, 7, 16, 16, false,
        [](const float* in, float* out, int64_t m, int64_t n,
           int64_t ip, int64_t op, bool at) { exp<float>(in, out, m, n, ip, op, at); },
        [](float v) { return std::exp(v); });
}

NNOPS_TEST(tiled_unary_mismatched_pitch) {
    // in_pitch != out_pitch
    auto input = random_f32(2 * 16, -2.0f, 2.0f);
    verify_kernel("exp mismatched pitch", input, 2, 6, 16, 8, false,
        [](const float* in, float* out, int64_t m, int64_t n,
           int64_t ip, int64_t op, bool at) { exp<float>(in, out, m, n, ip, op, at); },
        [](float v) { return std::exp(v); });
}

NNOPS_TEST(tiled_unary_add_to) {
    auto input = random_f32(2 * 8, 0.0f, 2.0f);
    verify_kernel("add_to", input, 2, 8, 8, 8, true,
        [](const float* in, float* out, int64_t m, int64_t n,
           int64_t ip, int64_t op, bool at) { exp<float>(in, out, m, n, ip, op, at); },
        [](float v) { return std::exp(v); });
}

NNOPS_TEST(tiled_unary_single_col) {
    // Single element per row with pitch=4
    auto input = random_f32(5 * 4, -2.0f, 2.0f);
    verify_kernel("sqrt single col", input, 5, 1, 4, 4, false,
        [](const float* in, float* out, int64_t m, int64_t n,
           int64_t ip, int64_t op, bool at) { sqrt<float>(in, out, m, n, ip, op, at); },
        [](float v) { return std::sqrt(v); });
}

NNOPS_TEST(tiled_unary_large_tile) {
    // Larger tile: 10 rows, 31 cols, pitch=64
    auto input = random_f32(10 * 64, -2.0f, 2.0f);
    verify_kernel("exp large tile", input, 10, 31, 64, 64, false,
        [](const float* in, float* out, int64_t m, int64_t n,
           int64_t ip, int64_t op, bool at) { exp<float>(in, out, m, n, ip, op, at); },
        [](float v) { return std::exp(v); });
}
