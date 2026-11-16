/// @file test_conv2d_winograd.cpp
/// @brief Unit tests for the channel-last Winograd Conv2D path (k3s1),
///        F(2×2,3×3) and F(4×4,3×3).
///
/// The weight transform is checked against a naive scalar reference written
/// longhand from each order's definition; the F(4,3) input and output
/// transforms are checked on channel-last planes. The order policy is pinned at
/// its fences: winograd_applicable's admission and wino_pick_order's F(2,3) /
/// F(4,3) decision.
///
/// The end-to-end Winograd-vs-reference cases live in
/// test_conv2d_channel_last, where the activation, the HWIO filter and the
/// prepacked U are all channel-last.
///
/// Winograd and direct convolution are algebraically equal but not bit-identical
/// (the transform reorders the sums and, at f16, adds a rounding step), so those
/// comparisons are tolerance-based.

#include "nnops/ops/conv2d.hpp"
#include "backend/cpu/conv/conv2d_winograd.h"
#include "backend/cpu/simd_kernel/simd_winograd.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace nnops;

// ============================================================
// Transform matrices, restated independently of the kernel
// ============================================================
//
// Bᵀ is applied as (d·B) then (Bᵀ·e); Aᵀ as (Aᵀ·m) then (·A); G as written.
// Longhand, so a typo in simd_winograd.hpp cannot hide behind a shared helper.

static void naive_gggt(const float g[3][3], float u[4][4]) {
    static const float G[4][3] = {
        {1.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f},
        {0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}};
    float t[4][3];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 3; ++k) { s += G[i][k] * g[k][j]; }
            t[i][j] = s;
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 3; ++k) { s += t[i][k] * G[j][k]; }  // ·Gᵀ
            u[i][j] = s;
        }
    }
}

// ---- the F(4,3) matrices, restated the same way ---------------------------

static void naive_btdb43(const float d[6][6], float v[6][6]) {
    static const float Bt[6][6] = {
        {4, 0, -5, 0, 1, 0}, {0, -4, -4, 1, 1, 0}, {0, 4, -4, -1, 1, 0},
        {0, -2, -1, 2, 1, 0}, {0, 2, -1, -2, 1, 0}, {0, 4, 0, -5, 0, 1}};
    float e[6][6];
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 6; ++k) { s += d[i][k] * Bt[j][k]; }  // d·B
            e[i][j] = s;
        }
    }
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 6; ++k) { s += Bt[i][k] * e[k][j]; }
            v[i][j] = s;
        }
    }
}

static void naive_atma43(const float m[6][6], float y[4][4]) {
    static const float At[4][6] = {
        {1, 1, 1, 1, 1, 0}, {0, 1, -1, 2, -2, 0},
        {0, 1, 1, 4, 4, 0}, {0, 1, -1, 8, -8, 1}};
    float t[4][6];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 6; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 6; ++k) { s += At[i][k] * m[k][j]; }
            t[i][j] = s;
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 6; ++k) { s += t[i][k] * At[j][k]; }
            y[i][j] = s;
        }
    }
}

static void naive_gggt43(const float g[3][3], float u[6][6]) {
    static const float G[6][3] = {
        {1.0f / 4.0f, 0.0f, 0.0f},
        {-1.0f / 6.0f, -1.0f / 6.0f, -1.0f / 6.0f},
        {-1.0f / 6.0f, 1.0f / 6.0f, -1.0f / 6.0f},
        {1.0f / 24.0f, 1.0f / 12.0f, 1.0f / 6.0f},
        {1.0f / 24.0f, -1.0f / 12.0f, 1.0f / 6.0f},
        {0.0f, 0.0f, 1.0f}};
    float t[6][3];
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 3; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 3; ++k) { s += G[i][k] * g[k][j]; }
            t[i][j] = s;
        }
    }
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 3; ++k) { s += t[i][k] * G[j][k]; }
            u[i][j] = s;
        }
    }
}

// ============================================================
// Transform unit tests
// ============================================================

NNOPS_TEST(wino_weight_transform_matches_naive) {
    test::XorShift128 rng(0xC0FFEE);
    for (int trial = 0; trial < 20; ++trial) {
        float g[3][3], ubuf[4][4];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) { g[i][j] = rng.next_float(-2.0f, 2.0f); }
        }
        nnops::kernel::winograd_weight_transform(&g[0][0], 3, &ubuf[0][0], 4);
        float want[4][4];
        naive_gggt(g, want);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                NNOPS_EXPECT_NEAR(ubuf[i][j], want[i][j], 1e-5f);
            }
        }
    }
}

NNOPS_TEST(wino_weight_transform_plane_strides) {
    // Same transform, but with the p = i·4 + j position scattered into 16
    // planes the way prepack stores U (ld_u = 4·plane, ld_c = plane).
    const int plane = 3;  // pretend OC·ICG == 3
    const int ld_u = 4 * plane;
    const int ld_c = plane;
    test::XorShift128 rng(42);
    float g[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) { g[i][j] = rng.next_float(-1.0f, 1.0f); }
    }
    std::vector<float> u(16 * plane, -99.0f);
    nnops::kernel::winograd_weight_transform(&g[0][0], 3, u.data(), ld_u, ld_c);

    float want[4][4];
    naive_gggt(g, want);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const int p = i * 4 + j;
            NNOPS_EXPECT_NEAR(u[p * plane + 0], want[i][j], 1e-5f);
        }
    }
}

// ============================================================
// F(4,3) transforms (channel-last route)
// ============================================================

NNOPS_TEST(wino43_weight_transform_matches_naive) {
    // p = i·6 + j, scattered into 36 planes the way prepack stores U.
    const int plane = 3;
    const int ld_u = 6 * plane;
    const int ld_c = plane;
    test::XorShift128 rng(0x43);
    for (int trial = 0; trial < 20; ++trial) {
        float g[3][3];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) { g[i][j] = rng.next_float(-2.0f, 2.0f); }
        }
        std::vector<float> u(36 * plane, -99.0f);
        nnops::kernel::winograd43_weight_transform(&g[0][0], 3, u.data(), ld_u, ld_c);

        float want[6][6];
        naive_gggt43(g, want);
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                NNOPS_EXPECT_NEAR(u[(i * 6 + j) * plane], want[i][j], 1e-6f);
            }
        }
    }
}

NNOPS_TEST(wino43_input_transform_matches_naive) {
    // A 12x12 channel-last plane holds four 4x4 tiles in a row and three tile rows; the
    // window of the corner tile reads four rows and columns of zeros.
    constexpr int IH = 12, IW = 12, IC = 8;
    constexpr int64_t kVPlane = 8;  // one tile per position
    std::vector<float> plane(static_cast<size_t>(IH * IW * IC));
    test::XorShift128 rng(0x4343);
    for (auto& x : plane) { x = rng.next_float(-3.0f, 3.0f); }

    std::vector<float> v(36 * kVPlane, -99.0f);
    const int th = 2, tw = 2;  // tile [8..11] x [8..11]: its window ends at 13
    nnops::kernel::winograd43_input_channel_last<float>(
        plane.data(), IW * IC, IC, 4 * th, 4 * tw, IH, IW,
        v.data(), kVPlane, IC);

    for (int k = 0; k < IC; ++k) {
        float d[6][6];
        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 6; ++c) {
                const int h = 4 * th + r, w = 4 * tw + c;
                d[r][c] = (h < IH && w < IW)
                        ? plane[static_cast<size_t>((h * IW + w) * IC + k)] : 0.0f;
            }
        }
        float want[6][6];
        naive_btdb43(d, want);
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                NNOPS_EXPECT_NEAR(v[(i * 6 + j) * kVPlane + k], want[i][j], 1e-3f);
            }
        }
    }
}

NNOPS_TEST(wino43_output_transform_matches_naive) {
    // m is [pos][oc] for one tile; the write-back adds the bias and applies the
    // epilogue on the four real outputs, and a null destination writes nothing.
    constexpr int kOcn = 8;
    std::vector<float> m(36 * kOcn);
    test::XorShift128 rng(0x434343);
    for (auto& x : m) { x = rng.next_float(-2.0f, 2.0f); }
    std::vector<float> bias(kOcn);
    for (auto& b : bias) { b = rng.next_float(-1.0f, 1.0f); }

    std::vector<float> out(16 * kOcn, -7.0f);
    float* p[4][4];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) { p[r][c] = out.data() + (r * 4 + c) * kOcn; }
    }
    p[3][2] = nullptr;  // past the output extent: must stay untouched
    const Epilogue ep{};
    nnops::kernel::winograd43_output_channel_last<float>(m.data(), kOcn, p, ep, bias.data(),
                                                 false, kOcn);

    for (int k = 0; k < kOcn; ++k) {
        float mm[6][6];
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) { mm[i][j] = m[(i * 6 + j) * kOcn + k]; }
        }
        float want[4][4];
        naive_atma43(mm, want);
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                const float got = out[static_cast<size_t>((r * 4 + c) * kOcn + k)];
                if (p[r][c] == nullptr) {
                    NNOPS_EXPECT_EQ(got, -7.0f);
                } else {
                    NNOPS_EXPECT_NEAR(got, want[r][c] + bias[k], 1e-3f);
                }
            }
        }
    }
}

// ============================================================
// Selection policy and the Auto algorithm
// ============================================================

NNOPS_TEST(wino_applicable_thresholds) {
    Conv2DAttributes a;
    a.kernel_size = {3, 3};
    a.padding = {1, 1};

    // Every condition met: k3s1, 32 channels, a 14x14 input.
    NNOPS_EXPECT_TRUE(winograd_applicable(a, 32, 14, 14));
    NNOPS_EXPECT_TRUE(winograd_applicable(a, 512, 56, 56));

    NNOPS_EXPECT_FALSE(winograd_applicable(a, 31, 14, 14));  // too few channels
    NNOPS_EXPECT_FALSE(winograd_applicable(a, 64, 6, 14));   // too short
    NNOPS_EXPECT_FALSE(winograd_applicable(a, 64, 14, 6));   // too narrow

    // The channel-last plane floor is 7x7: it flattens its tiles and keeps
    // winning down to one tile row.
    NNOPS_EXPECT_TRUE(winograd_applicable(a, 64, 7, 7));
    NNOPS_EXPECT_FALSE(winograd_applicable(a, 64, 6, 7));
    NNOPS_EXPECT_FALSE(winograd_applicable(a, 64, 7, 6));
    NNOPS_EXPECT_TRUE(winograd_applicable(a, 512, 14, 14));

    Conv2DAttributes k1 = a;
    k1.kernel_size = {1, 1};
    NNOPS_EXPECT_FALSE(winograd_applicable(k1, 64, 28, 28));

    Conv2DAttributes s2 = a;
    s2.stride = {2, 2};
    NNOPS_EXPECT_FALSE(winograd_applicable(s2, 64, 28, 28));

    Conv2DAttributes d2 = a;
    d2.dilation = {2, 2};
    NNOPS_EXPECT_FALSE(winograd_applicable(d2, 64, 28, 28));

    // The caller's off switch: ImplicitGemm keeps the prepack from happening.
    Conv2DAttributes off = a;
    off.algorithm = ConvAlgorithm::ImplicitGemm;
    NNOPS_EXPECT_FALSE(winograd_applicable(off, 64, 28, 28));
}

NNOPS_TEST(wino_pick_order_fences) {
    using nnops::backend::cpu::WinoOrder;
    using nnops::backend::cpu::wino_pick_order;
    // f16 keeps F(2,3) at every width and plane.
    NNOPS_EXPECT_TRUE(wino_pick_order(DataType::f16, 512, 512, 56, 56) ==
                      WinoOrder::F23);
    // The channel-last floor moves with the team: a 14x14 plane is one
    // whole-plane block on a small pool (1.47x of F(2,3) at 1 thread on
    // 256->256), and past four workers that block stops filling the pool
    // (0.60x at 20).
    NNOPS_EXPECT_TRUE(wino_pick_order(DataType::f32, 64, 64, 28, 28, 20) ==
                      WinoOrder::F43);
    NNOPS_EXPECT_TRUE(wino_pick_order(DataType::f32, 64, 64, 28, 28, 1) ==
                      WinoOrder::F43);
    NNOPS_EXPECT_TRUE(wino_pick_order(DataType::f32, 256, 256, 14, 14, 1) ==
                      WinoOrder::F43);
    NNOPS_EXPECT_TRUE(wino_pick_order(DataType::f32, 256, 256, 14, 14, 4) ==
                      WinoOrder::F43);
    NNOPS_EXPECT_TRUE(wino_pick_order(DataType::f32, 256, 256, 14, 14, 20) ==
                      WinoOrder::F23);
    // F(4,3) from a 28x28 plane up ...
    NNOPS_EXPECT_TRUE(wino_pick_order(DataType::f32, 64, 64, 28, 28) ==
                      WinoOrder::F43);
    NNOPS_EXPECT_TRUE(wino_pick_order(DataType::f32, 256, 256, 56, 56) ==
                      WinoOrder::F43);
    // ... and down to 14, where a single whole-plane block still pays. Under
    // that floor it does not: 7x7 is a 4-tile grid and F(4,3) measured 1.6x
    // F(2,3) there.
    NNOPS_EXPECT_TRUE(wino_pick_order(DataType::f32, 512, 64, 27, 56) ==
                      WinoOrder::F43);
    NNOPS_EXPECT_TRUE(wino_pick_order(DataType::f32, 512, 64, 13, 56) ==
                      WinoOrder::F23);
    NNOPS_EXPECT_TRUE(wino_pick_order(DataType::f32, 64, 64, 7, 7) ==
                      WinoOrder::F23);
    // An unknown plane takes the safe order.
    NNOPS_EXPECT_TRUE(wino_pick_order(DataType::f32, 64, 64, 0, 0) ==
                      WinoOrder::F23);
    // The U fence: 36 positions x IC x OC x 4 B under 16 MB. 512x512 is 37.7 MB
    // and loses; 256x256 is 9.4 MB and wins.
    NNOPS_EXPECT_TRUE(wino_pick_order(DataType::f32, 512, 512, 56, 56) ==
                      WinoOrder::F23);
    NNOPS_EXPECT_TRUE(wino_pick_order(DataType::f32, 512, 512, 28, 28) ==
                      WinoOrder::F23);
    NNOPS_EXPECT_TRUE(wino_pick_order(DataType::f32, 256, 256, 28, 28) ==
                      WinoOrder::F43);
    // 18.9 MB is over the line, 9.4 MB is under it.
    NNOPS_EXPECT_TRUE(wino_pick_order(DataType::f32, 512, 256, 28, 28) ==
                      WinoOrder::F23);
    NNOPS_EXPECT_TRUE(wino_pick_order(DataType::f32, 256, 128, 28, 28) ==
                      WinoOrder::F43);
}
