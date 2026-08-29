/// @file test_permute.cpp
/// @brief Unit tests for Permute operator (CPU backend).

#include "nnops/ops/permute.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/test_helpers.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <vector>
#include <algorithm>
#include <cstring>

using namespace nnops;

// ============================================================
// Hand-verified tests
// ============================================================

NNOPS_TEST(permute_2d_transpose) {
    // 2D transpose: [3, 4] → [4, 3]  (perm = [1, 0])
    float in_data[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
    };
    const int64_t in_shape[] = {3, 4};
    TensorView input(in_shape, DataType::f32, in_data);

    PermuteAttributes attrs;
    const int64_t p[] = {1, 0};
    attrs.perm = std::span<const int64_t>(p, 2);
    auto op = Permute::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 4);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 3);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(12);
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    // Expected: transpose of 3×4 → 4×3
    NNOPS_EXPECT_NEAR(out_buf[0], 1.0f, 1e-5f);   // [0,0] ← in[0,0]
    NNOPS_EXPECT_NEAR(out_buf[1], 5.0f, 1e-5f);   // [0,1] ← in[1,0]
    NNOPS_EXPECT_NEAR(out_buf[2], 9.0f, 1e-5f);   // [0,2] ← in[2,0]
    NNOPS_EXPECT_NEAR(out_buf[3], 2.0f, 1e-5f);   // [1,0] ← in[0,1]
    NNOPS_EXPECT_NEAR(out_buf[4], 6.0f, 1e-5f);   // [1,1] ← in[1,1]
    NNOPS_EXPECT_NEAR(out_buf[5], 10.0f, 1e-5f);  // [1,2] ← in[2,1]
    NNOPS_EXPECT_NEAR(out_buf[6], 3.0f, 1e-5f);   // [2,0] ← in[0,2]
    NNOPS_EXPECT_NEAR(out_buf[7], 7.0f, 1e-5f);   // [2,1] ← in[1,2]
    NNOPS_EXPECT_NEAR(out_buf[8], 11.0f, 1e-5f);  // [2,2] ← in[2,2]
    NNOPS_EXPECT_NEAR(out_buf[9], 4.0f, 1e-5f);   // [3,0] ← in[0,3]
    NNOPS_EXPECT_NEAR(out_buf[10], 8.0f, 1e-5f);  // [3,1] ← in[1,3]
    NNOPS_EXPECT_NEAR(out_buf[11], 12.0f, 1e-5f); // [3,2] ← in[2,3]
}

NNOPS_TEST(permute_3d_batch_transpose) {
    // 3D: [2, 3, 4] → [3, 2, 4]  (perm = [1, 0, 2])
    // Last dim stays the same → contiguous memcpy path
    const int64_t in_shape[] = {2, 3, 4};
    auto [in_vec, input] = test::make_random_tensor(in_shape, -1.0f, 1.0f, 111);

    PermuteAttributes attrs;
    const int64_t p1[] = {1, 0, 2};
    attrs.perm = std::span<const int64_t>(p1, 3);
    auto op = Permute::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 3);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[2], 4);

    std::vector<float> out_buf(24);
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    // Verify: output[s, b, d] == input[b, s, d]
    for (int64_t b = 0; b < 2; ++b) {
        for (int64_t s = 0; s < 3; ++s) {
            for (int64_t d = 0; d < 4; ++d) {
                float expected = in_vec[static_cast<size_t>(b * 12 + s * 4 + d)];
                float result   = out_buf[static_cast<size_t>(s * 8 + b * 4 + d)];
                NNOPS_EXPECT_NEAR(result, expected, 1e-5f);
            }
        }
    }
}

NNOPS_TEST(permute_4d_nchw_to_nhwc) {
    // 4D: [1, 3, 2, 2] → [1, 2, 2, 3]  (perm = [0, 2, 3, 1])
    float in_data[] = {
        // C=0
        1.0f, 2.0f,
        3.0f, 4.0f,
        // C=1
        5.0f, 6.0f,
        7.0f, 8.0f,
        // C=2
        9.0f, 10.0f,
        11.0f, 12.0f,
    };
    const int64_t in_shape[] = {1, 3, 2, 2};
    TensorView input(in_shape, DataType::f32, in_data);

    PermuteAttributes attrs;
    const int64_t p2[] = {0, 2, 3, 1};
    attrs.perm = std::span<const int64_t>(p2, 4);
    auto op = Permute::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 4);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 1);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[2], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[3], 3);

    std::vector<float> out_buf(12);
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    // output[n, h, w, c] == input[n, c, h, w]
    // For n=0, h=0, w=0: expects {1, 5, 9} (C dim)
    NNOPS_EXPECT_NEAR(out_buf[0], 1.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], 5.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[2], 9.0f, 1e-5f);
    // n=0, h=0, w=1: expects {2, 6, 10}
    NNOPS_EXPECT_NEAR(out_buf[3], 2.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[4], 6.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[5], 10.0f, 1e-5f);
}

NNOPS_TEST(permute_identity) {
    // Identity perm [0, 1, 2, 3] — output should equal input
    const int64_t shape[] = {2, 3, 4, 5};
    auto [in_vec, input] = test::make_random_tensor(shape, -1.0f, 1.0f, 222);

    PermuteAttributes attrs;
    const int64_t p3[] = {0, 1, 2, 3};
    attrs.perm = std::span<const int64_t>(p3, 4);
    auto op = Permute::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 4);
    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_EQ(descs[0].dims[i], shape[i]);
    }

    std::vector<float> out_buf(in_vec.size());
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    for (size_t i = 0; i < in_vec.size(); ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], in_vec[i], 1e-5f);
    }
}

NNOPS_TEST(permute_reverse) {
    // Reverse all dims: [2, 3, 4] → [4, 3, 2]  (perm = [2, 1, 0])
    // General path — not a last-2-dims swap
    const int64_t in_shape[] = {2, 3, 4};
    auto [in_vec, input] = test::make_random_tensor(in_shape, -1.0f, 1.0f, 333);

    PermuteAttributes attrs;
    const int64_t p4[] = {2, 1, 0};
    attrs.perm = std::span<const int64_t>(p4, 3);
    auto op = Permute::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 4);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 3);
    NNOPS_EXPECT_EQ(descs[0].dims[2], 2);

    std::vector<float> out_buf(24);
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    // output[k, j, i] == input[i, j, k]
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 3; ++j) {
            for (int64_t k = 0; k < 4; ++k) {
                float expected = in_vec[static_cast<size_t>(i * 12 + j * 4 + k)];
                float result   = out_buf[static_cast<size_t>(k * 6 + j * 2 + i)];
                NNOPS_EXPECT_NEAR(result, expected, 1e-5f);
            }
        }
    }
}

// ============================================================
// Batched last-2-dims swap tests (SIMD tiled path)
// ============================================================

NNOPS_TEST(permute_3d_last_two_swap) {
    // 3D: [B, M, K] → [B, K, M]  (perm = [0, 2, 1])
    // Batched 2D transpose: each of the B slices is [M, K] → [K, M]
    // Hits the SIMD tiled v_transpose_8x8 path
    const int64_t B = 3, M = 12, K = 16;
    auto [in_vec, input] = test::make_random_tensor({B, M, K}, -1.0f, 1.0f, 777);

    PermuteAttributes attrs;
    const int64_t p[] = {0, 2, 1};
    attrs.perm = std::span<const int64_t>(p, 3);
    auto op = Permute::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 3);
    NNOPS_EXPECT_EQ(descs[0].dims[0], B);
    NNOPS_EXPECT_EQ(descs[0].dims[1], K);
    NNOPS_EXPECT_EQ(descs[0].dims[2], M);

    size_t total = B * M * K;
    std::vector<float> out_buf(total);
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    // output[b, k, m] == input[b, m, k]
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t k = 0; k < K; ++k) {
                float expected = in_vec[static_cast<size_t>(b * M * K + m * K + k)];
                float result   = out_buf[static_cast<size_t>(b * K * M + k * M + m)];
                NNOPS_EXPECT_NEAR(result, expected, 1e-5f);
            }
        }
    }
}

NNOPS_TEST(permute_4d_last_two_swap) {
    // 4D: [N, C, H, W] → [N, C, W, H]  (perm = [0, 1, 3, 2])
    // N*C independent 2D transposes of [H, W] → [W, H]
    // Hits the SIMD tiled v_transpose_8x8 path
    const int64_t N = 2, C = 4, H = 16, W = 20;
    auto [in_vec, input] = test::make_random_tensor({N, C, H, W}, -1.0f, 1.0f, 888);

    PermuteAttributes attrs;
    const int64_t p[] = {0, 1, 3, 2};
    attrs.perm = std::span<const int64_t>(p, 4);
    auto op = Permute::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 4);
    NNOPS_EXPECT_EQ(descs[0].dims[0], N);
    NNOPS_EXPECT_EQ(descs[0].dims[1], C);
    NNOPS_EXPECT_EQ(descs[0].dims[2], W);
    NNOPS_EXPECT_EQ(descs[0].dims[3], H);

    size_t total = N * C * H * W;
    std::vector<float> out_buf(total);
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    // output[n, c, w, h] == input[n, c, h, w]
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    float expected = in_vec[static_cast<size_t>(
                        n * C * H * W + c * H * W + h * W + w)];
                    float result = out_buf[static_cast<size_t>(
                        n * C * W * H + c * W * H + w * H + h)];
                    NNOPS_EXPECT_NEAR(result, expected, 1e-5f);
                }
            }
        }
    }
}

NNOPS_TEST(permute_last_two_swap_f16) {
    // f16 batched 2D transpose: [B, M, K] → [B, K, M]
    // Hits the SIMD tiled v_transpose_8x8 path for f16
    const int64_t B = 2, M = 8, K = 10;
    auto [in_vec, _] = test::make_random_tensor({B, M, K}, -1.0f, 1.0f, 999);

    std::vector<nnops::backend::cpu::half> in_half(B * M * K);
    for (size_t i = 0; i < in_half.size(); ++i) {
        simd::s_store(&in_half[i], in_vec[i]);
    }

    const int64_t in_shape[] = {B, M, K};
    TensorView input(in_shape, DataType::f16, in_half.data());

    PermuteAttributes attrs;
    const int64_t p[] = {0, 2, 1};
    attrs.perm = std::span<const int64_t>(p, 3);
    auto op = Permute::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);
    NNOPS_EXPECT_EQ(descs[0].dims[0], B);
    NNOPS_EXPECT_EQ(descs[0].dims[1], K);
    NNOPS_EXPECT_EQ(descs[0].dims[2], M);

    std::vector<nnops::backend::cpu::half> out_buf(B * M * K);
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t k = 0; k < K; ++k) {
                float expected = in_vec[static_cast<size_t>(b * M * K + m * K + k)];
                float result = simd::s_load(
                    &out_buf[static_cast<size_t>(b * K * M + k * M + m)]);
                NNOPS_EXPECT_NEAR(result, expected, 5e-3f);
            }
        }
    }
}

NNOPS_TEST(permute_last_two_swap_partial_tile) {
    // Edge case: dims that are not multiples of TILE=8 exercise the
    // partial-tile scalar fallback inside the SIMD path
    const int64_t M = 10, K = 10;  // > 8 but not multiples
    auto [in_vec, input] = test::make_random_tensor({M, K}, -1.0f, 1.0f, 333);

    PermuteAttributes attrs;
    const int64_t p[] = {1, 0};
    attrs.perm = std::span<const int64_t>(p, 2);
    auto op = Permute::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].dims[0], K);
    NNOPS_EXPECT_EQ(descs[0].dims[1], M);

    std::vector<float> out_buf(M * K);
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float expected = in_vec[static_cast<size_t>(m * K + k)];
            float result   = out_buf[static_cast<size_t>(k * M + m)];
            NNOPS_EXPECT_NEAR(result, expected, 1e-5f);
        }
    }
}

// ============================================================
// Random data tests
// ============================================================

NNOPS_TEST(permute_random_f32) {
    const int64_t N = 4, C = 8, H = 6, W = 10;
    auto [in_vec, input] = test::make_random_tensor({N, C, H, W}, -5.0f, 5.0f, 444);

    PermuteAttributes attrs;
    const int64_t p5[] = {0, 2, 3, 1};  // NCHW → NHWC
    attrs.perm = std::span<const int64_t>(p5, 4);
    auto op = Permute::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].dims[0], N);
    NNOPS_EXPECT_EQ(descs[0].dims[1], H);
    NNOPS_EXPECT_EQ(descs[0].dims[2], W);
    NNOPS_EXPECT_EQ(descs[0].dims[3], C);

    size_t total = N * H * W * C;
    std::vector<float> out_buf(total);
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    // output[n, h, w, c] == input[n, c, h, w]
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    float expected = in_vec[static_cast<size_t>(
                        n * (C * H * W) + c * (H * W) + h * W + w)];
                    float result = out_buf[static_cast<size_t>(
                        n * (H * W * C) + h * (W * C) + w * C + c)];
                    NNOPS_EXPECT_NEAR(result, expected, 1e-5f);
                }
            }
        }
    }
}

NNOPS_TEST(permute_random_f16) {
    const int64_t M = 8, K = 12;
    auto [in_vec, _] = test::make_random_tensor({M, K}, -1.0f, 1.0f, 555);

    std::vector<nnops::backend::cpu::half> in_half(M * K);
    for (size_t i = 0; i < in_half.size(); ++i) {
        simd::s_store(&in_half[i], in_vec[i]);
    }

    const int64_t in_shape[] = {M, K};
    TensorView input(in_shape, DataType::f16, in_half.data());

    PermuteAttributes attrs;
    const int64_t p6[] = {1, 0};  // 2D transpose
    attrs.perm = std::span<const int64_t>(p6, 2);
    auto op = Permute::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);
    NNOPS_EXPECT_EQ(descs[0].dims[0], K);
    NNOPS_EXPECT_EQ(descs[0].dims[1], M);

    std::vector<nnops::backend::cpu::half> out_buf(K * M);
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    // output[k, m] == input[m, k]
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float expected = in_vec[static_cast<size_t>(m * K + k)];
            float result = simd::s_load(&out_buf[static_cast<size_t>(k * M + m)]);
            NNOPS_EXPECT_NEAR(result, expected, 5e-3f);
        }
    }
}

// ============================================================
// Shape inference test
// ============================================================

NNOPS_TEST(permute_shape_inference) {
    const int64_t in_shape[] = {2, 3, 5, 7};
    float dummy[1] = {};
    TensorView input(in_shape, DataType::f32, dummy);

    PermuteAttributes attrs;
    const int64_t p7[] = {0, 3, 1, 2};  // [2, 3, 5, 7] → [2, 7, 3, 5]
    attrs.perm = std::span<const int64_t>(p7, 4);
    auto op = Permute::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, 4);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 7);
    NNOPS_EXPECT_EQ(descs[0].dims[2], 3);
    NNOPS_EXPECT_EQ(descs[0].dims[3], 5);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
}

// ============================================================
// Class API test
// ============================================================

NNOPS_TEST(permute_class_api) {
    float in_data[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };
    const int64_t in_shape[] = {2, 3};
    TensorView input(in_shape, DataType::f32, in_data);

    PermuteAttributes attrs;
    const int64_t p8[] = {1, 0};
    attrs.perm = std::span<const int64_t>(p8, 2);

    // Class API
    std::vector<float> out_class(6);
    {
        auto op = Permute::create(attrs, Backend::CPU);
        const TensorDesc in_arr[] = {input.desc()};
        auto descs = op->getOutputTensorDesc(in_arr);
        TensorView output = test::make_planar(descs[0], out_class.data());
        const TensorView ins[] = {input};
        op->compute(output, ins);
    }

    // Class API (second instance)
    std::vector<float> out_func(6);
    {
        auto op = Permute::create(attrs, Backend::CPU);
        const TensorDesc in_arr[] = {input.desc()};
        auto descs = op->getOutputTensorDesc(in_arr);
        TensorView output = test::make_planar(descs[0], out_func.data());
        const TensorView ins[] = {input};
        op->compute(output, ins);
    }

    for (size_t i = 0; i < 6; ++i) {
        NNOPS_EXPECT_NEAR(out_class[i], out_func[i], 1e-5f);
    }
}
