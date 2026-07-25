/// Unit tests for Pooling operator (CPU reference) — 2D and 3D.

#include "nnops/ops/pooling.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

// ============================================================
// 2D Pooling tests
// ============================================================

NNOPS_TEST(pooling_2d_max_basic) {
    // 1x1x4x4 input, 2x2 kernel, stride=2, pad=0
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t oshape[] = {1, 1, 2, 2};
    float in_data[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    };
    float out_data[4] = {};

    TensorView input(ishape, DataType::f32, in_data);
    TensorView output(oshape, DataType::f32, out_data);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};  // KD=1, KH=2, KW=2
    attrs.stride       = {1, 2, 2};
    attrs.padding      = {0, 0, 0};

    pooling(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 6.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 8.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 14.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[3], 16.0f, 1e-6f);
}

NNOPS_TEST(pooling_2d_average_basic) {
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t oshape[] = {1, 1, 2, 2};
    float in_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out_data[4] = {};

    TensorView input(ishape, DataType::f32, in_data);
    TensorView output(oshape, DataType::f32, out_data);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};

    pooling(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 10.0f / 9.0f, 1e-4f);
}

NNOPS_TEST(pooling_2d_average_exclude_pad) {
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t oshape[] = {1, 1, 2, 2};
    float in_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out_data[4] = {};

    TensorView input(ishape, DataType::f32, in_data);
    TensorView output(oshape, DataType::f32, out_data);

    PoolingAttributes attrs;
    attrs.type = PoolingType::AverageExcludePad;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};

    pooling(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 2.5f, 1e-4f);
}

NNOPS_TEST(pooling_2d_lp) {
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t oshape[] = {1, 1, 1, 1};
    float in_data[4] = {3.0f, 4.0f, 0.0f, 0.0f};
    float out_data[1] = {};

    TensorView input(ishape, DataType::f32, in_data);
    TensorView output(oshape, DataType::f32, out_data);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Lp;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};
    attrs.padding      = {0, 0, 0};
    attrs.p_norm = 2;

    pooling(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 5.0f, 1e-4f);
}

NNOPS_TEST(pooling_2d_random) {
    auto [in_vec, input] = test::make_random_tensor({1, 3, 16, 16});
    std::vector<float> out_buf(1 * 3 * 8 * 8);
    const int64_t oshape[] = {1, 3, 8, 8};
    TensorView output(oshape, DataType::f32, out_buf.data());

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};

    pooling(input, output, attrs);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

NNOPS_TEST(pooling_2d_class_api) {
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t oshape[] = {1, 1, 2, 2};
    float in_data[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float out1_data[4] = {};
    float out2_data[4] = {};

    TensorView input(ishape, DataType::f32, in_data);
    TensorView out1(oshape, DataType::f32, out1_data);
    TensorView out2(oshape, DataType::f32, out2_data);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};

    // Functional
    pooling(input, out1, attrs);
    // Class
    auto op = Pooling::create(attrs, Backend::CPU);
    const TensorView ins[] = {input};
    op->compute(out2, ins);

    NNOPS_EXPECT_TRUE(test::allclose(out1, out2));
}

// ============================================================
// 3D Pooling tests
// ============================================================

NNOPS_TEST(pooling_3d_max_basic) {
    // 1x1x2x4x4 input, 2x2x2 kernel, stride=2, pad=0
    // Output: 1x1x1x2x2
    const int64_t ishape[] = {1, 1, 2, 4, 4};
    const int64_t oshape[] = {1, 1, 1, 2, 2};

    // Two depth slices, each 4x4
    std::vector<float> in_buf(2 * 4 * 4);
    for (int i = 0; i < 32; ++i) { in_buf[i] = static_cast<float>(i + 1); }
    std::vector<float> out_buf(4, 0.0f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView output(oshape, DataType::f32, out_buf.data());

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {2, 2, 2};   // KD=2, KH=2, KW=2
    attrs.stride       = {2, 2, 2};
    attrs.padding      = {0, 0, 0};

    pooling(input, output, attrs);

    // Output[0,0]: max of depth[0,0:2,0:2] + depth[1,0:2,0:2]
    // depth0[0:2,0:2] = {1,2,5,6}, depth1[0:2,0:2] = {17,18,21,22}, max = 22
    NNOPS_EXPECT_NEAR(out_buf[0], 22.0f, 1e-6f);
    // Output[0,1]: max of depth0[0:2,2:4] + depth1[0:2,2:4]
    // depth0 = {3,4,7,8}, depth1 = {19,20,23,24}, max = 24
    NNOPS_EXPECT_NEAR(out_buf[1], 24.0f, 1e-6f);
    // Output[1,0]: max from rows 2:4 of both depths
    // depth0 rows 2:4 cols 0:2 = {9,10,13,14}, depth1 = {25,26,29,30}, max = 30
    NNOPS_EXPECT_NEAR(out_buf[2], 30.0f, 1e-6f);
    // Output[1,1]: max from rows 2:4 cols 2:4 of both depths
    NNOPS_EXPECT_NEAR(out_buf[3], 32.0f, 1e-6f);
}

NNOPS_TEST(pooling_3d_average) {
    // 1x1x2x2x2 input, 2x2x2 kernel, stride=1, pad=0 -> output 1x1x1x1x1
    const int64_t ishape[] = {1, 1, 2, 2, 2};
    const int64_t oshape[] = {1, 1, 1, 1, 1};
    float in_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float out_data[1] = {};

    TensorView input(ishape, DataType::f32, in_data);
    TensorView output(oshape, DataType::f32, out_data);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {2, 2, 2};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 0, 0};

    pooling(input, output, attrs);

    // Average of all 8 elements = (1+2+3+4+5+6+7+8)/8 = 4.5
    NNOPS_EXPECT_NEAR(out_data[0], 4.5f, 1e-4f);
}

NNOPS_TEST(pooling_3d_random) {
    auto [in_vec, input] = test::make_random_tensor({1, 2, 8, 8, 8});
    std::vector<float> out_buf(1 * 2 * 4 * 4 * 4);
    const int64_t oshape[] = {1, 2, 4, 4, 4};
    TensorView output(oshape, DataType::f32, out_buf.data());

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {2, 2, 2};
    attrs.stride       = {2, 2, 2};

    pooling(input, output, attrs);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

// ============================================================
// SIMD vs Reference correctness tests (stride=1, Max/Avg)
// ============================================================

/// Inline reference for max pooling (NCHW, 2D).
/// Used to validate the SIMD-optimized kernel against a known-correct scalar path.
static void ref_pooling_2d_max(
    const float* input, float* output,
    int64_t N, int64_t C, int64_t IH, int64_t IW,
    int64_t OH, int64_t OW,
    int64_t KH, int64_t KW,
    int64_t SH, int64_t SW,
    int64_t PH, int64_t PW)
{
    const int64_t in_ch_stride = IH * IW;
    const int64_t out_ch_stride = OH * OW;
    for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
        const float* in_ch = input + n * C * in_ch_stride + c * in_ch_stride;
        float* out_ch = output + n * C * out_ch_stride + c * out_ch_stride;
        for (int64_t oh = 0; oh < OH; ++oh) {
        for (int64_t ow = 0; ow < OW; ++ow) {
            float max_val = -std::numeric_limits<float>::infinity();
            bool any = false;
            for (int64_t kh = 0; kh < KH; ++kh) {
                int64_t ih = oh * SH + kh - PH;
                if (ih < 0 || ih >= IH) { continue; }
                for (int64_t kw = 0; kw < KW; ++kw) {
                    int64_t iw = ow * SW + kw - PW;
                    if (iw < 0 || iw >= IW) { continue; }
                    float val = in_ch[ih * IW + iw];
                    if (val > max_val) { max_val = val; }
                    any = true;
                }
            }
            out_ch[oh * OW + ow] = any ? max_val : 0.0f;
        }}
    }}
}

/// Inline reference for average pooling (includes pad) (NCHW, 2D).
static void ref_pooling_2d_avg(
    const float* input, float* output,
    int64_t N, int64_t C, int64_t IH, int64_t IW,
    int64_t OH, int64_t OW,
    int64_t KH, int64_t KW,
    int64_t SH, int64_t SW,
    int64_t PH, int64_t PW)
{
    const float scale = 1.0f / static_cast<float>(KH * KW);
    const int64_t in_ch_stride = IH * IW;
    const int64_t out_ch_stride = OH * OW;
    for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
        const float* in_ch = input + n * C * in_ch_stride + c * in_ch_stride;
        float* out_ch = output + n * C * out_ch_stride + c * out_ch_stride;
        for (int64_t oh = 0; oh < OH; ++oh) {
        for (int64_t ow = 0; ow < OW; ++ow) {
            float sum = 0.0f;
            for (int64_t kh = 0; kh < KH; ++kh) {
                int64_t ih = oh * SH + kh - PH;
                if (ih < 0 || ih >= IH) { continue; }
                for (int64_t kw = 0; kw < KW; ++kw) {
                    int64_t iw = ow * SW + kw - PW;
                    if (iw < 0 || iw >= IW) { continue; }
                    sum += in_ch[ih * IW + iw];
                }
            }
            out_ch[oh * OW + ow] = sum * scale;
        }}
    }}
}

NNOPS_TEST(pooling_simd_max_vs_ref_small) {
    // Small tensor with no padding — exercises h4/h1 SIMD interior
    auto [in_vec, input] = test::make_random_tensor({1, 3, 16, 16});
    const int64_t oshape[] = {1, 3, 14, 14};
    std::vector<float> out_simd(1 * 3 * 14 * 14);
    std::vector<float> out_ref(1 * 3 * 14 * 14);

    TensorView out_s(oshape, DataType::f32, out_simd.data());
    TensorView out_r(oshape, DataType::f32, out_ref.data());

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};

    // SIMD path
    pooling(input, out_s, attrs);

    // Reference
    const int64_t N = 1, C = 3, IH = 16, IW = 16, OH = 14, OW = 14;
    ref_pooling_2d_max(input.ptr<float>(), out_ref.data(),
                       N, C, IH, IW, OH, OW, 3, 3, 1, 1, 0, 0);

    NNOPS_EXPECT_TRUE(test::allclose(out_s, out_r, 1e-4f, 1e-4f));
}

NNOPS_TEST(pooling_simd_max_vs_ref_with_pad) {
    // Padding exercises left/right/top/bottom scalar regions + SIMD interior
    auto [in_vec, input] = test::make_random_tensor({1, 4, 15, 15});
    const int64_t oshape[] = {1, 4, 15, 15};
    std::vector<float> out_simd(1 * 4 * 15 * 15);
    std::vector<float> out_ref(1 * 4 * 15 * 15);

    TensorView out_s(oshape, DataType::f32, out_simd.data());
    TensorView out_r(oshape, DataType::f32, out_ref.data());

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};

    pooling(input, out_s, attrs);

    const int64_t N = 1, C = 4, IH = 15, IW = 15, OH = 15, OW = 15;
    ref_pooling_2d_max(input.ptr<float>(), out_ref.data(),
                       N, C, IH, IW, OH, OW, 3, 3, 1, 1, 1, 1);

    NNOPS_EXPECT_TRUE(test::allclose(out_s, out_r, 1e-4f, 1e-4f));
}

NNOPS_TEST(pooling_simd_max_stride_2) {
    // stride=2 → SIMD disabled (needs gather), falls back to scalar path
    auto [in_vec, input] = test::make_random_tensor({1, 2, 16, 16});
    const int64_t oshape[] = {1, 2, 8, 8};
    std::vector<float> out_buf(1 * 2 * 8 * 8);
    std::vector<float> ref_buf(1 * 2 * 8 * 8);

    TensorView out_s(oshape, DataType::f32, out_buf.data());
    TensorView out_r(oshape, DataType::f32, ref_buf.data());

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};

    pooling(input, out_s, attrs);

    const int64_t N = 1, C = 2, IH = 16, IW = 16, OH = 8, OW = 8;
    ref_pooling_2d_max(input.ptr<float>(), ref_buf.data(),
                       N, C, IH, IW, OH, OW, 2, 2, 2, 2, 0, 0);

    NNOPS_EXPECT_TRUE(test::allclose(out_s, out_r, 1e-4f, 1e-4f));
}

NNOPS_TEST(pooling_simd_avg_vs_ref) {
    // Average pooling with padding — SIMD h4/h1 interior, scalar pad regions
    auto [in_vec, input] = test::make_random_tensor({1, 2, 16, 16});
    const int64_t oshape[] = {1, 2, 14, 14};
    std::vector<float> out_simd(1 * 2 * 14 * 14);
    std::vector<float> out_ref(1 * 2 * 14 * 14);

    TensorView out_s(oshape, DataType::f32, out_simd.data());
    TensorView out_r(oshape, DataType::f32, out_ref.data());

    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};

    pooling(input, out_s, attrs);

    const int64_t N = 1, C = 2, IH = 16, IW = 16, OH = 14, OW = 14;
    ref_pooling_2d_avg(input.ptr<float>(), out_ref.data(),
                       N, C, IH, IW, OH, OW, 3, 3, 1, 1, 1, 1);

    NNOPS_EXPECT_TRUE(test::allclose(out_s, out_r, 1e-4f, 1e-4f));
}

NNOPS_TEST(pooling_simd_max_add_to) {
    // add_to: output += pooling(input)
    auto [in_vec, input] = test::make_random_tensor({1, 2, 8, 8});
    const int64_t oshape[] = {1, 2, 6, 6};
    std::vector<float> initial_buf(1 * 2 * 6 * 6);
    std::vector<float> out_buf(1 * 2 * 6 * 6);
    std::vector<float> ref_buf(1 * 2 * 6 * 6);

    // Fill initial output with known values
    for (auto& v : initial_buf) { v = 0.5f; }
    out_buf = initial_buf;
    ref_buf = initial_buf;

    TensorView out_s(oshape, DataType::f32, out_buf.data());
    TensorView out_r(oshape, DataType::f32, ref_buf.data());

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.add_to = true;

    pooling(input, out_s, attrs);

    // Reference: compute maxpool without add_to, then add initial values
    PoolingAttributes attrs_no_add = attrs;
    attrs_no_add.add_to = false;

    const int64_t N = 1, C = 2, IH = 8, IW = 8, OH = 6, OW = 6;
    ref_pooling_2d_max(input.ptr<float>(), ref_buf.data(),
                       N, C, IH, IW, OH, OW, 3, 3, 1, 1, 0, 0);
    for (int64_t i = 0; i < N * C * OH * OW; ++i) {
        ref_buf[i] = initial_buf[i] + ref_buf[i];
    }

    NNOPS_EXPECT_TRUE(test::allclose(out_s, out_r, 1e-4f, 1e-4f));
}

NNOPS_TEST(pooling_simd_large_input) {
    // Large input to thoroughly exercise SIMD h4/h1 and block processing
    auto [in_vec, input] = test::make_random_tensor({2, 3, 64, 64});
    const int64_t oshape[] = {2, 3, 62, 62};
    std::vector<float> out_buf(2 * 3 * 62 * 62);
    std::vector<float> ref_buf(2 * 3 * 62 * 62);

    TensorView out_s(oshape, DataType::f32, out_buf.data());
    TensorView out_r(oshape, DataType::f32, ref_buf.data());

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};

    pooling(input, out_s, attrs);

    const int64_t N = 2, C = 3, IH = 64, IW = 64, OH = 62, OW = 62;
    ref_pooling_2d_max(input.ptr<float>(), ref_buf.data(),
                       N, C, IH, IW, OH, OW, 3, 3, 1, 1, 0, 0);

    NNOPS_EXPECT_TRUE(test::allclose(out_s, out_r, 1e-4f, 1e-4f));
}

NNOPS_TEST(pooling_simd_odd_width) {
    // Odd output width tests SIMD tail (ow_simd_end < ow_end) scalar path
    auto [in_vec, input] = test::make_random_tensor({1, 2, 10, 10});
    const int64_t oshape[] = {1, 2, 8, 7};  // 7 = odd width
    std::vector<float> out_buf(1 * 2 * 8 * 7);
    std::vector<float> ref_buf(1 * 2 * 8 * 7);

    TensorView out_s(oshape, DataType::f32, out_buf.data());
    TensorView out_r(oshape, DataType::f32, ref_buf.data());

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};

    pooling(input, out_s, attrs);

    const int64_t N = 1, C = 2, IH = 10, IW = 10, OH = 8, OW = 7;
    ref_pooling_2d_max(input.ptr<float>(), ref_buf.data(),
                       N, C, IH, IW, OH, OW, 3, 3, 1, 1, 0, 0);

    NNOPS_EXPECT_TRUE(test::allclose(out_s, out_r, 1e-4f, 1e-4f));
}

NNOPS_TEST(pooling_simd_avg_stride_2_no_simd) {
    // AvgPool with stride=2 — SIMD disabled but result must match reference
    auto [in_vec, input] = test::make_random_tensor({1, 2, 16, 16});
    const int64_t oshape[] = {1, 2, 8, 8};
    std::vector<float> out_buf(1 * 2 * 8 * 8);
    std::vector<float> ref_buf(1 * 2 * 8 * 8);

    TensorView out_s(oshape, DataType::f32, out_buf.data());
    TensorView out_r(oshape, DataType::f32, ref_buf.data());

    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};

    pooling(input, out_s, attrs);

    const int64_t N = 1, C = 2, IH = 16, IW = 16, OH = 8, OW = 8;
    ref_pooling_2d_avg(input.ptr<float>(), ref_buf.data(),
                       N, C, IH, IW, OH, OW, 2, 2, 2, 2, 0, 0);

    NNOPS_EXPECT_TRUE(test::allclose(out_s, out_r, 1e-4f, 1e-4f));
}

NNOPS_TEST(pooling_simd_3d_max_vs_ref) {
    // 3D MaxPool with stride=1 — SIMD h4/h1 for interior depth
    auto [in_vec, input] = test::make_random_tensor({1, 2, 8, 12, 12});
    const int64_t oshape[] = {1, 2, 6, 10, 10};
    std::vector<float> out_buf(1 * 2 * 6 * 10 * 10);
    std::vector<float> ref_buf(1 * 2 * 6 * 10 * 10);

    TensorView out_s(oshape, DataType::f32, out_buf.data());
    TensorView out_r(oshape, DataType::f32, ref_buf.data());

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {3, 3, 3};
    attrs.stride       = {1, 1, 1};

    pooling(input, out_s, attrs);

    // Inline 3D reference
    const int64_t N = 1, C = 2, ID = 8, IH = 12, IW = 12;
    const int64_t OD = 6, OH = 10, OW = 10;
    const int64_t KD = 3, KH = 3, KW = 3;
    const auto* in_ptr = input.ptr<float>();
    auto* ref_ptr = ref_buf.data();
    const int64_t in_ch_s = ID * IH * IW;
    const int64_t out_ch_s = OD * OH * OW;

    for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
        const float* in_ch = in_ptr + n * C * in_ch_s + c * in_ch_s;
        float* out_ch = ref_ptr + n * C * out_ch_s + c * out_ch_s;
        for (int64_t od = 0; od < OD; ++od) {
        for (int64_t oh = 0; oh < OH; ++oh) {
        for (int64_t ow = 0; ow < OW; ++ow) {
            float max_val = -std::numeric_limits<float>::infinity();
            bool any = false;
            for (int64_t kd = 0; kd < KD; ++kd) {
                int64_t id = od + kd;
                if (id < 0 || id >= ID) { continue; }
                for (int64_t kh = 0; kh < KH; ++kh) {
                    int64_t ih = oh + kh;
                    if (ih < 0 || ih >= IH) { continue; }
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        int64_t iw = ow + kw;
                        if (iw < 0 || iw >= IW) { continue; }
                        float val = in_ch[id * IH * IW + ih * IW + iw];
                        if (val > max_val) { max_val = val; }
                        any = true;
                    }
                }
            }
            out_ch[od * OH * OW + oh * OW + ow] = any ? max_val : 0.0f;
        }}}
    }}

    NNOPS_EXPECT_TRUE(test::allclose(out_s, out_r, 1e-4f, 1e-4f));
}

NNOPS_TEST(pooling_simd_3d_avg_vs_ref) {
    // 3D AvgPool including pad — SIMD path
    auto [in_vec, input] = test::make_random_tensor({1, 1, 8, 12, 12});
    const int64_t oshape[] = {1, 1, 6, 10, 10};
    std::vector<float> out_buf(1 * 1 * 6 * 10 * 10);
    std::vector<float> ref_buf(1 * 1 * 6 * 10 * 10);

    TensorView out_s(oshape, DataType::f32, out_buf.data());
    TensorView out_r(oshape, DataType::f32, ref_buf.data());

    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {3, 3, 3};
    attrs.stride       = {1, 1, 1};

    pooling(input, out_s, attrs);

    // Inline 3D reference
    const int64_t N = 1, C = 1, ID = 8, IH = 12, IW = 12;
    const int64_t OD = 6, OH = 10, OW = 10;
    const int64_t KD = 3, KH = 3, KW = 3;
    const float scale = 1.0f / static_cast<float>(KD * KH * KW);
    const auto* in_ptr = input.ptr<float>();
    auto* ref_ptr = ref_buf.data();
    const int64_t in_ch_s = ID * IH * IW;
    const int64_t out_ch_s = OD * OH * OW;

    for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
        const float* in_ch = in_ptr + n * C * in_ch_s + c * in_ch_s;
        float* out_ch = ref_ptr + n * C * out_ch_s + c * out_ch_s;
        for (int64_t od = 0; od < OD; ++od) {
        for (int64_t oh = 0; oh < OH; ++oh) {
        for (int64_t ow = 0; ow < OW; ++ow) {
            float sum = 0.0f;
            for (int64_t kd = 0; kd < KD; ++kd) {
                int64_t id = od + kd;
                if (id < 0 || id >= ID) { continue; }
                for (int64_t kh = 0; kh < KH; ++kh) {
                    int64_t ih = oh + kh;
                    if (ih < 0 || ih >= IH) { continue; }
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        int64_t iw = ow + kw;
                        if (iw < 0 || iw >= IW) { continue; }
                        sum += in_ch[id * IH * IW + ih * IW + iw];
                    }
                }
            }
            out_ch[od * OH * OW + oh * OW + ow] = sum * scale;
        }}}
    }}

    NNOPS_EXPECT_TRUE(test::allclose(out_s, out_r, 1e-4f, 1e-4f));
}

NNOPS_TEST(pooling_simd_avg_exclude_pad_vs_ref) {
    // AverageExcludePad — scalar-only path
    auto [in_vec, input] = test::make_random_tensor({1, 2, 4, 4});
    const int64_t oshape[] = {1, 2, 4, 4};
    std::vector<float> out_buf(1 * 2 * 4 * 4);
    std::vector<float> ref_buf(1 * 2 * 4 * 4);

    TensorView out_s(oshape, DataType::f32, out_buf.data());
    TensorView out_r(oshape, DataType::f32, ref_buf.data());

    PoolingAttributes attrs;
    attrs.type = PoolingType::AverageExcludePad;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};

    pooling(input, out_s, attrs);

    // Inline reference
    const int64_t N = 1, C = 2, IH = 4, IW = 4, OH = 4, OW = 4;
    const int64_t KH = 3, KW = 3, K_TOTAL = 9;
    const auto* in_ptr = input.ptr<float>();
    auto* ref_ptr = ref_buf.data();

    for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
        const float* in_ch = in_ptr + n * C * IH * IW + c * IH * IW;
        float* out_ch = ref_ptr + n * C * OH * OW + c * OH * OW;
        for (int64_t oh = 0; oh < OH; ++oh) {
        for (int64_t ow = 0; ow < OW; ++ow) {
            float sum = 0.0f;
            int valid = 0;
            for (int64_t kh = 0; kh < KH; ++kh) {
                int64_t ih = oh + kh - 1;
                for (int64_t kw = 0; kw < KW; ++kw) {
                    int64_t iw = ow + kw - 1;
                    if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                        sum += in_ch[ih * IW + iw];
                        ++valid;
                    }
                }
            }
            out_ch[oh * OW + ow] = valid > 0 ? sum / static_cast<float>(valid) : 0.0f;
        }}
    }}

    NNOPS_EXPECT_TRUE(test::allclose(out_s, out_r, 1e-4f, 1e-4f));
}

NNOPS_TEST(pooling_simd_lp_vs_ref) {
    // Lp pooling — scalar-only path
    auto [in_vec, input] = test::make_random_tensor({1, 1, 4, 4}, 0.0f, 2.0f);
    const int64_t oshape[] = {1, 1, 2, 2};
    std::vector<float> out_buf(1 * 1 * 2 * 2);
    std::vector<float> ref_buf(1 * 1 * 2 * 2);

    TensorView out_s(oshape, DataType::f32, out_buf.data());
    TensorView out_r(oshape, DataType::f32, ref_buf.data());

    PoolingAttributes attrs;
    attrs.type = PoolingType::Lp;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 2, 2};
    attrs.p_norm = 2;

    pooling(input, out_s, attrs);

    // Inline reference
    const int64_t N = 1, C = 1, IH = 4, IW = 4, OH = 2, OW = 2;
    const int64_t KH = 3, KW = 3, P = 2;
    const auto* in_ptr = input.ptr<float>();
    auto* ref_ptr = ref_buf.data();

    for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
        const float* in_ch = in_ptr + n * C * IH * IW + c * IH * IW;
        float* out_ch = ref_ptr + n * C * OH * OW + c * OH * OW;
        for (int64_t oh = 0; oh < OH; ++oh) {
        for (int64_t ow = 0; ow < OW; ++ow) {
            float sum = 0.0f;
            for (int64_t kh = 0; kh < KH; ++kh) {
                int64_t ih = oh * 2 + kh;
                if (ih < 0 || ih >= IH) { continue; }
                for (int64_t kw = 0; kw < KW; ++kw) {
                    int64_t iw = ow * 2 + kw;
                    if (iw < 0 || iw >= IW) { continue; }
                    sum += std::pow(std::abs(in_ch[ih * IW + iw]), static_cast<float>(P));
                }
            }
            out_ch[oh * OW + ow] = std::pow(sum, 1.0f / static_cast<float>(P));
        }}
    }}

    NNOPS_EXPECT_TRUE(test::allclose(out_s, out_r, 1e-4f, 1e-4f));
}

NNOPS_TEST(pooling_simd_3d_padding) {
    // 3D pooling with depth padding — exercises pad-depth front/back scalar regions
    auto [in_vec, input] = test::make_random_tensor({1, 1, 4, 8, 8});
    const int64_t oshape[] = {1, 1, 4, 6, 6};
    std::vector<float> out_buf(1 * 1 * 4 * 6 * 6);
    std::vector<float> ref_buf(1 * 1 * 4 * 6 * 6);

    TensorView out_s(oshape, DataType::f32, out_buf.data());
    TensorView out_r(oshape, DataType::f32, ref_buf.data());

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {3, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {1, 0, 0};

    pooling(input, out_s, attrs);

    // Inline 3D reference with padding
    const int64_t N = 1, C = 1, ID = 4, IH = 8, IW = 8;
    const int64_t OD = 4, OH = 6, OW = 6;
    const int64_t KD = 3, KH = 3, KW = 3;
    const auto* in_ptr = input.ptr<float>();
    auto* ref_ptr = ref_buf.data();

    for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
        const float* in_ch = in_ptr + n * C * ID * IH * IW + c * ID * IH * IW;
        float* out_ch = ref_ptr + n * C * OD * OH * OW + c * OD * OH * OW;
        for (int64_t od = 0; od < OD; ++od) {
        for (int64_t oh = 0; oh < OH; ++oh) {
        for (int64_t ow = 0; ow < OW; ++ow) {
            float max_val = -std::numeric_limits<float>::infinity();
            bool any = false;
            for (int64_t kd = 0; kd < KD; ++kd) {
                int64_t id = od + kd - 1;  // PD=1
                if (id < 0 || id >= ID) { continue; }
                for (int64_t kh = 0; kh < KH; ++kh) {
                    int64_t ih = oh + kh;
                    if (ih < 0 || ih >= IH) { continue; }
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        int64_t iw = ow + kw;
                        if (iw < 0 || iw >= IW) { continue; }
                        float val = in_ch[id * IH * IW + ih * IW + iw];
                        if (val > max_val) { max_val = val; }
                        any = true;
                    }
                }
            }
            out_ch[od * OH * OW + oh * OW + ow] = any ? max_val : 0.0f;
        }}}
    }}

    NNOPS_EXPECT_TRUE(test::allclose(out_s, out_r, 1e-4f, 1e-4f));
}
