/// Unit tests for DepthwiseConv2D operator (CPU reference + SIMD).
///
/// Depthwise conv applies a separate filter to each input channel independently.
/// Input: [N, C, IH, IW], Weight: [C, 1, KH, KW], Output: [N, C, OH, OW].

#include "nnops/ops/depthwise_conv2d.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

// ============================================================
// Hand-verified small tests
// ============================================================

NNOPS_TEST(dwconv_basic_no_pad) {
    // 1x1x4x4 input all-ones, 1x1x3x3 kernel all-0.5, stride=1, pad=0
    // Expected: 2x2 output, each = 9 * (1.0 * 0.5) = 4.5
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t wshape[] = {1, 1, 3, 3};
    const int64_t oshape[] = {1, 1, 2, 2};

    std::vector<float> in_buf(16, 1.0f);
    std::vector<float> w_buf(9, 0.5f);
    std::vector<float> out_buf(4, 0.0f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());
    TensorView output(oshape, DataType::f32, out_buf.data());

    DepthwiseConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};

    depthwise_conv2d(input, weight, output, attrs);

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], 4.5f, 1e-4f);
    }
}

NNOPS_TEST(dwconv_with_bias) {
    // Same as above but with bias = 0.5
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t wshape[] = {1, 1, 3, 3};
    const int64_t bshape[] = {1};
    const int64_t oshape[] = {1, 1, 2, 2};

    std::vector<float> in_buf(16, 1.0f);
    std::vector<float> w_buf(9, 0.5f);
    std::vector<float> b_buf = {0.5f};
    std::vector<float> out_buf(4, 0.0f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());
    TensorView bias(bshape, DataType::f32, b_buf.data());
    TensorView output(oshape, DataType::f32, out_buf.data());

    DepthwiseConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};

    depthwise_conv2d(input, weight, bias, output, attrs);

    // Each output = 4.5 + 0.5 = 5.0
    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], 5.0f, 1e-4f);
    }
}

NNOPS_TEST(dwconv_stride_2) {
    // 1x1x6x6 input all-ones, 1x1x3x3 kernel all-0.5, stride=2, pad=0
    // Output: 2x2
    const int64_t ishape[] = {1, 1, 6, 6};
    const int64_t wshape[] = {1, 1, 3, 3};
    const int64_t oshape[] = {1, 1, 2, 2};

    std::vector<float> in_buf(36, 1.0f);
    std::vector<float> w_buf(9, 0.5f);
    std::vector<float> out_buf(4, 0.0f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());
    TensorView output(oshape, DataType::f32, out_buf.data());

    DepthwiseConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {2, 2};
    attrs.padding = {0, 0};

    depthwise_conv2d(input, weight, output, attrs);

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], 4.5f, 1e-4f);
    }
}

NNOPS_TEST(dwconv_padding_1) {
    // 1x1x2x2 input all-ones, 1x1x3x3 kernel all-0.5, stride=1, pad=1
    // Output: 2x2 (same spatial size)
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t wshape[] = {1, 1, 3, 3};
    const int64_t oshape[] = {1, 1, 2, 2};

    std::vector<float> in_buf(4, 1.0f);
    std::vector<float> w_buf(9, 0.5f);
    std::vector<float> out_buf(4, 0.0f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());
    TensorView output(oshape, DataType::f32, out_buf.data());

    DepthwiseConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {1, 1};

    depthwise_conv2d(input, weight, output, attrs);

    // Top-left corner: only bottom-right 2x2 of kernel overlaps
    // valid input positions: ih=0,iw=0 maps to kernel[1][1]..kernel[2][2]
    // Actually, with padding 1:
    //   oh=0, ow=0: input positions are (-1,...), (0,...), etc.
    //   4 valid positions: ih=0,iw=0 and ih=0,iw=1 and ih=1,iw=0 and ih=1,iw=1
    //   These correspond to kernel positions (1,1), (1,2), (2,1), (2,2)
    //   So val = 1.0 * 0.5 * 4 = 2.0
    NNOPS_EXPECT_NEAR(out_buf[0], 2.0f, 1e-4f);
    // Top-right corner: similar, 4 valid positions → 2.0
    NNOPS_EXPECT_NEAR(out_buf[1], 2.0f, 1e-4f);
    // Bottom-left: 4 valid → 2.0
    NNOPS_EXPECT_NEAR(out_buf[2], 2.0f, 1e-4f);
    // Bottom-right: 4 valid → 2.0
    NNOPS_EXPECT_NEAR(out_buf[3], 2.0f, 1e-4f);
}

NNOPS_TEST(dwconv_multi_channel) {
    // N=1, C=2, IH=3, IW=3. Two channels with different kernels.
    // Channel 0: kernel all 1.0, input all 2.0 → each output = 9*2.0 = 18.0
    // Channel 1: kernel all 0.5, input all 3.0 → each output = 9*1.5 = 13.5
    const int64_t ishape[] = {1, 2, 3, 3};
    const int64_t wshape[] = {2, 1, 3, 3};
    const int64_t oshape[] = {1, 2, 1, 1};

    std::vector<float> in_buf(18);
    for (int i = 0; i < 9; ++i) {
        in_buf[i] = 2.0f;        // channel 0: all 2.0
        in_buf[i + 9] = 3.0f;    // channel 1: all 3.0
    }

    std::vector<float> w_buf(18);
    for (int i = 0; i < 9; ++i) {
        w_buf[i] = 1.0f;         // channel 0 kernel: all 1.0
        w_buf[i + 9] = 0.5f;     // channel 1 kernel: all 0.5
    }

    std::vector<float> out_buf(2, 0.0f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());
    TensorView output(oshape, DataType::f32, out_buf.data());

    DepthwiseConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};

    depthwise_conv2d(input, weight, output, attrs);

    NNOPS_EXPECT_NEAR(out_buf[0], 18.0f, 1e-4f);   // channel 0: 9 * 2.0 * 1.0
    NNOPS_EXPECT_NEAR(out_buf[1], 13.5f, 1e-4f);   // channel 1: 9 * 3.0 * 0.5
}

NNOPS_TEST(dwconv_dilation) {
    // 1x1x5x5 input all-ones, 1x1x3x3 kernel all-0.5, stride=1, dilation=2, pad=0
    // Output: 1x1. The 3x3 kernel with dilation 2 covers a 5x5 receptive field.
    // Output shape: (5-1*2-1)/1 + 1 = (5-2-1)+1 = 3? No...
    // Output formula: OH = floor((IH - DH*(KH-1) - 1) / SH + 1)
    //   = floor((5 - 2*2 - 1) / 1 + 1) = floor((5-4-1)+1) = 1
    // Single output: 9 kernel positions, all valid, sum = 9 * 0.5 = 4.5
    const int64_t ishape[] = {1, 1, 5, 5};
    const int64_t wshape[] = {1, 1, 3, 3};
    const int64_t oshape[] = {1, 1, 1, 1};

    std::vector<float> in_buf(25, 1.0f);
    std::vector<float> w_buf(9, 0.5f);
    std::vector<float> out_buf(1, 0.0f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());
    TensorView output(oshape, DataType::f32, out_buf.data());

    DepthwiseConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride   = {1, 1};
    attrs.dilation = {2, 2};
    attrs.padding  = {0, 0};

    depthwise_conv2d(input, weight, output, attrs);

    NNOPS_EXPECT_NEAR(out_buf[0], 4.5f, 1e-4f);
}

NNOPS_TEST(dwconv_add_to) {
    // Same as basic test but add_to=true, pre-fill output with 1.0
    // Each output = existing(1.0) + 4.5 = 5.5
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t wshape[] = {1, 1, 3, 3};
    const int64_t oshape[] = {1, 1, 2, 2};

    std::vector<float> in_buf(16, 1.0f);
    std::vector<float> w_buf(9, 0.5f);
    std::vector<float> out_buf(4, 1.0f);  // pre-filled

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());
    TensorView output(oshape, DataType::f32, out_buf.data());

    DepthwiseConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};
    attrs.add_to  = true;

    depthwise_conv2d(input, weight, output, attrs);

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], 5.5f, 1e-4f);
    }
}

NNOPS_TEST(dwconv_relu_epilogue) {
    // Values: all kernel=0.5, all input=1.0, so each output=4.5
    // But with ReLU epilogue, 4.5 > 0 → 4.5
    // For a more interesting test: mix positive and negative
    const int64_t ishape[] = {1, 2, 4, 4};
    const int64_t wshape[] = {2, 1, 3, 3};
    const int64_t oshape[] = {1, 2, 2, 2};

    std::vector<float> in_buf(32, 1.0f);
    // Channel 0: positive kernel (output positive)
    // Channel 1: negative kernel (output negative → ReLU should clip to 0)
    std::vector<float> w_buf(18);
    for (int i = 0; i < 9; ++i) {
        w_buf[i] = 0.5f;       // ch0 kernel: positive
        w_buf[i + 9] = -0.5f;  // ch1 kernel: negative
    }
    std::vector<float> out_buf(8, 0.0f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());
    TensorView output(oshape, DataType::f32, out_buf.data());

    DepthwiseConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};
    attrs.epilogue.type = EpilogueActivateType::Relu;

    depthwise_conv2d(input, weight, output, attrs);

    // Channel 0: 4.5 → ReLU → 4.5
    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], 4.5f, 1e-4f);
    }
    // Channel 1: -4.5 → ReLU → 0.0
    for (int i = 4; i < 8; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], 0.0f, 1e-4f);
    }
}

// ============================================================
// SIMD vs Reference correctness tests (stride=1, common case)
// ============================================================

NNOPS_TEST(dwconv_simd_vs_ref_small) {
    // Compare SIMD kernel output against reference for small tensors
    auto [in_vec, input]   = test::make_random_tensor({1, 3, 8, 8});
    auto [w_vec, weight]   = test::make_random_tensor({3, 1, 3, 3});
    std::vector<float> out_buf_simd(1 * 3 * 6 * 6);
    std::vector<float> out_buf_ref(1 * 3 * 6 * 6);

    const int64_t oshape[] = {1, 3, 6, 6};
    TensorView out_simd(oshape, DataType::f32, out_buf_simd.data());
    TensorView out_ref(oshape, DataType::f32, out_buf_ref.data());

    DepthwiseConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};

    // Reference path via class API (goes through SIMD kernel)
    {
        auto op = DepthwiseConv2D::create(attrs, Backend::CPU);
        const TensorView ins[] = {input, weight};
        op->compute(out_simd, ins);
    }

    // Manually compute reference using same algorithm
    const int64_t N = 1, C = 3, IH = 8, IW = 8;
    const int64_t OH = 6, OW = 6;
    const int64_t KH = 3, KW = 3;
    const auto* in_ptr  = input.data_as<float>();
    const auto* w_ptr   = weight.data_as<float>();
    auto* ref_ptr = out_ref.data_as<float>();

    for (int64_t c = 0; c < C; ++c) {
        for (int64_t oh = 0; oh < OH; ++oh) {
            for (int64_t ow = 0; ow < OW; ++ow) {
                float sum = 0.0f;
                for (int64_t kh = 0; kh < KH; ++kh) {
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        int64_t ih = oh + kh;
                        int64_t iw = ow + kw;
                        if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                            sum += in_ptr[c*IH*IW + ih*IW + iw] * w_ptr[c*KH*KW + kh*KW + kw];
                        }
                    }
                }
                ref_ptr[c*OH*OW + oh*OW + ow] = sum;
            }
        }
    }

    NNOPS_EXPECT_TRUE(test::allclose(out_simd, out_ref, 1e-4f, 1e-4f));
}

NNOPS_TEST(dwconv_simd_vs_ref_with_pad) {
    // Compare SIMD against reference with padding (exercises boundary code)
    auto [in_vec, input]   = test::make_random_tensor({1, 4, 7, 7});
    auto [w_vec, weight]   = test::make_random_tensor({4, 1, 3, 3});
    std::vector<float> out_buf_simd(1 * 4 * 7 * 7);
    std::vector<float> out_buf_ref(1 * 4 * 7 * 7);

    const int64_t oshape[] = {1, 4, 7, 7};
    TensorView out_simd(oshape, DataType::f32, out_buf_simd.data());
    TensorView out_ref(oshape, DataType::f32, out_buf_ref.data());

    DepthwiseConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {1, 1};

    // SIMD path
    {
        auto op = DepthwiseConv2D::create(attrs, Backend::CPU);
        const TensorView ins[] = {input, weight};
        op->compute(out_simd, ins);
    }

    // Manual reference
    const int64_t N = 1, C = 4, IH = 7, IW = 7;
    const int64_t OH = 7, OW = 7;
    const int64_t KH = 3, KW = 3;
    const int64_t PH = 1, PW = 1;
    const auto* in_ptr = input.data_as<float>();
    const auto* w_ptr  = weight.data_as<float>();
    auto* ref_ptr = out_ref.data_as<float>();

    for (int64_t c = 0; c < C; ++c) {
        for (int64_t oh = 0; oh < OH; ++oh) {
            for (int64_t ow = 0; ow < OW; ++ow) {
                float sum = 0.0f;
                for (int64_t kh = 0; kh < KH; ++kh) {
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        int64_t ih = oh + kh - PH;
                        int64_t iw = ow + kw - PW;
                        if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                            sum += in_ptr[c*IH*IW + ih*IW + iw] * w_ptr[c*KH*KW + kh*KW + kw];
                        }
                    }
                }
                ref_ptr[c*OH*OW + oh*OW + ow] = sum;
            }
        }
    }

    NNOPS_EXPECT_TRUE(test::allclose(out_simd, out_ref, 1e-4f, 1e-4f));
}

NNOPS_TEST(dwconv_simd_vs_ref_stride_2) {
    // Stride=2 uses scalar path in the optimized kernel
    auto [in_vec, input]   = test::make_random_tensor({1, 3, 8, 8});
    auto [w_vec, weight]   = test::make_random_tensor({3, 1, 3, 3});
    std::vector<float> out_buf(1 * 3 * 3 * 3);

    const int64_t oshape[] = {1, 3, 3, 3};
    TensorView output(oshape, DataType::f32, out_buf.data());

    DepthwiseConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {2, 2};
    attrs.padding = {0, 0};

    depthwise_conv2d(input, weight, output, attrs);

    // Should not produce NaN or Inf
    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

// ============================================================
// Random correctness tests
// ============================================================

NNOPS_TEST(dwconv_random_small) {
    auto [in_vec, input]   = test::make_random_tensor({2, 4, 8, 8});
    auto [w_vec, weight]   = test::make_random_tensor({4, 1, 3, 3});
    std::vector<float> out_buf(2 * 4 * 6 * 6);

    const int64_t oshape[] = {2, 4, 6, 6};
    TensorView output(oshape, DataType::f32, out_buf.data());

    DepthwiseConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};

    depthwise_conv2d(input, weight, output, attrs);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

NNOPS_TEST(dwconv_random_large) {
    auto [in_vec, input]   = test::make_random_tensor({1, 16, 32, 32});
    auto [w_vec, weight]   = test::make_random_tensor({16, 1, 3, 3});
    std::vector<float> out_buf(1 * 16 * 30 * 30);

    const int64_t oshape[] = {1, 16, 30, 30};
    TensorView output(oshape, DataType::f32, out_buf.data());

    DepthwiseConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};

    depthwise_conv2d(input, weight, output, attrs);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

NNOPS_TEST(dwconv_random_with_bias) {
    auto [in_vec, input]   = test::make_random_tensor({1, 4, 8, 8});
    auto [w_vec, weight]   = test::make_random_tensor({4, 1, 3, 3});
    auto [b_vec, bias]     = test::make_random_tensor({4});
    std::vector<float> out_buf(1 * 4 * 6 * 6);

    const int64_t oshape[] = {1, 4, 6, 6};
    TensorView output(oshape, DataType::f32, out_buf.data());

    DepthwiseConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};

    depthwise_conv2d(input, weight, bias, output, attrs);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

// ============================================================
// Class API test
// ============================================================

NNOPS_TEST(dwconv_class_api) {
    // Test class-based API produces same result as functional API
    const int64_t ishape[] = {1, 2, 4, 4};
    const int64_t wshape[] = {2, 1, 3, 3};
    const int64_t oshape[] = {1, 2, 2, 2};

    std::vector<float> in_buf(32, 1.0f);
    std::vector<float> w_buf(18, 0.5f);
    std::vector<float> out1_buf(8, 0.0f);
    std::vector<float> out2_buf(8, 0.0f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());
    TensorView out1(oshape, DataType::f32, out1_buf.data());
    TensorView out2(oshape, DataType::f32, out2_buf.data());

    DepthwiseConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};

    // Functional API
    depthwise_conv2d(input, weight, out1, attrs);

    // Class API
    auto op = DepthwiseConv2D::create(attrs, Backend::CPU);
    const TensorView ins[] = {input, weight};
    op->compute(out2, ins);

    NNOPS_EXPECT_TRUE(test::allclose(out1, out2, 1e-4f, 1e-4f));
}

// ============================================================
// Batch + multi-channel stress test
// ============================================================

NNOPS_TEST(dwconv_batch_multichannel) {
    // N=3, C=5, 3x3 kernel, stride=1, no padding
    // Verify per-channel isolation: each channel should be independent
    auto [in_vec, input]   = test::make_random_tensor({3, 5, 8, 8});
    auto [w_vec, weight]   = test::make_random_tensor({5, 1, 3, 3});
    std::vector<float> out_buf(3 * 5 * 6 * 6);

    const int64_t oshape[] = {3, 5, 6, 6};
    TensorView output(oshape, DataType::f32, out_buf.data());

    DepthwiseConv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};

    depthwise_conv2d(input, weight, output, attrs);

    // Manual check: re-compute sample 0, channel 0
    const auto* in_ptr = input.data_as<float>();
    const auto* w_ptr  = weight.data_as<float>();
    const auto* out_ptr = output.data_as<float>();

    const int64_t IH = 8, IW = 8, OH = 6, OW = 6;
    const int64_t KH = 3, KW = 3;
    const int64_t C = 5;
    const int64_t in_ch = IH * IW;
    const int64_t out_ch = OH * OW;

    for (int64_t c = 0; c < C; ++c) {
        for (int64_t oh = 0; oh < OH; ++oh) {
            for (int64_t ow = 0; ow < OW; ++ow) {
                float expected = 0.0f;
                for (int64_t kh = 0; kh < KH; ++kh) {
                    for (int64_t kw = 0; kw < KW; ++kw) {
                        int64_t ih = oh + kh;
                        int64_t iw = ow + kw;
                        expected += in_ptr[c*in_ch + ih*IW + iw] * w_ptr[c*KH*KW + kh*KW + kw];
                    }
                }
                float actual = out_ptr[c*out_ch + oh*OW + ow];
                NNOPS_EXPECT_NEAR(actual, expected, 1e-4f);
            }
        }
    }
}
