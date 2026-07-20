/// Unit tests for Conv2D operator (CPU reference).

#include "nnops/ops/conv2d.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cstring>

using namespace nnops;

// ============================================================
// Hand-verified small tests
// ============================================================

NNOPS_TEST(conv2d_basic_no_pad) {
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

    Conv2DAttributes attrs;
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};

    conv2d(input, weight, output, attrs);

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], 4.5f, 1e-4f);
    }
}

NNOPS_TEST(conv2d_with_bias) {
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

    Conv2DAttributes attrs;
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};

    conv2d(input, weight, bias, output, attrs);

    // Each output = 4.5 + 0.5 = 5.0
    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], 5.0f, 1e-4f);
    }
}

NNOPS_TEST(conv2d_stride_2) {
    // 1x1x6x6 input, 1x1x3x3 kernel, stride=2, pad=0
    // Expected: 2x2 output
    const int64_t ishape[] = {1, 1, 6, 6};
    const int64_t wshape[] = {1, 1, 3, 3};
    const int64_t oshape[] = {1, 1, 2, 2};

    std::vector<float> in_buf(36, 1.0f);
    std::vector<float> w_buf(9, 0.5f);
    std::vector<float> out_buf(4, 0.0f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());
    TensorView output(oshape, DataType::f32, out_buf.data());

    Conv2DAttributes attrs;
    attrs.stride  = {2, 2};
    attrs.padding = {0, 0};

    conv2d(input, weight, output, attrs);

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], 4.5f, 1e-4f);
    }
}

NNOPS_TEST(conv2d_padding_1) {
    // 1x1x2x2 input, 1x1x3x3 kernel, stride=1, pad=1
    // Output should be 2x2 (same spatial size)
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t wshape[] = {1, 1, 3, 3};
    const int64_t oshape[] = {1, 1, 2, 2};

    std::vector<float> in_buf(4, 1.0f);
    std::vector<float> w_buf(9, 0.5f);
    std::vector<float> out_buf(4, 0.0f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());
    TensorView output(oshape, DataType::f32, out_buf.data());

    Conv2DAttributes attrs;
    attrs.stride  = {1, 1};
    attrs.padding = {1, 1};

    conv2d(input, weight, output, attrs);

    // Each output element sees partial kernel region (padded zeros)
    // Top-left: only bottom-right 2x2 of kernel overlaps => 1.0 * 0.5 * 4 = 2.0
    NNOPS_EXPECT_NEAR(out_buf[0], 2.0f, 1e-4f);
    // Top-right: bottom-left 2x2 of kernel overlaps => 2.0
    NNOPS_EXPECT_NEAR(out_buf[1], 2.0f, 1e-4f);
}

NNOPS_TEST(conv2d_grouped) {
    // Grouped conv with groups=2, IC=2, OC=2
    // Each group: 1 input channel -> 1 output channel
    const int64_t ishape[] = {1, 2, 3, 3};
    const int64_t wshape[] = {2, 1, 2, 2};  // [OC, IC/G, KH, KW] = [2, 1, 2, 2]
    const int64_t oshape[] = {1, 2, 2, 2};

    std::vector<float> in_buf(18, 1.0f);
    std::vector<float> w_buf(8, 1.0f);
    std::vector<float> out_buf(8, 0.0f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());
    TensorView output(oshape, DataType::f32, out_buf.data());

    Conv2DAttributes attrs;
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};
    attrs.groups  = 2;

    conv2d(input, weight, output, attrs);

    // Each group: 1 input channel convolved with 1 filter => 4 products * 1.0 = 4.0
    for (int i = 0; i < 8; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], 4.0f, 1e-4f);
    }
}

// ============================================================
// Random comparison test (reference vs reference = identity)
// ============================================================

NNOPS_TEST(conv2d_random_small) {
    auto [in_vec, input]   = test::make_random_tensor({1, 3, 8, 8});
    auto [w_vec, weight]   = test::make_random_tensor({4, 3, 3, 3});
    std::vector<float> out_buf(1 * 4 * 6 * 6);

    const int64_t oshape[] = {1, 4, 6, 6};
    TensorView output(oshape, DataType::f32, out_buf.data());

    Conv2DAttributes attrs;
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};

    // Should not crash or produce NaN
    conv2d(input, weight, output, attrs);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

NNOPS_TEST(conv2d_class_api) {
    // Test class-based API produces same result as functional API
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t wshape[] = {1, 1, 3, 3};
    const int64_t oshape[] = {1, 1, 2, 2};

    std::vector<float> in_buf(16, 1.0f);
    std::vector<float> w_buf(9, 0.5f);
    std::vector<float> out1_buf(4, 0.0f);
    std::vector<float> out2_buf(4, 0.0f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());
    TensorView out1(oshape, DataType::f32, out1_buf.data());
    TensorView out2(oshape, DataType::f32, out2_buf.data());

    Conv2DAttributes attrs;
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};

    // Functional API
    conv2d(input, weight, out1, attrs);

    // Class API
    auto op = Conv2D::create(attrs, Backend::CPU);
    const TensorView ins[] = {input, weight};
    op->compute(out2, ins);

    NNOPS_EXPECT_TRUE(test::allclose(out1, out2, 1e-6f, 1e-6f));
}
