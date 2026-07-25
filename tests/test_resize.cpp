/// @file test_resize.cpp
/// @brief Tests for the Resize (interpolation) operator.

#include "nnops/ops/resize.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

// ============================================================
// 2D Nearest-neighbor tests
// ============================================================

NNOPS_TEST(resize_2d_nearest_upsample) {
    // 1x1x2x2 -> 1x1x4x4, nearest with Asymmetric mode
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t oshape[] = {1, 1, 4, 4};
    float in_data[4] = {1, 2, 3, 4};
    float out_data[16] = {};

    TensorView input(ishape, DataType::f32, in_data);
    TensorView output(oshape, DataType::f32, out_data);

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 4, 4};

    resize(input, output, attrs);

    // scale=0.5, each input pixel maps to a 2x2 block
    // Expected (row-major):
    //   row 0: [1, 2, 2, 2]
    //   row 1: [3, 4, 4, 4]
    //   row 2: [3, 4, 4, 4]
    //   row 3: [3, 4, 4, 4]
    NNOPS_EXPECT_NEAR(out_data[0],  1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[3],  2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[4],  3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[7],  4.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[12], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[15], 4.0f, 1e-6f);
}

NNOPS_TEST(resize_2d_nearest_downsample) {
    // 1x1x4x4 -> 1x1x2x2, nearest with Asymmetric mode
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t oshape[] = {1, 1, 2, 2};
    float in_data[16] = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
       13, 14, 15, 16
    };
    float out_data[4] = {};

    TensorView input(ishape, DataType::f32, in_data);
    TensorView output(oshape, DataType::f32, out_data);

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 2, 2};

    resize(input, output, attrs);

    // scale=2.0, dst 0->src 0, dst 1->src 2
    // Expected: in[0,0]=1, in[0,2]=3, in[2,0]=9, in[2,2]=11
    NNOPS_EXPECT_NEAR(out_data[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 9.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[3], 11.0f, 1e-6f);
}

NNOPS_TEST(resize_2d_nearest_identity) {
    // Identity resize: same size should be exact copy with Asymmetric mode
    auto [in_vec, input] = test::make_random_tensor({1, 3, 8, 8});
    std::vector<float> out_buf(1 * 3 * 8 * 8);
    const int64_t oshape[] = {1, 3, 8, 8};
    TensorView output(oshape, DataType::f32, out_buf.data());

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 8, 8};

    resize(input, output, attrs);

    NNOPS_EXPECT_TRUE(test::allclose(input, output, 1e-6f, 1e-6f));
}

// ============================================================
// 2D Linear (bilinear) tests
// ============================================================

NNOPS_TEST(resize_2d_linear_align_corners_corners_exact) {
    // 1x1x2x2 -> 1x1x4x4, bilinear with AlignCorners
    // With AlignCorners, the 4 corners of output exactly match input corners.
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t oshape[] = {1, 1, 4, 4};
    float in_data[4] = {1, 2, 3, 4};
    float out_data[16] = {};

    TensorView input(ishape, DataType::f32, in_data);
    TensorView output(oshape, DataType::f32, out_data);

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::AlignCorners;
    attrs.output_size = {0, 4, 4};

    resize(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0],  1.0f, 1e-5f);   // top-left
    NNOPS_EXPECT_NEAR(out_data[3],  2.0f, 1e-5f);   // top-right
    NNOPS_EXPECT_NEAR(out_data[12], 3.0f, 1e-5f);   // bottom-left
    NNOPS_EXPECT_NEAR(out_data[15], 4.0f, 1e-5f);   // bottom-right

    // Check no NaN/inf for interior
    for (size_t i = 0; i < 16; ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_data[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_data[i]));
    }
}

NNOPS_TEST(resize_2d_linear_identity) {
    // Identity resize should be exact copy in any mode that preserves identity
    auto [in_vec, input] = test::make_random_tensor({1, 1, 4, 4});
    std::vector<float> out_buf(16);
    const int64_t oshape[] = {1, 1, 4, 4};
    TensorView output(oshape, DataType::f32, out_buf.data());

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 4, 4};

    resize(input, output, attrs);

    NNOPS_EXPECT_TRUE(test::allclose(input, output, 1e-4f, 1e-4f));
}

NNOPS_TEST(resize_2d_linear_upsample_no_nan) {
    auto [in_vec, input] = test::make_random_tensor({1, 2, 8, 8});
    std::vector<float> out_buf(1 * 2 * 20 * 20);
    const int64_t oshape[] = {1, 2, 20, 20};
    TensorView output(oshape, DataType::f32, out_buf.data());

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 20, 20};

    resize(input, output, attrs);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

// ============================================================
// 3D Nearest-neighbor tests
// ============================================================

NNOPS_TEST(resize_3d_nearest_basic) {
    // 1x1x2x2x2 -> 1x1x4x4x4, nearest with Asymmetric
    const int64_t ishape[] = {1, 1, 2, 2, 2};
    const int64_t oshape[] = {1, 1, 4, 4, 4};
    std::vector<float> in_buf(8);
    for (int i = 0; i < 8; ++i) { in_buf[i] = static_cast<float>(i + 1); }
    std::vector<float> out_buf(64, 0.0f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView output(oshape, DataType::f32, out_buf.data());

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {4, 4, 4};

    resize(input, output, attrs);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

NNOPS_TEST(resize_3d_nearest_identity) {
    auto [in_vec, input] = test::make_random_tensor({1, 2, 4, 4, 4});
    std::vector<float> out_buf(1 * 2 * 4 * 4 * 4);
    const int64_t oshape[] = {1, 2, 4, 4, 4};
    TensorView output(oshape, DataType::f32, out_buf.data());

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {4, 4, 4};

    resize(input, output, attrs);

    NNOPS_EXPECT_TRUE(test::allclose(input, output, 1e-6f, 1e-6f));
}

// ============================================================
// 3D Linear (trilinear) tests
// ============================================================

NNOPS_TEST(resize_3d_linear_basic) {
    auto [in_vec, input] = test::make_random_tensor({1, 1, 4, 6, 6}, -1.0f, 1.0f);
    std::vector<float> out_buf(1 * 1 * 8 * 12 * 12);
    const int64_t oshape[] = {1, 1, 8, 12, 12};
    TensorView output(oshape, DataType::f32, out_buf.data());

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {8, 12, 12};

    resize(input, output, attrs);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

NNOPS_TEST(resize_3d_linear_identity) {
    auto [in_vec, input] = test::make_random_tensor({1, 2, 4, 4, 4});
    std::vector<float> out_buf(1 * 2 * 4 * 4 * 4);
    const int64_t oshape[] = {1, 2, 4, 4, 4};
    TensorView output(oshape, DataType::f32, out_buf.data());

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {4, 4, 4};

    resize(input, output, attrs);

    NNOPS_EXPECT_TRUE(test::allclose(input, output, 1e-4f, 1e-4f));
}

// ============================================================
// Random stress tests (multi-channel)
// ============================================================

NNOPS_TEST(resize_2d_multichannel) {
    auto [in_vec, input] = test::make_random_tensor({2, 3, 32, 32});
    std::vector<float> out_buf(2 * 3 * 16 * 16);
    const int64_t oshape[] = {2, 3, 16, 16};
    TensorView output(oshape, DataType::f32, out_buf.data());

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 16, 16};

    resize(input, output, attrs);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

NNOPS_TEST(resize_3d_multichannel) {
    auto [in_vec, input] = test::make_random_tensor({1, 3, 8, 16, 16});
    std::vector<float> out_buf(1 * 3 * 4 * 8 * 8);
    const int64_t oshape[] = {1, 3, 4, 8, 8};
    TensorView output(oshape, DataType::f32, out_buf.data());

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {4, 8, 8};

    resize(input, output, attrs);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

// ============================================================
// Class API vs Functional API
// ============================================================

NNOPS_TEST(resize_class_vs_functional) {
    auto [in_vec, input] = test::make_random_tensor({1, 2, 8, 8});
    std::vector<float> out1_buf(1 * 2 * 16 * 16);
    std::vector<float> out2_buf(1 * 2 * 16 * 16);
    const int64_t oshape[] = {1, 2, 16, 16};

    TensorView out1(oshape, DataType::f32, out1_buf.data());
    TensorView out2(oshape, DataType::f32, out2_buf.data());

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 16, 16};

    // Functional API
    resize(input, out1, attrs);

    // Class API
    auto op = Resize::create(attrs, Backend::CPU);
    const TensorView ins[] = {input};
    op->compute(out2, ins);

    NNOPS_EXPECT_TRUE(test::allclose(out1, out2, 1e-6f, 1e-6f));
}

// ============================================================
// Add-to mode
// ============================================================

NNOPS_TEST(resize_add_to) {
    auto [in_vec, input] = test::make_random_tensor({1, 1, 4, 4});
    std::vector<float> out_buf(1 * 1 * 8 * 8, 0.5f);  // pre-filled
    const int64_t oshape[] = {1, 1, 8, 8};
    TensorView output(oshape, DataType::f32, out_buf.data());

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 8, 8};
    attrs.add_to = true;

    resize(input, output, attrs);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

// ============================================================
// HalfPixel coordinate mode
// ============================================================

NNOPS_TEST(resize_2d_linear_half_pixel) {
    auto [in_vec, input] = test::make_random_tensor({1, 1, 8, 8});
    std::vector<float> out_buf(1 * 1 * 16 * 16);
    const int64_t oshape[] = {1, 1, 16, 16};
    TensorView output(oshape, DataType::f32, out_buf.data());

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 16, 16};

    resize(input, output, attrs);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}
