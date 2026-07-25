/// @file test_unary.cpp
/// @brief Unit tests for Unary operator (CPU backend).

#include "nnops/ops/unary.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

// ============================================================
// Hand-verified tests
// ============================================================

NNOPS_TEST(unary_exp) {
    const int64_t shape[] = {3};
    float in_data[]  = {0.0f, 1.0f, 2.0f};
    float out_data[3] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(shape, DataType::f32, out_data);

    UnaryAttributes attrs;
    attrs.type = UnaryType::Exp;
    unary(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], std::exp(0.0f), 1e-5f);
    NNOPS_EXPECT_NEAR(out_data[1], std::exp(1.0f), 1e-5f);
    NNOPS_EXPECT_NEAR(out_data[2], std::exp(2.0f), 1e-5f);
}

NNOPS_TEST(unary_log) {
    const int64_t shape[] = {3};
    float in_data[]  = {1.0f, 2.0f, 10.0f};
    float out_data[3] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(shape, DataType::f32, out_data);

    UnaryAttributes attrs;
    attrs.type = UnaryType::Log;
    unary(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], std::log(1.0f), 1e-5f);
    NNOPS_EXPECT_NEAR(out_data[1], std::log(2.0f), 1e-5f);
    NNOPS_EXPECT_NEAR(out_data[2], std::log(10.0f), 1e-5f);
}

NNOPS_TEST(unary_sin) {
    const int64_t shape[] = {3};
    float in_data[]  = {0.0f, 1.5707963f, 3.1415927f};
    float out_data[3] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(shape, DataType::f32, out_data);

    UnaryAttributes attrs;
    attrs.type = UnaryType::Sin;
    unary(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 0.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_data[1], 1.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_data[2], 0.0f, 1e-5f);
}

NNOPS_TEST(unary_cos) {
    const int64_t shape[] = {3};
    float in_data[]  = {0.0f, 1.5707963f, 3.1415927f};
    float out_data[3] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(shape, DataType::f32, out_data);

    UnaryAttributes attrs;
    attrs.type = UnaryType::Cos;
    unary(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 1.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_data[1], 0.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_data[2], -1.0f, 1e-4f);
}

NNOPS_TEST(unary_tanh) {
    const int64_t shape[] = {3};
    float in_data[]  = {0.0f, 1.0f, -1.0f};
    float out_data[3] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(shape, DataType::f32, out_data);

    UnaryAttributes attrs;
    attrs.type = UnaryType::Tanh;
    unary(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], std::tanh(0.0f), 1e-5f);
    NNOPS_EXPECT_NEAR(out_data[1], std::tanh(1.0f), 1e-5f);
    NNOPS_EXPECT_NEAR(out_data[2], std::tanh(-1.0f), 1e-5f);
}

NNOPS_TEST(unary_abs) {
    const int64_t shape[] = {4};
    float in_data[]  = {-3.0f, 0.0f, 5.0f, -2.5f};
    float out_data[4] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(shape, DataType::f32, out_data);

    UnaryAttributes attrs;
    attrs.type = UnaryType::Abs;
    unary(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 5.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[3], 2.5f, 1e-6f);
}

NNOPS_TEST(unary_neg) {
    const int64_t shape[] = {3};
    float in_data[]  = {1.0f, -2.0f, 0.0f};
    float out_data[3] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(shape, DataType::f32, out_data);

    UnaryAttributes attrs;
    attrs.type = UnaryType::Neg;
    unary(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], -1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 0.0f, 1e-6f);
}

NNOPS_TEST(unary_sqrt) {
    const int64_t shape[] = {3};
    float in_data[]  = {0.0f, 4.0f, 9.0f};
    float out_data[3] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(shape, DataType::f32, out_data);

    UnaryAttributes attrs;
    attrs.type = UnaryType::Sqrt;
    unary(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 3.0f, 1e-6f);
}

NNOPS_TEST(unary_add_to) {
    const int64_t shape[] = {3};
    float in_data[]  = {1.0f, 2.0f, 3.0f};
    float out_data[] = {10.0f, 20.0f, 30.0f};  // initial

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(shape, DataType::f32, out_data);

    UnaryAttributes attrs;
    attrs.type = UnaryType::Exp;
    attrs.add_to = true;
    unary(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 10.0f + std::exp(1.0f), 1e-5f);
    NNOPS_EXPECT_NEAR(out_data[1], 20.0f + std::exp(2.0f), 1e-5f);
    NNOPS_EXPECT_NEAR(out_data[2], 30.0f + std::exp(3.0f), 1e-5f);
}

// ============================================================
// Random data tests
// ============================================================

NNOPS_TEST(unary_random_abs) {
    auto [in_vec, input] = test::make_random_tensor({1000}, -10.0f, 10.0f, 111);
    std::vector<float> out_buf(1000);
    TensorView output(input.shape_span(), DataType::f32, out_buf.data());

    UnaryAttributes attrs;
    attrs.type = UnaryType::Abs;
    unary(input, output, attrs);

    for (int i = 0; i < 1000; ++i) {
        NNOPS_EXPECT_TRUE(out_buf[i] >= 0.0f);
        NNOPS_EXPECT_NEAR(out_buf[i], std::abs(in_vec[i]), 1e-5f);
    }
}

NNOPS_TEST(unary_random_exp) {
    auto [in_vec, input] = test::make_random_tensor({500}, -2.0f, 2.0f, 222);
    std::vector<float> out_buf(500);
    TensorView output(input.shape_span(), DataType::f32, out_buf.data());

    UnaryAttributes attrs;
    attrs.type = UnaryType::Exp;
    unary(input, output, attrs);

    for (int i = 0; i < 500; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], std::exp(in_vec[i]), 1e-4f);
    }
}

// ============================================================
// 2D tests
// ============================================================

NNOPS_TEST(unary_2d_sqrt) {
    const int64_t shape[] = {2, 3};
    float in_data[]  = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
    float out_data[6] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(shape, DataType::f32, out_data);

    UnaryAttributes attrs;
    attrs.type = UnaryType::Sqrt;
    unary(input, output, attrs);

    for (int i = 0; i < 6; ++i) {
        NNOPS_EXPECT_NEAR(out_data[i], std::sqrt(in_data[i]), 1e-5f);
    }
}

// ============================================================
// Class API parity
// ============================================================

NNOPS_TEST(unary_class_api) {
    const int64_t shape[] = {4};
    float in_data[]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float out1[4] = {}, out2[4] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView out1_view(shape, DataType::f32, out1);
    TensorView out2_view(shape, DataType::f32, out2);

    UnaryAttributes attrs;
    attrs.type = UnaryType::Log;

    // Functional
    unary(input, out1_view, attrs);
    // Class
    auto op = Unary::create(attrs, Backend::CPU);
    const TensorView ins[] = {input};
    op->compute(out2_view, ins);

    NNOPS_EXPECT_TRUE(test::allclose(out1_view, out2_view));
}
