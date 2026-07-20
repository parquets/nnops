/// Unit tests for Activation operator (CPU reference).

#include "nnops/ops/activation.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

NNOPS_TEST(activation_relu) {
    const int64_t shape[] = {8};
    float in_data[]  = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 10.0f, -10.0f};
    float out_data[8] = {};
    float expected[]  = { 0.0f,  0.0f, 0.0f, 1.0f, 2.0f,  0.0f, 10.0f,  0.0f};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(shape, DataType::f32, out_data);

    ActivationAttributes attrs;
    attrs.type = ActivationType::Relu;

    activation(input, output, attrs);

    for (int i = 0; i < 8; ++i) {
        NNOPS_EXPECT_NEAR(out_data[i], expected[i], 1e-6f);
    }
}

NNOPS_TEST(activation_leaky_relu) {
    const int64_t shape[] = {4};
    float in_data[]  = {-2.0f, 0.0f, 1.0f, -1.0f};
    float out_data[4] = {};
    const float alpha = 0.01f;

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(shape, DataType::f32, out_data);

    ActivationAttributes attrs;
    attrs.type = ActivationType::LeakyRelu;
    attrs.alpha = alpha;

    activation(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], alpha * (-2.0f), 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[3], alpha * (-1.0f), 1e-6f);
}

NNOPS_TEST(activation_sigmoid) {
    const int64_t shape[] = {3};
    float in_data[]  = {0.0f, 10.0f, -10.0f};
    float out_data[3] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(shape, DataType::f32, out_data);

    ActivationAttributes attrs;
    attrs.type = ActivationType::Sigmoid;

    activation(input, output, attrs);

    // sigmoid(0) = 0.5
    NNOPS_EXPECT_NEAR(out_data[0], 0.5f, 1e-6f);
    // sigmoid(10) ≈ 0.99995
    NNOPS_EXPECT_NEAR(out_data[1], 0.99995f, 1e-4f);
    // sigmoid(-10) ≈ 0.000045
    NNOPS_EXPECT_NEAR(out_data[2], 0.000045f, 1e-4f);
}

NNOPS_TEST(activation_tanh) {
    const int64_t shape[] = {3};
    float in_data[]  = {0.0f, 2.0f, -2.0f};
    float out_data[3] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(shape, DataType::f32, out_data);

    ActivationAttributes attrs;
    attrs.type = ActivationType::Tanh;

    activation(input, output, attrs);

    // tanh(0) = 0
    NNOPS_EXPECT_NEAR(out_data[0], 0.0f, 1e-6f);
    // tanh(2) ≈ 0.9640
    NNOPS_EXPECT_NEAR(out_data[1], 0.9640f, 1e-3f);
    // tanh(-2) ≈ -0.9640
    NNOPS_EXPECT_NEAR(out_data[2], -0.9640f, 1e-3f);
}

NNOPS_TEST(activation_gelu) {
    const int64_t shape[] = {2};
    float in_data[]  = {0.0f, 1.0f};
    float out_data[2] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(shape, DataType::f32, out_data);

    ActivationAttributes attrs;
    attrs.type = ActivationType::Gelu;

    activation(input, output, attrs);

    // GELU(0) ≈ 0.0
    NNOPS_EXPECT_NEAR(out_data[0], 0.0f, 1e-3f);
    // GELU(1) ≈ 0.8413
    NNOPS_EXPECT_NEAR(out_data[1], 0.8413f, 1e-3f);
}

NNOPS_TEST(activation_silu) {
    const int64_t shape[] = {2};
    float in_data[]  = {0.0f, 1.0f};
    float out_data[2] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(shape, DataType::f32, out_data);

    ActivationAttributes attrs;
    attrs.type = ActivationType::Silu;

    activation(input, output, attrs);

    // SiLU(0) = 0 * sigmoid(0) = 0
    NNOPS_EXPECT_NEAR(out_data[0], 0.0f, 1e-6f);
    // SiLU(1) = 1 * sigmoid(1) ≈ 0.7311
    NNOPS_EXPECT_NEAR(out_data[1], 0.7311f, 1e-3f);
}

NNOPS_TEST(activation_random_relu) {
    auto [in_vec, input] = test::make_random_tensor({1000}, -5.0f, 5.0f);
    std::vector<float> out_buf(1000);
    TensorView output(input.shape_span(), DataType::f32, out_buf.data());

    ActivationAttributes attrs;
    attrs.type = ActivationType::Relu;

    activation(input, output, attrs);

    for (int i = 0; i < 1000; ++i) {
        NNOPS_EXPECT_TRUE(out_buf[i] >= 0.0f);
    }
}

NNOPS_TEST(activation_class_api) {
    const int64_t shape[] = {4};
    float in_data[]  = {-2.0f, 0.0f, 1.0f, -1.0f};
    float out1_data[4] = {};
    float out2_data[4] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView out1(shape, DataType::f32, out1_data);
    TensorView out2(shape, DataType::f32, out2_data);

    ActivationAttributes attrs;
    attrs.type = ActivationType::Relu;

    // Functional
    activation(input, out1, attrs);
    // Class
    auto op = Activation::create(attrs, Backend::CPU);
    const TensorView ins[] = {input};
    op->compute(out2, ins);

    NNOPS_EXPECT_TRUE(test::allclose(out1, out2));
}
