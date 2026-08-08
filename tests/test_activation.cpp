/// Unit tests for Activation operator (CPU reference).

#include "nnops/ops/activation.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

NNOPS_TEST(activation_relu) {
    const int64_t shape[] = {8};
    float in_data[]  = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 10.0f, -10.0f};
    float expected[]  = { 0.0f,  0.0f, 0.0f, 1.0f, 2.0f,  0.0f, 10.0f,  0.0f};

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    ActivationAttributes attrs;
    attrs.type = ActivationType::Relu;
    auto op = Activation::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_TRUE(descs[0].rank == 1);
    NNOPS_EXPECT_TRUE(descs[0].dims[0] == 8);
    NNOPS_EXPECT_TRUE(descs[0].layout == TensorLayout::NCHW);
    NNOPS_EXPECT_TRUE(descs[0].dtype == DataType::f32);

    std::vector<float> buf(static_cast<size_t>(descs[0].numel()));
    auto out = nnops::test::make_planar(descs[0], buf.data());

    const TensorView ins[] = {input};
    op->compute(out, ins);

    for (int i = 0; i < 8; ++i) {
        NNOPS_EXPECT_NEAR(buf[i], expected[i], 1e-6f);
    }
}

NNOPS_TEST(activation_leaky_relu) {
    const int64_t shape[] = {4};
    float in_data[]  = {-2.0f, 0.0f, 1.0f, -1.0f};
    const float alpha = 0.01f;

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    ActivationAttributes attrs;
    attrs.type = ActivationType::LeakyRelu;
    attrs.alpha = alpha;
    auto op = Activation::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_TRUE(descs[0].rank == 1);
    NNOPS_EXPECT_TRUE(descs[0].dims[0] == 4);
    NNOPS_EXPECT_TRUE(descs[0].layout == TensorLayout::NCHW);
    NNOPS_EXPECT_TRUE(descs[0].dtype == DataType::f32);

    std::vector<float> buf(static_cast<size_t>(descs[0].numel()));
    auto out = nnops::test::make_planar(descs[0], buf.data());

    const TensorView ins[] = {input};
    op->compute(out, ins);

    NNOPS_EXPECT_NEAR(buf[0], alpha * (-2.0f), 1e-6f);
    NNOPS_EXPECT_NEAR(buf[1], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(buf[2], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(buf[3], alpha * (-1.0f), 1e-6f);
}

NNOPS_TEST(activation_sigmoid) {
    const int64_t shape[] = {3};
    float in_data[]  = {0.0f, 10.0f, -10.0f};

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    ActivationAttributes attrs;
    attrs.type = ActivationType::Sigmoid;
    auto op = Activation::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_TRUE(descs[0].rank == 1);
    NNOPS_EXPECT_TRUE(descs[0].dims[0] == 3);
    NNOPS_EXPECT_TRUE(descs[0].layout == TensorLayout::NCHW);
    NNOPS_EXPECT_TRUE(descs[0].dtype == DataType::f32);

    std::vector<float> buf(static_cast<size_t>(descs[0].numel()));
    auto out = nnops::test::make_planar(descs[0], buf.data());

    const TensorView ins[] = {input};
    op->compute(out, ins);

    // sigmoid(0) = 0.5
    NNOPS_EXPECT_NEAR(buf[0], 0.5f, 1e-6f);
    // sigmoid(10) ≈ 0.99995
    NNOPS_EXPECT_NEAR(buf[1], 0.99995f, 1e-4f);
    // sigmoid(-10) ≈ 0.000045
    NNOPS_EXPECT_NEAR(buf[2], 0.000045f, 1e-4f);
}

NNOPS_TEST(activation_tanh) {
    const int64_t shape[] = {3};
    float in_data[]  = {0.0f, 2.0f, -2.0f};

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    ActivationAttributes attrs;
    attrs.type = ActivationType::Tanh;
    auto op = Activation::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_TRUE(descs[0].rank == 1);
    NNOPS_EXPECT_TRUE(descs[0].dims[0] == 3);
    NNOPS_EXPECT_TRUE(descs[0].layout == TensorLayout::NCHW);
    NNOPS_EXPECT_TRUE(descs[0].dtype == DataType::f32);

    std::vector<float> buf(static_cast<size_t>(descs[0].numel()));
    auto out = nnops::test::make_planar(descs[0], buf.data());

    const TensorView ins[] = {input};
    op->compute(out, ins);

    // tanh(0) = 0
    NNOPS_EXPECT_NEAR(buf[0], 0.0f, 1e-6f);
    // tanh(2) ≈ 0.9640
    NNOPS_EXPECT_NEAR(buf[1], 0.9640f, 1e-3f);
    // tanh(-2) ≈ -0.9640
    NNOPS_EXPECT_NEAR(buf[2], -0.9640f, 1e-3f);
}

NNOPS_TEST(activation_gelu) {
    const int64_t shape[] = {2};
    float in_data[]  = {0.0f, 1.0f};

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    ActivationAttributes attrs;
    attrs.type = ActivationType::Gelu;
    auto op = Activation::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_TRUE(descs[0].rank == 1);
    NNOPS_EXPECT_TRUE(descs[0].dims[0] == 2);
    NNOPS_EXPECT_TRUE(descs[0].layout == TensorLayout::NCHW);
    NNOPS_EXPECT_TRUE(descs[0].dtype == DataType::f32);

    std::vector<float> buf(static_cast<size_t>(descs[0].numel()));
    auto out = nnops::test::make_planar(descs[0], buf.data());

    const TensorView ins[] = {input};
    op->compute(out, ins);

    // GELU(0) ≈ 0.0
    NNOPS_EXPECT_NEAR(buf[0], 0.0f, 1e-3f);
    // GELU(1) ≈ 0.8413
    NNOPS_EXPECT_NEAR(buf[1], 0.8413f, 1e-3f);
}

NNOPS_TEST(activation_silu) {
    const int64_t shape[] = {2};
    float in_data[]  = {0.0f, 1.0f};

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    ActivationAttributes attrs;
    attrs.type = ActivationType::Silu;
    auto op = Activation::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_TRUE(descs[0].rank == 1);
    NNOPS_EXPECT_TRUE(descs[0].dims[0] == 2);
    NNOPS_EXPECT_TRUE(descs[0].layout == TensorLayout::NCHW);
    NNOPS_EXPECT_TRUE(descs[0].dtype == DataType::f32);

    std::vector<float> buf(static_cast<size_t>(descs[0].numel()));
    auto out = nnops::test::make_planar(descs[0], buf.data());

    const TensorView ins[] = {input};
    op->compute(out, ins);

    // SiLU(0) = 0 * sigmoid(0) = 0
    NNOPS_EXPECT_NEAR(buf[0], 0.0f, 1e-6f);
    // SiLU(1) = 1 * sigmoid(1) ≈ 0.7311
    NNOPS_EXPECT_NEAR(buf[1], 0.7311f, 1e-3f);
}

NNOPS_TEST(activation_random_relu) {
    auto [in_vec, input] = test::make_random_tensor({1000}, -5.0f, 5.0f);
    auto d = input.desc();

    ActivationAttributes attrs;
    attrs.type = ActivationType::Relu;
    auto op = Activation::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_TRUE(descs[0].rank == 1);
    NNOPS_EXPECT_TRUE(descs[0].dims[0] == 1000);
    NNOPS_EXPECT_TRUE(descs[0].layout == TensorLayout::NCHW);
    NNOPS_EXPECT_TRUE(descs[0].dtype == DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    auto out = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(out, ins);

    for (int i = 0; i < 1000; ++i) {
        NNOPS_EXPECT_TRUE(out_buf[i] >= 0.0f);
    }
}

// ============================================================
// f16 tests — exercise the SIMD f16 code path
// ============================================================

NNOPS_TEST(activation_random_relu_f16) {
    auto [f32_vec, _] = test::make_random_tensor({500}, -5.0f, 5.0f, 800);

    auto f16_vec = test::f32_to_f16(f32_vec);

    const int64_t shape[] = {500};
    TensorView input(shape, DataType::f16, f16_vec.data());

    ActivationAttributes attrs;
    attrs.type = ActivationType::Relu;
    auto op = Activation::create(attrs, Backend::CPU);

    auto d = input.desc();
    const TensorDesc arr[] = {d};
    auto descs = op->getOutputTensorDesc(arr);

    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

    std::vector<nnops::backend::cpu::half> out_buf(500);
    auto out = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(out, ins);

    for (int i = 0; i < 500; ++i) {
        float result = simd::s_load(&out_buf[i]);
        NNOPS_EXPECT_TRUE(result >= -1e-5f);  // ReLU: >= 0
        float expected = f32_vec[i] > 0.0f ? f32_vec[i] : 0.0f;
        NNOPS_EXPECT_NEAR(result, expected, 1e-2f);
    }
}
