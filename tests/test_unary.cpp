/// @file test_unary.cpp
/// @brief Unit tests for Unary operator (CPU backend).

#include "nnops/ops/unary.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

// ============================================================
// Hand-verified tests
// ============================================================

NNOPS_TEST(unary_exp) {
    const int64_t shape[] = {3};
    float in_data[]  = {0.0f, 1.0f, 2.0f};
    float out_buf[3] = {};

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    UnaryAttributes attrs;
    attrs.type = UnaryType::Exp;
    auto op = Unary::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    TensorView output = test::make_planar(descs[0], out_buf);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], std::exp(0.0f), 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], std::exp(1.0f), 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[2], std::exp(2.0f), 1e-5f);
}

NNOPS_TEST(unary_log) {
    const int64_t shape[] = {3};
    float in_data[]  = {1.0f, 2.0f, 10.0f};
    float out_buf[3] = {};

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    UnaryAttributes attrs;
    attrs.type = UnaryType::Log;
    auto op = Unary::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    TensorView output = test::make_planar(descs[0], out_buf);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], std::log(1.0f), 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], std::log(2.0f), 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[2], std::log(10.0f), 1e-5f);
}

NNOPS_TEST(unary_sin) {
    const int64_t shape[] = {3};
    float in_data[]  = {0.0f, 1.5707963f, 3.1415927f};
    float out_buf[3] = {};

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    UnaryAttributes attrs;
    attrs.type = UnaryType::Sin;
    auto op = Unary::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    TensorView output = test::make_planar(descs[0], out_buf);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], 0.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], 1.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[2], 0.0f, 1e-5f);
}

NNOPS_TEST(unary_cos) {
    const int64_t shape[] = {3};
    float in_data[]  = {0.0f, 1.5707963f, 3.1415927f};
    float out_buf[3] = {};

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    UnaryAttributes attrs;
    attrs.type = UnaryType::Cos;
    auto op = Unary::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    TensorView output = test::make_planar(descs[0], out_buf);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], 1.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], 0.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[2], -1.0f, 1e-4f);
}

NNOPS_TEST(unary_tanh) {
    const int64_t shape[] = {3};
    float in_data[]  = {0.0f, 1.0f, -1.0f};
    float out_buf[3] = {};

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    UnaryAttributes attrs;
    attrs.type = UnaryType::Tanh;
    auto op = Unary::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    TensorView output = test::make_planar(descs[0], out_buf);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], std::tanh(0.0f), 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], std::tanh(1.0f), 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[2], std::tanh(-1.0f), 1e-5f);
}

NNOPS_TEST(unary_abs) {
    const int64_t shape[] = {4};
    float in_data[]  = {-3.0f, 0.0f, 5.0f, -2.5f};
    float out_buf[4] = {};

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    UnaryAttributes attrs;
    attrs.type = UnaryType::Abs;
    auto op = Unary::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 4);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    TensorView output = test::make_planar(descs[0], out_buf);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[1], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[2], 5.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[3], 2.5f, 1e-6f);
}

NNOPS_TEST(unary_neg) {
    const int64_t shape[] = {3};
    float in_data[]  = {1.0f, -2.0f, 0.0f};
    float out_buf[3] = {};

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    UnaryAttributes attrs;
    attrs.type = UnaryType::Neg;
    auto op = Unary::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    TensorView output = test::make_planar(descs[0], out_buf);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], -1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[1], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[2], 0.0f, 1e-6f);
}

NNOPS_TEST(unary_sqrt) {
    const int64_t shape[] = {3};
    float in_data[]  = {0.0f, 4.0f, 9.0f};
    float out_buf[3] = {};

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    UnaryAttributes attrs;
    attrs.type = UnaryType::Sqrt;
    auto op = Unary::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    TensorView output = test::make_planar(descs[0], out_buf);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[1], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[2], 3.0f, 1e-6f);
}

NNOPS_TEST(unary_add_to) {
    const int64_t shape[] = {3};
    float in_data[]  = {1.0f, 2.0f, 3.0f};
    float out_buf[] = {10.0f, 20.0f, 30.0f};  // initial

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    UnaryAttributes attrs;
    attrs.type = UnaryType::Exp;
    attrs.add_to = true;
    auto op = Unary::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    TensorView output = test::make_planar(descs[0], out_buf);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], 10.0f + std::exp(1.0f), 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], 20.0f + std::exp(2.0f), 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[2], 30.0f + std::exp(3.0f), 1e-5f);
}

// ============================================================
// Random data tests
// ============================================================

NNOPS_TEST(unary_random_abs) {
    auto [in_vec, input] = test::make_random_tensor({1000}, -10.0f, 10.0f, 111);
    auto d = input.desc();

    UnaryAttributes attrs;
    attrs.type = UnaryType::Abs;
    auto op = Unary::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 1000);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(1000);
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    for (int i = 0; i < 1000; ++i) {
        NNOPS_EXPECT_TRUE(out_buf[i] >= 0.0f);
        NNOPS_EXPECT_NEAR(out_buf[i], std::abs(in_vec[i]), 1e-5f);
    }
}

NNOPS_TEST(unary_random_exp) {
    auto [in_vec, input] = test::make_random_tensor({500}, -2.0f, 2.0f, 222);
    auto d = input.desc();

    UnaryAttributes attrs;
    attrs.type = UnaryType::Exp;
    auto op = Unary::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 500);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(500);
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

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
    float out_buf[6] = {};

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    UnaryAttributes attrs;
    attrs.type = UnaryType::Sqrt;
    auto op = Unary::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 3);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    TensorView output = test::make_planar(descs[0], out_buf);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    for (int i = 0; i < 6; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], std::sqrt(in_data[i]), 1e-5f);
    }
}

// ============================================================
// f16 tests — exercise the SIMD f16 code path
// ============================================================

NNOPS_TEST(unary_random_exp_f16) {
    auto [f32_vec, _] = test::make_random_tensor({400}, -2.0f, 2.0f, 900);

    auto f16_vec = test::f32_to_f16(f32_vec);

    const int64_t shape[] = {400};
    TensorView input(shape, DataType::f16, f16_vec.data());

    UnaryAttributes attrs;
    attrs.type = UnaryType::Exp;
    auto op = Unary::create(attrs, Backend::CPU);

    auto d = input.desc();
    const TensorDesc in_arr[] = {d};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

    std::vector<nnops::backend::cpu::half> out_buf(400);
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    for (int i = 0; i < 400; ++i) {
        float result = simd::s_load(&out_buf[i]);
        NNOPS_EXPECT_TRUE(std::isfinite(result));
        NNOPS_EXPECT_NEAR(result, std::exp(f32_vec[i]), 1e-2f);
    }
}

NNOPS_TEST(unary_random_abs_f16) {
    auto [f32_vec, _] = test::make_random_tensor({400}, -10.0f, 10.0f, 901);

    auto f16_vec = test::f32_to_f16(f32_vec);

    const int64_t shape[] = {400};
    TensorView input(shape, DataType::f16, f16_vec.data());

    UnaryAttributes attrs;
    attrs.type = UnaryType::Abs;
    auto op = Unary::create(attrs, Backend::CPU);

    auto d = input.desc();
    const TensorDesc in_arr[] = {d};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

    std::vector<nnops::backend::cpu::half> out_buf(400);
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    for (int i = 0; i < 400; ++i) {
        float result = simd::s_load(&out_buf[i]);
        NNOPS_EXPECT_TRUE(result >= 0.0f);
        NNOPS_EXPECT_NEAR(result, std::abs(f32_vec[i]), 1e-2f);
    }
}
