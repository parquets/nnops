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

// ============================================================
// Quantized input/output tests (fused dequant → unary → quant)
// ============================================================

NNOPS_TEST(unary_quant_per_tensor_s8_to_s8) {
    // s8 input → s8 output (fused), per-tensor scale=0.1/zp=0, Abs.
    const int64_t shape[] = {4};
    int8_t in_data[] = {-10, -5, 0, 5};

    QuantParams in_qp;
    in_qp.scale = 0.1f;
    in_qp.granularity = QuantGranularity::PerTensor;
    TensorView input(shape, DataType::s8, in_data, TensorLayout::NCHW, in_qp);

    UnaryAttributes attrs;
    attrs.type = UnaryType::Abs;
    auto op = Unary::create(attrs, Backend::CPU);

    int8_t out_buf[4] = {};
    QuantParams out_qp;
    out_qp.scale = 0.1f;
    out_qp.granularity = QuantGranularity::PerTensor;
    TensorView output(shape, DataType::s8, out_buf, TensorLayout::NCHW, out_qp);

    const TensorView ins[] = {input};
    op->compute(output, ins);

    // dequant: [-1, -0.5, 0, 0.5] → abs → [1, 0.5, 0, 0.5] → /0.1 → [10, 5, 0, 5]
    NNOPS_EXPECT_EQ(out_buf[0], 10);
    NNOPS_EXPECT_EQ(out_buf[1], 5);
    NNOPS_EXPECT_EQ(out_buf[2], 0);
    NNOPS_EXPECT_EQ(out_buf[3], 5);
}

NNOPS_TEST(unary_quant_per_token_s8_to_s8) {
    // Per-token s8 → s8 (fused), Neg, output uses the same per-token scales.
    const int64_t shape[] = {2, 4};
    int8_t in_data[] = {1, -2, 3, -4,  10, -20, 30, -40};

    float scale_data[] = {0.25f, 0.1f};
    QuantParams in_qp;
    in_qp.granularity = QuantGranularity::PerToken;
    in_qp.scale_data = scale_data;
    in_qp.num_scales = 2;
    TensorView input(shape, DataType::s8, in_data, TensorLayout::NCHW, in_qp);

    UnaryAttributes attrs;
    attrs.type = UnaryType::Neg;
    auto op = Unary::create(attrs, Backend::CPU);

    int8_t out_buf[8] = {};
    QuantParams out_qp;
    out_qp.granularity = QuantGranularity::PerToken;
    out_qp.scale_data = scale_data;   // reuse same scales
    out_qp.num_scales = 2;
    TensorView output(shape, DataType::s8, out_buf, TensorLayout::NCHW, out_qp);

    const TensorView ins[] = {input};
    op->compute(output, ins);

    // Row0: neg([0.25,-0.5,0.75,-1])/0.25 = [-1,2,-3,4]
    // Row1: neg([1,-2,3,-4])/0.1 = [-10,20,-30,40]
    const int8_t expected[] = {-1, 2, -3, 4, -10, 20, -30, 40};
    for (int i = 0; i < 8; ++i) {
        NNOPS_EXPECT_EQ(out_buf[i], expected[i]);
    }
}

NNOPS_TEST(unary_quant_per_tensor_u8_to_u8) {
    // Asymmetric u8 → u8 (fused), Abs. scale=0.5, zp=128.
    const int64_t shape[] = {3};
    uint8_t in_data[] = {128, 0, 255};

    QuantParams in_qp;
    in_qp.scale = 0.5f;
    in_qp.zero_point = 128;
    in_qp.granularity = QuantGranularity::PerTensor;
    TensorView input(shape, DataType::u8, in_data, TensorLayout::NCHW, in_qp);

    UnaryAttributes attrs;
    attrs.type = UnaryType::Abs;
    auto op = Unary::create(attrs, Backend::CPU);

    uint8_t out_buf[3] = {};
    QuantParams out_qp;
    out_qp.scale = 0.5f;
    out_qp.zero_point = 128;
    out_qp.granularity = QuantGranularity::PerTensor;
    TensorView output(shape, DataType::u8, out_buf, TensorLayout::NCHW, out_qp);

    const TensorView ins[] = {input};
    op->compute(output, ins);

    // dequant: (128-128)*0.5=0, (0-128)*0.5=-64, (255-128)*0.5=63.5
    // abs: [0, 64, 63.5] → /0.5 + 128 → [128, 256, 255] → clamp → [128, 255, 255]
    NNOPS_EXPECT_EQ(out_buf[0], 128);
    NNOPS_EXPECT_EQ(out_buf[1], 255);
    NNOPS_EXPECT_EQ(out_buf[2], 255);
}

NNOPS_TEST(unary_quant_per_tensor_s8_exp) {
    // Transcendental op (Exp) through the fused path: s8 → s8, scale=1/zp=0.
    const int64_t shape[] = {3};
    int8_t in_data[] = {0, 1, 2};

    QuantParams qp;
    qp.scale = 1.0f;
    qp.granularity = QuantGranularity::PerTensor;
    TensorView input(shape, DataType::s8, in_data, TensorLayout::NCHW, qp);

    UnaryAttributes attrs;
    attrs.type = UnaryType::Exp;
    auto op = Unary::create(attrs, Backend::CPU);

    int8_t out_buf[3] = {};
    TensorView output(shape, DataType::s8, out_buf, TensorLayout::NCHW, qp);

    const TensorView ins[] = {input};
    op->compute(output, ins);

    // exp([0,1,2]) ≈ [1, 2.718, 7.389] → round → [1, 3, 7]
    NNOPS_EXPECT_EQ(out_buf[0], 1);
    NNOPS_EXPECT_EQ(out_buf[1], 3);
    NNOPS_EXPECT_EQ(out_buf[2], 7);
}

