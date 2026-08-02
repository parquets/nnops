/// Unit tests for QuantizeLinear / DequantizeLinear operators.

#include "nnops/ops/quant_linear.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"

#include <vector>
#include <cmath>
#include <cstdint>
#include <random>

using namespace nnops;

// ============================================================
// QuantizeLinear tests
// ============================================================

NNOPS_TEST(quantize_linear_i8_basic) {
    const int64_t shape[] = {3};
    float in_data[] = {0.0f, 1.0f, 2.0f};
    TensorView input(shape, DataType::f32, in_data);

    QuantLinearAttributes attrs;
    attrs.output_dtype = DataType::i8;
    auto op = QuantizeLinear::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::i8);
    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);

    std::vector<int8_t> out_buf(3);
    TensorView output(shape, DataType::i8, out_buf.data());

    float scale_data[] = {1.0f};
    int8_t zp_data[] = {0};
    const int64_t s_shape[] = {1};
    TensorView scale(s_shape, DataType::f32, scale_data);
    TensorView zp(s_shape, DataType::i8, zp_data);

    const TensorView ins[] = {input, scale, zp};
    op->compute(output, ins);

    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[0]), 0);
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[1]), 1);
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[2]), 2);
}

NNOPS_TEST(quantize_linear_i8_scale) {
    const int64_t shape[] = {3};
    float in_data[] = {0.0f, 2.0f, 4.0f};
    TensorView input(shape, DataType::f32, in_data);

    QuantLinearAttributes attrs;
    attrs.output_dtype = DataType::i8;
    auto op = QuantizeLinear::create(attrs, Backend::CPU);

    std::vector<int8_t> out_buf(3);
    TensorView output(shape, DataType::i8, out_buf.data());

    float scale_data[] = {2.0f};
    int8_t zp_data[] = {0};
    const int64_t s_shape[] = {1};
    TensorView scale(s_shape, DataType::f32, scale_data);
    TensorView zp(s_shape, DataType::i8, zp_data);

    const TensorView ins[] = {input, scale, zp};
    op->compute(output, ins);

    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[0]), 0);
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[1]), 1);
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[2]), 2);
}

NNOPS_TEST(quantize_linear_i8_zero_point) {
    const int64_t shape[] = {3};
    float in_data[] = {0.0f, 1.0f, 2.0f};
    TensorView input(shape, DataType::f32, in_data);

    QuantLinearAttributes attrs;
    attrs.output_dtype = DataType::i8;
    auto op = QuantizeLinear::create(attrs, Backend::CPU);

    std::vector<int8_t> out_buf(3);
    TensorView output(shape, DataType::i8, out_buf.data());

    float scale_data[] = {1.0f};
    int8_t zp_data[] = {100};
    const int64_t s_shape[] = {1};
    TensorView scale(s_shape, DataType::f32, scale_data);
    TensorView zp(s_shape, DataType::i8, zp_data);

    const TensorView ins[] = {input, scale, zp};
    op->compute(output, ins);

    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[0]), 100);
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[1]), 101);
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[2]), 102);
}

NNOPS_TEST(quantize_linear_i8_clamp) {
    const int64_t shape[] = {3};
    float in_data[] = {0.0f, 200.0f, -200.0f};
    TensorView input(shape, DataType::f32, in_data);

    QuantLinearAttributes attrs;
    attrs.output_dtype = DataType::i8;
    auto op = QuantizeLinear::create(attrs, Backend::CPU);

    std::vector<int8_t> out_buf(3);
    TensorView output(shape, DataType::i8, out_buf.data());

    float scale_data[] = {1.0f};
    int8_t zp_data[] = {0};
    const int64_t s_shape[] = {1};
    TensorView scale(s_shape, DataType::f32, scale_data);
    TensorView zp(s_shape, DataType::i8, zp_data);

    const TensorView ins[] = {input, scale, zp};
    op->compute(output, ins);

    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[0]), 0);
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[1]), 127);
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[2]), -128);
}

NNOPS_TEST(quantize_linear_u8_basic) {
    const int64_t shape[] = {4};
    float in_data[] = {-1.0f, 0.0f, 1.0f, 300.0f};
    TensorView input(shape, DataType::f32, in_data);

    QuantLinearAttributes attrs;
    attrs.output_dtype = DataType::u8;
    auto op = QuantizeLinear::create(attrs, Backend::CPU);

    std::vector<uint8_t> out_buf(4);
    TensorView output(shape, DataType::u8, out_buf.data());

    float scale_data[] = {1.0f};
    uint8_t zp_data[] = {0};
    const int64_t s_shape[] = {1};
    TensorView scale(s_shape, DataType::f32, scale_data);
    TensorView zp(s_shape, DataType::u8, zp_data);

    const TensorView ins[] = {input, scale, zp};
    op->compute(output, ins);

    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[0]), 0);
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[1]), 0);
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[2]), 1);
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[3]), 255);
}

NNOPS_TEST(quantize_linear_u8_zp) {
    const int64_t shape[] = {3};
    float in_data[] = {0.0f, 1.0f, -1.0f};
    TensorView input(shape, DataType::f32, in_data);

    QuantLinearAttributes attrs;
    attrs.output_dtype = DataType::u8;
    auto op = QuantizeLinear::create(attrs, Backend::CPU);

    std::vector<uint8_t> out_buf(3);
    TensorView output(shape, DataType::u8, out_buf.data());

    float scale_data[] = {1.0f};
    uint8_t zp_data[] = {128};
    const int64_t s_shape[] = {1};
    TensorView scale(s_shape, DataType::f32, scale_data);
    TensorView zp(s_shape, DataType::u8, zp_data);

    const TensorView ins[] = {input, scale, zp};
    op->compute(output, ins);

    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[0]), 128);
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[1]), 129);
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[2]), 127);
}

NNOPS_TEST(quantize_linear_2d_per_channel) {
    const int64_t shape[] = {2, 3};
    float in_data[] = {0.0f, 1.0f, 2.0f, 10.0f, 11.0f, 12.0f};
    TensorView input(shape, DataType::f32, in_data);

    QuantLinearAttributes attrs;
    attrs.axis = 1;
    attrs.output_dtype = DataType::i8;
    auto op = QuantizeLinear::create(attrs, Backend::CPU);

    std::vector<int8_t> out_buf(6);
    TensorView output(shape, DataType::i8, out_buf.data());

    float scale_data[] = {1.0f, 2.0f, 3.0f};
    int8_t zp_data[] = {0, 0, 0};
    const int64_t s_shape[] = {3};
    TensorView scale(s_shape, DataType::f32, scale_data);
    TensorView zp(s_shape, DataType::i8, zp_data);

    const TensorView ins[] = {input, scale, zp};
    op->compute(output, ins);

    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[0]), 0);
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[3]), 10);
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[1]), 1);  // round(0.5)=1
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[4]), 6);
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[2]), 1);
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[5]), 4);
}

NNOPS_TEST(quantize_linear_random) {
    auto [in_vec, input] = test::make_random_tensor({4, 8}, -10.0f, 10.0f);

    QuantLinearAttributes attrs;
    attrs.output_dtype = DataType::i8;
    auto op = QuantizeLinear::create(attrs, Backend::CPU);

    std::vector<int8_t> out_buf(32);
    const int64_t shape[] = {4, 8};
    TensorView output(shape, DataType::i8, out_buf.data());

    float scale_data[] = {0.5f};
    int8_t zp_data[] = {0};
    const int64_t s_shape[] = {1};
    TensorView scale(s_shape, DataType::f32, scale_data);
    TensorView zp(s_shape, DataType::i8, zp_data);

    const TensorView ins[] = {input, scale, zp};
    op->compute(output, ins);

    for (int i = 0; i < 32; ++i) {
        int32_t expected = static_cast<int32_t>(std::lround(in_vec[i] / 0.5f));
        expected = std::max(-128, std::min(127, expected));
        NNOPS_EXPECT_EQ(static_cast<int32_t>(out_buf[i]), expected);
    }
}

// ============================================================
// DequantizeLinear tests
// ============================================================

NNOPS_TEST(dequantize_linear_i8_basic) {
    const int64_t shape[] = {3};
    int8_t in_data[] = {0, 1, -1};
    TensorView input(shape, DataType::i8, in_data);

    QuantLinearAttributes attrs;
    attrs.output_dtype = DataType::f32;
    auto op = DequantizeLinear::create(attrs, Backend::CPU);

    std::vector<float> out_buf(3);
    TensorView output(shape, DataType::f32, out_buf.data());

    float scale_data[] = {1.0f};
    int8_t zp_data[] = {0};
    const int64_t s_shape[] = {1};
    TensorView scale(s_shape, DataType::f32, scale_data);
    TensorView zp(s_shape, DataType::i8, zp_data);

    const TensorView ins[] = {input, scale, zp};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], 0.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], 1.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[2], -1.0f, 1e-5f);
}

NNOPS_TEST(dequantize_linear_i8_scale_zp) {
    const int64_t shape[] = {3};
    int8_t in_data[] = {100, 101, 102};
    TensorView input(shape, DataType::i8, in_data);

    QuantLinearAttributes attrs;
    attrs.output_dtype = DataType::f32;
    auto op = DequantizeLinear::create(attrs, Backend::CPU);

    std::vector<float> out_buf(3);
    TensorView output(shape, DataType::f32, out_buf.data());

    float scale_data[] = {2.0f};
    int8_t zp_data[] = {100};
    const int64_t s_shape[] = {1};
    TensorView scale(s_shape, DataType::f32, scale_data);
    TensorView zp(s_shape, DataType::i8, zp_data);

    const TensorView ins[] = {input, scale, zp};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], 0.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], 2.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[2], 4.0f, 1e-5f);
}

NNOPS_TEST(dequantize_linear_u8_basic) {
    const int64_t shape[] = {3};
    uint8_t in_data[] = {128, 129, 127};
    TensorView input(shape, DataType::u8, in_data);

    QuantLinearAttributes attrs;
    attrs.output_dtype = DataType::f32;
    auto op = DequantizeLinear::create(attrs, Backend::CPU);

    std::vector<float> out_buf(3);
    TensorView output(shape, DataType::f32, out_buf.data());

    float scale_data[] = {0.5f};
    uint8_t zp_data[] = {128};
    const int64_t s_shape[] = {1};
    TensorView scale(s_shape, DataType::f32, scale_data);
    TensorView zp(s_shape, DataType::u8, zp_data);

    const TensorView ins[] = {input, scale, zp};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], 0.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], 0.5f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[2], -0.5f, 1e-5f);
}

NNOPS_TEST(dequantize_linear_2d_per_channel) {
    const int64_t shape[] = {2, 3};
    int8_t in_data[] = {0, 0, 1, 10, 5, 4};
    TensorView input(shape, DataType::i8, in_data);

    QuantLinearAttributes attrs;
    attrs.axis = 1;
    attrs.output_dtype = DataType::f32;
    auto op = DequantizeLinear::create(attrs, Backend::CPU);

    std::vector<float> out_buf(6);
    TensorView output(shape, DataType::f32, out_buf.data());

    float scale_data[] = {1.0f, 2.0f, 3.0f};
    int8_t zp_data[] = {0, 0, 0};
    const int64_t s_shape[] = {3};
    TensorView scale(s_shape, DataType::f32, scale_data);
    TensorView zp(s_shape, DataType::i8, zp_data);

    const TensorView ins[] = {input, scale, zp};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], 0.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[3], 10.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], 0.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[4], 10.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[2], 3.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[5], 12.0f, 1e-5f);
}

NNOPS_TEST(dequantize_linear_random) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-128, 127);

    const int64_t shape[] = {4, 8};
    std::vector<int8_t> in_buf(32);
    for (auto& v : in_buf) v = static_cast<int8_t>(dist(rng));
    TensorView input(shape, DataType::i8, in_buf.data());

    QuantLinearAttributes attrs;
    attrs.output_dtype = DataType::f32;
    auto op = DequantizeLinear::create(attrs, Backend::CPU);

    std::vector<float> out_buf(32);
    TensorView output(shape, DataType::f32, out_buf.data());

    float scale_data[] = {1.5f};
    int8_t zp_data[] = {-3};
    const int64_t s_shape[] = {1};
    TensorView scale(s_shape, DataType::f32, scale_data);
    TensorView zp(s_shape, DataType::i8, zp_data);

    const TensorView ins[] = {input, scale, zp};
    op->compute(output, ins);

    for (int i = 0; i < 32; ++i) {
        float expected = (static_cast<float>(in_buf[i]) - static_cast<float>(-3)) * 1.5f;
        NNOPS_EXPECT_NEAR(out_buf[i], expected, 1e-4f);
    }
}

// ============================================================
// Roundtrip test
// ============================================================

NNOPS_TEST(quant_dequant_roundtrip_i8) {
    auto [in_vec, input] = test::make_random_tensor({4, 8}, -5.0f, 5.0f);

    float scale_val = 0.1f;
    int8_t zp_val = 0;

    // Quantize
    QuantLinearAttributes q_attrs;
    q_attrs.output_dtype = DataType::i8;
    auto q_op = QuantizeLinear::create(q_attrs, Backend::CPU);

    std::vector<int8_t> q_buf(32);
    const int64_t shape[] = {4, 8};
    TensorView q_out(shape, DataType::i8, q_buf.data());

    float q_scale_data[] = {scale_val};
    int8_t q_zp_data[] = {zp_val};
    const int64_t s_shape[] = {1};
    TensorView q_scale(s_shape, DataType::f32, q_scale_data);
    TensorView q_zp(s_shape, DataType::i8, q_zp_data);
    const TensorView q_ins[] = {input, q_scale, q_zp};
    q_op->compute(q_out, q_ins);

    // Dequantize
    QuantLinearAttributes dq_attrs;
    dq_attrs.output_dtype = DataType::f32;
    auto dq_op = DequantizeLinear::create(dq_attrs, Backend::CPU);

    std::vector<float> dq_buf(32);
    TensorView dq_out(shape, DataType::f32, dq_buf.data());
    const TensorView dq_ins[] = {q_out, q_scale, q_zp};
    dq_op->compute(dq_out, dq_ins);

    for (int i = 0; i < 32; ++i) {
        float err = std::abs(dq_buf[i] - in_vec[i]);
        NNOPS_EXPECT_TRUE(err <= scale_val * 0.55f);
    }
}

// ============================================================
// OpType tests
// ============================================================

NNOPS_TEST(quant_linear_op_type) {
    auto q_op = QuantizeLinear::create(Backend::CPU);
    NNOPS_EXPECT_EQ(q_op->getOpType(), OpType::QuantizeLinear);
    NNOPS_EXPECT_EQ(q_op->getBackend(), Backend::CPU);

    auto dq_op = DequantizeLinear::create(Backend::CPU);
    NNOPS_EXPECT_EQ(dq_op->getOpType(), OpType::DequantizeLinear);
    NNOPS_EXPECT_EQ(dq_op->getBackend(), Backend::CPU);
}
