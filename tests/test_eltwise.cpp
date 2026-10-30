/// @file test_eltwise.cpp
/// @brief Eltwise operator unit tests, CPU backend.
///
/// Desk-checked scalar cases for Add/Sub/Mul/Div/Min/Max/Pow with hand-computed
/// expectations, then a random-data sweep for Add/Mul/Min/Max/Pow checked
/// against the scalar expression. `add_to` is the accumulating form: the output
/// view arrives pre-filled and the op adds into it instead of overwriting.
///
/// The f16 cases cover Add and Mul through the half-precision path. Their
/// inputs are rounded to fp16 before the op while the reference is the
/// unrounded f32 expression, so their tolerances widen to 5e-2 (Add) and 3e-2
/// (Mul); f32 random cases hold 1e-4 and the desk-checked Pow holds 1e-5.
///
/// The quantized cases are the fused dequant → op → requant path: both input
/// views and the output view already carry QuantParams over pre-quantized data,
/// PerTensor and PerToken (here one scale per row of a rank-2 tensor) both
/// appear, and the u8 case is asymmetric with zero_point 128. `add_to` is never
/// set on them — only the float paths exercise it.

#include "nnops/ops/eltwise.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

NNOPS_TEST(eltwise_add_1d) {
    const int64_t shape[] = {4};
    float a_data[]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[]  = {5.0f, 6.0f, 7.0f, 8.0f};
    float out_data[4] = {};

    TensorView a(shape, DataType::f32, a_data);
    TensorView b(shape, DataType::f32, b_data);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Add;
    auto op = Eltwise::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 4);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    auto output = test::make_planar(descs[0], out_data);
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_data[0], 6.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 8.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 10.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[3], 12.0f, 1e-6f);
}

NNOPS_TEST(eltwise_sub_1d) {
    const int64_t shape[] = {3};
    float a_data[]  = {10.0f, 20.0f, 30.0f};
    float b_data[]  = {3.0f, 5.0f, 7.0f};
    float out_data[3] = {};

    TensorView a(shape, DataType::f32, a_data);
    TensorView b(shape, DataType::f32, b_data);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Sub;
    auto op = Eltwise::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    auto output = test::make_planar(descs[0], out_data);
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_data[0], 7.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 15.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 23.0f, 1e-6f);
}

NNOPS_TEST(eltwise_mul_1d) {
    const int64_t shape[] = {4};
    float a_data[]  = {2.0f, 3.0f, 4.0f, 5.0f};
    float b_data[]  = {3.0f, 4.0f, 5.0f, 6.0f};
    float out_data[4] = {};

    TensorView a(shape, DataType::f32, a_data);
    TensorView b(shape, DataType::f32, b_data);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Mul;
    auto op = Eltwise::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 4);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    auto output = test::make_planar(descs[0], out_data);
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_data[0], 6.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 12.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 20.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[3], 30.0f, 1e-6f);
}

NNOPS_TEST(eltwise_div_1d) {
    const int64_t shape[] = {3};
    float a_data[]  = {6.0f, 12.0f, 20.0f};
    float b_data[]  = {2.0f, 3.0f, 4.0f};
    float out_data[3] = {};

    TensorView a(shape, DataType::f32, a_data);
    TensorView b(shape, DataType::f32, b_data);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Div;
    auto op = Eltwise::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    auto output = test::make_planar(descs[0], out_data);
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_data[0], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 4.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 5.0f, 1e-6f);
}

NNOPS_TEST(eltwise_add_to) {
    const int64_t shape[] = {3};
    float a_data[]  = {2.0f, 3.0f, 4.0f};
    float b_data[]  = {1.0f, 2.0f, 3.0f};
    float out_data[] = {10.0f, 20.0f, 30.0f};  // initial values

    TensorView a(shape, DataType::f32, a_data);
    TensorView b(shape, DataType::f32, b_data);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Add;
    attrs.add_to = true;
    auto op = Eltwise::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    auto output = test::make_planar(descs[0], out_data);
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_data[0], 10.0f + 2.0f + 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 20.0f + 3.0f + 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 30.0f + 4.0f + 3.0f, 1e-6f);
}

NNOPS_TEST(eltwise_random_add) {
    auto [a_vec, a] = test::make_random_tensor({1000}, -10.0f, 10.0f, 42);
    auto [b_vec, b] = test::make_random_tensor({1000}, -10.0f, 10.0f, 99);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Add;
    auto op = Eltwise::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 1000);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    std::vector<float> out_buf(1000);
    auto output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    for (int i = 0; i < 1000; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], a_vec[i] + b_vec[i], 1e-4f);
    }
}

NNOPS_TEST(eltwise_random_mul) {
    auto [a_vec, a] = test::make_random_tensor({500}, -2.0f, 2.0f, 77);
    auto [b_vec, b] = test::make_random_tensor({500}, -2.0f, 2.0f, 88);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Mul;
    auto op = Eltwise::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 500);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    std::vector<float> out_buf(500);
    auto output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    for (int i = 0; i < 500; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], a_vec[i] * b_vec[i], 1e-4f);
    }
}

NNOPS_TEST(eltwise_2d_add) {
    const int64_t shape[] = {2, 3};
    float a_data[] = {1, 2, 3, 4, 5, 6};
    float b_data[] = {6, 5, 4, 3, 2, 1};
    float out_data[6] = {};

    TensorView a(shape, DataType::f32, a_data);
    TensorView b(shape, DataType::f32, b_data);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Add;
    auto op = Eltwise::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 3);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    auto output = test::make_planar(descs[0], out_data);
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    for (int i = 0; i < 6; ++i) {
        NNOPS_EXPECT_NEAR(out_data[i], 7.0f, 1e-6f);
    }
}

NNOPS_TEST(eltwise_min_1d) {
    const int64_t shape[] = {4};
    float a_data[]  = {1.0f, 5.0f, -3.0f, 8.0f};
    float b_data[]  = {3.0f, 2.0f, 7.0f, 4.0f};
    float out_data[4] = {};

    TensorView a(shape, DataType::f32, a_data);
    TensorView b(shape, DataType::f32, b_data);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Min;
    auto op = Eltwise::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 4);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    auto output = test::make_planar(descs[0], out_data);
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_data[0], 1.0f, 1e-6f);   // min(1, 3)
    NNOPS_EXPECT_NEAR(out_data[1], 2.0f, 1e-6f);   // min(5, 2)
    NNOPS_EXPECT_NEAR(out_data[2], -3.0f, 1e-6f);  // min(-3, 7)
    NNOPS_EXPECT_NEAR(out_data[3], 4.0f, 1e-6f);   // min(8, 4)
}

NNOPS_TEST(eltwise_max_1d) {
    const int64_t shape[] = {4};
    float a_data[]  = {1.0f, 5.0f, -3.0f, 8.0f};
    float b_data[]  = {3.0f, 2.0f, 7.0f, 4.0f};
    float out_data[4] = {};

    TensorView a(shape, DataType::f32, a_data);
    TensorView b(shape, DataType::f32, b_data);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Max;
    auto op = Eltwise::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 4);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    auto output = test::make_planar(descs[0], out_data);
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_data[0], 3.0f, 1e-6f);   // max(1, 3)
    NNOPS_EXPECT_NEAR(out_data[1], 5.0f, 1e-6f);   // max(5, 2)
    NNOPS_EXPECT_NEAR(out_data[2], 7.0f, 1e-6f);   // max(-3, 7)
    NNOPS_EXPECT_NEAR(out_data[3], 8.0f, 1e-6f);   // max(8, 4)
}

NNOPS_TEST(eltwise_min_add_to) {
    const int64_t shape[] = {3};
    float a_data[]  = {1.0f, 10.0f, 5.0f};
    float b_data[]  = {3.0f, 4.0f, 8.0f};
    float out_data[] = {100.0f, 200.0f, 300.0f};  // initial values

    TensorView a(shape, DataType::f32, a_data);
    TensorView b(shape, DataType::f32, b_data);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Min;
    attrs.add_to = true;
    auto op = Eltwise::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    auto output = test::make_planar(descs[0], out_data);
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_data[0], 100.0f + 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 200.0f + 4.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 300.0f + 5.0f, 1e-6f);
}

NNOPS_TEST(eltwise_random_min) {
    auto [a_vec, a] = test::make_random_tensor({1000}, -10.0f, 10.0f, 42);
    auto [b_vec, b] = test::make_random_tensor({1000}, -10.0f, 10.0f, 99);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Min;
    auto op = Eltwise::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    std::vector<float> out_buf(1000);
    auto output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    for (int i = 0; i < 1000; ++i) {
        float expected = a_vec[i] < b_vec[i] ? a_vec[i] : b_vec[i];
        NNOPS_EXPECT_NEAR(out_buf[i], expected, 1e-4f);
    }
}

NNOPS_TEST(eltwise_random_max) {
    auto [a_vec, a] = test::make_random_tensor({1000}, -10.0f, 10.0f, 77);
    auto [b_vec, b] = test::make_random_tensor({1000}, -10.0f, 10.0f, 88);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Max;
    auto op = Eltwise::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    std::vector<float> out_buf(1000);
    auto output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    for (int i = 0; i < 1000; ++i) {
        float expected = a_vec[i] > b_vec[i] ? a_vec[i] : b_vec[i];
        NNOPS_EXPECT_NEAR(out_buf[i], expected, 1e-4f);
    }
}

NNOPS_TEST(eltwise_pow_1d) {
    const int64_t shape[] = {4};
    float a_data[]  = {2.0f, 3.0f, 4.0f, 5.0f};
    float b_data[]  = {2.0f, 3.0f, 2.0f, 0.0f};
    float out_data[4] = {};

    TensorView a(shape, DataType::f32, a_data);
    TensorView b(shape, DataType::f32, b_data);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Pow;
    auto op = Eltwise::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 4);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    auto output = test::make_planar(descs[0], out_data);
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_data[0], 4.0f, 1e-5f);    // 2^2
    NNOPS_EXPECT_NEAR(out_data[1], 27.0f, 1e-5f);   // 3^3
    NNOPS_EXPECT_NEAR(out_data[2], 16.0f, 1e-5f);   // 4^2
    NNOPS_EXPECT_NEAR(out_data[3], 1.0f, 1e-5f);    // 5^0
}

NNOPS_TEST(eltwise_pow_add_to) {
    const int64_t shape[] = {3};
    float a_data[]  = {2.0f, 3.0f, 4.0f};
    float b_data[]  = {2.0f, 2.0f, 2.0f};
    float out_data[] = {10.0f, 20.0f, 30.0f};

    TensorView a(shape, DataType::f32, a_data);
    TensorView b(shape, DataType::f32, b_data);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Pow;
    attrs.add_to = true;
    auto op = Eltwise::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    auto output = test::make_planar(descs[0], out_data);
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_data[0], 10.0f + 4.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_data[1], 20.0f + 9.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_data[2], 30.0f + 16.0f, 1e-5f);
}

NNOPS_TEST(eltwise_random_pow) {
    auto [a_vec, a] = test::make_random_tensor({1000}, 0.1f, 3.0f, 42);
    auto [b_vec, b] = test::make_random_tensor({1000}, 0.1f, 2.0f, 99);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Pow;
    auto op = Eltwise::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    std::vector<float> out_buf(1000);
    auto output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    for (int i = 0; i < 1000; ++i) {
        float expected = std::pow(a_vec[i], b_vec[i]);
        NNOPS_EXPECT_NEAR(out_buf[i], expected, 1e-4f);
    }
}

NNOPS_TEST(eltwise_min_max_2d) {
    const int64_t shape[] = {2, 3};
    float a_data[] = {1, 5, 3, 9, 2, 7};
    float b_data[] = {4, 2, 6, 1, 8, 3};
    float min_out[6] = {};
    float max_out[6] = {};

    TensorView a(shape, DataType::f32, a_data);
    TensorView b(shape, DataType::f32, b_data);

    // Min
    {
        EltwiseAttributes attrs;
        attrs.type = EltwiseType::Min;
        auto op = Eltwise::create(attrs, Backend::CPU);

        auto d_a = a.desc();
        auto d_b = b.desc();
        const TensorDesc desc_arr[] = {d_a, d_b};
        auto descs = op->getOutputTensorDesc(desc_arr);

        NNOPS_EXPECT_EQ(descs[0].rank, 2);
        NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
        NNOPS_EXPECT_EQ(descs[0].dims[1], 3);

        auto output = test::make_planar(descs[0], min_out);
        const TensorView ins[] = {a, b};
        op->compute(output, ins);
    }

    NNOPS_EXPECT_NEAR(min_out[0], 1.0f, 1e-6f);  // min(1, 4)
    NNOPS_EXPECT_NEAR(min_out[1], 2.0f, 1e-6f);  // min(5, 2)
    NNOPS_EXPECT_NEAR(min_out[2], 3.0f, 1e-6f);  // min(3, 6)
    NNOPS_EXPECT_NEAR(min_out[3], 1.0f, 1e-6f);  // min(9, 1)
    NNOPS_EXPECT_NEAR(min_out[4], 2.0f, 1e-6f);  // min(2, 8)
    NNOPS_EXPECT_NEAR(min_out[5], 3.0f, 1e-6f);  // min(7, 3)

    // Max
    {
        EltwiseAttributes attrs;
        attrs.type = EltwiseType::Max;
        auto op = Eltwise::create(attrs, Backend::CPU);

        auto d_a = a.desc();
        auto d_b = b.desc();
        const TensorDesc desc_arr[] = {d_a, d_b};
        auto descs = op->getOutputTensorDesc(desc_arr);

        auto output = test::make_planar(descs[0], max_out);
        const TensorView ins[] = {a, b};
        op->compute(output, ins);
    }

    NNOPS_EXPECT_NEAR(max_out[0], 4.0f, 1e-6f);  // max(1, 4)
    NNOPS_EXPECT_NEAR(max_out[1], 5.0f, 1e-6f);  // max(5, 2)
    NNOPS_EXPECT_NEAR(max_out[2], 6.0f, 1e-6f);  // max(3, 6)
    NNOPS_EXPECT_NEAR(max_out[3], 9.0f, 1e-6f);  // max(9, 1)
    NNOPS_EXPECT_NEAR(max_out[4], 8.0f, 1e-6f);  // max(2, 8)
    NNOPS_EXPECT_NEAR(max_out[5], 7.0f, 1e-6f);  // max(7, 3)
}

NNOPS_TEST(eltwise_random_add_f16) {
    auto [a_f32_vec, _a] = test::make_random_tensor({500}, -10.0f, 10.0f, 600);
    auto [b_f32_vec, _b] = test::make_random_tensor({500}, -10.0f, 10.0f, 601);

    auto a_f16 = test::f32_to_f16(a_f32_vec);
    auto b_f16 = test::f32_to_f16(b_f32_vec);

    const int64_t shape[] = {500};
    TensorView a(shape, DataType::f16, a_f16.data());
    TensorView b(shape, DataType::f16, b_f16.data());

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Add;
    auto op = Eltwise::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

    std::vector<nnops::backend::cpu::half> out_buf(500);
    auto output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    for (int i = 0; i < 500; ++i) {
        float result = simd::s_load(&out_buf[i]);
        NNOPS_EXPECT_TRUE(std::isfinite(result));
        float expected = a_f32_vec[i] + b_f32_vec[i];
        NNOPS_EXPECT_NEAR(result, expected, 5e-2f);
    }
}

NNOPS_TEST(eltwise_random_mul_f16) {
    auto [a_f32_vec, _a] = test::make_random_tensor({300}, -2.0f, 2.0f, 700);
    auto [b_f32_vec, _b] = test::make_random_tensor({300}, -2.0f, 2.0f, 701);

    auto a_f16 = test::f32_to_f16(a_f32_vec);
    auto b_f16 = test::f32_to_f16(b_f32_vec);

    const int64_t shape[] = {300};
    TensorView a(shape, DataType::f16, a_f16.data());
    TensorView b(shape, DataType::f16, b_f16.data());

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Mul;
    auto op = Eltwise::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

    std::vector<nnops::backend::cpu::half> out_buf(300);
    auto output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    for (int i = 0; i < 300; ++i) {
        float result = simd::s_load(&out_buf[i]);
        NNOPS_EXPECT_TRUE(std::isfinite(result));
        NNOPS_EXPECT_NEAR(result, a_f32_vec[i] * b_f32_vec[i], 3e-2f);
    }
}

NNOPS_TEST(eltwise_quant_per_tensor_add_s8) {
    // PerTensor, scale 1 / zero_point 0: dequant and requant cancel, so the
    // integer result is the plain sum of the codes.
    const int64_t shape[] = {4};
    int8_t a_data[] = {1, 2, 3, 4};
    int8_t b_data[] = {5, 6, 7, 8};

    QuantParams qp;
    qp.scale = 1.0f;
    qp.granularity = QuantGranularity::PerTensor;
    TensorView a(shape, DataType::s8, a_data, TensorLayout::NCHW, qp);
    TensorView b(shape, DataType::s8, b_data, TensorLayout::NCHW, qp);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Add;
    auto op = Eltwise::create(attrs, Backend::CPU);

    int8_t out_buf[4] = {};
    TensorView output(shape, DataType::s8, out_buf, TensorLayout::NCHW, qp);

    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    NNOPS_EXPECT_EQ(out_buf[0], 6);
    NNOPS_EXPECT_EQ(out_buf[1], 8);
    NNOPS_EXPECT_EQ(out_buf[2], 10);
    NNOPS_EXPECT_EQ(out_buf[3], 12);
}

NNOPS_TEST(eltwise_quant_per_token_sub_s8) {
    // PerToken, one scale per row: 0.5 then 0.25.
    const int64_t shape[] = {2, 3};
    int8_t a_data[] = {6, 9, 12,  15, 18, 21};
    int8_t b_data[] = {1, 2, 3,   4, 5, 6};

    float scale_data[] = {0.5f, 0.25f};
    QuantParams qp;
    qp.granularity = QuantGranularity::PerToken;
    qp.scale_data = scale_data;
    qp.num_scales = 2;
    TensorView a(shape, DataType::s8, a_data, TensorLayout::NCHW, qp);
    TensorView b(shape, DataType::s8, b_data, TensorLayout::NCHW, qp);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Sub;
    auto op = Eltwise::create(attrs, Backend::CPU);

    int8_t out_buf[6] = {};
    TensorView output(shape, DataType::s8, out_buf, TensorLayout::NCHW, qp);

    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    // Row0 (scale 0.5): a=[3,4.5,6] b=[0.5,1,1.5] → sub=[2.5,3.5,4.5] → /0.5=[5,7,9]
    // Row1 (scale 0.25): a=[3.75,4.5,5.25] b=[1,1.25,1.5] → sub=[2.75,3.25,3.75] → /0.25=[11,13,15]
    const int8_t expected[] = {5, 7, 9, 11, 13, 15};
    for (int i = 0; i < 6; ++i) {
        NNOPS_EXPECT_EQ(out_buf[i], expected[i]);
    }
}

NNOPS_TEST(eltwise_quant_per_tensor_add_u8) {
    // Asymmetric u8: scale 0.5, zero_point 128.
    const int64_t shape[] = {3};
    uint8_t a_data[] = {128, 129, 130};
    uint8_t b_data[] = {128, 130, 132};

    QuantParams qp;
    qp.scale = 0.5f;
    qp.zero_point = 128;
    qp.granularity = QuantGranularity::PerTensor;
    TensorView a(shape, DataType::u8, a_data, TensorLayout::NCHW, qp);
    TensorView b(shape, DataType::u8, b_data, TensorLayout::NCHW, qp);

    EltwiseAttributes attrs;
    attrs.type = EltwiseType::Add;
    auto op = Eltwise::create(attrs, Backend::CPU);

    uint8_t out_buf[3] = {};
    TensorView output(shape, DataType::u8, out_buf, TensorLayout::NCHW, qp);

    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    // dequant a=[0,0.5,1] b=[0,1,2] → add=[0,1.5,3] → /0.5+128 → [128,131,134]
    NNOPS_EXPECT_EQ(out_buf[0], 128);
    NNOPS_EXPECT_EQ(out_buf[1], 131);
    NNOPS_EXPECT_EQ(out_buf[2], 134);
}
