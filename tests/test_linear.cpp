/// Unit tests for Linear operator (CPU reference).
/// Uses Linear class API exclusively — no functional linear() calls.

#include "nnops/ops/linear.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

// ============================================================
// Hand-verified tests
// ============================================================

NNOPS_TEST(linear_basic) {
    // input: [2, 3] (M=2, K=3)
    // weight: [4, 3] (N=4, K=3)
    // output: [2, 4]
    const int64_t ishape[] = {2, 3};
    const int64_t wshape[] = {4, 3};

    float in_data[6] = {1, 2, 3, 4, 5, 6};
    float w_data[12] = {
        1, 0, 0,   // weight[0] * input[m] => dot({1,2,3}, {1,0,0}) = 1
        0, 1, 0,   // dot({1,2,3}, {0,1,0}) = 2
        0, 0, 1,   // dot({1,2,3}, {0,0,1}) = 3
        1, 1, 1,   // dot({1,2,3}, {1,1,1}) = 6
    };

    TensorView input(ishape, DataType::f32, in_data);
    TensorView weight(wshape, DataType::f32, w_data);

    auto di = input.desc();
    auto dw = weight.desc();

    auto op = Linear::create(Backend::CPU);
    const TensorDesc desc_arr[] = {di, dw};
    auto descs = op->getOutputTensorDesc(desc_arr);

    // Validate output descriptor
    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 4);
    NNOPS_EXPECT_EQ(static_cast<int>(descs[0].layout), static_cast<int>(TensorLayout::NCHW));
    NNOPS_EXPECT_EQ(static_cast<int>(descs[0].dtype), static_cast<int>(DataType::f32));

    int64_t out_numel = 1;
    for (int64_t i = 0; i < descs[0].rank; ++i) {
        out_numel *= descs[0].dims[i];
    }
    std::vector<float> out_buf(static_cast<size_t>(out_numel), 0.0f);
    auto out = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {input, weight};
    op->compute(out, ins);

    // Row 0: dot({1,2,3}, w[i])
    NNOPS_EXPECT_NEAR(out_buf[0], 1.0f, 1e-6f);   // 1*1 + 2*0 + 3*0
    NNOPS_EXPECT_NEAR(out_buf[1], 2.0f, 1e-6f);   // 1*0 + 2*1 + 3*0
    NNOPS_EXPECT_NEAR(out_buf[2], 3.0f, 1e-6f);   // 1*0 + 2*0 + 3*1
    NNOPS_EXPECT_NEAR(out_buf[3], 6.0f, 1e-6f);   // 1*1 + 2*1 + 3*1

    // Row 1: dot({4,5,6}, w[i])
    NNOPS_EXPECT_NEAR(out_buf[4], 4.0f, 1e-6f);   // 4*1 + 5*0 + 6*0
    NNOPS_EXPECT_NEAR(out_buf[5], 5.0f, 1e-6f);   // 4*0 + 5*1 + 6*0
    NNOPS_EXPECT_NEAR(out_buf[6], 6.0f, 1e-6f);   // 4*0 + 5*0 + 6*1
    NNOPS_EXPECT_NEAR(out_buf[7], 15.0f, 1e-6f);  // 4*1 + 5*1 + 6*1
}

NNOPS_TEST(linear_with_bias) {
    const int64_t ishape[] = {1, 2};
    const int64_t wshape[] = {3, 2};
    const int64_t bshape[] = {3};

    float in_data[2] = {1, 2};
    float w_data[6] = {1, 0, 0, 1, 1, 1};
    float b_data[3] = {0.5f, -0.5f, 1.0f};

    TensorView input(ishape, DataType::f32, in_data);
    TensorView weight(wshape, DataType::f32, w_data);
    TensorView bias(bshape, DataType::f32, b_data);

    auto di = input.desc();
    auto dw = weight.desc();

    auto op = Linear::create(Backend::CPU);
    const TensorDesc desc_arr[] = {di, dw};
    auto descs = op->getOutputTensorDesc(desc_arr);

    // Validate output descriptor
    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 1);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 3);
    NNOPS_EXPECT_EQ(static_cast<int>(descs[0].layout), static_cast<int>(TensorLayout::NCHW));
    NNOPS_EXPECT_EQ(static_cast<int>(descs[0].dtype), static_cast<int>(DataType::f32));

    int64_t out_numel = 1;
    for (int64_t i = 0; i < descs[0].rank; ++i) {
        out_numel *= descs[0].dims[i];
    }
    std::vector<float> out_buf(static_cast<size_t>(out_numel), 0.0f);
    auto out = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {input, weight, bias};
    op->compute(out, ins);

    // dot({1,2}, {1,0}) + 0.5 = 1 + 0.5 = 1.5
    NNOPS_EXPECT_NEAR(out_buf[0], 1.5f, 1e-6f);
    // dot({1,2}, {0,1}) - 0.5 = 2 - 0.5 = 1.5
    NNOPS_EXPECT_NEAR(out_buf[1], 1.5f, 1e-6f);
    // dot({1,2}, {1,1}) + 1.0 = 3 + 1.0 = 4.0
    NNOPS_EXPECT_NEAR(out_buf[2], 4.0f, 1e-6f);
}

// ============================================================
// Random data test
// ============================================================

NNOPS_TEST(linear_random) {
    auto [in_vec, input] = test::make_random_tensor({8, 16});
    auto [w_vec, weight] = test::make_random_tensor({32, 16});

    auto di = input.desc();
    auto dw = weight.desc();

    auto op = Linear::create(Backend::CPU);
    const TensorDesc desc_arr[] = {di, dw};
    auto descs = op->getOutputTensorDesc(desc_arr);

    // Validate output descriptor
    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 8);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 32);
    NNOPS_EXPECT_EQ(static_cast<int>(descs[0].layout), static_cast<int>(TensorLayout::NCHW));
    NNOPS_EXPECT_EQ(static_cast<int>(descs[0].dtype), static_cast<int>(DataType::f32));

    int64_t out_numel = 1;
    for (int64_t i = 0; i < descs[0].rank; ++i) {
        out_numel *= descs[0].dims[i];
    }
    std::vector<float> out_buf(static_cast<size_t>(out_numel));
    auto out = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {input, weight};
    op->compute(out, ins);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}
