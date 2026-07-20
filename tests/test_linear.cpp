/// Unit tests for Linear operator (CPU reference).

#include "nnops/ops/linear.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

NNOPS_TEST(linear_basic) {
    // input: [2, 3] (M=2, K=3)
    // weight: [4, 3] (N=4, K=3)
    // output: [2, 4]
    const int64_t ishape[] = {2, 3};
    const int64_t wshape[] = {4, 3};
    const int64_t oshape[] = {2, 4};

    float in_data[6] = {1, 2, 3, 4, 5, 6};
    float w_data[12] = {
        1, 0, 0,   // weight[0] * input[m] => dot({1,2,3}, {1,0,0}) = 1
        0, 1, 0,   // dot({1,2,3}, {0,1,0}) = 2
        0, 0, 1,   // dot({1,2,3}, {0,0,1}) = 3
        1, 1, 1,   // dot({1,2,3}, {1,1,1}) = 6
    };
    float out_data[8] = {};

    TensorView input(ishape, DataType::f32, in_data);
    TensorView weight(wshape, DataType::f32, w_data);
    TensorView output(oshape, DataType::f32, out_data);

    linear(input, weight, output);

    // Row 0: dot({1,2,3}, w[i])
    NNOPS_EXPECT_NEAR(out_data[0], 1.0f, 1e-6f);   // 1*1 + 2*0 + 3*0
    NNOPS_EXPECT_NEAR(out_data[1], 2.0f, 1e-6f);   // 1*0 + 2*1 + 3*0
    NNOPS_EXPECT_NEAR(out_data[2], 3.0f, 1e-6f);   // 1*0 + 2*0 + 3*1
    NNOPS_EXPECT_NEAR(out_data[3], 6.0f, 1e-6f);   // 1*1 + 2*1 + 3*1

    // Row 1: dot({4,5,6}, w[i])
    NNOPS_EXPECT_NEAR(out_data[4], 4.0f, 1e-6f);   // 4*1 + 5*0 + 6*0
    NNOPS_EXPECT_NEAR(out_data[5], 5.0f, 1e-6f);   // 4*0 + 5*1 + 6*0
    NNOPS_EXPECT_NEAR(out_data[6], 6.0f, 1e-6f);   // 4*0 + 5*0 + 6*1
    NNOPS_EXPECT_NEAR(out_data[7], 15.0f, 1e-6f);  // 4*1 + 5*1 + 6*1
}

NNOPS_TEST(linear_with_bias) {
    const int64_t ishape[] = {1, 2};
    const int64_t wshape[] = {3, 2};
    const int64_t bshape[] = {3};
    const int64_t oshape[] = {1, 3};

    float in_data[2] = {1, 2};
    float w_data[6] = {1, 0, 0, 1, 1, 1};
    float b_data[3] = {0.5f, -0.5f, 1.0f};
    float out_data[3] = {};

    TensorView input(ishape, DataType::f32, in_data);
    TensorView weight(wshape, DataType::f32, w_data);
    TensorView bias(bshape, DataType::f32, b_data);
    TensorView output(oshape, DataType::f32, out_data);

    linear(input, weight, bias, output);

    // dot({1,2}, {1,0}) + 0.5 = 1 + 0.5 = 1.5
    NNOPS_EXPECT_NEAR(out_data[0], 1.5f, 1e-6f);
    // dot({1,2}, {0,1}) - 0.5 = 2 - 0.5 = 1.5
    NNOPS_EXPECT_NEAR(out_data[1], 1.5f, 1e-6f);
    // dot({1,2}, {1,1}) + 1.0 = 3 + 1.0 = 4.0
    NNOPS_EXPECT_NEAR(out_data[2], 4.0f, 1e-6f);
}

NNOPS_TEST(linear_random) {
    auto [in_vec, input] = test::make_random_tensor({8, 16});
    auto [w_vec, weight] = test::make_random_tensor({32, 16});
    std::vector<float> out_buf(8 * 32);
    const int64_t oshape[] = {8, 32};
    TensorView output(oshape, DataType::f32, out_buf.data());

    linear(input, weight, output);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

NNOPS_TEST(linear_class_api) {
    const int64_t ishape[] = {2, 3};
    const int64_t wshape[] = {4, 3};
    const int64_t oshape[] = {2, 4};
    float in_data[6] = {1,2,3,4,5,6};
    float w_data[12] = {1,0,0,0,1,0,0,0,1,1,1,1};
    float out1_data[8] = {};
    float out2_data[8] = {};

    TensorView input(ishape, DataType::f32, in_data);
    TensorView weight(wshape, DataType::f32, w_data);
    TensorView out1(oshape, DataType::f32, out1_data);
    TensorView out2(oshape, DataType::f32, out2_data);

    // Functional
    linear(input, weight, out1);
    // Class
    auto op = Linear::create(Backend::CPU);
    const TensorView ins[] = {input, weight};
    op->compute(out2, ins);

    NNOPS_EXPECT_TRUE(test::allclose(out1, out2));
}
