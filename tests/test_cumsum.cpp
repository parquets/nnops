/// Unit tests for CumSum operator (CPU reference).

#include "nnops/ops/cumsum.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>

using namespace nnops;

NNOPS_TEST(cumsum_1d_inclusive) {
    const int64_t shape[] = {5};
    float in_data[]  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float out_data[5] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(shape, DataType::f32, out_data);

    cumsum(input, output);  // default: inclusive, forward, axis=0

    NNOPS_EXPECT_NEAR(out_data[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 6.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[3], 10.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[4], 15.0f, 1e-6f);
}

NNOPS_TEST(cumsum_1d_exclusive) {
    const int64_t shape[] = {4};
    float in_data[]  = {2.0f, 4.0f, 6.0f, 8.0f};
    float out_data[4] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(shape, DataType::f32, out_data);

    CumSumAttributes attrs;
    attrs.exclusive = true;
    cumsum(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 6.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[3], 12.0f, 1e-6f);
}

NNOPS_TEST(cumsum_1d_reverse) {
    const int64_t shape[] = {4};
    float in_data[]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float out_data[4] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(shape, DataType::f32, out_data);

    CumSumAttributes attrs;
    attrs.reverse = true;
    cumsum(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 10.0f, 1e-6f);  // 1+2+3+4
    NNOPS_EXPECT_NEAR(out_data[1], 9.0f, 1e-6f);   // 2+3+4
    NNOPS_EXPECT_NEAR(out_data[2], 7.0f, 1e-6f);   // 3+4
    NNOPS_EXPECT_NEAR(out_data[3], 4.0f, 1e-6f);   // 4
}

NNOPS_TEST(cumsum_2d_axis_1) {
    // 2x3, cumsum along axis=1
    const int64_t shape[] = {2, 3};
    float in_data[]  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float out_data[6] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView output(shape, DataType::f32, out_data);

    CumSumAttributes attrs;
    attrs.axis = 1;
    cumsum(input, output, attrs);

    // Row 0: 1, 1+2=3, 1+2+3=6
    NNOPS_EXPECT_NEAR(out_data[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 6.0f, 1e-6f);
    // Row 1: 4, 4+5=9, 4+5+6=15
    NNOPS_EXPECT_NEAR(out_data[3], 4.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[4], 9.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[5], 15.0f, 1e-6f);
}

NNOPS_TEST(cumsum_random) {
    auto [in_vec, input] = test::make_random_tensor({3, 10}, -1.0f, 1.0f);
    std::vector<float> out_buf(30);
    TensorView output(input.shape_span(), DataType::f32, out_buf.data());

    CumSumAttributes attrs;
    attrs.axis = 1;
    cumsum(input, output, attrs);

    // Verify inclusive property: diff between consecutive elements matches input
    for (int r = 0; r < 3; ++r) {
        NNOPS_EXPECT_NEAR(out_buf[r * 10], in_vec[r * 10], 1e-5f);
        for (int c = 1; c < 10; ++c) {
            float diff = out_buf[r * 10 + c] - out_buf[r * 10 + c - 1];
            NNOPS_EXPECT_NEAR(diff, in_vec[r * 10 + c], 1e-5f);
        }
    }
}

NNOPS_TEST(cumsum_class_api) {
    const int64_t shape[] = {5};
    float in_data[]  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float out1[5] = {}, out2[5] = {};

    TensorView input(shape, DataType::f32, in_data);
    TensorView out1_view(shape, DataType::f32, out1);
    TensorView out2_view(shape, DataType::f32, out2);

    cumsum(input, out1_view);

    auto op = CumSum::create(Backend::CPU);
    const TensorView ins[] = {input};
    op->compute(out2_view, ins);

    NNOPS_EXPECT_TRUE(test::allclose(out1_view, out2_view));
}
