/// Unit tests for CumSum operator (CPU reference) — class API only.
///
/// All tests use the CumSum class interface:
///   1. auto op = CumSum::create(attrs, Backend::CPU)
///   2. auto d = input.desc(); const TensorDesc arr[] = {d};
///      auto descs = op->getOutputTensorDesc(arr);
///   3. Validate descs[0].rank / dims / layout / dtype
///   4. Create output via test::make_planar
///   5. const TensorView ins[] = {input}; op->compute(out, ins);

#include "nnops/ops/cumsum.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/test_helpers.hpp"

#include <vector>

using namespace nnops;

NNOPS_TEST(cumsum_1d_inclusive) {
    const int64_t shape[] = {5};
    float in_data[]  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float out_data[5] = {};

    TensorView input(shape, DataType::f32, in_data);

    auto d = input.desc();
    auto op = CumSum::create(Backend::CPU);
    const TensorDesc desc_arr[] = {d};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(5));
    NNOPS_EXPECT_EQ(static_cast<int>(descs[0].layout), static_cast<int>(TensorLayout::NCHW));
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    auto out = test::make_planar(descs[0], out_data);

    const TensorView ins[] = {input};
    op->compute(out, ins);

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

    CumSumAttributes attrs;
    attrs.exclusive = true;

    auto d = input.desc();
    auto op = CumSum::create(attrs, Backend::CPU);
    const TensorDesc desc_arr[] = {d};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(4));
    NNOPS_EXPECT_EQ(static_cast<int>(descs[0].layout), static_cast<int>(TensorLayout::NCHW));
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    auto out = test::make_planar(descs[0], out_data);

    const TensorView ins[] = {input};
    op->compute(out, ins);

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

    CumSumAttributes attrs;
    attrs.reverse = true;

    auto d = input.desc();
    auto op = CumSum::create(attrs, Backend::CPU);
    const TensorDesc desc_arr[] = {d};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(4));
    NNOPS_EXPECT_EQ(static_cast<int>(descs[0].layout), static_cast<int>(TensorLayout::NCHW));
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    auto out = test::make_planar(descs[0], out_data);

    const TensorView ins[] = {input};
    op->compute(out, ins);

    NNOPS_EXPECT_NEAR(out_data[0], 10.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 9.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 7.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[3], 4.0f, 1e-6f);
}

NNOPS_TEST(cumsum_2d_axis_1) {
    const int64_t shape[] = {2, 3};
    float in_data[]  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float out_data[6] = {};

    TensorView input(shape, DataType::f32, in_data);

    CumSumAttributes attrs;
    attrs.axis = 1;

    auto d = input.desc();
    auto op = CumSum::create(attrs, Backend::CPU);
    const TensorDesc desc_arr[] = {d};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(3));
    NNOPS_EXPECT_EQ(static_cast<int>(descs[0].layout), static_cast<int>(TensorLayout::NCHW));
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    auto out = test::make_planar(descs[0], out_data);

    const TensorView ins[] = {input};
    op->compute(out, ins);

    NNOPS_EXPECT_NEAR(out_data[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 6.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[3], 4.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[4], 9.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[5], 15.0f, 1e-6f);
}

NNOPS_TEST(cumsum_random) {
    auto [in_vec, input] = test::make_random_tensor({3, 10}, -1.0f, 1.0f);

    CumSumAttributes attrs;
    attrs.axis = 1;
    auto op = CumSum::create(attrs, Backend::CPU);

    auto d = input.desc();
    const TensorDesc desc_arr[] = {d};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(3));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(10));
    NNOPS_EXPECT_EQ(static_cast<int>(descs[0].layout), static_cast<int>(TensorLayout::NCHW));
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    auto out = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(out, ins);

    for (int r = 0; r < 3; ++r) {
        NNOPS_EXPECT_NEAR(out_buf[r * 10], in_vec[r * 10], 1e-5f);
        for (int c = 1; c < 10; ++c) {
            float diff = out_buf[r * 10 + c] - out_buf[r * 10 + c - 1];
            NNOPS_EXPECT_NEAR(diff, in_vec[r * 10 + c], 1e-5f);
        }
    }
}
