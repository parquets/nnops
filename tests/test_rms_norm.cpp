/// Unit tests for RMSNorm operator (CPU reference).
/// Uses RMSNorm class API exclusively — no functional rms_norm() calls.

#include "nnops/ops/rms_norm.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

// ============================================================
// Hand-verified small tests
// ============================================================

NNOPS_TEST(rmsnorm_simple_last_axis) {
    const int64_t shape[] = {4};
    const int64_t scale_shape[] = {4};

    float x_data[]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float s_data[]  = {1.0f, 1.0f, 1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    auto op = RMSNorm::create(Backend::CPU);

    auto d_x = x.desc();
    auto d_s = s.desc();
    const TensorDesc desc_arr[] = {d_x, d_s};
    auto descs = op->getOutputTensorDesc(desc_arr);

    // Validate output descriptor
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(4));
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(y, ins);

    // rms = sqrt(mean(x^2) + eps) = sqrt((1+4+9+16)/4 + 1e-5) = sqrt(7.5 + 1e-5)
    float sum_sq = 1.0f + 4.0f + 9.0f + 16.0f;
    float rms = std::sqrt(sum_sq / 4.0f + 1e-5f);
    for (int i = 0; i < 4; ++i) {
        float expected = x_data[i] / rms;
        NNOPS_EXPECT_NEAR(out_buf[i], expected, 1e-3f);
    }
}

NNOPS_TEST(rmsnorm_with_scale) {
    const int64_t shape[] = {2};
    const int64_t scale_shape[] = {2};

    float x_data[]  = {2.0f, 4.0f};
    float s_data[]  = {0.5f, 2.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    auto op = RMSNorm::create(Backend::CPU);

    auto d_x = x.desc();
    auto d_s = s.desc();
    const TensorDesc desc_arr[] = {d_x, d_s};
    auto descs = op->getOutputTensorDesc(desc_arr);

    // Validate output descriptor
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(y, ins);

    // rms = sqrt((4 + 16) / 2 + 1e-5) = sqrt(10 + 1e-5)
    float rms = std::sqrt(10.0f + 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[0], 2.0f / rms * 0.5f, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[1], 4.0f / rms * 2.0f, 1e-3f);
}

NNOPS_TEST(rmsnorm_2d_axis_1) {
    const int64_t shape[] = {2, 2};
    const int64_t scale_shape[] = {2};

    float x_data[]  = {1.0f, 1.0f,
                       2.0f, 0.0f};
    float s_data[]  = {1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    RMSNormAttributes attrs;
    attrs.axis = 1;
    auto op = RMSNorm::create(attrs, Backend::CPU);

    auto d_x = x.desc();
    auto d_s = s.desc();
    const TensorDesc desc_arr[] = {d_x, d_s};
    auto descs = op->getOutputTensorDesc(desc_arr);

    // Validate output descriptor
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(y, ins);

    // Row 0: [1,1] → rms = sqrt((1+1)/2 + eps) = sqrt(1 + eps) ≈ 1
    // Row 1: [2,0] → rms = sqrt((4+0)/2 + eps) = sqrt(2 + eps)
    float rms0 = std::sqrt(1.0f + 1e-5f);
    float rms1 = std::sqrt(2.0f + 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[0], 1.0f / rms0, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[1], 1.0f / rms0, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[2], 2.0f / rms1, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[3], 0.0f, 1e-3f);
}

NNOPS_TEST(rmsnorm_random) {
    auto [in_vec, input] = test::make_random_tensor({3, 6}, -1.0f, 1.0f);
    std::vector<float> scale_buf(6, 1.0f);

    const int64_t scale_shape[] = {6};
    TensorView s(scale_shape, DataType::f32, scale_buf.data());

    auto op = RMSNorm::create(Backend::CPU);

    auto d_x = input.desc();
    auto d_s = s.desc();
    const TensorDesc desc_arr[] = {d_x, d_s};
    auto descs = op->getOutputTensorDesc(desc_arr);

    // Validate output descriptor
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(3));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(6));
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {input, s};
    op->compute(y, ins);

    // With scale=1: each row's RMS should be ≈ 1
    for (int r = 0; r < 3; ++r) {
        float sum_sq = 0.0f;
        for (int c = 0; c < 6; ++c) {
            sum_sq += y.ptr<float>()[r * 6 + c] *
                      y.ptr<float>()[r * 6 + c];
        }
        float rms = std::sqrt(sum_sq / 6.0f);
        NNOPS_EXPECT_NEAR(rms, 1.0f, 0.1f);
    }
}
