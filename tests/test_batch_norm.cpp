/// Unit tests for BatchNorm operator (CPU reference, inference only).

#include "nnops/ops/batch_norm.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

NNOPS_TEST(batchnorm_simple_1d) {
    // 1D input [N=4], C=1
    const int64_t shape_x[] = {4};
    const int64_t shape_c[] = {1};

    float x_data[]     = {1.0f, 2.0f, 3.0f, 4.0f};
    float scale_data[] = {1.0f};
    float bias_data[]  = {0.0f};
    float mean_data[]  = {2.5f};  // (1+2+3+4)/4
    float var_data[]   = {1.25f}; // variance
    float out_data[4]  = {};

    TensorView x(shape_x, DataType::f32, x_data);
    TensorView s(shape_c, DataType::f32, scale_data);
    TensorView b(shape_c, DataType::f32, bias_data);
    TensorView m(shape_c, DataType::f32, mean_data);
    TensorView v(shape_c, DataType::f32, var_data);

    // Class API
    BatchNormAttributes attrs;
    auto op = BatchNorm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    auto dm = m.desc();
    auto dv = v.desc();
    const TensorDesc desc_arr[] = {dx, ds, db, dm, dv};
    auto descs = op->getOutputTensorDesc(desc_arr);

    // Validate output descriptor
    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 4);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    auto out = test::make_planar(descs[0], out_data);

    const TensorView ins[] = {x, s, b, m, v};
    TensorView outs[] = {out};
    op->compute(outs, ins);

    // Manual: (x - 2.5) / sqrt(1.25 + 1e-5) * 1.0 + 0.0
    float eps = 1e-5f;
    float inv_std = 1.0f / std::sqrt(1.25f + eps);
    for (int i = 0; i < 4; ++i) {
        float expected = (x_data[i] - 2.5f) * inv_std;
        NNOPS_EXPECT_NEAR(out_data[i], expected, 1e-4f);
    }
}

NNOPS_TEST(batchnorm_2d_with_scale_bias) {
    // N=2, C=2, spatial dim: 1x1 (total 2 elements per channel)
    const int64_t shape_x[] = {2, 2};
    const int64_t shape_c[] = {2};

    float x_data[]     = {1.0f, 3.0f,   // sample 0: ch0=1, ch1=3
                          5.0f, 7.0f};  // sample 1: ch0=5, ch1=7
    float scale_data[] = {2.0f, 3.0f};
    float bias_data[]  = {1.0f, -1.0f};
    float mean_data[]  = {0.0f, 0.0f};
    float var_data[]   = {1.0f, 1.0f};
    float out_data[4]  = {};

    TensorView x(shape_x, DataType::f32, x_data);
    TensorView s(shape_c, DataType::f32, scale_data);
    TensorView b(shape_c, DataType::f32, bias_data);
    TensorView m(shape_c, DataType::f32, mean_data);
    TensorView v(shape_c, DataType::f32, var_data);

    // Class API
    BatchNormAttributes attrs;
    auto op = BatchNorm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    auto dm = m.desc();
    auto dv = v.desc();
    const TensorDesc desc_arr[] = {dx, ds, db, dm, dv};
    auto descs = op->getOutputTensorDesc(desc_arr);

    // Validate output descriptor
    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 2);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    auto out = test::make_planar(descs[0], out_data);

    const TensorView ins[] = {x, s, b, m, v};
    TensorView outs[] = {out};
    op->compute(outs, ins);

    float eps = 1e-5f;
    float inv_std = 1.0f / std::sqrt(1.0f + eps);
    float ns0 = inv_std * 2.0f;  // ch0 new_scale
    float nb0 = 1.0f - 0.0f * ns0;  // ch0 new_bias
    float ns1 = inv_std * 3.0f;  // ch1 new_scale
    float nb1 = -1.0f - 0.0f * ns1;  // ch1 new_bias

    NNOPS_EXPECT_NEAR(out_data[0], 1.0f * ns0 + nb0, 1e-4f);   // ch0, sample 0
    NNOPS_EXPECT_NEAR(out_data[1], 3.0f * ns1 + nb1, 1e-4f);   // ch1, sample 0
    NNOPS_EXPECT_NEAR(out_data[2], 5.0f * ns0 + nb0, 1e-4f);   // ch0, sample 1
    NNOPS_EXPECT_NEAR(out_data[3], 7.0f * ns1 + nb1, 1e-4f);   // ch1, sample 1
}

NNOPS_TEST(batchnorm_random) {
    auto [in_vec, input] = test::make_random_tensor({2, 4, 3}, -1.0f, 1.0f);
    std::vector<float> scale_buf(4, 1.0f);
    std::vector<float> bias_buf(4, 0.0f);
    std::vector<float> mean_buf(4, 0.0f);
    std::vector<float> var_buf(4, 1.0f);
    std::vector<float> out_buf(2 * 4 * 3);

    int64_t shape_c[] = {4};
    int64_t shape_x[] = {2, 4, 3};

    TensorView x(shape_x, DataType::f32, in_vec.data());
    TensorView s(shape_c, DataType::f32, scale_buf.data());
    TensorView b(shape_c, DataType::f32, bias_buf.data());
    TensorView m(shape_c, DataType::f32, mean_buf.data());
    TensorView v(shape_c, DataType::f32, var_buf.data());

    // Class API
    BatchNormAttributes attrs;
    auto op = BatchNorm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    auto dm = m.desc();
    auto dv = v.desc();
    const TensorDesc desc_arr[] = {dx, ds, db, dm, dv};
    auto descs = op->getOutputTensorDesc(desc_arr);

    // Validate output descriptor
    NNOPS_EXPECT_EQ(descs[0].rank, 3);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 4);
    NNOPS_EXPECT_EQ(descs[0].dims[2], 3);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    auto out = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {x, s, b, m, v};
    TensorView outs[] = {out};
    op->compute(outs, ins);

    // With mean=0, var=1, scale=1, bias=0: output should equal input
    NNOPS_EXPECT_TRUE(test::allclose(x, out, 1e-3f, 1e-3f));
}
