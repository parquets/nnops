/// Unit tests for the consolidated Norm operator (CPU reference).
///
/// Covers BatchNorm, LayerNorm, RMSNorm, GroupNorm, and L2Norm through the
/// shared Norm class / NormAttributes, distinguished by NormType.
/// Uses the class API exclusively.

#include "nnops/ops/norm.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

using namespace nnops;

// ============================================================
// BatchNorm
// ============================================================

NNOPS_TEST(norm_batchnorm_simple_1d) {
    const int64_t shape_x[] = {4};
    const int64_t shape_c[] = {1};

    float x_data[]     = {1.0f, 2.0f, 3.0f, 4.0f};
    float scale_data[] = {1.0f};
    float bias_data[]  = {0.0f};
    float mean_data[]  = {2.5f};
    float var_data[]   = {1.25f};
    float out_data[4]  = {};

    TensorView x(shape_x, DataType::f32, x_data);
    TensorView s(shape_c, DataType::f32, scale_data);
    TensorView b(shape_c, DataType::f32, bias_data);
    TensorView m(shape_c, DataType::f32, mean_data);
    TensorView v(shape_c, DataType::f32, var_data);

    NormAttributes attrs;
    attrs.type = NormType::BatchNorm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    auto dm = m.desc();
    auto dv = v.desc();
    const TensorDesc desc_arr[] = {dx, ds, db, dm, dv};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 4);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    auto out = test::make_planar(descs[0], out_data);

    const TensorView ins[] = {x, s, b, m, v};
    TensorView outs[] = {out};
    op->compute(outs, ins);

    float eps = 1e-5f;
    float inv_std = 1.0f / std::sqrt(1.25f + eps);
    for (int i = 0; i < 4; ++i) {
        float expected = (x_data[i] - 2.5f) * inv_std;
        NNOPS_EXPECT_NEAR(out_data[i], expected, 1e-4f);
    }
}

NNOPS_TEST(norm_batchnorm_2d_with_scale_bias) {
    const int64_t shape_x[] = {2, 2};
    const int64_t shape_c[] = {2};

    float x_data[]     = {1.0f, 3.0f, 5.0f, 7.0f};
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

    NormAttributes attrs;
    attrs.type = NormType::BatchNorm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    auto dm = m.desc();
    auto dv = v.desc();
    const TensorDesc desc_arr[] = {dx, ds, db, dm, dv};
    auto descs = op->getOutputTensorDesc(desc_arr);

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
    float ns0 = inv_std * 2.0f;
    float nb0 = 1.0f - 0.0f * ns0;
    float ns1 = inv_std * 3.0f;
    float nb1 = -1.0f - 0.0f * ns1;

    NNOPS_EXPECT_NEAR(out_data[0], 1.0f * ns0 + nb0, 1e-4f);
    NNOPS_EXPECT_NEAR(out_data[1], 3.0f * ns1 + nb1, 1e-4f);
    NNOPS_EXPECT_NEAR(out_data[2], 5.0f * ns0 + nb0, 1e-4f);
    NNOPS_EXPECT_NEAR(out_data[3], 7.0f * ns1 + nb1, 1e-4f);
}

NNOPS_TEST(norm_batchnorm_random) {
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

    NormAttributes attrs;
    attrs.type = NormType::BatchNorm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    auto dm = m.desc();
    auto dv = v.desc();
    const TensorDesc desc_arr[] = {dx, ds, db, dm, dv};
    auto descs = op->getOutputTensorDesc(desc_arr);

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

NNOPS_TEST(norm_batchnorm_random_f16) {
    auto [x_f32_vec, _] = test::make_random_tensor({2, 4, 3}, -1.0f, 1.0f, 200);
    std::vector<float> scale_buf(4, 1.0f);
    std::vector<float> bias_buf(4, 0.0f);
    std::vector<float> mean_buf(4, 0.0f);
    std::vector<float> var_buf(4, 1.0f);

    auto x_f16 = test::f32_to_f16(x_f32_vec);
    auto s_f16 = test::f32_to_f16(scale_buf);
    auto b_f16 = test::f32_to_f16(bias_buf);
    auto m_f16 = test::f32_to_f16(mean_buf);
    auto v_f16 = test::f32_to_f16(var_buf);

    const int64_t shape_x[] = {2, 4, 3};
    const int64_t shape_c[] = {4};
    TensorView x(shape_x, DataType::f16, x_f16.data());
    TensorView s(shape_c, DataType::f16, s_f16.data());
    TensorView b(shape_c, DataType::f16, b_f16.data());
    TensorView m(shape_c, DataType::f16, m_f16.data());
    TensorView v(shape_c, DataType::f16, v_f16.data());

    NormAttributes attrs;
    attrs.type = NormType::BatchNorm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    auto dm = m.desc();
    auto dv = v.desc();
    const TensorDesc desc_arr[] = {dx, ds, db, dm, dv};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

    std::vector<nnops::backend::cpu::half> out_buf(static_cast<size_t>(descs[0].numel()));
    auto out = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {x, s, b, m, v};
    TensorView outs[] = {out};
    op->compute(outs, ins);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        float result = simd::s_load(&out_buf[i]);
        NNOPS_EXPECT_TRUE(std::isfinite(result));
        NNOPS_EXPECT_NEAR(result, x_f32_vec[i], 1e-2f);
    }
}

NNOPS_TEST(norm_batchnorm_nchwc8_spatial) {
    const int64_t N = 2, C = 8, H = 3, W = 4;

    auto [f32_buf, _] = test::make_random_tensor({N, C, H, W}, -1.0f, 1.0f, 300);

    const int64_t num_c8 = (C + 7) / 8;
    const int64_t row_stride = W * 8;
    size_t total = static_cast<size_t>(N * num_c8 * H * row_stride);
    std::vector<float> in_data(total, 0.0f);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            int64_t c8 = c / 8;
            int64_t lane = c % 8;
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    int64_t planar_idx = n * (C * H * W) + c * (H * W) + h * W + w;
                    int64_t packed_off = n * (num_c8 * H * row_stride)
                                         + c8 * (H * row_stride)
                                         + h * row_stride + w * 8 + lane;
                    in_data[static_cast<size_t>(packed_off)] = f32_buf[static_cast<size_t>(planar_idx)];
                }
            }
        }
    }

    TensorDesc desc;
    desc.rank = 4;
    desc.dims = {N, C, H, W};
    desc.dtype = DataType::f32;
    desc.layout = TensorLayout::NCHWC8;
    TensorView x = test::make_packed(desc, in_data.data());

    std::vector<float> scale(C, 1.0f);
    std::vector<float> bias(C, 0.0f);
    std::vector<float> mean(C, 0.0f);
    std::vector<float> var(C, 1.0f);

    const int64_t shape_c[] = {C};
    TensorView s(shape_c, DataType::f32, scale.data());
    TensorView b(shape_c, DataType::f32, bias.data());
    TensorView m(shape_c, DataType::f32, mean.data());
    TensorView v(shape_c, DataType::f32, var.data());

    NormAttributes attrs;
    attrs.type = NormType::BatchNorm;
    attrs.spatial = true;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    auto dm = m.desc();
    auto dv = v.desc();
    const TensorDesc desc_arr[] = {dx, ds, db, dm, dv};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHWC8);

    std::vector<float> out_buf(total, 0.0f);
    auto out = test::make_packed(descs[0], out_buf.data());

    const TensorView ins[] = {x, s, b, m, v};
    TensorView outs[] = {out};
    op->compute(outs, ins);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            int64_t c8 = c / 8;
            int64_t lane = c % 8;
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    int64_t planar_idx = n * (C * H * W) + c * (H * W) + h * W + w;
                    int64_t packed_off = n * (num_c8 * H * row_stride)
                                         + c8 * (H * row_stride)
                                         + h * row_stride + w * 8 + lane;
                    float expected = f32_buf[static_cast<size_t>(planar_idx)];
                    float result = out_buf[static_cast<size_t>(packed_off)];
                    NNOPS_EXPECT_TRUE(std::isfinite(result));
                    NNOPS_EXPECT_NEAR(result, expected, 1e-3f);
                }
            }
        }
    }
}

// ============================================================
// LayerNorm
// ============================================================

NNOPS_TEST(norm_layernorm_simple_last_axis) {
    const int64_t shape[] = {4};
    const int64_t scale_shape[] = {4};

    float x_data[]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float s_data[]  = {1.0f, 1.0f, 1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    NormAttributes attrs;
    attrs.type = NormType::LayerNorm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {dx, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(outs[0].rank, 1);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t{4});
    NNOPS_EXPECT_EQ(outs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(out, ins);

    float mean = 2.5f;
    float var = 1.25f;
    float eps = 1e-5f;
    float inv_std = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < 4; ++i) {
        float expected = (x_data[i] - mean) * inv_std;
        NNOPS_EXPECT_NEAR(out_buf[i], expected, 1e-3f);
    }
}

NNOPS_TEST(norm_layernorm_with_bias) {
    const int64_t shape[] = {2};
    const int64_t scale_shape[] = {2};

    float x_data[]  = {0.0f, 2.0f};
    float s_data[]  = {2.0f, 3.0f};
    float b_data[]  = {1.0f, -1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);
    TensorView b(scale_shape, DataType::f32, b_data);

    NormAttributes attrs;
    attrs.type = NormType::LayerNorm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    const TensorDesc desc_arr[] = {dx, ds, db};
    auto outs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(outs[0].rank, 1);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t{2});
    NNOPS_EXPECT_EQ(outs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {x, s, b};
    op->compute(out, ins);

    float mean = 1.0f;
    float var = 1.0f;
    float inv_std = 1.0f / std::sqrt(var + 1e-5f);
    for (int i = 0; i < 2; ++i) {
        float norm = (x_data[i] - mean) * inv_std;
        float expected = norm * s_data[i] + b_data[i];
        NNOPS_EXPECT_NEAR(out_buf[i], expected, 1e-3f);
    }
}

NNOPS_TEST(norm_layernorm_2d_axis_1) {
    const int64_t shape[] = {2, 3};
    const int64_t scale_shape[] = {3};

    float x_data[]  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float s_data[]  = {1.0f, 1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    NormAttributes attrs;
    attrs.type = NormType::LayerNorm;
    attrs.axis = 1;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {dx, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(outs[0].rank, 2);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t{2});
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t{3});
    NNOPS_EXPECT_EQ(outs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(out, ins);

    float var = (1.0f + 0.0f + 1.0f) / 3.0f;
    float inv_std = 1.0f / std::sqrt(var + 1e-5f);

    NNOPS_EXPECT_NEAR(out_buf[0], (1.0f - 2.0f) * inv_std, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[1], (2.0f - 2.0f) * inv_std, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[2], (3.0f - 2.0f) * inv_std, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[3], (4.0f - 5.0f) * inv_std, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[4], (5.0f - 5.0f) * inv_std, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[5], (6.0f - 5.0f) * inv_std, 1e-3f);
}

NNOPS_TEST(norm_layernorm_random) {
    auto [in_vec, input] = test::make_random_tensor({4, 8}, -1.0f, 1.0f);
    std::vector<float> scale_buf(8, 1.0f);

    int64_t shape_s[] = {8};
    TensorView s(shape_s, DataType::f32, scale_buf.data());

    NormAttributes attrs;
    attrs.type = NormType::LayerNorm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto di = input.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {di, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(outs[0].rank, 2);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t{4});
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t{8});
    NNOPS_EXPECT_EQ(outs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {input, s};
    op->compute(out, ins);

    for (int r = 0; r < 4; ++r) {
        float row_mean = 0.0f;
        for (int c = 0; c < 8; ++c) { row_mean += out_buf[r * 8 + c]; }
        row_mean /= 8.0f;
        NNOPS_EXPECT_NEAR(row_mean, 0.0f, 1e-3f);

        float row_var = 0.0f;
        for (int c = 0; c < 8; ++c) {
            float d = out_buf[r * 8 + c] - row_mean;
            row_var += d * d;
        }
        row_var /= 8.0f;
        NNOPS_EXPECT_NEAR(row_var, 1.0f, 0.1f);
    }
}

NNOPS_TEST(norm_layernorm_random_f16) {
    auto [x_f32_vec, _] = test::make_random_tensor({4, 8}, -1.0f, 1.0f, 400);
    auto [s_f32_vec, __] = test::make_random_tensor({8}, 0.5f, 2.0f, 401);
    auto [b_f32_vec, ___] = test::make_random_tensor({8}, -0.5f, 0.5f, 402);

    auto x_f16 = test::f32_to_f16(x_f32_vec);
    auto s_f16 = test::f32_to_f16(s_f32_vec);
    auto b_f16 = test::f32_to_f16(b_f32_vec);

    const int64_t x_shape[] = {4, 8};
    const int64_t c_shape[] = {8};
    TensorView x(x_shape, DataType::f16, x_f16.data());
    TensorView s(c_shape, DataType::f16, s_f16.data());
    TensorView b(c_shape, DataType::f16, b_f16.data());

    NormAttributes attrs;
    attrs.type = NormType::LayerNorm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    const TensorDesc desc_arr[] = {dx, ds, db};
    auto outs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::f16);

    std::vector<nnops::backend::cpu::half> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {x, s, b};
    op->compute(out, ins);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        float v = simd::s_load(&out_buf[i]);
        NNOPS_EXPECT_TRUE(std::isfinite(v));
        NNOPS_EXPECT_TRUE(std::abs(v) < 100.0f);
    }
}

// ============================================================
// RMSNorm
// ============================================================

NNOPS_TEST(norm_rmsnorm_simple_last_axis) {
    const int64_t shape[] = {4};
    const int64_t scale_shape[] = {4};

    float x_data[]  = {1.0f, 2.0f, 3.0f, 4.0f};
    float s_data[]  = {1.0f, 1.0f, 1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    NormAttributes attrs;
    attrs.type = NormType::RMSNorm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto d_x = x.desc();
    auto d_s = s.desc();
    const TensorDesc desc_arr[] = {d_x, d_s};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(4));
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(y, ins);

    float sum_sq = 1.0f + 4.0f + 9.0f + 16.0f;
    float rms = std::sqrt(sum_sq / 4.0f + 1e-5f);
    for (int i = 0; i < 4; ++i) {
        float expected = x_data[i] / rms;
        NNOPS_EXPECT_NEAR(out_buf[i], expected, 1e-3f);
    }
}

NNOPS_TEST(norm_rmsnorm_with_scale) {
    const int64_t shape[] = {2};
    const int64_t scale_shape[] = {2};

    float x_data[]  = {2.0f, 4.0f};
    float s_data[]  = {0.5f, 2.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    NormAttributes attrs;
    attrs.type = NormType::RMSNorm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto d_x = x.desc();
    auto d_s = s.desc();
    const TensorDesc desc_arr[] = {d_x, d_s};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(y, ins);

    float rms = std::sqrt(10.0f + 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[0], 2.0f / rms * 0.5f, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[1], 4.0f / rms * 2.0f, 1e-3f);
}

NNOPS_TEST(norm_rmsnorm_2d_axis_1) {
    const int64_t shape[] = {2, 2};
    const int64_t scale_shape[] = {2};

    float x_data[]  = {1.0f, 1.0f, 2.0f, 0.0f};
    float s_data[]  = {1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    NormAttributes attrs;
    attrs.type = NormType::RMSNorm;
    attrs.axis = 1;
    auto op = Norm::create(attrs, Backend::CPU);

    auto d_x = x.desc();
    auto d_s = s.desc();
    const TensorDesc desc_arr[] = {d_x, d_s};
    auto descs = op->getOutputTensorDesc(desc_arr);

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

    float rms0 = std::sqrt(1.0f + 1e-5f);
    float rms1 = std::sqrt(2.0f + 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[0], 1.0f / rms0, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[1], 1.0f / rms0, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[2], 2.0f / rms1, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[3], 0.0f, 1e-3f);
}

NNOPS_TEST(norm_rmsnorm_random) {
    auto [in_vec, input] = test::make_random_tensor({3, 6}, -1.0f, 1.0f);
    std::vector<float> scale_buf(6, 1.0f);

    const int64_t scale_shape[] = {6};
    TensorView s(scale_shape, DataType::f32, scale_buf.data());

    NormAttributes attrs;
    attrs.type = NormType::RMSNorm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto d_x = input.desc();
    auto d_s = s.desc();
    const TensorDesc desc_arr[] = {d_x, d_s};
    auto descs = op->getOutputTensorDesc(desc_arr);

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

    for (int r = 0; r < 3; ++r) {
        float sum_sq = 0.0f;
        for (int c = 0; c < 6; ++c) {
            sum_sq += y.ptr<float>()[r * 6 + c] * y.ptr<float>()[r * 6 + c];
        }
        float rms = std::sqrt(sum_sq / 6.0f);
        NNOPS_EXPECT_NEAR(rms, 1.0f, 0.1f);
    }
}

NNOPS_TEST(norm_rmsnorm_random_f16) {
    auto [x_f32_vec, _] = test::make_random_tensor({3, 6}, -1.0f, 1.0f, 500);
    auto [s_f32_vec, __] = test::make_random_tensor({6}, 0.5f, 2.0f, 501);

    auto x_f16 = test::f32_to_f16(x_f32_vec);
    auto s_f16 = test::f32_to_f16(s_f32_vec);

    const int64_t x_shape[] = {3, 6};
    const int64_t s_shape[] = {6};
    TensorView x(x_shape, DataType::f16, x_f16.data());
    TensorView s(s_shape, DataType::f16, s_f16.data());

    NormAttributes attrs;
    attrs.type = NormType::RMSNorm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto d_x = x.desc();
    auto d_s = s.desc();
    const TensorDesc desc_arr[] = {d_x, d_s};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

    std::vector<nnops::backend::cpu::half> out_buf(static_cast<size_t>(descs[0].numel()));
    auto y = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(y, ins);

    for (int r = 0; r < 3; ++r) {
        float sum_sq = 0.0f;
        for (int c = 0; c < 6; ++c) {
            float v = simd::s_load(&out_buf[static_cast<size_t>(r * 6 + c)]);
            NNOPS_EXPECT_TRUE(std::isfinite(v));
            sum_sq += v * v;
        }
        float rms = std::sqrt(sum_sq / 6.0f);
        NNOPS_EXPECT_TRUE(rms > 0.01f && rms < 100.0f);
    }
}

// ============================================================
// L2Norm
// ============================================================

NNOPS_TEST(norm_l2norm_simple_last_axis) {
    const int64_t shape[] = {2};

    float x_data[] = {3.0f, 4.0f};

    TensorView x(shape, DataType::f32, x_data);

    NormAttributes attrs;
    attrs.type = NormType::L2Norm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    const TensorDesc desc_arr[] = {dx};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {x};
    op->compute(y, ins);

    // norm = sqrt(9 + 16 + eps) ≈ 5
    float norm_val = std::sqrt(25.0f + 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[0], 3.0f / norm_val, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[1], 4.0f / norm_val, 1e-3f);
}

NNOPS_TEST(norm_l2norm_2d_axis_1) {
    const int64_t shape[] = {2, 2};

    float x_data[] = {3.0f, 4.0f, 0.0f, 5.0f};

    TensorView x(shape, DataType::f32, x_data);

    NormAttributes attrs;
    attrs.type = NormType::L2Norm;
    attrs.axis = 1;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    const TensorDesc desc_arr[] = {dx};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(2));

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {x};
    op->compute(y, ins);

    float n0 = std::sqrt(25.0f + 1e-5f);
    float n1 = std::sqrt(25.0f + 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[0], 3.0f / n0, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[1], 4.0f / n0, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[2], 0.0f / n1, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[3], 5.0f / n1, 1e-3f);
}

NNOPS_TEST(norm_l2norm_random) {
    auto [in_vec, input] = test::make_random_tensor({4, 8}, -1.0f, 1.0f);

    NormAttributes attrs;
    attrs.type = NormType::L2Norm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = input.desc();
    const TensorDesc desc_arr[] = {dx};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(4));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(8));

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    auto y = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(y, ins);

    // Each row's L2 norm (sum of squares) should be ≈ 1
    for (int r = 0; r < 4; ++r) {
        float sum_sq = 0.0f;
        for (int c = 0; c < 8; ++c) {
            sum_sq += out_buf[r * 8 + c] * out_buf[r * 8 + c];
        }
        NNOPS_EXPECT_NEAR(sum_sq, 1.0f, 0.1f);
    }
}

NNOPS_TEST(norm_l2norm_add_to) {
    const int64_t shape[] = {2};

    float x_data[] = {3.0f, 4.0f};

    TensorView x(shape, DataType::f32, x_data);

    NormAttributes attrs;
    attrs.type = NormType::L2Norm;
    attrs.add_to = true;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    const TensorDesc desc_arr[] = {dx};
    auto descs = op->getOutputTensorDesc(desc_arr);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 10.0f);
    auto y = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {x};
    op->compute(y, ins);

    float norm_val = std::sqrt(25.0f + 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[0], 10.0f + 3.0f / norm_val, 1e-3f);
    NNOPS_EXPECT_NEAR(out_buf[1], 10.0f + 4.0f / norm_val, 1e-3f);
}

NNOPS_TEST(norm_l2norm_random_f16) {
    auto [x_f32_vec, _] = test::make_random_tensor({3, 6}, -1.0f, 1.0f, 700);
    auto x_f16 = test::f32_to_f16(x_f32_vec);

    const int64_t x_shape[] = {3, 6};
    TensorView x(x_shape, DataType::f16, x_f16.data());

    NormAttributes attrs;
    attrs.type = NormType::L2Norm;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    const TensorDesc desc_arr[] = {dx};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

    std::vector<nnops::backend::cpu::half> out_buf(static_cast<size_t>(descs[0].numel()));
    auto y = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {x};
    op->compute(y, ins);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        float v = simd::s_load(&out_buf[i]);
        NNOPS_EXPECT_TRUE(std::isfinite(v));
    }
}

// ============================================================
// GroupNorm
// ============================================================

NNOPS_TEST(norm_group_norm_2_groups) {
    const int64_t shape[] = {1, 4, 1, 2};
    const int64_t scale_shape[] = {4};

    float x_data[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f,
    };
    float s_data[] = {1.0f, 1.0f, 1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    NormAttributes attrs;
    attrs.type = NormType::GroupNorm;
    attrs.num_groups = 2;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {dx, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(outs[0].rank, 4);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t{1});
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t{4});
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t{1});
    NNOPS_EXPECT_EQ(outs[0].dims[3], int64_t{2});

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(out, ins);

    float eps = 1e-5f;

    float g0_mean = 2.5f, g0_var = 1.25f;
    float g0_inv_std = 1.0f / std::sqrt(g0_var + eps);
    NNOPS_EXPECT_NEAR(out_buf[0], (1.0f - g0_mean) * g0_inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[1], (2.0f - g0_mean) * g0_inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[2], (3.0f - g0_mean) * g0_inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[3], (4.0f - g0_mean) * g0_inv_std, 1e-4f);

    float g1_mean = 6.5f, g1_var = 1.25f;
    float g1_inv_std = 1.0f / std::sqrt(g1_var + eps);
    NNOPS_EXPECT_NEAR(out_buf[4], (5.0f - g1_mean) * g1_inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[5], (6.0f - g1_mean) * g1_inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[6], (7.0f - g1_mean) * g1_inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[7], (8.0f - g1_mean) * g1_inv_std, 1e-4f);
}

NNOPS_TEST(norm_group_norm_with_scale_bias) {
    const int64_t shape[] = {1, 2, 1, 2};
    const int64_t scale_shape[] = {2};

    float x_data[] = {0.0f, 2.0f, 0.0f, 4.0f};
    float s_data[] = {2.0f, 3.0f};
    float b_data[] = {1.0f, -1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);
    TensorView b(scale_shape, DataType::f32, b_data);

    NormAttributes attrs;
    attrs.type = NormType::GroupNorm;
    attrs.num_groups = 1;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    const TensorDesc desc_arr[] = {dx, ds, db};
    auto outs = op->getOutputTensorDesc(desc_arr);

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {x, s, b};
    op->compute(out, ins);

    float mean = 1.5f, var = 2.75f;
    float eps = 1e-5f;
    float inv_std = 1.0f / std::sqrt(var + eps);

    NNOPS_EXPECT_NEAR(out_buf[0], (0.0f - mean) * inv_std * 2.0f + 1.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[1], (2.0f - mean) * inv_std * 2.0f + 1.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[2], (0.0f - mean) * inv_std * 3.0f + (-1.0f), 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[3], (4.0f - mean) * inv_std * 3.0f + (-1.0f), 1e-4f);
}

NNOPS_TEST(norm_group_norm_equals_instancenorm_when_g_equals_c) {
    const int64_t shape[] = {2, 4, 2};
    const int64_t scale_shape[] = {4};

    std::vector<float> x_data = {
        1.0f, 2.0f,   3.0f, 4.0f,   5.0f, 6.0f,   7.0f, 8.0f,
        9.0f, 10.0f,  11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
    };
    std::vector<float> s_data = {1.0f, 1.0f, 1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data.data());
    TensorView s(scale_shape, DataType::f32, s_data.data());

    NormAttributes attrs;
    attrs.type = NormType::GroupNorm;
    attrs.num_groups = 4;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {dx, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());
    const TensorView ins[] = {x, s};
    op->compute(out, ins);

    float eps = 1e-5f;

    for (int n = 0; n < 2; ++n) {
        for (int c = 0; c < 4; ++c) {
            int base = n * 8 + c * 2;
            float v0 = x_data[base];
            float v1 = x_data[base + 1];
            float mean = (v0 + v1) * 0.5f;
            float var = ((v0 - mean) * (v0 - mean) + (v1 - mean) * (v1 - mean)) * 0.5f;
            float inv_std = 1.0f / std::sqrt(var + eps);
            NNOPS_EXPECT_NEAR(out_buf[base], (v0 - mean) * inv_std, 1e-4f);
            NNOPS_EXPECT_NEAR(out_buf[base + 1], (v1 - mean) * inv_std, 1e-4f);
        }
    }
}

NNOPS_TEST(norm_group_norm_add_to) {
    const int64_t shape[] = {1, 2, 2};
    const int64_t scale_shape[] = {2};

    float x_data[]  = {1.0f, 3.0f, 5.0f, 7.0f};
    float s_data[]  = {1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    NormAttributes attrs;
    attrs.type = NormType::GroupNorm;
    attrs.num_groups = 1;
    attrs.add_to = true;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {dx, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()), 10.0f);
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(out, ins);

    float mean = 4.0f;
    float var = 5.0f;
    float eps = 1e-5f;
    float inv_std = 1.0f / std::sqrt(var + eps);

    NNOPS_EXPECT_NEAR(out_buf[0], 10.0f + (1.0f - mean) * inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[1], 10.0f + (3.0f - mean) * inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[2], 10.0f + (5.0f - mean) * inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[3], 10.0f + (7.0f - mean) * inv_std, 1e-4f);
}

NNOPS_TEST(norm_group_norm_random_f32) {
    const int64_t shape[] = {2, 6, 4, 8};
    const int64_t scale_shape[] = {6};

    auto [x_vec, x] = test::make_random_tensor(shape, -10.0f, 10.0f, 500);
    auto [s_vec, s] = test::make_random_tensor(scale_shape, 0.5f, 2.0f, 501);
    auto [b_vec, b] = test::make_random_tensor(scale_shape, -1.0f, 1.0f, 502);

    for (int64_t g : {1, 2, 3, 6}) {
        NormAttributes attrs;
        attrs.type = NormType::GroupNorm;
        attrs.num_groups = g;

        auto dx = x.desc();
        auto ds = s.desc();
        auto db = b.desc();

        auto op_simd = Norm::create(attrs, Backend::CPU);
        const TensorDesc desc_simd[] = {dx, ds, db};
        auto outs_simd = op_simd->getOutputTensorDesc(desc_simd);
        std::vector<float> out_simd_buf(static_cast<size_t>(outs_simd[0].numel()));
        auto out_simd = test::make_planar(outs_simd[0], out_simd_buf.data());
        const TensorView ins_simd[] = {x, s, b};
        op_simd->compute(out_simd, ins_simd);

        float out_mean = 0.0f, out_var = 0.0f;
        int64_t total = outs_simd[0].numel();
        for (int64_t i = 0; i < total; ++i) {
            out_mean += out_simd_buf[i];
        }
        out_mean /= static_cast<float>(total);
        for (int64_t i = 0; i < total; ++i) {
            float d = out_simd_buf[i] - out_mean;
            out_var += d * d;
        }
        out_var /= static_cast<float>(total);

        NNOPS_EXPECT_TRUE(out_var > 0.01f && out_var < 100.0f);
    }
}

NNOPS_TEST(norm_group_norm_random_f16) {
    const int64_t shape[] = {1, 4, 4, 4};
    const int64_t scale_shape[] = {4};

    using nnops::backend::cpu::half;

    auto [x_f32_vec, x_f32] = test::make_random_tensor(shape, -5.0f, 5.0f, 600);
    auto [s_f32_vec, s_f32] = test::make_random_tensor(scale_shape, 0.5f, 2.0f, 601);

    const int64_t x_numel = x_f32.numel();
    const int64_t s_numel = s_f32.numel();

    std::vector<half> x_data(static_cast<size_t>(x_numel));
    std::vector<half> s_data(static_cast<size_t>(s_numel));
    for (int64_t i = 0; i < x_numel; ++i) {
        simd::s_store(&x_data[static_cast<size_t>(i)], x_f32_vec[static_cast<size_t>(i)]);
    }
    for (int64_t i = 0; i < s_numel; ++i) {
        simd::s_store(&s_data[static_cast<size_t>(i)], s_f32_vec[static_cast<size_t>(i)]);
    }

    TensorView x(shape, DataType::f16, x_data.data());
    TensorView s(scale_shape, DataType::f16, s_data.data());

    NormAttributes attrs;
    attrs.type = NormType::GroupNorm;
    attrs.num_groups = 2;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {dx, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    std::vector<half> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(out, ins);

    int64_t total = outs[0].numel();
    for (int64_t i = 0; i < total; ++i) {
        float val = simd::s_load(&out_buf[static_cast<size_t>(i)]);
        NNOPS_EXPECT_TRUE(std::isfinite(val));
    }
}

NNOPS_TEST(norm_group_norm_rank3_ncw) {
    const int64_t shape[] = {1, 3, 4};
    const int64_t scale_shape[] = {3};

    float x_data[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
    };
    float s_data[] = {1.0f, 1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    NormAttributes attrs;
    attrs.type = NormType::GroupNorm;
    attrs.num_groups = 1;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {dx, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(outs[0].rank, 3);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t{1});
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t{3});
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t{4});

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());
    const TensorView ins[] = {x, s};
    op->compute(out, ins);

    float mean = 6.5f;
    float var = 0.0f;
    for (int i = 0; i < 12; ++i) {
        var += (x_data[i] - mean) * (x_data[i] - mean);
    }
    var /= 12.0f;
    float eps = 1e-5f;
    float inv_std = 1.0f / std::sqrt(var + eps);

    for (int i = 0; i < 12; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], (x_data[i] - mean) * inv_std, 1e-4f);
    }
}

NNOPS_TEST(norm_group_norm_class_api) {
    const int64_t shape[] = {1, 2, 2};
    const int64_t scale_shape[] = {2};

    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> s_data = {1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data.data());
    TensorView s(scale_shape, DataType::f32, s_data.data());

    auto dx = x.desc();
    auto out_desc = TensorDesc{dx.dims, dx.rank, dx.dtype, dx.layout};
    std::vector<float> out_buf1(static_cast<size_t>(out_desc.numel()));
    std::vector<float> out_buf2(static_cast<size_t>(out_desc.numel()));
    auto out1 = test::make_planar(out_desc, out_buf1.data());
    auto out2 = test::make_planar(out_desc, out_buf2.data());

    NormAttributes attrs;
    attrs.type = NormType::GroupNorm;
    attrs.num_groups = 1;

    auto op = Norm::create(attrs, Backend::CPU);
    const TensorView ins_class[] = {x, s};
    op->compute(out1, ins_class);

    op->compute(out2, ins_class);

    for (size_t i = 0; i < out_buf1.size(); ++i) {
        NNOPS_EXPECT_NEAR(out_buf1[i], out_buf2[i], 1e-6f);
    }
}

NNOPS_TEST(norm_group_norm_class_api_with_bias) {
    const int64_t shape[] = {1, 2, 2};
    const int64_t scale_shape[] = {2};

    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> s_data = {1.0f, 1.0f};
    std::vector<float> b_data = {0.5f, -0.5f};

    TensorView x(shape, DataType::f32, x_data.data());
    TensorView s(scale_shape, DataType::f32, s_data.data());
    TensorView b(scale_shape, DataType::f32, b_data.data());

    auto dx = x.desc();
    auto out_desc = TensorDesc{dx.dims, dx.rank, dx.dtype, dx.layout};
    std::vector<float> out_buf1(static_cast<size_t>(out_desc.numel()));
    std::vector<float> out_buf2(static_cast<size_t>(out_desc.numel()));
    auto out1 = test::make_planar(out_desc, out_buf1.data());
    auto out2 = test::make_planar(out_desc, out_buf2.data());

    NormAttributes attrs;
    attrs.type = NormType::GroupNorm;
    attrs.num_groups = 1;

    auto op = Norm::create(attrs, Backend::CPU);
    const TensorView ins_class[] = {x, s, b};
    op->compute(out1, ins_class);

    op->compute(out2, ins_class);

    for (size_t i = 0; i < out_buf1.size(); ++i) {
        NNOPS_EXPECT_NEAR(out_buf1[i], out_buf2[i], 1e-6f);
    }
}

NNOPS_TEST(norm_group_norm_shape_inference) {
    const int64_t shape[] = {2, 8, 4, 4};
    const int64_t scale_shape[] = {8};

    std::vector<float> x_data(2 * 8 * 4 * 4);
    std::vector<float> s_data(8);

    TensorView x(shape, DataType::f32, x_data.data());
    TensorView s(scale_shape, DataType::f32, s_data.data());

    NormAttributes attrs;
    attrs.type = NormType::GroupNorm;
    attrs.num_groups = 4;
    auto op = Norm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {dx, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(outs[0].rank, 4);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t{2});
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t{8});
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t{4});
    NNOPS_EXPECT_EQ(outs[0].dims[3], int64_t{4});
    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(outs[0].layout, TensorLayout::NCHW);
}

// ============================================================
// Quantized input (s8/u8) — LayerNorm / RMSNorm / L2Norm
// ============================================================
//
// X is s8/u8 carrying its own PerTensor / PerToken quant params; the kernel
// dequantizes one row at a time, runs the float norm row kernel on it, and then
// stores f32/f16 as-is or requantizes with the output TensorView's params.
// Only axis == rank-1 is defined, so one norm row is exactly one quantization
// row (the unit of a PerToken scale/zero_point).
//
// No reference kernel takes integer input, so the expected values are built the
// slow way: dequantize by hand to f32, run reference::norm_ref on that, and
// requantize by hand when the output is integer.

namespace nnops::backend::cpu::reference {
void norm_ref(const NormAttributes& attrs,
              TensorView& output,
              std::span<const TensorView> inputs,
              const ComputeContext& ctx,
              void* workspace);
}

namespace {

// Minimal fixed-size thread pool exposing a CpuBackend (test_conv2d pattern).
// Items in [begin, end) are claimed via an atomic counter; each worker runs
// body(i) until the range is exhausted.
struct SimplePool {
    explicit SimplePool(int nthreads) : nthreads_(nthreads) {
        cpu.parallel_for = [this](int64_t begin, int64_t end, const ParallelForBody& body) {
            this->parallel_for(begin, end, body);
        };
        cpu.num_threads = [this]() { return nthreads_; };
        cpu.thread_id = []() { return current_thread_id_; };
    }

    void parallel_for(int64_t begin, int64_t end, const ParallelForBody& body) {
        std::atomic<int64_t> next{begin};
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(nthreads_));
        for (int t = 0; t < nthreads_; ++t) {
            workers.emplace_back([&, t]() {
                current_thread_id_ = t;
                for (;;) {
                    int64_t i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= end) { break; }
                    body(i);
                }
            });
        }
        for (auto& w : workers) { w.join(); }
    }

    int nthreads_;
    CpuBackend cpu;
    inline static thread_local int current_thread_id_ = 0;
};

// ---- quant-param builders (test_matmul_int8 pattern) ----

QuantParams per_tensor(float scale, int32_t zp) {
    QuantParams qp;
    qp.granularity = QuantGranularity::PerTensor;
    qp.scale = scale;
    qp.zero_point = zp;
    return qp;
}

QuantParams per_token(const float* scale, const int32_t* zp, int64_t n) {
    QuantParams qp;
    qp.granularity = QuantGranularity::PerToken;
    qp.scale_data = scale;
    qp.zero_point_data = zp;
    qp.num_scales = n;
    return qp;
}

/// Planar s8/u8 view over `data` carrying explicit quant params.
TensorView make_q(std::span<const int64_t> shape, DataType dt, void* data,
                  const QuantParams& qp, TensorLayout layout = TensorLayout::NCHW) {
    return TensorView(shape, dt, data, layout, qp);
}

/// The (scale, zero_point) the kernel uses for row `row`: the per-row buffers
/// when present, else the per-tensor scalars. A missing zero_point buffer is 0.
void row_qparam(const QuantParams& qp, int64_t row, float& scale, int32_t& zp) {
    if (qp.scale_data != nullptr) {
        scale = qp.scale_data[row];
        zp = (qp.zero_point_data != nullptr) ? qp.zero_point_data[row] : 0;
    } else {
        scale = qp.scale;
        zp = qp.zero_point;
    }
}

/// x = (q - zero_point) * scale.
float dequant_one(int64_t q, float scale, int32_t zp) {
    return (static_cast<float>(q) - static_cast<float>(zp)) * scale;
}

int32_t qmin_for(DataType dt) { return dt == DataType::u8 ? 0 : -128; }
int32_t qmax_for(DataType dt) { return dt == DataType::u8 ? 255 : 127; }

/// Quantize one f32 value the way the arch kernel does: round to nearest, then
/// clamp to the target dtype's range.
int32_t requant_one(float v, float scale, int32_t zp, DataType dt) {
    const int32_t q = static_cast<int32_t>(std::nearbyintf(v / scale)) + zp;
    return std::min(std::max(q, qmin_for(dt)), qmax_for(dt));
}

/// Element `idx` of an s8/u8 buffer, read as a signed integer (both dtypes are
/// stored one byte per element).
int64_t raw_at(const uint8_t* bytes, int64_t idx, DataType dt) {
    if (dt == DataType::u8) { return static_cast<int64_t>(bytes[idx]); }
    return static_cast<int64_t>(reinterpret_cast<const int8_t*>(bytes)[idx]);
}

/// Quantize a contiguous f32 [rows, n] buffer with `qp`'s row params.
std::vector<uint8_t> quantize_rows(const float* src, int64_t rows, int64_t n,
                                   const QuantParams& qp, DataType dt) {
    std::vector<uint8_t> out(static_cast<size_t>(rows * n));
    for (int64_t r = 0; r < rows; ++r) {
        float s; int32_t z;
        row_qparam(qp, r, s, z);
        for (int64_t i = 0; i < n; ++i) {
            const int64_t idx = r * n + i;
            out[static_cast<size_t>(idx)] =
                static_cast<uint8_t>(requant_one(src[idx], s, z, dt));
        }
    }
    return out;
}

/// Run the quantized path once and check it against "dequantize by hand ->
/// reference norm -> requantize by hand".
///
/// `attrs` carries the norm type / axis / output_dtype; `x_q` is the s8/u8
/// input; `extra_inputs` holds [scale] or [scale, bias], f32. `num_rows` is the
/// norm row count (and the PerToken param count); the row width is derived.
void expect_quant_norm(const NormAttributes& attrs,
                       const TensorView& x_q,
                       std::span<const TensorView> extra_inputs,
                       const QuantParams& y_qp,
                       int64_t num_rows,
                       float tol,
                       const ComputeContext& ctx = {})
{
    const DataType   x_dt = x_q.data_type();
    const DataType   y_dt = attrs.output_dtype;
    const auto&      x_qp = x_q.quant_params();
    const uint8_t*   x_bytes = x_q.ptr<uint8_t>();
    const int64_t    n = x_q.numel() / num_rows;
    const auto       shape = x_q.shape_span();

    // ---- expected: hand-dequantized f32 through the f32 reference ----
    std::vector<float> x_f32(static_cast<size_t>(num_rows * n));
    for (int64_t r = 0; r < num_rows; ++r) {
        float s; int32_t z;
        row_qparam(x_qp, r, s, z);
        for (int64_t i = 0; i < n; ++i) {
            const int64_t idx = r * n + i;
            x_f32[static_cast<size_t>(idx)] = dequant_one(raw_at(x_bytes, idx, x_dt), s, z);
        }
    }

    TensorView x_ref(shape, DataType::f32, x_f32.data(), x_q.layout());
    std::vector<TensorView> ref_ins;
    ref_ins.push_back(x_ref);
    for (const auto& t : extra_inputs) { ref_ins.push_back(t); }

    std::vector<float> want(static_cast<size_t>(num_rows * n));
    TensorView want_view(shape, DataType::f32, want.data(), x_q.layout());
    nnops::backend::cpu::reference::norm_ref(
        attrs, want_view, std::span<const TensorView>(ref_ins), {}, nullptr);

    // ---- the op under test ----
    auto op = Norm::create(attrs, Backend::CPU);

    std::vector<TensorDesc> in_descs;
    in_descs.push_back(x_q.desc());
    for (const auto& t : extra_inputs) { in_descs.push_back(t.desc()); }
    const auto out_desc = op->getOutputTensorDesc(std::span<const TensorDesc>(in_descs));
    NNOPS_EXPECT_EQ(static_cast<int>(out_desc[0].dtype), static_cast<int>(y_dt));

    const size_t numel = static_cast<size_t>(num_rows * n);
    std::vector<float> y_f32(y_dt == DataType::f32 ? numel : 0);
    std::vector<nnops::backend::cpu::half> y_f16(y_dt == DataType::f16 ? numel : 0);
    std::vector<uint8_t> y_bytes(is_quantized_dtype(y_dt) ? numel : 0);
    void* y_data = (y_dt == DataType::f32) ? static_cast<void*>(y_f32.data())
                 : (y_dt == DataType::f16) ? static_cast<void*>(y_f16.data())
                 : static_cast<void*>(y_bytes.data());

    TensorView out = make_q(shape, y_dt, y_data,
                            is_quantized_dtype(y_dt) ? y_qp : QuantParams{},
                            x_q.layout());

    std::vector<TensorView> ins;
    ins.push_back(x_q);
    for (const auto& t : extra_inputs) { ins.push_back(t); }
    op->compute(out, std::span<const TensorView>(ins), ctx);

    // ---- compare ----
    if (y_dt == DataType::f32) {
        for (size_t i = 0; i < numel; ++i) {
            NNOPS_EXPECT_NEAR(y_f32[i], want[i], tol);
        }
    } else if (y_dt == DataType::f16) {
        for (size_t i = 0; i < numel; ++i) {
            NNOPS_EXPECT_NEAR(simd::s_load(&y_f16[i]), want[i], tol);
        }
    } else {
        // Integer output: compare in quantized units, allowing one LSB for
        // rounding-mode / reassociation differences.
        for (int64_t r = 0; r < num_rows; ++r) {
            float s; int32_t z;
            row_qparam(y_qp, r, s, z);
            for (int64_t i = 0; i < n; ++i) {
                const int64_t idx = r * n + i;
                const int64_t want_q = requant_one(want[static_cast<size_t>(idx)], s, z, y_dt);
                NNOPS_EXPECT_TRUE(std::abs(raw_at(y_bytes.data(), idx, y_dt) - want_q) <= 1);
            }
        }
    }
}

/// Sweep every supported combination — {PerTensor, PerToken} input params x
/// {f32, f16, s8} output dtype — for one norm type. The integer output reuses
/// the input's granularity, so both output-param paths are exercised.
void sweep_quant_norm(NormType type, bool with_bias, const ComputeContext& ctx = {})
{
    constexpr int64_t kRows = 4;
    constexpr int64_t kDim  = 12;

    auto [x_data, _x] = test::make_random_tensor({kRows, kDim}, -1.0f, 1.0f, 900);
    auto [s_data, s_view] = test::make_random_tensor({kDim}, 0.25f, 2.0f, 901);
    auto [b_data, b_view] = test::make_random_tensor({kDim}, -0.5f, 0.5f, 902);
    (void)_x; (void)s_data; (void)b_data;

    // PerTensor covers the input range exactly; the PerToken scales vary a
    // little per row (and carry non-zero zero points) so the row mapping is
    // observable.
    const QuantParams x_qp_pt = per_tensor(1.0f / 127.0f, 0);
    const float row_scale[] = {0.0079f, 0.0081f, 0.0077f, 0.0083f};
    const int32_t row_zp[]  = {0, -3, 5, -8};
    const QuantParams x_qp_pr = per_token(row_scale, row_zp, kRows);

    const QuantParams y_qp_pt = per_tensor(1.0f / 64.0f, 0);
    const float out_scale[] = {1.0f / 64.0f, 1.0f / 60.0f, 1.0f / 68.0f, 1.0f / 62.0f};
    const int32_t out_zp[]  = {0, 0, 0, 0};
    const QuantParams y_qp_pr = per_token(out_scale, out_zp, kRows);

    const int64_t shape[] = {kRows, kDim};

    for (int g = 0; g < 2; ++g) {
        const bool per_row = (g == 1);
        const QuantParams& x_qp = per_row ? x_qp_pr : x_qp_pt;
        const QuantParams& y_qp = per_row ? y_qp_pr : y_qp_pt;

        std::vector<uint8_t> x_bytes =
            quantize_rows(x_data.data(), kRows, kDim, x_qp, DataType::s8);
        TensorView x_q = make_q(shape, DataType::s8, x_bytes.data(), x_qp);

        std::vector<TensorView> extras;
        if (type != NormType::L2Norm) {
            extras.push_back(s_view);
            if (with_bias) { extras.push_back(b_view); }
        }

        for (DataType y_dt : {DataType::f32, DataType::f16, DataType::s8}) {
            NormAttributes attrs;
            attrs.type = type;
            attrs.axis = -1;
            attrs.output_dtype = y_dt;
            const float tol = (y_dt == DataType::f16) ? 1e-2f : 1e-3f;
            expect_quant_norm(attrs, x_q, std::span<const TensorView>(extras), y_qp,
                              kRows, tol, ctx);
        }
    }
}

}  // anonymous namespace

NNOPS_TEST(norm_quant_layernorm_all_combos) {
    sweep_quant_norm(NormType::LayerNorm, /*with_bias=*/true);
}

NNOPS_TEST(norm_quant_rmsnorm_all_combos) {
    sweep_quant_norm(NormType::RMSNorm, /*with_bias=*/false);
}

NNOPS_TEST(norm_quant_l2norm_all_combos) {
    sweep_quant_norm(NormType::L2Norm, /*with_bias=*/false);
}

NNOPS_TEST(norm_quant_threaded_matches_serial) {
    // The per-row staging buffer comes from the memory pool, so the parallel
    // path must produce the same values as the serial one.
    SimplePool pool(4);
    ComputeContext ctx;
    ctx.cpu = pool.cpu;
    sweep_quant_norm(NormType::LayerNorm, /*with_bias=*/true, ctx);
}

NNOPS_TEST(norm_quant_3d_per_token_no_zero_point) {
    // 3D [B, S, H]: PerToken row r maps to param r, and a null zero_point
    // buffer means zero_point == 0.
    constexpr int64_t kRows = 6;
    constexpr int64_t kDim  = 8;
    const int64_t shape[] = {2, 3, kDim};

    auto [x_data, _x] = test::make_random_tensor({2, 3, kDim}, -1.0f, 1.0f, 910);
    auto [s_data, s_view] = test::make_random_tensor({kDim}, 0.5f, 1.5f, 911);
    (void)_x; (void)s_data;

    const float row_scale[] = {0.0075f, 0.0078f, 0.0082f, 0.0073f, 0.0080f, 0.0076f};
    const QuantParams x_qp = per_token(row_scale, nullptr, kRows);
    std::vector<uint8_t> x_bytes =
        quantize_rows(x_data.data(), kRows, kDim, x_qp, DataType::s8);
    TensorView x_q = make_q(shape, DataType::s8, x_bytes.data(), x_qp);

    NormAttributes attrs;
    attrs.type = NormType::LayerNorm;   // no bias -> the HasBias=false instantiation
    attrs.axis = -1;
    const TensorView extras[] = {s_view};

    expect_quant_norm(attrs, x_q, std::span<const TensorView>(extras, 1),
                      QuantParams{}, kRows, 1e-3f);
}

NNOPS_TEST(norm_quant_u8_input) {
    // u8 with a non-zero zero_point: the dequantize/reference mapping flips to
    // the unsigned convention.
    constexpr int64_t kRows = 3;
    constexpr int64_t kDim  = 10;
    const int64_t shape[] = {kRows, kDim};

    auto [x_data, _x] = test::make_random_tensor({kRows, kDim}, -0.5f, 1.5f, 920);
    (void)_x;

    const QuantParams x_qp = per_tensor(2.0f / 255.0f, 128);
    std::vector<uint8_t> x_bytes =
        quantize_rows(x_data.data(), kRows, kDim, x_qp, DataType::u8);
    TensorView x_q = make_q(shape, DataType::u8, x_bytes.data(), x_qp);

    NormAttributes attrs;
    attrs.type = NormType::L2Norm;
    attrs.axis = -1;

    expect_quant_norm(attrs, x_q, std::span<const TensorView>(), QuantParams{},
                      kRows, 1e-3f);
}

NNOPS_TEST(norm_quant_output_desc_follows_output_dtype) {
    // A quantized X makes the output follow attrs.output_dtype; a float X still
    // follows X.
    const int64_t shape[] = {2, 4};
    uint8_t x_bytes[8] = {};
    const int64_t s_shape[] = {4};
    float s_data[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    TensorView s(s_shape, DataType::f32, s_data);

    const QuantParams x_qp = per_tensor(0.01f, 0);
    TensorView x_q = make_q(shape, DataType::s8, x_bytes, x_qp);

    const TensorDesc in_descs[] = {x_q.desc(), s.desc()};

    NormAttributes attrs;
    attrs.type = NormType::LayerNorm;
    auto op = Norm::create(attrs, Backend::CPU);
    NNOPS_EXPECT_EQ(static_cast<int>(op->getOutputTensorDesc(in_descs)[0].dtype),
                    static_cast<int>(DataType::f32));  // default

    attrs.output_dtype = DataType::s8;
    auto op_s8 = Norm::create(attrs, Backend::CPU);
    NNOPS_EXPECT_EQ(static_cast<int>(op_s8->getOutputTensorDesc(in_descs)[0].dtype),
                    static_cast<int>(DataType::s8));

    attrs.output_dtype = DataType::f16;
    auto op_f16 = Norm::create(attrs, Backend::CPU);
    NNOPS_EXPECT_EQ(static_cast<int>(op_f16->getOutputTensorDesc(in_descs)[0].dtype),
                    static_cast<int>(DataType::f16));

    // Float X is unaffected by output_dtype.
    float x_f32[8] = {};
    TensorView x_float(shape, DataType::f32, x_f32);
    const TensorDesc f_descs[] = {x_float.desc(), s.desc()};
    NNOPS_EXPECT_EQ(static_cast<int>(op_s8->getOutputTensorDesc(f_descs)[0].dtype),
                    static_cast<int>(DataType::f32));
}

// ============================================================
// OpType / Backend / default-attributes checks
// ============================================================

NNOPS_TEST(norm_create_optype_backend) {
    NormAttributes attrs;
    attrs.type = NormType::LayerNorm;
    auto op = Norm::create(attrs, Backend::CPU);
    NNOPS_EXPECT_EQ(static_cast<int>(op->getOpType()), static_cast<int>(OpType::Norm));
    NNOPS_EXPECT_EQ(static_cast<int>(op->getBackend()), static_cast<int>(Backend::CPU));
}

NNOPS_TEST(norm_default_attrs) {
    NormAttributes attrs;
    NNOPS_EXPECT_EQ(static_cast<int>(attrs.type), static_cast<int>(NormType::LayerNorm));
    NNOPS_EXPECT_EQ(attrs.axis, int64_t(-1));
    NNOPS_EXPECT_EQ(attrs.num_groups, int64_t(1));
    NNOPS_EXPECT_NEAR(attrs.epsilon, 1e-5f, 1e-9f);
    NNOPS_EXPECT_EQ(attrs.spatial, true);
    NNOPS_EXPECT_EQ(attrs.add_to, false);
    NNOPS_EXPECT_EQ(static_cast<int>(attrs.output_dtype), static_cast<int>(DataType::f32));
}
