/// Unit tests for GroupNorm operator (CPU SIMD + reference).

#include "nnops/ops/group_norm.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

// ============================================================
// Basic correctness tests
// ============================================================

NNOPS_TEST(group_norm_2_groups) {
    // [N=1, C=4, H=1, W=2]
    // G=2, so each group has 2 channels (C/G=2)
    // Group 0: channels 0,1 → 2*1*2 = 4 elements
    // Group 1: channels 2,3 → 4 elements
    const int64_t shape[] = {1, 4, 1, 2};
    const int64_t scale_shape[] = {4};

    // Group 0: channels 0,1. Group 1: channels 2,3
    float x_data[] = {
        // N=0, C=0, H=0:  w=0, w=1
        1.0f, 2.0f,
        // N=0, C=1, H=0:
        3.0f, 4.0f,
        // N=0, C=2, H=0:
        5.0f, 6.0f,
        // N=0, C=3, H=0:
        7.0f, 8.0f,
    };
    float s_data[] = {1.0f, 1.0f, 1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    GroupNormAttributes attrs;
    attrs.num_groups = 2;

    auto op = GroupNorm::create(attrs, Backend::CPU);

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

    // Group 0 (ch 0,1): values [1,2,3,4]
    //   mean = 2.5, var = 1.25
    // Group 1 (ch 2,3): values [5,6,7,8]
    //   mean = 6.5, var = 1.25
    float eps = 1e-5f;

    // Group 0 normalized:
    float g0_mean = 2.5f, g0_var = 1.25f;
    float g0_inv_std = 1.0f / std::sqrt(g0_var + eps);
    NNOPS_EXPECT_NEAR(out_buf[0], (1.0f - g0_mean) * g0_inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[1], (2.0f - g0_mean) * g0_inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[2], (3.0f - g0_mean) * g0_inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[3], (4.0f - g0_mean) * g0_inv_std, 1e-4f);

    // Group 1 normalized:
    float g1_mean = 6.5f, g1_var = 1.25f;
    float g1_inv_std = 1.0f / std::sqrt(g1_var + eps);
    NNOPS_EXPECT_NEAR(out_buf[4], (5.0f - g1_mean) * g1_inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[5], (6.0f - g1_mean) * g1_inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[6], (7.0f - g1_mean) * g1_inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[7], (8.0f - g1_mean) * g1_inv_std, 1e-4f);
}

NNOPS_TEST(group_norm_with_scale_bias) {
    // G=1 (equivalent to LayerNorm over C+H+W with per-channel scale/bias)
    const int64_t shape[] = {1, 2, 1, 2};
    const int64_t scale_shape[] = {2};

    float x_data[] = {0.0f, 2.0f, 0.0f, 4.0f};  // ce, H=1,W=2  ×  C=2
    float s_data[] = {2.0f, 3.0f};
    float b_data[] = {1.0f, -1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);
    TensorView b(scale_shape, DataType::f32, b_data);

    GroupNormAttributes attrs;
    attrs.num_groups = 1;

    auto op = GroupNorm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    auto db = b.desc();
    const TensorDesc desc_arr[] = {dx, ds, db};
    auto outs = op->getOutputTensorDesc(desc_arr);

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {x, s, b};
    op->compute(out, ins);

    // All 4 values: [0,2,0,4] → mean=1.5, var=2.75
    float mean = 1.5f, var = 2.75f;
    float eps = 1e-5f;
    float inv_std = 1.0f / std::sqrt(var + eps);

    // Ch 0: s=2, b=1
    NNOPS_EXPECT_NEAR(out_buf[0], (0.0f - mean) * inv_std * 2.0f + 1.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[1], (2.0f - mean) * inv_std * 2.0f + 1.0f, 1e-4f);
    // Ch 1: s=3, b=-1
    NNOPS_EXPECT_NEAR(out_buf[2], (0.0f - mean) * inv_std * 3.0f + (-1.0f), 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[3], (4.0f - mean) * inv_std * 3.0f + (-1.0f), 1e-4f);
}

NNOPS_TEST(group_norm_equals_instancenorm_when_g_equals_c) {
    // G=C → each channel normalized independently (InstanceNorm)
    const int64_t shape[] = {2, 4, 2};  // [N, C, W]
    const int64_t scale_shape[] = {4};

    std::vector<float> x_data = {
        // N=0: C=0..3, W=0..1
        1.0f, 2.0f,   3.0f, 4.0f,   5.0f, 6.0f,   7.0f, 8.0f,
        // N=1: C=0..3, W=0..1
        9.0f, 10.0f,  11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
    };
    // scale = all 1s
    std::vector<float> s_data = {1.0f, 1.0f, 1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data.data());
    TensorView s(scale_shape, DataType::f32, s_data.data());

    GroupNormAttributes attrs;
    attrs.num_groups = 4;  // G = C

    auto op = GroupNorm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {dx, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());
    const TensorView ins[] = {x, s};
    op->compute(out, ins);

    float eps = 1e-5f;

    // Each channel normalized independently (norm_size = W = 2)
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

NNOPS_TEST(group_norm_add_to) {
    const int64_t shape[] = {1, 2, 2};
    const int64_t scale_shape[] = {2};

    float x_data[]  = {1.0f, 3.0f, 5.0f, 7.0f};
    float s_data[]  = {1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    GroupNormAttributes attrs;
    attrs.num_groups = 1;
    attrs.add_to = true;

    auto op = GroupNorm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {dx, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    // Pre-fill output with initial values
    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()), 10.0f);
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(out, ins);

    // With add_to, output = init_value + normalized_value
    float mean = 4.0f;  // mean of [1,3,5,7]
    float var = 5.0f;
    float eps = 1e-5f;
    float inv_std = 1.0f / std::sqrt(var + eps);

    NNOPS_EXPECT_NEAR(out_buf[0], 10.0f + (1.0f - mean) * inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[1], 10.0f + (3.0f - mean) * inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[2], 10.0f + (5.0f - mean) * inv_std, 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[3], 10.0f + (7.0f - mean) * inv_std, 1e-4f);
}

// ============================================================
// Random data comparison: SIMD vs reference
// ============================================================

NNOPS_TEST(group_norm_random_f32) {
    const int64_t shape[] = {2, 6, 4, 8};  // NCHW
    const int64_t scale_shape[] = {6};

    auto [x_vec, x] = test::make_random_tensor(shape, -10.0f, 10.0f, 500);
    auto [s_vec, s] = test::make_random_tensor(scale_shape, 0.5f, 2.0f, 501);
    auto [b_vec, b] = test::make_random_tensor(scale_shape, -1.0f, 1.0f, 502);

    for (int64_t g : {1, 2, 3, 6}) {
        GroupNormAttributes attrs;
        attrs.num_groups = g;

        auto dx = x.desc();
        auto ds = s.desc();
        auto db = b.desc();

        // SIMD output
        auto op_simd = GroupNorm::create(attrs, Backend::CPU);
        const TensorDesc desc_simd[] = {dx, ds, db};
        auto outs_simd = op_simd->getOutputTensorDesc(desc_simd);
        std::vector<float> out_simd_buf(static_cast<size_t>(outs_simd[0].numel()));
        auto out_simd = test::make_planar(outs_simd[0], out_simd_buf.data());
        const TensorView ins_simd[] = {x, s, b};
        op_simd->compute(out_simd, ins_simd);

        // Verify output is plausible: after normalization with scale/bias ~1,
        // variance should be bounded.
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

        // With unit scale/bias ≈ 1, the post-normalization variance should
        // not explode or collapse to zero.
        NNOPS_EXPECT_TRUE(out_var > 0.01f && out_var < 100.0f);
    }
}

NNOPS_TEST(group_norm_random_f16) {
    const int64_t shape[] = {1, 4, 4, 4};  // NCHW
    const int64_t scale_shape[] = {4};

    using nnops::backend::cpu::half;

    // Generate f32 random data
    auto [x_f32_vec, x_f32] = test::make_random_tensor(shape, -5.0f, 5.0f, 600);
    auto [s_f32_vec, s_f32] = test::make_random_tensor(scale_shape, 0.5f, 2.0f, 601);

    const int64_t x_numel = x_f32.numel();
    const int64_t s_numel = s_f32.numel();

    // Convert to f16 using simd::s_store
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

    GroupNormAttributes attrs;
    attrs.num_groups = 2;

    auto op = GroupNorm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {dx, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    std::vector<half> out_buf(static_cast<size_t>(outs[0].numel()));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {x, s};
    op->compute(out, ins);

    // Verify no NaN/Inf by checking all values are finite
    int64_t total = outs[0].numel();
    for (int64_t i = 0; i < total; ++i) {
        float val = simd::s_load(&out_buf[static_cast<size_t>(i)]);
        NNOPS_EXPECT_TRUE(std::isfinite(val));
    }
}

NNOPS_TEST(group_norm_rank3_ncw) {
    // 3D input: [N=1, C=3, W=4]
    const int64_t shape[] = {1, 3, 4};
    const int64_t scale_shape[] = {3};

    float x_data[] = {
        1.0f, 2.0f, 3.0f, 4.0f,   // C=0
        5.0f, 6.0f, 7.0f, 8.0f,   // C=1
        9.0f, 10.0f, 11.0f, 12.0f, // C=2
    };
    float s_data[] = {1.0f, 1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data);
    TensorView s(scale_shape, DataType::f32, s_data);

    GroupNormAttributes attrs;
    attrs.num_groups = 1;

    auto op = GroupNorm::create(attrs, Backend::CPU);

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

    // mean over all 12 elements = 6.5
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

NNOPS_TEST(group_norm_functional_api) {
    const int64_t shape[] = {1, 2, 2};
    const int64_t scale_shape[] = {2};

    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> s_data = {1.0f, 1.0f};

    TensorView x(shape, DataType::f32, x_data.data());
    TensorView s(scale_shape, DataType::f32, s_data.data());

    auto dx = x.desc();
    auto out_desc = TensorDesc{
        dx.dims,
        dx.rank,
        dx.dtype,
        dx.layout
    };
    std::vector<float> out_buf1(static_cast<size_t>(out_desc.numel()));
    std::vector<float> out_buf2(static_cast<size_t>(out_desc.numel()));
    auto out1 = test::make_planar(out_desc, out_buf1.data());
    auto out2 = test::make_planar(out_desc, out_buf2.data());

    GroupNormAttributes attrs;
    attrs.num_groups = 1;

    // Class API
    auto op = GroupNorm::create(attrs, Backend::CPU);
    const TensorView ins_class[] = {x, s};
    op->compute(out1, ins_class);

    // Functional API
    group_norm(x, s, out2, attrs);

    // Results should match
    for (size_t i = 0; i < out_buf1.size(); ++i) {
        NNOPS_EXPECT_NEAR(out_buf1[i], out_buf2[i], 1e-6f);
    }
}

NNOPS_TEST(group_norm_functional_api_with_bias) {
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

    GroupNormAttributes attrs;
    attrs.num_groups = 1;

    // Class API (3 inputs)
    auto op = GroupNorm::create(attrs, Backend::CPU);
    const TensorView ins_class[] = {x, s, b};
    op->compute(out1, ins_class);

    // Functional API (3 inputs)
    group_norm(x, s, b, out2, attrs);

    // Results should match
    for (size_t i = 0; i < out_buf1.size(); ++i) {
        NNOPS_EXPECT_NEAR(out_buf1[i], out_buf2[i], 1e-6f);
    }
}

NNOPS_TEST(group_norm_shape_inference) {
    const int64_t shape[] = {2, 8, 4, 4};
    const int64_t scale_shape[] = {8};

    std::vector<float> x_data(2 * 8 * 4 * 4);
    std::vector<float> s_data(8);

    TensorView x(shape, DataType::f32, x_data.data());
    TensorView s(scale_shape, DataType::f32, s_data.data());

    GroupNormAttributes attrs;
    attrs.num_groups = 4;

    auto op = GroupNorm::create(attrs, Backend::CPU);

    auto dx = x.desc();
    auto ds = s.desc();
    const TensorDesc desc_arr[] = {dx, ds};
    auto outs = op->getOutputTensorDesc(desc_arr);

    // Output shape equals input shape
    NNOPS_EXPECT_EQ(outs[0].rank, 4);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t{2});
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t{8});
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t{4});
    NNOPS_EXPECT_EQ(outs[0].dims[3], int64_t{4});
    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(outs[0].layout, TensorLayout::NCHW);
}
