/// Unit tests for Pooling operator (CPU reference) — 2D and 3D.

#include "nnops/ops/pooling.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

// ============================================================
// 2D Pooling tests
// ============================================================

NNOPS_TEST(pooling_2d_max_basic) {
    // 1x1x4x4 input, 2x2 kernel, stride=2, pad=0
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t oshape[] = {1, 1, 2, 2};
    float in_data[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    };
    float out_data[4] = {};

    TensorView input(ishape, DataType::f32, in_data);
    TensorView output(oshape, DataType::f32, out_data);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};  // KD=1, KH=2, KW=2
    attrs.stride       = {1, 2, 2};
    attrs.padding      = {0, 0, 0};

    pooling(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 6.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 8.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 14.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[3], 16.0f, 1e-6f);
}

NNOPS_TEST(pooling_2d_average_basic) {
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t oshape[] = {1, 1, 2, 2};
    float in_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out_data[4] = {};

    TensorView input(ishape, DataType::f32, in_data);
    TensorView output(oshape, DataType::f32, out_data);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};

    pooling(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 10.0f / 9.0f, 1e-4f);
}

NNOPS_TEST(pooling_2d_average_exclude_pad) {
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t oshape[] = {1, 1, 2, 2};
    float in_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out_data[4] = {};

    TensorView input(ishape, DataType::f32, in_data);
    TensorView output(oshape, DataType::f32, out_data);

    PoolingAttributes attrs;
    attrs.type = PoolingType::AverageExcludePad;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};

    pooling(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 2.5f, 1e-4f);
}

NNOPS_TEST(pooling_2d_lp) {
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t oshape[] = {1, 1, 1, 1};
    float in_data[4] = {3.0f, 4.0f, 0.0f, 0.0f};
    float out_data[1] = {};

    TensorView input(ishape, DataType::f32, in_data);
    TensorView output(oshape, DataType::f32, out_data);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Lp;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};
    attrs.padding      = {0, 0, 0};
    attrs.p_norm = 2;

    pooling(input, output, attrs);

    NNOPS_EXPECT_NEAR(out_data[0], 5.0f, 1e-4f);
}

NNOPS_TEST(pooling_2d_random) {
    auto [in_vec, input] = test::make_random_tensor({1, 3, 16, 16});
    std::vector<float> out_buf(1 * 3 * 8 * 8);
    const int64_t oshape[] = {1, 3, 8, 8};
    TensorView output(oshape, DataType::f32, out_buf.data());

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};

    pooling(input, output, attrs);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

NNOPS_TEST(pooling_2d_class_api) {
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t oshape[] = {1, 1, 2, 2};
    float in_data[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float out1_data[4] = {};
    float out2_data[4] = {};

    TensorView input(ishape, DataType::f32, in_data);
    TensorView out1(oshape, DataType::f32, out1_data);
    TensorView out2(oshape, DataType::f32, out2_data);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};

    // Functional
    pooling(input, out1, attrs);
    // Class
    auto op = Pooling::create(attrs, Backend::CPU);
    const TensorView ins[] = {input};
    op->compute(out2, ins);

    NNOPS_EXPECT_TRUE(test::allclose(out1, out2));
}

// ============================================================
// 3D Pooling tests
// ============================================================

NNOPS_TEST(pooling_3d_max_basic) {
    // 1x1x2x4x4 input, 2x2x2 kernel, stride=2, pad=0
    // Output: 1x1x1x2x2
    const int64_t ishape[] = {1, 1, 2, 4, 4};
    const int64_t oshape[] = {1, 1, 1, 2, 2};

    // Two depth slices, each 4x4
    std::vector<float> in_buf(2 * 4 * 4);
    for (int i = 0; i < 32; ++i) in_buf[i] = static_cast<float>(i + 1);
    std::vector<float> out_buf(4, 0.0f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView output(oshape, DataType::f32, out_buf.data());

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {2, 2, 2};   // KD=2, KH=2, KW=2
    attrs.stride       = {2, 2, 2};
    attrs.padding      = {0, 0, 0};

    pooling(input, output, attrs);

    // Output[0,0]: max of depth[0,0:2,0:2] + depth[1,0:2,0:2]
    // depth0[0:2,0:2] = {1,2,5,6}, depth1[0:2,0:2] = {17,18,21,22}, max = 22
    NNOPS_EXPECT_NEAR(out_buf[0], 22.0f, 1e-6f);
    // Output[0,1]: max of depth0[0:2,2:4] + depth1[0:2,2:4]
    // depth0 = {3,4,7,8}, depth1 = {19,20,23,24}, max = 24
    NNOPS_EXPECT_NEAR(out_buf[1], 24.0f, 1e-6f);
    // Output[1,0]: max from rows 2:4 of both depths
    // depth0 rows 2:4 cols 0:2 = {9,10,13,14}, depth1 = {25,26,29,30}, max = 30
    NNOPS_EXPECT_NEAR(out_buf[2], 30.0f, 1e-6f);
    // Output[1,1]: max from rows 2:4 cols 2:4 of both depths
    NNOPS_EXPECT_NEAR(out_buf[3], 32.0f, 1e-6f);
}

NNOPS_TEST(pooling_3d_average) {
    // 1x1x2x2x2 input, 2x2x2 kernel, stride=1, pad=0 -> output 1x1x1x1x1
    const int64_t ishape[] = {1, 1, 2, 2, 2};
    const int64_t oshape[] = {1, 1, 1, 1, 1};
    float in_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float out_data[1] = {};

    TensorView input(ishape, DataType::f32, in_data);
    TensorView output(oshape, DataType::f32, out_data);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {2, 2, 2};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 0, 0};

    pooling(input, output, attrs);

    // Average of all 8 elements = (1+2+3+4+5+6+7+8)/8 = 4.5
    NNOPS_EXPECT_NEAR(out_data[0], 4.5f, 1e-4f);
}

NNOPS_TEST(pooling_3d_random) {
    auto [in_vec, input] = test::make_random_tensor({1, 2, 8, 8, 8});
    std::vector<float> out_buf(1 * 2 * 4 * 4 * 4);
    const int64_t oshape[] = {1, 2, 4, 4, 4};
    TensorView output(oshape, DataType::f32, out_buf.data());

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {2, 2, 2};
    attrs.stride       = {2, 2, 2};

    pooling(input, output, attrs);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}
