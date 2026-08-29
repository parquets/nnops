/// Unit tests for Conv2D operator (CPU reference).
/// Uses Conv2D class API exclusively.

#include "nnops/ops/conv2d.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cstring>
#include <cstdint>

using namespace nnops;

// ============================================================
// Hand-verified small tests
// ============================================================

NNOPS_TEST(conv2d_basic_no_pad) {
    // 1x1x4x4 input all-ones, 1x1x3x3 kernel all-0.5, stride=1, pad=0
    // Expected: 2x2 output, each = 9 * (1.0 * 0.5) = 4.5
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t wshape[] = {1, 1, 3, 3};

    std::vector<float> in_buf(16, 1.0f);
    std::vector<float> w_buf(9, 0.5f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};

    auto op = Conv2D::create(attrs, Backend::CPU);

    const TensorDesc desc_arr[] = {input.desc(), weight.desc()};
    auto outs = op->getOutputTensorDesc(desc_arr);

    int64_t out_numel = 1;
    for (int64_t i = 0; i < outs[0].rank; ++i) {
        out_numel *= outs[0].dims[i];
    }
    std::vector<float> out_buf(static_cast<size_t>(out_numel), 0.0f);
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {input, weight};
    op->compute(out, ins);

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], 4.5f, 1e-4f);
    }
}

NNOPS_TEST(conv2d_with_bias) {
    // Same as above but with bias = 0.5
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t wshape[] = {1, 1, 3, 3};
    const int64_t bshape[] = {1};

    std::vector<float> in_buf(16, 1.0f);
    std::vector<float> w_buf(9, 0.5f);
    std::vector<float> b_buf = {0.5f};

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());
    TensorView bias(bshape, DataType::f32, b_buf.data());

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};

    auto op = Conv2D::create(attrs, Backend::CPU);

    const TensorDesc desc_arr[] = {input.desc(), weight.desc()};
    auto outs = op->getOutputTensorDesc(desc_arr);

    int64_t out_numel = 1;
    for (int64_t i = 0; i < outs[0].rank; ++i) {
        out_numel *= outs[0].dims[i];
    }
    std::vector<float> out_buf(static_cast<size_t>(out_numel), 0.0f);
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {input, weight, bias};
    op->compute(out, ins);

    // Each output = 4.5 + 0.5 = 5.0
    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], 5.0f, 1e-4f);
    }
}

NNOPS_TEST(conv2d_stride_2) {
    // 1x1x6x6 input, 1x1x3x3 kernel, stride=2, pad=0
    // Expected: 2x2 output
    const int64_t ishape[] = {1, 1, 6, 6};
    const int64_t wshape[] = {1, 1, 3, 3};

    std::vector<float> in_buf(36, 1.0f);
    std::vector<float> w_buf(9, 0.5f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {2, 2};
    attrs.padding = {0, 0};

    auto op = Conv2D::create(attrs, Backend::CPU);

    const TensorDesc desc_arr[] = {input.desc(), weight.desc()};
    auto outs = op->getOutputTensorDesc(desc_arr);

    int64_t out_numel = 1;
    for (int64_t i = 0; i < outs[0].rank; ++i) {
        out_numel *= outs[0].dims[i];
    }
    std::vector<float> out_buf(static_cast<size_t>(out_numel), 0.0f);
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {input, weight};
    op->compute(out, ins);

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], 4.5f, 1e-4f);
    }
}

NNOPS_TEST(conv2d_padding_1) {
    // 1x1x2x2 input, 1x1x3x3 kernel, stride=1, pad=1
    // Output should be 2x2 (same spatial size)
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t wshape[] = {1, 1, 3, 3};

    std::vector<float> in_buf(4, 1.0f);
    std::vector<float> w_buf(9, 0.5f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {1, 1};

    auto op = Conv2D::create(attrs, Backend::CPU);

    const TensorDesc desc_arr[] = {input.desc(), weight.desc()};
    auto outs = op->getOutputTensorDesc(desc_arr);

    int64_t out_numel = 1;
    for (int64_t i = 0; i < outs[0].rank; ++i) {
        out_numel *= outs[0].dims[i];
    }
    std::vector<float> out_buf(static_cast<size_t>(out_numel), 0.0f);
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {input, weight};
    op->compute(out, ins);

    // Each output element sees partial kernel region (padded zeros)
    // Top-left: only bottom-right 2x2 of kernel overlaps => 1.0 * 0.5 * 4 = 2.0
    NNOPS_EXPECT_NEAR(out_buf[0], 2.0f, 1e-4f);
    // Top-right: bottom-left 2x2 of kernel overlaps => 2.0
    NNOPS_EXPECT_NEAR(out_buf[1], 2.0f, 1e-4f);
}

NNOPS_TEST(conv2d_grouped) {
    // Grouped conv with groups=2, IC=2, OC=2
    // Each group: 1 input channel -> 1 output channel
    const int64_t ishape[] = {1, 2, 3, 3};
    const int64_t wshape[] = {2, 1, 2, 2};  // [OC, IC/G, KH, KW] = [2, 1, 2, 2]

    std::vector<float> in_buf(18, 1.0f);
    std::vector<float> w_buf(8, 1.0f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());

    Conv2DAttributes attrs;
    attrs.kernel_size = {2, 2};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};
    attrs.groups  = 2;

    auto op = Conv2D::create(attrs, Backend::CPU);

    const TensorDesc desc_arr[] = {input.desc(), weight.desc()};
    auto outs = op->getOutputTensorDesc(desc_arr);

    int64_t out_numel = 1;
    for (int64_t i = 0; i < outs[0].rank; ++i) {
        out_numel *= outs[0].dims[i];
    }
    std::vector<float> out_buf(static_cast<size_t>(out_numel), 0.0f);
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {input, weight};
    op->compute(out, ins);

    // Each group: 1 input channel convolved with 1 filter => 4 products * 1.0 = 4.0
    for (int i = 0; i < 8; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], 4.0f, 1e-4f);
    }
}

// ============================================================
// Random comparison test
// ============================================================

NNOPS_TEST(conv2d_random_small) {
    auto [in_vec, input]   = test::make_random_tensor({1, 3, 8, 8});
    auto [w_vec, weight]   = test::make_random_tensor({4, 3, 3, 3});

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};

    auto op = Conv2D::create(attrs, Backend::CPU);

    const TensorDesc desc_arr[] = {input.desc(), weight.desc()};
    auto outs = op->getOutputTensorDesc(desc_arr);

    int64_t out_numel = 1;
    for (int64_t i = 0; i < outs[0].rank; ++i) {
        out_numel *= outs[0].dims[i];
    }
    std::vector<float> out_buf(static_cast<size_t>(out_numel));
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {input, weight};
    op->compute(out, ins);

    // Should not crash or produce NaN
    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

// ============================================================
// Class API standalone test
// ============================================================

NNOPS_TEST(conv2d_class_api) {
    // Validate Conv2D class API with getOutputTensorDesc + make_planar pipeline.
    // 1x1x4x4 input all-ones, 1x1x3x3 kernel all-0.5, stride=1, pad=0
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t wshape[] = {1, 1, 3, 3};

    std::vector<float> in_buf(16, 1.0f);
    std::vector<float> w_buf(9, 0.5f);

    TensorView input(ishape, DataType::f32, in_buf.data());
    TensorView weight(wshape, DataType::f32, w_buf.data());

    Conv2DAttributes attrs;
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};

    auto op = Conv2D::create(attrs, Backend::CPU);

    const TensorDesc desc_arr[] = {input.desc(), weight.desc()};
    auto outs = op->getOutputTensorDesc(desc_arr);

    int64_t out_numel = 1;
    for (int64_t i = 0; i < outs[0].rank; ++i) {
        out_numel *= outs[0].dims[i];
    }
    std::vector<float> out_buf(static_cast<size_t>(out_numel), 0.0f);
    auto out = test::make_planar(outs[0], out_buf.data());

    const TensorView ins[] = {input, weight};
    op->compute(out, ins);

    // Expected: 2x2 output, each = 9 * (1.0 * 0.5) = 4.5
    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], 4.5f, 1e-4f);
    }
}
