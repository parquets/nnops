/// @file test_slice.cpp
/// @brief Slice operator tests (CPU, planar, f32/f16).
///
/// Covers 1-D..4-D, multi-axis, negative axes, step > 1, empty axes (identity),
/// f16, and descriptor-only shape inference. Slice is a strided sub-tensor
/// extraction: unlisted axes are kept whole, and a listed axis maps output index
/// i to input index start + i*step, with output dim ceil((end - start) / step).
///
/// There is no slice_ref, so the randomized cases check each output element
/// against a hand-computed index into the source buffer; the small cases assert
/// literals. The innermost step is its own code path: the kernel memcpys a row
/// when it is 1 and takes the scalar strided loop otherwise. f16 values are
/// converted from an f32 source, hence the looser 0.01 tolerance.

#include "nnops/ops/slice.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/test_helpers.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <vector>
#include <algorithm>
#include <cstring>

using namespace nnops;

NNOPS_TEST(slice_1d_basic) {
    float in_data[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    const int64_t in_shape[] = {8};
    TensorView input(in_shape, DataType::f32, in_data);

    SliceAttributes attrs;
    const int64_t starts[] = {2};
    const int64_t ends[]   = {6};
    const int64_t axes[]   = {0};
    attrs.starts = std::span<const int64_t>(starts, 1);
    attrs.ends   = std::span<const int64_t>(ends, 1);
    attrs.axes   = std::span<const int64_t>(axes, 1);

    auto op = Slice::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 4);

    std::vector<float> out_buf(4);
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], 2.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], 3.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[2], 4.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[3], 5.0f, 1e-5f);
}

NNOPS_TEST(slice_1d_with_step) {
    float in_data[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    const int64_t in_shape[] = {8};
    TensorView input(in_shape, DataType::f32, in_data);

    SliceAttributes attrs;
    const int64_t starts[] = {0};
    const int64_t ends[]   = {8};
    const int64_t axes[]   = {0};
    const int64_t steps[]  = {2};
    attrs.starts = std::span<const int64_t>(starts, 1);
    attrs.ends   = std::span<const int64_t>(ends, 1);
    attrs.axes   = std::span<const int64_t>(axes, 1);
    attrs.steps  = std::span<const int64_t>(steps, 1);

    auto op = Slice::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].dims[0], 4);

    std::vector<float> out_buf(4);
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], 0.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], 2.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[2], 4.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[3], 6.0f, 1e-5f);
}

NNOPS_TEST(slice_2d_rows) {
    float in_data[] = {
        0.0f, 1.0f, 2.0f,
        3.0f, 4.0f, 5.0f,
        6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f,
    };
    const int64_t in_shape[] = {4, 3};
    TensorView input(in_shape, DataType::f32, in_data);

    SliceAttributes attrs;
    const int64_t starts[] = {1};
    const int64_t ends[]   = {3};
    const int64_t axes[]   = {0};
    attrs.starts = std::span<const int64_t>(starts, 1);
    attrs.ends   = std::span<const int64_t>(ends, 1);
    attrs.axes   = std::span<const int64_t>(axes, 1);

    auto op = Slice::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 3);

    std::vector<float> out_buf(6);
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    // Row 0 = input row 1: [3, 4, 5]
    NNOPS_EXPECT_NEAR(out_buf[0], 3.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], 4.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[2], 5.0f, 1e-5f);
    // Row 1 = input row 2: [6, 7, 8]
    NNOPS_EXPECT_NEAR(out_buf[3], 6.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[4], 7.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[5], 8.0f, 1e-5f);
}

NNOPS_TEST(slice_2d_cols_strided) {
    const int64_t in_shape[] = {4, 6};
    auto [in_vec, input] = test::make_random_tensor(in_shape, -1.0f, 1.0f, 222);

    SliceAttributes attrs;
    const int64_t starts[] = {0};
    const int64_t ends[]   = {6};
    const int64_t axes[]   = {1};
    const int64_t steps[]  = {2};
    attrs.starts = std::span<const int64_t>(starts, 1);
    attrs.ends   = std::span<const int64_t>(ends, 1);
    attrs.axes   = std::span<const int64_t>(axes, 1);
    attrs.steps  = std::span<const int64_t>(steps, 1);

    auto op = Slice::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 4);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 3);

    std::vector<float> out_buf(12);
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    // Verify: output[r, c] == input[r, c*2]
    for (int64_t r = 0; r < 4; ++r) {
        for (int64_t c = 0; c < 3; ++c) {
            float expected = in_vec[static_cast<size_t>(r * 6 + c * 2)];
            float result   = out_buf[static_cast<size_t>(r * 3 + c)];
            NNOPS_EXPECT_NEAR(result, expected, 1e-5f);
        }
    }
}

NNOPS_TEST(slice_3d_multi_axis) {
    const int64_t in_shape[] = {4, 6, 8};
    auto [in_vec, input] = test::make_random_tensor(in_shape, -1.0f, 1.0f, 333);

    SliceAttributes attrs;
    const int64_t starts[] = {1, 2};
    const int64_t ends[]   = {3, 6};
    const int64_t axes[]   = {0, 2};
    attrs.starts = std::span<const int64_t>(starts, 2);
    attrs.ends   = std::span<const int64_t>(ends, 2);
    attrs.axes   = std::span<const int64_t>(axes, 2);

    auto op = Slice::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 3);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 6);
    NNOPS_EXPECT_EQ(descs[0].dims[2], 4);

    std::vector<float> out_buf(48);  // 2 * 6 * 4
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    // Verify: output[i, j, k] == input[i+1, j, k+2]
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 6; ++j) {
            for (int64_t k = 0; k < 4; ++k) {
                float expected = in_vec[static_cast<size_t>((i + 1) * 48 + j * 8 + (k + 2))];
                float result   = out_buf[static_cast<size_t>(i * 24 + j * 4 + k)];
                NNOPS_EXPECT_NEAR(result, expected, 1e-5f);
            }
        }
    }
}

NNOPS_TEST(slice_identity) {
    const int64_t in_shape[] = {3, 4, 5};
    auto [in_vec, input] = test::make_random_tensor(in_shape, -1.0f, 1.0f, 444);

    SliceAttributes attrs;
    // Empty axes → identity

    auto op = Slice::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 3);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 4);
    NNOPS_EXPECT_EQ(descs[0].dims[2], 5);

    std::vector<float> out_buf(60);
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    for (size_t i = 0; i < 60; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], in_vec[i], 1e-5f);
    }
}

NNOPS_TEST(slice_negative_axes) {
    // -1 addresses the last axis
    const int64_t in_shape[] = {2, 3, 4};
    auto [in_vec, input] = test::make_random_tensor(in_shape, -1.0f, 1.0f, 555);

    SliceAttributes attrs;
    const int64_t starts[] = {1};
    const int64_t ends[]   = {3};
    const int64_t axes[]   = {-1};
    attrs.starts = std::span<const int64_t>(starts, 1);
    attrs.ends   = std::span<const int64_t>(ends, 1);
    attrs.axes   = std::span<const int64_t>(axes, 1);

    auto op = Slice::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].dims[2], 2);

    std::vector<float> out_buf(12);  // 2 * 3 * 2
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    // output[b, r, c] == input[b, r, c+1]
    for (int64_t b = 0; b < 2; ++b) {
        for (int64_t r = 0; r < 3; ++r) {
            for (int64_t c = 0; c < 2; ++c) {
                float expected = in_vec[static_cast<size_t>(b * 12 + r * 4 + (c + 1))];
                float result   = out_buf[static_cast<size_t>(b * 6 + r * 2 + c)];
                NNOPS_EXPECT_NEAR(result, expected, 1e-5f);
            }
        }
    }
}

NNOPS_TEST(slice_random_f32) {
    const int64_t in_shape[] = {4, 8, 12, 6};
    auto [in_vec, input] = test::make_random_tensor(in_shape, -1.0f, 1.0f, 666);

    SliceAttributes attrs;
    const int64_t starts[] = {1, 2, 3};
    const int64_t ends[]   = {3, 8, 6};
    const int64_t axes[]   = {0, 1, 2};
    attrs.starts = std::span<const int64_t>(starts, 3);
    attrs.ends   = std::span<const int64_t>(ends, 3);
    attrs.axes   = std::span<const int64_t>(axes, 3);

    auto op = Slice::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 6);
    NNOPS_EXPECT_EQ(descs[0].dims[2], 3);
    NNOPS_EXPECT_EQ(descs[0].dims[3], 6);

    int64_t numel = 2 * 6 * 3 * 6;
    std::vector<float> out_buf(static_cast<size_t>(numel));
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    // Verify: output[i,j,k,l] == input[i+1, j+2, k+3, l]
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 6; ++j) {
            for (int64_t k = 0; k < 3; ++k) {
                for (int64_t l = 0; l < 6; ++l) {
                    int64_t in_idx = ((i + 1) * 8 * 12 * 6) + ((j + 2) * 12 * 6) + ((k + 3) * 6) + l;
                    int64_t out_idx = (i * 6 * 3 * 6) + (j * 3 * 6) + (k * 6) + l;
                    NNOPS_EXPECT_NEAR(out_buf[static_cast<size_t>(out_idx)],
                                      in_vec[static_cast<size_t>(in_idx)], 1e-5f);
                }
            }
        }
    }
}

NNOPS_TEST(slice_random_f16) {
    const int64_t in_shape[] = {3, 8, 4};
    auto [f32_vec, input_f32] = test::make_random_tensor(in_shape, -1.0f, 1.0f, 777);

    const int64_t numel = 3 * 8 * 4;
    std::vector<nnops::backend::cpu::half> in_half(static_cast<size_t>(numel));
    for (int64_t i = 0; i < numel; ++i) {
        simd::s_store(&in_half[static_cast<size_t>(i)], f32_vec[static_cast<size_t>(i)]);
    }

    TensorView input(in_shape, DataType::f16, in_half.data());

    SliceAttributes attrs;
    const int64_t starts[] = {0};
    const int64_t ends[]   = {2};
    const int64_t axes[]   = {0};
    attrs.starts = std::span<const int64_t>(starts, 1);
    attrs.ends   = std::span<const int64_t>(ends, 1);
    attrs.axes   = std::span<const int64_t>(axes, 1);

    auto op = Slice::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 8);
    NNOPS_EXPECT_EQ(descs[0].dims[2], 4);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

    int64_t out_numel = 2 * 8 * 4;
    std::vector<nnops::backend::cpu::half> out_buf(static_cast<size_t>(out_numel));
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    // output[i,j,k] == input[i,j,k] for i in [0,1]
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 8; ++j) {
            for (int64_t k = 0; k < 4; ++k) {
                float expected = f32_vec[static_cast<size_t>(i * 32 + j * 4 + k)];
                float result   = simd::s_load(&out_buf[static_cast<size_t>(i * 32 + j * 4 + k)]);
                NNOPS_EXPECT_NEAR(result, expected, 0.01f);  // f16 tolerance
            }
        }
    }
}

NNOPS_TEST(slice_innermost_strided) {
    const int64_t in_shape[] = {2, 3, 8};
    auto [in_vec, input] = test::make_random_tensor(in_shape, -1.0f, 1.0f, 888);

    SliceAttributes attrs;
    const int64_t starts[] = {1};
    const int64_t ends[]   = {7};
    const int64_t axes[]   = {2};
    const int64_t steps[]  = {3};
    attrs.starts = std::span<const int64_t>(starts, 1);
    attrs.ends   = std::span<const int64_t>(ends, 1);
    attrs.axes   = std::span<const int64_t>(axes, 1);
    attrs.steps  = std::span<const int64_t>(steps, 1);

    auto op = Slice::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    // ceil((7-1)/3) = ceil(6/3) = 2
    NNOPS_EXPECT_EQ(descs[0].dims[2], 2);

    std::vector<float> out_buf(12);  // 2 * 3 * 2
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    // output[b, r, c] == input[b, r, 1 + c*3]
    for (int64_t b = 0; b < 2; ++b) {
        for (int64_t r = 0; r < 3; ++r) {
            for (int64_t c = 0; c < 2; ++c) {
                float expected = in_vec[static_cast<size_t>(b * 24 + r * 8 + (1 + c * 3))];
                float result   = out_buf[static_cast<size_t>(b * 6 + r * 2 + c)];
                NNOPS_EXPECT_NEAR(result, expected, 1e-5f);
            }
        }
    }
}

NNOPS_TEST(slice_shape_inference) {
    SliceAttributes attrs;
    const int64_t starts[] = {1, 0};
    const int64_t ends[]   = {3, 5};
    const int64_t axes[]   = {0, 2};
    attrs.starts = std::span<const int64_t>(starts, 2);
    attrs.ends   = std::span<const int64_t>(ends, 2);
    attrs.axes   = std::span<const int64_t>(axes, 2);

    auto op = Slice::create(attrs, Backend::CPU);

    TensorDesc in_desc;
    in_desc.layout = TensorLayout::NCHW;
    in_desc.dtype  = DataType::f32;
    in_desc.rank   = 3;
    in_desc.dims.resize(3);
    in_desc.dims[0] = 4;
    in_desc.dims[1] = 6;
    in_desc.dims[2] = 8;
    const TensorDesc in_arr[] = {in_desc};

    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 3);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 6);
    NNOPS_EXPECT_EQ(descs[0].dims[2], 5);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
}

NNOPS_TEST(slice_class_api) {
    const int64_t in_shape[] = {2, 3, 4};
    auto [in_vec, input] = test::make_random_tensor(in_shape, -1.0f, 1.0f, 999);

    SliceAttributes attrs;
    const int64_t starts[] = {0, 1};
    const int64_t ends[]   = {2, 3};
    const int64_t axes[]   = {0, 2};
    attrs.starts = std::span<const int64_t>(starts, 2);
    attrs.ends   = std::span<const int64_t>(ends, 2);
    attrs.axes   = std::span<const int64_t>(axes, 2);

    auto op = Slice::create(attrs, Backend::CPU);
    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    int64_t numel = 2 * 3 * 2;
    std::vector<float> out_class(static_cast<size_t>(numel));
    TensorView out_view1 = test::make_planar(descs[0], out_class.data());
    const TensorView ins[] = {input};
    op->compute(out_view1, ins);

    // A second compute through the same op must reproduce the first exactly.
    std::vector<float> out_func(static_cast<size_t>(numel));
    TensorView out_view2 = test::make_planar(descs[0], out_func.data());
    op->compute(out_view2, ins);

    for (size_t i = 0; i < static_cast<size_t>(numel); ++i) {
        NNOPS_EXPECT_NEAR(out_class[i], out_func[i], 1e-5f);
    }
}
