/// @file test_slice.cpp
/// @brief Unit tests for Slice operator (CPU backend).

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

// ============================================================
// Hand-verified tests
// ============================================================

NNOPS_TEST(slice_1d_basic) {
    // 1D slice: [0, 1, 2, 3, 4, 5, 6, 7] → [2, 3, 4, 5] (start=2, end=6)
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
    // 1D slice with step=2: [0,1,2,3,4,5,6,7] start=0 end=8 step=2 → [0,2,4,6]
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
    // 2D slice on dim 0: [4, 3] → [2, 3] (rows 1..2)
    // Input (row-major):
    //   [0, 1, 2]
    //   [3, 4, 5]
    //   [6, 7, 8]
    //   [9,10,11]
    // Output: rows 1,2 (start=1, end=3 on axis 0)
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
    // 2D slice on dim 1 with step=2: [4, 6] → [4, 3] (cols 0,2,4)
    // This tests the strided inner-dim path.
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
    // 3D: [4, 6, 8] → slice on dims 0 and 2
    // dim 0: start=1, end=3 → 2 elements
    // dim 2: start=2, end=6 → 4 elements (step=1 → memcpy path)
    // Output: [2, 6, 4]
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
    // Identity slice: output == input
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
    // Negative axis: -1 means last dim
    // 3D [2, 3, 4], slice axis=-1 with start=1, end=3 → [2, 3, 2]
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
    // 4D slice with multiple axes, random verification vs reference algorithm
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
    // f16 random slice
    const int64_t in_shape[] = {3, 8, 4};
    auto [f32_vec, input_f32] = test::make_random_tensor(in_shape, -1.0f, 1.0f, 777);

    // Convert to f16
    const int64_t numel = 3 * 8 * 4;
    std::vector<nnops::backend::cpu::half> in_half(static_cast<size_t>(numel));
    for (int64_t i = 0; i < numel; ++i)
        simd::s_store(&in_half[static_cast<size_t>(i)], f32_vec[static_cast<size_t>(i)]);

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
    // Slice on innermost dimension with step > 1 → strided inner path
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
    // Verify shape inference
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

NNOPS_TEST(slice_functional_api) {
    // Verify functional API produces same result as class API
    const int64_t in_shape[] = {2, 3, 4};
    auto [in_vec, input] = test::make_random_tensor(in_shape, -1.0f, 1.0f, 999);

    SliceAttributes attrs;
    const int64_t starts[] = {0, 1};
    const int64_t ends[]   = {2, 3};
    const int64_t axes[]   = {0, 2};
    attrs.starts = std::span<const int64_t>(starts, 2);
    attrs.ends   = std::span<const int64_t>(ends, 2);
    attrs.axes   = std::span<const int64_t>(axes, 2);

    // Class API
    auto op = Slice::create(attrs, Backend::CPU);
    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    int64_t numel = 2 * 3 * 2;
    std::vector<float> out_class(static_cast<size_t>(numel));
    TensorView out_view1 = test::make_planar(descs[0], out_class.data());
    const TensorView ins[] = {input};
    op->compute(out_view1, ins);

    // Functional API
    std::vector<float> out_func(static_cast<size_t>(numel));
    TensorView out_view2 = test::make_planar(descs[0], out_func.data());
    slice(input, out_view2, attrs);

    for (size_t i = 0; i < static_cast<size_t>(numel); ++i) {
        NNOPS_EXPECT_NEAR(out_class[i], out_func[i], 1e-5f);
    }
}
