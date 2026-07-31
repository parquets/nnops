/// @file test_grid_sample.cpp
/// @brief Tests for the GridSample operator — NCHWC8/NCDHWC8 only.
///
/// All tests use the NCHWC8 path:
///   NCHW in → pack_nchw_to_nchwc8 → grid_sample(NCHWC8, grid) → unpack → compare
///
/// Covers bilinear and nearest-neighbor interpolation with zeros, border,
/// and reflection padding modes.

#include "nnops/ops/grid_sample.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "backend/cpu/layout_convert.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

// Forward declarations
namespace nnops::backend::cpu {
    void grid_sample_cpu(const GridSampleAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         void* workspace);
}

namespace {

inline int64_t nchwc8_pitch(int64_t W, int64_t elem_size = 4) {
    return ((W * 8 * elem_size + 31) / 32) * 32;
}

}  // namespace

// ============================================================
// 2D Bilinear tests
// ============================================================

NNOPS_TEST(grid_sample_2d_bilinear_identity) {
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t oshape[] = {1, 1, 2, 2};
    const int64_t gshape[] = {1, 2, 2, 2};
    const int64_t C8 = 1, IH = 2, IW = 2, OH = 2, OW = 2;

    float in_data[4] = {1, 2, 3, 4};
    float grid_data[8] = {
        -0.5f, -0.5f,
        -0.5f,  0.5f,
         0.5f, -0.5f,
         0.5f,  0.5f,
    };

    TensorView in_nchw(ishape, DataType::f32, in_data, TensorLayout::NCHW);
    TensorView grid(gshape, DataType::f32, grid_data, TensorLayout::NCHW);

    // Pack to NCHWC8
    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OH * out_pitch / 4));
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    grid_sample(in_c8, grid, out_c8, {});

    // Unpack
    float result[4] = {};
    TensorView res_nchw(oshape, DataType::f32, result, TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    NNOPS_EXPECT_NEAR(result[0], 1.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(result[1], 2.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(result[2], 3.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(result[3], 4.0f, 1e-5f);
}

NNOPS_TEST(grid_sample_2d_bilinear_align_corners) {
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t oshape[] = {1, 1, 4, 4};
    const int64_t gshape[] = {1, 4, 4, 2};
    const int64_t C8 = 1, IH = 2, IW = 2, OH = 4, OW = 4;

    float in_data[4] = {1, 2, 3, 4};

    TensorView in_nchw(ishape, DataType::f32, in_data, TensorLayout::NCHW);

    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    std::vector<float> grid_buf(1 * 4 * 4 * 2);
    for (int64_t oh = 0; oh < 4; ++oh) {
        for (int64_t ow = 0; ow < 4; ++ow) {
            float gy = static_cast<float>(oh) / 3.0f * 2.0f - 1.0f;
            float gx = static_cast<float>(ow) / 3.0f * 2.0f - 1.0f;
            grid_buf[(oh * 4 + ow) * 2]     = gy;
            grid_buf[(oh * 4 + ow) * 2 + 1] = gx;
        }
    }

    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OH * out_pitch / 4));
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    GridSampleAttributes attrs;
    attrs.align_corners = true;
    grid_sample(in_c8, grid, out_c8, attrs);

    float result[16] = {};
    TensorView res_nchw(oshape, DataType::f32, result, TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    NNOPS_EXPECT_NEAR(result[0],  1.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(result[3],  2.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(result[12], 3.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(result[15], 4.0f, 1e-5f);

    for (size_t i = 0; i < 16; ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(result[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(result[i]));
    }
}

NNOPS_TEST(grid_sample_2d_bilinear_random) {
    auto [in_vec, in_nchw] = test::make_random_tensor({1, 2, 8, 8});
    const int64_t ishape[] = {1, 2, 8, 8};
    const int64_t oshape[] = {1, 2, 6, 6};
    const int64_t gshape[] = {1, 6, 6, 2};
    const int64_t C8 = 1, IH = 8, IW = 8, OH = 6, OW = 6;

    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    std::vector<float> grid_buf(1 * 6 * 6 * 2);
    for (size_t i = 0; i < grid_buf.size(); ++i) {
        grid_buf[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OH * out_pitch / 4));
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    grid_sample(in_c8, grid, out_c8, {});

    float result[1 * 2 * 6 * 6] = {};
    TensorView res_nchw(oshape, DataType::f32, result, TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    for (size_t i = 0; i < 72; ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(result[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(result[i]));
    }
}

// ============================================================
// 2D Nearest-neighbor tests
// ============================================================

NNOPS_TEST(grid_sample_2d_nearest_basic) {
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t oshape[] = {1, 1, 2, 2};
    const int64_t gshape[] = {1, 2, 2, 2};
    const int64_t C8 = 1, IH = 2, IW = 2, OH = 2, OW = 2;

    float in_data[4] = {1, 2, 3, 4};
    float grid_data[8] = {
        0.5f, 0.5f, 0.5f, 0.5f,
        0.5f, 0.5f, 0.5f, 0.5f,
    };

    TensorView in_nchw(ishape, DataType::f32, in_data, TensorLayout::NCHW);
    TensorView grid(gshape, DataType::f32, grid_data, TensorLayout::NCHW);

    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OH * out_pitch / 4));
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    GridSampleAttributes attrs;
    attrs.mode = GridSampleMode::Nearest;
    grid_sample(in_c8, grid, out_c8, attrs);

    float result[4] = {};
    TensorView res_nchw(oshape, DataType::f32, result, TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(result[i], 4.0f, 1e-5f);
    }
}

NNOPS_TEST(grid_sample_2d_nearest_random) {
    auto [in_vec, in_nchw] = test::make_random_tensor({1, 3, 10, 10});
    const int64_t ishape[] = {1, 3, 10, 10};
    const int64_t oshape[] = {1, 3, 5, 5};
    const int64_t gshape[] = {1, 5, 5, 2};
    const int64_t C8 = 1, IH = 10, IW = 10, OH = 5, OW = 5;

    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    std::vector<float> grid_buf(1 * 5 * 5 * 2);
    for (size_t i = 0; i < grid_buf.size(); ++i) {
        grid_buf[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OH * out_pitch / 4));
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    GridSampleAttributes attrs;
    attrs.mode = GridSampleMode::Nearest;
    grid_sample(in_c8, grid, out_c8, attrs);

    float result[1 * 3 * 5 * 5] = {};
    TensorView res_nchw(oshape, DataType::f32, result, TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    for (size_t i = 0; i < 75; ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(result[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(result[i]));
    }
}

// ============================================================
// Padding mode tests
// ============================================================

NNOPS_TEST(grid_sample_2d_zeros_padding) {
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t oshape[] = {1, 1, 2, 2};
    const int64_t gshape[] = {1, 2, 2, 2};
    const int64_t C8 = 1, IH = 2, IW = 2, OH = 2, OW = 2;

    float in_data[4] = {1, 2, 3, 4};
    float grid_data[8] = {
        10.0f, 10.0f, 10.0f, 10.0f,
        10.0f, 10.0f, 10.0f, 10.0f,
    };

    TensorView in_nchw(ishape, DataType::f32, in_data, TensorLayout::NCHW);
    TensorView grid(gshape, DataType::f32, grid_data, TensorLayout::NCHW);

    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OH * out_pitch / 4));
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    grid_sample(in_c8, grid, out_c8, {});

    float result[4] = {};
    TensorView res_nchw(oshape, DataType::f32, result, TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(result[i], 0.0f, 1e-5f);
    }
}

NNOPS_TEST(grid_sample_2d_border_padding) {
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t oshape[] = {1, 1, 2, 2};
    const int64_t gshape[] = {1, 2, 2, 2};
    const int64_t C8 = 1, IH = 2, IW = 2, OH = 2, OW = 2;

    float in_data[4] = {10, 20, 30, 40};
    float grid_data[8] = {
         0.0f, -10.0f, 0.0f, -10.0f,
         0.0f, -10.0f, 0.0f, -10.0f,
    };

    TensorView in_nchw(ishape, DataType::f32, in_data, TensorLayout::NCHW);
    TensorView grid(gshape, DataType::f32, grid_data, TensorLayout::NCHW);

    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OH * out_pitch / 4));
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    GridSampleAttributes attrs;
    attrs.padding_mode = GridSamplePaddingMode::Border;
    grid_sample(in_c8, grid, out_c8, attrs);

    float result[4] = {};
    TensorView res_nchw(oshape, DataType::f32, result, TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(result[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(result[i]));
    }
}

NNOPS_TEST(grid_sample_2d_reflection_padding) {
    auto [in_vec, in_nchw] = test::make_random_tensor({1, 1, 4, 4});
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t oshape[] = {1, 1, 4, 4};
    const int64_t gshape[] = {1, 4, 4, 2};
    const int64_t C8 = 1, IH = 4, IW = 4, OH = 4, OW = 4;

    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    std::vector<float> grid_buf(1 * 4 * 4 * 2);
    for (int64_t oh = 0; oh < 4; ++oh) {
        for (int64_t ow = 0; ow < 4; ++ow) {
            float gy = (static_cast<float>(oh) / 3.0f) * 2.4f - 1.2f;
            float gx = (static_cast<float>(ow) / 3.0f) * 2.4f - 1.2f;
            grid_buf[(oh * 4 + ow) * 2]     = gy;
            grid_buf[(oh * 4 + ow) * 2 + 1] = gx;
        }
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OH * out_pitch / 4));
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    GridSampleAttributes attrs;
    attrs.padding_mode = GridSamplePaddingMode::Reflection;
    grid_sample(in_c8, grid, out_c8, attrs);

    float result[1 * 1 * 4 * 4] = {};
    TensorView res_nchw(oshape, DataType::f32, result, TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    for (size_t i = 0; i < 16; ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(result[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(result[i]));
    }
}

// ============================================================
// 3D tests
// ============================================================

NNOPS_TEST(grid_sample_3d_bilinear_basic) {
    const int64_t ishape[] = {1, 1, 2, 2, 2};
    const int64_t oshape[] = {1, 1, 2, 2, 2};
    const int64_t gshape[] = {1, 2, 2, 2, 3};
    const int64_t C8 = 1, ID = 2, IH = 2, IW = 2, OD = 2, OH = 2, OW = 2;

    std::vector<float> in_buf(8);
    for (int i = 0; i < 8; ++i) in_buf[i] = static_cast<float>(i + 1);

    TensorView in_ncdhw(ishape, DataType::f32, in_buf.data(), TensorLayout::NCDHW);

    // Pack to NCDHWC8
    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * ID * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCDHWC8);
    pack_ncdhw_to_ncdhwc8(in_ncdhw, in_c8);

    std::vector<float> grid_buf(1 * 2 * 2 * 2 * 3);
    for (int64_t od = 0; od < 2; ++od) {
        for (int64_t oh = 0; oh < 2; ++oh) {
            for (int64_t ow = 0; ow < 2; ++ow) {
                float gd = (od == 0) ? -0.5f : 0.5f;
                float gh = (oh == 0) ? -0.5f : 0.5f;
                float gw = (ow == 0) ? -0.5f : 0.5f;
                size_t idx = (od * 2 * 2 + oh * 2 + ow) * 3;
                grid_buf[idx]     = gd;
                grid_buf[idx + 1] = gh;
                grid_buf[idx + 2] = gw;
            }
        }
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OD * OH * out_pitch / 4));
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCDHWC8);

    grid_sample(in_c8, grid, out_c8, {});

    std::vector<float> result(8);
    TensorView res_ncdhw(oshape, DataType::f32, result.data(), TensorLayout::NCDHW);
    unpack_ncdhwc8_to_ncdhw(out_c8, res_ncdhw);

    for (size_t i = 0; i < 8; ++i) {
        NNOPS_EXPECT_NEAR(result[i], static_cast<float>(i + 1), 1e-4f);
    }
}

NNOPS_TEST(grid_sample_3d_nearest_basic) {
    auto [in_vec, in_ncdhw] = test::make_random_tensor({1, 1, 4, 4, 4});
    const int64_t ishape[] = {1, 1, 4, 4, 4};
    const int64_t oshape[] = {1, 1, 3, 3, 3};
    const int64_t gshape[] = {1, 3, 3, 3, 3};
    const int64_t C8 = 1, ID = 4, IH = 4, IW = 4, OD = 3, OH = 3, OW = 3;

    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * ID * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCDHWC8);
    pack_ncdhw_to_ncdhwc8(in_ncdhw, in_c8);

    std::vector<float> grid_buf(1 * 3 * 3 * 3 * 3);
    for (size_t i = 0; i < grid_buf.size(); ++i) {
        grid_buf[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OD * OH * out_pitch / 4));
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCDHWC8);

    GridSampleAttributes attrs;
    attrs.mode = GridSampleMode::Nearest;
    grid_sample(in_c8, grid, out_c8, attrs);

    std::vector<float> result(1 * 1 * 3 * 3 * 3);
    TensorView res_ncdhw(oshape, DataType::f32, result.data(), TensorLayout::NCDHW);
    unpack_ncdhwc8_to_ncdhw(out_c8, res_ncdhw);

    for (size_t i = 0; i < 27; ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(result[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(result[i]));
    }
}

// ============================================================
// Class API vs Functional API
// ============================================================

NNOPS_TEST(grid_sample_class_vs_functional) {
    auto [in_vec, in_nchw] = test::make_random_tensor({1, 2, 4, 4});
    const int64_t ishape[] = {1, 2, 4, 4};
    const int64_t oshape[] = {1, 2, 3, 3};
    const int64_t gshape[] = {1, 3, 3, 2};
    const int64_t C8 = 1, IH = 4, IW = 4, OH = 3, OW = 3;

    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    std::vector<float> grid_buf(1 * 3 * 3 * 2);
    for (size_t i = 0; i < grid_buf.size(); ++i) {
        grid_buf[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    int64_t out_pitch = nchwc8_pitch(OW);

    // Functional
    std::vector<float> packed_out1(static_cast<size_t>(C8 * OH * out_pitch / 4));
    TensorView out1_c8(oshape, DataType::f32, packed_out1.data(), out_pitch, TensorLayout::NCHWC8);
    grid_sample(in_c8, grid, out1_c8, {});

    // Class
    std::vector<float> packed_out2(static_cast<size_t>(C8 * OH * out_pitch / 4));
    TensorView out2_c8(oshape, DataType::f32, packed_out2.data(), out_pitch, TensorLayout::NCHWC8);

    auto op = GridSample::create({}, Backend::CPU);
    const TensorView ins[] = {in_c8, grid};
    op->compute(out2_c8, ins);

    // Unpack and compare
    std::vector<float> out1_buf(1 * 2 * 3 * 3), out2_buf(1 * 2 * 3 * 3);
    TensorView out1(oshape, DataType::f32, out1_buf.data(), TensorLayout::NCHW);
    TensorView out2(oshape, DataType::f32, out2_buf.data(), TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out1_c8, out1);
    unpack_nchwc8_to_nchw(out2_c8, out2);

    NNOPS_EXPECT_TRUE(test::allclose(out1, out2, 1e-6f, 1e-6f));
}

// ============================================================
// Add-to mode
// ============================================================

NNOPS_TEST(grid_sample_add_to) {
    auto [in_vec, in_nchw] = test::make_random_tensor({1, 1, 4, 4});
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t oshape[] = {1, 1, 4, 4};
    const int64_t gshape[] = {1, 4, 4, 2};
    const int64_t C8 = 1, IH = 4, IW = 4, OH = 4, OW = 4;

    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    std::vector<float> grid_buf(1 * 4 * 4 * 2);
    for (int64_t oh = 0; oh < 4; ++oh) {
        for (int64_t ow = 0; ow < 4; ++ow) {
            float gy = (static_cast<float>(oh) + 0.5f) / 2.0f - 1.0f;
            float gx = (static_cast<float>(ow) + 0.5f) / 2.0f - 1.0f;
            grid_buf[(oh * 4 + ow) * 2]     = gy;
            grid_buf[(oh * 4 + ow) * 2 + 1] = gx;
        }
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OH * out_pitch / 4), 0.5f);
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    GridSampleAttributes attrs;
    attrs.add_to = true;
    grid_sample(in_c8, grid, out_c8, attrs);

    std::vector<float> result(1 * 1 * 4 * 4);
    TensorView res_nchw(oshape, DataType::f32, result.data(), TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    for (size_t i = 0; i < 16; ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(result[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(result[i]));
    }
}

// ============================================================
// Multi-sample batch test
// ============================================================

NNOPS_TEST(grid_sample_multisample) {
    auto [in_vec, in_nchw] = test::make_random_tensor({2, 1, 4, 4});
    const int64_t ishape[] = {2, 1, 4, 4};
    const int64_t oshape[] = {2, 1, 3, 3};
    const int64_t gshape[] = {2, 3, 3, 2};
    const int64_t N = 2, C8 = 1, IH = 4, IW = 4, OH = 3, OW = 3;

    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(N * C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    std::vector<float> grid_buf(2 * 3 * 3 * 2);
    for (size_t i = 0; i < grid_buf.size(); ++i) {
        grid_buf[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(N * C8 * OH * out_pitch / 4));
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    grid_sample(in_c8, grid, out_c8, {});

    std::vector<float> result(2 * 1 * 3 * 3);
    TensorView res_nchw(oshape, DataType::f32, result.data(), TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    for (size_t i = 0; i < 18; ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(result[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(result[i]));
    }
}

// ============================================================
// Align corners: 2D edge cases
// ============================================================

NNOPS_TEST(grid_sample_2d_align_corners_corners_exact) {
    const int64_t ishape[] = {1, 1, 3, 3};
    const int64_t oshape[] = {1, 1, 3, 3};
    const int64_t gshape[] = {1, 3, 3, 2};
    const int64_t C8 = 1, IH = 3, IW = 3, OH = 3, OW = 3;

    float in_data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    TensorView in_nchw(ishape, DataType::f32, in_data, TensorLayout::NCHW);

    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    std::vector<float> grid_buf(1 * 3 * 3 * 2);
    for (int64_t oh = 0; oh < 3; ++oh) {
        for (int64_t ow = 0; ow < 3; ++ow) {
            float gy = static_cast<float>(oh) / 1.0f - 1.0f;
            float gx = static_cast<float>(ow) / 1.0f - 1.0f;
            grid_buf[(oh * 3 + ow) * 2]     = gy;
            grid_buf[(oh * 3 + ow) * 2 + 1] = gx;
        }
    }
    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OH * out_pitch / 4));
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    GridSampleAttributes attrs;
    attrs.align_corners = true;
    grid_sample(in_c8, grid, out_c8, attrs);

    std::vector<float> result(9);
    TensorView res_nchw(oshape, DataType::f32, result.data(), TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    for (int i = 0; i < 9; ++i) {
        NNOPS_EXPECT_NEAR(result[i], in_data[i], 1e-5f);
    }
}

// ============================================================
// Partial C8 tests (zero-padded, all C8 blocks use SIMD)
// ============================================================

NNOPS_TEST(grid_sample_partial_c8) {
    auto [in_vec, in_nchw] = test::make_random_tensor({1, 3, 4, 4});
    const int64_t ishape[] = {1, 3, 4, 4};
    const int64_t oshape[] = {1, 3, 4, 4};
    const int64_t gshape[] = {1, 4, 4, 2};
    const int64_t C8 = 1, IH = 4, IW = 4, OH = 4, OW = 4;

    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    std::vector<float> grid_buf(1 * 4 * 4 * 2);
    for (int64_t oh = 0; oh < 4; ++oh) {
        for (int64_t ow = 0; ow < 4; ++ow) {
            float gy = (static_cast<float>(oh) + 0.5f) / 2.0f - 1.0f;
            float gx = (static_cast<float>(ow) + 0.5f) / 2.0f - 1.0f;
            grid_buf[(oh * 4 + ow) * 2]     = gy;
            grid_buf[(oh * 4 + ow) * 2 + 1] = gx;
        }
    }

    TensorView grid(gshape, DataType::f32, grid_buf.data(), TensorLayout::NCHW);

    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OH * out_pitch / 4));
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    grid_sample(in_c8, grid, out_c8, {});

    std::vector<float> result(1 * 3 * 4 * 4);
    TensorView res_nchw(oshape, DataType::f32, result.data(), TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    for (size_t i = 0; i < result.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(result[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(result[i]));
    }
}
