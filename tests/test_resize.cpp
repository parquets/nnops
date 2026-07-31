/// @file test_resize.cpp
/// @brief Tests for the Resize (interpolation) operator — NCHWC8/NCDHWC8 only.
///
/// All tests go through the NCHWC8 path:
///   NCHW in → pack_nchw_to_nchwc8 → resize(NCHWC8) → unpack → compare

#include "nnops/ops/resize.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "backend/cpu/layout_convert.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

// Forward declarations
namespace nnops::backend::cpu {
    void resize_cpu(const ResizeAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}
namespace nnops::backend::cpu::reference {
    void resize_ref(const ResizeAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

namespace {

/// Compute aligned pitch for NCHWC8 tensor.
inline int64_t nchwc8_pitch(int64_t W, int64_t elem_size = 4) {
    return ((W * 8 * elem_size + 31) / 32) * 32;
}

/// Pack → resize → unpack → compare against NCHW reference.
void test_nchwc8_vs_nchw(const std::vector<int64_t>& in_shape,
                          const ResizeAttributes& attrs)
{
    const int64_t N = in_shape[0], C = in_shape[1];
    const int64_t rank = static_cast<int64_t>(in_shape.size());
    const int64_t srank = rank - 2;  // 2 or 3
    const int64_t IH = in_shape[srank];
    const int64_t IW = in_shape[srank + 1];

    // Random NCHW input
    auto [in_vec, in_nchw] = test::make_random_tensor(in_shape);
    const int64_t C8 = (C + 7) / 8;

    // NCHW scalar reference (golden)
    std::vector<int64_t> oshape_ref = in_shape;
    if (srank == 3) {
        oshape_ref[2] = attrs.output_size[0];
        oshape_ref[3] = attrs.output_size[1];
        oshape_ref[4] = attrs.output_size[2];
    } else {
        oshape_ref[2] = attrs.output_size[1];
        oshape_ref[3] = attrs.output_size[2];
    }

    const int64_t O_H = (srank == 3) ? attrs.output_size[1] : attrs.output_size[1];
    const int64_t O_W = (srank == 3) ? attrs.output_size[2] : attrs.output_size[2];
    const int64_t O_D = (srank == 3) ? attrs.output_size[0] : 1;

    size_t ref_elems = 1;
    for (auto d : oshape_ref) ref_elems *= static_cast<size_t>(d);
    std::vector<float> ref_out(ref_elems);
    TensorView out_ref(oshape_ref, DataType::f32, ref_out.data(), TensorLayout::NCHW);

    // For add_to: pre-fill both reference and NCHWC8 output with the same value
    const float prefill_val = 0.5f;
    if (attrs.add_to) {
        for (auto& v : ref_out) v = prefill_val;
    }

    {
        ComputeContext ctx;
        const TensorView ref_arr[] = {in_nchw};
        backend::cpu::reference::resize_ref(attrs, out_ref, ref_arr, ctx, nullptr);
    }

    // Pack input to NCHWC8 (or NCDHWC8 for 3D)
    int64_t in_pitch = nchwc8_pitch(IW);
    int64_t in_pitch_elems = in_pitch / 4;
    int64_t IH_eff = (srank == 3) ? in_shape[2] * IH : IH;
    std::vector<float> packed_in(static_cast<size_t>(N * C8 * IH_eff * in_pitch_elems));

    TensorLayout packed_layout = (srank == 3) ? TensorLayout::NCDHWC8 : TensorLayout::NCHWC8;
    TensorView in_c8(in_shape, DataType::f32, packed_in.data(), in_pitch, packed_layout);

    if (srank == 3) {
        pack_ncdhw_to_ncdhwc8(in_nchw, in_c8);
    } else {
        pack_nchw_to_nchwc8(in_nchw, in_c8);
    }

    // NCHWC8 output
    int64_t out_pitch = nchwc8_pitch(O_W);
    int64_t out_pitch_elems = out_pitch / 4;
    int64_t OH_eff = (srank == 3) ? O_D * O_H : O_H;
    std::vector<float> packed_out(static_cast<size_t>(N * C8 * OH_eff * out_pitch_elems));
    TensorView out_c8(oshape_ref, DataType::f32, packed_out.data(), out_pitch, packed_layout);

    // For add_to: pre-fill NCHWC8 output (same value as reference pre-fill)
    if (attrs.add_to) {
        for (auto& v : packed_out) v = prefill_val;
    }

    // Run backend
    {
        ComputeContext ctx;
        const TensorView ins_arr[] = {in_c8};
        backend::cpu::resize_cpu(attrs, out_c8, ins_arr, ctx, nullptr);
    }

    // Unpack and compare
    std::vector<float> result(ref_elems);
    TensorView res_nchw(oshape_ref, DataType::f32, result.data(), TensorLayout::NCHW);

    if (srank == 3) {
        unpack_ncdhwc8_to_ncdhw(out_c8, res_nchw);
    } else {
        unpack_nchwc8_to_nchw(out_c8, res_nchw);
    }

    // Use a more relaxed tolerance for linear interpolation (floating point differences)
    float rtol = (attrs.mode == ResizeMode::Linear) ? 1e-3f : 1e-4f;
    float atol = (attrs.mode == ResizeMode::Linear) ? 1e-3f : 1e-4f;
    NNOPS_EXPECT_TRUE(test::allclose(res_nchw, out_ref, rtol, atol));
}

}  // anonymous namespace

// ============================================================
// 2D Nearest-neighbor hand-verified tests
// ============================================================

NNOPS_TEST(resize_2d_nearest_upsample) {
    // 1x1x2x2 -> 1x1x4x4, nearest with Asymmetric mode
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t oshape[] = {1, 1, 4, 4};
    const int64_t C8 = 1, IH = 2, IW = 2, OH = 4, OW = 4;
    float in_data[4] = {1, 2, 3, 4};

    TensorView in_nchw(ishape, DataType::f32, in_data, TensorLayout::NCHW);

    // Pack to NCHWC8
    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    // Output NCHWC8
    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OH * out_pitch / 4));
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 4, 4};

    resize(in_c8, out_c8, attrs);

    // Unpack
    float result[16] = {};
    TensorView res_nchw(oshape, DataType::f32, result, TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    // scale=0.5, each input pixel maps to a 2x2 block
    NNOPS_EXPECT_NEAR(result[0],  1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(result[3],  2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(result[4],  3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(result[7],  4.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(result[12], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(result[15], 4.0f, 1e-6f);
}

NNOPS_TEST(resize_2d_nearest_downsample) {
    const int64_t ishape[] = {1, 1, 4, 4};
    const int64_t oshape[] = {1, 1, 2, 2};
    const int64_t C8 = 1, IH = 4, IW = 4, OH = 2, OW = 2;
    float in_data[16] = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
       13, 14, 15, 16
    };

    TensorView in_nchw(ishape, DataType::f32, in_data, TensorLayout::NCHW);

    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OH * out_pitch / 4));
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 2, 2};

    resize(in_c8, out_c8, attrs);

    float result[4] = {};
    TensorView res_nchw(oshape, DataType::f32, result, TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    // scale=2.0, dst 0->src 0, dst 1->src 2
    NNOPS_EXPECT_NEAR(result[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(result[1], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(result[2], 9.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(result[3], 11.0f, 1e-6f);
}

NNOPS_TEST(resize_2d_nearest_identity) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 8, 8};
    test_nchwc8_vs_nchw({1, 3, 8, 8}, attrs);
}

// ============================================================
// 2D Linear (bilinear) tests
// ============================================================

NNOPS_TEST(resize_2d_linear_align_corners_corners_exact) {
    // 1x1x2x2 -> 1x1x4x4, bilinear with AlignCorners
    const int64_t ishape[] = {1, 1, 2, 2};
    const int64_t oshape[] = {1, 1, 4, 4};
    const int64_t C8 = 1, IH = 2, IW = 2, OH = 4, OW = 4;
    float in_data[4] = {1, 2, 3, 4};

    TensorView in_nchw(ishape, DataType::f32, in_data, TensorLayout::NCHW);

    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(static_cast<size_t>(C8 * OH * out_pitch / 4));
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::AlignCorners;
    attrs.output_size = {0, 4, 4};

    resize(in_c8, out_c8, attrs);

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

NNOPS_TEST(resize_2d_linear_identity) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 4, 4};
    test_nchwc8_vs_nchw({1, 1, 4, 4}, attrs);
}

NNOPS_TEST(resize_2d_linear_upsample_no_nan) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 20, 20};
    test_nchwc8_vs_nchw({1, 2, 8, 8}, attrs);
}

// ============================================================
// 3D Nearest-neighbor tests
// ============================================================

NNOPS_TEST(resize_3d_nearest_basic) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {4, 4, 4};
    test_nchwc8_vs_nchw({1, 1, 2, 2, 2}, attrs);
}

NNOPS_TEST(resize_3d_nearest_identity) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {4, 4, 4};
    test_nchwc8_vs_nchw({1, 2, 4, 4, 4}, attrs);
}

// ============================================================
// 3D Linear (trilinear) tests
// ============================================================

NNOPS_TEST(resize_3d_linear_basic) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {8, 12, 12};
    test_nchwc8_vs_nchw({1, 1, 4, 6, 6}, attrs);
}

NNOPS_TEST(resize_3d_linear_identity) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {4, 4, 4};
    test_nchwc8_vs_nchw({1, 2, 4, 4, 4}, attrs);
}

// ============================================================
// Random stress tests (multi-channel, partial C8)
// ============================================================

NNOPS_TEST(resize_2d_multichannel) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 16, 16};
    test_nchwc8_vs_nchw({2, 3, 32, 32}, attrs);
}

NNOPS_TEST(resize_3d_multichannel) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {4, 8, 8};
    test_nchwc8_vs_nchw({1, 3, 8, 16, 16}, attrs);
}

// ============================================================
// Class API vs Functional API
// ============================================================

NNOPS_TEST(resize_class_vs_functional) {
    auto [in_vec, in_nchw] = test::make_random_tensor({1, 2, 8, 8});
    const int64_t C8 = 1;
    const int64_t IH = 8, IW = 8, OH = 16, OW = 16;

    // Pack to NCHWC8
    const int64_t ishape[] = {1, 2, 8, 8};
    int64_t in_pitch = nchwc8_pitch(IW);
    std::vector<float> packed_in(static_cast<size_t>(C8 * IH * in_pitch / 4));
    TensorView in_c8(ishape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 16, 16};

    // Functional API
    const int64_t oshape[] = {1, 2, 16, 16};
    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out1(static_cast<size_t>(C8 * OH * out_pitch / 4));
    TensorView out1_c8(oshape, DataType::f32, packed_out1.data(), out_pitch, TensorLayout::NCHWC8);
    resize(in_c8, out1_c8, attrs);

    // Class API
    std::vector<float> packed_out2(static_cast<size_t>(C8 * OH * out_pitch / 4));
    TensorView out2_c8(oshape, DataType::f32, packed_out2.data(), out_pitch, TensorLayout::NCHWC8);

    auto op = Resize::create(attrs, Backend::CPU);
    const TensorView ins[] = {in_c8};
    op->compute(out2_c8, ins);

    // Unpack both and compare
    std::vector<float> out1_buf(1 * 2 * 16 * 16), out2_buf(1 * 2 * 16 * 16);
    TensorView out1(oshape, DataType::f32, out1_buf.data(), TensorLayout::NCHW);
    TensorView out2(oshape, DataType::f32, out2_buf.data(), TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out1_c8, out1);
    unpack_nchwc8_to_nchw(out2_c8, out2);

    NNOPS_EXPECT_TRUE(test::allclose(out1, out2, 1e-6f, 1e-6f));
}

// ============================================================
// Add-to mode
// ============================================================

NNOPS_TEST(resize_add_to) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 8, 8};
    attrs.add_to = true;
    test_nchwc8_vs_nchw({1, 1, 4, 4}, attrs);
}

// ============================================================
// HalfPixel coordinate mode
// ============================================================

NNOPS_TEST(resize_2d_linear_half_pixel) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 16, 16};
    test_nchwc8_vs_nchw({1, 1, 8, 8}, attrs);
}

// ============================================================
// Partial C8 tests (zero-padded, all C8 blocks use SIMD)
// ============================================================

NNOPS_TEST(resize_partial_c8_nearest) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 16, 16};
    test_nchwc8_vs_nchw({1, 5, 8, 8}, attrs);
}

NNOPS_TEST(resize_partial_c8_linear) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 16, 16};
    test_nchwc8_vs_nchw({1, 3, 8, 8}, attrs);
}

NNOPS_TEST(resize_single_channel) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 12, 12};
    test_nchwc8_vs_nchw({1, 1, 4, 4}, attrs);
}

NNOPS_TEST(resize_multi_c8_nearest) {
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 8, 8};
    test_nchwc8_vs_nchw({2, 17, 16, 16}, attrs);
}
