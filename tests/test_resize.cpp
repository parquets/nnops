/// @file test_resize.cpp
/// @brief Tests for the Resize (interpolation) operator — NCHWC8/NCDHWC8 only.
///
/// All tests use the class API path:
///   NCHW in → LayoutConvert(pack) → NCHWC8 → Resize::compute → LayoutConvert(unpack) → NCHW → compare

#include "nnops/ops/resize.hpp"
#include "nnops/ops/layout_convert.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"

#include <algorithm>
#include <vector>
#include <cmath>

using namespace nnops;

// Forward declaration — NCHW scalar reference used as ground truth.
namespace nnops::backend::cpu::reference {
    void resize_ref(const ResizeAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

namespace {

/// 2D+3D roundtrip: pack → resize on packed → unpack → compare with NCHW reference.
void test_nchwc8_vs_nchw(const std::vector<int64_t>& in_shape,
                          const ResizeAttributes& attrs)
{
    const int64_t rank = static_cast<int64_t>(in_shape.size());
    const int64_t srank = rank - 2;  // 2 or 3

    // Determine layouts for this spatial rank.
    const TensorLayout planar_layout = (srank == 3) ? TensorLayout::NCDHW : TensorLayout::NCHW;
    const TensorLayout packed_layout = (srank == 3) ? TensorLayout::NCDHWC8 : TensorLayout::NCHWC8;

    // Random planar input with correct layout for rank (NCHW for 4D, NCDHW for 5D).
    auto [in_vec, in_default] = test::make_random_tensor(in_shape);
    TensorView in_planar(in_default.shape_span(), DataType::f32,
                         in_default.ptr<float>(), planar_layout);

    // ---- Golden: compute reference on planar layout ----
    std::vector<int64_t> oshape_ref = in_shape;
    if (srank == 3) {
        oshape_ref[2] = attrs.output_size[0];
        oshape_ref[3] = attrs.output_size[1];
        oshape_ref[4] = attrs.output_size[2];
    } else {
        oshape_ref[2] = attrs.output_size[1];
        oshape_ref[3] = attrs.output_size[2];
    }

    size_t ref_elems = 1;
    for (auto d : oshape_ref) ref_elems *= static_cast<size_t>(d);
    std::vector<float> ref_out(ref_elems);
    TensorView out_ref(oshape_ref, DataType::f32, ref_out.data(), planar_layout);

    const float prefill_val = 0.5f;
    if (attrs.add_to) {
        for (auto& v : ref_out) v = prefill_val;
    }

    {
        ComputeContext ctx;
        const TensorView ref_ins[] = {in_planar};
        backend::cpu::reference::resize_ref(attrs, out_ref, ref_ins, ctx, nullptr);
    }

    // ---- LayoutConvert: planar → packed ----
    {
        auto pack_op = LayoutConvert::create(packed_layout, Backend::CPU);
        auto d = in_planar.desc();
        const TensorDesc pack_in[] = {d};
        auto pack_descs = pack_op->getOutputTensorDesc(pack_in);

        // Validate pack output descriptor
        NNOPS_EXPECT_EQ(pack_descs[0].rank, rank);
        NNOPS_EXPECT_EQ(pack_descs[0].layout, packed_layout);
        NNOPS_EXPECT_EQ(pack_descs[0].dtype, DataType::f32);
        NNOPS_EXPECT_EQ(pack_descs[0].dims[0], in_shape[0]);
        NNOPS_EXPECT_EQ(pack_descs[0].dims[1], in_shape[1]);

        std::vector<char> packed_buf(pack_descs[0].storage_bytes());
        auto in_packed = test::make_packed(pack_descs[0], packed_buf.data());
        const TensorView pack_ins[] = {in_planar};
        pack_op->compute(in_packed, pack_ins);

        // ---- Resize on packed layout ----
        auto resize_op = Resize::create(attrs, Backend::CPU);
        auto pd = in_packed.desc();
        const TensorDesc resize_in[] = {pd};
        auto resize_descs = resize_op->getOutputTensorDesc(resize_in);

        // Validate resize output descriptor
        NNOPS_EXPECT_EQ(resize_descs[0].rank, rank);
        NNOPS_EXPECT_EQ(resize_descs[0].layout, packed_layout);
        NNOPS_EXPECT_EQ(resize_descs[0].dtype, DataType::f32);
        NNOPS_EXPECT_EQ(resize_descs[0].dims[0], in_shape[0]);  // N unchanged
        NNOPS_EXPECT_EQ(resize_descs[0].dims[1], in_shape[1]);  // C unchanged
        if (srank == 3) {
            NNOPS_EXPECT_EQ(resize_descs[0].dims[2], attrs.output_size[0]);
            NNOPS_EXPECT_EQ(resize_descs[0].dims[3], attrs.output_size[1]);
            NNOPS_EXPECT_EQ(resize_descs[0].dims[4], attrs.output_size[2]);
        } else {
            NNOPS_EXPECT_EQ(resize_descs[0].dims[2], attrs.output_size[1]);
            NNOPS_EXPECT_EQ(resize_descs[0].dims[3], attrs.output_size[2]);
        }

        std::vector<char> resize_buf(resize_descs[0].storage_bytes());
        if (attrs.add_to) {
            float* fbuf = reinterpret_cast<float*>(resize_buf.data());
            size_t nfloats = resize_descs[0].storage_bytes() / sizeof(float);
            std::fill(fbuf, fbuf + nfloats, prefill_val);
        }
        auto out_packed = test::make_packed(resize_descs[0], resize_buf.data());
        const TensorView resize_ins[] = {in_packed};
        resize_op->compute(out_packed, resize_ins);

        // ---- LayoutConvert: packed → planar ----
        auto unpack_op = LayoutConvert::create(planar_layout, Backend::CPU);
        auto od = out_packed.desc();
        const TensorDesc unpack_in[] = {od};
        auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in);

        // Validate unpack output descriptor
        NNOPS_EXPECT_EQ(unpack_descs[0].rank, rank);
        NNOPS_EXPECT_EQ(unpack_descs[0].layout, planar_layout);
        NNOPS_EXPECT_EQ(unpack_descs[0].dtype, DataType::f32);
        NNOPS_EXPECT_EQ(unpack_descs[0].dims[0], in_shape[0]);
        NNOPS_EXPECT_EQ(unpack_descs[0].dims[1], in_shape[1]);
        if (srank == 3) {
            NNOPS_EXPECT_EQ(unpack_descs[0].dims[2], attrs.output_size[0]);
            NNOPS_EXPECT_EQ(unpack_descs[0].dims[3], attrs.output_size[1]);
            NNOPS_EXPECT_EQ(unpack_descs[0].dims[4], attrs.output_size[2]);
        } else {
            NNOPS_EXPECT_EQ(unpack_descs[0].dims[2], attrs.output_size[1]);
            NNOPS_EXPECT_EQ(unpack_descs[0].dims[3], attrs.output_size[2]);
        }

        std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
        auto res_planar = test::make_planar(unpack_descs[0], unpack_buf.data());
        const TensorView unpack_ins[] = {out_packed};
        unpack_op->compute(res_planar, unpack_ins);

        // ---- Compare with golden ----
        float rtol = (attrs.mode == ResizeMode::Linear) ? 1e-3f : 1e-4f;
        float atol = (attrs.mode == ResizeMode::Linear) ? 1e-3f : 1e-4f;
        NNOPS_EXPECT_TRUE(test::allclose(res_planar, out_ref, rtol, atol));
    }
}

/// Manual roundtrip helper for hand-verified tests with known input data.
/// pack → resize → unpack using class API, then call the user's validation callback.
template <typename ValidateFn>
void run_resize_roundtrip(const std::vector<int64_t>& in_shape,
                           const std::vector<int64_t>& out_shape,
                           const std::vector<float>& in_data,
                           const ResizeAttributes& attrs,
                           ValidateFn validate_fn)
{
    const int64_t rank = static_cast<int64_t>(in_shape.size());
    const int64_t srank = rank - 2;
    const TensorLayout planar_layout = (srank == 3) ? TensorLayout::NCDHW : TensorLayout::NCHW;
    const TensorLayout packed_layout = (srank == 3) ? TensorLayout::NCDHWC8 : TensorLayout::NCHWC8;

    // Copy input data into a vector (LayoutConvert reads from it).
    std::vector<float> in_buf = in_data;
    TensorView in_planar(std::span<const int64_t>(in_shape), DataType::f32, in_buf.data(), planar_layout);

    // Pack
    auto pack_op = LayoutConvert::create(packed_layout, Backend::CPU);
    auto d = in_planar.desc();
    const TensorDesc pack_in[] = {d};
    auto pack_descs = pack_op->getOutputTensorDesc(pack_in);

    NNOPS_EXPECT_EQ(pack_descs[0].rank, rank);
    NNOPS_EXPECT_EQ(pack_descs[0].layout, packed_layout);
    NNOPS_EXPECT_EQ(pack_descs[0].dtype, DataType::f32);

    std::vector<char> packed_buf(pack_descs[0].storage_bytes());
    auto in_packed = test::make_packed(pack_descs[0], packed_buf.data());
    const TensorView pack_ins[] = {in_planar};
    pack_op->compute(in_packed, pack_ins);

    // Resize
    auto resize_op = Resize::create(attrs, Backend::CPU);
    auto pd = in_packed.desc();
    const TensorDesc resize_in[] = {pd};
    auto resize_descs = resize_op->getOutputTensorDesc(resize_in);

    NNOPS_EXPECT_EQ(resize_descs[0].rank, rank);
    NNOPS_EXPECT_EQ(resize_descs[0].layout, packed_layout);
    NNOPS_EXPECT_EQ(resize_descs[0].dtype, DataType::f32);

    std::vector<char> resize_buf(resize_descs[0].storage_bytes());
    auto out_packed = test::make_packed(resize_descs[0], resize_buf.data());
    const TensorView resize_ins[] = {in_packed};
    resize_op->compute(out_packed, resize_ins);

    // Unpack
    auto unpack_op = LayoutConvert::create(planar_layout, Backend::CPU);
    auto od = out_packed.desc();
    const TensorDesc unpack_in[] = {od};
    auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in);

    NNOPS_EXPECT_EQ(unpack_descs[0].rank, rank);
    NNOPS_EXPECT_EQ(unpack_descs[0].layout, planar_layout);
    NNOPS_EXPECT_EQ(unpack_descs[0].dtype, DataType::f32);

    std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
    auto res_planar = test::make_planar(unpack_descs[0], unpack_buf.data());
    const TensorView unpack_ins[] = {out_packed};
    unpack_op->compute(res_planar, unpack_ins);

    // Validate
    validate_fn(res_planar);
}

}  // anonymous namespace

// ============================================================
// 2D Nearest-neighbor hand-verified tests
// ============================================================

NNOPS_TEST(resize_2d_nearest_upsample) {
    // 1x1x2x2 -> 1x1x4x4, nearest with Asymmetric mode
    const std::vector<int64_t> ishape = {1, 1, 2, 2};
    const std::vector<int64_t> oshape = {1, 1, 4, 4};
    const std::vector<float> in_data = {1, 2, 3, 4};

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 4, 4};

    run_resize_roundtrip(ishape, oshape, in_data, attrs, [](const TensorView& result) {
        const float* r = result.ptr<float>();
        // scale=0.5, each input pixel maps to a 2x2 block
        NNOPS_EXPECT_NEAR(r[0],  1.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[3],  2.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[4],  3.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[7],  4.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[12], 3.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[15], 4.0f, 1e-6f);
    });
}

NNOPS_TEST(resize_2d_nearest_downsample) {
    const std::vector<int64_t> ishape = {1, 1, 4, 4};
    const std::vector<int64_t> oshape = {1, 1, 2, 2};
    const std::vector<float> in_data = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
       13, 14, 15, 16
    };

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 2, 2};

    run_resize_roundtrip(ishape, oshape, in_data, attrs, [](const TensorView& result) {
        const float* r = result.ptr<float>();
        // scale=2.0, dst 0->src 0, dst 1->src 2
        NNOPS_EXPECT_NEAR(r[0], 1.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[1], 3.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[2], 9.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[3], 11.0f, 1e-6f);
    });
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
    const std::vector<int64_t> ishape = {1, 1, 2, 2};
    const std::vector<int64_t> oshape = {1, 1, 4, 4};
    const std::vector<float> in_data = {1, 2, 3, 4};

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::AlignCorners;
    attrs.output_size = {0, 4, 4};

    run_resize_roundtrip(ishape, oshape, in_data, attrs, [](const TensorView& result) {
        const float* r = result.ptr<float>();
        NNOPS_EXPECT_NEAR(r[0],  1.0f, 1e-5f);
        NNOPS_EXPECT_NEAR(r[3],  2.0f, 1e-5f);
        NNOPS_EXPECT_NEAR(r[12], 3.0f, 1e-5f);
        NNOPS_EXPECT_NEAR(r[15], 4.0f, 1e-5f);
        for (int64_t i = 0; i < result.numel(); ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(r[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(r[i]));
        }
    });
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

// ============================================================
// Crop + resize tests — crop a sub-region then resize to output
// ============================================================

NNOPS_TEST(resize_2d_crop_nearest_upsample) {
    // 1x1x8x8 input, crop center 4x4 region [2:6, 2:6], resize to 8x8
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 8, 8};
    attrs.crop_start = {0, 2, 2};
    attrs.crop_end   = {0, 6, 6};
    test_nchwc8_vs_nchw({1, 3, 8, 8}, attrs);
}

NNOPS_TEST(resize_2d_crop_nearest_downsample) {
    // Crop a 8x8 sub-region from 16x16, then nearest-downsample to 4x4
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 4, 4};
    attrs.crop_start = {0, 4, 4};
    attrs.crop_end   = {0, 12, 12};
    test_nchwc8_vs_nchw({1, 3, 16, 16}, attrs);
}

NNOPS_TEST(resize_2d_crop_bilinear_upsample) {
    // Crop center 4x4 from 10x10, bilinear upsample to 12x12
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 12, 12};
    attrs.crop_start = {0, 3, 3};
    attrs.crop_end   = {0, 7, 7};
    test_nchwc8_vs_nchw({1, 3, 10, 10}, attrs);
}

NNOPS_TEST(resize_2d_crop_bilinear_align_corners) {
    // Crop + bilinear with AlignCorners coordinate mode
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::AlignCorners;
    attrs.output_size = {0, 8, 8};
    attrs.crop_start = {0, 1, 1};
    attrs.crop_end   = {0, 5, 5};
    test_nchwc8_vs_nchw({1, 1, 6, 6}, attrs);
}

NNOPS_TEST(resize_2d_crop_full_image) {
    // Crop equals full image — should produce same result as no-crop
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 16, 16};
    attrs.crop_start = {0, 0, 0};
    attrs.crop_end   = {0, 8, 8};
    test_nchwc8_vs_nchw({1, 3, 8, 8}, attrs);
}

NNOPS_TEST(resize_2d_crop_single_pixel_output) {
    // Crop a thin 2x2 strip, resize down to 1x1 (single pixel)
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 1, 1};
    attrs.crop_start = {0, 3, 5};
    attrs.crop_end   = {0, 5, 7};
    test_nchwc8_vs_nchw({1, 3, 8, 12}, attrs);
}

NNOPS_TEST(resize_2d_crop_asymmetric_edge) {
    // Crop includes top-left corner (start at origin, partial end)
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 8, 8};
    attrs.crop_start = {0, 0, 0};
    attrs.crop_end   = {0, 6, 6};
    test_nchwc8_vs_nchw({1, 3, 10, 10}, attrs);
}

NNOPS_TEST(resize_2d_crop_bottom_right_corner) {
    // Crop includes bottom-right corner
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 8, 8};
    attrs.crop_start = {0, 6, 6};
    attrs.crop_end   = {0, 12, 12};
    test_nchwc8_vs_nchw({1, 3, 12, 12}, attrs);
}

NNOPS_TEST(resize_2d_crop_multichannel) {
    // Multi-channel crop + resize
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 8, 8};
    attrs.crop_start = {0, 4, 4};
    attrs.crop_end   = {0, 12, 12};
    test_nchwc8_vs_nchw({2, 7, 16, 16}, attrs);
}

NNOPS_TEST(resize_2d_crop_partial_c8) {
    // Crop with partial C8 channels (3 channels → 1 C8 block, zero-padded)
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {0, 16, 16};
    attrs.crop_start = {0, 2, 2};
    attrs.crop_end   = {0, 10, 10};
    test_nchwc8_vs_nchw({1, 3, 12, 12}, attrs);
}

NNOPS_TEST(resize_2d_crop_add_to) {
    // Crop + add_to mode
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 8, 8};
    attrs.add_to = true;
    attrs.crop_start = {0, 3, 3};
    attrs.crop_end   = {0, 9, 9};
    test_nchwc8_vs_nchw({1, 3, 12, 12}, attrs);
}

// ============================================================
// 3D crop + resize tests
// ============================================================

NNOPS_TEST(resize_3d_crop_nearest) {
    // Crop D=[1:3), H=[2:6), W=[2:6) from 4x8x8, then nearest to 4x4x4
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {4, 4, 4};
    attrs.crop_start = {1, 2, 2};
    attrs.crop_end   = {3, 6, 6};
    test_nchwc8_vs_nchw({1, 3, 4, 8, 8}, attrs);
}

NNOPS_TEST(resize_3d_crop_trilinear) {
    // Crop center region from 8x12x12, trilinear to 4x6x6
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {4, 6, 6};
    attrs.crop_start = {2, 2, 2};
    attrs.crop_end   = {6, 10, 10};
    test_nchwc8_vs_nchw({1, 3, 8, 12, 12}, attrs);
}

NNOPS_TEST(resize_3d_crop_full_volume) {
    // Crop equals entire 3D volume — same as no-crop
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::HalfPixel;
    attrs.output_size = {4, 8, 8};
    attrs.crop_start = {0, 0, 0};
    attrs.crop_end   = {4, 8, 8};
    test_nchwc8_vs_nchw({1, 1, 4, 8, 8}, attrs);
}

NNOPS_TEST(resize_3d_crop_partial_c8) {
    // 3D crop with partial C8 channels
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {4, 8, 8};
    attrs.crop_start = {1, 2, 2};
    attrs.crop_end   = {5, 10, 10};
    test_nchwc8_vs_nchw({1, 5, 6, 12, 12}, attrs);
}

NNOPS_TEST(resize_3d_crop_align_corners) {
    // 3D crop + trilinear with AlignCorners
    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::AlignCorners;
    attrs.output_size = {4, 6, 6};
    attrs.crop_start = {1, 1, 1};
    attrs.crop_end   = {5, 7, 7};
    test_nchwc8_vs_nchw({1, 2, 6, 8, 8}, attrs);
}

// ============================================================
// Hand-verified crop + resize: known input → known output
// ============================================================

NNOPS_TEST(resize_2d_crop_nearest_hand_verified) {
    // 1x1x4x4 input:
    //   1   2   3   4
    //   5   6   7   8
    //   9  10  11  12
    //  13  14  15  16
    //
    // Crop [1:3, 1:3) = {6,7; 10,11}, nearest-upsample Asymmetric to 4x4.
    //
    // scale = 2/4 = 0.5.  Asymmetric: src = dst * scale.
    //   dst=0 → src=0.0, round=0
    //   dst=1 → src=0.5, round=1  (std::round half-away-from-zero)
    //   dst=2 → src=1.0, round=1
    //   dst=3 → src=1.5, round=2 → clamp(2,2)=1
    //
    // Expected output (4x4):
    //   6   7   7   7
    //  10  11  11  11
    //  10  11  11  11
    //  10  11  11  11
    const std::vector<int64_t> ishape = {1, 1, 4, 4};
    const std::vector<float> in_data = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
       13, 14, 15, 16
    };

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Nearest;
    attrs.coord_mode = CoordinateTransformMode::Asymmetric;
    attrs.output_size = {0, 4, 4};
    attrs.crop_start = {0, 1, 1};
    attrs.crop_end   = {0, 3, 3};

    run_resize_roundtrip(ishape, {1, 1, 4, 4}, in_data, attrs, [](const TensorView& result) {
        const float* r = result.ptr<float>();
        NNOPS_EXPECT_NEAR(r[0],  6.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[1],  7.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[2],  7.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[3],  7.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[4], 10.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[5], 11.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[6], 11.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[7], 11.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[8], 10.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[9], 11.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[10], 11.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[11], 11.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[12], 10.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[13], 11.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[14], 11.0f, 1e-6f);
        NNOPS_EXPECT_NEAR(r[15], 11.0f, 1e-6f);
    });
}

NNOPS_TEST(resize_2d_crop_bilinear_hand_verified) {
    // 1x1x3x3 input:
    //   1  2  3
    //   4  5  6
    //   7  8  9
    //
    // Crop [0:2, 0:2) = {1,2; 4,5}, bilinear-upsample AlignCorners to 3x3
    // src_h = dst * (2-1)/(3-1) = dst * 0.5
    // dst_h=0 -> src_h=0.0 (row 0, col 0), dst_h=2 -> src_h=1.0 (row 1, col 1)
    // dst_h=1 -> src_h=0.5 (interpolate row 0/1)
    //
    // Output:
    //   1.0   1.5   2.0
    //   2.5   3.0   3.5
    //   4.0   4.5   5.0
    const std::vector<int64_t> ishape = {1, 1, 3, 3};
    const std::vector<float> in_data = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    ResizeAttributes attrs;
    attrs.mode = ResizeMode::Linear;
    attrs.coord_mode = CoordinateTransformMode::AlignCorners;
    attrs.output_size = {0, 3, 3};
    attrs.crop_start = {0, 0, 0};
    attrs.crop_end   = {0, 2, 2};

    run_resize_roundtrip(ishape, {1, 1, 3, 3}, in_data, attrs, [](const TensorView& result) {
        const float* r = result.ptr<float>();
        NNOPS_EXPECT_NEAR(r[0], 1.0f, 1e-5f);  // (0,0): v00=1
        NNOPS_EXPECT_NEAR(r[1], 1.5f, 1e-5f);  // (0,1): h=0,w=0.5 → 0.5*1+0.5*2=1.5
        NNOPS_EXPECT_NEAR(r[2], 2.0f, 1e-5f);  // (0,2): v01=2
        NNOPS_EXPECT_NEAR(r[3], 2.5f, 1e-5f);  // (1,0): h=0.5,w=0 → 0.5*1+0.5*4=2.5
        NNOPS_EXPECT_NEAR(r[4], 3.0f, 1e-5f);  // (1,1): center interp of all 4
        NNOPS_EXPECT_NEAR(r[5], 3.5f, 1e-5f);  // (1,2): h=0.5,w=1 → 0.5*2+0.5*5=3.5
        NNOPS_EXPECT_NEAR(r[6], 4.0f, 1e-5f);  // (2,0): v10=4
        NNOPS_EXPECT_NEAR(r[7], 4.5f, 1e-5f);  // (2,1): h=1,w=0.5 → 0.5*4+0.5*5=4.5
        NNOPS_EXPECT_NEAR(r[8], 5.0f, 1e-5f);  // (2,2): v11=5
        for (int64_t i = 0; i < result.numel(); ++i) {
            NNOPS_EXPECT_TRUE(!std::isnan(r[i]));
            NNOPS_EXPECT_TRUE(!std::isinf(r[i]));
        }
    });
}
