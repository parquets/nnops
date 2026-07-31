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
