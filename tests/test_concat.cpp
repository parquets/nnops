/// @file test_concat.cpp
/// @brief Unit tests for Concat operator (CPU backend).
///
/// Tests cover planar (NCHW/NCDHW) and packed (NCHWC8/NCDHWC8) layouts,
/// including C-axis concatenation with packed C8 lane-level merging.

#include "nnops/ops/concat.hpp"
#include "nnops/ops/layout_convert.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"

#include <vector>
#include <cstring>

using namespace nnops;

// Forward declaration — NCHW scalar reference used as ground truth.
namespace nnops::backend::cpu::reference {
    void concat_ref(const ConcatAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

namespace {

/// Helper: run concat on packed data (pack → concat → unpack roundtrip)
/// and compare against the planar reference concat.
void test_packed_concat(const std::vector<std::vector<int64_t>>& in_shapes,
                         int64_t axis,
                         DataType dtype = DataType::f32)
{
    const int64_t N = static_cast<int64_t>(in_shapes.size());
    NNOPS_ASSERT(N >= 2);

    const int64_t rank = static_cast<int64_t>(in_shapes[0].size());
    const int64_t srank = rank - 2;
    const TensorLayout planar_layout = (srank == 3) ? TensorLayout::NCDHW : TensorLayout::NCHW;
    const TensorLayout packed_layout = (srank == 3) ? TensorLayout::NCDHWC8 : TensorLayout::NCHWC8;

    // Create random planar inputs
    std::vector<std::vector<float>> in_bufs(N);
    std::vector<TensorView> in_planars(N);
    for (int64_t i = 0; i < N; ++i) {
        auto [buf, tv] = test::make_random_tensor(in_shapes[static_cast<size_t>(i)], -1.0f, 1.0f,
                                                   42 + static_cast<int>(i));
        in_bufs[static_cast<size_t>(i)] = std::move(buf);
        in_planars[static_cast<size_t>(i)] = tv;
    }

    // ---- Golden: reference concat on planar ----
    ConcatAttributes attrs;
    attrs.axis = axis;

    // Compute output shape
    auto op_ref = Concat::create(attrs, Backend::CPU);
    std::vector<TensorDesc> planar_descs(N);
    for (int64_t i = 0; i < N; ++i) {
        planar_descs[static_cast<size_t>(i)] = in_planars[static_cast<size_t>(i)].desc();
    }
    auto ref_out_descs = op_ref->getOutputTensorDesc(planar_descs);

    std::vector<float> ref_out(static_cast<size_t>(ref_out_descs[0].numel()));
    TensorView out_ref = test::make_planar(ref_out_descs[0], ref_out.data());

    std::vector<TensorView> ref_ins(N);
    for (int64_t i = 0; i < N; ++i) {
        ref_ins[static_cast<size_t>(i)] = in_planars[static_cast<size_t>(i)];
    }

    {
        ComputeContext ctx;
        backend::cpu::reference::concat_ref(attrs, out_ref, ref_ins, ctx, nullptr);
    }

    // ---- Packed path: pack each input → concat → unpack ----
    // Pack each input
    std::vector<std::vector<char>> packed_bufs(N);
    std::vector<TensorView> in_packeds(N);
    for (int64_t i = 0; i < N; ++i) {
        auto pack_op = LayoutConvert::create(packed_layout, Backend::CPU);
        auto d = in_planars[static_cast<size_t>(i)].desc();
        const TensorDesc pack_in[] = {d};
        auto pack_descs = pack_op->getOutputTensorDesc(pack_in);

        packed_bufs[static_cast<size_t>(i)].resize(
            static_cast<size_t>(pack_descs[0].storage_bytes()));
        in_packeds[static_cast<size_t>(i)] =
            test::make_packed(pack_descs[0], packed_bufs[static_cast<size_t>(i)].data());
        const TensorView pack_ins[] = {in_planars[static_cast<size_t>(i)]};
        pack_op->compute(in_packeds[static_cast<size_t>(i)], pack_ins);
    }

    // Concat on packed
    auto concat_op = Concat::create(attrs, Backend::CPU);
    std::vector<TensorDesc> packed_descs(N);
    for (int64_t i = 0; i < N; ++i) {
        packed_descs[static_cast<size_t>(i)] = in_packeds[static_cast<size_t>(i)].desc();
    }
    auto concat_descs = concat_op->getOutputTensorDesc(packed_descs);

    NNOPS_EXPECT_EQ(concat_descs[0].rank, rank);
    NNOPS_EXPECT_EQ(concat_descs[0].layout, packed_layout);
    NNOPS_EXPECT_EQ(concat_descs[0].dtype, dtype);

    std::vector<char> concat_buf(static_cast<size_t>(concat_descs[0].storage_bytes()));
    auto out_packed = test::make_packed(concat_descs[0], concat_buf.data());

    std::vector<TensorView> concat_ins(N);
    for (int64_t i = 0; i < N; ++i) {
        concat_ins[static_cast<size_t>(i)] = in_packeds[static_cast<size_t>(i)];
    }
    concat_op->compute(out_packed, concat_ins);

    // Unpack
    auto unpack_op = LayoutConvert::create(planar_layout, Backend::CPU);
    auto od = out_packed.desc();
    const TensorDesc unpack_in[] = {od};
    auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in);

    NNOPS_EXPECT_EQ(unpack_descs[0].rank, rank);
    NNOPS_EXPECT_EQ(unpack_descs[0].layout, planar_layout);
    NNOPS_EXPECT_EQ(unpack_descs[0].dtype, dtype);

    std::vector<char> unpack_buf(static_cast<size_t>(unpack_descs[0].storage_bytes()));
    auto res_planar = test::make_planar(unpack_descs[0], unpack_buf.data());
    const TensorView unpack_ins[] = {out_packed};
    unpack_op->compute(res_planar, unpack_ins);

    // Compare with planar reference
    NNOPS_EXPECT_TRUE(test::allclose(res_planar, out_ref, 1e-5f, 1e-5f));
}

}  // anonymous namespace

// ============================================================
// Hand-verified tests
// ============================================================

NNOPS_TEST(concat_1d_basic) {
    const int64_t shape[] = {3};
    float a_data[] = {1.0f, 2.0f, 3.0f};
    float b_data[] = {4.0f, 5.0f, 6.0f};
    float out_data[6] = {};

    TensorView a(shape, DataType::f32, a_data);
    TensorView b(shape, DataType::f32, b_data);

    ConcatAttributes attrs;
    attrs.axis = 0;
    auto op = Concat::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 6);

    auto output = test::make_planar(descs[0], out_data);
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_data[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[3], 4.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[4], 5.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[5], 6.0f, 1e-6f);
}

NNOPS_TEST(concat_2d_axis0) {
    // Concat along batch dim: [2,3] + [1,3] → [3,3]
    const int64_t shape_a[] = {2, 3};
    const int64_t shape_b[] = {1, 3};
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b_data[] = {7.0f, 8.0f, 9.0f};
    float out_data[9] = {};

    TensorView a(shape_a, DataType::f32, a_data);
    TensorView b(shape_b, DataType::f32, b_data);

    ConcatAttributes attrs;
    attrs.axis = 0;
    auto op = Concat::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 3);

    auto output = test::make_planar(descs[0], out_data);
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    // Row 0: from a
    NNOPS_EXPECT_NEAR(out_data[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 3.0f, 1e-6f);
    // Row 1: from a
    NNOPS_EXPECT_NEAR(out_data[3], 4.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[4], 5.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[5], 6.0f, 1e-6f);
    // Row 2: from b
    NNOPS_EXPECT_NEAR(out_data[6], 7.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[7], 8.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[8], 9.0f, 1e-6f);
}

NNOPS_TEST(concat_2d_axis1) {
    // Concat along column dim: [2,3] + [2,2] → [2,5]
    const int64_t shape_a[] = {2, 3};
    const int64_t shape_b[] = {2, 2};
    float a_data[] = {1.0f, 2.0f, 3.0f,
                      4.0f, 5.0f, 6.0f};
    float b_data[] = {7.0f, 8.0f,
                      9.0f, 10.0f};
    float out_data[10] = {};

    TensorView a(shape_a, DataType::f32, a_data);
    TensorView b(shape_b, DataType::f32, b_data);

    ConcatAttributes attrs;
    attrs.axis = 1;
    auto op = Concat::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 5);

    auto output = test::make_planar(descs[0], out_data);
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    // Row 0: 1,2,3,7,8
    NNOPS_EXPECT_NEAR(out_data[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[3], 7.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[4], 8.0f, 1e-6f);
    // Row 1: 4,5,6,9,10
    NNOPS_EXPECT_NEAR(out_data[5], 4.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[6], 5.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[7], 6.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[8], 9.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[9], 10.0f, 1e-6f);
}

NNOPS_TEST(concat_3d_axis1) {
    // Concat along channel dim: [1,2,3] + [1,3,3] → [1,5,3]
    const int64_t shape_a[] = {1, 2, 3};
    const int64_t shape_b[] = {1, 3, 3};
    // A: [a0,a1,a2, a3,a4,a5]
    // B: [b0,b1,b2, b3,b4,b5, b6,b7,b8]
    float a_data[] = {1.0f, 2.0f, 3.0f,
                      4.0f, 5.0f, 6.0f};
    float b_data[] = {7.0f, 8.0f, 9.0f,
                      10.0f, 11.0f, 12.0f,
                      13.0f, 14.0f, 15.0f};
    float out_data[15] = {};

    TensorView a(shape_a, DataType::f32, a_data);
    TensorView b(shape_b, DataType::f32, b_data);

    ConcatAttributes attrs;
    attrs.axis = 1;
    auto op = Concat::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 3);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 1);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 5);
    NNOPS_EXPECT_EQ(descs[0].dims[2], 3);

    auto output = test::make_planar(descs[0], out_data);
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    // Channel 0 (from A): 1,2,3
    NNOPS_EXPECT_NEAR(out_data[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 3.0f, 1e-6f);
    // Channel 1 (from A): 4,5,6
    NNOPS_EXPECT_NEAR(out_data[3], 4.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[4], 5.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[5], 6.0f, 1e-6f);
    // Channel 2 (from B): 7,8,9
    NNOPS_EXPECT_NEAR(out_data[6], 7.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[7], 8.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[8], 9.0f, 1e-6f);
    // Channel 3 (from B): 10,11,12
    NNOPS_EXPECT_NEAR(out_data[9], 10.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[10], 11.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[11], 12.0f, 1e-6f);
    // Channel 4 (from B): 13,14,15
    NNOPS_EXPECT_NEAR(out_data[12], 13.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[13], 14.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[14], 15.0f, 1e-6f);
}

NNOPS_TEST(concat_three_inputs) {
    // Three inputs: [2] + [3] + [2] → [7]
    float a_data[] = {1.0f, 2.0f};
    float b_data[] = {3.0f, 4.0f, 5.0f};
    float c_data[] = {6.0f, 7.0f};
    float out_data[7] = {};

    const int64_t shape_a[] = {2};
    const int64_t shape_b[] = {3};
    const int64_t shape_c[] = {2};

    TensorView a(shape_a, DataType::f32, a_data);
    TensorView b(shape_b, DataType::f32, b_data);
    TensorView c(shape_c, DataType::f32, c_data);

    ConcatAttributes attrs;
    attrs.axis = 0;
    auto op = Concat::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    auto d_c = c.desc();
    const TensorDesc desc_arr[] = {d_a, d_b, d_c};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].dims[0], 7);

    auto output = test::make_planar(descs[0], out_data);
    const TensorView ins[] = {a, b, c};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_data[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[3], 4.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[4], 5.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[5], 6.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[6], 7.0f, 1e-6f);
}

NNOPS_TEST(concat_negative_axis) {
    // axis=-1 means last axis (same as axis=1 for 2D)
    const int64_t shape[] = {2, 3};
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b_data[] = {7.0f, 8.0f, 9.0f, 10.0f};
    float out_data[10] = {};

    const int64_t shape_b[] = {2, 2};
    TensorView a(shape, DataType::f32, a_data);
    TensorView b(shape_b, DataType::f32, b_data);

    ConcatAttributes attrs;
    attrs.axis = -1;  // last axis
    auto op = Concat::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 5);

    auto output = test::make_planar(descs[0], out_data);
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    // Row 0: 1,2,3,7,8
    NNOPS_EXPECT_NEAR(out_data[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[1], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[2], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[3], 7.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[4], 8.0f, 1e-6f);
}

NNOPS_TEST(concat_nchw_axis1) {
    // NCHW [1,2,2,2] + [1,3,2,2] → [1,5,2,2]
    const int64_t shape_a[] = {1, 2, 2, 2};
    const int64_t shape_b[] = {1, 3, 2, 2};

    // A: 8 elements = 2 channels × 4 spatial each
    float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f,     // ch0
                      5.0f, 6.0f, 7.0f, 8.0f};    // ch1
    // B: 12 elements = 3 channels × 4 spatial each
    float b_data[] = {9.0f, 10.0f, 11.0f, 12.0f,   // ch2
                      13.0f, 14.0f, 15.0f, 16.0f,  // ch3
                      17.0f, 18.0f, 19.0f, 20.0f}; // ch4
    float out_data[20] = {};

    TensorView a(shape_a, DataType::f32, a_data);
    TensorView b(shape_b, DataType::f32, b_data);

    ConcatAttributes attrs;
    attrs.axis = 1;
    auto op = Concat::create(attrs, Backend::CPU);

    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 4);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 5);

    auto output = test::make_planar(descs[0], out_data);
    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    // Check channel 0 (1,2,3,4)
    NNOPS_EXPECT_NEAR(out_data[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[3], 4.0f, 1e-6f);
    // Check channel 1 (5,6,7,8)
    NNOPS_EXPECT_NEAR(out_data[4], 5.0f, 1e-6f);
    // Check channel 4 (17,18,19,20)
    NNOPS_EXPECT_NEAR(out_data[16], 17.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_data[19], 20.0f, 1e-6f);
}

NNOPS_TEST(concat_random_2d_ref_vs_simd) {
    // Verify class API and functional API produce identical results
    const int64_t shape_a[] = {4, 7};
    const int64_t shape_b[] = {4, 5};

    auto [a_data, a] = test::make_random_tensor({4, 7}, -1.0f, 1.0f, 42);
    auto [b_data, b] = test::make_random_tensor({4, 5}, -1.0f, 1.0f, 99);

    const int64_t total_out = 4 * 12;
    std::vector<float> out_simd(total_out, 0.0f);
    std::vector<float> out_ref(total_out, 0.0f);

    ConcatAttributes attrs;
    attrs.axis = 1;

    // Class API
    auto op = Concat::create(attrs, Backend::CPU);
    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    auto output1 = test::make_planar(descs[0], out_simd.data());
    const TensorView ins[] = {a, b};
    op->compute(output1, ins);

    // Functional API
    auto output2 = test::make_planar(descs[0], out_ref.data());
    concat(ins, output2, attrs);

    // Compare
    for (int64_t i = 0; i < total_out; ++i) {
        NNOPS_EXPECT_NEAR(out_simd[i], out_ref[i], 1e-6f);
    }
}

NNOPS_TEST(concat_op_type) {
    auto op = Concat::create(Backend::CPU);
    NNOPS_EXPECT_EQ(static_cast<int>(op->getOpType()), static_cast<int>(OpType::Concat));
}

// ============================================================
// Packed layout tests (NCHWC8 / NCDHWC8)
// ============================================================

NNOPS_TEST(concat_nchwc8_axis0) {
    // Concat on N axis with NCHWC8: [2, 3, 4, 4] + [1, 3, 4, 4] → [3, 3, 4, 4]
    test_packed_concat({{2, 3, 4, 4}, {1, 3, 4, 4}}, 0);
}

NNOPS_TEST(concat_nchwc8_axis1_c3_c5) {
    // Concat on C axis: 3ch + 5ch → 8ch (exactly 1 C8 block)
    test_packed_concat({{1, 3, 4, 4}, {1, 5, 4, 4}}, 1);
}

NNOPS_TEST(concat_nchwc8_axis1_c7_c5) {
    // C axis: 7ch + 5ch → 12ch (1 full + 1 partial C8 block)
    test_packed_concat({{1, 7, 4, 4}, {1, 5, 4, 4}}, 1);
}

NNOPS_TEST(concat_nchwc8_axis1_c10_c6) {
    // C axis: 10ch + 6ch → 16ch (exactly 2 C8 blocks)
    test_packed_concat({{1, 10, 4, 4}, {1, 6, 4, 4}}, 1);
}

NNOPS_TEST(concat_nchwc8_axis1_c2_c3_c3) {
    // Three inputs on C axis: 2ch + 3ch + 3ch → 8ch
    test_packed_concat({{1, 2, 4, 4}, {1, 3, 4, 4}, {1, 3, 4, 4}}, 1);
}

NNOPS_TEST(concat_nchwc8_axis1_c1_c1) {
    // Small C: 1 + 1 → 2 (partial C8 block)
    test_packed_concat({{1, 1, 8, 8}, {1, 1, 8, 8}}, 1);
}

NNOPS_TEST(concat_nchwc8_axis1_multibatch) {
    // Multi-batch C-axis concat: N=2
    test_packed_concat({{2, 5, 4, 4}, {2, 3, 4, 4}}, 1);
}

NNOPS_TEST(concat_nchwc8_axis2_H) {
    // Concat on H axis with NCHWC8: [1, 3, 4, 4] + [1, 3, 6, 4] → [1, 3, 10, 4]
    test_packed_concat({{1, 3, 4, 4}, {1, 3, 6, 4}}, 2);
}

NNOPS_TEST(concat_nchwc8_axis3_W) {
    // Concat on W axis with NCHWC8: [1, 3, 4, 6] + [1, 3, 4, 4] → [1, 3, 4, 10]
    test_packed_concat({{1, 3, 4, 6}, {1, 3, 4, 4}}, 3);
}

NNOPS_TEST(concat_nchwc8_axis2_H_multi_c8) {
    // H-axis concat with multiple C8 blocks (C=10 → 2 C8 blocks)
    test_packed_concat({{1, 10, 4, 4}, {1, 10, 6, 4}}, 2);
}

NNOPS_TEST(concat_nchwc8_axis0_multi_c8) {
    // N-axis concat with multiple C8 blocks
    test_packed_concat({{2, 10, 4, 4}, {1, 10, 4, 4}}, 0);
}

// ============================================================
// 3D packed layout tests (NCDHWC8)
// ============================================================

NNOPS_TEST(concat_ncdhwc8_axis1_C) {
    // C-axis concat: [1, 3, 4, 4, 4] + [1, 5, 4, 4, 4] → [1, 8, 4, 4, 4]
    test_packed_concat({{1, 3, 4, 4, 4}, {1, 5, 4, 4, 4}}, 1);
}

NNOPS_TEST(concat_ncdhwc8_axis1_C_multi) {
    // C-axis concat with multiple C8 blocks (10ch + 8ch → 18ch)
    test_packed_concat({{1, 10, 2, 4, 4}, {1, 8, 2, 4, 4}}, 1);
}

NNOPS_TEST(concat_ncdhwc8_axis2_D) {
    // D-axis concat: [1, 3, 2, 4, 4] + [1, 3, 3, 4, 4] → [1, 3, 5, 4, 4]
    test_packed_concat({{1, 3, 2, 4, 4}, {1, 3, 3, 4, 4}}, 2);
}

NNOPS_TEST(concat_ncdhwc8_axis3_H) {
    // H-axis concat
    test_packed_concat({{1, 3, 2, 4, 4}, {1, 3, 2, 6, 4}}, 3);
}

NNOPS_TEST(concat_ncdhwc8_axis4_W) {
    // W-axis concat
    test_packed_concat({{1, 3, 2, 4, 6}, {1, 3, 2, 4, 4}}, 4);
}
