/// @file test_concat.cpp
/// @brief Unit tests for Concat (CPU), planar NCHW/NCDHW and channel-last NHWC/NDHWC.
///
/// The channel-last cases are a convert → concat → convert-back roundtrip
/// compared against concat_ref on the planar inputs; the C-axis ones exercise
/// the interleaved-channel axis merge.

#include "nnops/ops/concat.hpp"
#include "nnops/ops/layout_convert.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"

#include <vector>
#include <cstring>

using namespace nnops;

// Reference kernel, declared test-locally (not exported through a header).
namespace nnops::backend::cpu::reference {
    void concat_ref(const ConcatAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* workspace);
}

namespace {

/// Convert → concat → convert-back, compared against the planar reference concat.
void test_channel_last_concat(const std::vector<std::vector<int64_t>>& in_shapes,
                               int64_t axis,
                               DataType dtype = DataType::f32)
{
    const int64_t N = static_cast<int64_t>(in_shapes.size());
    NNOPS_ASSERT(N >= 2);

    const int64_t rank = static_cast<int64_t>(in_shapes[0].size());
    const int64_t srank = rank - 2;
    const TensorLayout planar_layout = (srank == 3) ? TensorLayout::NCDHW : TensorLayout::NCHW;
    const TensorLayout cl_layout = (srank == 3) ? TensorLayout::NDHWC : TensorLayout::NHWC;

    std::vector<std::vector<float>> in_bufs(N);
    std::vector<TensorView> in_planars(N);
    for (int64_t i = 0; i < N; ++i) {
        auto [buf, tv] = test::make_random_tensor(in_shapes[static_cast<size_t>(i)], -1.0f, 1.0f,
                                                   42 + static_cast<int>(i));
        in_bufs[static_cast<size_t>(i)] = std::move(buf);
        in_planars[static_cast<size_t>(i)] = tv;
    }

    // Golden: concat_ref on the planar inputs.
    ConcatAttributes attrs;
    attrs.axis = axis;

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

    // Channel-last path: convert each input, concat, convert back.
    std::vector<std::vector<char>> cl_bufs(N);
    std::vector<TensorView> in_cls(N);
    for (int64_t i = 0; i < N; ++i) {
        auto to_cl = LayoutConvert::create(cl_layout, Backend::CPU);
        auto d = in_planars[static_cast<size_t>(i)].desc();
        const TensorDesc cl_in[] = {d};
        auto cl_descs = to_cl->getOutputTensorDesc(cl_in);

        cl_bufs[static_cast<size_t>(i)].resize(
            static_cast<size_t>(cl_descs[0].storage_bytes()));
        in_cls[static_cast<size_t>(i)] =
            test::make_packed(cl_descs[0], cl_bufs[static_cast<size_t>(i)].data());
        const TensorView cl_ins[] = {in_planars[static_cast<size_t>(i)]};
        to_cl->compute(in_cls[static_cast<size_t>(i)], cl_ins);
    }

    auto concat_op = Concat::create(attrs, Backend::CPU);
    std::vector<TensorDesc> cl_descs_in(N);
    for (int64_t i = 0; i < N; ++i) {
        cl_descs_in[static_cast<size_t>(i)] = in_cls[static_cast<size_t>(i)].desc();
    }
    auto concat_descs = concat_op->getOutputTensorDesc(cl_descs_in);

    NNOPS_EXPECT_EQ(concat_descs[0].rank, rank);
    NNOPS_EXPECT_EQ(concat_descs[0].layout, cl_layout);
    NNOPS_EXPECT_EQ(concat_descs[0].dtype, dtype);

    std::vector<char> concat_buf(static_cast<size_t>(concat_descs[0].storage_bytes()));
    auto out_cl = test::make_packed(concat_descs[0], concat_buf.data());

    std::vector<TensorView> concat_ins(N);
    for (int64_t i = 0; i < N; ++i) {
        concat_ins[static_cast<size_t>(i)] = in_cls[static_cast<size_t>(i)];
    }
    concat_op->compute(out_cl, concat_ins);

    auto to_planar = LayoutConvert::create(planar_layout, Backend::CPU);
    auto od = out_cl.desc();
    const TensorDesc back_in[] = {od};
    auto back_descs = to_planar->getOutputTensorDesc(back_in);

    NNOPS_EXPECT_EQ(back_descs[0].rank, rank);
    NNOPS_EXPECT_EQ(back_descs[0].layout, planar_layout);
    NNOPS_EXPECT_EQ(back_descs[0].dtype, dtype);

    std::vector<char> back_buf(static_cast<size_t>(back_descs[0].storage_bytes()));
    auto res_planar = test::make_planar(back_descs[0], back_buf.data());
    const TensorView back_ins[] = {out_cl};
    to_planar->compute(res_planar, back_ins);

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
    attrs.axis = -1;
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
    // One op instance, two output buffers: the two runs must agree.
    const int64_t shape_a[] = {4, 7};
    const int64_t shape_b[] = {4, 5};

    auto [a_data, a] = test::make_random_tensor({4, 7}, -1.0f, 1.0f, 42);
    auto [b_data, b] = test::make_random_tensor({4, 5}, -1.0f, 1.0f, 99);

    const int64_t total_out = 4 * 12;
    std::vector<float> out_simd(total_out, 0.0f);
    std::vector<float> out_ref(total_out, 0.0f);

    ConcatAttributes attrs;
    attrs.axis = 1;

    auto op = Concat::create(attrs, Backend::CPU);
    auto d_a = a.desc();
    auto d_b = b.desc();
    const TensorDesc desc_arr[] = {d_a, d_b};
    auto descs = op->getOutputTensorDesc(desc_arr);

    auto output1 = test::make_planar(descs[0], out_simd.data());
    const TensorView ins[] = {a, b};
    op->compute(output1, ins);

    auto output2 = test::make_planar(descs[0], out_ref.data());
    op->compute(output2, ins);

    for (int64_t i = 0; i < total_out; ++i) {
        NNOPS_EXPECT_NEAR(out_simd[i], out_ref[i], 1e-6f);
    }
}

NNOPS_TEST(concat_op_type) {
    auto op = Concat::create(Backend::CPU);
    NNOPS_EXPECT_EQ(static_cast<int>(op->getOpType()), static_cast<int>(OpType::Concat));
}

// ============================================================
// Channel-last layout tests (NHWC / NDHWC)
// ============================================================

NNOPS_TEST(concat_nhwc_axis0) {
    // Concat on N axis with NHWC: [2, 3, 4, 4] + [1, 3, 4, 4] -> [3, 3, 4, 4]
    test_channel_last_concat({{2, 3, 4, 4}, {1, 3, 4, 4}}, 0);
}

NNOPS_TEST(concat_nhwc_axis1_c3_c5) {
    // Concat on C axis: 3ch + 5ch → 8ch (single channel block)
    test_channel_last_concat({{1, 3, 4, 4}, {1, 5, 4, 4}}, 1);
}

NNOPS_TEST(concat_nhwc_axis1_c7_c5) {
    // C axis: 7ch + 5ch → 12ch 
    test_channel_last_concat({{1, 7, 4, 4}, {1, 5, 4, 4}}, 1);
}

NNOPS_TEST(concat_nhwc_axis1_c10_c6) {
    // C axis: 10ch + 6ch → 16ch 
    test_channel_last_concat({{1, 10, 4, 4}, {1, 6, 4, 4}}, 1);
}

NNOPS_TEST(concat_nhwc_axis1_c2_c3_c3) {
    // Three inputs on C axis: 2ch + 3ch + 3ch → 8ch
    test_channel_last_concat({{1, 2, 4, 4}, {1, 3, 4, 4}, {1, 3, 4, 4}}, 1);
}

NNOPS_TEST(concat_nhwc_axis1_c1_c1) {
    // Small C: 1 + 1 → 2 
    test_channel_last_concat({{1, 1, 8, 8}, {1, 1, 8, 8}}, 1);
}

NNOPS_TEST(concat_nhwc_axis1_multibatch) {
    // Multi-batch C-axis concat: N=2
    test_channel_last_concat({{2, 5, 4, 4}, {2, 3, 4, 4}}, 1);
}

NNOPS_TEST(concat_nhwc_axis2_H) {
    // Concat on H axis with NHWC: [1, 3, 4, 4] + [1, 3, 6, 4] -> [1, 3, 10, 4]
    test_channel_last_concat({{1, 3, 4, 4}, {1, 3, 6, 4}}, 2);
}

NNOPS_TEST(concat_nhwc_axis3_W) {
    // Concat on W axis with NHWC: [1, 3, 4, 6] + [1, 3, 4, 4] -> [1, 3, 4, 10]
    test_channel_last_concat({{1, 3, 4, 6}, {1, 3, 4, 4}}, 3);
}

NNOPS_TEST(concat_nhwc_axis2_H_multi_c8) {
    // H-axis concat with multiple channel planes (C=10)
    test_channel_last_concat({{1, 10, 4, 4}, {1, 10, 6, 4}}, 2);
}

NNOPS_TEST(concat_nhwc_axis0_multi_c8) {
    // N-axis concat with multiple channel planes
    test_channel_last_concat({{2, 10, 4, 4}, {1, 10, 4, 4}}, 0);
}

// ============================================================
// 3D channel-last layout tests (NDHWC)
// ============================================================

NNOPS_TEST(concat_ndhwc_axis1_C) {
    // C-axis concat: [1, 3, 4, 4, 4] + [1, 5, 4, 4, 4] → [1, 8, 4, 4, 4]
    test_channel_last_concat({{1, 3, 4, 4, 4}, {1, 5, 4, 4, 4}}, 1);
}

NNOPS_TEST(concat_ndhwc_axis1_C_multi) {
    // C-axis concat with multiple channel planes (10ch + 8ch → 18ch)
    test_channel_last_concat({{1, 10, 2, 4, 4}, {1, 8, 2, 4, 4}}, 1);
}

NNOPS_TEST(concat_ndhwc_axis2_D) {
    // D-axis concat: [1, 3, 2, 4, 4] + [1, 3, 3, 4, 4] → [1, 3, 5, 4, 4]
    test_channel_last_concat({{1, 3, 2, 4, 4}, {1, 3, 3, 4, 4}}, 2);
}

NNOPS_TEST(concat_ndhwc_axis3_H) {
    // H-axis concat
    test_channel_last_concat({{1, 3, 2, 4, 4}, {1, 3, 2, 6, 4}}, 3);
}

NNOPS_TEST(concat_ndhwc_axis4_W) {
    // W-axis concat
    test_channel_last_concat({{1, 3, 2, 4, 6}, {1, 3, 2, 4, 4}}, 4);
}
