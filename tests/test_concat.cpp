/// @file test_concat.cpp
/// @brief Unit tests for Concat operator (CPU backend).

#include "nnops/ops/concat.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"

#include <vector>
#include <cstring>

using namespace nnops;

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
