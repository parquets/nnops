/// @file test_scatter.cpp
/// @brief ScatterElements / ScatterND tests — the copy base, the three
/// reductions, negative indices, whole-slice updates and rejection.
///
/// Both operators first copy data into the output, then write updates, so an
/// element nothing addresses keeps its input value. Reduction None overwrites;
/// Add and Mul accumulate. ScatterND writes a data[k:] slice per tuple.

#include "nnops/ops/scatter.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <vector>

using namespace nnops;

NNOPS_TEST(scatter_elements_axis0) {
    float data[] = {1, 2, 3, 4, 5, 6};
    const int64_t d_shape[] = {2, 3};
    TensorView d(d_shape, DataType::f32, data);

    int64_t idx[] = {1, 0, 1};
    const int64_t i_shape[] = {1, 3};
    TensorView indices(i_shape, DataType::s64, idx);

    float upd[] = {10, 20, 30};
    TensorView updates(i_shape, DataType::f32, upd);

    auto op = ScatterElements::create(ScatterElementsAttributes{0});
    const TensorDesc in_arr[] = {d.desc(), indices.desc(), updates.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    float out[6] = {};
    TensorView ov = test::make_planar(descs[0], out);
    const TensorView ins[] = {d, indices, updates};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    const float want[6] = {1, 20, 3, 10, 5, 30};
    for (int i = 0; i < 6; ++i) { NNOPS_EXPECT_NEAR(out[i], want[i], 1e-6f); }
}

NNOPS_TEST(scatter_elements_add_and_mul) {
    float data[] = {1, 1, 1, 1};
    const int64_t d_shape[] = {2, 2};
    TensorView d(d_shape, DataType::f32, data);

    int64_t idx[] = {0, 1, 0, 1};
    const int64_t i_shape[] = {2, 2};
    TensorView indices(i_shape, DataType::s64, idx);

    float upd[] = {10, 20, 30, 40};
    TensorView updates(i_shape, DataType::f32, upd);

    ScatterElementsAttributes attrs;
    attrs.axis = 0;
    attrs.reduction = ScatterReduction::Add;
    auto op = ScatterElements::create(attrs);

    const TensorDesc in_arr[] = {d.desc(), indices.desc(), updates.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    float out_add[4] = {};
    TensorView ov = test::make_planar(descs[0], out_add);
    const TensorView ins[] = {d, indices, updates};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    NNOPS_EXPECT_NEAR(out_add[0], 41.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_add[3], 61.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_add[1], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_add[2], 1.0f, 1e-6f);

    float base2[] = {2, 2, 2, 2};
    TensorView d2(d_shape, DataType::f32, base2);
    attrs.reduction = ScatterReduction::Mul;
    auto op_mul = ScatterElements::create(attrs);

    float out_mul[4] = {};
    TensorView ov2 = test::make_planar(descs[0], out_mul);
    const TensorView ins2[] = {d2, indices, updates};
    NNOPS_EXPECT_TRUE(op_mul->compute(ov2, ins2) == Status::ok);

    NNOPS_EXPECT_NEAR(out_mul[0], 600.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(out_mul[3], 1600.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(out_mul[1], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_mul[2], 2.0f, 1e-6f);
}

NNOPS_TEST(scatter_elements_negative_index) {
    float data[] = {1, 2, 3, 4, 5, 6};
    const int64_t d_shape[] = {2, 3};
    TensorView d(d_shape, DataType::f32, data);

    int64_t idx[] = {-1, 0};
    const int64_t i_shape[] = {2, 1};
    TensorView indices(i_shape, DataType::s64, idx);

    float upd[] = {99, 88};
    TensorView updates(i_shape, DataType::f32, upd);

    auto op = ScatterElements::create(ScatterElementsAttributes{1});
    const TensorDesc in_arr[] = {d.desc(), indices.desc(), updates.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    float out[6] = {};
    TensorView ov = test::make_planar(descs[0], out);
    const TensorView ins[] = {d, indices, updates};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    NNOPS_EXPECT_NEAR(out[2], 99.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out[3], 88.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out[0], 1.0f, 1e-6f);
}

NNOPS_TEST(scatter_elements_f16_bits) {
    using nnops::backend::cpu::half;
    using nnops::backend::cpu::half_to_bits;
    half data[3];
    const float vals[3] = {1.5f, -2.0f, 0.25f};
    for (int i = 0; i < 3; ++i) { simd::s_store(&data[i], vals[i]); }

    const int64_t d_shape[] = {3};
    TensorView d(d_shape, DataType::f16, data);

    int64_t idx[] = {0};
    const int64_t i_shape[] = {1};
    TensorView indices(i_shape, DataType::s64, idx);

    half upd[1];
    simd::s_store(&upd[0], 7.5f);
    TensorView updates(i_shape, DataType::f16, upd);

    auto op = ScatterElements::create(ScatterElementsAttributes{0});
    const TensorDesc in_arr[] = {d.desc(), indices.desc(), updates.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    half out[3];
    TensorView ov = test::make_planar(descs[0], out);
    const TensorView ins[] = {d, indices, updates};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    NNOPS_EXPECT_EQ(half_to_bits(out[0]), half_to_bits(upd[0]));
    NNOPS_EXPECT_EQ(half_to_bits(out[1]), half_to_bits(data[1]));
    NNOPS_EXPECT_EQ(half_to_bits(out[2]), half_to_bits(data[2]));
}

NNOPS_TEST(scatter_nd_element) {
    float data[] = {1, 1, 1, 1};
    const int64_t d_shape[] = {2, 2};
    TensorView d(d_shape, DataType::f32, data);

    int64_t idx[] = {0, 1};
    const int64_t i_shape[] = {1, 2};
    TensorView indices(i_shape, DataType::s64, idx);

    float upd[] = {5};
    const int64_t u_shape[] = {1};
    TensorView updates(u_shape, DataType::f32, upd);

    auto op = ScatterND::create(ScatterNDAttributes{});
    const TensorDesc in_arr[] = {d.desc(), indices.desc(), updates.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    float out[4] = {};
    TensorView ov = test::make_planar(descs[0], out);
    const TensorView ins[] = {d, indices, updates};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    const float want[4] = {1, 5, 1, 1};
    for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_NEAR(out[i], want[i], 1e-6f); }
}

NNOPS_TEST(scatter_nd_slice_add) {
    float data[] = {1, 1, 1, 1, 1, 1, 1, 1};
    const int64_t d_shape[] = {2, 2, 2};
    TensorView d(d_shape, DataType::f32, data);

    int64_t idx[] = {0};
    const int64_t i_shape[] = {1, 1};
    TensorView indices(i_shape, DataType::s64, idx);

    float upd[] = {10, 20, 30, 40};
    const int64_t u_shape[] = {1, 2, 2};
    TensorView updates(u_shape, DataType::f32, upd);

    ScatterNDAttributes attrs;
    attrs.reduction = ScatterReduction::Add;
    auto op = ScatterND::create(attrs);

    const TensorDesc in_arr[] = {d.desc(), indices.desc(), updates.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    float out[8] = {};
    TensorView ov = test::make_planar(descs[0], out);
    const TensorView ins[] = {d, indices, updates};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    const float want[8] = {11, 21, 31, 41, 1, 1, 1, 1};
    for (int i = 0; i < 8; ++i) { NNOPS_EXPECT_NEAR(out[i], want[i], 1e-6f); }
}

NNOPS_TEST(scatter_rejects_out_of_range) {
    float data[] = {1, 2, 3};
    const int64_t d_shape[] = {3};
    TensorView d(d_shape, DataType::f32, data);

    int64_t idx[] = {3};
    const int64_t i_shape[] = {1};
    TensorView indices(i_shape, DataType::s64, idx);

    float upd[] = {9};
    TensorView updates(i_shape, DataType::f32, upd);

    auto op = ScatterElements::create(ScatterElementsAttributes{0});
    const TensorDesc in_arr[] = {d.desc(), indices.desc(), updates.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);

    float out[3] = {};
    TensorView ov = test::make_planar(descs[0], out);
    const TensorView ins[] = {d, indices, updates};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::invalid_argument);

    auto op_nd = ScatterND::create(ScatterNDAttributes{});
    int64_t idx_nd[] = {5};
    const int64_t i_nd_shape[] = {1, 1};
    TensorView indices_nd(i_nd_shape, DataType::s64, idx_nd);
    const TensorDesc in_nd[] = {d.desc(), indices_nd.desc(), updates.desc()};
    auto nd_descs = op_nd->getOutputTensorDesc(in_nd);
    TensorView ov_nd = test::make_planar(nd_descs[0], out);
    const TensorView ins_nd[] = {d, indices_nd, updates};
    NNOPS_EXPECT_TRUE(op_nd->compute(ov_nd, ins_nd) == Status::invalid_argument);
}
