/// @file test_where.cpp
/// @brief Where / Compare / Logical tests — broadcast selects, every relation,
/// the logic gates, and the u8 result convention.
///
/// A condition byte is true when nonzero, so u8 and s8 share one test. Compare
/// accepts f32/f16/s32/s64 and Logical s8/u8/s32/s64; both emit u8 0/1 and
/// broadcast their operands. NaN compares false in every relation, including
/// Equal against another NaN.

#include "nnops/ops/where.hpp"
#include "nnops/ops/compare.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <vector>
#include <limits>

using namespace nnops;

NNOPS_TEST(where_basic) {
    uint8_t cond[] = {1, 0, 1, 0};
    const int64_t c_shape[] = {4};
    TensorView cv(c_shape, DataType::u8, cond);

    float x[] = {1, 2, 3, 4};
    float y[] = {10, 20, 30, 40};
    TensorView xv(c_shape, DataType::f32, x);
    TensorView yv(c_shape, DataType::f32, y);

    auto op = Where::create();
    const TensorDesc in_arr[] = {cv.desc(), xv.desc(), yv.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 4);

    float out[4] = {};
    TensorView ov = test::make_channel_first(descs[0], out);
    const TensorView ins[] = {cv, xv, yv};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    const float want[4] = {1, 20, 3, 40};
    for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_NEAR(out[i], want[i], 1e-6f); }
}

NNOPS_TEST(where_broadcast_condition) {
    uint8_t cond[] = {1, 0};
    const int64_t c_shape[] = {2, 1};
    TensorView cv(c_shape, DataType::u8, cond);

    float x[] = {1, 2, 3, 4, 5, 6};
    const int64_t d_shape[] = {2, 3};
    TensorView xv(d_shape, DataType::f32, x);
    float y[] = {10, 20, 30, 40, 50, 60};
    TensorView yv(d_shape, DataType::f32, y);

    auto op = Where::create();
    const TensorDesc in_arr[] = {cv.desc(), xv.desc(), yv.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 3);

    float out[6] = {};
    TensorView ov = test::make_channel_first(descs[0], out);
    const TensorView ins[] = {cv, xv, yv};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    const float want[6] = {1, 2, 3, 40, 50, 60};
    for (int i = 0; i < 6; ++i) { NNOPS_EXPECT_NEAR(out[i], want[i], 1e-6f); }
}

NNOPS_TEST(where_f16_exact) {
    using nnops::backend::cpu::half;
    using nnops::backend::cpu::half_to_bits;
    uint8_t cond[] = {0, 1};
    const int64_t c_shape[] = {2};
    TensorView cv(c_shape, DataType::u8, cond);

    half x[2], y[2];
    simd::s_store(&x[0], 1.5f);
    simd::s_store(&x[1], 2.5f);
    simd::s_store(&y[0], -1.5f);
    simd::s_store(&y[1], -2.5f);
    TensorView xv(c_shape, DataType::f16, x);
    TensorView yv(c_shape, DataType::f16, y);

    auto op = Where::create();
    const TensorDesc in_arr[] = {cv.desc(), xv.desc(), yv.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

    half out[2];
    TensorView ov = test::make_channel_first(descs[0], out);
    const TensorView ins[] = {cv, xv, yv};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    NNOPS_EXPECT_EQ(half_to_bits(out[0]), half_to_bits(y[0]));
    NNOPS_EXPECT_EQ(half_to_bits(out[1]), half_to_bits(x[1]));
}

NNOPS_TEST(where_rejects_bad_operands) {
    float cond[] = {1, 0};  // f32 is not a condition
    const int64_t shape[] = {2};
    TensorView cv(shape, DataType::f32, cond);
    float a[] = {1, 2};
    TensorView av(shape, DataType::f32, a);
    using nnops::backend::cpu::half;
    half h[2] = {};
    TensorView hv(shape, DataType::f16, h);

    auto op = Where::create();
    const TensorDesc bad_cond[] = {cv.desc(), av.desc(), av.desc()};
    const TensorDesc out_desc[] = {av.desc()};
    NNOPS_EXPECT_TRUE(op->validate(bad_cond, out_desc) == Status::unsupported);

    uint8_t c8[] = {1, 0};
    TensorView c8v(shape, DataType::u8, c8);
    const TensorDesc bad_branch[] = {c8v.desc(), av.desc(), hv.desc()};
    NNOPS_EXPECT_TRUE(op->validate(bad_branch, out_desc) == Status::invalid_argument);
}

namespace {

void run_compare(CompareType type, const float* a, const float* b, size_t n,
                 uint8_t* out) {
    const int64_t shape[] = {static_cast<int64_t>(n)};
    TensorView av(shape, DataType::f32, const_cast<float*>(a));
    TensorView bv(shape, DataType::f32, const_cast<float*>(b));

    CompareAttributes attrs;
    attrs.type = type;
    auto op = Compare::create(attrs);
    const TensorDesc in_arr[] = {av.desc(), bv.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    TensorView ov = test::make_channel_first(descs[0], out);
    const TensorView ins[] = {av, bv};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);
}

}  // namespace

NNOPS_TEST(compare_relations) {
    const float a[4] = {1, 2, 3, 4};
    const float b[4] = {2, 2, 2, 2};
    uint8_t out[4];

    run_compare(CompareType::Equal, a, b, 4, out);
    const uint8_t eq[4] = {0, 1, 0, 0};
    for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_EQ(out[i], eq[i]); }

    run_compare(CompareType::Greater, a, b, 4, out);
    const uint8_t gt[4] = {0, 0, 1, 1};
    for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_EQ(out[i], gt[i]); }

    run_compare(CompareType::Less, a, b, 4, out);
    const uint8_t lt[4] = {1, 0, 0, 0};
    for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_EQ(out[i], lt[i]); }

    run_compare(CompareType::GreaterOrEqual, a, b, 4, out);
    const uint8_t ge[4] = {0, 1, 1, 1};
    for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_EQ(out[i], ge[i]); }

    run_compare(CompareType::LessOrEqual, a, b, 4, out);
    const uint8_t le[4] = {1, 1, 0, 0};
    for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_EQ(out[i], le[i]); }
}

NNOPS_TEST(compare_broadcast_and_f16) {
    using nnops::backend::cpu::half;
    half h[3];
    simd::s_store(&h[0], 1.0f);
    simd::s_store(&h[1], 2.0f);
    simd::s_store(&h[2], 3.0f);
    const int64_t h_shape[] = {3};
    TensorView hv(h_shape, DataType::f16, h);

    half t;
    simd::s_store(&t, 1.5f);
    const int64_t t_shape[] = {1};
    TensorView tv(t_shape, DataType::f16, &t);

    CompareAttributes attrs;
    attrs.type = CompareType::Greater;
    auto op = Compare::create(attrs);
    const TensorDesc in_arr[] = {hv.desc(), tv.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);

    uint8_t out[3] = {};
    TensorView ov = test::make_channel_first(descs[0], out);
    const TensorView ins[] = {hv, tv};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    const uint8_t want[3] = {0, 1, 1};
    for (int i = 0; i < 3; ++i) { NNOPS_EXPECT_EQ(out[i], want[i]); }
}

NNOPS_TEST(compare_nan) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float a[2] = {nan, 1.0f};
    const float b[2] = {nan, nan};
    uint8_t out[2];

    run_compare(CompareType::Equal, a, b, 2, out);
    NNOPS_EXPECT_EQ(out[0], uint8_t(0));
    NNOPS_EXPECT_EQ(out[1], uint8_t(0));

    run_compare(CompareType::Greater, a, b, 2, out);
    NNOPS_EXPECT_EQ(out[0], uint8_t(0));
    NNOPS_EXPECT_EQ(out[1], uint8_t(0));

    run_compare(CompareType::LessOrEqual, a, b, 2, out);
    NNOPS_EXPECT_EQ(out[0], uint8_t(0));
    NNOPS_EXPECT_EQ(out[1], uint8_t(0));
}

NNOPS_TEST(logical_gates) {
    uint8_t a[] = {1, 0, 2, 0};
    uint8_t b[] = {1, 1, 0, 0};
    const int64_t shape[] = {4};
    TensorView av(shape, DataType::u8, a);
    TensorView bv(shape, DataType::u8, b);

    auto run = [&](LogicalType type, const uint8_t* want) {
        LogicalAttributes attrs;
        attrs.type = type;
        auto op = Logical::create(attrs);
        const TensorDesc in_arr[] = {av.desc(), bv.desc()};
        auto descs = op->getOutputTensorDesc(in_arr);
        NNOPS_EXPECT_EQ(descs[0].dtype, DataType::u8);
        uint8_t out[4] = {};
        TensorView ov = test::make_channel_first(descs[0], out);
        const TensorView ins[] = {av, bv};
        NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);
        for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_EQ(out[i], want[i]); }
    };

    const uint8_t and_want[4] = {1, 0, 0, 0};
    const uint8_t or_want[4] = {1, 1, 1, 0};
    const uint8_t xor_want[4] = {0, 1, 1, 0};
    run(LogicalType::And, and_want);
    run(LogicalType::Or, or_want);
    run(LogicalType::Xor, xor_want);
}

NNOPS_TEST(logical_not_int32) {
    int32_t a[] = {0, 5, -3};
    const int64_t shape[] = {3};
    TensorView av(shape, DataType::s32, a);

    LogicalAttributes attrs;
    attrs.type = LogicalType::Not;
    auto op = Logical::create(attrs);
    const TensorDesc in_arr[] = {av.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);

    uint8_t out[3] = {};
    TensorView ov = test::make_channel_first(descs[0], out);
    const TensorView ins[] = {av};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::ok);

    const uint8_t want[3] = {1, 0, 0};
    for (int i = 0; i < 3; ++i) { NNOPS_EXPECT_EQ(out[i], want[i]); }
}

NNOPS_TEST(logical_rejects_float) {
    float a[] = {1.0f, 0.0f};
    const int64_t shape[] = {2};
    TensorView av(shape, DataType::f32, a);
    uint8_t out[2] = {};

    LogicalAttributes attrs;
    attrs.type = LogicalType::Not;
    auto op = Logical::create(attrs);
    const TensorDesc in_arr[] = {av.desc()};
    const TensorDesc out_desc[] = {av.desc()};
    NNOPS_EXPECT_TRUE(op->validate(in_arr, out_desc) == Status::unsupported);

    TensorView ov = test::make_channel_first(av.desc(), out);
    const TensorView ins[] = {av};
    NNOPS_EXPECT_TRUE(op->compute(ov, ins) == Status::unsupported);
}

NNOPS_TEST(compare_rejects_mixed_dtypes) {
    const int64_t shape[] = {2};
    float a[] = {1, 2};
    using nnops::backend::cpu::half;
    half h[2] = {};
    TensorView av(shape, DataType::f32, a);
    TensorView hv(shape, DataType::f16, h);

    auto op = Compare::create();
    const TensorDesc in_arr[] = {av.desc(), hv.desc()};
    const TensorDesc out_desc[] = {av.desc()};
    NNOPS_EXPECT_TRUE(op->validate(in_arr, out_desc) == Status::invalid_argument);
}
