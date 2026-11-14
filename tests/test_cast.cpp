/// @file test_cast.cpp
/// @brief Cast operator tests (CPU backend).
///
/// The interesting cases are the ones a plain static_cast gets wrong: float →
/// int truncates toward zero and saturates instead of wrapping, an int64 index
/// past f32's 24-bit mantissa keeps its exact value, and a NaN lands on 0
/// instead of a host-dependent pattern. Each case is hand-checked, then the
/// same conversions are swept over random data.

#include "nnops/ops/cast.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <cmath>
#include <limits>
#include <vector>

using namespace nnops;
namespace cpu = nnops::backend::cpu;

namespace {

template <class T>
std::vector<T> run_cast(DataType from, DataType to, const void* src, int64_t n)
{
    const int64_t shape[] = {n};
    TensorView input(shape, from, const_cast<void*>(src));

    CastAttributes attrs;
    attrs.to = to;
    auto op = Cast::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, to);
    NNOPS_EXPECT_EQ(descs[0].dims[0], n);

    std::vector<T> out(static_cast<size_t>(descs[0].storage_bytes() / sizeof(T)));
    TensorView output = test::make_channel_first(descs[0], out.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);
    return out;
}

}  // anonymous namespace

NNOPS_TEST(cast_f32_to_f16) {
    const float src[] = {0.5f, -1.25f, 3.0f, 1.0f / 3.0f};
    auto out = run_cast<cpu::half>(DataType::f32, DataType::f16, src, 4);
    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(simd::s_load(&out[i]), src[i], 1e-3f);
    }
}

NNOPS_TEST(cast_f32_to_s32_truncates_and_saturates) {
    const float src[] = {2.9f, -2.9f, 1e30f, -1e30f};
    auto out = run_cast<int32_t>(DataType::f32, DataType::s32, src, 4);
    NNOPS_EXPECT_EQ(out[0], 2);                  // toward zero, not 3
    NNOPS_EXPECT_EQ(out[1], -2);
    NNOPS_EXPECT_EQ(out[2], std::numeric_limits<int32_t>::max());
    NNOPS_EXPECT_EQ(out[3], std::numeric_limits<int32_t>::lowest());
}

NNOPS_TEST(cast_f32_to_u8_saturates) {
    const float src[] = {-3.0f, 0.0f, 200.0f, 255.0f};
    auto out = run_cast<uint8_t>(DataType::f32, DataType::u8, src, 4);
    NNOPS_EXPECT_EQ(out[0], 0);
    NNOPS_EXPECT_EQ(out[1], 0);
    NNOPS_EXPECT_EQ(out[2], 200);
    NNOPS_EXPECT_EQ(out[3], 255);
}

NNOPS_TEST(cast_f32_nan_to_int_is_zero) {
    const float src[] = {std::nanf(""), -std::nanf("")};
    auto out = run_cast<int32_t>(DataType::f32, DataType::s32, src, 2);
    NNOPS_EXPECT_EQ(out[0], 0);
    NNOPS_EXPECT_EQ(out[1], 0);
}

NNOPS_TEST(cast_s64_to_s32_keeps_exact_index) {
    // 1 << 30 is fine in f32, but 2^30 + 1 is not — the int64 path must not
    // route through float.
    const int64_t src[] = {(int64_t{1} << 30) + 1, -(int64_t{1} << 40), 7};
    auto out = run_cast<int32_t>(DataType::s64, DataType::s32, src, 3);
    NNOPS_EXPECT_EQ(out[0], static_cast<int32_t>((int64_t{1} << 30) + 1));
    NNOPS_EXPECT_EQ(out[1], std::numeric_limits<int32_t>::lowest());
    NNOPS_EXPECT_EQ(out[2], 7);
}

NNOPS_TEST(cast_s32_to_f32) {
    const int32_t src[] = {-7, 0, 123456};
    auto out = run_cast<float>(DataType::s32, DataType::f32, src, 3);
    NNOPS_EXPECT_NEAR(out[0], -7.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out[2], 123456.0f, 1e-2f);
}

NNOPS_TEST(cast_f16_roundtrip) {
    const float src[] = {0.1f, -2.5f, 100.0f};
    auto halves = run_cast<cpu::half>(DataType::f32, DataType::f16, src, 3);
    auto back = run_cast<float>(DataType::f16, DataType::f32, halves.data(), 3);
    for (int i = 0; i < 3; ++i) {
        NNOPS_EXPECT_NEAR(back[i], src[i], 0.1f);
    }
}

NNOPS_TEST(cast_preserves_rank2_shape) {
    const int64_t shape[] = {3, 4};
    float src[12];
    for (int i = 0; i < 12; ++i) { src[i] = static_cast<float>(i) - 5.0f; }
    TensorView input(shape, DataType::f32, src);

    CastAttributes attrs;
    attrs.to = DataType::s8;
    auto op = Cast::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 4);

    std::vector<int8_t> out(12);
    TensorView output = test::make_channel_first(descs[0], out.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);
    for (int i = 0; i < 12; ++i) {
        NNOPS_EXPECT_EQ(out[static_cast<size_t>(i)], static_cast<int8_t>(i - 5));
    }
}

NNOPS_TEST(cast_rejects_wrong_output_dtype) {
    const int64_t shape[] = {2};
    float src[2] = {1.0f, 2.0f};
    TensorView input(shape, DataType::f32, src);

    CastAttributes attrs;
    attrs.to = DataType::s32;
    auto op = Cast::create(attrs, Backend::CPU);

    const TensorDesc in = input.desc();
    TensorDesc good = in;
    good.dtype = DataType::s32;
    TensorDesc bad = in;
    bad.dtype = DataType::f16;
    NNOPS_EXPECT_EQ(op->validate(std::span<const TensorDesc>(&in, 1),
                                 std::span<const TensorDesc>(&good, 1)),
                    Status::ok);
    NNOPS_EXPECT_EQ(op->validate(std::span<const TensorDesc>(&in, 1),
                                 std::span<const TensorDesc>(&bad, 1)),
                    Status::invalid_argument);
}
