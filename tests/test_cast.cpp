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

// ---- large arrays: the element tile, the eight-lane paths, the scalar tail ----

/// Independent oracle for the ONNX rule: truncate toward zero, then saturate.
template <class Dst>
Dst ref_float_to_int(float v)
{
    if (std::isnan(v)) { return Dst{0}; }
    const double t = std::trunc(static_cast<double>(v));
    const double lo = static_cast<double>(std::numeric_limits<Dst>::lowest());
    const double hi = static_cast<double>(std::numeric_limits<Dst>::max());
    if (t <= lo) { return std::numeric_limits<Dst>::lowest(); }
    if (t >= hi) { return std::numeric_limits<Dst>::max(); }
    return static_cast<Dst>(static_cast<int64_t>(t));
}

NNOPS_TEST(cast_large_float_to_int) {
    // 200003 = 3 mod 8 and well past the 32768-element tile, so the vector
    // body, the tile split, and the scalar tail all run.
    const int64_t n = 200003;
    std::vector<float> src(static_cast<size_t>(n));
    uint32_t st = 12345u;
    for (int64_t i = 0; i < n; ++i) {
        st = st * 1664525u + 1013904223u;
        const float u = static_cast<float>(st >> 8) / 16777216.0f;
        src[static_cast<size_t>(i)] = (u - 0.5f) * 1200.0f;
    }
    const float specials[] = {std::nanf(""), INFINITY, -INFINITY, 1e30f, -1e30f,
                              0.0f, -0.0f, 127.9f, -128.9f};
    for (int k = 0; k < 9; ++k) {
        src[static_cast<size_t>(k) * 997] = specials[k];
    }

    auto i32 = run_cast<int32_t>(DataType::f32, DataType::s32, src.data(), n);
    auto i8  = run_cast<int8_t>(DataType::f32, DataType::s8, src.data(), n);
    auto u8  = run_cast<uint8_t>(DataType::f32, DataType::u8, src.data(), n);

    int64_t bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        const float v = src[static_cast<size_t>(i)];
        if (i32[static_cast<size_t>(i)] != ref_float_to_int<int32_t>(v)) { ++bad; }
        if (i8[static_cast<size_t>(i)] != ref_float_to_int<int8_t>(v)) { ++bad; }
        if (u8[static_cast<size_t>(i)] != ref_float_to_int<uint8_t>(v)) { ++bad; }
    }
    NNOPS_EXPECT_EQ(bad, 0);
}

NNOPS_TEST(cast_large_float_paths) {
    const int64_t n = 200003;
    std::vector<float> src(static_cast<size_t>(n));
    uint32_t st = 999u;
    for (int64_t i = 0; i < n; ++i) {
        st = st * 1664525u + 1013904223u;
        src[static_cast<size_t>(i)] =
            (static_cast<float>(st >> 8) / 16777216.0f - 0.5f) * 2000.0f;
    }

    auto halves = run_cast<cpu::half>(DataType::f32, DataType::f16, src.data(), n);
    auto back   = run_cast<float>(DataType::f16, DataType::f32, halves.data(), n);

    int64_t bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        // The eight-lane narrowing must round exactly where the scalar
        // conversion does, including the f16 subnormal range.
        if (halves[static_cast<size_t>(i)].bits !=
            cpu::float_to_half(src[static_cast<size_t>(i)]).bits) {
            ++bad;
        }
        if (back[static_cast<size_t>(i)] != cpu::half_to_float(halves[static_cast<size_t>(i)])) {
            ++bad;
        }
    }
    NNOPS_EXPECT_EQ(bad, 0);

    // Same-type casts are copies, and int8/uint8 → f32 is exact.
    auto copy32 = run_cast<float>(DataType::f32, DataType::f32, src.data(), n);
    auto copy16 = run_cast<cpu::half>(DataType::f16, DataType::f16, halves.data(), n);
    for (int64_t i = 0; i < n; ++i) {
        if (copy32[static_cast<size_t>(i)] != src[static_cast<size_t>(i)]) { ++bad; }
        if (copy16[static_cast<size_t>(i)].bits != halves[static_cast<size_t>(i)].bits) { ++bad; }
    }

    std::vector<int8_t>  s8(static_cast<size_t>(n));
    std::vector<uint8_t> u8(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        s8[static_cast<size_t>(i)] = static_cast<int8_t>(i % 251 - 125);
        u8[static_cast<size_t>(i)] = static_cast<uint8_t>(i % 256);
    }
    auto f32_from_s8 = run_cast<float>(DataType::s8, DataType::f32, s8.data(), n);
    auto f32_from_u8 = run_cast<float>(DataType::u8, DataType::f32, u8.data(), n);
    for (int64_t i = 0; i < n; ++i) {
        if (f32_from_s8[static_cast<size_t>(i)] != static_cast<float>(s8[static_cast<size_t>(i)])) {
            ++bad;
        }
        if (f32_from_u8[static_cast<size_t>(i)] != static_cast<float>(u8[static_cast<size_t>(i)])) {
            ++bad;
        }
    }
    NNOPS_EXPECT_EQ(bad, 0);
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
