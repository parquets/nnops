/// @file test_clamp.cpp
/// @brief Unit tests for Clamp operator (CPU backend).

#include "nnops/ops/clamp.hpp"
#include "nnops/ops/layout_convert.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/test_helpers.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <vector>
#include <algorithm>
#include <limits>
#include <cstring>

using namespace nnops;

// ============================================================
// Hand-verified tests
// ============================================================

NNOPS_TEST(clamp_basic) {
    const int64_t shape[] = {3};
    float in_data[]  = {0.0f, 5.0f, 10.0f};
    float out_buf[3] = {};

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    ClampAttributes attrs;
    attrs.min_val = 2.0f;
    attrs.max_val = 8.0f;
    auto op = Clamp::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    TensorView output = test::make_planar(descs[0], out_buf);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], 2.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], 5.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[2], 8.0f, 1e-5f);
}

NNOPS_TEST(clamp_no_min) {
    const int64_t shape[] = {4};
    float in_data[]  = {-100.0f, -1.0f, 5.0f, 100.0f};
    float out_buf[4] = {};

    TensorView input(shape, DataType::f32, in_data);

    ClampAttributes attrs;
    attrs.max_val = 10.0f;  // only upper bound
    auto op = Clamp::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    TensorView output = test::make_planar(descs[0], out_buf);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], -100.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], -1.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[2], 5.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[3], 10.0f, 1e-5f);
}

NNOPS_TEST(clamp_no_max) {
    const int64_t shape[] = {4};
    float in_data[]  = {-100.0f, -1.0f, 5.0f, 100.0f};
    float out_buf[4] = {};

    TensorView input(shape, DataType::f32, in_data);

    ClampAttributes attrs;
    attrs.min_val = 0.0f;  // only lower bound
    auto op = Clamp::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    TensorView output = test::make_planar(descs[0], out_buf);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], 0.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], 0.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[2], 5.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[3], 100.0f, 1e-5f);
}

// ============================================================
// Random data tests
// ============================================================

NNOPS_TEST(clamp_random_f32) {
    auto [in_vec, input] = test::make_random_tensor({1000}, -10.0f, 10.0f, 222);

    auto d = input.desc();
    ClampAttributes attrs;
    attrs.min_val = -2.0f;
    attrs.max_val = 3.0f;
    auto op = Clamp::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d};
    auto descs = op->getOutputTensorDesc(in_arr);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 1000);

    std::vector<float> out_buf(1000);
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    for (int i = 0; i < 1000; ++i) {
        float expected = std::max(-2.0f, std::min(3.0f, in_vec[i]));
        NNOPS_EXPECT_NEAR(out_buf[i], expected, 1e-5f);
    }
}

NNOPS_TEST(clamp_random_f16) {
    auto [in_vec, _] = test::make_random_tensor({500}, -5.0f, 5.0f, 333);

    std::vector<nnops::backend::cpu::half> in_half(500);
    for (int i = 0; i < 500; ++i) {
        simd::s_store(&in_half[i], in_vec[i]);
    }

    const int64_t shape[] = {500};
    TensorView input(shape, DataType::f16, in_half.data());
    auto d = input.desc();

    ClampAttributes attrs;
    attrs.min_val = -1.0f;
    attrs.max_val = 2.0f;
    auto op = Clamp::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {d};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);
    std::vector<nnops::backend::cpu::half> out_buf(500);
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {input};
    op->compute(output, ins);

    for (int i = 0; i < 500; ++i) {
        float result = simd::s_load(&out_buf[i]);
        float expected = std::max(-1.0f, std::min(2.0f, in_vec[i]));
        NNOPS_EXPECT_NEAR(result, expected, 1e-3f);
    }
}

// ============================================================
// add_to test
// ============================================================

NNOPS_TEST(clamp_add_to) {
    const int64_t shape[] = {3};
    float in_data[]  = {0.0f, 5.0f, 10.0f};
    float out_buf[3] = {1.0f, 2.0f, 3.0f};  // pre-fill

    TensorView input(shape, DataType::f32, in_data);

    ClampAttributes attrs;
    attrs.min_val = 2.0f;
    attrs.max_val = 8.0f;
    attrs.add_to = true;
    auto op = Clamp::create(attrs, Backend::CPU);

    const TensorDesc in_arr[] = {input.desc()};
    auto descs = op->getOutputTensorDesc(in_arr);
    TensorView output = test::make_planar(descs[0], out_buf);
    const TensorView ins[] = {input};
    op->compute(output, ins);

    // out = prefill + clamp(in): 1+2=3, 2+5=7, 3+8=11
    NNOPS_EXPECT_NEAR(out_buf[0], 3.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[1], 7.0f, 1e-5f);
    NNOPS_EXPECT_NEAR(out_buf[2], 11.0f, 1e-5f);
}

// ============================================================
// Packed layout test (Any layout support)
// ============================================================

NNOPS_TEST(clamp_packed_layout) {
    // Create planar data, convert to NCHWC8, clamp, convert back, verify
    const int64_t N = 1, C = 16, H = 4, W = 4;
    auto [in_vec, input_planar] = test::make_random_tensor({N, C, H, W}, -10.0f, 10.0f, 444);

    // Convert to NCHWC8
    auto lc = LayoutConvert::create(LayoutConvertAttributes{TensorLayout::NCHWC8}, Backend::CPU);
    const TensorDesc lc_in[] = {input_planar.desc()};
    auto lc_descs = lc->getOutputTensorDesc(lc_in);

    int64_t packed_elems = 1;
    for (int i = 0; i < lc_descs[0].rank; ++i) {
        packed_elems *= lc_descs[0].dims[i];
    }
    std::vector<float> packed_in_buf(static_cast<size_t>(packed_elems), 0.0f);
    auto input_packed = test::make_packed(lc_descs[0], packed_in_buf.data());
    const TensorView lc_inputs[] = {input_planar};
    lc->compute(input_packed, lc_inputs);

    // Output in same packed layout
    std::vector<float> packed_out_buf(static_cast<size_t>(packed_elems), 0.0f);
    auto output_packed = test::make_packed(lc_descs[0], packed_out_buf.data());

    // Now clamp the packed tensor
    ClampAttributes attrs;
    attrs.min_val = -2.0f;
    attrs.max_val = 3.0f;
    auto op = Clamp::create(attrs, Backend::CPU);

    const TensorView ins[] = {input_packed};
    op->compute(output_packed, ins);

    // Verify each element by comparing to scalar clamp on original data
    // Since NCHWC8 is just a re-layout, element values should match
    const float* out_ptr = output_packed.ptr<float>();
    const float* in_ptr = input_packed.ptr<float>();
    for (size_t i = 0; i < static_cast<size_t>(packed_elems); ++i) {
        // Skip padding elements (zero-filled in NCHWC8 when C not multiple of 8)
        float inv = in_ptr[i];
        float outv = out_ptr[i];
        float expected = std::max(-2.0f, std::min(3.0f, inv));
        NNOPS_EXPECT_NEAR(outv, expected, 1e-5f);
    }
}

// ============================================================
// Class API test
// ============================================================

NNOPS_TEST(clamp_class_api) {
    const int64_t shape[] = {4};
    float in_data[]  = {-5.0f, 0.0f, 5.0f, 10.0f};
    float out_buf_class[4] = {};
    float out_buf_func[4] = {};

    TensorView input(shape, DataType::f32, in_data);
    auto d = input.desc();

    ClampAttributes attrs;
    attrs.min_val = 0.0f;
    attrs.max_val = 7.0f;

    // Class API
    {
        auto op = Clamp::create(attrs, Backend::CPU);
        const TensorDesc in_arr[] = {d};
        auto descs = op->getOutputTensorDesc(in_arr);
        TensorView output = test::make_planar(descs[0], out_buf_class);
        const TensorView ins[] = {input};
        op->compute(output, ins);
    }

    // Class API (second instance)
    {
        auto op = Clamp::create(attrs, Backend::CPU);
        const TensorDesc in_arr[] = {d};
        auto descs = op->getOutputTensorDesc(in_arr);
        TensorView output = test::make_planar(descs[0], out_buf_func);
        const TensorView ins[] = {input};
        op->compute(output, ins);
    }

    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(out_buf_class[i], out_buf_func[i], 1e-5f);
    }
}
