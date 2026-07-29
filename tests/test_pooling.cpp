/// Unit tests for Pooling operator — NCHWC8 only.
///
/// All tests explicitly go through the NCHWC8 path:
///   NCHW input → pack_nchw_to_nchwc8 → pooling(NCHWC8) → unpack_nchwc8_to_nchw → compare
///
/// NCHW inline reference functions are used as the ground truth for comparison.

#include "nnops/ops/pooling.hpp"
#include "backend/cpu/layout_convert.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cmath>
#include <limits>

using namespace nnops;

// Forward declare the reference pooling kernel (NCHW scalar, ground truth).
namespace nnops::backend::cpu::reference {
    void pooling_ref(const PoolingAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);
}

// ============================================================
// Helpers
// ============================================================

/// Compute 32-byte-aligned pitch in bytes for NCHWC8 row stride.
inline int64_t nchwc8_pitch(int64_t W, int64_t elem_size = 4) {
    return ((W * 8 * elem_size + 31) / 32) * 32;
}

/// Compute output spatial dims for 2D pooling.
inline std::pair<int64_t, int64_t> pool_out_2d(int64_t H, int64_t W,
                                                 int64_t KH, int64_t KW,
                                                 int64_t SH, int64_t SW,
                                                 int64_t PH, int64_t PW,
                                                 int64_t DH = 1, int64_t DW = 1) {
    int64_t OH = (H + 2*PH - DH*(KH-1) - 1) / SH + 1;
    int64_t OW = (W + 2*PW - DW*(KW-1) - 1) / SW + 1;
    return {OH, OW};
}

/// Compute output spatial dims for 3D pooling.
inline std::tuple<int64_t, int64_t, int64_t> pool_out_3d(
    int64_t D, int64_t H, int64_t W,
    int64_t KD, int64_t KH, int64_t KW,
    int64_t SD, int64_t SH, int64_t SW,
    int64_t PD, int64_t PH, int64_t PW) {
    int64_t OD = (D + 2*PD - (KD-1) - 1) / SD + 1;
    int64_t OH = (H + 2*PH - (KH-1) - 1) / SH + 1;
    int64_t OW = (W + 2*PW - (KW-1) - 1) / SW + 1;
    return {OD, OH, OW};
}

// ============================================================
// Generic NCHWC8 roundtrip test helpers
// ============================================================

/// 2D: pack NCHW → pool on NCHWC8 → unpack → compare with NCHW reference.
static void test_nchwc8_vs_nchw(
    const std::vector<int64_t>& shape,
    PoolingAttributes attrs)
{
    const int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    const int64_t C8 = (C + 7) / 8;

    const int64_t KH = attrs.kernel_shape[1], KW = attrs.kernel_shape[2];
    const int64_t SH = attrs.stride[1], SW = attrs.stride[2];
    const int64_t PH = attrs.padding[1], PW = attrs.padding[2];
    const int64_t DH = attrs.dilation[1], DW = attrs.dilation[2];
    auto [OH, OW] = pool_out_2d(H, W, KH, KW, SH, SW, PH, PW, DH, DW);

    // Random NCHW input
    auto [in_vec, in_nchw] = test::make_random_tensor(shape);

    // NCHW scalar reference (ground truth)
    std::vector<float> ref_out(N * C * OH * OW);
    const std::vector<int64_t> oshape_ref = {N, C, OH, OW};
    TensorView out_nchw(oshape_ref, DataType::f32, ref_out.data(), TensorLayout::NCHW);
    if (attrs.add_to) {
        for (auto& v : ref_out) v = 1.0f;
    }
    {
        ComputeContext ctx;
        backend::cpu::reference::pooling_ref(attrs, out_nchw, {&in_nchw, 1}, ctx, nullptr);
    }

    // Pack input to NCHWC8
    int64_t in_pitch = nchwc8_pitch(W);
    int64_t in_pitch_elems = in_pitch / 4;
    std::vector<float> packed_in(N * C8 * H * in_pitch_elems);
    TensorView in_c8(shape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    // Pool on NCHWC8
    int64_t out_pitch = nchwc8_pitch(OW);
    int64_t out_pitch_elems = out_pitch / 4;
    std::vector<float> packed_out(N * C8 * OH * out_pitch_elems);
    TensorView out_c8(oshape_ref, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    // For add_to: pre-fill output with 1.0
    if (attrs.add_to) {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c8 = 0; c8 < C8; ++c8) {
                for (int64_t oh = 0; oh < OH; ++oh) {
                    float* row = packed_out.data() + ((n * C8 + c8) * OH + oh) * out_pitch_elems;
                    for (int64_t ow = 0; ow < OW; ++ow)
                        for (int64_t l = 0; l < 8; ++l)
                            row[ow * 8 + l] = 1.0f;
                }
            }
        }
    }

    pooling(in_c8, out_c8, attrs);

    // Unpack and compare
    std::vector<float> result(N * C * OH * OW);
    TensorView res_nchw(oshape_ref, DataType::f32, result.data(), TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);
    NNOPS_EXPECT_TRUE(test::allclose(res_nchw, out_nchw, 1e-4f, 1e-4f));
}

/// 3D: pack NCDHW → pool on NCDHWC8 → unpack → compare with NCDHW reference.
static void test_nchwc8_3d_vs_nchw(
    const std::vector<int64_t>& shape,
    PoolingAttributes attrs)
{
    const int64_t N = shape[0], C = shape[1], D = shape[2], H = shape[3], W = shape[4];
    const int64_t C8 = (C + 7) / 8;

    const int64_t KD = attrs.kernel_shape[0], KH = attrs.kernel_shape[1], KW = attrs.kernel_shape[2];
    const int64_t SD = attrs.stride[0], SH = attrs.stride[1], SW = attrs.stride[2];
    const int64_t PD = attrs.padding[0], PH = attrs.padding[1], PW = attrs.padding[2];
    auto [OD, OH, OW] = pool_out_3d(D, H, W, KD, KH, KW, SD, SH, SW, PD, PH, PW);

    // Random NCDHW input
    auto [in_vec, in_ncdhw] = test::make_random_tensor(shape);

    // NCDHW scalar reference (ground truth)
    std::vector<float> ref_out(N * C * OD * OH * OW);
    const std::vector<int64_t> oshape = {N, C, OD, OH, OW};
    TensorView out_ncdhw(oshape, DataType::f32, ref_out.data(), TensorLayout::NCDHW);
    {
        ComputeContext ctx;
        backend::cpu::reference::pooling_ref(attrs, out_ncdhw, {&in_ncdhw, 1}, ctx, nullptr);
    }

    // Pack input to NCDHWC8
    int64_t in_pitch = nchwc8_pitch(W);
    int64_t in_pitch_elems = in_pitch / 4;
    std::vector<float> packed_in(N * C8 * D * H * in_pitch_elems);
    TensorView in_c8(shape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCDHWC8);
    pack_ncdhw_to_ncdhwc8(in_ncdhw, in_c8);

    // Pool on NCDHWC8
    int64_t out_pitch = nchwc8_pitch(OW);
    int64_t out_pitch_elems = out_pitch / 4;
    std::vector<float> packed_out(N * C8 * OD * OH * out_pitch_elems);
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCDHWC8);
    pooling(in_c8, out_c8, attrs);

    // Unpack and compare
    std::vector<float> result(N * C * OD * OH * OW);
    TensorView res_ncdhw(oshape, DataType::f32, result.data(), TensorLayout::NCDHW);
    unpack_ncdhwc8_to_ncdhw(out_c8, res_ncdhw);
    NNOPS_EXPECT_TRUE(test::allclose(res_ncdhw, out_ncdhw, 1e-4f, 1e-4f));
}

// ============================================================
// 2D Basic deterministic tests (hardcoded values)
// ============================================================

NNOPS_TEST(pooling_2d_max_basic) {
    // 1x1x4x4 input, 2x2 kernel, stride=2, pad=0 → 1x1x2x2
    const int64_t shape[] = {1, 1, 4, 4};
    float in_data[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    TensorView in_nchw(shape, DataType::f32, in_data, TensorLayout::NCHW);

    // Pack to NCHWC8
    int64_t in_pitch = nchwc8_pitch(4);
    std::vector<float> packed_in(1 * 1 * 4 * (in_pitch / 4), 0.0f);
    TensorView in_c8(shape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    // Pool
    int64_t out_pitch = nchwc8_pitch(2);
    std::vector<float> packed_out(1 * 1 * 2 * (out_pitch / 4), 0.0f);
    const int64_t oshape[] = {1, 1, 2, 2};
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};
    attrs.padding      = {0, 0, 0};

    pooling(in_c8, out_c8, attrs);

    // Unpack and check
    float result[4] = {};
    TensorView res_nchw(oshape, DataType::f32, result, TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    NNOPS_EXPECT_NEAR(result[0], 6.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(result[1], 8.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(result[2], 14.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(result[3], 16.0f, 1e-6f);
}

NNOPS_TEST(pooling_2d_average_basic) {
    // 1x1x2x2 input, 3x3 kernel, stride=1, pad=1 → 1x1x2x2
    const int64_t shape[] = {1, 1, 2, 2};
    float in_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    TensorView in_nchw(shape, DataType::f32, in_data, TensorLayout::NCHW);

    int64_t in_pitch = nchwc8_pitch(2);
    std::vector<float> packed_in(1 * 1 * 2 * (in_pitch / 4), 0.0f);
    TensorView in_c8(shape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    int64_t out_pitch = nchwc8_pitch(2);
    std::vector<float> packed_out(1 * 1 * 2 * (out_pitch / 4), 0.0f);
    const int64_t oshape[] = {1, 1, 2, 2};
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};

    pooling(in_c8, out_c8, attrs);

    float result[4] = {};
    TensorView res_nchw(oshape, DataType::f32, result, TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    // sum=1+2+3+4=10, K_total=9, avg=10/9
    NNOPS_EXPECT_NEAR(result[0], 10.0f / 9.0f, 1e-4f);
}

NNOPS_TEST(pooling_2d_average_exclude_pad) {
    const int64_t shape[] = {1, 1, 2, 2};
    float in_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    TensorView in_nchw(shape, DataType::f32, in_data, TensorLayout::NCHW);

    int64_t in_pitch = nchwc8_pitch(2);
    std::vector<float> packed_in(1 * 1 * 2 * (in_pitch / 4), 0.0f);
    TensorView in_c8(shape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    int64_t out_pitch = nchwc8_pitch(2);
    std::vector<float> packed_out(1 * 1 * 2 * (out_pitch / 4), 0.0f);
    const int64_t oshape[] = {1, 1, 2, 2};
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.exclude_pad = true;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};

    pooling(in_c8, out_c8, attrs);

    float result[4] = {};
    TensorView res_nchw(oshape, DataType::f32, result, TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, res_nchw);

    // 2x2 input, 3x3 kernel, pad=1, exclude_pad:
    // At every output position, the kernel covers all 4 input elements.
    // sum=1+2+3+4=10, valid=4, avg=2.5
    NNOPS_EXPECT_NEAR(result[0], 2.5f, 1e-4f);
    NNOPS_EXPECT_NEAR(result[1], 2.5f, 1e-4f);
    NNOPS_EXPECT_NEAR(result[2], 2.5f, 1e-4f);
    NNOPS_EXPECT_NEAR(result[3], 2.5f, 1e-4f);
}

NNOPS_TEST(pooling_2d_class_api) {
    const int64_t shape[] = {1, 1, 4, 4};
    const int64_t oshape[] = {1, 1, 2, 2};
    float in_data[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    TensorView in_nchw(shape, DataType::f32, in_data, TensorLayout::NCHW);

    // Functional API (auto-converts)
    float out1_data[4] = {};
    TensorView out1(oshape, DataType::f32, out1_data, TensorLayout::NCHW);
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};
    pooling(in_nchw, out1, attrs);

    // Class API (NCHWC8)
    int64_t in_pitch = nchwc8_pitch(4);
    std::vector<float> packed_in(1 * 1 * 4 * (in_pitch / 4), 0.0f);
    TensorView in_c8(shape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    int64_t out_pitch = nchwc8_pitch(2);
    std::vector<float> packed_out(1 * 1 * 2 * (out_pitch / 4), 0.0f);
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    auto op = Pooling::create(attrs, Backend::CPU);
    const TensorView ins[] = {in_c8};
    op->compute(out_c8, ins);

    float out2_data[4] = {};
    TensorView out2(oshape, DataType::f32, out2_data, TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(out_c8, out2);

    NNOPS_EXPECT_TRUE(test::allclose(out1, out2, 1e-6f, 1e-6f));
}

// ============================================================
// 2D NCHWC8 roundtrip tests (pack→pool→unpack vs NCHW ref)
// ============================================================

NNOPS_TEST(pooling_nchwc8_max_small) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    test_nchwc8_vs_nchw({1, 3, 16, 16}, attrs);
}

NNOPS_TEST(pooling_nchwc8_max_padding) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};
    test_nchwc8_vs_nchw({1, 4, 15, 15}, attrs);
}

NNOPS_TEST(pooling_nchwc8_max_stride2) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 2};
    test_nchwc8_vs_nchw({2, 16, 32, 32}, attrs);
}

NNOPS_TEST(pooling_nchwc8_avg_padding) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};
    test_nchwc8_vs_nchw({1, 8, 12, 12}, attrs);
}

NNOPS_TEST(pooling_nchwc8_avg_stride2) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};
    test_nchwc8_vs_nchw({1, 2, 16, 16}, attrs);
}

NNOPS_TEST(pooling_nchwc8_large_input) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    test_nchwc8_vs_nchw({2, 3, 64, 64}, attrs);
}

NNOPS_TEST(pooling_nchwc8_odd_width) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    test_nchwc8_vs_nchw({1, 2, 10, 10}, attrs);
}

NNOPS_TEST(pooling_nchwc8_partial_c8) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    test_nchwc8_vs_nchw({1, 20, 8, 8}, attrs);
}

NNOPS_TEST(pooling_nchwc8_single_channel) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 1, 1};
    test_nchwc8_vs_nchw({1, 1, 4, 4}, attrs);
}

NNOPS_TEST(pooling_nchwc8_exclude_pad) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.exclude_pad = true;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};
    test_nchwc8_vs_nchw({1, 2, 4, 4}, attrs);
}

NNOPS_TEST(pooling_nchwc8_add_to) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.add_to = true;
    test_nchwc8_vs_nchw({1, 8, 8, 8}, attrs);
}

// ============================================================
// 2D Random tests (no NaN/Inf)
// ============================================================

NNOPS_TEST(pooling_nchwc8_random_2d) {
    auto [in_vec, in_nchw] = test::make_random_tensor({1, 3, 16, 16});
    int64_t W = 16, C = 3;
    int64_t C8 = (C + 7) / 8;

    int64_t in_pitch = nchwc8_pitch(W);
    std::vector<float> packed_in(1 * C8 * 16 * (in_pitch / 4));
    const int64_t shape[] = {1, 3, 16, 16};
    TensorView in_c8(shape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(in_nchw, in_c8);

    int64_t OW = 8;
    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(1 * C8 * 8 * (out_pitch / 4));
    const int64_t oshape[] = {1, 3, 8, 8};
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCHWC8);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};

    pooling(in_c8, out_c8, attrs);

    for (size_t i = 0; i < packed_out.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(packed_out[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(packed_out[i]));
    }
}

// ============================================================
// 3D Basic deterministic tests
// ============================================================

NNOPS_TEST(pooling_3d_max_basic) {
    // 1x1x2x4x4 input, 2x2x2 kernel, stride=2, pad=0 → 1x1x1x2x2
    const int64_t shape[] = {1, 1, 2, 4, 4};
    std::vector<float> in_buf(2 * 4 * 4);
    for (int i = 0; i < 32; ++i) in_buf[i] = static_cast<float>(i + 1);
    TensorView in_ncdhw(shape, DataType::f32, in_buf.data(), TensorLayout::NCDHW);

    int64_t in_pitch = nchwc8_pitch(4);
    std::vector<float> packed_in(1 * 1 * 2 * 4 * (in_pitch / 4));
    TensorView in_c8(shape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCDHWC8);
    pack_ncdhw_to_ncdhwc8(in_ncdhw, in_c8);

    int64_t out_pitch = nchwc8_pitch(2);
    std::vector<float> packed_out(1 * 1 * 1 * 2 * (out_pitch / 4));
    const int64_t oshape[] = {1, 1, 1, 2, 2};
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCDHWC8);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {2, 2, 2};
    attrs.stride       = {2, 2, 2};
    attrs.padding      = {0, 0, 0};

    pooling(in_c8, out_c8, attrs);

    float result[4] = {};
    TensorView res_ncdhw(oshape, DataType::f32, result, TensorLayout::NCDHW);
    unpack_ncdhwc8_to_ncdhw(out_c8, res_ncdhw);

    NNOPS_EXPECT_NEAR(result[0], 22.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(result[1], 24.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(result[2], 30.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(result[3], 32.0f, 1e-6f);
}

NNOPS_TEST(pooling_3d_average) {
    // 1x1x2x2x2 input, 2x2x2 kernel, stride=1, pad=0 → 1x1x1x1x1
    const int64_t shape[] = {1, 1, 2, 2, 2};
    float in_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    TensorView in_ncdhw(shape, DataType::f32, in_data, TensorLayout::NCDHW);

    int64_t in_pitch = nchwc8_pitch(2);
    std::vector<float> packed_in(1 * 1 * 2 * 2 * (in_pitch / 4), 0.0f);
    TensorView in_c8(shape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCDHWC8);
    pack_ncdhw_to_ncdhwc8(in_ncdhw, in_c8);

    int64_t out_pitch = nchwc8_pitch(1);
    std::vector<float> packed_out(1 * 1 * 1 * 1 * (out_pitch / 4), 0.0f);
    const int64_t oshape[] = {1, 1, 1, 1, 1};
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCDHWC8);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {2, 2, 2};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 0, 0};

    pooling(in_c8, out_c8, attrs);

    float result[1] = {};
    TensorView res_ncdhw(oshape, DataType::f32, result, TensorLayout::NCDHW);
    unpack_ncdhwc8_to_ncdhw(out_c8, res_ncdhw);

    NNOPS_EXPECT_NEAR(result[0], 4.5f, 1e-4f);
}

// ============================================================
// 3D NCHWC8 roundtrip tests
// ============================================================

NNOPS_TEST(pooling_nchwc8_3d_max) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {3, 3, 3};
    attrs.stride       = {1, 1, 1};
    test_nchwc8_3d_vs_nchw({1, 2, 8, 12, 12}, attrs);
}

NNOPS_TEST(pooling_nchwc8_3d_avg) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {3, 3, 3};
    attrs.stride       = {1, 1, 1};
    test_nchwc8_3d_vs_nchw({1, 1, 8, 12, 12}, attrs);
}

NNOPS_TEST(pooling_nchwc8_3d_padding) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {3, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {1, 0, 0};
    test_nchwc8_3d_vs_nchw({1, 1, 4, 8, 8}, attrs);
}

NNOPS_TEST(pooling_nchwc8_3d_random) {
    auto [in_vec, in_ncdhw] = test::make_random_tensor({1, 2, 8, 8, 8});
    int64_t D = 8, H = 8, W = 8, C = 2;
    int64_t C8 = (C + 7) / 8;

    int64_t in_pitch = nchwc8_pitch(W);
    std::vector<float> packed_in(1 * C8 * D * H * (in_pitch / 4));
    const int64_t shape[] = {1, 2, 8, 8, 8};
    TensorView in_c8(shape, DataType::f32, packed_in.data(), in_pitch, TensorLayout::NCDHWC8);
    pack_ncdhw_to_ncdhwc8(in_ncdhw, in_c8);

    int64_t OW = 4, OH = 4, OD = 4;
    int64_t out_pitch = nchwc8_pitch(OW);
    std::vector<float> packed_out(1 * C8 * OD * OH * (out_pitch / 4));
    const int64_t oshape[] = {1, 2, 4, 4, 4};
    TensorView out_c8(oshape, DataType::f32, packed_out.data(), out_pitch, TensorLayout::NCDHWC8);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {2, 2, 2};
    attrs.stride       = {2, 2, 2};

    pooling(in_c8, out_c8, attrs);

    for (size_t i = 0; i < packed_out.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(packed_out[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(packed_out[i]));
    }
}
