/// Unit tests for Pooling operator — NCHWC8 only.
///
/// All tests explicitly go through the class API path:
///   NCHW input → LayoutConvert(pack) → NCHWC8 → Pooling::compute → LayoutConvert(unpack) → NCHW → compare
///
/// NCHW inline reference functions are used as the ground truth for comparison.

#include "nnops/ops/pooling.hpp"
#include "nnops/ops/layout_convert.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"

#include <vector>
#include <cmath>
#include <limits>
#include <cstdint>
#include <type_traits>

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
        for (auto& v : ref_out) {
            v = 1.0f;
        }
    }
    {
        ComputeContext ctx;
        const TensorView ref_ins[] = {in_nchw};
        backend::cpu::reference::pooling_ref(attrs, out_nchw, ref_ins, ctx, nullptr);
    }

    // LayoutConvert: NCHW → NCHWC8 (pack)
    {
        auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
        TensorDesc in_desc = in_nchw.desc();
        const TensorDesc desc_arr[] = {in_desc};
        auto pack_descs = pack_op->getOutputTensorDesc(desc_arr);
        std::vector<char> packed_buf(pack_descs[0].storage_bytes());
        auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
        const TensorView ins[] = {in_nchw};
        pack_op->compute(in_c8, ins);

        // Pooling on NCHWC8
        auto pool_op = Pooling::create(attrs);
        TensorDesc c8_desc = in_c8.desc();
        const TensorDesc pool_in_arr[] = {c8_desc};
        auto pool_descs = pool_op->getOutputTensorDesc(pool_in_arr);
        std::vector<char> pool_buf(pool_descs[0].storage_bytes());
        auto out_c8 = test::make_packed(pool_descs[0], pool_buf.data());

        if (attrs.add_to) {
            int64_t rows = out_c8.total_rows();
            int64_t row_elems = out_c8.row_stride_elems();
            for (int64_t r = 0; r < rows; ++r) {
                float* row = out_c8.ptr<float>(r);
                for (int64_t e = 0; e < row_elems; ++e) {
                    row[e] = 1.0f;
                }
            }
        }

        const TensorView pool_ins[] = {in_c8};
        pool_op->compute(out_c8, pool_ins);

        // LayoutConvert: NCHWC8 → NCHW (unpack)
        auto unpack_op = LayoutConvert::create(TensorLayout::NCHW);
        TensorDesc c8_out_desc = out_c8.desc();
        const TensorDesc unpack_in_arr[] = {c8_out_desc};
        auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in_arr);
        std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
        auto res_nchw = test::make_planar(unpack_descs[0], unpack_buf.data());
        const TensorView unpack_ins[] = {out_c8};
        unpack_op->compute(res_nchw, unpack_ins);

        // Compare
        NNOPS_EXPECT_TRUE(test::allclose(res_nchw, out_nchw, 1e-4f, 1e-4f));
    }
}

/// 3D: pack NCDHW → pool on NCDHWC8 → unpack → compare with NCDHW reference.
static void test_nchwc8_3d_vs_nchw(
    const std::vector<int64_t>& shape,
    PoolingAttributes attrs)
{
    const int64_t N = shape[0], C = shape[1], D = shape[2], H = shape[3], W = shape[4];

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
        const TensorView ref_ins[] = {in_ncdhw};
        backend::cpu::reference::pooling_ref(attrs, out_ncdhw, ref_ins, ctx, nullptr);
    }

    // LayoutConvert: NCDHW → NCDHWC8 (pack)
    {
        auto pack_op = LayoutConvert::create(TensorLayout::NCDHWC8);
        TensorDesc in_desc = in_ncdhw.desc();
        const TensorDesc desc_arr[] = {in_desc};
        auto pack_descs = pack_op->getOutputTensorDesc(desc_arr);
        std::vector<char> packed_buf(pack_descs[0].storage_bytes());
        auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
        const TensorView ins[] = {in_ncdhw};
        pack_op->compute(in_c8, ins);

        // Pooling on NCDHWC8
        auto pool_op = Pooling::create(attrs);
        TensorDesc c8_desc = in_c8.desc();
        const TensorDesc pool_in_arr[] = {c8_desc};
        auto pool_descs = pool_op->getOutputTensorDesc(pool_in_arr);
        std::vector<char> pool_buf(pool_descs[0].storage_bytes());
        auto out_c8 = test::make_packed(pool_descs[0], pool_buf.data());
        const TensorView pool_ins[] = {in_c8};
        pool_op->compute(out_c8, pool_ins);

        // LayoutConvert: NCDHWC8 → NCDHW (unpack)
        auto unpack_op = LayoutConvert::create(TensorLayout::NCDHW);
        TensorDesc c8_out_desc = out_c8.desc();
        const TensorDesc unpack_in_arr[] = {c8_out_desc};
        auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in_arr);
        std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
        auto res_ncdhw = test::make_planar(unpack_descs[0], unpack_buf.data());
        const TensorView unpack_ins[] = {out_c8};
        unpack_op->compute(res_ncdhw, unpack_ins);

        NNOPS_EXPECT_TRUE(test::allclose(res_ncdhw, out_ncdhw, 1e-4f, 1e-4f));
    }
}

// ============================================================
// Planar path — the operator takes NCHW/NCDHW as it is, no layout conversion
// ============================================================

template <typename T>
static void test_planar_vs_ref(const PoolingAttributes& attrs, TensorView in_planar)
{
    auto pool_op = Pooling::create(attrs);
    TensorDesc in_desc = in_planar.desc();
    const TensorDesc in_arr[] = {in_desc};
    auto out_descs = pool_op->getOutputTensorDesc(in_arr);

    std::vector<T> out_buf(static_cast<size_t>(out_descs[0].numel()));
    auto out = test::make_planar(out_descs[0], out_buf.data());

    std::vector<T> ref_buf(out_buf.size());
    auto ref = test::make_planar(out_descs[0], ref_buf.data());
    if (attrs.add_to) {
        for (auto& v : out_buf) { simd::s_store(&v, 1.0f); }
        for (auto& v : ref_buf) { simd::s_store(&v, 1.0f); }
    }
    {
        ComputeContext ctx;
        const TensorView ref_ins[] = {in_planar};
        backend::cpu::reference::pooling_ref(attrs, ref, ref_ins, ctx, nullptr);
    }

    const TensorView ins[] = {in_planar};
    NNOPS_EXPECT_TRUE(is_ok(pool_op->compute(out, ins)));

    // allclose compares f32 storage, so an f16 pair goes through f32 first.
    if constexpr (std::is_same_v<T, float>) {
        NNOPS_EXPECT_TRUE(test::allclose(out, ref, 1e-4f, 1e-4f));
    } else {
        const float tol = 5e-3f;   // f16 accumulate: the sum runs in half lanes
        const auto a = test::f16_to_f32(out_buf);
        const auto b = test::f16_to_f32(ref_buf);
        for (size_t i = 0; i < a.size(); ++i) {
            NNOPS_EXPECT_NEAR(a[i], b[i], tol);
        }
    }
}

/// f32 random input, the common case.
static void test_planar_f32(const std::vector<int64_t>& shape,
                            const PoolingAttributes& attrs)
{
    auto [in_vec, in] = test::make_random_tensor(shape);
    test_planar_vs_ref<float>(attrs, in);
}

static void test_planar_f16(const std::vector<int64_t>& shape,
                            const PoolingAttributes& attrs)
{
    auto [in_vec, in] = test::make_random_f16_tensor(shape);
    test_planar_vs_ref<backend::cpu::half>(attrs, in);
}

// ============================================================
// 2D Basic deterministic tests (hardcoded values)
// ============================================================

NNOPS_TEST(pooling_2d_max_basic) {
    // 1x1x4x4 input, 2x2 kernel, stride=2, pad=0 → 1x1x2x2
    const int64_t shape[] = {1, 1, 4, 4};
    float in_data[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    TensorView in_nchw(shape, DataType::f32, in_data, TensorLayout::NCHW);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};
    attrs.padding      = {0, 0, 0};

    // LayoutConvert: NCHW → NCHWC8 (pack)
    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
    TensorDesc in_desc = in_nchw.desc();
    const TensorDesc desc_arr[] = {in_desc};
    auto pack_descs = pack_op->getOutputTensorDesc(desc_arr);
    std::vector<char> packed_buf(pack_descs[0].storage_bytes());
    auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
    const TensorView pack_ins[] = {in_nchw};
    pack_op->compute(in_c8, pack_ins);

    // Pooling on NCHWC8
    auto pool_op = Pooling::create(attrs);
    TensorDesc c8_desc = in_c8.desc();
    const TensorDesc pool_in_arr[] = {c8_desc};
    auto pool_descs = pool_op->getOutputTensorDesc(pool_in_arr);
    std::vector<char> pool_buf(pool_descs[0].storage_bytes());
    auto out_c8 = test::make_packed(pool_descs[0], pool_buf.data());
    const TensorView pool_ins[] = {in_c8};
    pool_op->compute(out_c8, pool_ins);

    // LayoutConvert: NCHWC8 → NCHW (unpack)
    auto unpack_op = LayoutConvert::create(TensorLayout::NCHW);
    TensorDesc c8_out_desc = out_c8.desc();
    const TensorDesc unpack_in_arr[] = {c8_out_desc};
    auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in_arr);
    std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
    auto res_nchw = test::make_planar(unpack_descs[0], unpack_buf.data());
    const TensorView unpack_ins[] = {out_c8};
    unpack_op->compute(res_nchw, unpack_ins);

    float expected[4] = {6.0f, 8.0f, 14.0f, 16.0f};
    const int64_t oshape[] = {1, 1, 2, 2};
    TensorView exp_view(oshape, DataType::f32, expected, TensorLayout::NCHW);
    NNOPS_EXPECT_TRUE(test::allclose(res_nchw, exp_view, 1e-6f, 1e-6f));
}

NNOPS_TEST(pooling_2d_average_basic) {
    // 1x1x2x2 input, 3x3 kernel, stride=1, pad=1 → 1x1x2x2
    const int64_t shape[] = {1, 1, 2, 2};
    float in_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    TensorView in_nchw(shape, DataType::f32, in_data, TensorLayout::NCHW);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};

    // LayoutConvert: NCHW → NCHWC8
    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
    TensorDesc in_desc = in_nchw.desc();
    const TensorDesc desc_arr[] = {in_desc};
    auto pack_descs = pack_op->getOutputTensorDesc(desc_arr);
    std::vector<char> packed_buf(pack_descs[0].storage_bytes());
    auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
    const TensorView pack_ins[] = {in_nchw};
    pack_op->compute(in_c8, pack_ins);

    // Pooling on NCHWC8
    auto pool_op = Pooling::create(attrs);
    TensorDesc c8_desc = in_c8.desc();
    const TensorDesc pool_in_arr[] = {c8_desc};
    auto pool_descs = pool_op->getOutputTensorDesc(pool_in_arr);
    std::vector<char> pool_buf(pool_descs[0].storage_bytes());
    auto out_c8 = test::make_packed(pool_descs[0], pool_buf.data());
    const TensorView pool_ins[] = {in_c8};
    pool_op->compute(out_c8, pool_ins);

    // LayoutConvert: NCHWC8 → NCHW
    auto unpack_op = LayoutConvert::create(TensorLayout::NCHW);
    TensorDesc c8_out_desc = out_c8.desc();
    const TensorDesc unpack_in_arr[] = {c8_out_desc};
    auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in_arr);
    std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
    auto res_nchw = test::make_planar(unpack_descs[0], unpack_buf.data());
    const TensorView unpack_ins[] = {out_c8};
    unpack_op->compute(res_nchw, unpack_ins);

    // sum=1+2+3+4=10, K_total=9, avg=10/9
    NNOPS_EXPECT_NEAR(res_nchw.ptr<float>()[0], 10.0f / 9.0f, 1e-4f);
}

NNOPS_TEST(pooling_2d_average_exclude_pad) {
    const int64_t shape[] = {1, 1, 2, 2};
    float in_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    TensorView in_nchw(shape, DataType::f32, in_data, TensorLayout::NCHW);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.exclude_pad = true;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};

    // LayoutConvert: NCHW → NCHWC8
    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
    TensorDesc in_desc = in_nchw.desc();
    const TensorDesc desc_arr[] = {in_desc};
    auto pack_descs = pack_op->getOutputTensorDesc(desc_arr);
    std::vector<char> packed_buf(pack_descs[0].storage_bytes());
    auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
    const TensorView pack_ins[] = {in_nchw};
    pack_op->compute(in_c8, pack_ins);

    // Pooling on NCHWC8
    auto pool_op = Pooling::create(attrs);
    TensorDesc c8_desc = in_c8.desc();
    const TensorDesc pool_in_arr[] = {c8_desc};
    auto pool_descs = pool_op->getOutputTensorDesc(pool_in_arr);
    std::vector<char> pool_buf(pool_descs[0].storage_bytes());
    auto out_c8 = test::make_packed(pool_descs[0], pool_buf.data());
    const TensorView pool_ins[] = {in_c8};
    pool_op->compute(out_c8, pool_ins);

    // LayoutConvert: NCHWC8 → NCHW
    auto unpack_op = LayoutConvert::create(TensorLayout::NCHW);
    TensorDesc c8_out_desc = out_c8.desc();
    const TensorDesc unpack_in_arr[] = {c8_out_desc};
    auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in_arr);
    std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
    auto res_nchw = test::make_planar(unpack_descs[0], unpack_buf.data());
    const TensorView unpack_ins[] = {out_c8};
    unpack_op->compute(res_nchw, unpack_ins);

    // 2x2 input, 3x3 kernel, pad=1, exclude_pad:
    // At every output position, the kernel covers all 4 input elements.
    // sum=1+2+3+4=10, valid=4, avg=2.5
    NNOPS_EXPECT_NEAR(res_nchw.ptr<float>()[0], 2.5f, 1e-4f);
    NNOPS_EXPECT_NEAR(res_nchw.ptr<float>()[1], 2.5f, 1e-4f);
    NNOPS_EXPECT_NEAR(res_nchw.ptr<float>()[2], 2.5f, 1e-4f);
    NNOPS_EXPECT_NEAR(res_nchw.ptr<float>()[3], 2.5f, 1e-4f);
}

NNOPS_TEST(pooling_2d_class_api) {
    const int64_t shape[] = {1, 1, 4, 4};
    float in_data[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    TensorView in_nchw(shape, DataType::f32, in_data, TensorLayout::NCHW);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};

    // Pack input to NCHWC8
    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
    TensorDesc in_desc = in_nchw.desc();
    const TensorDesc desc_arr[] = {in_desc};
    auto pack_descs = pack_op->getOutputTensorDesc(desc_arr);
    std::vector<char> packed_buf(pack_descs[0].storage_bytes());
    auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
    const TensorView pack_ins[] = {in_nchw};
    pack_op->compute(in_c8, pack_ins);

    // Class API path (first instance)
    TensorDesc c8_desc = in_c8.desc();
    const TensorDesc pool_in_desc[] = {c8_desc};

    // Get output desc from Pooling for the first class path
    auto pool_ref = Pooling::create(attrs);
    auto func_out_descs = pool_ref->getOutputTensorDesc(pool_in_desc);
    std::vector<char> func_buf(func_out_descs[0].storage_bytes());
    auto out1_c8 = test::make_packed(func_out_descs[0], func_buf.data());
    const TensorView func_ins[] = {in_c8};
    pool_ref->compute(out1_c8, func_ins);

    // Class API path (second instance)
    auto pool_op = Pooling::create(attrs);
    auto cls_out_descs = pool_op->getOutputTensorDesc(pool_in_desc);
    std::vector<char> cls_buf(cls_out_descs[0].storage_bytes());
    auto out2_c8 = test::make_packed(cls_out_descs[0], cls_buf.data());
    const TensorView cls_ins[] = {in_c8};
    pool_op->compute(out2_c8, cls_ins);

    // Unpack both and compare
    auto unpack_op = LayoutConvert::create(TensorLayout::NCHW);

    // First instance unpack
    TensorDesc func_c8_desc = out1_c8.desc();
    const TensorDesc func_unpack_in[] = {func_c8_desc};
    auto func_unpack_descs = unpack_op->getOutputTensorDesc(func_unpack_in);
    std::vector<char> func_unpack_buf(func_unpack_descs[0].storage_bytes());
    auto out1 = test::make_planar(func_unpack_descs[0], func_unpack_buf.data());
    const TensorView func_unpack_ins[] = {out1_c8};
    unpack_op->compute(out1, func_unpack_ins);

    // Class API path unpack
    TensorDesc cls_c8_desc = out2_c8.desc();
    const TensorDesc cls_unpack_in[] = {cls_c8_desc};
    auto cls_unpack_descs = unpack_op->getOutputTensorDesc(cls_unpack_in);
    std::vector<char> cls_unpack_buf(cls_unpack_descs[0].storage_bytes());
    auto out2 = test::make_planar(cls_unpack_descs[0], cls_unpack_buf.data());
    const TensorView cls_unpack_ins[] = {out2_c8};
    unpack_op->compute(out2, cls_unpack_ins);

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

    // LayoutConvert: NCHW → NCHWC8
    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
    TensorDesc in_desc = in_nchw.desc();
    const TensorDesc desc_arr[] = {in_desc};
    auto pack_descs = pack_op->getOutputTensorDesc(desc_arr);
    std::vector<char> packed_buf(pack_descs[0].storage_bytes());
    auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
    const TensorView pack_ins[] = {in_nchw};
    pack_op->compute(in_c8, pack_ins);

    // Pooling on NCHWC8
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};

    auto pool_op = Pooling::create(attrs);
    TensorDesc c8_desc = in_c8.desc();
    const TensorDesc pool_in_arr[] = {c8_desc};
    auto pool_descs = pool_op->getOutputTensorDesc(pool_in_arr);
    std::vector<char> pool_buf(pool_descs[0].storage_bytes());
    auto out_c8 = test::make_packed(pool_descs[0], pool_buf.data());
    const TensorView pool_ins[] = {in_c8};
    pool_op->compute(out_c8, pool_ins);

    // Check packed output for NaN/Inf
    int64_t rows = out_c8.total_rows();
    int64_t row_elems = out_c8.row_stride_elems();
    for (int64_t r = 0; r < rows; ++r) {
        float* row = out_c8.ptr<float>(r);
        for (int64_t e = 0; e < row_elems; ++e) {
            NNOPS_EXPECT_TRUE(!std::isnan(row[e]));
            NNOPS_EXPECT_TRUE(!std::isinf(row[e]));
        }
    }
}

// ============================================================
// 3D Basic deterministic tests
// ============================================================

NNOPS_TEST(pooling_3d_max_basic) {
    // 1x1x2x4x4 input, 2x2x2 kernel, stride=2, pad=0 → 1x1x1x2x2
    const int64_t shape[] = {1, 1, 2, 4, 4};
    std::vector<float> in_buf(2 * 4 * 4);
    for (int i = 0; i < 32; ++i) {
        in_buf[i] = static_cast<float>(i + 1);
    }
    TensorView in_ncdhw(shape, DataType::f32, in_buf.data(), TensorLayout::NCDHW);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {2, 2, 2};
    attrs.stride       = {2, 2, 2};
    attrs.padding      = {0, 0, 0};

    // LayoutConvert: NCDHW → NCDHWC8
    auto pack_op = LayoutConvert::create(TensorLayout::NCDHWC8);
    TensorDesc in_desc = in_ncdhw.desc();
    const TensorDesc desc_arr[] = {in_desc};
    auto pack_descs = pack_op->getOutputTensorDesc(desc_arr);
    std::vector<char> packed_buf(pack_descs[0].storage_bytes());
    auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
    const TensorView pack_ins[] = {in_ncdhw};
    pack_op->compute(in_c8, pack_ins);

    // Pooling on NCDHWC8
    auto pool_op = Pooling::create(attrs);
    TensorDesc c8_desc = in_c8.desc();
    const TensorDesc pool_in_arr[] = {c8_desc};
    auto pool_descs = pool_op->getOutputTensorDesc(pool_in_arr);
    std::vector<char> pool_buf(pool_descs[0].storage_bytes());
    auto out_c8 = test::make_packed(pool_descs[0], pool_buf.data());
    const TensorView pool_ins[] = {in_c8};
    pool_op->compute(out_c8, pool_ins);

    // LayoutConvert: NCDHWC8 → NCDHW
    auto unpack_op = LayoutConvert::create(TensorLayout::NCDHW);
    TensorDesc c8_out_desc = out_c8.desc();
    const TensorDesc unpack_in_arr[] = {c8_out_desc};
    auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in_arr);
    std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
    auto res_ncdhw = test::make_planar(unpack_descs[0], unpack_buf.data());
    const TensorView unpack_ins[] = {out_c8};
    unpack_op->compute(res_ncdhw, unpack_ins);

    float expected[4] = {22.0f, 24.0f, 30.0f, 32.0f};
    const int64_t oshape[] = {1, 1, 1, 2, 2};
    TensorView exp_view(oshape, DataType::f32, expected, TensorLayout::NCDHW);
    NNOPS_EXPECT_TRUE(test::allclose(res_ncdhw, exp_view, 1e-6f, 1e-6f));
}

NNOPS_TEST(pooling_3d_average) {
    // 1x1x2x2x2 input, 2x2x2 kernel, stride=1, pad=0 → 1x1x1x1x1
    const int64_t shape[] = {1, 1, 2, 2, 2};
    float in_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    TensorView in_ncdhw(shape, DataType::f32, in_data, TensorLayout::NCDHW);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {2, 2, 2};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 0, 0};

    // LayoutConvert: NCDHW → NCDHWC8
    auto pack_op = LayoutConvert::create(TensorLayout::NCDHWC8);
    TensorDesc in_desc = in_ncdhw.desc();
    const TensorDesc desc_arr[] = {in_desc};
    auto pack_descs = pack_op->getOutputTensorDesc(desc_arr);
    std::vector<char> packed_buf(pack_descs[0].storage_bytes());
    auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
    const TensorView pack_ins[] = {in_ncdhw};
    pack_op->compute(in_c8, pack_ins);

    // Pooling on NCDHWC8
    auto pool_op = Pooling::create(attrs);
    TensorDesc c8_desc = in_c8.desc();
    const TensorDesc pool_in_arr[] = {c8_desc};
    auto pool_descs = pool_op->getOutputTensorDesc(pool_in_arr);
    std::vector<char> pool_buf(pool_descs[0].storage_bytes());
    auto out_c8 = test::make_packed(pool_descs[0], pool_buf.data());
    const TensorView pool_ins[] = {in_c8};
    pool_op->compute(out_c8, pool_ins);

    // LayoutConvert: NCDHWC8 → NCDHW
    auto unpack_op = LayoutConvert::create(TensorLayout::NCDHW);
    TensorDesc c8_out_desc = out_c8.desc();
    const TensorDesc unpack_in_arr[] = {c8_out_desc};
    auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in_arr);
    std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
    auto res_ncdhw = test::make_planar(unpack_descs[0], unpack_buf.data());
    const TensorView unpack_ins[] = {out_c8};
    unpack_op->compute(res_ncdhw, unpack_ins);

    NNOPS_EXPECT_NEAR(res_ncdhw.ptr<float>()[0], 4.5f, 1e-4f);
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

    // LayoutConvert: NCDHW → NCDHWC8
    auto pack_op = LayoutConvert::create(TensorLayout::NCDHWC8);
    TensorDesc in_desc = in_ncdhw.desc();
    const TensorDesc desc_arr[] = {in_desc};
    auto pack_descs = pack_op->getOutputTensorDesc(desc_arr);
    std::vector<char> packed_buf(pack_descs[0].storage_bytes());
    auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
    const TensorView pack_ins[] = {in_ncdhw};
    pack_op->compute(in_c8, pack_ins);

    // Pooling on NCDHWC8
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {2, 2, 2};
    attrs.stride       = {2, 2, 2};

    auto pool_op = Pooling::create(attrs);
    TensorDesc c8_desc = in_c8.desc();
    const TensorDesc pool_in_arr[] = {c8_desc};
    auto pool_descs = pool_op->getOutputTensorDesc(pool_in_arr);
    std::vector<char> pool_buf(pool_descs[0].storage_bytes());
    auto out_c8 = test::make_packed(pool_descs[0], pool_buf.data());
    const TensorView pool_ins[] = {in_c8};
    pool_op->compute(out_c8, pool_ins);

    // Check packed output for NaN/Inf
    int64_t rows = out_c8.total_rows();
    int64_t row_elems = out_c8.row_stride_elems();
    for (int64_t r = 0; r < rows; ++r) {
        float* row = out_c8.ptr<float>(r);
        for (int64_t e = 0; e < row_elems; ++e) {
            NNOPS_EXPECT_TRUE(!std::isnan(row[e]));
            NNOPS_EXPECT_TRUE(!std::isinf(row[e]));
        }
    }
}

// ============================================================
// Planar (NCHW/NCDHW) tests — the operator on the tensor as a graph hands it
// over, compared straight against the scalar reference
// ============================================================

NNOPS_TEST(pooling_planar_max_padding) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.padding      = {0, 1, 1};
    test_planar_f32({1, 3, 15, 15}, attrs);
}

NNOPS_TEST(pooling_planar_max_stride2_odd_width) {
    // Stride 2 takes the deinterleaving load; W=13 leaves a partial last block.
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 2, 2};
    attrs.padding      = {0, 1, 1};
    test_planar_f32({2, 4, 16, 13}, attrs);
}

NNOPS_TEST(pooling_planar_avg_padding) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {1, 3, 3};
    attrs.padding      = {0, 1, 1};
    test_planar_f32({1, 5, 12, 9}, attrs);
}

NNOPS_TEST(pooling_planar_avg_exclude_pad) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.exclude_pad = true;
    attrs.kernel_shape = {1, 3, 3};
    attrs.padding      = {0, 1, 1};
    test_planar_f32({1, 2, 10, 11}, attrs);
}

NNOPS_TEST(pooling_planar_avg_stride2) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};
    test_planar_f32({1, 3, 17, 9}, attrs);
}

NNOPS_TEST(pooling_planar_add_to) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.padding      = {0, 1, 1};
    attrs.add_to = true;
    test_planar_f32({1, 3, 8, 8}, attrs);
}

NNOPS_TEST(pooling_planar_f16) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {1, 3, 3};
    attrs.padding      = {0, 1, 1};
    test_planar_f16({1, 3, 10, 10}, attrs);
}

NNOPS_TEST(pooling_planar_3d_max) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {3, 3, 3};
    attrs.padding      = {1, 1, 1};
    test_planar_f32({1, 2, 6, 9, 11}, attrs);
}

NNOPS_TEST(pooling_planar_3d_avg) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {2, 2, 2};
    attrs.stride       = {2, 2, 2};
    test_planar_f32({1, 2, 6, 8, 10}, attrs);
}

// ============================================================
// Global pooling — the window is the whole extent, so every spatial axis of
// the output is 1 and the window fields are not read
// ============================================================

NNOPS_TEST(pooling_global_deterministic) {
    const int64_t shape[] = {1, 2, 2, 2};
    float data[8] = {1, 2, 3, 4, -1, -2, -8, -4};
    TensorView in(shape, DataType::f32, data, TensorLayout::NCHW);

    const int64_t oshape[] = {1, 2, 1, 1};
    float out_buf[2] = {0.0f, 0.0f};
    TensorView out(oshape, DataType::f32, out_buf, TensorLayout::NCHW);
    const TensorView views[] = {in};

    PoolingAttributes avg;
    avg.type = PoolingType::Average;
    avg.global = true;
    auto avg_op = Pooling::create(avg);
    NNOPS_EXPECT_TRUE(is_ok(avg_op->compute(out, views)));
    NNOPS_EXPECT_NEAR(out_buf[0], 2.5f, 1e-6f);    // (1 + 2 + 3 + 4) / 4
    NNOPS_EXPECT_NEAR(out_buf[1], -3.75f, 1e-6f);  // (-1 - 2 - 8 - 4) / 4

    // Max ignores the window fields too: a kernel_shape/stride/padding triple
    // that could not produce a 1x1 output still pools the whole plane.
    PoolingAttributes mx;
    mx.type = PoolingType::Max;
    mx.global = true;
    mx.kernel_shape = {1, 9, 9};
    mx.stride = {1, 3, 3};
    mx.padding = {0, 2, 2};
    auto mx_op = Pooling::create(mx);
    NNOPS_EXPECT_TRUE(is_ok(mx_op->compute(out, views)));
    NNOPS_EXPECT_NEAR(out_buf[0], 4.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[1], -1.0f, 1e-6f);

    // Shape follows the input alone.
    TensorDesc d = in.desc();
    const TensorDesc descs[] = {d};
    auto out_descs = mx_op->getOutputTensorDesc(descs);
    NNOPS_EXPECT_TRUE(out_descs[0].rank == 4 && out_descs[0].dims[2] == 1 &&
                      out_descs[0].dims[3] == 1);
}

NNOPS_TEST(pooling_global_planar_shapes) {
    // Widths picked to hit the 4x-unrolled body (W >= 32), the 8-lane body and
    // the scalar tail; C = 5 also leaves a partial NCHWC8 block.
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.global = true;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride = {1, 2, 2};
    attrs.padding = {0, 1, 1};

    test_planar_f32({1, 5, 7, 37}, attrs);
    test_planar_f32({2, 3, 9, 13}, attrs);
    test_planar_f32({1, 1, 5, 3}, attrs);

    PoolingAttributes mx = attrs;
    mx.type = PoolingType::Max;
    test_planar_f32({1, 5, 7, 37}, mx);
    test_planar_f32({2, 3, 9, 13}, mx);
}

NNOPS_TEST(pooling_global_planar_f16) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.global = true;
    test_planar_f16({1, 3, 10, 10}, attrs);

    PoolingAttributes mx = attrs;
    mx.type = PoolingType::Max;
    test_planar_f16({1, 3, 10, 10}, mx);
}

NNOPS_TEST(pooling_global_planar_3d) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.global = true;
    test_planar_f32({1, 2, 4, 6, 9}, attrs);

    PoolingAttributes mx = attrs;
    mx.type = PoolingType::Max;
    test_planar_f32({1, 2, 4, 6, 9}, mx);
}

NNOPS_TEST(pooling_global_add_to) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.global = true;
    attrs.add_to = true;
    test_planar_f32({1, 3, 8, 8}, attrs);
}

/// Packed (NCHWC8) global pooling: the lanes are the eight channels, so this
/// is the one path where the reduction ends in a vector store.
NNOPS_TEST(pooling_global_packed) {
    const std::vector<int64_t> shape = {2, 11, 9, 13};
    auto [in_vec, in_nchw] = test::make_random_tensor(shape);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.global = true;

    std::vector<float> ref_buf(2 * 11, 0.0f);
    const std::vector<int64_t> oshape = {2, 11, 1, 1};
    TensorView ref(oshape, DataType::f32, ref_buf.data(), TensorLayout::NCHW);
    {
        ComputeContext ctx;
        const TensorView ref_ins[] = {in_nchw};
        backend::cpu::reference::pooling_ref(attrs, ref, ref_ins, ctx, nullptr);
    }

    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
    TensorDesc in_desc = in_nchw.desc();
    const TensorDesc desc_arr[] = {in_desc};
    auto pack_descs = pack_op->getOutputTensorDesc(desc_arr);
    std::vector<char> packed_buf(pack_descs[0].storage_bytes());
    auto in_c8 = test::make_packed(pack_descs[0], packed_buf.data());
    const TensorView pack_ins[] = {in_nchw};
    pack_op->compute(in_c8, pack_ins);

    auto pool_op = Pooling::create(attrs);
    TensorDesc c8_desc = in_c8.desc();
    const TensorDesc pool_in_arr[] = {c8_desc};
    auto pool_descs = pool_op->getOutputTensorDesc(pool_in_arr);
    NNOPS_EXPECT_TRUE(pool_descs[0].rank == 4 && pool_descs[0].dims[2] == 1 &&
                      pool_descs[0].dims[3] == 1);
    std::vector<char> pool_buf(pool_descs[0].storage_bytes());
    auto out_c8 = test::make_packed(pool_descs[0], pool_buf.data());
    const TensorView pool_ins[] = {in_c8};
    NNOPS_EXPECT_TRUE(is_ok(pool_op->compute(out_c8, pool_ins)));

    auto unpack_op = LayoutConvert::create(TensorLayout::NCHW);
    TensorDesc c8_out_desc = out_c8.desc();
    const TensorDesc unpack_in_arr[] = {c8_out_desc};
    auto unpack_descs = unpack_op->getOutputTensorDesc(unpack_in_arr);
    std::vector<char> unpack_buf(unpack_descs[0].storage_bytes());
    auto res_nchw = test::make_planar(unpack_descs[0], unpack_buf.data());
    const TensorView unpack_ins[] = {out_c8};
    unpack_op->compute(res_nchw, unpack_ins);

    NNOPS_EXPECT_TRUE(test::allclose(res_nchw, ref, 1e-4f, 1e-4f));
}

// ============================================================
// Quantized pooling (s8/u8, PerTensor activation) — NCHWC8 SIMD
// vs NCHW reference. MaxPooling with matching dtype/scale/zp takes
// the raw-integer fast path; everything else uses the float path.
// ============================================================

static QuantParams make_per_tensor_q(float scale, int32_t zp) {
    QuantParams qp;
    qp.granularity = QuantGranularity::PerTensor;
    qp.scale = scale;
    qp.zero_point = zp;
    return qp;
}

template <typename InT, typename OutT>
static void pooling_q_test(const std::vector<int64_t>& shape,
                           const PoolingAttributes& attrs,
                           const QuantParams& in_qp,
                           const QuantParams& out_qp)
{
    constexpr DataType in_dt  = std::is_same_v<InT, int8_t> ? DataType::s8 : DataType::u8;
    constexpr DataType out_dt = std::is_same_v<OutT, int8_t> ? DataType::s8 : DataType::u8;

    auto numel = [](const std::vector<int64_t>& s) {
        int64_t n = 1;
        for (int64_t d : s) n *= d;
        return n;
    };

    // Deterministic small-integer data (exact in float for scale == 1).
    std::vector<InT> in_data(static_cast<size_t>(numel(shape)));
    for (size_t i = 0; i < in_data.size(); ++i) {
        int32_t v = static_cast<int32_t>((i * 7 + 3) % 9) - 4;  // -4..4
        if constexpr (std::is_same_v<InT, uint8_t>) v += 4;      // 0..8
        in_data[i] = static_cast<InT>(v);
    }

    TensorView in_nchw(std::span<const int64_t>(shape), in_dt, in_data.data(),
                       TensorLayout::NCHW, in_qp);

    // ---- Reference (NCHW, ground truth) ----
    auto op_ref = Pooling::create(attrs, Backend::CPU);
    TensorDesc in_desc_ref = in_nchw.desc();
    const TensorDesc ref_arr[] = {in_desc_ref};
    auto ref_desc = op_ref->getOutputTensorDesc(ref_arr)[0];
    std::vector<OutT> ref_buf(static_cast<size_t>(ref_desc.numel()));
    TensorView ref_out(std::span<const int64_t>(ref_desc.dims.data(),
                       static_cast<size_t>(ref_desc.rank)), out_dt,
                       ref_buf.data(), TensorLayout::NCHW, out_qp);
    {
        ComputeContext ctx;
        const TensorView ins[] = {in_nchw};
        backend::cpu::reference::pooling_ref(attrs, ref_out, ins, ctx, nullptr);
    }

    // ---- SIMD path: pack -> pool -> unpack ----
    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
    TensorDesc in_desc = in_nchw.desc();
    const TensorDesc pack_arr[] = {in_desc};
    auto pack_desc = pack_op->getOutputTensorDesc(pack_arr)[0];
    std::vector<InT> in_c8_buf(static_cast<size_t>(pack_desc.storage_bytes() / sizeof(InT)));
    TensorView in_c8(std::span<const int64_t>(pack_desc.dims.data(),
                     static_cast<size_t>(pack_desc.rank)), in_dt, in_c8_buf.data(),
                     pack_desc.row_pitch(), TensorLayout::NCHWC8, in_qp);
    {
        const TensorView ins[] = {in_nchw};
        TensorView outs[] = {in_c8};
        pack_op->compute(outs, ins);
    }

    auto pool_op = Pooling::create(attrs);
    TensorDesc c8_desc = in_c8.desc();
    const TensorDesc pool_arr[] = {c8_desc};
    auto pool_desc = pool_op->getOutputTensorDesc(pool_arr)[0];
    std::vector<OutT> out_c8_buf(static_cast<size_t>(pool_desc.storage_bytes() / sizeof(OutT)));
    TensorView out_c8(std::span<const int64_t>(pool_desc.dims.data(),
                      static_cast<size_t>(pool_desc.rank)), out_dt, out_c8_buf.data(),
                      pool_desc.row_pitch(), TensorLayout::NCHWC8, out_qp);
    {
        const TensorView ins[] = {in_c8};
        TensorView outs[] = {out_c8};
        pool_op->compute(outs, ins);
    }

    auto unpack_op = LayoutConvert::create(TensorLayout::NCHW);
    TensorDesc oc8_desc = out_c8.desc();
    const TensorDesc unpack_arr[] = {oc8_desc};
    auto unpack_desc = unpack_op->getOutputTensorDesc(unpack_arr)[0];
    std::vector<OutT> simd_buf(static_cast<size_t>(unpack_desc.numel()));
    TensorView simd_out(std::span<const int64_t>(unpack_desc.dims.data(),
                        static_cast<size_t>(unpack_desc.rank)), out_dt,
                        simd_buf.data(), TensorLayout::NCHW, out_qp);
    {
        const TensorView ins[] = {out_c8};
        TensorView outs[] = {simd_out};
        unpack_op->compute(outs, ins);
    }

    // ---- Compare (dequantized, tolerance = one output quant step) ----
    const float out_scale = out_qp.scale;
    const float out_zero  = static_cast<float>(out_qp.zero_point);
    const int64_t n = ref_desc.numel();
    NNOPS_EXPECT_EQ(unpack_desc.numel(), n);
    for (int64_t i = 0; i < n; ++i) {
        const float fa = (static_cast<float>(ref_buf[static_cast<size_t>(i)]) - out_zero) * out_scale;
        const float fb = (static_cast<float>(simd_buf[static_cast<size_t>(i)]) - out_zero) * out_scale;
        NNOPS_EXPECT_NEAR(fa, fb, out_scale * 1.001f);
    }
}

NNOPS_TEST(pooling_q_max_fast_path) {
    // InT == OutT, same scale/zp, !add_to -> raw-integer fast path.
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};
    pooling_q_test<int8_t, int8_t>({1, 8, 8, 8}, attrs,
        make_per_tensor_q(1.0f, 0), make_per_tensor_q(1.0f, 0));
}

NNOPS_TEST(pooling_q_max_float_path_scale) {
    // Differing scale -> float dequant/max/requant path.
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};
    pooling_q_test<int8_t, uint8_t>({1, 4, 8, 8}, attrs,
        make_per_tensor_q(2.0f, 0), make_per_tensor_q(1.0f, 128));
}

NNOPS_TEST(pooling_q_max_u8_to_s8) {
    // Mixed dtype (u8 -> s8) exercises the float path.
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};
    pooling_q_test<uint8_t, int8_t>({1, 8, 8, 8}, attrs,
        make_per_tensor_q(1.0f, 0), make_per_tensor_q(1.0f, 0));
}

NNOPS_TEST(pooling_q_avg) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};
    pooling_q_test<int8_t, int8_t>({1, 4, 6, 6}, attrs,
        make_per_tensor_q(1.0f, 0), make_per_tensor_q(1.0f, 0));
}

NNOPS_TEST(pooling_q_avg_exclude_pad) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.exclude_pad = true;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};
    pooling_q_test<int8_t, int8_t>({1, 3, 5, 5}, attrs,
        make_per_tensor_q(1.0f, 0), make_per_tensor_q(1.0f, 0));
}

NNOPS_TEST(pooling_q_partial_c8) {
    // C=5 -> partial C8 exercises pad-lane zero_point fill.
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};
    pooling_q_test<int8_t, int8_t>({1, 5, 6, 6}, attrs,
        make_per_tensor_q(1.0f, 0), make_per_tensor_q(1.0f, 0));
}

NNOPS_TEST(pooling_q_global) {
    // The quantized kernels have no whole-plane form: the global window is
    // spelled out as the full extent and the window walk reduces it.
    const auto qp = make_per_tensor_q(1.0f, 0);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.global = true;
    pooling_q_test<int8_t, int8_t>({2, 11, 9, 13}, attrs, qp, qp);

    PoolingAttributes mx = attrs;
    mx.type = PoolingType::Max;
    pooling_q_test<int8_t, int8_t>({2, 11, 9, 13}, mx, qp, qp);
}

// ============================================================
// Layout contract — a quantized plane has no kernel, and the descriptor-level
// validate() says so before anything is allocated
// ============================================================

NNOPS_TEST(pooling_planar_rejects_quantized) {
    TensorDesc in;
    in.rank   = 4;
    in.layout = TensorLayout::NCHW;
    in.dtype  = DataType::s8;
    for (int64_t d : {1, 4, 8, 8}) { in.dims.push_back(d); }

    TensorDesc out;
    out.rank   = 4;
    out.layout = TensorLayout::NCHW;
    out.dtype  = DataType::s8;
    for (int64_t d : {1, 4, 4, 4}) { out.dims.push_back(d); }

    PoolingAttributes attrs;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};

    auto op = Pooling::create(attrs);
    const TensorDesc ins[] = {in};
    const TensorDesc outs[] = {out};
    NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::unsupported);
}
