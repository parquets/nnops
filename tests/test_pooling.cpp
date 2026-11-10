/// Unit tests for Pooling — channel-first and channel-last.
///
/// Channel-first: the operator takes the tensor in the layout the graph hands it, and
/// the result is compared against the scalar reference kernel, plus a few
/// hand-computed windows that pin the arithmetic on their own.
///
/// Channel-last: the channel-first reference holds the truth; the channel-last run is
/// converted back to channel-first for the comparison.
///
/// Window walks: the kernels, the reference and the tests share one convention —
/// taps are spaced by the dilation and only the leading pad exists explicitly, the
/// trailing half of a SAME pad being implicit. The auto_pad cases therefore hand the
/// operator auto_pad and the reference the leading pad it must resolve to, worked out
/// by hand; a wrong resolution shifts every window and no other test sees it.

#include "nnops/ops/pooling.hpp"

#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"
#include "common/channel_last.hpp"

#include <array>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdint>
#include <type_traits>

using namespace nnops;

// Forward declare the reference pooling kernel (channel-first scalar, ground truth).
namespace nnops::backend::cpu::reference {
    void pooling_ref(const PoolingAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);
}

template <typename T>
static void test_channel_first_vs_ref(const PoolingAttributes& attrs, TensorView in_channel_first)
{
    auto pool_op = Pooling::create(attrs);
    TensorDesc in_desc = in_channel_first.desc();
    const TensorDesc in_arr[] = {in_desc};
    auto out_descs = pool_op->getOutputTensorDesc(in_arr);

    std::vector<T> out_buf(static_cast<size_t>(out_descs[0].numel()));
    auto out = test::make_channel_first(out_descs[0], out_buf.data());

    std::vector<T> ref_buf(out_buf.size());
    auto ref = test::make_channel_first(out_descs[0], ref_buf.data());
    if (attrs.add_to) {
        for (auto& v : out_buf) { simd::s_store(&v, 1.0f); }
        for (auto& v : ref_buf) { simd::s_store(&v, 1.0f); }
    }
    {
        ComputeContext ctx;
        const TensorView ref_ins[] = {in_channel_first};
        backend::cpu::reference::pooling_ref(attrs, ref, ref_ins, ctx, nullptr);
    }

    const TensorView ins[] = {in_channel_first};
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

static void test_channel_first_f32(const std::vector<int64_t>& shape,
                            const PoolingAttributes& attrs)
{
    auto [in_vec, in] = test::make_random_tensor(shape);
    test_channel_first_vs_ref<float>(attrs, in);
}

static void test_channel_first_f16(const std::vector<int64_t>& shape,
                            const PoolingAttributes& attrs)
{
    auto [in_vec, in] = test::make_random_f16_tensor(shape);
    test_channel_first_vs_ref<backend::cpu::half>(attrs, in);
}

// ============================================================
// 2D hand-computed windows — no reference kernel, so these pin the arithmetic
// ============================================================

NNOPS_TEST(pooling_2d_max_basic) {
    // 1x1x4x4 input, 2x2 kernel, stride=2, pad=0 → 1x1x2x2
    const int64_t shape[] = {1, 1, 4, 4};
    float in_data[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    TensorView in(shape, DataType::f32, in_data, TensorLayout::NCHW);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};

    auto op = Pooling::create(attrs);
    const TensorDesc in_desc[] = {in.desc()};
    auto out_descs = op->getOutputTensorDesc(in_desc);
    NNOPS_EXPECT_TRUE(out_descs[0].rank == 4 && out_descs[0].dims[2] == 2 &&
                      out_descs[0].dims[3] == 2);

    float out_buf[4] = {};
    const int64_t oshape[] = {1, 1, 2, 2};
    TensorView out(oshape, DataType::f32, out_buf, TensorLayout::NCHW);
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));

    const float expected[4] = {6.0f, 8.0f, 14.0f, 16.0f};
    for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_NEAR(out_buf[i], expected[i], 1e-6f); }
}

NNOPS_TEST(pooling_2d_average_basic) {
    // 1x1x2x2 input, 3x3 kernel, stride=1, pad=1 → 1x1x2x2
    const int64_t shape[] = {1, 1, 2, 2};
    float in_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    TensorView in(shape, DataType::f32, in_data, TensorLayout::NCHW);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};

    auto op = Pooling::create(attrs);
    float out_buf[4] = {};
    const int64_t oshape[] = {1, 1, 2, 2};
    TensorView out(oshape, DataType::f32, out_buf, TensorLayout::NCHW);
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));

    // sum=1+2+3+4=10, K_total=9, avg=10/9 — the pad counts in the divisor.
    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], 10.0f / 9.0f, 1e-4f);
    }
}

NNOPS_TEST(pooling_2d_average_exclude_pad) {
    const int64_t shape[] = {1, 1, 2, 2};
    float in_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    TensorView in(shape, DataType::f32, in_data, TensorLayout::NCHW);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.exclude_pad = true;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};

    auto op = Pooling::create(attrs);
    float out_buf[4] = {};
    const int64_t oshape[] = {1, 1, 2, 2};
    TensorView out(oshape, DataType::f32, out_buf, TensorLayout::NCHW);
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));

    // Every window covers all 4 input elements: sum=10, valid=4, avg=2.5.
    for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_NEAR(out_buf[i], 2.5f, 1e-4f); }
}

NNOPS_TEST(pooling_2d_class_api) {
    // Two instances: create() shares no state, so both must match the window.
    const int64_t shape[] = {1, 1, 4, 4};
    float in_data[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    TensorView in(shape, DataType::f32, in_data, TensorLayout::NCHW);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};

    const int64_t oshape[] = {1, 1, 2, 2};
    float buf_a[4] = {};
    float buf_b[4] = {};
    TensorView out_a(oshape, DataType::f32, buf_a, TensorLayout::NCHW);
    TensorView out_b(oshape, DataType::f32, buf_b, TensorLayout::NCHW);

    auto op_a = Pooling::create(attrs);
    auto op_b = Pooling::create(attrs);
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(is_ok(op_a->compute(out_a, ins)));
    NNOPS_EXPECT_TRUE(is_ok(op_b->compute(out_b, ins)));

    const float expected[4] = {6.0f, 8.0f, 14.0f, 16.0f};
    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(buf_a[i], expected[i], 1e-6f);
        NNOPS_EXPECT_NEAR(buf_b[i], expected[i], 1e-6f);
    }
}

// ============================================================
// Channel-first shape coverage — widths, channel counts and batch sizes
// ============================================================

NNOPS_TEST(pooling_channel_first_batch_stride2_wide) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 2};
    test_channel_first_f32({2, 16, 32, 32}, attrs);
}

NNOPS_TEST(pooling_channel_first_large_input) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    test_channel_first_f32({2, 3, 64, 64}, attrs);
}

NNOPS_TEST(pooling_channel_first_many_channels) {
    // C = 20: each plane is pooled on its own, so this sweeps the plane loop.
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    test_channel_first_f32({1, 20, 8, 8}, attrs);
}

NNOPS_TEST(pooling_channel_first_single_channel) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    test_channel_first_f32({1, 1, 4, 4}, attrs);
}
// ============================================================
// 3D pooling — hand-computed windows, plus depth-only padding vs the reference
// ============================================================

NNOPS_TEST(pooling_3d_max_basic) {
    // 1x1x2x4x4 input, 2x2x2 kernel, stride=2, pad=0 → 1x1x1x2x2
    const int64_t shape[] = {1, 1, 2, 4, 4};
    std::vector<float> in_buf(2 * 4 * 4);
    for (int i = 0; i < 32; ++i) {
        in_buf[i] = static_cast<float>(i + 1);
    }
    TensorView in(shape, DataType::f32, in_buf.data(), TensorLayout::NCDHW);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {2, 2, 2};
    attrs.stride       = {2, 2, 2};

    auto op = Pooling::create(attrs);
    const TensorDesc in_desc[] = {in.desc()};
    auto out_descs = op->getOutputTensorDesc(in_desc);
    NNOPS_EXPECT_TRUE(out_descs[0].rank == 5 && out_descs[0].dims[2] == 1 &&
                      out_descs[0].dims[3] == 2 && out_descs[0].dims[4] == 2);

    float out_buf[4] = {};
    const int64_t oshape[] = {1, 1, 1, 2, 2};
    TensorView out(oshape, DataType::f32, out_buf, TensorLayout::NCDHW);
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));

    const float expected[4] = {22.0f, 24.0f, 30.0f, 32.0f};
    for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_NEAR(out_buf[i], expected[i], 1e-6f); }
}

NNOPS_TEST(pooling_3d_average) {
    // 1x1x2x2x2 input, 2x2x2 kernel, stride=1, pad=0 → 1x1x1x1x1
    const int64_t shape[] = {1, 1, 2, 2, 2};
    float in_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    TensorView in(shape, DataType::f32, in_data, TensorLayout::NCDHW);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {2, 2, 2};

    auto op = Pooling::create(attrs);
    float out_buf[1] = {0.0f};
    const int64_t oshape[] = {1, 1, 1, 1, 1};
    TensorView out(oshape, DataType::f32, out_buf, TensorLayout::NCDHW);
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));

    NNOPS_EXPECT_NEAR(out_buf[0], 4.5f, 1e-4f);
}

NNOPS_TEST(pooling_channel_first_3d_padding) {
    // Padding on the depth axis only, so the front and back windows see a
    // partial kernel.
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {3, 3, 3};
    attrs.padding      = {1, 0, 0};
    test_channel_first_f32({1, 1, 4, 8, 8}, attrs);
}

// ============================================================
// Channel-first window walk — padding, stride, exclude_pad, add_to, f16, 3D
// ============================================================

NNOPS_TEST(pooling_channel_first_max_padding) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.padding      = {0, 1, 1};
    test_channel_first_f32({1, 3, 15, 15}, attrs);
}

NNOPS_TEST(pooling_channel_first_max_stride2_odd_width) {
    // Stride 2 with an odd width: the deinterleaving load needs a 16-column
    // span, so this falls to the scalar gather, and OW=7 is a short last block.
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 2, 2};
    attrs.padding      = {0, 1, 1};
    test_channel_first_f32({2, 4, 16, 13}, attrs);
}

NNOPS_TEST(pooling_channel_first_avg_padding) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {1, 3, 3};
    attrs.padding      = {0, 1, 1};
    test_channel_first_f32({1, 5, 12, 9}, attrs);
}

NNOPS_TEST(pooling_channel_first_avg_exclude_pad) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.exclude_pad = true;
    attrs.kernel_shape = {1, 3, 3};
    attrs.padding      = {0, 1, 1};
    test_channel_first_f32({1, 2, 10, 11}, attrs);
}

NNOPS_TEST(pooling_channel_first_avg_stride2) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};
    test_channel_first_f32({1, 3, 17, 9}, attrs);
}

NNOPS_TEST(pooling_channel_first_add_to) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.padding      = {0, 1, 1};
    attrs.add_to = true;
    test_channel_first_f32({1, 3, 8, 8}, attrs);
}

NNOPS_TEST(pooling_channel_first_f16) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {1, 3, 3};
    attrs.padding      = {0, 1, 1};
    test_channel_first_f16({1, 3, 10, 10}, attrs);

    // Max with padding and stride 2, so most windows touch the pad: the lane a
    // pad contributes must be below every real value. `half` is a raw-bits
    // struct with no numeric_limits specialization, so lowest() was 0 — every
    // edge window whose real values were negative came out 0 instead.
    attrs.type   = PoolingType::Max;
    attrs.stride = {1, 2, 2};
    test_channel_first_f16({2, 3, 10, 9}, attrs);
}

// ============================================================
// Dilation — the window stretches, the tap count does not
// ============================================================

NNOPS_TEST(pooling_channel_first_2d_dilation) {
    // 3x3 at dilation 2 spans five rows and columns: 7x9 leaves OH 3, OW 5, so
    // no window is clipped and the tap spacing is the only difference from a
    // plain k3 walk. A dilated tap is not an extra tap, so an average still
    // divides by nine.
    for (auto type : {PoolingType::Max, PoolingType::Average}) {
        PoolingAttributes attrs;
        attrs.type         = type;
        attrs.kernel_shape = {1, 3, 3};
        attrs.dilation     = {1, 2, 2};
        test_channel_first_f32({2, 3, 7, 9}, attrs);
    }
}

NNOPS_TEST(pooling_channel_first_2d_dilation_per_axis) {
    // Dilation on H only, with padding on both axes, so side windows leave the
    // plane and exclude_pad has a dilated walk to count over.
    PoolingAttributes attrs;
    attrs.type         = PoolingType::Average;
    attrs.kernel_shape = {1, 3, 3};
    attrs.dilation     = {1, 2, 1};
    attrs.padding      = {0, 1, 1};
    test_channel_first_f32({1, 4, 9, 10}, attrs);

    attrs.exclude_pad = true;
    test_channel_first_f32({1, 4, 9, 10}, attrs);

    attrs.type = PoolingType::Max;
    attrs.exclude_pad = false;
    test_channel_first_f32({1, 4, 9, 10}, attrs);
}

NNOPS_TEST(pooling_channel_first_3d_dilation) {
    // Depth strides by two, the other two axes not: 5 deep with a 2-tap window
    // leaves OD 3.
    for (auto type : {PoolingType::Max, PoolingType::Average}) {
        PoolingAttributes attrs;
        attrs.type         = type;
        attrs.kernel_shape = {2, 2, 2};
        attrs.dilation     = {2, 2, 1};
        attrs.padding      = {0, 1, 1};
        test_channel_first_f32({1, 3, 5, 6, 7}, attrs);
    }
}

NNOPS_TEST(pooling_channel_first_f16_dilation) {
    PoolingAttributes attrs;
    attrs.type         = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.dilation     = {1, 2, 2};
    attrs.padding      = {0, 2, 2};
    test_channel_first_f16({1, 3, 8, 8}, attrs);
}

// ============================================================
// auto_pad — resolved to explicit leading padding before the kernel
// ============================================================

namespace {

// The operator sees auto_pad and resolves it; the reference sees the pad that
// resolution must produce, handed in by the case. The output shape still comes
// from the operator, so the two agree on the window count and a wrong leading
// pad moves every window rather than changing the shape.
void check_auto_pad_resolution(const PoolingAttributes& auto_attrs,
                               const std::vector<int64_t>& shape,
                               const std::array<int64_t, 3>& leading_pad)
{
    auto [in_vec, in] = test::make_random_tensor(shape);

    auto op = Pooling::create(auto_attrs);
    const TensorDesc ds[] = {in.desc()};
    auto out_descs = op->getOutputTensorDesc(ds);
    std::vector<float> out_buf(static_cast<size_t>(out_descs[0].numel()));
    auto out = test::make_channel_first(out_descs[0], out_buf.data());
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));

    PoolingAttributes resolved = auto_attrs;
    resolved.auto_pad = PoolingAttributes::AutoPad::NOTSET;
    resolved.padding  = leading_pad;
    std::vector<float> ref_buf(out_buf.size());
    auto ref = test::make_channel_first(out_descs[0], ref_buf.data());
    ComputeContext ctx;
    backend::cpu::reference::pooling_ref(resolved, ref, ins, ctx, nullptr);

    NNOPS_EXPECT_TRUE(test::allclose(out, ref, 1e-4f, 1e-4f));
}

}  // namespace

NNOPS_TEST(pooling_2d_auto_pad_resolution) {
    // Each row carries the leading padding the resolver must produce, worked
    // out by hand. SAME_LOWER on an even plane leads by one where SAME_UPPER
    // leads by zero.
    struct Case {
        std::vector<int64_t> shape;
        PoolingType type;
        int64_t kh, kw, sh, sw, dh, dw;
        PoolingAttributes::AutoPad ap;
        std::array<int64_t, 3> pad;
        bool exclude_pad;
    };
    const Case cases[] = {
        // SAME_UPPER, odd plane at stride 2: total pad 2, so one row and one
        // column lead.
        {{1, 3, 13, 15}, PoolingType::Max, 3, 3, 2, 2, 1, 1,
         PoolingAttributes::AutoPad::SAME_UPPER, {0, 1, 1}, false},
        // SAME_UPPER, even plane: total pad 1 on H, 2 on W — the odd element
        // goes after, so H leads by none.
        {{1, 3, 8, 9}, PoolingType::Average, 3, 3, 2, 2, 1, 1,
         PoolingAttributes::AutoPad::SAME_UPPER, {0, 0, 1}, false},
        // The same plane under SAME_LOWER: the odd H element goes before.
        {{1, 3, 8, 9}, PoolingType::Average, 3, 3, 2, 2, 1, 1,
         PoolingAttributes::AutoPad::SAME_LOWER, {0, 1, 1}, true},
        // VALID: no padding whatever the stride.
        {{2, 2, 13, 13}, PoolingType::Max, 3, 3, 2, 2, 1, 1,
         PoolingAttributes::AutoPad::VALID, {0, 0, 0}, false},
        // A dilated SAME window: the extent is dilation·(k-1)+1, so the total
        // pad is 4 on H and 3 on W.
        {{1, 4, 11, 12}, PoolingType::Average, 3, 3, 2, 2, 2, 2,
         PoolingAttributes::AutoPad::SAME_UPPER, {0, 2, 1}, false},
    };

    for (const Case& c : cases) {
        PoolingAttributes a;
        a.type         = c.type;
        a.kernel_shape = {1, c.kh, c.kw};
        a.stride       = {1, c.sh, c.sw};
        a.dilation     = {1, c.dh, c.dw};
        a.auto_pad     = c.ap;
        a.exclude_pad  = c.exclude_pad;
        check_auto_pad_resolution(a, c.shape, c.pad);
    }
}

NNOPS_TEST(pooling_3d_auto_pad_resolution) {
    // One row per axis: 5 deep, 7 high, 8 wide at kernel 3 stride 2 gives total
    // pads of 2, 2 and 1, so D and H lead by one and W by none.
    PoolingAttributes a;
    a.type         = PoolingType::Average;
    a.kernel_shape = {3, 3, 3};
    a.stride       = {2, 2, 2};
    a.auto_pad     = PoolingAttributes::AutoPad::SAME_UPPER;
    a.exclude_pad  = true;
    check_auto_pad_resolution(a, {1, 2, 5, 7, 8}, {1, 1, 0});
}

NNOPS_TEST(pooling_channel_first_3d_max) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {3, 3, 3};
    attrs.padding      = {1, 1, 1};
    test_channel_first_f32({1, 2, 6, 9, 11}, attrs);
}

NNOPS_TEST(pooling_channel_first_3d_avg) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {2, 2, 2};
    attrs.stride       = {2, 2, 2};
    test_channel_first_f32({1, 2, 6, 8, 10}, attrs);
}

// ============================================================
// Global pooling — the window is the whole extent, so one output pixel per
// plane and the window fields are not read
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

NNOPS_TEST(pooling_global_channel_first_shapes) {
    // Widths picked to hit the 4x-unrolled body (a plane of at least 32
    // elements), the 8-lane body and the scalar tail; C = 5 gives five planes.
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.global = true;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride = {1, 2, 2};
    attrs.padding = {0, 1, 1};

    test_channel_first_f32({1, 5, 7, 37}, attrs);
    test_channel_first_f32({2, 3, 9, 13}, attrs);
    test_channel_first_f32({1, 1, 5, 3}, attrs);

    PoolingAttributes mx = attrs;
    mx.type = PoolingType::Max;
    test_channel_first_f32({1, 5, 7, 37}, mx);
    test_channel_first_f32({2, 3, 9, 13}, mx);
}

NNOPS_TEST(pooling_global_channel_first_f16) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.global = true;
    test_channel_first_f16({1, 3, 10, 10}, attrs);

    PoolingAttributes mx = attrs;
    mx.type = PoolingType::Max;
    test_channel_first_f16({1, 3, 10, 10}, mx);
}

NNOPS_TEST(pooling_global_channel_first_3d) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.global = true;
    test_channel_first_f32({1, 2, 4, 6, 9}, attrs);

    PoolingAttributes mx = attrs;
    mx.type = PoolingType::Max;
    test_channel_first_f32({1, 2, 4, 6, 9}, mx);
}

NNOPS_TEST(pooling_global_add_to) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.global = true;
    attrs.add_to = true;
    test_channel_first_f32({1, 3, 8, 8}, attrs);
}

// ============================================================
// Quantized pooling (s8/u8, PerTensor) — channel-last only: the kernel lanes the
// channel, where a channel-first plane lanes output width
// ============================================================

static QuantParams make_per_tensor_q(float scale, int32_t zp) {
    QuantParams qp;
    qp.granularity = QuantGranularity::PerTensor;
    qp.scale = scale;
    qp.zero_point = zp;
    return qp;
}

// validate() rejects a quantized channel-first tensor at the descriptor level.
NNOPS_TEST(pooling_channel_first_rejects_quantized) {
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

// ============================================================
// Channel-last pooling — the lane is the channel, so a plane is
// stored [N][H][W][C]. Each case keeps a copy in both orders and compares the
// Channel-last result, converted back, against the channel-first reference.
// ============================================================

/// Product of the spatial dims of a [N, C, (D,) H, W] shape.
template <class Shape>
static int64_t pool_spatial_elems(const Shape& shape)
{
    int64_t s = 1;
    for (size_t i = 2; i < shape.size(); ++i) { s *= shape[i]; }
    return s;
}

template <typename T>
static void test_channel_last_vs_ref(const std::vector<int64_t>& shape,
                             const PoolingAttributes& attrs,
                             float tol)
{
    constexpr DataType dt = std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
    const auto to_T = [](float v) -> T {
        if constexpr (std::is_same_v<T, float>) {
            return v;
        } else {
            T h;
            simd::s_store(&h, v);
            return h;
        }
    };
    const auto as_f = [](T v) -> float {
        if constexpr (std::is_same_v<T, float>) { return v; }
        else { return simd::s_load(&v); }
    };

    const int64_t N   = shape[0];
    const int64_t C   = shape[1];
    const int64_t inS = pool_spatial_elems(shape);
    const TensorLayout cl = (shape.size() == 5) ? TensorLayout::NDHWC
                                                : TensorLayout::NHWC;

    auto [in_vec, in_channel_first] = test::make_random_tensor(shape);

    auto op = Pooling::create(attrs, Backend::CPU);

    // ---- Reference, in the channel-first order ----------------------------------
    const TensorDesc ref_in[] = {in_channel_first.desc()};
    auto ref_descs = op->getOutputTensorDesc(ref_in);
    const int64_t outS  = pool_spatial_elems(ref_descs[0].dims);
    const int64_t out_n = N * C * outS;

    std::vector<float> ref_buf(static_cast<size_t>(out_n),
                               attrs.add_to ? 1.0f : 0.0f);
    auto ref = test::make_channel_first(ref_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        const TensorView r[] = {in_channel_first};
        backend::cpu::reference::pooling_ref(attrs, ref, r, ctx, nullptr);
    }

    // ---- The same values, in the channel-last order ----------------------
    const auto in_cl_f = test::to_channel_last<float>(in_vec, N, C, inS);
    std::vector<T> in_cl(in_cl_f.size());
    for (size_t i = 0; i < in_cl.size(); ++i) { in_cl[i] = to_T(in_cl_f[i]); }

    TensorView in_v(std::span<const int64_t>(shape), dt, in_cl.data(), cl);
    const TensorDesc cl_in[] = {in_v.desc()};
    auto out_descs = op->getOutputTensorDesc(cl_in);
    NNOPS_EXPECT_TRUE(out_descs[0].layout == cl);
    NNOPS_EXPECT_EQ(out_descs[0].numel(), out_n);

    std::vector<T> out_buf(static_cast<size_t>(out_n),
                           to_T(attrs.add_to ? 1.0f : 0.0f));
    TensorView out_v(std::span<const int64_t>(out_descs[0].dims.data(),
                                              static_cast<size_t>(out_descs[0].rank)),
                     dt, out_buf.data(), cl);

    const TensorView ins[] = {in_v};
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out_v, ins)));

    // ---- Back to the channel-first order for the comparison ----------------------
    const auto got = test::to_channel_first<T>(out_buf, N, C, outS);
    for (int64_t i = 0; i < out_n; ++i) {
        NNOPS_EXPECT_NEAR(as_f(got[static_cast<size_t>(i)]),
                          ref_buf[static_cast<size_t>(i)], tol);
    }
}

NNOPS_TEST(pooling_channel_last_max_padding) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};
    test_channel_last_vs_ref<float>({1, 8, 10, 10}, attrs, 1e-4f);
}

NNOPS_TEST(pooling_channel_last_max_stride2_odd_width) {
    // Odd width and a short channel block (C=5) at once: the 16-lane chunk
    // runs with five live lanes.
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 2, 2};
    attrs.padding      = {0, 1, 1};
    test_channel_last_vs_ref<float>({2, 5, 13, 15}, attrs, 1e-4f);
}

NNOPS_TEST(pooling_channel_last_avg_padding) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};
    test_channel_last_vs_ref<float>({1, 16, 9, 9}, attrs, 1e-4f);
}

NNOPS_TEST(pooling_channel_last_avg_exclude_pad) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.exclude_pad  = true;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};
    test_channel_last_vs_ref<float>({1, 3, 7, 7}, attrs, 1e-4f);
}

NNOPS_TEST(pooling_channel_last_add_to) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.add_to       = true;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};
    test_channel_last_vs_ref<float>({1, 8, 8, 8}, attrs, 1e-4f);
}

NNOPS_TEST(pooling_channel_last_f16) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};
    test_channel_last_vs_ref<backend::cpu::half>({1, 8, 8, 8}, attrs, 5e-3f);
}

NNOPS_TEST(pooling_channel_last_global) {
    // The whole-plane form: one pixel out per image, so with few rows the
    // split has nothing but the channel axis.
    PoolingAttributes attrs;
    attrs.type   = PoolingType::Average;
    attrs.global = true;
    test_channel_last_vs_ref<float>({2, 11, 9, 13}, attrs, 1e-4f);

    PoolingAttributes mx = attrs;
    mx.type = PoolingType::Max;
    test_channel_last_vs_ref<float>({2, 11, 9, 13}, mx, 1e-4f);
}

NNOPS_TEST(pooling_channel_last_3d) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {2, 2, 2};
    attrs.stride       = {2, 2, 2};
    test_channel_last_vs_ref<float>({1, 4, 6, 8, 8}, attrs, 1e-4f);
}

NNOPS_TEST(pooling_channel_last_3d_avg_padding) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {2, 2, 2};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {1, 1, 1};
    test_channel_last_vs_ref<float>({1, 5, 4, 5, 6}, attrs, 1e-4f);
}

NNOPS_TEST(pooling_channel_last_2d_dilation) {
    // C = 5 ends on a short block, and the dilated window keeps the row-group
    // walk's bounds checks busy on the clipped side windows.
    PoolingAttributes attrs;
    attrs.type         = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.dilation     = {1, 2, 2};
    test_channel_last_vs_ref<float>({1, 5, 7, 9}, attrs, 1e-4f);

    attrs.type        = PoolingType::Average;
    attrs.exclude_pad = true;
    attrs.padding     = {0, 1, 1};
    test_channel_last_vs_ref<float>({2, 8, 9, 11}, attrs, 1e-4f);

    attrs.exclude_pad = false;
    attrs.padding     = {0, 0, 0};
    attrs.kernel_shape = {1, 2, 2};
    attrs.dilation     = {1, 2, 2};
    test_channel_last_vs_ref<backend::cpu::half>({1, 16, 8, 8}, attrs, 5e-3f);
}

NNOPS_TEST(pooling_channel_last_3d_dilation) {
    PoolingAttributes attrs;
    attrs.type         = PoolingType::Average;
    attrs.kernel_shape = {2, 2, 2};
    attrs.dilation     = {2, 2, 1};
    test_channel_last_vs_ref<float>({1, 6, 5, 6, 7}, attrs, 1e-4f);
}

// ---- Quantized, channel-last -------------------------------------------

template <typename InT, typename OutT>
static void pooling_q_channel_last_test(const std::vector<int64_t>& shape,
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
    const int64_t N   = shape[0];
    const int64_t C   = shape[1];
    const int64_t inS = pool_spatial_elems(shape);

    std::vector<InT> in_data(static_cast<size_t>(numel(shape)));
    for (size_t i = 0; i < in_data.size(); ++i) {
        int32_t v = static_cast<int32_t>((i * 7 + 3) % 9) - 4;
        if constexpr (std::is_same_v<InT, uint8_t>) { v += 4; }
        in_data[i] = static_cast<InT>(v);
    }

    TensorView in_channel_first(std::span<const int64_t>(shape), in_dt, in_data.data(),
                       TensorLayout::NCHW, in_qp);

    auto op = Pooling::create(attrs, Backend::CPU);

    // ---- Reference (channel-first, ground truth) ----
    const TensorDesc ref_in[] = {in_channel_first.desc()};
    auto ref_desc = op->getOutputTensorDesc(ref_in)[0];
    const int64_t outS = pool_spatial_elems(ref_desc.dims);
    std::vector<OutT> ref_buf(static_cast<size_t>(N * C * outS));
    TensorView ref_out(std::span<const int64_t>(ref_desc.dims.data(),
                                                static_cast<size_t>(ref_desc.rank)),
                       out_dt, ref_buf.data(), TensorLayout::NCHW, out_qp);
    {
        ComputeContext ctx;
        const TensorView ins[] = {in_channel_first};
        backend::cpu::reference::pooling_ref(attrs, ref_out, ins, ctx, nullptr);
    }

    // ---- Channel-last run --------------------------------------------------
    auto in_cl = test::to_channel_last<InT>(in_data, N, C, inS);
    const TensorLayout cl = (shape.size() == 5) ? TensorLayout::NDHWC
                                                : TensorLayout::NHWC;
    TensorView in_v(std::span<const int64_t>(shape), in_dt, in_cl.data(), cl, in_qp);

    const TensorDesc cl_in[] = {in_v.desc()};
    auto out_desc = op->getOutputTensorDesc(cl_in)[0];
    NNOPS_EXPECT_TRUE(out_desc.layout == cl);

    std::vector<OutT> out_buf(static_cast<size_t>(N * C * outS));
    TensorView out_v(std::span<const int64_t>(out_desc.dims.data(),
                                              static_cast<size_t>(out_desc.rank)),
                     out_dt, out_buf.data(), cl, out_qp);
    const TensorView ins[] = {in_v};
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out_v, ins)));

    // ---- Compare dequantized, within one output step ----------------------
    const auto got = test::to_channel_first<OutT>(out_buf, N, C, outS);
    const float out_scale = out_qp.scale;
    const float out_zero  = static_cast<float>(out_qp.zero_point);
    const int64_t n = N * C * outS;
    for (int64_t i = 0; i < n; ++i) {
        const float fa = (static_cast<float>(ref_buf[static_cast<size_t>(i)]) - out_zero) * out_scale;
        const float fb = (static_cast<float>(got[static_cast<size_t>(i)]) - out_zero) * out_scale;
        NNOPS_EXPECT_NEAR(fa, fb, out_scale * 1.001f);
    }
}

NNOPS_TEST(pooling_q_channel_last_max_fast_path) {
    // Matching scale/zero_point and dtype: the raw-integer max, on full blocks.
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};
    pooling_q_channel_last_test<int8_t, int8_t>({1, 8, 8, 8}, attrs,
        make_per_tensor_q(1.0f, 0), make_per_tensor_q(1.0f, 0));
}

NNOPS_TEST(pooling_q_channel_last_max_short_block) {
    // C=5: the block is five lanes, short of the eight-lane fast path, so it
    // dequantizes.
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};
    pooling_q_channel_last_test<int8_t, int8_t>({1, 5, 6, 6}, attrs,
        make_per_tensor_q(1.0f, 0), make_per_tensor_q(1.0f, 0));
}

NNOPS_TEST(pooling_q_channel_last_max_scale_change) {
    // Differing scale and zero_point: the max cannot run on the raw integers,
    // so it dequantizes, compares, and requantizes.
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};
    pooling_q_channel_last_test<int8_t, uint8_t>({1, 4, 8, 8}, attrs,
        make_per_tensor_q(2.0f, 0), make_per_tensor_q(1.0f, 128));
}

NNOPS_TEST(pooling_q_channel_last_u8_to_s8) {
    // Mixed dtype (u8 -> s8) takes the same float path.
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};
    pooling_q_channel_last_test<uint8_t, int8_t>({1, 8, 8, 8}, attrs,
        make_per_tensor_q(1.0f, 0), make_per_tensor_q(1.0f, 0));
}

NNOPS_TEST(pooling_q_channel_last_avg_exclude_pad) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.exclude_pad  = true;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};
    pooling_q_channel_last_test<int8_t, int8_t>({1, 3, 5, 5}, attrs,
        make_per_tensor_q(1.0f, 0), make_per_tensor_q(1.0f, 0));
}

NNOPS_TEST(pooling_q_channel_last_avg_reqantize) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};
    pooling_q_channel_last_test<int8_t, uint8_t>({1, 6, 6, 7}, attrs,
        make_per_tensor_q(2.0f, 0), make_per_tensor_q(1.0f, 128));
}

NNOPS_TEST(pooling_q_channel_last_global) {
    // Quantized global takes the window walk with the window spelled out; the
    // Channel-last driver then runs it a row group at a time.
    const auto qp = make_per_tensor_q(1.0f, 0);
    PoolingAttributes attrs;
    attrs.type   = PoolingType::Average;
    attrs.global = true;
    pooling_q_channel_last_test<int8_t, int8_t>({2, 11, 9, 13}, attrs, qp, qp);

    PoolingAttributes mx = attrs;
    mx.type = PoolingType::Max;
    pooling_q_channel_last_test<int8_t, int8_t>({2, 11, 9, 13}, mx, qp, qp);
}

// ============================================================
// Adaptive pooling — the window of output index o spans
// [floor(o·I/O), ceil((o+1)·I/O)) on each axis, so a size whose I % O != 0 has
// no fixed kernel/stride form. The channel-first reference walks the same bounds.
// ============================================================

NNOPS_TEST(pooling_adaptive_deterministic) {
    // 1x1x3x3 → 1x1x2x2: H windows are [0,2) and [1,3); W windows the same.
    const int64_t shape[] = {1, 1, 3, 3};
    float data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    TensorView in(shape, DataType::f32, data, TensorLayout::NCHW);

    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.output_size = {0, 2, 2};

    auto op = Pooling::create(attrs);
    const TensorDesc in_desc[] = {in.desc()};
    auto out_descs = op->getOutputTensorDesc(in_desc);
    NNOPS_EXPECT_TRUE(out_descs[0].rank == 4 && out_descs[0].dims[2] == 2 &&
                      out_descs[0].dims[3] == 2);

    const int64_t oshape[] = {1, 1, 2, 2};
    float out_buf[4] = {};
    TensorView out(oshape, DataType::f32, out_buf, TensorLayout::NCHW);
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));

    // (1,2,4,5)/4, (2,3,5,6)/4, (4,5,7,8)/4, (5,6,8,9)/4
    const float expected[4] = {3.0f, 4.0f, 6.0f, 7.0f};
    for (int i = 0; i < 4; ++i) { NNOPS_EXPECT_NEAR(out_buf[i], expected[i], 1e-6f); }

    PoolingAttributes mx = attrs;
    mx.type = PoolingType::Max;
    auto mx_op = Pooling::create(mx);
    NNOPS_EXPECT_TRUE(is_ok(mx_op->compute(out, ins)));
    const float expected_max[4] = {5.0f, 6.0f, 8.0f, 9.0f};
    for (int i = 0; i < 4; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], expected_max[i], 1e-6f);
    }
}

NNOPS_TEST(pooling_adaptive_ones_is_global) {
    // An all-ones output is the global window; the op routes it to the plane
    // scan, so the values must match a global request exactly.
    PoolingAttributes ad;
    ad.type = PoolingType::Average;
    ad.output_size = {0, 1, 1};

    PoolingAttributes gl = ad;
    gl.global = true;
    gl.output_size = {0, 0, 0};

    for (PoolingAttributes a : {ad, gl}) {
        auto [in_vec, in] = test::make_random_tensor({2, 3, 7, 9});
        auto op = Pooling::create(a);
        const TensorDesc in_desc[] = {in.desc()};
        auto out_descs = op->getOutputTensorDesc(in_desc);
        NNOPS_EXPECT_TRUE(out_descs[0].dims[2] == 1 && out_descs[0].dims[3] == 1);

        std::vector<float> out_buf(static_cast<size_t>(out_descs[0].numel()));
        auto out = test::make_channel_first(out_descs[0], out_buf.data());
        const TensorView ins[] = {in};
        NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));

        // Same plane, same mean.
        float sum = 0.0f;
        const float* p = in.ptr<float>();
        for (int64_t i = 0; i < 7 * 9; ++i) { sum += p[i]; }
        NNOPS_EXPECT_NEAR(out_buf[0], sum / 63.0f, 1e-4f);
    }
}

NNOPS_TEST(pooling_adaptive_channel_first_shapes) {
    // Odd input sizes and outputs that do not divide them: 7→3, 9→4, 13→5.
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.output_size = {0, 3, 4};
    test_channel_first_f32({1, 4, 7, 9}, attrs);
    test_channel_first_f32({2, 5, 13, 5}, attrs);

    PoolingAttributes mx = attrs;
    mx.type = PoolingType::Max;
    test_channel_first_f32({1, 4, 7, 9}, mx);
    test_channel_first_f32({2, 3, 11, 6}, mx);

    // A larger output than half the input, so most windows overlap.
    PoolingAttributes wide;
    wide.type = PoolingType::Average;
    wide.output_size = {0, 7, 11};
    test_channel_first_f32({1, 3, 8, 12}, wide);
}

NNOPS_TEST(pooling_adaptive_channel_first_3d) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.output_size = {2, 3, 4};
    test_channel_first_f32({1, 2, 5, 7, 9}, attrs);

    PoolingAttributes mx = attrs;
    mx.type = PoolingType::Max;
    test_channel_first_f32({1, 2, 5, 7, 9}, mx);
}

NNOPS_TEST(pooling_adaptive_channel_first_f16) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.output_size = {0, 3, 4};
    test_channel_first_f16({1, 3, 10, 10}, attrs);

    PoolingAttributes mx = attrs;
    mx.type = PoolingType::Max;
    test_channel_first_f16({1, 3, 10, 10}, mx);
}

NNOPS_TEST(pooling_adaptive_add_to) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.output_size = {0, 3, 5};
    attrs.add_to = true;
    test_channel_first_f32({1, 3, 9, 11}, attrs);

    PoolingAttributes mx = attrs;
    mx.type = PoolingType::Max;
    test_channel_first_f32({1, 3, 9, 11}, mx);
}

NNOPS_TEST(pooling_adaptive_channel_last) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.output_size = {0, 3, 4};
    test_channel_last_vs_ref<float>({2, 5, 9, 11}, attrs, 1e-4f);

    PoolingAttributes mx = attrs;
    mx.type = PoolingType::Max;
    test_channel_last_vs_ref<float>({2, 11, 7, 13}, mx, 1e-4f);
}

NNOPS_TEST(pooling_adaptive_channel_last_3d) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.output_size = {2, 2, 3};
    test_channel_last_vs_ref<float>({1, 4, 5, 6, 7}, attrs, 1e-4f);
}

NNOPS_TEST(pooling_adaptive_channel_last_f16) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.output_size = {0, 3, 3};
    test_channel_last_vs_ref<backend::cpu::half>({1, 8, 8, 8}, attrs, 5e-3f);
}

NNOPS_TEST(pooling_adaptive_rejects_quantized) {
    TensorDesc in;
    in.rank   = 4;
    in.layout = TensorLayout::NHWC;
    in.dtype  = DataType::s8;
    for (int64_t d : {1, 4, 8, 8}) { in.dims.push_back(d); }

    TensorDesc out;
    out.rank   = 4;
    out.layout = TensorLayout::NHWC;
    out.dtype  = DataType::s8;
    for (int64_t d : {1, 4, 3, 3}) { out.dims.push_back(d); }

    PoolingAttributes attrs;
    attrs.output_size = {0, 3, 3};

    auto op = Pooling::create(attrs);
    const TensorDesc ins[] = {in};
    const TensorDesc outs[] = {out};
    NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::unsupported);
}
