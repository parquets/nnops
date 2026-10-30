/// Unit tests for Pooling operator — planar (NCHW/NCDHW) and channel-last
/// (NHWC/NDHWC).
///
/// Planar: the operator takes the tensor as a graph hands it over, and the
/// result is compared with the scalar reference kernel (plus a few
/// hand-computed windows).
///
/// Channel-last: the same values in both orders — the planar reference holds
/// the truth, the channel-last run is converted back for the comparison.

#include "nnops/ops/pooling.hpp"

#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"
#include "common/channel_last.hpp"

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
// Planar path — the operator takes NCHW/NCDHW as it is, and the reference
// kernel is the thing to agree with
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
// 2D hand-computed windows — no reference kernel in the loop, so these pin the
// arithmetic itself rather than agreement between two implementations
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
    // Two instances: create() must not hand out shared state, so both runs
    // agree with each other and with the hand-computed window.
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
// Planar shape coverage — the widths, channel counts and batch sizes the
// channel-packed cases used to sweep
// ============================================================

NNOPS_TEST(pooling_planar_batch_stride2_wide) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 2};
    test_planar_f32({2, 16, 32, 32}, attrs);
}

NNOPS_TEST(pooling_planar_large_input) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    test_planar_f32({2, 3, 64, 64}, attrs);
}

NNOPS_TEST(pooling_planar_many_channels) {
    // C = 20: each plane is pooled on its own, so the plane loop is what this
    // sweeps.
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    test_planar_f32({1, 20, 8, 8}, attrs);
}

NNOPS_TEST(pooling_planar_single_channel) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    test_planar_f32({1, 1, 4, 4}, attrs);
}
// ============================================================
// 3D hand-computed windows
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

NNOPS_TEST(pooling_planar_3d_padding) {
    // Padding on the depth axis only, so the front and back windows see a
    // partial kernel.
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {3, 3, 3};
    attrs.padding      = {1, 0, 0};
    test_planar_f32({1, 1, 4, 8, 8}, attrs);
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
    // the scalar tail; C = 5 leaves an odd plane count over the batch.
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


// ============================================================
// Quantized pooling (s8/u8, PerTensor activation) — channel-last only: the
// quantization axis is the channel, which a planar plane does not keep
// together (pooling_planar_rejects_quantized pins the rejection)
// ============================================================

static QuantParams make_per_tensor_q(float scale, int32_t zp) {
    QuantParams qp;
    qp.granularity = QuantGranularity::PerTensor;
    qp.scale = scale;
    qp.zero_point = zp;
    return qp;
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

// ============================================================
// Channel-last (NHWC/NDHWC) pooling — the lane is the channel, so the plane
// is stored [N][H][W][C]. Every case keeps one copy in each order and compares
// the channel-last result, converted back, against the planar reference.
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
static void test_nhwc_vs_ref(const std::vector<int64_t>& shape,
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

    auto [in_vec, in_planar] = test::make_random_tensor(shape);

    auto op = Pooling::create(attrs, Backend::CPU);

    // ---- Reference, in the planar order ----------------------------------
    const TensorDesc ref_in[] = {in_planar.desc()};
    auto ref_descs = op->getOutputTensorDesc(ref_in);
    const int64_t outS  = pool_spatial_elems(ref_descs[0].dims);
    const int64_t out_n = N * C * outS;

    std::vector<float> ref_buf(static_cast<size_t>(out_n),
                               attrs.add_to ? 1.0f : 0.0f);
    auto ref = test::make_planar(ref_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        const TensorView r[] = {in_planar};
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

    // ---- Back to the planar order for the comparison ----------------------
    const auto got = test::to_channel_first<T>(out_buf, N, C, outS);
    for (int64_t i = 0; i < out_n; ++i) {
        NNOPS_EXPECT_NEAR(as_f(got[static_cast<size_t>(i)]),
                          ref_buf[static_cast<size_t>(i)], tol);
    }
}

NNOPS_TEST(pooling_nhwc_max_padding) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};
    test_nhwc_vs_ref<float>({1, 8, 10, 10}, attrs, 1e-4f);
}

NNOPS_TEST(pooling_nhwc_max_stride2_odd_width) {
    // Odd width and a partial block (C=5) at once: the tail of every row is
    // shorter than a vector, and the row group never fills a full W block.
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 2, 2};
    attrs.padding      = {0, 1, 1};
    test_nhwc_vs_ref<float>({2, 5, 13, 15}, attrs, 1e-4f);
}

NNOPS_TEST(pooling_nhwc_avg_padding) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};
    test_nhwc_vs_ref<float>({1, 16, 9, 9}, attrs, 1e-4f);
}

NNOPS_TEST(pooling_nhwc_avg_exclude_pad) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.exclude_pad  = true;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};
    test_nhwc_vs_ref<float>({1, 3, 7, 7}, attrs, 1e-4f);
}

NNOPS_TEST(pooling_nhwc_add_to) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.add_to       = true;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};
    test_nhwc_vs_ref<float>({1, 8, 8, 8}, attrs, 1e-4f);
}

NNOPS_TEST(pooling_nhwc_f16) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};
    test_nhwc_vs_ref<backend::cpu::half>({1, 8, 8, 8}, attrs, 5e-3f);
}

NNOPS_TEST(pooling_nhwc_global) {
    // The whole-plane form: one pixel out per image, so a poor task split
    // (many workers against few rows) has nothing left but the channel axis.
    PoolingAttributes attrs;
    attrs.type   = PoolingType::Average;
    attrs.global = true;
    test_nhwc_vs_ref<float>({2, 11, 9, 13}, attrs, 1e-4f);

    PoolingAttributes mx = attrs;
    mx.type = PoolingType::Max;
    test_nhwc_vs_ref<float>({2, 11, 9, 13}, mx, 1e-4f);
}

NNOPS_TEST(pooling_nhwc_3d) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {2, 2, 2};
    attrs.stride       = {2, 2, 2};
    test_nhwc_vs_ref<float>({1, 4, 6, 8, 8}, attrs, 1e-4f);
}

NNOPS_TEST(pooling_nhwc_3d_avg_padding) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {2, 2, 2};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {1, 1, 1};
    test_nhwc_vs_ref<float>({1, 5, 4, 5, 6}, attrs, 1e-4f);
}

// ---- Quantized, channel-last -------------------------------------------

template <typename InT, typename OutT>
static void pooling_q_nhwc_test(const std::vector<int64_t>& shape,
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

    TensorView in_nchw(std::span<const int64_t>(shape), in_dt, in_data.data(),
                       TensorLayout::NCHW, in_qp);

    auto op = Pooling::create(attrs, Backend::CPU);

    // ---- Reference (NCHW, ground truth) ----
    const TensorDesc ref_in[] = {in_nchw.desc()};
    auto ref_desc = op->getOutputTensorDesc(ref_in)[0];
    const int64_t outS = pool_spatial_elems(ref_desc.dims);
    std::vector<OutT> ref_buf(static_cast<size_t>(N * C * outS));
    TensorView ref_out(std::span<const int64_t>(ref_desc.dims.data(),
                                                static_cast<size_t>(ref_desc.rank)),
                       out_dt, ref_buf.data(), TensorLayout::NCHW, out_qp);
    {
        ComputeContext ctx;
        const TensorView ins[] = {in_nchw};
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

NNOPS_TEST(pooling_q_nhwc_max_fast_path) {
    // Matching scale/zero_point and dtype: the raw-integer max, on full blocks.
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};
    pooling_q_nhwc_test<int8_t, int8_t>({1, 8, 8, 8}, attrs,
        make_per_tensor_q(1.0f, 0), make_per_tensor_q(1.0f, 0));
}

NNOPS_TEST(pooling_q_nhwc_max_short_block) {
    // C=5: the last block is three lanes, so it takes the dequantize path.
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};
    pooling_q_nhwc_test<int8_t, int8_t>({1, 5, 6, 6}, attrs,
        make_per_tensor_q(1.0f, 0), make_per_tensor_q(1.0f, 0));
}

NNOPS_TEST(pooling_q_nhwc_max_scale_change) {
    // Differing scale and zero_point: the max cannot run on the raw integers,
    // so it dequantizes, compares, and requantizes.
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};
    pooling_q_nhwc_test<int8_t, uint8_t>({1, 4, 8, 8}, attrs,
        make_per_tensor_q(2.0f, 0), make_per_tensor_q(1.0f, 128));
}

NNOPS_TEST(pooling_q_nhwc_u8_to_s8) {
    // Mixed dtype (u8 -> s8) takes the same float path.
    PoolingAttributes attrs;
    attrs.type = PoolingType::Max;
    attrs.kernel_shape = {1, 2, 2};
    attrs.stride       = {1, 2, 2};
    pooling_q_nhwc_test<uint8_t, int8_t>({1, 8, 8, 8}, attrs,
        make_per_tensor_q(1.0f, 0), make_per_tensor_q(1.0f, 0));
}

NNOPS_TEST(pooling_q_nhwc_avg_exclude_pad) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.exclude_pad  = true;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};
    pooling_q_nhwc_test<int8_t, int8_t>({1, 3, 5, 5}, attrs,
        make_per_tensor_q(1.0f, 0), make_per_tensor_q(1.0f, 0));
}

NNOPS_TEST(pooling_q_nhwc_avg_reqantize) {
    PoolingAttributes attrs;
    attrs.type = PoolingType::Average;
    attrs.kernel_shape = {1, 3, 3};
    attrs.stride       = {1, 1, 1};
    attrs.padding      = {0, 1, 1};
    pooling_q_nhwc_test<int8_t, uint8_t>({1, 6, 6, 7}, attrs,
        make_per_tensor_q(2.0f, 0), make_per_tensor_q(1.0f, 128));
}

NNOPS_TEST(pooling_q_nhwc_global) {
    // Quantized global takes the window walk with the window spelled out; the
    // channel-last driver then runs it a row group at a time.
    const auto qp = make_per_tensor_q(1.0f, 0);
    PoolingAttributes attrs;
    attrs.type   = PoolingType::Average;
    attrs.global = true;
    pooling_q_nhwc_test<int8_t, int8_t>({2, 11, 9, 13}, attrs, qp, qp);

    PoolingAttributes mx = attrs;
    mx.type = PoolingType::Max;
    pooling_q_nhwc_test<int8_t, int8_t>({2, 11, 9, 13}, mx, qp, qp);
}
