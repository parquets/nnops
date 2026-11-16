/// Unit tests for Pooling — channel-last.
///
/// The channel-first scalar reference holds the truth; the channel-last run is
/// converted back to channel-first for the comparison. The quantization lane is
/// the channel, so a quantized plane is channel-last as well.
///
/// Window walks: the kernels, the reference and the tests share one convention —
/// taps are spaced by the dilation and only the leading pad exists explicitly, the
/// trailing half of a SAME pad being implicit.

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

// ============================================================
// Channel-last pooling — the lane is the channel, so a plane is
// stored [N][H][W][C]. Each case keeps a copy in both orders and compares the
// channel-last result, converted back, against the channel-first reference.
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
                       TensorLayout::Dense, in_qp);

    auto op = Pooling::create(attrs, Backend::CPU);

    // ---- Reference (channel-first, ground truth) ----
    const TensorDesc ref_in[] = {in_channel_first.desc()};
    auto ref_desc = op->getOutputTensorDesc(ref_in)[0];
    const int64_t outS = pool_spatial_elems(ref_desc.dims);
    std::vector<OutT> ref_buf(static_cast<size_t>(N * C * outS));
    TensorView ref_out(std::span<const int64_t>(ref_desc.dims.data(),
                                                static_cast<size_t>(ref_desc.rank)),
                       out_dt, ref_buf.data(), TensorLayout::Dense, out_qp);
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
