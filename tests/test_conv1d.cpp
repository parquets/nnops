/// @file test_conv1d.cpp
/// @brief Conv1D against the 2D reference computed on the equivalent lift.
///
/// Conv1D is the 2D convolution on [N, C, 1, L]: the oracle here is conv2d_ref
/// handed a rank-4 view of the same bytes, with the lift written out locally
/// rather than taken from conv1d_common.h — so a lift the operator got wrong
/// cannot hide behind the same mistake in the expectation.
///
/// Covered: stride/dilation/padding, groups, depthwise (the hand-off to the
/// depthwise kernels), bias, both layouts, f32/f16, the quantized route
/// (u8/s8 activations, s8 per-channel weights, fp and integer destinations) and
/// the L-chunked parallel path, which must be bit-exact against the serial one.

#include "nnops/ops/conv1d.hpp"
#include "nnops/ops/conv2d.hpp"   // the reference's attribute type
#include "nnops/core/compute_context.hpp"
#include "nnops/detail/half.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/channel_last.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <type_traits>
#include <vector>

using namespace nnops;
using nnops::backend::cpu::half;

namespace nnops::backend::cpu::reference {
void conv2d_ref(const Conv2DAttributes& attrs, TensorView& output,
                std::span<const TensorView> inputs, const ComputeContext& ctx,
                void* workspace);
void conv2d_int8_ref(const Conv2DAttributes& attrs, TensorView& output,
                     std::span<const TensorView> inputs, const ComputeContext& ctx,
                     void* workspace);
}

namespace {

void* vdata(const TensorView& v) {
    return const_cast<void*>(static_cast<const void*>(v.ptr<uint8_t>()));
}

/// [N, C, L] → [N, C, 1, L] — the local restatement of the operator's lift.
TensorView wide_act(const TensorView& v) {
    const int64_t s[4] = {v.shape(0), v.shape(1), 1, v.shape(2)};
    return TensorView(std::span<const int64_t>(s, 4), v.data_type(), vdata(v),
                      v.pitch(), v.layout(), v.quant_params());
}

/// OIHW [OC, IC/G, K] → [OC, IC/G, 1, K]; HWIO [K, IC/G, OC] → [1, K, IC/G, OC].
TensorView wide_w(const TensorView& w) {
    const bool hwio = is_channel_last_weight(w.layout());
    const int64_t s[4] = {hwio ? 1 : w.shape(0), w.shape(hwio ? 0 : 1),
                          hwio ? w.shape(1) : 1, w.shape(2)};
    return TensorView(std::span<const int64_t>(s, 4), w.data_type(), vdata(w),
                      w.pitch(), w.layout(), w.quant_params());
}

Conv2DAttributes to_2d(const Conv1DAttributes& a) {
    Conv2DAttributes c{};
    c.kernel_size = {1, a.kernel_size};
    c.stride      = {1, a.stride};
    c.dilation    = {1, a.dilation};
    c.padding     = {0, a.padding};
    c.groups      = a.groups;
    return c;
}

struct Geom {
    const char* name;
    int64_t N, IC, L, OC, G, K, S, D, P;
    bool bias;
};

/// Run the operator on a channel-first f32 tensor and compare with the 2D
/// reference on the lifted views.
void run_f32(const Geom& g)
{
    const int64_t ICG = g.IC / g.G;
    auto [in_vec, in] = test::make_random_tensor({g.N, g.IC, g.L});
    auto [w_vec, w]   = test::make_random_tensor({g.OC, ICG, g.K});
    std::vector<float> b_vec;
    TensorView b;
    if (g.bias) {
        auto p = test::make_random_tensor({g.OC});
        b_vec = std::move(p.first);
        b = p.second;
    }

    Conv1DAttributes attrs;
    attrs.kernel_size = g.K;
    attrs.stride      = g.S;
    attrs.dilation    = g.D;
    attrs.padding     = g.P;
    attrs.groups      = g.G;

    auto op = Conv1D::create(attrs);
    const TensorDesc in_d = in.desc();
    const TensorDesc w_d  = w.desc();
    const TensorDesc b_d  = b.desc();
    std::vector<TensorDesc> out_d;
    if (g.bias) {
        const TensorDesc ds[] = {in_d, w_d, b_d};
        out_d = op->getOutputTensorDesc(ds);
    } else {
        const TensorDesc ds[] = {in_d, w_d};
        out_d = op->getOutputTensorDesc(ds);
    }
    NNOPS_EXPECT_TRUE(out_d[0].rank == 3);
    NNOPS_EXPECT_TRUE(out_d[0].dims[0] == g.N && out_d[0].dims[1] == g.OC);

    std::vector<float> out_buf(static_cast<size_t>(out_d[0].numel()));
    TensorView out = test::make_channel_first(out_d[0], out_buf.data());

    std::vector<float> ref_buf(out_buf.size());
    TensorView ref = test::make_channel_first(out_d[0], ref_buf.data());

    {
        ComputeContext ctx;
        const TensorView in4 = wide_act(in);
        const TensorView w4  = wide_w(w);
        TensorView ref4 = wide_act(ref);
        if (g.bias) {
            const TensorView ins[] = {in4, w4, b};
            backend::cpu::reference::conv2d_ref(to_2d(attrs), ref4, ins, ctx, nullptr);
        } else {
            const TensorView ins[] = {in4, w4};
            backend::cpu::reference::conv2d_ref(to_2d(attrs), ref4, ins, ctx, nullptr);
        }
    }

    if (g.bias) {
        const TensorView ins[] = {in, w, b};
        NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));
    } else {
        const TensorView ins[] = {in, w};
        NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));
    }
    NNOPS_EXPECT_TRUE(test::allclose(out, ref, 1e-4f, 1e-4f));
}

void run_f16(const Geom& g)
{
    const int64_t ICG = g.IC / g.G;
    auto [in_f32, in_view] = test::make_random_tensor({g.N, g.IC, g.L});
    auto [w_f32, w_view]   = test::make_random_tensor({g.OC, ICG, g.K});

    // The operator sees the f16-rounded values; the reference sees those same
    // values in f32, so the only difference left is the f16 accumulation.
    auto in16_vec = test::f32_to_f16(in_f32);
    auto w16_vec  = test::f32_to_f16(w_f32);
    auto in_ref   = test::f16_to_f32(in16_vec);
    auto w_ref    = test::f16_to_f32(w16_vec);
    const int64_t in_shape[] = {g.N, g.IC, g.L};
    const int64_t w_shape[]  = {g.OC, ICG, g.K};
    TensorView in16(std::span<const int64_t>(in_shape, 3), DataType::f16, in16_vec.data(),
                    TensorLayout::NCHW);
    TensorView w16(std::span<const int64_t>(w_shape, 3), DataType::f16, w16_vec.data(),
                   TensorLayout::NCHW);
    TensorView in32(std::span<const int64_t>(in_shape, 3), DataType::f32, in_ref.data(),
                    TensorLayout::NCHW);
    TensorView w32(std::span<const int64_t>(w_shape, 3), DataType::f32, w_ref.data(),
                   TensorLayout::NCHW);

    Conv1DAttributes attrs;
    attrs.kernel_size = g.K;
    attrs.stride      = g.S;
    attrs.dilation    = g.D;
    attrs.padding     = g.P;
    attrs.groups      = g.G;

    auto op = Conv1D::create(attrs);
    const TensorDesc ds[] = {in16.desc(), w16.desc()};
    auto out_d = op->getOutputTensorDesc(ds);
    NNOPS_EXPECT_TRUE(out_d[0].dtype == DataType::f16);

    std::vector<half> out_buf(static_cast<size_t>(out_d[0].numel()));
    TensorView out = test::make_channel_first(out_d[0], out_buf.data());
    std::vector<float> ref_buf(out_buf.size());
    TensorView ref = test::make_channel_first(out_d[0], ref_buf.data());

    {
        ComputeContext ctx;
        TensorView ref4 = wide_act(ref);
        const TensorView ins[] = {wide_act(in32), wide_w(w32)};
        backend::cpu::reference::conv2d_ref(to_2d(attrs), ref4, ins, ctx, nullptr);
    }
    {
        const TensorView ins[] = {in16, w16};
        NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));
    }

    const auto got = test::f16_to_f32(out_buf);
    for (size_t i = 0; i < got.size(); ++i) {
        NNOPS_EXPECT_NEAR(got[i], ref_buf[i], test::kF16AccumTol);
    }
}

/// Channel-last activation + HWIO filter, compared with the channel-first
/// operator run on the same logical values.
void run_channel_last(const Geom& g)
{
    const int64_t ICG = g.IC / g.G;
    auto [in_vec, in] = test::make_random_tensor({g.N, g.IC, g.L});
    auto [w_vec, w]   = test::make_random_tensor({g.OC, ICG, g.K});

    Conv1DAttributes attrs;
    attrs.kernel_size = g.K;
    attrs.stride      = g.S;
    attrs.dilation    = g.D;
    attrs.padding     = g.P;
    attrs.groups      = g.G;

    auto op = Conv1D::create(attrs);
    const TensorDesc ds[] = {in.desc(), w.desc()};
    auto out_d = op->getOutputTensorDesc(ds);
    const int64_t OL = out_d[0].dims[2];

    // Channel-first arm — the expectation.
    std::vector<float> cf_buf(static_cast<size_t>(out_d[0].numel()));
    TensorView cf_out = test::make_channel_first(out_d[0], cf_buf.data());
    {
        const TensorView ins[] = {in, w};
        NNOPS_EXPECT_TRUE(is_ok(op->compute(cf_out, ins)));
    }

    // Channel-last arm: activation [N, L, C], filter [K, IC/G, OC].
    auto cl_in_vec = test::to_channel_last<float>(in_vec, g.N, g.IC, g.L);
    auto hwio_vec = test::oihw_to_hwio<float>(w_vec, g.OC, ICG, 1, g.K);
    const int64_t cl_shape[] = {g.N, g.IC, g.L};
    const int64_t hw_shape[] = {g.K, ICG, g.OC};
    TensorView cl_in(std::span<const int64_t>(cl_shape, 3), DataType::f32, cl_in_vec.data(),
                     TensorLayout::NHWC);
    TensorView hwio(std::span<const int64_t>(hw_shape, 3), DataType::f32, hwio_vec.data(),
                    TensorLayout::HWIO);

    const TensorDesc cl_ds[] = {cl_in.desc(), hwio.desc()};
    auto cl_out_d = op->getOutputTensorDesc(cl_ds);
    NNOPS_EXPECT_TRUE(cl_out_d[0].layout == TensorLayout::NHWC);
    NNOPS_EXPECT_TRUE(cl_out_d[0].dims[2] == OL);

    std::vector<float> cl_buf(static_cast<size_t>(cl_out_d[0].numel()));
    TensorView cl_out = test::make_channel_first(cl_out_d[0], cl_buf.data());
    {
        const TensorView ins[] = {cl_in, hwio};
        NNOPS_EXPECT_TRUE(is_ok(op->compute(cl_out, ins)));
    }

    const auto back = test::to_channel_first<float>(cl_buf, g.N, g.OC, OL);
    for (size_t i = 0; i < cf_buf.size(); ++i) {
        NNOPS_EXPECT_NEAR(back[i], cf_buf[i], 1e-4f);
    }
}

// ---- quantized route -----------------------------------------------------

QuantParams per_tensor(float scale, int32_t zp) {
    QuantParams qp;
    qp.granularity = QuantGranularity::PerTensor;
    qp.scale = scale;
    qp.zero_point = zp;
    return qp;
}

QuantParams per_channel(const float* scale, const int32_t* zp, int64_t n) {
    QuantParams qp;
    qp.granularity = QuantGranularity::PerChannel;
    qp.scale_data = scale;
    qp.zero_point_data = zp;
    qp.num_scales = n;
    return qp;
}

/// The quantized route is integer; both sides share the dequant expression, so
/// the comparison is bit for bit.
template <class TAct, class TOut>
void run_int8(const Geom& g, const QuantParams& act_qp, const QuantParams& out_qp,
              bool int_out)
{
    const int64_t ICG = g.IC / g.G;
    const int64_t in_n = g.N * g.IC * g.L;
    const int64_t w_n  = g.OC * ICG * g.K;

    std::vector<TAct> in_vec(static_cast<size_t>(in_n));
    test::XorShift128 rng(7);
    const int32_t lo = std::is_same_v<TAct, uint8_t> ? 0 : -128;
    const int32_t hi = std::is_same_v<TAct, uint8_t> ? 255 : 127;
    for (auto& v : in_vec) {
        v = static_cast<TAct>(lo + static_cast<int32_t>(rng.next_u64() % (hi - lo + 1)));
    }
    std::vector<int8_t> w_vec(static_cast<size_t>(w_n));
    for (auto& v : w_vec) {
        v = static_cast<int8_t>(static_cast<int32_t>(rng.next_u64() % 255) - 127);
    }
    std::vector<float> w_scale(static_cast<size_t>(g.OC));
    std::vector<int32_t> w_zp(static_cast<size_t>(g.OC), 0);
    for (int64_t i = 0; i < g.OC; ++i) {
        w_scale[static_cast<size_t>(i)] = 1.0f / static_cast<float>(32 + 8 * (i % 5));
    }
    const QuantParams w_qp = per_channel(w_scale.data(), w_zp.data(), g.OC);

    const int64_t in_shape[] = {g.N, g.IC, g.L};
    const int64_t w_shape[]  = {g.OC, ICG, g.K};
    TensorView in(std::span<const int64_t>(in_shape, 3),
                  std::is_same_v<TAct, uint8_t> ? DataType::u8 : DataType::s8,
                  in_vec.data(), TensorLayout::NCHW, act_qp);
    TensorView w(std::span<const int64_t>(w_shape, 3), DataType::s8, w_vec.data(),
                 TensorLayout::NCHW, w_qp);

    Conv1DAttributes attrs;
    attrs.kernel_size  = g.K;
    attrs.stride       = g.S;
    attrs.dilation     = g.D;
    attrs.padding      = g.P;
    attrs.groups       = g.G;
    attrs.output_dtype = std::is_same_v<TOut, int8_t> ? DataType::s8
                        : std::is_same_v<TOut, uint8_t> ? DataType::u8
                                                        : DataType::f32;

    auto op = Conv1D::create(attrs);
    const TensorDesc ds[] = {in.desc(), w.desc()};
    auto out_d = op->getOutputTensorDesc(ds);
    NNOPS_EXPECT_TRUE(out_d[0].dtype == attrs.output_dtype);

    std::vector<TOut> out_buf(static_cast<size_t>(out_d[0].numel()));
    TensorView out(std::span<const int64_t>(out_d[0].dims.data(), 3), attrs.output_dtype,
                   out_buf.data(), TensorLayout::NCHW, out_qp);
    std::vector<TOut> ref_buf(out_buf.size());
    TensorView ref(std::span<const int64_t>(out_d[0].dims.data(), 3), attrs.output_dtype,
                   ref_buf.data(), TensorLayout::NCHW, out_qp);

    {
        ComputeContext ctx;
        TensorView ref4 = wide_act(ref);
        const TensorView ins[] = {wide_act(in), wide_w(w)};
        backend::cpu::reference::conv2d_int8_ref(to_2d(attrs), ref4, ins, ctx, nullptr);
    }
    {
        const TensorView ins[] = {in, w};
        NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));
    }
    NNOPS_EXPECT_TRUE(std::memcmp(out_buf.data(), ref_buf.data(),
                                  out_buf.size() * sizeof(TOut)) == 0);
    (void)int_out;
}

// ---- the L-chunked parallel path -----------------------------------------

/// A CpuBackend that reports n workers but runs serially: the pool is large
/// enough to trigger the L split while the calls stay deterministic.
struct FakePool {
    explicit FakePool(int n) : nthreads(n) {
        cpu.parallel_for = [](int64_t begin, int64_t end, const ParallelForBody& body) {
            for (int64_t i = begin; i < end; ++i) { body(i); }
        };
        cpu.num_threads = [this]() { return nthreads; };
        cpu.thread_id   = []() { return 0; };
    }
    int nthreads;
    CpuBackend cpu;
};

void run_chunked(const Geom& g, int workers)
{
    const int64_t ICG = g.IC / g.G;
    auto [in_vec, in] = test::make_random_tensor({g.N, g.IC, g.L});
    auto [w_vec, w]   = test::make_random_tensor({g.OC, ICG, g.K});

    Conv1DAttributes attrs;
    attrs.kernel_size = g.K;
    attrs.stride      = g.S;
    attrs.dilation    = g.D;
    attrs.padding     = g.P;
    attrs.groups      = g.G;

    auto op = Conv1D::create(attrs);
    const TensorDesc ds[] = {in.desc(), w.desc()};
    auto out_d = op->getOutputTensorDesc(ds);

    std::vector<float> serial_buf(static_cast<size_t>(out_d[0].numel()));
    std::vector<float> pool_buf(serial_buf.size());
    TensorView serial_out = test::make_channel_first(out_d[0], serial_buf.data());
    TensorView pool_out   = test::make_channel_first(out_d[0], pool_buf.data());

    {
        const TensorView ins[] = {in, w};
        NNOPS_EXPECT_TRUE(is_ok(op->compute(serial_out, ins)));
    }
    {
        FakePool pool(workers);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        NNOPS_EXPECT_TRUE(ctx.cpu.worker_count() == workers);
        const TensorView ins[] = {in, w};
        NNOPS_EXPECT_TRUE(is_ok(op->compute(pool_out, ins, ctx)));
    }
    NNOPS_EXPECT_TRUE(pool_buf == serial_buf);
}

}  // namespace

NNOPS_TEST(conv1d_f32_basic) {
    run_f32({"basic",        2, 3, 17, 5, 1, 3, 1, 1, 0, false});
    run_f32({"basic_bias",   1, 8, 33, 4, 1, 3, 1, 1, 1, true});
    run_f32({"k1_pointwise", 2, 16, 40, 24, 1, 1, 1, 1, 0, true});
    run_f32({"stride2",      1, 6, 31, 9, 1, 4, 2, 1, 1, false});
    run_f32({"dilated",      1, 5, 29, 7, 1, 3, 1, 2, 2, false});
    run_f32({"k7_stride3",   2, 4, 64, 6, 1, 7, 3, 1, 3, true});
}

NNOPS_TEST(conv1d_f32_groups) {
    run_f32({"groups2",  2, 8, 21, 10, 2, 3, 1, 1, 1, false});
    run_f32({"groups4",  1, 12, 25, 8, 4, 5, 2, 1, 2, true});
    run_f32({"depthwise", 2, 6, 19, 6, 6, 3, 1, 1, 1, false});
    run_f32({"depthwise_stride2", 1, 8, 40, 8, 8, 3, 2, 1, 1, true});
}

NNOPS_TEST(conv1d_f16_basic) {
    run_f16({"f16_basic",  2, 4, 20, 6, 1, 3, 1, 1, 1, false});
    run_f16({"f16_stride2", 1, 8, 33, 5, 1, 3, 2, 1, 1, false});
    run_f16({"f16_groups",  1, 8, 18, 8, 2, 3, 1, 1, 1, false});
}

NNOPS_TEST(conv1d_channel_last_f32) {
    run_channel_last({"cl_basic",  2, 4, 23, 6, 1, 3, 1, 1, 1, false});
    run_channel_last({"cl_stride2", 1, 6, 37, 8, 1, 3, 2, 1, 1, false});
    run_channel_last({"cl_groups", 1, 8, 19, 8, 2, 3, 1, 1, 1, false});
    run_channel_last({"cl_depthwise", 1, 6, 27, 6, 6, 3, 1, 1, 1, false});
}

NNOPS_TEST(conv1d_int8_u8_act_f32_out) {
    const QuantParams act_qp = per_tensor(1.0f / 128.0f, 128);
    const QuantParams out_qp = per_tensor(1.0f / 64.0f, 0);
    run_int8<uint8_t, float>({"q_u8", 2, 4, 15, 6, 1, 3, 1, 1, 1, false},
                             act_qp, out_qp, false);
    run_int8<uint8_t, float>({"q_u8_stride2", 1, 8, 32, 4, 1, 3, 2, 1, 1, false},
                             act_qp, out_qp, false);
}

NNOPS_TEST(conv1d_int8_s8_act_s8_out) {
    const QuantParams act_qp = per_tensor(1.0f / 64.0f, 0);
    const QuantParams out_qp = per_tensor(1.0f / 32.0f, 0);
    run_int8<int8_t, int8_t>({"q_s8", 1, 4, 21, 6, 1, 3, 1, 1, 1, false},
                             act_qp, out_qp, true);
    run_int8<int8_t, int8_t>({"q_s8_groups", 1, 8, 17, 8, 2, 3, 1, 1, 1, false},
                             act_qp, out_qp, true);
}

NNOPS_TEST(conv1d_chunk_split_matches_serial) {
    // N = G = 1 and one output row leave the 2D plan a single task, so these
    // shapes take the L split. Each stride/padding/dilation variant exercises a
    // different window origin at the chunk boundary.
    run_chunked({"chunk_s1p1",  1, 4, 2048, 8, 1, 3, 1, 1, 1, false}, 8);
    run_chunked({"chunk_s2p1",  1, 4, 4096, 8, 1, 5, 2, 1, 1, false}, 4);
    run_chunked({"chunk_s1p2d2", 1, 2, 2000, 4, 1, 3, 1, 2, 2, false}, 4);
    run_chunked({"chunk_s3p0",  1, 2, 3000, 4, 1, 4, 3, 1, 0, false}, 8);
}

NNOPS_TEST(conv1d_validate_rejects_bad_rank) {
    // A rank-4 input has no 1D lift, so it must be refused rather than silently
    // misdescribed by shape inference.
    auto op = Conv1D::create(Conv1DAttributes{});
    TensorDesc in;
    in.rank = 4;
    in.layout = TensorLayout::NCHW;
    in.dtype = DataType::f32;
    in.dims = {1, 2, 3, 4};
    TensorDesc w;
    w.rank = 3;
    w.layout = TensorLayout::NCHW;
    w.dtype = DataType::f32;
    w.dims = {3, 2, 3};
    TensorDesc out;
    out.rank = 3;
    out.layout = TensorLayout::NCHW;
    out.dtype = DataType::f32;
    out.dims = {1, 3, 2};
    const TensorDesc ins[] = {in, w};
    const TensorDesc outs[] = {out};
    NNOPS_EXPECT_TRUE(!is_ok(op->validate(ins, outs)));
    NNOPS_EXPECT_TRUE(op->getWorkspaceSize(ins, outs) == 0);
}
