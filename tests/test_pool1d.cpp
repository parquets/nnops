/// @file test_pool1d.cpp
/// @brief Pool1D against a scalar 1D oracle and, where it applies, the 2D
///        pooling reference.
///
/// Pool1D is Pooling on [N, C, 1, L] with a {1, K} window. The f16 and
/// quantized channel-last arms hand pooling_ref a rank-4 view of the same
/// bytes, lifted here rather than by the operator's own helper. The f32 arm
/// cannot: the reference's windowed walk computes `id = od·SD + kd - PD` and so
/// never applies dilation, while a 1D window's dilation stretches the very axis
/// that walk advances along. That arm uses ref1d, written out below. Covered:
/// max/average, stride/dilation/padding, exclude_pad, global, adaptive
/// (including the length-1 fast path), auto_pad, both layouts, f32/f16, the
/// quantized channel-last route, and a pitched input — the lift keeps the
/// caller's row pitch, and pooling reads rows through it.

#include "nnops/ops/pool1d.hpp"
#include "nnops/ops/pooling.hpp"   // the reference's attribute type
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
#include <limits>
#include <type_traits>
#include <vector>

using namespace nnops;
using nnops::backend::cpu::half;

namespace nnops::backend::cpu::reference {
void pooling_ref(const PoolingAttributes& attrs, TensorView& output,
                 std::span<const TensorView> inputs, const ComputeContext& ctx,
                 void* workspace);
}

namespace {

void* vdata(const TensorView& v) {
    return const_cast<void*>(static_cast<const void*>(v.ptr<uint8_t>()));
}

TensorView wide_act(const TensorView& v) {
    const int64_t s[4] = {v.shape(0), v.shape(1), 1, v.shape(2)};
    return TensorView(std::span<const int64_t>(s, 4), v.data_type(), vdata(v),
                      v.pitch(), v.layout(), v.quant_params());
}

PoolingAttributes to_2d(const Pool1DAttributes& a) {
    PoolingAttributes c{};
    c.type         = (a.type == Pool1DType::Max) ? PoolingType::Max : PoolingType::Average;
    c.kernel_shape = {1, 1, a.kernel_shape};
    c.stride       = {1, 1, a.stride};
    c.padding      = {0, 0, a.padding};
    c.dilation     = {1, 1, a.dilation};
    c.exclude_pad  = a.exclude_pad;
    c.global       = a.global;
    c.output_size  = {0, 1, a.output_size};
    return c;
}

/// The 1D pooling semantics written out once, over channel-first data. The 2D
/// reference's windowed walk does not apply dilation (`id = od·SD + kd - PD`),
/// so a dilated case needs an oracle that does — this one.
void ref1d(const Pool1DAttributes& a, const std::vector<float>& in, int64_t N,
           int64_t C, int64_t L, int64_t OL, std::vector<float>& out)
{
    const float ninf = -std::numeric_limits<float>::infinity();
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const float* src = in.data() + (n * C + c) * L;
            float* dst = out.data() + (n * C + c) * OL;
            for (int64_t o = 0; o < OL; ++o) {
                int64_t i0, i1, step;
                if (a.adaptive()) {
                    i0 = (o * L) / OL;
                    i1 = ((o + 1) * L + OL - 1) / OL;
                    step = 1;
                } else if (a.global) {
                    i0 = 0; i1 = L; step = 1;
                } else {
                    i0 = o * a.stride - a.padding;
                    i1 = i0 + a.dilation * (a.kernel_shape - 1) + 1;
                    step = a.dilation;
                }
                float acc = (a.type == Pool1DType::Max) ? ninf : 0.0f;
                int64_t cnt = 0;
                for (int64_t i = i0; i < i1; i += step) {
                    if (i < 0 || i >= L) { continue; }
                    acc = (a.type == Pool1DType::Max) ? std::max(acc, src[i])
                                                      : acc + src[i];
                    ++cnt;
                }
                if (a.type == Pool1DType::Max) {
                    dst[o] = (cnt > 0) ? acc : 0.0f;
                } else {
                    const int64_t div = a.exclude_pad || a.adaptive() || a.global
                        ? cnt : a.kernel_shape;
                    dst[o] = (div > 0) ? acc / static_cast<float>(div) : 0.0f;
                }
            }
        }
    }
}

/// Channel-first f32 against the scalar oracle, plus the output length.
void run_f32(const Pool1DAttributes& attrs, const TensorView& in)
{
    const int64_t N = in.shape(0);
    const int64_t C = in.shape(1);
    const int64_t L = in.shape(2);
    int64_t exp_ol;
    if (attrs.global) {
        exp_ol = 1;
    } else if (attrs.adaptive()) {
        exp_ol = attrs.output_size;
    } else {
        exp_ol = (L + 2 * attrs.padding - attrs.dilation * (attrs.kernel_shape - 1) - 1)
                     / attrs.stride + 1;
    }

    auto op = Pool1D::create(attrs);
    const TensorDesc ds[] = {in.desc()};
    auto out_d = op->getOutputTensorDesc(ds);
    NNOPS_EXPECT_TRUE(out_d[0].rank == 3);
    NNOPS_EXPECT_TRUE(out_d[0].dims[0] == N && out_d[0].dims[1] == C);
    NNOPS_EXPECT_TRUE(out_d[0].dims[2] == exp_ol);

    std::vector<float> out_buf(static_cast<size_t>(out_d[0].numel()));
    TensorView out = test::make_channel_first(out_d[0], out_buf.data());
    std::vector<float> ref_buf(out_buf.size());
    TensorView ref = test::make_channel_first(out_d[0], ref_buf.data());

    {
        // The operator reads the rows by their pitch; so must the expectation,
        // which is handed the same values sorted by (n, c).
        std::vector<float> flat(static_cast<size_t>(N * C * L));
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                const float* row = in.ptr<float>() + (n * C + c) * in.row_stride_elems();
                std::copy(row, row + L, flat.data() + (n * C + c) * L);
            }
        }
        ref1d(attrs, flat, N, C, L, exp_ol, ref_buf);
    }
    {
        const TensorView ins[] = {in};
        NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));
    }
    NNOPS_EXPECT_TRUE(test::allclose(out, ref, 1e-4f, 1e-4f));
}

void run_f32_case(const Pool1DAttributes& attrs, std::initializer_list<int64_t> shape,
                  uint64_t seed = 3)
{
    auto [vec, in] = test::make_random_tensor(shape, -4.0f, 4.0f, seed);
    run_f32(attrs, in);
}

void run_f16(const Pool1DAttributes& attrs, std::initializer_list<int64_t> shape)
{
    auto [in_f32, in_view] = test::make_random_tensor(shape, -2.0f, 2.0f, 5);
    auto in16_vec = test::f32_to_f16(in_f32);
    auto in_ref   = test::f16_to_f32(in16_vec);
    const int64_t dims[] = {*shape.begin(), *(shape.begin() + 1), *(shape.begin() + 2)};
    TensorView in16(std::span<const int64_t>(dims, 3), DataType::f16, in16_vec.data(),
                    TensorLayout::NCHW);
    TensorView in32(std::span<const int64_t>(dims, 3), DataType::f32, in_ref.data(),
                    TensorLayout::NCHW);

    auto op = Pool1D::create(attrs);
    const TensorDesc ds[] = {in16.desc()};
    auto out_d = op->getOutputTensorDesc(ds);
    NNOPS_EXPECT_TRUE(out_d[0].dtype == DataType::f16);

    std::vector<half> out_buf(static_cast<size_t>(out_d[0].numel()));
    TensorView out = test::make_channel_first(out_d[0], out_buf.data());
    std::vector<float> ref_buf(out_buf.size());
    TensorView ref = test::make_channel_first(out_d[0], ref_buf.data());

    {
        ComputeContext ctx;
        TensorView ref4 = wide_act(ref);
        const TensorView ins[] = {wide_act(in32)};
        backend::cpu::reference::pooling_ref(to_2d(attrs), ref4, ins, ctx, nullptr);
    }
    {
        const TensorView ins[] = {in16};
        NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));
    }
    const auto got = test::f16_to_f32(out_buf);
    for (size_t i = 0; i < got.size(); ++i) {
        NNOPS_EXPECT_NEAR(got[i], ref_buf[i], 5e-3f);
    }
}

/// Channel-last activation against the channel-first operator on the same
/// logical values.
void run_channel_last(const Pool1DAttributes& attrs, std::initializer_list<int64_t> shape)
{
    auto [in_vec, in] = test::make_random_tensor(shape, -4.0f, 4.0f, 9);
    const int64_t N = *shape.begin();
    const int64_t C = *(shape.begin() + 1);
    const int64_t L = *(shape.begin() + 2);

    auto op = Pool1D::create(attrs);
    const TensorDesc ds[] = {in.desc()};
    auto out_d = op->getOutputTensorDesc(ds);
    const int64_t OL = out_d[0].dims[2];

    std::vector<float> cf_buf(static_cast<size_t>(out_d[0].numel()));
    TensorView cf_out = test::make_channel_first(out_d[0], cf_buf.data());
    {
        const TensorView ins[] = {in};
        NNOPS_EXPECT_TRUE(is_ok(op->compute(cf_out, ins)));
    }

    auto cl_vec = test::to_channel_last<float>(in_vec, N, C, L);
    const int64_t cl_shape[] = {N, C, L};
    TensorView cl_in(std::span<const int64_t>(cl_shape, 3), DataType::f32, cl_vec.data(),
                     TensorLayout::NHWC);
    const TensorDesc cl_ds[] = {cl_in.desc()};
    auto cl_out_d = op->getOutputTensorDesc(cl_ds);
    NNOPS_EXPECT_TRUE(cl_out_d[0].layout == TensorLayout::NHWC);

    std::vector<float> cl_buf(static_cast<size_t>(cl_out_d[0].numel()));
    TensorView cl_out = test::make_channel_first(cl_out_d[0], cl_buf.data());
    {
        const TensorView ins[] = {cl_in};
        NNOPS_EXPECT_TRUE(is_ok(op->compute(cl_out, ins)));
    }

    const auto back = test::to_channel_first<float>(cl_buf, N, C, OL);
    for (size_t i = 0; i < cf_buf.size(); ++i) {
        NNOPS_EXPECT_NEAR(back[i], cf_buf[i], 1e-5f);
    }
}

QuantParams per_tensor(float scale, int32_t zp) {
    QuantParams qp;
    qp.granularity = QuantGranularity::PerTensor;
    qp.scale = scale;
    qp.zero_point = zp;
    return qp;
}

/// Quantized pooling is channel-last only; the reference is channel-first, so
/// the two arms hold the same values in different orders and are compared
/// dequantized, within one output step.
template <class T>
void run_int8(const Pool1DAttributes& attrs, std::initializer_list<int64_t> shape,
              const QuantParams& qp)
{
    const int64_t N = *shape.begin();
    const int64_t C = *(shape.begin() + 1);
    const int64_t L = *(shape.begin() + 2);
    constexpr DataType dt = std::is_same_v<T, int8_t> ? DataType::s8 : DataType::u8;

    std::vector<T> cf_vec(static_cast<size_t>(N * C * L));
    test::XorShift128 rng(13);
    for (auto& v : cf_vec) {
        int32_t x = static_cast<int32_t>(rng.next_u64() % 9) - 4;
        if constexpr (std::is_same_v<T, uint8_t>) { x += 4; }
        v = static_cast<T>(x);
    }
    const int64_t cf_shape[] = {N, C, L};
    TensorView cf_in(std::span<const int64_t>(cf_shape, 3), dt, cf_vec.data(),
                     TensorLayout::NCHW, qp);

    auto op = Pool1D::create(attrs);
    const TensorDesc cf_ds[] = {cf_in.desc()};
    auto out_d = op->getOutputTensorDesc(cf_ds);
    const int64_t OL = out_d[0].dims[2];

    std::vector<T> ref_buf(static_cast<size_t>(N * C * OL));
    TensorView ref(std::span<const int64_t>(out_d[0].dims.data(), 3), dt, ref_buf.data(),
                   TensorLayout::NCHW, qp);
    {
        ComputeContext ctx;
        TensorView ref4 = wide_act(ref);
        const TensorView ins[] = {wide_act(cf_in)};
        backend::cpu::reference::pooling_ref(to_2d(attrs), ref4, ins, ctx, nullptr);
    }

    auto cl_vec = test::to_channel_last<T>(cf_vec, N, C, L);
    TensorView cl_in(std::span<const int64_t>(cf_shape, 3), dt, cl_vec.data(),
                     TensorLayout::NHWC, qp);
    const TensorDesc cl_ds[] = {cl_in.desc()};
    auto cl_out_d = op->getOutputTensorDesc(cl_ds);
    NNOPS_EXPECT_TRUE(cl_out_d[0].layout == TensorLayout::NHWC);

    std::vector<T> out_buf(static_cast<size_t>(cl_out_d[0].numel()));
    TensorView out(std::span<const int64_t>(cl_out_d[0].dims.data(), 3), dt,
                   out_buf.data(), TensorLayout::NHWC, qp);
    {
        const TensorView ins[] = {cl_in};
        NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));
    }

    const auto back = test::to_channel_first<T>(out_buf, N, C, OL);
    for (size_t i = 0; i < ref_buf.size(); ++i) {
        NNOPS_EXPECT_NEAR(static_cast<float>(back[i]), static_cast<float>(ref_buf[i]), 1.0f);
    }
}

}  // namespace

NNOPS_TEST(pool1d_max_f32) {
    Pool1DAttributes a;
    a.type = Pool1DType::Max;

    a.kernel_shape = 2; a.stride = 2; a.padding = 0; a.dilation = 1;
    run_f32_case(a, {2, 3, 16});
    a.kernel_shape = 3; a.stride = 1; a.padding = 1;
    run_f32_case(a, {1, 4, 15});
    a.kernel_shape = 3; a.stride = 2; a.padding = 1;
    run_f32_case(a, {2, 2, 21});
    a.kernel_shape = 5; a.stride = 1; a.padding = 2; a.dilation = 2;
    run_f32_case(a, {1, 3, 19});
}

NNOPS_TEST(pool1d_average_f32) {
    Pool1DAttributes a;
    a.type = Pool1DType::Average;

    a.kernel_shape = 2; a.stride = 2; a.padding = 0; a.dilation = 1; a.exclude_pad = false;
    run_f32_case(a, {1, 3, 14});
    // Padding included in the divisor: the padded taps count as zeros.
    a.kernel_shape = 3; a.stride = 2; a.padding = 1;
    run_f32_case(a, {1, 2, 11});
    // exclude_pad: the divisor shrinks at the edges.
    a.exclude_pad = true;
    run_f32_case(a, {1, 2, 11});
    run_f32_case(a, {2, 4, 17}, 7);
    a.exclude_pad = false;
    a.kernel_shape = 4; a.stride = 3; a.padding = 2; a.dilation = 2;
    run_f32_case(a, {1, 2, 23});
}

NNOPS_TEST(pool1d_global) {
    Pool1DAttributes a;
    a.global = true;
    a.type = Pool1DType::Max;
    run_f32_case(a, {2, 3, 40});
    a.type = Pool1DType::Average;
    run_f32_case(a, {1, 5, 33});

    auto op = Pool1D::create(a);
    const int64_t shape[] = {1, 5, 33};
    TensorDesc d;
    d.rank = 3;
    d.dtype = DataType::f32;
    d.layout = TensorLayout::NCHW;
    d.dims = std::span<const int64_t>(shape, 3);
    const TensorDesc ds[] = {d};
    auto out_d = op->getOutputTensorDesc(ds);
    NNOPS_EXPECT_TRUE(out_d[0].dims[2] == 1);
}

NNOPS_TEST(pool1d_adaptive) {
    Pool1DAttributes a;
    a.type = Pool1DType::Average;
    a.output_size = 3;
    run_f32_case(a, {1, 3, 20});
    a.output_size = 7;
    run_f32_case(a, {2, 2, 23});
    a.type = Pool1DType::Max;
    a.output_size = 4;
    run_f32_case(a, {1, 4, 19});
    // Adaptive to length 1 is the global case; it must give the same number.
    a.output_size = 1;
    run_f32_case(a, {1, 4, 19});
}

NNOPS_TEST(pool1d_auto_pad) {
    // k3 s1 SAME_UPPER is padding 1 on both edges.
    Pool1DAttributes same;
    same.type = Pool1DType::Max;
    same.kernel_shape = 3;
    same.auto_pad = Pool1DAttributes::AutoPad::SAME_UPPER;
    Pool1DAttributes explicit_pad = same;
    explicit_pad.auto_pad = Pool1DAttributes::AutoPad::NOTSET;
    explicit_pad.padding = 1;

    auto [vec, in] = test::make_random_tensor({1, 3, 17}, -4.0f, 4.0f, 21);
    auto op_same = Pool1D::create(same);
    auto op_pad  = Pool1D::create(explicit_pad);
    const TensorDesc ds[] = {in.desc()};
    auto d1 = op_same->getOutputTensorDesc(ds);
    auto d2 = op_pad->getOutputTensorDesc(ds);
    NNOPS_EXPECT_TRUE(d1[0].dims[2] == 17 && d2[0].dims[2] == 17);

    std::vector<float> b1(static_cast<size_t>(d1[0].numel())), b2(b1.size());
    TensorView o1 = test::make_channel_first(d1[0], b1.data());
    TensorView o2 = test::make_channel_first(d2[0], b2.data());
    const TensorView ins[] = {in};
    NNOPS_EXPECT_TRUE(is_ok(op_same->compute(o1, ins)));
    NNOPS_EXPECT_TRUE(is_ok(op_pad->compute(o2, ins)));
    NNOPS_EXPECT_TRUE(b1 == b2);

    // VALID resolves to no padding at all.
    Pool1DAttributes valid = same;
    valid.auto_pad = Pool1DAttributes::AutoPad::VALID;
    auto op_valid = Pool1D::create(valid);
    auto d3 = op_valid->getOutputTensorDesc(ds);
    NNOPS_EXPECT_TRUE(d3[0].dims[2] == 15);

    // Stride 2 with SAME_UPPER: ceil(17/2) = 9 outputs.
    Pool1DAttributes same_s2 = same;
    same_s2.stride = 2;
    auto op_s2 = Pool1D::create(same_s2);
    auto d4 = op_s2->getOutputTensorDesc(ds);
    NNOPS_EXPECT_TRUE(d4[0].dims[2] == 9);
}

NNOPS_TEST(pool1d_f16) {
    Pool1DAttributes a;
    a.type = Pool1DType::Max;
    a.kernel_shape = 3;
    a.padding = 1;
    run_f16(a, {1, 4, 21});
    a.type = Pool1DType::Average;
    a.exclude_pad = true;
    a.stride = 2;
    run_f16(a, {2, 3, 18});
}

NNOPS_TEST(pool1d_channel_last_f32) {
    Pool1DAttributes a;
    a.type = Pool1DType::Max;
    a.kernel_shape = 3;
    a.stride = 2;
    a.padding = 1;
    run_channel_last(a, {1, 5, 23});
    a.type = Pool1DType::Average;
    a.exclude_pad = true;
    run_channel_last(a, {2, 3, 17});
    a.global = true;
    run_channel_last(a, {1, 4, 31});
}

NNOPS_TEST(pool1d_int8_channel_last) {
    const QuantParams qp = per_tensor(1.0f / 32.0f, 0);
    Pool1DAttributes a;
    a.type = Pool1DType::Max;
    a.kernel_shape = 2;
    a.stride = 2;
    run_int8<int8_t>(a, {1, 8, 24}, qp);
    a.kernel_shape = 3;
    a.stride = 1;
    a.padding = 1;
    run_int8<int8_t>(a, {2, 3, 13}, qp);
    a.type = Pool1DType::Average;
    a.exclude_pad = true;
    a.stride = 2;
    run_int8<int8_t>(a, {1, 4, 15}, qp);

    const QuantParams u8qp = per_tensor(1.0f / 32.0f, 128);
    a.type = Pool1DType::Max;
    a.exclude_pad = false;
    run_int8<uint8_t>(a, {1, 4, 15}, u8qp);
}

NNOPS_TEST(pool1d_pitched_input) {
    // A padded row pitch must be read through, not assumed dense.
    auto [dense_vec, dense] = test::make_random_tensor({1, 4, 13}, -4.0f, 4.0f, 31);
    Pool1DAttributes a;
    a.type = Pool1DType::Max;
    a.kernel_shape = 3;
    a.padding = 1;

    auto op = Pool1D::create(a);
    const TensorDesc ds[] = {dense.desc()};
    auto out_d = op->getOutputTensorDesc(ds);

    std::vector<float> ref_buf(static_cast<size_t>(out_d[0].numel()));
    TensorView ref = test::make_channel_first(out_d[0], ref_buf.data());
    {
        const TensorView ins[] = {dense};
        NNOPS_EXPECT_TRUE(is_ok(op->compute(ref, ins)));
    }

    // Same values, each row followed by its pitch padding.
    TensorDesc pd = dense.desc();
    const int64_t pitch_bytes = pd.row_pitch(32);
    NNOPS_EXPECT_TRUE(pitch_bytes > static_cast<int64_t>(13 * sizeof(float)));
    std::vector<float> pitched_buf(static_cast<size_t>(4 * pitch_bytes / sizeof(float)), 0.0f);
    for (int64_t c = 0; c < 4; ++c) {
        const float* src = dense_vec.data() + c * 13;
        std::copy(src, src + 13, pitched_buf.data() + c * (pitch_bytes / sizeof(float)));
    }
    const int64_t shape[] = {1, 4, 13};
    TensorView pitched(std::span<const int64_t>(shape, 3), DataType::f32, pitched_buf.data(),
                       pitch_bytes, TensorLayout::NCHW);

    std::vector<float> out_buf(ref_buf.size());
    TensorView out = test::make_channel_first(out_d[0], out_buf.data());
    {
        const TensorView ins[] = {pitched};
        NNOPS_EXPECT_TRUE(is_ok(op->compute(out, ins)));
    }
    NNOPS_EXPECT_TRUE(out_buf == ref_buf);
}

NNOPS_TEST(pool1d_validate_rejects_bad_rank) {
    auto op = Pool1D::create(Pool1DAttributes{});
    TensorDesc in;
    in.rank = 4;
    in.dtype = DataType::f32;
    in.layout = TensorLayout::NCHW;
    in.dims = {1, 2, 3, 4};
    TensorDesc out;
    out.rank = 3;
    out.dtype = DataType::f32;
    out.layout = TensorLayout::NCHW;
    out.dims = {1, 2, 1};
    const TensorDesc ins[] = {in};
    const TensorDesc outs[] = {out};
    NNOPS_EXPECT_TRUE(!is_ok(op->validate(ins, outs)));

    // Quantized pooling is channel-last only.
    in.rank = 3;
    in.dims = {1, 2, 8};
    in.dtype = DataType::s8;
    const TensorDesc ins_q[] = {in};
    NNOPS_EXPECT_TRUE(!is_ok(op->validate(ins_q, outs)));
}
