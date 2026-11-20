/// @file test_pool1d.cpp
/// @brief Pool1D against a scalar 1D oracle and, where it applies, the 2D
///        pooling reference.
///
/// Pool1D is Pooling on the channel-last [N, 1, L, C] with a {1, K} window. The
/// f32 arm uses ref1d, written out below: the 2D
/// reference's windowed walk computes `id = od·SD + kd - PD` and so never
/// applies dilation, while a 1D window's dilation stretches the very axis that
/// walk advances along. The quantized arm hands pooling_ref a rank-4 view of
/// the same bytes, lifted here rather than by the operator's own helper.
/// Covered: the f32 route against the scalar oracle, and the quantized route.

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
#include <cstdio>
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

/// The 1D pooling semantics written out once, over dense-ordered data. It
/// carries its own window walk rather than reusing the 2D reference: the rows
/// are handed to it flat, and a 1D window has no H axis to fold into.
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

/// The NHWC route against the scalar 1D oracle on the same values.
void run_case(const Pool1DAttributes& attrs, std::initializer_list<int64_t> shape)
{
    auto [in_vec, in_view] = test::make_random_tensor(shape, -4.0f, 4.0f, 9);
    (void)in_view;
    const int64_t N = *shape.begin();
    const int64_t C = *(shape.begin() + 1);
    const int64_t L = *(shape.begin() + 2);

    auto cl_vec = test::to_channel_last<float>(in_vec, N, C, L);
    const int64_t cl_shape[] = {N, L, C};   // channel-last
    TensorView cl_in(std::span<const int64_t>(cl_shape, 3), DataType::f32, cl_vec.data(),
                     TensorLayout::NHWC);

    auto op = Pool1D::create(attrs);
    const TensorDesc cl_ds[] = {cl_in.desc()};
    auto cl_out_d = op->getOutputTensorDesc(cl_ds);
    NNOPS_EXPECT_TRUE(cl_out_d[0].layout == TensorLayout::NHWC);
    const int64_t OL = cl_out_d[0].dims[1];   // [N, OL, C]

    // Ground truth: the scalar oracle over the dense-order values.
    std::vector<float> ref_buf(static_cast<size_t>(N * C * OL));
    ref1d(attrs, in_vec, N, C, L, OL, ref_buf);

    std::vector<float> cl_buf(static_cast<size_t>(cl_out_d[0].numel()));
    TensorView cl_out = test::make_channel_first(cl_out_d[0], cl_buf.data());
    {
        const TensorView ins[] = {cl_in};
        NNOPS_EXPECT_TRUE(is_ok(op->compute(cl_out, ins)));
    }

    const auto back = test::to_channel_first<float>(cl_buf, N, C, OL);
    for (size_t i = 0; i < ref_buf.size(); ++i) {
    }
    for (size_t i = 0; i < ref_buf.size(); ++i) {
        NNOPS_EXPECT_NEAR(back[i], ref_buf[i], 1e-5f);
    }
}

QuantParams per_tensor(float scale, int32_t zp) {
    QuantParams qp;
    qp.granularity = QuantGranularity::PerTensor;
    qp.scale = scale;
    qp.zero_point = zp;
    return qp;
}

/// The reference reads dense order, so the two arms hold the same values in
/// different orders and are compared
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
    const int64_t cl_shape8[] = {N, L, C};   // channel-last
    TensorView cf_in(std::span<const int64_t>(cf_shape, 3), dt, cf_vec.data(),
                     TensorLayout::Dense, qp);

    auto op = Pool1D::create(attrs);
    // Shape inference speaks channel-last whatever the storage.
    TensorDesc cf_cl = cf_in.desc();
    cf_cl.dims = std::span<const int64_t>(cl_shape8, 3);
    cf_cl.layout = TensorLayout::NHWC;
    const TensorDesc cf_ds[] = {cf_cl};
    auto out_d = op->getOutputTensorDesc(cf_ds);
    const int64_t OL = out_d[0].dims[1];   // [N, OL, C]
    // The reference reads dense [N, C, OL].
    const int64_t ref_dims[] = {N, C, OL};

    std::vector<T> ref_buf(static_cast<size_t>(N * C * OL));
    TensorView ref(std::span<const int64_t>(ref_dims, 3), dt, ref_buf.data(),
                   TensorLayout::Dense, qp);
    {
        ComputeContext ctx;
        TensorView ref4 = wide_act(ref);
        const TensorView ins[] = {wide_act(cf_in)};
        backend::cpu::reference::pooling_ref(to_2d(attrs), ref4, ins, ctx, nullptr);
    }

    auto cl_vec = test::to_channel_last<T>(cf_vec, N, C, L);
    TensorView cl_in(std::span<const int64_t>(cl_shape8, 3), dt, cl_vec.data(),
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

NNOPS_TEST(pool1d_f32) {
    Pool1DAttributes a;
    a.type = Pool1DType::Max;
    a.kernel_shape = 3;
    a.stride = 2;
    a.padding = 1;
    run_case(a, {1, 5, 23});
    a.type = Pool1DType::Average;
    a.exclude_pad = true;
    run_case(a, {2, 3, 17});
    a.global = true;
    run_case(a, {1, 4, 31});
}

NNOPS_TEST(pool1d_int8) {
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

NNOPS_TEST(pool1d_validate_rejects_bad_rank) {
    auto op = Pool1D::create(Pool1DAttributes{});
    TensorDesc in;
    in.rank = 4;
    in.dtype = DataType::f32;
    in.layout = TensorLayout::NHWC;
    in.dims = {1, 2, 3, 4};
    TensorDesc out;
    out.rank = 3;
    out.dtype = DataType::f32;
    out.layout = TensorLayout::NHWC;
    out.dims = {1, 2, 1};
    const TensorDesc ins[] = {in};
    const TensorDesc outs[] = {out};
    NNOPS_EXPECT_TRUE(!is_ok(op->validate(ins, outs)));

    // Channel-first has no route.
    in.rank = 3;
    in.dims = {1, 2, 8};
    in.dtype = DataType::s8;
    in.layout = TensorLayout::Dense;
    const TensorDesc ins_q[] = {in};
    NNOPS_EXPECT_TRUE(!is_ok(op->validate(ins_q, outs)));
}
