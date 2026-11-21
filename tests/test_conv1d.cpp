/// @file test_conv1d.cpp
/// @brief Conv1D against the 2D reference computed on the equivalent lift.
///
/// Conv1D takes an NHWC activation [N, L, C] and an HWIO filter
/// [K, IC/G, OC]. The oracle here is conv2d_ref handed a rank-4 dense
/// view of the same logical values, with the lift written out locally rather
/// than taken from conv_util.h — so a lift the operator got wrong cannot
/// hide behind the same mistake in the expectation.
///
/// Covered: stride/dilation/padding and groups (including the depthwise
/// hand-off to the depthwise kernels).

#include "nnops/ops/conv1d.hpp"
#include "nnops/ops/conv2d.hpp"   // the reference's attribute type
#include "nnops/core/compute_context.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/channel_last.hpp"

#include <span>
#include <vector>

using namespace nnops;

namespace nnops::backend::cpu::reference {
void conv2d_ref(const Conv2DAttributes& attrs, TensorView& output,
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

/// NHWC activation + HWIO filter, compared with the 2D reference on the same
/// logical values lifted into dense order.
void run_case(const Geom& g)
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

    // NHWC activation [N, L, C], filter [K, IC/G, OC].
    auto cl_in_vec = test::to_channel_last<float>(in_vec, g.N, g.IC, g.L);
    auto hwio_vec = test::oihw_to_hwio<float>(w_vec, g.OC, ICG, 1, g.K);
    const int64_t cl_shape[] = {g.N, g.L, g.IC};   // channel-last
    const int64_t hw_shape[] = {g.K, ICG, g.OC};
    TensorView cl_in(std::span<const int64_t>(cl_shape, 3), DataType::f32, cl_in_vec.data(),
                     TensorLayout::NHWC);
    TensorView hwio(std::span<const int64_t>(hw_shape, 3), DataType::f32, hwio_vec.data(),
                    TensorLayout::HWIO);

    const TensorDesc cl_ds[] = {cl_in.desc(), hwio.desc()};
    auto cl_out_d = op->getOutputTensorDesc(cl_ds);
    NNOPS_EXPECT_TRUE(cl_out_d[0].layout == TensorLayout::NHWC);
    const int64_t OL = cl_out_d[0].dims[1];   // [N, OL, OC]

    // The expectation: the 2D reference on the same logical values, in the
    // dense order it reads.
    std::vector<float> ref_buf(static_cast<size_t>(cl_out_d[0].numel()));
    {
        // Dense [N, OC, OL]: the order the 2D reference reads.
        const int64_t ref_dims[] = {g.N, g.OC, OL};
        TensorView ref3(std::span<const int64_t>(ref_dims, 3),
                        DataType::f32, ref_buf.data(), TensorLayout::Dense);
        ComputeContext ctx;
        TensorView ref4 = wide_act(ref3);
        const TensorView ins[] = {wide_act(in), wide_w(w)};
        backend::cpu::reference::conv2d_ref(to_2d(attrs), ref4, ins, ctx, nullptr);
    }

    std::vector<float> cl_buf(static_cast<size_t>(cl_out_d[0].numel()));
    TensorView cl_out = test::make_channel_first(cl_out_d[0], cl_buf.data());
    {
        const TensorView ins[] = {cl_in, hwio};
        NNOPS_EXPECT_TRUE(is_ok(op->compute(cl_out, ins)));
    }

    const auto back = test::to_channel_first<float>(cl_buf, g.N, g.OC, OL);
    for (size_t i = 0; i < ref_buf.size(); ++i) {
        NNOPS_EXPECT_NEAR(back[i], ref_buf[i], 1e-4f);
    }
}

}  // namespace

NNOPS_TEST(conv1d_nhwc_f32) {
    run_case({"basic",  2, 4, 23, 6, 1, 3, 1, 1, 1, false});
    run_case({"stride2", 1, 6, 37, 8, 1, 3, 2, 1, 1, false});
    run_case({"groups", 1, 8, 19, 8, 2, 3, 1, 1, 1, false});
    run_case({"depthwise", 1, 6, 27, 6, 6, 3, 1, 1, 1, false});
}
