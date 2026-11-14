/// @file test_cuda_bf16.cpp
/// @brief CUDA bf16 against the CPU backend, value for value.
///
/// bf16's availability is per backend, and off AArch64 only the CUDA backend has
/// it, so the CPU cannot run a bf16 tensor as the oracle. Every case therefore
/// goes through run_op_bf16: the f32 inputs are rounded to bf16 on the host, the
/// CPU operator runs over those exact values in f32, the CUDA operator runs over
/// the same values held as bf16, and both outputs are compared as bf16. The
/// device rounds its f32 accumulator at the store and the oracle is rounded the
/// same way, so what is left is accumulation order — not input or output
/// precision.
///
/// The cases cover each kernel family the backend dispatches: the elementwise
/// pack8 path, the CuTe GEMM with its bf16 MMA atom (and the split-K reduce,
/// whose partial planes are f32), the conv / transpose-conv / depthwise routes in
/// both layouts, and the movement, index and k-selection operators. Operands of
/// equal shape only: the CUDA elementwise kernel walks both in lockstep, so a
/// broadcast second operand is unsupported for every dtype, not just bf16.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/activation.hpp"
#include "nnops/ops/argminmax.hpp"
#include "nnops/ops/concat.hpp"
#include "nnops/ops/conv2d.hpp"
#include "nnops/ops/cumsum.hpp"
#include "nnops/ops/depthwise_conv.hpp"
#include "nnops/ops/depthwise_transpose_conv.hpp"
#include "nnops/ops/eltwise.hpp"
#include "nnops/ops/gather.hpp"
#include "nnops/ops/linear.hpp"
#include "nnops/ops/matmul.hpp"
#include "nnops/ops/norm.hpp"
#include "nnops/ops/pad.hpp"
#include "nnops/ops/permute.hpp"
#include "nnops/ops/pooling.hpp"
#include "nnops/ops/reduce.hpp"
#include "nnops/ops/resize.hpp"
#include "nnops/ops/scatter.hpp"
#include "nnops/ops/slice.hpp"
#include "nnops/ops/softmax.hpp"
#include "nnops/ops/split.hpp"
#include "nnops/ops/tile.hpp"
#include "nnops/ops/topk.hpp"
#include "nnops/ops/transpose_conv2d.hpp"
#include "nnops/ops/unary.hpp"

#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"

#include <span>
#include <utility>
#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

/// bf16 keeps 8 mantissa bits, so one ulp is 0.4% of a value. Both sides round
/// the same f32 arithmetic away from that, so the budget is one ulp of relative
/// slack plus a floor for the sums that land near zero.
constexpr float kTolAbs = 2e-3f;
constexpr float kTolRel = 2e-2f;

HostTensor rnd(std::vector<int64_t> shape, bool padded = false, uint64_t seed = 17,
               float lo = -0.5f, float hi = 0.5f,
               TensorLayout layout = TensorLayout::NCHW)
{
    return make_host_random(shape, DataType::f32, padded, seed, lo, hi, layout);
}

/// An index tensor, always f32-free so run_op_bf16 passes it through untouched.
HostTensor idxs(std::vector<int64_t> shape, std::vector<int64_t> values)
{
    HostTensor t;
    const TensorView probe(shape, DataType::s64, nullptr);
    t.bytes.assign(probe.storage_bytes(), std::byte{0});
    auto* p = reinterpret_cast<int64_t*>(t.bytes.data());
    const size_t n = std::min(values.size(), static_cast<size_t>(probe.numel()));
    for (size_t i = 0; i < n; ++i) { p[i] = values[i]; }
    t.view = TensorView(shape, DataType::s64, t.bytes.data());
    return t;
}

/// Values that survive the bf16 round trip exactly and are distinct along
/// `axis`, so TopK and ArgMinMax must agree on which element they picked.
HostTensor distinct(std::vector<int64_t> shape, int64_t axis)
{
    HostTensor t = make_host_zeroed(shape, DataType::f32, false);
    TensorView& v = t.view;
    const int64_t rank = v.rank();
    int64_t coords[TensorView::kMaxRank] = {};
    for (int64_t e = 0; e < v.numel(); ++e) {
        int64_t off = 0;
        for (int64_t d = 0; d < rank; ++d) { off += coords[d] * v.stride_elems(d); }
        v.ptr<float>()[off] = static_cast<float>(coords[axis]) * 0.5f;
        for (int64_t d = rank - 1; d >= 0; --d) {
            if (++coords[d] < v.shape(d)) { break; }
            coords[d] = 0;
        }
    }
    return t;
}

PoolingAttributes window(PoolingType type, int64_t k, int64_t stride = 1,
                         int64_t pad = 0)
{
    PoolingAttributes a;
    a.type = type;
    a.kernel_shape = {k, k, k};
    a.stride = {stride, stride, stride};
    a.padding = {pad, pad, pad};
    a.dilation = {1, 1, 1};
    return a;
}

}  // namespace

NNOPS_TEST(cuda_bf16_unary) {
    // The pack8 path (two 16 B vectors per iteration) needs a 2D row long
    // enough for it; the rank-1 case takes the flat grid-stride script.
    for (UnaryType type : {UnaryType::Exp, UnaryType::Tanh, UnaryType::Abs,
                           UnaryType::Neg, UnaryType::Sign, UnaryType::Recip,
                           UnaryType::Round, UnaryType::Ceil, UnaryType::Floor}) {
        HostTensor a = rnd({2, 3, 7, 11}, true, 17);
        const HostTensor in[] = {a};
        run_op_bf16([type](Backend be) {
                        return Unary::create(UnaryAttributes{type, false}, be);
                    },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_unary");
    }
    {
        HostTensor a = rnd({257}, false, 19, 0.05f, 2.0f);
        const HostTensor in[] = {a};
        run_op_bf16([](Backend be) {
                        return Unary::create(UnaryAttributes{UnaryType::Sqrt, false}, be);
                    },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_unary_sqrt");
    }
}

NNOPS_TEST(cuda_bf16_eltwise) {
    for (EltwiseType type : {EltwiseType::Add, EltwiseType::Sub, EltwiseType::Mul,
                             EltwiseType::Min, EltwiseType::Max}) {
        HostTensor a = rnd({2, 3, 5, 13}, true, 17);
        HostTensor b = rnd({2, 3, 5, 13}, true, 29);
        const HostTensor in[] = {a, b};
        run_op_bf16([type](Backend be) {
                        return Eltwise::create(EltwiseAttributes{type, false}, be);
                    },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_eltwise");
    }
    {
        HostTensor a = rnd({4, 6, 9}, true, 17);
        HostTensor b = rnd({4, 6, 9}, true, 29, 0.25f, 1.0f);
        const HostTensor in[] = {a, b};
        run_op_bf16([](Backend be) {
                        return Eltwise::create(
                            EltwiseAttributes{EltwiseType::Div, false}, be);
                    },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_eltwise_div");
    }
}

NNOPS_TEST(cuda_bf16_activation) {
    for (ActivationType type : {ActivationType::Relu, ActivationType::Sigmoid,
                                ActivationType::Tanh, ActivationType::Gelu,
                                ActivationType::Silu, ActivationType::Elu,
                                ActivationType::HardSwish,
                                ActivationType::HardSigmoid,
                                ActivationType::Softplus, ActivationType::Mish,
                                ActivationType::Selu, ActivationType::GeluExact,
                                ActivationType::QuickGelu}) {
        ActivationAttributes attrs;
        attrs.type = type;
        attrs.alpha = type == ActivationType::Selu ? 1.6732632f
                      : type == ActivationType::HardSigmoid ? 0.2f
                      : type == ActivationType::QuickGelu ? 1.702f
                                                          : 0.01f;
        attrs.beta = type == ActivationType::Selu ? 1.0507010f
                     : type == ActivationType::HardSigmoid ? 0.5f
                                                           : 1.0f;
        HostTensor a = rnd({2, 5, 9}, true, 17, -2.0f, 2.0f);
        const HostTensor in[] = {a};
        run_op_bf16([attrs](Backend be) { return Activation::create(attrs, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_activation");
    }
}

// ============================================================
// GEMM family
// ============================================================

NNOPS_TEST(cuda_bf16_matmul) {
    const int64_t shapes[][2] = {{64, 64}, {65, 63}, {5, 7}, {129, 250}};
    for (const auto& s : shapes) {
        HostTensor a = rnd({s[0], 96}, true, 17);
        HostTensor b = rnd({96, s[1]}, true, 29);
        const HostTensor in[] = {a, b};
        run_op_bf16([](Backend be) { return MatMul::create(MatMulAttributes{}, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_matmul");
    }

    // K shorter than one 16-element MMA step, and a k tile plus a partial one.
    for (int64_t k : {1, 17, 130}) {
        HostTensor a = rnd({40, k}, true, 17);
        HostTensor b = rnd({k, 40}, true, 29);
        const HostTensor in[] = {a, b};
        run_op_bf16([](Backend be) { return MatMul::create(MatMulAttributes{}, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_matmul_ragged_k");
    }

    // Both transpose flags: each picks a different copy atom and LDSM operand.
    for (int flags = 0; flags < 4; ++flags) {
        MatMulAttributes attrs;
        attrs.transpose_a = (flags & 1) != 0;
        attrs.transpose_b = (flags & 2) != 0;
        HostTensor a = rnd({attrs.transpose_a ? 48 : 37, attrs.transpose_a ? 37 : 48}, true, 17);
        HostTensor b = rnd({attrs.transpose_b ? 91 : 48, attrs.transpose_b ? 48 : 91}, true, 29);
        const HostTensor in[] = {a, b};
        run_op_bf16([attrs](Backend be) { return MatMul::create(attrs, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_matmul_transposes");
    }

    {
        // A grid one tile deep against a long reduction: the split-K cut, whose
        // partial planes are f32 whatever the operand type.
        HostTensor a = rnd({16, 2048}, true, 17);
        HostTensor b = rnd({2048, 256}, true, 29);
        const HostTensor in[] = {a, b};
        run_op_bf16([](Backend be) { return MatMul::create(MatMulAttributes{}, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_matmul_splitk");
    }

    {
        // Bias, alpha and a relu epilogue: bf16 flows through the epilogue too.
        MatMulAttributes attrs;
        attrs.alpha = 0.5f;
        attrs.epilogue.type = EpilogueActivateType::Relu;
        HostTensor a = rnd({37, 65}, false, 17);
        HostTensor b = rnd({65, 43}, false, 29);
        HostTensor bias = rnd({43}, false, 31, -0.25f, 0.25f);
        const HostTensor in[] = {a, b, bias};
        run_op_bf16([attrs](Backend be) { return MatMul::create(attrs, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_matmul_epilogue");
    }
}

NNOPS_TEST(cuda_bf16_linear) {
    for (int64_t m : {1, 7, 65}) {
        LinearAttributes attrs;
        HostTensor x = rnd({m, 96}, true, 11);
        HostTensor w = rnd({64, 96}, false, 13);
        HostTensor bias = rnd({64}, false, 19, -0.25f, 0.25f);
        const HostTensor in[] = {x, w, bias};
        run_op_bf16([attrs](Backend be) { return Linear::create(attrs, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_linear");
    }
    {
        LinearAttributes attrs;
        attrs.epilogue.type = EpilogueActivateType::Gelu;
        HostTensor x = rnd({33, 130}, true, 11);
        HostTensor w = rnd({40, 130}, false, 13);
        const HostTensor in[] = {x, w};
        run_op_bf16([attrs](Backend be) { return Linear::create(attrs, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_linear_epilogue");
    }
}

// ============================================================
// Convolution family
// ============================================================

NNOPS_TEST(cuda_bf16_conv2d) {
    struct Case {
        int64_t ic = 8, oc = 16, kh = 3, kw = 3, sh = 1, sw = 1, ph = 1, pw = 1;
        bool channel_last = false;
        bool bias = false;
        bool padded = false;
    };
    std::vector<Case> cases;
    {
        Case c;                      // 3x3 s1 p1, channel-first pitched
        c.padded = true;
        cases.push_back(c);
    }
    {
        Case c;                      // stride 2
        c.sh = c.sw = 2;
        c.ph = c.pw = 0;
        c.bias = true;
        cases.push_back(c);
    }
    {
        Case c;                      // the channel-last route
        c.channel_last = true;
        cases.push_back(c);
    }
    {
        Case c;                      // 1x1 pointwise, channel-last
        c.channel_last = true;
        c.kh = c.kw = 1;
        c.ph = c.pw = 0;
        c.oc = 24;
        cases.push_back(c);
    }

    for (const Case& c : cases) {
        Conv2DAttributes attrs;
        attrs.kernel_size = {c.kh, c.kw};
        attrs.stride = {c.sh, c.sw};
        attrs.padding = {c.ph, c.pw};

        const TensorLayout act_layout =
            c.channel_last ? TensorLayout::NHWC : TensorLayout::NCHW;
        const TensorLayout wgt_layout =
            c.channel_last ? TensorLayout::HWIO : TensorLayout::NCHW;

        HostTensor act = rnd({1, c.ic, 10, 12}, c.padded, 23, -0.5f, 0.5f, act_layout);
        HostTensor wgt =
            c.channel_last
                ? rnd({c.kh, c.kw, c.ic, c.oc}, false, 29, -0.25f, 0.25f, wgt_layout)
                : rnd({c.oc, c.ic, c.kh, c.kw}, false, 29, -0.25f, 0.25f, wgt_layout);
        std::vector<HostTensor> own = {act, wgt};
        if (c.bias) { own.push_back(rnd({c.oc}, false, 31, -0.25f, 0.25f)); }
        run_op_bf16([attrs](Backend be) { return Conv2D::create(attrs, be); },
                    std::span<const HostTensor>(own), kTolAbs, kTolRel,
                    "cuda_bf16_conv2d");
    }
}

NNOPS_TEST(cuda_bf16_transpose_conv2d) {
    for (bool channel_last : {false, true}) {
        TransposeConv2DAttributes attrs;
        attrs.kernel_size = {3, 3};
        attrs.stride = {2, 2};
        attrs.padding = {1, 1};

        const TensorLayout act_layout =
            channel_last ? TensorLayout::NHWC : TensorLayout::NCHW;
        HostTensor act = rnd({1, 6, 7, 9}, !channel_last, 23, -0.5f, 0.5f, act_layout);
        HostTensor wgt = rnd({6, 4, 3, 3}, false, 29, -0.25f, 0.25f);
        HostTensor bias = rnd({4}, false, 31, -0.25f, 0.25f);
        const HostTensor in[] = {act, wgt, bias};
        run_op_bf16([attrs](Backend be) { return TransposeConv2D::create(attrs, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_transpose_conv2d");
    }
}

NNOPS_TEST(cuda_bf16_depthwise_conv) {
    for (bool channel_last : {false, true}) {
        DepthwiseConvAttributes attrs;
        attrs.kernel_size = {1, 3, 3};
        attrs.padding = {0, 1, 1};

        const int64_t C = 8;
        const TensorLayout in_layout =
            channel_last ? TensorLayout::NHWC : TensorLayout::NCHW;
        HostTensor in = rnd({1, C, 9, 11}, !channel_last, 17, -1.0f, 1.0f, in_layout);

        std::vector<int64_t> w_shape;
        TensorLayout w_layout = in_layout;
        if (channel_last) {
            w_shape = {3, 3, C};
            w_layout = TensorLayout::HWCM;
        } else {
            w_shape = {C, 1, 3, 3};
        }
        HostTensor w = rnd(w_shape, false, 23, -0.5f, 0.5f, w_layout);
        HostTensor b = rnd({C}, false, 29, -0.25f, 0.25f);
        const HostTensor ins[] = {in, w, b};
        run_op_bf16([attrs](Backend be) { return DepthwiseConv::create(attrs, be); },
                    std::span<const HostTensor>(ins), kTolAbs, kTolRel,
                    "cuda_bf16_depthwise_conv");
    }
}

NNOPS_TEST(cuda_bf16_depthwise_transpose_conv) {
    for (bool channel_last : {false, true}) {
        DepthwiseTransposeConv2DAttributes attrs;
        attrs.kernel_size = {2, 2};
        attrs.stride = {2, 2};

        const int64_t C = 7;
        const TensorLayout layout =
            channel_last ? TensorLayout::NHWC : TensorLayout::NCHW;
        HostTensor act = rnd({1, C, 5, 6}, !channel_last, 23, -0.5f, 0.5f, layout);
        HostTensor wgt = channel_last
            ? rnd({2, 2, C}, false, 29, -0.25f, 0.25f, TensorLayout::HWCM)
            : rnd({C, 1, 2, 2}, false, 29, -0.25f, 0.25f);
        const HostTensor in[] = {act, wgt};
        run_op_bf16([attrs](Backend be) {
                        return DepthwiseTransposeConv2D::create(attrs, be);
                    },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_depthwise_transpose_conv");
    }
}

// ============================================================
// Reductions and windows
// ============================================================

NNOPS_TEST(cuda_bf16_pooling) {
    for (bool channel_last : {false, true}) {
        for (PoolingType type : {PoolingType::Max, PoolingType::Average}) {
            PoolingAttributes attrs = window(type, 3, 2, 1);
            if (type == PoolingType::Average) { attrs.exclude_pad = true; }
            HostTensor a = rnd({1, 8, 9, 11}, !channel_last, 13, -1.0f, 1.0f,
                               channel_last ? TensorLayout::NHWC
                                            : TensorLayout::NCHW);
            const HostTensor in[] = {a};
            run_op_bf16([attrs](Backend be) { return Pooling::create(attrs, be); },
                        std::span<const HostTensor>(in), kTolAbs, kTolRel,
                        "cuda_bf16_pooling");
        }
    }
}

NNOPS_TEST(cuda_bf16_softmax) {
    for (bool log_softmax : {false, true}) {
        SoftmaxAttributes attrs;
        attrs.axis = -1;
        attrs.log_softmax = log_softmax;
        HostTensor a = rnd({2, 5, 37}, true, 17, -2.0f, 2.0f);
        const HostTensor in[] = {a};
        run_op_bf16([attrs](Backend be) { return Softmax::create(attrs, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_softmax");
    }
}

NNOPS_TEST(cuda_bf16_norm) {
    {
        NormAttributes attrs;
        attrs.type = NormType::LayerNorm;
        HostTensor x = rnd({4, 8, 16}, true, 23, -1.0f, 1.0f);
        HostTensor scale = rnd({16}, false, 81, -1.0f, 1.0f);
        HostTensor bias = rnd({16}, false, 82, -1.0f, 1.0f);
        const HostTensor in[] = {x, scale, bias};
        run_op_bf16([attrs](Backend be) { return Norm::create(attrs, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_layernorm");
    }
    {
        NormAttributes attrs;
        attrs.type = NormType::RMSNorm;
        HostTensor x = rnd({4, 7, 32}, true, 23, -1.0f, 1.0f);
        HostTensor scale = rnd({32}, false, 91, -1.0f, 1.0f);
        const HostTensor in[] = {x, scale};
        run_op_bf16([attrs](Backend be) { return Norm::create(attrs, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_rmsnorm");
    }
    {
        NormAttributes attrs;
        attrs.type = NormType::GroupNorm;
        attrs.num_groups = 4;
        HostTensor x = rnd({2, 8, 6, 7}, true, 23, -1.0f, 1.0f);
        HostTensor scale = rnd({8}, false, 95, -1.0f, 1.0f);
        HostTensor bias = rnd({8}, false, 96, -1.0f, 1.0f);
        const HostTensor in[] = {x, scale, bias};
        run_op_bf16([attrs](Backend be) { return Norm::create(attrs, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_groupnorm");
    }
}

NNOPS_TEST(cuda_bf16_reduce) {
    for (ReduceType type : {ReduceType::Sum, ReduceType::Mean, ReduceType::Max,
                            ReduceType::Min}) {
        ReduceAttributes attrs;
        attrs.type = type;
        attrs.axes.push_back(-1);
        attrs.keepdims = true;
        HostTensor a = rnd({3, 4, 13}, true, 17, -1.0f, 1.0f);
        const HostTensor in[] = {a};
        run_op_bf16([attrs](Backend be) { return Reduce::create(attrs, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_reduce");
    }
}

NNOPS_TEST(cuda_bf16_resize) {
    for (ResizeMode mode : {ResizeMode::Nearest, ResizeMode::Linear,
                            ResizeMode::Cubic}) {
        ResizeAttributes attrs;
        attrs.mode = mode;
        attrs.output_size = {0, 13, 17};
        HostTensor a = rnd({1, 4, 7, 9}, true, 17, -1.0f, 1.0f);
        const HostTensor in[] = {a};
        run_op_bf16([attrs](Backend be) { return Resize::create(attrs, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_resize");
    }
}

NNOPS_TEST(cuda_bf16_cumsum) {
    for (bool reverse : {false, true}) {
        CumSumAttributes attrs;
        attrs.axis = 1;
        attrs.reverse = reverse;
        HostTensor a = rnd({3, 11, 5}, true, 17, -1.0f, 1.0f);
        const HostTensor in[] = {a};
        run_op_bf16([attrs](Backend be) { return CumSum::create(attrs, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_cumsum");
    }
}

// ============================================================
// Movement
// ============================================================

NNOPS_TEST(cuda_bf16_move) {
    {
        HostTensor a = rnd({2, 3, 4, 5}, true, 17);
        const HostTensor in[] = {a};
        run_op_bf16([](Backend be) {
                        return Permute::create(PermuteAttributes{{0, 1, 3, 2}}, be);
                    },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_permute");
    }
    {
        SliceAttributes attrs;
        attrs.starts = {1, 2};
        attrs.ends = {5, 7};
        attrs.axes = {1, 2};
        HostTensor a = rnd({3, 8, 9}, true, 17);
        const HostTensor in[] = {a};
        run_op_bf16([attrs](Backend be) { return Slice::create(attrs, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_slice");
    }
    {
        HostTensor a = rnd({2, 3, 4}, true, 17);
        HostTensor b = rnd({2, 5, 4}, true, 29);
        const HostTensor in[] = {a, b};
        run_op_bf16([](Backend be) { return Concat::create(ConcatAttributes{1}, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_concat");
    }
    {
        HostTensor a = rnd({2, 6, 4}, true, 17);
        const HostTensor in[] = {a};
        run_op_bf16([](Backend be) { return Split::create(SplitAttributes{1, {2, 4}, 0}, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_split");
    }
    {
        HostTensor a = rnd({3, 5}, true, 17);
        TileAttributes attrs;
        attrs.repeats = {2, 3};
        const HostTensor in[] = {a};
        run_op_bf16([attrs](Backend be) { return Tile::create(attrs, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_tile");
    }
    for (PadAttributes::Mode mode : {PadAttributes::Mode::Constant,
                                     PadAttributes::Mode::Edge,
                                     PadAttributes::Mode::Reflect}) {
        PadAttributes attrs;
        attrs.pads = {0, 1, 1, 1, 0, 1, 1, 1};
        attrs.mode = mode;
        attrs.constant_value = 0.25f;
        HostTensor a = rnd({2, 3, 6, 7}, true, 17);
        const HostTensor in[] = {a};
        run_op_bf16([attrs](Backend be) { return Pad::create(attrs, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_pad");
    }
}

// ============================================================
// Index-driven
// ============================================================

NNOPS_TEST(cuda_bf16_gather_scatter) {
    {
        GatherAttributes attrs;
        attrs.axis = 1;
        HostTensor data = rnd({4, 6, 3}, true, 17);
        HostTensor index = idxs({4, 3, 3}, {0, 2, 5, 1, 4, 3, 5, 0, 2, 2, 3, 1});
        const HostTensor in[] = {data, index};
        run_op_bf16([attrs](Backend be) { return Gather::create(attrs, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_gather");
    }
    {
        // Index sets are duplicate-free, so each destination element sees one
        // update and both routes apply the same single operation.
        ScatterElementsAttributes attrs;
        attrs.axis = 1;
        HostTensor data = rnd({4, 6}, false, 17);
        HostTensor index = idxs({4, 3}, {0, 2, 5, 1, 4, 0, 5, 2, 3, 4, 1, 0});
        HostTensor updates = rnd({4, 3}, false, 29);
        const HostTensor in[] = {data, index, updates};
        run_op_bf16([attrs](Backend be) { return ScatterElements::create(attrs, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_scatter");
    }
    {
        ScatterElementsAttributes attrs;
        attrs.axis = 1;
        attrs.reduction = ScatterReduction::Add;
        HostTensor data = rnd({4, 6}, false, 17);
        HostTensor index = idxs({4, 3}, {0, 2, 5, 1, 4, 0, 5, 2, 3, 4, 1, 0});
        HostTensor updates = rnd({4, 3}, false, 29);
        const HostTensor in[] = {data, index, updates};
        run_op_bf16([attrs](Backend be) { return ScatterElements::create(attrs, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_scatter_add");
    }
}

NNOPS_TEST(cuda_bf16_topk_argmax) {
    {
        TopKAttributes attrs;
        attrs.axis = -1;
        attrs.k = 3;
        HostTensor a = distinct({3, 9}, 1);
        const HostTensor in[] = {a};
        run_op_bf16([attrs](Backend be) { return TopK::create(attrs, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_topk");
    }
    {
        ArgMinMaxAttributes attrs;
        attrs.axis = 1;
        HostTensor a = distinct({4, 7}, 1);
        const HostTensor in[] = {a};
        run_op_bf16([attrs](Backend be) { return ArgMax::create(attrs, be); },
                    std::span<const HostTensor>(in), kTolAbs, kTolRel,
                    "cuda_bf16_argmax");
    }
}

#endif  // NNOPS_HAS_CUDA
