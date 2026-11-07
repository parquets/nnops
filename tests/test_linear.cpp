/// Unit tests for the Linear operator.
///
/// Linear is X × W^T + bias on the MatMul kernel; most cases compare against
/// `linear_ref` on the same operands: f32/f16, u8×s8 and s8×s8 with per-tensor,
/// per-token and per-channel parameters, optional bias, epilogue, add_to, and
/// rank-1 / batched inputs.
///
/// add_to rides MatMul's beta only for fp operands with an identity epilogue —
/// the residual must not pass through the activation. Any other fp-output case
/// takes a compact output-shaped temporary and adds the residual afterwards: a
/// non-identity epilogue, or any int8 product (the integer route ignores
/// alpha/beta and starts C at zero). An integer output (s32 compensated
/// accumulate, s8/u8 requantized with the output's QuantParams) follows
/// `output_dtype`, not the operand dtype, and takes an int32 [N] bias.
///
/// prepackWeights is the query-then-pack contract: an empty output view is the
/// query, answering with layout PackedB (rank 2, N leading; empty for a dtype
/// with no packed form, a u8 weight among them) and a numel the caller sizes the
/// buffer from. The float blob is the GEMM's nr-wide panels over the whole K, so
/// it removes the per-call RHS pack and nothing else: where the raw weight also
/// takes the tiled GEMM the two outputs must be bit-identical. Below the
/// small-product thresholds (M ≤ 4 or M·N·K < 1024) the raw weight runs the
/// pack-free kernel, which a panel-formatted B cannot, so there only the numbers
/// are compared. The int8 blob is the same panels cut into k segments, read by
/// an s8 activation only; its raw arm always takes the tiled GEMM, so those two
/// outputs are bit-identical at every size.

#include "nnops/ops/linear.hpp"
#include "nnops/core/quant_params.hpp"
#include "nnops/detail/q4_1.hpp"
#include "nnops/detail/half.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

using namespace nnops;
using nnops::backend::cpu::half;
using nnops::backend::cpu::half_to_float;
using nnops::backend::cpu::float_to_half;

namespace nnops::backend::cpu::reference {
void linear_ref(const LinearAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* workspace);
}

namespace {

float to_f(float v) { return v; }
float to_f(half v) { return half_to_float(v); }

TensorDesc make_desc(std::initializer_list<int64_t> dims,
                     DataType dtype = DataType::f32) {
    TensorDesc d;
    d.rank = static_cast<int64_t>(dims.size());
    d.dims = std::span<const int64_t>(dims.begin(), dims.size());
    d.dtype = dtype;
    d.layout = TensorLayout::NCHW;
    return d;
}

/// Run Linear and linear_ref over the same inputs; @p initial seeds add_to's residual.
template <class OutT>
void check_linear(const LinearAttributes& attrs,
                  const std::vector<TensorView>& ins,
                  float tol = 1e-5f,
                  std::span<const OutT> initial = {})
{
    std::vector<TensorDesc> descs;
    descs.reserve(ins.size());
    for (const auto& t : ins) { descs.push_back(t.desc()); }

    auto op = Linear::create(attrs, Backend::CPU);
    auto odescs = op->getOutputTensorDesc(descs);
    const TensorDesc& od = odescs[0];

    if constexpr (std::is_same_v<OutT, float>) {
        NNOPS_EXPECT_EQ(od.dtype, DataType::f32);
    } else {
        NNOPS_EXPECT_EQ(od.dtype, DataType::f16);
    }

    std::vector<OutT> got(static_cast<size_t>(od.numel()), OutT{});
    std::vector<OutT> want(static_cast<size_t>(od.numel()), OutT{});
    if (!initial.empty()) {
        got.assign(initial.begin(), initial.end());
        want.assign(initial.begin(), initial.end());
    }

    auto shape = std::span<const int64_t>(od.dims.data(),
                                          static_cast<size_t>(od.rank));
    TensorView og(shape, od.dtype, got.data());
    TensorView ow(shape, od.dtype, want.data());

    op->compute(og, ins);
    nnops::backend::cpu::reference::linear_ref(attrs, ow, ins, {}, nullptr);

    for (size_t i = 0; i < got.size(); ++i) {
        const float g = to_f(got[i]);
        const float e = to_f(want[i]);
        NNOPS_EXPECT_NEAR(g, e, tol * std::max(1.0f, std::abs(e)));
    }
}

std::vector<int8_t> random_s8(int64_t n, uint64_t seed) {
    std::vector<int8_t> v(static_cast<size_t>(n));
    nnops::test::XorShift128 rng(seed);
    for (auto& e : v) {
        e = static_cast<int8_t>(static_cast<int32_t>(rng.next_u64() % 255) - 127);
    }
    return v;
}

std::vector<uint8_t> random_u8(int64_t n, uint64_t seed) {
    std::vector<uint8_t> v(static_cast<size_t>(n));
    nnops::test::XorShift128 rng(seed);
    for (auto& e : v) {
        e = static_cast<uint8_t>(rng.next_u64() & 0xFF);
    }
    return v;
}

QuantParams per_tensor(float scale, int32_t zp) {
    QuantParams qp;
    qp.granularity = QuantGranularity::PerTensor;
    qp.scale = scale;
    qp.zero_point = zp;
    return qp;
}

QuantParams quant_arrays(QuantGranularity g, const float* scale,
                         const int32_t* zp, int64_t n) {
    QuantParams qp;
    qp.granularity = g;
    qp.scale_data = scale;
    qp.zero_point_data = zp;
    qp.num_scales = n;
    return qp;
}

TensorView make_view(std::span<const int64_t> shape, DataType dt, void* data,
                     const QuantParams& qp = {}) {
    return TensorView(shape, dt, data, TensorLayout::NCHW, qp);
}

}  // anonymous namespace

// ---- Hand-verified tests ----

NNOPS_TEST(linear_basic) {
    const int64_t ishape[] = {2, 3};
    const int64_t wshape[] = {4, 3};

    float in_data[6] = {1, 2, 3, 4, 5, 6};
    float w_data[12] = {
        1, 0, 0,   // dot({1,2,3}, {1,0,0}) = 1
        0, 1, 0,   // dot({1,2,3}, {0,1,0}) = 2
        0, 0, 1,   // dot({1,2,3}, {0,0,1}) = 3
        1, 1, 1,   // dot({1,2,3}, {1,1,1}) = 6
    };

    TensorView input(ishape, DataType::f32, in_data);
    TensorView weight(wshape, DataType::f32, w_data);

    auto di = input.desc();
    auto dw = weight.desc();

    auto op = Linear::create(Backend::CPU);
    const TensorDesc desc_arr[] = {di, dw};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 4);
    NNOPS_EXPECT_EQ(static_cast<int>(descs[0].layout), static_cast<int>(TensorLayout::NCHW));
    NNOPS_EXPECT_EQ(static_cast<int>(descs[0].dtype), static_cast<int>(DataType::f32));

    int64_t out_numel = 1;
    for (int64_t i = 0; i < descs[0].rank; ++i) {
        out_numel *= descs[0].dims[i];
    }
    std::vector<float> out_buf(static_cast<size_t>(out_numel), 0.0f);
    auto out = test::make_channel_first(descs[0], out_buf.data());

    const TensorView ins[] = {input, weight};
    op->compute(out, ins);

    // Row 0: dot({1,2,3}, w[i])
    NNOPS_EXPECT_NEAR(out_buf[0], 1.0f, 1e-6f);   // 1*1 + 2*0 + 3*0
    NNOPS_EXPECT_NEAR(out_buf[1], 2.0f, 1e-6f);   // 1*0 + 2*1 + 3*0
    NNOPS_EXPECT_NEAR(out_buf[2], 3.0f, 1e-6f);   // 1*0 + 2*0 + 3*1
    NNOPS_EXPECT_NEAR(out_buf[3], 6.0f, 1e-6f);   // 1*1 + 2*1 + 3*1

    // Row 1: dot({4,5,6}, w[i])
    NNOPS_EXPECT_NEAR(out_buf[4], 4.0f, 1e-6f);   // 4*1 + 5*0 + 6*0
    NNOPS_EXPECT_NEAR(out_buf[5], 5.0f, 1e-6f);   // 4*0 + 5*1 + 6*0
    NNOPS_EXPECT_NEAR(out_buf[6], 6.0f, 1e-6f);   // 4*0 + 5*0 + 6*1
    NNOPS_EXPECT_NEAR(out_buf[7], 15.0f, 1e-6f);  // 4*1 + 5*1 + 6*1
}

NNOPS_TEST(linear_with_bias) {
    const int64_t ishape[] = {1, 2};
    const int64_t wshape[] = {3, 2};
    const int64_t bshape[] = {3};

    float in_data[2] = {1, 2};
    float w_data[6] = {1, 0, 0, 1, 1, 1};
    float b_data[3] = {0.5f, -0.5f, 1.0f};

    TensorView input(ishape, DataType::f32, in_data);
    TensorView weight(wshape, DataType::f32, w_data);
    TensorView bias(bshape, DataType::f32, b_data);

    auto di = input.desc();
    auto dw = weight.desc();

    auto op = Linear::create(Backend::CPU);
    const TensorDesc desc_arr[] = {di, dw};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 1);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 3);
    NNOPS_EXPECT_EQ(static_cast<int>(descs[0].layout), static_cast<int>(TensorLayout::NCHW));
    NNOPS_EXPECT_EQ(static_cast<int>(descs[0].dtype), static_cast<int>(DataType::f32));

    int64_t out_numel = 1;
    for (int64_t i = 0; i < descs[0].rank; ++i) {
        out_numel *= descs[0].dims[i];
    }
    std::vector<float> out_buf(static_cast<size_t>(out_numel), 0.0f);
    auto out = test::make_channel_first(descs[0], out_buf.data());

    const TensorView ins[] = {input, weight, bias};
    op->compute(out, ins);

    // dot({1,2}, {1,0}) + 0.5 = 1 + 0.5 = 1.5
    NNOPS_EXPECT_NEAR(out_buf[0], 1.5f, 1e-6f);
    // dot({1,2}, {0,1}) - 0.5 = 2 - 0.5 = 1.5
    NNOPS_EXPECT_NEAR(out_buf[1], 1.5f, 1e-6f);
    // dot({1,2}, {1,1}) + 1.0 = 3 + 1.0 = 4.0
    NNOPS_EXPECT_NEAR(out_buf[2], 4.0f, 1e-6f);
}

// ---- fp32 / fp16 against the reference ----

NNOPS_TEST(linear_f32_random) {
    const int64_t M = 7, K = 37, N = 19;   // odd sizes exercise the tails
    auto [xv, x] = test::make_random_tensor({M, K}, -1.0f, 1.0f, 1);
    auto [wv, w] = test::make_random_tensor({N, K}, -1.0f, 1.0f, 2);
    auto [bv, b] = test::make_random_tensor({N}, -0.5f, 0.5f, 3);

    check_linear<float>(LinearAttributes{}, {x, w, b});
    check_linear<float>(LinearAttributes{}, {x, w});
}

NNOPS_TEST(linear_gated_swiglu) {
    // The gated form: weight [2N, K], bias [2N], output [..., N]. Compared
    // against the reference, which combines the two halves itself — so the
    // shape rule (N, not 2N), the bias split, the gate formula and the epilogue
    // ordering all have to agree.
    const int64_t M = 7, K = 37, N = 19;
    const int64_t n2 = 2 * N;
    auto [xv, x] = test::make_random_tensor({M, K}, -1.0f, 1.0f, 11);
    auto [wv, w] = test::make_random_tensor({n2, K}, -1.0f, 1.0f, 12);
    auto [bv, b] = test::make_random_tensor({n2}, -0.5f, 0.5f, 13);

    LinearAttributes plain;
    plain.gated = true;
    check_linear<float>(plain, {x, w, b});
    check_linear<float>(plain, {x, w});

    // GPT-OSS's SwiGLU: gate scaled 1.702, value offset 1.0, both clamped at 7.
    LinearAttributes oss;
    oss.gated = true;
    oss.gate_alpha = 1.702f;
    oss.gate_beta = 1.0f;
    oss.gate_limit = 7.0f;
    check_linear<float>(oss, {x, w, b});

    // The epilogue and add_to belong to the combine, not to the two halves.
    Epilogue relu;
    relu.type = EpilogueActivateType::Relu;
    LinearAttributes fused;
    fused.gated = true;
    fused.epilogue = relu;
    fused.add_to = true;
    const std::vector<float> seed(static_cast<size_t>(M * N), 0.25f);
    check_linear<float>(fused, {x, w, b}, 1e-5f, seed);

    // Rank-1 X and the f16 storage.
    auto [xv1, x1] = test::make_random_tensor({K}, -1.0f, 1.0f, 14);
    check_linear<float>(plain, {x1, w, b});
    {
        auto x16 = test::f32_to_f16(xv);
        auto w16 = test::f32_to_f16(wv);
        auto b16 = test::f32_to_f16(bv);
        const int64_t xs[] = {M, K};
        const int64_t ws[] = {n2, K};
        const int64_t bs[] = {n2};
        TensorView xf(xs, DataType::f16, x16.data());
        TensorView wf(ws, DataType::f16, w16.data());
        TensorView bf(bs, DataType::f16, b16.data());
        check_linear<half>(plain, {xf, wf, bf}, 2e-2f);
    }
}

NNOPS_TEST(linear_gated_shape_and_reject) {
    const int64_t ishape[] = {2, 8};
    const int64_t wshape[] = {11, 8};   // odd row count
    float xd[16] = {};
    float wd[88] = {};
    TensorView x(ishape, DataType::f32, xd);
    TensorView w(wshape, DataType::f32, wd);

    LinearAttributes attrs;
    attrs.gated = true;
    auto op = Linear::create(attrs, Backend::CPU);
    const TensorDesc desc_arr[] = {x.desc(), w.desc()};
    const auto outs = op->getOutputTensorDesc(desc_arr);
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t(5));   // 11 / 2

    std::vector<float> out_buf(static_cast<size_t>(outs[0].numel()), 0.0f);
    auto y = test::make_channel_first(outs[0], out_buf.data());
    const TensorView ins[] = {x, w};
    NNOPS_EXPECT_TRUE(op->compute(y, ins) == Status::invalid_argument);

    // int8 has no gated route.
    LinearAttributes q = attrs;
    std::vector<int8_t> xi(16), wi(88);
    TensorView xq(ishape, DataType::s8, xi.data());
    TensorView wq(wshape, DataType::s8, wi.data());
    auto qop = Linear::create(q, Backend::CPU);
    const TensorDesc qd[] = {xq.desc(), wq.desc()};
    const auto qouts = qop->getOutputTensorDesc(qd);
    std::vector<float> qbuf(static_cast<size_t>(qouts[0].numel()), 0.0f);
    auto qy = test::make_channel_first(qouts[0], qbuf.data());
    const TensorView qins[] = {xq, wq};
    NNOPS_EXPECT_TRUE(qop->compute(qy, qins) == Status::unsupported);
}

NNOPS_TEST(linear_batched) {
    const int64_t B = 3, M = 4, K = 16, N = 8;
    auto [xv, x] = test::make_random_tensor({B, M, K}, -1.0f, 1.0f, 4);
    auto [wv, w] = test::make_random_tensor({N, K}, -1.0f, 1.0f, 5);
    auto [bv, b] = test::make_random_tensor({N}, -0.5f, 0.5f, 6);

    check_linear<float>(LinearAttributes{}, {x, w, b});
}

NNOPS_TEST(linear_rank1) {
    const int64_t K = 24, N = 11;
    auto [xv, x] = test::make_random_tensor({K}, -1.0f, 1.0f, 7);
    auto [wv, w] = test::make_random_tensor({N, K}, -1.0f, 1.0f, 8);
    auto [bv, b] = test::make_random_tensor({N}, -0.5f, 0.5f, 9);

    check_linear<float>(LinearAttributes{}, {x, w, b});
}

NNOPS_TEST(linear_f16) {
    const int64_t M = 5, K = 32, N = 13;
    auto [xv, x] = test::make_random_f16_tensor({M, K}, -1.0f, 1.0f, 10);
    auto [wv, w] = test::make_random_f16_tensor({N, K}, -1.0f, 1.0f, 11);
    auto [bv, b] = test::make_random_f16_tensor({N}, -0.5f, 0.5f, 12);

    check_linear<half>(LinearAttributes{}, {x, w, b}, test::kF16AccumTol);
}

NNOPS_TEST(linear_epilogue_addto) {
    const int64_t M = 6, K = 20, N = 9;
    auto [xv, x] = test::make_random_tensor({M, K}, -1.0f, 1.0f, 13);
    auto [wv, w] = test::make_random_tensor({N, K}, -1.0f, 1.0f, 14);
    auto [bv, b] = test::make_random_tensor({N}, -0.5f, 0.5f, 15);

    LinearAttributes attrs;
    attrs.epilogue.type = EpilogueActivateType::Relu;
    attrs.add_to = true;

    // Non-zero residual so add_to is actually exercised.
    nnops::test::XorShift128 rng(16);
    std::vector<float> initial(static_cast<size_t>(M * N));
    rng.fill_float(initial.data(), M * N, -1.0f, 1.0f);

    check_linear<float>(attrs, {x, w, b}, 1e-5f, initial);
}

NNOPS_TEST(linear_addto_identity) {
    const int64_t M = 4, K = 12, N = 6;
    auto [xv, x] = test::make_random_tensor({M, K}, -1.0f, 1.0f, 17);
    auto [wv, w] = test::make_random_tensor({N, K}, -1.0f, 1.0f, 18);
    auto [bv, b] = test::make_random_tensor({N}, -0.5f, 0.5f, 19);

    LinearAttributes attrs;
    attrs.add_to = true;   // identity epilogue → MatMul's beta residual

    nnops::test::XorShift128 rng(20);
    std::vector<float> initial(static_cast<size_t>(M * N));
    rng.fill_float(initial.data(), M * N, -1.0f, 1.0f);

    check_linear<float>(attrs, {x, w, b}, 1e-5f, initial);
}

// ---- Quantized: u8|s8 × s8 → fp ----

NNOPS_TEST(linear_u8s8_per_tensor) {
    const int64_t M = 5, K = 32, N = 17;
    auto xv = random_u8(M * K, 21);
    auto wv = random_s8(N * K, 22);

    const int64_t xshape[] = {M, K};
    const int64_t wshape[] = {N, K};
    auto x = make_view(xshape, DataType::u8, xv.data(),
                       per_tensor(0.02f, 128));
    auto w = make_view(wshape, DataType::s8, wv.data(),
                       per_tensor(0.01f, 0));

    auto [bv, b] = test::make_random_tensor({N}, -0.5f, 0.5f, 23);

    LinearAttributes attrs;
    attrs.output_dtype = DataType::f32;
    check_linear<float>(attrs, {x, w, b}, 1e-4f);
    check_linear<float>(attrs, {x, w}, 1e-4f);
}

NNOPS_TEST(linear_s8s8_asymmetric) {
    const int64_t M = 6, K = 40, N = 13;
    auto xv = random_s8(M * K, 24);
    auto wv = random_s8(N * K, 25);

    std::vector<float> xs(static_cast<size_t>(M));
    std::vector<int32_t> xzp(static_cast<size_t>(M));
    std::vector<float> ws(static_cast<size_t>(N));
    std::vector<int32_t> wzp(static_cast<size_t>(N));
    for (int64_t i = 0; i < M; ++i) {
        xs[static_cast<size_t>(i)] = 0.01f + 0.001f * static_cast<float>(i);
        xzp[static_cast<size_t>(i)] = static_cast<int32_t>(i) - 3;
    }
    for (int64_t i = 0; i < N; ++i) {
        ws[static_cast<size_t>(i)] = 0.005f + 0.001f * static_cast<float>(i);
        wzp[static_cast<size_t>(i)] = static_cast<int32_t>(i % 5) - 2;
    }

    const int64_t xshape[] = {M, K};
    const int64_t wshape[] = {N, K};
    auto x = make_view(xshape, DataType::s8, xv.data(),
                       quant_arrays(QuantGranularity::PerToken,
                                    xs.data(), xzp.data(), M));
    auto w = make_view(wshape, DataType::s8, wv.data(),
                       quant_arrays(QuantGranularity::PerChannel,
                                    ws.data(), wzp.data(), N));

    auto [bv, b] = test::make_random_tensor({N}, -0.5f, 0.5f, 27);

    check_linear<float>(LinearAttributes{}, {x, w, b}, 1e-4f);
}

NNOPS_TEST(linear_s8s8_f16_out) {
    const int64_t M = 4, K = 32, N = 11;
    auto xv = random_s8(M * K, 28);
    auto wv = random_s8(N * K, 29);

    const int64_t xshape[] = {M, K};
    const int64_t wshape[] = {N, K};
    auto x = make_view(xshape, DataType::s8, xv.data(),
                       per_tensor(0.02f, 0));
    auto w = make_view(wshape, DataType::s8, wv.data(),
                       per_tensor(0.01f, 0));

    auto [bv, b] = test::make_random_f16_tensor({N}, -0.5f, 0.5f, 30);

    LinearAttributes attrs;
    attrs.output_dtype = DataType::f16;
    check_linear<half>(attrs, {x, w, b}, test::kF16AccumTol);
}

NNOPS_TEST(linear_int8_addto) {
    const int64_t M = 5, K = 24, N = 7;
    auto xv = random_u8(M * K, 31);
    auto wv = random_s8(N * K, 32);

    const int64_t xshape[] = {M, K};
    const int64_t wshape[] = {N, K};
    auto x = make_view(xshape, DataType::u8, xv.data(),
                       per_tensor(0.02f, 128));
    auto w = make_view(wshape, DataType::s8, wv.data(),
                       per_tensor(0.01f, 0));

    LinearAttributes attrs;
    attrs.add_to = true;   // int8 route ignores beta → temporary + combine

    nnops::test::XorShift128 rng(33);
    std::vector<float> initial(static_cast<size_t>(M * N));
    rng.fill_float(initial.data(), M * N, -1.0f, 1.0f);

    check_linear<float>(attrs, {x, w}, 1e-4f, initial);
}

NNOPS_TEST(linear_quant_output_dtype) {
    // The quantized output follows output_dtype, not the u8/s8 operand dtype.
    const TensorDesc desc_arr[] = {
        make_desc({4, 8}, DataType::u8),
        make_desc({6, 8}, DataType::s8),
    };

    auto f32_op = Linear::create(LinearAttributes{}, Backend::CPU);
    auto f32_out = f32_op->getOutputTensorDesc(desc_arr);
    NNOPS_EXPECT_EQ(f32_out[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(f32_out[0].dims[0], int64_t(4));
    NNOPS_EXPECT_EQ(f32_out[0].dims[1], int64_t(6));

    LinearAttributes attrs;
    attrs.output_dtype = DataType::f16;
    auto f16_op = Linear::create(attrs, Backend::CPU);
    auto f16_out = f16_op->getOutputTensorDesc(desc_arr);
    NNOPS_EXPECT_EQ(f16_out[0].dtype, DataType::f16);

    for (DataType odt : {DataType::s32, DataType::s8, DataType::u8}) {
        LinearAttributes iattrs;
        iattrs.output_dtype = odt;
        auto op = Linear::create(iattrs, Backend::CPU);
        NNOPS_EXPECT_EQ(op->getOutputTensorDesc(desc_arr)[0].dtype, odt);
    }
}

NNOPS_TEST(linear_int8_output_matches_ref) {
    // s32/s8/u8 against linear_ref bit for bit: int32 [N] bias, Relu, per-token
    // X, per-channel W, and K spanning several k-blocks.
    for (DataType odt : {DataType::s32, DataType::s8, DataType::u8}) {
        const int64_t M = 5, K = 600, N = 13;
        auto xv = random_s8(M * K, 41);
        auto wv = random_s8(N * K, 42);

        std::vector<float> xs(static_cast<size_t>(M));
        std::vector<int32_t> xzp(static_cast<size_t>(M));
        std::vector<float> ws(static_cast<size_t>(N));
        std::vector<int32_t> wzp(static_cast<size_t>(N));
        for (int64_t i = 0; i < M; ++i) {
            xs[static_cast<size_t>(i)] = 0.01f + 0.001f * static_cast<float>(i);
            xzp[static_cast<size_t>(i)] = static_cast<int32_t>(i % 7) - 3;
        }
        for (int64_t i = 0; i < N; ++i) {
            ws[static_cast<size_t>(i)] = 0.005f + 0.001f * static_cast<float>(i);
            wzp[static_cast<size_t>(i)] = static_cast<int32_t>(i % 5) - 2;
        }

        const int64_t xshape[] = {M, K};
        const int64_t wshape[] = {N, K};
        auto x = make_view(xshape, DataType::s8, xv.data(),
                           quant_arrays(QuantGranularity::PerToken,
                                        xs.data(), xzp.data(), M));
        auto w = make_view(wshape, DataType::s8, wv.data(),
                           quant_arrays(QuantGranularity::PerChannel,
                                        ws.data(), wzp.data(), N));

        std::vector<int32_t> bias_buf(static_cast<size_t>(N));
        nnops::test::XorShift128 rng(43);
        for (auto& b : bias_buf) {
            b = static_cast<int32_t>(rng.next_u64() % 20001) - 10000;
        }
        const int64_t bshape[] = {N};
        TensorView b(bshape, DataType::s32, bias_buf.data());

        LinearAttributes attrs;
        attrs.output_dtype = odt;
        attrs.epilogue.type = EpilogueActivateType::Relu;

        auto op = Linear::create(attrs, Backend::CPU);
        const TensorDesc descs[] = {x.desc(), w.desc()};
        const TensorDesc od = op->getOutputTensorDesc(descs)[0];
        NNOPS_EXPECT_EQ(od.dtype, odt);

        // 1024 is a power of two: the requant scale adds no rounding of its own.
        const QuantParams oq = per_tensor(1024.0f, odt == DataType::u8 ? 128 : -7);
        std::vector<uint8_t> got(static_cast<size_t>(od.nbytes()), 0);
        std::vector<uint8_t> want(static_cast<size_t>(od.nbytes()), 0);
        const auto shape = std::span<const int64_t>(od.dims.data(),
                                                    static_cast<size_t>(od.rank));
        TensorView og(shape, odt, got.data(), TensorLayout::NCHW, oq);
        TensorView ow(shape, odt, want.data(), TensorLayout::NCHW, oq);

        const TensorView ins[] = {x, w, b};
        op->compute(og, ins);
        nnops::backend::cpu::reference::linear_ref(attrs, ow, ins, {}, nullptr);

        NNOPS_EXPECT_EQ(std::memcmp(got.data(), want.data(), got.size()), 0);
    }
}

// ---- The prepacked weight — Linear::prepackWeights ----

namespace {

struct PrepackShape {
    int64_t M, K, N;
    bool bias, relu;
};

/// Raw weight vs the same weight through the two-step contract; @p bitwise when
/// both take the tiled GEMM, so not a single bit may move.
template <class T>
void check_prepacked_weight(const PrepackShape& c, uint64_t seed,
                            bool bitwise = true)
{
    const DataType dt = std::is_same_v<T, half> ? DataType::f16 : DataType::f32;
    const int64_t ishape[] = {c.M, c.K};
    const int64_t wshape[] = {c.N, c.K};
    const int64_t bshape[] = {c.N};

    auto fill = [](std::vector<T>& v, float lo, float hi, uint64_t s) {
        nnops::test::XorShift128 rng(s);
        for (auto& e : v) {
            const float f = lo + (hi - lo) *
                static_cast<float>(rng.next_u64() % 10001) / 10000.0f;
            if constexpr (std::is_same_v<T, half>) { e = float_to_half(f); }
            else { e = f; }
        }
    };

    std::vector<T> xv(static_cast<size_t>(c.M * c.K));
    std::vector<T> wv(static_cast<size_t>(c.N * c.K));
    std::vector<T> bv(static_cast<size_t>(c.N));
    fill(xv, -1.0f, 1.0f, seed);
    fill(wv, -1.0f, 1.0f, seed + 1);
    fill(bv, -0.5f, 0.5f, seed + 2);

    TensorView x(ishape, dt, xv.data());
    TensorView weight(wshape, dt, wv.data());
    TensorView bias;
    if (c.bias) { bias = TensorView(bshape, dt, bv.data()); }

    LinearAttributes attrs;
    if (c.relu) { attrs.epilogue.type = EpilogueActivateType::Relu; }
    auto op = Linear::create(attrs, Backend::CPU);

    std::vector<TensorView> ins = {x, weight};
    if (c.bias) { ins.push_back(bias); }
    std::vector<TensorDesc> descs;
    for (const auto& t : ins) { descs.push_back(t.desc()); }
    const TensorDesc od = op->getOutputTensorDesc(descs)[0];

    std::vector<T> got(static_cast<size_t>(od.numel()));
    std::vector<T> want(static_cast<size_t>(od.numel()));
    const auto oshape = std::span<const int64_t>(od.dims.data(),
                                                 static_cast<size_t>(od.rank));
    TensorView og(oshape, od.dtype, got.data());
    TensorView ow(oshape, od.dtype, want.data());

    NNOPS_EXPECT_EQ(op->compute(og, ins, {}, nullptr), Status::ok);

    // ---- two-step contract: query the packed descriptor, allocate, fill
    TensorView q;
    const TensorView wq[] = {weight};
    TensorView oq[] = {q};
    op->prepackWeights(wq, oq);
    q = oq[0];
    // A query answer carries no storage: rank 2 is what separates it from
    // "this weight has no packed form".
    NNOPS_EXPECT_EQ(q.rank(), int64_t(2));
    NNOPS_EXPECT_EQ(static_cast<int>(q.layout()),
                    static_cast<int>(TensorLayout::PackedB));
    // N leads — the dim shape inference reads — and numel is the buffer a caller sizes.
    NNOPS_EXPECT_EQ(q.shape(0), c.N);
    NNOPS_EXPECT_TRUE(q.desc().numel() >= c.N * c.K);
    NNOPS_EXPECT_EQ(q.data_type(), dt);

    std::vector<T> blob(static_cast<size_t>(q.desc().numel()));
    TensorView packed(q.shape_span(), q.data_type(), blob.data(), q.pitch(),
                      q.layout());
    const TensorView wp[] = {weight};
    TensorView opk[] = {packed};
    op->prepackWeights(wp, opk);

    std::vector<TensorView> pins = ins;
    pins[1] = packed;
    NNOPS_EXPECT_EQ(op->compute(ow, pins, {}, nullptr), Status::ok);

    if (bitwise) {
        NNOPS_EXPECT_TRUE(std::memcmp(got.data(), want.data(),
                                      got.size() * sizeof(T)) == 0);
        return;
    }
    const float tol = std::is_same_v<T, half> ? test::kF16AccumTol : 1e-5f;
    for (size_t i = 0; i < got.size(); ++i) {
        const float e = to_f(want[i]);
        NNOPS_EXPECT_NEAR(to_f(got[i]), e, tol * std::max(1.0f, std::abs(e)));
    }
}

}  // anonymous namespace

NNOPS_TEST(linear_prepacked_weight) {
    // Shapes for the pack's seams: partial n panel, K not a k-block multiple,
    // N below one panel, deep K.
    struct Row { PrepackShape s; bool bitwise; };
    const Row cases[] = {
        // Both tails; M > 4 and past the small-product threshold, so both arms
        // take the tiled GEMM.
        {{  7,   37,  19, true,  false}, true},
        // One row: the raw weight runs the pack-free kernel — numbers only.
        {{  1,    8,   5, false, false}, false},
        {{ 33,  300,  48, true,  true},  true},   // several k-blocks, Relu
        {{ 64,  256,  96, false, false}, true},   // a layer shape, no bias
        {{  5, 1024,  17, true,  false}, true},   // deep K, narrow N
    };

    const size_t ncases = sizeof(cases) / sizeof(cases[0]);
    for (size_t i = 0; i < ncases; ++i) {
        const uint64_t seed = 700 + 10 * static_cast<uint64_t>(i);
        check_prepacked_weight<float>(cases[i].s, seed, cases[i].bitwise);
        check_prepacked_weight<half>(cases[i].s, seed + 5, cases[i].bitwise);
    }
}

NNOPS_TEST(linear_prepacked_weight_wrong_k) {
    // The blob carries no K, so packing against one K and computing another must
    // be refused rather than read past its panels.
    const int64_t wshape[] = {16, 37};
    auto [wv, weight] = test::make_random_tensor({16, 37}, -1.0f, 1.0f, 41);

    auto op = Linear::create(Backend::CPU);
    TensorView q;
    const TensorView wq[] = {weight};
    TensorView oq[] = {q};
    op->prepackWeights(wq, oq);
    q = oq[0];

    std::vector<float> blob(static_cast<size_t>(q.desc().numel()));
    TensorView packed(q.shape_span(), q.data_type(), blob.data(), q.pitch(),
                      q.layout());
    const TensorView wp[] = {weight};
    TensorView opk[] = {packed};
    op->prepackWeights(wp, opk);

    auto [xv, x] = test::make_random_tensor({4, 32}, -1.0f, 1.0f, 42);  // K = 32, not 37
    const int64_t oshape[] = {4, 16};
    std::vector<float> obuf(static_cast<size_t>(4 * 16));
    TensorView out(oshape, DataType::f32, obuf.data());

    const TensorView ins[] = {x, packed};
    NNOPS_EXPECT_EQ(op->compute(out, ins, {}, nullptr),
                    Status::invalid_argument);
}

namespace {

/// s8 weight, packed once vs packed per call, same operands. Both arms take the
/// tiled int8 GEMM — the blob only changes the k-block split, and integer
/// accumulation is exact — so the outputs are bit-identical. @p asym drives a
/// non-zero activation zero point, the case whose output pass still needs Σ_k B
/// and therefore reads the packed panels.
void check_prepacked_s8(int64_t M, int64_t K, int64_t N, bool asym,
                        DataType odt, uint64_t seed)
{
    auto xv = random_s8(M * K, seed);
    auto wv = random_s8(N * K, seed + 1);

    std::vector<float> xs(static_cast<size_t>(M), 0.01f);
    std::vector<int32_t> xzp(static_cast<size_t>(M), 0);
    std::vector<float> ws(static_cast<size_t>(N), 0.005f);
    std::vector<int32_t> wzp(static_cast<size_t>(N), 0);
    QuantParams xq = per_tensor(0.01f, 0);
    QuantParams wq = per_tensor(0.005f, 0);
    if (asym) {
        for (int64_t i = 0; i < M; ++i) {
            xs[static_cast<size_t>(i)] = 0.01f + 0.001f * static_cast<float>(i);
            xzp[static_cast<size_t>(i)] = static_cast<int32_t>(i % 7) - 3;
        }
        for (int64_t i = 0; i < N; ++i) {
            ws[static_cast<size_t>(i)] = 0.005f + 0.001f * static_cast<float>(i);
            wzp[static_cast<size_t>(i)] = static_cast<int32_t>(i % 5) - 2;
        }
        xq = quant_arrays(QuantGranularity::PerToken, xs.data(), xzp.data(), M);
        wq = quant_arrays(QuantGranularity::PerChannel, ws.data(), wzp.data(), N);
    }

    const int64_t xshape[] = {M, K};
    const int64_t wshape[] = {N, K};
    TensorView x = make_view(xshape, DataType::s8, xv.data(), xq);
    TensorView weight = make_view(wshape, DataType::s8, wv.data(), wq);

    LinearAttributes attrs;
    attrs.output_dtype = odt;
    auto op = Linear::create(attrs, Backend::CPU);

    const TensorDesc descs[] = {x.desc(), weight.desc()};
    const TensorDesc od = op->getOutputTensorDesc(descs)[0];
    const auto oshape = std::span<const int64_t>(od.dims.data(),
                                                 static_cast<size_t>(od.rank));
    std::vector<std::byte> raw_buf(static_cast<size_t>(od.numel()) *
                                   data_type_size(od.dtype));
    std::vector<std::byte> pk_buf(raw_buf.size());
    TensorView raw(oshape, od.dtype, raw_buf.data());
    TensorView pk(oshape, od.dtype, pk_buf.data());

    const TensorView ins[] = {x, weight};
    NNOPS_EXPECT_EQ(op->compute(raw, ins, {}, nullptr), Status::ok);

    // Query, allocate, fill. The blob carries the weight's quantization — the
    // output pass reads the per-channel scales off the view it is handed.
    TensorView q;
    TensorView oq[] = {q};
    const TensorView wq_in[] = {weight};
    op->prepackWeights(wq_in, oq);
    q = oq[0];
    NNOPS_EXPECT_EQ(q.rank(), int64_t(2));
    NNOPS_EXPECT_EQ(static_cast<int>(q.layout()),
                    static_cast<int>(TensorLayout::PackedB));
    NNOPS_EXPECT_EQ(q.shape(0), N);
    NNOPS_EXPECT_TRUE(q.desc().numel() >= N * K);

    std::vector<int8_t> blob(static_cast<size_t>(q.desc().numel()));
    TensorView packed(q.shape_span(), q.data_type(), blob.data(), q.pitch(),
                      q.layout(), weight.quant_params());
    const TensorView wp[] = {weight};
    TensorView opk[] = {packed};
    op->prepackWeights(wp, opk);

    const TensorView pins[] = {x, packed};
    NNOPS_EXPECT_EQ(op->compute(pk, pins, {}, nullptr), Status::ok);
    NNOPS_EXPECT_TRUE(std::memcmp(raw_buf.data(), pk_buf.data(),
                                  raw_buf.size()) == 0);
}

}  // anonymous namespace

NNOPS_TEST(linear_prepacked_weight_int8) {
    // 33: one short segment. 1031: three k segments (512, 512, 7) — a final
    // segment shorter than a dot group. 512/1024: whole segments. The asym rows
    // are the ones whose output pass needs the B column sums.
    check_prepacked_s8(  7,   33, 19, false, DataType::f32, 900);
    check_prepacked_s8(  6, 1031, 21, false, DataType::f32, 910);
    check_prepacked_s8(  6,   40, 13, true,  DataType::f32, 920);
    check_prepacked_s8(  9, 1024, 17, true,  DataType::f32, 930);
    check_prepacked_s8(  5,  512, 40, false, DataType::s32, 940);
    // One row and two: the narrow-M kernels read the blob too.
    check_prepacked_s8(  1, 4096, 37, false, DataType::f32, 950);
    check_prepacked_s8(  2,  777, 21, true,  DataType::f32, 960);
}

NNOPS_TEST(linear_prepacked_weight_quantized_has_no_blob) {
    // Only s8 has a packed form: an s8 blob's panel correction block is the s8
    // activation shift, so a u8 weight (and every other dtype) answers rank 0.
    const int64_t wshape[] = {8, 16};
    std::vector<uint8_t> wv(static_cast<size_t>(8 * 16), 3);
    TensorView weight(wshape, DataType::u8, wv.data());

    auto op = Linear::create(Backend::CPU);
    TensorView q;
    const TensorView wq[] = {weight};
    TensorView oq[] = {q};
    op->prepackWeights(wq, oq);
    NNOPS_EXPECT_TRUE(oq[0].is_empty());
    NNOPS_EXPECT_EQ(oq[0].rank(), int64_t(0));
}

// ---- Q4_1 weights ----------------------------------------------------------

namespace {

/// One [rows, cols] Q4_1 matrix (whole-block row padding) plus its decoded
/// values; d and m are f16-exact, so the decoded values are exact in f32.
struct Q4Weight {
    std::vector<uint8_t> bytes;
    std::vector<float>   decoded;
    int64_t rows = 0, cols = 0, blocks = 0;
};

Q4Weight make_q4_weight(int64_t rows, int64_t cols, uint32_t seed) {
    Q4Weight w;
    w.rows = rows;
    w.cols = cols;
    w.blocks = (cols + q4_1_block_elems - 1) / q4_1_block_elems;
    w.bytes.assign(static_cast<size_t>(rows * w.blocks
                                       * static_cast<int64_t>(q4_1_block_bytes)), 0);
    w.decoded.assign(static_cast<size_t>(rows * cols), 0.0f);

    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t b = 0; b < w.blocks; ++b) {
            uint8_t* blk = w.bytes.data() + static_cast<size_t>(
                (r * w.blocks + b) * static_cast<int64_t>(q4_1_block_bytes));
            const float d = 0.125f * static_cast<float>(1 + (seed + r + b) % 8);
            const float m = -0.25f * static_cast<float>((seed + 2 * r + b) % 4);
            const uint16_t d_bits = backend::cpu::half_to_bits(float_to_half(d));
            const uint16_t m_bits = backend::cpu::half_to_bits(float_to_half(m));
            std::memcpy(blk + 0, &d_bits, 2);
            std::memcpy(blk + 2, &m_bits, 2);
            for (int j = 0; j < 16; ++j) {
                const int q0 = static_cast<int>((seed + r * 13 + b * 7 + j * 5) & 0x0F);
                const int q1 = static_cast<int>((seed + r * 7 + b * 11 + j * 3) & 0x0F);
                blk[4 + j] = static_cast<uint8_t>(q0 | (q1 << 4));
            }
            const int64_t base = r * cols + b * q4_1_block_elems;
            const int n = static_cast<int>(std::min<int64_t>(
                q4_1_block_elems, cols - b * q4_1_block_elems));
            for (int j = 0; j < n; ++j) {
                const int q = (j < 16) ? (blk[4 + j] & 0x0F) : (blk[4 + j - 16] >> 4);
                w.decoded[static_cast<size_t>(base + j)] =
                    half_to_float(float_to_half(d)) * static_cast<float>(q) +
                    half_to_float(float_to_half(m));
            }
        }
    }
    return w;
}

template <class T>
T from_f32(float f) {
    if constexpr (std::is_same_v<T, half>) { return float_to_half(f); }
    else { return f; }
}

/// Linear with a q4_1 weight against C = A × Wᵀ from the decoded values, then
/// the bias and the activation.
template <class T>
void check_linear_q4(int64_t M, int64_t K, int64_t N, bool with_bias,
                     EpilogueActivateType act, float tol, uint32_t seed)
{
    Q4Weight w = make_q4_weight(N, K, seed);

    std::vector<T> a_buf(static_cast<size_t>(M * K));
    nnops::test::XorShift128 rng(seed + 1);
    for (auto& v : a_buf) {
        v = from_f32<T>(rng.next_float(-1.0f, 1.0f));
    }
    std::vector<T> b_buf(static_cast<size_t>(N));
    for (auto& v : b_buf) {
        v = from_f32<T>(rng.next_float(-0.5f, 0.5f));
    }

    const int64_t ashape[] = {M, K};
    const int64_t wshape[] = {N, K};
    const int64_t bshape[] = {N};
    TensorView a(ashape, std::is_same_v<T, half> ? DataType::f16 : DataType::f32,
                 a_buf.data());
    TensorView wv(wshape, DataType::q4_1, w.bytes.data());
    TensorView bv(bshape, std::is_same_v<T, half> ? DataType::f16 : DataType::f32,
                  b_buf.data());

    LinearAttributes attrs;
    attrs.epilogue.type = act;
    auto op = Linear::create(attrs, Backend::CPU);

    std::vector<TensorDesc> descs = {a.desc(), wv.desc()};
    if (with_bias) { descs.push_back(bv.desc()); }
    const TensorDesc od = op->getOutputTensorDesc(descs)[0];
    NNOPS_EXPECT_EQ(od.dtype, a.data_type());
    NNOPS_EXPECT_EQ(od.dims[0], M);
    NNOPS_EXPECT_EQ(od.dims[1], N);

    std::vector<T> out(static_cast<size_t>(M * N), T{});
    TensorView ov(std::span<const int64_t>(od.dims.data(),
                                           static_cast<size_t>(od.rank)),
                  od.dtype, out.data());
    const std::vector<TensorView> ins = with_bias
        ? std::vector<TensorView>{a, wv, bv} : std::vector<TensorView>{a, wv};
    NNOPS_EXPECT_EQ(op->compute(ov, ins), Status::ok);

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                acc += to_f(a_buf[static_cast<size_t>(m * K + k)]) *
                       w.decoded[static_cast<size_t>(n * K + k)];
            }
            if (with_bias) { acc += to_f(b_buf[static_cast<size_t>(n)]); }
            if (act == EpilogueActivateType::Relu6) {
                acc = std::min(std::max(acc, 0.0f), 6.0f);
            } else if (act == EpilogueActivateType::Relu) {
                acc = std::max(acc, 0.0f);
            }
            const size_t i = static_cast<size_t>(m * N + n);
            NNOPS_EXPECT_NEAR(to_f(out[i]), acc, tol * std::max(1.0f, std::abs(acc)));
        }
    }
}

}  // anonymous namespace

NNOPS_TEST(linear_q4_1_weight) {
    const float tol = 2e-4f;
    // M = 1 is the decode shape and the fused kernel's home; 17 is past the f32
    // dot kernel's M limit (16), so it sweeps the packed plan instead. K = 70 and
    // N = 37 are off the 32-value block and the 16-wide panel.
    check_linear_q4<float>(1, 70, 37, false, EpilogueActivateType::None, tol, 31);
    check_linear_q4<float>(1, 70, 37, true, EpilogueActivateType::None, tol, 32);
    check_linear_q4<float>(17, 70, 37, true, EpilogueActivateType::Relu, tol, 33);
    check_linear_q4<float>(5, 33, 18, true, EpilogueActivateType::Relu6, tol, 34);
    check_linear_q4<float>(64, 96, 44, false, EpilogueActivateType::None, tol, 35);
}

NNOPS_TEST(linear_q4_1_weight_f16) {
    // f16 activations are the w4a16 decode case: the weight decodes into f32
    // registers and the output round-trips through f16 once.
    const float tol = test::kF16AccumTol;
    check_linear_q4<half>(1, 71, 33, false, EpilogueActivateType::None, tol, 41);
    check_linear_q4<half>(32, 71, 33, true, EpilogueActivateType::None, tol, 42);
    check_linear_q4<half>(33, 71, 33, true, EpilogueActivateType::Relu6, tol, 43);
    check_linear_q4<half>(8, 37, 70, true, EpilogueActivateType::None, tol, 44);
}

NNOPS_TEST(linear_q4_1_rank1_and_batch) {
    // A rank-1 X rides the same buffers as one row; a batched X takes MatMul's
    // per-batch offset over the same weight.
    const int64_t K = 40, N = 9;
    Q4Weight w = make_q4_weight(N, K, 51);
    const int64_t wshape[] = {N, K};
    TensorView wv(wshape, DataType::q4_1, w.bytes.data());

    std::vector<float> x(static_cast<size_t>(K));
    nnops::test::XorShift128 rng(52);
    rng.fill_float(x.data(), K, -1.0f, 1.0f);

    const int64_t xshape[] = {K};
    const int64_t oshape[] = {N};
    TensorView xv(xshape, DataType::f32, x.data());
    std::vector<float> out(static_cast<size_t>(N), 0.0f);
    TensorView ov(oshape, DataType::f32, out.data());

    auto op = Linear::create(Backend::CPU);
    const TensorView in_a[] = {xv, wv};
    NNOPS_EXPECT_EQ(op->compute(ov, in_a), Status::ok);
    for (int64_t n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            acc += x[static_cast<size_t>(k)] * w.decoded[static_cast<size_t>(n * K + k)];
        }
        NNOPS_EXPECT_NEAR(out[static_cast<size_t>(n)], acc, 2e-4f * std::max(1.0f, std::abs(acc)));
    }

    const int64_t B = 3;
    std::vector<float> xb(static_cast<size_t>(B * K));
    for (auto& v : xb) { v = rng.next_float(-1.0f, 1.0f); }
    const int64_t xbshape[] = {B, K};
    const int64_t obshape[] = {B, N};
    TensorView xbv(xbshape, DataType::f32, xb.data());
    std::vector<float> outb(static_cast<size_t>(B * N), 0.0f);
    TensorView obv(obshape, DataType::f32, outb.data());

    const TensorView in_b[] = {xbv, wv};
    NNOPS_EXPECT_EQ(op->compute(obv, in_b), Status::ok);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                acc += xb[static_cast<size_t>(b * K + k)] *
                       w.decoded[static_cast<size_t>(n * K + k)];
            }
            NNOPS_EXPECT_NEAR(outb[static_cast<size_t>(b * N + n)], acc,
                              2e-4f * std::max(1.0f, std::abs(acc)));
        }
    }
}

NNOPS_TEST(linear_q4_1_has_no_packed_blob) {
    // A q4_1 weight is already its own compact form; the query answers rank 0,
    // so a caller's prepack path falls back to the raw weight.
    Q4Weight w = make_q4_weight(8, 64, 61);
    const int64_t wshape[] = {8, 64};
    TensorView wv(wshape, DataType::q4_1, w.bytes.data());

    auto op = Linear::create(Backend::CPU);
    TensorView q;
    const TensorView wq[] = {wv};
    TensorView oq[] = {q};
    op->prepackWeights(wq, oq);
    NNOPS_EXPECT_TRUE(oq[0].is_empty());
    NNOPS_EXPECT_EQ(oq[0].rank(), int64_t(0));
}
