/// Unit tests for the Linear operator.
///
/// Linear is X × W^T + bias, driven by the MatMul kernel. Every case is checked
/// against `linear_ref` (naive triple loop) on the same operands:
///   - f32 / f16
///   - u8×s8 and s8×s8 with per-tensor, per-token and per-channel parameters
///   - optional bias, epilogue and add_to
///   - rank-1 and batched (rank-3) inputs

#include "nnops/ops/linear.hpp"
#include "nnops/core/quant_params.hpp"
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

/// Run Linear and linear_ref over the same inputs and require agreement.
/// @p initial optionally seeds the add_to residual in both outputs.
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

// ============================================================
// Hand-verified tests
// ============================================================

NNOPS_TEST(linear_basic) {
    // input: [2, 3] (M=2, K=3)
    // weight: [4, 3] (N=4, K=3)
    // output: [2, 4]
    const int64_t ishape[] = {2, 3};
    const int64_t wshape[] = {4, 3};

    float in_data[6] = {1, 2, 3, 4, 5, 6};
    float w_data[12] = {
        1, 0, 0,   // weight[0] * input[m] => dot({1,2,3}, {1,0,0}) = 1
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

    // Validate output descriptor
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
    auto out = test::make_planar(descs[0], out_buf.data());

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

    // Validate output descriptor
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
    auto out = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {input, weight, bias};
    op->compute(out, ins);

    // dot({1,2}, {1,0}) + 0.5 = 1 + 0.5 = 1.5
    NNOPS_EXPECT_NEAR(out_buf[0], 1.5f, 1e-6f);
    // dot({1,2}, {0,1}) - 0.5 = 2 - 0.5 = 1.5
    NNOPS_EXPECT_NEAR(out_buf[1], 1.5f, 1e-6f);
    // dot({1,2}, {1,1}) + 1.0 = 3 + 1.0 = 4.0
    NNOPS_EXPECT_NEAR(out_buf[2], 4.0f, 1e-6f);
}

// ============================================================
// fp32 / fp16 against the reference
// ============================================================

NNOPS_TEST(linear_f32_random) {
    const int64_t M = 7, K = 37, N = 19;   // odd sizes exercise the tails
    auto [xv, x] = test::make_random_tensor({M, K}, -1.0f, 1.0f, 1);
    auto [wv, w] = test::make_random_tensor({N, K}, -1.0f, 1.0f, 2);
    auto [bv, b] = test::make_random_tensor({N}, -0.5f, 0.5f, 3);

    check_linear<float>(LinearAttributes{}, {x, w, b});
    check_linear<float>(LinearAttributes{}, {x, w});
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

    // A non-identity epilogue with a residual cannot ride MatMul's beta, so
    // this drives the compact-temporary path.
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

// ============================================================
// Quantized: u8|s8 × s8 → fp
// ============================================================

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

    // Per-token activation parameters, per-channel weight parameters.
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
    // getOutputTensorDesc must resolve the quantized output to fp, following
    // output_dtype rather than the u8/s8 operand dtype.
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

    // Integer outputs are also resolved from output_dtype.
    for (DataType odt : {DataType::s32, DataType::s8, DataType::u8}) {
        LinearAttributes iattrs;
        iattrs.output_dtype = odt;
        auto op = Linear::create(iattrs, Backend::CPU);
        NNOPS_EXPECT_EQ(op->getOutputTensorDesc(desc_arr)[0].dtype, odt);
    }
}

NNOPS_TEST(linear_int8_output_matches_ref) {
    // Integer outputs: s32 (compensated accumulate), s8 and u8 (requantized),
    // each with an int32 [N] bias and Relu. Per-token X, per-channel W, and
    // K spanning several k-blocks — kernel against linear_ref, bit for bit.
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

// ============================================================
// The prepacked weight — Linear::prepackWeights
// ============================================================
//
// The weight packed once at load time is the same product as the raw one: this
// route removes the per-call RHS pack and nothing else, so on shapes that the
// GEMM tiles both ways the two outputs must agree to the bit — same panels,
// same k-blocks, same order. Below the small product's thresholds the raw
// weight takes the pack-free kernel instead (a prepacked B is panel-formatted
// and cannot), so there the two are compared as numbers.

namespace {

struct PrepackShape {
    int64_t M, K, N;
    bool bias, relu;
};

/// Run one shape twice — raw weight, then the same weight through the two-step
/// contract — and require the two outputs to agree (@p bitwise: the raw weight
/// takes the tiled GEMM too, so not a single bit may move).
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

    // The raw weight: the reference for this route.
    NNOPS_EXPECT_EQ(op->compute(og, ins, {}, nullptr), Status::ok);

    // ---- two-step contract: query the packed descriptor, allocate, fill
    TensorView q;
    const TensorView wq[] = {weight};
    TensorView oq[] = {q};
    op->prepackWeights(wq, oq);
    q = oq[0];
    // A query answer carries no storage (as Conv2D's does): the rank is what
    // separates it from "this weight has no packed form".
    NNOPS_EXPECT_EQ(q.rank(), int64_t(2));
    NNOPS_EXPECT_EQ(static_cast<int>(q.layout()),
                    static_cast<int>(TensorLayout::PackedB));
    // The descriptor leads with N — the dim shape inference reads — and its
    // numel is exactly the buffer a caller sizes (net_io's alloc does).
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
    // Shapes chosen for the pack's own seams: a partial n panel, a K that is
    // not a multiple of the k-block, N below one panel, and a deep K where A
    // gets packed too.
    struct Row { PrepackShape s; bool bitwise; };
    const Row cases[] = {
        // Odd on every side: both tails. Above the small-product threshold and
        // M > 4, so the raw weight takes the tiled GEMM too.
        {{  7,   37,  19, true,  false}, true},
        // N below one panel and one row: below the small-product threshold the
        // raw weight runs the pack-free kernel instead — a prepacked B is
        // panel-formatted and cannot — so only the numbers are compared.
        {{  1,    8,   5, false, false}, false},
        {{ 33,  300,  48, true,  true},  true},   // several k-blocks, Relu
        {{ 64,  256,  96, false, false}, true},   // a layer shape, no bias
        {{  5, 1024,  17, true,  false}, true},   // deep K: pack_a on, narrow N
    };

    const size_t ncases = sizeof(cases) / sizeof(cases[0]);
    for (size_t i = 0; i < ncases; ++i) {
        const uint64_t seed = 700 + 10 * static_cast<uint64_t>(i);
        check_prepacked_weight<float>(cases[i].s, seed, cases[i].bitwise);
        check_prepacked_weight<half>(cases[i].s, seed + 5, cases[i].bitwise);
    }
}

NNOPS_TEST(linear_prepacked_weight_wrong_k) {
    // The blob carries no K (its descriptor only knows N), so a caller who packs
    // against one K and computes against another must be refused rather than
    // read past its panels.
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

NNOPS_TEST(linear_prepacked_weight_quantized_has_no_blob) {
    // A quantized weight has no packed form (its panels would have to carry the
    // int32 column sums too), so the query answers with nothing — a graph
    // engine asks unconditionally.
    const int64_t wshape[] = {8, 16};
    std::vector<int8_t> wv(static_cast<size_t>(8 * 16), 3);
    TensorView weight(wshape, DataType::s8, wv.data());

    auto op = Linear::create(Backend::CPU);
    TensorView q;
    const TensorView wq[] = {weight};
    TensorView oq[] = {q};
    op->prepackWeights(wq, oq);
    NNOPS_EXPECT_TRUE(oq[0].is_empty());
    NNOPS_EXPECT_EQ(oq[0].rank(), int64_t(0));
}
