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
