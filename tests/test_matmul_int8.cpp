/// @file test_matmul_int8.cpp
/// @brief Integer (s8|u8 × s8) MatMul — the tiled kernel against matmul_int8_ref,
/// plus exact-value cases that pin the zero-point compensation and requantizing
/// formula.
///
/// Everything here is integer, so every comparison is exact. u8 is not a second
/// kernel: x86 VNNI takes u8 A natively and shifts an s8 A (+128, the correction
/// folded out in the B panel), while aarch64's SDOT is s8×s8, so u8 A shifts
/// (-128) and the epilogue takes that back (int8_a_pack_bias /
/// int8_epilogue_offset). The u8 cases therefore pin the compensation, not a
/// separate code path.
///
/// Tiling is the shared fp plan; int8 only forks on its geometry: A is always
/// packed, kc is capped at 512, and an M-split hoists the B pack (num_slots == 1)
/// while still packing A per m-block, so the two slot counts diverge there.
/// Below the cap, kc follows the L2 the host reports — which is why the plan test
/// pins the rule on L2 values it chooses rather than on this host's kc.
///
/// Per-token / per-channel params are batch-independent (indexed by m and n), so
/// the rank-3 cases reuse the unbatched arrays; their buffers are still sized for
/// the batch, since a [B,M,K] view over a 2-D-sized buffer reads past it.

#include "nnops/ops/matmul.hpp"
#include "backend/cpu/matmul/matmul_helper.h"   // get_matmul_plan — the int8 path is plan-driven
#include "common/test_harness.hpp"
#include "common/cpu_pool.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"

#include <atomic>
#include <cstdint>
#include <thread>
#include <type_traits>
#include <vector>

using namespace nnops;

namespace nnops::backend::cpu::reference {
void matmul_int8_ref(const MatMulAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);
}

namespace {

QuantParams per_tensor(float scale, int32_t zp) {
    QuantParams qp;
    qp.granularity = QuantGranularity::PerTensor;
    qp.scale = scale;
    qp.zero_point = zp;
    return qp;
}

QuantParams per_token(const float* scale, const int32_t* zp, int64_t n) {
    QuantParams qp;
    qp.granularity = QuantGranularity::PerToken;
    qp.scale_data = scale;
    qp.zero_point_data = zp;
    qp.num_scales = n;
    return qp;
}

// [rows, cols] view with explicit quant params; make_q_nd is the any-rank form.
TensorView make_q(int64_t rows, int64_t cols, DataType dt, void* data, const QuantParams& qp) {
    const int64_t shape[] = {rows, cols};
    return TensorView(shape, dt, data, TensorLayout::NCHW, qp);
}

TensorView make_q_nd(std::span<const int64_t> shape, DataType dt, void* data, const QuantParams& qp) {
    return TensorView(shape, dt, data, TensorLayout::NCHW, qp);
}

// The shape getOutputTensorDesc resolved, as a span for make_q_nd.
std::span<const int64_t> desc_shape(const TensorDesc& d) {
    return std::span<const int64_t>(d.dims.data(), static_cast<size_t>(d.rank));
}

/// Tiled kernel and matmul_int8_ref over the same inputs; exact per element.
/// `out_qp` must reach both output views.
template <typename T>
void expect_matches_ref(const MatMulAttributes& attrs, const TensorDesc& odesc,
                        std::span<const TensorView> ins, QuantParams out_qp = {}) {
    constexpr DataType dt = std::is_same_v<T, int32_t> ? DataType::s32
                          : std::is_same_v<T, uint8_t> ? DataType::u8
                                                       : DataType::s8;
    std::vector<T> got(static_cast<size_t>(odesc.numel()));
    std::vector<T> want(static_cast<size_t>(odesc.numel()));
    TensorView output = make_q_nd(desc_shape(odesc), dt, got.data(), out_qp);
    TensorView ref_out = make_q_nd(desc_shape(odesc), dt, want.data(), out_qp);

    auto op = MatMul::create(attrs, Backend::CPU);
    op->compute(output, ins, {}, nullptr);
    nnops::backend::cpu::reference::matmul_int8_ref(attrs, ref_out, ins, {}, nullptr);

    for (size_t i = 0; i < got.size(); ++i) {
        NNOPS_EXPECT_EQ(static_cast<long>(got[i]), static_cast<long>(want[i]));
    }
}

}  // anonymous namespace

// ---- exact-value cases: zero-point compensation and requantizing ----

NNOPS_TEST(matmul_int8_basic_s32) {
    const int64_t ashape[] = {2, 3};
    const int64_t bshape[] = {3, 2};
    int8_t a_data[6] = {1, -2, 3, -4, 5, -6};   // A = [[1,-2,3],[-4,5,-6]]
    int8_t b_data[6] = {1, 2, 3, 4, 5, 6};       // B = [[1,2],[3,4],[5,6]]

    TensorView a(ashape, DataType::s8, a_data);
    TensorView b(bshape, DataType::s8, b_data);

    MatMulAttributes attrs{};
    attrs.output_dtype = DataType::s32;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::s32);
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(2));

    std::vector<int32_t> out_buf(descs[0].numel());
    auto output = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins, {}, nullptr);

    // C = A·B = [[10, 12], [-19, -24]]
    NNOPS_EXPECT_EQ(out_buf[0], 10);
    NNOPS_EXPECT_EQ(out_buf[1], 12);
    NNOPS_EXPECT_EQ(out_buf[2], -19);
    NNOPS_EXPECT_EQ(out_buf[3], -24);
}

NNOPS_TEST(matmul_int8_zeropoint_s32) {
    // Per-token: zp_a indexed by m, zp_w by n; the product is A @ W^T.
    const int64_t ashape[] = {2, 3};
    const int64_t wshape[] = {2, 3};
    int8_t a_data[6] = {1, 2, 3, 4, 5, 6};
    int8_t w_data[6] = {7, 8, 9, 10, 11, 12};

    float a_scale[2] = {1.0f, 1.0f};
    int32_t a_zp[2] = {1, 2};
    float w_scale[2] = {1.0f, 1.0f};
    int32_t w_zp[2] = {3, 4};

    TensorView a = make_q(2, 3, DataType::s8, a_data, per_token(a_scale, a_zp, 2));
    TensorView w = make_q(2, 3, DataType::s8, w_data, per_token(w_scale, w_zp, 2));

    MatMulAttributes attrs{};
    attrs.transpose_b = true;  // A @ W^T
    attrs.output_dtype = DataType::s32;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto w_desc = w.desc();
    const TensorDesc arr[] = {a_desc, w_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::s32);
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(2));

    std::vector<int32_t> out_buf(descs[0].numel());
    auto output = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, w};
    op->compute(output, ins, {}, nullptr);

    // MatMulInteger: Σ (qa−zp_a[m])·(qw−zp_w[n]) = [[17, 23], [47, 65]]
    NNOPS_EXPECT_EQ(out_buf[0], 17);
    NNOPS_EXPECT_EQ(out_buf[1], 23);
    NNOPS_EXPECT_EQ(out_buf[2], 47);
    NNOPS_EXPECT_EQ(out_buf[3], 65);
}

NNOPS_TEST(matmul_int8_u8_zeropoint_s32) {
    // u8 A: on x86 VNNI takes it as stored, on aarch64 the pack shifts it (-128)
    // and the epilogue takes that back. zp_a = 128 (the u8 midpoint) on row 0 and
    // a non-128 value on row 1 pin the compensation algebra, not the identity.
    const int64_t ashape[] = {2, 3};
    const int64_t bshape[] = {3, 2};
    uint8_t a_data[6] = {130, 128, 3, 200, 255, 0};   // A = [[130,128,3],[200,255,0]]
    int8_t  b_data[6] = {1, -4, 2, 5, 3, -6};         // B = [[1,-4],[2,5],[3,-6]]

    float   a_scale[2] = {1.0f, 1.0f};
    int32_t a_zp[2]    = {128, 100};
    TensorView a = make_q(2, 3, DataType::u8, a_data, per_token(a_scale, a_zp, 2));
    TensorView b = make_q(3, 2, DataType::s8, b_data, per_tensor(1.0f, 3));

    MatMulAttributes attrs{};
    attrs.output_dtype = DataType::s32;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::s32);
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(2));

    std::vector<int32_t> out_buf(descs[0].numel());
    auto output = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins, {}, nullptr);

    // Dezero-pointed A = [[2,0,-125],[100,155,-100]], B = [[-2,-7],[-1,2],[0,-9]]
    //   → out = [[-4, 1111], [-355, 510]]
    NNOPS_EXPECT_EQ(out_buf[0], -4);
    NNOPS_EXPECT_EQ(out_buf[1], 1111);
    NNOPS_EXPECT_EQ(out_buf[2], -355);
    NNOPS_EXPECT_EQ(out_buf[3], 510);
}

NNOPS_TEST(matmul_int8_requant_s8) {
    const int64_t ashape[] = {1, 2};
    const int64_t wshape[] = {2, 2};
    int8_t a_data[2] = {4, 6};
    int8_t w_data[4] = {1, 1, 1, 1};   // W = [[1,1],[1,1]]

    float a_scale[1] = {1.0f};
    int32_t a_zp[1] = {0};
    float w_scale[2] = {1.0f, 1.0f};
    int32_t w_zp[2] = {0, 0};

    TensorView a = make_q(1, 2, DataType::s8, a_data, per_token(a_scale, a_zp, 1));
    TensorView w = make_q(2, 2, DataType::s8, w_data, per_token(w_scale, w_zp, 2));

    // scale_out = 2, zp_out = 0 → q = round(acc·(1·1/2)) + 0.
    MatMulAttributes attrs{};
    attrs.transpose_b = true;
    attrs.output_dtype = DataType::s8;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto w_desc = w.desc();
    const TensorDesc arr[] = {a_desc, w_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::s8);

    std::vector<int8_t> out_buf(descs[0].numel());
    TensorView output = make_q(1, 2, DataType::s8, out_buf.data(), per_tensor(2.0f, 0));

    const TensorView ins[] = {a, w};
    op->compute(output, ins, {}, nullptr);

    // acc = dot({4,6},{1,1}) = 10 → requant = round(10/2) = 5 (both columns).
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[0]), 5);
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[1]), 5);
}

NNOPS_TEST(matmul_int8_requant_clamp_s8) {
    // scale_out = 1, so a dot product past 127 has to saturate.
    const int64_t ashape[] = {1, 2};
    const int64_t wshape[] = {2, 2};
    int8_t a_data[2] = {100, 100};
    int8_t w_data[4] = {100, 100, 100, 100};

    TensorView a(ashape, DataType::s8, a_data);
    TensorView w(wshape, DataType::s8, w_data);

    MatMulAttributes attrs{};
    attrs.transpose_b = true;
    attrs.output_dtype = DataType::s8;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto w_desc = w.desc();
    const TensorDesc arr[] = {a_desc, w_desc};
    auto descs = op->getOutputTensorDesc(arr);

    std::vector<int8_t> out_buf(descs[0].numel());
    TensorView output = make_q(1, 2, DataType::s8, out_buf.data(), per_tensor(1.0f, 0));

    const TensorView ins[] = {a, w};
    op->compute(output, ins, {}, nullptr);

    // acc = 100*100 + 100*100 = 20000 → clamps to 127.
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[0]), 127);
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[1]), 127);
}

// ---- random correctness against matmul_int8_ref ----

namespace {

// Operands for one GEMM; the buffers may hold more than a/b's 2-D extent
// (batch_a/batch_b).
struct I8Inputs {
    std::vector<int8_t> a_buf, b_buf;
    std::vector<float> a_scale, b_scale;
    std::vector<int32_t> a_zp, b_zp;
    TensorView a, b;
};

/// Operands for one GEMM, or `batch_a` × `batch_b` stacked along a leading batch
/// dim (the quant arrays are per-token / per-channel, so they are not batched).
/// `a_dt` picks A's fill: s8 spans ±127, u8 spans 0..255; bytes either way.
I8Inputs make_i8_inputs(int64_t M, int64_t K, int64_t N, bool transpose_b, uint64_t seed,
                        int64_t batch_a = 1, int64_t batch_b = 1,
                        DataType a_dt = DataType::s8) {
    const int64_t b_rows = transpose_b ? N : K;
    const int64_t b_cols = transpose_b ? K : N;

    nnops::test::XorShift128 rng(seed);
    auto fill_i8 = [&](std::vector<int8_t>& buf) {
        for (auto& v : buf) { v = static_cast<int8_t>(rng.next_u64() % 255 - 127); }
    };
    auto fill_a = [&](std::vector<int8_t>& buf) {
        for (auto& v : buf) {
            v = (a_dt == DataType::u8) ? static_cast<int8_t>(rng.next_u64() % 256)
                                       : static_cast<int8_t>(rng.next_u64() % 255 - 127);
        }
    };
    auto fill_zp = [&](std::vector<int32_t>& buf) {
        for (auto& v : buf) { v = static_cast<int32_t>(rng.next_u64() % 11 - 5); }
    };
    auto fill_scale = [&](std::vector<float>& buf) {
        for (auto& v : buf) { v = 0.5f + (rng.next_u64() % 1000) / 1000.0f; }
    };

    I8Inputs in;
    in.a_buf.resize(static_cast<size_t>(batch_a * M * K));
    in.b_buf.resize(static_cast<size_t>(batch_b * b_rows * b_cols));
    fill_a(in.a_buf);
    fill_i8(in.b_buf);

    in.a_scale.resize(static_cast<size_t>(M));
    in.b_scale.resize(static_cast<size_t>(N));
    in.a_zp.resize(static_cast<size_t>(M));
    in.b_zp.resize(static_cast<size_t>(N));
    fill_scale(in.a_scale);
    fill_scale(in.b_scale);
    fill_zp(in.a_zp);
    fill_zp(in.b_zp);

    in.a = make_q(M, K, a_dt, in.a_buf.data(),
                  per_token(in.a_scale.data(), in.a_zp.data(), M));
    in.b = make_q(b_rows, b_cols, DataType::s8, in.b_buf.data(),
                  per_token(in.b_scale.data(), in.b_zp.data(), N));
    return in;
}

}  // anonymous namespace

NNOPS_TEST(matmul_int8_random_s32) {
    for (bool tb : {false, true}) {
        const int64_t M = 13, K = 21, N = 17;
        I8Inputs in = make_i8_inputs(M, K, N, tb, 1234);

        MatMulAttributes attrs{};
        attrs.transpose_b = tb;
        attrs.output_dtype = DataType::s32;
        auto op = MatMul::create(attrs, Backend::CPU);

        auto a_desc = in.a.desc();
        auto b_desc = in.b.desc();
        const TensorDesc arr[] = {a_desc, b_desc};
        auto descs = op->getOutputTensorDesc(arr);

        std::vector<int32_t> out_buf(descs[0].numel());
        auto output = test::make_planar(descs[0], out_buf.data());
        const TensorView ins[] = {in.a, in.b};
        op->compute(output, ins, {}, nullptr);

        std::vector<int32_t> ref_buf(descs[0].numel());
        auto ref_out = test::make_planar(descs[0], ref_buf.data());
        nnops::backend::cpu::reference::matmul_int8_ref(attrs, ref_out, ins, {}, nullptr);

        for (size_t i = 0; i < out_buf.size(); ++i) {
            NNOPS_EXPECT_EQ(out_buf[i], ref_buf[i]);
        }
    }
}

NNOPS_TEST(matmul_int8_random_s8) {
    const int64_t M = 9, K = 17, N = 11;
    I8Inputs in = make_i8_inputs(M, K, N, /*transpose_b=*/true, 999);

    MatMulAttributes attrs{};
    attrs.transpose_b = true;
    attrs.output_dtype = DataType::s8;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = in.a.desc();
    auto b_desc = in.b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);

    const float out_scale = 0.25f;
    const int32_t out_zp = -7;
    std::vector<int8_t> out_buf(descs[0].numel());
    TensorView output = make_q(M, N, DataType::s8, out_buf.data(), per_tensor(out_scale, out_zp));

    const TensorView ins[] = {in.a, in.b};
    op->compute(output, ins, {}, nullptr);

    std::vector<int8_t> ref_buf(descs[0].numel());
    TensorView ref_out = make_q(M, N, DataType::s8, ref_buf.data(), per_tensor(out_scale, out_zp));
    nnops::backend::cpu::reference::matmul_int8_ref(attrs, ref_out, ins, {}, nullptr);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_EQ(static_cast<int>(out_buf[i]), static_cast<int>(ref_buf[i]));
    }
}

NNOPS_TEST(matmul_int8_u8_random_matches_ref) {
    // u8 activation against the reference; K = 1100 > KC_CAP_I8 spans several
    // k-blocks, so the accumulator is carried across them.
    for (bool tb : {false, true}) {
        const int64_t M = 14, K = 1100, N = 19;
        I8Inputs in = make_i8_inputs(M, K, N, tb, 8642, 1, 1, DataType::u8);

        MatMulAttributes attrs{};
        attrs.transpose_b = tb;
        attrs.output_dtype = DataType::s32;
        auto op = MatMul::create(attrs, Backend::CPU);

        const TensorDesc arr[] = {in.a.desc(), in.b.desc()};
        auto descs = op->getOutputTensorDesc(arr);
        NNOPS_EXPECT_EQ(descs[0].dtype, DataType::s32);

        const TensorView ins[] = {in.a, in.b};
        expect_matches_ref<int32_t>(attrs, descs[0], ins);
    }
}

NNOPS_TEST(matmul_int8_u8_requant_s8_matches_ref) {
    // The requantizing epilogue on the u8 path. 2048 is a power of two, so
    // dividing by it is exact: a mismatch can only be the accumulator.
    for (bool tb : {false, true}) {
        const int64_t M = 9, K = 17, N = 11;
        I8Inputs in = make_i8_inputs(M, K, N, tb, 777777, 1, 1, DataType::u8);

        MatMulAttributes attrs{};
        attrs.transpose_b = tb;
        attrs.output_dtype = DataType::s8;
        auto op = MatMul::create(attrs, Backend::CPU);

        const TensorDesc arr[] = {in.a.desc(), in.b.desc()};
        auto descs = op->getOutputTensorDesc(arr);
        NNOPS_EXPECT_EQ(descs[0].dtype, DataType::s8);

        const TensorView ins[] = {in.a, in.b};
        expect_matches_ref<int8_t>(attrs, descs[0], ins, per_tensor(2048.0f, -7));
    }
}

// ---- bias, saturating epilogues, multi-k-block and batching ----

NNOPS_TEST(matmul_int8_bias_s32) {
    // No zero points and no A shift left over: the int32 bias alone must turn the
    // output pass on, without sweeping either reduction.
    const int64_t ashape[] = {2, 3};
    const int64_t bshape[] = {3, 2};
    int8_t a_data[6] = {1, -2, 3, -4, 5, -6};
    int8_t b_data[6] = {1, 2, 3, 4, 5, 6};
    int32_t bias_data[2] = {-5, 7};

    TensorView a(ashape, DataType::s8, a_data);
    TensorView b(bshape, DataType::s8, b_data);
    const int64_t bias_shape[] = {2};
    TensorView bias(bias_shape, DataType::s32, bias_data);

    MatMulAttributes attrs{};
    attrs.output_dtype = DataType::s32;
    auto op = MatMul::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    std::vector<int32_t> out_buf(descs[0].numel());
    auto output = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b, bias};
    op->compute(output, ins, {}, nullptr);

    // C = [[10, 12], [-19, -24]] + bias [-5, 7]
    NNOPS_EXPECT_EQ(out_buf[0], 5);
    NNOPS_EXPECT_EQ(out_buf[1], 19);
    NNOPS_EXPECT_EQ(out_buf[2], -24);
    NNOPS_EXPECT_EQ(out_buf[3], -17);
}

NNOPS_TEST(matmul_int8_bias_requant_matches_ref) {
    // Per-token / per-channel zero points, K across several k-blocks, an int32
    // bias, Relu and both integer destinations, kernel against reference.
    for (DataType odt : {DataType::s8, DataType::u8}) {
        const int64_t M = 7, K = 1200, N = 13;
        I8Inputs in = make_i8_inputs(M, K, N, /*transpose_b=*/true, 5150, 1, 1,
                                     DataType::u8);

        std::vector<int32_t> bias_buf(static_cast<size_t>(N));
        nnops::test::XorShift128 rng(77);
        for (auto& v : bias_buf) {
            v = static_cast<int32_t>(rng.next_u64() % 20001) - 10000;
        }
        const int64_t bias_shape[] = {N};
        TensorView bias(bias_shape, DataType::s32, bias_buf.data());

        MatMulAttributes attrs{};
        attrs.transpose_b = true;
        attrs.output_dtype = odt;
        attrs.epilogue.type = EpilogueActivateType::Relu;
        auto op = MatMul::create(attrs, Backend::CPU);

        const TensorDesc arr[] = {in.a.desc(), in.b.desc()};
        auto descs = op->getOutputTensorDesc(arr);
        NNOPS_EXPECT_EQ(descs[0].dtype, odt);

        // 1024 is a power of two: the division is exact.
        const QuantParams oq =
            per_tensor(1024.0f, odt == DataType::u8 ? 128 : -7);
        const TensorView ins[] = {in.a, in.b, bias};
        if (odt == DataType::s8) {
            expect_matches_ref<int8_t>(attrs, descs[0], ins, oq);
        } else {
            expect_matches_ref<uint8_t>(attrs, descs[0], ins, oq);
        }
    }
}

NNOPS_TEST(matmul_int8_requant_clamp_u8) {
    // The u8 arm of the requant: the same saturating epilogue, clamped to
    // [0, 255] instead of [-128, 127]. acc = 20000 and acc = −100, zp_out = 10.
    const int64_t ashape[] = {2, 2};
    const int64_t wshape[] = {2, 2};
    int8_t a_data[4] = {100, 100, 1, -2};
    //  W = [[100,100],[100,100]] → row 0: 20000, row 1: 1·100 + (−2)·100 = −100
    int8_t w_data[4] = {100, 100, 100, 100};

    TensorView a(ashape, DataType::s8, a_data);
    TensorView w(wshape, DataType::s8, w_data);

    MatMulAttributes attrs{};
    attrs.transpose_b = true;
    attrs.output_dtype = DataType::u8;
    auto op = MatMul::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {a.desc(), w.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::u8);

    std::vector<uint8_t> out_buf(descs[0].numel());
    TensorView output = make_q(2, 2, DataType::u8, out_buf.data(), per_tensor(1.0f, 10));

    const TensorView ins[] = {a, w};
    op->compute(output, ins, {}, nullptr);

    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[0]), 255);   // 20000 + 10 clamps
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[1]), 255);
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[2]), 0);     // −100 + 10 clamps
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[3]), 0);
}

NNOPS_TEST(matmul_int8_multik_s32) {
    const int64_t M = 8, K = 1200, N = 6;   // K = 1200 > KC_CAP_I8 (512)
    I8Inputs in = make_i8_inputs(M, K, N, /*transpose_b=*/true, 4242);

    MatMulAttributes attrs{};
    attrs.transpose_b = true;
    attrs.output_dtype = DataType::s32;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = in.a.desc();
    auto b_desc = in.b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);

    std::vector<int32_t> out_buf(descs[0].numel());
    auto output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {in.a, in.b};
    op->compute(output, ins, {}, nullptr);

    std::vector<int32_t> ref_buf(descs[0].numel());
    auto ref_out = test::make_planar(descs[0], ref_buf.data());
    nnops::backend::cpu::reference::matmul_int8_ref(attrs, ref_out, ins, {}, nullptr);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_EQ(out_buf[i], ref_buf[i]);
    }
}

NNOPS_TEST(matmul_int8_batched_s32) {
    const int64_t B = 3, M = 7, K = 13, N = 5;
    I8Inputs in = make_i8_inputs(M, K, N, /*transpose_b=*/false, 31337, /*batch_a=*/B, /*batch_b=*/B);

    const int64_t ashape[] = {B, M, K};
    const int64_t bshape[] = {B, K, N};
    TensorView a = make_q_nd(ashape, DataType::s8, in.a_buf.data(),
                             per_token(in.a_scale.data(), in.a_zp.data(), M));
    TensorView b = make_q_nd(bshape, DataType::s8, in.b_buf.data(),
                             per_token(in.b_scale.data(), in.b_zp.data(), N));

    MatMulAttributes attrs{};
    attrs.output_dtype = DataType::s32;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(3));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(B));

    std::vector<int32_t> out_buf(descs[0].numel());
    auto output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {a, b};
    op->compute(output, ins, {}, nullptr);

    std::vector<int32_t> ref_buf(descs[0].numel());
    auto ref_out = test::make_planar(descs[0], ref_buf.data());
    nnops::backend::cpu::reference::matmul_int8_ref(attrs, ref_out, ins, {}, nullptr);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_EQ(out_buf[i], ref_buf[i]);
    }
}

NNOPS_TEST(matmul_int8_transpose_a_s32) {
    const int64_t M = 6, K = 15, N = 9;
    I8Inputs in = make_i8_inputs(M, K, N, /*transpose_b=*/false, 777);

    // The same buffer read as [K, M] instead of [M, K]; the per-token params
    // still index the output row m.
    const int64_t ashape[] = {K, M};
    TensorView a = make_q_nd(ashape, DataType::s8, in.a_buf.data(),
                             per_token(in.a_scale.data(), in.a_zp.data(), M));

    MatMulAttributes attrs{};
    attrs.transpose_a = true;
    attrs.output_dtype = DataType::s32;
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), in.b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dims[0], M);
    NNOPS_EXPECT_EQ(descs[0].dims[1], N);

    const TensorView ins[] = {a, in.b};
    expect_matches_ref<int32_t>(attrs, descs[0], ins);
}

NNOPS_TEST(matmul_int8_batched_broadcast_s32) {
    const int64_t B = 3, M = 5, K = 11, N = 4;
    // A batches over B; B stays 2-D and broadcasts against every batch element.
    I8Inputs in = make_i8_inputs(M, K, N, /*transpose_b=*/false, 5150, /*batch_a=*/B);

    const int64_t ashape[] = {B, M, K};
    TensorView a = make_q_nd(ashape, DataType::s8, in.a_buf.data(),
                             per_token(in.a_scale.data(), in.a_zp.data(), M));

    MatMulAttributes attrs{};
    attrs.output_dtype = DataType::s32;
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), in.b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(3));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(B));

    const TensorView ins[] = {a, in.b};
    expect_matches_ref<int32_t>(attrs, descs[0], ins);
}

NNOPS_TEST(matmul_int8_batched_s8) {
    const int64_t B = 2, M = 9, K = 40, N = 6;
    I8Inputs in = make_i8_inputs(M, K, N, /*transpose_b=*/false, 909, B, B);

    const int64_t ashape[] = {B, M, K};
    const int64_t bshape[] = {B, K, N};
    TensorView a = make_q_nd(ashape, DataType::s8, in.a_buf.data(),
                             per_token(in.a_scale.data(), in.a_zp.data(), M));
    TensorView b = make_q_nd(bshape, DataType::s8, in.b_buf.data(),
                             per_token(in.b_scale.data(), in.b_zp.data(), N));

    MatMulAttributes attrs{};
    attrs.output_dtype = DataType::s8;
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::s8);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(3));

    // The requantized path needs a per-tensor output scale / zero point.
    const TensorView ins[] = {a, b};
    expect_matches_ref<int8_t>(attrs, descs[0], ins, per_tensor(0.3f, 3));
}

NNOPS_TEST(matmul_int8_batched_u8_s32) {
    const int64_t B = 3, M = 5, K = 40, N = 7;
    I8Inputs in = make_i8_inputs(M, K, N, /*transpose_b=*/false, 5151, B, B,
                                 DataType::u8);

    const int64_t ashape[] = {B, M, K};
    const int64_t bshape[] = {B, K, N};
    TensorView a = make_q_nd(ashape, DataType::u8, in.a_buf.data(),
                             per_token(in.a_scale.data(), in.a_zp.data(), M));
    TensorView b = make_q_nd(bshape, DataType::s8, in.b_buf.data(),
                             per_token(in.b_scale.data(), in.b_zp.data(), N));

    MatMulAttributes attrs{};
    attrs.output_dtype = DataType::s32;
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::s32);
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(B));

    const TensorView ins[] = {a, b};
    expect_matches_ref<int32_t>(attrs, descs[0], ins);
}

NNOPS_TEST(matmul_int8_batched_nsplit_s32) {
    // N > M selects the N-split: blocks are n-slabs, not m-slabs. K = 1200 at
    // the 512 cap spans three k-blocks.
    const int64_t B = 2, M = 6, K = 1200, N = 70;
    I8Inputs in = make_i8_inputs(M, K, N, /*transpose_b=*/false, 616, B, B);

    const int64_t ashape[] = {B, M, K};
    const int64_t bshape[] = {B, K, N};
    TensorView a = make_q_nd(ashape, DataType::s8, in.a_buf.data(),
                             per_token(in.a_scale.data(), in.a_zp.data(), M));
    TensorView b = make_q_nd(bshape, DataType::s8, in.b_buf.data(),
                             per_token(in.b_scale.data(), in.b_zp.data(), N));

    MatMulAttributes attrs{};
    attrs.output_dtype = DataType::s32;
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dims[1], M);
    NNOPS_EXPECT_EQ(descs[0].dims[2], N);

    const TensorView ins[] = {a, b};
    expect_matches_ref<int32_t>(attrs, descs[0], ins);
}

NNOPS_TEST(matmul_int8_packed_a_bad_remainder_s32) {
    // The packed-A region is sized by num_panels_max(mc, MR_I8), not
    // num_panels(mc, MR_I8): the largest-first panel decomposition is not
    // monotonic in mc, so a smaller m-slab can need more panels than the tile's
    // own mc. These M values reach such slab heights through the public op.
    for (int64_t M : {1152, 2048, 2304, 2560, 2688, 3072, 4096}) {
        const int64_t K = 512, N = 256;   // K == KC_CAP_I8, one k-block at the cap
        I8Inputs in = make_i8_inputs(M, K, N, /*transpose_b=*/false, 900 + static_cast<uint64_t>(M));

        MatMulAttributes attrs{};
        attrs.output_dtype = DataType::s32;
        auto op = MatMul::create(attrs, Backend::CPU);

        auto a_desc = in.a.desc();
        auto b_desc = in.b.desc();
        const TensorDesc arr[] = {a_desc, b_desc};
        auto descs = op->getOutputTensorDesc(arr);

        std::vector<int32_t> out_buf(descs[0].numel());
        auto output = test::make_planar(descs[0], out_buf.data());
        const TensorView ins[] = {in.a, in.b};
        op->compute(output, ins, {}, nullptr);

        std::vector<int32_t> ref_buf(descs[0].numel());
        auto ref_out = test::make_planar(descs[0], ref_buf.data());
        nnops::backend::cpu::reference::matmul_int8_ref(attrs, ref_out, ins, {}, nullptr);

        for (size_t i = 0; i < out_buf.size(); ++i) {
            NNOPS_EXPECT_EQ(out_buf[i], ref_buf[i]);
        }
    }
}

// ---- plan: int8 shares the fp plan, so tiles shrink with the pool ----

namespace {

nnops::backend::cpu::MatMulPlan i8_plan(int64_t M, int64_t K, int64_t N, int nt) {
    std::vector<int8_t> ab(static_cast<size_t>(M * K), 1);
    std::vector<int8_t> bb(static_cast<size_t>(K * N), 2);
    const int64_t as[] = {M, K};
    const int64_t bs[] = {K, N};
    TensorView a(as, DataType::s8, ab.data());
    TensorView b(bs, DataType::s8, bb.data());
    MatMulAttributes attrs{};
    return nnops::backend::cpu::get_matmul_plan(attrs, a.desc(), b.desc(), nt);
}

}  // anonymous namespace

NNOPS_TEST(matmul_int8_plan_shrinks_tiles_with_threads) {
    namespace cpu = nnops::backend::cpu;

    // mc is bounded by MC_MAX (768) and the shape, then balanced on the M-split:
    // the block count is rounded up to a multiple of the pool and the tile is
    // that even share, rounded up to MC_ALIGN (24). It is the plan's thread
    // awareness that makes the block count grow with the pool.
    const auto p1  = i8_plan(256, 256, 256, 1);
    const auto p10 = i8_plan(256, 256, 256, 10);
    NNOPS_EXPECT_TRUE(!p10.split_n);                     // M >= N → split on M
    NNOPS_EXPECT_EQ(p1.mc, int64_t{240});                // nt = 1: the M cap, snapped to 24
    NNOPS_EXPECT_EQ(cpu::split_block_count(256, p1.mc), int64_t{2});
    NNOPS_EXPECT_EQ(p10.mc, int64_t{48});                // ceil24(256 / 10) → 6 blocks
    NNOPS_EXPECT_EQ(cpu::split_block_count(256, p10.mc), int64_t{6});

    // At one thread mc is the 768 ceiling, so the pool is what moves the block
    // count.
    const auto b1  = i8_plan(1024, 512, 1024, 1);
    const auto b10 = i8_plan(1024, 512, 1024, 10);
    NNOPS_EXPECT_EQ(cpu::split_block_count(1024, b1.mc), int64_t{2});   // mc = 768
    NNOPS_EXPECT_EQ(b10.mc, int64_t{120});               // ceil24(1024 / 10)
    NNOPS_EXPECT_EQ(cpu::split_block_count(1024, b10.mc), int64_t{9});

    // int8 has no direct-A route, so A is packed however narrow its row stride.
    NNOPS_EXPECT_TRUE(b10.pack_a);

    // M-split: int8 hoists the B pack, so num_slots is 1 (one full-N slice)
    // while A is still packed per m-block — the one place num_slots and
    // num_slots_a disagree, which is why the dispatch indexes A by num_slots_a.
    //
    // kc is the one tile size that follows the host: a per-dtype cap, and below
    // it the L2 residency budget. A literal would assert this host's cache
    // instead of the rule, so the rule is checked on an L2 the test chooses. The
    // plan's own kc is only bounded here; everything sized from it is checked
    // against its own value.
    {
        int64_t mc = 0, nc = 0, kc = 0;
        // K = 4096: every other limit — K itself, the residency budget (5461 at
        // this L2) and the L1 prefetch bound — is above the 512 cap, so the cap
        // is what comes back.
        cpu::choose_matmul_tile(1024, 1024, 4096, cpu::MR_MAX_I8, cpu::NR_MAX_I8,
                                1, 1, /*split_n=*/false, cpu::KC_CAP_I8,
                                /*l1=*/65536, size_t{16} << 20, mc, nc, kc);
        NNOPS_EXPECT_EQ(kc, cpu::KC_CAP_I8);         // 16 MiB: the cap decides
        cpu::choose_matmul_tile(1024, 1024, 4096, cpu::MR_MAX_I8, cpu::NR_MAX_I8,
                                1, 10, /*split_n=*/false, cpu::KC_CAP_I8,
                                /*l1=*/65536, size_t{2} << 20, mc, nc, kc);
        NNOPS_EXPECT_TRUE(kc < cpu::KC_CAP_I8);      // 2 MiB / 10 threads: neither cap
        NNOPS_EXPECT_TRUE(kc > cpu::KC_MIN);         // nor floor — the budget decides
    }
    NNOPS_EXPECT_TRUE(b1.kc <= cpu::KC_CAP_I8);      // and K = 512 is the cap
    NNOPS_EXPECT_TRUE(b10.kc <= cpu::KC_CAP_I8);
    NNOPS_EXPECT_EQ(b1.kc % cpu::KC_ALIGN, int64_t{0});
    NNOPS_EXPECT_EQ(b10.kc % cpu::KC_ALIGN, int64_t{0});
    // B panels are uniform nr_max-wide slots (correction block included), so a
    // slice is ceil(N / nr_max) of them.
    const int64_t ldd_b1  = cpu::int8_panel_stride(b1.kc);
    const int64_t ldd_b10 = cpu::int8_panel_stride(b10.kc);
    const int64_t np_n    = (int64_t{1024} + cpu::NR_MAX_I8 - 1) / cpu::NR_MAX_I8;
    const int64_t one_slice1  = np_n * ldd_b1;
    const int64_t one_slice10 = np_n * ldd_b10;
    NNOPS_EXPECT_EQ(b1.ldd_b, ldd_b1);
    NNOPS_EXPECT_EQ(b10.ldd_b, ldd_b10);
    NNOPS_EXPECT_EQ(b1.num_slots, int64_t{1});
    NNOPS_EXPECT_EQ(b10.num_slots, int64_t{1});
    NNOPS_EXPECT_EQ(b1.pack_a_offset, one_slice1);
    NNOPS_EXPECT_EQ(b10.pack_a_offset, one_slice10);

    // nt = 1 gets one A slot whatever the block count, so b1's A region is one
    // tile: workspace_size is pack_a_offset plus exactly that. i8_plan leaves
    // use_thread_slots false, so b10 falls back to one slot per m-block even at
    // 10 threads — the thread count alone does not shrink the A region.
    const int64_t b1_ldd_a = static_cast<int64_t>(cpu::mma_pack_align(cpu::MR_MAX_I8 * b1.kc));
    const int64_t b1_np_a  = cpu::num_panels_max(static_cast<int>(b1.mc), cpu::MR_I8);
    NNOPS_EXPECT_EQ(b1.num_slots_a, int64_t{1});
    NNOPS_EXPECT_EQ(b1.ldd_a, b1_ldd_a);
    NNOPS_EXPECT_EQ(b1.np_a, b1_np_a);
    NNOPS_EXPECT_EQ(b1.workspace_size, one_slice1 + b1_np_a * b1_ldd_a);

    const int64_t b10_blocks = cpu::split_block_count(int64_t{1024}, b10.mc);
    const int64_t b10_np_a   = cpu::num_panels_max(static_cast<int>(b10.mc), cpu::MR_I8);
    NNOPS_EXPECT_EQ(b10.num_slots_a, b10_blocks);
    NNOPS_EXPECT_EQ(b10.workspace_size, one_slice10 + b10_blocks * b10_np_a * b10.ldd_a);

    // N-split (N > M): each n-block owns one slice. Uniform panels make the count
    // monotonic, so ceil(nc / nr_max) is exact even for the short last block.
    const auto n10 = i8_plan(64, 512, 1024, 10);
    NNOPS_EXPECT_TRUE(n10.split_n);
    NNOPS_EXPECT_EQ(n10.nc % cpu::NR_MAX_I8, int64_t{0});
    const int64_t n_blocks = (int64_t{1024} + n10.nc - 1) / n10.nc;
    const int64_t np_max = (n10.nc + cpu::NR_MAX_I8 - 1) / cpu::NR_MAX_I8;
    NNOPS_EXPECT_EQ(n10.ldd_b, cpu::int8_panel_stride(n10.kc));
    NNOPS_EXPECT_EQ(n10.num_slots, n_blocks);
    NNOPS_EXPECT_EQ(n10.pack_a_offset, n_blocks * np_max * n10.ldd_b);
    NNOPS_EXPECT_EQ(n10.num_slots_a, n_blocks);   // per-block: no thread ids here
    NNOPS_EXPECT_EQ(n10.workspace_size,
                    n10.pack_a_offset
                        + n_blocks * cpu::num_panels_max(static_cast<int>(n10.mc), cpu::MR_I8)
                          * n10.ldd_a);
}

// ---- threaded correctness: the block decomposition moves with the pool ----

namespace {

// Minimal fixed-size pool exposing a CpuBackend (as in test_matmul.cpp): an
// atomic counter hands out [begin, end), one item per worker take.
struct SimplePool {
    explicit SimplePool(int nthreads) : nthreads_(nthreads) {
        // Use the build's OpenMP adapter when there is one (common/cpu_pool.hpp).
        if (nnops::test::bench_use_openmp()) {
            cpu = nnops::openmp_backend(nthreads);
            return;
        }
        cpu.parallel_for = [this](int64_t begin, int64_t end, const ParallelForBody& body) {
            this->parallel_for(begin, end, body);
        };
        cpu.num_threads = [this]() { return nthreads_; };
        cpu.thread_id = []() { return current_thread_id_; };
    }

    void parallel_for(int64_t begin, int64_t end, const ParallelForBody& body) {
        std::atomic<int64_t> next{begin};
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(nthreads_));
        for (int t = 0; t < nthreads_; ++t) {
            workers.emplace_back([&, t]() {
                current_thread_id_ = t;
                for (;;) {
                    int64_t i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= end) { break; }
                    body(i);
                }
            });
        }
        for (auto& w : workers) { w.join(); }
    }

    int nthreads_;
    CpuBackend cpu;
    inline static thread_local int current_thread_id_ = 0;
};

/// Run one s8×s8 GEMM through the public op with `nt` workers, exact against
/// matmul_int8_ref. `ta`/`tb` are separate loop nests (the reductions run along
/// a different axis; A is physical K×M, B physical N×K), so each needs its case.
template <typename T>
void expect_i8_threaded_matches_ref(int64_t M, int64_t K, int64_t N, int nt, uint64_t seed,
                                    bool ta = false, bool tb = false,
                                    DataType a_dt = DataType::s8) {
    I8Inputs in = make_i8_inputs(M, K, N, tb, seed, 1, 1, a_dt);

    // make_i8_inputs lays A out M×K; transpose_a needs a K×M operand over its own
    // buffer. Scales stay per logical token (M of them) — the kernel indexes qa[m].
    std::vector<int8_t> a_t_buf;
    std::vector<float> a_t_scale;
    std::vector<int32_t> a_t_zp;
    if (ta) {
        nnops::test::XorShift128 rng(seed ^ 0x5eedULL);
        a_t_buf.resize(static_cast<size_t>(K * M));
        for (auto& v : a_t_buf) {
            v = (a_dt == DataType::u8) ? static_cast<int8_t>(rng.next_u64() % 256)
                                       : static_cast<int8_t>(rng.next_u64() % 255 - 127);
        }
        a_t_scale.resize(static_cast<size_t>(M));
        a_t_zp.resize(static_cast<size_t>(M));
        for (auto& v : a_t_scale) v = 0.5f + (rng.next_u64() % 1000) / 1000.0f;
        for (auto& v : a_t_zp) v = static_cast<int32_t>(rng.next_u64() % 11 - 5);
    }
    const TensorView a_view =
        ta ? make_q(K, M, a_dt, a_t_buf.data(),
                    per_token(a_t_scale.data(), a_t_zp.data(), M))
           : in.a;

    constexpr DataType odt = std::is_same_v<T, int32_t> ? DataType::s32 : DataType::s8;
    MatMulAttributes attrs{};
    attrs.output_dtype = odt;
    attrs.transpose_a = ta;
    attrs.transpose_b = tb;
    auto op = MatMul::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {a_view.desc(), in.b.desc()};
    auto descs = op->getOutputTensorDesc(arr);

    const QuantParams oq = std::is_same_v<T, int32_t> ? QuantParams{} : per_tensor(0.25f, -7);
    std::vector<T> got(static_cast<size_t>(descs[0].numel()));
    std::vector<T> want(got.size());
    TensorView output  = make_q(descs[0].dims[0], descs[0].dims[1], odt, got.data(), oq);
    TensorView ref_out = make_q(descs[0].dims[0], descs[0].dims[1], odt, want.data(), oq);

    SimplePool pool(nt);
    ComputeContext ctx;
    ctx.cpu = pool.cpu;

    const TensorView ins[] = {a_view, in.b};
    op->compute(output, ins, ctx, nullptr);
    nnops::backend::cpu::reference::matmul_int8_ref(attrs, ref_out, ins, {}, nullptr);

    for (size_t i = 0; i < got.size(); ++i) {
        NNOPS_EXPECT_EQ(static_cast<int>(got[i]), static_cast<int>(want[i]));
    }
}

}  // anonymous namespace

NNOPS_TEST(matmul_int8_threaded_matches_ref) {
    // A pool shrinks mc (M-split, once M/nt drops below MC_MAX or the L2 budget)
    // and nc (N-split), so the decomposition differs per thread count: both
    // splits plus the s8 accumulator and requant must still match exactly.
    for (int nt : {2, 4, 8}) {
        expect_i8_threaded_matches_ref<int32_t>(512, 512, 128, nt, 11);   // M-split
        expect_i8_threaded_matches_ref<int32_t>(128, 512, 512, nt, 12);   // N-split
        expect_i8_threaded_matches_ref<int32_t>(256, 256, 256, nt, 13);   // mc < MC_MAX
        expect_i8_threaded_matches_ref<int32_t>(1024, 512, 1024, nt, 14); // the plan test's shape
    }
    expect_i8_threaded_matches_ref<int8_t>(512, 512, 128, 4, 15);
    expect_i8_threaded_matches_ref<int8_t>(128, 512, 512, 4, 16);

    // All four transpose combinations, each a separate loop nest.
    for (int nt : {2, 4, 8}) {
        expect_i8_threaded_matches_ref<int32_t>(512, 512, 128, nt, 21, /*ta=*/true);
        expect_i8_threaded_matches_ref<int32_t>(512, 512, 128, nt, 22, /*ta=*/false, /*tb=*/true);
        expect_i8_threaded_matches_ref<int32_t>(512, 512, 128, nt, 23, /*ta=*/true,  /*tb=*/true);
        // Wide-N (N-split) with both transposes, where nc is also shrunk.
        expect_i8_threaded_matches_ref<int32_t>(128, 512, 512, nt, 24, /*ta=*/true,  /*tb=*/true);
        expect_i8_threaded_matches_ref<int8_t>(256, 256, 256, nt, 25, /*ta=*/true,  /*tb=*/true);
    }

    // u8 through a pool: the dtype path is separate from the shape paths, so
    // both splits and a transposed reduction get an unsigned case.
    for (int nt : {2, 4}) {
        expect_i8_threaded_matches_ref<int32_t>(256, 256, 256, nt, 31, /*ta=*/false, /*tb=*/false,
                                                DataType::u8);
        expect_i8_threaded_matches_ref<int32_t>(128, 512, 512, nt, 32, /*ta=*/false, /*tb=*/true,
                                                DataType::u8);
        expect_i8_threaded_matches_ref<int32_t>(512, 512, 128, nt, 33, /*ta=*/true, /*tb=*/false,
                                                DataType::u8);
    }
    expect_i8_threaded_matches_ref<int8_t>(256, 256, 256, 4, 34, /*ta=*/true, /*tb=*/true,
                                           DataType::u8);
}

NNOPS_TEST(matmul_int8_per_tensor_weight_zp_matches_ref) {
    // A per-tensor (non-zero) zp_w is the one shape that folds the row half of
    // the compensation into one int32 per output row (r_fold); with symmetric A
    // that fold is the whole compensation, and a per-token A zero point leaves
    // the column half swept. Both destinations are checked against the reference.
    for (int symmetric_a : {1, 0}) {
        const int64_t M = 15, K = 23, N = 12;
        I8Inputs in = make_i8_inputs(M, K, N, /*transpose_b=*/true, 4711 + 2 * symmetric_a);
        TensorView a = make_q(M, K, DataType::s8, in.a_buf.data(),
                              symmetric_a
                                  ? per_tensor(1.0f, 0)
                                  : per_token(in.a_scale.data(), in.a_zp.data(), M));
        TensorView b = make_q(N, K, DataType::s8, in.b_buf.data(), per_tensor(1.0f, -3));
        const TensorView ins[] = {a, b};

        for (DataType odt : {DataType::s32, DataType::s8}) {
            MatMulAttributes attrs{};
            attrs.transpose_b = true;
            attrs.output_dtype = odt;
            auto op = MatMul::create(attrs, Backend::CPU);
            const TensorDesc arr[] = {a.desc(), b.desc()};
            auto descs = op->getOutputTensorDesc(arr);
            if (odt == DataType::s32) {
                expect_matches_ref<int32_t>(attrs, descs[0], ins);
            } else {
                expect_matches_ref<int8_t>(attrs, descs[0], ins, per_tensor(0.125f, 3));
            }
        }
    }
}
