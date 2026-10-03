/// Unit tests for integer (s8×s8) MatMul — per-token activation × per-channel
/// weight, s32 (MatMulInteger) or requantized s8 output.
///
/// The tiled kernel (packed VNNI / SDOT / scalar path) is compared against
/// matmul_int8_ref, plus a few exact-value cases that pin the zero-point
/// compensation and requantization formulas.

#include "nnops/ops/matmul.hpp"
#include "backend/cpu/matmul_helper.h"   // get_matmul_plan — the int8 path is plan-driven
#include "common/test_harness.hpp"
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

// ---- quant-param builders ----
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

// Build a 2D TensorView with explicit quant params.
TensorView make_q(int64_t rows, int64_t cols, DataType dt, void* data, const QuantParams& qp) {
    const int64_t shape[] = {rows, cols};
    return TensorView(shape, dt, data, TensorLayout::NCHW, qp);
}

// Same, for any rank (the batched cases).
TensorView make_q_nd(std::span<const int64_t> shape, DataType dt, void* data, const QuantParams& qp) {
    return TensorView(shape, dt, data, TensorLayout::NCHW, qp);
}

// The output shape op->getOutputTensorDesc() resolved, as a span for make_q_nd.
std::span<const int64_t> desc_shape(const TensorDesc& d) {
    return std::span<const int64_t>(d.dims.data(), static_cast<size_t>(d.rank));
}

/// Run the tiled kernel and matmul_int8_ref over the same inputs and require
/// exact agreement on every element. `out_qp` is only read for an s8 output.
template <typename T>
void expect_matches_ref(const MatMulAttributes& attrs, const TensorDesc& odesc,
                        std::span<const TensorView> ins, QuantParams out_qp = {}) {
    const DataType dt = std::is_same_v<T, int32_t> ? DataType::s32 : DataType::s8;
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

// ============================================================
// Basic s8×s8 → s32 (no zero-points, per-tensor identity)
// ============================================================

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

    // C[0] = dot({1,-2,3},{1,3,5}) = 10, C[1] = dot({1,-2,3},{2,4,6}) = 12
    // C[2] = dot({-4,5,-6},{1,3,5}) = -19, C[3] = dot({-4,5,-6},{2,4,6}) = -24
    NNOPS_EXPECT_EQ(out_buf[0], 10);
    NNOPS_EXPECT_EQ(out_buf[1], 12);
    NNOPS_EXPECT_EQ(out_buf[2], -19);
    NNOPS_EXPECT_EQ(out_buf[3], -24);
}

// ============================================================
// Zero-point compensation → s32 (per-token A, per-channel W via A@W^T)
// ============================================================

NNOPS_TEST(matmul_int8_zeropoint_s32) {
    // A: [2, 3] per-token (zp_a[m]); W: [2, 3] per-token (zp_w[n]); A @ W^T.
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

    // MatMulInteger: Σ (qa−zp_a[m])·(qw−zp_w[n])
    //   out[0,0] = dot({0,1,2},{4,5,6}) = 17, out[0,1] = dot({0,1,2},{6,7,8}) = 23
    //   out[1,0] = dot({2,3,4},{4,5,6}) = 47, out[1,1] = dot({2,3,4},{6,7,8}) = 65
    NNOPS_EXPECT_EQ(out_buf[0], 17);
    NNOPS_EXPECT_EQ(out_buf[1], 23);
    NNOPS_EXPECT_EQ(out_buf[2], 47);
    NNOPS_EXPECT_EQ(out_buf[3], 65);
}

// ============================================================
// Requantized s8 output
// ============================================================

NNOPS_TEST(matmul_int8_requant_s8) {
    // A: [1, 2] per-token, W: [2, 2] per-token, A @ W^T.
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

    // Output: scale_out = 2, zp_out = 0 → q = round(acc·(1·1/2)) + 0.
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
    // Same inputs, scale_out = 0.25 → acc=10 → round(40) = 40 (no clamp), and a
    // larger dot product that clamps to 127.
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

// ============================================================
// Random correctness vs reference (both transpose variants, s32 + s8)
// ============================================================

namespace {

// Random int8 tensors with per-token quant params over [lo, hi). `a`/`b` are
// the 2-D views; the buffer behind them may hold more (see batch_a/batch_b).
struct I8Inputs {
    std::vector<int8_t> a_buf, b_buf;
    std::vector<float> a_scale, b_scale;
    std::vector<int32_t> a_zp, b_zp;
    TensorView a, b;
};

/// Build the operands for one GEMM, or for `batch_a` × `batch_b` independent
/// GEMMs stacked along a leading batch dim. The buffers are sized for the batch
/// so an [B, M, K] / [B, K, N] view over them stays in bounds — a rank-3 view
/// over a 2-D-sized buffer reads past the allocation. Weights and activations
/// batch separately because the N-D cases broadcast one against the other.
I8Inputs make_i8_inputs(int64_t M, int64_t K, int64_t N, bool transpose_b, uint64_t seed,
                        int64_t batch_a = 1, int64_t batch_b = 1) {
    const int64_t b_rows = transpose_b ? N : K;
    const int64_t b_cols = transpose_b ? K : N;

    nnops::test::XorShift128 rng(seed);
    auto fill_i8 = [&](std::vector<int8_t>& buf) {
        for (auto& v : buf) { v = static_cast<int8_t>(rng.next_u64() % 255 - 127); }
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
    fill_i8(in.a_buf);
    fill_i8(in.b_buf);

    in.a_scale.resize(static_cast<size_t>(M));
    in.b_scale.resize(static_cast<size_t>(N));
    in.a_zp.resize(static_cast<size_t>(M));
    in.b_zp.resize(static_cast<size_t>(N));
    fill_scale(in.a_scale);
    fill_scale(in.b_scale);
    fill_zp(in.a_zp);
    fill_zp(in.b_zp);

    in.a = make_q(M, K, DataType::s8, in.a_buf.data(),
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

// ============================================================
// Multi-k-block (K > KC_I8) and batched
// ============================================================

NNOPS_TEST(matmul_int8_multik_s32) {
    const int64_t M = 8, K = 1200, N = 6;   // K = 1200 > KC_I8 (512)
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
    // Per-token / per-channel params are batch-independent (indexed by m and n),
    // so every batch element shares the same scales and zero points.
    I8Inputs in = make_i8_inputs(M, K, N, /*transpose_b=*/false, 31337, /*batch_a=*/B, /*batch_b=*/B);

    // A is [B, M, K], B is [B, K, N].
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

// ============================================================
// transpose_a, batch broadcast, batched s8 and the batched N-split
// ============================================================

NNOPS_TEST(matmul_int8_transpose_a_s32) {
    const int64_t M = 6, K = 15, N = 9;
    I8Inputs in = make_i8_inputs(M, K, N, /*transpose_b=*/false, 777);

    // The same bytes in.a reads as [M, K], read instead as [K, M] — A^T. The
    // per-token params still index the output row m.
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

NNOPS_TEST(matmul_int8_batched_nsplit_s32) {
    // N > M selects the N-split (one n-block per task) rather than the M-split,
    // and K = 1200 spans three k-blocks (KC_I8 = 512).
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
    // Regression: the on-stack packed-A tile is sized by the panel count of
    // MC_TARGET itself (mc = 144 -> 18 panels of mr = 8), but the greedy
    // {8,4,1} decomposition the pack actually emits is NOT monotonic in mc —
    // a *smaller* tile needs more panels (142 -> 20, 143 -> 21). An m-slab
    // landing on one of those wrote up to 3 panels (12 KB) past the buffer.
    //
    // Reached through the public op: an M-split whose slab height is ~M /
    // ceil(M / 144), with K == KC_I8 so the pack uses the full-Kc panel stride
    // the buffer is sized at. Slab heights of 142/143 appear for many M, which
    // is why this is worth a regression case rather than a comment.
    for (int64_t M : {1152, 2048, 2304, 2560, 2688, 3072, 4096}) {
        const int64_t K = 512, N = 256;   // K == KC_I8 -> full-Kc stride
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

// ============================================================
// Plan — int8 shares the fp plan, so the tiles shrink with the pool
// ============================================================

namespace {

// The plan for an s8×s8 GEMM of the given shape at `nt` worker threads.
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

    // int8 used to resolve its own tiles with no thread awareness (mc was always
    // min(MC_TARGET, M)), so a small M produced two blocks however big the pool
    // was. It now goes through get_matmul_plan, whose thread-aware mc/nc shrink
    // is what makes the block count grow with the pool.
    const auto p1  = i8_plan(256, 256, 256, 1);
    const auto p10 = i8_plan(256, 256, 256, 10);
    NNOPS_EXPECT_TRUE(!p10.split_n);                     // M >= N → split on M
    NNOPS_EXPECT_EQ(p1.mc, cpu::MC_TARGET);              // 256 > MC_TARGET → clamped
    NNOPS_EXPECT_EQ(cpu::split_block_count(256, p1.mc), int64_t{2});
    NNOPS_EXPECT_EQ(p10.mc, int64_t{25});                // 256 / 10
    NNOPS_EXPECT_EQ(cpu::split_block_count(256, p10.mc), int64_t{11});

    // The shape the int8 scaling was measured on: at one thread mc is already
    // MC_TARGET, so the pool is what moves the block count.
    const auto b1  = i8_plan(1024, 512, 1024, 1);
    const auto b10 = i8_plan(1024, 512, 1024, 10);
    NNOPS_EXPECT_EQ(cpu::split_block_count(1024, b1.mc), int64_t{8});
    NNOPS_EXPECT_EQ(b10.mc, int64_t{102});
    NNOPS_EXPECT_EQ(cpu::split_block_count(1024, b10.mc), int64_t{11});

    // int8 has no direct-A route, so A is packed however narrow its row stride.
    NNOPS_EXPECT_TRUE(b10.pack_a);

    // M-split: the B pack is hoisted out of the parallel region, so one slice
    // serves every m-block and the workspace is one full-N slice either way.
    const int64_t ldd_b = cpu::align_up<PANEL_ALIGN_BYTES>(cpu::NR_MAX_I8 * 512);
    const int64_t one_slice = cpu::num_panels4(1024, cpu::NR_I8) * ldd_b;
    NNOPS_EXPECT_EQ(b10.kc, int64_t{512});               // min(KC_I8, K)
    NNOPS_EXPECT_EQ(b10.ldd_b, ldd_b);
    NNOPS_EXPECT_EQ(b1.num_slots, int64_t{1});
    NNOPS_EXPECT_EQ(b10.num_slots, int64_t{1});
    NNOPS_EXPECT_EQ(b1.workspace_size, one_slice);
    NNOPS_EXPECT_EQ(b10.workspace_size, one_slice);

    // N-split (N > M): the n-blocks tile one full-N slice, so the workspace is
    // that slice however many blocks there are.
    const auto n10 = i8_plan(64, 512, 1024, 10);
    NNOPS_EXPECT_TRUE(n10.split_n);
    NNOPS_EXPECT_EQ(n10.nc % cpu::NR_MAX_I8, int64_t{0});
    NNOPS_EXPECT_EQ(n10.num_slots, (int64_t{1024} + n10.nc - 1) / n10.nc);
    NNOPS_EXPECT_EQ(n10.workspace_size, one_slice);
}

// ============================================================
// Threaded correctness — the block decomposition moves with the pool
// ============================================================

namespace {

// Minimal fixed-size thread pool exposing a CpuBackend (as in test_matmul.cpp):
// items in [begin, end) are claimed via an atomic counter, each worker runs
// body(i) until the range is exhausted.
struct SimplePool {
    explicit SimplePool(int nthreads) : nthreads_(nthreads) {
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

/// Run one s8×s8 GEMM through the public op with `nt` workers and require exact
/// agreement with matmul_int8_ref — the int8 path is integer, so agreement is
/// element-exact, not approximate.
///
/// `ta`/`tb` select the transposed operand layouts, which change which axis each
/// pre-epilogue reduction runs along (A physical K×M under transpose_a, B
/// physical N×K under transpose_b) — a different loop nest per combination, so
/// each needs its own pooled coverage.
template <typename T>
void expect_i8_threaded_matches_ref(int64_t M, int64_t K, int64_t N, int nt, uint64_t seed,
                                    bool ta = false, bool tb = false) {
    I8Inputs in = make_i8_inputs(M, K, N, tb, seed);

    // make_i8_inputs always lays A out as M×K; for transpose_a build a K×M
    // operand over its own buffer. Scales stay per logical token (M of them),
    // matching how the kernel indexes qa[m].
    std::vector<int8_t> a_t_buf;
    std::vector<float> a_t_scale;
    std::vector<int32_t> a_t_zp;
    if (ta) {
        nnops::test::XorShift128 rng(seed ^ 0x5eedULL);
        a_t_buf.resize(static_cast<size_t>(K * M));
        for (auto& v : a_t_buf) v = static_cast<int8_t>(rng.next_u64() % 255 - 127);
        a_t_scale.resize(static_cast<size_t>(M));
        a_t_zp.resize(static_cast<size_t>(M));
        for (auto& v : a_t_scale) v = 0.5f + (rng.next_u64() % 1000) / 1000.0f;
        for (auto& v : a_t_zp) v = static_cast<int32_t>(rng.next_u64() % 11 - 5);
    }
    const TensorView a_view =
        ta ? make_q(K, M, DataType::s8, a_t_buf.data(),
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
    // With a pool the plan shrinks mc (M-split, once M/nt drops below
    // MC_TARGET) and nc (N-split), so the decomposition differs per thread
    // count. Both int8 splits, plus the s8 accumulator and requant epilogue,
    // must still match the reference element for element.
    for (int nt : {2, 4, 8}) {
        expect_i8_threaded_matches_ref<int32_t>(512, 512, 128, nt, 11);   // M-split
        expect_i8_threaded_matches_ref<int32_t>(128, 512, 512, nt, 12);   // N-split
        expect_i8_threaded_matches_ref<int32_t>(256, 256, 256, nt, 13);   // mc < MC_TARGET
        expect_i8_threaded_matches_ref<int32_t>(1024, 512, 1024, nt, 14); // the measured shape
    }
    expect_i8_threaded_matches_ref<int8_t>(512, 512, 128, 4, 15);
    expect_i8_threaded_matches_ref<int8_t>(128, 512, 512, 4, 16);

    // Transposed operands: the reductions run along a different axis, and each
    // combination is a separate loop nest, so cover all four with a pool.
    for (int nt : {2, 4, 8}) {
        expect_i8_threaded_matches_ref<int32_t>(512, 512, 128, nt, 21, /*ta=*/true);
        expect_i8_threaded_matches_ref<int32_t>(512, 512, 128, nt, 22, /*ta=*/false, /*tb=*/true);
        expect_i8_threaded_matches_ref<int32_t>(512, 512, 128, nt, 23, /*ta=*/true,  /*tb=*/true);
        // Wide-N (N-split) with both transposes, where nc is also shrunk.
        expect_i8_threaded_matches_ref<int32_t>(128, 512, 512, nt, 24, /*ta=*/true,  /*tb=*/true);
        expect_i8_threaded_matches_ref<int8_t>(256, 256, 256, nt, 25, /*ta=*/true,  /*tb=*/true);
    }
}
