/// Unit tests for integer (s8×s8) MatMul — per-token activation × per-channel
/// weight, s32 (MatMulInteger) or requantized s8 output.
///
/// The tiled kernel (packed VNNI / SDOT / scalar path) is compared against
/// matmul_int8_ref, plus a few exact-value cases that pin the zero-point
/// compensation and requantization formulas.

#include "nnops/ops/matmul.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"

#include <cstdint>
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

    std::vector<char> workspace(op->getWorkspaceSize(arr, descs));
    NNOPS_EXPECT_TRUE(workspace.size() > 0);

    const TensorView ins[] = {a, b};
    op->compute(output, ins, {}, workspace.data());

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

    std::vector<char> workspace(op->getWorkspaceSize(arr, descs));
    const TensorView ins[] = {a, w};
    op->compute(output, ins, {}, workspace.data());

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

    std::vector<char> workspace(op->getWorkspaceSize(arr, descs));
    NNOPS_EXPECT_TRUE(workspace.size() > 0);
    const TensorView ins[] = {a, w};
    op->compute(output, ins, {}, workspace.data());

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

    std::vector<char> workspace(op->getWorkspaceSize(arr, descs));
    const TensorView ins[] = {a, w};
    op->compute(output, ins, {}, workspace.data());

    // acc = 100*100 + 100*100 = 20000 → clamps to 127.
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[0]), 127);
    NNOPS_EXPECT_EQ(static_cast<int>(out_buf[1]), 127);
}

// ============================================================
// Random correctness vs reference (both transpose variants, s32 + s8)
// ============================================================

namespace {

// Random int8 tensors with per-token quant params over [lo, hi).
struct I8Inputs {
    std::vector<int8_t> a_buf, b_buf;
    std::vector<float> a_scale, b_scale;
    std::vector<int32_t> a_zp, b_zp;
    TensorView a, b;
};

I8Inputs make_i8_inputs(int64_t M, int64_t K, int64_t N, bool transpose_b, uint64_t seed) {
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
    in.a_buf.resize(static_cast<size_t>(M * K));
    in.b_buf.resize(static_cast<size_t>(b_rows * b_cols));
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
        std::vector<char> workspace(op->getWorkspaceSize(arr, descs));
        const TensorView ins[] = {in.a, in.b};
        op->compute(output, ins, {}, workspace.data());

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

    std::vector<char> workspace(op->getWorkspaceSize(arr, descs));
    const TensorView ins[] = {in.a, in.b};
    op->compute(output, ins, {}, workspace.data());

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
    std::vector<char> workspace(op->getWorkspaceSize(arr, descs));
    const TensorView ins[] = {in.a, in.b};
    op->compute(output, ins, {}, workspace.data());

    std::vector<int32_t> ref_buf(descs[0].numel());
    auto ref_out = test::make_planar(descs[0], ref_buf.data());
    nnops::backend::cpu::reference::matmul_int8_ref(attrs, ref_out, ins, {}, nullptr);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_EQ(out_buf[i], ref_buf[i]);
    }
}

NNOPS_TEST(matmul_int8_batched_s32) {
    const int64_t B = 3, M = 7, K = 13, N = 5;
    // Batch 0..2 share the same per-token quant params; build three views.
    I8Inputs in = make_i8_inputs(B * M, K, N, /*transpose_b=*/false, 31337);

    // A is [B, M, K], B is [B, K, N] — reuse the flat buffers.
    const int64_t ashape[] = {B, M, K};
    const int64_t bshape[] = {B, K, N};
    TensorView a(ashape, DataType::s8, in.a_buf.data());
    TensorView b(bshape, DataType::s8, in.b_buf.data());

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
    std::vector<char> workspace(op->getWorkspaceSize(arr, descs));
    const TensorView ins[] = {a, b};
    op->compute(output, ins, {}, workspace.data());

    std::vector<int32_t> ref_buf(descs[0].numel());
    auto ref_out = test::make_planar(descs[0], ref_buf.data());
    nnops::backend::cpu::reference::matmul_int8_ref(attrs, ref_out, ins, {}, nullptr);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_EQ(out_buf[i], ref_buf[i]);
    }
}
