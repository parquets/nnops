/// Unit tests for the offline-quantized-activation GEMM:
/// fp16/fp32 A (per-tensor or per-token offline params) × s8 B, quantized to
/// u8/s8 inside the kernel, int8 GEMM, dequantized back to A's dtype.
///
/// Every case is compared bit for bit against `matmul_dq_ref`: the GEMM is
/// integer, and both sides share the dequant expression, so agreement is exact
/// (f16 outputs are compared by their bit patterns).

#include "nnops/ops/matmul.hpp"
#include "nnops/detail/half.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <type_traits>
#include <vector>

using namespace nnops;
using nnops::backend::cpu::half;
using nnops::backend::cpu::float_to_half;
using nnops::backend::cpu::half_to_bits;

namespace nnops::backend::cpu::reference {
void matmul_dq_ref(const MatMulAttributes& attrs,
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

QuantParams per_channel(const float* scale, const int32_t* zp, int64_t n) {
    QuantParams qp;
    qp.granularity = QuantGranularity::PerChannel;
    qp.scale_data = scale;
    qp.zero_point_data = zp;
    qp.num_scales = n;
    return qp;
}

TensorView make_q(int64_t rows, int64_t cols, DataType dt, void* data, const QuantParams& qp) {
    const int64_t shape[] = {rows, cols};
    return TensorView(shape, dt, data, TensorLayout::NCHW, qp);
}

TensorView make_q_nd(std::span<const int64_t> shape, DataType dt, void* data, const QuantParams& qp) {
    return TensorView(shape, dt, data, TensorLayout::NCHW, qp);
}

std::span<const int64_t> desc_shape(const TensorDesc& d) {
    return std::span<const int64_t>(d.dims.data(), static_cast<size_t>(d.rank));
}

// ---- random operands ----

/// fp A (f32 and the f16 view of the same values), s8 B, and the per-row /
/// per-column parameters both use. Build TensorViews *after* this returns — the
/// vectors own the storage the views point at.
struct Quad {
    std::vector<float>   a;     // M×K
    std::vector<half>    ah;    // M×K
    std::vector<float>   a_s;   // M
    std::vector<int32_t> a_zp;  // M
    std::vector<int8_t>  b;     // K×N
    std::vector<float>   b_s;   // N
    std::vector<int32_t> b_zp;  // N
};

Quad make_quad(int64_t M, int64_t K, int64_t N, uint64_t seed) {
    Quad q;
    q.a.resize(static_cast<size_t>(M * K));
    q.ah.resize(q.a.size());
    q.a_s.resize(static_cast<size_t>(M));
    q.a_zp.resize(static_cast<size_t>(M));
    q.b.resize(static_cast<size_t>(K * N));
    q.b_s.resize(static_cast<size_t>(N));
    q.b_zp.resize(static_cast<size_t>(N));

    nnops::test::XorShift128 rng(seed);
    rng.fill_float(q.a.data(), M * K, -2.0f, 2.0f);
    for (size_t i = 0; i < q.a.size(); ++i) { q.ah[i] = float_to_half(q.a[i]); }

    for (int64_t m = 0; m < M; ++m) {
        q.a_s[static_cast<size_t>(m)]  = 0.02f + 0.01f * (rng.next_u64() % 8) / 7.0f;
        q.a_zp[static_cast<size_t>(m)] = static_cast<int32_t>(rng.next_u64() % 21) - 10;
    }
    for (auto& v : q.b) {
        v = static_cast<int8_t>(static_cast<int32_t>(rng.next_u64() % 255) - 127);
    }
    for (int64_t n = 0; n < N; ++n) {
        q.b_s[static_cast<size_t>(n)]  = 0.004f + 0.004f * (rng.next_u64() % 8) / 7.0f;
        q.b_zp[static_cast<size_t>(n)] = static_cast<int32_t>(rng.next_u64() % 21) - 10;
    }
    return q;
}

// ---- exact comparison ----

template <class T>
uint32_t bits_of(T v) {
    if constexpr (std::is_same_v<T, half>) {
        return half_to_bits(v);
    } else {
        uint32_t b;
        std::memcpy(&b, &v, sizeof(b));
        return b;
    }
}

/// Run the op and matmul_dq_ref on the same inputs; the outputs must agree bit
/// for bit. `odesc` is the descriptor the op itself resolved.
template <class T>
void expect_dq_matches_ref(const MatMulAttributes& attrs, const TensorDesc& odesc,
                           std::span<const TensorView> ins,
                           const ComputeContext& ctx = {})
{
    std::vector<T> got(static_cast<size_t>(odesc.numel()));
    std::vector<T> want(got.size());
    TensorView output  = test::make_planar(odesc, got.data());
    TensorView ref_out = test::make_planar(odesc, want.data());

    auto op = MatMul::create(attrs, Backend::CPU);
    op->compute(output, ins, ctx, nullptr);
    nnops::backend::cpu::reference::matmul_dq_ref(attrs, ref_out, ins, {}, nullptr);

    for (size_t i = 0; i < got.size(); ++i) {
        NNOPS_EXPECT_EQ(static_cast<long>(bits_of(got[i])),
                        static_cast<long>(bits_of(want[i])));
    }
}

/// The quantization the kernel performs, one element at a time.
template <class QInt>
QInt quant_one(float x, float scale, int32_t zp) {
    constexpr int32_t qmin = std::is_same_v<QInt, uint8_t> ? 0 : -128;
    constexpr int32_t qmax = std::is_same_v<QInt, uint8_t> ? 255 : 127;
    int32_t q = static_cast<int32_t>(std::nearbyintf(std::fmaf(x, 1.0f / scale,
                                                              static_cast<float>(zp))));
    q = std::min(std::max(q, qmin), qmax);
    return static_cast<QInt>(q);
}

}  // anonymous namespace

// ============================================================
// Exact values — pin the quantize / compensate / dequantize formula
// ============================================================

NNOPS_TEST(matmul_dq_exact_f32) {
    // per-tensor s_a=0.25, zp_a=0 → A_q = round(A·4).
    const int64_t ashape[] = {2, 3};
    const int64_t bshape[] = {3, 2};
    float  a_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, -5.0f, 6.0f};
    int8_t b_data[6] = {1, 2, 3, 4, 5, 6};

    TensorView a = make_q(2, 3, DataType::f32, a_data, per_tensor(0.25f, 0));
    TensorView b = make_q(3, 2, DataType::s8, b_data, per_tensor(1.0f, 0));

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);   // output follows A

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {a, b};
    op->compute(output, ins, {}, nullptr);

    // A_q = [[4,8,12],[16,-20,24]]; row dots with B columns {1,3,5} / {2,4,6}.
    // C = acc · 0.25.
    NNOPS_EXPECT_TRUE(out_buf[0] == 22.0f);
    NNOPS_EXPECT_TRUE(out_buf[1] == 28.0f);
    NNOPS_EXPECT_TRUE(out_buf[2] == 19.0f);
    NNOPS_EXPECT_TRUE(out_buf[3] == 24.0f);
}

NNOPS_TEST(matmul_dq_per_token_scale) {
    // s_a = {0.5, 0.25}, zp_a = {2, -1}: the A_q values differ from the
    // symmetric case but the dezero-pointed values (and thus C) do not.
    const int64_t ashape[] = {2, 3};
    const int64_t bshape[] = {3, 2};
    float   a_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int8_t  b_data[6] = {1, 2, 3, 4, 5, 6};
    float   a_s[2] = {0.5f, 0.25f};
    int32_t a_zp[2] = {2, -1};

    TensorView a = make_q(2, 3, DataType::f32, a_data, per_token(a_s, a_zp, 2));
    TensorView b = make_q(3, 2, DataType::s8, b_data, per_tensor(1.0f, 0));

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    TensorView output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {a, b};
    op->compute(output, ins, {}, nullptr);

    // row 0: A_q = [4,6,8] → dezeroed [2,4,6]; row 1: A_q = [15,19,23] →
    // dezeroed [16,20,24]. Same C as the symmetric case: 22,28 / 49,64.
    NNOPS_EXPECT_TRUE(out_buf[0] == 22.0f);
    NNOPS_EXPECT_TRUE(out_buf[1] == 28.0f);
    NNOPS_EXPECT_TRUE(out_buf[2] == 49.0f);
    NNOPS_EXPECT_TRUE(out_buf[3] == 64.0f);
}

// ============================================================
// Random operands against the reference
// ============================================================

NNOPS_TEST(matmul_dq_per_tensor_matches_ref) {
    constexpr int64_t M = 13, K = 21, N = 17;   // N is not a multiple of nr
    Quad q = make_quad(M, K, N, 101);
    TensorView a = make_q(M, K, DataType::f32, q.a.data(), per_tensor(0.025f, 3));
    TensorView b = make_q(K, N, DataType::s8, q.b.data(), per_tensor(0.006f, -2));

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    const TensorView ins[] = {a, b};
    expect_dq_matches_ref<float>(attrs, descs[0], ins);
}

NNOPS_TEST(matmul_dq_symmetric_matches_ref) {
    // Both subtractions in compensate() vanish — A has no zero points and every
    // zp_b is zero — so the dispatch skips the reduction sweep and the row pass
    // takes its no-compensation path. B keeps an all-zero zp *array* so the
    // "is any zero point non-zero" scan runs over a buffer, not a scalar, and A
    // keeps its per-token scale array so only the zero side is the identity.
    constexpr int64_t M = 21, K = 33, N = 19;
    Quad q = make_quad(M, K, N, 105);
    std::fill(q.b_zp.begin(), q.b_zp.end(), 0);
    TensorView a = make_q(M, K, DataType::f32, q.a.data(),
                          per_token(q.a_s.data(), nullptr, M));
    TensorView b = make_q(K, N, DataType::s8, q.b.data(),
                          per_channel(q.b_s.data(), q.b_zp.data(), N));

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    const TensorView ins[] = {a, b};
    expect_dq_matches_ref<float>(attrs, descs[0], ins);

    // With an activation the flag must skip only the two subtractions — the
    // activation still runs. (No custom clip bounds here: matmul_cpu folds
    // Relu/Relu6 into min_clip/max_clip and a caller-supplied bound is
    // overwritten, which the reference does not do.)
    MatMulAttributes attrs_act{};
    attrs_act.epilogue.type = EpilogueActivateType::Relu;
    expect_dq_matches_ref<float>(attrs_act, descs[0], ins);
}

NNOPS_TEST(matmul_dq_symmetric_f16_matches_ref) {
    // The same identity compensation through the f16 output pass.
    constexpr int64_t M = 17, K = 29, N = 15;
    Quad q = make_quad(M, K, N, 116);
    std::fill(q.b_zp.begin(), q.b_zp.end(), 0);
    TensorView a = make_q(M, K, DataType::f16, q.ah.data(),
                          per_token(q.a_s.data(), nullptr, M));
    TensorView b = make_q(K, N, DataType::s8, q.b.data(),
                          per_channel(q.b_s.data(), q.b_zp.data(), N));

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);
    const TensorView ins[] = {a, b};
    expect_dq_matches_ref<half>(attrs, descs[0], ins);
}

NNOPS_TEST(matmul_dq_per_token_matches_ref) {
    // Per-token A × per-channel B: both sides of the compensation vary.
    constexpr int64_t M = 13, K = 21, N = 17;
    Quad q = make_quad(M, K, N, 102);
    TensorView a = make_q(M, K, DataType::f32, q.a.data(), per_token(q.a_s.data(), q.a_zp.data(), M));
    TensorView b = make_q(K, N, DataType::s8, q.b.data(), per_channel(q.b_s.data(), q.b_zp.data(), N));

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    const TensorView ins[] = {a, b};
    expect_dq_matches_ref<float>(attrs, descs[0], ins);
}

NNOPS_TEST(matmul_dq_per_token_transpose_b_matches_ref) {
    // [N,K] weights: the per-channel arrays index the output column n.
    constexpr int64_t M = 9, K = 13, N = 11;
    Quad q = make_quad(M, K, N, 103);
    // Re-lay B as N×K with the same column-major values.
    std::vector<int8_t> bt(static_cast<size_t>(N * K));
    for (int64_t k = 0; k < K; ++k) {
        for (int64_t n = 0; n < N; ++n) {
            bt[static_cast<size_t>(n * K + k)] = q.b[static_cast<size_t>(k * N + n)];
        }
    }
    TensorView a = make_q(M, K, DataType::f32, q.a.data(), per_token(q.a_s.data(), q.a_zp.data(), M));
    TensorView b = make_q(N, K, DataType::s8, bt.data(), per_channel(q.b_s.data(), q.b_zp.data(), N));

    MatMulAttributes attrs{};
    attrs.transpose_b = true;
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    const TensorView ins[] = {a, b};
    expect_dq_matches_ref<float>(attrs, descs[0], ins);
}

NNOPS_TEST(matmul_dq_per_tensor_b_zp_symmetric_a_matches_ref) {
    // Symmetric A and a per-tensor (not per-channel) non-zero zp_b: the whole
    // compensation is a row constant, so the dispatch folds it into a per-row
    // offset and sweeps neither A nor B.
    constexpr int64_t M = 12, K = 20, N = 16;
    Quad q = make_quad(M, K, N, 105);
    TensorView a = make_q(M, K, DataType::f32, q.a.data(), per_tensor(0.03f, 0));
    TensorView b = make_q(K, N, DataType::s8, q.b.data(), per_tensor(0.006f, -7));

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    const TensorView ins[] = {a, b};
    expect_dq_matches_ref<float>(attrs, descs[0], ins);
}

NNOPS_TEST(matmul_dq_per_tensor_b_zp_asymmetric_a_matches_ref) {
    // Both halves live: the column term sweeps B, while the per-element term is
    // still a row constant and folds — r_a then feeds only the fold.
    constexpr int64_t M = 10, K = 18, N = 14;
    Quad q = make_quad(M, K, N, 106);
    TensorView a = make_q(M, K, DataType::f32, q.a.data(), per_token(q.a_s.data(), q.a_zp.data(), M));
    TensorView b = make_q(K, N, DataType::s8, q.b.data(), per_tensor(0.006f, 5));

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    const TensorView ins[] = {a, b};
    expect_dq_matches_ref<float>(attrs, descs[0], ins);
}

NNOPS_TEST(matmul_dq_symmetric_a_per_channel_b_matches_ref) {
    // A symmetric, so the column term is identically zero and the B sweep is
    // skipped; the per-element half has no row fold (zp_b varies per channel)
    // and is still applied element by element.
    constexpr int64_t M = 13, K = 21, N = 17;
    Quad q = make_quad(M, K, N, 107);
    TensorView a = make_q(M, K, DataType::f32, q.a.data(), per_tensor(0.03f, 0));
    TensorView b = make_q(K, N, DataType::s8, q.b.data(), per_channel(q.b_s.data(), q.b_zp.data(), N));

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    const TensorView ins[] = {a, b};
    expect_dq_matches_ref<float>(attrs, descs[0], ins);
}

NNOPS_TEST(matmul_dq_u8_target_matches_ref) {
    // act_quant_dtype = u8: the pack shifts A into signed range for the
    // signed-dot kernels, and the shift comes back out of the accumulator.
    constexpr int64_t M = 15, K = 19, N = 17;
    Quad q = make_quad(M, K, N, 104);
    TensorView a = make_q(M, K, DataType::f32, q.a.data(), per_token(q.a_s.data(), q.a_zp.data(), M));
    TensorView b = make_q(K, N, DataType::s8, q.b.data(), per_channel(q.b_s.data(), q.b_zp.data(), N));

    MatMulAttributes attrs{};
    attrs.act_quant_dtype = DataType::u8;
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    const TensorView ins[] = {a, b};
    expect_dq_matches_ref<float>(attrs, descs[0], ins);
}

NNOPS_TEST(matmul_dq_u8_negative_zp_matches_ref) {
    // A u8 target with a negative zero point: quantization clamps at 0, and the
    // compensation still has to recover the right values.
    constexpr int64_t M = 7, K = 9, N = 5;
    Quad q = make_quad(M, K, N, 105);
    TensorView a = make_q(M, K, DataType::f32, q.a.data(), per_tensor(0.02f, -30));
    TensorView b = make_q(K, N, DataType::s8, q.b.data(), per_tensor(0.005f, 4));

    MatMulAttributes attrs{};
    attrs.act_quant_dtype = DataType::u8;
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    const TensorView ins[] = {a, b};
    expect_dq_matches_ref<float>(attrs, descs[0], ins);
}

NNOPS_TEST(matmul_dq_f16_matches_ref) {
    constexpr int64_t M = 13, K = 21, N = 17;
    Quad q = make_quad(M, K, N, 106);
    TensorView a = make_q(M, K, DataType::f16, q.ah.data(), per_token(q.a_s.data(), q.a_zp.data(), M));
    TensorView b = make_q(K, N, DataType::s8, q.b.data(), per_channel(q.b_s.data(), q.b_zp.data(), N));

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);
    const TensorView ins[] = {a, b};
    expect_dq_matches_ref<half>(attrs, descs[0], ins);
}

NNOPS_TEST(matmul_dq_multik_matches_ref) {
    // K past KC_CAP_I8 (512): several k-blocks, so only the last one runs the
    // dequant epilogue.
    constexpr int64_t M = 13, K = 1200, N = 17;
    Quad q = make_quad(M, K, N, 107);
    TensorView a = make_q(M, K, DataType::f32, q.a.data(), per_token(q.a_s.data(), q.a_zp.data(), M));
    TensorView b = make_q(K, N, DataType::s8, q.b.data(), per_channel(q.b_s.data(), q.b_zp.data(), N));

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    const TensorView ins[] = {a, b};
    expect_dq_matches_ref<float>(attrs, descs[0], ins);
}

NNOPS_TEST(matmul_dq_nsplit_matches_ref) {
    // N ≫ M: the plan splits over N (NKM), so the per-token arrays are read by
    // every n-block and the accumulators are per-column-range.
    constexpr int64_t M = 16, K = 512, N = 1024;
    Quad q = make_quad(M, K, N, 108);
    TensorView a = make_q(M, K, DataType::f32, q.a.data(), per_token(q.a_s.data(), q.a_zp.data(), M));
    TensorView b = make_q(K, N, DataType::s8, q.b.data(), per_channel(q.b_s.data(), q.b_zp.data(), N));

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    const TensorView ins[] = {a, b};
    expect_dq_matches_ref<float>(attrs, descs[0], ins);
}

// ============================================================
// Shape sweep — every tiling boundary, bit-exact against the reference
// ============================================================

namespace {

struct DqShape { int64_t M, K, N; };

/// The boundaries the tiling actually has, as (M, K, N):
///   - M = 1, 2 and 5: the int8 route has no GEMV kernel, so these are the
///     m-tail micro-kernels running one to five rows;
///   - M = 6, 7: MR_MAX_I8 and the first row past it;
///   - N = 15, 16, 17: the padded 16-wide B panel — untouched, exact, and one
///     past it (copy packer with a masked-load tail);
///   - N = 240, 241: nc on this class of host, and one past it;
///   - K = 511, 512, 513: either side of KC_CAP_I8, i.e. the k-block boundary
///     and the multi-k-block epilogue;
///   - the rest: mixed primes, powers of two, and the two rectangular corners
///     (large-M/small-N, small-M/large-N).
const DqShape kDqShapes[] = {
    {1, 1, 1},        {1, 16, 16},     {1, 512, 1024},  {2, 3, 2},
    {3, 7, 5},        {5, 16, 15},     {6, 16, 16},     {7, 17, 17},
    {13, 21, 17},     {15, 31, 15},    {16, 32, 16},    {17, 33, 31},
    {31, 63, 33},     {33, 65, 47},    {63, 127, 63},   {64, 128, 64},
    {65, 129, 65},    {127, 255, 127}, {128, 256, 128}, {129, 257, 240},
    {240, 511, 241},  {256, 512, 256}, {128, 512, 512}, {256, 513, 256},
    {64, 1024, 64},   {16, 1200, 17},  {1024, 256, 64},
};

/// One shape through the same bit-exact comparison. @p act_dt selects the
/// integer storage; @p scalar_params swaps the per-token / per-channel arrays
/// for one (scale, zp) per tensor; @p symmetric zeroes every zero point, which
/// makes the compensation the identity (and, on an M-split, lets the pack
/// quantize A directly).
template <class T>
void expect_dq_shape(int64_t M, int64_t K, int64_t N, uint64_t seed,
                     DataType act_dt, bool scalar_params, bool symmetric = false) {
    Quad q = make_quad(M, K, N, seed);
    const bool u8 = (act_dt == DataType::u8);
    if (symmetric) { std::fill(q.b_zp.begin(), q.b_zp.end(), 0); }

    const QuantParams qa = scalar_params
        ? per_tensor(u8 ? 0.008f : 0.02f, symmetric ? 0 : (u8 ? 12 : 3))
        : per_token(q.a_s.data(), symmetric ? nullptr : q.a_zp.data(), M);
    // B keeps its (all-zero, when symmetric) zp array so the "is any zero point
    // non-zero" scan still runs over a buffer rather than a scalar.
    const QuantParams qb = scalar_params
        ? per_tensor(0.006f, symmetric ? 0 : -2)
        : per_channel(q.b_s.data(), q.b_zp.data(), N);

    constexpr DataType dt = std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
    void* a_data = std::is_same_v<T, float> ? static_cast<void*>(q.a.data())
                                            : static_cast<void*>(q.ah.data());
    TensorView a = make_q(M, K, dt, a_data, qa);
    TensorView b = make_q(K, N, DataType::s8, q.b.data(), qb);

    MatMulAttributes attrs{};
    attrs.act_quant_dtype = act_dt;
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, dt);
    NNOPS_EXPECT_EQ(descs[0].dims[0], M);
    NNOPS_EXPECT_EQ(descs[0].dims[1], N);
    const TensorView ins[] = {a, b};
    expect_dq_matches_ref<T>(attrs, descs[0], ins);
}

/// Run the whole table, tagging a failure with the shape that produced it.
template <class T>
void run_dq_shape_sweep(const char* tag, DataType act_dt, bool scalar_params,
                        bool symmetric = false) {
    int64_t i = 0;
    for (const DqShape& s : kDqShapes) {
        try {
            expect_dq_shape<T>(s.M, s.K, s.N, 300 + static_cast<uint64_t>(i++),
                               act_dt, scalar_params, symmetric);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string(tag) + " " + std::to_string(s.M) + "x" +
                                     std::to_string(s.K) + "x" + std::to_string(s.N) +
                                     ": " + e.what());
        }
    }
}

}  // anonymous namespace

NNOPS_TEST(matmul_dq_shape_sweep) {
    // The shipped configuration — f32 activations, s8 target, per-token A ×
    // per-channel B — over every tiling boundary.
    run_dq_shape_sweep<float>("f32/s8/token", DataType::s8, /*scalar_params=*/false);
}

NNOPS_TEST(matmul_dq_shape_sweep_u8) {
    // u8 storage with per-token parameters: the sign handling in the pack and
    // the epilogue's row vectors are both live.
    run_dq_shape_sweep<float>("f32/u8/token", DataType::u8, /*scalar_params=*/false);
}

NNOPS_TEST(matmul_dq_shape_sweep_f16_u8) {
    // f16 activations into u8 storage with per-tensor parameters: the quantize
    // pass widens f16, the arch shifts u8 into signed range, and the dequant
    // multiply runs in f32 before the f16 narrowing.
    run_dq_shape_sweep<half>("f16/u8/tensor", DataType::u8, /*scalar_params=*/true);
}

NNOPS_TEST(matmul_dq_shape_sweep_symmetric) {
    // No zero point on either side: compensate() is the identity, so the
    // reduction sweep is skipped — and, on an M-split, nothing ever reads a
    // materialised A_q, so the A pack quantizes the fp activations itself
    // (the fused route). The table still covers every tiling boundary, the
    // multi-k-block epilogue, and the N-split shapes that fall back.
    run_dq_shape_sweep<float>("f32/s8/symmetric", DataType::s8, /*scalar_params=*/false, true);
    run_dq_shape_sweep<float>("f32/s8/tensor/symmetric", DataType::s8, /*scalar_params=*/true, true);
    run_dq_shape_sweep<half>("f16/s8/symmetric", DataType::s8, /*scalar_params=*/true, true);
    // u8 storage whose zero point is 0: the target's clamp range still applies.
    run_dq_shape_sweep<float>("f32/u8/symmetric", DataType::u8, /*scalar_params=*/true, true);
}

NNOPS_TEST(matmul_dq_saturated_rows) {
    // Rows whose magnitudes dwarf scale·127 clamp to the storage range; an
    // all-zero row quantizes to zp_a everywhere (and must still dequantize to
    // A ≈ 0's contribution, i.e. the compensated dot is exact).
    constexpr int64_t M = 5, K = 33, N = 7;
    Quad q = make_quad(M, K, N, 109);
    for (int64_t k = 0; k < K; ++k) {
        q.a[static_cast<size_t>(0 * K + k)] = 900.0f;    // saturate high
        q.a[static_cast<size_t>(1 * K + k)] = -900.0f;   // saturate low
        q.a[static_cast<size_t>(2 * K + k)] = 0.0f;      // all zero
    }
    TensorView a = make_q(M, K, DataType::f32, q.a.data(), per_token(q.a_s.data(), q.a_zp.data(), M));
    TensorView b = make_q(K, N, DataType::s8, q.b.data(), per_token(q.b_s.data(), q.b_zp.data(), N));

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);

    // The saturated rows clamp identically in both implementations.
    const TensorView ins[] = {a, b};
    expect_dq_matches_ref<float>(attrs, descs[0], ins);
}

NNOPS_TEST(matmul_dq_padded_a_pitch) {
    // A's row pitch exceeds K: the quantize pass must step by lda, not K.
    constexpr int64_t M = 9, K = 13, N = 7, pitch_elems = 20;
    Quad q = make_quad(M, K, N, 110);
    std::vector<float> padded(static_cast<size_t>(M * pitch_elems), -12345.0f);
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            padded[static_cast<size_t>(m * pitch_elems + k)] = q.a[static_cast<size_t>(m * K + k)];
        }
    }
    const int64_t ashape[] = {M, K};
    TensorView a(ashape, DataType::f32, padded.data(),
                 pitch_elems * static_cast<int64_t>(sizeof(float)),
                 TensorLayout::NCHW, per_token(q.a_s.data(), q.a_zp.data(), M));
    TensorView b = make_q(K, N, DataType::s8, q.b.data(), per_channel(q.b_s.data(), q.b_zp.data(), N));

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    const TensorView ins[] = {a, b};
    expect_dq_matches_ref<float>(attrs, descs[0], ins);
}

// ============================================================
// Epilogue — clip then activation, in the fp route's order
// ============================================================

NNOPS_TEST(matmul_dq_epilogue) {
    constexpr int64_t M = 11, K = 17, N = 9;
    Quad q = make_quad(M, K, N, 111);
    TensorView a = make_q(M, K, DataType::f32, q.a.data(), per_token(q.a_s.data(), q.a_zp.data(), M));
    TensorView b = make_q(K, N, DataType::s8, q.b.data(), per_channel(q.b_s.data(), q.b_zp.data(), N));

    const auto run = [&](EpilogueActivateType type, float alpha = 0.0f,
                         float min_clip = -1e30f, float max_clip = 1e30f) {
        MatMulAttributes attrs{};
        attrs.epilogue.type = type;
        attrs.epilogue.alpha = alpha;
        attrs.epilogue.min_clip = min_clip;
        attrs.epilogue.max_clip = max_clip;
        auto op = MatMul::create(attrs, Backend::CPU);
        const TensorDesc arr[] = {a.desc(), b.desc()};
        auto descs = op->getOutputTensorDesc(arr);
        const TensorView ins[] = {a, b};
        expect_dq_matches_ref<float>(attrs, descs[0], ins);
    };

    run(EpilogueActivateType::None);
    run(EpilogueActivateType::Relu);
    run(EpilogueActivateType::Relu6);
    run(EpilogueActivateType::LeakyRelu, 0.1f);
    run(EpilogueActivateType::Sigmoid);
    run(EpilogueActivateType::Tanh);
    run(EpilogueActivateType::None, 0.0f, -3.0f, 5.0f);       // clip only
    run(EpilogueActivateType::Sigmoid, 0.0f, -0.5f, 1.5f);    // clip + activation
}

// ============================================================
// Batched / broadcast
// ============================================================

NNOPS_TEST(matmul_dq_batched) {
    constexpr int64_t B = 2, M = 9, K = 13, N = 7;
    Quad q = make_quad(B * M, K, N, 112);
    const int64_t ashape[] = {B, M, K};
    const int64_t bshape[] = {K, N};
    TensorView a = make_q_nd(ashape, DataType::f32, q.a.data(),
                             per_token(q.a_s.data(), q.a_zp.data(), B * M));
    TensorView b = make_q_nd(bshape, DataType::s8, q.b.data(),
                             per_channel(q.b_s.data(), q.b_zp.data(), N));

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dims[0], B);
    const TensorView ins[] = {a, b};
    expect_dq_matches_ref<float>(attrs, descs[0], ins);
}

NNOPS_TEST(matmul_dq_batched_symmetric) {
    // The fused pack path with a batch: every batch indexes the per-token scale
    // array from its own first row (`row_base = a_offset / lda`), which the
    // pack reads through the same pointer the epilogue does.
    constexpr int64_t B = 3, M = 9, K = 550, N = 7;   // K crosses KC_CAP_I8 too
    Quad q = make_quad(B * M, K, N, 118);
    std::fill(q.b_zp.begin(), q.b_zp.end(), 0);   // symmetric: identity compensation
    std::vector<int8_t> bb(static_cast<size_t>(B * K * N));
    for (int64_t i = 0; i < B; ++i) {
        std::copy(q.b.begin(), q.b.end(), bb.begin() + i * K * N);
    }
    const int64_t ashape[] = {B, M, K};
    const int64_t bshape[] = {B, K, N};
    TensorView a = make_q_nd(ashape, DataType::f32, q.a.data(),
                             per_token(q.a_s.data(), nullptr, B * M));
    TensorView b = make_q_nd(bshape, DataType::s8, bb.data(),
                             per_channel(q.b_s.data(), q.b_zp.data(), N));

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dims[0], B);
    const TensorView ins[] = {a, b};
    expect_dq_matches_ref<float>(attrs, descs[0], ins);
}

NNOPS_TEST(matmul_dq_batched_broadcast_a) {
    // A broadcast across a 3-way batch: every batch still reads A's own rows,
    // so the per-token row base stays 0.
    constexpr int64_t B = 3, M = 5, K = 9, N = 6;
    Quad q = make_quad(M, K, N, 113);
    const int64_t ashape[] = {M, K};
    const int64_t bshape[] = {B, K, N};
    std::vector<int8_t> bb(static_cast<size_t>(B * K * N));
    for (int64_t i = 0; i < B; ++i) {
        std::copy(q.b.begin(), q.b.end(), bb.begin() + i * K * N);
    }
    TensorView a = make_q_nd(ashape, DataType::f32, q.a.data(),
                             per_token(q.a_s.data(), q.a_zp.data(), M));
    TensorView b = make_q_nd(bshape, DataType::s8, bb.data(),
                             per_channel(q.b_s.data(), q.b_zp.data(), N));

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dims[0], B);
    const TensorView ins[] = {a, b};
    expect_dq_matches_ref<float>(attrs, descs[0], ins);
}

// ============================================================
// Compatibility: same result as "quantize A yourself + s32 int8 route"
// ============================================================

NNOPS_TEST(matmul_dq_matches_prequantized_int8) {
    // The reason the route exists: quantizing A outside and running the int8
    // route must produce the same numbers as the fused fp-activation route.
    constexpr int64_t M = 15, K = 77, N = 13;
    Quad q = make_quad(M, K, N, 114);

    std::vector<int8_t> aq(static_cast<size_t>(M * K));
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            aq[static_cast<size_t>(m * K + k)] =
                quant_one<int8_t>(q.a[static_cast<size_t>(m * K + k)],
                                  q.a_s[static_cast<size_t>(m)], q.a_zp[static_cast<size_t>(m)]);
        }
    }

    TensorView a_fp = make_q(M, K, DataType::f32, q.a.data(),
                             per_token(q.a_s.data(), q.a_zp.data(), M));
    TensorView b = make_q(K, N, DataType::s8, q.b.data(),
                          per_channel(q.b_s.data(), q.b_zp.data(), N));

    // Fused route.
    MatMulAttributes dq_attrs{};
    auto dq_op = MatMul::create(dq_attrs, Backend::CPU);
    const TensorDesc arr[] = {a_fp.desc(), b.desc()};
    auto descs = dq_op->getOutputTensorDesc(arr);
    std::vector<float> fused(static_cast<size_t>(descs[0].numel()));
    TensorView fused_out = test::make_planar(descs[0], fused.data());
    const TensorView ins_fp[] = {a_fp, b};
    dq_op->compute(fused_out, ins_fp, {}, nullptr);

    // Pre-quantized int8 route → s32, then dequantize by hand.
    TensorView a_q = make_q(M, K, DataType::s8, aq.data(),
                            per_token(q.a_s.data(), q.a_zp.data(), M));
    MatMulAttributes i8_attrs{};
    i8_attrs.output_dtype = DataType::s32;
    auto i8_op = MatMul::create(i8_attrs, Backend::CPU);
    const TensorDesc i8_arr[] = {a_q.desc(), b.desc()};
    auto i8_descs = i8_op->getOutputTensorDesc(i8_arr);
    NNOPS_EXPECT_EQ(i8_descs[0].dtype, DataType::s32);
    std::vector<int32_t> acc(static_cast<size_t>(i8_descs[0].numel()));
    TensorView acc_out = test::make_planar(i8_descs[0], acc.data());
    const TensorView ins_q[] = {a_q, b};
    i8_op->compute(acc_out, ins_q, {}, nullptr);

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            const float scale = q.a_s[static_cast<size_t>(m)] * q.b_s[static_cast<size_t>(n)];
            const float want = static_cast<float>(acc[static_cast<size_t>(m * N + n)]) * scale;
            NNOPS_EXPECT_EQ(bits_of(fused[static_cast<size_t>(m * N + n)]),
                            bits_of(want));
        }
    }
}

NNOPS_TEST(matmul_dq_prequantized_fp_output) {
    // v1.1: A is already quantized (s8 + params) and the caller asks for an
    // f32 output. The same dequant epilogue serves both entry points.
    constexpr int64_t M = 11, K = 19, N = 8;
    Quad q = make_quad(M, K, N, 115);
    std::vector<int8_t> aq(static_cast<size_t>(M * K));
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            aq[static_cast<size_t>(m * K + k)] =
                quant_one<int8_t>(q.a[static_cast<size_t>(m * K + k)],
                                  q.a_s[static_cast<size_t>(m)], q.a_zp[static_cast<size_t>(m)]);
        }
    }
    TensorView a_s8 = make_q(M, K, DataType::s8, aq.data(),
                             per_token(q.a_s.data(), q.a_zp.data(), M));
    TensorView b = make_q(K, N, DataType::s8, q.b.data(),
                          per_channel(q.b_s.data(), q.b_zp.data(), N));

    MatMulAttributes attrs{};
    attrs.output_dtype = DataType::f32;
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a_s8.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    std::vector<float> got(static_cast<size_t>(descs[0].numel()));
    TensorView output = test::make_planar(descs[0], got.data());
    const TensorView ins[] = {a_s8, b};
    op->compute(output, ins, {}, nullptr);

    // Compare against the s32 route dequantized by hand.
    MatMulAttributes i8_attrs{};
    i8_attrs.output_dtype = DataType::s32;
    auto i8_op = MatMul::create(i8_attrs, Backend::CPU);
    const TensorDesc i8_arr[] = {a_s8.desc(), b.desc()};
    auto i8_descs = i8_op->getOutputTensorDesc(i8_arr);
    std::vector<int32_t> acc(static_cast<size_t>(i8_descs[0].numel()));
    TensorView acc_out = test::make_planar(i8_descs[0], acc.data());
    i8_op->compute(acc_out, ins, {}, nullptr);

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            const float scale = q.a_s[static_cast<size_t>(m)] * q.b_s[static_cast<size_t>(n)];
            const float want = static_cast<float>(acc[static_cast<size_t>(m * N + n)]) * scale;
            NNOPS_EXPECT_EQ(bits_of(got[static_cast<size_t>(m * N + n)]), bits_of(want));
        }
    }
}

// transpose_a is outside the fast path; it must still be correct via the
// reference.
NNOPS_TEST(matmul_dq_transpose_a_ref) {
    constexpr int64_t M = 7, K = 11, N = 5;
    Quad q = make_quad(M, K, N, 116);
    // A physical K×M over the same logical values (A_log[m][k] = a[k*M+m]).
    std::vector<float> at(static_cast<size_t>(K * M));
    for (int64_t k = 0; k < K; ++k) {
        for (int64_t m = 0; m < M; ++m) {
            at[static_cast<size_t>(k * M + m)] = q.a[static_cast<size_t>(m * K + k)];
        }
    }
    TensorView a = make_q(K, M, DataType::f32, at.data(),
                          per_token(q.a_s.data(), q.a_zp.data(), M));
    TensorView b = make_q(K, N, DataType::s8, q.b.data(),
                          per_channel(q.b_s.data(), q.b_zp.data(), N));

    MatMulAttributes attrs{};
    attrs.transpose_a = true;
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dims[0], M);
    NNOPS_EXPECT_EQ(descs[0].dims[1], N);
    const TensorView ins[] = {a, b};
    expect_dq_matches_ref<float>(attrs, descs[0], ins);
}

// ============================================================
// Threaded correctness
// ============================================================

namespace {

// Minimal fixed-size thread pool exposing a CpuBackend (as in test_matmul.cpp).
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

void expect_dq_threaded_matches_ref(int64_t M, int64_t K, int64_t N, int nt, uint64_t seed) {
    Quad q = make_quad(M, K, N, seed);
    TensorView a = make_q(M, K, DataType::f32, q.a.data(),
                          per_token(q.a_s.data(), q.a_zp.data(), M));
    TensorView b = make_q(K, N, DataType::s8, q.b.data(),
                          per_channel(q.b_s.data(), q.b_zp.data(), N));

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);

    SimplePool pool(nt);
    ComputeContext ctx;
    ctx.cpu = pool.cpu;

    const TensorView ins[] = {a, b};
    expect_dq_matches_ref<float>(attrs, descs[0], ins, ctx);
}

}  // anonymous namespace

NNOPS_TEST(matmul_dq_threaded) {
    // With a pool the plan shrinks mc (M-split) and nc (N-split); the quantize
    // pass and the per-token arrays are shared read-only state, so every
    // decomposition must still match the serial reference exactly.
    for (int nt : {2, 4, 8}) {
        expect_dq_threaded_matches_ref(512, 512, 128, nt, 201);   // M-split
        expect_dq_threaded_matches_ref(128, 512, 512, nt, 202);   // N-split
        expect_dq_threaded_matches_ref(256, 256, 256, nt, 203);   // mc < MC_MAX
        expect_dq_threaded_matches_ref(1024, 512, 1024, nt, 204); // the measured shape
    }
}
