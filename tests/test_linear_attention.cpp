/// @file test_linear_attention.cpp
/// @brief Unit tests for the LinearAttention operator.
///
/// The SIMD kernel is checked against `reference::linear_attention_ref`, which
/// lives in its own translation unit and shares no code with it — so agreement
/// is evidence about the math rather than about a shared bug. On top of that sit
/// a hand-computed case, invariants (incremental decoding, conv causality, the
/// decay/gate semantics), an f16 case, and a genuinely threaded run.
///
/// Two things the tests pin that a kernel-vs-reference compare cannot:
///  - Optional input slots are positional: conv_w is index 4, beta is index 5.
///    A beta supplied in the fifth slot is read as conv taps, and both sides
///    misread the list identically, so the compare still passes.
///  - outputs[1] (the state) is read, then overwritten in place; decoding one
///    token at a time (each call S == 1) is what pins the in-place update.
///
/// The absolute tolerance is what decides near zero: the delta step subtracts
/// two nearly equal quantities (`v[j] - r[j]`), so a near-zero entry carries an
/// absolute error set by the magnitude of the terms that cancelled — ~eps times
/// the largest state entry, not eps times its own. A problem whose state grows
/// past order 1 therefore needs a looser atol.

#include "nnops/ops/linear_attention.hpp"
#include "common/test_harness.hpp"
#include "common/cpu_pool.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"

#include <atomic>
#include <cmath>
#include <span>
#include <string>
#include <thread>
#include <vector>

using namespace nnops;

// Test-local declaration; the reference is defined in its own translation unit.
namespace nnops::backend::cpu::reference {
void linear_attention_ref(const LinearAttentionAttributes& attrs,
                          std::span<TensorView> outputs,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx,
                          void* workspace);
}

// Minimal fixed-size pool exposing a CpuBackend; items in [begin, end) are
// claimed via an atomic counter. Without it the op only ever runs through the
// default ComputeContext, whose `run()` is sequential — the parallel path would
// never execute.
struct SimplePool {
    explicit SimplePool(int nthreads) : nthreads_(nthreads) {
        // OpenMP when the build has it, so the adapter runs under the same
        // comparisons as the kernels.
        if (nnops::test::bench_use_openmp()) {
            cpu = nnops::openmp_backend(nthreads);
            return;
        }
        cpu.parallel_for = [this](int64_t begin, int64_t end, const ParallelForBody& body) {
            this->run(begin, end, body);
        };
        cpu.num_threads = [this]() { return nthreads_; };
        cpu.thread_id = []() { return current_thread_id_; };
    }

    void run(int64_t begin, int64_t end, const ParallelForBody& body) {
        const int64_t count = end - begin;
        const int nw = nthreads_;
        std::atomic<int64_t> next{0};

        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(nw));
        for (int t = 0; t < nw; ++t) {
            workers.emplace_back([&, t]() {
                current_thread_id_ = t;
                for (;;) {
                    const int64_t i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= count) { break; }
                    body(begin + i);
                }
            });
        }
        for (auto& w : workers) { w.join(); }
    }

    ComputeContext context() const {
        ComputeContext c;
        c.cpu = cpu;
        return c;
    }

    int nthreads_;
    CpuBackend cpu;
    inline static thread_local int current_thread_id_ = 0;
};

namespace {

using nnops::backend::cpu::half;
using nnops::test::XorShift128;

// ============================================================
// Problem setup
// ============================================================

inline TensorView f32_view(std::vector<float>& buf, const std::vector<int64_t>& shape) {
    return TensorView(std::span<const int64_t>(shape.data(), shape.size()),
                      DataType::f32, buf.data(), TensorLayout::NCHW);
}

inline TensorView f16_view(std::vector<half>& buf, const std::vector<int64_t>& shape) {
    return TensorView(std::span<const int64_t>(shape.data(), shape.size()),
                      DataType::f16, buf.data(), TensorLayout::NCHW);
}

/// Attributes, input buffers and initial state for one problem; shapes derive
/// from (B, H_q, H_k, H_v, S, Dk, Dv). H_v / Dv of 0 mean "same as H_kv / Dk",
/// which is the plain Gated Delta Net.
struct Problem {
    LinearAttentionAttributes attrs;
    int64_t B = 1, H = 1, H_kv = 1, S = 4, D = 8;
    int64_t H_v = 0;         // value heads; 0 -> H_kv
    int64_t D_v = 0;         // value width; 0 -> D
    int64_t CK = 0;          // convolution taps; 0 => the conv weights are absent
    bool has_beta = false;

    std::vector<float> q, k, v, gate, conv_w, beta, state0;

    int64_t hv() const { return H_v > 0 ? H_v : H_kv; }
    int64_t dv() const { return D_v > 0 ? D_v : D; }

    std::vector<int64_t> sh_q()  const { return {B, H, S, D}; }
    std::vector<int64_t> sh_o()  const { return {B, H, S, dv()}; }
    std::vector<int64_t> sh_k()  const { return {B, H_kv, S, D}; }
    std::vector<int64_t> sh_v()  const { return {B, hv(), S, dv()}; }
    std::vector<int64_t> sh_g()  const { return {B, hv(), S, D}; }
    std::vector<int64_t> sh_st() const { return {B, hv(), D, dv()}; }
    std::vector<int64_t> sh_cw() const { return {(H + H_kv) * D + hv() * dv(), 1, CK}; }
    std::vector<int64_t> sh_bt() const { return {B, hv(), S}; }
};

void fill(std::vector<float>& buf, int64_t n, float lo, float hi, uint64_t seed) {
    buf.assign(static_cast<size_t>(n), 0.0f);
    XorShift128 rng(seed);
    rng.fill_float(buf.data(), n, lo, hi);
}

/// Random problem. `gate_lo/hi` bound the gate, which the kernel exponentiates:
/// [-2, 0] gives decay factors around [0.14, 1].
Problem make_problem(LinearAttentionAttributes attrs, int64_t B, int64_t H, int64_t H_kv,
                     int64_t S, int64_t D, int64_t CK = 0, bool has_beta = false,
                     float gate_lo = -2.0f, float gate_hi = 0.0f, uint64_t seed = 7,
                     int64_t H_v = 0, int64_t D_v = 0)
{
    Problem p;
    p.attrs = attrs;
    p.B = B; p.H = H; p.H_kv = H_kv; p.S = S; p.D = D;
    p.H_v = H_v; p.D_v = D_v;
    p.CK = CK; p.has_beta = has_beta;

    const int64_t hv = p.hv(), dv = p.dv();
    fill(p.q, B * H * S * D, -1.0f, 1.0f, seed + 1);
    fill(p.k, B * H_kv * S * D, -1.0f, 1.0f, seed + 2);
    fill(p.v, B * hv * S * dv, -1.0f, 1.0f, seed + 3);
    fill(p.gate, B * hv * S * D, gate_lo, gate_hi, seed + 4);
    fill(p.state0, B * hv * D * dv, -1.0f, 1.0f, seed + 5);
    if (CK > 0) {
        fill(p.conv_w, ((H + H_kv) * D + hv * dv) * CK, -0.5f, 0.5f, seed + 6);
    }
    if (has_beta) {
        fill(p.beta, B * hv * S, 0.0f, 1.0f, seed + 7);
    }
    return p;
}

struct Result {
    std::vector<float> out, state;
};

/// Run the problem through either the optimized kernel or the reference.
Result run(Problem& p, bool use_reference, const ComputeContext& ctx = {})
{
    const auto sq = p.sh_q(), so = p.sh_o(), sk = p.sh_k(),
               sv = p.sh_v(), sg = p.sh_g(), sst = p.sh_st();

    Result r;
    r.out.assign(static_cast<size_t>(p.B * p.H * p.S * p.dv()), 0.0f);
    r.state = p.state0;   // State is read, then overwritten in place

    TensorView q  = f32_view(p.q, sq);
    TensorView k  = f32_view(p.k, sk);
    TensorView v  = f32_view(p.v, sv);
    TensorView g  = f32_view(p.gate, sg);
    TensorView o  = f32_view(r.out, so);
    TensorView st = f32_view(r.state, sst);

    std::vector<TensorView> outputs = {o, st};
    std::vector<TensorView> ins = {q, k, v, g};
    // Optional slots are positional (conv_w is 4, beta is 5), so conv_w must be
    // occupied — by an empty view — whenever beta is supplied.
    TensorView cw, bt;
    if (p.CK > 0) { cw = f32_view(p.conv_w, p.sh_cw()); }
    if (p.CK > 0 || p.has_beta) { ins.push_back(cw); }
    if (p.has_beta) {
        bt = f32_view(p.beta, p.sh_bt());
        ins.push_back(bt);
    }

    if (use_reference) {
        nnops::backend::cpu::reference::linear_attention_ref(p.attrs, outputs, ins, ctx, nullptr);
    } else {
        auto op = LinearAttention::create(p.attrs, Backend::CPU);
        op->compute(outputs, ins, ctx, nullptr);
    }
    return r;
}

/// Kernel vs reference on the same data, for both outputs. `atol` is absolute;
/// see the file header for why that is the meaningful measure here.
void expect_kernel_matches_reference(Problem& p, const ComputeContext& ctx = {},
                                     const std::string& what = "linear_attention",
                                     float atol = 1e-6f)
{
    Result want = run(p, /*use_reference=*/true);
    Result got  = run(p, /*use_reference=*/false, ctx);

    const auto so = p.sh_o(), sst = p.sh_st();
    {
        TensorView gv = f32_view(got.out, so);
        TensorView wv = f32_view(want.out, so);
        if (!test::allclose(gv, wv, 1e-4f, atol)) {
            std::cerr << "  [" << what << "] output max_abs_diff = "
                      << test::max_diff(gv, wv) << "\n";
        }
        NNOPS_EXPECT_TRUE(test::allclose(gv, wv, 1e-4f, atol));
    }
    {
        TensorView gv = f32_view(got.state, sst);
        TensorView wv = f32_view(want.state, sst);
        if (!test::allclose(gv, wv, 1e-4f, atol)) {
            std::cerr << "  [" << what << "] state max_abs_diff = "
                      << test::max_diff(gv, wv) << "\n";
        }
        NNOPS_EXPECT_TRUE(test::allclose(gv, wv, 1e-4f, atol));
    }
}

}  // anonymous namespace

// ============================================================
// Operator creation and attributes
// ============================================================

NNOPS_TEST(linear_attention_create) {
    auto op = LinearAttention::create(Backend::CPU);
    NNOPS_EXPECT_EQ(static_cast<int>(op->getOpType()),
                    static_cast<int>(OpType::LinearAttention));
    NNOPS_EXPECT_EQ(static_cast<int>(op->getBackend()),
                    static_cast<int>(Backend::CPU));
}

NNOPS_TEST(linear_attention_default_attrs) {
    LinearAttentionAttributes attrs;
    NNOPS_EXPECT_EQ(attrs.head_dim, int64_t(64));
    NNOPS_EXPECT_EQ(attrs.num_heads, int64_t(8));
    NNOPS_EXPECT_EQ(attrs.num_kv_heads, int64_t(0));
    NNOPS_EXPECT_NEAR(attrs.scale, 0.0f, 1e-6f);
    NNOPS_EXPECT_EQ(attrs.conv_kernel_size, int64_t(4));
}

NNOPS_TEST(linear_attention_custom_attrs) {
    LinearAttentionAttributes attrs;
    attrs.head_dim = 128;
    attrs.num_heads = 32;
    attrs.num_kv_heads = 8;
    attrs.scale = 0.1f;
    attrs.conv_kernel_size = 3;

    auto op = LinearAttention::create(attrs, Backend::CPU);
    NNOPS_EXPECT_EQ(op->attributes().head_dim, int64_t(128));
    NNOPS_EXPECT_EQ(op->attributes().num_heads, int64_t(32));
    NNOPS_EXPECT_EQ(op->attributes().num_kv_heads, int64_t(8));
    NNOPS_EXPECT_NEAR(op->attributes().scale, 0.1f, 1e-6f);
    NNOPS_EXPECT_EQ(op->attributes().conv_kernel_size, int64_t(3));
}

// ============================================================
// Shape inference
// ============================================================

NNOPS_TEST(linear_attention_shape_inference) {
    LinearAttentionAttributes attrs;
    attrs.head_dim = 64;
    attrs.num_heads = 4;
    attrs.num_kv_heads = 2;
    auto op = LinearAttention::create(attrs, Backend::CPU);

    // Q: [B=1, H=4, S=1, D=64]
    TensorDesc q_desc;
    q_desc.rank = 4;
    q_desc.dims = {1, 4, 1, 64};
    q_desc.dtype = DataType::f32;
    q_desc.layout = TensorLayout::NCHW;

    // K: [B=1, H_kv=2, S=1, D=64]
    TensorDesc k_desc;
    k_desc.rank = 4;
    k_desc.dims = {1, 2, 1, 64};
    k_desc.dtype = DataType::f32;
    k_desc.layout = TensorLayout::NCHW;

    // V: same as K
    TensorDesc v_desc = k_desc;

    // Gate: [B=1, H_kv=2, S=1, D=64] — per *kv* head, not per query head. Under
    // GQA several query heads share one recurrent state, so they must share one
    // decay too; a per-query-head gate would be ambiguous.
    TensorDesc g_desc = k_desc;

    // 4 inputs: Q, K, V, Gate (State is in outputs, not inputs)
    const TensorDesc desc_arr[] = {q_desc, k_desc, v_desc, g_desc};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs.size(), size_t(2));

    // outputs[0] = Output, same shape as Q
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(4));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(4));
    NNOPS_EXPECT_EQ(descs[0].dims[2], int64_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[3], int64_t(64));
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    // outputs[1] = State [B=1, H_kv=2, D=64, D=64]
    NNOPS_EXPECT_EQ(descs[1].rank, int64_t(4));
    NNOPS_EXPECT_EQ(descs[1].dims[0], int64_t(1));
    NNOPS_EXPECT_EQ(descs[1].dims[1], int64_t(2));
    NNOPS_EXPECT_EQ(descs[1].dims[2], int64_t(64));
    NNOPS_EXPECT_EQ(descs[1].dims[3], int64_t(64));
    NNOPS_EXPECT_EQ(descs[1].dtype, DataType::f32);
}

// ============================================================
// Hand-computed case — pins the whole pipeline
// ============================================================
//
// B = H = H_kv = 1, D = 2, S = 2, no conv, no beta (=> 1.0), gate = 0 (no
// decay), scale = 1, initial state = 0.
//
//   q0 = [1, 1]  k0 = [1, 2]  v0 = [3, 4]
//   q1 = [1, 0]  k1 = [1, 1]  v1 = [1, 0]
//
// t = 0: r = 0, so d = v0 and S = k0 (x) v0 = [[3, 4], [6, 8]];
//        o0 = q0 . S = [9, 12].
// t = 1: r = S[0] + S[1] = [9, 12], d = v1 - r = [-8, -12];
//        S += k1 (x) d = [[-5, -8], [-2, -4]], o1 = q1 . S = [-5, -8].
//
// d != v at t = 1, so a kernel that dropped the (V - K@S) correction would
// produce o1 = [1, 0] and be caught.

NNOPS_TEST(linear_attention_hand_computed) {
    Problem p;
    p.attrs.scale = 1.0f;   // fixed scale keeps the expected values exact
    p.attrs.num_heads = 1;
    p.attrs.num_kv_heads = 1;
    p.B = 1; p.H = 1; p.H_kv = 1; p.S = 2; p.D = 2;
    p.CK = 0;

    p.q = {1.0f, 1.0f,  1.0f, 0.0f};   // [B,H,S,D]
    p.k = {1.0f, 2.0f,  1.0f, 1.0f};
    p.v = {3.0f, 4.0f,  1.0f, 0.0f};
    p.gate = {0.0f, 0.0f,  0.0f, 0.0f};
    p.state0.assign(4, 0.0f);

    const std::vector<float> want_out   = {9.0f, 12.0f,  -5.0f, -8.0f};
    const std::vector<float> want_state = {-5.0f, -8.0f,  -2.0f, -4.0f};

    Result got  = run(p, /*use_reference=*/false);
    Result ref  = run(p, /*use_reference=*/true);

    for (size_t i = 0; i < want_out.size(); ++i) {
        NNOPS_EXPECT_NEAR(got.out[i], want_out[i], 1e-5f);
        NNOPS_EXPECT_NEAR(ref.out[i], want_out[i], 1e-5f);
    }
    for (size_t i = 0; i < want_state.size(); ++i) {
        NNOPS_EXPECT_NEAR(got.state[i], want_state[i], 1e-5f);
        NNOPS_EXPECT_NEAR(ref.state[i], want_state[i], 1e-5f);
    }
}

// ============================================================
// Kernel vs reference (f32)
// ============================================================

NNOPS_TEST(linear_attention_matches_reference_basic) {
    // No GQA; S is long enough for the state to stop being dominated by state0.
    LinearAttentionAttributes a;
    a.head_dim = 16; a.num_heads = 2; a.num_kv_heads = 2;
    Problem p = make_problem(a, /*B=*/2, /*H=*/2, /*H_kv=*/2, /*S=*/6, /*D=*/16);
    expect_kernel_matches_reference(p, {}, "basic");
}

NNOPS_TEST(linear_attention_matches_reference_gqa) {
    // 4 query heads over 2 kv heads: each group reads one state and one shared
    // per-kv-head gate.
    LinearAttentionAttributes a;
    a.head_dim = 32; a.num_heads = 4; a.num_kv_heads = 2;
    Problem p = make_problem(a, /*B=*/2, /*H=*/4, /*H_kv=*/2, /*S=*/5, /*D=*/32);
    expect_kernel_matches_reference(p, {}, "gqa");
}

NNOPS_TEST(linear_attention_matches_reference_odd_head_dim) {
    // D = 100 is not a multiple of the SIMD lane count (8), so this exercises
    // the scalar tails in every vectorized stage.
    LinearAttentionAttributes a;
    a.head_dim = 100; a.num_heads = 3; a.num_kv_heads = 3;
    Problem p = make_problem(a, /*B=*/1, /*H=*/3, /*H_kv=*/3, /*S=*/4, /*D=*/100);
    // Reordered accumulation (vector blocks of eight, then a scalar tail, on top
    // of fma-vs-mul-add) and a state that grows past order 1: the absolute error
    // scales with the state, so 1e-4 — still far too tight to hide a real
    // indexing or recurrence error.
    expect_kernel_matches_reference(p, {}, "odd_head_dim", 1e-4f);
}

NNOPS_TEST(linear_attention_matches_reference_head_dim_one) {
    // Degenerate D = 1: the state is a 1x1 scalar, every vector tail runs.
    LinearAttentionAttributes a;
    a.head_dim = 1; a.num_heads = 2; a.num_kv_heads = 2;
    Problem p = make_problem(a, /*B=*/1, /*H=*/2, /*H_kv=*/2, /*S=*/3, /*D=*/1);
    expect_kernel_matches_reference(p, {}, "head_dim_one");
}

NNOPS_TEST(linear_attention_matches_reference_with_conv) {
    // Causal depthwise conv1d over the packed Q | K | V channels.
    LinearAttentionAttributes a;
    a.head_dim = 16; a.num_heads = 4; a.num_kv_heads = 2;
    a.conv_kernel_size = 4;
    Problem p = make_problem(a, /*B=*/1, /*H=*/4, /*H_kv=*/2, /*S=*/7, /*D=*/16,
                             /*CK=*/4);
    expect_kernel_matches_reference(p, {}, "conv");
}

NNOPS_TEST(linear_attention_matches_reference_with_beta) {
    // Explicit per-token update rate in place of the default 1.0.
    LinearAttentionAttributes a;
    a.head_dim = 16; a.num_heads = 2; a.num_kv_heads = 2;
    Problem p = make_problem(a, /*B=*/2, /*H=*/2, /*H_kv=*/2, /*S=*/5, /*D=*/16,
                             /*CK=*/0, /*has_beta=*/true);
    expect_kernel_matches_reference(p, {}, "beta");
}

NNOPS_TEST(linear_attention_matches_reference_conv_and_beta) {
    LinearAttentionAttributes a;
    a.head_dim = 24; a.num_heads = 4; a.num_kv_heads = 2;
    a.conv_kernel_size = 3;
    Problem p = make_problem(a, /*B=*/2, /*H=*/4, /*H_kv=*/2, /*S=*/6, /*D=*/24,
                             /*CK=*/3, /*has_beta=*/true);
    expect_kernel_matches_reference(p, {}, "conv+beta");
}

NNOPS_TEST(linear_attention_matches_reference_explicit_scale) {
    LinearAttentionAttributes a;
    a.head_dim = 16; a.num_heads = 2; a.num_kv_heads = 1;
    a.scale = 0.37f;   // overrides the 1/sqrt(D) default
    Problem p = make_problem(a, /*B=*/1, /*H=*/2, /*H_kv=*/1, /*S=*/4, /*D=*/16);
    expect_kernel_matches_reference(p, {}, "explicit_scale");
}

// ============================================================
// Threaded run
// ============================================================
//
// B * H_kv = 8 independent state matrices over a 4-worker pool.

NNOPS_TEST(linear_attention_matches_reference_threaded) {
    LinearAttentionAttributes a;
    a.head_dim = 16; a.num_heads = 4; a.num_kv_heads = 4;
    a.conv_kernel_size = 3;

    SimplePool pool(4);
    Problem p = make_problem(a, /*B=*/2, /*H=*/4, /*H_kv=*/4, /*S=*/8, /*D=*/16,
                             /*CK=*/3, /*has_beta=*/true);
    expect_kernel_matches_reference(p, pool.context(), "threaded");
}

// ============================================================
// Invariants
// ============================================================

// Feeding the sequence one token at a time, carrying the state forward, must
// reproduce the whole-sequence run — the core contract for incremental decoding.
//
// B = H = H_kv = 1 keeps each token's slice contiguous, so a per-token view is
// just a pointer offset. A convolution would break that: a length-1 call has no
// left context, so convolution and incremental decoding are mutually exclusive
// (see linear_attention.hpp).
NNOPS_TEST(linear_attention_incremental_decode_matches_full_sequence) {
    LinearAttentionAttributes a;
    a.head_dim = 8; a.num_heads = 1; a.num_kv_heads = 1;
    const int64_t S = 5, D = 8;

    Problem p = make_problem(a, /*B=*/1, /*H=*/1, /*H_kv=*/1, S, D,
                             /*CK=*/0, /*has_beta=*/true);
    p.state0.assign(static_cast<size_t>(D * D), 0.0f);   // decode starts from a cold state

    Result full = run(p, /*use_reference=*/false);

    std::vector<float> got_out(static_cast<size_t>(S * D), 0.0f);
    std::vector<float> state = p.state0;

    for (int64_t t = 0; t < S; ++t) {
        const std::vector<int64_t> sh_q1  = {1, 1, 1, D};
        const std::vector<int64_t> sh_kv1 = {1, 1, 1, D};
        const std::vector<int64_t> sh_bt1 = {1, 1, 1};
        const std::vector<int64_t> sh_st  = {1, 1, D, D};

        // Narrow views onto token t of each input, and into row t of the output.
        TensorView q(std::span<const int64_t>(sh_q1.data(), sh_q1.size()),
                     DataType::f32, p.q.data() + t * D, TensorLayout::NCHW);
        TensorView k(std::span<const int64_t>(sh_kv1.data(), sh_kv1.size()),
                     DataType::f32, p.k.data() + t * D, TensorLayout::NCHW);
        TensorView v(std::span<const int64_t>(sh_kv1.data(), sh_kv1.size()),
                     DataType::f32, p.v.data() + t * D, TensorLayout::NCHW);
        TensorView g(std::span<const int64_t>(sh_kv1.data(), sh_kv1.size()),
                     DataType::f32, p.gate.data() + t * D, TensorLayout::NCHW);
        TensorView b(std::span<const int64_t>(sh_bt1.data(), sh_bt1.size()),
                     DataType::f32, p.beta.data() + t, TensorLayout::NCHW);
        TensorView o(std::span<const int64_t>(sh_q1.data(), sh_q1.size()),
                     DataType::f32, got_out.data() + t * D, TensorLayout::NCHW);
        TensorView st(std::span<const int64_t>(sh_st.data(), sh_st.size()),
                      DataType::f32, state.data(), TensorLayout::NCHW);

        std::vector<TensorView> outputs = {o, st};
        // Empty conv_w slot: this problem has beta but no conv, and the slots
        // are positional.
        std::vector<TensorView> ins = {q, k, v, g, TensorView{}, b};
        auto op = LinearAttention::create(a, Backend::CPU);
        op->compute(outputs, ins, {}, nullptr);
    }

    NNOPS_EXPECT_EQ(got_out.size(), full.out.size());
    for (size_t i = 0; i < full.out.size(); ++i) {
        NNOPS_EXPECT_NEAR(got_out[i], full.out[i], 1e-5f);
    }
    for (size_t i = 0; i < full.state.size(); ++i) {
        NNOPS_EXPECT_NEAR(state[i], full.state[i], 1e-5f);
    }
}

// The carried state must actually be read: running from a nonzero state has to
// differ from running from a cold one.
NNOPS_TEST(linear_attention_initial_state_is_read) {
    LinearAttentionAttributes a;
    a.head_dim = 8; a.num_heads = 2; a.num_kv_heads = 2;

    Problem p = make_problem(a, /*B=*/1, /*H=*/2, /*H_kv=*/2, /*S=*/3, /*D=*/8);
    Result with_state = run(p, /*use_reference=*/false);

    Problem cold = p;
    cold.state0.assign(cold.state0.size(), 0.0f);
    Result from_zero = run(cold, /*use_reference=*/false);

    float diff = 0.0f;
    for (size_t i = 0; i < with_state.out.size(); ++i) {
        diff = std::max(diff, std::abs(with_state.out[i] - from_zero.out[i]));
    }
    NNOPS_EXPECT_TRUE(diff > 1e-3f);

    // ... and the kernel must agree with the reference either way.
    expect_kernel_matches_reference(p, {}, "initial_state");
}

// beta sits at index 5 and defaults to 1.0; both halves are pinned here because
// the failure mode is silent — a beta in the fifth slot is read as conv taps,
// and a kernel-vs-reference compare cannot see it, since both sides misread the
// list identically and neither has a conv to disagree about.
NNOPS_TEST(linear_attention_beta_default_is_one) {
    LinearAttentionAttributes a;
    a.head_dim = 8; a.num_heads = 2; a.num_kv_heads = 1;

    Problem p = make_problem(a, /*B=*/2, /*H=*/2, /*H_kv=*/1, /*S=*/4, /*D=*/8,
                             /*CK=*/0, /*has_beta=*/true);

    // beta == 1.0 everywhere must reproduce the no-beta default exactly: the
    // update is `1.0f * (v - r)`, and multiplying by 1.0 is exact.
    Problem ones = p;
    ones.beta.assign(ones.beta.size(), 1.0f);
    Result got = run(ones, /*use_reference=*/false);

    Problem none = p;
    none.has_beta = false;
    none.beta.clear();
    Result want = run(none, /*use_reference=*/false);

    for (size_t i = 0; i < got.out.size(); ++i) {
        NNOPS_EXPECT_EQ(got.out[i], want.out[i]);
    }
    for (size_t i = 0; i < got.state.size(); ++i) {
        NNOPS_EXPECT_EQ(got.state[i], want.state[i]);
    }

    // And a beta that is *not* 1.0 has to change the result — otherwise the
    // equality above is satisfied by ignoring the input.
    Problem half = p;
    half.beta.assign(half.beta.size(), 0.5f);
    Result shrunk = run(half, /*use_reference=*/false);

    float diff = 0.0f;
    for (size_t i = 0; i < shrunk.out.size(); ++i) {
        diff = std::max(diff, std::abs(shrunk.out[i] - want.out[i]));
    }
    NNOPS_EXPECT_TRUE(diff > 1e-3f);
}

// A gate that decays the state to nothing must erase the carried state, making
// the initial value irrelevant.
NNOPS_TEST(linear_attention_decay_forgets_initial_state) {
    LinearAttentionAttributes a;
    a.head_dim = 8; a.num_heads = 1; a.num_kv_heads = 1;

    Problem p = make_problem(a, /*B=*/1, /*H=*/1, /*H_kv=*/1, /*S=*/4, /*D=*/8,
                             /*CK=*/0, /*has_beta=*/false,
                             /*gate_lo=*/-1e4f, /*gate_hi=*/-1e4f);   // exp(-1e4) == 0

    Result with_state = run(p, /*use_reference=*/false);

    Problem cold = p;
    cold.state0.assign(cold.state0.size(), 0.0f);
    Result from_zero = run(cold, /*use_reference=*/false);

    for (size_t i = 0; i < with_state.out.size(); ++i) {
        NNOPS_EXPECT_NEAR(with_state.out[i], from_zero.out[i], 1e-4f);
    }
    for (size_t i = 0; i < with_state.state.size(); ++i) {
        NNOPS_EXPECT_NEAR(with_state.state[i], from_zero.state[i], 1e-4f);
    }

    // The reference must agree that the state was forgotten.
    expect_kernel_matches_reference(p, {}, "decay_forgets");
}

// The convolution is causal: perturbing the inputs at t >= t0 must leave every
// output at t < t0 bit-identical. This is the guard against a conv that leaks
// future tokens (e.g. by centring the kernel or reading past t).
NNOPS_TEST(linear_attention_conv_is_causal) {
    LinearAttentionAttributes a;
    a.head_dim = 8; a.num_heads = 1; a.num_kv_heads = 1;
    a.conv_kernel_size = 4;

    Problem p = make_problem(a, /*B=*/1, /*H=*/1, /*H_kv=*/1, /*S=*/6, /*D=*/8, /*CK=*/4);
    const int64_t S = p.S, D = p.D;
    const int64_t t0 = 3;

    Result before = run(p, /*use_reference=*/false);

    // Perturb Q/K/V at every token from t0 on.
    for (int64_t t = t0; t < S; ++t) {
        for (int64_t d = 0; d < D; ++d) {
            p.q[static_cast<size_t>(t * D + d)] += 5.0f;
            p.k[static_cast<size_t>(t * D + d)] -= 3.0f;
            p.v[static_cast<size_t>(t * D + d)] += 2.0f;
        }
    }
    Result after = run(p, /*use_reference=*/false);

    for (int64_t t = 0; t < t0; ++t) {
        for (int64_t d = 0; d < D; ++d) {
            const size_t i = static_cast<size_t>(t * D + d);
            NNOPS_EXPECT_NEAR(before.out[i], after.out[i], 0.0f);
        }
    }
    // The perturbation must actually have changed something, or the test is vacuous.
    float changed = 0.0f;
    for (int64_t t = t0; t < S; ++t) {
        for (int64_t d = 0; d < D; ++d) {
            const size_t i = static_cast<size_t>(t * D + d);
            changed = std::max(changed, std::abs(before.out[i] - after.out[i]));
        }
    }
    NNOPS_EXPECT_TRUE(changed > 1e-3f);
}

// ============================================================
// f16
// ============================================================
//
// The reference computes in f32 on the exact values the f16 inputs represent, so
// the comparison isolates accumulation from input quantization. The kernel
// accumulates in f32 anyway — f16 only appears at the I/O boundary — so the
// error is a couple of roundings, well inside kF16AccumTol.

NNOPS_TEST(linear_attention_f16_matches_reference) {
    LinearAttentionAttributes a;
    a.num_kv_heads = 2;
    a.conv_kernel_size = 3;

    // [B, H_q, H_k, H_v, S, Dk, Dv, CK, beta]
    struct G {
        int64_t B, H, H_kv, H_v, S, D, D_v, CK;
        bool beta;
    };
    const G cases[] = {
        {1, 2, 2, 0, 3, 8, 0, 3, true},   // plain: Dk == Dv, H_v == H_k
        {1, 4, 2, 4, 3, 8, 0, 0, false},  // Qwen3-Next head ratio
        {1, 4, 2, 4, 3, 8, 0, 3, true},   // ... with the conv over all three
        {1, 2, 2, 2, 3, 8, 4, 0, false},  // narrower value
    };

    int64_t ci = 0;
    for (const auto& c : cases) {
        const int64_t case_idx = ci++;
        Problem p32 = make_problem(a, c.B, c.H, c.H_kv, c.S, c.D, c.CK, c.beta,
                                   /*gate_lo=*/-1.0f, /*gate_hi=*/0.0f,
                                   /*seed=*/7, c.H_v, c.D_v);

        // Round every input through f16, then use the *rounded* values as the
        // f32 reference input.
        Problem p = p32;
        p.q      = test::f16_to_f32(test::f32_to_f16(p32.q));
        p.k      = test::f16_to_f32(test::f32_to_f16(p32.k));
        p.v      = test::f16_to_f32(test::f32_to_f16(p32.v));
        p.gate   = test::f16_to_f32(test::f32_to_f16(p32.gate));
        p.conv_w = test::f16_to_f32(test::f32_to_f16(p32.conv_w));
        p.beta   = test::f16_to_f32(test::f32_to_f16(p32.beta));
        p.state0 = test::f16_to_f32(test::f32_to_f16(p32.state0));

        Result want = run(p, /*use_reference=*/true);

        // Now the same problem in f16, through the optimized kernel.
        const auto sq = p.sh_q(), so = p.sh_o(), sk = p.sh_k(), sv = p.sh_v(),
                   sg = p.sh_g(), sst = p.sh_st(), scw = p.sh_cw(), sbt = p.sh_bt();

        auto q16 = test::f32_to_f16(p.q);
        auto k16 = test::f32_to_f16(p.k);
        auto v16 = test::f32_to_f16(p.v);
        auto g16 = test::f32_to_f16(p.gate);
        auto w16 = test::f32_to_f16(p.conv_w);
        auto b16 = test::f32_to_f16(p.beta);

        std::vector<half> out16(static_cast<size_t>(c.B * c.H * c.S * p.dv()));
        std::vector<half> st16 = test::f32_to_f16(p.state0);

        TensorView q  = f16_view(q16, sq);
        TensorView k  = f16_view(k16, sk);
        TensorView v  = f16_view(v16, sv);
        TensorView g  = f16_view(g16, sg);
        TensorView o  = f16_view(out16, so);
        TensorView st = f16_view(st16, sst);

        // Optional slots are positional; an absent one is a default TensorView
        // (rank 0), not a shaped view over an empty buffer.
        std::vector<TensorView> ins = {q, k, v, g};
        TensorView cw, bt;
        if (c.CK > 0) { cw = f16_view(w16, scw); }
        if (c.CK > 0 || c.beta) { ins.push_back(cw); }
        if (c.beta) {
            bt = f16_view(b16, sbt);
            ins.push_back(bt);
        }

        std::vector<TensorView> outputs = {o, st};

        auto op = LinearAttention::create(a, Backend::CPU);
        const Status sts = op->compute(outputs, ins, {}, nullptr);
        NNOPS_EXPECT_TRUE(sts == Status::ok);

        float worst_out = 0.0f, worst_st = 0.0f;
        for (size_t i = 0; i < want.out.size(); ++i) {
            worst_out = std::max(worst_out,
                                 std::abs(simd::s_load(&out16[i]) - want.out[i]));
        }
        for (size_t i = 0; i < want.state.size(); ++i) {
            worst_st = std::max(worst_st,
                                std::abs(simd::s_load(&st16[i]) - want.state[i]));
        }
        if (worst_out > test::kF16AccumTol || worst_st > test::kF16AccumTol) {
            std::cerr << "  [f16 case " << case_idx << "] max_abs_diff = "
                      << worst_out << " / " << worst_st
                      << " (tol " << test::kF16AccumTol << ")\n";
        }
        NNOPS_EXPECT_TRUE(worst_out <= test::kF16AccumTol);
        NNOPS_EXPECT_TRUE(worst_st <= test::kF16AccumTol);
    }
}

// ============================================================
// Value geometry: H_v != H_k and D_v != D_k
// ============================================================

NNOPS_TEST(linear_attention_matches_reference_more_value_heads) {
    // H_v = 2·H_k (the Qwen3-Next ratio) with the query heads expanded over the
    // value heads: each key head feeds two independent states, and its K row is
    // read once per state.
    LinearAttentionAttributes a;
    a.head_dim = 16; a.num_heads = 4; a.num_kv_heads = 2;
    Problem p = make_problem(a, /*B=*/2, /*H=*/4, /*H_kv=*/2, /*S=*/5, /*D=*/16,
                             /*CK=*/0, /*has_beta=*/false,
                             /*gate_lo=*/-2.0f, /*gate_hi=*/0.0f, /*seed=*/7,
                             /*H_v=*/4, /*D_v=*/0);
    expect_kernel_matches_reference(p, {}, "more_value_heads");
}

NNOPS_TEST(linear_attention_matches_reference_narrow_value) {
    // D_v < D_k: the state is [Dk, Dv], the decay and the rank-1 update run one
    // row per key channel but only Dv columns, and the readout is Dv wide.
    LinearAttentionAttributes a;
    a.head_dim = 32; a.num_heads = 2; a.num_kv_heads = 2;
    Problem p = make_problem(a, /*B=*/2, /*H=*/2, /*H_kv=*/2, /*S=*/5, /*D=*/32,
                             /*CK=*/0, /*has_beta=*/false,
                             /*gate_lo=*/-2.0f, /*gate_hi=*/0.0f, /*seed=*/7,
                             /*H_v=*/2, /*D_v=*/16);
    expect_kernel_matches_reference(p, {}, "narrow_value");
}

NNOPS_TEST(linear_attention_matches_reference_both_unequal) {
    // Both axes differ, with the conv and beta: the packed conv channels are
    // [H_q*Dk Q | H_k*Dk K | H_v*Dv V] and each row convolves at its own width.
    LinearAttentionAttributes a;
    a.head_dim = 24; a.num_heads = 4; a.num_kv_heads = 2;
    a.conv_kernel_size = 3;
    Problem p = make_problem(a, /*B=*/2, /*H=*/4, /*H_kv=*/2, /*S=*/6, /*D=*/24,
                             /*CK=*/3, /*has_beta=*/true,
                             /*gate_lo=*/-2.0f, /*gate_hi=*/0.0f, /*seed=*/7,
                             /*H_v=*/4, /*D_v=*/16);
    expect_kernel_matches_reference(p, {}, "both_unequal");
}

NNOPS_TEST(linear_attention_matches_reference_grouped_narrow) {
    // Two query heads over one value head (H_q = 2·H_v) *and* a narrower value:
    // both query heads read one state, at Dv columns each.
    LinearAttentionAttributes a;
    a.head_dim = 16; a.num_heads = 2; a.num_kv_heads = 1;
    Problem p = make_problem(a, /*B=*/2, /*H=*/2, /*H_kv=*/1, /*S=*/5, /*D=*/16,
                             /*CK=*/0, /*has_beta=*/false,
                             /*gate_lo=*/-2.0f, /*gate_hi=*/0.0f, /*seed=*/7,
                             /*H_v=*/1, /*D_v=*/8);
    expect_kernel_matches_reference(p, {}, "grouped_narrow");
}

// One state, two query heads: the two output rows are two different reads of the
// same state, so zeroing one query row must not move the other.
NNOPS_TEST(linear_attention_query_heads_read_shared_state) {
    LinearAttentionAttributes a;
    a.head_dim = 8; a.num_heads = 2; a.num_kv_heads = 1;

    Problem p = make_problem(a, /*B=*/1, /*H=*/2, /*H_kv=*/1, /*S=*/3, /*D=*/8);
    const int64_t dv = p.dv(), S = p.S;
    Result before = run(p, /*use_reference=*/false);

    Problem zero_q1 = p;
    for (int64_t t = 0; t < S; ++t) {
        for (int64_t d = 0; d < p.D; ++d) {
            zero_q1.q[static_cast<size_t>((1 * S + t) * p.D + d)] = 0.0f;
        }
    }
    Result after = run(zero_q1, /*use_reference=*/false);

    // Head 0 is untouched...
    for (int64_t t = 0; t < S; ++t) {
        for (int64_t d = 0; d < dv; ++d) {
            NNOPS_EXPECT_NEAR(after.out[static_cast<size_t>(t * dv + d)],
                              before.out[static_cast<size_t>(t * dv + d)], 0.0f);
        }
    }
    // ... and head 1 is now zero, so its readout contributes nothing.
    for (int64_t t = 0; t < S; ++t) {
        for (int64_t d = 0; d < dv; ++d) {
            NNOPS_EXPECT_NEAR(after.out[static_cast<size_t>((1 * S + t) * dv + d)],
                              0.0f, 0.0f);
        }
    }
    // The state itself is shared, so it did not move either.
    for (size_t i = 0; i < before.state.size(); ++i) {
        NNOPS_EXPECT_NEAR(after.state[i], before.state[i], 0.0f);
    }
}

// ============================================================
// Validation
// ============================================================

static TensorDesc td(std::initializer_list<int64_t> dims) {
    TensorDesc d;
    d.rank = static_cast<int64_t>(dims.size());
    for (int64_t v : dims) { d.dims.push_back(v); }
    d.dtype  = DataType::f32;
    d.layout = TensorLayout::NCHW;
    return d;
}

NNOPS_TEST(linear_attention_validate_value_geometry) {
    // H_q = 4, H_k = 2, H_v = 4, Dk = 16, Dv = 8: the value head owns the state
    // and the output columns.
    LinearAttentionAttributes a;
    a.head_dim = 16; a.num_heads = 4; a.num_kv_heads = 2;
    a.conv_kernel_size = 3;
    auto op = LinearAttention::create(a, Backend::CPU);

    const TensorDesc q  = td({1, 4, 5, 16});
    const TensorDesc k  = td({1, 2, 5, 16});
    const TensorDesc v  = td({1, 4, 5, 8});
    const TensorDesc g  = td({1, 4, 5, 16});
    const TensorDesc o  = td({1, 4, 5, 8});
    const TensorDesc st = td({1, 4, 16, 8});

    {
        const TensorDesc ins[] = {q, k, v, g};
        const TensorDesc outs[] = {o, st};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::ok);
    }
    {
        // The inferred descriptors agree with the ones validate accepts.
        const TensorDesc ins[] = {q, k, v, g};
        auto descs = op->getOutputTensorDesc(ins);
        NNOPS_EXPECT_EQ(descs.size(), size_t(2));
        NNOPS_EXPECT_EQ(descs[0].dims[3], int64_t(8));
        NNOPS_EXPECT_EQ(descs[1].dims[1], int64_t(4));
        NNOPS_EXPECT_EQ(descs[1].dims[2], int64_t(16));
        NNOPS_EXPECT_EQ(descs[1].dims[3], int64_t(8));
    }
    {
        // The gate decays one state row per key channel: per value head, Dk wide.
        const TensorDesc g_bad = td({1, 4, 5, 8});
        const TensorDesc ins[] = {q, k, v, g_bad};
        const TensorDesc outs[] = {o, st};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::invalid_argument);
    }
    {
        // V's head count must be a multiple of the key heads'.
        const TensorDesc v_bad = td({1, 3, 5, 8});
        const TensorDesc ins[] = {q, k, v_bad, g};
        const TensorDesc outs[] = {o, st};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::invalid_argument);
    }
    {
        // 6 query heads over 4 value heads neither expand nor follow the keys.
        const TensorDesc q_bad = td({1, 6, 5, 16});
        const TensorDesc o_bad = td({1, 6, 5, 8});
        const TensorDesc ins[] = {q_bad, k, v, g};
        const TensorDesc outs[] = {o_bad, st};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::invalid_argument);
    }
    {
        // The state is [B, H_v, Dk, Dv]; the last two axes cannot be swapped.
        const TensorDesc st_bad = td({1, 4, 8, 16});
        const TensorDesc ins[] = {q, k, v, g};
        const TensorDesc outs[] = {o, st_bad};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::invalid_argument);
    }
    {
        // Conv channels pack Q | K | V at their own widths: 4*16 + 2*16 + 4*8 = 128.
        const TensorDesc w     = td({128, 1, 3});
        const TensorDesc ins[] = {q, k, v, g, w};
        const TensorDesc outs[] = {o, st};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::ok);

        const TensorDesc w_bad = td({120, 1, 3});   // Dv taken for Dk
        const TensorDesc ins_bad[] = {q, k, v, g, w_bad};
        NNOPS_EXPECT_TRUE(op->validate(ins_bad, outs) == Status::invalid_argument);
    }
    {
        // beta: [B, H_v, S].
        const TensorDesc w     = td({128, 1, 3});
        const TensorDesc beta  = td({1, 4, 5});
        const TensorDesc ins[] = {q, k, v, g, w, beta};
        const TensorDesc outs[] = {o, st};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::ok);

        const TensorDesc b_bad = td({1, 2, 5});   // key heads, not value heads
        const TensorDesc ins_bad[] = {q, k, v, g, w, b_bad};
        NNOPS_EXPECT_TRUE(op->validate(ins_bad, outs) == Status::invalid_argument);
    }
}

// ============================================================
// Chunked prefill
// ============================================================
//
// From kChunkMinSeq tokens up the kernel runs the chunked ("WY") route. It is
// not bit-identical to the per-token loop — it reassociates the state's
// contributions into a per-chunk sum — so it is held to the same naive
// reference as everything else. NNOPS_LINEAR_ATTENTION_ROUTE forces one route
// or the other, which is how the two are compared on one problem: the automatic
// choice against the explicit arms, and the arms against each other.

namespace {

void set_route(const char* value) {
#ifdef _WIN32
    _putenv_s("NNOPS_LINEAR_ATTENTION_ROUTE", value ? value : "");
#else
    if (value) { setenv("NNOPS_LINEAR_ATTENTION_ROUTE", value, 1); }
    else { unsetenv("NNOPS_LINEAR_ATTENTION_ROUTE"); }
#endif
}

/// The kernel under a forced route; `nullptr` restores the automatic choice.
Result run_route(Problem& p, const char* route, const ComputeContext& ctx = {}) {
    set_route(route);
    Result r = run(p, /*use_reference=*/false, ctx);
    set_route(nullptr);
    return r;
}

/// Max |a - b| over the output and the state, as one number.
float result_diff(const Problem& p, Result a, Result b) {
    const auto so = p.sh_o(), sst = p.sh_st();
    TensorView av_o = f32_view(a.out, so), bv_o = f32_view(b.out, so);
    TensorView av_s = f32_view(a.state, sst), bv_s = f32_view(b.state, sst);
    return std::max(test::max_diff(av_o, bv_o), test::max_diff(av_s, bv_s));
}

}  // anonymous namespace

NNOPS_TEST(linear_attention_chunked_matches_reference) {
    // 128 tokens = four full chunks, with the head grouping, the convolution
    // and beta all live at once.
    LinearAttentionAttributes a;
    a.head_dim = 32; a.num_heads = 4; a.num_kv_heads = 2; a.conv_kernel_size = 3;
    Problem p = make_problem(a, /*B=*/2, /*H=*/4, /*H_kv=*/2, /*S=*/128, /*D=*/32,
                             /*CK=*/3, /*has_beta=*/true, /*gate_lo=*/-1.0f,
                             /*gate_hi=*/0.0f, /*seed=*/7, /*H_v=*/2, /*D_v=*/16);
    expect_kernel_matches_reference(p, {}, "chunked gqa conv beta", 1e-5f);
    // ... and that really was the chunked route.
    NNOPS_EXPECT_TRUE(result_diff(p, run_route(p, nullptr), run_route(p, "chunk")) == 0.0f);
}

NNOPS_TEST(linear_attention_chunked_tail_matches_reference) {
    // 100 tokens = three chunks and a 4-token tail: the tail chunk is where a
    // chunk shorter than kChunk has to behave.
    LinearAttentionAttributes a;
    a.head_dim = 24; a.num_heads = 2; a.num_kv_heads = 2; a.conv_kernel_size = 3;
    Problem p = make_problem(a, /*B=*/1, /*H=*/2, /*H_kv=*/2, /*S=*/100, /*D=*/24,
                             /*CK=*/3, /*has_beta=*/true);
    expect_kernel_matches_reference(p, {}, "chunked tail", 1e-5f);
}

NNOPS_TEST(linear_attention_chunked_matches_token_route) {
    // Both routes over one problem, at a key width that is not a multiple of
    // the SIMD lane count and a value width that is neither.
    LinearAttentionAttributes a;
    a.head_dim = 20; a.num_heads = 3; a.num_kv_heads = 1; a.conv_kernel_size = 2;
    Problem p = make_problem(a, /*B=*/1, /*H=*/3, /*H_kv=*/1, /*S=*/77, /*D=*/20,
                             /*CK=*/2, /*has_beta=*/true, /*gate_lo=*/-1.0f,
                             /*gate_hi=*/0.0f, /*seed=*/11, /*H_v=*/3, /*D_v=*/12);
    NNOPS_EXPECT_TRUE(result_diff(p, run_route(p, "chunk"), run_route(p, "token")) < 1e-5f);

    // Forcing the chunked route below kChunkMinSeq is legal: the whole sequence
    // is one short chunk.
    LinearAttentionAttributes b;
    b.head_dim = 16; b.num_heads = 1; b.num_kv_heads = 1;
    Problem s = make_problem(b, /*B=*/1, /*H=*/1, /*H_kv=*/1, /*S=*/5, /*D=*/16,
                             /*CK=*/0, /*has_beta=*/false);
    NNOPS_EXPECT_TRUE(result_diff(s, run_route(s, "chunk"), run_route(s, "token")) < 1e-6f);
}

NNOPS_TEST(linear_attention_chunked_long_sequence) {
    // Enough chunks that the state's error has room to accumulate.
    LinearAttentionAttributes a;
    a.head_dim = 16; a.num_heads = 2; a.num_kv_heads = 2;
    Problem p = make_problem(a, /*B=*/1, /*H=*/2, /*H_kv=*/2, /*S=*/1024, /*D=*/16,
                             /*CK=*/0, /*has_beta=*/false, /*gate_lo=*/-1.0f);
    expect_kernel_matches_reference(p, {}, "chunked 1024", 1e-5f);
}

NNOPS_TEST(linear_attention_chunked_threaded) {
    SimplePool pool(4);
    LinearAttentionAttributes a;
    a.head_dim = 16; a.num_heads = 2; a.num_kv_heads = 1; a.conv_kernel_size = 3;
    Problem p = make_problem(a, /*B=*/2, /*H=*/2, /*H_kv=*/1, /*S=*/192, /*D=*/16,
                             /*CK=*/3, /*has_beta=*/true);
    expect_kernel_matches_reference(p, pool.context(), "chunked threaded", 1e-5f);
}

NNOPS_TEST(linear_attention_chunked_f16) {
    LinearAttentionAttributes a;
    a.head_dim = 16; a.num_heads = 2; a.num_kv_heads = 2; a.conv_kernel_size = 3;
    Problem p32 = make_problem(a, /*B=*/1, /*H=*/2, /*H_kv=*/2, /*S=*/96, /*D=*/16,
                               /*CK=*/3, /*has_beta=*/true, /*gate_lo=*/-1.0f);

    // The reference reads the f16-rounded inputs, so the compare is about the
    // route and not about the input precision.
    Problem p = p32;
    p.q      = test::f16_to_f32(test::f32_to_f16(p32.q));
    p.k      = test::f16_to_f32(test::f32_to_f16(p32.k));
    p.v      = test::f16_to_f32(test::f32_to_f16(p32.v));
    p.gate   = test::f16_to_f32(test::f32_to_f16(p32.gate));
    p.conv_w = test::f16_to_f32(test::f32_to_f16(p32.conv_w));
    p.beta   = test::f16_to_f32(test::f32_to_f16(p32.beta));
    p.state0 = test::f16_to_f32(test::f32_to_f16(p32.state0));
    const Result want = run(p, /*use_reference=*/true);

    const auto sq = p.sh_q(), so = p.sh_o(), sk = p.sh_k(), sv = p.sh_v(),
               sg = p.sh_g(), sst = p.sh_st(), scw = p.sh_cw(), sbt = p.sh_bt();
    auto q16 = test::f32_to_f16(p.q);
    auto k16 = test::f32_to_f16(p.k);
    auto v16 = test::f32_to_f16(p.v);
    auto g16 = test::f32_to_f16(p.gate);
    auto w16 = test::f32_to_f16(p.conv_w);
    auto b16 = test::f32_to_f16(p.beta);
    std::vector<half> out16(static_cast<size_t>(p.B * p.H * p.S * p.dv()));
    std::vector<half> st16 = test::f32_to_f16(p.state0);

    TensorView q = f16_view(q16, sq), k = f16_view(k16, sk),
               v = f16_view(v16, sv), g = f16_view(g16, sg);
    TensorView o = f16_view(out16, so), st = f16_view(st16, sst);
    TensorView cw = f16_view(w16, scw), bt = f16_view(b16, sbt);

    std::vector<TensorView> outs = {o, st};
    const TensorView ins[] = {q, k, v, g, cw, bt};
    auto op = LinearAttention::create(p.attrs, Backend::CPU);
    NNOPS_EXPECT_TRUE(op->compute(outs, ins) == Status::ok);

    const auto got_o = test::f16_to_f32(out16);
    const auto got_s = test::f16_to_f32(st16);
    float worst = 0.0f;
    for (size_t i = 0; i < got_o.size(); ++i) {
        worst = std::max(worst, std::fabs(got_o[i] - want.out[i]));
    }
    for (size_t i = 0; i < got_s.size(); ++i) {
        worst = std::max(worst, std::fabs(got_s[i] - want.state[i]));
    }
    NNOPS_EXPECT_TRUE(worst < 2e-3f);
}

NNOPS_TEST(linear_attention_chunked_falls_back_when_decay_overflows) {
    // A gate this negative pushes one chunk's cumulative log decay past what
    // its +- scaling can hold (kChunk * 6 > 80), so the route must hand the
    // sequence back to the per-token loop before computing anything — which
    // makes the result bit-identical to the forced per-token run. A guard that
    // did not fire would scale by inf and come back with NaNs.
    LinearAttentionAttributes a;
    a.head_dim = 16; a.num_heads = 2; a.num_kv_heads = 2;
    Problem p = make_problem(a, /*B=*/1, /*H=*/2, /*H_kv=*/2, /*S=*/96, /*D=*/16,
                             /*CK=*/0, /*has_beta=*/false,
                             /*gate_lo=*/-7.0f, /*gate_hi=*/-6.0f);
    const Result auto_r = run_route(p, nullptr);
    // The first tokens still carry the initial state, so the outputs are not
    // all zeros and the compare below is not vacuous.
    float peak = 0.0f;
    for (float v : auto_r.out) { peak = std::max(peak, std::fabs(v)); }
    NNOPS_EXPECT_TRUE(peak > 1e-6f);
    NNOPS_EXPECT_TRUE(result_diff(p, auto_r, run_route(p, "token")) == 0.0f);
    expect_kernel_matches_reference(p, {}, "fallback", 1e-6f);
}

