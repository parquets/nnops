/// Unit tests for the LinearAttention operator.
///
/// The SIMD kernel is checked against `reference::linear_attention_ref`, which
/// lives in its own translation unit and shares no code with it — so agreement
/// is evidence about the math rather than about a shared bug. On top of that sit
/// a hand-computed case, a set of invariants (incremental decoding, causality of
/// the convolution, decay semantics), an f16 case, and a genuinely threaded run.

#include "nnops/ops/linear_attention.hpp"
#include "common/test_harness.hpp"
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

// Reference kernel for comparison (test-local declaration, following the
// test_moe.cpp / test_matmul.cpp pattern).
namespace nnops::backend::cpu::reference {
void linear_attention_ref(const LinearAttentionAttributes& attrs,
                          std::span<TensorView> outputs,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx,
                          void* workspace);
}

// Minimal fixed-size thread pool exposing a CpuBackend (test_conv2d.cpp /
// test_moe.cpp pattern). Items in [begin, end) are claimed via an atomic
// counter. Without a pool like this the op would be exercised only through the
// default ComputeContext, whose `run()` degrades to a sequential loop — the
// parallel path would never execute.
struct SimplePool {
    explicit SimplePool(int nthreads) : nthreads_(nthreads) {
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

    /// A ComputeContext wired to this pool's hooks.
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

/// One LinearAttention problem: attributes, every input buffer, and the initial
/// recurrent state. Shapes are derived from (B, H, H_kv, S, D).
struct Problem {
    LinearAttentionAttributes attrs;
    int64_t B = 1, H = 1, H_kv = 1, S = 4, D = 8;
    int64_t CK = 0;          // convolution taps; 0 => the conv weights are absent
    bool has_beta = false;

    std::vector<float> q, k, v, gate, conv_w, beta, state0;

    std::vector<int64_t> sh_q()  const { return {B, H, S, D}; }
    std::vector<int64_t> sh_kv() const { return {B, H_kv, S, D}; }
    std::vector<int64_t> sh_st() const { return {B, H_kv, D, D}; }
    std::vector<int64_t> sh_cw() const { return {(H + 2 * H_kv) * D, 1, CK}; }
    std::vector<int64_t> sh_bt() const { return {B, H_kv, S}; }
};

void fill(std::vector<float>& buf, int64_t n, float lo, float hi, uint64_t seed) {
    buf.assign(static_cast<size_t>(n), 0.0f);
    XorShift128 rng(seed);
    rng.fill_float(buf.data(), n, lo, hi);
}

/// Random problem. `gate_lo/hi` control the decay: exponentiate the gate, so a
/// range of ~[-2, 0] gives decay factors in ~[0.14, 1].
Problem make_problem(LinearAttentionAttributes attrs, int64_t B, int64_t H, int64_t H_kv,
                     int64_t S, int64_t D, int64_t CK = 0, bool has_beta = false,
                     float gate_lo = -2.0f, float gate_hi = 0.0f, uint64_t seed = 7)
{
    Problem p;
    p.attrs = attrs;
    p.B = B; p.H = H; p.H_kv = H_kv; p.S = S; p.D = D;
    p.CK = CK; p.has_beta = has_beta;

    fill(p.q, B * H * S * D, -1.0f, 1.0f, seed + 1);
    fill(p.k, B * H_kv * S * D, -1.0f, 1.0f, seed + 2);
    fill(p.v, B * H_kv * S * D, -1.0f, 1.0f, seed + 3);
    fill(p.gate, B * H_kv * S * D, gate_lo, gate_hi, seed + 4);
    fill(p.state0, B * H_kv * D * D, -1.0f, 1.0f, seed + 5);
    if (CK > 0) {
        fill(p.conv_w, (H + 2 * H_kv) * D * CK, -0.5f, 0.5f, seed + 6);
    }
    if (has_beta) {
        fill(p.beta, B * H_kv * S, 0.0f, 1.0f, seed + 7);
    }
    return p;
}

struct Result {
    std::vector<float> out, state;
};

/// Run the problem through either the optimized kernel or the reference.
Result run(Problem& p, bool use_reference, const ComputeContext& ctx = {})
{
    const auto sq = p.sh_q(), skv = p.sh_kv(), sst = p.sh_st();

    Result r;
    r.out.assign(static_cast<size_t>(p.B * p.H * p.S * p.D), 0.0f);
    r.state = p.state0;   // State is read, then overwritten in place

    TensorView q  = f32_view(p.q, sq);
    TensorView k  = f32_view(p.k, skv);
    TensorView v  = f32_view(p.v, skv);
    TensorView g  = f32_view(p.gate, skv);
    TensorView o  = f32_view(r.out, sq);
    TensorView st = f32_view(r.state, sst);

    std::vector<TensorView> outputs = {o, st};
    std::vector<TensorView> ins = {q, k, v, g};
    // The optional slots are positional (conv_w is index 4, beta is index 5),
    // so conv_w's slot has to be occupied — by an empty view — whenever beta is
    // supplied. Pushing beta alone would alias it into the conv_w slot.
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

/// Kernel vs reference on the same data, for both outputs.
///
/// `atol` is in absolute terms, and for the recurrence that is the meaningful
/// one: the retrieval/delta step subtracts two nearly equal quantities
/// (`v[j] - r[j]`), so an entry that ends up near zero still carries an absolute
/// error set by the magnitude of the terms that cancelled — roughly `eps` times
/// the largest state entry, not `eps` times its own. The default atol only has
/// to cover a state of order 1; a problem whose state grows larger needs a
/// larger one to be comparable at all.
void expect_kernel_matches_reference(Problem& p, const ComputeContext& ctx = {},
                                     const std::string& what = "linear_attention",
                                     float atol = 1e-6f)
{
    Result want = run(p, /*use_reference=*/true);
    Result got  = run(p, /*use_reference=*/false, ctx);

    const auto sq = p.sh_q(), sst = p.sh_st();
    {
        TensorView gv = f32_view(got.out, sq);
        TensorView wv = f32_view(want.out, sq);
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
// B = H = H_kv = 1, D = 2, S = 2, no convolution, no beta (=> 1.0), gate = 0
// (so exp(gate) = 1, i.e. no decay), scale = 1, initial state = 0.
//
//   q0 = [1, 1]  k0 = [1, 2]  v0 = [3, 4]
//   q1 = [1, 0]  k1 = [1, 1]  v1 = [1, 0]
//
// t = 0: S = 0, so r = 0 and d = v0. S = k0 (x) v0 = [[3, 4], [6, 8]].
//        o0 = q0 . S = [1*3 + 1*6, 1*4 + 1*8] = [9, 12].
// t = 1: r[j] = S[0][j]*k1[0] + S[1][j]*k1[1] = S[0][j] + S[1][j] = [9, 12].
//        d = v1 - r = [1-9, 0-12] = [-8, -12].
//        S += k1 (x) d, k1 = [1, 1]:
//          row 0: [3, 4]   + [-8, -12] = [-5, -8]
//          row 1: [6, 8]   + [-8, -12] = [-2, -4]
//        o1 = q1 . S_new = row 0 = [-5, -8].
//
// The delta term matters here: d != v at t = 1, so a kernel that dropped the
// (V - K@S) correction would produce o1 = [1, 0] and be caught.

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
    // No GQA, moderate D, sequence long enough for the state to stop being
    // dominated by the initial value.
    LinearAttentionAttributes a;
    a.head_dim = 16; a.num_heads = 2; a.num_kv_heads = 2;
    Problem p = make_problem(a, /*B=*/2, /*H=*/2, /*H_kv=*/2, /*S=*/6, /*D=*/16);
    expect_kernel_matches_reference(p, {}, "basic");
}

NNOPS_TEST(linear_attention_matches_reference_gqa) {
    // 4 query heads over 2 kv heads: the group's query heads each read the same
    // state, and the decay comes from the single shared per-kv-head gate.
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
    // The tail-free stages reduce in different orders here (vector blocks of
    // eight, then a scalar tail) on top of fma-vs-mul-add, and a 100-wide
    // recurrence lets the state reach ~25. The measured kernel/reference gap is
    // 4.8e-7 at S=1, 1.4e-6 at S=2 and 2.0e-5 at S=4 — i.e. it tracks the state
    // magnitude (2e-5/24.4 is under 8 ulp), which is exactly the absolute error
    // a cancelling `v[j] - r[j]` leaves on a near-zero entry. An atol of 1e-4 is
    // therefore about four ulp of the largest state entry: loose enough to admit
    // reordering, far too tight to hide a real indexing or recurrence error.
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
// B * H_kv = 8 independent state matrices over a 4-worker pool. Without a real
// pool the kernel's parallel path is never entered.

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

// Feeding the sequence one token at a time, carrying the State forward, must
// reproduce the whole-sequence run. This is the operator's core contract for
// incremental decoding — and with `S == 1` it is also the only thing that pins
// the "read the old state, overwrite in place" semantics of outputs[1].
//
// B = H = H_kv = 1 keeps each token's slice contiguous, so a per-token view is
// just a pointer offset. (With a convolution this would not hold: a length-1
// call has no left context to convolve against. Convolution and incremental
// decoding are therefore mutually exclusive — see linear_attention.hpp.)
NNOPS_TEST(linear_attention_incremental_decode_matches_full_sequence) {
    LinearAttentionAttributes a;
    a.head_dim = 8; a.num_heads = 1; a.num_kv_heads = 1;
    const int64_t S = 5, D = 8;

    Problem p = make_problem(a, /*B=*/1, /*H=*/1, /*H_kv=*/1, S, D,
                             /*CK=*/0, /*has_beta=*/true);
    p.state0.assign(static_cast<size_t>(D * D), 0.0f);   // decode starts from a cold state

    Result full = run(p, /*use_reference=*/false);

    // Step the same problem one token at a time.
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
        // Empty conv_w slot: this problem has beta but no convolution, and the
        // optional slots are positional.
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

// The carried state must actually be read, not assumed zero. Running from a
// nonzero state has to differ from running from a cold one.
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

// beta occupies index 5 — the *sixth* slot — and the default is 1.0. Both halves
// of that are pinned here, because the failure mode is silent: a beta supplied
// as the fifth entry lands in the conv_w slot and gets read as tap weights, and
// a kernel-vs-reference comparison cannot see it (both sides misread the list
// identically, and neither has a conv to disagree about). Concretely, an
// implementation that aliased the two slots would multiply the outputs by a
// five-tap convolution of all-ones weights rather than by beta.
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
// The kernel takes f16 input/output; the reference computes in f32 on the exact
// values those f16 inputs represent (so the comparison isolates accumulation,
// not input quantization). Everything inside the kernel accumulates in f32
// anyway — f16 only appears at the I/O boundary — so the error is bounded by a
// couple of roundings, well inside kF16AccumTol.

NNOPS_TEST(linear_attention_f16_matches_reference) {
    LinearAttentionAttributes a;
    a.head_dim = 8; a.num_heads = 2; a.num_kv_heads = 2;
    a.conv_kernel_size = 3;

    const int64_t B = 1, H = 2, H_kv = 2, S = 3, D = 8, CK = 3;

    Problem p32 = make_problem(a, B, H, H_kv, S, D, CK, /*has_beta=*/true,
                               /*gate_lo=*/-1.0f, /*gate_hi=*/0.0f);

    // Round every input through f16, then use the *rounded* values as the f32
    // reference input.
    Problem p = p32;
    p.q      = test::f16_to_f32(test::f32_to_f16(p32.q));
    p.k      = test::f16_to_f32(test::f32_to_f16(p32.k));
    p.v      = test::f16_to_f32(test::f32_to_f16(p32.v));
    p.gate   = test::f16_to_f32(test::f32_to_f16(p32.gate));
    p.conv_w = test::f16_to_f32(test::f32_to_f16(p32.conv_w));
    p.beta   = test::f16_to_f32(test::f32_to_f16(p32.beta));

    Result want = run(p, /*use_reference=*/true);

    // Now the same problem in f16, through the optimized kernel.
    const auto sq = p.sh_q(), skv = p.sh_kv(), sst = p.sh_st(),
               scw = p.sh_cw(), sbt = p.sh_bt();

    auto q16 = test::f32_to_f16(p.q);
    auto k16 = test::f32_to_f16(p.k);
    auto v16 = test::f32_to_f16(p.v);
    auto g16 = test::f32_to_f16(p.gate);
    auto w16 = test::f32_to_f16(p.conv_w);
    auto b16 = test::f32_to_f16(p.beta);

    std::vector<half> out16(static_cast<size_t>(B * H * S * D));
    std::vector<half> st16 = test::f32_to_f16(p.state0);

    TensorView q  = f16_view(q16, sq);
    TensorView k  = f16_view(k16, skv);
    TensorView v  = f16_view(v16, skv);
    TensorView g  = f16_view(g16, skv);
    TensorView cw = f16_view(w16, scw);
    TensorView bt = f16_view(b16, sbt);
    TensorView o  = f16_view(out16, sq);
    TensorView st = f16_view(st16, sst);

    std::vector<TensorView> outputs = {o, st};
    std::vector<TensorView> ins = {q, k, v, g, cw, bt};

    auto op = LinearAttention::create(a, Backend::CPU);
    op->compute(outputs, ins, {}, nullptr);

    auto expect_f16 = [&](std::vector<half>& got, std::vector<float>& ref,
                          const char* what) {
        float worst = 0.0f;
        for (size_t i = 0; i < ref.size(); ++i) {
            worst = std::max(worst, std::abs(simd::s_load(&got[i]) - ref[i]));
        }
        std::cerr << "  [f16 " << what << "] max_abs_diff = " << worst
                  << " (tol " << test::kF16AccumTol << ")\n";
        NNOPS_EXPECT_TRUE(worst <= test::kF16AccumTol);
    };

    expect_f16(out16, want.out, "output");
    expect_f16(st16, want.state, "state");
}
