/// @file test_moe.cpp
/// @brief Tests for the MoE operator (CPU, f32/f16).
///
/// The optimized kernel is checked against `reference::moe_ref`, which shares no
/// code with it. The top-K > 1 cases get their own tests with a *real* thread
/// pool injected through ComputeContext — under the default (sequential) context
/// a lost update in the router/accumulate step would never surface, so the
/// scheduling fix would be untested.

#include "nnops/ops/moe.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/detail/half.hpp"

#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace nnops;

// Reference kernel for comparison (test-local declaration, following the
// test_matmul.cpp / test_conv2d.cpp pattern).
namespace nnops::backend::cpu::reference {
void moe_ref(const MoEAttributes& attrs,
             TensorView& output,
             std::span<const TensorView> inputs,
             const ComputeContext& ctx,
             void* workspace);
}

// Minimal fixed-size thread pool exposing a CpuBackend (test_conv2d.cpp
// pattern). Items in [begin, end) are claimed via an atomic counter; each
// worker runs body(i) until the range is exhausted.
//
// Unlike the bench/test pools this one *rendezvouses* the workers before any
// body runs. That matters for the MoE tests: a broken kernel that accumulates
// into output[token] from the per-expert loop only loses an update if two
// workers are inside their accumulate loops at the same moment. With ordinary
// thread-creation stagger the tasks fan out and a lost update is merely
// *probable*. Measured against a knowingly-broken kernel, the stagger version
// caught it in about two runs out of three; starting every worker at the same
// instant improved that to about four out of five. Neither is a guarantee — see
// the note on moe_f32_topk_threaded.
struct SimpleMoEPool {
    explicit SimpleMoEPool(int nthreads) : nthreads_(nthreads) {
        cpu.parallel_for = [this](int64_t begin, int64_t end, const ParallelForBody& body) {
            this->parallel_for(begin, end, body);
        };
        cpu.num_threads = [this]() { return nthreads_; };
        cpu.thread_id = []() { return current_thread_id_; };
    }

    void parallel_for(int64_t begin, int64_t end, const ParallelForBody& body) {
        const int64_t count = end - begin;
        const int nw = nthreads_;

        std::atomic<int64_t> next{0};
        std::atomic<int> arrived{0};
        std::atomic<bool> go{false};

        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(nw));
        for (int t = 0; t < nw; ++t) {
            workers.emplace_back([&, t]() {
                current_thread_id_ = t;
                if (arrived.fetch_add(1, std::memory_order_acq_rel) == nw - 1) {
                    go.store(true, std::memory_order_release);
                }
                while (!go.load(std::memory_order_acquire)) { /* spin */ }
                for (;;) {
                    const int64_t i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= count) { break; }
                    body(begin + i);
                }
            });
        }
        for (auto& w : workers) { w.join(); }
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

/// One tensor of the MoE problem: the shape, and the f32 / f16 host buffers
/// behind it. Only one of the two is live at a time (see MoEProblem::dtype).
struct Buf {
    std::vector<int64_t> shape;
    std::vector<float> f32;
    std::vector<half>  f16;

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape) { n *= d; }
        return n;
    }

    void fill(uint64_t seed, float lo = -1.0f, float hi = 1.0f) {
        f32.resize(static_cast<size_t>(numel()));
        XorShift128 rng(seed);
        rng.fill_float(f32.data(), numel(), lo, hi);
    }

    void to_f16() { f16 = nnops::test::f32_to_f16(f32); }

    /// Replace the f32 values with their f16 round-trip, so a reference run sees
    /// exactly the numbers the f16 kernel saw.
    void round_trip_f16() { f32 = nnops::test::f16_to_f32(f16); }

    TensorView view(DataType dt) {
        if (dt == DataType::f16) {
            return TensorView(shape, dt, f16.data(), TensorLayout::NCHW);
        }
        return TensorView(shape, dt, f32.data(), TensorLayout::NCHW);
    }
};

/// A whole MoE problem: shapes, the nine input buffers, and which optional ones
/// are actually supplied.
struct MoEProblem {
    int64_t T = 4, H = 8, E = 2, I = 8;
    DataType dtype = DataType::f32;

    Buf input, router, fc1w, fc1b, fc2w, fc2b, fc3w, fc3b, rw;
    bool has_fc1b = true, has_fc2b = true;
    bool has_fc3w = false, has_fc3b = false, has_rw = false;

    /// Build with random data. fc3w (and its bias) is only materialized when
    /// requested — SwiGLU Separate is the only path that uses it.
    void build(uint64_t seed = 1, bool with_fc3 = false) {
        input.shape  = {T, H};
        router.shape = {T, E};
        fc1w.shape   = {E, I, H};
        fc1b.shape   = {E, I};
        fc2w.shape   = {E, H, I};
        fc2b.shape   = {E, H};
        fc3w.shape   = {E, I, H};
        fc3b.shape   = {E, I};
        rw.shape     = {T, E};

        input.fill(seed + 1);
        router.fill(seed + 2);
        fc1w.fill(seed + 3);
        fc1b.fill(seed + 4);
        fc2w.fill(seed + 5);
        fc2b.fill(seed + 6);
        fc3w.fill(seed + 7);
        fc3b.fill(seed + 8);
        rw.fill(seed + 9);

        has_fc3w = with_fc3;
        has_fc3b = with_fc3;
    }

    void to_f16_all() {
        for (Buf* b : all()) { b->to_f16(); }
        dtype = DataType::f16;
    }

    void round_trip_f16_all() {
        for (Buf* b : all()) { b->round_trip_f16(); }
        dtype = DataType::f32;
    }

    std::vector<Buf*> all() {
        std::vector<Buf*> bufs = {&input, &router, &fc1w, &fc2w};
        if (has_fc1b) { bufs.push_back(&fc1b); }
        if (has_fc2b) { bufs.push_back(&fc2b); }
        if (has_fc3w) { bufs.push_back(&fc3w); }
        if (has_fc3b) { bufs.push_back(&fc3b); }
        if (has_rw)   { bufs.push_back(&rw); }
        return bufs;
    }

    /// All nine inputs, with the absent optional ones as empty views. Passing
    /// the empties explicitly (rather than truncating) exercises the op's
    /// `is_empty()` handling.
    std::vector<TensorView> inputs(DataType dt) {
        std::vector<TensorView> ins;
        ins.reserve(9);
        ins.push_back(input.view(dt));
        ins.push_back(router.view(dt));
        ins.push_back(fc1w.view(dt));
        ins.push_back(has_fc1b ? fc1b.view(dt) : TensorView{});
        ins.push_back(fc2w.view(dt));
        ins.push_back(has_fc2b ? fc2b.view(dt) : TensorView{});
        ins.push_back(has_fc3w ? fc3w.view(dt) : TensorView{});
        ins.push_back(has_fc3b ? fc3b.view(dt) : TensorView{});
        ins.push_back(has_rw   ? rw.view(dt)   : TensorView{});
        return ins;
    }

    std::vector<int64_t> out_shape() const { return {T, H}; }
};

// ============================================================
// Runners
// ============================================================

/// Run the op (f16) or the reference (always f32) and return the output as f32.
std::vector<float> run_moe(MoEProblem& p, const MoEAttributes& attrs,
                           const ComputeContext& ctx, DataType dt, bool use_ref)
{
    auto op = MoE::create(attrs, Backend::CPU);
    auto ins = p.inputs(dt);
    const auto oshape = p.out_shape();
    const size_t n_out = static_cast<size_t>(p.T * p.H);

    std::vector<float> out_f32;
    std::vector<half>  out_f16;
    TensorView out;
    if (dt == DataType::f16) {
        out_f16.resize(n_out);
        out = TensorView(oshape, DataType::f16, out_f16.data(), TensorLayout::NCHW);
    } else {
        out_f32.assign(n_out, 0.0f);
        out = TensorView(oshape, DataType::f32, out_f32.data(), TensorLayout::NCHW);
    }

    if (use_ref) {
        nnops::backend::cpu::reference::moe_ref(attrs, out, ins, ctx, nullptr);
    } else {
        TensorView outs[] = {out};
        op->compute(outs, ins, ctx, nullptr);
    }

    if (dt == DataType::f16) { return nnops::test::f16_to_f32(out_f16); }
    return out_f32;
}

std::vector<float> run_kernel(MoEProblem& p, const MoEAttributes& attrs,
                              const ComputeContext& ctx = {})
{
    return run_moe(p, attrs, ctx, p.dtype, /*use_ref=*/false);
}

std::vector<float> run_ref(MoEProblem& p, const MoEAttributes& attrs)
{
    return run_moe(p, attrs, ComputeContext{}, DataType::f32, /*use_ref=*/true);
}

// ============================================================
// Comparison
// ============================================================

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b)
{
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::abs(a[i] - b[i]));
    }
    return worst;
}

/// allclose-style check (atol + rtol*|expected|) with a message naming the
/// worst element, so a failure is diagnosable without a debugger.
void expect_close(const std::vector<float>& got, const std::vector<float>& want,
                  float rtol, float atol, const std::string& what)
{
    if (got.size() != want.size()) {
        throw std::runtime_error(what + ": size mismatch");
    }
    bool bad = false;
    float worst_excess = 0.0f;
    size_t worst_i = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float d = std::abs(got[i] - want[i]);
        const float thr = atol + rtol * std::max(std::abs(got[i]), std::abs(want[i]));
        if (d > thr && (d - thr) > worst_excess) {
            worst_excess = d - thr;
            worst_i = i;
            bad = true;
        }
    }
    if (bad) {
        throw std::runtime_error(what + ": element " + std::to_string(worst_i) +
                                 " got " + std::to_string(got[worst_i]) +
                                 " want " + std::to_string(want[worst_i]) +
                                 " (excess " + std::to_string(worst_excess) + ")");
    }
}

/// Kernel vs reference on the same problem, the standard assertion of this file.
void expect_matches_ref(MoEProblem& p, const MoEAttributes& attrs,
                        const ComputeContext& ctx = {},
                        float rtol = 1e-4f, float atol = 1e-5f,
                        const std::string& what = "moe")
{
    auto got  = run_kernel(p, attrs, ctx);
    auto want = run_ref(p, attrs);
    expect_close(got, want, rtol, atol, what);
}

// ============================================================
// Tests
// ============================================================

// Descriptor / workspace contract: shape-preserving, dtype-preserving, and the
// op allocates no caller-side workspace (staging comes from the internal pool).
NNOPS_TEST(moe_output_desc)
{
    MoEAttributes attrs;
    auto op = MoE::create(attrs, Backend::CPU);

    TensorDesc d;
    d.layout = TensorLayout::NCHW;
    d.dtype  = DataType::f32;
    d.rank   = 2;
    d.dims   = {4, 8};

    const std::vector<TensorDesc> ins(5, d);
    const auto descs = op->getOutputTensorDesc(ins);

    NNOPS_EXPECT_EQ(descs.size(), size_t(1));
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(4));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(8));
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(op->getWorkspaceSize(ins, descs), size_t(0));
    NNOPS_EXPECT_EQ(static_cast<int>(op->getOpType()), static_cast<int>(OpType::MoE));

    // Rank is carried through as-is: only the trailing dim is the hidden size.
    TensorDesc d3 = d;
    d3.rank = 3;
    d3.dims = {2, 3, 8};
    const std::vector<TensorDesc> ins3(5, d3);
    const auto descs3 = op->getOutputTensorDesc(ins3);
    NNOPS_EXPECT_EQ(descs3[0].rank, int64_t(3));
    NNOPS_EXPECT_EQ(descs3[0].dims[2], int64_t(8));
}

// Small expert: M*N*K stays under GEMM_FAST_PATH_THRESHOLD, so the inner GEMMs
// take the f32 fast path that routes to matmul_ref — the one that reads the old
// C. Exercises the zero-before-GEMM guard against NaN-contaminated pool memory.
NNOPS_TEST(moe_f32_small_experts)
{
    MoEProblem p;
    p.T = 3; p.H = 4; p.E = 2; p.I = 4;
    p.build(/*seed=*/11);

    MoEAttributes attrs;
    attrs.k = 1;
    attrs.activation = MoEActivation::Silu;

    expect_matches_ref(p, attrs, {}, 1e-4f, 1e-5f, "moe_f32_small k=1");
}

// Large enough that H*I*n_e clears the threshold and the tiled MMA path runs.
NNOPS_TEST(moe_f32_tiled_path)
{
    MoEProblem p;
    p.T = 64; p.H = 128; p.E = 8; p.I = 256;
    p.build(/*seed=*/21);

    MoEAttributes attrs;
    attrs.k = 2;
    attrs.activation = MoEActivation::Silu;

    expect_matches_ref(p, attrs, {}, 1e-4f, 1e-4f, "moe_f32_tiled k=2");
}

// No biases at all: inputs[3] and inputs[5] arrive empty.
NNOPS_TEST(moe_f32_no_biases)
{
    MoEProblem p;
    p.T = 8; p.H = 16; p.E = 4; p.I = 16;
    p.build(/*seed=*/31);
    p.has_fc1b = false;
    p.has_fc2b = false;

    MoEAttributes attrs;
    attrs.k = 2;

    expect_matches_ref(p, attrs, {}, 1e-4f, 1e-5f, "moe_f32_no_biases");
}

// Only one of the two biases present, to pin down the optional-input indexing.
NNOPS_TEST(moe_f32_fc2_bias_only)
{
    MoEProblem p;
    p.T = 6; p.H = 16; p.E = 3; p.I = 16;
    p.build(/*seed=*/41);
    p.has_fc1b = false;

    MoEAttributes attrs;
    attrs.k = 2;

    expect_matches_ref(p, attrs, {}, 1e-4f, 1e-5f, "moe_f32_fc2_bias_only");
}

// The core top-K > 1 case, under real threads. A token served by k == E experts
// is written by every worker: accumulating straight into output[token] from the
// per-expert loop loses updates. E is pinned to the worker count so all experts
// are in flight simultaneously, and the case is repeated because a lost update
// is a timing accident, not a certainty.
//
// Detection is probabilistic, and deliberately so — no amount of shape tuning
// made it deterministic. Measured against a knowingly-broken kernel (phase A
// accumulating into the output directly), this pair of threaded tests catches it
// in roughly four runs out of five; each run is a fresh chance. That is a
// regression guard, not a proof. The proof is structural and lives elsewhere:
// moe_hand_computed pins the two-phase arithmetic for k == 2 with an exact
// expected value, and every non-threaded case compares against the reference.
NNOPS_TEST(moe_f32_topk_threaded)
{
    MoEProblem p;
    p.T = 4096; p.H = 64; p.E = 4; p.I = 8;
    p.build(/*seed=*/51);

    MoEAttributes attrs;
    attrs.k = 4;
    attrs.activation = MoEActivation::Gelu;

    SimpleMoEPool pool(4);
    ComputeContext ctx;
    ctx.cpu = pool.cpu;

    auto want = run_ref(p, attrs);
    for (int rep = 0; rep < 3; ++rep) {
        auto got = run_kernel(p, attrs, ctx);
        expect_close(got, want, 1e-4f, 1e-4f,
                     "moe_f32_topk_threaded rep " + std::to_string(rep));
    }
}

// Same, with an expert count that does not divide the worker count and a k that
// splits every expert group unevenly, so the tasks finish at different times
// instead of all at once.
NNOPS_TEST(moe_f32_topk_threaded_uneven)
{
    MoEProblem p;
    p.T = 1024; p.H = 32; p.E = 7; p.I = 8;
    p.build(/*seed=*/61);

    MoEAttributes attrs;
    attrs.k = 5;
    attrs.activation = MoEActivation::Relu;

    SimpleMoEPool pool(6);
    ComputeContext ctx;
    ctx.cpu = pool.cpu;

    auto want = run_ref(p, attrs);
    for (int rep = 0; rep < 4; ++rep) {
        auto got = run_kernel(p, attrs, ctx);
        expect_close(got, want, 1e-4f, 1e-4f,
                     "moe_f32_topk_threaded_uneven rep " + std::to_string(rep));
    }
}

// Renormalization on and off must both track the reference.
NNOPS_TEST(moe_normalize_routing_weights)
{
    MoEProblem p;
    p.T = 12; p.H = 16; p.E = 4; p.I = 16;
    p.build(/*seed=*/71);

    MoEAttributes plain;
    plain.k = 2;
    expect_matches_ref(p, plain, {}, 1e-4f, 1e-5f, "moe normalize=off");

    MoEAttributes norm = plain;
    norm.normalize_routing_weights = true;
    expect_matches_ref(p, norm, {}, 1e-4f, 1e-5f, "moe normalize=on");

    // Renormalizing the top-2 of a 4-way softmax must actually change the
    // output, otherwise the case above proves nothing.
    NNOPS_EXPECT_TRUE(max_abs_diff(run_kernel(p, plain), run_kernel(p, norm)) > 1e-3f);
}

// A separate mixing-weight table: selection still comes from softmax(logits),
// the weights come from router_weights[t, id]. Checked with and without the
// renormalization on top.
NNOPS_TEST(moe_router_weights)
{
    MoEProblem p;
    p.T = 10; p.H = 16; p.E = 4; p.I = 16;
    p.build(/*seed=*/81);
    p.has_rw = true;
    // Small positive weights in [0.1, 0.6) so renormalization is well-conditioned.
    XorShift128 rng(999);
    rng.fill_float(p.rw.f32.data(), p.rw.numel(), 0.1f, 0.6f);

    MoEAttributes attrs;
    attrs.k = 2;
    expect_matches_ref(p, attrs, {}, 1e-4f, 1e-5f, "moe router_weights");

    attrs.normalize_routing_weights = true;
    expect_matches_ref(p, attrs, {}, 1e-4f, 1e-5f, "moe router_weights + normalize");

    // router_weights must be *used*: q has the same seed but no weight table, so
    // it falls back to the softmax probabilities and must differ.
    MoEProblem q;
    q.T = 10; q.H = 16; q.E = 4; q.I = 16;
    q.build(/*seed=*/81);

    MoEAttributes plain;
    plain.k = 2;
    NNOPS_EXPECT_TRUE(max_abs_diff(run_kernel(p, plain), run_kernel(q, plain)) > 1e-3f);
}

// Every router gating function must track the reference, and each must actually
// change the result — a kernel-vs-reference comparison alone would pass
// vacuously if the attribute were ignored, since both sides would ignore it
// together.
NNOPS_TEST(moe_router_gating_functions)
{
    const MoERouterGating modes[] = {MoERouterGating::Softmax, MoERouterGating::Sigmoid,
                                     MoERouterGating::SqrtSoftplus};
    const char* names[] = {"softmax", "sigmoid", "sqrt_softplus"};

    MoEProblem p;
    p.T = 12; p.H = 16; p.E = 4; p.I = 16;
    p.build(/*seed=*/61);

    std::vector<std::vector<float>> outs;
    for (int i = 0; i < 3; ++i) {
        MoEAttributes attrs;
        attrs.k = 2;
        attrs.router_gating = modes[i];
        expect_matches_ref(p, attrs, {}, 1e-4f, 1e-5f,
                           std::string("moe gating ") + names[i]);
        outs.push_back(run_kernel(p, attrs));
    }

    // Pairwise distinct: the three score the same logits differently, so their
    // mixing weights differ (softmax renormalizes the row, the other two do not).
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            NNOPS_EXPECT_TRUE(max_abs_diff(outs[i], outs[j]) > 1e-3f);
        }
    }
}

// Hand-computed routing weights, one gating function at a time, on a problem
// where the output *is* the selected routing weight:
//
//   T = 1, H = 1, E = 3, I = 1, k = 1, Identity activation, no biases
//   x = [1], every expert weight = 1  =>  out = the selected expert's weight
//
//   logits = [0, 1, 2]  ->  argmax is expert 2 under all three functions:
//     softmax      e^2/(1+e+e^2)        = 0.665240956
//     sigmoid      1/(1+e^-2)           = 0.880797078
//     sqrt_softplus sqrt(ln(1+e^2))     = 1.458399418
//
// The three are all monotonic in the logits, so they select the *same* expert
// and only disagree on the weight — which is exactly the property being pinned
// here. (It also means a 2-expert case would prove nothing: softmax([0,b]) is
// sigmoid(b), so the two coincide.)
NNOPS_TEST(moe_sigmoid_routing_hand_computed)
{
    MoEProblem p;
    p.T = 1; p.H = 1; p.E = 3; p.I = 1;
    p.build(/*seed=*/601);
    p.has_fc1b = false;
    p.has_fc2b = false;

    p.input.f32  = {1.0f};
    p.router.f32 = {0.0f, 1.0f, 2.0f};
    p.fc1w.f32   = {1.0f, 1.0f, 1.0f};
    p.fc2w.f32   = {1.0f, 1.0f, 1.0f};

    const MoERouterGating modes[] = {MoERouterGating::Softmax, MoERouterGating::Sigmoid,
                                     MoERouterGating::SqrtSoftplus};
    const float expect[] = {0.665240956f, 0.880797078f, 1.458399418f};

    for (int i = 0; i < 3; ++i) {
        MoEAttributes attrs;
        attrs.k = 1;
        attrs.activation = MoEActivation::Identity;
        attrs.router_gating = modes[i];

        auto got = run_kernel(p, attrs);
        NNOPS_EXPECT_EQ(got.size(), size_t(1));
        NNOPS_EXPECT_NEAR(got[0], expect[i], 1e-5f);

        // Same numbers through the reference: the two agree on the score, not
        // just on the final value.
        expect_close(got, run_ref(p, attrs), 0.0f, 1e-5f, "moe gating hand computed");
    }
}

// The routed scaling factor is llama.cpp's w_scale / DeepSeek-V3's 2.5. It is a
// pure multiplier on the routed output — the experts are combined linearly in
// the routing weight — which pins its semantics independently of the reference.
NNOPS_TEST(moe_routed_scaling_factor)
{
    MoEProblem p;
    p.T = 10; p.H = 16; p.E = 4; p.I = 16;
    p.build(/*seed=*/91);

    MoEAttributes attrs;
    attrs.k = 2;
    expect_matches_ref(p, attrs, {}, 1e-4f, 1e-5f, "moe scale=1");

    MoEAttributes scaled = attrs;
    scaled.routed_scaling_factor = 2.5f;
    expect_matches_ref(p, scaled, {}, 1e-4f, 1e-5f, "moe scale=2.5");

    auto plain_out  = run_kernel(p, attrs);
    auto scaled_out = run_kernel(p, scaled);
    NNOPS_EXPECT_TRUE(max_abs_diff(plain_out, scaled_out) > 1e-3f);
    for (size_t i = 0; i < plain_out.size(); ++i) {
        NNOPS_EXPECT_NEAR(scaled_out[i], 2.5f * plain_out[i], 1e-4f);
    }
}

// The DeepSeek-V3 combination: sigmoid scoring, renormalize the top-K, then
// scale. Sigmoid scores do not sum to 1 and the renormalization is what puts
// them back on the softmax scale — so it must genuinely matter here.
NNOPS_TEST(moe_sigmoid_routing_normalized)
{
    MoEProblem p;
    p.T = 12; p.H = 16; p.E = 8; p.I = 16;
    p.build(/*seed=*/101);

    MoEAttributes attrs;
    attrs.k = 2;
    attrs.router_gating = MoERouterGating::Sigmoid;
    attrs.normalize_routing_weights = true;
    attrs.routed_scaling_factor = 2.5f;
    expect_matches_ref(p, attrs, {}, 1e-4f, 1e-5f, "moe sigmoid+normalize+scale");

    MoEAttributes raw = attrs;
    raw.normalize_routing_weights = false;
    NNOPS_EXPECT_TRUE(max_abs_diff(run_kernel(p, attrs), run_kernel(p, raw)) > 1e-3f);
}

// SqrtSoftplus over logits that run past the softplus guard (llama.cpp's
// ggml_compute_softplus_f32 switches to the identity above 20 to keep
// log1p(exp(x)) from overflowing). The reference takes the same branch, so
// agreement alone proves nothing here — the finiteness check is what catches a
// missing guard, since exp(100) is inf and sqrt(inf) is still inf.
//
// The logits are laid out explicitly rather than randomly: the guard only
// matters above ln(FLT_MAX) ~ 88.7, which a uniform draw of f32 values reaches
// only some of the time. A random fill here would be a test that passes or
// fails by luck.
NNOPS_TEST(moe_sqrt_softplus_routing_large_logits)
{
    MoEProblem p;
    p.T = 6; p.H = 8; p.E = 4; p.I = 8;
    p.build(/*seed=*/111);
    p.router.f32 = {
        -1000.0f,  -50.0f,    0.0f,   95.0f,   // -1000 underflows to softplus 0
         -95.0f,   50.0f,   20.0f,  100.0f,
          80.0f,   88.0f,   90.0f,  120.0f,   // 88.7 is the overflow threshold
         -120.0f,  -90.0f,  -20.0f,   30.0f,
           0.0f,     0.0f,    0.0f,    0.0f,
         200.0f, -200.0f,   89.0f,  -89.0f,
    };

    MoEAttributes attrs;
    attrs.k = 2;
    attrs.router_gating = MoERouterGating::SqrtSoftplus;
    expect_matches_ref(p, attrs, {}, 1e-4f, 1e-5f, "moe sqrt_softplus large logits");

    // sqrt(softplus(95)) is sqrt(95) ~ 9.7 under the guard; without it the score
    // is inf and so is the output.
    for (float v : run_kernel(p, attrs)) {
        NNOPS_EXPECT_TRUE(std::isfinite(v));
    }
}

// Every non-gated activation.
NNOPS_TEST(moe_activations)
{
    const MoEActivation acts[] = {MoEActivation::Relu, MoEActivation::Gelu,
                                  MoEActivation::Silu, MoEActivation::Identity};
    const char* names[] = {"relu", "gelu", "silu", "identity"};

    for (int i = 0; i < 4; ++i) {
        MoEProblem p;
        p.T = 9; p.H = 16; p.E = 3; p.I = 16;
        p.build(static_cast<uint64_t>(100 + i));

        MoEAttributes attrs;
        attrs.k = 2;
        attrs.activation = acts[i];

        expect_matches_ref(p, attrs, {}, 1e-4f, 1e-5f,
                           std::string("moe activation ") + names[i]);
    }
}

// SwiGLU with the Separate (fc3) layout, including the non-default GPT-OSS
// scaling and a finite clamp — the clamping branch is otherwise never entered.
NNOPS_TEST(moe_swiglu_separate)
{
    MoEProblem p;
    p.T = 12; p.H = 16; p.E = 4; p.I = 24;
    p.build(/*seed=*/201, /*with_fc3=*/true);

    MoEAttributes attrs;
    attrs.k = 2;
    attrs.activation = MoEActivation::SwiGLU;
    attrs.swiglu_layout = SwiGLULayout::Separate;

    // Standard SwiGLU first (alpha=1, beta=0, no clamp).
    expect_matches_ref(p, attrs, {}, 1e-4f, 1e-5f, "moe swiglu swiglu");

    // GPT-OSS parameters, with the clamp active.
    attrs.activation_alpha = 1.702f;
    attrs.activation_beta  = 1.0f;
    attrs.swiglu_limit     = 7.0f;
    expect_matches_ref(p, attrs, {}, 1e-4f, 1e-5f, "moe swiglu gpt-oss params");

    // A very tight limit forces the clamp on essentially every element, so the
    // clamped and unclamped results must differ.
    MoEAttributes tight = attrs;
    tight.swiglu_limit = 0.1f;
    NNOPS_EXPECT_TRUE(max_abs_diff(run_kernel(p, attrs), run_kernel(p, tight)) > 1e-3f);
}

// Every token routed to one expert, leaving the others with n_e == 0. The
// empty-expert path must skip its GEMMs rather than launch zero-sized ones.
NNOPS_TEST(moe_skewed_routing_empty_experts)
{
    MoEProblem p;
    p.T = 16; p.H = 16; p.E = 4; p.I = 16;
    p.build(/*seed=*/301);

    // Expert 0 dominates; experts 1..3 are pushed far below it.
    for (int64_t t = 0; t < p.T; ++t) {
        for (int64_t e = 0; e < p.E; ++e) {
            p.router.f32[static_cast<size_t>(t * p.E + e)] = (e == 0) ? 20.0f : -20.0f;
        }
    }

    MoEAttributes attrs;
    attrs.k = 1;

    expect_matches_ref(p, attrs, {}, 1e-4f, 1e-5f, "moe skewed routing");

    // Sanity: the empty experts really were skipped. Every token went to expert
    // 0, so scrambling the weights of experts 1..3 must not move the output.
    auto before = run_kernel(p, attrs);
    for (size_t i = static_cast<size_t>(p.I * p.H); i < p.fc1w.f32.size(); ++i) {
        p.fc1w.f32[i] = 1234.0f;
    }
    for (size_t i = static_cast<size_t>(p.H * p.I); i < p.fc2w.f32.size(); ++i) {
        p.fc2w.f32[i] = -1234.0f;
    }
    auto after = run_kernel(p, attrs);
    expect_close(after, before, 0.0f, 0.0f, "moe skewed routing ignores empty experts");
}

// Ties go to the lower expert index. Both experts carry identical weights, so
// the only thing distinguishing them is the mixing weight this test injects via
// router_weights: rw[t, e] = e + 1. Selecting expert 0 scales by 1, expert 1 by 2.
NNOPS_TEST(moe_tie_break_prefers_lower_index)
{
    MoEProblem p;
    p.T = 6; p.H = 8; p.E = 2; p.I = 8;
    p.build(/*seed=*/401);
    p.has_rw = true;

    // Identical experts: copy expert 0's weights over expert 1's.
    const auto copy_expert = [](Buf& b, int64_t per_expert) {
        std::copy_n(b.f32.begin(), static_cast<size_t>(per_expert),
                    b.f32.begin() + static_cast<size_t>(per_expert));
    };
    copy_expert(p.fc1w, p.I * p.H);
    copy_expert(p.fc1b, p.I);
    copy_expert(p.fc2w, p.H * p.I);
    copy_expert(p.fc2b, p.H);

    // Identical logits -> an exact tie in the softmax.
    std::fill(p.router.f32.begin(), p.router.f32.end(), 0.0f);

    MoEAttributes attrs;
    attrs.k = 1;

    // Weight table A: expert 0 -> 1.0, expert 1 -> 2.0.
    for (int64_t t = 0; t < p.T; ++t) {
        p.rw.f32[static_cast<size_t>(t * 2 + 0)] = 1.0f;
        p.rw.f32[static_cast<size_t>(t * 2 + 1)] = 2.0f;
    }
    auto lower = run_kernel(p, attrs);
    auto lower_ref = run_ref(p, attrs);  // must see table A as well

    // Weight table B: expert 0 -> 2.0, expert 1 -> 1.0. Same weights, swapped.
    for (int64_t t = 0; t < p.T; ++t) {
        p.rw.f32[static_cast<size_t>(t * 2 + 0)] = 2.0f;
        p.rw.f32[static_cast<size_t>(t * 2 + 1)] = 1.0f;
    }
    auto alt = run_kernel(p, attrs);

    // Table A must land on expert 0's weight (1.0), table B on the same expert's
    // weight (2.0): so alt == 2 * lower, and neither is the expert-1 branch.
    NNOPS_EXPECT_TRUE(max_abs_diff(lower, alt) > 1e-3f);
    for (size_t i = 0; i < lower.size(); ++i) {
        NNOPS_EXPECT_NEAR(alt[i], 2.0f * lower[i], 1e-5f);
    }

    // And the reference must agree with the kernel on the same tie.
    expect_close(lower, lower_ref, 1e-4f, 1e-5f, "moe tie-break");
}

// Hand-computed 2x2 case: no pooling, no accumulation subtleties, just the
// routing arithmetic checked against numbers worked out on paper.
//
//   x = [1, 2], logits = [0, 0] -> p = [0.5, 0.5], k = 2 (both, in index order)
//   expert 0: W1 = I, W2 = I        -> out0 = [1, 2]
//   expert 1: W1 = swap, W2 = I     -> out1 = [2, 1]
//   output = 0.5*[1,2] + 0.5*[2,1]  = [1.5, 1.5]
//
// Both experts serve the same single token, which is exactly the case that a
// per-expert `out[token] +=` update would corrupt.
NNOPS_TEST(moe_hand_computed)
{
    MoEProblem p;
    p.T = 1; p.H = 2; p.E = 2; p.I = 2;
    p.build(/*seed=*/501);
    p.has_fc1b = false;
    p.has_fc2b = false;

    p.input.f32 = {1.0f, 2.0f};
    p.router.f32 = {0.0f, 0.0f};

    p.fc1w.f32 = {1.0f, 0.0f,   0.0f, 1.0f,    // expert 0: identity
                  0.0f, 1.0f,   1.0f, 0.0f};  // expert 1: swap
    p.fc2w.f32 = {1.0f, 0.0f,   0.0f, 1.0f,    // expert 0: identity
                  1.0f, 0.0f,   0.0f, 1.0f};  // expert 1: identity

    MoEAttributes attrs;
    attrs.k = 2;
    attrs.activation = MoEActivation::Identity;

    auto got = run_kernel(p, attrs);
    NNOPS_EXPECT_EQ(got.size(), size_t(2));
    NNOPS_EXPECT_NEAR(got[0], 1.5f, 1e-6f);
    NNOPS_EXPECT_NEAR(got[1], 1.5f, 1e-6f);

    // Same numbers via the reference, so the two implementations agree on the
    // routing order and the mixing arithmetic, not just on the final value.
    auto want = run_ref(p, attrs);
    expect_close(got, want, 0.0f, 1e-6f, "moe hand computed");
}

// A single expert with k == 1: the degenerate case where top-K is trivial.
NNOPS_TEST(moe_single_expert)
{
    MoEProblem p;
    p.T = 5; p.H = 16; p.E = 1; p.I = 16;
    p.build(/*seed=*/601);

    MoEAttributes attrs;
    attrs.k = 1;

    expect_matches_ref(p, attrs, {}, 1e-4f, 1e-5f, "moe single expert");
}

// f16: the kernel runs the whole FC1 -> activation -> FC2 chain in f16 (aarch64
// accumulates in native fp16), while the reference computes the same problem in
// f32. Both see bit-identical inputs — the f16 buffers are converted straight
// back to f32 for the reference — so the tolerance only has to cover the f16
// chain, and kF16AccumTol is the arch-specific bound the other f16 tests use.
// It is a loose bound by design (it is sized for K near 256; this case runs
// K = 32 twice), so the margin is wide: measured at 0.010 against a 0.2 budget
// on aarch64. Tightening it here would just re-derive a second arch-dependent
// constant for no extra coverage.
NNOPS_TEST(moe_f16)
{
    MoEProblem p;
    p.T = 8; p.H = 32; p.E = 4; p.I = 32;
    p.build(/*seed=*/701);

    MoEAttributes attrs;
    attrs.k = 2;
    attrs.activation = MoEActivation::Silu;

    p.to_f16_all();
    auto got = run_kernel(p, attrs);

    // Feed the reference exactly the values the kernel consumed.
    p.round_trip_f16_all();
    auto want = run_ref(p, attrs);

    expect_close(got, want, 0.0f, nnops::test::kF16AccumTol, "moe f16");
}

}  // anonymous namespace
