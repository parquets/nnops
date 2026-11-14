/// @file bench_linear_attention.cpp
/// @brief LinearAttention prefill and decode on the Gated Delta Net shapes.
///
/// The op carries one state matrix per (batch, value head), so the task count is
/// B * H_v and nothing else: a 32-value-head layer on a 20-thread host is two
/// tasks short of a full wave, while an 8-head layer leaves more than half the
/// machine idle. Each row here is one such geometry over the sequence lengths a
/// server sees — decode (S = 1), a short prompt, and a 4k prefill — with the
/// per-token recurrence's own arithmetic as the denominator, so the GFLOPS
/// column says how much of the algebra the machine actually delivered.
///
/// The recurrence is 4 FMA per state element per token (decay, retrieval,
/// update, one readout head), i.e. (3 + H_q/H_v) * Dk * Dv. A chunked prefill
/// moves work into the intra-chunk terms instead, which is reported by the same
/// denominator — it is the model's arithmetic, not the kernel's instruction
/// count, and the two routes are compared in the same unit.
///
/// Measurement follows the other benches: per-round min of two back-to-back
/// calls, median across rounds. `NNOPS_BENCH_THREADS=1,4` pins the ladder.

#include "nnops/ops/linear_attention.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "common/bench_harness.hpp"
#include "common/cpu_pool.hpp"
#include "common/random_tensor.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using namespace nnops;
using half = nnops::backend::cpu::half;

namespace {

struct BenchPool {
    explicit BenchPool(int nthreads) : nthreads_(nthreads) {
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

/// NNOPS_LINEAR_ATTENTION_ROUTE, re-read by the kernel on every call, so this
/// one process can time both routes over the same buffers.
void set_route(const char* value) {
#ifdef _WIN32
    _putenv_s("NNOPS_LINEAR_ATTENTION_ROUTE", value ? value : "");
#else
    if (value) { setenv("NNOPS_LINEAR_ATTENTION_ROUTE", value, 1); }
    else { unsetenv("NNOPS_LINEAR_ATTENTION_ROUTE"); }
#endif
}

template <class Fn>
double time_ms(Fn&& fn, int iters) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) { fn(); }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() /
           std::max(iters, 1);
}

double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

std::vector<int> bench_threads() {
    std::vector<int> threads;
    if (const char* env = std::getenv("NNOPS_BENCH_THREADS")) {
        for (const char* p = env; *p;) {
            char* end = nullptr;
            const long v = std::strtol(p, &end, 10);
            if (end != p && v > 0) { threads.push_back(static_cast<int>(v)); }
            p = (end != p) ? end : p + 1;
            if (*p == ',') { ++p; }
        }
        if (!threads.empty()) { return threads; }
    }
    threads = {1};
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 1) { threads.push_back(hw); }
    return threads;
}

template <class T>
constexpr DataType dtype_of() {
    return std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
}

template <class T>
void fill_random(std::vector<T>& v, uint64_t seed, float lo, float hi) {
    nnops::test::XorShift128 rng(seed);
    for (size_t i = 0; i < v.size(); ++i) {
        const float f = rng.next_float(lo, hi);
        if constexpr (std::is_same_v<T, float>) {
            v[i] = f;
        } else {
            nnops::simd::s_store(&v[i], f);
        }
    }
}

/// One layer: B x H_q query heads, H_kv key heads, H_v value heads (the state
/// owners), Dk x Dv state, S tokens. H_v <= 0 means "same as H_kv"; Ck > 0 adds
/// the short convolution and beta, as a trained GDN layer carries them.
struct Shape {
    const char* name;
    int64_t B, H, Hkv, Hv, S, Dk, Dv, Ck;
};

const std::vector<Shape>& sweep() {
    static const std::vector<Shape> s = {
        // GDN as Qwen3-Next wires it: 32 value heads over 16 key heads, and
        // one query head per value head.
        {"gdn 16k/32v d128 decode",      1, 32, 16, 32,    1, 128, 128, 4},
        {"gdn 16k/32v d128  S=128",      1, 32, 16, 32,  128, 128, 128, 4},
        {"gdn 16k/32v d128  S=1024",     1, 32, 16, 32, 1024, 128, 128, 4},
        {"gdn 16k/32v d128  S=4096",     1, 32, 16, 32, 4096, 128, 128, 4},
        // A narrow layer: 8 value heads cannot fill the machine on a single
        // sequence, which is the case a chunked prefill is meant to open up.
        {"gdn  8k/8v  d128  S=1024",     1,  8,  8,  8, 1024, 128, 128, 4},
        {"gdn  8k/8v  d128  S=4096",     1,  8,  8,  8, 4096, 128, 128, 4},
        // Batched prefill: the task grid scales with B instead.
        {"gdn 16k/32v d128  B4 S=512",   4, 32, 16, 32,  512, 128, 128, 4},
        // Kimi's narrow value and the Qwen grouping, one row each.
        {"gdn 16k/32v d64v   S=1024",    1, 32, 16, 32, 1024, 128,  64, 4},
        {"gdn 16k/32v 2q/1v  S=1024",    1, 64, 16, 32, 1024, 128, 128, 4},
    };
    return s;
}

template <class T>
struct Problem {
    explicit Problem(const Shape& c) : s(c) {
        constexpr DataType dt = dtype_of<T>();
        const int64_t hv = c.Hv > 0 ? c.Hv : c.Hkv;
        const int64_t q[]  = {c.B, c.H,   c.S, c.Dk};
        const int64_t o[]  = {c.B, c.H,   c.S, c.Dv};
        const int64_t k[]  = {c.B, c.Hkv, c.S, c.Dk};
        const int64_t v[]  = {c.B, hv,    c.S, c.Dv};
        const int64_t g[]  = {c.B, hv,    c.S, c.Dk};
        const int64_t st[] = {c.B, hv,    c.Dk, c.Dv};
        const int64_t cw[] = {(c.H + c.Hkv) * c.Dk + hv * c.Dv, 1, std::max<int64_t>(c.Ck, 1)};
        const int64_t bt[] = {c.B, hv,    c.S};

        auto alloc = [](std::vector<T>& b, const int64_t* d, int rank, uint64_t seed,
                        float lo, float hi) {
            int64_t n = 1;
            for (int i = 0; i < rank; ++i) { n *= d[i]; }
            b.resize(static_cast<size_t>(n));
            fill_random(b, seed, lo, hi);
        };
        alloc(qb, q, 4, 1, -1.0f, 1.0f);
        alloc(kb, k, 4, 2, -1.0f, 1.0f);
        alloc(vb, v, 4, 3, -1.0f, 1.0f);
        // The gate is a log decay: real GDN states see a few 1e-3 to 1e-1 per
        // step, so a uniform [-1, 0] keeps the 4k cumulative product meaningful
        // in f32.
        alloc(gb, g, 4, 4, -1.0f, 0.0f);
        alloc(sb, st, 4, 5, -1.0f, 1.0f);
        alloc(ob, o, 4, 0, 0.0f, 0.0f);
        if (c.Ck > 0) {
            alloc(cwb, cw, 3, 6, -0.5f, 0.5f);
            alloc(bb, bt, 3, 7, 0.0f, 1.0f);
            cwv = TensorView(std::span<const int64_t>(cw, 3), dt, cwb.data());
            bv  = TensorView(std::span<const int64_t>(bt, 3), dt, bb.data());
        }

        qv  = TensorView(std::span<const int64_t>(q, 4), dt, qb.data());
        kv  = TensorView(std::span<const int64_t>(k, 4), dt, kb.data());
        vv  = TensorView(std::span<const int64_t>(v, 4), dt, vb.data());
        gv  = TensorView(std::span<const int64_t>(g, 4), dt, gb.data());
        stv = TensorView(std::span<const int64_t>(st, 4), dt, sb.data());
        ov  = TensorView(std::span<const int64_t>(o, 4), dt, ob.data());

        attrs.head_dim = c.Dk;
        attrs.num_heads = c.H;
        attrs.num_kv_heads = c.Hkv;
        attrs.conv_kernel_size = c.Ck;
        op = LinearAttention::create(attrs, Backend::CPU);
    }

    LinearAttentionAttributes attrs;
    Shape s;
    std::vector<T> qb, kb, vb, gb, sb, ob, cwb, bb;
    TensorView qv, kv, vv, gv, stv, ov, cwv, bv;
    std::unique_ptr<OpBase> op;
};

template <class T>
void run_shape(const Shape& c, const std::vector<int>& threads) {
    Problem<T> p(c);
    const int64_t hv = c.Hv > 0 ? c.Hv : c.Hkv;
    const int64_t gqq = c.H / hv;
    // The recurrence's own arithmetic: 3 state passes plus one readout per
    // query head, 2 FLOP per FMA, over B * H_v states of Dk * Dv each.
    const double flops = 2.0 * static_cast<double>((3 + gqq) * c.Dk * c.Dv) *
                         static_cast<double>(c.S * c.B * hv);
    const int iters = c.S <= 64 ? 20 : c.S <= 512 ? 5 : 2;

    std::vector<TensorView> outs = {p.ov, p.stv};
    std::vector<TensorView> ins = {p.qv, p.kv, p.vv, p.gv};
    if (c.Ck > 0) {
        ins.push_back(p.cwv);
        ins.push_back(p.bv);
    }

    std::printf("  %-28s %s |", c.name, std::is_same_v<T, float> ? "f32" : "f16");
    for (int nt : threads) {
        BenchPool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

        std::vector<double> chunk, token;
        for (int round = 0; round < 3; ++round) {
            set_route("chunk");
            chunk.push_back(time_ms([&] { p.op->compute(outs, ins, use_ctx, nullptr); }, iters));
            set_route("token");
            token.push_back(time_ms([&] { p.op->compute(outs, ins, use_ctx, nullptr); }, iters));
        }
        set_route(nullptr);
        const double tc = median_of(chunk), tt = median_of(token);
        std::printf("  %2dT %8.3f ms %7.1f GF", nt, tt, flops / (tt * 1e-3) / 1e9);
        if (c.S >= 64) {
            std::printf(" -> chunk %7.3f ms = %.2fx |", tc, tt / tc);
        } else {
            std::printf(" %-28s |", "(decode: per-token)");
        }
    }
    std::printf("\n");
}

}  // namespace

NNOPS_BENCH(linear_attention_prefill) {
    std::printf("  LinearAttention: B*H_v state matrices, per-token recurrence\n");
    const std::vector<int> threads = bench_threads();
    for (const Shape& c : sweep()) {
        run_shape<float>(c, threads);
    }
    std::printf("\n  f16\n");
    for (const Shape& c : sweep()) {
        if (c.S == 1) { continue; }
        run_shape<half>(c, threads);
    }
}
