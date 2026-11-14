/// @file bench_norm.cpp
/// @brief Norm's pre-add residual against the graph chain it replaces.
///
/// Three arms over the same [M, H] activation:
///   norm   RMSNorm alone                       — two passes, 3·N elements moved
///   chain  Add(x, r) then RMSNorm              — four passes, 6·N
///   fused  RMSNorm with the residual input     — two passes, 5·N
/// The fused arm drops the intermediate tensor and one dispatch, so the row
/// shows what a transformer block's `norm(x + sublayer(x))` costs when the add
/// is folded in. `NNOPS_BENCH_THREADS=1,4` pins the ladder, as in the others.

#include "nnops/ops/norm.hpp"
#include "nnops/ops/activation.hpp"
#include "nnops/ops/eltwise.hpp"
#include "common/bench_harness.hpp"
#include "common/cpu_pool.hpp"
#include "common/random_tensor.hpp"
#include "common/test_helpers.hpp"
#include "nnops/detail/half.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <thread>
#include <vector>

using namespace nnops;

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
    threads = {1, 2, 4};
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 4) { threads.push_back(hw); }
    return threads;
}

struct Shape { int64_t M, H; };

void run_shape(const Shape& s, DataType dt, const std::vector<int>& threads) {
    const int64_t elems = s.M * s.H;
    auto [xv, x] = test::make_random_tensor({s.M, s.H}, -1.0f, 1.0f, 31);
    auto [rv, r] = test::make_random_tensor({s.M, s.H}, -1.0f, 1.0f, 32);
    auto [sv, sc] = test::make_random_tensor({s.H}, 0.5f, 1.5f, 33);

    std::vector<float> xb(xv.begin(), xv.end());
    std::vector<float> rb(rv.begin(), rv.end());
    std::vector<float> sb(sv.begin(), sv.end());
    std::vector<nnops::backend::cpu::half> xh, rh, sh;
    if (dt == DataType::f16) {
        xh = test::f32_to_f16(xb); rh = test::f32_to_f16(rb); sh = test::f32_to_f16(sb);
    }

    const int64_t shape[] = {s.M, s.H};
    const int64_t sshape[] = {s.H};
    TensorView xv_(std::span<const int64_t>(shape, 2), dt,
                   dt == DataType::f16 ? static_cast<void*>(xh.data()) : xb.data());
    TensorView rv_(std::span<const int64_t>(shape, 2), dt,
                   dt == DataType::f16 ? static_cast<void*>(rh.data()) : rb.data());
    TensorView sv_(std::span<const int64_t>(sshape, 1), dt,
                   dt == DataType::f16 ? static_cast<void*>(sh.data()) : sb.data());

    std::vector<float> ob(static_cast<size_t>(elems)), tb(static_cast<size_t>(elems));
    std::vector<nnops::backend::cpu::half> oh(static_cast<size_t>(elems)),
                                           th(static_cast<size_t>(elems));
    TensorView out(std::span<const int64_t>(shape, 2), dt,
                   dt == DataType::f16 ? static_cast<void*>(oh.data()) : ob.data());
    TensorView tmp(std::span<const int64_t>(shape, 2), dt,
                   dt == DataType::f16 ? static_cast<void*>(th.data()) : tb.data());

    NormAttributes nattrs;
    nattrs.type = NormType::RMSNorm;
    auto norm = Norm::create(nattrs, Backend::CPU);

    NormAttributes fattrs = nattrs;
    fattrs.add_residual = true;
    auto fused = Norm::create(fattrs, Backend::CPU);

    EltwiseAttributes eattrs;
    eattrs.type = EltwiseType::Add;
    auto add = Eltwise::create(eattrs, Backend::CPU);

    const TensorView norm_in[]  = {xv_, sv_};
    const TensorView fused_in[] = {xv_, sv_, rv_};
    const TensorView add_in[]   = {xv_, rv_};
    const TensorView tmp_in[]   = {tmp, sv_};

    const int iters = elems <= (int64_t{1} << 16) ? 50
                    : elems <= (int64_t{1} << 20) ? 10 : 3;

    std::printf("    %4lldx%-6lld %s |", (long long)s.M, (long long)s.H,
                dt == DataType::f16 ? "f16" : "f32");
    for (int nt : threads) {
        BenchPool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

        std::vector<double> vn, vc, vf;
        for (int round = 0; round < 3; ++round) {
            vn.push_back(time_ms([&] { norm->compute(out, norm_in, use_ctx, nullptr); }, iters));
            vc.push_back(time_ms([&] {
                add->compute(tmp, add_in, use_ctx, nullptr);
                norm->compute(out, tmp_in, use_ctx, nullptr);
            }, iters));
            vf.push_back(time_ms([&] { fused->compute(out, fused_in, use_ctx, nullptr); }, iters));
        }
        const double tn = median_of(vn), tc = median_of(vc), tf = median_of(vf);
        std::printf("  %2dT chain %8.3f fused %8.3f = %.2fx |", nt, tc, tf,
                    tc / tf);
    }
    std::printf("\n");
}

}  // namespace

NNOPS_BENCH(norm_pre_add_residual) {
    std::printf("  RMSNorm(x + r) fused against Add + RMSNorm  (ms/call, GB/s = 5 reads/writes)\n");
    const std::vector<int> threads = bench_threads();
    const Shape shapes[] = {{1, 4096}, {16, 4096}, {512, 4096}, {16, 11008}};
    for (const Shape& s : shapes) {
        run_shape(s, DataType::f32, threads);
    }
    run_shape({16, 4096}, DataType::f16, threads);
}

// ---------------------------------------------------------------------------
// RMSNormGated: the fused gate against the three-op graph chain it replaces.
// ---------------------------------------------------------------------------

void run_gated_shape(const Shape& s, DataType dt, const std::vector<int>& threads)
{
    const int64_t elems = s.M * s.H;
    auto [xb, x0] = test::make_random_tensor({s.M, s.H}, -1.0f, 1.0f, 41);
    auto [gb, g0] = test::make_random_tensor({s.M, s.H}, -2.0f, 2.0f, 42);
    auto [sb, s0] = test::make_random_tensor({s.H}, 0.5f, 1.5f, 43);
    (void)x0; (void)g0; (void)s0;

    std::vector<nnops::backend::cpu::half> xh, gh, sh;
    if (dt == DataType::f16) {
        xh = test::f32_to_f16(xb);
        gh = test::f32_to_f16(gb);
        sh = test::f32_to_f16(sb);
    }
    void* xp = dt == DataType::f16 ? static_cast<void*>(xh.data()) : xb.data();
    void* gp = dt == DataType::f16 ? static_cast<void*>(gh.data()) : gb.data();
    void* sp = dt == DataType::f16 ? static_cast<void*>(sh.data()) : sb.data();

    std::vector<float> ob(static_cast<size_t>(elems)), tb(static_cast<size_t>(elems)),
                       ab(static_cast<size_t>(elems));
    std::vector<nnops::backend::cpu::half> oh(static_cast<size_t>(elems)),
                                           th(static_cast<size_t>(elems)),
                                           ah(static_cast<size_t>(elems));
    void* op = dt == DataType::f16 ? static_cast<void*>(oh.data()) : ob.data();
    void* tp = dt == DataType::f16 ? static_cast<void*>(th.data()) : tb.data();
    void* ap = dt == DataType::f16 ? static_cast<void*>(ah.data()) : ab.data();

    const int64_t shape[] = {s.M, s.H};
    const int64_t sshape[] = {s.H};
    TensorView x(std::span<const int64_t>(shape, 2), dt, xp);
    TensorView g(std::span<const int64_t>(shape, 2), dt, gp);
    TensorView sc(std::span<const int64_t>(sshape, 1), dt, sp);
    TensorView out(std::span<const int64_t>(shape, 2), dt, op);
    TensorView tmp(std::span<const int64_t>(shape, 2), dt, tp);
    TensorView act(std::span<const int64_t>(shape, 2), dt, ap);

    NormAttributes nattrs;
    nattrs.type = NormType::RMSNorm;
    auto norm = Norm::create(nattrs, Backend::CPU);
    NormAttributes gattrs = nattrs;
    gattrs.gate_activation = EpilogueActivateType::Silu;
    auto gnorm = Norm::create(gattrs, Backend::CPU);
    ActivationAttributes aattrs;
    aattrs.type = ActivationType::Silu;
    auto silu = Activation::create(aattrs, Backend::CPU);
    EltwiseAttributes eattrs;
    eattrs.type = EltwiseType::Mul;
    auto mul = Eltwise::create(eattrs, Backend::CPU);

    const TensorView norm_in[] = {x, sc};
    const TensorView gate_in[] = {x, sc, g};
    const TensorView act_in[] = {g};
    const TensorView mul_in[] = {tmp, act};

    const int iters = elems <= (int64_t{1} << 16) ? 50
                    : elems <= (int64_t{1} << 20) ? 10 : 3;

    std::printf("    %4lldx%-6lld %s |", (long long)s.M, (long long)s.H,
                dt == DataType::f16 ? "f16" : "f32");
    for (int nt : threads) {
        BenchPool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

        std::vector<double> chain, fused;
        for (int round = 0; round < 3; ++round) {
            chain.push_back(time_ms([&] {
                norm->compute(tmp, norm_in, use_ctx, nullptr);
                silu->compute(act, act_in, use_ctx, nullptr);
                mul->compute(out, mul_in, use_ctx, nullptr);
            }, iters));
            fused.push_back(time_ms([&] {
                gnorm->compute(out, gate_in, use_ctx, nullptr);
            }, iters));
        }
        const double tc = median_of(chain), tf = median_of(fused);
        std::printf("  %2dT chain %8.3f fused %8.3f = %.2fx |", nt, tc, tf, tc / tf);
    }
    std::printf("\n");
}

NNOPS_BENCH(norm_gated) {
    std::printf("  RMSNormGated fused against RMSNorm -> Silu -> Mul\n");
    const std::vector<int> threads = bench_threads();
    const Shape shapes[] = {{1, 4096}, {16, 4096}, {512, 4096}, {16, 11008}};
    for (const Shape& s : shapes) {
        run_gated_shape(s, DataType::f32, threads);
    }
    run_gated_shape({16, 4096}, DataType::f16, threads);
    run_gated_shape({512, 4096}, DataType::f16, threads);
    run_gated_shape({16, 11008}, DataType::f16, threads);
}
