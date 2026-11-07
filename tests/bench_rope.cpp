/// @file bench_rope.cpp
/// @brief RoPE: what partial rotary costs, and what fusing the Q/K RMSNorm saves.
///
/// Three arms over the same [B, H, S, D] image:
///   rope       RoPE alone — read + write (two tensor passes).
///   norm+rope  the pair a graph would otherwise run: RMSNorm into an
///              intermediate, then RoPE — four passes.
///   fused      RoPE with the normalization folded in before the rotation —
///              two passes, no intermediate.
/// A bandwidth-bound op should put the fused arm near half the two-op arm's
/// time; what is left over is the rotation's own arithmetic and the table.
///
/// The partial-rotary rows (rotary_dim < head_dim) read and write the whole
/// head but rotate only the prefix: the GB/s stays the same, so the cost of the
/// copied-through tail shows up as a smaller element/page footprint, not as
/// less traffic.
///
/// Both pair layouts are timed because they take different kernels: split-half
/// (HF rotate_half / GPT-NeoX) rotates a fixed offset and runs full SIMD for
/// both dtypes, while interleaved (llama.cpp's own layout) is a per-pair
/// scalar loop — the row pair is its cost.
///
/// `NNOPS_BENCH_THREADS=1` (or "1,20") pins the ladder, as in the other benches.

#include "nnops/ops/rope.hpp"
#include "nnops/ops/norm.hpp"
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
void fill_random(std::vector<T>& v, uint64_t seed, float lo = -1.0f, float hi = 1.0f) {
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

/// [B, H, S, D] with the rotary prefix; 0 means the whole head rotates.
/// `interleaved` picks the pair layout: true is (2k, 2k+1) (llama.cpp GGUF),
/// false is (d, d + rotary_dim/2) (HF rotate_half / GPT-NeoX).
struct Shape {
    const char* name;
    int64_t B, H, S, D;
    int64_t rotary_dim;
    bool interleaved = true;
};

const std::vector<Shape>& sweep() {
    static const std::vector<Shape> s = {
        {"q prefill B1 H32 S2048 D128 full intlv",   1, 32, 2048, 128,  0},
        {"q prefill B1 H32 S2048 D128 r=64 intlv",   1, 32, 2048, 128, 64},
        {"q prefill B1 H32 S2048 D128 full split",   1, 32, 2048, 128,  0, false},
        {"q prefill B1 H32 S2048 D128 r=64 split",   1, 32, 2048, 128, 64, false},
        {"k prefill B1 H8  S2048 D128 full intlv",   1,  8, 2048, 128,  0},
        {"k prefill B1 H8  S2048 D128 full split",   1,  8, 2048, 128,  0, false},
        {"q decode  B1 H32 S1    D128 full intlv",   1, 32,    1, 128,  0},
        {"q decode  B1 H32 S1    D128 full split",   1, 32,    1, 128,  0, false},
    };
    return s;
}

template <class T>
void run_table(const char* tag) {
    const std::vector<int> threads = bench_threads();
    const int rounds = 3;

    std::printf("  %s\n", tag);
    std::printf("    %-36s %5s %9s %9s %9s %9s %9s %6s\n",
                "case", "nt", "norm ms", "rope ms", "n+r ms", "fused ms", "GB/s", "n/f");

    for (const Shape& c : sweep()) {
        constexpr DataType dt = dtype_of<T>();
        const int64_t shp[] = {c.B, c.H, c.S, c.D};
        const int64_t sc[] = {c.D};
        const size_t elems = static_cast<size_t>(c.B * c.H * c.S * c.D);

        std::vector<T> x_v(elems), out_v(elems), tmp_v(elems), scale_v(static_cast<size_t>(c.D));
        fill_random(x_v, 1);
        fill_random(scale_v, 2, 0.5f, 1.5f);

        TensorView x(shp, dt, x_v.data());
        TensorView out(shp, dt, out_v.data());
        TensorView tmp(shp, dt, tmp_v.data());
        TensorView scale(sc, dt, scale_v.data());

        RoPEAttributes rattrs{};
        rattrs.rotary_dim = c.rotary_dim;
        rattrs.interleaved = c.interleaved;
        auto rope = RoPE::create(rattrs, Backend::CPU);

        NormAttributes nattrs{};
        nattrs.type = NormType::RMSNorm;
        nattrs.axis = -1;
        nattrs.epsilon = 1e-5f;
        auto norm = Norm::create(nattrs, Backend::CPU);

        const TensorView rope_in[]  = {x};
        const TensorView fused_in[] = {x, scale};
        const TensorView norm_in[]  = {x, scale};
        TensorView norm_out[] = {tmp};
        TensorView out_view[] = {out};

        const int iters = elems <= (size_t{1} << 16) ? 50
                        : elems <= (size_t{1} << 20) ? 10 : 3;
        const double bytes2 = 2.0 * static_cast<double>(elems) * sizeof(T);

        for (int nt : threads) {
            BenchPool pool(nt);
            ComputeContext ctx;
            ctx.cpu = pool.cpu;
            const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

            const auto arm_norm = [&] {
                norm->compute(norm_out, norm_in, use_ctx, nullptr);
            };
            const auto arm_rope = [&] {
                rope->compute(out_view, rope_in, use_ctx, nullptr);
            };
            // Two-op arm: Norm writes tmp, then RoPE reads tmp.
            const TensorView pair_in[] = {tmp};
            const auto run_pair = [&] {
                norm->compute(norm_out, norm_in, use_ctx, nullptr);
                rope->compute(out_view, pair_in, use_ctx, nullptr);
            };
            const auto run_fused = [&] {
                rope->compute(out_view, fused_in, use_ctx, nullptr);
            };

            run_fused();  // warm-up / page faults

            std::vector<double> v_norm, v_rope, v_pair, v_fused;
            for (int r = 0; r < rounds; ++r) {
                v_norm.push_back(time_ms(arm_norm, iters));
                v_rope.push_back(time_ms(arm_rope, iters));
                if (r % 2 == 0) {
                    v_pair.push_back(time_ms(run_pair, iters));
                    v_fused.push_back(time_ms(run_fused, iters));
                } else {
                    v_fused.push_back(time_ms(run_fused, iters));
                    v_pair.push_back(time_ms(run_pair, iters));
                }
            }
            const double t_norm = median_of(v_norm);
            const double t_rope = median_of(v_rope);
            const double t_pair = median_of(v_pair);
            const double t_fused = median_of(v_fused);

            std::printf("    %-36s %5d %9.3f %9.3f %9.3f %9.3f %9.1f %6.3f\n",
                        (nt == threads.front()) ? c.name : "",
                        nt, t_norm, t_rope, t_pair, t_fused,
                        bytes2 / t_fused / 1e6,
                        t_fused > 0.0 ? t_pair / t_fused : 0.0);
        }
    }
}

}  // anonymous namespace

NNOPS_BENCH(rope_f32) { run_table<float>("f32"); }
NNOPS_BENCH(rope_f16) { run_table<half>("f16"); }
