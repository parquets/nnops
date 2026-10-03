/// @file bench_matmul.cpp
/// @brief MatMul micro-benchmarks — pack decision + thread scaling.
///
/// Every case runs the plan-driven fused path (B always packed, A packed only on
/// transpose_a or a wide row stride) across {nn, ta, tb, tatb} and padded
/// strides. For each case the harness reports:
///   - the split direction (NKM when split on N, MKN when split on M),
///   - the resolved tile sizes (mc, nc) and block count from get_matmul_plan(),
///   - GFLOPS at 1, 2, 4, 8 and hardware threads.
///
/// The point is to see where block-granularity threading scales (and where it
/// can't — e.g. when Nc is so large that an N-split yields only one block).

#include "nnops/ops/matmul.hpp"
#include "backend/cpu/matmul_helper.h"   // resolve_tile_sizes, PACK_A_STRIDE_THRESHOLD, tile constants
#include "common/bench_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using namespace nnops;
using half = nnops::backend::cpu::half;
namespace cpu = nnops::backend::cpu;

namespace {

// Minimal fixed-size thread pool exposing a CpuBackend (same as the test).
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

// Build a [rows, cols] tensor of type T with an explicit row pitch (elements);
// only the logical region is filled (padding lanes stay 0).
template <typename T>
std::pair<std::vector<T>, TensorView>
make_tensor(int64_t rows, int64_t cols, int64_t pitch_elems, uint64_t seed) {
    std::vector<float> f32(static_cast<size_t>(rows * pitch_elems), 0.0f);
    nnops::test::XorShift128 rng(seed);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            f32[static_cast<size_t>(r * pitch_elems + c)] = rng.next_float(-1.0f, 1.0f);
        }
    }
    std::vector<T> buf(static_cast<size_t>(rows * pitch_elems));
    for (size_t i = 0; i < buf.size(); ++i) {
        if constexpr (std::is_same_v<T, float>) {
            buf[i] = f32[i];
        } else {
            simd::s_store(&buf[i], f32[i]);
        }
    }
    constexpr DataType dt = std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
    const int64_t shape[] = {rows, cols};
    TensorView tv(shape, dt, buf.data(),
                  pitch_elems * static_cast<int64_t>(sizeof(T)));
    return {std::move(buf), tv};
}

// Build a compact [rows, cols] s8 tensor. Quant params are left at the default
// identity (zp=0, scale=1), so the int8 kernel accumulates the raw dot product —
// the qgemm / MatMulInteger analog — with a trivial zero-point compensation.
std::pair<std::vector<int8_t>, TensorView>
make_tensor_i8(int64_t rows, int64_t cols, uint64_t seed) {
    std::vector<int8_t> buf(static_cast<size_t>(rows * cols));
    nnops::test::XorShift128 rng(seed);
    for (auto& v : buf) { v = static_cast<int8_t>(rng.next_u64() % 255 - 127); }
    const int64_t shape[] = {rows, cols};
    TensorView tv(shape, DataType::s8, buf.data());
    return {std::move(buf), tv};
}

struct GeMMCase {
    const char* name;
    int64_t M, K, N;
    bool ta = false, tb = false;
    int64_t pad_a = 0;  // extra elements on A's row pitch (0 = compact)
    int64_t pad_b = 0;
    bool f16 = false;
};

struct GeMMReport {
    const char* name;
    const char* route;   // "NKM-fused" (fp) or "int8-NKM"
    const char* unit;    // "GFLOPS" or "GOPS"
    bool pack_a, pack_b;
    int mc, nc;
    int64_t num_blocks;
    std::vector<int> threads;
    std::vector<double> gflops;
};

// Time one case at the given thread counts; fills the report.
template <typename T>
GeMMReport run_geMM(const GeMMCase& c, const std::vector<int>& threads, int iters) {
    const int64_t a_rows = c.ta ? c.K : c.M;
    const int64_t a_cols = c.ta ? c.M : c.K;
    const int64_t b_rows = c.tb ? c.N : c.K;
    const int64_t b_cols = c.tb ? c.K : c.N;

    auto [a_vec, a] = make_tensor<T>(a_rows, a_cols, a_cols + c.pad_a, 1);
    auto [b_vec, b] = make_tensor<T>(b_rows, b_cols, b_cols + c.pad_b, 2);

    MatMulAttributes attrs{};
    attrs.transpose_a = c.ta;
    attrs.transpose_b = c.tb;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);

    // --- static metadata from the plan (mirrors matmul_kernel) ---
    const cpu::MatMulPlan plan = cpu::get_matmul_plan(attrs, a_desc, b_desc, 1);

    std::vector<T> out_buf(static_cast<size_t>(descs[0].numel()));
    auto output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {a, b};

    GeMMReport rep;
    rep.name = c.name;
    rep.route = plan.split_n ? "NKM" : "MKN";
    rep.unit = "GFLOPS";
    rep.pack_a = plan.pack_a;
    rep.pack_b = true;
    rep.mc = static_cast<int>(plan.mc);
    rep.nc = static_cast<int>(plan.nc);
    rep.num_blocks = plan.split_n
        ? (c.N + plan.nc - 1) / plan.nc
        : cpu::split_block_count(c.M, plan.mc);
    rep.threads = threads;

    for (int nt : threads) {
        SimplePool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

        // Warm-up (also faults/allocates pages).
        for (int i = 0; i < 2; ++i) {
            op->compute(output, ins, use_ctx, nullptr);
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            op->compute(output, ins, use_ctx, nullptr);
        }
        auto end = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        rep.gflops.push_back(
            2.0 * static_cast<double>(c.M) * static_cast<double>(c.N)
                * static_cast<double>(c.K) * static_cast<double>(iters) / (ms * 1e6));
    }
    return rep;
}

void print_report(const GeMMReport& r) {
    std::printf("  %s\n", r.name);
    std::printf("    route=%-12s pack_a=%d pack_b=%d  mc=%d nc=%d  blocks=%lld\n",
                r.route, (int)r.pack_a, (int)r.pack_b, r.mc, r.nc,
                static_cast<long long>(r.num_blocks));

    const double g1 = r.gflops.empty() ? 0.0 : r.gflops[0];
    std::printf("    ");
    for (size_t i = 0; i < r.threads.size(); ++i) {
        const double speedup = g1 > 0.0 ? r.gflops[i] / g1 : 0.0;
        std::printf(" %2dT %7.1f %s (%.2fx)", r.threads[i], r.gflops[i], r.unit, speedup);
        if (i + 1 < r.threads.size()) { std::printf(" |"); }
    }
    std::printf("\n");
}

void bench_case(const GeMMCase& c) {
    const int iters = 20;
    std::vector<int> threads{1, 2, 4, 8};
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 1 && std::find(threads.begin(), threads.end(), hw) == threads.end()) {
        threads.push_back(hw);
    }
    GeMMReport rep = c.f16 ? run_geMM<half>(c, threads, iters)
                           : run_geMM<float>(c, threads, iters);
    print_report(rep);
}

// Time one int8 (s8×s8 → s32) case; mirrors run_geMM but uses the int8 panel
// constants and a s32 output buffer. GFLOPS is relabeled GOPS for int8.
GeMMReport run_geMM_i8(const GeMMCase& c, const std::vector<int>& threads, int iters) {
    const int64_t a_rows = c.ta ? c.K : c.M;
    const int64_t a_cols = c.ta ? c.M : c.K;
    const int64_t b_rows = c.tb ? c.N : c.K;
    const int64_t b_cols = c.tb ? c.K : c.N;

    auto [a_vec, a] = make_tensor_i8(a_rows, a_cols, 1);
    auto [b_vec, b] = make_tensor_i8(b_rows, b_cols, 2);

    MatMulAttributes attrs{};
    attrs.transpose_a = c.ta;
    attrs.transpose_b = c.tb;
    attrs.output_dtype = DataType::s32;   // MatMulInteger semantics
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);

    // int8 tiling: elements are 1 byte; panel sizes come from the int8 lists.
    int mc, nc;
    cpu::resolve_tile_sizes(static_cast<int>(c.M), static_cast<int>(c.N),
                            cpu::MR_MAX_I8, cpu::NR_MAX_I8, cpu::KC_I8, 1, mc, nc);
    const bool split_n = (c.N > c.M);
    const int64_t num_blocks = split_n
        ? (c.N + nc - 1) / nc
        : cpu::split_block_count(c.M, mc);

    std::vector<int32_t> out_buf(static_cast<size_t>(descs[0].numel()));
    auto output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {a, b};

    GeMMReport rep;
    rep.name = c.name;
    rep.route = split_n ? "int8-NKM" : "int8-MKN";
    rep.unit = "GOPS";
    rep.pack_a = true;   // int8 always packs both A and B
    rep.pack_b = true;
    rep.mc = mc;
    rep.nc = nc;
    rep.num_blocks = num_blocks;
    rep.threads = threads;

    for (int nt : threads) {
        SimplePool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

        for (int i = 0; i < 2; ++i) {
            op->compute(output, ins, use_ctx, nullptr);
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            op->compute(output, ins, use_ctx, nullptr);
        }
        auto end = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        rep.gflops.push_back(
            2.0 * static_cast<double>(c.M) * static_cast<double>(c.N)
                * static_cast<double>(c.K) * static_cast<double>(iters) / (ms * 1e6));
    }
    return rep;
}

void bench_i8_case(const GeMMCase& c) {
    const int iters = 20;
    std::vector<int> threads{1, 2, 4, 8};
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 1 && std::find(threads.begin(), threads.end(), hw) == threads.end()) {
        threads.push_back(hw);
    }
    print_report(run_geMM_i8(c, threads, iters));
}

// ---- controlled A/B between two routes ------------------------------------
//
// bench_case() runs each route as its own case, so the two numbers are taken in
// different thermal/frequency states — on a fanless host, two runs of the *same*
// route differ by >30%, which swamps the pack_a=0 vs pack_a=1 gap this suite
// exists to expose. This builds both routes up front and alternates them round
// by round inside one process, so the ratio is measured under matched
// conditions. Read the per-round ratios, not the absolute GFLOPS.
//
// The A-side of each pair is built to take the unpacked-A ("direct") route and
// the B-side the packed route. Two ways to force packing exist and they answer
// different questions:
//   * transpose_a=true  — packing is required for correctness (A is stored
//                         K×M), so the pair also changes A's memory layout;
//   * a wide A row pitch — trips the stride heuristic (lda > PACK_A_STRIDE_THRESHOLD)
//                         with the layout held fixed.
// Both are exposed; read the per-round ratios.
template <typename T>
struct RouteAB {
    using OpPtr = decltype(MatMul::create(std::declval<MatMulAttributes>(), Backend::CPU));

    OpPtr op;
    std::vector<T> a_buf, b_buf, out_buf;
    TensorView a, b, out;
    TensorView ins[2];
    int64_t M, N, K;
    bool pack_a;

    RouteAB(int64_t m, int64_t k, int64_t n, bool transpose_a, uint64_t seed,
            int64_t pad_a = 0)
        : M(m), N(n), K(k) {
        const int64_t a_rows = transpose_a ? k : m;
        const int64_t a_cols = transpose_a ? m : k;
        auto [a_vec, a_view] = make_tensor<T>(a_rows, a_cols, a_cols + pad_a, seed);
        a_buf = std::move(a_vec);
        a = a_view;

        auto [b_vec, b_view] = make_tensor<T>(k, n, n, seed + 1);
        b_buf = std::move(b_vec);
        b = b_view;

        MatMulAttributes attrs{};
        attrs.transpose_a = transpose_a;
        op = MatMul::create(attrs, Backend::CPU);

        const TensorDesc arr[] = {a.desc(), b.desc()};
        auto descs = op->getOutputTensorDesc(arr);
        out_buf.assign(static_cast<size_t>(descs[0].numel()), T{});
        out = test::make_planar(descs[0], out_buf.data());

        ins[0] = a;
        ins[1] = b;
        pack_a = cpu::get_matmul_plan(attrs, a.desc(), b.desc(), 1).pack_a;
    }

    double time(int iters) {
        for (int i = 0; i < 2; ++i) { op->compute(out, ins, {}, nullptr); }
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) { op->compute(out, ins, {}, nullptr); }
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return 2.0 * static_cast<double>(M) * static_cast<double>(N)
             * static_cast<double>(K) * static_cast<double>(iters) / (ms * 1e6);
    }
};

/// Build two routes, alternate them round by round, and report the per-round
/// ratio plus the median. @p la is the expected unpacked-A (direct) side.
template <typename T>
void bench_route_ab(const char* tag, const char* la, const char* lb,
                    int64_t M, int64_t K, int64_t N,
                    bool ta_a, int64_t pad_a, bool ta_b, int64_t pad_b,
                    int rounds = 6, int iters = 20) {
    RouteAB<T> ra(M, K, N, ta_a, 1, pad_a);
    RouteAB<T> rb(M, K, N, ta_b, 1, pad_b);

    std::printf("  route A/B %s  %lldx%lldx%lld  1T  rounds=%d iters=%d\n",
                tag, static_cast<long long>(M), static_cast<long long>(K),
                static_cast<long long>(N), rounds, iters);
    std::printf("    pack_a: %s=%d %s=%d   (0 = unpacked-A direct route)\n",
                la, static_cast<int>(ra.pack_a), lb, static_cast<int>(rb.pack_a));

    std::vector<double> va, vb;
    for (int r = 0; r < rounds; ++r) {
        const double ga = ra.time(iters);
        const double gb = rb.time(iters);
        va.push_back(ga);
        vb.push_back(gb);
        std::printf("    round %d:  %s=%7.1f  %s=%7.1f   %s/%s=%.3f\n",
                    r, la, ga, lb, gb, la, lb, ga / gb);
    }

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    const double ma = median(va), mb = median(vb);
    std::printf("    median:   %s=%7.1f  %s=%7.1f   %s/%s=%.3f\n",
                la, ma, lb, mb, la, lb, ma / mb);
}

}  // anonymous namespace

// ============================================================
// f32 — routing table sweep
// ============================================================

NNOPS_BENCH(matmul_f32_nn_direct_256) {
    bench_case({"nn 256^3 direct", 256, 256, 256});
}

NNOPS_BENCH(matmul_f32_nn_direct_large) {
    bench_case({"nn 1024x512x1024", 1024, 512, 1024});
}

NNOPS_BENCH(matmul_f32_route_ab) {
    // Paired with matmul_f32_nn_direct_large / matmul_f32_ta_mkn, which measure
    // the same shapes but in separate, individually-unreliable runs. On aarch64
    // the two routes also differ in M panel height (pack mr=8, direct mr=6), so
    // the nn/ta ratio folds in the narrower direct panel.
    bench_route_ab<float>("f32", "nn", "ta", 1024, 512, 1024,
                          /*ta_a=*/false, /*pad_a=*/0,
                          /*ta_b=*/true,  /*pad_b=*/0);
}

NNOPS_BENCH(matmul_f32_ta_mkn) {
    bench_case({"ta 1024x512x1024 MKN", 1024, 512, 1024, /*ta=*/true});
}

NNOPS_BENCH(matmul_f32_tb_nkm) {
    bench_case({"tb 1024x512x1024 NKM", 1024, 512, 1024, /*ta=*/false, /*tb=*/true});
}

NNOPS_BENCH(matmul_f32_tatb_nkm) {
    bench_case({"tatb 1024x512x1024 NKM", 1024, 512, 1024, /*ta=*/true, /*tb=*/true});
}

NNOPS_BENCH(matmul_f32_padded_b_stride) {
    // Wide non-transposed B row stride forces pack_b via the stride heuristic.
    bench_case({"nn pad_b stride pack_b", 1024, 512, 1024, /*ta=*/false, /*tb=*/false,
                /*pad_a=*/0, /*pad_b=*/4096});
}

// ============================================================
// f16 — fused path spot checks
// ============================================================

NNOPS_BENCH(matmul_f16_nn) {
    bench_case({"f16 nn 512x512x256", 512, 512, 256, /*ta=*/false, /*tb=*/false,
                /*pad_a=*/0, /*pad_b=*/0, /*f16=*/true});
}

NNOPS_BENCH(matmul_f16_ta) {
    bench_case({"f16 ta 512x512x256 MKN", 512, 512, 256, /*ta=*/true, /*tb=*/false,
                /*pad_a=*/0, /*pad_b=*/0, /*f16=*/true});
}

// f16 pack-vs-direct, measured under matched conditions (see bench_route_ab).
// Unlike f32, f16 tiles M identically on both routes (mr={8,4,1}; there is no
// mr_f16_direct), so the ratio here is not tangled up with panel height.
NNOPS_BENCH(matmul_f16_route_ab) {
    bench_route_ab<half>("f16", "nn", "ta", 512, 512, 256,
                         /*ta_a=*/false, /*pad_a=*/0,
                         /*ta_b=*/true,  /*pad_b=*/0);
}

NNOPS_BENCH(matmul_f16_route_ab_large) {
    bench_route_ab<half>("f16", "nn", "ta", 1024, 1024, 1024,
                         /*ta_a=*/false, /*pad_a=*/0,
                         /*ta_b=*/true,  /*pad_b=*/0,
                         /*rounds=*/4, /*iters=*/10);
}

NNOPS_BENCH(matmul_f16_route_ab_deep_k) {
    // Deep K. NOTE: with a compact A, lda == K, so K > PACK_A_STRIDE_THRESHOLD
    // means the stride heuristic fires on *both* sides — this measures packed
    // vs packed-transposed, not pack vs direct (the plan prints nn=1 ta=1).
    // A direct route is simply unreachable at this depth without raising the
    // threshold.
    bench_route_ab<half>("f16 deep-K (both packed)", "nn", "ta", 512, 2048, 512,
                         /*ta_a=*/false, /*pad_a=*/0,
                         /*ta_b=*/true,  /*pad_b=*/0,
                         /*rounds=*/4, /*iters=*/10);
}

NNOPS_BENCH(matmul_f16_a_pitch_sweep) {
    // Walk the A row pitch across PACK_A_STRIDE_THRESHOLD with transpose_a=false
    // on every row, so lda == pitch and the only thing changing is the plan's
    // pack_a decision (and the row footprint that drives it). Shows where the
    // heuristic flips and what the flip costs. Rounds are interleaved across all
    // pitches so the comparison is thermal-matched (see bench_route_ab).
    const int64_t M = 512, K = 512, N = 256;
    const int64_t pads[] = {0, 128, 256, 384, 512, 513, 640, 1024, 3072};
    const int rounds = 5, iters = 20;

    std::vector<RouteAB<half>> routes;
    routes.reserve(std::size(pads));
    for (int64_t pad : pads) {
        routes.emplace_back(M, K, N, /*transpose_a=*/false, 1, pad);
    }

    std::printf("  f16 A row-pitch sweep  %lldx%lldx%lld  1T  rounds=%d iters=%d  "
                "threshold=%lld elems\n",
                static_cast<long long>(M), static_cast<long long>(K),
                static_cast<long long>(N), rounds, iters,
                static_cast<long long>(cpu::PACK_A_STRIDE_THRESHOLD));

    std::vector<std::vector<double>> samples(routes.size());
    for (int r = 0; r < rounds; ++r) {
        for (size_t i = 0; i < routes.size(); ++i) {
            samples[i].push_back(routes[i].time(iters));
        }
    }
    for (size_t i = 0; i < routes.size(); ++i) {
        auto v = samples[i];
        std::sort(v.begin(), v.end());
        const int64_t lda = K + pads[i];
        std::printf("    lda=%5lld elems (%6.1f KB/row)  pack_a=%d   %7.1f GFLOPS\n",
                    static_cast<long long>(lda),
                    static_cast<double>(lda) * sizeof(half) / 1024.0,
                    static_cast<int>(routes[i].pack_a),
                    v[v.size() / 2]);
    }
}

NNOPS_BENCH(matmul_f16_route_ab_stride) {
    // Layout held fixed (transpose_a=false on both sides) so this isolates the
    // pack *copy* from the transpose relayout: the direct route reads a compact
    // A (lda = K, below the heuristic), the packed route trips the stride
    // heuristic with a wide row pitch (lda = K + PACK_A_STRIDE_THRESHOLD + 512).
    // The wide pitch is part of the decision the plan makes — read it as "when A
    // is page-scattered, does packing it back pay off?", not a pure toggle.
    bench_route_ab<half>("f16 stride-pack", "nn", "nn-wide", 512, 512, 256,
                         /*ta_a=*/false, /*pad_a=*/0,
                         /*ta_b=*/false, /*pad_b=*/cpu::PACK_A_STRIDE_THRESHOLD + 512);
}

// ============================================================
// int8 (s8×s8 → s32) — qgemm path
// ============================================================

NNOPS_BENCH(matmul_i8_nn_large) {
    bench_i8_case({"i8 nn 1024x512x1024", 1024, 512, 1024});
}

NNOPS_BENCH(matmul_i8_nn_256) {
    bench_i8_case({"i8 nn 256^3", 256, 256, 256});
}

NNOPS_BENCH(matmul_i8_tb_large) {
    // LLM-style A @ W^T (transpose_b): weight-B is [N, K] per-token.
    bench_i8_case({"i8 tb 1024x512x1024", 1024, 512, 1024, /*ta=*/false, /*tb=*/true});
}
