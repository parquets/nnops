/// @file bench_matmul.cpp
/// @brief MatMul micro-benchmarks — pack decision + thread scaling.
///
/// Every case runs the plan-driven fused path (B always packed, A packed only on
/// transpose_a or a deep K) across {nn, ta, tb, tatb} and padded
/// strides. For each case the harness reports:
///   - the split direction (NKM when split on N, MKN when split on M),
///   - the resolved tile sizes (mc, nc) and block count from get_matmul_plan(),
///   - GFLOPS at 1, 2, 4, 8 and hardware threads.
///
/// The point is to see where block-granularity threading scales (and where it
/// can't — e.g. when Nc is so large that an N-split yields only one block).

#include "nnops/ops/matmul.hpp"
#include "nnops/detail/q5_1.hpp"         // q5_1_block, float_to_half, half_to_bits
#include "backend/cpu/matmul_helper.h"   // get_matmul_plan, PACK_A_K_THRESHOLD, tile constants
#include "common/bench_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(NNOPS_ARCH_X86_64)
#include <immintrin.h>
#endif

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

// Build a compact [rows, cols] int8 tensor (s8 by default, u8 on request; the
// buffer is bytes either way). Quant params are left at the default identity
// (zp=0, scale=1), so the int8 kernel accumulates the raw dot product — the
// qgemm / MatMulInteger analog — with a trivial zero-point compensation.
std::pair<std::vector<int8_t>, TensorView>
make_tensor_i8(int64_t rows, int64_t cols, uint64_t seed,
               DataType dt = DataType::s8) {
    std::vector<int8_t> buf(static_cast<size_t>(rows * cols));
    nnops::test::XorShift128 rng(seed);
    if (dt == DataType::u8) {
        for (auto& v : buf) { v = static_cast<int8_t>(rng.next_u64() % 256); }
    } else {
        for (auto& v : buf) { v = static_cast<int8_t>(rng.next_u64() % 255 - 127); }
    }
    const int64_t shape[] = {rows, cols};
    TensorView tv(shape, dt, buf.data());
    return {std::move(buf), tv};
}

// Build a compact [rows, cols] Q5_1 (llama.cpp block) tensor: 32 five-bit
// quants per 24-byte block, with a per-block fp16 scale/offset in the block
// itself. Only the storage matters here, but the quants and d/m are drawn from
// realistic ranges so no lane is denormal or inf.
std::pair<std::vector<uint8_t>, TensorView>
make_tensor_q5(int64_t rows, int64_t cols, uint64_t seed) {
    const int64_t blocks_per_row = (cols + q5_1_block_elems - 1) / q5_1_block_elems;
    std::vector<uint8_t> buf(static_cast<size_t>(
        rows * blocks_per_row * static_cast<int64_t>(q5_1_block_bytes)));

    nnops::test::XorShift128 rng(seed);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t b = 0; b < blocks_per_row; ++b) {
            q5_1_block blk{};
            blk.d = cpu::half_to_bits(cpu::float_to_half(rng.next_float(0.002f, 0.02f)));
            blk.m = cpu::half_to_bits(cpu::float_to_half(rng.next_float(-0.1f, 0.1f)));
            uint32_t qh = 0;
            for (int j = 0; j < 16; ++j) {
                const uint32_t q0 = static_cast<uint32_t>(rng.next_u64() & 0x1F);
                const uint32_t q1 = static_cast<uint32_t>(rng.next_u64() & 0x1F);
                blk.qs[j] = static_cast<uint8_t>(q0 | (q1 << 4));
                qh |= ((q0 >> 4) & 1u) << j;
                qh |= ((q1 >> 4) & 1u) << (j + 16);
            }
            std::memcpy(blk.qh, &qh, sizeof(qh));
            std::memcpy(buf.data() + static_cast<size_t>(
                            (r * blocks_per_row + b)
                            * static_cast<int64_t>(q5_1_block_bytes)),
                        &blk, q5_1_block_bytes);
        }
    }

    const int64_t shape[] = {rows, cols};
    TensorView tv(shape, DataType::q5_1, buf.data());
    return {std::move(buf), tv};
}

struct GeMMCase {
    const char* name;
    int64_t M, K, N;
    bool ta = false, tb = false;
    int64_t pad_a = 0;  // extra elements on A's row pitch (0 = compact)
    int64_t pad_b = 0;
    bool f16 = false;
    DataType a_dt = DataType::s8;   // int8 cases only: s8 (default) or u8
};

struct GeMMReport {
    const char* name;
    const char* route;   // "NKM-fused" (fp) or "int8-NKM"
    const char* unit;    // "GFLOPS" or "GOPS"
    bool pack_a, pack_b;
    int mc, nc, kc;
    int64_t num_blocks;
    double b_mb = 0.0;   // B storage footprint (0 = don't print)
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

    // --- static metadata from the plan (mirrors matmul_cpu) ---
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
    rep.kc = static_cast<int>(plan.kc);
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
    std::printf("    route=%-12s pack_a=%d pack_b=%d  mc=%d nc=%d kc=%d  blocks=%lld",
                r.route, (int)r.pack_a, (int)r.pack_b, r.mc, r.nc, r.kc,
                static_cast<long long>(r.num_blocks));
    if (r.b_mb > 0.0) { std::printf("  B=%.2f MB", r.b_mb); }
    std::printf("\n");

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

// Time one int8 (s8|u8 × s8 → s32) case; mirrors run_geMM but uses the int8
// panel constants and a s32 output buffer. GFLOPS is relabeled GOPS for int8.
GeMMReport run_geMM_i8(const GeMMCase& c, const std::vector<int>& threads, int iters) {
    const int64_t a_rows = c.ta ? c.K : c.M;
    const int64_t a_cols = c.ta ? c.M : c.K;
    const int64_t b_rows = c.tb ? c.N : c.K;
    const int64_t b_cols = c.tb ? c.K : c.N;

    auto [a_vec, a] = make_tensor_i8(a_rows, a_cols, 1, c.a_dt);
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

    // Static metadata from the plan, exactly as matmul_cpu builds it. The
    // int8 path is plan-driven like fp, so mc/nc/blocks are what the kernel
    // actually runs — resolvable only for a given pool size. Report the largest
    // pool in the row: those numbers describe the (fastest) run printed last,
    // whereas the 1-thread plan would describe no run at all.
    const int meta_threads = *std::max_element(threads.begin(), threads.end());
    const cpu::MatMulPlan plan = cpu::get_matmul_plan(attrs, a_desc, b_desc, meta_threads);

    std::vector<int32_t> out_buf(static_cast<size_t>(descs[0].numel()));
    auto output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {a, b};

    GeMMReport rep;
    rep.name = c.name;
    rep.route = plan.split_n ? "int8-NKM" : "int8-MKN";
    rep.unit = "GOPS";
    rep.pack_a = plan.pack_a;   // int8 always packs both A and B
    rep.pack_b = true;
    rep.mc = static_cast<int>(plan.mc);
    rep.nc = static_cast<int>(plan.nc);
    rep.kc = static_cast<int>(plan.kc);
    rep.num_blocks = plan.split_n
        ? (c.N + plan.nc - 1) / plan.nc
        : cpu::split_block_count(c.M, plan.mc);
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

// ---- Q5_1 right-hand side --------------------------------------------------
//
// B is a llama.cpp block stream (24 bytes per 32 values) rather than floats.
// The pack step decodes blocks straight into the GEMM's packed panels, so the
// kernel never materialises a dense dequantized B; the arithmetic is still f32
// (or f16 for an f16 A), which is what keeps these GFLOPS comparable with the
// f32/f16 rows above. The difference under test is that B arrives at 0.75
// bytes/value instead of 4 (f32) or 2 (f16), so the pack/decode cost trades
// against weight bandwidth — the balance shifts with M.
template <typename T>
GeMMReport run_geMM_q5(const GeMMCase& c, const std::vector<int>& threads, int iters) {
    const int64_t a_rows = c.ta ? c.K : c.M;
    const int64_t a_cols = c.ta ? c.M : c.K;
    const int64_t b_rows = c.tb ? c.N : c.K;
    const int64_t b_cols = c.tb ? c.K : c.N;

    auto [a_vec, a] = make_tensor<T>(a_rows, a_cols, a_cols + c.pad_a, 1);
    auto [b_vec, b] = make_tensor_q5(b_rows, b_cols, 2);

    MatMulAttributes attrs{};
    attrs.transpose_a = c.ta;
    attrs.transpose_b = c.tb;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);

    const cpu::MatMulPlan plan = cpu::get_matmul_plan(attrs, a_desc, b_desc, 1);

    std::vector<T> out_buf(static_cast<size_t>(descs[0].numel()));
    auto output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {a, b};

    GeMMReport rep;
    rep.name = c.name;
    rep.route = plan.split_n ? "q5-NKM" : "q5-MKN";
    rep.unit = "GFLOPS";
    rep.pack_a = plan.pack_a;
    rep.pack_b = true;
    rep.mc = static_cast<int>(plan.mc);
    rep.nc = static_cast<int>(plan.nc);
    rep.kc = static_cast<int>(plan.kc);
    rep.num_blocks = plan.split_n
        ? (c.N + plan.nc - 1) / plan.nc
        : cpu::split_block_count(c.M, plan.mc);
    rep.b_mb = static_cast<double>(b.storage_bytes()) / (1024.0 * 1024.0);
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

template <typename T>
void bench_q5_case(const GeMMCase& c) {
    const int iters = 20;
    std::vector<int> threads{1, 2, 4, 8};
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 1 && std::find(threads.begin(), threads.end(), hw) == threads.end()) {
        threads.push_back(hw);
    }
    print_report(run_geMM_q5<T>(c, threads, iters));
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
// the B-side the packed route. Packing A is forced by transpose_a (required for
// correctness) or by K > PACK_A_K_THRESHOLD; the row pitch never enters the
// decision, so a pad_a/pad_b pair past the threshold still runs one route twice.
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

// ---- size / shape sweeps ---------------------------------------------------
//
// The point cases above each pin one shape. That answers the routing questions
// but not the one the 20T column kept raising: 256^3 resolves to two M-blocks,
// so its 20T row measures the thread pool start-up, not the kernel, and the
// useful question is where the knee between "pool dominates" and "kernel
// dominates" sits. These sweep it.
//
// Iterations are adaptive here — every cell runs until a fixed wall-clock
// budget elapses — rather than the flat 20 the point cases use. At 2048^3
// twenty iterations is ~30 s per thread count; at 64^3 it is well under a
// millisecond, too short to time at all. A fixed budget also leaves every row
// on the machine for the same wall time, which is what makes the rows
// comparable on a host that drifts under load (see the A/B section above for
// how much it drifts).
//
// The tile columns are the plan resolved at the *largest* pool in the row, not
// at 1 thread: kc shares the residency budget with mc and nc, so the plan is
// per-pool, and the widest pool is the run whose block count bounds the others.

struct SweepShape { int64_t M, K, N; };

std::vector<int> sweep_threads() {
    std::vector<int> threads{1, 2, 4, 8};
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 1 && std::find(threads.begin(), threads.end(), hw) == threads.end()) {
        threads.push_back(hw);
    }
    return threads;
}

void print_sweep_header(const std::vector<int>& threads, const char* unit) {
    std::printf("   %5s %5s %5s  %-5s %4s %4s %4s |", "M", "K", "N", "route", "mc", "nc", "blk");
    for (int t : threads) {
        const std::string label = std::to_string(t) + "T";
        std::printf(" %7s", label.c_str());
    }
    std::printf("   (%s)\n", unit);
}

void print_sweep_row(const SweepShape& s, const char* route, int mc, int nc, int64_t blocks,
                     const std::vector<double>& g) {
    std::printf("   %5lld %5lld %5lld  %-5s %4d %4d %4lld |",
                static_cast<long long>(s.M), static_cast<long long>(s.K),
                static_cast<long long>(s.N), route, mc, nc,
                static_cast<long long>(blocks));
    for (double v : g) { std::printf(" %7.1f", v); }
    std::printf("\n");
}

/// One (M,K,N) fp problem, built once and reused across pool sizes.
template <typename T>
struct SweepOp {
    using OpPtr = decltype(MatMul::create(std::declval<MatMulAttributes>(), Backend::CPU));

    OpPtr op;
    std::vector<T> a_buf, b_buf, out_buf;
    TensorView ins[2], out;
    int64_t M, K, N, blocks;
    int mc, nc;
    bool split_n;
    double flops;

    SweepOp(int64_t m, int64_t k, int64_t n, uint64_t seed, int64_t meta_threads,
            bool tb = false)
        : M(m), K(k), N(n) {
        auto [av, a] = make_tensor<T>(M, K, K, seed);
        auto [bv, b] = tb ? make_tensor<T>(N, K, K, seed + 1)
                          : make_tensor<T>(K, N, N, seed + 1);
        a_buf = std::move(av);
        b_buf = std::move(bv);

        MatMulAttributes attrs{};
        attrs.transpose_b = tb;
        op = MatMul::create(attrs, Backend::CPU);
        const TensorDesc arr[] = {a.desc(), b.desc()};
        auto descs = op->getOutputTensorDesc(arr);
        out_buf.assign(static_cast<size_t>(descs[0].numel()), T{});
        out = test::make_planar(descs[0], out_buf.data());
        ins[0] = a;
        ins[1] = b;

        const cpu::MatMulPlan plan =
            cpu::get_matmul_plan(attrs, a.desc(), b.desc(), meta_threads);
        split_n = plan.split_n;
        mc = static_cast<int>(plan.mc);
        nc = static_cast<int>(plan.nc);
        blocks = split_n ? (N + plan.nc - 1) / plan.nc : cpu::split_block_count(M, plan.mc);
        flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    }

    /// GFLOPS of one budgeted block: warm up, then iterate until @p target_ms
    /// of wall time has elapsed. One block rather than a median of blocks —
    /// at ~100 ms a block is long enough that the spread within it sits far
    /// below the between-size differences this table is read for.
    double time_budget(const ComputeContext& ctx, double target_ms) {
        for (int i = 0; i < 2; ++i) { op->compute(out, ins, ctx, nullptr); }
        const auto t0 = std::chrono::high_resolution_clock::now();
        int64_t iters = 0;
        double ms = 0.0;
        constexpr int64_t kMaxIters = 20000;   // the floor is 64^3 against 100 ms
        do {
            op->compute(out, ins, ctx, nullptr);
            ++iters;
            ms = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - t0).count();
        } while (ms < target_ms && iters < kMaxIters);
        return flops * static_cast<double>(iters) / (ms * 1e6);
    }
};

template <typename T>
void sweep_fp(const char* tag, const std::vector<SweepShape>& shapes,
              const std::vector<int>& threads, double target_ms) {
    const int64_t meta_threads = *std::max_element(threads.begin(), threads.end());
    std::printf("  %s nn sweep  budget=%.0f ms/cell\n", tag, target_ms);
    print_sweep_header(threads, "GFLOPS");
    for (const SweepShape& s : shapes) {
        SweepOp<T> g(s.M, s.K, s.N, 7, meta_threads);
        std::vector<double> gflops;
        for (int nt : threads) {
            SimplePool pool(nt);
            ComputeContext ctx;
            if (nt > 1) { ctx.cpu = pool.cpu; }
            gflops.push_back(g.time_budget(ctx, target_ms));
        }
        print_sweep_row(s, g.split_n ? "NKM" : "MKN", g.mc, g.nc, g.blocks, gflops);
    }
}

/// The int8 counterpart. Same shape as SweepOp, on the s8|u8 × s8 → s32 path.
struct SweepOpI8 {
    using OpPtr = decltype(MatMul::create(std::declval<MatMulAttributes>(), Backend::CPU));

    OpPtr op;
    std::vector<int8_t> a_buf, b_buf;
    std::vector<int32_t> out_buf;
    TensorView ins[2], out;
    int64_t M, K, N, blocks;
    int mc, nc;
    bool split_n;
    double flops;

    SweepOpI8(int64_t m, int64_t k, int64_t n, uint64_t seed, int64_t meta_threads,
              DataType a_dt = DataType::s8)
        : M(m), K(k), N(n) {
        auto [av, a] = make_tensor_i8(M, K, seed, a_dt);
        auto [bv, b] = make_tensor_i8(K, N, seed + 1);
        a_buf = std::move(av);
        b_buf = std::move(bv);

        MatMulAttributes attrs{};
        attrs.output_dtype = DataType::s32;
        op = MatMul::create(attrs, Backend::CPU);
        const TensorDesc arr[] = {a.desc(), b.desc()};
        auto descs = op->getOutputTensorDesc(arr);
        out_buf.assign(static_cast<size_t>(descs[0].numel()), 0);
        out = test::make_planar(descs[0], out_buf.data());
        ins[0] = a;
        ins[1] = b;

        const cpu::MatMulPlan plan =
            cpu::get_matmul_plan(attrs, a.desc(), b.desc(), meta_threads);
        split_n = plan.split_n;
        mc = static_cast<int>(plan.mc);
        nc = static_cast<int>(plan.nc);
        blocks = split_n ? (N + plan.nc - 1) / plan.nc : cpu::split_block_count(M, plan.mc);
        flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    }

    double time_budget(const ComputeContext& ctx, double target_ms) {
        for (int i = 0; i < 2; ++i) { op->compute(out, ins, ctx, nullptr); }
        const auto t0 = std::chrono::high_resolution_clock::now();
        int64_t iters = 0;
        double ms = 0.0;
        constexpr int64_t kMaxIters = 20000;
        do {
            op->compute(out, ins, ctx, nullptr);
            ++iters;
            ms = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - t0).count();
        } while (ms < target_ms && iters < kMaxIters);
        return flops * static_cast<double>(iters) / (ms * 1e6);
    }
};

void sweep_i8(const std::vector<SweepShape>& shapes, const std::vector<int>& threads,
              double target_ms, DataType a_dt = DataType::s8) {
    const int64_t meta_threads = *std::max_element(threads.begin(), threads.end());
    std::printf("  %s nn sweep  budget=%.0f ms/cell\n",
                a_dt == DataType::u8 ? "u8s8" : "s8s8", target_ms);
    print_sweep_header(threads, "GOPS");
    for (const SweepShape& s : shapes) {
        SweepOpI8 g(s.M, s.K, s.N, 7, meta_threads, a_dt);
        std::vector<double> gops;
        for (int nt : threads) {
            SimplePool pool(nt);
            ComputeContext ctx;
            if (nt > 1) { ctx.cpu = pool.cpu; }
            gops.push_back(g.time_budget(ctx, target_ms));
        }
        print_sweep_row(s, g.split_n ? "NKM" : "MKN", g.mc, g.nc, g.blocks, gops);
    }
}

/// The Q5_1 counterpart to SweepOp: A is dense f32/f16, B is a block stream.
template <typename T>
struct SweepOpQ5 {
    using OpPtr = decltype(MatMul::create(std::declval<MatMulAttributes>(), Backend::CPU));

    OpPtr op;
    std::vector<T> a_buf, out_buf;
    std::vector<uint8_t> b_buf;
    TensorView ins[2], out;
    int64_t M, K, N, blocks, b_bytes;
    int mc, nc;
    bool split_n;
    double flops;

    SweepOpQ5(int64_t m, int64_t k, int64_t n, uint64_t seed, int64_t meta_threads,
              bool tb = false)
        : M(m), K(k), N(n) {
        auto [av, a] = make_tensor<T>(M, K, K, seed);
        auto [bv, b] = tb ? make_tensor_q5(N, K, seed + 1)
                          : make_tensor_q5(K, N, seed + 1);
        a_buf = std::move(av);
        b_buf = std::move(bv);
        b_bytes = static_cast<int64_t>(b.storage_bytes());

        MatMulAttributes attrs{};
        attrs.transpose_b = tb;
        op = MatMul::create(attrs, Backend::CPU);
        const TensorDesc arr[] = {a.desc(), b.desc()};
        auto descs = op->getOutputTensorDesc(arr);
        out_buf.assign(static_cast<size_t>(descs[0].numel()), T{});
        out = test::make_planar(descs[0], out_buf.data());
        ins[0] = a;
        ins[1] = b;

        const cpu::MatMulPlan plan =
            cpu::get_matmul_plan(attrs, a.desc(), b.desc(), meta_threads);
        split_n = plan.split_n;
        mc = static_cast<int>(plan.mc);
        nc = static_cast<int>(plan.nc);
        blocks = split_n ? (N + plan.nc - 1) / plan.nc : cpu::split_block_count(M, plan.mc);
        flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    }

    /// GFLOPS of one budgeted block, plus the effective weight bandwidth:
    /// the block stream is swept once per call, so b_bytes / time is what the
    /// pack step actually pulls through the cache hierarchy.
    double time_budget(const ComputeContext& ctx, double target_ms, double* gbps) {
        for (int i = 0; i < 2; ++i) { op->compute(out, ins, ctx, nullptr); }
        const auto t0 = std::chrono::high_resolution_clock::now();
        int64_t iters = 0;
        double ms = 0.0;
        constexpr int64_t kMaxIters = 20000;
        do {
            op->compute(out, ins, ctx, nullptr);
            ++iters;
            ms = std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - t0).count();
        } while (ms < target_ms && iters < kMaxIters);
        if (gbps) {
            *gbps = static_cast<double>(b_bytes) * static_cast<double>(iters)
                  / (ms * 1e6);
        }
        return flops * static_cast<double>(iters) / (ms * 1e6);
    }
};

template <typename T>
void sweep_q5(const char* tag, const std::vector<SweepShape>& shapes,
              const std::vector<int>& threads, double target_ms) {
    const int64_t meta_threads = *std::max_element(threads.begin(), threads.end());
    std::printf("  %s nn sweep (B = q5_1)  budget=%.0f ms/cell\n", tag, target_ms);
    print_sweep_header(threads, "GFLOPS");
    for (const SweepShape& s : shapes) {
        SweepOpQ5<T> g(s.M, s.K, s.N, 7, meta_threads);
        std::vector<double> gflops;
        double bw_first = 0.0, bw_last = 0.0;
        for (size_t i = 0; i < threads.size(); ++i) {
            SimplePool pool(threads[i]);
            ComputeContext ctx;
            if (threads[i] > 1) { ctx.cpu = pool.cpu; }
            double gbps = 0.0;
            gflops.push_back(g.time_budget(ctx, target_ms, &gbps));
            if (i == 0) { bw_first = gbps; }
            bw_last = gbps;
        }
        print_sweep_row(s, g.split_n ? "NKM" : "MKN", g.mc, g.nc, g.blocks, gflops);
        std::printf("        B=%6.2f MB  weight BW: %2dT %5.1f GB/s  %2dT %5.1f GB/s\n",
                    static_cast<double>(g.b_bytes) / (1024.0 * 1024.0),
                    threads.front(), bw_first, threads.back(), bw_last);
    }
}

/// M ladder at a fixed K×N: the packed B panel is reused once per row, so the
/// decode cost per FLOP falls as 1/M until the MMA (or memory) takes over.
/// Each M gets a plan resolved at the thread count it is timed with, and the
/// dense f32 twin is built for the same shape so the ratio shows the knee
/// rather than the host's absolute speed.
void m_ladder_q5(int64_t K, int64_t N, const std::vector<int64_t>& ms,
                 const std::vector<int>& threads, double target_ms,
                 bool tb = false) {
    std::printf("  M ladder q5_1 vs dense f32  K=%lld N=%lld  B=%s  budget=%.0f ms/cell\n",
                static_cast<long long>(K), static_cast<long long>(N),
                tb ? "[N][K]" : "[K][N]", target_ms);
    std::printf("   %5s |", "M");
    for (int nt : threads) {
        std::printf("  %2dT:    q5_1      dense f32     ratio |", nt);
    }
    std::printf("\n");

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    for (int64_t M : ms) {
        std::printf("   %5lld |", static_cast<long long>(M));
        for (int nt : threads) {
            SweepOpQ5<float> q(M, K, N, 7, nt, tb);
            SweepOp<float> d(M, K, N, 7, nt, tb);
            SimplePool pool(nt);
            ComputeContext ctx;
            if (nt > 1) { ctx.cpu = pool.cpu; }

            // Alternate the two formats round by round: on this host a run of
            // one format drifts a few percent against the next, which is the
            // same order as the ratio being measured.
            std::vector<double> vq, vd;
            double gbps = 0.0;
            for (int r = 0; r < 3; ++r) {
                vq.push_back(q.time_budget(ctx, target_ms, &gbps));
                vd.push_back(d.time_budget(ctx, target_ms));
            }
            const double gq = median(vq);
            const double gd = median(vd);
            std::printf(" %10.1f %10.1f %6.3f |", gq, gd, gq / gd);
        }
        std::printf("\n");
    }
}

/// Same A, two weight formats: dense f16 and Q5_1, alternated round by round so
/// the ratio is measured under matched thermal conditions (see the A-side A/B
/// section above). With M small the pair is weight-bandwidth bound and Q5_1
/// should win outright; as M grows the decode overhead is amortised over more
/// FLOPs and the gap narrows toward the pack cost.
template <typename T>
void bench_b_format_ab(const char* tag, int64_t M, int64_t K, int64_t N,
                       int rounds = 5, int iters = 20) {
    using OpPtr = decltype(MatMul::create(std::declval<MatMulAttributes>(), Backend::CPU));

    auto [av, a] = make_tensor<T>(M, K, K, 1);
    auto [bfv, bf] = make_tensor<T>(K, N, N, 2);
    auto [bqv, bq] = make_tensor_q5(K, N, 2);

    MatMulAttributes attrs{};
    auto op_f = MatMul::create(attrs, Backend::CPU);
    auto op_q = MatMul::create(attrs, Backend::CPU);

    const TensorDesc arr_f[] = {a.desc(), bf.desc()};
    const TensorDesc arr_q[] = {a.desc(), bq.desc()};
    auto descs_f = op_f->getOutputTensorDesc(arr_f);
    auto descs_q = op_q->getOutputTensorDesc(arr_q);
    std::vector<T> out_f(static_cast<size_t>(descs_f[0].numel()));
    std::vector<T> out_q(static_cast<size_t>(descs_q[0].numel()));
    auto vout_f = test::make_planar(descs_f[0], out_f.data());
    auto vout_q = test::make_planar(descs_q[0], out_q.data());

    const TensorView ins_f[] = {a, bf};
    const TensorView ins_q[] = {a, bq};

    const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N)
                       * static_cast<double>(K);

    auto time_f = [&]() {
        for (int i = 0; i < 2; ++i) { op_f->compute(vout_f, ins_f, {}, nullptr); }
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) { op_f->compute(vout_f, ins_f, {}, nullptr); }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return flops * iters / (ms * 1e6);
    };
    auto time_q = [&]() {
        for (int i = 0; i < 2; ++i) { op_q->compute(vout_q, ins_q, {}, nullptr); }
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) { op_q->compute(vout_q, ins_q, {}, nullptr); }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return flops * iters / (ms * 1e6);
    };

    const double b_f_mb = static_cast<double>(bf.storage_bytes()) / (1024.0 * 1024.0);
    const double b_q_mb = static_cast<double>(bq.storage_bytes()) / (1024.0 * 1024.0);

    const char* dense = std::is_same_v<T, float> ? "f32" : "f16";
    std::printf("  %s B-format A/B  %lldx%lldx%lld  1T  rounds=%d iters=%d  "
                "B: %s %.2f MB, q5_1 %.2f MB\n",
                tag, static_cast<long long>(M), static_cast<long long>(K),
                static_cast<long long>(N), rounds, iters, dense, b_f_mb, b_q_mb);

    std::vector<double> vf, vq;
    for (int r = 0; r < rounds; ++r) {
        const double gf = time_f();
        const double gq = time_q();
        vf.push_back(gf);
        vq.push_back(gq);
        std::printf("    round %d:  f16=%7.1f  q5_1=%7.1f   q5/f16=%.3f\n",
                    r, gf, gq, gq / gf);
    }

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    const double mf = median(vf), mq = median(vq);
    std::printf("    median:   f16=%7.1f  q5_1=%7.1f   q5/f16=%.3f\n", mf, mq, mq / mf);
}

// Interleaved A/B of the two activation dtypes at identical memory: one byte
// buffer is viewed as s8 and as u8, so the only difference is the dtype
// contract. On x86 the s8 A pack XORs every byte into unsigned range and the B
// packer writes the column sums the accumulators are seeded from; the u8 A
// packs verbatim and the sums block is zeroed. The kernel itself is the same
// code on both sides, so this measures the cost of the bias bookkeeping alone.
void bench_i8_u8_ab(const char* tag, int64_t M, int64_t K, int64_t N,
                    int rounds = 5, int iters = 20) {
    std::vector<int8_t> a_bytes(static_cast<size_t>(M * K));
    nnops::test::XorShift128 rng(1);
    for (auto& v : a_bytes) { v = static_cast<int8_t>(rng.next_u64() % 256); }
    const int64_t ashape[] = {M, K};
    TensorView a_s8(ashape, DataType::s8, a_bytes.data());
    TensorView a_u8(ashape, DataType::u8, a_bytes.data());
    auto [bv, b] = make_tensor_i8(K, N, 2);

    MatMulAttributes attrs{};
    attrs.output_dtype = DataType::s32;
    auto op_s8 = MatMul::create(attrs, Backend::CPU);
    auto op_u8 = MatMul::create(attrs, Backend::CPU);

    const TensorDesc arr_s8[] = {a_s8.desc(), b.desc()};
    const TensorDesc arr_u8[] = {a_u8.desc(), b.desc()};
    auto descs_s8 = op_s8->getOutputTensorDesc(arr_s8);
    auto descs_u8 = op_u8->getOutputTensorDesc(arr_u8);
    std::vector<int32_t> out_s8(static_cast<size_t>(descs_s8[0].numel()));
    std::vector<int32_t> out_u8(static_cast<size_t>(descs_u8[0].numel()));
    auto vout_s8 = test::make_planar(descs_s8[0], out_s8.data());
    auto vout_u8 = test::make_planar(descs_u8[0], out_u8.data());

    const TensorView ins_s8[] = {a_s8, b};
    const TensorView ins_u8[] = {a_u8, b};

    const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N)
                       * static_cast<double>(K);

    auto time_s8 = [&]() {
        for (int i = 0; i < 2; ++i) { op_s8->compute(vout_s8, ins_s8, {}, nullptr); }
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) { op_s8->compute(vout_s8, ins_s8, {}, nullptr); }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return flops * iters / (ms * 1e6);
    };
    auto time_u8 = [&]() {
        for (int i = 0; i < 2; ++i) { op_u8->compute(vout_u8, ins_u8, {}, nullptr); }
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) { op_u8->compute(vout_u8, ins_u8, {}, nullptr); }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return flops * iters / (ms * 1e6);
    };

    std::printf("  int8 A-dtype A/B  %s  %lldx%lldx%lld  1T  rounds=%d iters=%d\n",
                tag, static_cast<long long>(M), static_cast<long long>(K),
                static_cast<long long>(N), rounds, iters);

    std::vector<double> vs, vu;
    for (int r = 0; r < rounds; ++r) {
        const double gs = time_s8();
        const double gu = time_u8();
        vs.push_back(gs);
        vu.push_back(gu);
        std::printf("    round %d:  s8=%7.1f  u8=%7.1f   u8/s8=%.3f\n",
                    r, gs, gu, gu / gs);
    }

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    const double ms = median(vs), mu = median(vu);
    std::printf("    median:   s8=%7.1f  u8=%7.1f   u8/s8=%.3f\n", ms, mu, mu / ms);
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

NNOPS_BENCH(matmul_f32_n_split) {
    // N > M takes the N-split (NKM), where each n-block packs its own B slice.
    // N is not a multiple of nc, so the last block packs a short tile — the case
    // the packed-B slice sizing is about (see num_panels_max).
    bench_case({"nn 256x512x10240 N-split", 256, 512, 10240});
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
    // Deep K. K > PACK_A_K_THRESHOLD packs A on *both* sides — this measures
    // packed vs packed-transposed, not pack vs direct (the plan prints nn=1
    // ta=1). A direct route is unreachable at this depth without raising the
    // threshold.
    bench_route_ab<half>("f16 deep-K (both packed)", "nn", "ta", 512, 2048, 512,
                         /*ta_a=*/false, /*pad_a=*/0,
                         /*ta_b=*/true,  /*pad_b=*/0,
                         /*rounds=*/4, /*iters=*/10);
}

NNOPS_BENCH(matmul_f16_a_pitch_sweep) {
    // How much a wide A row pitch costs the *direct* route. K is below
    // PACK_A_K_THRESHOLD, so pack_a=0 at every pitch and the only variable is
    // the row footprint the direct reads scatter across. Rounds are interleaved
    // across all pitches so the comparison is thermal-matched.
    const int64_t M = 512, K = 512, N = 256;
    const int64_t pads[] = {0, 128, 256, 384, 512, 513, 640, 1024, 3072};
    const int rounds = 5, iters = 20;

    std::vector<RouteAB<half>> routes;
    routes.reserve(std::size(pads));
    for (int64_t pad : pads) {
        routes.emplace_back(M, K, N, /*transpose_a=*/false, 1, pad);
    }

    std::printf("  f16 A row-pitch sweep  %lldx%lldx%lld  1T  rounds=%d iters=%d\n",
                static_cast<long long>(M), static_cast<long long>(K),
                static_cast<long long>(N), rounds, iters);

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

NNOPS_BENCH(matmul_f16_pack_k_flip) {
    // The pack-A threshold crossed. Layout and M/N are held fixed and K goes
    // from PACK_A_K_THRESHOLD to +1 — a 0.1% work difference that flips pack_a
    // from 0 to 1 — so the interleaved ratio is what the threshold buys at the
    // boundary. Rounds alternate the two routes in one process (see
    // bench_route_ab).
    const int64_t M = 512, N = 256;
    const int64_t k_direct = cpu::PACK_A_K_THRESHOLD;
    const int64_t k_packed = cpu::PACK_A_K_THRESHOLD + 1;
    const int rounds = 5, iters = 10;

    RouteAB<half> rd(M, k_direct, N, /*transpose_a=*/false, 1);
    RouteAB<half> rp(M, k_packed, N, /*transpose_a=*/false, 1);

    std::printf("  f16 pack-A K flip  M=%lld N=%lld  1T  rounds=%d iters=%d\n",
                static_cast<long long>(M), static_cast<long long>(N),
                rounds, iters);
    std::printf("    K=%lld pack_a=%d   K=%lld pack_a=%d\n",
                static_cast<long long>(k_direct), static_cast<int>(rd.pack_a),
                static_cast<long long>(k_packed), static_cast<int>(rp.pack_a));

    std::vector<double> vd, vp;
    for (int r = 0; r < rounds; ++r) {
        const double gd = rd.time(iters);
        const double gp = rp.time(iters);
        vd.push_back(gd);
        vp.push_back(gp);
        std::printf("    round %d:  direct=%7.1f  packed=%7.1f   d/p=%.3f\n",
                    r, gd, gp, gd / gp);
    }

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    std::printf("    median:   direct=%7.1f  packed=%7.1f   d/p=%.3f\n",
                median(vd), median(vp), median(vd) / median(vp));
}

// Small GEMMs (M*N*K below GEMM_FAST_PATH_THRESHOLD) take the per-arch naive
// SIMD kernel instead of the scalar reference. Interleaved A/B in one process,
// as elsewhere — but the unit is ns/call, since at these sizes the call overhead
// and not the FLOPs sets the time.
namespace nnops::backend::cpu::reference {
void matmul_ref(const MatMulAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* workspace);
}

NNOPS_BENCH(matmul_small_naive_ab) {
    struct Shape { int64_t M, N, K; };
    const Shape shapes[] = {{1, 512, 1}, {4, 64, 2}, {16, 16, 3},
                            {32, 8, 3},  {1, 1, 512}, {8, 32, 3}};
    const int rounds = 5, iters = 2000;

    std::printf("  small-GEMM naive vs scalar reference  1T  rounds=%d iters=%d  "
                "(ns/call)\n", rounds, iters);

    for (const Shape& s : shapes) {
        auto [a_vec, a] = make_tensor<float>(s.M, s.K, s.K, 1);
        auto [b_vec, b] = make_tensor<float>(s.K, s.N, s.N, 2);

        MatMulAttributes attrs{};
        auto op = MatMul::create(attrs, Backend::CPU);
        const TensorDesc arr[] = {a.desc(), b.desc()};
        auto descs = op->getOutputTensorDesc(arr);
        std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
        auto output = test::make_planar(descs[0], out_buf.data());
        const TensorView ins[] = {a, b};

        const auto ns_per_call = [&](auto&& fn) {
            for (int i = 0; i < 20; ++i) { fn(); }
            const auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; ++i) { fn(); }
            const auto t1 = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
        };

        std::vector<double> vn, vr;
        for (int r = 0; r < rounds; ++r) {
            vn.push_back(ns_per_call([&] { op->compute(output, ins, {}, nullptr); }));
            vr.push_back(ns_per_call([&] {
                cpu::reference::matmul_ref(attrs, output, ins, ComputeContext{}, nullptr);
            }));
        }

        auto median = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        const double mn = median(vn), mr = median(vr);
        std::printf("    M=%2lld N=%4lld K=%3lld   naive=%8.1f  ref=%8.1f   "
                    "ref/naive=%.2f\n",
                    static_cast<long long>(s.M), static_cast<long long>(s.N),
                    static_cast<long long>(s.K), mn, mr, mr / mn);
    }
}

// GEMV (M <= 4): the packed route sweeps B twice (write the packed copy, then
// read it back) for a handful of output rows; the direct route streams B once.
// The packed side is forced by handing the same GEMM over as ta+tb, which is
// the one layout the direct kernel declines. Interleaved A/B at 1T, then a
// thread ladder for the direct route (its N blocks are the parallel unit).
template <typename T>
void bench_gemv_dtype(const char* tag) {
    struct Shape { int64_t M, N, K; };
    const Shape shapes[] = {{1, 2048, 2048}, {2, 2048, 2048}, {3, 2048, 2048},
                            {4, 2048, 2048}, {8, 2048, 2048}};
    const int rounds = 5, iters = 10;

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    std::vector<int> threads{1, 2, 4, 8};
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 1 && std::find(threads.begin(), threads.end(), hw) == threads.end()) {
        threads.push_back(hw);
    }

    std::printf("  GEMV direct (no pack) vs packed  %s  rounds=%d iters=%d\n",
                tag, rounds, iters);

    for (const Shape& s : shapes) {
        // Direct: A [M][K], B [K][N] (the GEMV kernel's own layout).
        auto [a_vec, a] = make_tensor<T>(s.M, s.K, s.K, 1);
        auto [b_vec, b] = make_tensor<T>(s.K, s.N, s.N, 2);
        // Packed: the same GEMM as A^T [K][M] and B^T [N][K].
        auto [at_vec, at] = make_tensor<T>(s.K, s.M, s.M, 3);
        auto [bt_vec, bt] = make_tensor<T>(s.N, s.K, s.K, 4);

        MatMulAttributes attrs_direct{};
        MatMulAttributes attrs_packed{};
        attrs_packed.transpose_a = true;
        attrs_packed.transpose_b = true;
        auto op_direct = MatMul::create(attrs_direct, Backend::CPU);
        auto op_packed = MatMul::create(attrs_packed, Backend::CPU);

        const TensorDesc descs_in[] = {a.desc(), b.desc()};
        auto descs = op_direct->getOutputTensorDesc(descs_in);
        std::vector<T> out_buf(static_cast<size_t>(descs[0].numel()));
        auto output = test::make_planar(descs[0], out_buf.data());
        const TensorView ins_direct[] = {a, b};
        const TensorView ins_packed[] = {at, bt};

        const auto ns_per_call = [&](auto&& fn) {
            for (int i = 0; i < 2; ++i) { fn(); }
            const auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; ++i) { fn(); }
            const auto t1 = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
        };

        std::vector<double> vd, vp;
        for (int r = 0; r < rounds; ++r) {
            vd.push_back(ns_per_call(
                [&] { op_direct->compute(output, ins_direct, ComputeContext{}, nullptr); }));
            vp.push_back(ns_per_call(
                [&] { op_packed->compute(output, ins_packed, ComputeContext{}, nullptr); }));
        }

        const double md = median(vd), mp = median(vp);
        const double flops = 2.0 * static_cast<double>(s.M) *
                             static_cast<double>(s.N) * static_cast<double>(s.K);
        std::printf("    M=%lld N=%lld K=%lld   direct=%8.2f GFLOPS  "
                    "packed=%8.2f GFLOPS   d/p=%.2f\n",
                    static_cast<long long>(s.M), static_cast<long long>(s.N),
                    static_cast<long long>(s.K), flops / md, flops / mp,
                    mp / md);

        // The direct route splits N across threads instead of batching the
        // packing work, so its ladder starts at one row of output.
        std::printf("      direct ladder:");
        const ComputeContext serial{};
        for (int nt : threads) {
            SimplePool pool(nt);
            ComputeContext ctx;
            ctx.cpu = pool.cpu;
            const ComputeContext& use_ctx = (nt > 1) ? ctx : serial;
            const double ns = ns_per_call([&] {
                op_direct->compute(output, ins_direct, use_ctx, nullptr);
            });
            std::printf("  %dT %6.1f", nt, flops / ns);
        }
        std::printf("  GFLOPS\n");
    }
}

NNOPS_BENCH(matmul_gemv_direct_ab) {
    bench_gemv_dtype<float>("f32");
    bench_gemv_dtype<half>("f16");
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

NNOPS_BENCH(matmul_u8s8_nn_large) {
    // Same shape as matmul_i8_nn_large with a u8 activation, so the two rows
    // answer "what does the unsigned-A contract cost?" from the same table.
    bench_i8_case({"u8s8 nn 1024x512x1024", 1024, 512, 1024,
                   /*ta=*/false, /*tb=*/false, /*pad_a=*/0, /*pad_b=*/0,
                   /*f16=*/false, DataType::u8});
}

NNOPS_BENCH(matmul_u8s8_tb_large) {
    bench_i8_case({"u8s8 tb 1024x512x1024", 1024, 512, 1024,
                   /*ta=*/false, /*tb=*/true, /*pad_a=*/0, /*pad_b=*/0,
                   /*f16=*/false, DataType::u8});
}

NNOPS_BENCH(matmul_u8s8_vs_s8_ab) {
    // Interleaved, same process, identical A bytes: the dtype is the only
    // variable, so the ratio is the bias bookkeeping and nothing else.
    bench_i8_u8_ab("nn", 1024, 512, 1024);
}

// ============================================================
// Q5_1 weights — llama.cpp block stream on the RHS
// ============================================================

NNOPS_BENCH(matmul_q5_f32_nn_large) {
    bench_q5_case<float>({"q5/f32 nn 1024x512x1024", 1024, 512, 1024});
}

NNOPS_BENCH(matmul_q5_f32_tb_large) {
    // LLM-style: the weight is stored [N][K] and consumed as A @ W^T, which is
    // the layout a quantized checkpoint is loaded in.
    bench_q5_case<float>({"q5/f32 tb 1024x512x1024", 1024, 512, 1024,
                          /*ta=*/false, /*tb=*/true});
}

NNOPS_BENCH(matmul_q5_f16_nn) {
    bench_q5_case<half>({"q5/f16 nn 1024x512x1024", 1024, 512, 1024,
                         /*ta=*/false, /*tb=*/false, /*pad_a=*/0, /*pad_b=*/0,
                         /*f16=*/true});
}

NNOPS_BENCH(matmul_q5_f16_tb) {
    bench_q5_case<half>({"q5/f16 tb 1024x512x1024", 1024, 512, 1024,
                         /*ta=*/false, /*tb=*/true, /*pad_a=*/0, /*pad_b=*/0,
                         /*f16=*/true});
}

NNOPS_BENCH(matmul_q5_decode) {
    // One token. There is almost no A-side work, so this row is the weight
    // stream's bandwidth limit: the block bytes swept per call are printed as
    // "B=... MB" next to the tile sizes.
    bench_q5_case<float>({"q5/f32 decode 1x4096x4096", 1, 4096, 4096,
                          /*ta=*/false, /*tb=*/true});
}

NNOPS_BENCH(matmul_q5_pack_only) {
    // Isolate the RHS pack from the MMA at the decode shape: the same tile
    // decomposition matmul_block_fused walks (n-block × k-block, row-major B),
    // but no A pack and no micro-kernel. One sweep is one full-B pass, i.e. the
    // pack half of a single matmul call — this is what dominates when M = 1 and
    // the packed panel has no second row to be reused by.
    constexpr int64_t K = 4096, N = 4096;
    auto [av, a] = make_tensor<float>(1, K, K, 1);
    auto [fv, f] = make_tensor<float>(K, N, N, 2);   // dense f32 twin
    auto [bv, b] = make_tensor_q5(K, N, 2);          // block stream

    MatMulAttributes attrs{};
    const cpu::MatMulPlan plan = cpu::get_matmul_plan(attrs, a.desc(), b.desc(), 1);
    const int kc = static_cast<int>(plan.kc);
    const int nc = static_cast<int>(plan.nc);
    constexpr int nr = 16;
    const int64_t row_blocks = b.row_stride_elems();
    std::vector<float> buf(static_cast<size_t>(plan.np_slice * plan.ldd_b));

    // Both physical layouts; the same square buffers serve as [K][N] (nn) and
    // [N][K] (tb) since K == N.
    auto sweep = [&](bool q5, bool trans) {
        for (int n0 = 0; n0 < static_cast<int>(N); n0 += nc) {
            const int width = std::min(nc, static_cast<int>(N - n0));
            for (int k = 0; k < static_cast<int>(K); k += kc) {
                const int actual_kc = std::min(kc, static_cast<int>(K - k));
                const int ldd = static_cast<int>(
                    cpu::mma_pack_align(nr * actual_kc * sizeof(float)) / sizeof(float));
                if (q5) {
                    cpu::tile_pack_rhs_q5_1(trans, n0, width, k, actual_kc,
                                            buf.data(), ldd, bv.data(), row_blocks);
                } else {
                    const float* src = trans
                        ? fv.data() + static_cast<int64_t>(n0) * K + k
                        : fv.data() + static_cast<int64_t>(k) * N + n0;
                    cpu::tile_pack_rhs(trans, width, actual_kc, buf.data(), ldd,
                                       src, static_cast<int>(N), 1.0f);
                }
            }
        }
    };

    const int iters = 20;
    double ms[2][2] = {};
    for (int q5 = 0; q5 < 2; ++q5) {
        for (int tr = 0; tr < 2; ++tr) {
            sweep(q5 == 1, tr == 1);
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; ++i) { sweep(q5 == 1, tr == 1); }
            ms[q5][tr] = std::chrono::duration<double, std::milli>(
                             std::chrono::high_resolution_clock::now() - t0).count() / iters;
        }
    }

    const double values = static_cast<double>(K) * static_cast<double>(N);
    std::printf("  q5_1 pack-only  K=%lld N=%lld  kc=%d nc=%d  (one full-B sweep)\n",
                static_cast<long long>(K), static_cast<long long>(N), kc, nc);
    static constexpr const char* kLabels[2][2] = {
        {"dense f32 nn [K][N]", "dense f32 tb [N][K]"},
        {"q5_1      nn [K][N]", "q5_1      tb [N][K]"},
    };
    for (int q5 = 0; q5 < 2; ++q5) {
        for (int tr = 0; tr < 2; ++tr) {
            std::printf("    %-22s : %7.3f ms  = %6.2f Gvalues/s   (%.2fx dense)\n",
                        kLabels[q5][tr], ms[q5][tr], values / (ms[q5][tr] * 1e6),
                        ms[q5][tr] / ms[0][tr]);
        }
    }
}

NNOPS_BENCH(matmul_q5_m_ladder) {
    // Where does the fused decode stop dominating? Small-M rows are weight
    // (decode) bound, large-M rows are FLOP bound; the ratio column crossing
    // back toward 1 is the knee. Two B sizes: 12 MB (out of cache) and 3 MB.
    const std::vector<int64_t> ms{1, 2, 4, 8, 16, 32, 64, 128, 256};
    m_ladder_q5(4096, 4096, ms, {1, 4}, 100.0);
    m_ladder_q5(2048, 2048, ms, {1, 4}, 100.0);
}

NNOPS_BENCH(matmul_q5_fused_ab) {
#if defined(NNOPS_ARCH_X86_64)
    // The fused route against the packed route, alternated round by round in
    // one process (the only way this host gives a stable ratio). The fused row
    // forces the fused route for every M so the table shows where it stops
    // winning; the shipped gate (Q5_FUSED_MAX_M_*) is the crossover itself.
    // Override < 0 keeps the built-in limits, 0 forces the packed route,
    // > 0 overrides the M limit.
    const int64_t K = 4096, N = 4096;
    const std::vector<int64_t> ms{1, 2, 4, 8, 16, 32, 64, 128};
    const std::vector<int> threads{1, 4};

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    for (int tb = 0; tb < 2; ++tb) {
        std::printf("  fused vs packed q5_1  K=%lld N=%lld  B=%s\n",
                    static_cast<long long>(K), static_cast<long long>(N),
                    tb ? "[N][K]" : "[K][N]");
        std::printf("   %5s |", "M");
        for (int nt : threads) {
            std::printf("   %2dT:    fused     packed     ratio |", nt);
        }
        std::printf("\n");

        for (int64_t M : ms) {
            std::printf("   %5lld |", static_cast<long long>(M));
            for (int nt : threads) {
                SweepOpQ5<float> q(M, K, N, 7, nt, tb != 0);
                SimplePool pool(nt);
                ComputeContext ctx;
                if (nt > 1) { ctx.cpu = pool.cpu; }

                // Order matters as much as host drift: the route measured
                // second inherits the other one's cache/bandwidth state and
                // reads up to 2x slower. Alternate the order per round and
                // take the median of four samples each.
                std::vector<double> vf, vp;
                for (int r = 0; r < 4; ++r) {
                    const bool fused_first = (r % 2) == 0;
                    const int first = fused_first ? 128 : 0;
                    const int second = fused_first ? 0 : 128;
                    cpu::q5_fused_max_m_override = first;
                    const double t1 = q.time_budget(ctx, 40.0, nullptr);
                    cpu::q5_fused_max_m_override = second;
                    const double t2 = q.time_budget(ctx, 40.0, nullptr);
                    (fused_first ? vf : vp).push_back(t1);
                    (fused_first ? vp : vf).push_back(t2);
                }
                cpu::q5_fused_max_m_override = -1;

                const double gf = median(vf), gp = median(vp);
                std::printf(" %9.1f %9.1f %8.3f |", gf, gp, gf / gp);
            }
            std::printf("\n");
        }
    }
#else
    std::printf("  fused Q5_1 route: x86_64 only\n");
#endif
}

NNOPS_BENCH(matmul_q5_m_ladder_tb) {
    // Same ladder against a [N][K] weight (the Linear layout), which is the one
    // the fused dot kernel serves.
    const std::vector<int64_t> ms{1, 2, 4, 8, 16, 32, 64, 128, 256};
    m_ladder_q5(4096, 4096, ms, {1, 4}, 100.0, /*tb=*/true);
    m_ladder_q5(2048, 2048, ms, {1, 4}, 100.0, /*tb=*/true);
}

// Weight-format A/B under matched conditions: same A, dense f16/f32 B vs Q5_1 B.
NNOPS_BENCH(matmul_q5_vs_f16_ab) {
    bench_b_format_ab<half>("f16-act", 512, 512, 256);
}

NNOPS_BENCH(matmul_q5_vs_f32_ab) {
    bench_b_format_ab<float>("f32-act", 512, 512, 256);
}

NNOPS_BENCH(matmul_q5_vs_f16_ab_large) {
    bench_b_format_ab<half>("f16-act", 1024, 1024, 1024, /*rounds=*/4, /*iters=*/10);
}

// ============================================================
// Size and shape sweeps — where the thread pool starts to pay
// ============================================================
//
// Same binary, same shapes as above but read as a table instead of one case at
// a time: the point cases answer "which route", these answer "at what size".
// Everything is capped at 2048^3.

NNOPS_BENCH(matmul_f32_size_sweep) {
    // Square, doubling-ish from 64 to the 2048 cap. The left end is where the
    // pool has nothing to hand out (256^3 = 2 M-blocks, which is why its point
    // case reads 0.48x at 20T) and the right end is past L2/L3.
    sweep_fp<float>("f32", {
        {64, 64, 64},       {96, 96, 96},       {128, 128, 128},   {192, 192, 192},
        {256, 256, 256},    {384, 384, 384},    {512, 512, 512},   {768, 768, 768},
        {1024, 1024, 1024}, {1536, 1536, 1536}, {2048, 2048, 2048},
    }, sweep_threads(), 120.0);
}

NNOPS_BENCH(matmul_f32_shape_sweep) {
    // Asymmetric shapes, each isolating one axis as the one that has to carry
    // the parallelism. M=1 decode is the extreme: there is no M to split at
    // all, so the block count is whatever N yields.
    sweep_fp<float>("f32", {
        {1, 2048, 2048},      // decode: one token, M is not a splittable axis
        {16, 2048, 2048},     // small-batch decode
        {2048, 512, 128},     // tall-skinny: only M splits
        {128, 512, 2048},     // short-fat: N > M, takes the NKM split
        {512, 2048, 512},     // deep K: kc blocking, not tile count, dominates
        {2048, 2048, 2048},   // the cap, as the square reference point
    }, sweep_threads(), 120.0);
}

NNOPS_BENCH(matmul_f16_size_sweep) {
    // f16 runs at the F16C cvt→compute→cvt rate (~78 GFLOPS at 1T on this
    // host), so 2048^3 is a ~220 ms iteration — one per cell here.
    sweep_fp<half>("f16", {
        {128, 128, 128},   {256, 256, 256},   {384, 384, 384},   {512, 512, 512},
        {768, 768, 768},   {1024, 1024, 1024},{1536, 1536, 1536}, {2048, 2048, 2048},
    }, sweep_threads(), 120.0);
}

NNOPS_BENCH(matmul_i8_size_sweep) {
    // The int8 point case 256^3 collapses to 0.08x at 20T (mc=24, nc=240, 11
    // blocks — the kc budget shrinks the tiles as the pool grows). This walks
    // the same curve at a size where the tiles have room as well.
    sweep_i8({
        {128, 128, 128},    {192, 192, 192},    {256, 256, 256},   {384, 384, 384},
        {512, 512, 512},    {768, 768, 768},    {1024, 1024, 1024},
        {1536, 1536, 1536}, {2048, 2048, 2048},
    }, sweep_threads(), 120.0);
}

NNOPS_BENCH(matmul_q5_size_sweep) {
    // Same square ladder as f32 so the two tables can be read side by side:
    // the delta at each size is what the block-stream pack costs (or saves).
    sweep_q5<float>("f32", {
        {64, 64, 64},       {96, 96, 96},       {128, 128, 128},   {192, 192, 192},
        {256, 256, 256},    {384, 384, 384},    {512, 512, 512},   {768, 768, 768},
        {1024, 1024, 1024}, {1536, 1536, 1536}, {2048, 2048, 2048},
    }, sweep_threads(), 120.0);
}

NNOPS_BENCH(matmul_q5_decode_sweep) {
    // Decode-shaped rows: M stays small while K and N grow, so B is the whole
    // working set and the weight-BW line is the real limit. The M ladder at
    // fixed 4096^2 shows where the decode cost stops dominating.
    sweep_q5<float>("f32 decode", {
        {1, 1024, 1024},  {1, 2048, 2048},  {1, 4096, 4096},
        {16, 4096, 4096}, {64, 4096, 4096}, {256, 4096, 4096},
    }, sweep_threads(), 120.0);
}

NNOPS_BENCH(matmul_q5_decode_rate) {
#if defined(NNOPS_ARCH_X86_64)
    // Raw block-decode throughput: no A, no panel, no output traffic. This is
    // the floor the fused route can reach — the packed route additionally
    // writes and re-reads every decoded value.
    const int64_t rows = 4096, blocks_per_row = 128;   // 4096 x 4096 values
    auto [bv, b] = make_tensor_q5(rows, blocks_per_row * 32, 3);
    const uint8_t* p = bv.data();
    const int64_t nblk = rows * blocks_per_row;
    const int64_t values = nblk * 32;

    double best = 1e30;
    for (int r = 0; r < 5; ++r) {
        __m256 acc = _mm256_setzero_ps();
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int64_t i = 0; i < nblk; ++i) {
            __m256 v0, v1, v2, v3;
            cpu::arch::decode_q5_1_block_ps(p + i * 24, v0, v1, v2, v3);
            acc = _mm256_add_ps(acc, _mm256_add_ps(_mm256_add_ps(v0, v1),
                                                   _mm256_add_ps(v2, v3)));
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        if (_mm256_cvtss_f32(acc) == 12345.0f) { std::printf(" "); }
        best = std::min(best, ms);
    }
    std::printf("  q5_1 decode-only  4096x4096 blocks: %.3f ms  = %.2f Gvalues/s\n",
                best, static_cast<double>(values) / (best * 1e6));

    // Same stream through the packer's 32-value decode (writes a small stack
    // array per block, no panel). All four vectors are read back out, or the
    // compiler drops the stores and the number is meaningless.
    best = 1e30;
    for (int r = 0; r < 5; ++r) {
        float vals[32];
        __m256 acc = _mm256_setzero_ps();
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int64_t i = 0; i < nblk; ++i) {
            cpu::arch::decode_q5_1_block<float>(p + i * 24, vals);
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(vals));
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(vals + 8));
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(vals + 16));
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(vals + 24));
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        if (_mm256_cvtss_f32(acc) == 12345.0f) { std::printf(" "); }
        best = std::min(best, ms);
    }
    std::printf("  q5_1 decode+stack 4096x4096 blocks: %.3f ms  = %.2f Gvalues/s\n",
                best, static_cast<double>(values) / (best * 1e6));
#else
    std::printf("  q5_1 decode rate: x86_64 only\n");
#endif
}
