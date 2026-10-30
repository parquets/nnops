/// @file bench_matmul.cpp
/// @brief MatMul micro-benchmarks — route decisions, weight formats, thread scaling.
///
/// Most cases run the plan-driven path: B packed, A packed on transpose_a, K >
/// PACK_A_K_THRESHOLD or an int8 A (the row pitch never enters that decision),
/// across {nn, ta, tb, tatb} and padded strides; the GEMV, small-GEMM and fused
/// block-decode cases deliberately take the pack-free route. Point cases report
/// the split direction (NKM / MKN), the resolved (mc, nc, kc) and block count
/// from get_matmul_plan(), and GFLOPS at 1, 2, 4, 8 and hardware threads. The
/// sweeps print the same binary as a table, answering "at what size" rather than
/// "which route". Units are GFLOPS (GOPS for int8, ns/call for the small-GEMM
/// case, Gvalues/s for the pack and decode probes).
///
/// Reading the tables:
///   - BenchPool is an OpenMP team when the build has one (unless
///     NNOPS_BENCH_OMP opts out), otherwise a pool that spawns its workers per
///     call. That spawn is why the 8T/20T columns sag for *every* route here;
///     the dq materialized path adds a quantize `ctx.cpu.run`, so it sags a
///     little earlier.
///   - The host drifts: two separate runs of the *same* route differ by >30% on
///     a fanless machine, and sustained 4T load drops it into throttled episodes
///     lasting seconds, during which both arms read 4-5x slow. Every A/B here
///     builds both arms up front and interleaves them round by round in one
///     process; a row whose multi-threaded numbers matter has its own
///     registration so it can be run in its own process. Read the ratios and the
///     medians, not the absolute GFLOPS.
///   - Order matters as much as drift: the route measured second inherits the
///     first one's cache and bandwidth state and reads up to 2x slower. Every
///     A/B interleaves both arms in one process; the dq, fused-vs-packed and
///     granularity loops also swap which arm goes first each round.
///   - Sweep cells use a wall-clock budget instead of a fixed count — twenty
///     iterations is ~30 s at 2048^3 and under a millisecond at 64^3 — and a
///     fixed budget also leaves every row on the machine for the same time.
///   - Sweep tile columns are the plan resolved at the *largest* pool in the
///     row: kc shares the residency budget with mc and nc, so the plan is
///     per-pool (the fp and q4 point cases report the 1-thread plan instead).
///
/// 256^3 resolves to two M-blocks, so its 20T column measures thread-pool
/// start-up rather than the kernel; finding the knee between the two is what the
/// size sweeps are for.
///
/// Q4_1 right-hand side: B is a llama.cpp block stream (20 bytes per 32 values)
/// and the pack step decodes blocks straight into the GEMM's packed panels, so no
/// dense dequantized B is ever materialised. The arithmetic stays f32 (f16 for an
/// f16 A), which keeps these rows comparable with the dense ones; what is under
/// test is 0.625 bytes/value against 4 or 2 — pack/decode cost traded against
/// weight bandwidth. The GEMV gate and the fused small-M kernels are the two ends
/// of that trade, and the balance shifts with M.
///
/// dq route (fp activations × s8 weights, fp out): quantization parameters are
/// fixed offline, A arrives as f32/f16, B is s8 with one scale per output
/// channel, C comes back in A's dtype. The materialized call does three passes —
/// quantize A, run the int8 GEMM, dequantize the accumulate — yet the GFLOPS are
/// still 2·M·N·K, so a dq row reads directly against the dense fp row of the
/// same shape. The s8 A rows are symmetric (zp 0), making the compensation the
/// identity and letting the A pack quantize on its own during an M-split with no
/// A_q materialised; the u8 rows (zp 128) keep the materialised A_q. The fused
/// pack drops the quantize pass and its `ctx.cpu.run`: measured +1/+4/+14/+20%
/// at 1/2/4/8T on 1024x512x1024, a wash at 256^3 1T (the win scales with thread
/// count, not FLOPs). Baseline on the dev host: dq/dense is ~3.3 at 1T for f32
/// activations and ~3.3 at 4T for f16 (the dense B is only half as wide there,
/// but the f16 dense route also scales worse); 1024x512x1024 dq runs at ~87% of
/// matmul_i8_nn_large on one thread — the int32 accumulator round trip plus the
/// quantize pass — and skipping the reduction sweep on the identity compensation
/// is worth +13% at 256^3, one thread.

#include "nnops/ops/matmul.hpp"
#include "nnops/detail/q4_1.hpp"         // q4_1_block, float_to_half, half_to_bits
#include "backend/cpu/matmul/matmul_helper.h"   // get_matmul_plan, PACK_A_K_THRESHOLD, tile constants
#include "backend/cpu/simd_kernel/simd_quant.hpp"  // quant_kernel::quantization (the dq quantize pass)
#include "common/bench_harness.hpp"
#include "common/cpu_pool.hpp"
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

// The bench's CpuBackend: OpenMP team or spawn-per-call pool, behind the same
// hooks; a table differs only in its scheduler (see common/cpu_pool.hpp).
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

/// [rows, cols] tensor of T with an explicit row pitch; padding lanes stay 0.
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

/// Compact i8 tensor (s8 by default, u8 on request) with identity quant params,
/// so the kernel accumulates the raw dot — the qgemm / MatMulInteger analog.
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

/// Compact Q4_1 [rows, cols] block stream (32 four-bit quants per 20-byte block);
/// realistic d/m so no lane is denormal or inf.
std::pair<std::vector<uint8_t>, TensorView>
make_tensor_q4(int64_t rows, int64_t cols, uint64_t seed) {
    const int64_t blocks_per_row = (cols + q4_1_block_elems - 1) / q4_1_block_elems;
    std::vector<uint8_t> buf(static_cast<size_t>(
        rows * blocks_per_row * static_cast<int64_t>(q4_1_block_bytes)));

    nnops::test::XorShift128 rng(seed);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t b = 0; b < blocks_per_row; ++b) {
            q4_1_block blk{};
            blk.d = cpu::half_to_bits(cpu::float_to_half(rng.next_float(0.002f, 0.02f)));
            blk.m = cpu::half_to_bits(cpu::float_to_half(rng.next_float(-0.1f, 0.1f)));
            for (int j = 0; j < 16; ++j) {
                const uint32_t q0 = static_cast<uint32_t>(rng.next_u64() & 0x0F);
                const uint32_t q1 = static_cast<uint32_t>(rng.next_u64() & 0x0F);
                blk.qs[j] = static_cast<uint8_t>(q0 | (q1 << 4));
            }
            std::memcpy(buf.data() + static_cast<size_t>(
                            (r * blocks_per_row + b)
                            * static_cast<int64_t>(q4_1_block_bytes)),
                        &blk, q4_1_block_bytes);
        }
    }

    const int64_t shape[] = {rows, cols};
    TensorView tv(shape, DataType::q4_1, buf.data());
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
    const char* route;   // "NKM"/"MKN", format-prefixed on the quantized rows
    const char* unit;    // "GFLOPS" or "GOPS"
    bool pack_a, pack_b;
    int mc, nc, kc;
    int64_t num_blocks;
    double b_mb = 0.0;   // B storage footprint (0 = don't print)
    std::vector<int> threads;
    std::vector<double> gflops;
};

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
        BenchPool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

        // Warm-up; also faults the pages.
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

/// int8 (s8|u8 × s8 → s32) twin of run_geMM; the unit is GOPS.
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

    // Tiles are per-pool, so describe the largest pool's plan.
    const int meta_threads = *std::max_element(threads.begin(), threads.end());
    const cpu::MatMulPlan plan = cpu::get_matmul_plan(attrs, a_desc, b_desc, meta_threads);

    std::vector<int32_t> out_buf(static_cast<size_t>(descs[0].numel()));
    auto output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {a, b};

    GeMMReport rep;
    rep.name = c.name;
    rep.route = plan.split_n ? "int8-NKM" : "int8-MKN";
    rep.unit = "GOPS";
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
        BenchPool pool(nt);
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

// ---- Q4_1 right-hand side --------------------------------------------------
//
// B arrives at 0.625 bytes/value instead of 4 (f32) or 2 (f16); the file header
// covers what the pack step does with it.
template <typename T>
GeMMReport run_geMM_q4(const GeMMCase& c, const std::vector<int>& threads, int iters) {
    const int64_t a_rows = c.ta ? c.K : c.M;
    const int64_t a_cols = c.ta ? c.M : c.K;
    const int64_t b_rows = c.tb ? c.N : c.K;
    const int64_t b_cols = c.tb ? c.K : c.N;

    auto [a_vec, a] = make_tensor<T>(a_rows, a_cols, a_cols + c.pad_a, 1);
    auto [b_vec, b] = make_tensor_q4(b_rows, b_cols, 2);

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
    rep.route = plan.split_n ? "q4-NKM" : "q4-MKN";
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
        BenchPool pool(nt);
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
void bench_q4_case(const GeMMCase& c) {
    const int iters = 20;
    std::vector<int> threads{1, 2, 4, 8};
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 1 && std::find(threads.begin(), threads.end(), hw) == threads.end()) {
        threads.push_back(hw);
    }
    print_report(run_geMM_q4<T>(c, threads, iters));
}

// ---- controlled A/B between two routes ------------------------------------
//
// The A-side of each pair is built to take the unpacked-A ("direct") route and
// the B-side the packed route; a pad_a/pad_b pair past the threshold runs one
// route twice, since the row pitch never enters the pack decision.
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

/// Interleaves @p la and @p lb round by round and reports per-round ratios plus
/// the median; @p la is the expected unpacked-A (direct) side.
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
// Point cases pin one shape each; these walk the knee between "pool dominates"
// and "kernel dominates", with a wall-clock budget per cell and the tiles
// resolved at the largest pool in the row (both explained in the file header).

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
    /// elapses. One block is long enough that its spread sits below the
    /// between-size differences this table is read for.
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
            BenchPool pool(nt);
            ComputeContext ctx;
            if (nt > 1) { ctx.cpu = pool.cpu; }
            gflops.push_back(g.time_budget(ctx, target_ms));
        }
        print_sweep_row(s, g.split_n ? "NKM" : "MKN", g.mc, g.nc, g.blocks, gflops);
    }
}

/// int8 counterpart of SweepOp, on the s8|u8 × s8 → s32 path.
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
            BenchPool pool(nt);
            ComputeContext ctx;
            if (nt > 1) { ctx.cpu = pool.cpu; }
            gops.push_back(g.time_budget(ctx, target_ms));
        }
        print_sweep_row(s, g.split_n ? "NKM" : "MKN", g.mc, g.nc, g.blocks, gops);
    }
}

/// The Q4_1 counterpart to SweepOp: A is dense f32/f16, B is a block stream.
template <typename T>
struct SweepOpQ4 {
    using OpPtr = decltype(MatMul::create(std::declval<MatMulAttributes>(), Backend::CPU));

    OpPtr op;
    std::vector<T> a_buf, out_buf;
    std::vector<uint8_t> b_buf;
    TensorView ins[2], out;
    int64_t M, K, N, blocks, b_bytes;
    int mc, nc;
    bool split_n;
    double flops;

    SweepOpQ4(int64_t m, int64_t k, int64_t n, uint64_t seed, int64_t meta_threads,
              bool tb = false)
        : M(m), K(k), N(n) {
        auto [av, a] = make_tensor<T>(M, K, K, seed);
        auto [bv, b] = tb ? make_tensor_q4(N, K, seed + 1)
                          : make_tensor_q4(K, N, seed + 1);
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

    /// GFLOPS of one budgeted block, plus the weight bandwidth it pulls: the
    /// block stream is swept once per call, so this is b_bytes / time.
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
void sweep_q4(const char* tag, const std::vector<SweepShape>& shapes,
              const std::vector<int>& threads, double target_ms) {
    const int64_t meta_threads = *std::max_element(threads.begin(), threads.end());
    std::printf("  %s nn sweep (B = q4_1)  budget=%.0f ms/cell\n", tag, target_ms);
    print_sweep_header(threads, "GFLOPS");
    for (const SweepShape& s : shapes) {
        SweepOpQ4<T> g(s.M, s.K, s.N, 7, meta_threads);
        std::vector<double> gflops;
        double bw_first = 0.0, bw_last = 0.0;
        for (size_t i = 0; i < threads.size(); ++i) {
            BenchPool pool(threads[i]);
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

/// M ladder at a fixed KxN: the packed B panel is reused once per output row, so
/// the decode cost per FLOP falls as 1/M until the MMA (or memory) takes over.
/// The dense f32 twin is built at the same thread count so the ratio shows the
/// knee, not the host's absolute speed.
///
/// @p packed pins the block format to the packed route (the fused small-M kernels
/// off), so the small-M cells compare formats, not routes; the dense arm has no
/// such switch and GEMVs to M = 4 regardless.
void m_ladder_q4(int64_t K, int64_t N, const std::vector<int64_t>& ms,
                 const std::vector<int>& threads, double target_ms,
                 bool tb = false, bool packed = false) {
    std::printf("  M ladder q4_1 vs dense f32  K=%lld N=%lld  B=%s  budget=%.0f ms/cell\n",
                static_cast<long long>(K), static_cast<long long>(N),
                tb ? "[N][K]" : "[K][N]", target_ms);
    if (packed) {
        std::printf("    q4_1 pinned to the packed route; the dense arm GEMVs to M = 4,\n"
                    "    so the small-M cells compare the format, not two routes\n");
    } else if (tb) {
        std::printf("    as routed: q4_1 fuses to M = 16 on [N][K]; the dense arm\n"
                    "    GEMVs to M = 4, so the small-M cells compare routes\n");
    } else {
        std::printf("    as routed: q4_1 fuses to M = 8 on [K][N]; the dense arm\n"
                    "    GEMVs to M = 4, so the small-M cells compare routes\n");
    }
    std::printf("   %5s |", "M");
    for (int nt : threads) {
        std::printf("  %2dT:    q4_1      dense f32     ratio |", nt);
    }
    std::printf("\n");

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    for (int64_t M : ms) {
        std::printf("   %5lld |", static_cast<long long>(M));
        for (int nt : threads) {
            SweepOpQ4<float> q(M, K, N, 7, nt, tb);
            SweepOp<float> d(M, K, N, 7, nt, tb);
            BenchPool pool(nt);
            ComputeContext ctx;
            if (nt > 1) { ctx.cpu = pool.cpu; }
#if defined(NNOPS_ARCH_X86_64)
            if (packed) { cpu::qblock_fused_max_m_override = 0; }
#endif

            // Interleave the formats: host drift is the same order as the ratio.
            std::vector<double> vq, vd;
            double gbps = 0.0;
            for (int r = 0; r < 3; ++r) {
                vq.push_back(q.time_budget(ctx, target_ms, &gbps));
                vd.push_back(d.time_budget(ctx, target_ms));
            }
#if defined(NNOPS_ARCH_X86_64)
            cpu::qblock_fused_max_m_override = -1;
#endif
            const double gq = median(vq);
            const double gd = median(vd);
            std::printf(" %10.1f %10.1f %6.3f |", gq, gd, gq / gd);
        }
        std::printf("\n");
    }
}

/// Same A, dense vs Q4_1 B, interleaved round by round. Small M is
/// weight-bandwidth bound and Q4_1 wins outright; as M grows the decode is
/// amortised over more FLOPs and the gap narrows toward the pack cost.
template <typename T>
void bench_b_format_ab(const char* tag, int64_t M, int64_t K, int64_t N,
                       int rounds = 5, int iters = 20) {
    using OpPtr = decltype(MatMul::create(std::declval<MatMulAttributes>(), Backend::CPU));

    auto [av, a] = make_tensor<T>(M, K, K, 1);
    auto [bfv, bf] = make_tensor<T>(K, N, N, 2);
    auto [bqv, bq] = make_tensor_q4(K, N, 2);

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
                "B: %s %.2f MB, q4_1 %.2f MB\n",
                tag, static_cast<long long>(M), static_cast<long long>(K),
                static_cast<long long>(N), rounds, iters, dense, b_f_mb, b_q_mb);

    std::vector<double> vf, vq;
    for (int r = 0; r < rounds; ++r) {
        const double gf = time_f();
        const double gq = time_q();
        vf.push_back(gf);
        vq.push_back(gq);
        std::printf("    round %d:  %s=%7.1f  q4_1=%7.1f   q4/%s=%.3f\n",
                    r, dense, gf, gq, dense, gq / gf);
    }

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    const double mf = median(vf), mq = median(vq);
    std::printf("    median:   %s=%7.1f  q4_1=%7.1f   q4/%s=%.3f\n", dense, mf, mq,
                dense, mq / mf);
}

// Interleaved A/B of the two activation dtypes over identical memory: one byte
// buffer viewed as s8 and as u8. On x86 the s8 A pack XORs every byte into
// unsigned range and the B packer writes the column sums the accumulators are
// seeded from; u8 packs verbatim and zeroes that block. Same kernel both sides,
// so this is the bias bookkeeping alone.
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

// ---- integer destinations: s32 vs s8 vs u8 ---------------------------------
//
// Same GEMM, same operands, only the output pass differs:
//   s32  a bias-only pass over the raw accumulate, in place — no reductions,
//        since `int8_passes` stays off when the compensation is the identity
//        and a bias alone is no reason to run it (int8_epilogue_row_bias);
//   s8   the reductions sweep (the requant reads r_a / r_b), then
//   u8   compensate + bias + requant + the saturating pack, per element.
//
// s8/s32 is what an integer destination costs over the raw accumulate; u8/s8 is
// the wider clamp (packus vs packs) alone. Zero points are symmetric (W8A8).
void bench_i8_out_ab(const char* tag, int64_t M, int64_t K, int64_t N,
                     int rounds = 5, int iters = 20) {
    auto [a_buf, a] = make_tensor_i8(M, K, 1);
    auto [b_buf, b] = make_tensor_i8(K, N, 2);

    std::vector<int32_t> bias_buf(static_cast<size_t>(N));
    {
        nnops::test::XorShift128 rng(3);
        for (auto& v : bias_buf) { v = static_cast<int32_t>(rng.next_u64() % 4096) - 2048; }
    }
    const int64_t bshape[] = {N};
    TensorView bias(bshape, DataType::s32, bias_buf.data());

    // The requant reads the output's own scale / zero point.
    auto per_tensor = [](float s, int32_t zp) {
        QuantParams qp;
        qp.granularity = QuantGranularity::PerTensor;
        qp.scale = s;
        qp.zero_point = zp;
        return qp;
    };
    const QuantParams oq_s8 = per_tensor(0.25f, -7);
    const QuantParams oq_u8 = per_tensor(0.25f, 128);

    MatMulAttributes attrs_s32{}, attrs_s8{}, attrs_u8{};
    attrs_s32.output_dtype = DataType::s32;
    attrs_s8.output_dtype = DataType::s8;
    attrs_u8.output_dtype = DataType::u8;
    auto op_s32 = MatMul::create(attrs_s32, Backend::CPU);
    auto op_s8 = MatMul::create(attrs_s8, Backend::CPU);
    auto op_u8 = MatMul::create(attrs_u8, Backend::CPU);

    const TensorDesc arr[] = {a.desc(), b.desc(), bias.desc()};
    const TensorDesc od_s32 = op_s32->getOutputTensorDesc(arr)[0];
    const TensorDesc od_s8 = op_s8->getOutputTensorDesc(arr)[0];
    const TensorDesc od_u8 = op_u8->getOutputTensorDesc(arr)[0];

    std::vector<int32_t> out_s32(static_cast<size_t>(od_s32.numel()));
    std::vector<int8_t> out_s8(static_cast<size_t>(od_s8.numel()));
    std::vector<uint8_t> out_u8(static_cast<size_t>(od_u8.numel()));
    const int64_t oshape[] = {M, N};
    const std::span<const int64_t> ospan(oshape, 2);
    auto vout_s32 = test::make_planar(od_s32, out_s32.data());
    auto vout_s8 = TensorView(ospan, DataType::s8, out_s8.data(),
                              TensorLayout::NCHW, oq_s8);
    auto vout_u8 = TensorView(ospan, DataType::u8, out_u8.data(),
                              TensorLayout::NCHW, oq_u8);

    const TensorView ins[] = {a, b, bias};
    const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N)
                       * static_cast<double>(K);

    auto timed = [&](auto&& fn) {
        for (int i = 0; i < 2; ++i) { fn(); }
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) { fn(); }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return flops * iters / (ms * 1e6);
    };

    std::printf("  int8 out-dtype A/B  %s  %lldx%lldx%lld  1T  rounds=%d iters=%d\n",
                tag, static_cast<long long>(M), static_cast<long long>(K),
                static_cast<long long>(N), rounds, iters);

    std::vector<double> v32, v8, vu8;
    for (int r = 0; r < rounds; ++r) {
        const double g32 = timed([&] { op_s32->compute(vout_s32, ins, {}, nullptr); });
        const double g8  = timed([&] { op_s8->compute(vout_s8, ins, {}, nullptr); });
        const double gu8 = timed([&] { op_u8->compute(vout_u8, ins, {}, nullptr); });
        v32.push_back(g32);
        v8.push_back(g8);
        vu8.push_back(gu8);
        std::printf("    round %d:  s32=%7.1f  s8=%7.1f  u8=%7.1f   s8/s32=%.3f"
                    "  u8/s32=%.3f\n", r, g32, g8, gu8, g8 / g32, gu8 / g32);
    }

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    const double m32 = median(v32), m8 = median(v8), mu8 = median(vu8);
    std::printf("    median:   s32=%7.1f  s8=%7.1f  u8=%7.1f   s8/s32=%.3f"
                "  u8/s32=%.3f  u8/s8=%.3f\n",
                m32, m8, mu8, m8 / m32, mu8 / m32, mu8 / m8);
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
    // Paired with matmul_f32_nn_direct_large / matmul_f32_ta_mkn; packed and
    // direct share the M panel cascade, so nn/ta isolates pack-vs-direct alone.
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
    bench_case({"nn pad_b stride pack_b", 1024, 512, 1024, /*ta=*/false, /*tb=*/false,
                /*pad_a=*/0, /*pad_b=*/4096});
}

NNOPS_BENCH(matmul_f32_n_split) {
    // N > M takes the N-split (NKM), each n-block packing its own B slice; N is
    // not a multiple of nc, so the last block packs a short tile.
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

// As for f32, packed and direct tile M with the same cascade, so the ratio is
// not tangled up with panel height.
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
    // K > PACK_A_K_THRESHOLD packs A on *both* sides: packed vs
    // packed-transposed, not pack vs direct (the plan prints nn=1 ta=1). A
    // direct route is unreachable at this depth without raising the threshold.
    bench_route_ab<half>("f16 deep-K (both packed)", "nn", "ta", 512, 2048, 512,
                         /*ta_a=*/false, /*pad_a=*/0,
                         /*ta_b=*/true,  /*pad_b=*/0,
                         /*rounds=*/4, /*iters=*/10);
}

NNOPS_BENCH(matmul_f16_a_pitch_sweep) {
    // What a wide A row pitch costs the *direct* route: K is below
    // PACK_A_K_THRESHOLD, so pack_a=0 at every pitch and the row footprint the
    // direct reads scatter across is the only variable. Rounds interleave the
    // pitches, so the comparison is thermal-matched.
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
    // K goes from PACK_A_K_THRESHOLD to +1 — a 0.1% work difference that flips
    // pack_a from 0 to 1 — so the interleaved ratio is what the threshold buys
    // at the boundary.
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

// Below GEMM_FAST_PATH_THRESHOLD the per-arch naive SIMD kernel replaces the
// scalar reference. ns/call, because at these sizes the call overhead and not
// the FLOPs sets the time.
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

// GEMV (M <= 4): the packed route sweeps B twice (write the packed copy, read it
// back) for a handful of output rows, the direct route streams B once. Three
// direct layouts are timed — [K][N] cache-resident, [K][N] at 64 MB, and [N][K]
// (transpose_b, the decode layout) — with the packed side forced by ta+tb, the
// one layout the direct kernel declines. Interleaved at 1T, then a thread ladder
// for the direct route.
template <typename T>
void bench_gemv_dtype(const char* tag) {
    struct Shape { int64_t M, N, K; };
    const Shape shapes[] = {{1, 2048, 2048}, {2, 2048, 2048}, {4, 2048, 2048},
                            {1, 4096, 4096}, {2, 4096, 4096}, {4, 4096, 4096},
                            {8, 2048, 2048}};
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

        // transpose_b: A [M][K] against a [N][K] weight, the decode layout.
        auto [bt2_vec, bt2] = make_tensor<T>(s.N, s.K, s.K, 5);

        MatMulAttributes attrs_direct{};
        MatMulAttributes attrs_tb{};
        attrs_tb.transpose_b = true;
        MatMulAttributes attrs_packed{};
        attrs_packed.transpose_a = true;
        attrs_packed.transpose_b = true;
        auto op_direct = MatMul::create(attrs_direct, Backend::CPU);
        auto op_tb = MatMul::create(attrs_tb, Backend::CPU);
        auto op_packed = MatMul::create(attrs_packed, Backend::CPU);

        const TensorDesc descs_in[] = {a.desc(), b.desc()};
        auto descs = op_direct->getOutputTensorDesc(descs_in);
        std::vector<T> out_buf(static_cast<size_t>(descs[0].numel()));
        auto output = test::make_planar(descs[0], out_buf.data());
        const TensorView ins_direct[] = {a, b};
        const TensorView ins_tb[] = {a, bt2};
        const TensorView ins_packed[] = {at, bt};

        const auto ns_per_call = [&](auto&& fn) {
            for (int i = 0; i < 2; ++i) { fn(); }
            const auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; ++i) { fn(); }
            const auto t1 = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
        };

        std::vector<double> vd, vt, vp;
        for (int r = 0; r < rounds; ++r) {
            vd.push_back(ns_per_call(
                [&] { op_direct->compute(output, ins_direct, ComputeContext{}, nullptr); }));
            vt.push_back(ns_per_call(
                [&] { op_tb->compute(output, ins_tb, ComputeContext{}, nullptr); }));
            vp.push_back(ns_per_call(
                [&] { op_packed->compute(output, ins_packed, ComputeContext{}, nullptr); }));
        }

        const double md = median(vd), mt = median(vt), mp = median(vp);
        const double flops = 2.0 * static_cast<double>(s.M) *
                             static_cast<double>(s.N) * static_cast<double>(s.K);
        std::printf("    M=%lld N=%lld K=%lld   direct=%8.2f  tb=%8.2f  "
                    "packed=%8.2f GFLOPS   d/p=%.2f  tb/p=%.2f\n",
                    static_cast<long long>(s.M), static_cast<long long>(s.N),
                    static_cast<long long>(s.K), flops / md, flops / mt, flops / mp,
                    mp / md, mp / mt);

        // The direct route's parallel unit is an N block, not a packing task.
        std::printf("      direct ladder:");
        const ComputeContext serial{};
        for (int nt : threads) {
            BenchPool pool(nt);
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
    // LLM-style A @ W^T (transpose_b): the weight is [N][K], one row per output
    // channel.
    bench_i8_case({"i8 tb 1024x512x1024", 1024, 512, 1024, /*ta=*/false, /*tb=*/true});
}

NNOPS_BENCH(matmul_u8s8_nn_large) {
    // Same shape as matmul_i8_nn_large with a u8 activation: what the
    // unsigned-A contract costs, from the same table.
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
    // Identical A bytes, the dtype the only variable: the ratio is the bias
    // bookkeeping and nothing else.
    bench_i8_u8_ab("nn", 1024, 512, 1024);
}

// ============================================================
// int8 with an integer destination (s32 / s8 / u8) + int32 bias
// ============================================================

NNOPS_BENCH(matmul_i8_out_256) {
    // Small enough that the output pass is a real share of the call.
    bench_i8_out_ab("nn", 256, 256, 256);
}

NNOPS_BENCH(matmul_i8_out_large) {
    // GEMM-bound: the same pass amortized over 4x the work.
    bench_i8_out_ab("nn", 1024, 512, 1024);
}

NNOPS_BENCH(matmul_i8_out_decode) {
    // One token: the pass is M rows, the GEMM is M·K·N, so s8 costs the most
    // relative to s32 here.
    bench_i8_out_ab("nn", 8, 4096, 4096);
}

// ============================================================
// Q4_1 weights — llama.cpp block stream on the RHS
// ============================================================

NNOPS_BENCH(matmul_q4_f32_nn_large) {
    bench_q4_case<float>({"q4/f32 nn 1024x512x1024", 1024, 512, 1024});
}

NNOPS_BENCH(matmul_q4_f32_tb_large) {
    // LLM-style: the weight is stored [N][K] and consumed as A @ W^T, which is
    // the layout a quantized checkpoint is loaded in.
    bench_q4_case<float>({"q4/f32 tb 1024x512x1024", 1024, 512, 1024,
                          /*ta=*/false, /*tb=*/true});
}

NNOPS_BENCH(matmul_q4_f16_nn) {
    bench_q4_case<half>({"q4/f16 nn 1024x512x1024", 1024, 512, 1024,
                         /*ta=*/false, /*tb=*/false, /*pad_a=*/0, /*pad_b=*/0,
                         /*f16=*/true});
}

NNOPS_BENCH(matmul_q4_f16_tb) {
    bench_q4_case<half>({"q4/f16 tb 1024x512x1024", 1024, 512, 1024,
                         /*ta=*/false, /*tb=*/true, /*pad_a=*/0, /*pad_b=*/0,
                         /*f16=*/true});
}

NNOPS_BENCH(matmul_q4_decode) {
    // One token, almost no A-side work: the weight stream's bandwidth limit.
    // The block bytes swept per call print as "B=... MB" next to the tiles.
    bench_q4_case<float>({"q4/f32 decode 1x4096x4096", 1, 4096, 4096,
                          /*ta=*/false, /*tb=*/true});
}

NNOPS_BENCH(matmul_q4_pack_only) {
    // Isolate the RHS pack from the MMA at the decode shape: the same n-block ×
    // k-block walk as matmul_block_fused, no A pack, no micro-kernel. One sweep
    // is one full-B pass — the pack half of a call, which dominates when M = 1
    // and the packed panel has no second row to be reused by.
    constexpr int64_t K = 4096, N = 4096;
    auto [av, a] = make_tensor<float>(1, K, K, 1);
    auto [fv, f] = make_tensor<float>(K, N, N, 2);   // dense f32 twin
    auto [bv, b] = make_tensor_q4(K, N, 2);          // 20-byte block stream

    MatMulAttributes attrs{};
    const cpu::MatMulPlan plan = cpu::get_matmul_plan(attrs, a.desc(), b.desc(), 1);
    const int kc = static_cast<int>(plan.kc);
    const int nc = static_cast<int>(plan.nc);
    constexpr int nr = 16;
    const int64_t row_blocks = b.row_stride_elems();
    std::vector<float> buf(static_cast<size_t>(plan.np_slice * plan.ldd_b));

    // Both physical layouts; the same square buffers serve as [K][N] (nn) and
    // [N][K] (tb) since K == N.
    enum Arm { kDense, kQ4, kArms };
    const uint8_t* streams[kArms] = {nullptr, bv.data()};
    auto sweep = [&](Arm arm, bool trans, float scale) {
        for (int n0 = 0; n0 < static_cast<int>(N); n0 += nc) {
            const int width = std::min(nc, static_cast<int>(N - n0));
            for (int k = 0; k < static_cast<int>(K); k += kc) {
                const int actual_kc = std::min(kc, static_cast<int>(K - k));
                const int ldd = static_cast<int>(
                    cpu::mma_pack_align(nr * actual_kc * sizeof(float)) / sizeof(float));
                if (arm == kQ4) {
                    cpu::tile_pack_rhs_q4_1(trans, n0, width, k, actual_kc,
                                            buf.data(), ldd, streams[arm], row_blocks,
                                            scale);
                } else {
                    const float* src = trans
                        ? fv.data() + static_cast<int64_t>(n0) * K + k
                        : fv.data() + static_cast<int64_t>(k) * N + n0;
                    cpu::tile_pack_rhs(trans, width, actual_kc, buf.data(), ldd,
                                       src, static_cast<int>(N), scale);
                }
            }
        }
    };

    // [arm][trans][scale]; scale 2 is the alpha path. Both arms scale as they
    // store (dense per element, Q4_1 folded into the block's d and m), so alpha
    // costs a multiply per block, not a second sweep over the panel.
    const int iters = 20;
    double ms[kArms][2][2] = {};
    for (int arm = 0; arm < kArms; ++arm) {
        for (int tr = 0; tr < 2; ++tr) {
            for (int sc = 0; sc < 2; ++sc) {
                const float scale = sc == 0 ? 1.0f : 2.0f;
                sweep(static_cast<Arm>(arm), tr == 1, scale);
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < iters; ++i) {
                    sweep(static_cast<Arm>(arm), tr == 1, scale);
                }
                ms[arm][tr][sc] = std::chrono::duration<double, std::milli>(
                                      std::chrono::high_resolution_clock::now() - t0).count()
                                  / iters;
            }
        }
    }

    const double values = static_cast<double>(K) * static_cast<double>(N);
    std::printf("  block pack-only  K=%lld N=%lld  kc=%d nc=%d  (one full-B sweep)\n",
                static_cast<long long>(K), static_cast<long long>(N), kc, nc);
    static constexpr const char* kLabels[kArms][2] = {
        {"dense f32 nn [K][N]", "dense f32 tb [N][K]"},
        {"q4_1      nn [K][N]", "q4_1      tb [N][K]"},
    };
    for (int arm = 0; arm < kArms; ++arm) {
        for (int tr = 0; tr < 2; ++tr) {
            std::printf("    %-22s : %7.3f ms  = %6.2f Gvalues/s   (%.2fx dense)\n",
                        kLabels[arm][tr], ms[arm][tr][0],
                        values / (ms[arm][tr][0] * 1e6),
                        ms[arm][tr][0] / ms[kDense][tr][0]);
            std::printf("    %-22s : %7.3f ms  = %6.2f Gvalues/s   (alpha=2: %.2fx plain)\n",
                        "  same, alpha=2", ms[arm][tr][1],
                        values / (ms[arm][tr][1] * 1e6),
                        ms[arm][tr][1] / ms[arm][tr][0]);
        }
    }
}

NNOPS_BENCH(matmul_q4_m_ladder) {
    // Where does the fused decode stop dominating? Small-M rows are weight
    // (decode) bound, large-M rows are FLOP bound; the ratio column crossing
    // back toward 1 is the knee. Two B sizes: 10 MB (out of cache) and 2.5 MB.
    const std::vector<int64_t> ms{1, 2, 4, 8, 16, 32, 64, 128, 256};
    m_ladder_q4(4096, 4096, ms, {1, 4}, 100.0);
    m_ladder_q4(2048, 2048, ms, {1, 4}, 100.0);
}

NNOPS_BENCH(matmul_qblock_fused_ab) {
#if defined(NNOPS_ARCH_X86_64)
    // Fused vs packed, alternated in one process. The fused arm forces the fused
    // route for every M, so the table shows where it stops winning; the shipped
    // gate is Q4Block::fused_max_m_trans/copy. Override < 0 keeps the built-in
    // limits, 0 forces packed, > 0 overrides the M limit.
    const int64_t K = 4096, N = 4096;
    const std::vector<int64_t> ms{1, 2, 4, 8, 16, 32, 64, 128};
    const std::vector<int> threads{1, 4};

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    for (int tb = 0; tb < 2; ++tb) {
        std::printf("  fused vs packed q4_1  K=%lld N=%lld  B=%s\n",
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
                SweepOpQ4<float> q(M, K, N, 7, nt, tb != 0);
                BenchPool pool(nt);
                ComputeContext ctx;
                if (nt > 1) { ctx.cpu = pool.cpu; }

                // The route measured second inherits the other's cache state
                // and reads up to 2x slower, so alternate it per round.
                std::vector<double> vf, vp;
                for (int r = 0; r < 4; ++r) {
                    const bool fused_first = (r % 2) == 0;
                    const int first = fused_first ? 128 : 0;
                    const int second = fused_first ? 0 : 128;
                    cpu::qblock_fused_max_m_override = first;
                    const double t1 = q.time_budget(ctx, 40.0, nullptr);
                    cpu::qblock_fused_max_m_override = second;
                    const double t2 = q.time_budget(ctx, 40.0, nullptr);
                    (fused_first ? vf : vp).push_back(t1);
                    (fused_first ? vp : vf).push_back(t2);
                }
                cpu::qblock_fused_max_m_override = -1;

                const double gf = median(vf), gp = median(vp);
                std::printf(" %9.1f %9.1f %8.3f |", gf, gp, gf / gp);
            }
            std::printf("\n");
        }
    }
#else
    std::printf("  fused block-quant route: x86_64 only\n");
#endif
}

NNOPS_BENCH(matmul_q4_vs_f32_gemv) {
    // Both arms pack-free: at M <= 4 a dense f32 B goes through matmul_gemv (B
    // streamed once, never packed) and a q4_1 B through the fused block decode
    // (x86_64; on aarch64 the block side falls back to the packed route). That
    // pair is the whole trade at batch 1: 6.4x fewer weight bytes against a
    // per-element decode. Above M = 4 the dense side leaves this route, which is
    // matmul_q4_m_ladder's comparison, not this one.
    const std::vector<int64_t> ms{1, 2, 4};
    const std::vector<int> threads{1, 2, 4, 8};
    m_ladder_q4(4096, 4096, ms, threads, 60.0);
    m_ladder_q4(2048, 2048, ms, threads, 60.0);
    m_ladder_q4(1024, 1024, ms, threads, 60.0);
    m_ladder_q4(4096, 4096, ms, threads, 60.0, /*tb=*/true);
    m_ladder_q4(2048, 2048, ms, threads, 60.0, /*tb=*/true);
    m_ladder_q4(1024, 1024, ms, threads, 60.0, /*tb=*/true);
}

NNOPS_BENCH(matmul_q4_f32_packed) {
    // Format-only: the block format pinned to the packed route, so the small-M
    // cells compare Q4_1 against dense f32 rather than two routes. The dense
    // arm has no switch — its GEMV is the route a dense B takes to M = 4 anyway.
    const std::vector<int64_t> ms{1, 2, 4, 8, 16};
    const std::vector<int> threads{1, 8};
    m_ladder_q4(4096, 4096, ms, threads, 40.0, false, /*packed=*/true);
    m_ladder_q4(4096, 4096, ms, threads, 40.0, true,  /*packed=*/true);
    m_ladder_q4(2048, 2048, ms, threads, 40.0, false, /*packed=*/true);
    m_ladder_q4(2048, 2048, ms, threads, 40.0, true,  /*packed=*/true);
}

NNOPS_BENCH(matmul_q4_m_ladder_tb) {
    // Same ladder against a [N][K] weight (the Linear layout): the fused dot
    // kernel's shape.
    const std::vector<int64_t> ms{1, 2, 4, 8, 16, 32, 64, 128, 256};
    m_ladder_q4(4096, 4096, ms, {1, 4}, 100.0, /*tb=*/true);
    m_ladder_q4(2048, 2048, ms, {1, 4}, 100.0, /*tb=*/true);
}

// Weight-format A/B under matched conditions: same A, dense f16/f32 B vs Q4_1 B.
NNOPS_BENCH(matmul_q4_vs_f16_ab) {
    bench_b_format_ab<half>("f16-act", 512, 512, 256);
}

NNOPS_BENCH(matmul_q4_vs_f32_ab) {
    bench_b_format_ab<float>("f32-act", 512, 512, 256);
}

NNOPS_BENCH(matmul_q4_vs_f16_ab_large) {
    bench_b_format_ab<half>("f16-act", 1024, 1024, 1024, /*rounds=*/4, /*iters=*/10);
}

// ============================================================
// Offline-quantized fp activations × s8 weights (W8A8, fp in / fp out)
// ============================================================
//
// The deployment shape and the s8/u8 A-path split are in the file header. A
// carries QuantParams (per-token for the point cases, per-tensor for the
// granularity split); B one scale per output channel. Each case keeps a dense fp
// copy of B, so dq/dense multiplies exactly the same weights.

/// Run @p f until @p budget_ms elapses (at least once); return ms per call.
/// A wall-clock budget leaves every row of a ladder on the machine the same
/// time, which is what makes the rows comparable on a host that drifts.
template <typename F>
double time_budget_ms(F&& f, double budget_ms) {
    f();
    f();
    const auto t0 = std::chrono::high_resolution_clock::now();
    int iters = 0;
    double ms = 0.0;
    do {
        f();
        ++iters;
        ms = std::chrono::duration<double, std::milli>(
                 std::chrono::high_resolution_clock::now() - t0).count();
    } while (ms < budget_ms && iters < 20000);
    return ms / static_cast<double>(iters);
}

template <typename F>
double time_budget_gflops(F&& f, double flops, double budget_ms) {
    return flops / (time_budget_ms(std::forward<F>(f), budget_ms) * 1e6);
}

/// Everything one dq case needs: fp A with offline QuantParams, s8 B with a
/// per-output-channel scale, and a dense fp copy of B.
///
/// @p b_zp is the weight zero point: 0 keeps B's per-channel array (all zeros,
/// the symmetric layout); non-zero declares one zp for the whole tensor and no
/// per-channel array — the shape whose per-element half folds into a per-row
/// offset.
template <typename T>
struct DqCase {
    int64_t M, K, N;
    bool tb;
    QuantGranularity gran;
    DataType act_dt;

    std::vector<T>       a_buf, bf_buf, c_buf, c_fp_buf;
    std::vector<float>   s_a, s_b;
    std::vector<int32_t> zp_a, zp_b;
    std::vector<int8_t>  b_buf, aq_buf;

    TensorView a, b, bf, c, c_fp, aq;
    TensorView ins[2], ins_fp[2];

    DqCase(int64_t m, int64_t k, int64_t n, bool transpose_b,
           QuantGranularity g, DataType qdt, int32_t b_zp = 0,
           bool b_zp_per_channel = false)
        : M(m), K(k), N(n), tb(transpose_b), gran(g), act_dt(qdt)
    {
        nnops::test::XorShift128 rng(1);
        constexpr DataType dt = std::is_same_v<T, float> ? DataType::f32 : DataType::f16;

        // A
        {
            std::vector<float> f(static_cast<size_t>(M * K));
            for (auto& v : f) { v = rng.next_float(-1.0f, 1.0f); }
            a_buf.resize(f.size());
            for (size_t i = 0; i < f.size(); ++i) {
                if constexpr (std::is_same_v<T, float>) { a_buf[i] = f[i]; }
                else { simd::s_store(&a_buf[i], f[i]); }
            }
        }

        // Offline A scale just covers [-1, 1], varied per row so nothing folds.
        const float span = (act_dt == DataType::u8) ? 2.0f / 255.0f : 1.0f / 127.0f;
        const int32_t zp0 = (act_dt == DataType::u8) ? 128 : 0;
        const bool per_token = (gran == QuantGranularity::PerToken);
        const size_t ns = static_cast<size_t>(per_token ? M : 1);
        s_a.resize(ns);
        zp_a.assign(ns, zp0);
        for (auto& s : s_a) { s = span * (0.8f + 0.4f * rng.next_float(0.0f, 1.0f)); }

        QuantParams qa;
        qa.granularity = gran;
        qa.scale = s_a[0];
        qa.zero_point = zp0;
        if (per_token) {
            qa.scale_data = s_a.data();
            qa.zero_point_data = zp_a.data();
            qa.num_scales = M;
        }

        // B: s8 weights. A second generator keeps B identical between a
        // per-token and a per-tensor case, which draw different numbers of
        // values for A's parameters.
        nnops::test::XorShift128 rng_b(2);
        const int64_t b_rows = tb ? N : K;
        const int64_t b_cols = tb ? K : N;
        b_buf.resize(static_cast<size_t>(b_rows * b_cols));
        for (auto& v : b_buf) { v = static_cast<int8_t>(rng_b.next_u64() % 255 - 127); }
        s_b.resize(static_cast<size_t>(N));
        zp_b.assign(static_cast<size_t>(N), 0);   // symmetric weight quant
        for (auto& s : s_b) { s = 0.008f * (0.8f + 0.4f * rng_b.next_float(0.0f, 1.0f)); }

        QuantParams qb;
        qb.granularity = QuantGranularity::PerChannel;
        qb.scale = s_b[0];
        qb.scale_data = s_b.data();
        qb.num_scales = N;
        if (b_zp == 0) {
            qb.zero_point_data = zp_b.data();
        } else if (b_zp_per_channel) {
            std::fill(zp_b.begin(), zp_b.end(), b_zp);
            qb.zero_point_data = zp_b.data();
        } else {
            // One zp for the whole tensor, no per-channel array: the per-element
            // half becomes a row constant (num_scales still describes the scales).
            qb.zero_point = b_zp;
        }

        // The fp route's B: exactly the values the s8 B represents.
        bf_buf.resize(b_buf.size());
        for (int64_t r = 0; r < b_rows; ++r) {
            for (int64_t cidx = 0; cidx < b_cols; ++cidx) {
                const size_t i = static_cast<size_t>(r * b_cols + cidx);
                const int64_t col = tb ? r : cidx;
                const float v = s_b[static_cast<size_t>(col)] * static_cast<float>(b_buf[i]);
                if constexpr (std::is_same_v<T, float>) { bf_buf[i] = v; }
                else { simd::s_store(&bf_buf[i], v); }
            }
        }

        // views (compact pitches)
        const int64_t as[] = {M, K};
        const int64_t bs[] = {b_rows, b_cols};
        const int64_t cs[] = {M, N};
        const int64_t ts = static_cast<int64_t>(sizeof(T));

        c_buf.resize(static_cast<size_t>(M * N));
        c_fp_buf.resize(static_cast<size_t>(M * N));
        aq_buf.resize(static_cast<size_t>(M * K));

        a  = TensorView(as, dt, a_buf.data(), K * ts, TensorLayout::NCHW, qa);
        b  = TensorView(bs, DataType::s8, b_buf.data(), b_cols, TensorLayout::NCHW, qb);
        bf = TensorView(bs, dt, bf_buf.data(), b_cols * ts);
        c    = TensorView(cs, dt, c_buf.data(), N * ts);
        c_fp = TensorView(cs, dt, c_fp_buf.data(), N * ts);
        // Desc-only view of the compact A_q the backend materializes on every
        // call; only its desc is read (the backend allocates the real one).
        aq = TensorView(as, act_dt, aq_buf.data(), K, TensorLayout::NCHW, qa);

        ins[0] = a;     ins[1] = b;
        ins_fp[0] = a;  ins_fp[1] = bf;
    }
};

/// Time one dq case at the given thread counts; mirrors run_geMM_i8.
template <typename T>
GeMMReport run_geMM_dq(DqCase<T>& dc, const char* name,
                       const std::vector<int>& threads, int iters) {
    MatMulAttributes attrs{};
    attrs.transpose_b = dc.tb;
    attrs.act_quant_dtype = dc.act_dt;
    auto op = MatMul::create(attrs, Backend::CPU);

    // The backend plans off the compact A_q desc, so the reported tiles are the
    // ones that actually run.
    const int meta_threads = *std::max_element(threads.begin(), threads.end());
    const cpu::MatMulPlan plan =
        cpu::get_matmul_plan(attrs, dc.aq.desc(), dc.b.desc(), meta_threads);

    GeMMReport rep;
    rep.name = name;
    rep.route = plan.split_n ? "dq-NKM" : "dq-MKN";
    rep.unit = "GFLOPS";
    rep.pack_a = plan.pack_a;
    rep.pack_b = true;
    rep.mc = static_cast<int>(plan.mc);
    rep.nc = static_cast<int>(plan.nc);
    rep.kc = static_cast<int>(plan.kc);
    rep.num_blocks = plan.split_n
        ? (dc.N + plan.nc - 1) / plan.nc
        : cpu::split_block_count(dc.M, plan.mc);
    rep.b_mb = static_cast<double>(dc.b.storage_bytes()) / (1024.0 * 1024.0);
    rep.threads = threads;

    for (int nt : threads) {
        BenchPool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

        for (int i = 0; i < 2; ++i) {
            op->compute(dc.c, dc.ins, use_ctx, nullptr);
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            op->compute(dc.c, dc.ins, use_ctx, nullptr);
        }
        auto end = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        rep.gflops.push_back(
            2.0 * static_cast<double>(dc.M) * static_cast<double>(dc.N)
                * static_cast<double>(dc.K) * static_cast<double>(iters) / (ms * 1e6));
    }
    return rep;
}

template <typename T>
void bench_dq_case(const char* name, int64_t M, int64_t K, int64_t N,
                   bool tb = false,
                   QuantGranularity gran = QuantGranularity::PerToken,
                   DataType act_dt = DataType::s8) {
    const int iters = 20;
    std::vector<int> threads{1, 2, 4, 8};
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 1 && std::find(threads.begin(), threads.end(), hw) == threads.end()) {
        threads.push_back(hw);
    }
    DqCase<T> dc(M, K, N, tb, gran, act_dt);
    print_report(run_geMM_dq(dc, name, threads, iters));
}

/// Table counterpart of `sweep_i8`. Every row includes the quantize pass and the
/// dequantizing epilogue, so the per-size gap against `matmul_i8_size_sweep` is
/// the fp activation boundary, not the GEMM.
template <typename T>
void sweep_dq(const char* tag, const std::vector<SweepShape>& shapes,
              const std::vector<int>& threads, double target_ms,
              bool tb = false,
              QuantGranularity gran = QuantGranularity::PerToken) {
    const char* dense = std::is_same_v<T, float> ? "f32" : "f16";
    std::printf("  %s-act dq sweep  B=%s  budget=%.0f ms/cell\n", dense,
                tb ? "[N][K]" : "[K][N]", target_ms);
    print_sweep_header(threads, "GFLOPS");

    const int meta_threads = *std::max_element(threads.begin(), threads.end());
    for (const SweepShape& s : shapes) {
        DqCase<T> dc(s.M, s.K, s.N, tb, gran, DataType::s8);

        MatMulAttributes attrs{};
        attrs.transpose_b = tb;
        attrs.act_quant_dtype = DataType::s8;
        auto op = MatMul::create(attrs, Backend::CPU);
        // Plan off the A_q desc, as matmul_cpu does — see run_geMM_dq.
        const cpu::MatMulPlan plan =
            cpu::get_matmul_plan(attrs, dc.aq.desc(), dc.b.desc(), meta_threads);

        const double flops = 2.0 * static_cast<double>(s.M) * static_cast<double>(s.N)
                           * static_cast<double>(s.K);
        std::vector<double> g;
        for (int nt : threads) {
            BenchPool pool(nt);
            ComputeContext ctx;
            if (nt > 1) { ctx.cpu = pool.cpu; }
            g.push_back(time_budget_gflops(
                [&] { op->compute(dc.c, dc.ins, ctx, nullptr); }, flops, target_ms));
        }
        print_sweep_row(s, plan.split_n ? "NKM" : "MKN",
                        static_cast<int>(plan.mc), static_cast<int>(plan.nc),
                        plan.split_n ? (s.N + plan.nc - 1) / plan.nc
                                     : cpu::split_block_count(s.M, plan.mc), g);
    }
}

/// Same A and same weight values, once as dense fp B and once as s8 B. Reports
/// dq/fp; >1 means dq is the faster way to get these numbers.
template <typename T>
void bench_dq_vs_fp_ab(int64_t M, int64_t K, int64_t N, bool tb = false,
                       QuantGranularity gran = QuantGranularity::PerToken,
                       int nt = 1, int rounds = 3, double budget_ms = 150.0) {
    DqCase<T> dc(M, K, N, tb, gran, DataType::s8);

    MatMulAttributes attrs_dq{};
    attrs_dq.transpose_b = tb;
    attrs_dq.act_quant_dtype = DataType::s8;
    MatMulAttributes attrs_fp{};
    attrs_fp.transpose_b = tb;

    auto op_dq = MatMul::create(attrs_dq, Backend::CPU);
    auto op_fp = MatMul::create(attrs_fp, Backend::CPU);

    BenchPool pool(nt);
    ComputeContext ctx;
    ctx.cpu = pool.cpu;
    const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

    const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N)
                       * static_cast<double>(K);
    auto time_dq = [&] {
        return time_budget_gflops(
            [&] { op_dq->compute(dc.c, dc.ins, use_ctx, nullptr); }, flops, budget_ms);
    };
    auto time_fp = [&] {
        return time_budget_gflops(
            [&] { op_fp->compute(dc.c_fp, dc.ins_fp, use_ctx, nullptr); }, flops, budget_ms);
    };

    std::printf("  dq vs dense  %s%s  %lldx%lldx%lld  %dT  rounds=%d  "
                "B: s8 %.2f MB, dense %.2f MB\n",
                std::is_same_v<T, float> ? "f32" : "f16", tb ? "-act  tb" : "-act  nn",
                static_cast<long long>(M), static_cast<long long>(K),
                static_cast<long long>(N), nt, rounds,
                static_cast<double>(K * N) / (1024.0 * 1024.0),
                static_cast<double>(K * N * static_cast<int64_t>(sizeof(T)))
                    / (1024.0 * 1024.0));

    std::vector<double> vd, vf;
    for (int r = 0; r < rounds; ++r) {
        // Alternate which route goes first: the second inherits the other's
        // cache state and reads slower.
        const bool dq_first = (r % 2) == 0;
        const double t1 = dq_first ? time_dq() : time_fp();
        const double t2 = dq_first ? time_fp() : time_dq();
        (dq_first ? vd : vf).push_back(t1);
        (dq_first ? vf : vd).push_back(t2);
        std::printf("    round %d:  dq=%7.1f  dense=%7.1f   dq/dense=%.3f\n",
                    r, vd.back(), vf.back(), vd.back() / vf.back());
    }

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    const double md = median(vd), mf = median(vf);
    std::printf("    median:   dq=%7.1f  dense=%7.1f   dq/dense=%.3f\n",
                md, mf, md / mf);
}

/// PerTensor vs PerToken over identical tensors: the only difference is whether
/// the epilogue broadcasts one (scale, zp) or loads a per-row vector.
template <typename T>
void bench_dq_granularity_ab(int64_t M, int64_t K, int64_t N,
                             int rounds = 3, double budget_ms = 150.0) {
    DqCase<T> dc_tok(M, K, N, /*tb=*/false, QuantGranularity::PerToken, DataType::s8);
    DqCase<T> dc_ten(M, K, N, /*tb=*/false, QuantGranularity::PerTensor, DataType::s8);

    MatMulAttributes attrs{};
    attrs.act_quant_dtype = DataType::s8;
    auto op_tok = MatMul::create(attrs, Backend::CPU);
    auto op_ten = MatMul::create(attrs, Backend::CPU);

    const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N)
                       * static_cast<double>(K);
    auto time_tok = [&] {
        return time_budget_gflops(
            [&] { op_tok->compute(dc_tok.c, dc_tok.ins, {}, nullptr); }, flops, budget_ms);
    };
    auto time_ten = [&] {
        return time_budget_gflops(
            [&] { op_ten->compute(dc_ten.c, dc_ten.ins, {}, nullptr); }, flops, budget_ms);
    };

    std::printf("  dq granularity  %lldx%lldx%lld  1T  rounds=%d\n",
                static_cast<long long>(M), static_cast<long long>(K),
                static_cast<long long>(N), rounds);

    std::vector<double> vt, vn;
    for (int r = 0; r < rounds; ++r) {
        const bool tok_first = (r % 2) == 0;
        const double t1 = tok_first ? time_tok() : time_ten();
        const double t2 = tok_first ? time_ten() : time_tok();
        (tok_first ? vt : vn).push_back(t1);
        (tok_first ? vn : vt).push_back(t2);
        std::printf("    round %d:  per_token=%7.1f  per_tensor=%7.1f   tok/ten=%.3f\n",
                    r, vt.back(), vn.back(), vt.back() / vn.back());
    }

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    const double mt = median(vt), mn = median(vn);
    std::printf("    median:   per_token=%7.1f  per_tensor=%7.1f   tok/ten=%.3f\n",
                mt, mn, mt / mn);
}

NNOPS_BENCH(matmul_dq_f32_nn_large) {
    bench_dq_case<float>("dq/f32 nn 1024x512x1024", 1024, 512, 1024);
}

NNOPS_BENCH(matmul_dq_f32_tb_large) {
    // A @ W^T from a checkpoint ([N][K]); the per-channel scale still indexes the
    // output column.
    bench_dq_case<float>("dq/f32 tb 1024x512x1024", 1024, 512, 1024, /*tb=*/true);
}

NNOPS_BENCH(matmul_dq_f32_nn_256) {
    // Smallest useful shape: the quantize (M×K) and dequantize (M×N) passes are a
    // much larger share.
    bench_dq_case<float>("dq/f32 nn 256^3", 256, 256, 256);
}

NNOPS_BENCH(matmul_dq_f32_nn_512) {
    bench_dq_case<float>("dq/f32 nn 512^3", 512, 512, 512);
}

NNOPS_BENCH(matmul_dq_f32_tb_256) {
    bench_dq_case<float>("dq/f32 tb 256^3", 256, 256, 256, /*tb=*/true);
}

NNOPS_BENCH(matmul_dq_f32_odd_n) {
    // N=1000 is not a multiple of nc or of the 16-wide B panel, so the last
    // n-block runs the masked pack and the epilogue's scalar tail.
    bench_dq_case<float>("dq/f32 nn 1024x512x1000", 1024, 512, 1000);
}

NNOPS_BENCH(matmul_dq_f32_nsplit) {
    // N ≫ M: the plan splits over N, so the quantized A is packed by every
    // n-block task and the per-token arrays are read by all of them.
    bench_dq_case<float>("dq/f32 nn 128x512x4096", 128, 512, 4096);
}

NNOPS_BENCH(matmul_dq_f32_deep_k) {
    // K ≫ M, N: several k-blocks per tile, small tiles, and the quantize pass
    // is K-wide per row.
    bench_dq_case<float>("dq/f32 nn 64x4096x64", 64, 4096, 64);
}

NNOPS_BENCH(matmul_dq_f16_nn) {
    bench_dq_case<half>("dq/f16 nn 1024x512x1024", 1024, 512, 1024);
}

NNOPS_BENCH(matmul_dq_f16_tb) {
    bench_dq_case<half>("dq/f16 tb 1024x512x1024", 1024, 512, 1024, /*tb=*/true);
}

NNOPS_BENCH(matmul_dq_u8_act_large) {
    // u8 activations (zp=128): the materialized-A_q path.
    bench_dq_case<float>("dq/u8act nn 1024x512x1024", 1024, 512, 1024,
                         /*tb=*/false, QuantGranularity::PerToken, DataType::u8);
}

NNOPS_BENCH(matmul_dq_u8_act_tb) {
    bench_dq_case<float>("dq/u8act tb 1024x512x1024", 1024, 512, 1024,
                         /*tb=*/true, QuantGranularity::PerToken, DataType::u8);
}

NNOPS_BENCH(matmul_dq_per_tensor_large) {
    // Per-tensor parameters: one (scale, zp) broadcasts into the epilogue
    // instead of a row vector. Same kernel, one load fewer per row.
    bench_dq_case<float>("dq/f32 nn 1024x512x1024 per-tensor", 1024, 512, 1024,
                         /*tb=*/false, QuantGranularity::PerTensor);
}

/// The weight zero point: 0 (a per-channel array of zeros, the symmetric
/// checkpoint layout) against one non-zero zp for the tensor, both with a u8
/// activation, so the column half of the compensation is live on both sides and
/// only the row half — the folded one — differs. The fold applies a row constant
/// once per output row instead of once per element; a non-zero-zp arm slower
/// than its twin means the per-element term or the B sweep came back.
template <typename T>
void bench_dq_weight_zp_ab(int64_t M, int64_t K, int64_t N, bool tb, int nt,
                           DataType act_dt = DataType::u8, int32_t b_zp = -5,
                           bool b_zp_per_channel = false,
                           int rounds = 3, double budget_ms = 150.0) {
    DqCase<T> sym(M, K, N, tb, QuantGranularity::PerToken, act_dt, /*b_zp=*/0);
    DqCase<T> fold(M, K, N, tb, QuantGranularity::PerToken, act_dt, b_zp,
                   b_zp_per_channel);

    MatMulAttributes attrs{};
    attrs.transpose_b = tb;
    attrs.act_quant_dtype = act_dt;
    auto op_sym = MatMul::create(attrs, Backend::CPU);
    auto op_fold = MatMul::create(attrs, Backend::CPU);

    BenchPool pool(nt);
    ComputeContext ctx;
    if (nt > 1) { ctx.cpu = pool.cpu; }
    const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

    const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N)
                       * static_cast<double>(K);
    auto time_sym = [&] {
        return time_budget_gflops(
            [&] { op_sym->compute(sym.c, sym.ins, use_ctx, nullptr); }, flops, budget_ms);
    };
    auto time_fold = [&] {
        return time_budget_gflops(
            [&] { op_fold->compute(fold.c, fold.ins, use_ctx, nullptr); }, flops, budget_ms);
    };

    std::printf("  dq weight zp  %s  %lldx%lldx%lld  %dT  act=%s  zp_b=%d%s  rounds=%d\n",
                tb ? "tb" : "nn", static_cast<long long>(M), static_cast<long long>(K),
                static_cast<long long>(N), nt,
                act_dt == DataType::u8 ? "u8" : "s8", b_zp,
                b_zp_per_channel ? " per-channel" : "", rounds);

    std::vector<double> vs, vf, vnull;
    for (int r = 0; r < rounds; ++r) {
        // A B A B, and the per-round minimum for each arm: the second A is the
        // null control (same op, same buffers) and sets the noise floor.
        const double s1 = time_sym();
        const double f1 = time_fold();
        const double s2 = time_sym();
        const double f2 = time_fold();
        vs.push_back(std::min(s1, s2));
        vf.push_back(std::min(f1, f2));
        vnull.push_back(std::abs(s2 - s1) / std::min(s1, s2) * 100.0);
        std::printf("    round %d:  zp0=%7.1f  zp5=%7.1f   zp5/zp0=%.3f   null %.1f%%\n",
                    r, vs.back(), vf.back(), vf.back() / vs.back(), vnull.back());
    }

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    const double ms = median(vs), mf = median(vf), mn = median(vnull);
    std::printf("    median:   zp0=%7.1f  zp5=%7.1f   zp5/zp0=%.3f   null %.1f%%%s\n",
                ms, mf, mf / ms, mn, std::abs(mf - ms) / ms * 100.0 > mn
                                       ? "" : "  (within noise)");
}

NNOPS_BENCH(matmul_dq_weight_zp_ab) {
    // 256^3 on one thread: the output pass is a real share of the call.
    bench_dq_weight_zp_ab<float>(256, 256, 256, /*tb=*/false, 1);
    // Full-size GEMM: the fold's per-row work is amortized, so the ratio should
    // sit at 1.
    bench_dq_weight_zp_ab<float>(1024, 512, 1024, /*tb=*/true, 1);
    bench_dq_weight_zp_ab<float>(1024, 512, 1024, /*tb=*/true, 4);
    // Decode: the A sweep the fold keeps is O(K), the per-element term it
    // retires is O(N).
    bench_dq_weight_zp_ab<float>(1, 4096, 4096, /*tb=*/true, 1);
    // Per-channel asymmetric weights under symmetric activations: no fold (it
    // needs one zp_b), but the column half is zero, so B is never swept.
    bench_dq_weight_zp_ab<float>(1024, 512, 1024, /*tb=*/true, 1, DataType::s8, -5,
                                 /*b_zp_per_channel=*/true);
    bench_dq_weight_zp_ab<float>(1024, 512, 1024, /*tb=*/true, 4, DataType::s8, -5,
                                 /*b_zp_per_channel=*/true);
}

NNOPS_BENCH(matmul_dq_vs_fp_ab_nn) {
    // Same A, same weight values: s8 B vs a dense B. Read the ratio.
    for (int nt : {1, 4}) {
        bench_dq_vs_fp_ab<float>(1024, 512, 1024, /*tb=*/false,
                                 QuantGranularity::PerToken, nt);
    }
}

NNOPS_BENCH(matmul_dq_vs_fp_ab_tb) {
    bench_dq_vs_fp_ab<float>(1024, 512, 1024, /*tb=*/true,
                             QuantGranularity::PerToken, 4);
}

NNOPS_BENCH(matmul_dq_vs_fp_ab_f16) {
    bench_dq_vs_fp_ab<half>(1024, 512, 1024, /*tb=*/false,
                            QuantGranularity::PerToken, 4);
}

NNOPS_BENCH(matmul_dq_vs_fp_sweep) {
    // The same comparison as a table over the square ladder: dq/dense moves with
    // the B footprint (B is 4x smaller as s8), so the win grows until the GEMM
    // stops being bandwidth bound and shrinks once both fit in cache.
    const std::vector<int64_t> sizes{128, 256, 384, 512, 768, 1024, 1536};
    std::printf("  dq vs dense f32-act  nn  budget=100 ms/cell   (GFLOPS)\n");
    std::printf("   %5s | %9s %9s %7s | %9s %9s %7s\n", "size",
                "dq 1T", "f32 1T", "dq/f32", "dq 4T", "f32 4T", "dq/f32");
    // Below ~768 the 4T ratio is pool start-up (dq issues one more
    // ctx.cpu.run); the 1T ratio is the honest curve.
    for (int64_t S : sizes) {
        DqCase<float> dc(S, S, S, /*tb=*/false, QuantGranularity::PerToken, DataType::s8);
        MatMulAttributes a_dq{};
        a_dq.act_quant_dtype = DataType::s8;
        auto op_dq = MatMul::create(a_dq, Backend::CPU);
        auto op_fp = MatMul::create(MatMulAttributes{}, Backend::CPU);
        const double flops = 2.0 * static_cast<double>(S) * static_cast<double>(S)
                           * static_cast<double>(S);

        std::printf("   %5lld |", static_cast<long long>(S));
        for (int nt : {1, 4}) {
            BenchPool pool(nt);
            ComputeContext ctx;
            if (nt > 1) { ctx.cpu = pool.cpu; }
            const double g_dq = time_budget_gflops(
                [&] { op_dq->compute(dc.c, dc.ins, ctx, nullptr); }, flops, 100.0);
            const double g_fp = time_budget_gflops(
                [&] { op_fp->compute(dc.c_fp, dc.ins_fp, ctx, nullptr); }, flops, 100.0);
            std::printf(" %9.1f %9.1f %7.2f |", g_dq, g_fp, g_dq / g_fp);
        }
        std::printf("\n");
    }
}

NNOPS_BENCH(matmul_dq_size_sweep) {
    // matmul_i8_size_sweep's ladder, so the two tables read side by side; the gap
    // at each size is the quantize pass plus the int32 accumulator round trip.
    sweep_dq<float>("f32", {
        {128, 128, 128},    {192, 192, 192},    {256, 256, 256},   {384, 384, 384},
        {512, 512, 512},    {768, 768, 768},    {1024, 1024, 1024},
        {1536, 1536, 1536}, {2048, 2048, 2048},
    }, sweep_threads(), 120.0);
}

NNOPS_BENCH(matmul_dq_tb_size_sweep) {
    // Same ladder with a [N][K] weight — the Linear/checkpoint layout.
    sweep_dq<float>("f32", {
        {128, 128, 128},    {256, 256, 256},    {512, 512, 512},
        {1024, 1024, 1024}, {1536, 1536, 1536}, {2048, 2048, 2048},
    }, sweep_threads(), 120.0, /*tb=*/true);
}

NNOPS_BENCH(matmul_dq_f16_size_sweep) {
    // f16 activations: the quantize pass reads half the bytes, the output pass
    // writes half. The int8 GEMM is unchanged.
    sweep_dq<half>("f16", {
        {128, 128, 128},    {256, 256, 256},    {512, 512, 512},
        {1024, 1024, 1024}, {1536, 1536, 1536},
    }, sweep_threads(), 120.0);
}

NNOPS_BENCH(matmul_dq_rect_sweep) {
    // Rectangles, not cubes: corners where one dimension is 1 or far from the
    // tile size, so the plan changes shape.
    sweep_dq<float>("f32", {
        {1, 512, 4096},   {1, 4096, 4096},   {16, 4096, 4096},  {64, 4096, 4096},
        {1024, 256, 64},  {1024, 512, 1000}, {128, 512, 4096},  {4096, 256, 256},
    }, sweep_threads(), 120.0);
}

NNOPS_BENCH(matmul_dq_granularity_ab) {
    bench_dq_granularity_ab<float>(1024, 512, 1024);
    bench_dq_granularity_ab<float>(256, 256, 256);
}

NNOPS_BENCH(matmul_dq_quantize_only) {
    // The quantize pass alone against the full dq call on the same shape: how
    // much of a call the fp→int8 conversion is. If this share ever matters, the
    // pass folds into A's pack (see skills/how_to_optimize_gemm.md).
    constexpr int64_t M = 1024, K = 512, N = 1024;
    std::vector<float>  src(static_cast<size_t>(M * K));
    std::vector<int8_t> dst(src.size());
    std::vector<float>  scale(static_cast<size_t>(M));
    nnops::test::XorShift128 rng(1);
    for (auto& v : src) { v = rng.next_float(-1.0f, 1.0f); }
    for (auto& s : scale) { s = (0.8f + 0.4f * rng.next_float(0.0f, 1.0f)) / 127.0f; }

    auto pass = [&] {
        kernel::quant_kernel::quantization<int8_t>(
            static_cast<int>(M), static_cast<int>(K), dst.data(), static_cast<int>(K),
            src.data(), static_cast<int>(K), scale.data(), nullptr);
    };
    pass();   // fault the pages before the timed runs
    const double pass_ms = time_budget_ms(pass, 200.0);
    const double vals = static_cast<double>(M) * static_cast<double>(K);
    const double bytes = vals * (sizeof(float) + 1);   // fp read + byte write

    DqCase<float> dc(M, K, N, /*tb=*/false, QuantGranularity::PerToken, DataType::s8);
    MatMulAttributes attrs{};
    attrs.act_quant_dtype = DataType::s8;
    auto op = MatMul::create(attrs, Backend::CPU);
    const double op_ms = time_budget_ms([&] { op->compute(dc.c, dc.ins, {}, nullptr); }, 200.0);

    std::printf("  dq quantize pass  %lldx%lld  1T : %.3f ms  = %.2f Gvalues/s, %.2f GB/s\n",
                static_cast<long long>(M), static_cast<long long>(K), pass_ms,
                vals / (pass_ms * 1e6), bytes / (pass_ms * 1e6));
    std::printf("  full dq call      %lldx%lldx%lld 1T : %.3f ms  (pass = %.1f%% of it)\n",
                static_cast<long long>(M), static_cast<long long>(K),
                static_cast<long long>(N), op_ms, 100.0 * pass_ms / op_ms);
}

NNOPS_BENCH(matmul_dq_m_ladder) {
    // Decode-to-prefill sweep. At small M both routes stream B and the dq side
    // wins only on B's width — it has no GEMV kernel, so it runs the tiled GEMM
    // for one row — hence a ratio near 1.2 until M is large enough for the int8
    // FLOP rate to matter (K=N=2048 keeps B at 4 MB: out of L2, in L3). On the
    // dev host the ratios climb 1.2, 1.2, 2.6, 2.9, 3.4, 3.6.
    const std::vector<int64_t> ms{1, 4, 16, 64, 256, 1024};
    for (int64_t M : ms) {
        bench_dq_vs_fp_ab<float>(M, 2048, 2048, /*tb=*/false,
                                 QuantGranularity::PerToken, /*nt=*/1,
                                 /*rounds=*/3, /*budget_ms=*/100.0);
    }
}

// ============================================================
// Size and shape sweeps — where the thread pool starts to pay
// ============================================================
//
// Same binary, read as a table: the point cases answer "which route", these
// answer "at what size". Everything is capped at 2048^3.

NNOPS_BENCH(matmul_f32_size_sweep) {
    // Square, doubling-ish from 64 to the 2048 cap. The left end is where the
    // pool has nothing to hand out (256^3 = 2 M-blocks), the right end past L2/L3.
    sweep_fp<float>("f32", {
        {64, 64, 64},       {96, 96, 96},       {128, 128, 128},   {192, 192, 192},
        {256, 256, 256},    {384, 384, 384},    {512, 512, 512},   {768, 768, 768},
        {1024, 1024, 1024}, {1536, 1536, 1536}, {2048, 2048, 2048},
    }, sweep_threads(), 120.0);
}

NNOPS_BENCH(matmul_f32_shape_sweep) {
    // Asymmetric shapes, each isolating one axis as the one that must carry the
    // parallelism; at M=1 the block count is whatever N yields.
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
    // The int8 point case 256^3 collapses at 20T (the kc budget shrinks the tiles
    // as the pool grows); this walks the same curve where the tiles have room.
    sweep_i8({
        {128, 128, 128},    {192, 192, 192},    {256, 256, 256},   {384, 384, 384},
        {512, 512, 512},    {768, 768, 768},    {1024, 1024, 1024},
        {1536, 1536, 1536}, {2048, 2048, 2048},
    }, sweep_threads(), 120.0);
}

NNOPS_BENCH(matmul_q4_size_sweep) {
    // Same square ladder as f32 so the two tables can be read side by side:
    // the delta at each size is what the block-stream pack costs (or saves).
    sweep_q4<float>("f32", {
        {64, 64, 64},       {96, 96, 96},       {128, 128, 128},   {192, 192, 192},
        {256, 256, 256},    {384, 384, 384},    {512, 512, 512},   {768, 768, 768},
        {1024, 1024, 1024}, {1536, 1536, 1536}, {2048, 2048, 2048},
    }, sweep_threads(), 120.0);
}

NNOPS_BENCH(matmul_q4_decode_sweep) {
    // Decode-shaped rows: M stays small while K and N grow, so B is the whole
    // working set and the weight-BW line is the real limit.
    sweep_q4<float>("f32 decode", {
        {1, 1024, 1024},  {1, 2048, 2048},  {1, 4096, 4096},
        {16, 4096, 4096}, {64, 4096, 4096}, {256, 4096, 4096},
    }, sweep_threads(), 120.0);
}

NNOPS_BENCH(matmul_q4_decode_rate) {
#if defined(NNOPS_ARCH_X86_64)
    // Raw block-decode throughput: no A, no panel, no output traffic. This is
    // the floor the fused route can reach — the packed route additionally
    // writes and re-reads every decoded value.
    const int64_t rows = 4096, blocks_per_row = 128;   // 4096 x 4096 values
    auto [bv, b] = make_tensor_q4(rows, blocks_per_row * 32, 3);
    const uint8_t* p = bv.data();
    const int64_t nblk = rows * blocks_per_row;
    const int64_t values = nblk * 32;

    double best = 1e30;
    for (int r = 0; r < 5; ++r) {
        __m256 acc = _mm256_setzero_ps();
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int64_t i = 0; i < nblk; ++i) {
            __m256 v0, v1, v2, v3;
            cpu::arch::decode_q4_1_block_ps(p + i * 20, v0, v1, v2, v3);
            acc = _mm256_add_ps(acc, _mm256_add_ps(_mm256_add_ps(v0, v1),
                                                   _mm256_add_ps(v2, v3)));
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        if (_mm256_cvtss_f32(acc) == 12345.0f) { std::printf(" "); }
        best = std::min(best, ms);
    }
    std::printf("  q4_1 decode-only  4096x4096 blocks: %.3f ms  = %.2f Gvalues/s\n",
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
            cpu::arch::decode_q4_1_block<float>(p + i * 20, vals);
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
    std::printf("  q4_1 decode+stack 4096x4096 blocks: %.3f ms  = %.2f Gvalues/s\n",
                best, static_cast<double>(values) / (best * 1e6));
#else
    std::printf("  q4_1 decode rate: x86_64 only\n");
#endif
}

// ---- bias A/B --------------------------------------------------------------
//
// A biased call costs the kernels nothing: matmul_cpu strips the activation (and
// the Relu/Relu6 fold) from the attrs it dispatches with, runs the bare product,
// then sweeps C once with epilogue_inplace(bias, activation). The price is one
// extra read-modify-write pass over M×N plus a third input span — invisible next
// to a large GEMM, worth measuring only at decode shapes. The four variants
// alternate round by round in one process, so bias/plain is not a thermal
// artifact.

template <typename T>
struct BiasAB {
    using OpPtr = decltype(MatMul::create(std::declval<MatMulAttributes>(), Backend::CPU));

    struct Variant {
        const char* label;
        OpPtr op;
        std::vector<TensorView> ins;
    };

    std::vector<T> a_buf, b_buf, z_buf, out_buf;
    std::vector<Variant> variants;   // 0 plain, 1 relu, 2 bias, 3 bias+relu
    TensorView out;
    int64_t M, N, K;

    BiasAB(int64_t m, int64_t k, int64_t n, bool transpose_b, uint64_t seed)
        : M(m), N(n), K(k) {
        auto [av, a] = make_tensor<T>(m, k, k, seed);
        a_buf = std::move(av);

        const int64_t b_rows = transpose_b ? n : k;
        const int64_t b_cols = transpose_b ? k : n;
        auto [bv, b] = make_tensor<T>(b_rows, b_cols, b_cols, seed + 1);
        b_buf = std::move(bv);

        // A rank-1 [N] bias; make_tensor builds the [1,N] pitch view we fill.
        auto [zv, _] = make_tensor<T>(1, n, n, seed + 2);
        z_buf = std::move(zv);
        constexpr DataType dt = std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
        const int64_t zshape[] = {n};
        TensorView bias(zshape, dt, z_buf.data());

        MatMulAttributes plain_attrs{};
        plain_attrs.transpose_b = transpose_b;
        MatMulAttributes relu_attrs = plain_attrs;
        relu_attrs.epilogue.type = EpilogueActivateType::Relu;

        auto add = [&](const char* label, const MatMulAttributes& attrs, bool with_bias) {
            Variant v;
            v.label = label;
            v.op = MatMul::create(attrs, Backend::CPU);
            v.ins.push_back(a);
            v.ins.push_back(b);
            if (with_bias) { v.ins.push_back(bias); }
            variants.push_back(std::move(v));
        };
        add("plain", plain_attrs, false);
        add("relu", relu_attrs, false);
        add("bias", plain_attrs, true);
        add("bias+relu", relu_attrs, true);

        const TensorDesc arr[] = {a.desc(), b.desc()};
        auto descs = variants[0].op->getOutputTensorDesc(arr);
        out_buf.assign(static_cast<size_t>(descs[0].numel()), T{});
        out = test::make_planar(descs[0], out_buf.data());
    }

    double time(Variant& v, int iters) {
        for (int i = 0; i < 2; ++i) { v.op->compute(out, v.ins, {}, nullptr); }
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) { v.op->compute(out, v.ins, {}, nullptr); }
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return 2.0 * static_cast<double>(M) * static_cast<double>(N)
             * static_cast<double>(K) * static_cast<double>(iters) / (ms * 1e6);
    }
};

template <typename T>
void bench_bias_ab(const char* tag, int64_t M, int64_t K, int64_t N,
                   bool transpose_b = false, int rounds = 6, int iters = 8) {
    BiasAB<T> p(M, K, N, transpose_b, 11);
    std::printf("  bias A/B %s%s  %lldx%lldx%lld  1T  rounds=%d iters=%d\n",
                tag, transpose_b ? " tb" : "",
                static_cast<long long>(M), static_cast<long long>(K),
                static_cast<long long>(N), rounds, iters);

    const size_t nv = p.variants.size();
    std::vector<std::vector<double>> vals(nv);
    for (int r = 0; r < rounds; ++r) {
        std::printf("    round %d:", r);
        for (size_t v = 0; v < nv; ++v) {
            const double g = p.time(p.variants[v], iters);
            vals[v].push_back(g);
            std::printf("  %s=%7.1f", p.variants[v].label, g);
        }
        const double base = vals[0].back();
        std::printf("   bias/plain=%.3f  bias+relu/plain=%.3f\n",
                    vals[2].back() / base, vals[3].back() / base);
    }

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    const double base = median(vals[0]);
    std::printf("    median GFLOPS (vs plain):");
    for (size_t v = 0; v < nv; ++v) {
        const double m = median(vals[v]);
        std::printf("  %s=%.1f (%.3f)", p.variants[v].label, m, m / base);
    }
    std::printf("\n");
}

// ---- alpha A/B -------------------------------------------------------------
//
// On the tiled route alpha folds into the RHS pack, so the dot product comes out
// scaled and the MMA never sees it. The small-M kernels have no pack to ride and
// scale the finished accumulator instead: one multiply per output element, which
// is what M=1 measures.

template <typename T>
struct AlphaAB {
    using OpPtr = decltype(MatMul::create(std::declval<MatMulAttributes>(), Backend::CPU));

    struct Variant {
        const char* label;
        OpPtr op;
    };

    std::vector<T> a_buf, b_buf, out_buf;
    std::vector<Variant> variants;   // 0 plain, 1 alpha=2, 2 alpha=0.5
    std::vector<TensorView> ins;
    TensorView out;
    int64_t M, N, K;

    AlphaAB(int64_t m, int64_t k, int64_t n, bool transpose_b, uint64_t seed)
        : M(m), N(n), K(k) {
        auto [av, a] = make_tensor<T>(m, k, k, seed);
        a_buf = std::move(av);

        const int64_t b_rows = transpose_b ? n : k;
        const int64_t b_cols = transpose_b ? k : n;
        auto [bv, b] = make_tensor<T>(b_rows, b_cols, b_cols, seed + 1);
        b_buf = std::move(bv);
        ins.push_back(a);
        ins.push_back(b);

        auto add = [&](const char* label, float alpha) {
            MatMulAttributes attrs{};
            attrs.transpose_b = transpose_b;
            attrs.alpha = alpha;
            variants.push_back({label, MatMul::create(attrs, Backend::CPU)});
        };
        add("plain", 1.0f);
        add("a=2", 2.0f);
        add("a=0.5", 0.5f);

        const TensorDesc arr[] = {ins[0].desc(), ins[1].desc()};
        auto descs = variants[0].op->getOutputTensorDesc(arr);
        out_buf.assign(static_cast<size_t>(descs[0].numel()), T{});
        out = test::make_planar(descs[0], out_buf.data());
    }

    double time(Variant& v, int iters) {
        for (int i = 0; i < 2; ++i) { v.op->compute(out, ins, {}, nullptr); }
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) { v.op->compute(out, ins, {}, nullptr); }
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return 2.0 * static_cast<double>(M) * static_cast<double>(N)
             * static_cast<double>(K) * static_cast<double>(iters) / (ms * 1e6);
    }
};

template <typename T>
void bench_alpha_ab(const char* tag, int64_t M, int64_t K, int64_t N,
                    bool transpose_b = false, int rounds = 6, int iters = 8) {
    AlphaAB<T> p(M, K, N, transpose_b, 13);
    std::printf("  alpha A/B %s%s  %lldx%lldx%lld  1T  rounds=%d iters=%d\n",
                tag, transpose_b ? " tb" : "",
                static_cast<long long>(M), static_cast<long long>(K),
                static_cast<long long>(N), rounds, iters);

    const size_t nv = p.variants.size();
    std::vector<std::vector<double>> vals(nv);
    for (int r = 0; r < rounds; ++r) {
        std::printf("    round %d:", r);
        for (size_t v = 0; v < nv; ++v) {
            const double g = p.time(p.variants[v], iters);
            vals[v].push_back(g);
            std::printf("  %s=%7.1f", p.variants[v].label, g);
        }
        const double base = vals[0].back();
        std::printf("   a2/plain=%.3f  a05/plain=%.3f\n",
                    vals[1].back() / base, vals[2].back() / base);
    }

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    const double base = median(vals[0]);
    std::printf("    median GFLOPS (vs plain):");
    for (size_t v = 0; v < nv; ++v) {
        const double m = median(vals[v]);
        std::printf("  %s=%.1f (%.3f)", p.variants[v].label, m, m / base);
    }
    std::printf("\n");
}

NNOPS_BENCH(matmul_alpha_ab) {
    // Decode (small-M kernels), packed prefill, square, and Linear's [N][K]
    // weight layout — the four ways alpha reaches the product.
    bench_alpha_ab<float>("f32", 1, 4096, 4096, false, 6, 30);
    bench_alpha_ab<float>("f32", 128, 4096, 4096, false, 6, 4);
    bench_alpha_ab<float>("f32", 1024, 1024, 1024, false, 6, 8);
    bench_alpha_ab<float>("f32", 128, 4096, 4096, true, 6, 4);
    bench_alpha_ab<half>("f16", 1, 4096, 4096, false, 6, 30);
    bench_alpha_ab<half>("f16", 128, 4096, 4096, false, 6, 4);
}

NNOPS_BENCH(matmul_bias_ab) {
    // Decode / prefill / square, then Linear's transposed layout. `iters` keeps
    // each timing at 50-250 ms; the output-pass delta drowns in a 2 ms sample.
    bench_bias_ab<float>("f32", 1, 4096, 4096, false, 6, 30);
    bench_bias_ab<float>("f32", 128, 4096, 4096, false, 6, 4);
    bench_bias_ab<float>("f32", 1024, 1024, 1024, false, 6, 8);
    bench_bias_ab<float>("f32", 128, 4096, 4096, true, 6, 4);
    bench_bias_ab<half>("f16", 1, 4096, 4096, false, 6, 30);
    bench_bias_ab<half>("f16", 128, 4096, 4096, false, 6, 4);
}
