/// @file bench_matmul.cpp
/// @brief MatMul micro-benchmarks — pack-decision routing + thread scaling.
///
/// Each case exercises one cell of the pack-decision routing table (direct /
/// NKM-fused / MKN-fused) across {nn, ta, tb, tatb} and padded strides. For each
/// case the harness reports:
///   - the static PackPlan routing decision (pack_a / pack_b / loop order),
///   - the resolved tile sizes (mc, nc) and split block count,
///   - the workspace size,
///   - GFLOPS at 1, 2, 4, 8 and hardware threads.
///
/// The point is to confirm the static plan picks a sane path and to see where
/// block-granularity threading scales (and where it can't — e.g. when Nc is so
/// large that an N-split yields only one block).

#include "nnops/ops/matmul.hpp"
#include "backend/cpu/matmul_helper.h"   // make_pack_plan, resolve_tile_sizes, tile constants
#include "common/bench_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <type_traits>
#include <vector>

using namespace nnops;
using half = nnops::backend::cpu::half;
namespace cpu = nnops::backend::cpu;

namespace {

// Minimal fixed-size thread pool injecting cpu_parallel_for (same as the test).
struct SimplePool {
    explicit SimplePool(int nthreads) : nthreads_(nthreads) {}

    void parallel_for(int64_t begin, int64_t end, const ParallelForBody& body) {
        std::atomic<int64_t> next{begin};
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(nthreads_));
        for (int t = 0; t < nthreads_; ++t) {
            workers.emplace_back([&]() {
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
    bool pack_a, pack_b, mkn_order;
    int mc, nc;
    int64_t num_blocks;
    size_t workspace_bytes;
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

    // --- static plan metadata (mirrors matmul_dispatch_2d) ---
    const int64_t lda = a.row_stride_elems();
    const int64_t ldb = b.row_stride_elems();
    const cpu::PackPlan plan = cpu::make_pack_plan<T>(attrs, c.M, c.N, lda, ldb);
    constexpr int mr_max = cpu::mr_max_flt<T>();
    constexpr int nr_max = cpu::nr_max_flt<T>();
    constexpr int kc = std::is_same_v<T, float> ? cpu::KC_F32 : cpu::KC_F16;
    int mc, nc;
    cpu::resolve_tile_sizes(static_cast<int>(c.M), static_cast<int>(c.N),
                            mr_max, nr_max, kc, static_cast<int>(sizeof(T)), mc, nc);
    const int64_t num_blocks = plan.mkn_order
        ? (c.M + mc - 1) / mc
        : (c.N + nc - 1) / nc;

    std::vector<char> workspace(op->getWorkspaceSize(arr, descs));
    std::vector<T> out_buf(static_cast<size_t>(descs[0].numel()));
    auto output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {a, b};

    GeMMReport rep;
    rep.name = c.name;
    rep.pack_a = plan.pack_a;
    rep.pack_b = plan.pack_b;
    rep.mkn_order = plan.mkn_order;
    rep.mc = mc;
    rep.nc = nc;
    rep.num_blocks = num_blocks;
    rep.workspace_bytes = workspace.size();
    rep.threads = threads;

    for (int nt : threads) {
        SimplePool pool(nt);
        ComputeContext ctx;
        ctx.cpu_parallel_for = [&pool](int64_t b0, int64_t e, const ParallelForBody& body) {
            pool.parallel_for(b0, e, body);
        };
        const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

        // Warm-up (also faults/allocates pages).
        for (int i = 0; i < 2; ++i) {
            op->compute(output, ins, use_ctx, workspace.data());
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            op->compute(output, ins, use_ctx, workspace.data());
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
    const char* route = r.mkn_order ? "MKN (A-pack)"
                       : (r.pack_a || r.pack_b) ? "NKM-fused" : "direct";
    std::printf("  %s\n", r.name);
    std::printf("    route=%-12s pack_a=%d pack_b=%d  mc=%d nc=%d  blocks=%lld  ws=%zu B\n",
                route, (int)r.pack_a, (int)r.pack_b, r.mc, r.nc,
                static_cast<long long>(r.num_blocks), r.workspace_bytes);

    const double g1 = r.gflops.empty() ? 0.0 : r.gflops[0];
    std::printf("    ");
    for (size_t i = 0; i < r.threads.size(); ++i) {
        const double speedup = g1 > 0.0 ? r.gflops[i] / g1 : 0.0;
        std::printf(" %2dT %7.1f GFLOPS (%.2fx)", r.threads[i], r.gflops[i], speedup);
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
