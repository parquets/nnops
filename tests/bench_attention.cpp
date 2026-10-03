/// @file bench_attention.cpp
/// @brief Attention micro-benchmark — plan (standard vs flash) + thread scaling.
///
/// Mirrors bench_matmul.cpp: every case runs the plan-driven fused fast path and
/// reports the resolved plan plus GFLOPS at 1, 2, 4, 8 and hardware threads.
///
/// QK^T (transpose-B GEMM) then softmax then attn @ V (direct GEMM), per
/// (batch, head) — or the online-softmax FlashAttention path for large Sq*Sk.
/// The report shows which route fired and the resolved tile sizes.

#include "nnops/ops/attention.hpp"
#include "backend/cpu/attention.h"        // get_attention_plan, AttentionPlan
#include "common/bench_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

using namespace nnops;
namespace cpu = nnops::backend::cpu;

namespace {

// Minimal fixed-size thread pool exposing a CpuBackend (same as bench_matmul).
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

struct AttentionCase {
    const char* name;
    int64_t B, S, H, D;     // merged [B, S, H*D], self-attention (Sq = S)
    int64_t Sk = 0;         // key seq len; 0 -> S
    bool mask = false;
};

void run_attention(const AttentionCase& c, const std::vector<int>& threads, int iters) {
    const int64_t Sq = c.S;
    const int64_t Sk = (c.Sk == 0) ? c.S : c.Sk;
    const int64_t H  = c.H;
    const int64_t D  = c.D;

    const int64_t qshape[] = {c.B, Sq, H * D};
    const int64_t kshape[] = {c.B, Sk, H * D};
    const int64_t vshape[] = {c.B, Sk, H * D};

    auto [q_vec, q] = test::make_random_tensor(qshape, -1.0f, 1.0f, 11);
    auto [k_vec, k] = test::make_random_tensor(kshape, -1.0f, 1.0f, 12);
    auto [v_vec, v] = test::make_random_tensor(vshape, -1.0f, 1.0f, 13);

    std::vector<float> m_vec;
    TensorView mask_view;
    if (c.mask) {
        const int64_t mshape[] = {Sq, Sk};
        auto [mv, m] = test::make_random_tensor(mshape, -2.0f, 2.0f, 14);
        m_vec = std::move(mv);
        mask_view = m;
    }

    AttentionAttributes attrs{};
    attrs.num_heads = H;
    attrs.scale = 0.0f;   // auto 1/sqrt(D)

    auto op = Attention::create(attrs, Backend::CPU);

    std::vector<TensorDesc> descs = {q.desc(), k.desc(), v.desc()};
    std::vector<TensorView> ins   = {q, k, v};
    if (c.mask) { descs.push_back(mask_view.desc()); ins.push_back(mask_view); }
    auto out_descs = op->getOutputTensorDesc(descs);

    const cpu::AttentionPlan plan = cpu::get_attention_plan(attrs, descs, out_descs,
                                                            threads.back(),
                                                            /*use_thread_slots=*/true);

    // 2 matmuls per head: QK^T (Sq*Sk*D) + attn@V (Sq*D*Sk) = 2*Sq*Sk*D MACs.
    const double flops_per_iter = 4.0 * static_cast<double>(c.B) * H * Sq * Sk * D;

    std::vector<float> out_buf(static_cast<size_t>(out_descs[0].numel()));
    auto output = test::make_planar(out_descs[0], out_buf.data());

    std::printf("  %s\n", c.name);
    std::printf("    B=%lld H=%lld Sq=%lld Sk=%lld D=%lld  (Sq*Sk=%lld)%s\n",
                (long long)c.B, (long long)H, (long long)Sq, (long long)Sk,
                (long long)D, (long long)(Sq * Sk), c.mask ? "  mask" : "");
    if (plan.use_flash) {
        std::printf("    route=flash Br=%d Bc=%d  ws=%zuB\n",
                    plan.Br, plan.Bc, plan.workspace_size);
    } else {
        std::printf("    route=standard GEMM1(mc=%d nc=%d) GEMM2(mc=%d nc=%d)  ws=%zuB\n",
                    plan.mc1, plan.nc1, plan.mc2, plan.nc2, plan.workspace_size);
    }

    auto compute = [&](const ComputeContext& ctx) { op->compute(output, ins, ctx, nullptr); };
    std::vector<double> gflops;
    for (int nt : threads) {
        SimplePool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

        for (int i = 0; i < 2; ++i) { compute(use_ctx); }   // warm-up / page faults

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) { compute(use_ctx); }
        auto end = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        gflops.push_back(flops_per_iter * static_cast<double>(iters) / (ms * 1e6));
    }

    const double g1 = gflops.empty() ? 0.0 : gflops[0];
    std::printf("    ");
    for (size_t i = 0; i < threads.size(); ++i) {
        const double speedup = g1 > 0.0 ? gflops[i] / g1 : 0.0;
        std::printf(" %2dT %7.1f GFLOPS (%.2fx)", threads[i], gflops[i], speedup);
        if (i + 1 < threads.size()) { std::printf(" |"); }
    }
    std::printf("\n");
}

void bench_attention_case(const AttentionCase& c) {
    const int iters = 20;
    std::vector<int> threads{1, 2, 4, 8};
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 1 && std::find(threads.begin(), threads.end(), hw) == threads.end()) {
        threads.push_back(hw);
    }
    run_attention(c, threads, iters);
}

}  // anonymous namespace

// ============================================================
// Standard vs flash routing sweep
// ============================================================

NNOPS_BENCH(attention_f32_standard) {
    // Small Sq*Sk — tiled GEMM + softmax path.
    bench_attention_case({"attn f32 standard B2 S128 H8 D64", 2, 128, 8, 64});
}

NNOPS_BENCH(attention_f32_standard_wide) {
    bench_attention_case({"attn f32 standard-wide B1 S128 H8 D128", 1, 128, 8, 128});
}

NNOPS_BENCH(attention_f32_flash) {
    // Sq*Sk = 262144 -> online-softmax FlashAttention path.
    bench_attention_case({"attn f32 flash B1 S512 H8 D64", 1, 512, 8, 64});
}

NNOPS_BENCH(attention_f32_flash_128) {
    // D=128 -> multi-KV-block flash path.
    bench_attention_case({"attn f32 flash D128 B1 S512 H16 D128", 1, 512, 16, 128});
}

NNOPS_BENCH(attention_f32_flash_masked) {
    // Flash + additive mask.
    bench_attention_case({"attn f32 flash masked B2 S256 H8 D64", 2, 256, 8, 64,
                          /*Sk=*/0, /*mask=*/true});
}