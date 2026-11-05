/// @file bench_attention.cpp
/// @brief Attention-family micro-benchmark — route (standard vs flash) and
///        thread scaling for MultiHeadAttention, GroupQueryAttention and
///        MultiLatentAttention.
///
/// Every case runs the plan-driven fused fast path and reports GFLOPS at 1, 2,
/// 4, 8 and hardware threads. QK^T (transpose-B GEMM) then softmax then
/// attn @ V (direct GEMM), per (batch, query head) — or the online-softmax
/// FlashAttention path once Sq*Sk reaches kFlashMinScores (1<<16).
///
/// FLOPs count head work, not distinct kv heads: under GQA every query head
/// still runs its own QK and AV, even though they read a shared K/V slice.
/// MLA counts both widths, qk and value, since the query-key product is over
/// kv_lora_rank + qk_rope_head_dim while the value product is over v_head_dim.
///
/// BenchPool carries the two arms of common/cpu_pool.hpp: an OpenMP team when
/// the build has one (unless NNOPS_BENCH_OMP opts out), else a pool that spawns
/// per call. Both sit behind the same hooks, so two tables differ only in their
/// scheduler.

#include "nnops/ops/group_query_attention.hpp"
#include "nnops/ops/multi_head_attention.hpp"
#include "nnops/ops/multi_latent_attention.hpp"
#include "backend/cpu/attention/attention.h"        // get_attention_plan, AttentionConfig
#include "common/bench_harness.hpp"
#include "common/cpu_pool.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <type_traits>
#include <vector>

using namespace nnops;
namespace cpu = nnops::backend::cpu;
using cpu::half;

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

template <typename T>
auto make_random(std::span<const int64_t> shape, uint64_t seed) {
    if constexpr (std::is_same_v<T, float>) {
        return test::make_random_tensor(shape, -1.0f, 1.0f, seed);
    } else {
        return test::make_random_f16_tensor(shape, -1.0f, 1.0f, seed);
    }
}

std::vector<int> thread_list() {
    std::vector<int> threads{1, 2, 4, 8};
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 1 && std::find(threads.begin(), threads.end(), hw) == threads.end()) {
        threads.push_back(hw);
    }
    return threads;
}

/// Time `body(ctx)` over the thread list; returns GFLOPS at each width.
template <typename Body>
std::vector<double> measure(const std::vector<int>& threads, int iters,
                            double flops_per_iter, Body&& body)
{
    std::vector<double> gflops;
    for (int nt : threads) {
        BenchPool pool(nt);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        const ComputeContext& use_ctx = (nt > 1) ? ctx : ComputeContext{};

        for (int i = 0; i < 2; ++i) { body(use_ctx); }   // warm-up / page faults

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) { body(use_ctx); }
        auto end = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        gflops.push_back(flops_per_iter * static_cast<double>(iters) / (ms * 1e6));
    }
    return gflops;
}

void print_row(const std::vector<int>& threads, const std::vector<double>& gflops) {
    const double g1 = gflops.empty() ? 0.0 : gflops[0];
    std::printf("    ");
    for (size_t i = 0; i < threads.size(); ++i) {
        const double speedup = g1 > 0.0 ? gflops[i] / g1 : 0.0;
        std::printf(" %2dT %7.1f GFLOPS (%.2fx)", threads[i], gflops[i], speedup);
        if (i + 1 < threads.size()) { std::printf(" |"); }
    }
    std::printf("\n");
}

// ---- MultiHeadAttention ---------------------------------------------------

struct MhaCase {
    const char* name;
    int64_t B, S, H, D;     // merged [B, S, H*D], self-attention (Sq = S)
    int64_t Sk = 0;         // key seq len; 0 -> S
    bool mask = false;
    bool f16 = false;
    bool causal = false;
};

template <typename T>
void run_mha(const MhaCase& c, const std::vector<int>& threads, int iters) {
    const int64_t Sq = c.S;
    const int64_t Sk = (c.Sk == 0) ? c.S : c.Sk;
    const int64_t H  = c.H;
    const int64_t D  = c.D;

    const int64_t qshape[] = {c.B, Sq, H * D};
    const int64_t kshape[] = {c.B, Sk, H * D};

    auto [q_vec, q] = make_random<T>(qshape, 11);
    auto [k_vec, k] = make_random<T>(kshape, 12);
    auto [v_vec, v] = make_random<T>(kshape, 13);

    std::vector<T> m_vec;
    TensorView mask_view;
    if (c.mask) {
        const int64_t mshape[] = {Sq, Sk};
        auto [mv, m] = make_random<T>(mshape, 14);
        m_vec = std::move(mv);
        mask_view = m;
    }

    MultiHeadAttentionAttributes attrs{};
    attrs.num_heads = H;
    attrs.scale = 0.0f;   // auto 1/sqrt(D)
    attrs.causal = c.causal;

    auto op = MultiHeadAttention::create(attrs, Backend::CPU);

    std::vector<TensorDesc> descs = {q.desc(), k.desc(), v.desc()};
    std::vector<TensorView> ins   = {q, k, v};
    if (c.mask) { descs.push_back(mask_view.desc()); ins.push_back(mask_view); }
    auto out_descs = op->getOutputTensorDesc(descs);

    const cpu::AttentionPlan plan = cpu::get_attention_plan({H}, descs, out_descs,
                                                            threads.back(),
                                                            /*use_thread_slots=*/true);

    // 2 matmuls per head: QK^T (Sq*Sk*D) + attn@V (Sq*D*Sk) = 2*Sq*Sk*D MACs.
    const double flops = 4.0 * static_cast<double>(c.B) * H * Sq * Sk * D;

    std::vector<T> out_buf(static_cast<size_t>(out_descs[0].numel()));
    auto output = test::make_planar(out_descs[0], out_buf.data());

    std::printf("  %s\n", c.name);
    std::printf("    B=%lld H=%lld Sq=%lld Sk=%lld D=%lld  (Sq*Sk=%lld)  %s%s%s\n",
                (long long)c.B, (long long)H, (long long)Sq, (long long)Sk, (long long)D,
                (long long)(Sq * Sk), c.f16 ? "f16" : "f32",
                c.mask ? "  mask" : "", c.causal ? "  causal" : "");
    if (plan.use_flash) {
        std::printf("    route=flash Br=%d Bc=%d  ws=%zuB\n",
                    plan.Br, plan.Bc, plan.workspace_size);
    } else {
        std::printf("    route=standard GEMM1(mc=%d nc=%d) GEMM2(mc=%d nc=%d)  ws=%zuB\n",
                    plan.mc1, plan.nc1, plan.mc2, plan.nc2, plan.workspace_size);
    }

    const auto gflops = measure(threads, iters, flops, [&](const ComputeContext& ctx) {
        op->compute(output, ins, ctx, nullptr);
    });
    print_row(threads, gflops);
}

void bench_mha(const MhaCase& c) {
    const int iters = 20;
    const auto threads = thread_list();
    if (c.f16) { run_mha<half>(c, threads, iters); }
    else       { run_mha<float>(c, threads, iters); }
}

// ---- GroupQueryAttention --------------------------------------------------

struct GqaCase {
    const char* name;
    int64_t B, S, H, Hkv, D;
    int64_t Sk = 0;         // 0 -> S
    bool causal = true;
    bool f16 = false;
    int64_t past = 0;       // rows in the past segment, keyed before the new ones
};

template <typename T>
void run_gqa(const GqaCase& c, const std::vector<int>& threads, int iters) {
    const int64_t Sq = c.S;
    const int64_t Sk_new = (c.Sk == 0) ? c.S : c.Sk;
    const int64_t Sk_past = c.past;   // chunked/step input: the run is [past | new]
    const int64_t Sk = Sk_new + Sk_past;

    const int64_t qshape[]  = {c.B, Sq, c.H * c.D};
    const int64_t kvshape[] = {c.B, Sk_new, c.Hkv * c.D};
    const int64_t pshape[]  = {c.B, Sk_past, c.Hkv * c.D};

    auto [q_vec, q]   = make_random<T>(qshape, 21);
    auto [k_vec, k]   = make_random<T>(kvshape, 22);
    auto [v_vec, v]   = make_random<T>(kvshape, 23);
    std::vector<T> pk_vec, pv_vec;
    TensorView past_k, past_v;
    if (Sk_past > 0) {
        auto [a, ka] = make_random<T>(pshape, 24);
        pk_vec = std::move(a);
        past_k = ka;
        auto [b, vb] = make_random<T>(pshape, 25);
        pv_vec = std::move(b);
        past_v = vb;
    }

    GroupQueryAttentionAttributes attrs{};
    attrs.num_heads    = c.H;
    attrs.kv_num_heads = c.Hkv;
    attrs.causal       = c.causal;
    attrs.position_offset = Sk_past > 0 ? Sk_past : -1;

    auto op = GroupQueryAttention::create(attrs, Backend::CPU);

    std::vector<TensorDesc> descs = {q.desc(), k.desc(), v.desc()};
    std::vector<TensorView> ins   = {q, k, v};
    if (Sk_past > 0) {
        descs.push_back(TensorDesc{});
        ins.push_back(TensorView{});
        descs.push_back(past_k.desc());
        ins.push_back(past_k);
        descs.push_back(past_v.desc());
        ins.push_back(past_v);
    }
    auto out_descs = op->getOutputTensorDesc(descs);

    cpu::AttentionConfig cfg;
    cfg.num_heads = c.H;
    cfg.num_kv_heads = c.Hkv;
    cfg.causal = c.causal;
    cfg.position_offset = attrs.position_offset;
    const cpu::AttentionPlan plan = cpu::get_attention_plan(cfg, descs, out_descs,
                                                            threads.back(), true);

    // Every query head runs its own QK and AV, whether or not it shares a kv
    // head, so the count is over H — not H_kv.
    const double flops = 4.0 * static_cast<double>(c.B) * c.H * Sq * Sk * c.D;

    std::vector<T> out_buf(static_cast<size_t>(out_descs[0].numel()));
    auto output = test::make_planar(out_descs[0], out_buf.data());

    std::printf("  %s\n", c.name);
    std::printf("    B=%lld H=%lld Hkv=%lld Sq=%lld Sk=%lld D=%lld  (Sq*Sk=%lld)  %s%s%s\n",
                (long long)c.B, (long long)c.H, (long long)c.Hkv, (long long)Sq,
                (long long)Sk, (long long)c.D, (long long)(Sq * Sk),
                c.f16 ? "f16" : "f32", c.causal ? "  causal" : "",
                Sk_past > 0 ? "  step" : "");
    std::printf("    route=%s  ws=%zuB\n",
                plan.use_decode ? "decode" : (plan.use_flash ? "flash" : "standard"),
                plan.workspace_size);

    const auto gflops = measure(threads, iters, flops, [&](const ComputeContext& ctx) {
        op->compute(output, ins, ctx, nullptr);
    });
    print_row(threads, gflops);
}

void bench_gqa(const GqaCase& c) {
    const int iters = 20;
    const auto threads = thread_list();
    if (c.f16) { run_gqa<half>(c, threads, iters); }
    else       { run_gqa<float>(c, threads, iters); }
}

// ---- MultiLatentAttention -------------------------------------------------

struct MlaCase {
    const char* name;
    int64_t B, S, H, Hkv, rank, rope, vdim;
    int64_t Sk = 0;         // 0 -> S
    bool causal = true;
    bool f16 = false;
    int64_t past = 0;       // latent rows in the past segment
};

template <typename T>
void run_mla(const MlaCase& c, const std::vector<int>& threads, int iters) {
    const int64_t Sq = c.S;
    const int64_t Sk_new = (c.Sk == 0) ? c.S : c.Sk;
    const int64_t Sk_past = c.past;   // chunked/step input: the run is [past | new]
    const int64_t Sk = Sk_new + Sk_past;
    const int64_t dkv = c.rank + c.rope;

    const int64_t q_width  = c.H * dkv;
    const int64_t kv_width = c.Hkv * dkv;

    const int64_t q_dims[]  = {c.B, Sq, q_width};
    const int64_t kv_dims[] = {c.B, Sk_new, kv_width};
    const int64_t p_dims[]  = {c.B, Sk_past, kv_width};

    auto [q_vec, q]   = make_random<T>(q_dims, 31);
    auto [kv_vec, kv] = make_random<T>(kv_dims, 32);
    std::vector<T> pkv_vec;
    TensorView past_kv;
    if (Sk_past > 0) {
        auto [a, pa] = make_random<T>(p_dims, 33);
        pkv_vec = std::move(a);
        past_kv = pa;
    }

    MultiLatentAttentionAttributes attrs{};
    attrs.num_heads        = c.H;
    attrs.kv_num_heads     = c.Hkv;
    attrs.kv_lora_rank     = c.rank;
    attrs.qk_rope_head_dim = c.rope;
    attrs.v_head_dim       = c.vdim;
    attrs.causal           = c.causal;
    attrs.position_offset  = Sk_past > 0 ? Sk_past : -1;

    auto op = MultiLatentAttention::create(attrs, Backend::CPU);

    std::vector<TensorDesc> descs = {q.desc(), kv.desc()};
    std::vector<TensorView> ins   = {q, kv};
    if (Sk_past > 0) {
        descs.push_back(TensorDesc{});
        ins.push_back(TensorView{});
        descs.push_back(past_kv.desc());
        ins.push_back(past_kv);
    }
    auto out_descs = op->getOutputTensorDesc(descs);

    cpu::AttentionConfig cfg;
    cfg.num_heads = c.H;
    cfg.num_kv_heads = c.Hkv;
    cfg.v_head_dim = c.vdim > 0 ? c.vdim : c.rank;
    cfg.v_is_k = true;
    cfg.causal = c.causal;
    cfg.position_offset = attrs.position_offset;
    const cpu::AttentionPlan plan = cpu::get_attention_plan(cfg, descs, out_descs,
                                                            threads.back(), true);

    // QK over the latent width, AV over the value width.
    const double flops = 2.0 * static_cast<double>(c.B) * c.H * Sq * Sk *
                         static_cast<double>(dkv + (c.vdim > 0 ? c.vdim : c.rank));

    std::vector<T> out_buf(static_cast<size_t>(out_descs[0].numel()));
    auto output = test::make_planar(out_descs[0], out_buf.data());

    std::printf("  %s\n", c.name);
    std::printf("    B=%lld H=%lld Hkv=%lld Sq=%lld Sk=%lld Dkv=%lld v=%lld  %s%s%s\n",
                (long long)c.B, (long long)c.H, (long long)c.Hkv, (long long)Sq,
                (long long)Sk, (long long)dkv,
                (long long)(c.vdim > 0 ? c.vdim : c.rank),
                c.f16 ? "f16" : "f32", c.causal ? "  causal" : "",
                Sk_past > 0 ? "  step" : "");
    std::printf("    route=%s  ws=%zuB\n",
                plan.use_decode ? "decode" : (plan.use_flash ? "flash" : "standard"),
                plan.workspace_size);

    const auto gflops = measure(threads, iters, flops, [&](const ComputeContext& ctx) {
        op->compute(output, ins, ctx, nullptr);
    });
    print_row(threads, gflops);
}

void bench_mla(const MlaCase& c) {
    const int iters = 20;
    const auto threads = thread_list();
    if (c.f16) { run_mla<half>(c, threads, iters); }
    else       { run_mla<float>(c, threads, iters); }
}

}  // anonymous namespace

// ============================================================
// MultiHeadAttention — standard vs flash routing sweep
// ============================================================

NNOPS_BENCH(attention_f32_standard) {
    // Small Sq*Sk — tiled GEMM + softmax path.
    bench_mha({"attn f32 standard B2 S128 H8 D64", 2, 128, 8, 64});
}

NNOPS_BENCH(attention_f32_standard_wide) {
    bench_mha({"attn f32 standard-wide B1 S128 H8 D128", 1, 128, 8, 128});
}

NNOPS_BENCH(attention_f32_standard_d256) {
    // D=256 -> GEMM1's K loop runs twice at Kc=128.
    bench_mha({"attn f32 standard D256 B2 S128 H8 D256", 2, 128, 8, 256});
}

NNOPS_BENCH(attention_f32_flash) {
    // Sq*Sk = 262144 -> online-softmax FlashAttention path.
    bench_mha({"attn f32 flash B1 S512 H8 D64", 1, 512, 8, 64});
}

NNOPS_BENCH(attention_f32_flash_128) {
    // D=128 doubles head_pairs, halving the L2-derived Bc before the Sk cap.
    bench_mha({"attn f32 flash D128 B1 S512 H16 D128", 1, 512, 16, 128});
}

NNOPS_BENCH(attention_f32_flash_d256) {
    // D=256 in the flash path -> GEMM1's K loop runs twice at Kc=128.
    bench_mha({"attn f32 flash D256 B1 S1024 H8 D256", 1, 1024, 8, 256});
}

NNOPS_BENCH(attention_f32_flash_d192) {
    // D=192 -> GEMM1's K loop splits 128 + 64 at Kc=128.
    bench_mha({"attn f32 flash D192 B1 S1024 H8 D192", 1, 1024, 8, 192});
}

NNOPS_BENCH(attention_f32_flash_masked) {
    // Flash + additive mask.
    bench_mha({"attn f32 flash masked B2 S256 H8 D64", 2, 256, 8, 64,
               /*Sk=*/0, /*mask=*/true});
}

NNOPS_BENCH(attention_f32_standard_causal) {
    // Causal prefill on the standard route.
    bench_mha({"attn f32 causal B2 S128 H8 D64", 2, 128, 8, 64,
               /*Sk=*/0, /*mask=*/false, /*f16=*/false, /*causal=*/true});
}

// ============================================================
// f16 twins of the f32 sweep, minus the D=192 shape
// ============================================================

NNOPS_BENCH(attention_f16_standard) {
    bench_mha({"attn f16 standard B2 S128 H8 D64", 2, 128, 8, 64,
               /*Sk=*/0, /*mask=*/false, /*f16=*/true});
}

NNOPS_BENCH(attention_f16_standard_wide) {
    bench_mha({"attn f16 standard-wide B1 S128 H8 D128", 1, 128, 8, 128,
               /*Sk=*/0, /*mask=*/false, /*f16=*/true});
}

NNOPS_BENCH(attention_f16_standard_d256) {
    bench_mha({"attn f16 standard D256 B2 S128 H8 D256", 2, 128, 8, 256,
               /*Sk=*/0, /*mask=*/false, /*f16=*/true});
}

NNOPS_BENCH(attention_f16_flash) {
    bench_mha({"attn f16 flash B1 S512 H8 D64", 1, 512, 8, 64,
               /*Sk=*/0, /*mask=*/false, /*f16=*/true});
}

NNOPS_BENCH(attention_f16_flash_128) {
    bench_mha({"attn f16 flash D128 B1 S512 H16 D128", 1, 512, 16, 128,
               /*Sk=*/0, /*mask=*/false, /*f16=*/true});
}

NNOPS_BENCH(attention_f16_flash_d256) {
    bench_mha({"attn f16 flash D256 B1 S1024 H8 D256", 1, 1024, 8, 256,
               /*Sk=*/0, /*mask=*/false, /*f16=*/true});
}

NNOPS_BENCH(attention_f16_flash_masked) {
    bench_mha({"attn f16 flash masked B2 S256 H8 D64", 2, 256, 8, 64,
               /*Sk=*/0, /*mask=*/true, /*f16=*/true});
}

// ============================================================
// GroupQueryAttention — prefill and decode shapes
// ============================================================

NNOPS_BENCH(gqa_f32_prefill) {
    // 4:1 grouping, 512-token prefill: the flash path.
    bench_gqa({"gqa f32 prefill B1 S512 H32 Hkv8 D128", 1, 512, 32, 8, 128});
}

NNOPS_BENCH(gqa_f32_prefill_long) {
    // 2048-token causal prefill: the flash path's triangle skip and its tile
    // sizes decide the result here.
    bench_gqa({"gqa f32 prefill B1 S2048 H32 Hkv8 D128", 1, 2048, 32, 8, 128});
}

NNOPS_BENCH(gqa_f32_decode) {
    // One new token against a 2048-token key run: the GEMV decode route, and
    // shape where a kv head's reuse actually shows up.
    bench_gqa({"gqa f32 decode B1 S1 H32 Hkv8 D128", 1, 1, 32, 8, 128, /*Sk=*/2048});
}

NNOPS_BENCH(gqa_f32_step_decode) {
    // The same decode shape, but the key run arrives as [past | new] — the
    // cache buffer plus the token just written — so the split's cost shows
    // against the single-run bench above.
    bench_gqa({"gqa f32 step decode B1 S1 H32 Hkv8 D128 past2048", 1, 1, 32, 8, 128,
               /*Sk=*/0, /*causal=*/true, /*f16=*/false, /*past=*/2048});
}

NNOPS_BENCH(gqa_f32_step_prefill) {
    // Chunked prefill: a 256-token chunk over 1024 cached keys.
    bench_gqa({"gqa f32 step prefill B1 S256 H32 Hkv8 D128 past1024", 1, 256, 32, 8, 128,
               /*Sk=*/0, /*causal=*/true, /*f16=*/false, /*past=*/1024});
}

NNOPS_BENCH(gqa_f32_mqa) {
    // MQA: one kv head for all 32 query heads.
    bench_gqa({"gqa f32 mqa B1 S512 H16 Hkv1 D64", 1, 512, 16, 1, 64});
}

NNOPS_BENCH(gqa_f16_prefill) {
    bench_gqa({"gqa f16 prefill B1 S512 H32 Hkv8 D128", 1, 512, 32, 8, 128,
               /*Sk=*/0, /*causal=*/true, /*f16=*/true});
}

// ============================================================
// MultiLatentAttention — DeepSeek-shaped latent, prefill and decode
// ============================================================

NNOPS_BENCH(mla_f32_prefill) {
    // kv_lora_rank 512 + rope 64, 128 query heads over one latent head.
    bench_mla({"mla f32 prefill B1 S128 H32 Hkv1 rank512 rope64 v512",
               1, 128, 32, 1, 512, 64, 512});
}

NNOPS_BENCH(mla_f32_prefill_long) {
    // 1024-token causal prefill over the latent: eight 64-row query blocks, so
    // the triangle skip and the latent width both show in the rate.
    bench_mla({"mla f32 prefill B1 S1024 H32 Hkv1 rank512 rope64 v512",
               1, 1024, 32, 1, 512, 64, 512});
}

NNOPS_BENCH(mla_f32_decode) {
    // Single-token decode against a 2048-token latent run.
    bench_mla({"mla f32 decode B1 S1 H64 Hkv1 rank512 rope64 v512",
               1, 1, 64, 1, 512, 64, 512, /*Sk=*/2048});
}

NNOPS_BENCH(mla_f32_step_decode) {
    // Decode whose latent run is [past | new]: the paged-cache step form.
    bench_mla({"mla f32 step decode B1 S1 H64 Hkv1 rank512 rope64 v512 past2048",
               1, 1, 64, 1, 512, 64, 512, /*Sk=*/0, /*causal=*/true, /*f16=*/false,
               /*past=*/2048});
}

NNOPS_BENCH(mla_f16_prefill) {
    bench_mla({"mla f16 prefill B1 S128 H32 Hkv1 rank512 rope64 v512",
               1, 128, 32, 1, 512, 64, 512, /*Sk=*/0, /*causal=*/true, /*f16=*/true});
}
