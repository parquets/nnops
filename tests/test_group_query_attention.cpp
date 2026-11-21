/// @file test_group_query_attention.cpp
/// @brief GroupQueryAttention tests — the class API against the CPU reference.
///
/// reference::group_query_attention_ref is f32 only, so f16 is compared against
/// the f32 reference on the same values with the arch-aware kF16AccumTol.
///
/// The operator has two input contracts, selected by `causal`:
///   - explicit (causal = false): Q, K, V and an external mask — the ViT /
///     encoder form, tested in merged and per-head layouts with padding masks,
///     additive bias, broadcast bias and packed documents;
///   - cache (causal = true): Q and a [B, H_kv, S, D] cache the caller wrote —
///     the decoder form, tested across prefill, decode, per-batch positions and
///     the standard / flash / decode routes, in fp and int8.
///
/// Both forms cover the three head ratios that matter (MQA, a grouped ratio
/// and H_kv == H) and both routes (standard, and flash once Sq*Sk reaches
/// kFlashMinScores = 1 << 16). The threaded cases drive the decode route over a
/// long run so the plan splits it into KV slices.

#include "nnops/ops/group_query_attention.hpp"
#include "nnops/ops/multi_head_attention.hpp"
#include "backend/cpu/attention/attention.h"   // AttentionConfig / get_attention_plan
#include "common/test_harness.hpp"
#include "common/cpu_pool.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace nnops;
namespace cpu = nnops::backend::cpu;

namespace nnops::backend::cpu::reference {
void group_query_attention_ref(const GroupQueryAttentionAttributes& attrs,
                               TensorView& output,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               void* workspace);
}

namespace {

TensorDesc td(std::initializer_list<int64_t> dims) {
    TensorDesc d;
    d.rank = static_cast<int64_t>(dims.size());
    for (int64_t v : dims) { d.dims.push_back(v); }
    return d;
}

/// Descriptor list addressed by slot; unset slots stay empty.
struct SlotDescs {
    std::vector<TensorDesc> d;
    SlotDescs& put(size_t slot, const TensorDesc& t) {
        while (d.size() <= slot) { d.push_back(TensorDesc{}); }
        d[slot] = t;
        return *this;
    }
};

struct Case {
    std::vector<int64_t> qshape;   // [B, Sq, H*D] or a rank-4 form
    std::vector<int64_t> kshape;   // the step's K/V block, same rank as Q
    int64_t heads = 4;
    int64_t kv_heads = 2;
    float scale = 0.0f;
    bool causal = true;            // selects the cache form
    bool mask = false;             // explicit form only
    int64_t past = 0;              // cache rows before the step's block
    int route = -1;                // 0 standard, 1 flash, 2 decode; -1 unchecked
    std::vector<int32_t> seqlens;  // per-batch valid run length; empty = full
    float softcap = 0.0f;
    int64_t window = 0;
    bool head_bias = false;        // [B, H, Sq, Sk] rather than [Sq, Sk]
    EpilogueActivateType gate = EpilogueActivateType::None;
    AttentionHeadLayout layout = AttentionHeadLayout::Auto;
    bool int8 = false;             // quantize the cache
    bool per_tensor = false;       // one scale instead of one per token
    int64_t sink = 0;              // always-visible leading cache rows
    std::vector<int64_t> pos_ids;  // per-batch first query row; empty = derived
};

/// Element of a Q/K/V-like view: rank 3 is merged [B, S, H*D], rank 4 is
/// [B, H, S, D] or, with @p bshd, [B, S, H, D].
float head_elem(const TensorView& v, bool bshd, int64_t b, int64_t s, int64_t h, int64_t d,
                int64_t dhead)
{
    const float* p = v.ptr<float>();
    if (v.rank() == 3) {
        return p[b * v.stride_elems(0) + s * v.stride_elems(1) + h * dhead + d];
    }
    if (bshd) {
        return p[b * v.stride_elems(0) + s * v.stride_elems(1) + h * v.stride_elems(2) + d];
    }
    return p[b * v.stride_elems(0) + h * v.stride_elems(1) + s * v.stride_elems(2) + d];
}

/// Fixed-size pool over a CpuBackend (test_conv2d.cpp pattern); items are
/// claimed atomically. Installed only for the cases that need a worker grid.
struct SimplePool {
    explicit SimplePool(int nthreads) : nthreads_(nthreads) {
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

/// Everything one call needs, with the buffers kept alive beside the views.
struct Built {
    GroupQueryAttentionAttributes attrs;
    std::vector<std::vector<float>> keep_f32;
    std::vector<std::vector<int8_t>> keep_i8;
    std::vector<int32_t> sl;
    std::vector<int64_t> pos;
    std::vector<TensorDesc> descs;
    std::vector<TensorView> ins;
    bool ok = false;
};

/// Symmetric int8 quantization of one [B, Hkv, Skv, D] cache: one scale per
/// token ([B, Hkv, Skv]) or one for the whole tensor.
void quantize_cache(const std::vector<float>& x, int64_t B, int64_t Hkv, int64_t Skv,
                    int64_t D, bool per_tensor, std::vector<int8_t>& q,
                    std::vector<float>& scale)
{
    const int64_t n = static_cast<int64_t>(x.size());
    q.resize(static_cast<size_t>(n));
    const auto quant = [](float v, float s) {
        const float t = std::round(v / s);
        return static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, t)));
    };
    if (per_tensor) {
        float amax = 0.0f;
        for (float v : x) { amax = std::max(amax, std::fabs(v)); }
        const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        scale.assign(1, s);
        for (int64_t i = 0; i < n; ++i) {
            q[static_cast<size_t>(i)] = quant(x[static_cast<size_t>(i)], s);
        }
        return;
    }
    scale.assign(static_cast<size_t>(B * Hkv * Skv), 1.0f);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < Hkv; ++h) {
            for (int64_t t = 0; t < Skv; ++t) {
                const int64_t base = ((b * Hkv + h) * Skv + t) * D;
                float amax = 0.0f;
                for (int64_t d = 0; d < D; ++d) {
                    amax = std::max(amax, std::fabs(x[static_cast<size_t>(base + d)]));
                }
                const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
                scale[static_cast<size_t>((b * Hkv + h) * Skv + t)] = s;
                for (int64_t d = 0; d < D; ++d) {
                    q[static_cast<size_t>(base + d)] =
                        quant(x[static_cast<size_t>(base + d)], s);
                }
            }
        }
    }
}

/// Assemble the input list of @p c. In the cache form the step's rows are
/// copied into a [B, H_kv, Skv, D] cache at each batch's own tail, so the
/// query block is always the last Sq rows of the run.
Built build(const Case& c, uint64_t seed)
{
    Built b;
    b.attrs.num_heads = c.heads;
    b.attrs.kv_num_heads = c.kv_heads;
    b.attrs.scale = c.scale;
    b.attrs.causal = c.causal;
    b.attrs.softcap = c.softcap;
    b.attrs.local_window_size = c.window > 0 ? c.window : -1;
    b.attrs.sink_tokens = c.sink;
    b.attrs.output_gate = c.gate;
    b.attrs.head_layout = c.layout;

    const bool bshd = c.layout == AttentionHeadLayout::BSHD;
    const bool merged = c.kshape.size() == 3;
    const int64_t B = c.qshape[0];
    const int64_t Hkv = c.kv_heads;
    const int64_t D = merged ? c.kshape[2] / Hkv : c.kshape[3];
    const int64_t Sk_new = merged ? c.kshape[1] : (bshd ? c.kshape[1] : c.kshape[2]);
    const int64_t Sq = merged ? c.qshape[1] : (bshd ? c.qshape[1] : c.qshape[2]);
    if (Sq <= 0 || Sk_new <= 0 || D <= 0 || B <= 0) {
        return b;
    }

    auto [qv, q] = test::make_random_tensor(c.qshape, -1.0f, 1.0f, seed);
    b.keep_f32.push_back(std::move(qv));
    auto [kv, k] = test::make_random_tensor(c.kshape, -1.0f, 1.0f, seed + 1);
    b.keep_f32.push_back(std::move(kv));
    auto [vv, v] = test::make_random_tensor(c.kshape, -1.0f, 1.0f, seed + 2);
    b.keep_f32.push_back(std::move(vv));

    TensorView past_k, past_v;
    if (c.past > 0) {
        auto ps = c.kshape;
        ps[merged ? 1 : (bshd ? 1 : 2)] = c.past;
        auto [a, ka] = test::make_random_tensor(ps, -1.0f, 1.0f, seed + 3);
        b.keep_f32.push_back(std::move(a));
        past_k = ka;
        auto [bv, vb] = test::make_random_tensor(ps, -1.0f, 1.0f, seed + 4);
        b.keep_f32.push_back(std::move(bv));
        past_v = vb;
    }

    b.ins.assign(9, TensorView{});
    b.ins[0] = q;

    if (!c.causal) {
        b.ins[1] = k;
        b.ins[2] = v;
        if (c.mask) {
            const std::vector<int64_t> ms = c.head_bias
                ? std::vector<int64_t>{B, c.heads, Sq, Sk_new}
                : std::vector<int64_t>{Sq, Sk_new};
            auto [mv, m] = test::make_random_tensor(ms, -2.0f, 2.0f, seed + 5);
            b.keep_f32.push_back(std::move(mv));
            b.ins[3] = m;
        }
        if (!c.seqlens.empty()) {
            b.sl = c.seqlens;
            b.ins[4] = TensorView(std::vector<int64_t>{static_cast<int64_t>(b.sl.size())},
                                  DataType::s32, b.sl.data());
        }
    } else {
        if (c.mask) {
            return b; // the cache form derives its mask from the positions
        }
        int64_t Skv = c.past + Sk_new;
        for (const int32_t L : c.seqlens) {
            Skv = std::max<int64_t>(Skv, L);
        }
        if (Skv < Sq) {
            return b;
        }

        const auto build_one = [&](const TensorView& nv, const TensorView& pv,
                                   std::vector<float>& out) -> bool {
            out.assign(static_cast<size_t>(B * Hkv * Skv * D), 0.0f);
            for (int64_t bb = 0; bb < B; ++bb) {
                const int64_t L = c.seqlens.empty() ? Skv : c.seqlens[static_cast<size_t>(bb)];
                const int64_t start = L - Sk_new;
                if (L < Sq || start < 0 || start > c.past) {
                    return false;
                }
                for (int64_t h = 0; h < Hkv; ++h) {
                    for (int64_t t = 0; t < L; ++t) {
                        for (int64_t d = 0; d < D; ++d) {
                            const float e = (t < start)
                                ? head_elem(pv, bshd, bb, c.past - start + t, h, d, D)
                                : head_elem(nv, bshd, bb, t - start, h, d, D);
                            out[static_cast<size_t>(((bb * Hkv + h) * Skv + t) * D + d)] = e;
                        }
                    }
                }
            }
            return true;
        };
        std::vector<float> ck, cv;
        if (!build_one(k, past_k, ck) || !build_one(v, past_v, cv)) {
            return b;
        }

        const std::vector<int64_t> cdims = {B, Hkv, Skv, D};
        if (!c.int8) {
            b.keep_f32.push_back(std::move(ck));
            b.ins[1] = TensorView(std::span<const int64_t>(cdims), DataType::f32,
                                  b.keep_f32.back().data());
            b.keep_f32.push_back(std::move(cv));
            b.ins[2] = TensorView(std::span<const int64_t>(cdims), DataType::f32,
                                  b.keep_f32.back().data());
        } else {
            std::vector<int8_t> qk, qv;
            std::vector<float> sk, sv;
            quantize_cache(ck, B, Hkv, Skv, D, c.per_tensor, qk, sk);
            quantize_cache(cv, B, Hkv, Skv, D, c.per_tensor, qv, sv);
            b.keep_i8.push_back(std::move(qk));
            b.ins[1] = TensorView(std::span<const int64_t>(cdims), DataType::s8,
                                  b.keep_i8.back().data());
            b.keep_i8.push_back(std::move(qv));
            b.ins[2] = TensorView(std::span<const int64_t>(cdims), DataType::s8,
                                  b.keep_i8.back().data());
            const std::vector<int64_t> sdim = c.per_tensor
                ? std::vector<int64_t>{1}
                : std::vector<int64_t>{B, Hkv, Skv};
            b.keep_f32.push_back(std::move(sk));
            b.ins[5] = TensorView(sdim, DataType::f32, b.keep_f32.back().data());
            b.keep_f32.push_back(std::move(sv));
            b.ins[6] = TensorView(sdim, DataType::f32, b.keep_f32.back().data());
        }
        if (!c.seqlens.empty()) {
            b.sl = c.seqlens;
            b.ins[3] = TensorView(std::vector<int64_t>{static_cast<int64_t>(b.sl.size())},
                                  DataType::s32, b.sl.data());
        }
        if (!c.pos_ids.empty()) {
            b.pos = c.pos_ids;
            b.ins[4] = TensorView(std::vector<int64_t>{static_cast<int64_t>(b.pos.size())},
                                  DataType::s64, b.pos.data());
        }
    }

    if (c.gate != EpilogueActivateType::None) {
        auto [gv, g] = test::make_random_tensor(c.qshape, -4.0f, 4.0f, seed + 6);
        b.keep_f32.push_back(std::move(gv));
        b.ins[8] = g;
    }

    b.descs.clear();
    for (const auto& t : b.ins) { b.descs.push_back(t.desc()); }
    b.ok = true;
    return b;
}

/// fast path vs reference on f32; returns false on the first mismatch.
bool fast_vs_ref(const Case& c, bool& used_flash, cpu::AttentionPlan* plan_out = nullptr,
                 const CpuBackend* pool = nullptr)
{
    Built b = build(c, 100);
    if (!b.ok) {
        std::printf("        case could not be assembled\n");
        return false;
    }
    auto op = GroupQueryAttention::create(b.attrs, Backend::CPU);
    const auto out_descs = op->getOutputTensorDesc(b.descs);
    if (out_descs.size() != 1) {
        std::printf("        expected one output, got %zu\n", out_descs.size());
        return false;
    }
    const int64_t out_numel = out_descs[0].numel();

    cpu::AttentionConfig pcfg;
    pcfg.num_heads = c.heads;
    pcfg.num_kv_heads = c.kv_heads;
    pcfg.causal = c.causal;
    pcfg.softcap = c.softcap;
    pcfg.window = c.window;
    pcfg.bshd = c.layout == AttentionHeadLayout::BSHD;
    if (c.causal) {
        pcfg.seqlens_slot = 3;
        pcfg.seqlens_is_length = true;
        pcfg.pos_ids_slot = 4;
        pcfg.k_scale_slot = 5;
        pcfg.v_scale_slot = 6;
        pcfg.sink_slot = 7;
        pcfg.gate_slot = 8;
    } else {
        pcfg.bias_slot = 3;
        pcfg.kpm_slot = 4;
        pcfg.sink_slot = 5;
        pcfg.cu_q_slot = 6;
        pcfg.cu_k_slot = 7;
        pcfg.gate_slot = 8;
        pcfg.kv_bshd = pcfg.bshd;
    }
    const cpu::AttentionPlan plan = cpu::get_attention_plan(
        pcfg, b.descs, out_descs, pool != nullptr ? pool->worker_count() : 1,
        pool != nullptr && pool->thread_slot_count() > 0);
    used_flash = plan.use_flash;
    if (plan_out) { *plan_out = plan; }
    if (c.route >= 0) {
        const int got = plan.use_decode ? 2 : (plan.use_flash ? 1 : 0);
        if (got != c.route) {
            std::printf("        route mismatch: want %d, got %d\n", c.route, got);
            return false;
        }
    }

    std::vector<float> fast_buf(static_cast<size_t>(out_numel));
    auto fast_out = test::make_channel_first(out_descs[0], fast_buf.data());
    ComputeContext fast_ctx;
    if (pool != nullptr) { fast_ctx.cpu = *pool; }
    if (op->compute(fast_out, b.ins, fast_ctx, nullptr) != Status::ok) {
        std::printf("        compute rejected the call\n");
        return false;
    }

    std::vector<float> ref_buf(static_cast<size_t>(out_numel));
    auto ref_out = test::make_channel_first(out_descs[0], ref_buf.data());
    nnops::backend::cpu::reference::group_query_attention_ref(b.attrs, ref_out, b.ins, {},
                                                              nullptr);

    for (int64_t i = 0; i < out_numel; ++i) {
        const float diff = std::abs(fast_buf[static_cast<size_t>(i)] -
                                    ref_buf[static_cast<size_t>(i)]);
        const float thr = 1e-4f + 1e-3f * std::max(std::abs(fast_buf[static_cast<size_t>(i)]),
                                                   std::abs(ref_buf[static_cast<size_t>(i)]));
        if (diff > thr) {
            std::printf("        worst mismatch at %lld: fast %.6f vs ref %.6f\n",
                        (long long)i, (double)fast_buf[static_cast<size_t>(i)],
                        (double)ref_buf[static_cast<size_t>(i)]);
            return false;
        }
    }
    return true;
}

} // anonymous namespace

// ============================================================
// Shapes and descriptor rules
// ============================================================

NNOPS_TEST(gqa_output_shape_follows_q) {
    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = 8;
    attrs.kv_num_heads = 2;
    attrs.causal = false;

    auto op = GroupQueryAttention::create(attrs, Backend::CPU);

    TensorDesc q = td({1, 5, 8 * 16});     // [B, Sq, H*D]
    TensorDesc k = td({1, 9, 2 * 16});     // [B, Sk, H_kv*D]
    TensorDesc v = td({1, 9, 2 * 16});
    const TensorDesc descs[] = {q, k, v};
    auto outs = op->getOutputTensorDesc(descs);

    NNOPS_EXPECT_EQ(outs.size(), size_t(1));
    NNOPS_EXPECT_EQ(outs[0].rank, int64_t(3));
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t(1));
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t(5));
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t(8 * 16));

    const TensorDesc out_descs[] = {outs[0]};
    NNOPS_EXPECT_TRUE(op->validate(descs, out_descs) == Status::ok);

    // H % H_kv != 0 is not a legal grouping.
    GroupQueryAttentionAttributes bad;
    bad.num_heads = 8;
    bad.kv_num_heads = 3;
    bad.causal = false;
    auto bad_op = GroupQueryAttention::create(bad, Backend::CPU);
    NNOPS_EXPECT_TRUE(bad_op->validate(descs, out_descs) == Status::invalid_argument);

    // K's kv-head width must match kv_num_heads.
    TensorDesc k_bad = td({1, 9, 8 * 16});
    const TensorDesc bad_descs[] = {q, k_bad, v};
    NNOPS_EXPECT_TRUE(op->validate(bad_descs, out_descs) == Status::invalid_argument);
}

NNOPS_TEST(gqa_rejects_seq_mismatch) {
    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = 4;
    attrs.kv_num_heads = 2;
    attrs.causal = false;
    auto op = GroupQueryAttention::create(attrs, Backend::CPU);

    TensorDesc q = td({1, 5, 4 * 8});
    TensorDesc k = td({1, 9, 2 * 8});
    TensorDesc v = td({1, 7, 2 * 8});   // shorter value run
    const TensorDesc descs[] = {q, k, v};
    const TensorDesc out_descs[] = {td({1, 5, 4 * 8})};
    NNOPS_EXPECT_TRUE(op->validate(descs, out_descs) == Status::invalid_argument);
}

// ============================================================
// Explicit form (ViT / encoder)
// ============================================================

NNOPS_TEST(gqa_explicit_matches_ref) {
    const Case cases[] = {
        // MQA: one kv head for every query head, merged.
        {{1, 6, 8 * 8},   {1, 6, 1 * 8},  8, 1, 0.0f, false},
        // Grouped ratio 2, merged.
        {{2, 5, 6 * 8},   {2, 5, 3 * 8},  6, 3, 0.0f, false},
        // Ratio 4 with a non-default scale.
        {{1, 4, 8 * 16},  {1, 4, 2 * 16}, 8, 2, 0.5f, false},
        // Additive mask, shared by every batch and head.
        {{1, 6, 4 * 8},   {1, 6, 2 * 8},  4, 2, 0.0f, false, true},
        // Per-head additive bias.
        {{2, 5, 4 * 8},   {2, 5, 2 * 8},  4, 2, 0.0f, false, true, 0, -1, {}, 0.0f, 0, true},
        // Cross attention: a shorter query run reading a longer key run.
        {{1, 3, 4 * 8},   {1, 9, 1 * 8},  4, 1, 0.0f, false},
        // Per-batch padding lengths: each batch reads its own prefix.
        {{2, 4, 4 * 8},   {2, 7, 2 * 8},  4, 2, 0.0f, false, false, 0, -1, {7, 3}},
        // Explicit [B, H, S, D] / [B, H_kv, S, D].
        {{2, 6, 5, 8},    {2, 3, 5, 8},   6, 3, 0.0f, false},
        // H_kv == H is plain MHA.
        {{1, 7, 4 * 8},   {1, 7, 4 * 8},  4, 4, 0.0f, false},
        // Multi-k-block head dim (D = 192 splits 128 + 64 at Kc = 128).
        {{1, 4, 2 * 192}, {1, 4, 1 * 192}, 2, 1, 0.0f, false},
    };

    for (const auto& c : cases) {
        bool flash = false;
        NNOPS_EXPECT_TRUE(fast_vs_ref(c, flash));
        (void)flash;
    }
}

NNOPS_TEST(gqa_explicit_flash_matches_ref) {
    // Sq*Sk >= 65536 routes through the online-softmax flash path.
    const Case cases[] = {
        {{1, 256, 8 * 64}, {1, 256, 1 * 64}, 8, 1, 0.0f, false, false, 0, -1, {}, 0.0f, 0, true},
        {{1, 320, 4 * 32}, {1, 320, 2 * 32}, 4, 2, 0.0f, false},
        {{1, 256, 2 * 128}, {1, 512, 1 * 128}, 2, 1, 0.0f, false},
    };

    for (const auto& c : cases) {
        bool flash = false;
        const bool ok = fast_vs_ref(c, flash);
        NNOPS_EXPECT_TRUE(flash);
        NNOPS_EXPECT_TRUE(ok);
    }
}

// ============================================================
// Cache form (decoder)
// ============================================================

NNOPS_TEST(gqa_cache_prefill_matches_ref) {
    // A full prefill: the step's rows are the whole cache, so every row reads
    // the bottom-right causal prefix.
    const Case cases[] = {
        {{1, 6, 8 * 8},   {1, 6, 1 * 8},  8, 1, 0.0f, true},
        {{2, 5, 6 * 8},   {2, 5, 3 * 8},  6, 3, 0.0f, true},
        {{1, 4, 8 * 16},  {1, 4, 2 * 16}, 8, 2, 0.5f, true},
        // Explicit [B, H, S, D] query over the BNSH cache.
        {{2, 6, 5, 8},    {2, 3, 5, 8},   6, 3, 0.0f, true},
        // H_kv == H.
        {{1, 7, 4 * 8},   {1, 7, 4 * 8},  4, 4, 0.0f, true},
        {{1, 4, 2 * 192}, {1, 4, 1 * 192}, 2, 1, 0.0f, true},
    };

    for (const auto& c : cases) {
        bool flash = false;
        NNOPS_EXPECT_TRUE(fast_vs_ref(c, flash));
    }
}

NNOPS_TEST(gqa_cache_decode_matches_ref) {
    // A single query row against a run the caller wrote earlier: the GEMV
    // route streams K and V once, and the current row is the last one.
    const Case cases[] = {
        {{1, 1, 8 * 8},    {1, 1, 1 * 8},   8, 1, 0.0f, true, false, 11, 2},
        {{1, 1, 8 * 16},   {1, 1, 4 * 16},  8, 4, 0.0f, true, false, 10, 2},
        {{1, 1, 2 * 192},  {1, 1, 1 * 192}, 2, 1, 0.0f, true, false, 6, 2},
        {{2, 4, 1, 8},     {2, 2, 1, 8},    4, 2, 0.0f, true, false, 4, 2},
        {{1, 1, 4 * 8},    {1, 1, 4 * 8},   4, 4, 0.0f, true, false, 0, 2},
        {{1, 1, 4 * 16},   {1, 1, 2 * 16},  4, 2, 0.5f, true, false, 12, 2},
        // Two kDecodeChunk blocks in one task: the online rescale.
        {{1, 1, 8 * 8},    {1, 1, 2 * 8},   8, 2, 0.0f, true, false, 2047, 2},
    };

    for (const auto& c : cases) {
        bool flash = false;
        cpu::AttentionPlan plan;
        NNOPS_EXPECT_TRUE(fast_vs_ref(c, flash, &plan));
        NNOPS_EXPECT_TRUE(plan.use_decode);
    }
}

NNOPS_TEST(gqa_cache_seqlens_matches_ref) {
    // Each batch's run length comes from kv_seqlens: the query block sits at
    // the end of that run, so batches at different positions share one call.
    const Case cases[] = {
        {{2, 6, 4 * 8}, {2, 6, 1 * 8},  4, 1, 0.0f, true, false, 3, -1, {9, 6}},
        {{2, 5, 4 * 8}, {2, 5, 2 * 8},  4, 2, 0.0f, true, false, 3, -1, {8, 5}},
        {{2, 4, 4 * 8}, {2, 4, 2 * 8},  4, 2, 0.0f, true, false, 3, -1, {7, 4}},
        {{2, 4, 3, 8},  {2, 2, 4, 8},   4, 2, 0.0f, true, false, 2, -1, {6, 4}},
        {{2, 1, 8 * 8}, {2, 1, 1 * 8},  8, 1, 0.0f, true, false, 11, 2, {12, 6}},
    };

    for (const auto& c : cases) {
        bool flash = false;
        NNOPS_EXPECT_TRUE(fast_vs_ref(c, flash));
        (void)flash;
    }
}

NNOPS_TEST(gqa_cache_flash_matches_ref) {
    // The flash route bounds its KV stream by the batch's run.
    const Case cases[] = {
        {{1, 256, 8 * 64}, {1, 256, 1 * 64}, 8, 1, 0.0f, true},
        {{1, 320, 4 * 32}, {1, 320, 2 * 32}, 4, 2, 0.0f, true},
        {{1, 256, 2 * 128}, {1, 256, 1 * 128}, 2, 1, 0.0f, true},
        // Per-batch runs on the flash route.
        {{2, 256, 4 * 64}, {2, 256, 2 * 64}, 4, 2, 0.0f, true, false, 44, -1, {300, 256}},
    };

    for (const auto& c : cases) {
        bool flash = false;
        const bool ok = fast_vs_ref(c, flash);
        NNOPS_EXPECT_TRUE(flash);
        NNOPS_EXPECT_TRUE(ok);
    }
}

NNOPS_TEST(gqa_cache_decode_split_matches_ref) {
    // A long decode run hands the head-batched plan fewer tasks than workers,
    // so it cuts the run into KV slices and merges the partial states.
    SimplePool pool(8);
    const Case cases[] = {
        {{1, 1, 8 * 64},   {1, 1, 2 * 64}, 8, 2, 0.0f, true, false, 2047, 2},
        {{1, 1, 16 * 32},  {1, 1, 1 * 32}, 16, 1, 0.0f, true, false, 1499, 2},
        {{1, 1, 8 * 32},   {1, 1, 2 * 32}, 8, 2, 0.0f, true, false, 1000, 2},
        // Per-batch runs with a split.
        {{2, 1, 8 * 32},   {2, 1, 2 * 32}, 8, 2, 0.0f, true, false, 999, 2, {1000, 700}},
    };

    for (const auto& c : cases) {
        bool flash = false;
        cpu::AttentionPlan plan;
        NNOPS_EXPECT_TRUE(fast_vs_ref(c, flash, &plan, &pool.cpu));
        NNOPS_EXPECT_TRUE(plan.use_decode);
        NNOPS_EXPECT_TRUE(plan.decode_splits > 1);
    }
}

// ============================================================
// int8 KV cache
// ============================================================

NNOPS_TEST(gqa_int8_cache_matches_ref) {
    // The cache stores symmetric integers and the external scale comes back on
    // read: per tensor here, across the standard, flash and decode routes.
    const Case cases[] = {
        // Standard, prefill.
        {{1, 6, 8 * 8},    {1, 6, 1 * 8},   8, 1, 0.0f, true, false, 0, 0, {}, 0, 0, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, true},
        // Decode over a written run.
        {{1, 1, 8 * 16},   {1, 1, 4 * 16},  8, 4, 0.0f, true, false, 10, 2, {}, 0, 0, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, true},
        // Flash.
        {{1, 256, 2 * 64}, {1, 256, 1 * 64}, 2, 1, 0.0f, true, false, 0, 1, {}, 0, 0, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, true},
        // Per-batch runs.
        {{2, 4, 4 * 8},    {2, 4, 2 * 8},   4, 2, 0.0f, true, false, 2, -1, {6, 4}, 0, 0, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, true},
    };

    for (const auto& c : cases) {
        bool flash = false;
        NNOPS_EXPECT_TRUE(fast_vs_ref(c, flash));
    }
}

NNOPS_TEST(gqa_int8_per_tensor_matches_ref) {
    // One scale for the whole cache, the other supported granularity.
    const Case cases[] = {
        {{1, 6, 8 * 8},    {1, 6, 1 * 8},  8, 1, 0.0f, true, false, 0, 0, {}, 0, 0, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, true, true},
        {{1, 1, 4 * 16},   {1, 1, 2 * 16}, 4, 2, 0.0f, true, false, 9, 2, {}, 0, 0, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, true, true},
        {{1, 256, 2 * 64}, {1, 256, 1 * 64}, 2, 1, 0.0f, true, false, 0, 1, {}, 0, 0, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, true, true},
    };

    for (const auto& c : cases) {
        bool flash = false;
        NNOPS_EXPECT_TRUE(fast_vs_ref(c, flash));
    }
}

// ============================================================
// Logit softcap and sliding window (cache form)
// ============================================================

NNOPS_TEST(gqa_softcap_matches_ref) {
    const Case cases[] = {
        // Standard route.
        {{1, 6, 4 * 8},   {1, 6, 2 * 8},   4, 2, 0.0f, true, false, 0, 0, {}, 2.0f},
        // Flash route.
        {{1, 256, 2 * 64}, {1, 256, 1 * 64}, 2, 1, 0.0f, true, false, 0, 1, {}, 1.5f},
        // Decode: the scale folds into the cap pass.
        {{2, 1, 8 * 8},   {2, 1, 2 * 8},   8, 2, 0.0f, true, false, 299, 2, {}, 3.0f},
        // A run that starts before the step's block.
        {{1, 4, 4 * 16},  {1, 4, 2 * 16},  4, 2, 0.0f, true, false, 61, -1, {}, 2.0f},
    };

    for (const auto& c : cases) {
        bool flash = false;
        NNOPS_EXPECT_TRUE(fast_vs_ref(c, flash));
    }
}

NNOPS_TEST(gqa_window_matches_ref) {
    // Row i's valid range is [pos_i - window + 1, pos_i]: the window raises the
    // lower bound instead of lowering the upper one.
    const Case cases[] = {
        {{1, 8, 4 * 8},    {1, 8, 2 * 8},    4, 2, 0.0f, true, false, 0, 0, {}, 0.0f, 3},
        {{1, 5, 4 * 8},    {1, 5, 2 * 8},    4, 2, 0.0f, true, false, 0, 0, {}, 0.0f, 64},
        {{1, 1, 8 * 8},    {1, 1, 1 * 8},    8, 1, 0.0f, true, false, 499, 2, {}, 0.0f, 16},
        {{1, 4, 4 * 16},   {1, 4, 2 * 16},   4, 2, 0.0f, true, false, 61, -1, {}, 0.0f, 8},
        {{1, 256, 2 * 64}, {1, 256, 1 * 64}, 2, 1, 0.0f, true, false, 0, 1, {}, 0.0f, 100},
        {{2, 6, 4 * 8},    {2, 6, 1 * 8},    4, 1, 0.0f, true, false, 3, 0, {9, 6}, 0.0f, 3},
        {{1, 8, 4 * 8},    {1, 8, 2 * 8},    4, 2, 0.0f, true, false, 0, 0, {}, 2.0f, 5},
    };

    for (const auto& c : cases) {
        bool flash = false;
        NNOPS_EXPECT_TRUE(fast_vs_ref(c, flash));
    }
}

NNOPS_TEST(gqa_window_sink_matches_ref) {
    // Streaming sinks: the first `sink` cache rows stay visible outside the
    // sliding window, on every route (Gemma-3-style local layers keep 4).
    const Case cases[] = {
        // Standard: 8 query rows over an 8-row run, window 3, sinks 4.
        {{1, 8, 4 * 8}, {1, 8, 2 * 8}, 4, 2, 0.0f, true, false, 0, 0, {}, 0.0f, 3, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, false, false, 4},
        // Flash over a long cache.
        {{1, 256, 2 * 64}, {1, 256, 1 * 64}, 2, 1, 0.0f, true, false, 200, 1, {}, 0.0f, 64,
         false, EpilogueActivateType::None, AttentionHeadLayout::Auto, false, false, 4},
        // Decode over a long cache.
        {{1, 1, 8 * 8}, {1, 1, 2 * 8}, 8, 2, 0.0f, true, false, 500, 2, {}, 0.0f, 16, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, false, false, 4},
        // Per-batch run lengths.
        {{2, 6, 4 * 8}, {2, 6, 1 * 8}, 4, 1, 0.0f, true, false, 3, 0, {9, 7}, 0.0f, 4, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, false, false, 4},
        // Sinks wider than the window.
        {{1, 6, 4 * 8}, {1, 6, 2 * 8}, 4, 2, 0.0f, true, false, 2, -1, {}, 0.0f, 2, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, false, false, 5},
        // Sinks + softcap.
        {{1, 6, 4 * 8}, {1, 6, 2 * 8}, 4, 2, 0.0f, true, false, 3, 0, {}, 2.0f, 3, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, false, false, 4},
    };

    for (const auto& c : cases) {
        bool flash = false;
        NNOPS_EXPECT_TRUE(fast_vs_ref(c, flash));
    }
}

NNOPS_TEST(gqa_position_ids_matches_ref) {
    // Per-batch first-row positions, the same int64 [B] input RoPE takes: the
    // derived bottom-right value, a block that starts elsewhere (its causal
    // limit clamps to the run), and a visible prefix shorter than the chunk.
    const Case cases[] = {
        // Equal to kv_seqlens - Sq: identical to the derived path.
        {{2, 4, 4 * 8}, {2, 4, 2 * 8}, 4, 2, 0.0f, true, false, 4, 0, {}, 0.0f, 0, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, false, false, 0, {4, 4}},
        // The block sits one position later than the cache tail.
        {{2, 4, 4 * 8}, {2, 4, 2 * 8}, 4, 2, 0.0f, true, false, 4, 0, {}, 0.0f, 0, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, false, false, 0, {5, 4}},
        // Visible prefix shorter than the query block: the draft rows are in
        // the cache, only the committed prefix is visible.
        {{2, 4, 4 * 8}, {2, 4, 2 * 8}, 4, 2, 0.0f, true, false, 4, 0, {6, 4}, 0.0f, 0, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, false, false, 0, {4, 2}},
        // Decode, per batch: the first row is the cache tail for batch 0 and
        // one behind it for batch 1.
        {{2, 1, 8 * 8}, {2, 1, 2 * 8}, 8, 2, 0.0f, true, false, 6, 2, {}, 0.0f, 0, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, false, false, 0, {6, 5}},
        // Window measured from the position, not from the cache row.
        {{1, 8, 4 * 8}, {1, 8, 2 * 8}, 4, 2, 0.0f, true, false, 0, -1, {}, 0.0f, 3, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, false, false, 0, {2}},
        // Softcap rides the same positions.
        {{1, 6, 4 * 8}, {1, 6, 2 * 8}, 4, 2, 0.0f, true, false, 2, 0, {}, 2.0f, 3, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, false, false, 0, {4}},
    };

    for (const auto& c : cases) {
        bool flash = false;
        NNOPS_EXPECT_TRUE(fast_vs_ref(c, flash));
    }
}

// ============================================================
// f16
// ============================================================

NNOPS_TEST(gqa_f16_matches_ref) {
    const Case cases[] = {
        // Explicit standard, explicit flash.
        {{1, 6, 8 * 8},    {1, 6, 1 * 8},  8, 1, 0.0f, false},
        {{1, 256, 8 * 64}, {1, 256, 1 * 64}, 8, 1, 0.0f, false},
        // Cache standard, cache flash, cache decode.
        {{2, 5, 6 * 16},   {2, 5, 3 * 16}, 6, 3, 0.0f, true},
        {{1, 256, 8 * 64}, {1, 256, 1 * 64}, 8, 1, 0.0f, true},
        {{1, 1, 4 * 16},   {1, 1, 2 * 16}, 4, 2, 0.0f, true, false, 9, 2},
        // f16 query over an int8 cache.
        {{1, 6, 4 * 8},    {1, 6, 2 * 8},  4, 2, 0.0f, true, false, 0, 0, {}, 0, 0, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, true},
    };

    const float tol = test::kF16AccumTol;
    for (const auto& c : cases) {
        Built b = build(c, 300);
        NNOPS_EXPECT_TRUE(b.ok);
        auto op = GroupQueryAttention::create(b.attrs, Backend::CPU);

        const auto out_descs = op->getOutputTensorDesc(b.descs);
        const int64_t n = out_descs[0].numel();

        std::vector<float> ref_buf(static_cast<size_t>(n));
        auto ref_out = test::make_channel_first(out_descs[0], ref_buf.data());
        nnops::backend::cpu::reference::group_query_attention_ref(b.attrs, ref_out, b.ins, {},
                                                                  nullptr);

        // Convert every f32 activation and metadata view to f16 by hand: the
        // cache keeps its dtype, the query/output/mask move to half.
        std::vector<std::vector<cpu::half>> half_bufs;
        std::vector<TensorView> ins16 = b.ins;
        for (size_t i = 0; i < b.ins.size(); ++i) {
            const TensorView& t = b.ins[i];
            if (t.is_empty() || t.data_type() != DataType::f32 ||
                (c.causal && (i == 5 || i == 6))) {
                continue;
            }
            const int64_t elems = t.numel();
            std::vector<cpu::half> h(static_cast<size_t>(elems));
            const float* src = t.ptr<float>();
            for (int64_t e = 0; e < elems; ++e) {
                h[static_cast<size_t>(e)] = cpu::float_to_half(src[e]);
            }
            half_bufs.push_back(std::move(h));
            ins16[i] = TensorView(t.shape_span(), DataType::f16, half_bufs.back().data(),
                                  t.layout());
        }
        std::vector<TensorDesc> descs16;
        for (const auto& t : ins16) { descs16.push_back(t.desc()); }
        const auto out16_desc = op->getOutputTensorDesc(descs16);

        std::vector<cpu::half> got_buf(static_cast<size_t>(n));
        auto got = test::make_channel_first(out16_desc[0], got_buf.data());
        NNOPS_EXPECT_TRUE(op->compute(got, ins16) == Status::ok);

        const auto got_f32 = test::f16_to_f32(got_buf);
        float worst = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            worst = std::max(worst, std::abs(got_f32[static_cast<size_t>(i)] -
                                             ref_buf[static_cast<size_t>(i)]));
        }
        if (worst > tol) {
            std::printf("        f16 deviation %f > tol %f\n", (double)worst, (double)tol);
        }
        NNOPS_EXPECT_TRUE(worst <= tol);
    }
}

// ============================================================
// Packed varlen (cu_seqlens), explicit form
// ============================================================

namespace {

/// One packed varlen call over [1, total, W] tensors, each segment run as its
/// own [1, len, W] problem.
struct VarlenCase {
    std::vector<int32_t> cu_q;
    std::vector<int32_t> cu_k;
    int64_t heads = 4;
    int64_t kv_heads = 2;
    int64_t dim = 8;
    int route = -1;        // per segment; -1 unchecked
    float softcap = 0.0f;
    int64_t window = 0;
    EpilogueActivateType gate = EpilogueActivateType::None;
};

bool varlen_vs_ref(const VarlenCase& c)
{
    const int64_t total_q = c.cu_q.back();
    const int64_t total_k = c.cu_k.back();
    auto [q_vec, q] = test::make_random_tensor({1, total_q, c.heads * c.dim},
                                               -1.0f, 1.0f, 700);
    auto [k_vec, k] = test::make_random_tensor({1, total_k, c.kv_heads * c.dim},
                                               -1.0f, 1.0f, 701);
    auto [v_vec, v] = test::make_random_tensor({1, total_k, c.kv_heads * c.dim},
                                               -1.0f, 1.0f, 702);
    (void)q_vec; (void)k_vec; (void)v_vec;

    std::vector<int32_t> qo(c.cu_q), ko(c.cu_k);
    const int64_t nseg[1] = {static_cast<int64_t>(qo.size())};
    TensorView cu_q_view(nseg, DataType::s32, qo.data());
    TensorView cu_k_view(nseg, DataType::s32, ko.data());

    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = c.heads;
    attrs.kv_num_heads = c.kv_heads;
    attrs.causal = false;
    attrs.softcap = c.softcap;
    attrs.local_window_size = c.window > 0 ? c.window : -1;
    attrs.output_gate = c.gate;

    auto op = GroupQueryAttention::create(attrs, Backend::CPU);

    std::vector<TensorView> ins(9, TensorView{});
    ins[0] = q;
    ins[1] = k;
    ins[2] = v;
    ins[6] = cu_q_view;
    ins[7] = cu_k_view;
    std::vector<float> g_vec;
    if (c.gate != EpilogueActivateType::None) {
        auto [gv, gt] = test::make_random_tensor(
            {1, total_q, c.heads * c.dim}, -3.0f, 3.0f, 703);
        g_vec = std::move(gv);
        ins[8] = gt;
    }
    std::vector<TensorDesc> descs;
    descs.reserve(ins.size());
    for (const auto& t : ins) { descs.push_back(t.desc()); }

    const TensorDesc out_desc = td({1, total_q, c.heads * c.dim});
    const TensorDesc out_descs[] = {out_desc};
    if (op->validate(descs, out_descs) != Status::ok) {
        std::printf("        varlen validate rejected the call\n");
        return false;
    }

    // Every segment must take the route the case asked for.
    if (c.route >= 0) {
        for (size_t s = 0; s + 1 < qo.size(); ++s) {
            const int64_t lq = qo[s + 1] - qo[s];
            if (lq == 0) { continue; }
            const int64_t lk = ko[s + 1] - ko[s];
            cpu::AttentionConfig pcfg;
            pcfg.num_heads = c.heads;
            pcfg.num_kv_heads = c.kv_heads;
            pcfg.softcap = c.softcap;
            pcfg.window = c.window;
            const TensorDesc sd[] = {td({1, lq, c.heads * c.dim}),
                                     td({1, lk, c.kv_heads * c.dim}),
                                     td({1, lk, c.kv_heads * c.dim})};
            const TensorDesc so[] = {td({1, lq, c.heads * c.dim})};
            const cpu::AttentionPlan plan = cpu::get_attention_plan(pcfg, sd, so, 1, false);
            const int got = plan.use_decode ? 2 : (plan.use_flash ? 1 : 0);
            if (got != c.route) {
                std::printf("        segment %zu route mismatch: want %d, got %d\n",
                            s, c.route, got);
                return false;
            }
        }
    }

    const int64_t n = out_desc.numel();
    std::vector<float> fast_buf(static_cast<size_t>(n));
    auto fast_out = test::make_channel_first(out_desc, fast_buf.data());
    ComputeContext ctx;
    if (op->compute(fast_out, ins, ctx) != Status::ok) {
        std::printf("        varlen compute failed\n");
        return false;
    }

    std::vector<float> ref_buf(static_cast<size_t>(n));
    auto ref_out = test::make_channel_first(out_desc, ref_buf.data());
    nnops::backend::cpu::reference::group_query_attention_ref(attrs, ref_out, ins, {},
                                                              nullptr);

    for (int64_t i = 0; i < n; ++i) {
        const float diff = std::abs(fast_buf[static_cast<size_t>(i)] -
                                    ref_buf[static_cast<size_t>(i)]);
        const float thr = 1e-4f + 1e-3f * std::max(std::abs(fast_buf[static_cast<size_t>(i)]),
                                                   std::abs(ref_buf[static_cast<size_t>(i)]));
        if (diff > thr) {
            std::printf("        varlen mismatch at %lld: fast %.6f vs ref %.6f\n",
                        (long long)i, (double)fast_buf[static_cast<size_t>(i)],
                        (double)ref_buf[static_cast<size_t>(i)]);
            return false;
        }
    }
    return true;
}

} // anonymous namespace

NNOPS_TEST(gqa_varlen_matches_ref) {
    // Packed segments are independent documents: no query reads a key from a
    // neighbouring segment.
    const VarlenCase cases[] = {
        {{0, 5, 6, 10}, {0, 6, 7, 15}, 4, 2, 8, -1},
        {{0, 4, 9},     {0, 4, 12},    4, 1, 16, 0},
        {{0, 0, 3},     {0, 4, 7},     4, 2, 8, 0},
        {{0, 256, 512}, {0, 300, 600}, 4, 2, 64, 1},
        {{0, 1, 2},     {0, 300, 500}, 8, 1, 8, 2},
        {{0, 6, 9},     {0, 6, 12},    4, 2, 8, 0, 2.0f, 3},
        {{0, 3, 7},     {0, 3, 9},     2, 1, 24, 0},
        {{0, 5, 6, 10}, {0, 6, 7, 15}, 4, 2, 8, -1, 0.0f, 0,
         EpilogueActivateType::Sigmoid},
    };

    for (const auto& c : cases) {
        NNOPS_EXPECT_TRUE(varlen_vs_ref(c));
    }
}

NNOPS_TEST(gqa_varlen_validation) {
    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = 4;
    attrs.kv_num_heads = 2;
    attrs.causal = false;
    auto op = GroupQueryAttention::create(attrs, Backend::CPU);

    const TensorDesc q = td({1, 10, 4 * 8});
    const TensorDesc k = td({1, 15, 2 * 8});
    const TensorDesc v = td({1, 15, 2 * 8});
    const TensorDesc out = td({1, 10, 4 * 8});
    TensorDesc cu;
    cu.rank = 1;
    cu.dims = {4};
    cu.dtype = DataType::s32;

    const TensorDesc outs[] = {out};
    const auto packed = [&](const TensorDesc& qq, const TensorDesc& kk, const TensorDesc& vv) {
        SlotDescs s;
        s.put(0, qq).put(1, kk).put(2, vv).put(6, cu).put(7, cu);
        return s.d;
    };
    NNOPS_EXPECT_TRUE(op->validate(packed(q, k, v), outs) == Status::ok);
    {
        SlotDescs s;
        s.put(0, q).put(1, k).put(2, v).put(6, cu);
        NNOPS_EXPECT_TRUE(op->validate(s.d, outs) == Status::invalid_argument);
    }
    {
        TensorDesc cu_k = cu;
        cu_k.dims = {3};
        SlotDescs s;
        s.put(0, q).put(1, k).put(2, v).put(6, cu).put(7, cu_k);
        NNOPS_EXPECT_TRUE(op->validate(s.d, outs) == Status::invalid_argument);
    }
    {
        TensorDesc one = cu;
        one.dims = {1};
        SlotDescs s;
        s.put(0, q).put(1, k).put(2, v).put(6, one).put(7, one);
        NNOPS_EXPECT_TRUE(op->validate(s.d, outs) == Status::invalid_argument);
    }
    {
        TensorDesc wrong_dtype = cu;
        wrong_dtype.dtype = DataType::f32;
        SlotDescs s;
        s.put(0, q).put(1, k).put(2, v).put(6, wrong_dtype).put(7, cu);
        NNOPS_EXPECT_TRUE(op->validate(s.d, outs) == Status::invalid_argument);
    }
    {
        // A shared bias does not apply across documents.
        SlotDescs s;
        s.put(0, q).put(1, k).put(2, v).put(3, td({10, 15})).put(6, cu).put(7, cu);
        NNOPS_EXPECT_TRUE(op->validate(s.d, outs) == Status::invalid_argument);
    }
    {
        // A packed batch is merged only.
        const TensorDesc q4 = td({1, 4, 10, 8});
        const TensorDesc k4 = td({1, 2, 15, 8});
        NNOPS_EXPECT_TRUE(op->validate(packed(q4, k4, k4), outs) ==
                          Status::invalid_argument);
    }
}

// ============================================================
// Output gate (Qwen3-Next / 3.5 / 4)
// ============================================================

NNOPS_TEST(gqa_output_gate_hand) {
    // The gate is one pass over the finished output, so the gated call is
    // exactly plain * act(gate) element for element.
    const std::vector<int64_t> qshape = {2, 3, 2 * 8};
    const std::vector<int64_t> kshape = {2, 5, 1 * 8};

    Case plain_case;
    plain_case.qshape = qshape;
    plain_case.kshape = kshape;
    plain_case.heads = 2;
    plain_case.kv_heads = 1;
    plain_case.causal = false;
    Built plain_b = build(plain_case, 200);
    NNOPS_EXPECT_TRUE(plain_b.ok);

    Case gate_case = plain_case;
    gate_case.gate = EpilogueActivateType::Sigmoid;
    Built gate_b = build(gate_case, 200);
    NNOPS_EXPECT_TRUE(gate_b.ok);

    auto op = GroupQueryAttention::create(plain_b.attrs, Backend::CPU);
    const auto out_descs = op->getOutputTensorDesc(plain_b.descs);
    const int64_t n = out_descs[0].numel();
    std::vector<float> plain(static_cast<size_t>(n));
    auto plain_out = test::make_channel_first(out_descs[0], plain.data());
    NNOPS_EXPECT_TRUE(op->compute(plain_out, plain_b.ins) == Status::ok);

    auto gop = GroupQueryAttention::create(gate_b.attrs, Backend::CPU);
    std::vector<float> got(static_cast<size_t>(n));
    auto got_out = test::make_channel_first(out_descs[0], got.data());
    NNOPS_EXPECT_TRUE(gop->compute(got_out, gate_b.ins) == Status::ok);

    const float* gate = gate_b.ins[8].ptr<float>();
    for (int64_t i = 0; i < n; ++i) {
        const float want = plain[static_cast<size_t>(i)] / (1.0f + std::exp(-gate[i]));
        NNOPS_EXPECT_NEAR(got[static_cast<size_t>(i)], want, 1e-5f);
    }
}

NNOPS_TEST(gqa_output_gate_matches_ref) {
    const EpilogueActivateType acts[] = {
        EpilogueActivateType::Sigmoid,
        EpilogueActivateType::Silu,
        EpilogueActivateType::Gelu,
        EpilogueActivateType::Tanh,
        EpilogueActivateType::Relu,
    };
    for (const auto act : acts) {
        Case e = {};
        e.qshape = {2, 4, 4 * 8};
        e.kshape = {2, 7, 2 * 8};
        e.heads = 4;
        e.kv_heads = 2;
        e.causal = false;
        e.gate = act;
        bool flash = false;
        NNOPS_EXPECT_TRUE(fast_vs_ref(e, flash));

        Case k = {};
        k.qshape = {1, 4, 4 * 8};
        k.kshape = {1, 4, 2 * 8};
        k.heads = 4;
        k.kv_heads = 2;
        k.causal = true;
        k.gate = act;
        NNOPS_EXPECT_TRUE(fast_vs_ref(k, flash));

        Case d = {};
        d.qshape = {1, 1, 4 * 8};
        d.kshape = {1, 1, 2 * 8};
        d.heads = 4;
        d.kv_heads = 2;
        d.causal = true;
        d.past = 11;
        d.gate = act;
        NNOPS_EXPECT_TRUE(fast_vs_ref(d, flash));

        Case f = {};
        f.qshape = {1, 256, 2 * 64};
        f.kshape = {1, 256, 1 * 64};
        f.heads = 2;
        f.kv_heads = 1;
        f.causal = true;
        f.gate = act;
        NNOPS_EXPECT_TRUE(fast_vs_ref(f, flash));
    }
}

NNOPS_TEST(gqa_output_gate_validation) {
    // The gate is the trailing slot, present exactly when the attribute names
    // an activation, and shaped like Q (the output).
    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = 4;
    attrs.kv_num_heads = 2;
    attrs.causal = false;
    attrs.output_gate = EpilogueActivateType::Sigmoid;
    auto op = GroupQueryAttention::create(attrs, Backend::CPU);

    const TensorDesc q = td({1, 5, 4 * 8});
    const TensorDesc k = td({1, 9, 2 * 8});
    const TensorDesc out = td({1, 5, 4 * 8});
    const TensorDesc outs[] = {out};
    const auto with_gate = [&](const TensorDesc& g) {
        SlotDescs s;
        s.put(0, q).put(1, k).put(2, k).put(8, g);
        return s.d;
    };
    NNOPS_EXPECT_TRUE(op->validate(with_gate(q), outs) == Status::ok);
    {
        const TensorDesc ins[] = {q, k, k};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::invalid_argument);
    }
    NNOPS_EXPECT_TRUE(op->validate(with_gate(td({1, 5, 4 * 7})), outs) ==
                      Status::invalid_argument);
    {
        TensorDesc g = q;
        g.dtype = DataType::f16;
        NNOPS_EXPECT_TRUE(op->validate(with_gate(g), outs) == Status::invalid_argument);
    }
    {
        GroupQueryAttentionAttributes plain;
        plain.num_heads = 4;
        plain.kv_num_heads = 2;
        plain.causal = false;
        auto plain_op = GroupQueryAttention::create(plain, Backend::CPU);
        NNOPS_EXPECT_TRUE(plain_op->validate(with_gate(q), outs) ==
                          Status::invalid_argument);
    }
    {
        GroupQueryAttentionAttributes bad = attrs;
        bad.output_gate = EpilogueActivateType::LeakyRelu;
        auto bad_op = GroupQueryAttention::create(bad, Backend::CPU);
        NNOPS_EXPECT_TRUE(bad_op->validate(with_gate(q), outs) == Status::unsupported);
    }
}

// ============================================================
// Layouts, broadcast bias and the slot contract
// ============================================================

NNOPS_TEST(gqa_bshd_matches_ref) {
    // rank-4 [B, S, H, D] Q/K/V in the explicit form, and a BSHD query over
    // the BNSH cache in the cache form.
    Case e = {};
    e.qshape = {2, 3, 4, 8};
    e.kshape = {2, 5, 2, 8};
    e.heads = 4;
    e.kv_heads = 2;
    e.causal = false;
    e.layout = AttentionHeadLayout::BSHD;
    {
        bool flash = false;
        NNOPS_EXPECT_TRUE(fast_vs_ref(e, flash));
    }

    Case k = {};
    k.qshape = {2, 3, 4, 8};
    k.kshape = {2, 5, 2, 8};
    k.heads = 4;
    k.kv_heads = 2;
    k.causal = true;
    k.layout = AttentionHeadLayout::BSHD;
    {
        bool flash = false;
        NNOPS_EXPECT_TRUE(fast_vs_ref(k, flash));
    }
}

NNOPS_TEST(gqa_bias_broadcast_matches_ref) {
    // attention_bias admits a batch or head extent of 1, exactly as
    // onnxruntime's shape inference declares it.
    const int64_t B = 2, H = 4, Hkv = 2, Sq = 3, Sk = 5, D = 8;

    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = H;
    attrs.kv_num_heads = Hkv;
    attrs.causal = false;
    auto op = GroupQueryAttention::create(attrs, Backend::CPU);

    const std::vector<std::vector<int64_t>> shapes = {
        {1, H, Sq, Sk}, {B, 1, Sq, Sk}, {1, 1, Sq, Sk}, {B, H, Sq, Sk}, {Sq, Sk}};
    for (const auto& ms : shapes) {
        auto [qv, q] = test::make_random_tensor({B, Sq, H * D}, -1.0f, 1.0f, 320);
        auto [kv, k] = test::make_random_tensor({B, Sk, Hkv * D}, -1.0f, 1.0f, 321);
        auto [vv, v] = test::make_random_tensor({B, Sk, Hkv * D}, -1.0f, 1.0f, 322);
        auto [mv, m] = test::make_random_tensor(ms, -2.0f, 2.0f, 323);

        std::vector<TensorView> ins(9, TensorView{});
        ins[0] = q;
        ins[1] = k;
        ins[2] = v;
        ins[3] = m;
        std::vector<TensorDesc> descs;
        for (const auto& t : ins) { descs.push_back(t.desc()); }
        const auto out_descs = op->getOutputTensorDesc(descs);
        std::vector<float> got(static_cast<size_t>(out_descs[0].numel()));
        auto got_out = test::make_channel_first(out_descs[0], got.data());
        ComputeContext ctx;
        NNOPS_EXPECT_TRUE(op->compute(got_out, ins, ctx) == Status::ok);

        std::vector<float> want(got.size());
        auto want_out = test::make_channel_first(out_descs[0], want.data());
        nnops::backend::cpu::reference::group_query_attention_ref(attrs, want_out, ins, {},
                                                                  nullptr);
        for (size_t i = 0; i < got.size(); ++i) { NNOPS_EXPECT_NEAR(got[i], want[i], 1e-4f); }
    }
}

NNOPS_TEST(gqa_key_padding_mask_matches_ref) {
    // The int32 keep-mask forms: one row per batch, one per query row, and the
    // rank-1 length vector.
    const int64_t B = 2, H = 4, Hkv = 2, Sq = 3, Sk = 6, D = 8;

    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = H;
    attrs.kv_num_heads = Hkv;
    attrs.causal = false;
    auto op = GroupQueryAttention::create(attrs, Backend::CPU);

    for (int form = 0; form < 3; ++form) {
        auto [qv, q] = test::make_random_tensor({B, Sq, H * D}, -1.0f, 1.0f, 330);
        auto [kv, k] = test::make_random_tensor({B, Sk, Hkv * D}, -1.0f, 1.0f, 331);
        auto [vv, v] = test::make_random_tensor({B, Sk, Hkv * D}, -1.0f, 1.0f, 332);
        (void)qv; (void)kv; (void)vv;

        std::vector<int32_t> mask;
        std::vector<int64_t> shape;
        if (form == 0) {
            shape = {B, Sk};
            mask.assign(static_cast<size_t>(B * Sk), 1);
            for (int64_t b = 0; b < B; ++b) {
                for (int64_t j = 0; j < Sk; ++j) {
                    mask[static_cast<size_t>(b * Sk + j)] = (j + b) % 3 == 0 ? 0 : 1;
                }
            }
        } else if (form == 1) {
            shape = {B, Sq, Sk};
            mask.assign(static_cast<size_t>(B * Sq * Sk), 1);
            for (int64_t b = 0; b < B; ++b) {
                for (int64_t i = 0; i < Sq; ++i) {
                    for (int64_t j = 0; j < Sk; ++j) {
                        mask[static_cast<size_t>((b * Sq + i) * Sk + j)] =
                            (j > i + b) ? 0 : 1;
                    }
                }
            }
        } else {
            shape = {B};
            mask = {6, 4};
        }
        TensorView mv(std::span<const int64_t>(shape), DataType::s32, mask.data());

        std::vector<TensorView> ins(9, TensorView{});
        ins[0] = q;
        ins[1] = k;
        ins[2] = v;
        ins[4] = mv;
        std::vector<TensorDesc> descs;
        for (const auto& t : ins) { descs.push_back(t.desc()); }
        const auto out_descs = op->getOutputTensorDesc(descs);
        std::vector<float> got(static_cast<size_t>(out_descs[0].numel()));
        auto got_out = test::make_channel_first(out_descs[0], got.data());
        ComputeContext ctx;
        NNOPS_EXPECT_TRUE(op->compute(got_out, ins, ctx) == Status::ok);

        std::vector<float> want(got.size());
        auto want_out = test::make_channel_first(out_descs[0], want.data());
        nnops::backend::cpu::reference::group_query_attention_ref(attrs, want_out, ins, {},
                                                                  nullptr);
        for (size_t i = 0; i < got.size(); ++i) { NNOPS_EXPECT_NEAR(got[i], want[i], 1e-4f); }
    }
}

NNOPS_TEST(gqa_slot_contract) {
    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = 4;
    attrs.kv_num_heads = 2;
    attrs.causal = false;
    auto op = GroupQueryAttention::create(attrs, Backend::CPU);

    const TensorDesc q = td({2, 3, 4 * 8});
    const TensorDesc k = td({2, 5, 2 * 8});
    const TensorDesc out = td({2, 3, 4 * 8});
    const TensorDesc outs[] = {out};
    {
        SlotDescs s;
        s.put(0, q).put(1, k).put(2, k).put(3, td({3, 3}));   // bias of the wrong shape
        NNOPS_EXPECT_TRUE(op->validate(s.d, outs) == Status::invalid_argument);
    }
    {
        // A second output is not part of the contract any more.
        SlotDescs s;
        s.put(0, q).put(1, k).put(2, k);
        const TensorDesc o2[] = {out, td({2, 2, 5, 8})};
        NNOPS_EXPECT_TRUE(op->validate(s.d, o2) == Status::invalid_argument);
    }
    {
        SlotDescs s;
        s.put(0, q).put(1, k).put(2, k);
        NNOPS_EXPECT_TRUE(op->validate(s.d, outs) == Status::ok);
    }
    {
        // A rank-1 key_padding_mask is one length per batch.
        TensorDesc sl;
        sl.rank = 1;
        sl.dims = {3};
        sl.dtype = DataType::s32;
        SlotDescs s;
        s.put(0, q).put(1, k).put(2, k).put(4, sl);
        NNOPS_EXPECT_TRUE(op->validate(s.d, outs) == Status::invalid_argument);
    }
}

NNOPS_TEST(gqa_cache_validation) {
    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = 4;
    attrs.kv_num_heads = 2;
    attrs.causal = true;
    auto op = GroupQueryAttention::create(attrs, Backend::CPU);

    const TensorDesc q = td({2, 2, 4 * 8});
    const TensorDesc ck = td({2, 2, 6, 8});
    const TensorDesc out = td({2, 2, 4 * 8});
    const TensorDesc outs[] = {out};
    TensorDesc sl;
    sl.rank = 1;
    sl.dims = {2};
    sl.dtype = DataType::s32;
    TensorDesc one;
    one.rank = 1;
    one.dims = {1};
    one.dtype = DataType::f32;

    const auto base = [&](SlotDescs& s) { s.put(0, q).put(1, ck).put(2, ck); };
    {
        SlotDescs s;
        base(s);
        s.put(3, sl);
        NNOPS_EXPECT_TRUE(op->validate(s.d, outs) == Status::ok);
    }
    {
        // A cache the query cannot fit in.
        SlotDescs s;
        s.put(0, td({2, 7, 4 * 8})).put(1, ck).put(2, ck);
        NNOPS_EXPECT_TRUE(op->validate(s.d, outs) == Status::invalid_argument);
    }
    {
        // The caches must be a matching pair of rank-4 BNSH tensors.
        SlotDescs s;
        s.put(0, q).put(1, td({2, 6, 2, 8})).put(2, ck);
        NNOPS_EXPECT_TRUE(op->validate(s.d, outs) == Status::invalid_argument);
    }
    {
        // the seqlens vector is int32 [B].
        TensorDesc bad = sl;
        bad.dtype = DataType::f32;
        SlotDescs s;
        base(s);
        s.put(3, bad);
        NNOPS_EXPECT_TRUE(op->validate(s.d, outs) == Status::invalid_argument);
    }
    {
        // An int8 cache without scales is incomplete.
        TensorDesc c8 = ck;
        c8.dtype = DataType::s8;
        SlotDescs s;
        s.put(0, q).put(1, c8).put(2, c8);
        NNOPS_EXPECT_TRUE(op->validate(s.d, outs) == Status::invalid_argument);
    }
    {
        // ... and the scales must be f32, one value or [B, H_kv, Skv].
        TensorDesc c8 = ck;
        c8.dtype = DataType::s8;
        SlotDescs s;
        s.put(0, q).put(1, c8).put(2, c8).put(5, one).put(6, one);
        NNOPS_EXPECT_TRUE(op->validate(s.d, outs) == Status::ok);
        // Per token: [B, H_kv, Skv].
        SlotDescs t;
        t.put(0, q).put(1, c8).put(2, c8).put(5, td({2, 2, 6})).put(6, one);
        NNOPS_EXPECT_TRUE(op->validate(t.d, outs) == Status::ok);
        SlotDescs u;
        u.put(0, q).put(1, c8).put(2, c8).put(5, td({2, 2, 5})).put(6, one);
        NNOPS_EXPECT_TRUE(op->validate(u.d, outs) == Status::invalid_argument);
        SlotDescs w;
        w.put(0, q).put(1, c8).put(2, c8).put(5, td({3})).put(6, one);
        NNOPS_EXPECT_TRUE(op->validate(w.d, outs) == Status::invalid_argument);
        // An fp cache carries no scales.
        SlotDescs v;
        base(v);
        v.put(5, one).put(6, one);
        NNOPS_EXPECT_TRUE(op->validate(v.d, outs) == Status::invalid_argument);
    }
    {
        // position_ids is int64 [B].
        TensorDesc pos;
        pos.rank = 1;
        pos.dims = {2};
        pos.dtype = DataType::s64;
        SlotDescs s;
        base(s);
        s.put(4, pos);
        NNOPS_EXPECT_TRUE(op->validate(s.d, outs) == Status::ok);
        TensorDesc bad = pos;
        bad.dtype = DataType::s32;
        SlotDescs t;
        base(t);
        t.put(4, bad);
        NNOPS_EXPECT_TRUE(op->validate(t.d, outs) == Status::invalid_argument);
        TensorDesc shortv = pos;
        shortv.dims = {3};
        SlotDescs u;
        base(u);
        u.put(4, shortv);
        NNOPS_EXPECT_TRUE(op->validate(u.d, outs) == Status::invalid_argument);
    }
    {
        // The cache form has no mask input.
        SlotDescs s;
        base(s);
        s.put(3, td({2, 8}));
        NNOPS_EXPECT_TRUE(op->validate(s.d, outs) == Status::invalid_argument);
    }
    {
        // head_sink sits after the scales.
        SlotDescs s;
        base(s);
        s.put(7, td({4}));
        NNOPS_EXPECT_TRUE(op->validate(s.d, outs) == Status::ok);
    }
}

// ============================================================
// H_kv == H must agree with MultiHeadAttention
// ============================================================

NNOPS_TEST(gqa_degenerate_equals_mha) {
    auto [q_vec, q] = test::make_random_tensor({1, 6, 4 * 8}, -1.0f, 1.0f, 200);
    auto [k_vec, k] = test::make_random_tensor({1, 6, 4 * 8}, -1.0f, 1.0f, 201);
    auto [v_vec, v] = test::make_random_tensor({1, 6, 4 * 8}, -1.0f, 1.0f, 202);

    MultiHeadAttentionAttributes mha_attrs;
    mha_attrs.num_heads = 4;
    mha_attrs.causal = false;

    GroupQueryAttentionAttributes gqa_attrs;
    gqa_attrs.num_heads = 4;
    gqa_attrs.kv_num_heads = 4;
    gqa_attrs.causal = false;

    const TensorDesc descs[] = {q.desc(), k.desc(), v.desc()};
    const TensorView ins[] = {q, k, v};

    auto mha = MultiHeadAttention::create(mha_attrs, Backend::CPU);
    auto gqa = GroupQueryAttention::create(gqa_attrs, Backend::CPU);

    auto mha_desc = mha->getOutputTensorDesc(descs)[0];
    auto gqa_desc = gqa->getOutputTensorDesc(descs)[0];
    NNOPS_EXPECT_EQ(mha_desc.dims[2], gqa_desc.dims[2]);

    std::vector<float> a_buf(static_cast<size_t>(mha_desc.numel()));
    std::vector<float> b_buf(static_cast<size_t>(gqa_desc.numel()));
    auto a = test::make_channel_first(mha_desc, a_buf.data());
    auto b = test::make_channel_first(gqa_desc, b_buf.data());

    mha->compute(a, ins);
    gqa->compute(b, ins);

    NNOPS_EXPECT_TRUE(test::allclose(a, b, 1e-6f, 1e-6f));
}

// ============================================================
// Attention sink (head_sink)
// ============================================================

namespace {

bool gqa_sink_case(const Case& c, uint64_t seed)
{
    Built b = build(c, seed);
    if (!b.ok) {
        return false;
    }
    auto [sv, sink] = test::make_random_tensor({c.heads}, -3.0f, 3.0f, seed + 5);
    b.keep_f32.push_back(std::move(sv));
    b.ins[c.causal ? 7 : 5] = sink;
    b.descs.clear();
    for (const auto& t : b.ins) { b.descs.push_back(t.desc()); }

    auto op = GroupQueryAttention::create(b.attrs, Backend::CPU);
    const auto out_descs = op->getOutputTensorDesc(b.descs);
    const int64_t n = out_descs[0].numel();
    std::vector<float> got(static_cast<size_t>(n));
    auto got_out = test::make_channel_first(out_descs[0], got.data());
    ComputeContext ctx;
    if (op->compute(got_out, b.ins, ctx) != Status::ok) {
        std::printf("        sink compute rejected the call\n");
        return false;
    }

    std::vector<float> want(static_cast<size_t>(n));
    auto want_out = test::make_channel_first(out_descs[0], want.data());
    nnops::backend::cpu::reference::group_query_attention_ref(b.attrs, want_out, b.ins, {},
                                                              nullptr);
    for (int64_t i = 0; i < n; ++i) {
        if (std::abs(got[static_cast<size_t>(i)] - want[static_cast<size_t>(i)]) > 1e-4f) {
            std::printf("        sink mismatch at %lld: %.6f vs %.6f\n", (long long)i,
                        (double)got[static_cast<size_t>(i)],
                        (double)want[static_cast<size_t>(i)]);
            return false;
        }
    }
    return true;
}

} // anonymous namespace

NNOPS_TEST(gqa_head_sink_matches_ref) {
    // The sink is a virtual key with value zero: it only grows the denominator.
    Case standard = {};
    standard.qshape = {2, 3, 4 * 8};
    standard.kshape = {2, 3, 2 * 8};
    standard.heads = 4;
    standard.kv_heads = 2;
    NNOPS_EXPECT_TRUE(gqa_sink_case(standard, 500));

    Case flash = {};
    flash.qshape = {1, 256, 2 * 64};
    flash.kshape = {1, 256, 1 * 64};
    flash.heads = 2;
    flash.kv_heads = 1;
    NNOPS_EXPECT_TRUE(gqa_sink_case(flash, 510));

    Case decode = {};
    decode.qshape = {1, 1, 8 * 16};
    decode.kshape = {1, 1, 2 * 16};
    decode.heads = 8;
    decode.kv_heads = 2;
    decode.past = 60;
    NNOPS_EXPECT_TRUE(gqa_sink_case(decode, 520));

    // Explicit form, with the sink in its own slot.
    Case explicit_case = {};
    explicit_case.qshape = {2, 3, 4 * 8};
    explicit_case.kshape = {2, 5, 2 * 8};
    explicit_case.heads = 4;
    explicit_case.kv_heads = 2;
    explicit_case.causal = false;
    NNOPS_EXPECT_TRUE(gqa_sink_case(explicit_case, 530));
}

NNOPS_TEST(gqa_head_sink_hand) {
    // One key, one head, D = 1: q = k = 1 with scale 1 gives score 1; the sink
    // logit 0 holds e^-1 of the weight, so the output is 3 / (1 + e^-1).
    const int64_t qshape[] = {1, 1, 1};
    const int64_t kshape[] = {1, 1, 1};
    std::vector<float> qd{1.0f}, kd{1.0f}, vd{3.0f}, sd{0.0f};
    TensorView q(qshape, DataType::f32, qd.data());
    TensorView k(kshape, DataType::f32, kd.data());
    TensorView v(kshape, DataType::f32, vd.data());
    TensorView sink(std::vector<int64_t>{1}, DataType::f32, sd.data());

    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = 1;
    attrs.kv_num_heads = 1;
    attrs.causal = false;
    attrs.scale = 1.0f;
    auto op = GroupQueryAttention::create(attrs, Backend::CPU);

    std::vector<TensorView> ins(9, TensorView{});
    ins[0] = q;
    ins[1] = k;
    ins[2] = v;
    ins[5] = sink;
    std::vector<TensorDesc> descs;
    for (const auto& t : ins) { descs.push_back(t.desc()); }
    const auto out_descs = op->getOutputTensorDesc(descs);
    std::vector<float> out_buf(1);
    auto out = test::make_channel_first(out_descs[0], out_buf.data());
    NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::ok);
    NNOPS_EXPECT_NEAR(out_buf[0], 3.0f / (1.0f + std::exp(-1.0f)), 1e-5f);
}
