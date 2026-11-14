/// @file test_group_query_attention.cpp
/// @brief GroupQueryAttention tests — the class API against the CPU reference.
///
/// reference::group_query_attention_ref is f32 only, so f16 is compared against
/// the f32 reference on the same values with the arch-aware kF16AccumTol.
///
/// The cases cover the three head ratios that matter — MQA (H_kv == 1), a
/// grouped ratio, and H_kv == H (which must match MultiHeadAttention) — in both
/// merged [B, S, H*D] and explicit [B, H, S, D] layouts, with and without the
/// causal mask, with an additive mask, with a shorter query run than the key
/// run (the KV-cache decode shape), and on both routes: standard, and flash
/// once Sq*Sk reaches kFlashMinScores (1 << 16). The step cases feed the key
/// run as [past | new] and check both the decode and the forced-flash route,
/// including a segment boundary that is not aligned to the KV tile. The
/// threaded cases drive the decode route over a long run, so the plan splits it
/// into KV slices and the partial softmax states are merged.

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

struct Case {
    std::vector<int64_t> qshape;   // [B, Sq, H*D] or [B, H, Sq, D]
    std::vector<int64_t> kshape;   // [B, Sk, Hkv*D] or [B, Hkv, Sk, D]
    int64_t heads;
    int64_t kv_heads;
    float scale = 0.0f;
    bool causal = true;
    bool mask = false;
    int64_t past = 0;              // rows in the past K/V segment
    int64_t position_offset = -1;  // absolute position of query row 0
    int route = -1;                // 0 standard, 1 flash, 2 decode; -1 unchecked
    std::vector<int32_t> seqlens;  // per-batch key-run lengths; empty = absent
    float softcap = 0.0f;          // logit cap; 0 = off
    int64_t window = 0;            // sliding window; 0 = off
    bool head_bias = false;        // mask as [B, H, Sq, Sk] instead of [Sq, Sk]
    EpilogueActivateType gate = EpilogueActivateType::None;  // output gate
};

/// @p past_shape with the sequence dim replaced by @p past.
std::vector<int64_t> with_seq(const std::vector<int64_t>& shape, int64_t seq)
{
    auto s = shape;
    s[shape.size() == 3 ? 1 : 2] = seq;
    return s;
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

/// fast path vs reference on f32; returns false on the first mismatch.
bool gqa_fast_vs_ref(const Case& c, bool& used_flash, cpu::AttentionPlan* plan_out = nullptr,
                     const CpuBackend* pool = nullptr)
{
    auto [q_vec, q] = test::make_random_tensor(c.qshape, -1.0f, 1.0f, 100);
    auto [k_vec, k] = test::make_random_tensor(c.kshape, -1.0f, 1.0f, 101);
    auto [v_vec, v] = test::make_random_tensor(c.kshape, -1.0f, 1.0f, 102);

    std::vector<float> pk_vec, pv_vec;
    TensorView past_k, past_v;
    if (c.past > 0) {
        const auto ps = with_seq(c.kshape, c.past);
        auto [a, ka] = test::make_random_tensor(ps, -1.0f, 1.0f, 104);
        pk_vec = std::move(a);
        past_k = ka;
        auto [b, vb] = test::make_random_tensor(ps, -1.0f, 1.0f, 105);
        pv_vec = std::move(b);
        past_v = vb;
    }

    const int64_t sk_all = (c.kshape.size() == 3 ? c.kshape[1] : c.kshape[2]) + c.past;
    std::vector<float> m_vec;
    TensorView mask;
    if (c.mask) {
        const int64_t sq = c.qshape.size() == 3 ? c.qshape[1] : c.qshape[2];
        // A head bias carries the batch and head axes; the flat mask does not.
        const std::vector<int64_t> mshape = c.head_bias
            ? std::vector<int64_t>{c.qshape[0], c.heads, sq, sk_all}
            : std::vector<int64_t>{sq, sk_all};
        auto [mv, m] = test::make_random_tensor(mshape, -2.0f, 2.0f, 103);
        m_vec = std::move(mv);
        mask  = m;
    }

    GroupQueryAttentionAttributes attrs;
    attrs.num_heads    = c.heads;
    attrs.kv_num_heads = c.kv_heads;
    attrs.scale        = c.scale;
    attrs.causal       = c.causal;
    attrs.position_offset = c.position_offset;
    attrs.softcap      = c.softcap;
    attrs.window       = c.window;
    attrs.output_gate  = c.gate;

    auto op = GroupQueryAttention::create(attrs, Backend::CPU);

    // Per-batch run lengths, positional slot 6: an int32 [B] vector.
    std::vector<int32_t> sl_vec(c.seqlens);
    std::vector<int64_t> sl_shape;
    TensorView sl_view;
    if (!sl_vec.empty()) {
        sl_shape.push_back(static_cast<int64_t>(sl_vec.size()));
        sl_view = TensorView(sl_shape, DataType::s32, sl_vec.data());
    }

    std::vector<float> gate_vec;
    const bool has_seqlens = !sl_vec.empty();
    std::vector<TensorDesc> descs = {q.desc(), k.desc(), v.desc()};
    std::vector<TensorView> ins   = {q, k, v};
    if (c.mask) {
        descs.push_back(mask.desc());
        ins.push_back(mask);
    } else if (c.past > 0 || has_seqlens) {
        // The optional slots are positional: an empty view marks the absent mask.
        descs.push_back(TensorDesc{});
        ins.push_back(TensorView{});
    }
    if (c.past > 0) {
        descs.push_back(past_k.desc());
        ins.push_back(past_k);
        descs.push_back(past_v.desc());
        ins.push_back(past_v);
    } else if (has_seqlens) {
        // seqlens sits after the past pair, so its slots must exist too.
        descs.push_back(TensorDesc{});
        ins.push_back(TensorView{});
        descs.push_back(TensorDesc{});
        ins.push_back(TensorView{});
    }
    if (has_seqlens) {
        descs.push_back(sl_view.desc());
        ins.push_back(sl_view);
    }
    if (c.gate != EpilogueActivateType::None) {
        // The gate is the last slot and is shaped like the output, so the
        // absent metadata slots before it must still be filled positionally.
        auto [gv, gt] = test::make_random_tensor(c.qshape, -3.0f, 3.0f, 106);
        while (ins.size() < 9) {
            descs.push_back(TensorDesc{});
            ins.push_back(TensorView{});
        }
        descs.push_back(gt.desc());
        ins.push_back(gt);
        gate_vec = std::move(gv);
    }
    auto out_descs = op->getOutputTensorDesc(descs);
    const int64_t out_numel = out_descs[0].numel();

    cpu::AttentionConfig pcfg;
    pcfg.num_heads = c.heads;
    pcfg.num_kv_heads = c.kv_heads;
    pcfg.causal = c.causal;
    pcfg.position_offset = c.position_offset;
    pcfg.softcap = c.softcap;
    pcfg.window = c.window;
    const cpu::AttentionPlan plan = cpu::get_attention_plan(
        pcfg, descs, out_descs,
        pool != nullptr ? pool->worker_count() : 1,
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
    op->compute(fast_out, ins, fast_ctx, nullptr);

    std::vector<float> ref_buf(static_cast<size_t>(out_numel));
    auto ref_out = test::make_channel_first(out_descs[0], ref_buf.data());
    nnops::backend::cpu::reference::group_query_attention_ref(attrs, ref_out, ins, {}, nullptr);

    for (int64_t i = 0; i < out_numel; ++i) {
        const float diff = std::abs(fast_buf[static_cast<size_t>(i)] -
                                    ref_buf[static_cast<size_t>(i)]);
        const float thr  = 1e-4f + 1e-3f * std::max(std::abs(fast_buf[static_cast<size_t>(i)]),
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

}  // anonymous namespace

// ============================================================
// Shapes and descriptor rules
// ============================================================

NNOPS_TEST(gqa_output_shape_follows_q) {
    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = 8;
    attrs.kv_num_heads = 2;

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
    auto op = GroupQueryAttention::create(attrs, Backend::CPU);

    TensorDesc q = td({1, 5, 4 * 8});
    TensorDesc k = td({1, 9, 2 * 8});
    TensorDesc v = td({1, 7, 2 * 8});   // shorter value run
    const TensorDesc descs[] = {q, k, v};
    const TensorDesc out_descs[] = {td({1, 5, 4 * 8})};
    NNOPS_EXPECT_TRUE(op->validate(descs, out_descs) == Status::invalid_argument);
}

// ============================================================
// Fast path vs reference
// ============================================================

NNOPS_TEST(gqa_matches_ref) {
    const Case cases[] = {
        // MQA: one kv head for every query head, merged, causal.
        {{1, 6, 8 * 8},   {1, 6, 1 * 8},  8, 1, 0.0f, true},
        // Grouped ratio 2, merged, no causal.
        {{2, 5, 6 * 8},   {2, 5, 3 * 8},  6, 3, 0.0f, false},
        // Ratio 4 with a non-default scale.
        {{1, 4, 8 * 16},  {1, 4, 2 * 16}, 8, 2, 0.5f, true},
        // Additive mask, no causal.
        {{1, 6, 4 * 8},   {1, 6, 2 * 8},  4, 2, 0.0f, false, true},
        // Mask + causal together.
        {{1, 5, 4 * 8},   {1, 5, 1 * 8},  4, 1, 0.0f, true, true},
        // Decode shape: three new queries against a nine-token key run.
        {{1, 3, 4 * 8},   {1, 9, 1 * 8},  4, 1, 0.0f, true},
        // Explicit [B, H, S, D] / [B, H_kv, S, D].
        {{2, 6, 5, 8},    {2, 3, 5, 8},   6, 3, 0.0f, true},
        // H_kv == H is plain MHA.
        {{1, 7, 4 * 8},   {1, 7, 4 * 8},  4, 4, 0.0f, true},
        // Multi-k-block head dim (D = 192 splits 128 + 64 at Kc = 128).
        {{1, 4, 2 * 192}, {1, 4, 1 * 192}, 2, 1, 0.0f, true},
    };

    for (const auto& c : cases) {
        bool flash = false;
        NNOPS_EXPECT_TRUE(gqa_fast_vs_ref(c, flash));
        (void)flash;
    }
}

NNOPS_TEST(gqa_flash_matches_ref) {
    // Sq*Sk >= 65536 routes through the online-softmax flash path: MQA with a
    // long key run, and a grouped ratio on a square sequence.
    const Case cases[] = {
        {{1, 256, 8 * 64}, {1, 256, 1 * 64}, 8, 1, 0.0f, true},
        {{1, 320, 4 * 32}, {1, 320, 2 * 32}, 4, 2, 0.0f, true},
        {{1, 256, 2 * 128}, {1, 256, 1 * 128}, 2, 1, 0.0f, false},
    };

    for (const auto& c : cases) {
        bool flash = false;
        const bool ok = gqa_fast_vs_ref(c, flash);
        NNOPS_EXPECT_TRUE(flash);
        NNOPS_EXPECT_TRUE(ok);
    }
}

// ============================================================
// Single-token decode: the GEMV route (Sq == 1, no mask)
// ============================================================

NNOPS_TEST(gqa_decode_sq1_matches_ref) {
    // One query row against the whole key run: the decode route streams K and
    // V once, with no packing and no M == 1 tiles. The key runs pick the four
    // shapes the dot_row4 batch has to handle — a multiple of 4, a remainder,
    // a multi-k-block head dim, and a single key — plus the explicit layout.
    const Case cases[] = {
        {{1, 1, 8 * 8},    {1, 12, 1 * 8},   8, 1, 0.0f, true},   // MQA, Sk % 4 == 0
        {{1, 1, 8 * 16},   {1, 11, 4 * 16},  8, 4, 0.0f, true},   // ratio 2, Sk % 4 == 3
        {{1, 1, 2 * 192},  {1, 7, 1 * 192},  2, 1, 0.0f, false},  // D = 192, two k-blocks
        {{2, 4, 1, 8},     {2, 2, 5, 8},     4, 2, 0.0f, true},   // explicit layout
        {{1, 1, 4 * 8},    {1, 1, 4 * 8},    4, 4, 0.0f, true},   // single key, Hkv == H
        {{1, 1, 4 * 16},   {1, 13, 2 * 16},  4, 2, 0.5f, true},   // non-default scale
    };

    for (const auto& c : cases) {
        bool flash = false;
        cpu::AttentionPlan plan;
        NNOPS_EXPECT_TRUE(gqa_fast_vs_ref(c, flash, &plan));
        NNOPS_EXPECT_TRUE(plan.use_decode);
    }
}

NNOPS_TEST(gqa_decode_sq1_masked_matches_ref) {
    // An additive mask keeps the call on the standard route, so the two routes
    // have to agree on the same Sq == 1 shape.
    const Case cases[] = {
        {{1, 1, 8 * 8}, {1, 9, 2 * 8}, 8, 2, 0.0f, false, true},
        {{1, 1, 4 * 8}, {1, 9, 1 * 8}, 4, 1, 0.0f, true,  true},
    };

    for (const auto& c : cases) {
        bool flash = false;
        cpu::AttentionPlan plan;
        NNOPS_EXPECT_TRUE(gqa_fast_vs_ref(c, flash, &plan));
        NNOPS_EXPECT_TRUE(!plan.use_decode);
    }
}

NNOPS_TEST(gqa_decode_kv_split_matches_ref) {
    // A long decode run hands the head-batched plan fewer tasks than workers,
    // so it cuts the run into KV slices: each slice keeps its own online
    // softmax and a second dispatch merges the partial output/max/sum. The
    // reference over the whole run is the ground truth for that merge.
    SimplePool pool(8);

    const Case cases[] = {
        // H = 8 over 2 kv heads → 2 units; 2048 keys → 4 slices of 512.
        {{1, 1, 8 * 64},   {1, 2048, 2 * 64}, 8, 2, 0.0f, true, false, 0, -1, 2},
        // G = 16 caps the head batch at 8: two head blocks per kv head, and
        // 1500 keys → 4 slices of 375.
        {{1, 1, 16 * 32},  {1, 1500, 1 * 32}, 16, 1, 0.0f, true, false, 0, -1, 2},
        // [past | new] with the slice edges inside the 1000-row past; 1001
        // keys land on 3 slices of 334, so the edges are not tile-aligned.
        {{1, 1, 8 * 32},   {1, 1, 2 * 32},    8, 2, 0.0f, true, false, 1000, -1, 2},
    };

    for (const auto& c : cases) {
        bool flash = false;
        cpu::AttentionPlan plan;
        NNOPS_EXPECT_TRUE(gqa_fast_vs_ref(c, flash, &plan, &pool.cpu));
        NNOPS_EXPECT_TRUE(plan.use_decode);
        NNOPS_EXPECT_TRUE(plan.decode_splits > 1);
    }
}

// ============================================================
// Chunked input: a past segment and an explicit position offset
// ============================================================

NNOPS_TEST(gqa_step_matches_ref) {
    // The run the operator reads is [past | new]; the kernels stream the two
    // segments in place, so the reference — which indexes the same run by
    // logical position — is the ground truth for both the segment plumbing and
    // the per-row causal limit.
    const Case cases[] = {
        // Single new token after a 37-row past: the GEMV decode route.
        {{1, 1, 8 * 8},   {1, 1, 1 * 8},   8, 1, 0.0f, true,  false, 37,  37, 2},
        // Masked step: off the decode route, mask spanning [Sq, past + new].
        {{1, 1, 4 * 16},  {1, 1, 2 * 16},  4, 2, 0.0f, true,  true,  11,  11, 1},
        // Chunked prefill, D = 128: a 64-wide KV tile does not line up with the
        // 100-row segment boundary.
        {{1, 8, 4 * 128}, {1, 8, 2 * 128}, 4, 2, 0.0f, true,  false, 100, 100, 1},
        // Past shorter than one KV tile.
        {{1, 5, 4 * 8},   {1, 5, 2 * 8},   4, 2, 0.0f, true,  false, 3,   3,   1},
        // Explicit [B, H_kv, S, D] layout, past + new.
        {{2, 4, 3, 8},    {2, 2, 4, 8},    4, 2, 0.0f, true,  false, 6,   6,   1},
        // Non-tail chunk of a full run: no past tensor, only the offset — the
        // rows see the 8..15 key prefix, not the bottom-right one.
        {{1, 8, 4 * 16},  {1, 40, 2 * 16}, 4, 2, 0.0f, true,  false, 0,   7,   0},
        // Same chunk on the flash route, and with a mask.
        {{1, 256, 2 * 64}, {1, 512, 1 * 64}, 2, 1, 0.0f, true, false, 0,  128, 1},
        {{1, 8, 4 * 16},  {1, 40, 2 * 16}, 4, 2, 0.0f, true,  true,  0,   7,   0},
        // Decode whose row sits inside the run, not at its end: the score pass
        // stops at the causal limit instead of walking every key.
        {{1, 1, 8 * 8},   {1, 1, 1 * 8},   8, 1, 0.0f, true,  false, 37,  20, 2},
        {{1, 1, 4 * 16},  {1, 40, 2 * 16}, 4, 2, 0.0f, true,  false, 0,   17, 2},
    };

    for (const auto& c : cases) {
        bool flash = false;
        NNOPS_EXPECT_TRUE(gqa_fast_vs_ref(c, flash));
    }
}

// ============================================================
// Per-batch key-run lengths (seqlens_k)
// ============================================================

NNOPS_TEST(gqa_seqlens_matches_ref) {
    // Each batch reads the prefix [0, seqlens_k[b]): a shorter entry drops the
    // right padding and moves that batch's bottom-right causal diagonal, so
    // batches at different positions share one call.
    const Case cases[] = {
        // Standard route, causal, two batches at different positions.
        {{2, 6, 4 * 8}, {2, 9, 1 * 8}, 4, 1, 0.0f, true,  false, 0, -1, 0, {9, 5}},
        // Non-causal: the valid range is the run prefix and nothing else.
        {{2, 5, 4 * 8}, {2, 9, 2 * 8}, 4, 2, 0.0f, false, false, 0, -1, 0, {9, 3}},
        // A batch with an empty run: every key masked, its rows are zeros.
        {{2, 4, 4 * 8}, {2, 7, 2 * 8}, 4, 2, 0.0f, true,  false, 0, -1, 0, {7, 0}},
        // Explicit [B, H, S, D] layout.
        {{2, 4, 3, 8},  {2, 2, 5, 8},  4, 2, 0.0f, true,  false, 0, -1, 0, {5, 2}},
        // With a past: seqlens counts the whole [past | new] run (a split run
        // routes through flash).
        {{2, 5, 4 * 8}, {2, 5, 2 * 8}, 4, 2, 0.0f, true,  false, 3, -1, 1, {8, 4}},
        // Mask + seqlens: the mask stays one flat [Sq, Sk] for all batches.
        {{2, 5, 4 * 8}, {2, 9, 1 * 8}, 4, 1, 0.0f, true,  true,  0, -1, 0, {9, 6}},
        // Decode: Sq == 1, a per-batch limit inside one GEMV run.
        {{2, 1, 8 * 8}, {2, 12, 1 * 8}, 8, 1, 0.0f, true, false, 0, -1, 2, {12, 5}},
        // Decode with the run inside a past and one batch empty.
        {{2, 1, 4 * 8}, {2, 1, 1 * 8},  4, 1, 0.0f, true, false, 20, -1, 2, {21, 0}},
    };

    for (const auto& c : cases) {
        bool flash = false;
        NNOPS_EXPECT_TRUE(gqa_fast_vs_ref(c, flash));
        (void)flash;
    }
}

NNOPS_TEST(gqa_seqlens_flash_matches_ref) {
    // The flash route bounds its KV stream by the batch's run: the causal tile
    // skip and the row limit both use it.
    const Case cases[] = {
        {{2, 256, 4 * 64}, {2, 256, 2 * 64}, 4, 2, 0.0f, true,  false, 0, -1, 1, {256, 200}},
        // Non-causal flash: the run prefix is the whole valid range.
        {{1, 256, 2 * 128}, {1, 256, 1 * 128}, 2, 1, 0.0f, false, false, 0, -1, 1, {180}},
        // Split [past | new] run, one batch short of its tensor.
        {{2, 512, 4 * 64}, {2, 64, 2 * 64}, 4, 2, 0.0f, true, false, 100, -1, 1, {164, 120}},
    };

    for (const auto& c : cases) {
        bool flash = false;
        const bool ok = gqa_fast_vs_ref(c, flash);
        NNOPS_EXPECT_TRUE(flash);
        NNOPS_EXPECT_TRUE(ok);
    }
}

NNOPS_TEST(gqa_seqlens_decode_split_matches_ref) {
    // Variable-length decode down the sliced route: each batch computes its own
    // limit inside the task, and the merge sees per-slice states that may be
    // empty on one side.
    SimplePool pool(8);
    const Case cases[] = {
        {{2, 1, 8 * 32}, {2, 1, 2 * 32}, 8, 2, 0.0f, true, false, 1000, -1, 2, {1001, 700}},
    };

    for (const auto& c : cases) {
        bool flash = false;
        cpu::AttentionPlan plan;
        NNOPS_EXPECT_TRUE(gqa_fast_vs_ref(c, flash, &plan, &pool.cpu));
        NNOPS_EXPECT_TRUE(plan.use_decode);
        NNOPS_EXPECT_TRUE(plan.decode_splits > 1);
    }
}

NNOPS_TEST(gqa_seqlens_validation) {
    // Q, K, V + mask + past K/V + seqlens_k: the vector must be int32, rank 1
    // and one entry per batch.
    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = 4;
    attrs.kv_num_heads = 2;
    auto op = GroupQueryAttention::create(attrs, Backend::CPU);

    const TensorDesc q = td({2, 5, 4 * 8});
    const TensorDesc k = td({2, 9, 2 * 8});
    const TensorDesc v = td({2, 9, 2 * 8});
    const TensorDesc out = td({2, 5, 4 * 8});
    TensorDesc sl;
    sl.rank = 1;
    sl.dims = {2};
    sl.dtype = DataType::s32;

    {
        const TensorDesc ins[] = {q, k, v, TensorDesc{}, TensorDesc{}, TensorDesc{}, sl};
        const TensorDesc outs[] = {out};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::ok);
    }
    {
        TensorDesc wrong_dtype = sl;
        wrong_dtype.dtype = DataType::f32;
        const TensorDesc ins[] = {q, k, v, TensorDesc{}, TensorDesc{}, TensorDesc{}, wrong_dtype};
        const TensorDesc outs[] = {out};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::invalid_argument);
    }
    {
        TensorDesc wrong_rank = sl;
        wrong_rank.rank = 2;
        wrong_rank.dims = {2, 1};
        const TensorDesc ins[] = {q, k, v, TensorDesc{}, TensorDesc{}, TensorDesc{}, wrong_rank};
        const TensorDesc outs[] = {out};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::invalid_argument);
    }
    {
        TensorDesc wrong_len = sl;
        wrong_len.dims = {3};
        const TensorDesc ins[] = {q, k, v, TensorDesc{}, TensorDesc{}, TensorDesc{}, wrong_len};
        const TensorDesc outs[] = {out};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::invalid_argument);
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
    mha_attrs.causal = true;

    GroupQueryAttentionAttributes gqa_attrs;
    gqa_attrs.num_heads = 4;
    gqa_attrs.kv_num_heads = 4;
    gqa_attrs.causal = true;

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
// f16 fast path
// ============================================================

NNOPS_TEST(gqa_f16_matches_ref) {
    const Case cases[] = {
        {{1, 6, 8 * 8},   {1, 6, 1 * 8},  8, 1, 0.0f, true},
        {{2, 5, 6 * 16},  {2, 5, 3 * 16}, 6, 3, 0.0f, false, true},
        {{1, 256, 8 * 64}, {1, 256, 1 * 64}, 8, 1, 0.0f, true},   // flash
    };

    const float tol = test::kF16AccumTol;
    for (const auto& c : cases) {
        auto [q32v, q32] = test::make_random_tensor(c.qshape, -1.0f, 1.0f, 300);
        auto [k32v, k32] = test::make_random_tensor(c.kshape, -1.0f, 1.0f, 301);
        auto [v32v, v32] = test::make_random_tensor(c.kshape, -1.0f, 1.0f, 302);

        GroupQueryAttentionAttributes attrs;
        attrs.num_heads = c.heads;
        attrs.kv_num_heads = c.kv_heads;
        attrs.scale = c.scale;
        attrs.causal = c.causal;

        auto op = GroupQueryAttention::create(attrs, Backend::CPU);

        const TensorDesc descs32[] = {q32.desc(), k32.desc(), v32.desc()};
        const TensorView ins32[] = {q32, k32, v32};
        auto out_desc = op->getOutputTensorDesc(descs32)[0];
        const int64_t n = out_desc.numel();

        std::vector<float> ref_buf(static_cast<size_t>(n));
        auto ref_out = test::make_channel_first(out_desc, ref_buf.data());
        nnops::backend::cpu::reference::group_query_attention_ref(attrs, ref_out, ins32, {}, nullptr);

        auto q16v = test::f32_to_f16(q32v);
        auto k16v = test::f32_to_f16(k32v);
        auto v16v = test::f32_to_f16(v32v);
        TensorView q16(std::span<const int64_t>(c.qshape), DataType::f16, q16v.data());
        TensorView k16(std::span<const int64_t>(c.kshape), DataType::f16, k16v.data());
        TensorView v16(std::span<const int64_t>(c.kshape), DataType::f16, v16v.data());

        // The output view must carry the f16 descriptor — one taken from the
        // f32 descriptors makes compute() reject the dtype and write nothing.
        const TensorDesc fdescs[] = {q16.desc(), k16.desc(), v16.desc()};
        const auto fout_desc = op->getOutputTensorDesc(fdescs)[0];
        std::vector<cpu::half> got_buf(static_cast<size_t>(n));
        auto got = test::make_channel_first(fout_desc, got_buf.data());
        const TensorView ins16[] = {q16, k16, v16};
        NNOPS_EXPECT_TRUE(op->compute(got, ins16) == Status::ok);

        const auto got_f32 = test::f16_to_f32(got_buf);
        float worst = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            worst = std::max(worst, std::abs(got_f32[static_cast<size_t>(i)] -
                                             ref_buf[static_cast<size_t>(i)]));
        }
        if (worst > tol) {
            throw std::runtime_error("gqa f16 deviation " + std::to_string(worst) +
                                     " > tol " + std::to_string(tol));
        }
    }
}

NNOPS_TEST(gqa_f16_decode_split_matches_ref) {
    // f16 down the sliced decode route: the f32 partials and the merge are
    // element-type independent, only the loads, the stores and the running
    // state move to half.
    SimplePool pool(8);
    const Case c = {{1, 1, 8 * 32}, {1, 1200, 2 * 32}, 8, 2, 0.0f, true};

    auto [q32v, q32] = test::make_random_tensor(c.qshape, -1.0f, 1.0f, 400);
    auto [k32v, k32] = test::make_random_tensor(c.kshape, -1.0f, 1.0f, 401);
    auto [v32v, v32] = test::make_random_tensor(c.kshape, -1.0f, 1.0f, 402);

    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = c.heads;
    attrs.kv_num_heads = c.kv_heads;
    attrs.causal = c.causal;
    auto op = GroupQueryAttention::create(attrs, Backend::CPU);

    const TensorDesc descs32[] = {q32.desc(), k32.desc(), v32.desc()};
    const TensorView ins32[] = {q32, k32, v32};
    auto out_desc = op->getOutputTensorDesc(descs32)[0];
    const int64_t n = out_desc.numel();

    std::vector<float> ref_buf(static_cast<size_t>(n));
    auto ref_out = test::make_channel_first(out_desc, ref_buf.data());
    nnops::backend::cpu::reference::group_query_attention_ref(attrs, ref_out, ins32, {}, nullptr);

    auto q16v = test::f32_to_f16(q32v);
    auto k16v = test::f32_to_f16(k32v);
    auto v16v = test::f32_to_f16(v32v);
    TensorView q16(std::span<const int64_t>(c.qshape), DataType::f16, q16v.data());
    TensorView k16(std::span<const int64_t>(c.kshape), DataType::f16, k16v.data());
    TensorView v16(std::span<const int64_t>(c.kshape), DataType::f16, v16v.data());

    const TensorDesc fdescs[] = {q16.desc(), k16.desc(), v16.desc()};
    const auto fout_desc = op->getOutputTensorDesc(fdescs)[0];
    std::vector<cpu::half> got_buf(static_cast<size_t>(n));
    auto got = test::make_channel_first(fout_desc, got_buf.data());
    const TensorView ins16[] = {q16, k16, v16};
    ComputeContext ctx;
    ctx.cpu = pool.cpu;
    NNOPS_EXPECT_TRUE(op->compute(got, ins16, ctx) == Status::ok);

    const auto got_f32 = test::f16_to_f32(got_buf);
    float worst = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        worst = std::max(worst, std::abs(got_f32[static_cast<size_t>(i)] -
                                         ref_buf[static_cast<size_t>(i)]));
    }
    if (worst > test::kF16AccumTol) {
        throw std::runtime_error("gqa f16 decode deviation " + std::to_string(worst) +
                                 " > tol " + std::to_string(test::kF16AccumTol));
    }
}

// ============================================================
// Logit softcap (Gemma2/3)
// ============================================================

NNOPS_TEST(gqa_softcap_matches_ref) {
    // s = cap·tanh(s/cap) on the *scaled* logit and before the mask, so the cap
    // sees q·k alone and the bias is added to the already-capped score. Every
    // route has its own spelling of that order: the standard row pass, the
    // flash tile pass, and the decode route, where the cap absorbs the scale
    // and the exponential runs unscaled.
    const Case cases[] = {
        // Standard route.
        {{1, 6, 4 * 8},   {1, 9, 2 * 8},   4, 2, 0.0f, true,  false, 0, -1, 0,
         {}, 2.0f},
        // Cap + mask: capping the biased logit instead would move the result.
        {{1, 6, 4 * 8},   {1, 9, 2 * 8},   4, 2, 0.0f, true,  true,  0, -1, 0,
         {}, 1.0f},
        // Non-causal: every key enters the cap.
        {{1, 5, 2 * 16},  {1, 7, 1 * 16},  2, 1, 0.0f, false, false, 0, -1, 0,
         {}, 0.75f},
        // Flash route.
        {{1, 256, 2 * 64}, {1, 300, 1 * 64}, 2, 1, 0.0f, true, false, 0, -1, 1,
         {}, 1.5f},
        // Flash + head bias.
        {{1, 256, 2 * 64}, {1, 300, 1 * 64}, 2, 1, 0.0f, true, true,  0, -1, 1,
         {}, 1.5f, 0, true},
        // Decode: the scale folds into the cap pass.
        {{2, 1, 8 * 8},   {2, 300, 2 * 8}, 8, 2, 0.0f, true,  false, 0, -1, 2,
         {}, 3.0f},
        // Split [past | new] run down the flash route.
        {{1, 4, 4 * 16},  {1, 4, 2 * 16},  4, 2, 0.0f, true,  false, 61, 61, 1,
         {}, 2.0f},
    };

    for (const auto& c : cases) {
        bool flash = false;
        NNOPS_EXPECT_TRUE(gqa_fast_vs_ref(c, flash));
    }
}

NNOPS_TEST(gqa_softcap_decode_split_matches_ref) {
    // A KV-sliced decode with a softcap: each slice caps its own raw dots and
    // leads with an unscaled max, and the merge reweights by
    // exp((m - M)·1) — the cap already owns the scale.
    SimplePool pool(8);
    const Case cases[] = {
        {{2, 1, 8 * 32}, {2, 1, 2 * 32}, 8, 2, 0.0f, true, false, 1000, -1, 2,
         {}, 2.0f},
    };

    for (const auto& c : cases) {
        bool flash = false;
        cpu::AttentionPlan plan;
        NNOPS_EXPECT_TRUE(gqa_fast_vs_ref(c, flash, &plan, &pool.cpu));
        NNOPS_EXPECT_TRUE(plan.use_decode);
        NNOPS_EXPECT_TRUE(plan.decode_splits > 1);
    }
}

// ============================================================
// Sliding-window attention
// ============================================================

NNOPS_TEST(gqa_window_matches_ref) {
    // Row i's valid range is [pos_i - window + 1, causal limit): the window
    // raises the lower bound instead of lowering the upper one, and both clamp
    // to the batch's run.
    const Case cases[] = {
        // Causal square: row i sees [i - 2, i].
        {{1, 8, 4 * 8},   {1, 8, 2 * 8},   4, 2, 0.0f, true,  false, 0, -1, 0,
         {}, 0.0f, 3},
        // Non-causal: the upper bound stays the run, only the start moves.
        {{1, 8, 4 * 8},   {1, 12, 2 * 8},  4, 2, 0.0f, false, false, 0, -1, 0,
         {}, 0.0f, 4},
        // A window wider than the run is a no-op.
        {{1, 5, 4 * 8},   {1, 9, 2 * 8},   4, 2, 0.0f, true,  false, 0, -1, 0,
         {}, 0.0f, 64},
        // Sliding decode.
        {{1, 1, 8 * 8},   {1, 500, 1 * 8}, 8, 1, 0.0f, true,  false, 0, -1, 2,
         {}, 0.0f, 16},
        // Split [past | new] whose window reaches across the boundary.
        {{1, 4, 4 * 16},  {1, 4, 2 * 16},  4, 2, 0.0f, true,  false, 61, 61, 1,
         {}, 0.0f, 8},
        // Flash: the tile skip starts at the window's lower bound.
        {{1, 256, 2 * 64}, {1, 512, 1 * 64}, 2, 1, 0.0f, true, false, 0, -1, 1,
         {}, 0.0f, 100},
        // Window + mask.
        {{1, 8, 4 * 8},   {1, 9, 2 * 8},   4, 2, 0.0f, true,  true,  0, -1, 0,
         {}, 0.0f, 4},
        // Window + per-batch runs: the window clamps to each batch's run.
        {{2, 6, 4 * 8},   {2, 9, 1 * 8},   4, 1, 0.0f, true,  false, 0, -1, 0,
         {9, 5}, 0.0f, 3},
        // Window + softcap together.
        {{1, 8, 4 * 8},   {1, 10, 2 * 8},  4, 2, 0.0f, true,  false, 0, -1, 0,
         {}, 2.0f, 5},
    };

    for (const auto& c : cases) {
        bool flash = false;
        NNOPS_EXPECT_TRUE(gqa_fast_vs_ref(c, flash));
    }
}

// ============================================================
// Per-(batch, head) additive bias, carried as a rank-4 mask
// ============================================================

NNOPS_TEST(gqa_head_bias_matches_ref) {
    // A rank-4 [B, H, Sq, Sk] mask is an additive bias with its own entry per
    // query head; a rank-2 one is shared by every head. Both add after the cap
    // and before the softmax.
    const Case cases[] = {
        // Standard route, causal.
        {{2, 5, 4 * 8},   {2, 9, 2 * 8},   4, 2, 0.0f, true,  true,  0, -1, 0,
         {}, 0.0f, 0, true},
        // Non-causal: every bias entry is read.
        {{1, 6, 2 * 8},   {1, 6, 1 * 8},   2, 1, 0.0f, false, true,  0, -1, 0,
         {}, 0.0f, 0, true},
        // Explicit [B, H, S, D] value layout, merged bias.
        {{2, 4, 3, 8},    {2, 2, 5, 8},    4, 2, 0.0f, true,  true,  0, -1, 0,
         {}, 0.0f, 0, true},
        // Flash route.
        {{1, 256, 2 * 64}, {1, 256, 1 * 64}, 2, 1, 0.0f, true, true,  0, -1, 1,
         {}, 0.0f, 0, true},
        // Past segment: the bias spans [past | new].
        {{1, 4, 4 * 16},  {1, 4, 2 * 16},  4, 2, 0.0f, true,  true,  32, 32, 1,
         {}, 0.0f, 0, true},
    };

    for (const auto& c : cases) {
        bool flash = false;
        NNOPS_EXPECT_TRUE(gqa_fast_vs_ref(c, flash));
    }
}

// ============================================================
// Packed varlen (cu_seqlens)
// ============================================================

namespace {

/// One packed varlen call: [Q, K, V, {}, {}, {}, {}, cu_q, cu_k] over
/// [1, total, W] tensors, each segment run as its own [1, len, W] problem.
struct VarlenCase {
    std::vector<int32_t> cu_q;
    std::vector<int32_t> cu_k;
    int64_t heads = 4;
    int64_t kv_heads = 2;
    int64_t dim = 8;
    bool causal = true;
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
    attrs.causal = c.causal;
    attrs.softcap = c.softcap;
    attrs.window = c.window;
    attrs.output_gate = c.gate;

    auto op = GroupQueryAttention::create(attrs, Backend::CPU);

    const TensorView none;
    std::vector<TensorView> ins = {q, k, v, none, none, none, none,
                                   cu_q_view, cu_k_view};
    std::vector<float> g_vec;
    if (c.gate != EpilogueActivateType::None) {
        // The gate is the parent's, shaped like the whole packed output.
        auto [gv, gt] = test::make_random_tensor(
            {1, total_q, c.heads * c.dim}, -3.0f, 3.0f, 703);
        g_vec = std::move(gv);
        ins.push_back(gt);
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

    // Every segment must take the route the case asked for; each packed slice
    // is planned on its own [1, len, W] shape.
    if (c.route >= 0) {
        for (size_t s = 0; s + 1 < qo.size(); ++s) {
            const int64_t lq = qo[s + 1] - qo[s];
            if (lq == 0) { continue; }
            const int64_t lk = ko[s + 1] - ko[s];
            cpu::AttentionConfig pcfg;
            pcfg.num_heads = c.heads;
            pcfg.num_kv_heads = c.kv_heads;
            pcfg.causal = c.causal;
            pcfg.softcap = c.softcap;
            pcfg.window = c.window;
            const TensorDesc sd[] = {td({1, lq, c.heads * c.dim}),
                                     td({1, lk, c.kv_heads * c.dim}),
                                     td({1, lk, c.kv_heads * c.dim})};
            const TensorDesc so[] = {td({1, lq, c.heads * c.dim})};
            const cpu::AttentionPlan plan =
                cpu::get_attention_plan(pcfg, sd, so, 1, false);
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
    nnops::backend::cpu::reference::group_query_attention_ref(attrs, ref_out, ins,
                                                              {}, nullptr);

    for (int64_t i = 0; i < n; ++i) {
        const float diff = std::abs(fast_buf[static_cast<size_t>(i)] -
                                    ref_buf[static_cast<size_t>(i)]);
        const float thr  = 1e-4f + 1e-3f * std::max(std::abs(fast_buf[static_cast<size_t>(i)]),
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

}  // anonymous namespace

NNOPS_TEST(gqa_varlen_matches_ref) {
    // Packed segments restart the causal diagonal at each cu offset and see no
    // key from a neighbouring document. The segments differ in both lengths, so
    // the bottom-right alignment has to move the query window per segment.
    const VarlenCase cases[] = {
        // Three segments whose middle one is a single row: the routes are
        // chosen per segment, so this call mixes standard and decode.
        {{0, 5, 6, 10}, {0, 6, 7, 15}, 4, 2, 8, true,  -1},
        // MQA and a non-causal packing.
        {{0, 4, 9},     {0, 4, 12},    4, 1, 16, false, 0},
        // An empty leading segment, which must be skipped.
        {{0, 0, 3},     {0, 4, 7},     4, 2, 8, true,  0},
        // Two long segments on the flash route.
        {{0, 256, 512}, {0, 300, 600}, 4, 2, 64, true, 1},
        // Two single-row segments down the decode route.
        {{0, 1, 2},     {0, 300, 500}, 8, 1, 8, true,  2},
        // Softcap + window on packed segments.
        {{0, 6, 9},     {0, 6, 12},    4, 2, 8, true,  0, 2.0f, 3},
        // Explicitly non-multiple head dim (D = 24).
        {{0, 3, 7},     {0, 3, 9},     2, 1, 24, true, 0},
        // Output gate over a packed batch: one pass over the parent output,
        // after every segment already ran.
        {{0, 5, 6, 10}, {0, 6, 7, 15}, 4, 2, 8, true,  -1, 0.0f, 0,
         EpilogueActivateType::Sigmoid},
    };

    for (const auto& c : cases) {
        NNOPS_EXPECT_TRUE(varlen_vs_ref(c));
    }
}

NNOPS_TEST(gqa_varlen_validation) {
    // The two cu vectors arrive together, are int32 [segments + 1], and a
    // packed batch rules out the shared mask, the past pair, seqlens_k and a
    // global position_offset.
    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = 4;
    attrs.kv_num_heads = 2;
    auto op = GroupQueryAttention::create(attrs, Backend::CPU);

    const TensorDesc q = td({1, 10, 4 * 8});
    const TensorDesc k = td({1, 15, 2 * 8});
    const TensorDesc v = td({1, 15, 2 * 8});
    const TensorDesc out = td({1, 10, 4 * 8});
    TensorDesc cu;
    cu.rank = 1;
    cu.dims = {4};
    cu.dtype = DataType::s32;

    const TensorDesc none;
    {
        const TensorDesc ins[] = {q, k, v, none, none, none, none, cu, cu};
        const TensorDesc outs[] = {out};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::ok);
    }
    {
        // cu_k missing: the offsets are a pair.
        const TensorDesc ins[] = {q, k, v, none, none, none, none, cu, none};
        const TensorDesc outs[] = {out};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::invalid_argument);
    }
    {
        // Different segment counts.
        TensorDesc cu_k = cu;
        cu_k.dims = {3};
        const TensorDesc ins[] = {q, k, v, none, none, none, none, cu, cu_k};
        const TensorDesc outs[] = {out};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::invalid_argument);
    }
    {
        // A single entry cannot describe one segment.
        TensorDesc one = cu;
        one.dims = {1};
        const TensorDesc ins[] = {q, k, v, none, none, none, none, one, one};
        const TensorDesc outs[] = {out};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::invalid_argument);
    }
    {
        // The offsets are int32 metadata.
        TensorDesc wrong_dtype = cu;
        wrong_dtype.dtype = DataType::f32;
        const TensorDesc ins[] = {q, k, v, none, none, none, none, wrong_dtype, cu};
        const TensorDesc outs[] = {out};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::invalid_argument);
    }
    {
        // A shared mask does not apply across documents.
        const TensorDesc mask = td({10, 15});
        const TensorDesc ins[] = {q, k, v, mask, none, none, none, cu, cu};
        const TensorDesc outs[] = {out};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::invalid_argument);
    }
    {
        // Nor does a past segment or seqlens_k.
        const TensorDesc past = td({1, 3, 2 * 8});
        const TensorDesc ins[] = {q, k, v, none, past, past, none, cu, cu};
        const TensorDesc outs[] = {out};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::invalid_argument);
    }
    {
        TensorDesc sl;
        sl.rank = 1;
        sl.dims = {1};
        sl.dtype = DataType::s32;
        const TensorDesc ins[] = {q, k, v, none, none, none, sl, cu, cu};
        const TensorDesc outs[] = {out};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::invalid_argument);
    }
    {
        // The packing is merged: an explicit [B, H, S, D] batch has no packed
        // rows to slice.
        const TensorDesc q4 = td({1, 4, 10, 8});
        const TensorDesc k4 = td({1, 2, 15, 8});
        const TensorDesc ins[] = {q4, k4, k4, none, none, none, none, cu, cu};
        const TensorDesc outs[] = {out};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::invalid_argument);
    }

    // A global position_offset cannot be expressed for a packed batch.
    GroupQueryAttentionAttributes offset;
    offset.num_heads = 4;
    offset.kv_num_heads = 2;
    offset.position_offset = 2;
    auto offset_op = GroupQueryAttention::create(offset, Backend::CPU);
    {
        const TensorDesc ins[] = {q, k, v, none, none, none, none, cu, cu};
        const TensorDesc outs[] = {out};
        NNOPS_EXPECT_TRUE(offset_op->validate(ins, outs) == Status::invalid_argument);
    }
}

// ============================================================
// Output gate (Qwen3-Next / 3.5 / 4)
// ============================================================

NNOPS_TEST(gqa_output_gate_hand) {
    // The gate is one pass over the finished output, so the gated call is
    // exactly plain * act(gate) element for element — Sigmoid is the Qwen form.
    const std::vector<int64_t> qshape = {2, 3, 2 * 8};
    const std::vector<int64_t> kshape = {2, 5, 1 * 8};
    auto [q_vec, q] = test::make_random_tensor(qshape, -1.0f, 1.0f, 200);
    auto [k_vec, k] = test::make_random_tensor(kshape, -1.0f, 1.0f, 201);
    auto [v_vec, v] = test::make_random_tensor(kshape, -1.0f, 1.0f, 202);
    auto [g_vec, g] = test::make_random_tensor(qshape, -4.0f, 4.0f, 203);

    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = 2;
    attrs.kv_num_heads = 1;
    attrs.causal = true;

    const TensorDesc descs[] = {q.desc(), k.desc(), v.desc()};
    auto op = GroupQueryAttention::create(attrs, Backend::CPU);
    const auto out_descs = op->getOutputTensorDesc(descs);
    const int64_t n = out_descs[0].numel();

    std::vector<float> plain(static_cast<size_t>(n));
    auto plain_out = test::make_channel_first(out_descs[0], plain.data());
    const TensorView ins_plain[] = {q, k, v};
    NNOPS_EXPECT_TRUE(op->compute(plain_out, ins_plain) == Status::ok);

    GroupQueryAttentionAttributes gated = attrs;
    gated.output_gate = EpilogueActivateType::Sigmoid;
    auto gop = GroupQueryAttention::create(gated, Backend::CPU);
    std::vector<float> got(static_cast<size_t>(n));
    auto got_out = test::make_channel_first(out_descs[0], got.data());
    const TensorView empty;
    const TensorView ins_gated[] = {q, k, v, empty, empty, empty, empty,
                                    empty, empty, g};
    NNOPS_EXPECT_TRUE(gop->compute(got_out, ins_gated) == Status::ok);

    for (int64_t i = 0; i < n; ++i) {
        const float want = plain[static_cast<size_t>(i)] /
                           (1.0f + std::exp(-g_vec[static_cast<size_t>(i)]));
        NNOPS_EXPECT_NEAR(got[static_cast<size_t>(i)], want, 1e-5f);
    }
}

NNOPS_TEST(gqa_output_gate_matches_ref) {
    // Every activation the gated forms implement, across the routes: the
    // standard path, the decode path (Sq == 1), the flash path, the explicit
    // [B, H, S, D] layout, and a [past | new] run.
    const EpilogueActivateType acts[] = {
        EpilogueActivateType::Sigmoid,
        EpilogueActivateType::Silu,
        EpilogueActivateType::Gelu,
        EpilogueActivateType::Tanh,
        EpilogueActivateType::Relu,
    };
    for (const auto act : acts) {
        const Case cases[] = {
            // Standard route, causal.
            {{2, 4, 4 * 8},   {2, 7, 2 * 8},  4, 2, 0.0f, true,  false, 0, -1, 0,
             {}, 0.0f, 0, false, act},
            // Non-causal: every key contributes.
            {{1, 5, 2 * 8},   {1, 9, 1 * 8},  2, 1, 0.0f, false, false, 0, -1, 0,
             {}, 0.0f, 0, false, act},
            // Decode: one query row against the run.
            {{1, 1, 4 * 8},   {1, 12, 2 * 8}, 4, 2, 0.0f, true,  false, 0, -1, 2,
             {}, 0.0f, 0, false, act},
            // Flash route.
            {{1, 256, 2 * 64}, {1, 256, 1 * 64}, 2, 1, 0.0f, true, false, 0, -1, 1,
             {}, 0.0f, 0, false, act},
            // Explicit [B, H, S, D] layout.
            {{2, 4, 3, 8},    {2, 2, 5, 8},   4, 2, 0.0f, true,  false, 0, -1, 0,
             {}, 0.0f, 0, false, act},
            // [past | new]: the gate spans the query rows only.
            {{1, 4, 4 * 16},  {1, 4, 2 * 16}, 4, 2, 0.0f, true,  false, 32, 32, 1,
             {}, 0.0f, 0, false, act},
        };
        for (const auto& c : cases) {
            bool flash = false;
            NNOPS_EXPECT_TRUE(gqa_fast_vs_ref(c, flash));
        }
    }
}

NNOPS_TEST(gqa_output_gate_f16) {
    // f16 down the fast path: the gate pass is dtype-templated like the routes.
    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = 4;
    attrs.kv_num_heads = 2;
    attrs.causal = true;
    attrs.output_gate = EpilogueActivateType::Sigmoid;

    const std::vector<int64_t> qshape = {1, 4, 4 * 16};
    const std::vector<int64_t> kshape = {1, 6, 2 * 16};
    auto [q32v, q32] = test::make_random_tensor(qshape, -1.0f, 1.0f, 210);
    auto [k32v, k32] = test::make_random_tensor(kshape, -1.0f, 1.0f, 211);
    auto [v32v, v32] = test::make_random_tensor(kshape, -1.0f, 1.0f, 212);
    auto [g32v, g32] = test::make_random_tensor(qshape, -3.0f, 3.0f, 213);

    auto op = GroupQueryAttention::create(attrs, Backend::CPU);
    const TensorDesc descs32[] = {q32.desc(), k32.desc(), v32.desc()};
    const auto out_descs = op->getOutputTensorDesc(descs32);
    const int64_t n = out_descs[0].numel();

    std::vector<float> ref_buf(static_cast<size_t>(n));
    auto ref_out = test::make_channel_first(out_descs[0], ref_buf.data());
    const TensorView empty;
    const TensorView ins32[] = {q32, k32, v32, empty, empty, empty, empty,
                                empty, empty, g32};
    nnops::backend::cpu::reference::group_query_attention_ref(
        attrs, ref_out, ins32, {}, nullptr);

    auto q16v = test::f32_to_f16(q32v);
    auto k16v = test::f32_to_f16(k32v);
    auto v16v = test::f32_to_f16(v32v);
    auto g16v = test::f32_to_f16(g32v);
    TensorView q16(std::span<const int64_t>(qshape), DataType::f16, q16v.data());
    TensorView k16(std::span<const int64_t>(kshape), DataType::f16, k16v.data());
    TensorView v16(std::span<const int64_t>(kshape), DataType::f16, v16v.data());
    TensorView g16(std::span<const int64_t>(qshape), DataType::f16, g16v.data());

    std::vector<cpu::half> got_buf(static_cast<size_t>(n));
    TensorDesc out16 = out_descs[0];
    out16.dtype = DataType::f16;
    auto got_out = test::make_channel_first(out16, got_buf.data());
    const TensorView ins16[] = {q16, k16, v16, empty, empty, empty, empty,
                                empty, empty, g16};
    NNOPS_EXPECT_TRUE(op->compute(got_out, ins16) == Status::ok);

    const auto got_f32 = test::f16_to_f32(got_buf);
    for (int64_t i = 0; i < n; ++i) {
        const float diff = std::abs(got_f32[static_cast<size_t>(i)] -
                                    ref_buf[static_cast<size_t>(i)]);
        if (diff > 2e-3f) {
            std::printf("        gate f16 deviation %f at %lld\n",
                        (double)diff, (long long)i);
            NNOPS_EXPECT_TRUE(false);
            break;
        }
    }
}

NNOPS_TEST(gqa_output_gate_validation) {
    // The gate is slot 9, present exactly when the attribute names an
    // activation, and shaped like Q (the output).
    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = 4;
    attrs.kv_num_heads = 2;
    attrs.output_gate = EpilogueActivateType::Sigmoid;
    auto op = GroupQueryAttention::create(attrs, Backend::CPU);

    const TensorDesc q = td({1, 5, 4 * 8});
    const TensorDesc k = td({1, 9, 2 * 8});
    const TensorDesc out = td({1, 5, 4 * 8});
    const TensorDesc none;
    {
        const TensorDesc ins[] = {q, k, k, none, none, none, none, none, none, q};
        const TensorDesc outs[] = {out};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::ok);
    }
    {
        // The attribute is set but the gate is absent.
        const TensorDesc ins[] = {q, k, k};
        const TensorDesc outs[] = {out};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::invalid_argument);
    }
    {
        // A gate of the wrong width cannot multiply the output.
        const TensorDesc narrow = td({1, 5, 4 * 7});
        const TensorDesc ins[] = {q, k, k, none, none, none, none, none, none, narrow};
        const TensorDesc outs[] = {out};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::invalid_argument);
    }
    {
        // Nor can one of another dtype.
        TensorDesc g = q;
        g.dtype = DataType::f16;
        const TensorDesc ins[] = {q, k, k, none, none, none, none, none, none, g};
        const TensorDesc outs[] = {out};
        NNOPS_EXPECT_TRUE(op->validate(ins, outs) == Status::invalid_argument);
    }
    {
        // A gate with no attribute to activate it is ambiguous.
        GroupQueryAttentionAttributes plain;
        plain.num_heads = 4;
        plain.kv_num_heads = 2;
        auto plain_op = GroupQueryAttention::create(plain, Backend::CPU);
        const TensorDesc ins[] = {q, k, k, none, none, none, none, none, none, q};
        const TensorDesc outs[] = {out};
        NNOPS_EXPECT_TRUE(plain_op->validate(ins, outs) == Status::invalid_argument);
    }
    {
        // Only the activations the gated forms implement.
        GroupQueryAttentionAttributes bad = attrs;
        bad.output_gate = EpilogueActivateType::LeakyRelu;
        auto bad_op = GroupQueryAttention::create(bad, Backend::CPU);
        const TensorDesc ins[] = {q, k, k, none, none, none, none, none, none, q};
        const TensorDesc outs[] = {out};
        NNOPS_EXPECT_TRUE(bad_op->validate(ins, outs) == Status::unsupported);
    }
}
