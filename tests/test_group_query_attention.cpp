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
/// once Sq*Sk reaches kFlashMinScores (1 << 16).

#include "nnops/ops/group_query_attention.hpp"
#include "nnops/ops/multi_head_attention.hpp"
#include "backend/cpu/attention/attention.h"   // AttentionConfig / get_attention_plan
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
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
};

/// fast path vs reference on f32; returns false on the first mismatch.
bool gqa_fast_vs_ref(const Case& c, bool& used_flash, cpu::AttentionPlan* plan_out = nullptr)
{
    auto [q_vec, q] = test::make_random_tensor(c.qshape, -1.0f, 1.0f, 100);
    auto [k_vec, k] = test::make_random_tensor(c.kshape, -1.0f, 1.0f, 101);
    auto [v_vec, v] = test::make_random_tensor(c.kshape, -1.0f, 1.0f, 102);

    std::vector<float> m_vec;
    TensorView mask;
    if (c.mask) {
        const int64_t sq = c.qshape.size() == 3 ? c.qshape[1] : c.qshape[2];
        const int64_t sk = c.kshape.size() == 3 ? c.kshape[1] : c.kshape[2];
        auto [mv, m] = test::make_random_tensor({sq, sk}, -2.0f, 2.0f, 103);
        m_vec = std::move(mv);
        mask  = m;
    }

    GroupQueryAttentionAttributes attrs;
    attrs.num_heads    = c.heads;
    attrs.kv_num_heads = c.kv_heads;
    attrs.scale        = c.scale;
    attrs.causal       = c.causal;

    auto op = GroupQueryAttention::create(attrs, Backend::CPU);

    std::vector<TensorDesc> descs = {q.desc(), k.desc(), v.desc()};
    std::vector<TensorView> ins   = {q, k, v};
    if (c.mask) {
        descs.push_back(mask.desc());
        ins.push_back(mask);
    }
    auto out_descs = op->getOutputTensorDesc(descs);
    const int64_t out_numel = out_descs[0].numel();

    const cpu::AttentionPlan plan = cpu::get_attention_plan({c.heads, c.kv_heads},
                                                             descs, out_descs);
    used_flash = plan.use_flash;
    if (plan_out) { *plan_out = plan; }

    std::vector<float> fast_buf(static_cast<size_t>(out_numel));
    auto fast_out = test::make_planar(out_descs[0], fast_buf.data());
    op->compute(fast_out, ins, {}, nullptr);

    std::vector<float> ref_buf(static_cast<size_t>(out_numel));
    auto ref_out = test::make_planar(out_descs[0], ref_buf.data());
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
    auto a = test::make_planar(mha_desc, a_buf.data());
    auto b = test::make_planar(gqa_desc, b_buf.data());

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
        auto ref_out = test::make_planar(out_desc, ref_buf.data());
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
        auto got = test::make_planar(fout_desc, got_buf.data());
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
