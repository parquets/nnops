/// @file test_multi_latent_attention.cpp
/// @brief MultiLatentAttention tests — the class API against the CPU reference.
///
/// reference::multi_latent_attention_ref is f32 only, so f16 is compared against
/// the f32 reference on the same values with the arch-aware kF16AccumTol.
///
/// Cases cover the absorbed geometry: every query head over one shared latent
/// head, a two-head latent, v_head_dim equal to and narrower than kv_lora_rank,
/// merged [B, S, H*Dkv] and explicit [B, H, S, Dkv] layouts, the causal mask on
/// and off, an additive mask, a decode-shaped shorter query run, and both
/// routes — standard, and flash once Sq*Sk reaches kFlashMinScores (1 << 16).
/// The step cases put latent rows in a past segment and set position_offset,
/// which also exercises the flash route's two-segment KV stream.

#include "nnops/ops/multi_latent_attention.hpp"
#include "backend/cpu/attention/attention.h"
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
void multi_latent_attention_ref(const MultiLatentAttentionAttributes& attrs,
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
    std::vector<int64_t> qshape;    // [B, Sq, H*Dkv] or [B, H, Sq, Dkv]
    std::vector<int64_t> kvshape;   // [B, Sk, Hkv*Dkv] or [B, Hkv, Sk, Dkv]
    int64_t heads;
    int64_t kv_heads;
    int64_t rank;
    int64_t rope;
    int64_t nope;
    int64_t vdim;                   // 0 -> rank
    float scale = 0.0f;
    bool causal = true;
    bool mask = false;
    int64_t past = 0;              // latent rows in the past segment
    int64_t position_offset = -1;  // absolute position of query row 0
    int route = -1;                // 0 standard, 1 flash, 2 decode; -1 unchecked
};

MultiLatentAttentionAttributes attrs_of(const Case& c)
{
    MultiLatentAttentionAttributes a;
    a.num_heads         = c.heads;
    a.kv_num_heads      = c.kv_heads;
    a.kv_lora_rank      = c.rank;
    a.qk_rope_head_dim  = c.rope;
    a.qk_nope_head_dim  = c.nope;
    a.v_head_dim        = c.vdim;
    a.scale             = c.scale;
    a.causal            = c.causal;
    a.position_offset   = c.position_offset;
    return a;
}

bool mla_fast_vs_ref(const Case& c, bool& used_flash, cpu::AttentionPlan* plan_out = nullptr)
{
    auto [q_vec, q]   = test::make_random_tensor(c.qshape, -1.0f, 1.0f, 100);
    auto [kv_vec, kv] = test::make_random_tensor(c.kvshape, -1.0f, 1.0f, 101);

    std::vector<float> pkv_vec;
    TensorView past_kv;
    if (c.past > 0) {
        auto ps = c.kvshape;
        ps[c.kvshape.size() == 3 ? 1 : 2] = c.past;
        auto [pv, p] = test::make_random_tensor(ps, -1.0f, 1.0f, 103);
        pkv_vec = std::move(pv);
        past_kv = p;
    }

    const int64_t sk_all = (c.kvshape.size() == 3 ? c.kvshape[1] : c.kvshape[2]) + c.past;
    std::vector<float> m_vec;
    TensorView mask;
    if (c.mask) {
        const int64_t sq = c.qshape.size() == 3 ? c.qshape[1] : c.qshape[2];
        auto [mv, m] = test::make_random_tensor({sq, sk_all}, -2.0f, 2.0f, 102);
        m_vec = std::move(mv);
        mask  = m;
    }

    const auto attrs = attrs_of(c);
    auto op = MultiLatentAttention::create(attrs, Backend::CPU);

    std::vector<TensorDesc> descs = {q.desc(), kv.desc()};
    std::vector<TensorView> ins   = {q, kv};
    if (c.mask) {
        descs.push_back(mask.desc());
        ins.push_back(mask);
    } else if (c.past > 0) {
        // The optional slots are positional: an empty view marks the absent mask.
        descs.push_back(TensorDesc{});
        ins.push_back(TensorView{});
    }
    if (c.past > 0) {
        descs.push_back(past_kv.desc());
        ins.push_back(past_kv);
    }
    auto out_descs = op->getOutputTensorDesc(descs);
    const int64_t out_numel = out_descs[0].numel();

    cpu::AttentionConfig cfg;
    cfg.num_heads = c.heads;
    cfg.num_kv_heads = c.kv_heads;
    cfg.v_head_dim = c.vdim > 0 ? c.vdim : c.rank;
    cfg.v_is_k = true;
    cfg.causal = c.causal;
    cfg.position_offset = c.position_offset;
    const cpu::AttentionPlan plan = cpu::get_attention_plan(cfg, descs, out_descs);
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
    auto fast_out = test::make_planar(out_descs[0], fast_buf.data());
    op->compute(fast_out, ins, {}, nullptr);

    std::vector<float> ref_buf(static_cast<size_t>(out_numel));
    auto ref_out = test::make_planar(out_descs[0], ref_buf.data());
    nnops::backend::cpu::reference::multi_latent_attention_ref(attrs, ref_out, ins, {}, nullptr);

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

NNOPS_TEST(mla_output_shape_uses_value_width) {
    MultiLatentAttentionAttributes attrs;
    attrs.num_heads = 8;
    attrs.kv_num_heads = 1;
    attrs.kv_lora_rank = 24;
    attrs.qk_rope_head_dim = 8;
    attrs.qk_nope_head_dim = 32;

    auto op = MultiLatentAttention::create(attrs, Backend::CPU);

    const int64_t dkv = 24 + 8;
    TensorDesc q  = td({1, 7, 8 * dkv});
    TensorDesc kv = td({1, 11, 1 * dkv});
    const TensorDesc descs[] = {q, kv};
    auto outs = op->getOutputTensorDesc(descs);

    NNOPS_EXPECT_EQ(outs.size(), size_t(1));
    NNOPS_EXPECT_EQ(outs[0].rank, int64_t(3));
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t(1));
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t(7));
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t(8 * 24));   // H * kv_lora_rank

    // An explicit v_head_dim narrows the output.
    attrs.v_head_dim = 16;
    auto narrow = MultiLatentAttention::create(attrs, Backend::CPU);
    auto narrow_outs = narrow->getOutputTensorDesc(descs);
    NNOPS_EXPECT_EQ(narrow_outs[0].dims[2], int64_t(8 * 16));

    const TensorDesc out_descs[] = {narrow_outs[0]};
    NNOPS_EXPECT_TRUE(narrow->validate(descs, out_descs) == Status::ok);

    // The latent row must be kv_lora_rank + qk_rope_head_dim wide.
    TensorDesc kv_bad = td({1, 11, 1 * (dkv - 4)});
    const TensorDesc bad_descs[] = {q, kv_bad};
    NNOPS_EXPECT_TRUE(narrow->validate(bad_descs, out_descs) == Status::invalid_argument);

    // A value wider than the latent is rejected.
    attrs.v_head_dim = 32;
    auto too_wide = MultiLatentAttention::create(attrs, Backend::CPU);
    auto wide_outs = too_wide->getOutputTensorDesc(descs);
    const TensorDesc wide_descs[] = {wide_outs[0]};
    NNOPS_EXPECT_TRUE(too_wide->validate(descs, wide_descs) == Status::invalid_argument);
}

NNOPS_TEST(mla_needs_two_inputs) {
    MultiLatentAttentionAttributes attrs;
    auto op = MultiLatentAttention::create(attrs, Backend::CPU);

    const int64_t dkv = attrs.kv_lora_rank + attrs.qk_rope_head_dim;
    TensorDesc q = td({1, 4, attrs.num_heads * dkv});
    const TensorDesc one[] = {q};
    auto outs = op->getOutputTensorDesc(one);
    const TensorDesc out_descs[] = {outs[0]};
    NNOPS_EXPECT_TRUE(op->validate(one, out_descs) == Status::invalid_argument);
}

// ============================================================
// Fast path vs reference
// ============================================================

NNOPS_TEST(mla_matches_ref) {
    const Case cases[] = {
        // DeepSeek-ish ratio: 8 query heads over one latent head.
        {{1, 6, 8 * 24},  {1, 6, 1 * 24},  8, 1, 16, 8, 32, 0, 0.0f, true},
        // v_head_dim narrower than the latent.
        {{1, 6, 4 * 24},  {1, 6, 1 * 24},  4, 1, 16, 8, 32, 8, 0.0f, true},
        // Two latent heads under four query heads.
        {{2, 5, 4 * 24},  {2, 5, 2 * 24},  4, 2, 16, 8, 32, 0, 0.0f, false},
        // Explicit [B, H, S, D] / [B, H_kv, S, D].
        {{2, 3, 5, 24},   {2, 1, 5, 24},   3, 1, 16, 8, 32, 0, 0.0f, true},
        // Additive mask, causal off.
        {{1, 6, 4 * 24},  {1, 6, 1 * 24},  4, 1, 16, 8, 32, 0, 0.0f, false, true},
        // Mask + causal.
        {{1, 5, 4 * 24},  {1, 5, 1 * 24},  4, 1, 16, 8, 32, 0, 0.0f, true, true},
        // Decode shape: two new queries against a ten-token latent run.
        {{1, 2, 4 * 24},  {1, 10, 1 * 24}, 4, 1, 16, 8, 32, 0, 0.0f, true},
        // Non-default scale.
        {{1, 4, 2 * 40},  {1, 4, 1 * 40},  2, 1, 32, 8, 64, 0, 0.25f, true},
    };

    for (const auto& c : cases) {
        bool flash = false;
        NNOPS_EXPECT_TRUE(mla_fast_vs_ref(c, flash));
        (void)flash;
    }
}

NNOPS_TEST(mla_flash_matches_ref) {
    // Sq*Sk >= 65536 routes through the online-softmax flash path.
    const Case cases[] = {
        {{1, 256, 8 * 72}, {1, 256, 1 * 72}, 8, 1, 64, 8, 128, 0, 0.0f, true},
        {{1, 320, 4 * 40}, {1, 320, 2 * 40}, 4, 2, 32, 8, 64, 24, 0.0f, false},
    };

    for (const auto& c : cases) {
        bool flash = false;
        const bool ok = mla_fast_vs_ref(c, flash);
        NNOPS_EXPECT_TRUE(flash);
        NNOPS_EXPECT_TRUE(ok);
    }
}

// ============================================================
// Single-token decode: the GEMV route (Sq == 1, no mask)
// ============================================================

NNOPS_TEST(mla_decode_sq1_matches_ref) {
    // One latent row read twice — as keys over kv_lora_rank + rope, as values
    // over the leading v_head_dim — with no packing and no M == 1 tiles.
    const Case cases[] = {
        {{1, 1, 8 * 24},  {1, 12, 1 * 24}, 8, 1, 16, 8, 32, 0,  0.0f, true},
        {{1, 1, 4 * 24},  {1, 11, 1 * 24}, 4, 1, 16, 8, 32, 8,  0.0f, true},   // narrow value
        {{2, 2, 1, 24},   {2, 1, 7, 24},   2, 1, 16, 8, 32, 0,  0.0f, false},  // explicit layout
        {{1, 1, 2 * 40},  {1, 9, 1 * 40},  2, 1, 32, 8, 64, 0,  0.25f, true},  // non-default scale
        {{1, 1, 4 * 24},  {1, 1, 1 * 24},  4, 1, 16, 8, 32, 0,  0.0f, true},   // single key
    };

    for (const auto& c : cases) {
        bool flash = false;
        cpu::AttentionPlan plan;
        NNOPS_EXPECT_TRUE(mla_fast_vs_ref(c, flash, &plan));
        NNOPS_EXPECT_TRUE(plan.use_decode);
    }
}

NNOPS_TEST(mla_decode_sq1_masked_matches_ref) {
    // A mask keeps the call on the standard route; both routes must agree on
    // the same Sq == 1 shape.
    const Case cases[] = {
        {{1, 1, 8 * 24}, {1, 9, 1 * 24}, 8, 1, 16, 8, 32, 8, 0.0f, false, true},
        {{1, 1, 4 * 24}, {1, 9, 1 * 24}, 4, 1, 16, 8, 32, 0, 0.0f, true,  true},
    };

    for (const auto& c : cases) {
        bool flash = false;
        cpu::AttentionPlan plan;
        NNOPS_EXPECT_TRUE(mla_fast_vs_ref(c, flash, &plan));
        NNOPS_EXPECT_TRUE(!plan.use_decode);
    }
}

// ============================================================
// Chunked input: a past segment and an explicit position offset
// ============================================================

NNOPS_TEST(mla_step_matches_ref) {
    // The latent run is [past | new] — exactly the rows a paged cache gathers
    // into its two buffers — and position_offset places the chunk in the run.
    const Case cases[] = {
        // Single new latent row after a 25-row past: the GEMV decode route.
        {{1, 1, 8 * 24}, {1, 1, 1 * 24},  8, 1, 16, 8, 32, 0,  0.0f, true, false, 25, 25, 2},
        // Chunked prefill over four new rows.
        {{1, 4, 4 * 24}, {1, 4, 1 * 24},  4, 1, 16, 8, 32, 0,  0.0f, true, false, 37, 37, 1},
        // Explicit [B, H, S, D] / [B, H_kv, S, D].
        {{1, 2, 3, 24},  {1, 1, 3, 24},   2, 1, 16, 8, 32, 0,  0.0f, true, false, 5,  5,  1},
        // Narrow value width with a past.
        {{1, 3, 4 * 24}, {1, 3, 1 * 24},  4, 1, 16, 8, 32, 8,  0.0f, true, false, 9,  9,  1},
        // Non-tail chunk of a full run, and the same with a mask.
        {{1, 4, 4 * 24}, {1, 40, 1 * 24}, 4, 1, 16, 8, 32, 0,  0.0f, true, false, 0,  7,  0},
        {{1, 4, 4 * 24}, {1, 40, 1 * 24}, 4, 1, 16, 8, 32, 0,  0.0f, true, true,  0,  7,  0},
        // Masked step after a past: the mask spans the whole run.
        {{1, 1, 4 * 24}, {1, 1, 1 * 24},  4, 1, 16, 8, 32, 0,  0.0f, true, true,  13, 13, 1},
    };

    for (const auto& c : cases) {
        bool flash = false;
        NNOPS_EXPECT_TRUE(mla_fast_vs_ref(c, flash));
    }
}

// ============================================================
// f16 fast path
// ============================================================

NNOPS_TEST(mla_f16_matches_ref) {
    const Case cases[] = {
        {{1, 6, 8 * 24},   {1, 6, 1 * 24},  8, 1, 16, 8, 32, 0, 0.0f, true},
        {{2, 5, 4 * 24},   {2, 5, 2 * 24},  4, 2, 16, 8, 32, 8, 0.0f, false, true},
        {{1, 256, 8 * 72}, {1, 256, 1 * 72}, 8, 1, 64, 8, 128, 0, 0.0f, true},   // flash
    };

    const float tol = test::kF16AccumTol;
    for (const auto& c : cases) {
        auto [q32v, q32]   = test::make_random_tensor(c.qshape, -1.0f, 1.0f, 300);
        auto [kv32v, kv32] = test::make_random_tensor(c.kvshape, -1.0f, 1.0f, 301);

        const auto attrs = attrs_of(c);
        auto op = MultiLatentAttention::create(attrs, Backend::CPU);

        const TensorDesc descs32[] = {q32.desc(), kv32.desc()};
        const TensorView ins32[] = {q32, kv32};
        auto out_desc = op->getOutputTensorDesc(descs32)[0];
        const int64_t n = out_desc.numel();

        std::vector<float> ref_buf(static_cast<size_t>(n));
        auto ref_out = test::make_planar(out_desc, ref_buf.data());
        nnops::backend::cpu::reference::multi_latent_attention_ref(attrs, ref_out, ins32, {}, nullptr);

        auto q16v  = test::f32_to_f16(q32v);
        auto kv16v = test::f32_to_f16(kv32v);
        TensorView q16(std::span<const int64_t>(c.qshape), DataType::f16, q16v.data());
        TensorView kv16(std::span<const int64_t>(c.kvshape), DataType::f16, kv16v.data());

        // The output view must carry the f16 descriptor — one taken from the
        // f32 descriptors makes compute() reject the dtype and write nothing.
        const TensorDesc fdescs[] = {q16.desc(), kv16.desc()};
        const auto fout_desc = op->getOutputTensorDesc(fdescs)[0];
        std::vector<cpu::half> got_buf(static_cast<size_t>(n));
        auto got = test::make_planar(fout_desc, got_buf.data());
        const TensorView ins16[] = {q16, kv16};
        NNOPS_EXPECT_TRUE(op->compute(got, ins16) == Status::ok);

        const auto got_f32 = test::f16_to_f32(got_buf);
        float worst = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            worst = std::max(worst, std::abs(got_f32[static_cast<size_t>(i)] -
                                             ref_buf[static_cast<size_t>(i)]));
        }
        if (worst > tol) {
            throw std::runtime_error("mla f16 deviation " + std::to_string(worst) +
                                     " > tol " + std::to_string(tol));
        }
    }
}
