/// Unit tests for Attention operator — class API with CPU reference.
///
/// All tests use the Attention class interface:
///   1. auto op = Attention::create(attrs, Backend::CPU);
///   2. auto d = tv.desc();  // store desc in lvalue
///   3. const TensorDesc arr[] = {d_q, d_k, d_v};  // C array
///      auto descs = op->getOutputTensorDesc(arr);
///   4. Validate descs[0].rank / dims / layout / dtype
///   5. Create output TensorView via nnops::test::make_planar
///   6. const TensorView ins[] = {q, k, v}; op->compute(output, ins);

#include "nnops/ops/attention.hpp"
#include "backend/cpu/attention.h"      // get_attention_plan — to pin the f16 fast path
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"

#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

using namespace nnops;

// ============================================================
// Basic single-head attention
// ============================================================

NNOPS_TEST(attention_basic_single_head) {
    // B=1, H=1, Sq=2, Sk=2, D=4
    // Test scaled dot-product attention without mask
    const int64_t qshape[] = {1, 2, 4};  // [B, S, D]
    const int64_t kshape[] = {1, 2, 4};
    const int64_t vshape[] = {1, 2, 4};

    float q_data[8] = {1,0,0,0, 0,1,0,0};
    float k_data[8] = {1,0,0,0, 0,1,0,0};
    float v_data[8] = {1,2,3,4, 5,6,7,8};

    TensorView q(qshape, DataType::f32, q_data);
    TensorView k(kshape, DataType::f32, k_data);
    TensorView v(vshape, DataType::f32, v_data);

    AttentionAttributes attrs;
    attrs.num_heads = 1;

    auto op = Attention::create(attrs, Backend::CPU);

    auto d_q = q.desc();
    auto d_k = k.desc();
    auto d_v = v.desc();
    const TensorDesc desc_arr[] = {d_q, d_k, d_v};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs.size(), size_t(1));
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(3));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[2], int64_t(4));
    NNOPS_EXPECT_EQ(static_cast<int>(descs[0].layout), static_cast<int>(TensorLayout::NCHW));
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    auto out = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {q, k, v};
    op->compute(out, ins);

    // QK^T: [[1,0],[0,1]] → scores after softmax ≈ [[0.73, 0.27], [0.27, 0.73]]
    // attn @ V should be close to V with some mixing
    for (int i = 0; i < 8; ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

// ============================================================
// Multi-head attention
// ============================================================

NNOPS_TEST(attention_multi_head) {
    // B=1, H=2, Sq=2, Sk=2, D=2 (merged: [B, S, H*D] = [1, 2, 4])
    const int64_t qshape[] = {1, 2, 4};
    const int64_t kshape[] = {1, 2, 4};
    const int64_t vshape[] = {1, 2, 4};

    std::vector<float> q_buf(8);
    std::vector<float> k_buf(8);
    std::vector<float> v_buf(8);

    // Identity-like Q and K (each head gets its own subspace)
    for (int i = 0; i < 8; ++i) { q_buf[i] = (i % 3 == 0) ? 1.0f : 0.0f; }
    for (int i = 0; i < 8; ++i) { k_buf[i] = q_buf[i]; }
    for (int i = 0; i < 8; ++i) { v_buf[i] = static_cast<float>(i + 1); }

    TensorView q(qshape, DataType::f32, q_buf.data());
    TensorView k(kshape, DataType::f32, k_buf.data());
    TensorView v(vshape, DataType::f32, v_buf.data());

    AttentionAttributes attrs;
    attrs.num_heads = 2;

    auto op = Attention::create(attrs, Backend::CPU);

    auto d_q = q.desc();
    auto d_k = k.desc();
    auto d_v = v.desc();
    const TensorDesc desc_arr[] = {d_q, d_k, d_v};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs.size(), size_t(1));
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(3));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[2], int64_t(4));
    NNOPS_EXPECT_EQ(static_cast<int>(descs[0].layout), static_cast<int>(TensorLayout::NCHW));
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    auto out = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {q, k, v};
    op->compute(out, ins);

    for (int i = 0; i < 8; ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

// ============================================================
// Random data (causal)
// ============================================================

NNOPS_TEST(attention_random) {
    auto [q_vec, q] = test::make_random_tensor({2, 1, 16});  // B=2, H=1, S=4, D=4 (merged)
    auto [k_vec, k] = test::make_random_tensor({2, 1, 16});
    auto [v_vec, v] = test::make_random_tensor({2, 1, 16});

    AttentionAttributes attrs;
    attrs.num_heads = 1;

    auto op = Attention::create(attrs, Backend::CPU);

    auto d_q = q.desc();
    auto d_k = k.desc();
    auto d_v = v.desc();
    const TensorDesc desc_arr[] = {d_q, d_k, d_v};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs.size(), size_t(1));
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(3));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[2], int64_t(16));
    NNOPS_EXPECT_EQ(static_cast<int>(descs[0].layout), static_cast<int>(TensorLayout::NCHW));
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    auto out = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {q, k, v};
    op->compute(out, ins);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

// ============================================================
// Tiled attention fast path (vs reference)
// ============================================================

// Reference kernel for comparison (test-local declaration, following the
// test_conv2d.cpp pattern).
namespace nnops::backend::cpu::reference {
void attention_ref(const AttentionAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx,
                   void* workspace);
}

namespace {

/// Run the fast path and the reference, and report whether every output element
/// agrees within (rtol, atol).
bool attention_fast_vs_ref(const AttentionAttributes& attrs,
                           const TensorView& q, const TensorView& k, const TensorView& v,
                           const TensorView* mask,
                           float rtol, float atol)
{
    std::vector<TensorDesc> descs = {q.desc(), k.desc(), v.desc()};
    std::vector<TensorView> ins   = {q, k, v};
    if (mask) {
        descs.push_back(mask->desc());
        ins.push_back(*mask);
    }

    auto op = Attention::create(attrs, Backend::CPU);
    auto out_descs = op->getOutputTensorDesc(descs);
    const int64_t out_numel = out_descs[0].numel();

    // Fast path (scratch pooled internally by the kernel).
    std::vector<float> fast_buf(static_cast<size_t>(out_numel));
    auto fast_out = test::make_planar(out_descs[0], fast_buf.data());
    op->compute(fast_out, ins, {}, nullptr);

    // Reference baseline.
    std::vector<float> ref_buf(static_cast<size_t>(out_numel));
    auto ref_out = test::make_planar(out_descs[0], ref_buf.data());
    nnops::backend::cpu::reference::attention_ref(attrs, ref_out, ins, {}, nullptr);

    for (int64_t i = 0; i < out_numel; ++i) {
        const float diff = std::abs(fast_buf[i] - ref_buf[i]);
        const float thr  = atol + rtol * std::max(std::abs(fast_buf[i]), std::abs(ref_buf[i]));
        if (diff > thr) {
            return false;
        }
    }
    return true;
}

}  // anonymous namespace

NNOPS_TEST(attention_tiled_matches_ref) {
    // Merged [B, S, H*D] self-attention across a spread of batch / heads /
    // sequence / head-dim sizes, exercising multi-k-block and multi-n-block
    // tiling paths, with and without a mask.
    struct Case { std::array<int64_t, 3> qshape; std::array<int64_t, 3> kshape;
                  std::array<int64_t, 3> vshape; int64_t heads; float scale; bool mask; };
    const Case cases[] = {
        {{1, 4, 4},     {1, 4, 4},     {1, 4, 4},     1, 0.0f, false},
        {{2, 8, 64},    {2, 8, 64},    {2, 8, 64},    4, 0.0f, false},
        {{1, 16, 32},   {1, 16, 32},   {1, 16, 32},   1, 0.0f, false},
        {{2, 6, 24},    {2, 6, 24},    {2, 6, 24},    3, 0.5f, true},
        {{1, 32, 128},  {1, 32, 128},  {1, 32, 128},  8, 0.0f, true},
        // head_dim > Kc: exercises the multi-k-block GEMM1 / multi-n-block GEMM2
        {{1, 4, 512},   {1, 4, 512},   {1, 4, 512},   1, 0.0f, false},
    };

    for (const auto& c : cases) {
        auto [q_vec, q] = test::make_random_tensor(c.qshape, -1.0f, 1.0f, 200);
        auto [k_vec, k] = test::make_random_tensor(c.kshape, -1.0f, 1.0f, 201);
        auto [v_vec, v] = test::make_random_tensor(c.vshape, -1.0f, 1.0f, 202);

        std::vector<float> m_vec;
        TensorView mask;
        if (c.mask) {
            // Reference reads the mask flat [Sq, Sk]; keep it 2-D here.
            auto [mv, m] = test::make_random_tensor(
                {c.qshape[1], c.kshape[1]}, -2.0f, 2.0f, 203);
            m_vec = std::move(mv);
            mask  = m;
        }

        AttentionAttributes attrs;
        attrs.num_heads = c.heads;
        attrs.scale     = c.scale;

        const TensorView* mp = c.mask ? &mask : nullptr;
        NNOPS_EXPECT_TRUE(attention_fast_vs_ref(attrs, q, k, v, mp, 1e-3f, 1e-4f));
    }
}

NNOPS_TEST(attention_tiled_explicit_matches_ref) {
    // Explicit [B, H, S, D] layout (one head per (batch, head) block).
    const int64_t qshape[] = {2, 3, 5, 8};
    const int64_t kshape[] = {2, 3, 5, 8};
    const int64_t vshape[] = {2, 3, 5, 8};

    auto [q_vec, q] = test::make_random_tensor(qshape, -1.0f, 1.0f, 210);
    auto [k_vec, k] = test::make_random_tensor(kshape, -1.0f, 1.0f, 211);
    auto [v_vec, v] = test::make_random_tensor(vshape, -1.0f, 1.0f, 212);

    AttentionAttributes attrs;
    attrs.num_heads = 3;

    NNOPS_EXPECT_TRUE(attention_fast_vs_ref(attrs, q, k, v, nullptr, 1e-3f, 1e-4f));
}

// ============================================================
// FlashAttention path (large sequence length)
// ============================================================

NNOPS_TEST(attention_flash_matches_ref) {
    // Large-sequence cases that route through the tiled FlashAttention path
    // (Sq × Sk ≥ 65536), exercising single- and multi-KV-block online softmax,
    // with and without an additive mask.
    struct Case { std::array<int64_t, 3> qshape; int64_t heads; bool mask; };
    const Case cases[] = {
        {{1, 256, 64},  2, false},   // Sq*Sk = 65536, H*D = 64
        {{1, 320, 32},  1, true},    // Sq*Sk = 102400, masked
        {{1, 512, 128}, 1, false},   // Sq*Sk = 262144, D=128 → multi-KV-block
    };

    for (const auto& c : cases) {
        auto [q_vec, q] = test::make_random_tensor(c.qshape, -1.0f, 1.0f, 300);
        auto [k_vec, k] = test::make_random_tensor(c.qshape, -1.0f, 1.0f, 301);
        auto [v_vec, v] = test::make_random_tensor(c.qshape, -1.0f, 1.0f, 302);

        std::vector<float> m_vec;
        TensorView mask;
        if (c.mask) {
            auto [mv, m] = test::make_random_tensor(
                {c.qshape[1], c.qshape[1]}, -2.0f, 2.0f, 303);
            m_vec = std::move(mv);
            mask  = m;
        }

        AttentionAttributes attrs;
        attrs.num_heads = c.heads;

        const TensorView* mp = c.mask ? &mask : nullptr;
        NNOPS_EXPECT_TRUE(attention_fast_vs_ref(attrs, q, k, v, mp, 1e-3f, 1e-4f));
    }
}

// ============================================================
// f16 fast path (vs the f32 reference on the same data)
// ============================================================

namespace {

using nnops::backend::cpu::half;

/// f16 fast path vs the f32 reference computed on the *same* underlying data —
/// the pattern shared by the other f16 GEMM tests (matmul, conv2d). Returns the
/// max absolute element-wise deviation so the caller can compare it to a
/// tolerance and report the margin.
float attention_f16_fast_vs_ref(const AttentionAttributes& attrs,
                                const TensorView& q32, const TensorView& k32, const TensorView& v32,
                                const TensorView* m32,
                                const TensorView& q16, const TensorView& k16, const TensorView& v16,
                                const TensorView* m16)
{
    auto op = Attention::create(attrs, Backend::CPU);

    // f32 reference baseline.
    std::vector<TensorDesc> rdescs = {q32.desc(), k32.desc(), v32.desc()};
    std::vector<TensorView> rins   = {q32, k32, v32};
    if (m32) { rdescs.push_back(m32->desc()); rins.push_back(*m32); }
    auto rout_descs = op->getOutputTensorDesc(rdescs);
    const int64_t out_numel = rout_descs[0].numel();
    std::vector<float> ref_buf(static_cast<size_t>(out_numel));
    auto ref_out = test::make_planar(rout_descs[0], ref_buf.data());
    nnops::backend::cpu::reference::attention_ref(attrs, ref_out, rins, {}, nullptr);

    // f16 fast path (scratch pooled internally by the kernel).
    std::vector<TensorDesc> fdescs = {q16.desc(), k16.desc(), v16.desc()};
    std::vector<TensorView> fins   = {q16, k16, v16};
    if (m16) { fdescs.push_back(m16->desc()); fins.push_back(*m16); }
    auto fout_descs = op->getOutputTensorDesc(fdescs);
    std::vector<half> fast_buf(static_cast<size_t>(out_numel));
    auto fast_out = test::make_planar(fout_descs[0], fast_buf.data());
    op->compute(fast_out, fins, {}, nullptr);

    const auto fast_f32 = test::f16_to_f32(fast_buf);
    float max_dev = 0.0f;
    for (int64_t i = 0; i < out_numel; ++i) {
        max_dev = std::max(max_dev,
            std::abs(fast_f32[static_cast<size_t>(i)] - ref_buf[static_cast<size_t>(i)]));
    }
    return max_dev;
}

/// Build the f32 + f16 views for one case (merged rank-3 or explicit rank-4,
/// decided by the shapes), run fast-vs-ref and return the max deviation.
/// @p mshape empty means "no mask".
float run_f16_case(const AttentionAttributes& attrs,
                   const std::vector<int64_t>& qshape,
                   const std::vector<int64_t>& kshape,
                   const std::vector<int64_t>& vshape,
                   const std::vector<int64_t>& mshape,
                   uint64_t seed)
{
    auto [q32v, q32] = test::make_random_tensor(qshape, -1.0f, 1.0f, seed);
    auto [k32v, k32] = test::make_random_tensor(kshape, -1.0f, 1.0f, seed + 1);
    auto [v32v, v32] = test::make_random_tensor(vshape, -1.0f, 1.0f, seed + 2);

    const bool has_mask = !mshape.empty();
    std::vector<float> m32v;
    std::vector<half>  m16v;
    TensorView m32, m16;
    if (has_mask) {
        auto [mv, m] = test::make_random_tensor(mshape, -2.0f, 2.0f, seed + 3);
        m32v = std::move(mv);
        m32  = m;
        m16v = test::f32_to_f16(m32v);
        m16  = TensorView(std::span<const int64_t>(mshape), DataType::f16, m16v.data());
    }

    auto q16v = test::f32_to_f16(q32v);
    auto k16v = test::f32_to_f16(k32v);
    auto v16v = test::f32_to_f16(v32v);
    TensorView q16(std::span<const int64_t>(qshape), DataType::f16, q16v.data());
    TensorView k16(std::span<const int64_t>(kshape), DataType::f16, k16v.data());
    TensorView v16(std::span<const int64_t>(vshape), DataType::f16, v16v.data());

    return attention_f16_fast_vs_ref(attrs, q32, k32, v32, has_mask ? &m32 : nullptr,
                                     q16, k16, v16, has_mask ? &m16 : nullptr);
}

}  // anonymous namespace

NNOPS_TEST(attention_f16_routes_to_fast_path) {
    // f16 must resolve a real plan (workspace_size != 0), i.e. take the tiled
    // fast path rather than falling through to the f32-only reference — which
    // would silently interpret the f16 buffers as f32.
    const int64_t qshape[] = {1, 8, 32};
    auto [q16v, q16] = test::make_random_f16_tensor(qshape, -1.0f, 1.0f, 400);
    auto [k16v, k16] = test::make_random_f16_tensor(qshape, -1.0f, 1.0f, 401);
    auto [v16v, v16] = test::make_random_f16_tensor(qshape, -1.0f, 1.0f, 402);

    AttentionAttributes attrs;
    attrs.num_heads = 2;

    const TensorDesc descs[] = {q16.desc(), k16.desc(), v16.desc()};
    const auto plan = nnops::backend::cpu::get_attention_plan(attrs, descs, {}, 1, false);
    NNOPS_EXPECT_TRUE(plan.workspace_size != 0);
}

NNOPS_TEST(attention_f16_matches_ref) {
    // f16 tiled standard path (Sq*Sk < kFlashMinScores): merged and explicit
    // layouts, multi-head, an additive mask, and head_dim > Kc (multi-k-block
    // GEMM1). The f32 reference on the same data is the ground truth.
    struct Case { std::vector<int64_t> qshape, kshape, vshape, mshape;
                  int64_t heads; float scale; };
    const Case cases[] = {
        // merged [B, S, H*D], single head.
        {{1, 4, 8},    {1, 4, 8},    {1, 4, 8},    {},        1, 0.0f},
        // merged, multi-head (H=2 -> per-head D=32).
        {{2, 8, 64},   {2, 8, 64},   {2, 8, 64},   {},        2, 0.0f},
        // merged, additive mask + non-default scale, H=3 (per-head D=8).
        {{2, 6, 24},   {2, 6, 24},   {2, 6, 24},   {6, 6},    3, 0.5f},
        // merged, H=8 (per-head D=16), masked.
        {{1, 16, 128}, {1, 16, 128}, {1, 16, 128}, {16, 16},  8, 0.0f},
        // head_dim (512) > Kc (128): multi-k-block GEMM1.
        {{1, 4, 512},  {1, 4, 512},  {1, 4, 512},  {},        1, 0.0f},
        // explicit [B, H, S, D] layout, H=3, per-head D=8.
        {{2, 3, 5, 8}, {2, 3, 5, 8}, {2, 3, 5, 8}, {},        3, 0.0f},
    };

    const float tol = nnops::test::kF16AccumTol;
    float worst = 0.0f;
    for (const auto& c : cases) {
        AttentionAttributes attrs;
        attrs.num_heads = c.heads;
        attrs.scale     = c.scale;
        const float dev = run_f16_case(attrs, c.qshape, c.kshape, c.vshape, c.mshape, 500);
        worst = std::max(worst, dev);
    }
    std::printf("      f16 attention (standard): max deviation %.4f (tol %.3f)\n", worst, tol);
    if (worst > tol) {
        throw std::runtime_error("f16 attention standard-path deviation " +
                                 std::to_string(worst) + " > tol " + std::to_string(tol));
    }
}

NNOPS_TEST(attention_f16_flash_matches_ref) {
    // Large Sq*Sk routes through the f16 FlashAttention path (online softmax
    // with f16 running max/sum), single- and multi-KV-block, with and without a
    // mask.
    struct Case { std::vector<int64_t> qshape, mshape; int64_t heads; };
    const Case cases[] = {
        {{1, 256, 64},  {},        2},   // Sq*Sk = 65536, H*D = 64
        {{1, 320, 32},  {320, 320},1},   // masked
        {{1, 512, 128}, {},        1},   // D=128 -> multi-KV-block
    };

    const float tol = nnops::test::kF16AccumTol;
    float worst = 0.0f;
    for (const auto& c : cases) {
        AttentionAttributes attrs;
        attrs.num_heads = c.heads;
        worst = std::max(worst, run_f16_case(attrs, c.qshape, c.qshape, c.qshape, c.mshape, 600));
    }
    std::printf("      f16 attention (flash): max deviation %.4f (tol %.3f)\n", worst, tol);
    if (worst > tol) {
        throw std::runtime_error("f16 attention flash-path deviation " +
                                 std::to_string(worst) + " > tol " + std::to_string(tol));
    }
}
