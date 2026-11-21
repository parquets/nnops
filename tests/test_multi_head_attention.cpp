/// @file test_multi_head_attention.cpp
/// @brief MultiHeadAttention operator tests — the class API against the CPU
///        reference.
///
/// The operator shares GroupQueryAttention's two input contracts (H_kv == H
/// here): `causal = false` is explicit attention with an external mask, and
/// `causal = true` is KV-cache attention over a [B, H, S, D] cache the caller
/// wrote. The reference is f32 only, so f16 is compared against the f32
/// reference on the same values with the arch-aware kF16AccumTol.
///
/// The pack-scratch test calls get_attention_plan directly: the scratch is
/// sized from uniform nr_max-wide panels while the loops pack a last block
/// narrower than a full tile.

#include "nnops/ops/multi_head_attention.hpp"
#include "backend/cpu/attention/attention.h"   // get_attention_plan
#include "backend/cpu/matmul/matmul_helper.h"  // nr_max_flt / KC_* / mma_pack_align
#include "common/test_harness.hpp"
#include "common/cpu_pool.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"

#include <algorithm>
#include <array>
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
void multi_head_attention_ref(const MultiHeadAttentionAttributes& attrs,
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
    float scale = 0.0f;
    bool causal = true;            // selects the cache form
    bool mask = false;             // explicit form only
    int64_t past = 0;              // cache rows before the step's block
    int route = -1;                // 0 standard, 1 flash, 2 decode; -1 unchecked
    std::vector<int32_t> seqlens;  // per-batch valid run length; empty = full
    float softcap = 0.0f;
    int64_t window = 0;
    bool head_bias = false;
    EpilogueActivateType gate = EpilogueActivateType::None;
    AttentionHeadLayout layout = AttentionHeadLayout::Auto;
    bool int8 = false;
    bool per_tensor = false;
    int64_t sink = 0;              // always-visible leading cache rows
    std::vector<int64_t> pos_ids;  // per-batch first query row; empty = derived
};

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

struct Built {
    MultiHeadAttentionAttributes attrs;
    std::vector<std::vector<float>> keep_f32;
    std::vector<std::vector<int8_t>> keep_i8;
    std::vector<int32_t> sl;
    std::vector<int64_t> pos;
    std::vector<TensorDesc> descs;
    std::vector<TensorView> ins;
    bool ok = false;
};

void quantize_cache(const std::vector<float>& x, int64_t B, int64_t H, int64_t Skv, int64_t D,
                    bool per_tensor, std::vector<int8_t>& q, std::vector<float>& scale)
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
    scale.assign(static_cast<size_t>(B * H * Skv), 1.0f);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t t = 0; t < Skv; ++t) {
                const int64_t base = ((b * H + h) * Skv + t) * D;
                float amax = 0.0f;
                for (int64_t d = 0; d < D; ++d) {
                    amax = std::max(amax, std::fabs(x[static_cast<size_t>(base + d)]));
                }
                const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
                scale[static_cast<size_t>((b * H + h) * Skv + t)] = s;
                for (int64_t d = 0; d < D; ++d) {
                    q[static_cast<size_t>(base + d)] =
                        quant(x[static_cast<size_t>(base + d)], s);
                }
            }
        }
    }
}

Built build(const Case& c, uint64_t seed)
{
    Built b;
    b.attrs.num_heads = c.heads;
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
    const int64_t H = c.heads;
    const int64_t D = merged ? c.kshape[2] / H : c.kshape[3];
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
                ? std::vector<int64_t>{B, H, Sq, Sk_new}
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
            return b;
        }
        int64_t Skv = c.past + Sk_new;
        for (const int32_t L : c.seqlens) { Skv = std::max<int64_t>(Skv, L); }
        if (Skv < Sq) { return b; }

        const auto build_one = [&](const TensorView& nv, const TensorView& pv,
                                   std::vector<float>& out) -> bool {
            out.assign(static_cast<size_t>(B * H * Skv * D), 0.0f);
            for (int64_t bb = 0; bb < B; ++bb) {
                const int64_t L = c.seqlens.empty() ? Skv : c.seqlens[static_cast<size_t>(bb)];
                const int64_t start = L - Sk_new;
                if (L < Sq || start < 0 || start > c.past) { return false; }
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t t = 0; t < L; ++t) {
                        for (int64_t d = 0; d < D; ++d) {
                            const float e = (t < start)
                                ? head_elem(pv, bshd, bb, c.past - start + t, h, d, D)
                                : head_elem(nv, bshd, bb, t - start, h, d, D);
                            out[static_cast<size_t>(((bb * H + h) * Skv + t) * D + d)] = e;
                        }
                    }
                }
            }
            return true;
        };
        std::vector<float> ck, cv;
        if (!build_one(k, past_k, ck) || !build_one(v, past_v, cv)) { return b; }

        const std::vector<int64_t> cdims = {B, H, Skv, D};
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
            quantize_cache(ck, B, H, Skv, D, c.per_tensor, qk, sk);
            quantize_cache(cv, B, H, Skv, D, c.per_tensor, qv, sv);
            b.keep_i8.push_back(std::move(qk));
            b.ins[1] = TensorView(std::span<const int64_t>(cdims), DataType::s8,
                                  b.keep_i8.back().data());
            b.keep_i8.push_back(std::move(qv));
            b.ins[2] = TensorView(std::span<const int64_t>(cdims), DataType::s8,
                                  b.keep_i8.back().data());
            const std::vector<int64_t> sdim = c.per_tensor
                ? std::vector<int64_t>{1}
                : std::vector<int64_t>{B, H, Skv};
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

bool fast_vs_ref(const Case& c, bool& used_flash, cpu::AttentionPlan* plan_out = nullptr,
                 const CpuBackend* pool = nullptr)
{
    Built b = build(c, 100);
    if (!b.ok) {
        std::printf("        case could not be assembled\n");
        return false;
    }
    auto op = MultiHeadAttention::create(b.attrs, Backend::CPU);
    const auto out_descs = op->getOutputTensorDesc(b.descs);
    if (out_descs.size() != 1) {
        std::printf("        expected one output, got %zu\n", out_descs.size());
        return false;
    }
    const int64_t out_numel = out_descs[0].numel();

    cpu::AttentionConfig pcfg;
    pcfg.num_heads = c.heads;
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
    nnops::backend::cpu::reference::multi_head_attention_ref(b.attrs, ref_out, b.ins, {},
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
// Hand-checked shapes
// ============================================================

NNOPS_TEST(attention_basic_single_head) {
    const int64_t qshape[] = {1, 2, 4};  // B=1, H=1, Sq=2, D=4
    const int64_t kshape[] = {1, 2, 4};
    float q_data[8] = {1, 0, 0, 0, 0, 1, 0, 0};
    float k_data[8] = {1, 0, 0, 0, 0, 1, 0, 0};
    float v_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    TensorView q(qshape, DataType::f32, q_data);
    TensorView k(kshape, DataType::f32, k_data);
    TensorView v(kshape, DataType::f32, v_data);

    MultiHeadAttentionAttributes attrs;
    attrs.num_heads = 1;   // explicit form by default

    auto op = MultiHeadAttention::create(attrs, Backend::CPU);
    const TensorDesc desc_arr[] = {q.desc(), k.desc(), v.desc()};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs.size(), size_t(1));
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(3));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[2], int64_t(4));
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    auto out = test::make_channel_first(descs[0], out_buf.data());
    const TensorView ins[] = {q, k, v};
    NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::ok);

    // Non-causal: row 0 reads both keys. Scores are 0.5 and 0 (scale
    // 1/sqrt(4)), so p = [e^0.5, 1] / (e^0.5 + 1) and the output mixes v row 0
    // and v row 1.
    const float e = std::exp(0.5f);
    const float p0 = e / (e + 1.0f);
    const float p1 = 1.0f / (e + 1.0f);
    for (int i = 0; i < 4; ++i) {
        const float want = p0 * float(i + 1) + p1 * float(i + 5);
        NNOPS_EXPECT_NEAR(out_buf[static_cast<size_t>(i)], want, 1e-5f);
    }
    for (int i = 0; i < 8; ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[static_cast<size_t>(i)]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[static_cast<size_t>(i)]));
    }
}

NNOPS_TEST(attention_multi_head) {
    // B=1, H=2, S=2, D=2 (merged [1, 2, 4]).
    const int64_t shape[] = {1, 2, 4};
    auto [q_vec, q] = test::make_random_tensor(shape, -1.0f, 1.0f, 100);
    auto [k_vec, k] = test::make_random_tensor(shape, -1.0f, 1.0f, 101);
    auto [v_vec, v] = test::make_random_tensor(shape, -1.0f, 1.0f, 102);
    (void)q_vec; (void)k_vec; (void)v_vec;

    MultiHeadAttentionAttributes attrs;
    attrs.num_heads = 2;
    auto op = MultiHeadAttention::create(attrs, Backend::CPU);
    const TensorDesc desc_arr[] = {q.desc(), k.desc(), v.desc()};
    auto descs = op->getOutputTensorDesc(desc_arr);
    NNOPS_EXPECT_EQ(descs[0].dims[2], int64_t(4));

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    auto out = test::make_channel_first(descs[0], out_buf.data());
    const TensorView ins[] = {q, k, v};
    NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::ok);
    for (float f : out_buf) { NNOPS_EXPECT_TRUE(!std::isnan(f) && !std::isinf(f)); }
}

// ============================================================
// Explicit form (ViT / encoder)
// ============================================================

NNOPS_TEST(attention_tiled_matches_ref) {
    const Case cases[] = {
        {{1, 4, 4},    {1, 4, 4},    1, 0.0f, false},
        {{2, 8, 64},   {2, 8, 64},   4, 0.0f, false},
        {{1, 16, 32},  {1, 16, 32},  1, 0.0f, false},
        {{2, 6, 24},   {2, 6, 24},   3, 0.5f, false, true},
        {{1, 32, 128}, {1, 32, 128}, 8, 0.0f, false, true},
        // head_dim just above Kc (128): GEMM1's K loop splits 128 + 64.
        {{1, 8, 192},  {1, 8, 192},  1, 0.0f, false},
        {{1, 8, 256},  {1, 8, 256},  1, 0.0f, false, true},
        {{1, 4, 512},  {1, 4, 512},  1, 0.0f, false},
        // Cross attention: a shorter query run.
        {{1, 3, 32},   {1, 9, 32},   2, 0.0f, false},
    };

    for (const auto& c : cases) {
        bool flash = false;
        NNOPS_EXPECT_TRUE(fast_vs_ref(c, flash));
    }
}

NNOPS_TEST(attention_tiled_explicit_matches_ref) {
    // Explicit [B, H, S, D] layout.
    Case c = {};
    c.qshape = {2, 3, 5, 8};
    c.kshape = {2, 3, 5, 8};
    c.heads = 3;
    c.causal = false;
    bool flash = false;
    NNOPS_EXPECT_TRUE(fast_vs_ref(c, flash));
}

NNOPS_TEST(attention_decode_sq1_matches_ref) {
    // One query row against a longer key run, both forms: explicit
    // cross-attention and cache decode.
    Case explicit_case = {};
    explicit_case.qshape = {1, 1, 8};
    explicit_case.kshape = {1, 12, 8};
    explicit_case.heads = 1;
    explicit_case.causal = false;
    {
        bool flash = false;
        cpu::AttentionPlan plan;
        NNOPS_EXPECT_TRUE(fast_vs_ref(explicit_case, flash, &plan));
        NNOPS_EXPECT_TRUE(plan.use_decode);
    }

    Case cross = {};
    cross.qshape = {2, 1, 64};
    cross.kshape = {2, 11, 64};
    cross.heads = 4;
    cross.causal = false;
    cross.scale = 0.3f;
    {
        bool flash = false;
        NNOPS_EXPECT_TRUE(fast_vs_ref(cross, flash));
    }

    Case cache = {};
    cache.qshape = {2, 1, 32};
    cache.kshape = {2, 1, 32};
    cache.heads = 2;
    cache.past = 11;
    {
        bool flash = false;
        cpu::AttentionPlan plan;
        NNOPS_EXPECT_TRUE(fast_vs_ref(cache, flash, &plan));
        NNOPS_EXPECT_TRUE(plan.use_decode);
    }
}

NNOPS_TEST(attention_keep_mask_matches_ref) {
    // The int32 keep-mask forms: [B, Sk] and [B, Sq, Sk].
    const int64_t B = 2, H = 4, Sq = 3, Sk = 5, D = 8;

    MultiHeadAttentionAttributes attrs;
    attrs.num_heads = H;
    auto op = MultiHeadAttention::create(attrs, Backend::CPU);

    for (int form = 0; form < 2; ++form) {
        auto [qv, q] = test::make_random_tensor({B, Sq, H * D}, -1.0f, 1.0f, 330);
        auto [kv, k] = test::make_random_tensor({B, Sk, H * D}, -1.0f, 1.0f, 331);
        auto [vv, v] = test::make_random_tensor({B, Sk, H * D}, -1.0f, 1.0f, 332);
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
        } else {
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
        NNOPS_EXPECT_TRUE(op->compute(got_out, ins) == Status::ok);

        std::vector<float> want(got.size());
        auto want_out = test::make_channel_first(out_descs[0], want.data());
        nnops::backend::cpu::reference::multi_head_attention_ref(attrs, want_out, ins, {},
                                                                nullptr);
        for (size_t i = 0; i < got.size(); ++i) { NNOPS_EXPECT_NEAR(got[i], want[i], 1e-4f); }
    }
}

// ============================================================
// Cache form (decoder)
// ============================================================

NNOPS_TEST(attention_cache_matches_ref) {
    const Case cases[] = {
        // Prefill.
        {{1, 4, 4 * 8},  {1, 4, 4 * 8},  4, 0.0f, true},
        {{2, 5, 6 * 8},  {2, 5, 6 * 8},  6, 0.0f, true},
        // Explicit [B, H, S, D] query over the BNSH cache.
        {{2, 4, 3, 8},   {2, 4, 3, 8},   4, 0.0f, true},
        // Decode against a written run.
        {{1, 1, 8 * 16}, {1, 1, 8 * 16}, 8, 0.0f, true, false, 12, 2},
        // Per-batch run lengths.
        {{2, 4, 4 * 8},  {2, 4, 4 * 8},  4, 0.0f, true, false, 2, -1, {6, 4}},
        // Softcap, window.
        {{1, 4, 4 * 8},  {1, 4, 4 * 8},  4, 0.0f, true, false, 0, -1, {}, 2.0f},
        {{1, 4, 4 * 8},  {1, 4, 4 * 8},  4, 0.0f, true, false, 0, -1, {}, 0.0f, 3},
    };

    for (const auto& c : cases) {
        bool flash = false;
        NNOPS_EXPECT_TRUE(fast_vs_ref(c, flash));
    }
}

NNOPS_TEST(attention_window_sink_matches_ref) {
    // Streaming sinks: the first `sink` cache rows stay visible outside the
    // sliding window on every route.
    const Case cases[] = {
        {{1, 8, 4 * 8}, {1, 8, 4 * 8}, 4, 0.0f, true, false, 0, 0, {}, 0.0f, 3, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, false, false, 4},
        {{1, 1, 8 * 8}, {1, 1, 8 * 8}, 8, 0.0f, true, false, 500, 2, {}, 0.0f, 16, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, false, false, 4},
        {{1, 256, 2 * 64}, {1, 256, 2 * 64}, 2, 0.0f, true, false, 128, 1, {}, 0.0f, 64, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, false, false, 4},
    };

    for (const auto& c : cases) {
        bool flash = false;
        NNOPS_EXPECT_TRUE(fast_vs_ref(c, flash));
    }
}

NNOPS_TEST(attention_cache_split_matches_ref) {
    // A long decode run: single-threaded it walks several kDecodeChunk blocks
    // through the online softmax; with a worker grid the plan cuts it into KV
    // slices and merges the partial states.
    {
        Case c = {};
        c.qshape = {1, 1, 8 * 8};
        c.kshape = {1, 1, 8 * 8};
        c.heads = 8;
        c.past = 2047;
        bool flash = false;
        NNOPS_EXPECT_TRUE(fast_vs_ref(c, flash));
    }

    SimplePool pool(8);
    Case c = {};
    c.qshape = {1, 1, 4 * 32};
    c.kshape = {1, 1, 4 * 32};
    c.heads = 4;
    c.past = 2047;
    bool flash = false;
    cpu::AttentionPlan plan;
    NNOPS_EXPECT_TRUE(fast_vs_ref(c, flash, &plan, &pool.cpu));
    NNOPS_EXPECT_TRUE(plan.use_decode);
    NNOPS_EXPECT_TRUE(plan.decode_splits > 1);
}

NNOPS_TEST(attention_position_ids_matches_ref) {
    // Per-batch first-row positions, the same int64 [B] input RoPE takes.
    const Case cases[] = {
        // Equal to kv_seqlens - Sq: identical to the derived path.
        {{2, 4, 4 * 8}, {2, 4, 4 * 8}, 4, 0.0f, true, false, 4, 0, {}, 0.0f, 0, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, false, false, 0, {4, 4}},
        // One position later than the cache tail.
        {{2, 4, 4 * 8}, {2, 4, 4 * 8}, 4, 0.0f, true, false, 4, 0, {}, 0.0f, 0, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, false, false, 0, {5, 4}},
        // Visible prefix shorter than the query block.
        {{2, 4, 4 * 8}, {2, 4, 4 * 8}, 4, 0.0f, true, false, 4, 0, {6, 4}, 0.0f, 0, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, false, false, 0, {4, 2}},
        // Decode with a window measured from the position.
        {{2, 1, 8 * 8}, {2, 1, 8 * 8}, 8, 0.0f, true, false, 6, 2, {}, 0.0f, 4, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, false, false, 0, {6, 5}},
        // Flash route.
        {{1, 256, 2 * 64}, {1, 256, 2 * 64}, 2, 0.0f, true, false, 128, 1, {}, 0.0f, 0, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, false, false, 0, {150}},
    };

    for (const auto& c : cases) {
        bool flash = false;
        NNOPS_EXPECT_TRUE(fast_vs_ref(c, flash));
    }
}

// ============================================================
// FlashAttention path
// ============================================================

NNOPS_TEST(attention_flash_matches_ref) {
    const Case cases[] = {
        {{1, 256, 64},   {1, 256, 64},  2, 0.0f, false},
        {{1, 320, 32},   {1, 320, 32},  1, 0.0f, false, true},
        {{1, 512, 128},  {1, 512, 128}, 1, 0.0f, false},
        {{1, 256, 256},  {1, 256, 256}, 1, 0.0f, false},
        // Cache form, large prefill.
        {{1, 256, 64},   {1, 256, 64},  2, 0.0f, true},
    };

    for (const auto& c : cases) {
        bool flash = false;
        const bool ok = fast_vs_ref(c, flash);
        NNOPS_EXPECT_TRUE(flash);
        NNOPS_EXPECT_TRUE(ok);
    }
}

// ============================================================
// f16 and int8
// ============================================================

NNOPS_TEST(attention_f16_matches_ref) {
    const Case cases[] = {
        {{1, 8, 64},     {1, 8, 64},    4, 0.0f, false},
        {{1, 256, 64},   {1, 256, 64},  2, 0.0f, true},
        {{1, 1, 8 * 16}, {1, 1, 8 * 16}, 8, 0.0f, true, false, 9, 2},
        // f16 query over an int8 cache.
        {{1, 8, 4 * 8},  {1, 8, 4 * 8},  4, 0.0f, true, false, 0, -1, {}, 0.0f, 0, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, true},
    };

    const float tol = test::kF16AccumTol;
    for (const auto& c : cases) {
        Built b = build(c, 300);
        NNOPS_EXPECT_TRUE(b.ok);
        auto op = MultiHeadAttention::create(b.attrs, Backend::CPU);

        const auto out_descs = op->getOutputTensorDesc(b.descs);
        const int64_t n = out_descs[0].numel();

        std::vector<float> ref_buf(static_cast<size_t>(n));
        auto ref_out = test::make_channel_first(out_descs[0], ref_buf.data());
        nnops::backend::cpu::reference::multi_head_attention_ref(b.attrs, ref_out, b.ins, {},
                                                                 nullptr);

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

NNOPS_TEST(attention_int8_cache_matches_ref) {
    const Case cases[] = {
        // Standard, per tensor.
        {{1, 4, 4 * 8}, {1, 4, 4 * 8}, 4, 0.0f, true, false, 0, 0, {}, 0.0f, 0, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, true},
        // Decode.
        {{1, 1, 4 * 8}, {1, 1, 4 * 8}, 4, 0.0f, true, false, 6, 2, {}, 0.0f, 0, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, true},
        // Flash.
        {{1, 256, 64},  {1, 256, 64},  2, 0.0f, true, false, 0, 1, {}, 0.0f, 0, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, true},
        // Per tensor.
        {{1, 5, 4 * 8}, {1, 5, 4 * 8}, 4, 0.0f, true, false, 0, 0, {}, 0.0f, 0, false,
         EpilogueActivateType::None, AttentionHeadLayout::Auto, true, true},
    };

    for (const auto& c : cases) {
        bool flash = false;
        NNOPS_EXPECT_TRUE(fast_vs_ref(c, flash));
    }
}

NNOPS_TEST(attention_f16_routes_to_fast_path) {
    // f16 takes the tiled path, not the naive fallback.
    Case c = {};
    c.qshape = {1, 8, 64};
    c.kshape = {1, 8, 64};
    c.heads = 4;
    c.causal = false;
    Built b = build(c, 310);
    NNOPS_EXPECT_TRUE(b.ok);
    cpu::AttentionConfig pcfg;
    pcfg.num_heads = c.heads;
    pcfg.bias_slot = 3;
    for (auto& d : b.descs) { d.dtype = DataType::f16; }
    const auto plan = cpu::get_attention_plan(pcfg, b.descs, {}, 1, false);
    NNOPS_EXPECT_TRUE(plan.workspace_size > 0);
}

// ============================================================
// Output gate
// ============================================================

NNOPS_TEST(attention_output_gate_hand) {
    const std::vector<int64_t> shape = {2, 3, 2 * 8};

    Case plain_case = {};
    plain_case.qshape = shape;
    plain_case.kshape = shape;
    plain_case.heads = 2;
    plain_case.causal = false;
    Built plain_b = build(plain_case, 200);
    NNOPS_EXPECT_TRUE(plain_b.ok);

    Case gate_case = plain_case;
    gate_case.gate = EpilogueActivateType::Sigmoid;
    Built gate_b = build(gate_case, 200);
    NNOPS_EXPECT_TRUE(gate_b.ok);

    auto op = MultiHeadAttention::create(plain_b.attrs, Backend::CPU);
    const auto out_descs = op->getOutputTensorDesc(plain_b.descs);
    const int64_t n = out_descs[0].numel();
    std::vector<float> plain(static_cast<size_t>(n));
    auto plain_out = test::make_channel_first(out_descs[0], plain.data());
    NNOPS_EXPECT_TRUE(op->compute(plain_out, plain_b.ins) == Status::ok);

    auto gop = MultiHeadAttention::create(gate_b.attrs, Backend::CPU);
    std::vector<float> got(static_cast<size_t>(n));
    auto got_out = test::make_channel_first(out_descs[0], got.data());
    NNOPS_EXPECT_TRUE(gop->compute(got_out, gate_b.ins) == Status::ok);

    const float* gate = gate_b.ins[8].ptr<float>();
    for (int64_t i = 0; i < n; ++i) {
        const float want = plain[static_cast<size_t>(i)] / (1.0f + std::exp(-gate[i]));
        NNOPS_EXPECT_NEAR(got[static_cast<size_t>(i)], want, 1e-5f);
    }
}

// ============================================================
// Layouts and slot contract
// ============================================================

NNOPS_TEST(attention_bshd_matches_ref) {
    Case e = {};
    e.qshape = {2, 3, 4, 8};
    e.kshape = {2, 5, 4, 8};
    e.heads = 4;
    e.causal = false;
    e.layout = AttentionHeadLayout::BSHD;
    {
        bool flash = false;
        NNOPS_EXPECT_TRUE(fast_vs_ref(e, flash));
    }

    Case k = {};
    k.qshape = {2, 3, 4, 8};
    k.kshape = {2, 5, 4, 8};
    k.heads = 4;
    k.causal = true;
    k.layout = AttentionHeadLayout::BSHD;
    {
        bool flash = false;
        NNOPS_EXPECT_TRUE(fast_vs_ref(k, flash));
    }
}

NNOPS_TEST(attention_slot_contract) {
    MultiHeadAttentionAttributes attrs;
    attrs.num_heads = 4;
    auto op = MultiHeadAttention::create(attrs, Backend::CPU);

    const TensorDesc q = td({2, 3, 4 * 8});
    const TensorDesc k = td({2, 5, 4 * 8});
    const TensorDesc out = td({2, 3, 4 * 8});
    const TensorDesc outs[] = {out};
    {
        SlotDescs s;
        s.put(0, q).put(1, k).put(2, k).put(3, td({3, 3}));   // wrong bias shape
        NNOPS_EXPECT_TRUE(op->validate(s.d, outs) == Status::invalid_argument);
    }
    {
        SlotDescs s;
        s.put(0, q).put(1, k).put(2, k);
        const TensorDesc o2[] = {out, td({2, 4, 5, 8})};
        NNOPS_EXPECT_TRUE(op->validate(s.d, o2) == Status::invalid_argument);
    }
    {
        SlotDescs s;
        s.put(0, q).put(1, k).put(2, k);
        NNOPS_EXPECT_TRUE(op->validate(s.d, outs) == Status::ok);
    }
    {
        // The cache form wants a [B, H, S, D] pair.
        MultiHeadAttentionAttributes cached;
        cached.num_heads = 4;
        cached.causal = true;
        auto cop = MultiHeadAttention::create(cached, Backend::CPU);
        SlotDescs s;
        s.put(0, q).put(1, k).put(2, k);
        NNOPS_EXPECT_TRUE(cop->validate(s.d, outs) == Status::invalid_argument);
        SlotDescs t;
        t.put(0, td({2, 2, 4 * 8})).put(1, td({2, 4, 6, 8})).put(2, td({2, 4, 6, 8}));
        NNOPS_EXPECT_TRUE(cop->validate(t.d, outs) == Status::ok);
    }
}

// ============================================================
// Pack-scratch sizing
// ============================================================

namespace {

/// Elements a packed-B block of @p n columns occupies at the plan's stride.
template <class T>
int64_t pack_demand_elems(int64_t n, int64_t D, int64_t Sk)
{
    constexpr int nr_max = cpu::nr_max_flt<T>();
    constexpr int64_t kc = std::is_same_v<T, cpu::half> ? cpu::KC_F16 : cpu::KC_F32;
    constexpr int esz = static_cast<int>(sizeof(T));
    const int64_t k_bound = std::min<int64_t>(kc, std::max(D, Sk));
    const int64_t ldd = static_cast<int64_t>(cpu::mma_pack_align(
        static_cast<int>(nr_max * k_bound * esz)) / esz);
    const int64_t panels = (n + nr_max - 1) / nr_max;
    return panels * ldd;
}

}  // anonymous namespace

NNOPS_TEST(attention_pack_scratch_covers_partial_block) {
    struct Case { bool f16; int64_t Sq, D, Sk; };
    const Case cases[] = {
        {false,  8, 128, 7679},   // standard path, Sk = 10*nc1 - 1
        {false, 33, 128, 2047},   // flash
        {false, 65, 256, 1023},   // flash, D = 2*Kc
        {true,  17, 128, 4095},   // flash, f16
    };

    for (const auto& c : cases) {
        TensorDesc q, k, v;
        q.rank = k.rank = v.rank = 3;
        q.dtype = k.dtype = v.dtype = c.f16 ? DataType::f16 : DataType::f32;
        q.dims = {1, c.Sq, c.D};
        k.dims = v.dims = {1, c.Sk, c.D};

        cpu::AttentionConfig cfg;
        cfg.num_heads = 1;

        const TensorDesc descs[] = {q, k, v};
        const cpu::AttentionPlan plan = cpu::get_attention_plan(cfg, descs, {}, 1, false);
        const int64_t esz = c.f16 ? 2 : 4;

        const int64_t tile = plan.use_flash ? plan.Bc : plan.nc1;
        const int64_t prefix = plan.use_flash
            ? static_cast<int64_t>(plan.Br) * 2
              + static_cast<int64_t>(plan.Br) * plan.Bc
              + static_cast<int64_t>(plan.Br) * c.D
            : c.Sq * c.Sk;
        const int64_t capacity = static_cast<int64_t>(plan.workspace_size) / esz - prefix;

        const int64_t actual = ((c.Sk - 1) % tile) + 1;
        const int64_t demand = c.f16 ? pack_demand_elems<cpu::half>(actual, c.D, c.Sk)
                                     : pack_demand_elems<float>(actual, c.D, c.Sk);
        NNOPS_EXPECT_TRUE(demand <= capacity);
    }
}
