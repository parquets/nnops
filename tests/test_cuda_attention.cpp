/// @file test_cuda_attention.cpp
/// @brief CUDA attention against the CPU backend on the same inputs.
///
/// Every case builds one random host input list and runs the operator twice —
/// once with Backend::CPU and once with Backend::CUDA over device copies — and
/// compares the outputs. The CPU core is the oracle: it defines the two input
/// contracts (explicit vs KV cache), the causal/window/sink bounds, the mask
/// order, the int8-cache dequantization and the output-gate placement. Both
/// routes read the identical views, so a difference is a CUDA bug.
///
/// The cases follow the CUDA kernel's surface: the explicit form (merged,
/// BNSD and BSHD) with the additive bias, the keep-mask, per-batch lengths,
/// varlen and the gate; the cache form with per-batch lengths, position_ids,
/// window + sink, softcap, the head sink, the gate and an int8 cache (per
/// tensor and per token); MLA absorbed (narrow value, past segment, seqlens_k,
/// varlen) and non-absorbed. f16 is compared against the f32-tolerance CPU
/// route with the arch-aware kF16AccumTol.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/group_query_attention.hpp"
#include "nnops/ops/multi_latent_attention.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <span>
#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

// ============================================================
// Host-tensor helpers
// ============================================================

HostTensor i32_tensor(const std::vector<int32_t>& v)
{
    HostTensor t;
    t.bytes.resize(v.size() * sizeof(int32_t));
    std::memcpy(t.bytes.data(), v.data(), t.bytes.size());
    const int64_t d[1] = {static_cast<int64_t>(v.size())};
    t.view = TensorView(std::span<const int64_t>(d, 1), DataType::s32, t.bytes.data());
    return t;
}

HostTensor i64_tensor(const std::vector<int64_t>& v)
{
    HostTensor t;
    t.bytes.resize(v.size() * sizeof(int64_t));
    std::memcpy(t.bytes.data(), v.data(), t.bytes.size());
    const int64_t d[1] = {static_cast<int64_t>(v.size())};
    t.view = TensorView(std::span<const int64_t>(d, 1), DataType::s64, t.bytes.data());
    return t;
}

/// One element of a Q/K/V-like view, addressed like the operators do.
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

void quantize_cache(const std::vector<float>& x, int64_t B, int64_t Hkv, int64_t Skv, int64_t D,
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
        for (float v : x) {
            amax = std::max(amax, std::fabs(v));
        }
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

/// Wrap an f32 vector as a dense HostTensor.
HostTensor f32_tensor(std::vector<float> v, std::vector<int64_t> dims)
{
    HostTensor t;
    t.bytes.resize(v.size() * sizeof(float));
    std::memcpy(t.bytes.data(), v.data(), t.bytes.size());
    t.view = TensorView(dims, DataType::f32, t.bytes.data());
    return t;
}

/// The same vector rounded to `dt` and stored as that type.
HostTensor tensor_from_f32(const std::vector<float>& v, std::vector<int64_t> dims, DataType dt)
{
    if (dt == DataType::f32) {
        return f32_tensor(v, dims);
    }
    HostTensor t = make_host_zeroed(dims, dt, false);
    if (dt == DataType::f16) {
        auto* p = reinterpret_cast<nnops::backend::cpu::half*>(t.bytes.data());
        for (size_t i = 0; i < v.size(); ++i) {
            simd::s_store(&p[i], v[i]);
        }
    } else {
        auto* p = reinterpret_cast<nnops::backend::cpu::bfloat16*>(t.bytes.data());
        for (size_t i = 0; i < v.size(); ++i) {
            simd::s_store(&p[i], v[i]);
        }
    }
    return t;
}

// ============================================================
// Explicit-form builder
// ============================================================

enum class KpmKind { None, Lengths, Keep2, Keep3 };

struct ExplicitOpt {
    bool bias = false;
    bool head_bias = false;
    KpmKind kpm = KpmKind::None;
    std::vector<int32_t> lengths;
    bool gate = false;
    bool bshd = false;
    bool causal = false;
};

std::vector<HostTensor> build_explicit(const std::vector<int64_t>& qdims, int64_t Sk, int64_t H,
                                       int64_t Hkv, const ExplicitOpt& o, DataType dt,
                                       uint64_t seed)
{
    const bool merged = qdims.size() == 3;
    const int64_t B = qdims[0];
    const int64_t Sq = merged ? qdims[1] : (o.bshd ? qdims[1] : qdims[2]);
    const int64_t D = merged ? qdims[2] / H : qdims[3];

    std::vector<int64_t> kd;
    if (merged) {
        kd = {B, Sk, Hkv * D};
    } else if (o.bshd) {
        kd = {B, Sk, Hkv, D};
    } else {
        kd = {B, Hkv, Sk, D};
    }

    std::vector<HostTensor> in(9);
    in[0] = make_host_random(qdims, dt, false, seed, -1.0f, 1.0f);
    in[1] = make_host_random(kd, dt, false, seed + 1, -1.0f, 1.0f);
    in[2] = make_host_random(kd, dt, false, seed + 2, -1.0f, 1.0f);
    if (o.bias) {
        const std::vector<int64_t> md = o.head_bias ? std::vector<int64_t>{B, H, Sq, Sk}
                                                    : std::vector<int64_t>{Sq, Sk};
        in[3] = make_host_random(md, dt, false, seed + 3, -2.0f, 2.0f);
    }
    switch (o.kpm) {
    case KpmKind::Lengths:
        in[4] = i32_tensor(o.lengths);
        break;
    case KpmKind::Keep2: {
        HostTensor t = make_host_zeroed(std::vector<int64_t>{B, Sk}, DataType::s32, false);
        auto* p = reinterpret_cast<int32_t*>(t.bytes.data());
        for (int64_t i = 0; i < B * Sk; ++i) {
            p[i] = (i % 5 == 0) ? 0 : 1;
        }
        in[4] = std::move(t);
        break;
    }
    case KpmKind::Keep3: {
        HostTensor t = make_host_zeroed(std::vector<int64_t>{B, Sq, Sk}, DataType::s32, false);
        auto* p = reinterpret_cast<int32_t*>(t.bytes.data());
        for (int64_t i = 0; i < B * Sq * Sk; ++i) {
            p[i] = (i % 7 == 0) ? 0 : 1;
        }
        in[4] = std::move(t);
        break;
    }
    default:
        break;
    }
    if (o.gate) {
        in[8] = make_host_random(qdims, dt, false, seed + 6, -4.0f, 4.0f);
    }
    return in;
}

void expect_explicit(const std::vector<int64_t>& qdims, int64_t Sk, int64_t H, int64_t Hkv,
                     const ExplicitOpt& o, DataType dt, float tol_abs, float tol_rel,
                     const char* what)
{
    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = H;
    attrs.kv_num_heads = Hkv;
    attrs.causal = o.causal;
    attrs.head_layout = o.bshd ? AttentionHeadLayout::BSHD : AttentionHeadLayout::Auto
        ;
    if (o.gate) {
        attrs.output_gate = EpilogueActivateType::Silu;
    }
    auto in = build_explicit(qdims, Sk, H, Hkv, o, dt, 101);
    auto make = [attrs](Backend be) { return GroupQueryAttention::create(attrs, be); };
    run_op_close(make, in, tol_abs, tol_rel, what);
}

// ============================================================
// Cache-form builder
// ============================================================

struct CacheOpt {
    int64_t past = 0;
    std::vector<int32_t> seqlens;
    std::vector<int64_t> pos_ids;
    bool int8 = false;
    bool per_tensor = false;
    int64_t window = -1;
    int64_t sink = 0;
    float softcap = 0.0f;
    bool head_sink = false;
    bool gate = false;
    bool bshd = false;
    DataType dt = DataType::f32;
};

std::vector<HostTensor> build_cache(const std::vector<int64_t>& qdims, int64_t Sk_new, int64_t H,
                                    int64_t Hkv, const CacheOpt& o, uint64_t seed)
{
    const bool merged = qdims.size() == 3;
    const int64_t B = qdims[0];
    const int64_t Sq = merged ? qdims[1] : (o.bshd ? qdims[1] : qdims[2]);
    const int64_t D = merged ? qdims[2] / H : qdims[3];

    int64_t Skv = o.past + Sk_new;
    for (int32_t L : o.seqlens) {
        Skv = std::max<int64_t>(Skv, L);
    }

    // The step's K/V and the optional past, both merged [B, S, Hkv*D].
    HostTensor knew = make_host_random(std::vector<int64_t>{B, Sk_new, Hkv * D}, DataType::f32, false, seed + 1);
    HostTensor vnew = make_host_random(std::vector<int64_t>{B, Sk_new, Hkv * D}, DataType::f32, false, seed + 2);
    HostTensor kpast = make_host_random(std::vector<int64_t>{B, std::max<int64_t>(o.past, 1), Hkv * D}, DataType::f32,
                                        false, seed + 3);
    HostTensor vpast = make_host_random(std::vector<int64_t>{B, std::max<int64_t>(o.past, 1), Hkv * D}, DataType::f32,
                                        false, seed + 4);

    std::vector<float> ck(static_cast<size_t>(B * Hkv * Skv * D), 0.0f);
    std::vector<float> cv(static_cast<size_t>(B * Hkv * Skv * D), 0.0f);
    for (int64_t b = 0; b < B; ++b) {
        const int64_t L = o.seqlens.empty() ? Skv : o.seqlens[static_cast<size_t>(b)];
        const int64_t start = L - Sk_new;
        for (int64_t h = 0; h < Hkv; ++h) {
            for (int64_t t = 0; t < L; ++t) {
                for (int64_t d = 0; d < D; ++d) {
                    const float ke = (t < start)
                        ? head_elem(kpast.view, false, b, o.past - start + t, h, d, D)
                        : head_elem(knew.view, false, b, t - start, h, d, D);
                    const float ve = (t < start)
                        ? head_elem(vpast.view, false, b, o.past - start + t, h, d, D)
                        : head_elem(vnew.view, false, b, t - start, h, d, D);
                    const size_t idx = static_cast<size_t>(((b * Hkv + h) * Skv + t) * D + d);
                    ck[idx] = ke;
                    cv[idx] = ve;
                }
            }
        }
    }

    std::vector<HostTensor> in(9);
    in[0] = make_host_random(qdims, o.dt, false, seed, -1.0f, 1.0f);
    const std::vector<int64_t> cdims = {B, Hkv, Skv, D};
    if (o.int8) {
        std::vector<int8_t> qk, qv;
        std::vector<float> sk, sv;
        quantize_cache(ck, B, Hkv, Skv, D, o.per_tensor, qk, sk);
        quantize_cache(cv, B, Hkv, Skv, D, o.per_tensor, qv, sv);
        HostTensor tk = make_host_zeroed(cdims, DataType::s8, false);
        HostTensor tv = make_host_zeroed(cdims, DataType::s8, false);
        std::memcpy(tk.bytes.data(), qk.data(), qk.size());
        std::memcpy(tv.bytes.data(), qv.data(), qv.size());
        in[1] = std::move(tk);
        in[2] = std::move(tv);
        const std::vector<int64_t> sd =
            o.per_tensor ? std::vector<int64_t>{1} : std::vector<int64_t>{B, Hkv, Skv};
        HostTensor s1 = make_host_zeroed(sd, DataType::f32, false);
        HostTensor s2 = make_host_zeroed(sd, DataType::f32, false);
        std::memcpy(s1.bytes.data(), sk.data(), sk.size() * sizeof(float));
        std::memcpy(s2.bytes.data(), sv.data(), sv.size() * sizeof(float));
        in[5] = std::move(s1);
        in[6] = std::move(s2);
    } else {
        in[1] = tensor_from_f32(ck, cdims, o.dt);
        in[2] = tensor_from_f32(cv, cdims, o.dt);
    }
    if (!o.seqlens.empty()) {
        in[3] = i32_tensor(o.seqlens);
    }
    if (!o.pos_ids.empty()) {
        in[4] = i64_tensor(o.pos_ids);
    }
    if (o.head_sink) {
        in[7] = make_host_random(std::vector<int64_t>{H}, DataType::f32, false, seed + 7, -3.0f, 3.0f);
    }
    if (o.gate) {
        in[8] = make_host_random(qdims, o.dt, false, seed + 6, -4.0f, 4.0f);
    }
    return in;
}

void expect_cache(const std::vector<int64_t>& qdims, int64_t Sk_new, int64_t H, int64_t Hkv,
                  const CacheOpt& o, float tol_abs, float tol_rel, const char* what)
{
    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = H;
    attrs.kv_num_heads = Hkv;
    attrs.causal = true;
    attrs.softcap = o.softcap;
    attrs.local_window_size = o.window;
    attrs.sink_tokens = o.sink;
    attrs.head_layout = o.bshd ? AttentionHeadLayout::BSHD : AttentionHeadLayout::Auto;
    if (o.gate) {
        attrs.output_gate = EpilogueActivateType::Silu;
    }
    auto in = build_cache(qdims, Sk_new, H, Hkv, o, 201);
    auto make = [attrs](Backend be) { return GroupQueryAttention::create(attrs, be); };
    run_op_close(make, in, tol_abs, tol_rel, what);
}

// ============================================================
// MLA builder
// ============================================================

struct MlaOpt {
    bool absorbed = true;
    bool causal = true;
    int64_t past = 0;
    bool mask = false;
    std::vector<int32_t> seqlens;
    int64_t vdim = 0;
    int64_t position_offset = -1;
    DataType dt = DataType::f32;
};

void expect_mla(int64_t B, int64_t Sq, int64_t Sk, int64_t H, int64_t Hkv, int64_t rank,
                int64_t rope, const MlaOpt& o, float tol_abs, float tol_rel, const char* what)
{
    MultiLatentAttentionAttributes attrs;
    attrs.absorbed = o.absorbed;
    attrs.num_heads = H;
    attrs.kv_num_heads = Hkv;
    attrs.kv_lora_rank = rank;
    attrs.qk_rope_head_dim = rope;
    attrs.qk_nope_head_dim = o.absorbed ? rank : rank; // only the scale default reads it
    attrs.v_head_dim = o.vdim;
    attrs.causal = o.causal;
    attrs.position_offset = o.position_offset;

    const int64_t dk = o.absorbed ? (rank + rope) : (attrs.qk_nope_head_dim + rope);
    const int64_t vd = o.vdim > 0 ? o.vdim : rank;
    const int64_t Sk_all = Sk + o.past;

    std::vector<HostTensor> in(o.absorbed ? 8 : 10);
    in[0] = make_host_random(std::vector<int64_t>{B, Sq, H * dk}, o.dt, false, 301, -1.0f, 1.0f);
    in[1] = make_host_random(std::vector<int64_t>{B, Sk, Hkv * dk}, o.dt, false, 302, -1.0f, 1.0f);
    if (!o.absorbed) {
        in[2] = make_host_random(std::vector<int64_t>{B, Sk, Hkv * vd}, o.dt, false, 303, -1.0f, 1.0f);
    }
    const size_t mask_slot = o.absorbed ? 2 : 3;
    const size_t past_slot = o.absorbed ? 3 : 4;
    const size_t seq_slot = o.absorbed ? 4 : 6;
    if (o.mask) {
        in[mask_slot] = make_host_random(std::vector<int64_t>{Sq, Sk_all}, DataType::f32, false, 304, -2.0f, 2.0f);
    }
    if (o.past > 0) {
        in[past_slot] = make_host_random(std::vector<int64_t>{B, o.past, Hkv * dk}, o.dt, false, 305);
        if (!o.absorbed) {
            in[past_slot + 1] =
                make_host_random(std::vector<int64_t>{B, o.past, Hkv * vd}, o.dt, false, 306);
        }
    }
    if (!o.seqlens.empty()) {
        in[seq_slot] = i32_tensor(o.seqlens);
    }

    auto make = [attrs](Backend be) { return MultiLatentAttention::create(attrs, be); };
    run_op_close(make, in, tol_abs, tol_rel, what);
}

} // namespace

// ============================================================
// Tests
// ============================================================

NNOPS_TEST(cuda_gqa_explicit_matches_cpu) {
    const float tol = 2e-4f, rel = 2e-3f;
    // MQA, grouped and MHA ratios, merged.
    expect_explicit({1, 6, 8 * 8}, 6, 8, 1, {}, DataType::f32, tol, rel, "explicit MQA");
    expect_explicit({2, 5, 6 * 8}, 5, 6, 3, {}, DataType::f32, tol, rel, "explicit grouped");
    expect_explicit({1, 7, 4 * 8}, 7, 4, 4, {}, DataType::f32, tol, rel, "explicit MHA");
    // Cross attention: a shorter query run over a longer key run.
    expect_explicit({1, 3, 4 * 8}, 9, 4, 1, {}, DataType::f32, tol, rel, "explicit cross");
    // Rank-4 BNSH and BSHD.
    ExplicitOpt bnsd;
    expect_explicit({2, 6, 5, 8}, 5, 6, 3, bnsd, DataType::f32, tol, rel, "explicit BNSD");
    ExplicitOpt bshd;
    bshd.bshd = true;
    expect_explicit({2, 4, 6, 8}, 6, 6, 2, bshd, DataType::f32, tol, rel, "explicit BSHD");
    // f16.
    expect_explicit({1, 6, 8 * 16}, 6, 8, 2, {}, DataType::f16, kF16AccumTol, 5e-2f,
                    "explicit f16");
}

NNOPS_TEST(cuda_gqa_explicit_masks_matches_cpu) {
    const float tol = 2e-4f, rel = 2e-3f;
    ExplicitOpt bias;
    bias.bias = true;
    expect_explicit({1, 6, 4 * 8}, 6, 4, 2, bias, DataType::f32, tol, rel, "explicit bias");
    ExplicitOpt hb;
    hb.bias = true;
    hb.head_bias = true;
    expect_explicit({2, 5, 4 * 8}, 5, 4, 2, hb, DataType::f32, tol, rel, "explicit head bias");
    ExplicitOpt keep2;
    keep2.kpm = KpmKind::Keep2;
    expect_explicit({2, 5, 4 * 8}, 5, 4, 2, keep2, DataType::f32, tol, rel, "explicit keep2");
    ExplicitOpt keep3;
    keep3.kpm = KpmKind::Keep3;
    expect_explicit({2, 4, 4 * 8}, 6, 4, 2, keep3, DataType::f32, tol, rel, "explicit keep3");
    ExplicitOpt lens;
    lens.kpm = KpmKind::Lengths;
    lens.lengths = {7, 3};
    expect_explicit({2, 4, 4 * 8}, 7, 4, 2, lens, DataType::f32, tol, rel, "explicit lengths");
    ExplicitOpt gate;
    gate.gate = true;
    expect_explicit({1, 5, 4 * 8}, 5, 4, 2, gate, DataType::f32, tol, rel, "explicit gate");
    // Single-query-row explicit form takes the decode kernel, bias included.
    ExplicitOpt d;
    d.bias = true;
    expect_explicit({1, 1, 4 * 8}, 9, 4, 1, d, DataType::f32, tol, rel, "explicit decode");
}

NNOPS_TEST(cuda_gqa_long_shapes_matches_cpu) {
    const float tol = 5e-4f, rel = 3e-3f;
    // Several query blocks (Sq = 40) over several KV tiles (Sk = 100): the
    // block grid and the online-softmax rescaling both leave their single-tile
    // special case here.
    expect_explicit({1, 40, 4 * 32}, 100, 4, 2, {}, DataType::f32, tol, rel, "explicit long");
    expect_explicit({1, 40, 4 * 32}, 100, 4, 2, {}, DataType::f16, kF16AccumTol, 5e-2f,
                    "explicit long f16");
    CacheOpt pre;
    expect_cache({1, 40, 4 * 32}, 100, 4, 2, pre, tol, rel, "cache long prefill");
    // Decode against a 200-row run: many tiles, one valid query row.
    CacheOpt dec;
    dec.past = 200;
    expect_cache({1, 1, 4 * 32}, 1, 4, 2, dec, tol, rel, "cache long decode");
    // Window + sink over a long run: the hole between the sink band and the
    // window band is where a per-row bound bug shows up.
    CacheOpt win;
    win.past = 40;
    win.window = 8;
    win.sink = 3;
    expect_cache({1, 20, 4 * 16}, 48, 4, 2, win, tol, rel, "cache long window+sink");
    // int8 per token over a long run.
    CacheOpt q8;
    q8.past = 20;
    q8.int8 = true;
    expect_cache({1, 20, 4 * 16}, 40, 4, 2, q8, tol, rel, "int8 long");
}

// The 16-bit tensor-core tile (attention_tc.cuh): 128 query rows x 64 keys a
// block, 16 rows a warp, m16n8k16 with fp32 accumulators. It carries the shapes
// with a head and value width of 64, 128 or 192/128 and no score-side gather
// (bias, keep mask, softcap), so these cases are the ones that reach it.
NNOPS_TEST(cuda_gqa_tensor_core_matches_cpu) {
    // `causal` is the cache form here (validate() reads it as the route), so
    // the causal cases are cache calls over a written run.
    CacheOpt c;
    c.dt = DataType::f16;
    // Two query blocks over four key chunks.
    expect_cache({1, 256, 4 * 128}, 256, 4, 2, c, kF16AccumTol, 5e-2f, "tc f16 causal");
    // A ragged last query block and a run that is not a multiple of the chunk.
    expect_cache({1, 200, 4 * 128}, 200, 4, 2, c, kF16AccumTol, 5e-2f, "tc f16 causal ragged");
    // The 64-wide instantiation over a short run.
    expect_cache({1, 128, 4 * 64}, 192, 4, 2, c, kF16AccumTol, 5e-2f, "tc f16 d64");
    // One kv head over a run four times the query block (MQA, offset causal).
    CacheOpt mqa = c;
    mqa.past = 384;
    expect_cache({1, 128, 4 * 128}, 128, 4, 1, mqa, kF16AccumTol, 5e-2f, "tc f16 mqa");
    // The output gate rides the epilogue.
    CacheOpt gated = c;
    gated.gate = true;
    expect_cache({1, 128, 4 * 128}, 128, 4, 2, gated, kF16AccumTol, 5e-2f, "tc f16 gate");
    // The explicit form is the non-causal one: every key visible.
    expect_explicit({1, 128, 4 * 64}, 96, 4, 2, {}, DataType::f16, kF16AccumTol, 5e-2f,
                    "tc f16 full");
}

NNOPS_TEST(cuda_gqa_tensor_core_cache_matches_cpu) {
    // The mask arms a cache call can add to the tile: window + sink over a
    // prefilled past, and the gate.
    CacheOpt win;
    win.past = 40;
    win.window = 8;
    win.sink = 3;
    win.dt = DataType::f16;
    expect_cache({1, 128, 4 * 128}, 128, 4, 2, win, kF16AccumTol, 5e-2f, "tc f16 window+sink");
    CacheOpt gated;
    gated.past = 16;
    gated.gate = true;
    gated.dt = DataType::f16;
    expect_cache({1, 64, 4 * 128}, 64, 4, 2, gated, kF16AccumTol, 5e-2f, "tc f16 cache gate");
}

NNOPS_TEST(cuda_mla_tensor_core_matches_cpu) {
    // Dqk 192 (rank 128 + rope 64) by Dv 128 is the third instantiation, and the
    // absorbed form reads V off the key row.
    MlaOpt f16;
    f16.dt = DataType::f16;
    expect_mla(1, 128, 128, 4, 1, 128, 64, f16, kF16AccumTol, 5e-2f, "mla f16 tc");
    MlaOpt past = f16;
    past.past = 40;
    expect_mla(1, 64, 64, 2, 1, 128, 64, past, kF16AccumTol, 5e-2f, "mla f16 tc past");
}

NNOPS_TEST(cuda_gqa_tensor_core_bf16_matches_cpu) {
    // bf16 rides the same tile through the bf16 mma; the cache is built in f32
    // storage because run_op_bf16 rounds both sides itself.
    CacheOpt c;
    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = 4;
    attrs.kv_num_heads = 2;
    attrs.causal = true;
    auto in = build_cache({1, 128, 4 * 128}, 128, 4, 2, c, 211);
    auto make = [attrs](Backend be) { return GroupQueryAttention::create(attrs, be); };
    run_op_bf16(make, in, kF16AccumTol, 5e-2f, "tc bf16 causal");
}

NNOPS_TEST(cuda_gqa_bf16_matches_cpu) {
    auto in = build_explicit({1, 6, 4 * 8}, 6, 4, 2, {}, DataType::f32, 601);
    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = 4;
    attrs.kv_num_heads = 2;
    attrs.causal = false;
    auto make = [attrs](Backend be) { return GroupQueryAttention::create(attrs, be); };
    run_op_bf16(make, in, kF16AccumTol, 5e-2f, "gqa bf16");
}

NNOPS_TEST(cuda_gqa_cache_matches_cpu) {
    const float tol = 2e-4f, rel = 2e-3f;
    CacheOpt prefill;
    expect_cache({1, 6, 8 * 8}, 6, 8, 1, prefill, tol, rel, "cache prefill MQA");
    CacheOpt grouped;
    expect_cache({2, 5, 6 * 8}, 5, 6, 3, grouped, tol, rel, "cache prefill grouped");
    // Decode: one query row against a written run.
    CacheOpt dec;
    dec.past = 11;
    expect_cache({1, 1, 8 * 8}, 1, 8, 2, dec, tol, rel, "cache decode");
    // Per-batch run lengths (the query block sits at each batch's tail).
    CacheOpt sl;
    sl.past = 3;
    sl.seqlens = {9, 6};
    expect_cache({2, 6, 4 * 8}, 6, 4, 1, sl, tol, rel, "cache seqlens");
    // position_ids override the bottom-right diagonal.
    CacheOpt pos;
    pos.past = 11;
    pos.pos_ids = {10, 6};
    expect_cache({2, 1, 8 * 8}, 1, 8, 2, pos, tol, rel, "cache pos_ids");
    // BNSH query over the BNSH cache.
    CacheOpt bnsd;
    bnsd.past = 3;
    expect_cache({2, 6, 5, 8}, 5, 6, 3, bnsd, tol, rel, "cache BNSD");
}

NNOPS_TEST(cuda_gqa_cache_features_matches_cpu) {
    const float tol = 2e-4f, rel = 2e-3f;
    CacheOpt win;
    win.past = 7;
    win.window = 4;
    win.sink = 2;
    expect_cache({1, 6, 8 * 8}, 6, 8, 2, win, tol, rel, "cache window+sink");
    CacheOpt cap;
    cap.past = 5;
    cap.softcap = 2.0f;
    expect_cache({1, 5, 4 * 8}, 5, 4, 2, cap, tol, rel, "cache softcap");
    CacheOpt hs;
    hs.past = 5;
    hs.head_sink = true;
    expect_cache({1, 4, 4 * 8}, 4, 4, 2, hs, tol, rel, "cache head sink");
    CacheOpt gate;
    gate.past = 3;
    gate.gate = true;
    expect_cache({1, 4, 4 * 8}, 4, 4, 2, gate, tol, rel, "cache gate");

    // The decode kernel owns the Sq == 1 shape, so the feature flags need one
    // single-row case each.
    CacheOpt dcap;
    dcap.past = 9;
    dcap.softcap = 1.5f;
    expect_cache({1, 1, 4 * 8}, 1, 4, 2, dcap, tol, rel, "decode softcap");
    CacheOpt dwin;
    dwin.past = 20;
    dwin.window = 5;
    dwin.sink = 2;
    expect_cache({1, 1, 4 * 8}, 1, 4, 2, dwin, tol, rel, "decode window+sink");
    CacheOpt dhs;
    dhs.past = 9;
    dhs.head_sink = true;
    expect_cache({1, 1, 8 * 8}, 1, 8, 2, dhs, tol, rel, "decode head sink");
    CacheOpt dgate;
    dgate.past = 9;
    dgate.gate = true;
    expect_cache({1, 1, 4 * 8}, 1, 4, 2, dgate, tol, rel, "decode gate");
    CacheOpt dpos;
    dpos.past = 9;
    dpos.pos_ids = {4};
    expect_cache({1, 1, 4 * 8}, 1, 4, 2, dpos, tol, rel, "decode pos_ids");
}

NNOPS_TEST(cuda_gqa_int8_cache_matches_cpu) {
    const float tol = 1e-3f, rel = 5e-3f;
    CacheOpt per_tensor;
    per_tensor.past = 7;
    per_tensor.int8 = true;
    per_tensor.per_tensor = true;
    expect_cache({1, 6, 8 * 8}, 6, 8, 2, per_tensor, tol, rel, "int8 per tensor");
    CacheOpt per_token;
    per_token.past = 7;
    per_token.int8 = true;
    expect_cache({1, 5, 8 * 16}, 5, 8, 4, per_token, tol, rel, "int8 per token");
    CacheOpt dec;
    dec.past = 15;
    dec.int8 = true;
    expect_cache({1, 1, 8 * 8}, 1, 8, 2, dec, tol, rel, "int8 decode");
}

NNOPS_TEST(cuda_gqa_varlen_matches_cpu) {
    const float tol = 2e-4f, rel = 2e-3f;
    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = 4;
    attrs.kv_num_heads = 2;
    attrs.causal = false;

    const int64_t D = 8, total_q = 6, total_k = 9;
    auto in = build_explicit({1, total_q, 4 * D}, total_k, 4, 2, {}, DataType::f32, 401);
    in[6] = i32_tensor({0, 3, 6});
    in[7] = i32_tensor({0, 5, 9});
    auto make = [attrs](Backend be) { return GroupQueryAttention::create(attrs, be); };
    run_op_close(make, in, tol, rel, "gqa varlen");
}

NNOPS_TEST(cuda_mla_matches_cpu) {
    const float tol = 2e-4f, rel = 2e-3f;
    expect_mla(1, 6, 6, 4, 1, 16, 8, {}, tol, rel, "mla absorbed");
    MlaOpt narrow;
    narrow.vdim = 8;
    expect_mla(1, 5, 5, 4, 1, 16, 8, narrow, tol, rel, "mla absorbed narrow value");
    MlaOpt masked;
    masked.mask = true;
    expect_mla(2, 4, 5, 3, 1, 16, 8, masked, tol, rel, "mla absorbed mask");
    // Non-absorbed takes its own V.
    MlaOpt nonabs;
    nonabs.absorbed = false;
    expect_mla(1, 5, 5, 4, 2, 16, 8, nonabs, tol, rel, "mla non-absorbed");
    // A past segment spliced ahead of the step's rows.
    MlaOpt past;
    past.past = 4;
    expect_mla(1, 5, 4, 4, 1, 16, 8, past, tol, rel, "mla past segment");
    MlaOpt seq;
    seq.past = 3;
    seq.seqlens = {7, 5};
    expect_mla(2, 4, 4, 4, 1, 16, 8, seq, tol, rel, "mla seqlens_k");
    // A long causal run: several query blocks over several KV tiles.
    expect_mla(1, 40, 100, 4, 1, 16, 8, {}, 5e-4f, 3e-3f, "mla long");
    // Decode over a long run, and a wide value (Dv=512 exercises the decode
    // kernel's two-column-per-thread path).
    expect_mla(1, 1, 200, 4, 1, 16, 8, {}, 5e-4f, 3e-3f, "mla decode long");
    MlaOpt dm;
    dm.mask = true;
    expect_mla(1, 1, 6, 4, 1, 16, 8, dm, 5e-4f, 3e-3f, "mla decode mask");
    expect_mla(1, 1, 300, 4, 1, 512, 64, {}, 1e-3f, 5e-3f, "mla decode wide value");
}

#endif // NNOPS_HAS_CUDA
