/// @file bench_cuda_attention.cpp
/// @brief CUDA attention benchmark — GroupQueryAttention (prefill and decode)
///        and MultiLatentAttention, f32 and f16.
///
/// The rows are the two shapes an LLM actually runs: a causal prefill (many
/// query rows over a growing run) and a decode step (one query row over the
/// whole cache). Both are cache-form GQA, so the K/V tensors are the run the
/// caller wrote. The FLOP count is the standard 2·H·Sq·Sk·D per product, with
/// the causal prefill discounted by half for the skipped upper triangle.
///
/// The decode rows are here rather than hidden because a decode step is the
/// worst case for the tiled kernels (one query row leaves fifteen of sixteen
/// row slots idle) and the one the dedicated kernel — flash decoding's split-KV
/// over a register dot, with a block per head group — exists to carry. The arms
/// below are the two knobs that shape it: how far the run is split, and how many
/// query heads one block serves.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/group_query_attention.hpp"
#include "nnops/ops/multi_latent_attention.hpp"
#include "detail/bench_knobs.hpp"
#include "common/bench_harness.hpp"
#include "common/cuda_bench_helper.hpp"
#include "common/cuda_test_helper.hpp"

#include <cstdio>
#include <string>
#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

std::string gqa_geom(int64_t B, int64_t H, int64_t Hkv, int64_t D, int64_t Sq, int64_t Sk,
                     DataType dt)
{
    char buf[160];
    std::snprintf(buf, sizeof(buf), "B=%lld Sq=%lld Sk=%lld H=%lld Hkv=%lld D=%lld %s",
                  static_cast<long long>(B), static_cast<long long>(Sq),
                  static_cast<long long>(Sk), static_cast<long long>(H),
                  static_cast<long long>(Hkv), static_cast<long long>(D),
                  dt == DataType::f32 ? "f32" : "f16");
    return std::string(buf);
}

void run_gqa(const char* tag, int64_t B, int64_t H, int64_t Hkv, int64_t D, int64_t Sq,
             int64_t Sk, bool causal, DataType dt, double flops)
{
    std::vector<HostTensor> in(9);
    in[0] = make_host_random(std::vector<int64_t>{B, Sq, H * D}, dt, false, 11, -0.5f, 0.5f);
    in[1] = make_host_random(std::vector<int64_t>{B, Hkv, Sk, D}, dt, false, 12, -0.5f, 0.5f);
    in[2] = make_host_random(std::vector<int64_t>{B, Hkv, Sk, D}, dt, false, 13, -0.5f, 0.5f);

    std::vector<DeviceTensor> dins;
    for (auto& t : in) {
        dins.push_back(to_device(t));
    }
    std::vector<TensorView> dv;
    for (auto& d : dins) {
        dv.push_back(d.view);
    }
    HostTensor got = make_host_zeroed(std::vector<int64_t>{B, Sq, H * D}, dt, false);
    CudaBuffer obuf(got.bytes.size());
    TensorView ov(got.view.shape_span(), dt, obuf.data(), got.view.pitch());

    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = H;
    attrs.kv_num_heads = Hkv;
    attrs.causal = causal;
    auto op = GroupQueryAttention::create(attrs, Backend::CUDA);

    const double ms = cuda_median_ms([&] { cuda_launch(*op, ov, dv); });
    print_cuda_row(tag, gqa_geom(B, H, Hkv, D, Sq, Sk, dt), ms, cuda_gflops(ms, flops));
}

void run_mla(const char* tag, int64_t B, int64_t H, int64_t Hkv, int64_t rank, int64_t rope,
             int64_t vdim, int64_t Sq, int64_t Sk, bool causal, DataType dt, double flops)
{
    const int64_t dk = rank + rope;
    std::vector<HostTensor> in(8);
    in[0] = make_host_random(std::vector<int64_t>{B, Sq, H * dk}, dt, false, 21, -0.5f, 0.5f);
    in[1] = make_host_random(std::vector<int64_t>{B, Sk, Hkv * dk}, dt, false, 22, -0.5f, 0.5f);

    std::vector<DeviceTensor> dins;
    for (auto& t : in) {
        dins.push_back(to_device(t));
    }
    std::vector<TensorView> dv;
    for (auto& d : dins) {
        dv.push_back(d.view);
    }
    const int64_t vd = vdim > 0 ? vdim : rank;
    HostTensor got = make_host_zeroed(std::vector<int64_t>{B, Sq, H * vd}, dt, false);
    CudaBuffer obuf(got.bytes.size());
    TensorView ov(got.view.shape_span(), dt, obuf.data(), got.view.pitch());

    MultiLatentAttentionAttributes attrs;
    attrs.absorbed = true;
    attrs.num_heads = H;
    attrs.kv_num_heads = Hkv;
    attrs.kv_lora_rank = rank;
    attrs.qk_rope_head_dim = rope;
    attrs.qk_nope_head_dim = 128;
    attrs.v_head_dim = vdim;
    attrs.causal = causal;
    auto op = MultiLatentAttention::create(attrs, Backend::CUDA);

    const double ms = cuda_median_ms([&] { cuda_launch(*op, ov, dv); });
    char geom[180];
    std::snprintf(geom, sizeof(geom), "B=%lld Sq=%lld Sk=%lld H=%lld Hkv=%lld Dkv=%lld %s",
                  static_cast<long long>(B), static_cast<long long>(Sq),
                  static_cast<long long>(Sk), static_cast<long long>(H),
                  static_cast<long long>(Hkv), static_cast<long long>(dk),
                  dt == DataType::f32 ? "f32" : "f16");
    print_cuda_row(tag, geom, ms, cuda_gflops(ms, flops));
}

} // namespace

NNOPS_BENCH(cuda_attention_gqa_prefill) {
    const int64_t B = 1, H = 32, Hkv = 8, D = 128, S = 512;
    // Causal prefill skips the upper triangle: 2·H·Sq·Sk·D over both products,
    // halved.
    const double flops = 2.0 * B * H * S * S * D;
    print_cuda_header("GQA causal prefill");
    run_gqa("gqa prefill", B, H, Hkv, D, S, S, true, DataType::f32, flops);
    run_gqa("gqa prefill", B, H, Hkv, D, S, S, true, DataType::f16, flops);
    run_gqa("gqa prefill MQA", B, H, 1, D, S, S, true, DataType::f32, flops);
}

/// The int8-cache arms. Same run of keys, same query, three operators: the fp
/// route (which widens the s8 cache on read), the dynamic-query int8 route, and
/// that route with the integer P·V. The ratio isolates each step — the first
/// answers "what is the widening tax", the second what the 32-k product adds.
void run_gqa_i8(const char* tag, int64_t B, int64_t H, int64_t Hkv, int64_t D, int64_t Sq,
                int64_t Sk, DataType dt, double flops, bool q8, bool pv8)
{
    std::vector<int8_t> qk(static_cast<size_t>(B * Hkv * Sk * D));
    std::vector<int8_t> qv(qk.size());
    std::vector<float> sk(static_cast<size_t>(B * Hkv * Sk)), sv(sk.size());
    HostTensor kf = make_host_random(std::vector<int64_t>{B, Hkv, Sk, D}, DataType::f32, false, 12,
                                     -0.5f, 0.5f);
    HostTensor vf = make_host_random(std::vector<int64_t>{B, Hkv, Sk, D}, DataType::f32, false, 13,
                                     -0.5f, 0.5f);
    const float* kp = kf.view.ptr<float>();
    const float* vp = vf.view.ptr<float>();
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < Hkv; ++h) {
            for (int64_t t = 0; t < Sk; ++t) {
                const int64_t row = ((b * Hkv + h) * Sk + t) * D;
                float ak = 0.0f, av = 0.0f;
                for (int64_t d = 0; d < D; ++d) {
                    ak = std::max(ak, std::fabs(kp[row + d]));
                    av = std::max(av, std::fabs(vp[row + d]));
                }
                const float sc_k = ak > 0.0f ? ak / 127.0f : 1.0f;
                const float sc_v = av > 0.0f ? av / 127.0f : 1.0f;
                const size_t srow = static_cast<size_t>((b * Hkv + h) * Sk + t);
                sk[srow] = sc_k;
                sv[srow] = sc_v;
                for (int64_t d = 0; d < D; ++d) {
                    qk[static_cast<size_t>(row + d)] =
                        static_cast<int8_t>(std::nearbyintf(kp[row + d] / sc_k));
                    qv[static_cast<size_t>(row + d)] =
                        static_cast<int8_t>(std::nearbyintf(vp[row + d] / sc_v));
                }
            }
        }
    }

    std::vector<HostTensor> in(9);
    in[0] = make_host_random(std::vector<int64_t>{B, Sq, H * D}, dt, false, 11, -0.5f, 0.5f);
    HostTensor tk = make_host_zeroed(std::vector<int64_t>{B, Hkv, Sk, D}, DataType::s8, false);
    HostTensor tv = make_host_zeroed(std::vector<int64_t>{B, Hkv, Sk, D}, DataType::s8, false);
    std::memcpy(tk.bytes.data(), qk.data(), qk.size());
    std::memcpy(tv.bytes.data(), qv.data(), qv.size());
    in[1] = std::move(tk);
    in[2] = std::move(tv);
    HostTensor s1 = make_host_zeroed(std::vector<int64_t>{B, Hkv, Sk}, DataType::f32, false);
    HostTensor s2 = make_host_zeroed(std::vector<int64_t>{B, Hkv, Sk}, DataType::f32, false);
    std::memcpy(s1.bytes.data(), sk.data(), sk.size() * sizeof(float));
    std::memcpy(s2.bytes.data(), sv.data(), sv.size() * sizeof(float));
    in[5] = std::move(s1);
    in[6] = std::move(s2);
    std::vector<DeviceTensor> dins;
    for (auto& t : in) {
        dins.push_back(to_device(t));
    }
    std::vector<TensorView> dv;
    for (auto& d : dins) {
        dv.push_back(d.view);
    }
    HostTensor got = make_host_zeroed(std::vector<int64_t>{B, Sq, H * D}, dt, false);
    CudaBuffer obuf(got.bytes.size());
    TensorView ov(got.view.shape_span(), dt, obuf.data(), got.view.pitch());

    GroupQueryAttentionAttributes attrs;
    attrs.num_heads = H;
    attrs.kv_num_heads = Hkv;
    attrs.causal = true;
    attrs.quantize_query = q8;
    attrs.quantize_pv = pv8;
    auto op = GroupQueryAttention::create(attrs, Backend::CUDA);
    const double ms = cuda_median_ms([&] { cuda_launch(*op, ov, dv); });
    print_cuda_row(tag, gqa_geom(B, H, Hkv, D, Sq, Sk, dt), ms, cuda_gflops(ms, flops));
}

NNOPS_BENCH(cuda_attention_gqa_decode) {
    const int64_t B = 1, H = 32, Hkv = 8, D = 128, Sk = 2048;
    // One query row reads the whole run: 2 products × 2·H·Sk·D.
    const double flops = 4.0 * B * H * Sk * D;
    print_cuda_header("GQA decode (one query row)");
    run_gqa("gqa decode", B, H, Hkv, D, 1, Sk, true, DataType::f32, flops);
    run_gqa("gqa decode", B, H, Hkv, D, 1, Sk, true, DataType::f16, flops);
    run_gqa("gqa decode MQA", B, H, 1, D, 1, Sk, true, DataType::f32, flops);
}

// The decode head-group arms. The query heads of a K/V head read the same rows,
// so a block serving `hb` of them cuts its requests by `hb`; the sweep shows
// what that buys against the single-head block the automatic choice starts
// from, on the GQA pair and on MLA's sixteen-heads-to-one-K/V shape.
NNOPS_BENCH(cuda_attention_decode_headgroup) {
    print_cuda_header("decode head-group arms");
    const double gq = 4.0 * 32 * 2048 * 128;
    for (int hb : {1, 2, 4}) {
        nnops::detail::cuda_attn_headbatch.store(hb);
        char tag[64];
        std::snprintf(tag, sizeof(tag), "gqa decode hb=%d", hb);
        run_gqa(tag, 1, 32, 8, 128, 1, 2048, true, DataType::f32, gq);
        std::snprintf(tag, sizeof(tag), "gqa decode f16 hb=%d", hb);
        run_gqa(tag, 1, 32, 8, 128, 1, 2048, true, DataType::f16, gq);
        std::snprintf(tag, sizeof(tag), "mla decode hb=%d", hb);
        run_mla(tag, 1, 16, 1, 128, 64, 128, 1, 1024, true, DataType::f32,
                4.0 * 16 * 1024 * 192);
    }
    nnops::detail::cuda_attn_headbatch.store(-1);
}

// The int8-cache arms: one s8 cache, one query, and the three routes over it.
// The fp row is the baseline the integer tile has to beat — on an s8 cache it
// widens every element on read and multiplies in FFMA, which is what the tiled
// kernel does today.
NNOPS_BENCH(cuda_attention_int8_cache) {
    const int64_t B = 1, H = 32, Hkv = 8, D = 128, S = 512;
    const double flops = 2.0 * B * H * S * S * D;
    print_cuda_header("int8 cache (causal prefill, 512 x 512)");
    for (int arm = 0; arm < 3; ++arm) {
        char tag[64];
        std::snprintf(tag, sizeof(tag), "gqa i8cache f32 arm=%d", arm);
        run_gqa_i8(tag, B, H, Hkv, D, S, S, DataType::f32, flops, arm >= 1, arm >= 2);
        std::snprintf(tag, sizeof(tag), "gqa i8cache f16 arm=%d", arm);
        run_gqa_i8(tag, B, H, Hkv, D, S, S, DataType::f16, flops, arm >= 1, arm >= 2);
    }
    // The same routes at four times the scores: the integer tile's per-element
    // cost is what decides whether a longer run moves it off the scalar side.
    const double flops2 = 2.0 * B * H * 1024 * 1024 * D;
    for (int arm = 1; arm < 3; ++arm) {
        char tag[64];
        std::snprintf(tag, sizeof(tag), "gqa i8cache 1024 f32 arm=%d", arm);
        run_gqa_i8(tag, B, H, Hkv, D, 1024, 1024, DataType::f32, flops2, true, arm >= 2);
        std::snprintf(tag, sizeof(tag), "gqa i8cache 1024 f16 arm=%d", arm);
        run_gqa_i8(tag, B, H, Hkv, D, 1024, 1024, DataType::f16, flops2, true, arm >= 2);
    }
}

NNOPS_BENCH(cuda_attention_mla) {
    const int64_t B = 1, H = 16, Hkv = 1, rank = 128, rope = 64, vdim = 128;
    print_cuda_header("MLA absorbed");
    const double pre = 2.0 * B * H * 512 * 512 * (rank + rope);
    run_mla("mla prefill", B, H, Hkv, rank, rope, vdim, 512, 512, true, DataType::f32, pre);
    const double dec = 4.0 * B * H * 1024 * (rank + rope);
    run_mla("mla decode", B, H, Hkv, rank, rope, vdim, 1, 1024, true, DataType::f32, dec);
}

// The prefill tile arms: the two register-blocked tiles against the older
// one-row-per-thread tiled kernel on the same rows. blocked=0 is the tiled
// kernel; blocked=1,tc=0 the FFMA register blocking the f32 GEMM uses; tc=1 the
// m16n8k16 tile a 16-bit cache takes (it declines f32, so those rows repeat).
NNOPS_BENCH(cuda_attention_prefill_tile) {
    print_cuda_header("prefill tile arms");
    const int arms[3][2] = {{0, 0}, {1, 0}, {1, 1}};
    for (const auto& a : arms) {
        nnops::detail::cuda_attn_blocked.store(a[0]);
        nnops::detail::cuda_attn_tc.store(a[1]);
        char tag[80];
        std::snprintf(tag, sizeof(tag), "gqa prefill blocked=%d tc=%d", a[0], a[1]);
        run_gqa(tag, 1, 32, 8, 128, 512, 512, true, DataType::f32, 2.0 * 32 * 512 * 512 * 128);
        std::snprintf(tag, sizeof(tag), "gqa prefill f16 blocked=%d tc=%d", a[0], a[1]);
        run_gqa(tag, 1, 32, 8, 128, 512, 512, true, DataType::f16, 2.0 * 32 * 512 * 512 * 128);
        std::snprintf(tag, sizeof(tag), "gqa prefill 1024 blocked=%d tc=%d", a[0], a[1]);
        run_gqa(tag, 1, 32, 8, 128, 1024, 1024, true, DataType::f32,
                2.0 * 32 * 1024 * 1024 * 128);
        std::snprintf(tag, sizeof(tag), "gqa prefill f16 1024 blocked=%d tc=%d", a[0], a[1]);
        run_gqa(tag, 1, 32, 8, 128, 1024, 1024, true, DataType::f16,
                2.0 * 32 * 1024 * 1024 * 128);
        std::snprintf(tag, sizeof(tag), "mla prefill 512 blocked=%d tc=%d", a[0], a[1]);
        run_mla(tag, 1, 16, 1, 128, 64, 128, 512, 512, true, DataType::f32,
                2.0 * 16 * 512 * 512 * 192);
        std::snprintf(tag, sizeof(tag), "mla prefill f16 512 blocked=%d tc=%d", a[0], a[1]);
        run_mla(tag, 1, 16, 1, 128, 64, 128, 512, 512, true, DataType::f16,
                2.0 * 16 * 512 * 512 * 192);
    }
    nnops::detail::cuda_attn_blocked.store(-1);
    nnops::detail::cuda_attn_tc.store(-1);
}

// The split-KV arms. A decode launches one block per (query head, batch), so
// splitting the run across the key axis is the only way to put more of the
// machine to work; the sweep shows where that stops paying (each split repeats
// its chunk tail and the merge pass) and where the plain grid already fills the
// device.
NNOPS_BENCH(cuda_attention_decode_splits) {
    print_cuda_header("decode split-KV arms");
    const int arms[] = {0, 1, 2, 4, 8, 16, 32};
    for (int a : arms) {
        nnops::detail::cuda_attn_split.store(a);
        char tag[64];
        std::snprintf(tag, sizeof(tag), "gqa decode split=%d", a);
        run_gqa(tag, 1, 32, 8, 128, 1, 2048, true, DataType::f32, 4.0 * 32 * 2048 * 128);
        std::snprintf(tag, sizeof(tag), "mla decode split=%d", a);
        run_mla(tag, 1, 16, 1, 128, 64, 128, 1, 1024, true, DataType::f32,
                4.0 * 16 * 1024 * 192);
    }
    nnops::detail::cuda_attn_split.store(-1);
}

#endif // NNOPS_HAS_CUDA
