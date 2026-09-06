/// @file test_causal_attention.cpp
/// @brief Unit tests for CausalAttention operator (CPU backend).

#include "nnops/ops/causal_attention.hpp"
#include "nnops/detail/bf16.hpp"
#include "nnops/detail/half.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"

#include <vector>
#include <cmath>
#include <cstring>

using namespace nnops;
using namespace nnops::backend::cpu;  // for half, half_to_float, float_to_half

namespace {

// ============================================================
// Helper functions
// ============================================================

inline TensorView make_scalar_i64(int64_t& val) {
    const int64_t shape[] = {1};
    return TensorView(shape, DataType::s64, &val);
}

/// Create cache tensor from f32 data with specified dtype.
TensorView make_cache(const std::vector<int64_t>& shape,
                      const std::vector<float>& f32_data,
                      std::vector<char>& storage,
                      DataType cache_dtype)
{
    int64_t total = 1;
    for (auto d : shape) {
        total *= d;
    }

    switch (cache_dtype) {
    case DataType::f32:
        storage.resize(total * sizeof(float));
        std::memcpy(storage.data(), f32_data.data(), total * sizeof(float));
        return TensorView(shape, DataType::f32, storage.data(), TensorLayout::NCHW);

    case DataType::f16:
        storage.resize(total * sizeof(uint16_t));
        {
            auto* p = reinterpret_cast<uint16_t*>(storage.data());
            for (int64_t i = 0; i < total; ++i) {
                p[i] = float_to_half(f32_data[i]).bits;
            }
        }
        return TensorView(shape, DataType::f16, storage.data(), TensorLayout::NCHW);

    case DataType::bf16:
        storage.resize(total * sizeof(uint16_t));
        {
            auto* p = reinterpret_cast<uint16_t*>(storage.data());
            for (int64_t i = 0; i < total; ++i) {
                p[i] = float_to_bf16(f32_data[i]);
            }
        }
        return TensorView(shape, DataType::bf16, storage.data(), TensorLayout::NCHW);

    default:
        return TensorView();
    }
}

/// Create int8 cache tensor with quantization.
TensorView make_i8_cache(const std::vector<int64_t>& shape,
                          const std::vector<float>& f32_data,
                          std::vector<char>& storage,
                          float scale, int32_t zp)
{
    int64_t total = 1;
    for (auto d : shape) {
        total *= d;
    }

    storage.resize(total * sizeof(int8_t));
    auto* p = reinterpret_cast<int8_t*>(storage.data());
    for (int64_t i = 0; i < total; ++i) {
        float iv = std::round(f32_data[i] / scale) + static_cast<float>(zp);
        p[i] = static_cast<int8_t>(std::clamp(iv, -128.0f, 127.0f));
    }

    QuantParams qp;
    qp.scale = scale;
    qp.zero_point = zp;
    return TensorView(shape, DataType::s8, storage.data(), TensorLayout::NCHW, qp);
}

/// Helpers to create dense float TensorViews
inline TensorView make_f32tv(const std::vector<int64_t>& sh, std::vector<float>& buf) {
    int64_t n = 1; for (auto d : sh) n *= d;
    buf.resize(static_cast<size_t>(n));
    return TensorView(sh, DataType::f32, buf.data(), TensorLayout::NCHW);
}

}  // anonymous namespace

// ============================================================
// Test 1: First token (empty cache)
// ============================================================

NNOPS_TEST(causal_attention_first_token) {
    const int64_t B = 1, H = 2, D = 4, max_len = 8;
    const std::vector<int64_t> qk_sh = {B, H, 1, D};
    const std::vector<int64_t> cache_sh = {B, H, max_len, D};

    auto [q_vec, q_tv]  = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 42);
    auto [k_vec, k_tv]  = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 43);
    auto [v_vec, v_tv]  = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 44);

    std::vector<float> kc_buf, vc_buf, out_buf;
    auto k_cache = make_f32tv(cache_sh, kc_buf);
    std::memset(kc_buf.data(), 0, kc_buf.size() * sizeof(float));
    auto v_cache = make_f32tv(cache_sh, vc_buf);
    std::memset(vc_buf.data(), 0, vc_buf.size() * sizeof(float));
    auto output = make_f32tv(qk_sh, out_buf);

    CausalAttentionAttributes attrs;
    attrs.num_heads = H;
    attrs.max_cache_seq_len = max_len;
    attrs.max_chunk_size = 1;
    auto op = CausalAttention::create(attrs, Backend::CPU);

    // Verify shape inference
    auto d_q = q_tv.desc();
    const TensorDesc desc_arr[] = {d_q, d_q, d_q};
    auto descs = op->getOutputTensorDesc(desc_arr);
    NNOPS_EXPECT_EQ(descs.size(), size_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[2], int64_t(1));

    int64_t pos_buf = 0, len_buf = 0;
    auto cache_pos = make_scalar_i64(pos_buf);
    auto cache_len = make_scalar_i64(len_buf);

    const TensorView ins[] = {q_tv, k_tv, v_tv, cache_pos, cache_len};
    TensorView outs[] = {output, k_cache, v_cache};
    op->compute(outs, ins, {}, nullptr);

    // Output should equal V_new (softmax over 1 element = identity)
    const float* v_ptr = v_tv.ptr<float>();
    for (int64_t i = 0; i < H * D; ++i) {
        NNOPS_EXPECT_NEAR(out_buf[i], v_ptr[i], 1e-5f);
    }
}

// ============================================================
// Test 2: Two-step decode, hand-verified values
// ============================================================

NNOPS_TEST(causal_attention_two_steps_f32) {
    const int64_t B = 1, H = 1, D = 4, max_len = 4;
    const std::vector<int64_t> qk_sh = {B, H, 1, D};
    const std::vector<int64_t> cache_sh = {B, H, max_len, D};

    CausalAttentionAttributes attrs;
    attrs.num_heads = H;
    attrs.max_cache_seq_len = max_len;
    attrs.max_chunk_size = 1;
    attrs.scale = 1.0f;  // explicit to simplify hand-verified values
    auto op = CausalAttention::create(attrs, Backend::CPU);

    // Step 1 data
    std::vector<float> q1_v = {1,0,0,0}, k1_v = {1,0,0,0}, v1_v = {1,2,3,4};
    TensorView q1(qk_sh, DataType::f32, q1_v.data(), TensorLayout::NCHW);
    TensorView k1(qk_sh, DataType::f32, k1_v.data(), TensorLayout::NCHW);
    TensorView v1(qk_sh, DataType::f32, v1_v.data(), TensorLayout::NCHW);

    std::vector<float> kc, vc, out;
    auto k_cache = make_f32tv(cache_sh, kc);
    auto v_cache = make_f32tv(cache_sh, vc);
    auto output = make_f32tv(qk_sh, out);

    int64_t pos_buf = 0, len_buf = 0;
    auto cache_pos = make_scalar_i64(pos_buf);
    auto cache_len = make_scalar_i64(len_buf);

    // Step 1
    {
        const TensorView ins[] = {q1, k1, v1, cache_pos, cache_len};
        TensorView outs[] = {output, k_cache, v_cache};
        op->compute(outs, ins, {}, nullptr);
    }
    for (int64_t d = 0; d < D; ++d) {
        NNOPS_EXPECT_NEAR(out[d], v1_v[d], 1e-5f);
    }

    // Step 2
    pos_buf = 1; len_buf = 1;
    std::vector<float> q2_v = {0,1,0,0}, k2_v = {0,1,0,0}, v2_v = {5,6,7,8};
    TensorView q2(qk_sh, DataType::f32, q2_v.data(), TensorLayout::NCHW);
    TensorView k2(qk_sh, DataType::f32, k2_v.data(), TensorLayout::NCHW);
    TensorView v2(qk_sh, DataType::f32, v2_v.data(), TensorLayout::NCHW);

    {
        const TensorView ins[] = {q2, k2, v2, cache_pos, cache_len};
        TensorView outs[] = {output, k_cache, v_cache};
        op->compute(outs, ins, {}, nullptr);
    }

    // Q2·K1=0, Q2·K2=1: softmax([0,1]) → [e^0/(e^0+e^1), e^1/(e^0+e^1)]
    float s0 = 1.0f / (1.0f + std::exp(1.0f));
    float s1 = 1.0f - s0;
    for (int64_t d = 0; d < D; ++d) {
        NNOPS_EXPECT_NEAR(out[d], s0 * v1_v[d] + s1 * v2_v[d], 1e-4f);
    }
}

// ============================================================
// Test 3: Chunk prefill (Sq=2, no past cache)
// ============================================================

NNOPS_TEST(causal_attention_chunk_prefill) {
    const int64_t B = 1, H = 1, D = 2, Sq = 2, max_len = 4;
    const std::vector<int64_t> qk_sh = {B, H, Sq, D};
    const std::vector<int64_t> cache_sh = {B, H, max_len, D};

    CausalAttentionAttributes attrs;
    attrs.num_heads = H;
    attrs.max_cache_seq_len = max_len;
    attrs.max_chunk_size = Sq;
    attrs.scale = 1.0f;  // explicit to simplify hand-verified values
    auto op = CausalAttention::create(attrs, Backend::CPU);

    std::vector<float> q_v = {1,0, 0,1}, k_v = {1,0, 0,1}, v_v = {1,2, 3,4};
    TensorView q(qk_sh, DataType::f32, q_v.data(), TensorLayout::NCHW);
    TensorView k(qk_sh, DataType::f32, k_v.data(), TensorLayout::NCHW);
    TensorView v(qk_sh, DataType::f32, v_v.data(), TensorLayout::NCHW);

    std::vector<float> kc, vc, out;
    auto k_cache = make_f32tv(cache_sh, kc);
    auto v_cache = make_f32tv(cache_sh, vc);
    auto output = make_f32tv(qk_sh, out);

    int64_t pos_buf = 0, len_buf = 0;
    auto cache_pos = make_scalar_i64(pos_buf);
    auto cache_len = make_scalar_i64(len_buf);

    const TensorView ins[] = {q, k, v, cache_pos, cache_len};
    TensorView outs[] = {output, k_cache, v_cache};
    op->compute(outs, ins, {}, nullptr);

    // Row 0 (i=0): Q[0]=[1,0], K[0]=[1,0], K[1]=[0,1]
    //   scores: dot(Q[0],K[0])=1, dot(Q[0],K[1])=0
    //   causal mask: j=0 valid, j=1 MASKED (j>i) → softmax([1,-inf]) = [1,0]
    //   → output row 0 = V[0] = [1,2]
    NNOPS_EXPECT_NEAR(out[0], v_v[0], 1e-4f);
    NNOPS_EXPECT_NEAR(out[1], v_v[1], 1e-4f);

    // Row 1 (i=1): Q[1]=[0,1], K[0]=[1,0], K[1]=[0,1]
    //   scores: dot(Q[1],K[0])=0, dot(Q[1],K[1])=1
    //   causal mask: j=0 valid, j=1 valid (j<=i) → softmax([0,1])
    float e = std::exp(1.0f);
    float w0 = 1.0f/(1.0f+e), w1 = e/(1.0f+e);
    NNOPS_EXPECT_NEAR(out[2], w0*v_v[0]+w1*v_v[2], 1e-4f);
    NNOPS_EXPECT_NEAR(out[3], w0*v_v[1]+w1*v_v[3], 1e-4f);
}

// ============================================================
// Test 4: f16 cache (random data, compare with f32 reference)
// ============================================================

NNOPS_TEST(causal_attention_f16_cache) {
    const int64_t B = 1, H = 2, D = 8, max_len = 4;
    const std::vector<int64_t> qk_sh = {B, H, 1, D};
    const std::vector<int64_t> cache_sh = {B, H, max_len, D};

    auto [q_vec, q_tv] = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 100);
    auto [k_vec, k_tv] = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 101);
    auto [v_vec, v_tv] = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 102);

    // Pre-fill cache with random data
    std::vector<float> kc_f32;
    auto k_cache_f32 = make_f32tv(cache_sh, kc_f32);
    std::vector<float> vc_f32;
    auto v_cache_f32 = make_f32tv(cache_sh, vc_f32);
    for (auto& x : kc_f32) {
        x = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
    for (auto& x : vc_f32) {
        x = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }

    std::vector<char> kc_f16_st, vc_f16_st;
    auto k_cache_f16 = make_cache(cache_sh, kc_f32, kc_f16_st, DataType::f16);
    auto v_cache_f16 = make_cache(cache_sh, vc_f32, vc_f16_st, DataType::f16);

    std::vector<float> out_f32, out_f16;
    auto output_f32 = make_f32tv(qk_sh, out_f32);
    auto output_f16 = make_f32tv(qk_sh, out_f16);

    CausalAttentionAttributes attrs;
    attrs.num_heads = H;
    attrs.max_cache_seq_len = max_len;
    attrs.max_chunk_size = 1;
    auto op = CausalAttention::create(attrs, Backend::CPU);

    int64_t pos_buf = 1, len_buf = 1;
    auto cache_pos = make_scalar_i64(pos_buf);
    auto cache_len = make_scalar_i64(len_buf);

    // f32 reference
    {
        const TensorView ins[] = {q_tv, k_tv, v_tv, cache_pos, cache_len};
        TensorView outs[] = {output_f32, k_cache_f32, v_cache_f32};
        op->compute(outs, ins, {}, nullptr);
    }
    // f16 cache
    {
        const TensorView ins[] = {q_tv, k_tv, v_tv, cache_pos, cache_len};
        TensorView outs[] = {output_f16, k_cache_f16, v_cache_f16};
        op->compute(outs, ins, {}, nullptr);
    }

    for (size_t i = 0; i < out_f32.size(); ++i) {
        NNOPS_EXPECT_NEAR(out_f16[i], out_f32[i], 1e-2f);
    }
}

// ============================================================
// Test 5: Multi-head
// ============================================================

NNOPS_TEST(causal_attention_multi_head) {
    const int64_t B = 1, H = 4, D = 4, max_len = 4;
    const std::vector<int64_t> qk_sh = {B, H, 1, D};
    const std::vector<int64_t> cache_sh = {B, H, max_len, D};

    auto [q_vec, q_tv] = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 200);
    auto [k_vec, k_tv] = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 201);
    auto [v_vec, v_tv] = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 202);

    std::vector<float> kc, vc, out;
    auto k_cache = make_f32tv(cache_sh, kc);
    auto v_cache = make_f32tv(cache_sh, vc);
    auto output = make_f32tv(qk_sh, out);

    CausalAttentionAttributes attrs;
    attrs.num_heads = H;
    attrs.max_cache_seq_len = max_len;
    attrs.max_chunk_size = 1;
    auto op = CausalAttention::create(attrs, Backend::CPU);

    int64_t pos_buf = 0, len_buf = 0;
    auto cache_pos = make_scalar_i64(pos_buf);
    auto cache_len = make_scalar_i64(len_buf);

    const TensorView ins[] = {q_tv, k_tv, v_tv, cache_pos, cache_len};
    TensorView outs[] = {output, k_cache, v_cache};
    op->compute(outs, ins, {}, nullptr);

    for (size_t i = 0; i < out.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out[i]));
    }
}

// ============================================================
// Test 6: Multi-batch
// ============================================================

NNOPS_TEST(causal_attention_multi_batch) {
    const int64_t B = 2, H = 2, D = 4, max_len = 4;
    const std::vector<int64_t> qk_sh = {B, H, 1, D};
    const std::vector<int64_t> cache_sh = {B, H, max_len, D};

    auto [q_vec, q_tv] = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 300);
    auto [k_vec, k_tv] = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 301);
    auto [v_vec, v_tv] = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 302);

    std::vector<float> kc, vc, out;
    auto k_cache = make_f32tv(cache_sh, kc);
    auto v_cache = make_f32tv(cache_sh, vc);
    auto output = make_f32tv(qk_sh, out);

    CausalAttentionAttributes attrs;
    attrs.num_heads = H;
    attrs.max_cache_seq_len = max_len;
    attrs.max_chunk_size = 1;
    auto op = CausalAttention::create(attrs, Backend::CPU);

    int64_t pos_buf = 0, len_buf = 0;
    auto cache_pos = make_scalar_i64(pos_buf);
    auto cache_len = make_scalar_i64(len_buf);

    const TensorView ins[] = {q_tv, k_tv, v_tv, cache_pos, cache_len};
    TensorView outs[] = {output, k_cache, v_cache};
    op->compute(outs, ins, {}, nullptr);

    for (size_t i = 0; i < out.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out[i]));
    }
}

// ============================================================
// Test 7: Temperature scaling
// ============================================================

NNOPS_TEST(causal_attention_temperature) {
    const int64_t B = 1, H = 1, D = 4, Sq = 2, max_len = 4;
    const std::vector<int64_t> qk_sh = {B, H, Sq, D};
    const std::vector<int64_t> cache_sh = {B, H, max_len, D};

    auto [q_vec, q_tv] = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 400);
    auto [k_vec, k_tv] = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 401);
    auto [v_vec, v_tv] = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 402);

    std::vector<float> kc_t1, vc_t1, out_t1;
    auto kc1 = make_f32tv(cache_sh, kc_t1);
    auto vc1 = make_f32tv(cache_sh, vc_t1);
    auto o1 = make_f32tv(qk_sh, out_t1);

    std::vector<float> kc_t2, vc_t2, out_t2;
    auto kc2 = make_f32tv(cache_sh, kc_t2);
    auto vc2 = make_f32tv(cache_sh, vc_t2);
    auto o2 = make_f32tv(qk_sh, out_t2);

    CausalAttentionAttributes attrs_t1;
    attrs_t1.num_heads = H;
    attrs_t1.max_cache_seq_len = max_len;
    attrs_t1.max_chunk_size = Sq;
    attrs_t1.temperature = 1.0f;
    auto op_t1 = CausalAttention::create(attrs_t1, Backend::CPU);

    CausalAttentionAttributes attrs_t2 = attrs_t1;
    attrs_t2.temperature = 2.0f;
    auto op_t2 = CausalAttention::create(attrs_t2, Backend::CPU);

    int64_t pos_buf = 0, len_buf = 0;
    auto cache_pos = make_scalar_i64(pos_buf);
    auto cache_len = make_scalar_i64(len_buf);

    {
        const TensorView ins[] = {q_tv, k_tv, v_tv, cache_pos, cache_len};
        TensorView outs[] = {o1, kc1, vc1};
        op_t1->compute(outs, ins, {}, nullptr);
    }
    {
        const TensorView ins[] = {q_tv, k_tv, v_tv, cache_pos, cache_len};
        TensorView outs[] = {o2, kc2, vc2};
        op_t2->compute(outs, ins, {}, nullptr);
    }

    // T=2.0 gives softer distribution — results should differ
    bool any_diff = false;
    for (size_t i = 0; i < out_t1.size(); ++i) {
        if (std::abs(out_t1[i] - out_t2[i]) > 1e-6f) { any_diff = true; break; }
    }
    NNOPS_EXPECT_TRUE(any_diff);
}

// ============================================================
// Test 8: Auto scale
// ============================================================

NNOPS_TEST(causal_attention_auto_scale) {
    const int64_t B = 1, H = 1, D = 16, max_len = 4;
    const std::vector<int64_t> qk_sh = {B, H, 1, D};
    const std::vector<int64_t> cache_sh = {B, H, max_len, D};

    auto [q_vec, q_tv] = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 500);
    auto [k_vec, k_tv] = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 501);
    auto [v_vec, v_tv] = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 502);

    std::vector<float> kc1, vc1, out1, kc2, vc2, out2;
    auto kc_a = make_f32tv(cache_sh, kc1);
    auto vc_a = make_f32tv(cache_sh, vc1);
    auto o_a  = make_f32tv(qk_sh, out1);
    auto kc_e = make_f32tv(cache_sh, kc2);
    auto vc_e = make_f32tv(cache_sh, vc2);
    auto o_e  = make_f32tv(qk_sh, out2);

    CausalAttentionAttributes a1;
    a1.num_heads = H; a1.max_cache_seq_len = max_len; a1.max_chunk_size = 1;
    a1.scale = 0.0f;
    auto op_a = CausalAttention::create(a1, Backend::CPU);

    CausalAttentionAttributes a2 = a1;
    a2.scale = 1.0f / std::sqrt(static_cast<float>(D));
    auto op_e = CausalAttention::create(a2, Backend::CPU);

    int64_t pos_buf = 0, len_buf = 0;
    auto cache_pos = make_scalar_i64(pos_buf);
    auto cache_len = make_scalar_i64(len_buf);

    {
        const TensorView ins[] = {q_tv, k_tv, v_tv, cache_pos, cache_len};
        TensorView outs[] = {o_a, kc_a, vc_a};
        op_a->compute(outs, ins, {}, nullptr);
    }
    {
        const TensorView ins[] = {q_tv, k_tv, v_tv, cache_pos, cache_len};
        TensorView outs[] = {o_e, kc_e, vc_e};
        op_e->compute(outs, ins, {}, nullptr);
    }

    for (size_t i = 0; i < out1.size(); ++i) {
        NNOPS_EXPECT_NEAR(out1[i], out2[i], 1e-5f);
    }
}

// ============================================================
// Test 9: int8 cache vs f32 reference
// ============================================================

NNOPS_TEST(causal_attention_int8_cache) {
    const int64_t B = 1, H = 1, D = 8, max_len = 4;
    const std::vector<int64_t> qk_sh = {B, H, 1, D};
    const std::vector<int64_t> cache_sh = {B, H, max_len, D};

    auto [q_vec, q_tv] = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 600);
    auto [k_vec, k_tv] = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 601);
    auto [v_vec, v_tv] = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 602);

    std::vector<float> kc_f32;
    auto kc_f32_tv = make_f32tv(cache_sh, kc_f32);
    std::vector<float> vc_f32;
    auto vc_f32_tv = make_f32tv(cache_sh, vc_f32);
    for (auto& x : kc_f32) {
        x = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }
    for (auto& x : vc_f32) {
        x = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    float scale = 0.01f;
    int32_t zp = 0;
    std::vector<char> kc_i8_st, vc_i8_st;
    auto kc_i8 = make_i8_cache(cache_sh, kc_f32, kc_i8_st, scale, zp);
    auto vc_i8 = make_i8_cache(cache_sh, vc_f32, vc_i8_st, scale, zp);

    std::vector<float> out_f32, out_i8;
    auto o_f32 = make_f32tv(qk_sh, out_f32);
    auto o_i8  = make_f32tv(qk_sh, out_i8);

    CausalAttentionAttributes attrs;
    attrs.num_heads = H;
    attrs.max_cache_seq_len = max_len;
    attrs.max_chunk_size = 1;
    auto op = CausalAttention::create(attrs, Backend::CPU);

    int64_t pos_buf = 1, len_buf = 1;
    auto cache_pos = make_scalar_i64(pos_buf);
    auto cache_len = make_scalar_i64(len_buf);

    {
        const TensorView ins[] = {q_tv, k_tv, v_tv, cache_pos, cache_len};
        TensorView outs[] = {o_f32, kc_f32_tv, vc_f32_tv};
        op->compute(outs, ins, {}, nullptr);
    }
    {
        const TensorView ins[] = {q_tv, k_tv, v_tv, cache_pos, cache_len};
        TensorView outs[] = {o_i8, kc_i8, vc_i8};
        op->compute(outs, ins, {}, nullptr);
    }

    for (size_t i = 0; i < out_f32.size(); ++i) {
        NNOPS_EXPECT_NEAR(out_i8[i], out_f32[i], 5e-2f);
    }
}

// ============================================================
// Test 10: bf16 cache vs f32 reference
// ============================================================

NNOPS_TEST(causal_attention_bf16_cache) {
    const int64_t B = 1, H = 2, D = 8, max_len = 4;
    const std::vector<int64_t> qk_sh = {B, H, 1, D};
    const std::vector<int64_t> cache_sh = {B, H, max_len, D};

    auto [q_vec, q_tv] = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 700);
    auto [k_vec, k_tv] = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 701);
    auto [v_vec, v_tv] = test::make_random_tensor(qk_sh, -1.0f, 1.0f, 702);

    std::vector<float> kc_f32;
    auto kc_f32_tv = make_f32tv(cache_sh, kc_f32);
    std::vector<float> vc_f32;
    auto vc_f32_tv = make_f32tv(cache_sh, vc_f32);
    for (auto& x : kc_f32) {
        x = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
    for (auto& x : vc_f32) {
        x = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }

    std::vector<char> kc_bf16_st, vc_bf16_st;
    auto kc_bf16 = make_cache(cache_sh, kc_f32, kc_bf16_st, DataType::bf16);
    auto vc_bf16 = make_cache(cache_sh, vc_f32, vc_bf16_st, DataType::bf16);

    std::vector<float> out_f32, out_bf16;
    auto o_f32 = make_f32tv(qk_sh, out_f32);
    auto o_bf16 = make_f32tv(qk_sh, out_bf16);

    CausalAttentionAttributes attrs;
    attrs.num_heads = H;
    attrs.max_cache_seq_len = max_len;
    attrs.max_chunk_size = 1;
    auto op = CausalAttention::create(attrs, Backend::CPU);

    int64_t pos_buf = 1, len_buf = 1;
    auto cache_pos = make_scalar_i64(pos_buf);
    auto cache_len = make_scalar_i64(len_buf);

    {
        const TensorView ins[] = {q_tv, k_tv, v_tv, cache_pos, cache_len};
        TensorView outs[] = {o_f32, kc_f32_tv, vc_f32_tv};
        op->compute(outs, ins, {}, nullptr);
    }
    {
        const TensorView ins[] = {q_tv, k_tv, v_tv, cache_pos, cache_len};
        TensorView outs[] = {o_bf16, kc_bf16, vc_bf16};
        op->compute(outs, ins, {}, nullptr);
    }

    for (size_t i = 0; i < out_f32.size(); ++i) {
        NNOPS_EXPECT_NEAR(out_bf16[i], out_f32[i], 5e-2f);
    }
}

// ============================================================
// Test 11: Shape inference
// ============================================================

NNOPS_TEST(causal_attention_output_shape) {
    CausalAttentionAttributes attrs;
    attrs.num_heads = 4;
    attrs.max_cache_seq_len = 64;
    attrs.max_chunk_size = 1;
    auto op = CausalAttention::create(attrs, Backend::CPU);

    TensorDesc q_desc;
    q_desc.rank = 4;
    q_desc.dims = {2, 4, 1, 128};
    q_desc.dtype = DataType::f32;
    q_desc.layout = TensorLayout::NCHW;

    const TensorDesc desc_arr[] = {q_desc, q_desc, q_desc};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs.size(), size_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[2], int64_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[3], int64_t(128));
}

// ============================================================
// Test 12: OpType
// ============================================================

NNOPS_TEST(causal_attention_op_type) {
    auto op = CausalAttention::create(Backend::CPU);
    NNOPS_EXPECT_EQ(static_cast<int>(op->getOpType()), static_cast<int>(OpType::CausalAttention));
    NNOPS_EXPECT_EQ(static_cast<int>(op->getBackend()), static_cast<int>(Backend::CPU));
}

// ============================================================
// Test 13: Chunk prefill with past cache
// ============================================================

NNOPS_TEST(causal_attention_chunk_with_past) {
    const int64_t B = 1, H = 1, D = 4, Sq = 2, max_len = 8, past_len = 2;
    const std::vector<int64_t> qk_sh = {B, H, Sq, D};
    const std::vector<int64_t> cache_sh = {B, H, max_len, D};

    CausalAttentionAttributes attrs;
    attrs.num_heads = H;
    attrs.max_cache_seq_len = max_len;
    attrs.max_chunk_size = Sq;
    auto op = CausalAttention::create(attrs, Backend::CPU);

    // Pre-fill cache with past data
    std::vector<float> kc, vc, out;
    auto k_cache = make_f32tv(cache_sh, kc);
    auto v_cache = make_f32tv(cache_sh, vc);
    auto output = make_f32tv(qk_sh, out);

    // Past token 0: K=[1,0,0,0], V=[1,1,1,1]
    // Past token 1: K=[0,1,0,0], V=[2,2,2,2]
    std::memset(kc.data(), 0, kc.size() * sizeof(float));
    std::memset(vc.data(), 0, vc.size() * sizeof(float));
    kc[0]=1; kc[4]=1; // K[0]=[1,0,0,0], K[1]=[0,1,0,0]
    vc[0]=1; vc[1]=1; vc[2]=1; vc[3]=1;
    vc[4]=2; vc[5]=2; vc[6]=2; vc[7]=2;

    // Current chunk
    std::vector<float> q_v = {0,0,1,0, 0,0,0,1};
    std::vector<float> k_v = {0,0,1,0, 0,0,0,1};
    std::vector<float> v_v = {3,3,3,3, 4,4,4,4};
    TensorView q(qk_sh, DataType::f32, q_v.data(), TensorLayout::NCHW);
    TensorView k(qk_sh, DataType::f32, k_v.data(), TensorLayout::NCHW);
    TensorView v(qk_sh, DataType::f32, v_v.data(), TensorLayout::NCHW);

    int64_t pos_buf = past_len, len_buf = past_len;
    auto cache_pos = make_scalar_i64(pos_buf);
    auto cache_len = make_scalar_i64(len_buf);

    const TensorView ins[] = {q, k, v, cache_pos, cache_len};
    TensorView outs[] = {output, k_cache, v_cache};
    op->compute(outs, ins, {}, nullptr);

    for (size_t i = 0; i < out.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out[i]));
    }

    // Cache should be written at past_len (position 2)
    int64_t off = past_len * D;
    NNOPS_EXPECT_NEAR(kc[off+2], 1.0f, 1e-5f);  // K[2][2] = 1
}
