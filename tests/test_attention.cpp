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
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"

#include <vector>
#include <cmath>
#include <algorithm>

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
// Causal mask
// ============================================================

NNOPS_TEST(attention_causal_mask) {
    // B=1, H=1, Sq=3, Sk=3, D=2
    const int64_t qshape[] = {1, 3, 2};
    const int64_t kshape[] = {1, 3, 2};
    const int64_t vshape[] = {1, 3, 2};

    float q_data[6] = {1,0, 0,1, 1,1};
    float k_data[6] = {1,0, 0,1, 1,1};
    float v_data[6] = {1,0, 2,0, 3,0};

    TensorView q(qshape, DataType::f32, q_data);
    TensorView k(kshape, DataType::f32, k_data);
    TensorView v(vshape, DataType::f32, v_data);

    AttentionAttributes attrs;
    attrs.num_heads = 1;
    attrs.use_causal_mask = true;

    auto op = Attention::create(attrs, Backend::CPU);

    auto d_q = q.desc();
    auto d_k = k.desc();
    auto d_v = v.desc();
    const TensorDesc desc_arr[] = {d_q, d_k, d_v};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs.size(), size_t(1));
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(3));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(3));
    NNOPS_EXPECT_EQ(descs[0].dims[2], int64_t(2));
    NNOPS_EXPECT_EQ(static_cast<int>(descs[0].layout), static_cast<int>(TensorLayout::NCHW));
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    auto out = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {q, k, v};
    op->compute(out, ins);

    // With causal mask, position 0 can only attend to position 0
    // position 1 can attend to 0,1; position 2 can attend to 0,1,2
    for (int i = 0; i < 6; ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
    // Position 0 output should equal V[0] since it only attends to itself
    NNOPS_EXPECT_NEAR(out_buf[0], v_data[0], 1e-4f);
    NNOPS_EXPECT_NEAR(out_buf[1], v_data[1], 1e-4f);
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
    attrs.use_causal_mask = true;

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
