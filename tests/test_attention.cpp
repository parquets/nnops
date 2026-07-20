/// Unit tests for Attention operator (CPU reference).

#include "nnops/ops/attention.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cmath>
#include <algorithm>

using namespace nnops;

NNOPS_TEST(attention_basic_single_head) {
    // B=1, H=1, Sq=2, Sk=2, D=4
    // Test scaled dot-product attention without mask
    const int64_t qshape[] = {1, 2, 4};  // [B, S, D]
    const int64_t kshape[] = {1, 2, 4};
    const int64_t vshape[] = {1, 2, 4};
    const int64_t oshape[] = {1, 2, 4};

    float q_data[8] = {1,0,0,0, 0,1,0,0};
    float k_data[8] = {1,0,0,0, 0,1,0,0};
    float v_data[8] = {1,2,3,4, 5,6,7,8};
    float out_data[8] = {};

    TensorView Q(qshape, DataType::f32, q_data);
    TensorView K(kshape, DataType::f32, k_data);
    TensorView V(vshape, DataType::f32, v_data);
    TensorView output(oshape, DataType::f32, out_data);

    AttentionAttributes attrs;
    attrs.num_heads = 1;

    attention(Q, K, V, output, attrs);

    // QK^T: [[1,0],[0,1]] → scores after softmax ≈ [[0.73, 0.27], [0.27, 0.73]]
    // attn @ V should be close to V with some mixing
    for (int i = 0; i < 8; ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_data[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_data[i]));
    }
}

NNOPS_TEST(attention_causal_mask) {
    // B=1, H=1, Sq=3, Sk=3, D=2
    const int64_t qshape[] = {1, 3, 2};
    const int64_t kshape[] = {1, 3, 2};
    const int64_t vshape[] = {1, 3, 2};
    const int64_t oshape[] = {1, 3, 2};

    float q_data[6] = {1,0, 0,1, 1,1};
    float k_data[6] = {1,0, 0,1, 1,1};
    float v_data[6] = {1,0, 2,0, 3,0};
    float out_data[6] = {};

    TensorView Q(qshape, DataType::f32, q_data);
    TensorView K(kshape, DataType::f32, k_data);
    TensorView V(vshape, DataType::f32, v_data);
    TensorView output(oshape, DataType::f32, out_data);

    AttentionAttributes attrs;
    attrs.num_heads = 1;
    attrs.use_causal_mask = true;

    attention(Q, K, V, output, attrs);

    // With causal mask, position 0 can only attend to position 0
    // position 1 can attend to 0,1; position 2 can attend to 0,1,2
    for (int i = 0; i < 6; ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_data[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_data[i]));
    }
    // Position 0 output should equal V[0] since it only attends to itself
    NNOPS_EXPECT_NEAR(out_data[0], v_data[0], 1e-4f);
    NNOPS_EXPECT_NEAR(out_data[1], v_data[1], 1e-4f);
}

NNOPS_TEST(attention_multi_head) {
    // B=1, H=2, Sq=2, Sk=2, D=2 (merged: [B, S, H*D] = [1, 2, 4])
    const int64_t qshape[] = {1, 2, 4};
    const int64_t kshape[] = {1, 2, 4};
    const int64_t vshape[] = {1, 2, 4};
    const int64_t oshape[] = {1, 2, 4};

    std::vector<float> q_buf(8);
    std::vector<float> k_buf(8);
    std::vector<float> v_buf(8);
    std::vector<float> out_buf(8);

    // Identity-like Q and K (each head gets its own subspace)
    for (int i = 0; i < 8; ++i) q_buf[i] = (i % 3 == 0) ? 1.0f : 0.0f;
    for (int i = 0; i < 8; ++i) k_buf[i] = q_buf[i];
    for (int i = 0; i < 8; ++i) v_buf[i] = static_cast<float>(i + 1);

    TensorView Q(qshape, DataType::f32, q_buf.data());
    TensorView K(kshape, DataType::f32, k_buf.data());
    TensorView V(vshape, DataType::f32, v_buf.data());
    TensorView output(oshape, DataType::f32, out_buf.data());

    AttentionAttributes attrs;
    attrs.num_heads = 2;

    attention(Q, K, V, output, attrs);

    for (int i = 0; i < 8; ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

NNOPS_TEST(attention_random) {
    auto [q_vec, Q] = test::make_random_tensor({2, 1, 16});  // B=2, H=1, S=4, D=4 (merged)
    auto [k_vec, K] = test::make_random_tensor({2, 1, 16});
    auto [v_vec, V] = test::make_random_tensor({2, 1, 16});
    std::vector<float> out_buf(2 * 1 * 16);
    const int64_t oshape[] = {2, 1, 16};
    TensorView output(oshape, DataType::f32, out_buf.data());

    AttentionAttributes attrs;
    attrs.num_heads = 1;
    attrs.use_causal_mask = true;

    attention(Q, K, V, output, attrs);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

NNOPS_TEST(attention_class_api) {
    const int64_t qshape[] = {1, 2, 4};
    const int64_t kshape[] = {1, 2, 4};
    const int64_t vshape[] = {1, 2, 4};
    const int64_t oshape[] = {1, 2, 4};

    float q_data[8] = {1,0,0,0, 0,1,0,0};
    float k_data[8] = {1,0,0,0, 0,1,0,0};
    float v_data[8] = {1,2,3,4, 5,6,7,8};
    float out1_data[8] = {};
    float out2_data[8] = {};

    TensorView Q(qshape, DataType::f32, q_data);
    TensorView K(kshape, DataType::f32, k_data);
    TensorView V(vshape, DataType::f32, v_data);
    TensorView out1(oshape, DataType::f32, out1_data);
    TensorView out2(oshape, DataType::f32, out2_data);

    AttentionAttributes attrs;
    attrs.num_heads = 1;

    // Functional
    attention(Q, K, V, out1, attrs);
    // Class
    auto op = Attention::create(attrs, Backend::CPU);
    const TensorView ins[] = {Q, K, V};
    op->compute(out2, ins);

    NNOPS_EXPECT_TRUE(test::allclose(out1, out2));
}
