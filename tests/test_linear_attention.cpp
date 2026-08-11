/// Unit tests for LinearAttention operator — interface validation only.
///
/// The kernel is not yet implemented; tests verify operator creation,
/// OpType/Backend, and shape inference for the linear attention interface.

#include "nnops/ops/linear_attention.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"

#include <vector>

using namespace nnops;

// ============================================================
// Operator creation
// ============================================================

NNOPS_TEST(linear_attention_create) {
    auto op = LinearAttention::create(Backend::CPU);
    NNOPS_EXPECT_EQ(static_cast<int>(op->getOpType()),
                    static_cast<int>(OpType::LinearAttention));
    NNOPS_EXPECT_EQ(static_cast<int>(op->getBackend()),
                    static_cast<int>(Backend::CPU));
}

// ============================================================
// Default attributes
// ============================================================

NNOPS_TEST(linear_attention_default_attrs) {
    LinearAttentionAttributes attrs;
    NNOPS_EXPECT_EQ(attrs.head_dim, int64_t(64));
    NNOPS_EXPECT_EQ(attrs.num_heads, int64_t(8));
    NNOPS_EXPECT_EQ(attrs.num_kv_heads, int64_t(0));
    NNOPS_EXPECT_NEAR(attrs.scale, 0.0f, 1e-6f);
    NNOPS_EXPECT_EQ(attrs.conv_kernel_size, int64_t(4));
}

// ============================================================
// Custom attributes
// ============================================================

NNOPS_TEST(linear_attention_custom_attrs) {
    LinearAttentionAttributes attrs;
    attrs.head_dim = 128;
    attrs.num_heads = 32;
    attrs.num_kv_heads = 8;
    attrs.scale = 0.1f;
    attrs.conv_kernel_size = 3;

    auto op = LinearAttention::create(attrs, Backend::CPU);
    NNOPS_EXPECT_EQ(op->attributes().head_dim, int64_t(128));
    NNOPS_EXPECT_EQ(op->attributes().num_heads, int64_t(32));
    NNOPS_EXPECT_EQ(op->attributes().num_kv_heads, int64_t(8));
    NNOPS_EXPECT_NEAR(op->attributes().scale, 0.1f, 1e-6f);
    NNOPS_EXPECT_EQ(op->attributes().conv_kernel_size, int64_t(3));
}

// ============================================================
// Shape inference
// ============================================================

NNOPS_TEST(linear_attention_shape_inference) {
    LinearAttentionAttributes attrs;
    attrs.head_dim = 64;
    attrs.num_heads = 4;
    attrs.num_kv_heads = 2;
    auto op = LinearAttention::create(attrs, Backend::CPU);

    // Q: [B=1, H=4, Sq=1, D=64]
    TensorDesc q_desc;
    q_desc.rank = 4;
    q_desc.dims = {1, 4, 1, 64};
    q_desc.dtype = DataType::f32;
    q_desc.layout = TensorLayout::NCHW;

    // K: [B=1, H_kv=2, Sk=1, D=64]
    TensorDesc k_desc;
    k_desc.rank = 4;
    k_desc.dims = {1, 2, 1, 64};
    k_desc.dtype = DataType::f32;
    k_desc.layout = TensorLayout::NCHW;

    // V: same as K
    TensorDesc v_desc = k_desc;

    // Gate: [B=1, H=4, Sq=1, D=64]
    TensorDesc g_desc = q_desc;

    // 4 inputs: Q, K, V, Gate (State is in outputs, not inputs)
    const TensorDesc desc_arr[] = {q_desc, k_desc, v_desc, g_desc};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs.size(), size_t(2));

    // outputs[0] = Output, same shape as Q
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(4));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(4));
    NNOPS_EXPECT_EQ(descs[0].dims[2], int64_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[3], int64_t(64));
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    // outputs[1] = State [B=1, H_kv=2, D=64, D=64]
    NNOPS_EXPECT_EQ(descs[1].rank, int64_t(4));
    NNOPS_EXPECT_EQ(descs[1].dims[0], int64_t(1));
    NNOPS_EXPECT_EQ(descs[1].dims[1], int64_t(2));
    NNOPS_EXPECT_EQ(descs[1].dims[2], int64_t(64));
    NNOPS_EXPECT_EQ(descs[1].dims[3], int64_t(64));
    NNOPS_EXPECT_EQ(descs[1].dtype, DataType::f32);
}
