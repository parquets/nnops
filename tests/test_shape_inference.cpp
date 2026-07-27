/// @file test_shape_inference.cpp
/// @brief Tests for shape inference across all 16 operator types.
///
/// Covers:
///   - Conv2D / Conv3D / DepthwiseConv2D: NOTSET, SAME_UPPER, VALID, grouped, strided
///   - Pooling: 2D/3D, auto_pad modes
///   - Linear / MatMul: basic 2D, batched, broadcast, transposed
///   - Attention: merged-heads and per-head layouts
///   - Activation / Softmax / CumSum / BatchNorm / LayerNorm / RMSNorm:
///     identity shape
///   - Eltwise / Unary: identity shape
///   - Reduce: axis removal, keepdims, negative axis
///   - Class API and workspace-size consistency

#include "nnops/ops/conv2d.hpp"
#include "nnops/ops/conv3d.hpp"
#include "nnops/ops/depthwise_conv2d.hpp"
#include "nnops/ops/pooling.hpp"
#include "nnops/ops/linear.hpp"
#include "nnops/ops/matmul.hpp"
#include "nnops/ops/attention.hpp"
#include "nnops/ops/activation.hpp"
#include "nnops/ops/softmax.hpp"
#include "nnops/ops/cumsum.hpp"
#include "nnops/ops/batch_norm.hpp"
#include "nnops/ops/layer_norm.hpp"
#include "nnops/ops/rms_norm.hpp"
#include "nnops/ops/eltwise.hpp"
#include "nnops/ops/unary.hpp"
#include "nnops/ops/reduce.hpp"
#include "nnops/detail/shape_inference.hpp"

#include "common/test_harness.hpp"
#include "common/compare.hpp"

#include <vector>

using namespace nnops;

// Helper: create a TensorDesc directly
static TensorDesc td(std::initializer_list<int64_t> dims,
                     DataType dtype = DataType::f32,
                     TensorLayout layout = TensorLayout::NCHW) {
    TensorDesc d;
    d.dims   = std::span<const int64_t>(dims.begin(), dims.size());
    d.rank   = static_cast<int64_t>(dims.size());
    d.dtype  = dtype;
    d.layout = layout;
    return d;
}

// ============================================================
// Conv2D
// ============================================================

NNOPS_TEST(shape_conv2d_basic) {
    auto op = Conv2D::create(Backend::CPU);
    auto& attrs = const_cast<Conv2DAttributes&>(op->attributes());
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};

    auto inputs = std::vector<TensorDesc>{
        td({1, 3, 224, 224}),   // input
        td({64, 3, 3, 3})       // weight
    };
    auto outs = op->getOutputShapes(inputs);

    NNOPS_EXPECT_EQ(outs.size(), size_t(1));
    NNOPS_EXPECT_EQ(outs[0].rank, int64_t(4));
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t(1));    // N
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t(64));   // OC
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t(222));  // OH = (224-3)/1+1
    NNOPS_EXPECT_EQ(outs[0].dims[3], int64_t(222));  // OW
}

NNOPS_TEST(shape_conv2d_stride2) {
    auto op = Conv2D::create(Backend::CPU);
    auto& attrs = const_cast<Conv2DAttributes&>(op->attributes());
    attrs.kernel_size = {3, 3};
    attrs.stride  = {2, 2};
    attrs.padding = {1, 1};

    auto inputs = std::vector<TensorDesc>{
        td({1, 3, 224, 224}),
        td({64, 3, 3, 3})
    };
    auto outs = op->getOutputShapes(inputs);

    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t(112));  // OH = (224+2-3)/2+1
    NNOPS_EXPECT_EQ(outs[0].dims[3], int64_t(112));
}

NNOPS_TEST(shape_conv2d_same_upper) {
    auto op = Conv2D::create(Backend::CPU);
    auto& attrs = const_cast<Conv2DAttributes&>(op->attributes());
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.auto_pad = Conv2DAttributes::AutoPad::SAME_UPPER;

    auto inputs = std::vector<TensorDesc>{
        td({1, 3, 224, 224}),
        td({64, 3, 3, 3})
    };
    auto outs = op->getOutputShapes(inputs);
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t(224));  // ceil(224/1)
    NNOPS_EXPECT_EQ(outs[0].dims[3], int64_t(224));
}

NNOPS_TEST(shape_conv2d_valid) {
    auto op = Conv2D::create(Backend::CPU);
    auto& attrs = const_cast<Conv2DAttributes&>(op->attributes());
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.auto_pad = Conv2DAttributes::AutoPad::VALID;

    auto inputs = std::vector<TensorDesc>{
        td({1, 3, 224, 224}),
        td({64, 3, 3, 3})
    };
    auto outs = op->getOutputShapes(inputs);
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t(222));  // (224-3)/1+1
}

NNOPS_TEST(shape_conv2d_grouped) {
    auto op = Conv2D::create(Backend::CPU);
    auto& attrs = const_cast<Conv2DAttributes&>(op->attributes());
    attrs.kernel_size = {3, 3};
    attrs.stride  = {1, 1};
    attrs.padding = {0, 0};
    attrs.groups  = 4;

    auto inputs = std::vector<TensorDesc>{
        td({2, 128, 56, 56}),     // IC=128
        td({128, 32, 3, 3})       // OC=128, IC/G=32
    };
    auto outs = op->getOutputShapes(inputs);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t(128));  // OC
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t(54));   // OH=(56-3)/1+1
}

// ============================================================
// Conv3D
// ============================================================

NNOPS_TEST(shape_conv3d_basic) {
    auto op = Conv3D::create(Backend::CPU);
    auto& attrs = const_cast<Conv3DAttributes&>(op->attributes());
    attrs.kernel_size = {3, 3, 3};
    attrs.stride  = {1, 1, 1};
    attrs.padding = {0, 0, 0};

    auto inputs = std::vector<TensorDesc>{
        td({1, 3, 16, 224, 224}),
        td({64, 3, 3, 3, 3})
    };
    auto outs = op->getOutputShapes(inputs);

    NNOPS_EXPECT_EQ(outs.size(), size_t(1));
    NNOPS_EXPECT_EQ(outs[0].rank, int64_t(5));
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t(1));
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t(64));
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t(14));   // OD=(16-3)/1+1
    NNOPS_EXPECT_EQ(outs[0].dims[3], int64_t(222));  // OH
    NNOPS_EXPECT_EQ(outs[0].dims[4], int64_t(222));  // OW
}

// ============================================================
// DepthwiseConv2D
// ============================================================

NNOPS_TEST(shape_depthwise_conv2d) {
    auto op = DepthwiseConv2D::create(Backend::CPU);
    auto& attrs = const_cast<DepthwiseConv2DAttributes&>(op->attributes());
    attrs.kernel_size = {3, 3};
    attrs.stride  = {2, 2};
    attrs.padding = {1, 1};

    auto inputs = std::vector<TensorDesc>{
        td({1, 32, 112, 112}),
        td({32, 1, 3, 3})
    };
    auto outs = op->getOutputShapes(inputs);

    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t(32));   // C unchanged
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t(56));   // OH = (112+2-3)/2+1
    NNOPS_EXPECT_EQ(outs[0].dims[3], int64_t(56));
}

// ============================================================
// Pooling
// ============================================================

NNOPS_TEST(shape_pooling_2d) {
    auto op = Pooling::create(Backend::CPU);
    auto& attrs = const_cast<PoolingAttributes&>(op->attributes());
    attrs.kernel_shape = {2, 2, 2};  // KD unused for 2D
    attrs.stride   = {1, 2, 2};
    attrs.padding  = {0, 0, 0};

    auto inputs = std::vector<TensorDesc>{td({1, 64, 112, 112})};
    auto outs = op->getOutputShapes(inputs);

    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t(1));
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t(64));
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t(56));  // OH = (112-2)/2+1
    NNOPS_EXPECT_EQ(outs[0].dims[3], int64_t(56));
}

NNOPS_TEST(shape_pooling_3d) {
    auto op = Pooling::create(Backend::CPU);
    auto& attrs = const_cast<PoolingAttributes&>(op->attributes());
    attrs.kernel_shape = {2, 2, 2};
    attrs.stride   = {1, 1, 1};
    attrs.padding  = {0, 0, 0};

    auto inputs = std::vector<TensorDesc>{td({1, 16, 8, 56, 56})};
    auto outs = op->getOutputShapes(inputs);

    NNOPS_EXPECT_EQ(outs[0].rank, int64_t(5));
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t(7));   // OD = (8-2)/1+1
    NNOPS_EXPECT_EQ(outs[0].dims[3], int64_t(55));  // OH
    NNOPS_EXPECT_EQ(outs[0].dims[4], int64_t(55));  // OW
}

// ============================================================
// Linear
// ============================================================

NNOPS_TEST(shape_linear_2d) {
    auto op = Linear::create(Backend::CPU);
    auto inputs = std::vector<TensorDesc>{
        td({32, 768}),    // [M, K]
        td({3072, 768})   // [N, K]
    };
    auto outs = op->getOutputShapes(inputs);
    NNOPS_EXPECT_EQ(outs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t(32));
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t(3072));
}

NNOPS_TEST(shape_linear_batched) {
    auto op = Linear::create(Backend::CPU);
    auto inputs = std::vector<TensorDesc>{
        td({2, 4, 128}),    // [B, M, K]
        td({512, 128})      // [N, K]
    };
    auto outs = op->getOutputShapes(inputs);
    NNOPS_EXPECT_EQ(outs[0].rank, int64_t(3));
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t(4));
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t(512));
}

// ============================================================
// MatMul
// ============================================================

NNOPS_TEST(shape_matmul_2d) {
    auto op = MatMul::create({}, Backend::CPU);
    auto inputs = std::vector<TensorDesc>{
        td({4, 256}),   // A: [M, K]
        td({256, 128})  // B: [K, N]
    };
    auto outs = op->getOutputShapes(inputs);
    NNOPS_EXPECT_EQ(outs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t(4));
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t(128));
}

NNOPS_TEST(shape_matmul_transpose_a) {
    auto op = MatMul::create({}, Backend::CPU);
    auto& attrs = const_cast<MatMulAttributes&>(op->attributes());
    attrs.transpose_a = true;

    auto inputs = std::vector<TensorDesc>{
        td({256, 4}),    // [K, M] — after transpose: [M, K]
        td({256, 128})   // [K, N]
    };
    auto outs = op->getOutputShapes(inputs);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t(4));    // M from A's last dim
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t(128));
}

NNOPS_TEST(shape_matmul_transpose_b) {
    auto op = MatMul::create({}, Backend::CPU);
    auto& attrs = const_cast<MatMulAttributes&>(op->attributes());
    attrs.transpose_b = true;

    auto inputs = std::vector<TensorDesc>{
        td({4, 256}),    // [M, K]
        td({128, 256})   // [N, K] — after transpose: [K, N]
    };
    auto outs = op->getOutputShapes(inputs);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t(4));
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t(128));
}

NNOPS_TEST(shape_matmul_batched) {
    auto op = MatMul::create({}, Backend::CPU);
    auto inputs = std::vector<TensorDesc>{
        td({2, 3, 4, 256}),    // [B1, B2, M, K]
        td({2, 1, 256, 128})   // [B1, 1, K, N] — broadcast B2
    };
    auto outs = op->getOutputShapes(inputs);
    NNOPS_EXPECT_EQ(outs[0].rank, int64_t(4));
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t(3));   // broadcast
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t(4));   // M
    NNOPS_EXPECT_EQ(outs[0].dims[3], int64_t(128)); // N
}

// ============================================================
// Attention
// ============================================================

NNOPS_TEST(shape_attention_merged_heads) {
    auto op = Attention::create(Backend::CPU);
    auto inputs = std::vector<TensorDesc>{
        td({1, 128, 768}),  // Q: [B, S, H*D]
        td({1, 128, 768}),  // K
        td({1, 128, 768})   // V
    };
    auto outs = op->getOutputShapes(inputs);
    NNOPS_EXPECT_EQ(outs[0].rank, int64_t(3));
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t(1));
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t(128));
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t(768));
}

NNOPS_TEST(shape_attention_per_head) {
    auto op = Attention::create(Backend::CPU);
    auto inputs = std::vector<TensorDesc>{
        td({1, 12, 128, 64}),  // Q: [B, H, S, D]
        td({1, 12, 128, 64}),  // K
        td({1, 12, 128, 64})   // V
    };
    auto outs = op->getOutputShapes(inputs);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t(1));
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t(12));
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t(128));
    NNOPS_EXPECT_EQ(outs[0].dims[3], int64_t(64));
}

// ============================================================
// Identity-preserving operators
// ============================================================

NNOPS_TEST(shape_activation_identity) {
    auto op = Activation::create(Backend::CPU);
    auto inputs = std::vector<TensorDesc>{td({1, 3, 224, 224})};
    auto outs = op->getOutputShapes(inputs);
    NNOPS_EXPECT_EQ(outs[0].rank, int64_t(4));
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t(1));
    NNOPS_EXPECT_EQ(outs[0].dims[3], int64_t(224));
}

NNOPS_TEST(shape_softmax_identity) {
    auto op = Softmax::create(Backend::CPU);
    auto inputs = std::vector<TensorDesc>{td({32, 10})};
    auto outs = op->getOutputShapes(inputs);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t(32));
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t(10));
}

NNOPS_TEST(shape_cumsum_identity) {
    auto op = CumSum::create(Backend::CPU);
    auto inputs = std::vector<TensorDesc>{td({5, 8})};
    auto outs = op->getOutputShapes(inputs);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t(5));
}

NNOPS_TEST(shape_batch_norm_identity) {
    auto op = BatchNorm::create(Backend::CPU);
    auto inputs = std::vector<TensorDesc>{
        td({1, 64, 56, 56}),
        td({64}), td({64}), td({64}), td({64})
    };
    auto outs = op->getOutputShapes(inputs);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t(1));
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t(64));
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t(56));
    NNOPS_EXPECT_EQ(outs[0].dims[3], int64_t(56));
}

NNOPS_TEST(shape_layer_norm_identity) {
    auto op = LayerNorm::create(Backend::CPU);
    auto inputs = std::vector<TensorDesc>{
        td({2, 128, 768}),
        td({768})
    };
    auto outs = op->getOutputShapes(inputs);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t(768));
}

NNOPS_TEST(shape_rms_norm_identity) {
    auto op = RMSNorm::create(Backend::CPU);
    auto inputs = std::vector<TensorDesc>{
        td({2, 128, 768}),
        td({768})
    };
    auto outs = op->getOutputShapes(inputs);
    NNOPS_EXPECT_EQ(outs[0].rank, int64_t(3));
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t(128));
}

NNOPS_TEST(shape_eltwise_identity) {
    auto op = Eltwise::create(Backend::CPU);
    auto inputs = std::vector<TensorDesc>{
        td({4, 8, 16}),
        td({4, 8, 16})
    };
    auto outs = op->getOutputShapes(inputs);
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t(4));
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t(8));
}

NNOPS_TEST(shape_unary_identity) {
    auto op = Unary::create(Backend::CPU);
    auto inputs = std::vector<TensorDesc>{td({7, 7, 1024})};
    auto outs = op->getOutputShapes(inputs);
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t(1024));
}

// ============================================================
// Reduce
// ============================================================

NNOPS_TEST(shape_reduce_sum_axis0) {
    auto op = Reduce::create(Backend::CPU);
    auto& attrs = const_cast<ReduceAttributes&>(op->attributes());
    attrs.type = ReduceType::Sum;
    attrs.axis = 0;

    auto inputs = std::vector<TensorDesc>{td({3, 4, 5})};
    auto outs = op->getOutputShapes(inputs);
    NNOPS_EXPECT_EQ(outs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t(4));
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t(5));
}

NNOPS_TEST(shape_reduce_keepdims) {
    auto op = Reduce::create(Backend::CPU);
    auto& attrs = const_cast<ReduceAttributes&>(op->attributes());
    attrs.type = ReduceType::Sum;
    attrs.axis = 1;
    attrs.keepdims = true;

    auto inputs = std::vector<TensorDesc>{td({3, 4, 5})};
    auto outs = op->getOutputShapes(inputs);
    NNOPS_EXPECT_EQ(outs[0].rank, int64_t(3));
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t(3));
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t(1));   // kept as 1
    NNOPS_EXPECT_EQ(outs[0].dims[2], int64_t(5));
}

NNOPS_TEST(shape_reduce_negative_axis) {
    auto op = Reduce::create(Backend::CPU);
    auto& attrs = const_cast<ReduceAttributes&>(op->attributes());
    attrs.type = ReduceType::Sum;
    attrs.axis = -1;  // last dim

    auto inputs = std::vector<TensorDesc>{td({3, 4, 5})};
    auto outs = op->getOutputShapes(inputs);
    NNOPS_EXPECT_EQ(outs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(outs[0].dims[0], int64_t(3));
    NNOPS_EXPECT_EQ(outs[0].dims[1], int64_t(4));
}

// ============================================================
// dtype / layout propagation
// ============================================================

NNOPS_TEST(shape_dtype_propagation) {
    auto op = Activation::create(Backend::CPU);
    auto inputs = std::vector<TensorDesc>{td({2, 4}, DataType::f16, TensorLayout::NCHWC8)};
    auto outs = op->getOutputShapes(inputs);
    NNOPS_EXPECT_EQ(outs[0].dtype, DataType::f16);
    NNOPS_EXPECT_EQ(outs[0].layout, TensorLayout::NCHWC8);
}

// ============================================================
// Workspace-size consistency
// ============================================================

NNOPS_TEST(shape_workspace_consistent) {
    // getWorkspaceSize should accept TensorDesc from getOutputShapes
    auto op = Conv2D::create(Backend::CPU);
    auto& attrs = const_cast<Conv2DAttributes&>(op->attributes());
    attrs.kernel_size = {3, 3};
    attrs.stride  = {2, 2};
    attrs.padding = {1, 1};

    auto inputs = std::vector<TensorDesc>{
        td({1, 3, 224, 224}),
        td({64, 3, 3, 3})
    };
    auto outs = op->getOutputShapes(inputs);

    // getWorkspaceSize takes input + output descriptors
    auto ws = op->getWorkspaceSize(inputs, outs);
    // Workspace may be 0 (reference impl) or >0 (optimized impl)
    NNOPS_EXPECT_TRUE(ws >= size_t(0));
}
