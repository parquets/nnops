/// Unit tests for MatMul operator — class API with CPU reference.
///
/// All tests use the MatMul class interface:
///   1. auto op = MatMul::create(attrs, Backend::CPU);
///   2. const TensorDesc arr[] = {a_desc, b_desc}; auto descs = op->getOutputTensorDesc(arr);
///   3. Validate descs[0].rank/dims/layout/dtype
///   4. Create output via nnops::test::make_planar(descs[0], buf.data())
///   5. const TensorView ins[] = {a, b}; op->compute(out, ins);

#include "nnops/ops/matmul.hpp"
#include "nnops/core/quant_params.hpp"
#include "backend/cpu/matmul/matmul_helper.h"   // get_matmul_plan — to pin the pack_a route
#include "common/test_harness.hpp"
#include "common/cpu_pool.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <vector>
#include <thread>
#include <cmath>

using namespace nnops;

// Reference kernels for comparison (test-local declarations, following the
// test_concat.cpp pattern).
namespace nnops::backend::cpu::reference {
void matmul_ref(const MatMulAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* workspace);
void matmul_dq_ref(const MatMulAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx,
                   void* workspace);
}

// f16 GEMM comparison tolerance: the arch-aware shared constant from
// common/test_helpers.hpp (x86 fp32 accumulators vs aarch64 native fp16).
using nnops::test::kF16AccumTol;

// ============================================================
// Basic 2D MatMul
// ============================================================

NNOPS_TEST(matmul_basic_2d) {
    // A: [2, 3], B: [3, 2]  →  C: [2, 2]
    const int64_t ashape[] = {2, 3};
    const int64_t bshape[] = {3, 2};

    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    float a_data[6] = {1, 2, 3, 4, 5, 6};
    // B = [[1, 0],
    //      [0, 1],
    //      [1, 1]]
    float b_data[6] = {1, 0, 0, 1, 1, 1};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    int64_t nelem = descs[0].numel();
    NNOPS_EXPECT_EQ(nelem, int64_t(4));
    std::vector<float> out_buf(nelem);
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    float* c = out_buf.data();
    // Row 0: dot({1,2,3}, B[:,0])=4, dot({1,2,3}, B[:,1])=5
    NNOPS_EXPECT_NEAR(c[0], 4.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[1], 5.0f, 1e-6f);
    // Row 1: dot({4,5,6}, B[:,0])=10, dot({4,5,6}, B[:,1])=11
    NNOPS_EXPECT_NEAR(c[2], 10.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[3], 11.0f, 1e-6f);
}

// ============================================================
// Transpose A
// ============================================================

NNOPS_TEST(matmul_transpose_a) {
    // A phys: [3, 2]  →  logical A^T: [2, 3]
    // B:      [3, 2]
    // C:      [2, 2]
    const int64_t ashape[] = {3, 2};
    const int64_t bshape[] = {3, 2};

    // Physical A = [[1, 2],     Logical A^T = [[1, 3, 5],
    //               [3, 4],                    [2, 4, 6]]
    //               [5, 6]]
    float a_data[6] = {1, 2, 3, 4, 5, 6};
    // B = [[1, 0],
    //      [0, 1],
    //      [1, 1]]
    float b_data[6] = {1, 0, 0, 1, 1, 1};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);

    MatMulAttributes attrs{};
    attrs.transpose_a = true;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));  // M = a.shape[1] = 2
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(2));  // N = b.shape[1] = 2
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    int64_t nelem = descs[0].numel();
    std::vector<float> out_buf(nelem);
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    float* c = out_buf.data();
    // Logical A^T = [[1,3,5],[2,4,6]]
    // Row 0: dot({1,3,5}, {1,0,1}) = 6, dot({1,3,5}, {0,1,1}) = 8
    NNOPS_EXPECT_NEAR(c[0], 6.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[1], 8.0f, 1e-6f);
    // Row 1: dot({2,4,6}, {1,0,1}) = 8, dot({2,4,6}, {0,1,1}) = 10
    NNOPS_EXPECT_NEAR(c[2], 8.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[3], 10.0f, 1e-6f);
}

// ============================================================
// Transpose B
// ============================================================

NNOPS_TEST(matmul_transpose_b) {
    // A:      [2, 3]
    // B phys: [2, 3]  →  logical B^T: [3, 2]
    // C:      [2, 2]
    const int64_t ashape[] = {2, 3};
    const int64_t bshape[] = {2, 3};

    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    float a_data[6] = {1, 2, 3, 4, 5, 6};
    // Physical B = [[1, 0, 0],    Logical B^T = [[1, 0],
    //               [0, 1, 0]]                   [0, 1],
    //                                            [0, 0]]
    float b_data[6] = {1, 0, 0, 0, 1, 0};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);

    MatMulAttributes attrs{};
    attrs.transpose_b = true;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));  // M
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(2));  // N = b.shape[0] = 2
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    int64_t nelem = descs[0].numel();
    std::vector<float> out_buf(nelem);
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    float* c = out_buf.data();
    // B^T logical = [[1,0],[0,1],[0,0]]
    // Row 0: dot({1,2,3}, B^T[:,0]) = 1, dot({1,2,3}, B^T[:,1]) = 2
    NNOPS_EXPECT_NEAR(c[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[1], 2.0f, 1e-6f);
    // Row 1: dot({4,5,6}, B^T[:,0]) = 4, dot({4,5,6}, B^T[:,1]) = 5
    NNOPS_EXPECT_NEAR(c[2], 4.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[3], 5.0f, 1e-6f);
}

// ============================================================
// Transpose Both
// ============================================================

NNOPS_TEST(matmul_transpose_both) {
    // A phys: [3, 2]  →  logical A^T: [2, 3]
    // B phys: [2, 3]  →  logical B^T: [3, 2]
    // C:      [2, 2]
    const int64_t ashape[] = {3, 2};
    const int64_t bshape[] = {2, 3};

    // Physical A = [[1, 2],     A^T logical = [[1, 3, 5],
    //               [3, 4],                    [2, 4, 6]]
    //               [5, 6]]
    float a_data[6] = {1, 2, 3, 4, 5, 6};
    // Physical B = [[1, 0, 0],   B^T logical = [[1, 0],
    //               [0, 1, 0]]                   [0, 1],
    //                                            [0, 0]]
    float b_data[6] = {1, 0, 0, 0, 1, 0};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);

    MatMulAttributes attrs{};
    attrs.transpose_a = true;
    attrs.transpose_b = true;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));  // M = a.shape[1] = 2
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(2));  // N = b.shape[0] = 2
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    int64_t nelem = descs[0].numel();
    std::vector<float> out_buf(nelem);
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    float* c = out_buf.data();
    // A^T = [[1,3,5],[2,4,6]], B^T = [[1,0],[0,1],[0,0]]
    // Row 0: dot({1,3,5}, {1,0,0}) = 1, dot({1,3,5}, {0,1,0}) = 3
    NNOPS_EXPECT_NEAR(c[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[1], 3.0f, 1e-6f);
    // Row 1: dot({2,4,6}, {1,0,0}) = 2, dot({2,4,6}, {0,1,0}) = 4
    NNOPS_EXPECT_NEAR(c[2], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[3], 4.0f, 1e-6f);
}

// ============================================================
// Batch MatMul (3D × 3D)
// ============================================================

NNOPS_TEST(matmul_batch_3d) {
    // A: [2, 2, 3], B: [2, 3, 2]  →  C: [2, 2, 2]
    const int64_t ashape[] = {2, 2, 3};
    const int64_t bshape[] = {2, 3, 2};

    // Batch 0 A = [[1,2,3],[4,5,6]]
    // Batch 1 A = [[7,8,9],[10,11,12]]
    float a_data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    // Batch 0 B = [[1,0],[0,1],[1,1]]
    // Batch 1 B = [[1,0],[0,1],[0,0]]
    float b_data[12] = {1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(3));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));  // batch
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(2));  // M
    NNOPS_EXPECT_EQ(descs[0].dims[2], int64_t(2));  // N
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    int64_t nelem = descs[0].numel();
    NNOPS_EXPECT_EQ(nelem, int64_t(8));
    std::vector<float> out_buf(nelem);
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    float* c = out_buf.data();

    // Batch 0: same as basic_2d
    NNOPS_EXPECT_NEAR(c[0], 4.0f, 1e-6f);   // dot({1,2,3}, {1,0,1})
    NNOPS_EXPECT_NEAR(c[1], 5.0f, 1e-6f);   // dot({1,2,3}, {0,1,1})
    NNOPS_EXPECT_NEAR(c[2], 10.0f, 1e-6f);  // dot({4,5,6}, {1,0,1})
    NNOPS_EXPECT_NEAR(c[3], 11.0f, 1e-6f);  // dot({4,5,6}, {0,1,1})

    // Batch 1: B=[[1,0],[0,1],[0,0]], so C = upper-left 2×2 of A
    NNOPS_EXPECT_NEAR(c[4], 7.0f, 1e-6f);   // dot({7,8,9}, {1,0,0})
    NNOPS_EXPECT_NEAR(c[5], 8.0f, 1e-6f);   // dot({7,8,9}, {0,1,0})
    NNOPS_EXPECT_NEAR(c[6], 10.0f, 1e-6f);  // dot({10,11,12}, {1,0,0})
    NNOPS_EXPECT_NEAR(c[7], 11.0f, 1e-6f);  // dot({10,11,12}, {0,1,0})
}

// ============================================================
// Broadcast A across batch (2D A × 3D B)
// ============================================================

NNOPS_TEST(matmul_broadcast_a_2d_x_3d) {
    // A: [2, 3] broadcast × B: [3, 3, 3]  →  C: [3, 2, 3]
    // (M=2, K=3 from A; N=3 from B; batch=3 from B)
    const int64_t ashape[] = {2, 3};
    const int64_t bshape[] = {3, 3, 3};

    // A = [[1, 0, 0],
    //      [0, 1, 0]]
    float a_data[6] = {1, 0, 0, 0, 1, 0};

    // B batch 0 = identity-like: [[1,0,0],[0,1,0],[0,0,1]]
    // B batch 1 = all ones:       [[1,1,1],[1,1,1],[1,1,1]]
    // B batch 2 = zeros:          [[0,0,0],[0,0,0],[0,0,0]]
    float b_data[27] = {
        1,0,0, 0,1,0, 0,0,1,   // batch 0
        1,1,1, 1,1,1, 1,1,1,   // batch 1
        0,0,0, 0,0,0, 0,0,0,   // batch 2
    };

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(3));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(3));  // batch (from B)
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(2));  // M (from A)
    NNOPS_EXPECT_EQ(descs[0].dims[2], int64_t(3));  // N (from B)
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    int64_t nelem = descs[0].numel();
    NNOPS_EXPECT_EQ(nelem, int64_t(18));
    std::vector<float> out_buf(nelem);
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    float* c = out_buf.data();

    // Batch 0: A × I = A  →  [[1,0,0],[0,1,0]]
    NNOPS_EXPECT_NEAR(c[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[1], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[2], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[3], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[4], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[5], 0.0f, 1e-6f);

    // Batch 1: A × all-ones → each row sum of A's row: [[1,1,1],[1,1,1]]
    NNOPS_EXPECT_NEAR(c[6], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[7], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[8], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[9], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[10], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[11], 1.0f, 1e-6f);

    // Batch 2: A × zeros = all zeros
    NNOPS_EXPECT_NEAR(c[12], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[13], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[14], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[15], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[16], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[17], 0.0f, 1e-6f);
}

// ============================================================
// Broadcast B across batch (3D A × 2D B)
// ============================================================

NNOPS_TEST(matmul_broadcast_b_3d_x_2d) {
    // A: [3, 2, 3] × B: [3, 2]  →  C: [3, 2, 2]
    const int64_t ashape[] = {3, 2, 3};
    const int64_t bshape[] = {3, 2};

    // A batch 0 = [[1,2,3],[4,5,6]]
    // A batch 1 = [[1,0,0],[0,1,0]]
    // A batch 2 = [[0,0,0],[0,0,0]]
    float a_data[18] = {
        1,2,3, 4,5,6,
        1,0,0, 0,1,0,
        0,0,0, 0,0,0,
    };

    // B = [[1, 0],
    //      [0, 1],
    //      [1, 1]]
    float b_data[6] = {1, 0, 0, 1, 1, 1};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(3));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(3));  // batch
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(2));  // M
    NNOPS_EXPECT_EQ(descs[0].dims[2], int64_t(2));  // N
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    int64_t nelem = descs[0].numel();
    NNOPS_EXPECT_EQ(nelem, int64_t(12));
    std::vector<float> out_buf(nelem);
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    float* c = out_buf.data();

    // Batch 0: same as basic_2d
    NNOPS_EXPECT_NEAR(c[0], 4.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[1], 5.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[2], 10.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[3], 11.0f, 1e-6f);

    // Batch 1: A = [[1,0,0],[0,1,0]], B same
    // C[0,0] = dot({1,0,0},{1,0,1}) = 1
    // C[0,1] = dot({1,0,0},{0,1,1}) = 0
    // C[1,0] = dot({0,1,0},{1,0,1}) = 0
    // C[1,1] = dot({0,1,0},{0,1,1}) = 1
    NNOPS_EXPECT_NEAR(c[4], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[5], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[6], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[7], 1.0f, 1e-6f);

    // Batch 2: A = zeros → C = zeros
    NNOPS_EXPECT_NEAR(c[8], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[9], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[10], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[11], 0.0f, 1e-6f);
}

// ============================================================
// Multi-dim broadcast (higher rank)
// ============================================================

NNOPS_TEST(matmul_broadcast_multi_dim) {
    // A: [2, 1, 3, 4]  →  batch = [2, 1, 3], M=3, K=4
    // B: [1, 4, 4, 2]  →  batch = [1, 4, 4], K=4, N=2
    // broadcast batch: [2, 4, 3]
    // C: [2, 4, 3, 2]
    const int64_t ashape[] = {2, 1, 3, 4};
    const int64_t bshape[] = {1, 4, 4, 2};

    // A[0,0,:,:] = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
    // A[1,0,:,:] = [[0,0,0,1],[0,0,0,0],[0,0,0,0]]
    // Total elements: 2*1*3*4 = 24
    std::vector<float> a_data(24, 0.0f);
    a_data[0] = 1.0f;                       // A[0,0,0,0]
    a_data[5] = 1.0f;                       // A[0,0,1,1]
    a_data[10] = 1.0f;                      // A[0,0,2,2]
    a_data[15] = 1.0f;                      // A[1,0,0,3]
    // Everything else is 0

    // B[0,0,:,:] = [[1,0],[0,1],[0,0],[0,0]]
    // B[0,1,:,:] = [[0,0],[0,0],[1,0],[0,1]]
    // B[0,2,:,:] = [[0,0],[0,0],[0,0],[0,0]]
    // B[0,3,:,:] = [[1,1],[1,1],[1,1],[1,1]]
    // Total elements: 1*4*4*2 = 32
    std::vector<float> b_data(32, 0.0f);
    // B[0,0,:,:]: row 0=[1,0], row 1=[0,1]
    b_data[0] = 1.0f; b_data[3] = 1.0f;
    // B[0,1,:,:]: row 2=[1,0], row 3=[0,1]
    b_data[8+4+0] = 1.0f;   // B[0,1,2,0] = 1
    b_data[8+6+1] = 1.0f;   // B[0,1,3,1] = 1
    // B[0,3,:,:]: all ones
    for (int i = 0; i < 8; ++i) {
        b_data[24 + i] = 1.0f;
    }

    TensorView a(ashape, DataType::f32, a_data.data());
    TensorView b(bshape, DataType::f32, b_data.data());

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(4));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));   // batch dim 0
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(4));   // batch dim 1
    NNOPS_EXPECT_EQ(descs[0].dims[2], int64_t(3));   // M
    NNOPS_EXPECT_EQ(descs[0].dims[3], int64_t(2));   // N
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    int64_t nelem = descs[0].numel();
    NNOPS_EXPECT_EQ(nelem, int64_t(2 * 4 * 3 * 2));
    std::vector<float> out_buf(nelem);
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    float* c = out_buf.data();

    // Check all outputs are finite (no NaN/Inf)
    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(c[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(c[i]));
    }

    // Verify specific positions:
    // C[b0=0, b1=0, :, :] = A[0,0,:,:] × B[0,0,:,:]
    // A[0,0] has 1s at (0,0), (1,1), (2,2)
    // B[0,0] has 1s at (0,0), (1,1)
    // C[0,0,0,0] = dot({1,0,0,0}, {1,0,0,0}) = 1
    // C[0,0,0,1] = dot({1,0,0,0}, {0,1,0,0}) = 0
    // C[0,0,1,0] = dot({0,1,0,0}, {1,0,0,0}) = 0
    // C[0,0,1,1] = dot({0,1,0,0}, {0,1,0,0}) = 1
    // C[0,0,2,0] = dot({0,0,1,0}, {1,0,0,0}) = 0
    // C[0,0,2,1] = dot({0,0,1,0}, {0,1,0,0}) = 0
    int64_t s0 = 0;  // offset for b0=0, b1=0: 0*4*3*2 + 0*3*2 = 0
    NNOPS_EXPECT_NEAR(c[s0 + 0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[s0 + 1], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[s0 + 2], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[s0 + 3], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[s0 + 4], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[s0 + 5], 0.0f, 1e-6f);

    // C[b0=0, b1=1, :, :] = A[0,0,:,:] × B[0,1,:,:]
    // B[0,1] has 1s at (2,0) and (3,1)
    // C[0,1,0,0] = dot({1,0,0,0}, {0,0,1,0}) = 0
    // C[0,1,0,1] = dot({1,0,0,0}, {0,0,0,1}) = 0
    // C[0,1,2,0] = dot({0,0,1,0}, {0,0,1,0}) = 1
    // C[0,1,2,1] = dot({0,0,1,0}, {0,0,0,1}) = 0
    int64_t s1 = 1 * 3 * 2;  // b0=0, b1=1: offset = 1 * (M*N)
    NNOPS_EXPECT_NEAR(c[s1 + 4], 1.0f, 1e-6f);   // C[0,1,2,0] = 1
    NNOPS_EXPECT_NEAR(c[s1 + 5], 0.0f, 1e-6f);   // C[0,1,2,1] = 0

    // C[b0=0, b1=3, :, :] = A[0,0,:,:] × B[0,3,:,:]
    // B[0,3] is all ones
    // C[0,3,0,0] = dot({1,0,0,0}, {1,1,1,1}) = 1
    // C[0,3,0,1] = dot({1,0,0,0}, {1,1,1,1}) = 1
    // C[0,3,1,0] = dot({0,1,0,0}, {1,1,1,1}) = 1
    // C[0,3,1,1] = dot({0,1,0,0}, {1,1,1,1}) = 1
    // C[0,3,2,0] = dot({0,0,1,0}, {1,1,1,1}) = 1
    // C[0,3,2,1] = dot({0,0,1,0}, {1,1,1,1}) = 1
    int64_t s3 = 3 * 3 * 2;  // b0=0, b1=3
    for (int j = 0; j < 6; ++j) {
        NNOPS_EXPECT_NEAR(c[s3 + j], 1.0f, 1e-6f);
    }

    // C[b0=1, b1=0, :, :] = A[1,0,:,:] × B[0,0,:,:]
    // A[1,0] has 1 at (0,3) only, B[0,0] has 1s at (0,0) and (1,1)
    // C[1,0,0,0] = dot({0,0,0,1}, {1,0,0,0}) = 0
    // C[1,0,0,1] = dot({0,0,0,1}, {0,1,0,0}) = 0
    int64_t s10 = 1 * 4 * 3 * 2 + 0 * 3 * 2;  // b0=1, b1=0
    NNOPS_EXPECT_NEAR(c[s10 + 0], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[s10 + 1], 0.0f, 1e-6f);

    // C[b0=1, b1=1, :, :] = A[1,0,:,:] × B[0,1,:,:]
    // A[1,0] has 1 at (0,3) only, B[0,1] has 1 at (3,1) only
    // C[1,1,0,1] = dot({0,0,0,1}, {0,0,0,1}) = 1
    int64_t s11 = 1 * 4 * 3 * 2 + 1 * 3 * 2;  // b0=1, b1=1
    NNOPS_EXPECT_NEAR(c[s11 + 1], 1.0f, 1e-6f);

    // C[b0=1, b1=3, :, :] = A[1,0,:,:] × B[0,3,:,:]
    // A[1,0] row 0 = {0,0,0,1}, B[0,3] all ones
    // C[1,3,0,0] = dot({0,0,0,1}, {1,1,1,1}) = 1
    // C[1,3,0,1] = dot({0,0,0,1}, {1,1,1,1}) = 1
    // C[1,3,1,:] = 0, C[1,3,2,:] = 0 (A rows 1,2 are all zero)
    int64_t s13 = 1 * 4 * 3 * 2 + 3 * 3 * 2;  // b0=1, b1=3
    NNOPS_EXPECT_NEAR(c[s13 + 0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[s13 + 1], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[s13 + 2], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[s13 + 3], 0.0f, 1e-6f);
}

// ============================================================
// Beta (add to existing output — fused C_out = A×B + beta*C_old)
// ============================================================

NNOPS_TEST(matmul_beta_add_to_output) {
    // A: [2, 3], B: [3, 2]  →  C: [2, 2]
    const int64_t ashape[] = {2, 3};
    const int64_t bshape[] = {3, 2};

    float a_data[6] = {1, 2, 3, 4, 5, 6};
    float b_data[6] = {1, 0, 0, 1, 1, 1};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);

    MatMulAttributes attrs{};
    attrs.beta = 0.5f;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    int64_t nelem = descs[0].numel();
    std::vector<float> out_buf(nelem);
    // Pre-fill C_old with 10.0
    for (auto& v : out_buf) {
        v = 10.0f;
    }
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    float* c = out_buf.data();
    // C_new = A×B + 0.5 * C_old
    // Row 0: A×B = (4,5), +0.5*10 = (4+5, 5+5) = (9, 10)
    NNOPS_EXPECT_NEAR(c[0], 9.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[1], 10.0f, 1e-6f);
    // Row 1: A×B = (10,11), +0.5*10 = (10+5, 11+5) = (15, 16)
    NNOPS_EXPECT_NEAR(c[2], 15.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[3], 16.0f, 1e-6f);
}

// ============================================================
// Beta = 0 (default — overwrite output)
// ============================================================

NNOPS_TEST(matmul_beta_zero_overwrite) {
    // A: [2, 3], B: [3, 2]  →  C: [2, 2]
    // Default beta=0 means C is fully overwritten (old values ignored).
    const int64_t ashape[] = {2, 3};
    const int64_t bshape[] = {3, 2};

    float a_data[6] = {1, 2, 3, 4, 5, 6};
    float b_data[6] = {1, 0, 0, 1, 1, 1};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);

    MatMulAttributes attrs{};  // beta = 0.0 (default)
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    int64_t nelem = descs[0].numel();
    std::vector<float> out_buf(nelem);
    // Pre-fill with garbage: should be ignored since beta=0
    for (auto& v : out_buf) {
        v = 999.0f;
    }
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    float* c = out_buf.data();
    // C = A×B + 0*999 = A×B (same as basic_2d)
    NNOPS_EXPECT_NEAR(c[0], 4.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[1], 5.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[2], 10.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[3], 11.0f, 1e-6f);
}

// ============================================================
// Epilogue: RELU
// ============================================================

NNOPS_TEST(matmul_epilogue_relu) {
    // A: [2, 3], B: [3, 2]  →  C: [2, 2]
    // Using B that produces both positive and negative outputs.
    const int64_t ashape[] = {2, 3};
    const int64_t bshape[] = {3, 2};

    // A = [[ 1,  2,  3],
    //      [-4, -5, -6]]
    float a_data[6] = {1, 2, 3, -4, -5, -6};
    // B = [[ 1, -1],
    //      [ 1, -1],
    //      [ 1, -1]]
    float b_data[6] = {1, -1, 1, -1, 1, -1};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);

    MatMulAttributes attrs{};
    attrs.epilogue.type = EpilogueActivateType::Relu;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    int64_t nelem = descs[0].numel();
    std::vector<float> out_buf(nelem);
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    float* c = out_buf.data();
    // A×B before RELU:
    // Row 0: dot({1,2,3}, {1,1,1})=6, dot({1,2,3}, {-1,-1,-1})=-6
    // Row 1: dot({-4,-5,-6}, {1,1,1})=-15, dot({-4,-5,-6}, {-1,-1,-1})=15
    // After RELU:
    NNOPS_EXPECT_NEAR(c[0], 6.0f, 1e-6f);    // relu(6) = 6
    NNOPS_EXPECT_NEAR(c[1], 0.0f, 1e-6f);    // relu(-6) = 0
    NNOPS_EXPECT_NEAR(c[2], 0.0f, 1e-6f);    // relu(-15) = 0
    NNOPS_EXPECT_NEAR(c[3], 15.0f, 1e-6f);   // relu(15) = 15
}

// ============================================================
// Epilogue + Beta together
// ============================================================

NNOPS_TEST(matmul_epilogue_relu_with_beta) {
    // A: [1, 2], B: [2, 2]  →  C: [1, 2]
    const int64_t ashape[] = {1, 2};
    const int64_t bshape[] = {2, 2};

    // A = [[1, -2]]
    float a_data[2] = {1, -2};
    // B = [[3, -4],
    //      [5, -6]]
    float b_data[4] = {3, -4, 5, -6};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);

    MatMulAttributes attrs{};
    attrs.beta = 0.5f;
    attrs.epilogue.type = EpilogueActivateType::Relu;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    int64_t nelem = descs[0].numel();
    std::vector<float> out_buf(nelem);
    // Pre-fill C_old with 10.0
    for (auto& v : out_buf) {
        v = 10.0f;
    }
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    float* c = out_buf.data();
    // A×B before epilogue:
    // C[0] = dot({1,-2}, {3,5}) = 1*3 + (-2)*5 = 3 - 10 = -7
    // C[1] = dot({1,-2}, {-4,-6}) = 1*(-4) + (-2)*(-6) = -4 + 12 = 8
    // After beta: -7 + 0.5*10 = -2  → relu(-2) = 0
    //             8 + 0.5*10 = 13  → relu(13) = 13
    NNOPS_EXPECT_NEAR(c[0], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[1], 13.0f, 1e-6f);
}

// ============================================================
// Bias — the optional third input
//
// C = epilogue(A×B + bias + beta×C_old), the bias broadcast down the columns.
// The order matters: it sits between the product and the activation, so Relu
// must see (product + bias), not product.
// ============================================================

namespace {

QuantParams bias_per_tensor(float scale, int32_t zp) {
    QuantParams qp;
    qp.granularity = QuantGranularity::PerTensor;
    qp.scale = scale;
    qp.zero_point = zp;
    return qp;
}

TensorView quant_view(const int64_t* shape, int64_t rank, DataType dt, void* data,
                      const QuantParams& qp) {
    return TensorView(std::span<const int64_t>(shape, static_cast<size_t>(rank)),
                      dt, data, TensorLayout::NCHW, qp);
}

/// Run a MatMul and matmul_ref over the same f32 operands and require
/// agreement. @p initial optionally seeds C in both outputs (for beta).
void expect_matches_ref(const MatMulAttributes& attrs,
                        const std::vector<TensorView>& ins,
                        std::span<const float> initial = {},
                        float tol = 1e-4f)
{
    std::vector<TensorDesc> descs;
    descs.reserve(ins.size());
    for (const auto& t : ins) { descs.push_back(t.desc()); }

    auto op = MatMul::create(attrs, Backend::CPU);
    const auto odescs = op->getOutputTensorDesc(descs);
    const TensorDesc& od = odescs[0];
    NNOPS_EXPECT_EQ(od.dtype, DataType::f32);

    std::vector<float> got(static_cast<size_t>(od.numel()), 0.0f);
    std::vector<float> want(got.size(), 0.0f);
    if (!initial.empty()) {
        std::copy(initial.begin(), initial.end(), got.begin());
        std::copy(initial.begin(), initial.end(), want.begin());
    }

    TensorView og = nnops::test::make_planar(od, got.data());
    TensorView ow = nnops::test::make_planar(od, want.data());

    op->compute(og, ins);
    nnops::backend::cpu::reference::matmul_ref(attrs, ow, ins, {}, nullptr);

    for (size_t i = 0; i < got.size(); ++i) {
        NNOPS_EXPECT_NEAR(got[i], want[i], tol * std::max(1.0f, std::abs(want[i])));
    }
}

}  // anonymous namespace

NNOPS_TEST(matmul_bias_basic_2d) {
    // A = [[1,2,3],[4,5,6]], B = [[1,0],[0,1],[1,1]]  →  A×B = [[4,5],[10,11]]
    // bias = [10,-20] adds down the columns: [[14,-15],[20,-9]].
    const int64_t ashape[] = {2, 3};
    const int64_t bshape[] = {3, 2};
    const int64_t bias_shape[] = {2};

    float a_data[6] = {1, 2, 3, 4, 5, 6};
    float b_data[6] = {1, 0, 0, 1, 1, 1};
    float bias_data[2] = {10, -20};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);
    TensorView bias(bias_shape, DataType::f32, bias_data);

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {a.desc(), b.desc(), bias.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b, bias};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], 14.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[1], -15.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[2], 20.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[3], -9.0f, 1e-6f);
}

NNOPS_TEST(matmul_bias_precedes_activation) {
    // Same product as matmul_epilogue_relu: [[6,-6],[-15,15]].
    // With bias [-10,10]: relu(6-10)=0, relu(-6+10)=4, relu(-15-10)=0,
    // relu(15+10)=25. Adding the bias after the Relu instead would give
    // [[0,10],[0,25]] — the 4 in cell [0,1] pins the order.
    const int64_t ashape[] = {2, 3};
    const int64_t bshape[] = {3, 2};
    const int64_t bias_shape[] = {2};

    float a_data[6] = {1, 2, 3, -4, -5, -6};
    float b_data[6] = {1, -1, 1, -1, 1, -1};
    float bias_data[2] = {-10, 10};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);
    TensorView bias(bias_shape, DataType::f32, bias_data);

    MatMulAttributes attrs{};
    attrs.epilogue.type = EpilogueActivateType::Relu;
    auto op = MatMul::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {a.desc(), b.desc(), bias.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b, bias};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[1], 4.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[2], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[3], 25.0f, 1e-6f);
}

NNOPS_TEST(matmul_bias_with_beta) {
    // Small route: A = [[1,-2]], B = [[3,-4],[5,-6]]  →  [[-7,8]].
    // +bias [1,-1] = [-6,7], +0.5×C_old(10) = [-1,12].
    const int64_t ashape[] = {1, 2};
    const int64_t bshape[] = {2, 2};
    const int64_t bias_shape[] = {2};

    float a_data[2] = {1, -2};
    float b_data[4] = {3, -4, 5, -6};
    float bias_data[2] = {1, -1};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);
    TensorView bias(bias_shape, DataType::f32, bias_data);

    MatMulAttributes attrs{};
    attrs.beta = 0.5f;
    auto op = MatMul::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {a.desc(), b.desc(), bias.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 10.0f);
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b, bias};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], -1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[1], 12.0f, 1e-6f);
}

NNOPS_TEST(matmul_bias_tiled_with_beta) {
    // A tiled shape, a random bias and beta = 1.5 over a seeded C — the
    // accumulate-into-C path under a biased call, against matmul_ref.
    auto [av, a] = test::make_random_tensor({37, 65}, -1.0f, 1.0f, 910);
    auto [bv, b] = test::make_random_tensor({65, 41}, -1.0f, 1.0f, 911);
    auto [zbv, bias] = test::make_random_tensor({41}, -0.5f, 0.5f, 912);

    MatMulAttributes attrs{};
    attrs.beta = 1.5f;
    std::vector<float> initial(37 * 41, 0.25f);
    expect_matches_ref(attrs, {a, b, bias}, initial);
}

NNOPS_TEST(matmul_bias_transpose_b) {
    // [M,K] × [N,K] (transpose_b) — the Linear weight layout — with a bias
    // over the N columns.
    auto [av, a] = test::make_random_tensor({23, 71}, -1.0f, 1.0f, 920);
    auto [bvt, bt] = test::make_random_tensor({19, 71}, -1.0f, 1.0f, 921);
    auto [zbv, bias] = test::make_random_tensor({19}, -0.5f, 0.5f, 922);

    MatMulAttributes attrs{};
    attrs.transpose_b = true;
    expect_matches_ref(attrs, {a, bt, bias});
}

NNOPS_TEST(matmul_bias_batched_broadcast) {
    // C[2,3,5] from a 3D A and a 2D B: the bias broadcasts over both the batch
    // and the M axis.
    {
        auto [av, a] = test::make_random_tensor({2, 3, 8}, -1.0f, 1.0f, 930);
        auto [bv, b] = test::make_random_tensor({8, 5}, -1.0f, 1.0f, 931);
        auto [zbv, bias] = test::make_random_tensor({5}, -0.5f, 0.5f, 932);
        expect_matches_ref(MatMulAttributes{}, {a, b, bias});
    }
    {
        // Both batched with a broadcast batch dim: [2,1,M,K] × [2,K,N].
        auto [av, a] = test::make_random_tensor({2, 1, 4, 9}, -1.0f, 1.0f, 933);
        auto [bv, b] = test::make_random_tensor({2, 9, 7}, -1.0f, 1.0f, 934);
        auto [zbv, bias] = test::make_random_tensor({7}, -0.5f, 0.5f, 935);
        expect_matches_ref(MatMulAttributes{}, {a, b, bias});
    }
}

NNOPS_TEST(matmul_bias_small_and_gemv_routes) {
    // M=1 decode (GEMV), a single output column, and a K that is not a
    // multiple of the pack width.
    {
        auto [av, a] = test::make_random_tensor({1, 17}, -1.0f, 1.0f, 940);
        auto [bv, b] = test::make_random_tensor({17, 13}, -1.0f, 1.0f, 941);
        auto [zbv, bias] = test::make_random_tensor({13}, -0.5f, 0.5f, 942);
        expect_matches_ref(MatMulAttributes{}, {a, b, bias});
    }
    {
        auto [av, a] = test::make_random_tensor({3, 23}, -1.0f, 1.0f, 943);
        auto [bv, b] = test::make_random_tensor({23, 1}, -1.0f, 1.0f, 944);
        auto [zbv, bias] = test::make_random_tensor({1}, -0.5f, 0.5f, 945);
        expect_matches_ref(MatMulAttributes{}, {a, b, bias});
    }
    {
        // The streaming GEMV route (M <= 4, above the naive MAC threshold),
        // with beta and alpha riding the same accumulator prime as the bias.
        auto [av, a] = test::make_random_tensor({2, 128}, -1.0f, 1.0f, 946);
        auto [bv, b] = test::make_random_tensor({128, 256}, -1.0f, 1.0f, 947);
        auto [zbv, bias] = test::make_random_tensor({256}, -0.5f, 0.5f, 948);
        expect_matches_ref(MatMulAttributes{}, {a, b, bias});

        MatMulAttributes attrs{};
        attrs.alpha = 1.5f;
        attrs.beta = 0.5f;
        const std::vector<float> initial(2 * 256, 0.25f);
        expect_matches_ref(attrs, {a, b, bias}, initial);
    }
}

NNOPS_TEST(matmul_bias_f16) {
    // f16 operands and an f16 bias, against the f32 reference on the same
    // values (f16 is computed in f32 on x86 and native f16 on aarch64, so the
    // comparison uses the shared accumulate tolerance).
    auto [a32, _] = test::make_random_tensor({12, 40}, -1.0f, 1.0f, 950);
    auto [b32, __] = test::make_random_tensor({40, 9}, -1.0f, 1.0f, 951);
    auto [z32, ___] = test::make_random_tensor({9}, -0.5f, 0.5f, 952);
    auto a_f16 = test::f32_to_f16(a32);
    auto b_f16 = test::f32_to_f16(b32);
    auto bias_f16 = test::f32_to_f16(z32);

    const int64_t a_shape[] = {12, 40};
    const int64_t b_shape[] = {40, 9};
    const int64_t bias_shape[] = {9};
    TensorView a(a_shape, DataType::f16, a_f16.data());
    TensorView b(b_shape, DataType::f16, b_f16.data());
    TensorView bias(bias_shape, DataType::f16, bias_f16.data());

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {a.desc(), b.desc(), bias.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

    std::vector<nnops::backend::cpu::half> out_buf(
        static_cast<size_t>(descs[0].numel()));
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b, bias};
    op->compute(output, ins);

    TensorView a32v(a_shape, DataType::f32, a32.data());
    TensorView b32v(b_shape, DataType::f32, b32.data());
    TensorView z32v(bias_shape, DataType::f32, z32.data());
    const TensorDesc ref_arr[] = {a32v.desc(), b32v.desc(), z32v.desc()};
    auto ref_descs = op->getOutputTensorDesc(ref_arr);
    std::vector<float> ref_buf(static_cast<size_t>(ref_descs[0].numel()));
    auto ref_out = nnops::test::make_planar(ref_descs[0], ref_buf.data());
    const TensorView ref_ins[] = {a32v, b32v, z32v};
    nnops::backend::cpu::reference::matmul_ref(attrs, ref_out, ref_ins, {}, nullptr);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_NEAR(simd::s_load(&out_buf[i]), ref_buf[i], kF16AccumTol);
    }
}

NNOPS_TEST(matmul_bias_int8_dequant) {
    // s8 × s8 → f32: the bias lands on the dequantized accumulate, before the
    // activation. C = acc·(0.5·0.25) + bias, then Relu.
    //   acc = [[1,2],[3,4]]  →  [[0.125,0.25],[0.375,0.5]]
    //   +bias [1,-1]         →  [[1.125,-0.75],[1.375,-0.5]]
    //   relu                 →  [[1.125,0],[1.375,0]]
    const int64_t ashape[] = {2, 2};
    const int64_t bshape[] = {2, 2};
    const int64_t bias_shape[] = {2};

    int8_t a_data[4] = {1, 2, 3, 4};
    int8_t b_data[4] = {1, 0, 0, 1};
    float bias_data[2] = {1.0f, -1.0f};

    TensorView a = quant_view(ashape, 2, DataType::s8, a_data,
                              bias_per_tensor(0.5f, 0));
    TensorView b = quant_view(bshape, 2, DataType::s8, b_data,
                              bias_per_tensor(0.25f, 0));
    TensorView bias(bias_shape, DataType::f32, bias_data);

    MatMulAttributes attrs{};
    attrs.output_dtype = DataType::f32;
    attrs.epilogue.type = EpilogueActivateType::Relu;
    auto op = MatMul::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {a.desc(), b.desc(), bias.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));

    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b, bias};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], 1.125f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[1], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[2], 1.375f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[3], 0.0f, 1e-6f);
}

NNOPS_TEST(matmul_bias_fp_activation_dq) {
    // fp A (offline per-token params) × s8 B (per-channel): the dequantizing
    // route, biased. matmul_dq_ref carries the same bias-then-activation order,
    // so the two agree to the last bit; the tolerance only absorbs the
    // multiply order.
    const int64_t M = 6, K = 32, N = 13;
    auto [av, a_f32] = test::make_random_tensor({M, K}, -2.0f, 2.0f, 960);

    std::vector<float> a_s(static_cast<size_t>(M));
    std::vector<int32_t> a_zp(static_cast<size_t>(M));
    for (int64_t m = 0; m < M; ++m) {
        a_s[static_cast<size_t>(m)] = 0.02f + 0.005f * static_cast<float>(m);
        a_zp[static_cast<size_t>(m)] = static_cast<int32_t>(m) - 3;
    }

    std::vector<int8_t> bv(static_cast<size_t>(K * N));
    nnops::test::XorShift128 rng(961);
    for (auto& v : bv) {
        v = static_cast<int8_t>(static_cast<int32_t>(rng.next_u64() % 255) - 127);
    }
    std::vector<float> b_s(static_cast<size_t>(N));
    std::vector<int32_t> b_zp(static_cast<size_t>(N));
    for (int64_t n = 0; n < N; ++n) {
        b_s[static_cast<size_t>(n)] = 0.004f + 0.004f * static_cast<float>(n % 8) / 7.0f;
        b_zp[static_cast<size_t>(n)] = static_cast<int32_t>(n % 5) - 2;
    }

    QuantParams qa;
    qa.granularity = QuantGranularity::PerToken;
    qa.scale_data = a_s.data();
    qa.zero_point_data = a_zp.data();
    qa.num_scales = M;

    QuantParams qb;
    qb.granularity = QuantGranularity::PerChannel;
    qb.scale_data = b_s.data();
    qb.zero_point_data = b_zp.data();
    qb.num_scales = N;

    const int64_t ashape[] = {M, K};
    const int64_t bshape[] = {K, N};
    const int64_t bias_shape[] = {N};
    TensorView a = quant_view(ashape, 2, DataType::f32, av.data(), qa);
    TensorView b = quant_view(bshape, 2, DataType::s8, bv.data(), qb);
    auto [zv, bias] = test::make_random_tensor({N}, -0.5f, 0.5f, 962);

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {a.desc(), b.desc(), bias.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> got(static_cast<size_t>(descs[0].numel()));
    std::vector<float> want(got.size());
    auto og = nnops::test::make_planar(descs[0], got.data());
    auto ow = nnops::test::make_planar(descs[0], want.data());

    const TensorView ins[] = {a, b, bias};
    op->compute(og, ins);
    nnops::backend::cpu::reference::matmul_dq_ref(attrs, ow, ins, {}, nullptr);

    for (size_t i = 0; i < got.size(); ++i) {
        NNOPS_EXPECT_NEAR(got[i], want[i], 1e-6f * std::max(1.0f, std::abs(want[i])));
    }
}

NNOPS_TEST(matmul_bias_clip_bounds) {
    // The epilogue's clamp bounds belong to the same slot as the bias:
    // C = clamp(A×B + bias, -15, 15). Product [[4,5],[10,11]], +[10,-20]
    // gives [[14,-15],[20,-9]], which the bounds pull to [[14,-15],[15,-9]].
    // A biased call runs its kernels with a neutral epilogue, so this pins
    // that the post-pass still applies the caller's bounds.
    const int64_t ashape[] = {2, 3};
    const int64_t bshape[] = {3, 2};
    const int64_t bias_shape[] = {2};

    float a_data[6] = {1, 2, 3, 4, 5, 6};
    float b_data[6] = {1, 0, 0, 1, 1, 1};
    float bias_data[2] = {10, -20};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);
    TensorView bias(bias_shape, DataType::f32, bias_data);

    MatMulAttributes attrs{};
    attrs.epilogue.min_clip = -15.0f;
    attrs.epilogue.max_clip = 15.0f;
    auto op = MatMul::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {a.desc(), b.desc(), bias.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b, bias};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], 14.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[1], -15.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[2], 15.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[3], -9.0f, 1e-6f);
}

NNOPS_TEST(matmul_bias_relu6) {
    // Bias then Relu6 — both ends of the activation are hit:
    // [[4,5],[10,11]] + [10,-20] = [[14,-15],[20,-9]] → [[6,0],[6,0]].
    const int64_t ashape[] = {2, 3};
    const int64_t bshape[] = {3, 2};
    const int64_t bias_shape[] = {2};

    float a_data[6] = {1, 2, 3, 4, 5, 6};
    float b_data[6] = {1, 0, 0, 1, 1, 1};
    float bias_data[2] = {10, -20};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);
    TensorView bias(bias_shape, DataType::f32, bias_data);

    MatMulAttributes attrs{};
    attrs.epilogue.type = EpilogueActivateType::Relu6;
    auto op = MatMul::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {a.desc(), b.desc(), bias.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b, bias};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], 6.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[1], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[2], 6.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[3], 0.0f, 1e-6f);
}

NNOPS_TEST(matmul_bias_transpose_a) {
    // The same product as matmul_bias_basic_2d with A stored [K, M], which
    // takes the transposed-A pack. A^T = [[1,2,3],[4,5,6]].
    const int64_t ashape[] = {3, 2};
    const int64_t bshape[] = {3, 2};
    const int64_t bias_shape[] = {2};

    float a_data[6] = {1, 4, 2, 5, 3, 6};
    float b_data[6] = {1, 0, 0, 1, 1, 1};
    float bias_data[2] = {10, -20};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);
    TensorView bias(bias_shape, DataType::f32, bias_data);

    MatMulAttributes attrs{};
    attrs.transpose_a = true;
    auto op = MatMul::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {a.desc(), b.desc(), bias.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(2));
    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b, bias};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], 14.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[1], -15.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[2], 20.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[3], -9.0f, 1e-6f);
}

NNOPS_TEST(matmul_bias_int8_dequant_clip) {
    // The dequantizing int8 route with clip bounds and an activation: the order
    // is dequantize → bias → clamp → activate, the same expression
    // matmul_dq_ref evaluates.
    //   acc = [[1,2],[3,4]] · (0.5·0.25) = [[0.125,0.25],[0.375,0.5]]
    //   +[1,-1]  = [[1.125,-0.75],[1.375,-0.5]]
    //   clamp -1..1 → [[1,-0.75],[1,-0.5]] → relu → [[1,0],[1,0]]
    const int64_t ashape[] = {2, 2};
    const int64_t bshape[] = {2, 2};
    const int64_t bias_shape[] = {2};

    int8_t a_data[4] = {1, 2, 3, 4};
    int8_t b_data[4] = {1, 0, 0, 1};
    float bias_data[2] = {1.0f, -1.0f};

    TensorView a = quant_view(ashape, 2, DataType::s8, a_data,
                              bias_per_tensor(0.5f, 0));
    TensorView b = quant_view(bshape, 2, DataType::s8, b_data,
                              bias_per_tensor(0.25f, 0));
    TensorView bias(bias_shape, DataType::f32, bias_data);

    MatMulAttributes attrs{};
    attrs.output_dtype = DataType::f32;
    attrs.epilogue.type = EpilogueActivateType::Relu;
    attrs.epilogue.min_clip = -1.0f;
    attrs.epilogue.max_clip = 1.0f;
    auto op = MatMul::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {a.desc(), b.desc(), bias.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    std::vector<float> got(static_cast<size_t>(descs[0].numel()));
    auto og = nnops::test::make_planar(descs[0], got.data());

    // No reference comparison here: matmul_dq_ref only backs the fp-A route,
    // so for a quantized A the hand-computed values below are the spec.
    const TensorView ins[] = {a, b, bias};
    op->compute(og, ins);

    NNOPS_EXPECT_NEAR(got[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(got[1], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(got[2], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(got[3], 0.0f, 1e-6f);
}

// ============================================================
// alpha: C = epilogue(alpha·(A×B) + bias + beta·C_old)
// ============================================================
//
// alpha scales the product only — neither the bias nor beta·C_old. The
// reference kernel honours the same attributes, so it is the oracle for the
// routes; the hand-computed cases below pin the order independently.

NNOPS_TEST(matmul_alpha_fp_routes) {
    // One case per routing decision: the packed MMA tile (NN, NT, and a
    // multi-k-block TT), the small-M kernels, and the direct GEMV. alpha ∉
    // {0, 1} on every one.
    struct Case {
        int64_t M, N, K;
        bool ta, tb;
        float alpha, beta;
    };
    const Case cases[] = {
        {64, 48, 96, false, false, 2.0f, 0.0f},   // packed, NN
        {64, 48, 96, false, true,  -0.5f, 0.0f},  // packed, NT
        {13, 21, 300, true, true,  1.5f, 0.5f},   // packed, TT, multi-k-block
        {8, 37, 40, false, false,  3.0f, 0.0f},   // small-M / packed boundary
        {2, 33, 70, false, true,  -2.0f, 1.0f},   // small-M
        {1, 64, 40, false, false,  0.25f, 0.0f},  // GEMV, M=1
        {3, 64, 40, false, true,   4.0f, -1.0f},  // GEMV, transposed B
    };

    uint64_t seed = 7100;
    for (const Case& c : cases) {
        int64_t a_shape[2];
        int64_t b_shape[2];
        if (c.ta) { a_shape[0] = c.K; a_shape[1] = c.M; }
        else      { a_shape[0] = c.M; a_shape[1] = c.K; }
        if (c.tb) { b_shape[0] = c.N; b_shape[1] = c.K; }
        else      { b_shape[0] = c.K; b_shape[1] = c.N; }

        auto [a_vec, a] = test::make_random_tensor(
            std::span<const int64_t>(a_shape, 2), -2.0f, 2.0f, seed++);
        auto [b_vec, b] = test::make_random_tensor(
            std::span<const int64_t>(b_shape, 2), -2.0f, 2.0f, seed++);

        MatMulAttributes attrs{};
        attrs.transpose_a = c.ta;
        attrs.transpose_b = c.tb;
        attrs.alpha = c.alpha;
        attrs.beta = c.beta;
        auto op = MatMul::create(attrs, Backend::CPU);

        const TensorDesc arr[] = {a.desc(), b.desc()};
        auto descs = op->getOutputTensorDesc(arr);
        NNOPS_EXPECT_EQ(descs[0].dims[0], c.M);
        NNOPS_EXPECT_EQ(descs[0].dims[1], c.N);

        std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
        auto output = nnops::test::make_planar(descs[0], out_buf.data());
        // beta ∉ {0}: give the accumulate real prior data.
        for (size_t i = 0; i < out_buf.size(); ++i) {
            out_buf[i] = 0.25f * static_cast<float>(i % 7) - 0.5f;
        }

        const TensorView ins[] = {a, b};
        op->compute(output, ins);

        std::vector<float> ref_buf(out_buf.size());
        auto ref_out = nnops::test::make_planar(descs[0], ref_buf.data());
        for (size_t i = 0; i < ref_buf.size(); ++i) {
            ref_buf[i] = 0.25f * static_cast<float>(i % 7) - 0.5f;
        }
        nnops::backend::cpu::reference::matmul_ref(attrs, ref_out, ins, {}, nullptr);

        NNOPS_EXPECT_TRUE(test::allclose(output, ref_out, 1e-4f, 1e-5f));
    }
}

NNOPS_TEST(matmul_alpha_scales_product_only) {
    // A×B = [[4,5],[10,11]]. With alpha = 2: [[8,10],[20,22]].
    //   + bias [10,-20]  → [[18,-10],[30,2]]      (bias is not scaled)
    //   + beta·C_old with beta = 1, C_old = [[1,2],[3,4]]
    //                    → [[19,-8],[33,6]]       (C_old is not scaled either)
    const int64_t ashape[] = {2, 3};
    const int64_t bshape[] = {3, 2};
    const int64_t bias_shape[] = {2};

    float a_data[6] = {1, 2, 3, 4, 5, 6};
    float b_data[6] = {1, 0, 0, 1, 1, 1};
    float bias_data[2] = {10, -20};
    float c_init[4] = {1, 2, 3, 4};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);
    TensorView bias(bias_shape, DataType::f32, bias_data);

    MatMulAttributes attrs{};
    attrs.alpha = 2.0f;
    attrs.beta = 1.0f;
    auto op = MatMul::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {a.desc(), b.desc(), bias.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()), 0.0f);
    std::copy(std::begin(c_init), std::end(c_init), out_buf.begin());
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b, bias};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], 19.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[1], -8.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[2], 33.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[3], 6.0f, 1e-6f);
}

NNOPS_TEST(matmul_alpha_before_activation) {
    // The clamp/activation sees alpha·(A×B) + bias, not the raw product:
    // alpha=2, bias [10,-20] → [[18,-10],[30,2]] → clamp -15..15
    // → [[15,-10],[15,2]] → relu → [[15,0],[15,2]].
    const int64_t ashape[] = {2, 3};
    const int64_t bshape[] = {3, 2};
    const int64_t bias_shape[] = {2};

    float a_data[6] = {1, 2, 3, 4, 5, 6};
    float b_data[6] = {1, 0, 0, 1, 1, 1};
    float bias_data[2] = {10, -20};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);
    TensorView bias(bias_shape, DataType::f32, bias_data);

    MatMulAttributes attrs{};
    attrs.alpha = 2.0f;
    attrs.epilogue.type = EpilogueActivateType::Relu;
    attrs.epilogue.min_clip = -15.0f;
    attrs.epilogue.max_clip = 15.0f;
    auto op = MatMul::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {a.desc(), b.desc(), bias.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    std::vector<float> out_buf(static_cast<size_t>(descs[0].numel()));
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b, bias};
    op->compute(output, ins);

    NNOPS_EXPECT_NEAR(out_buf[0], 15.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[1], 0.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[2], 15.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(out_buf[3], 2.0f, 1e-6f);
}

NNOPS_TEST(matmul_alpha_f16) {
    // f16 operands, alpha folded into the f16 RHS pack. The oracle is the f32
    // reference on the same values, hence the arch-aware f16 tolerance.
    auto [a32, _] = test::make_random_tensor({12, 40}, -1.0f, 1.0f, 960);
    auto [b32, __] = test::make_random_tensor({40, 9}, -1.0f, 1.0f, 961);
    auto a_f16 = test::f32_to_f16(a32);
    auto b_f16 = test::f32_to_f16(b32);

    const int64_t a_shape[] = {12, 40};
    const int64_t b_shape[] = {40, 9};
    TensorView a(a_shape, DataType::f16, a_f16.data());
    TensorView b(b_shape, DataType::f16, b_f16.data());

    MatMulAttributes attrs{};
    attrs.alpha = 2.5f;
    auto op = MatMul::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

    std::vector<nnops::backend::cpu::half> out_buf(
        static_cast<size_t>(descs[0].numel()));
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    TensorView a32v(a_shape, DataType::f32, a32.data());
    TensorView b32v(b_shape, DataType::f32, b32.data());
    const TensorDesc ref_arr[] = {a32v.desc(), b32v.desc()};
    auto ref_descs = op->getOutputTensorDesc(ref_arr);
    std::vector<float> ref_buf(static_cast<size_t>(ref_descs[0].numel()));
    auto ref_out = nnops::test::make_planar(ref_descs[0], ref_buf.data());
    const TensorView ref_ins[] = {a32v, b32v};
    nnops::backend::cpu::reference::matmul_ref(attrs, ref_out, ref_ins, {}, nullptr);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_NEAR(simd::s_load(&out_buf[i]), ref_buf[i], kF16AccumTol);
    }
}

NNOPS_TEST(matmul_alpha_ignored_on_integer_route) {
    // alpha is a floating-point scale; the int8 accumulator has none of its
    // own, so the integer route ignores it (see MatMulAttributes::alpha).
    const int64_t ashape[] = {2, 2};
    const int64_t bshape[] = {2, 2};

    int8_t a_data[4] = {1, 2, 3, 4};
    int8_t b_data[4] = {1, 0, 0, 1};

    TensorView a = quant_view(ashape, 2, DataType::s8, a_data, bias_per_tensor(0.5f, 0));
    TensorView b = quant_view(bshape, 2, DataType::s8, b_data, bias_per_tensor(0.25f, 0));

    auto run = [&](float alpha, std::vector<float>& out) {
        MatMulAttributes attrs{};
        attrs.output_dtype = DataType::f32;
        attrs.alpha = alpha;
        auto op = MatMul::create(attrs, Backend::CPU);
        const TensorDesc arr[] = {a.desc(), b.desc()};
        auto descs = op->getOutputTensorDesc(arr);
        out.assign(static_cast<size_t>(descs[0].numel()), 0.0f);
        auto output = nnops::test::make_planar(descs[0], out.data());
        const TensorView ins[] = {a, b};
        op->compute(output, ins);
    };

    std::vector<float> plain, scaled;
    run(1.0f, plain);
    run(2.0f, scaled);
    for (size_t i = 0; i < plain.size(); ++i) {
        NNOPS_EXPECT_NEAR(scaled[i], plain[i], 1e-6f);
    }

    // Sanity: the unscaled result really is the product × (0.5·0.25) = 0.125.
    NNOPS_EXPECT_NEAR(plain[0], 0.125f, 1e-6f);
    NNOPS_EXPECT_NEAR(plain[3], 0.5f, 1e-6f);
}

// ============================================================
// Single-element dot product (M=1, N=1)
// ============================================================

NNOPS_TEST(matmul_1x1_dot) {
    // A: [1, 4], B: [4, 1]  →  C: [1, 1]
    const int64_t ashape[] = {1, 4};
    const int64_t bshape[] = {4, 1};

    float a_data[4] = {1, 2, 3, 4};
    float b_data[4] = {5, 6, 7, 8};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(1));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(1));
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    int64_t nelem = descs[0].numel();
    std::vector<float> out_buf(nelem);
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    // dot({1,2,3,4}, {5,6,7,8}) = 5+12+21+32 = 70
    NNOPS_EXPECT_NEAR(out_buf[0], 70.0f, 1e-6f);
}

// ============================================================
// Square matrices
// ============================================================

NNOPS_TEST(matmul_square) {
    // A: [3, 3], B: [3, 3]  →  C: [3, 3]
    const int64_t ashape[] = {3, 3};
    const int64_t bshape[] = {3, 3};

    // A = identity
    float a_data[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    // B = [[1, 2, 3],
    //      [4, 5, 6],
    //      [7, 8, 9]]
    float b_data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(3));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(3));
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    int64_t nelem = descs[0].numel();
    std::vector<float> out_buf(nelem);
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    float* c = out_buf.data();
    // I × B = B
    NNOPS_EXPECT_NEAR(c[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[1], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[2], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[3], 4.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[4], 5.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[5], 6.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[6], 7.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[7], 8.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(c[8], 9.0f, 1e-6f);
}

// ============================================================
// Random data — validate shape inference and no NaN/Inf
// ============================================================

NNOPS_TEST(matmul_random) {
    auto [a_vec, a] = test::make_random_tensor({8, 16});
    auto [b_vec, b] = test::make_random_tensor({16, 32});

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(8));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(32));
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    int64_t nelem = descs[0].numel();
    NNOPS_EXPECT_EQ(nelem, int64_t(8 * 32));
    std::vector<float> out_buf(nelem);
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

// ============================================================
// Random batch matmul — validate no NaN/Inf
// ============================================================

NNOPS_TEST(matmul_random_batch) {
    auto [a_vec, a] = test::make_random_tensor({4, 8, 16});
    auto [b_vec, b] = test::make_random_tensor({4, 16, 32});

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(3));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(4));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(8));
    NNOPS_EXPECT_EQ(descs[0].dims[2], int64_t(32));
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    int64_t nelem = descs[0].numel();
    NNOPS_EXPECT_EQ(nelem, int64_t(4 * 8 * 32));
    std::vector<float> out_buf(nelem);
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

// ============================================================
// Random with transpose_a — validate no NaN/Inf
// ============================================================

NNOPS_TEST(matmul_random_transpose_a) {
    auto [a_vec, a] = test::make_random_tensor({16, 8});   // phys [K, M]
    auto [b_vec, b] = test::make_random_tensor({16, 32});  // [K, N]

    MatMulAttributes attrs{};
    attrs.transpose_a = true;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(8));   // M
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(32));  // N
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    int64_t nelem = descs[0].numel();
    std::vector<float> out_buf(nelem);
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

// ============================================================
// Random with transpose_b — validate no NaN/Inf
// ============================================================

NNOPS_TEST(matmul_random_transpose_b) {
    auto [a_vec, a] = test::make_random_tensor({8, 16});   // [M, K]
    auto [b_vec, b] = test::make_random_tensor({32, 16});  // phys [N, K]

    MatMulAttributes attrs{};
    attrs.transpose_b = true;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs.size(), 1u);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(8));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(32));  // N = b.shape[0]
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);

    int64_t nelem = descs[0].numel();
    std::vector<float> out_buf(nelem);
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins);

    for (size_t i = 0; i < out_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(out_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(out_buf[i]));
    }
}

// ============================================================
// Fused path — SIMD pack + MMA kernels
// ============================================================
//
// These tests exercise the fused pack + MMA path (packed A lives on the kernel
// stack, packed B in kernel-pooled scratch), and results are compared against
// matmul_ref. Scratch is pooled internally by the operator, so compute() is
// called with a null workspace pointer.

NNOPS_TEST(matmul_packed_transpose_both_workspace) {
    auto [a_vec, a] = test::make_random_tensor({13, 21});   // phys [K, M] (transpose_a)
    auto [b_vec, b] = test::make_random_tensor({29, 13});   // phys [N, K] (transpose_b)

    MatMulAttributes attrs{};
    attrs.transpose_a = true;
    attrs.transpose_b = true;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    // M = A.shape[1] (21), N = B.shape[0] (29)
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(21));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(29));

    std::vector<float> out_buf(descs[0].numel());
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins, {}, nullptr);

    // Reference oracle on a separate buffer.
    std::vector<float> ref_buf(descs[0].numel());
    auto ref_out = nnops::test::make_planar(descs[0], ref_buf.data());
    nnops::backend::cpu::reference::matmul_ref(attrs, ref_out, ins, {}, nullptr);

    NNOPS_EXPECT_TRUE(test::allclose(output, ref_out, 1e-4f, 1e-5f));
}

NNOPS_TEST(matmul_packed_multiple_kblocks) {
    // K = 300 > the plan's f32 k-block (128 here, capped at 256): exercises
    // multi-k-block accumulation and the last-k-block epilogue/clamp logic in
    // the packed path.
    auto [a_vec, a] = test::make_random_tensor({6, 300});    // [M, K]
    auto [b_vec, b] = test::make_random_tensor({9, 300});    // phys [N, K]

    MatMulAttributes attrs{};
    attrs.transpose_b = true;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(6));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(9));

    std::vector<float> out_buf(descs[0].numel());
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins, {}, nullptr);

    std::vector<float> ref_buf(descs[0].numel());
    auto ref_out = nnops::test::make_planar(descs[0], ref_buf.data());
    nnops::backend::cpu::reference::matmul_ref(attrs, ref_out, ins, {}, nullptr);

    NNOPS_EXPECT_TRUE(test::allclose(output, ref_out, 1e-3f, 1e-4f));
}

NNOPS_TEST(matmul_packed_beta_relu_workspace) {
    // Beta + Relu epilogue in the packed path must match reference semantics:
    // C = relu(A×B + beta×C_old).
    auto [a_vec, a] = test::make_random_tensor({64, 64});  // [M, K]
    auto [b_vec, b] = test::make_random_tensor({32, 64});  // phys [N, K]

    MatMulAttributes attrs{};
    attrs.transpose_b = true;
    attrs.beta = 0.5f;
    attrs.epilogue.type = EpilogueActivateType::Relu;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);

    std::vector<float> out_buf(descs[0].numel());
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    // Seed C with old values so beta scaling is observable.
    std::vector<float> ref_buf(descs[0].numel());
    auto ref_out = nnops::test::make_planar(descs[0], ref_buf.data());
    for (size_t i = 0; i < out_buf.size(); ++i) {
        out_buf[i] = static_cast<float>(i) * 0.25f;
        ref_buf[i] = out_buf[i];
    }

    const TensorView ins[] = {a, b};
    op->compute(output, ins, {}, nullptr);
    nnops::backend::cpu::reference::matmul_ref(attrs, ref_out, ins, {}, nullptr);

    NNOPS_EXPECT_TRUE(test::allclose(output, ref_out, 1e-4f, 1e-5f));
}

NNOPS_TEST(matmul_packed_batched_workspace) {
    // Batched (rank 3) transpose_b through the packed path.
    auto [a_vec, a] = test::make_random_tensor({2, 16, 16});   // [B, M, K]
    auto [b_vec, b] = test::make_random_tensor({2, 16, 16});   // phys [B, N, K]

    MatMulAttributes attrs{};
    attrs.transpose_b = true;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].rank, int64_t(3));
    NNOPS_EXPECT_EQ(descs[0].dims[0], int64_t(2));
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(16));
    NNOPS_EXPECT_EQ(descs[0].dims[2], int64_t(16));

    std::vector<float> out_buf(descs[0].numel());
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins, {}, nullptr);

    std::vector<float> ref_buf(descs[0].numel());
    auto ref_out = nnops::test::make_planar(descs[0], ref_buf.data());
    nnops::backend::cpu::reference::matmul_ref(attrs, ref_out, ins, {}, nullptr);

    NNOPS_EXPECT_TRUE(test::allclose(output, ref_out, 1e-4f, 1e-5f));
}

NNOPS_TEST(matmul_packed_f16_workspace) {
    // f16 transpose_a through the fused path (A packed on the stack, B packed
    // in kernel-pooled scratch), compared against the f32 reference computed
    // from the f16 inputs (loose tolerance for f16).
    auto [a_f32, _] = test::make_random_tensor({7, 6}, -1.0f, 1.0f, 777);  // phys [K, M]
    auto [b_f32, __] = test::make_random_tensor({7, 5}, -1.0f, 1.0f, 888);  // [K, N]
    auto a_f16 = test::f32_to_f16(a_f32);
    auto b_f16 = test::f32_to_f16(b_f32);

    const int64_t a_shape[] = {7, 6};
    const int64_t b_shape[] = {7, 5};
    TensorView a(a_shape, DataType::f16, a_f16.data());
    TensorView b(b_shape, DataType::f16, b_f16.data());

    MatMulAttributes attrs{};
    attrs.transpose_a = true;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

    std::vector<nnops::backend::cpu::half> out_buf(descs[0].numel());
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins, {}, nullptr);

    // f32 reference on converted buffers.
    const int64_t a32_shape[] = {7, 6};
    const int64_t b32_shape[] = {7, 5};
    TensorView a32(a32_shape, DataType::f32, a_f32.data());
    TensorView b32(b32_shape, DataType::f32, b_f32.data());
    const TensorDesc ref_arr[] = {a32.desc(), b32.desc()};
    auto ref_descs = op->getOutputTensorDesc(ref_arr);
    std::vector<float> ref_buf(ref_descs[0].numel());
    auto ref_out = nnops::test::make_planar(ref_descs[0], ref_buf.data());
    const TensorView ref_ins[] = {a32, b32};
    nnops::backend::cpu::reference::matmul_ref(attrs, ref_out, ref_ins, {}, nullptr);

    for (int64_t i = 0; i < descs[0].numel(); ++i) {
        float v = simd::s_load(&out_buf[static_cast<size_t>(i)]);
        NNOPS_EXPECT_NEAR(v, ref_buf[static_cast<size_t>(i)], kF16AccumTol);
    }
}

NNOPS_TEST(matmul_f16_deep_k_accumulation) {
    namespace cpu = nnops::backend::cpu;

    // On aarch64 the f16 kernel accumulates in f16 (mma_pack_8x8_f16 keeps
    // float16x8_t accumulators), so kc bounds how many products round into the
    // accumulator before the C read-modify-write. The shared tile rule settles
    // this shape on kc = 128 (KC_CAP_F16), so pin the deep-K worst case against
    // the reference rather than assuming the accumulation still holds.
    const int64_t M = 64, K = 4096, N = 64;
    auto [a_f32, _]  = test::make_random_tensor({M, K}, -1.0f, 1.0f, 4242);
    auto [b_f32, __] = test::make_random_tensor({K, N}, -1.0f, 1.0f, 4243);
    auto a_f16 = test::f32_to_f16(a_f32);
    auto b_f16 = test::f32_to_f16(b_f32);

    const int64_t a_shape[] = {M, K};
    const int64_t b_shape[] = {K, N};
    TensorView a(a_shape, DataType::f16, a_f16.data());
    TensorView b(b_shape, DataType::f16, b_f16.data());

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);
    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);

    // Pin the kc this shape actually gets; if the rule moves it the test still
    // checks correctness, it just stops covering this accumulation depth.
    const auto plan = cpu::get_matmul_plan(attrs, a.desc(), b.desc(), 1);

    std::vector<cpu::half> out_buf(descs[0].numel());
    auto output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {a, b};
    op->compute(output, ins, {}, nullptr);

    const int64_t a32_shape[] = {M, K};
    const int64_t b32_shape[] = {K, N};
    TensorView a32(a32_shape, DataType::f32, a_f32.data());
    TensorView b32(b32_shape, DataType::f32, b_f32.data());
    const TensorDesc ref_arr[] = {a32.desc(), b32.desc()};
    auto ref_descs = op->getOutputTensorDesc(ref_arr);
    std::vector<float> ref_buf(ref_descs[0].numel());
    auto ref_out = test::make_planar(ref_descs[0], ref_buf.data());
    const TensorView ref_ins[] = {a32, b32};
    cpu::reference::matmul_ref(attrs, ref_out, ref_ins, {}, nullptr);

    float worst = 0.0f, maxref = 0.0f;
    for (int64_t i = 0; i < descs[0].numel(); ++i) {
        const float v = simd::s_load(&out_buf[static_cast<size_t>(i)]);
        const float r = ref_buf[static_cast<size_t>(i)];
        worst = std::max(worst, std::fabs(v - r));
        maxref = std::max(maxref, std::fabs(r));
    }
    NNOPS_EXPECT_EQ(plan.kc, int64_t{128});

    // The deviation scales with the output magnitude — max |ref| is ~80 here
    // from a 4096-long reduction — so the absolute kF16AccumTol the small-GEMM
    // f16 tests use does not apply. Bound it relative to that magnitude instead.
    // Measured: 0.23% at kc = 128 (0.29% at kc = 256, which the rule no longer
    // reaches here). The bound has ~2x headroom, so it catches a real
    // accumulation break, not drift.
    NNOPS_EXPECT_TRUE(worst <= 0.005f * maxref);
}

NNOPS_TEST(matmul_f16_transpose_b_packed) {
    // f16 transpose_b through the packed path — the route that transposes B into
    // the pooled panel scratch. The shape makes N hit a full 16-wide panel plus
    // a narrow tail (25 = 16 + 9; x86 zero-pads that tail, aarch64 decomposes it
    // into 8 + 1) and K exercise an 8-block plus a scalar tail (13 = 8 + 5).
    auto [a_f32, _]  = test::make_random_tensor({7, 13},  -1.0f, 1.0f, 901);  // [M, K]
    auto [b_f32, __] = test::make_random_tensor({25, 13}, -1.0f, 1.0f, 902);  // phys [N, K]
    auto a_f16 = test::f32_to_f16(a_f32);
    auto b_f16 = test::f32_to_f16(b_f32);

    const int64_t a_shape[] = {7, 13};
    const int64_t b_shape[] = {25, 13};
    TensorView a(a_shape, DataType::f16, a_f16.data());
    TensorView b(b_shape, DataType::f16, b_f16.data());

    MatMulAttributes attrs{};
    attrs.transpose_b = true;
    auto op = MatMul::create(attrs, Backend::CPU);

    const TensorDesc arr[] = {a.desc(), b.desc()};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);
    NNOPS_EXPECT_EQ(descs[0].dims[1], int64_t(25));

    std::vector<nnops::backend::cpu::half> out_buf(descs[0].numel());
    auto output = nnops::test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {a, b};
    op->compute(output, ins, {}, nullptr);

    // f32 reference on the same values.
    const int64_t a32_shape[] = {7, 13};
    const int64_t b32_shape[] = {25, 13};
    TensorView a32(a32_shape, DataType::f32, a_f32.data());
    TensorView b32(b32_shape, DataType::f32, b_f32.data());
    const TensorDesc ref_arr[] = {a32.desc(), b32.desc()};
    auto ref_descs = op->getOutputTensorDesc(ref_arr);
    std::vector<float> ref_buf(ref_descs[0].numel());
    auto ref_out = nnops::test::make_planar(ref_descs[0], ref_buf.data());
    const TensorView ref_ins[] = {a32, b32};
    nnops::backend::cpu::reference::matmul_ref(attrs, ref_out, ref_ins, {}, nullptr);

    for (int64_t i = 0; i < descs[0].numel(); ++i) {
        float v = simd::s_load(&out_buf[static_cast<size_t>(i)]);
        NNOPS_EXPECT_NEAR(v, ref_buf[static_cast<size_t>(i)], kF16AccumTol);
    }
}

// ============================================================
// Pack-decision + threading helpers
// ============================================================

// Build a 2D TensorView with an explicit row pitch (elements), filling only the
// logical [rows, cols] region with random data (padding lanes stay 0 so any
// accidental read of padding would corrupt the comparison).
static std::pair<std::vector<float>, TensorView>
make_padded_f32(int64_t rows, int64_t cols, int64_t pitch_elems, uint64_t seed) {
    std::vector<float> buf(static_cast<size_t>(rows * pitch_elems), 0.0f);
    nnops::test::XorShift128 rng(seed);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            buf[static_cast<size_t>(r * pitch_elems + c)] = rng.next_float(-1.0f, 1.0f);
        }
    }
    const int64_t shape[] = {rows, cols};
    TensorView tv(shape, DataType::f32, buf.data(),
                  pitch_elems * static_cast<int64_t>(sizeof(float)));
    return {std::move(buf), tv};
}

// Minimal fixed-size thread pool exposing a CpuBackend. Items in [begin, end)
// are claimed via an atomic counter; each worker runs body(i) until the range
// is exhausted. Exposes parallel_for + num_threads + thread_id for injection
// into ComputeContext.
struct SimplePool {
    explicit SimplePool(int nthreads) : nthreads_(nthreads) {
        // OpenMP when the build has it (common/cpu_pool.hpp): the adapter then
        // runs under the same bit-exact comparisons as the kernels.
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

// ============================================================
// Wide-stride pack decision
// ============================================================

NNOPS_TEST(matmul_padded_b_stride_pack_b_workspace) {
    // Non-transposed B with a wide row stride: pack_b triggers on stride
    // (M > mc re-reads B, ldb > 1024 page-scattered); the packed-B scratch is
    // pooled internally and sized for the whole N (multiple n-blocks tile it).
    const int64_t M = 256, K = 128, N = 2048;
    auto [a_vec, a] = test::make_random_tensor({M, K}, -1.0f, 1.0f, 101);
    auto [b_vec, b] = make_padded_f32(K, N, /*pitch_elems=*/4096, 202);

    NNOPS_EXPECT_EQ(b.desc().row_stride_elems, 4096);

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);

    std::vector<float> out_buf(descs[0].numel());
    auto output = nnops::test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {a, b};
    op->compute(output, ins, {}, nullptr);

    std::vector<float> ref_buf(descs[0].numel());
    auto ref_out = nnops::test::make_planar(descs[0], ref_buf.data());
    nnops::backend::cpu::reference::matmul_ref(attrs, ref_out, ins, {}, nullptr);

    NNOPS_EXPECT_TRUE(test::allclose(output, ref_out, 1e-3f, 1e-4f));
}

NNOPS_TEST(matmul_padded_b_stride_f16_workspace) {
    // f16 variant: stride threshold is 2048 (4096/sizeof(half)), so a 4096-elem
    // pitch on B triggers pack_b; the packed result must match the f32 reference.
    const int64_t M = 256, K = 256, N = 1024;
    const int64_t pitch_elems = 4096;

    auto [a_f32, _] = test::make_random_tensor({M, K}, -1.0f, 1.0f, 303);
    auto [b_f32, __] = make_padded_f32(K, N, pitch_elems, 404);
    auto a_f16 = test::f32_to_f16(a_f32);
    auto b_f16 = test::f32_to_f16(b_f32);

    const int64_t a_shape[] = {M, K};
    const int64_t b_shape[] = {K, N};
    TensorView a(a_shape, DataType::f16, a_f16.data());
    TensorView b(b_shape, DataType::f16, b_f16.data(),
                 pitch_elems * static_cast<int64_t>(sizeof(nnops::backend::cpu::half)));

    NNOPS_EXPECT_EQ(b.desc().row_stride_elems, pitch_elems);

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);
    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

    std::vector<nnops::backend::cpu::half> out_buf(descs[0].numel());
    auto output = nnops::test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {a, b};
    op->compute(output, ins, {}, nullptr);

    const int64_t a32_shape[] = {M, K};
    const int64_t b32_shape[] = {K, N};
    TensorView a32(a32_shape, DataType::f32, a_f32.data());
    TensorView b32(b32_shape, DataType::f32, b_f32.data(),
                   pitch_elems * static_cast<int64_t>(sizeof(float)));
    const TensorDesc ref_arr[] = {a32.desc(), b32.desc()};
    auto ref_descs = op->getOutputTensorDesc(ref_arr);
    std::vector<float> ref_buf(ref_descs[0].numel());
    auto ref_out = nnops::test::make_planar(ref_descs[0], ref_buf.data());
    const TensorView ref_ins[] = {a32, b32};
    nnops::backend::cpu::reference::matmul_ref(attrs, ref_out, ref_ins, {}, nullptr);

    for (int64_t i = 0; i < descs[0].numel(); ++i) {
        float v = simd::s_load(&out_buf[static_cast<size_t>(i)]);
        NNOPS_EXPECT_NEAR(v, ref_buf[static_cast<size_t>(i)], kF16AccumTol);
    }
}

NNOPS_TEST(matmul_padded_a_stride_correctness) {
    // Wide row stride on A (> 1024) triggers pack_a; the packed result must
    // match the reference computed from the padded (wide-stride) source.
    const int64_t M = 512, K = 512, N = 512;
    auto [a_vec, a] = make_padded_f32(M, K, /*pitch_elems=*/2048, 707);
    auto [b_vec, b] = test::make_random_tensor({K, N}, -1.0f, 1.0f, 808);

    NNOPS_EXPECT_EQ(a.desc().row_stride_elems, 2048);

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);

    std::vector<float> out_buf(descs[0].numel());
    auto output = nnops::test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {a, b};
    op->compute(output, ins, {}, nullptr);

    std::vector<float> ref_buf(descs[0].numel());
    auto ref_out = nnops::test::make_planar(descs[0], ref_buf.data());
    nnops::backend::cpu::reference::matmul_ref(attrs, ref_out, ins, {}, nullptr);

    NNOPS_EXPECT_TRUE(test::allclose(output, ref_out, 1e-3f, 1e-4f));
}

NNOPS_TEST(matmul_f32_nn_direct_tile_remainders) {
    // The complement of matmul_padded_a_stride_correctness: a *compact*
    // non-transposed A stays on the unpacked-A (pack_a=0) route, and that route
    // decomposes M by the shared MR_F32 table, {8,4,1} on aarch64, so Mc splits
    // as 8a + 4b + 1c. Walk M across that boundary and K across both the
    // 4-wide inner unroll and the plan's k-block (128 for these shapes), so the
    // tall tile, the 4/1 remainder kernels and the scalar k-tail are each
    // checked against the reference.
    // N walks the nr panel widths too — the kernel is selected per (mr, nr), so
    // 13 exercises the 12+1 columns and 17 the 12+4+1 ones.
    const int64_t Ms[] = {1, 5, 6, 7, 8, 13, 143, 144, 145};
    const int64_t Ks[] = {1, 3, 4, 7, 128, 129};
    const int64_t Ns[] = {1, 13, 17};

    for (int64_t M : Ms) {
        for (int64_t K : Ks) {
            for (int64_t N : Ns) {
                auto [a_vec, a] = test::make_random_tensor({M, K});
                auto [b_vec, b] = test::make_random_tensor({K, N});

                MatMulAttributes attrs{};
                auto op = MatMul::create(attrs, Backend::CPU);

                // Pin the route. A compact non-transposed A stays direct only
                // while K <= PACK_A_K_THRESHOLD; if that threshold is ever
                // retuned below K the test would silently start covering the
                // packed kernels instead of the direct ones.
                const auto plan = nnops::backend::cpu::get_matmul_plan(
                    attrs, a.desc(), b.desc(), 1);
                if (plan.pack_a) {
                    throw std::runtime_error(
                        "matmul_f32_nn_direct_tile_remainders: expected pack_a=0 at M="
                        + std::to_string(M) + " K=" + std::to_string(K));
                }

                const TensorDesc arr[] = {a.desc(), b.desc()};
                auto descs = op->getOutputTensorDesc(arr);

                std::vector<float> out_buf(descs[0].numel());
                auto output = nnops::test::make_planar(descs[0], out_buf.data());
                const TensorView ins[] = {a, b};
                op->compute(output, ins, {}, nullptr);

                std::vector<float> ref_buf(descs[0].numel());
                auto ref_out = nnops::test::make_planar(descs[0], ref_buf.data());
                nnops::backend::cpu::reference::matmul_ref(attrs, ref_out, ins, {}, nullptr);

                if (!test::allclose(output, ref_out, 1e-4f, 1e-5f)) {
                    throw std::runtime_error(
                        "matmul_f32_nn_direct_tile_remainders: mismatch at M="
                        + std::to_string(M) + " K=" + std::to_string(K)
                        + " N=" + std::to_string(N));
                }
            }
        }
    }
}

NNOPS_TEST(matmul_f16_nn_direct_tile_remainders) {
    // f16 complement of matmul_f32_nn_direct_tile_remainders: a compact
    // non-transposed A stays on the unpacked-A (pack_a=0) route. f16 has no
    // mr_f16_direct — the direct kernels read row-major A at the same height
    // they pack — so the route decomposes M with MR_F16 = {8,4,1}, i.e. 8a+4b+1c,
    // not the f32 direct {6,4,1}. Walk M across those boundaries, K across the
    // 4-wide inner unroll and the plan's k-block (128 for these shapes; the cap
    // is 256), and N across the NR_F16 = {16,8,1} panel widths, checking each
    // tile against the f32 reference.
    const int64_t Ms[] = {1, 4, 5, 8, 9, 12, 13, 143, 144, 145};
    const int64_t Ks[] = {1, 3, 4, 5, 128, 129};
    const int64_t Ns[] = {1, 17, 25};  // 16+1 and 16+8+1

    for (int64_t M : Ms) {
        for (int64_t K : Ks) {
            for (int64_t N : Ns) {
                auto [a_f32, _] = test::make_random_tensor({M, K}, -1.0f, 1.0f, 909);
                auto [b_f32, __] = test::make_random_tensor({K, N}, -1.0f, 1.0f, 1010);
                auto a_f16 = test::f32_to_f16(a_f32);
                auto b_f16 = test::f32_to_f16(b_f32);

                const int64_t a_shape[] = {M, K};
                const int64_t b_shape[] = {K, N};
                TensorView a(a_shape, DataType::f16, a_f16.data());
                TensorView b(b_shape, DataType::f16, b_f16.data());

                MatMulAttributes attrs{};
                auto op = MatMul::create(attrs, Backend::CPU);

                // Pin the route. This stays direct only while
                // K <= PACK_A_K_THRESHOLD; if that threshold is ever retuned
                // below K the test would silently start covering the packed
                // kernels instead.
                const auto plan = nnops::backend::cpu::get_matmul_plan(
                    attrs, a.desc(), b.desc(), 1);
                if (plan.pack_a) {
                    throw std::runtime_error(
                        "matmul_f16_nn_direct_tile_remainders: expected pack_a=0 at M="
                        + std::to_string(M) + " K=" + std::to_string(K));
                }

                const TensorDesc arr[] = {a.desc(), b.desc()};
                auto descs = op->getOutputTensorDesc(arr);
                NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f16);

                std::vector<nnops::backend::cpu::half> out_buf(descs[0].numel());
                auto output = nnops::test::make_planar(descs[0], out_buf.data());

                const TensorView ins[] = {a, b};
                op->compute(output, ins, {}, nullptr);

                // f32 reference computed from the f16 inputs (loose f16 tol).
                const int64_t a32_shape[] = {M, K};
                const int64_t b32_shape[] = {K, N};
                TensorView a32(a32_shape, DataType::f32, a_f32.data());
                TensorView b32(b32_shape, DataType::f32, b_f32.data());
                const TensorDesc ref_arr[] = {a32.desc(), b32.desc()};
                auto ref_descs = op->getOutputTensorDesc(ref_arr);
                std::vector<float> ref_buf(ref_descs[0].numel());
                auto ref_out = nnops::test::make_planar(ref_descs[0], ref_buf.data());
                const TensorView ref_ins[] = {a32, b32};
                nnops::backend::cpu::reference::matmul_ref(attrs, ref_out, ref_ins, {}, nullptr);

                for (int64_t i = 0; i < descs[0].numel(); ++i) {
                    const float v = simd::s_load(&out_buf[static_cast<size_t>(i)]);
                    const float r = ref_buf[static_cast<size_t>(i)];
                    if (std::fabs(v - r) > kF16AccumTol) {
                        throw std::runtime_error(
                            "matmul_f16_nn_direct_tile_remainders: mismatch at M="
                            + std::to_string(M) + " K=" + std::to_string(K)
                            + " N=" + std::to_string(N) + " idx=" + std::to_string(i)
                            + " got=" + std::to_string(v) + " want=" + std::to_string(r));
                    }
                }
            }
        }
    }
}

NNOPS_TEST(matmul_mkn_transpose_a_multik) {
    // transpose_a → pack_a (NKM order, B always packed). Multi-k and multi-m
    // blocks exercise beta@k==0, Relu clamp, and the last-k epilogue.
    const int64_t K = 300, M = 256, N = 128;
    auto [a_vec, a] = test::make_random_tensor({K, M}, -1.0f, 1.0f, 505);  // phys [K, M]
    auto [b_vec, b] = test::make_random_tensor({K, N}, -1.0f, 1.0f, 606);  // [K, N]

    MatMulAttributes attrs{};
    attrs.transpose_a = true;
    attrs.beta = 0.5f;
    attrs.epilogue.type = EpilogueActivateType::Relu;
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);

    std::vector<float> out_buf(descs[0].numel(), 1.0f);  // prefill for beta
    auto output = nnops::test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {a, b};
    op->compute(output, ins, {}, nullptr);

    std::vector<float> ref_buf(descs[0].numel(), 1.0f);
    auto ref_out = nnops::test::make_planar(descs[0], ref_buf.data());
    nnops::backend::cpu::reference::matmul_ref(attrs, ref_out, ins, {}, nullptr);

    NNOPS_EXPECT_TRUE(test::allclose(output, ref_out, 1e-3f, 1e-4f));
}

// ============================================================
// Multithreading: bit-identical to serial
// ============================================================

namespace {
struct MatmulInputs {
    std::vector<float> a_buf, b_buf;
    TensorView a, b;
};

MatmulInputs make_matmul_inputs(int64_t M, int64_t K, int64_t N,
                                bool ta, bool tb, bool pad_b, uint64_t seed)
{
    const int64_t a_rows = ta ? K : M, a_cols = ta ? M : K;
    const int64_t b_rows = tb ? N : K, b_cols = tb ? K : N;

    auto [a_buf, a] = test::make_random_tensor({a_rows, a_cols}, -1.0f, 1.0f, seed);
    std::vector<float> b_buf;
    TensorView b;
    if (pad_b) {
        const int64_t pitch = b_cols + 2048;  // wide stride (> 1024 → pack_b)
        b_buf.assign(static_cast<size_t>(b_rows * pitch), 0.0f);
        nnops::test::XorShift128 rng(seed + 1);
        for (int64_t r = 0; r < b_rows; ++r) {
            for (int64_t c = 0; c < b_cols; ++c) {
                b_buf[static_cast<size_t>(r * pitch + c)] = rng.next_float(-1.0f, 1.0f);
            }
        }
        const int64_t shape[] = {b_rows, b_cols};
        b = TensorView(shape, DataType::f32, b_buf.data(),
                       pitch * static_cast<int64_t>(sizeof(float)));
    } else {
        auto [buf, view] = test::make_random_tensor({b_rows, b_cols}, -1.0f, 1.0f, seed + 1);
        b_buf = std::move(buf);
        b = view;
    }
    return {std::move(a_buf), std::move(b_buf), a, b};
}
}  // anonymous namespace

NNOPS_TEST(matmul_threaded_matches_serial) {
    // Threading is bit-identical to serial: blocks write disjoint C tiles with
    // no reductions, so the thread count must not change any result.
    const int64_t M = 200, K = 160, N = 300;

    struct Case { const char* name; bool ta; bool tb; bool pad_b; };
    const Case cases[] = {
        {"nn", false, false, false},
        {"ta", true, false, false},
        {"tb", false, true, false},
        {"tatb", true, true, false},
        {"pad_b", false, false, true},
    };

    for (const Case& c : cases) {
        for (int nthreads : {2, 4}) {
            MatmulInputs in = make_matmul_inputs(M, K, N, c.ta, c.tb, c.pad_b, 1234);

            MatMulAttributes attrs{};
            attrs.transpose_a = c.ta;
            attrs.transpose_b = c.tb;
            auto op = MatMul::create(attrs, Backend::CPU);

            auto a_desc = in.a.desc();
            auto b_desc = in.b.desc();
            const TensorDesc arr[] = {a_desc, b_desc};
            auto descs = op->getOutputTensorDesc(arr);

            const TensorView ins[] = {in.a, in.b};

            // Serial (no parallelism hook).
            std::vector<float> serial_buf(descs[0].numel());
            auto serial_out = test::make_planar(descs[0], serial_buf.data());
            op->compute(serial_out, ins, {}, nullptr);

            // Threaded via SimplePool.
            SimplePool pool(nthreads);
            ComputeContext ctx;
            ctx.cpu = pool.cpu;
            std::vector<float> threaded_buf(descs[0].numel());
            auto threaded_out = test::make_planar(descs[0], threaded_buf.data());
            op->compute(threaded_out, ins, ctx, nullptr);

            // Bit-identical: disjoint C tiles, no reductions.
            for (int64_t i = 0; i < descs[0].numel(); ++i) {
                if (serial_buf[static_cast<size_t>(i)] != threaded_buf[static_cast<size_t>(i)]) {
                    throw std::runtime_error(
                        std::string("matmul_threaded_matches_serial: bit mismatch at ") +
                        std::to_string(i) + " (case " + c.name + ", " +
                        std::to_string(nthreads) + " threads)");
                }
            }

            // Both match the reference.
            std::vector<float> ref_buf(descs[0].numel());
            auto ref_out = test::make_planar(descs[0], ref_buf.data());
            nnops::backend::cpu::reference::matmul_ref(attrs, ref_out, ins, {}, nullptr);
            NNOPS_EXPECT_TRUE(test::allclose(serial_out, ref_out, 1e-3f, 1e-4f));
        }
    }
}

NNOPS_TEST(matmul_packed_a_region_holds_every_tile_height) {
    namespace cpu = nnops::backend::cpu;

    // The packed-A region must hold the panels of whatever mc the dispatch
    // actually picks — a per-block height <= plan.mc, not plan.mc itself. The
    // greedy {mr,..,1} decomposition the pack emits is NOT monotonic in mc, so
    // sizing the region at num_panels(plan.mc, mr) under-counts: {8,4,1} needs
    // 21 panels at mc = 143 (17×8 + 4 + 1 + 1 + 1) but only 18 at mc = 144, so
    // a region sized for a 144-row tile overruns by 3 panels when the shape
    // clips the last block to 143 rows.
    //
    // The tile used to be a stack array sized at the compile-time worst case;
    // it is now a slot in the plan's workspace, so this pins the plan's own
    // arithmetic. Heights straddle the {8,4,1} boundary so the non-monotonic
    // case is really reached, and transpose_a forces the packed-A route.
    for (int64_t M : {int64_t{143}, int64_t{144}, int64_t{145}, int64_t{572},
                      int64_t{1024}, int64_t{1152}}) {
        MatmulInputs in = make_matmul_inputs(M, 128, 64, /*ta=*/true, false, false, 7);
        MatMulAttributes attrs{};
        attrs.transpose_a = true;
        const auto a_desc = in.a.desc();
        const auto b_desc = in.b.desc();

        for (int nt : {1, 2, 4}) {
            const auto plan = cpu::get_matmul_plan(attrs, a_desc, b_desc, nt);
            NNOPS_EXPECT_TRUE(plan.pack_a);

            // The worst panel count over every height the block can hand the
            // pack, demanded at the widest stride it can use: the kernel packs
            // at actual_kc <= plan.kc, and ldd_a grows with kc, so plan.ldd_a
            // is the bound. This is the whole invariant — num_panels_max, not
            // num_panels(plan.mc, ..).
            int worst = 0;
            for (int mc = 1; mc <= static_cast<int>(plan.mc); ++mc) {
                const int c = cpu::num_panels(mc, cpu::MR_F32);
                worst = (c > worst) ? c : worst;
            }
            NNOPS_EXPECT_EQ(plan.np_a, int64_t{worst});

            // Geometry: sized from the plan's own kc, and every part of the
            // region 64-byte aligned so the slots after it stay aligned (the
            // pack/MMA kernels assume that alignment).
            const int64_t ldd_a = static_cast<int64_t>(
                cpu::mma_pack_align(cpu::MR_MAX_F32 * static_cast<int>(plan.kc) * 4) / 4);
            NNOPS_EXPECT_EQ(plan.ldd_a, ldd_a);
            NNOPS_EXPECT_EQ(plan.pack_a_offset % 64, int64_t{0});
            NNOPS_EXPECT_EQ((plan.np_a * plan.ldd_a * 4) % 64, int64_t{0});
            NNOPS_EXPECT_EQ(plan.workspace_size,
                            plan.pack_a_offset
                                + plan.num_slots_a * plan.np_a * plan.ldd_a * 4);
        }
    }
}

NNOPS_TEST(matmul_panel_counts_match_their_stepwise_definitions) {
    namespace cpu = nnops::backend::cpu;

    // num_panels and num_panels_max are now closed
    // forms of the loops they used to run: "take nr[s] panels while they fit"
    // leaves `rem % nr[s]` behind, so that loop is the quotient chain; and the
    // largest count over every width in [1, n] sits at one of two points (a full
    // q = n/nr[0], or a short q-1 with a worst-case remainder) rather than
    // anywhere in the range.
    //
    // The rewrite is what takes the per-plan path from O(n*levels) to
    // O(levels) — the scan ran at n = the tile width (up to NC_MAX) on every
    // get_matmul_plan and every attention tile sizing, 2.6 us at n = 1024 for
    // {12,4,1}. It is worth exactly nothing if the recursion is wrong, so pin
    // both identities against the loops themselves, for every NR/MR list the
    // kernels use, past every width the tiling heuristic can ask for.
    auto stepwise = [](int n, const int* nr, int levels) {
        int count = 0;
        int i = 0;
        for (int s = 0; s < levels; ++s) {
            for (; i + nr[s] <= n; i += nr[s]) { ++count; }
        }
        return count;
    };

    struct List { const int* nr; int levels; };
    for (const List& l : {List{cpu::MR_F32, 3}, List{cpu::MR_F16, 3}, List{cpu::NR_F32, 3},
                          List{cpu::NR_F16, 3}, List{cpu::MR_I8, 3}}) {
        for (int n = 1; n <= 8192; ++n) {
            NNOPS_EXPECT_EQ(cpu::num_panels(n, l.nr, l.levels), stepwise(n, l.nr, l.levels));
            int worst = 0;
            for (int mc = 1; mc <= n; ++mc) {
                const int c = stepwise(mc, l.nr, l.levels);
                worst = c > worst ? c : worst;
            }
            NNOPS_EXPECT_EQ(cpu::num_panels_max(n, l.nr, l.levels), worst);
        }
    }
}

NNOPS_TEST(matmul_tile_rule_properties) {
    namespace cpu = nnops::backend::cpu;

    // One rule now picks mc/nc/kc for every dtype: maximise the tile's reuse
    // subject to (a) the aggregate L2 the concurrent blocks occupy, (b) an L1
    // prefetch-residency bound, (c) the MC_MAX / NC_MAX ceiling, and (d) the
    // panel grids — mc on MC_ALIGN, nc on NC_ALIGN, kc on KC_ALIGN. Pin all
    // four so a change to the rule has to say so here instead of drifting.
    //
    // This is the guard on the rule itself; matmul_threaded_matches_serial
    // remains the guard on its output.
    struct Dtype {
        DataType dt;
        int mr_max, nr_max;
        int64_t esz, kc_cap;
    };
    const Dtype dtypes[] = {
        {DataType::f32, cpu::MR_MAX_F32, cpu::NR_MAX_F32, 4, cpu::KC_CAP_F32},
        {DataType::f16, cpu::MR_MAX_F16, cpu::NR_MAX_F16, 2, cpu::KC_CAP_F16},
        {DataType::s8,  cpu::MR_MAX_I8,  cpu::NR_MAX_I8,  1, cpu::KC_CAP_I8},
    };

    const auto& f = nnops::simd::CpuFeatures::get();
    const int64_t l1 = static_cast<int64_t>(f.l1_cache_size());
    const int64_t l2 = static_cast<int64_t>(f.l2_shared_cache_size());

    const int64_t Ms[]  = {1, 7, 24, 100, 143, 256, 512, 1024, 4096};
    const int64_t Ns[]  = {1, 8, 48, 100, 256, 1024, 2048};
    const int64_t Ks[]  = {1, 32, 128, 129, 256, 512, 3000};
    const int64_t nts[] = {1, 2, 4, 8, 16};

    for (const Dtype& d : dtypes) {
        for (int64_t M : Ms) {
            for (int64_t N : Ns) {
                for (int64_t K : Ks) {
                    TensorDesc ad, bd;
                    ad.rank = bd.rank = 2;
                    ad.dtype = bd.dtype = d.dt;
                    ad.dims.push_back(M); ad.dims.push_back(K);
                    bd.dims.push_back(K); bd.dims.push_back(N);
                    MatMulAttributes attrs{};
                    for (int64_t nt : nts) {
                        const auto p = cpu::get_matmul_plan(attrs, ad, bd, nt);

                        // (c) ceiling, and (d) grid: below the align width the
                        // tile sits on the mr/nr grid instead, so only the
                        // large-tile case pins the alignment exactly.
                        NNOPS_EXPECT_TRUE(p.mc <= cpu::MC_MAX);
                        NNOPS_EXPECT_TRUE(p.mc % cpu::MC_ALIGN == 0 ||
                                          p.mc < cpu::MC_ALIGN);
                        NNOPS_EXPECT_TRUE(p.nc <= cpu::NC_MAX);
                        NNOPS_EXPECT_TRUE(p.nc % cpu::NC_ALIGN == 0 ||
                                          p.nc < cpu::NC_ALIGN);
                        NNOPS_EXPECT_TRUE(p.kc % cpu::KC_ALIGN == 0 || p.kc == K);

                        // kc never exceeds its dtype cap nor the reduction length.
                        NNOPS_EXPECT_TRUE(p.kc <= d.kc_cap);
                        NNOPS_EXPECT_TRUE(p.kc <= K);

                        // (b) prefetch residency: mr_max rows of A plus the
                        // current and next nr_max columns of B fit in L1.
                        NNOPS_EXPECT_TRUE((d.mr_max + 2 * static_cast<int64_t>(d.nr_max)) *
                                              p.kc * d.esz <= l1);

                        // (a) aggregate L2 the concurrent blocks occupy. The
                        // KC_MIN floor is the one escape — it can raise kc back
                        // over the budget for a very large span at a high thread
                        // count — so a floor hit is exempt rather than failing.
                        const bool floored =
                            p.kc == std::min<int64_t>(cpu::KC_MIN, K);
                        NNOPS_EXPECT_TRUE(floored ||
                                          nt * p.kc * (p.mc + p.nc) * d.esz <= l2 / 2);
                    }
                }
            }
        }
    }
}

NNOPS_TEST(matmul_n_split_slice_holds_every_tile_width) {
    namespace cpu = nnops::backend::cpu;

    // The N-side sibling of matmul_packed_a_stack_holds_every_tile_height. The
    // N-split packs each n-block's B into its own slice, indexed by the
    // dispatch as `blk * np_slice * ldd_b`. Only the blocks before the last are
    // exactly nc wide — the last packs N mod nc columns. The uniform route
    // emits ceil(n/nr_max) panels for n columns, so the widest tile (nc) is
    // also the worst case, and np_slice must cover it. A tile below nc emits
    // fewer panels, never more.
    //
    // These shapes resolve kc == K (256), so the kernel's per-k-block panel
    // stride equals plan.ldd_b and the demand checked below is exact, not merely
    // bounded. If the rule ever drops kc below K here the check still holds, it
    // just stops being tight — the np_slice guard above is what pins the sizing.
    for (int64_t N : {int64_t{2000}, int64_t{2001}, int64_t{2047}, int64_t{2048}}) {
        MatmulInputs in = make_matmul_inputs(64, 256, N, false, false, false, 3);
        MatMulAttributes attrs{};
        const auto plan = cpu::get_matmul_plan(attrs, in.a.desc(), in.b.desc(), 4);

        NNOPS_EXPECT_TRUE(plan.split_n);
        // The A row stride is K = 256, so this is the direct-A route and the
        // packed-A region is empty — the offsets checked below are then pure
        // packed-B, which is what makes them exact rather than merely bounded.
        NNOPS_EXPECT_TRUE(!plan.pack_a);
        NNOPS_EXPECT_EQ(plan.num_slots, (N + plan.nc - 1) / plan.nc);

        // Panels the uniform pack emits for a tile of n columns.
        const int nr0 = cpu::NR_F32[0];
        const auto panels_for = [nr0](int64_t n) {
            return (n + nr0 - 1) / nr0;
        };
        NNOPS_EXPECT_EQ(plan.np_slice, panels_for(plan.nc));

        // Every block, at the offset the dispatch gives it, stays inside the
        // workspace the plan hands over.
        const int64_t nb = (N + plan.nc - 1) / plan.nc;
        const int64_t capacity = plan.workspace_size / static_cast<int64_t>(sizeof(float));
        for (int64_t blk = 0; blk < nb; ++blk) {
            const int64_t left = N - blk * plan.nc;
            const int64_t n_count = (left < plan.nc) ? left : plan.nc;
            const int64_t demand = panels_for(n_count) * plan.ldd_b;
            NNOPS_EXPECT_TRUE(blk * plan.np_slice * plan.ldd_b + demand <= capacity);
        }
    }
}

NNOPS_TEST(matmul_packed_a_bad_remainder_threaded) {
    // End-to-end cover for the {8,4,1} non-monotonicity the region sizing is
    // there for: a short final block packs more panels than the tile's own
    // height does (143 rows -> 21 panels, 144 -> 18), so a region sized at
    // num_panels(plan.mc) is the one that overruns. mc is a multiple of
    // MC_ALIGN = 24, and the worst count always sits one row under it (24k-1
    // needs 3k+3 panels, 24k only 3k), so num_panels_max exceeds
    // num_panels(plan.mc) for every tile the rule produces — which is why
    // sizing must use the former.
    //
    // Both shapes reach the pack through the packed-A kernel (transpose_a) with
    // K == plan.kc, so the pack uses the full-Kc stride the region is sized at.
    // The rule sizes the M-split so the block count is a multiple of the worker
    // count, which makes the last block exactly mc-1 rows here — the worst case
    // is not hypothetical, it is what these two shapes actually pack:
    //   - M = 575 with 4 workers: mc = 144 (575/4 rounded up to the 24-grid),
    //     blocks 144,144,144,143.
    //   - M = 287 with 2 workers: mc = 144, blocks 144,143.
    struct Case { int64_t M; int nthreads; int64_t want_mc; };
    const Case cases[] = {
        {575, 4, 144},
        {287, 2, 144},
    };

    for (const Case& c : cases) {
        const int64_t K = 128, N = 64;
        MatmulInputs in = make_matmul_inputs(c.M, K, N, /*ta=*/true, /*tb=*/false,
                                             /*pad_b=*/false, 7);

        MatMulAttributes attrs{};
        attrs.transpose_a = true;
        auto op = MatMul::create(attrs, Backend::CPU);

        auto a_desc = in.a.desc();
        auto b_desc = in.b.desc();
        const TensorDesc arr[] = {a_desc, b_desc};
        auto descs = op->getOutputTensorDesc(arr);

        // Pin the premise: if the tile-shrink or clamp rule changes, these stop
        // covering the bad remainder and the test should say so.
        const auto plan = nnops::backend::cpu::get_matmul_plan(attrs, a_desc, b_desc,
                                                               c.nthreads);
        NNOPS_EXPECT_EQ(plan.mc, c.want_mc);
        NNOPS_EXPECT_TRUE(plan.pack_a);
        NNOPS_EXPECT_EQ(plan.np_a,
                        int64_t{nnops::backend::cpu::num_panels_max(static_cast<int>(plan.mc), nnops::backend::cpu::MR_F32)});
        // And np_a really does exceed num_panels(plan.mc) — the count a naive
        // region sizing would use, and the one that would put the short block's
        // extra panels past the end of the region.
        NNOPS_EXPECT_TRUE(plan.np_a >
                          nnops::backend::cpu::num_panels(static_cast<int>(plan.mc),
                                                          nnops::backend::cpu::MR_F32));

        const TensorView ins[] = {in.a, in.b};
        std::vector<float> got(descs[0].numel()), want(descs[0].numel());
        auto out = test::make_planar(descs[0], got.data());
        auto ref_out = test::make_planar(descs[0], want.data());

        SimplePool pool(c.nthreads);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        op->compute(out, ins, ctx, nullptr);
        nnops::backend::cpu::reference::matmul_ref(attrs, ref_out, ins, {}, nullptr);

        NNOPS_EXPECT_TRUE(test::allclose(out, ref_out, 1e-3f, 1e-4f));
    }
}

NNOPS_TEST(matmul_threaded_multi_block_slots) {
    // M-split with far more m-blocks than workers. The packed-B scratch is one
    // slot per worker thread, so a thread reuses its slot across every block it
    // claims rather than owning one slice per block. The case above uses
    // M=200/N=300, where the split collapses to a single block, so this is the
    // one that actually exercises slot reuse. The result must stay bit-identical
    // to serial: blocks write disjoint C tiles and no reduction is reordered.
    //
    // The rule now sizes the M-split so the block count is a multiple of the
    // worker count (balance_tile), so blocks == nthreads whenever M is under
    // MC_MAX * nthreads = 768 * nthreads — and with one block per thread there
    // is nothing to reuse a slot across. Slot mode therefore needs a big M now,
    // not the M=1024 the old rule shrank to 8 blocks: every row below is in the
    // thousands. The "ta" rows are the same shape as the "nn" ones with A
    // stored transposed, so they reach the pack by the other route rather than
    // by a different block count.
    struct Case { const char* name; int64_t M, K, N; bool ta; int nthreads; };
    const Case cases[] = {
        {"nn",      4096, 128, 512, false, 2},
        {"nn",      4096, 128, 512, false, 4},
        {"ta",      4096, 128, 512, true,  2},
        {"ta",      4096, 128, 512, true,  4},
        {"nn-big",  8192, 512, 256, false, 2},
        {"nn-big",  8192, 512, 256, false, 4},
        {"nn-big",  8192, 512, 256, false, 8},
    };

    for (const Case& c : cases) {
        MatmulInputs in = make_matmul_inputs(c.M, c.K, c.N, c.ta, /*tb=*/false,
                                             /*pad_b=*/false, 77);

        MatMulAttributes attrs{};
        attrs.transpose_a = c.ta;
        auto op = MatMul::create(attrs, Backend::CPU);

        const TensorDesc arr[] = {in.a.desc(), in.b.desc()};
        auto descs = op->getOutputTensorDesc(arr);
        const TensorView ins[] = {in.a, in.b};

        // Guard against the test going vacuous: the plan must really be in
        // slot mode (fewer slots than blocks) for this shape and pool.
        const auto plan = nnops::backend::cpu::get_matmul_plan(
            attrs, in.a.desc(), in.b.desc(), c.nthreads, /*use_thread_slots=*/true);
        const int64_t num_blocks =
            nnops::backend::cpu::split_block_count(c.M, plan.mc);
        if (!(plan.num_slots < num_blocks)) {
            throw std::runtime_error(
                std::string("matmul_threaded_multi_block_slots: expected slot mode (") +
                c.name + ", " + std::to_string(c.nthreads) + "t), got num_slots=" +
                std::to_string(plan.num_slots) + " blocks=" + std::to_string(num_blocks));
        }

        std::vector<float> serial_buf(descs[0].numel());
        auto serial_out = test::make_planar(descs[0], serial_buf.data());
        op->compute(serial_out, ins, {}, nullptr);

        SimplePool pool(c.nthreads);
        ComputeContext ctx;
        ctx.cpu = pool.cpu;
        std::vector<float> threaded_buf(descs[0].numel());
        auto threaded_out = test::make_planar(descs[0], threaded_buf.data());
        op->compute(threaded_out, ins, ctx, nullptr);

        for (int64_t i = 0; i < descs[0].numel(); ++i) {
            if (serial_buf[static_cast<size_t>(i)] != threaded_buf[static_cast<size_t>(i)]) {
                throw std::runtime_error(
                    std::string("matmul_threaded_multi_block_slots: bit mismatch at ") +
                    std::to_string(i) + " (case " + c.name + ", " +
                    std::to_string(c.nthreads) + " threads)");
            }
        }
    }
}

NNOPS_TEST(matmul_threaded_batched) {
    // Batched matmul with the threading pool: each batch element parallelizes
    // independently; the result must still match the reference.
    const int64_t B = 3, M = 96, K = 128, N = 160;
    auto [a_vec, a] = test::make_random_tensor({B, M, K}, -1.0f, 1.0f, 909);
    auto [b_vec, b] = test::make_random_tensor({B, K, N}, -1.0f, 1.0f, 1010);

    MatMulAttributes attrs{};
    auto op = MatMul::create(attrs, Backend::CPU);

    auto a_desc = a.desc();
    auto b_desc = b.desc();
    const TensorDesc arr[] = {a_desc, b_desc};
    auto descs = op->getOutputTensorDesc(arr);

    SimplePool pool(4);
    ComputeContext ctx;
    ctx.cpu = pool.cpu;

    std::vector<float> out_buf(descs[0].numel());
    auto output = test::make_planar(descs[0], out_buf.data());
    const TensorView ins[] = {a, b};
    op->compute(output, ins, ctx, nullptr);

    std::vector<float> ref_buf(descs[0].numel());
    auto ref_out = test::make_planar(descs[0], ref_buf.data());
    nnops::backend::cpu::reference::matmul_ref(attrs, ref_out, ins, {}, nullptr);

    NNOPS_EXPECT_TRUE(test::allclose(output, ref_out, 1e-3f, 1e-4f));
}

// ============================================================
// Small-GEMM kernel (arch::matmul_naive), called directly
// ============================================================

NNOPS_TEST(matmul_naive_kernel_layouts) {
    // The small-GEMM kernel bypasses the operator, so its own coverage has to
    // name the layouts: all four transpose combinations, N on both sides of the
    // vector width (wide sweep vs per-output dots), k tails, padded pitches on A,
    // B and C, and beta folded into the output.
    namespace cpu = nnops::backend::cpu;
    using cpu::half;

    const int64_t shapes[][3] = {{1, 1, 1}, {2, 3, 4}, {5, 4, 8},
                                 {9, 8, 13}, {3, 16, 5}, {4, 21, 9}};

    const auto run = [&](auto dtype_tag) {
        using T = decltype(dtype_tag);
        constexpr float tol = std::is_same_v<T, float> ? 1e-5f : kF16AccumTol;

        const auto cast = [](float v) -> T {
            if constexpr (std::is_same_v<T, float>) { return v; }
            else { return cpu::float_to_half(v); }
        };
        const auto to_f = [](T v) -> float {
            if constexpr (std::is_same_v<T, float>) { return v; }
            else { return cpu::half_to_float(v); }
        };

        for (const auto& s : shapes) {
            const int64_t M = s[0], N = s[1], K = s[2];
            for (int ta = 0; ta < 2; ++ta) {
                for (int tb = 0; tb < 2; ++tb) {
                    for (int64_t pad : {int64_t{0}, int64_t{3}}) {
                        const int64_t a_rows = ta ? K : M, a_cols = ta ? M : K;
                        const int64_t b_rows = tb ? N : K, b_cols = tb ? K : N;
                        const int64_t apitch = a_cols + pad;
                        const int64_t bpitch = b_cols + pad;
                        const int64_t ldc = N + 1;
                        const float beta = pad ? 0.5f : 0.0f;

                        nnops::test::XorShift128 rng(17 + static_cast<uint64_t>(pad));
                        std::vector<T> abuf(static_cast<size_t>(a_rows * apitch));
                        std::vector<T> bbuf(static_cast<size_t>(b_rows * bpitch));
                        std::vector<T> cbuf(static_cast<size_t>(M * ldc));
                        for (int64_t r = 0; r < a_rows; ++r) {
                            for (int64_t c = 0; c < a_cols; ++c) {
                                abuf[static_cast<size_t>(r * apitch + c)] = cast(rng.next_float());
                            }
                        }
                        for (int64_t r = 0; r < b_rows; ++r) {
                            for (int64_t c = 0; c < b_cols; ++c) {
                                bbuf[static_cast<size_t>(r * bpitch + c)] = cast(rng.next_float());
                            }
                        }
                        for (auto& v : cbuf) { v = cast(1.0f); }

                        cpu::arch::matmul_naive(static_cast<int>(M), static_cast<int>(N),
                                                static_cast<int>(K),
                                                abuf.data(), static_cast<int>(apitch), ta != 0,
                                                bbuf.data(), static_cast<int>(bpitch), tb != 0,
                                                cbuf.data(), static_cast<int>(ldc), 1.0f, beta);

                        for (int64_t m = 0; m < M; ++m) {
                            for (int64_t n = 0; n < N; ++n) {
                                float want = 0.0f;
                                for (int64_t k = 0; k < K; ++k) {
                                    const float av = to_f(abuf[static_cast<size_t>(
                                        ta ? k * apitch + m : m * apitch + k)]);
                                    const float bv = to_f(bbuf[static_cast<size_t>(
                                        tb ? n * bpitch + k : k * bpitch + n)]);
                                    want += av * bv;
                                }
                                if (beta != 0.0f) { want += beta * 1.0f; }
                                NNOPS_EXPECT_NEAR(to_f(cbuf[static_cast<size_t>(m * ldc + n)]),
                                                  want, tol);
                            }
                        }
                    }
                }
            }
        }
    };

    run(0.0f);
    run(half{});
}

// ============================================================
// GEMV route (M <= GEMV_M_MAX): direct kernels, no packing
// ============================================================

NNOPS_TEST(matmul_gemv_route_layouts) {
    // Small M leaves the packed route entirely: the direct kernel streams B
    // once. Every shape below is far above GEMM_FAST_PATH_THRESHOLD, so each
    // really takes that route — except ta+tb, which the kernel declines and the
    // packed path must still answer. Covered: both dtypes, every transpose
    // pair, k and n tails, N on both sides of the 1024-wide GEMV block, beta, a
    // Relu epilogue, a broadcast batch, and B either side of the size cutoff
    // that switches the nn kernel from the register tile to the stream.
    namespace cpu = nnops::backend::cpu;
    using cpu::half;

    struct Case {
        int64_t M, N, K;
        bool ta, tb, batch, beta, relu;
    };
    const Case cases[] = {
        {1, 3000, 64,   false, false, false, false, false},
        {2, 129,  512,  false, true,  false, true,  true},
        {3, 1024, 300,  true,  false, false, false, false},
        {4, 33,   4096, false, false, false, true,  false},
        {1, 700,  900,  false, false, true,  false, true},
        {2, 100,  700,  true,  true,  false, false, false},
        // B over the 24 MB cutoff at f32 (25 MB) and at f16 (25 MB), which is
        // the streaming kernel at M=2 and M=1 respectively; 3073 leaves a k
        // row below the unroll depth.
        {2, 2048, 3073, false, false, false, true,  false},
        {1, 4096, 3073, false, false, false, false, false},
    };

    const auto run = [&](auto dtype_tag) {
        using T = decltype(dtype_tag);
        constexpr float tol = std::is_same_v<T, float> ? 1e-4f : kF16AccumTol;
        constexpr DataType dt = std::is_same_v<T, float> ? DataType::f32
                                                         : DataType::f16;

        const auto cast = [](float v) -> T {
            if constexpr (std::is_same_v<T, float>) { return v; }
            else { return cpu::float_to_half(v); }
        };
        const auto to_f = [](T v) -> float {
            if constexpr (std::is_same_v<T, float>) { return v; }
            else { return cpu::half_to_float(v); }
        };

        for (const Case& c : cases) {
            const int64_t a_rows = c.ta ? c.K : c.M;
            const int64_t a_cols = c.ta ? c.M : c.K;
            const int64_t b_rows = c.tb ? c.N : c.K;
            const int64_t b_cols = c.tb ? c.K : c.N;
            const int64_t batch = c.batch ? 2 : 1;

            // The reference is f32-only, so it reads f32 mirrors of the same
            // values (the f16 inputs are already rounded to what the op saw).
            nnops::test::XorShift128 rng(31);
            std::vector<float> af32(static_cast<size_t>(batch * a_rows * a_cols));
            std::vector<float> bf32(static_cast<size_t>(batch * b_rows * b_cols));
            for (auto& v : af32) { v = rng.next_float(-1.0f, 1.0f); }
            for (auto& v : bf32) { v = rng.next_float(-1.0f, 1.0f); }
            std::vector<T> abuf(af32.size());
            std::vector<T> bbuf(bf32.size());
            for (size_t i = 0; i < af32.size(); ++i) { abuf[i] = cast(af32[i]); }
            for (size_t i = 0; i < bf32.size(); ++i) { bbuf[i] = cast(bf32[i]); }

            TensorView a, b, a32, b32;
            if (c.batch) {
                const int64_t as[] = {batch, a_rows, a_cols};
                const int64_t bs[] = {batch, b_rows, b_cols};
                a   = TensorView(as, dt, abuf.data());
                b   = TensorView(bs, dt, bbuf.data());
                a32 = TensorView(as, DataType::f32, af32.data());
                b32 = TensorView(bs, DataType::f32, bf32.data());
            } else {
                const int64_t as[] = {a_rows, a_cols};
                const int64_t bs[] = {b_rows, b_cols};
                a   = TensorView(as, dt, abuf.data());
                b   = TensorView(bs, dt, bbuf.data());
                a32 = TensorView(as, DataType::f32, af32.data());
                b32 = TensorView(bs, DataType::f32, bf32.data());
            }

            MatMulAttributes attrs{};
            attrs.transpose_a = c.ta;
            attrs.transpose_b = c.tb;
            if (c.beta) { attrs.beta = 0.5f; }
            if (c.relu) { attrs.epilogue.type = EpilogueActivateType::Relu; }
            auto op = MatMul::create(attrs, Backend::CPU);

            const TensorDesc arr[] = {a.desc(), b.desc()};
            auto descs = op->getOutputTensorDesc(arr);
            std::vector<T> got_buf(static_cast<size_t>(descs[0].numel()));
            std::vector<float> want_buf(static_cast<size_t>(descs[0].numel()));
            // Seed C so a beta would be observable.
            for (size_t i = 0; i < got_buf.size(); ++i) {
                got_buf[i] = cast(static_cast<float>(i % 17) * 0.25f);
                want_buf[i] = to_f(got_buf[i]);
            }
            auto got = nnops::test::make_planar(descs[0], got_buf.data());

            const TensorView ins[] = {a, b};
            op->compute(got, ins, {}, nullptr);

            const TensorView ref_ins[] = {a32, b32};
            const TensorDesc ref_arr[] = {a32.desc(), b32.desc()};
            auto ref_descs = op->getOutputTensorDesc(ref_arr);
            auto want = nnops::test::make_planar(ref_descs[0], want_buf.data());
            cpu::reference::matmul_ref(attrs, want, ref_ins, {}, nullptr);

            for (size_t i = 0; i < got_buf.size(); ++i) {
                NNOPS_EXPECT_NEAR(to_f(got_buf[i]), want_buf[i], tol);
            }
        }
    };

    run(float{});
    run(half{});
}

// ============================================================
// The prepacked B — MatMul::prepackWeights
// ============================================================
//
// Packing B once at load time takes the per-call RHS pack out of the GEMM and
// changes nothing else — the same uniform panels, the same k-blocks, the same
// accumulation order — so where the raw B also takes the tiled GEMM the two
// products must agree to the bit. Below the small product's thresholds the raw
// B takes the pack-free kernel instead (a prepacked B is panel-formatted and
// cannot), and there the two are compared as numbers.

NNOPS_TEST(matmul_prepacked_b) {
    namespace cpu = nnops::backend::cpu;

    struct Case {
        int64_t B, M, K, N;
        bool tb;
        bool bitwise;
    };
    const Case cases[] = {
        {1,  5,   37,  19, false, true},   // tails on both axes
        {1,  5,   37,  19, true,  true},   // the same product, B stored [N, K]
        {1,  1,    8,   5, true,  false},  // one row: the raw B is a GEMV
        {3, 16,  300,  48, false, true},   // batched A, several k-blocks
        {2, 64,  256,  96, true,  true},   // a layer shape, B broadcast
        {1,  4, 1024,  17, false, false},  // deep K, M = 4: the raw B is a GEMV
    };

    const auto run = [&](auto dtype_tag) {
        using T = decltype(dtype_tag);
        const DataType dt = std::is_same_v<T, float> ? DataType::f32
                                                    : DataType::f16;
        const auto cast = [](float v) -> T {
            if constexpr (std::is_same_v<T, float>) { return v; }
            else { return cpu::float_to_half(v); }
        };

        for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
            const Case& c = cases[ci];
            const int64_t ashape[] = {c.B, c.M, c.K};
            const int64_t bshape[] = {c.tb ? c.N : c.K, c.tb ? c.K : c.N};

            std::vector<T> av(static_cast<size_t>(c.B * c.M * c.K));
            std::vector<T> bv(static_cast<size_t>(c.K * c.N));
            nnops::test::XorShift128 rng(500 + 10 * static_cast<uint64_t>(ci));
            for (auto& e : av) { e = cast(rng.next_float()); }
            for (auto& e : bv) { e = cast(rng.next_float()); }

            TensorView a(ashape, dt, av.data());
            TensorView b(bshape, dt, bv.data());

            MatMulAttributes attrs;
            attrs.transpose_b = c.tb;
            auto op = MatMul::create(attrs, Backend::CPU);

            const TensorDesc descs[] = {a.desc(), b.desc()};
            const auto od = op->getOutputTensorDesc(descs);

            std::vector<T> raw(static_cast<size_t>(od[0].numel()));
            std::vector<T> pk(static_cast<size_t>(od[0].numel()));
            auto ro = nnops::test::make_planar(od[0], raw.data());
            auto po = nnops::test::make_planar(od[0], pk.data());

            const TensorView ins[] = {a, b};
            NNOPS_EXPECT_EQ(op->compute(ro, ins, {}, nullptr), Status::ok);

            // ---- two-step contract: query, allocate, fill
            TensorView q;
            const TensorView bq[] = {b};
            TensorView oq[] = {q};
            op->prepackWeights(bq, oq);
            q = oq[0];
            // A query answer carries no storage (as Conv2D's does): the rank
            // is what separates it from "this B has no packed form".
            NNOPS_EXPECT_EQ(q.rank(), int64_t(2));
            NNOPS_EXPECT_EQ(static_cast<int>(q.layout()),
                            static_cast<int>(TensorLayout::PackedB));
            // Leading N: shape inference reads it, and it is what the panel
            // count is built from.
            NNOPS_EXPECT_EQ(q.shape(0), c.N);
            NNOPS_EXPECT_TRUE(q.desc().numel() >= c.N * c.K);

            std::vector<T> blob(static_cast<size_t>(q.desc().numel()));
            TensorView packed(q.shape_span(), q.data_type(), blob.data(),
                              q.pitch(), q.layout());
            const TensorView bp[] = {b};
            TensorView opk[] = {packed};
            op->prepackWeights(bp, opk);

            // The blob's descriptor resolves the same product shape as the raw
            // B did.
            const TensorDesc pdescs[] = {a.desc(), packed.desc()};
            const auto pod = op->getOutputTensorDesc(pdescs);
            NNOPS_EXPECT_EQ(pod[0].rank, od[0].rank);
            for (int64_t d = 0; d < od[0].rank; ++d) {
                NNOPS_EXPECT_EQ(pod[0].dims[static_cast<size_t>(d)],
                                od[0].dims[static_cast<size_t>(d)]);
            }

            const TensorView pins[] = {a, packed};
            NNOPS_EXPECT_EQ(op->compute(po, pins, {}, nullptr), Status::ok);

            if (c.bitwise) {
                NNOPS_EXPECT_TRUE(std::memcmp(raw.data(), pk.data(),
                                              raw.size() * sizeof(T)) == 0);
                continue;
            }
            const float tol = std::is_same_v<T, float> ? 1e-5f : kF16AccumTol;
            const auto to_f = [](T v) -> float {
                if constexpr (std::is_same_v<T, float>) { return v; }
                else { return cpu::half_to_float(v); }
            };
            for (size_t i = 0; i < raw.size(); ++i) {
                const float e = to_f(raw[i]);
                NNOPS_EXPECT_NEAR(to_f(pk[i]), e, tol * std::max(1.0f, std::abs(e)));
            }
        }
    };

    run(float{});
    run(half{});
}

NNOPS_TEST(matmul_prepacked_b_contract_errors) {
    const int64_t bshape[] = {37, 19};
    auto [bv, b] = test::make_random_tensor({37, 19}, -1.0f, 1.0f, 61);

    auto op = MatMul::create(MatMulAttributes{}, Backend::CPU);
    TensorView q;
    const TensorView bq[] = {b};
    TensorView oq[] = {q};
    op->prepackWeights(bq, oq);
    q = oq[0];

    std::vector<float> blob(static_cast<size_t>(q.desc().numel()));
    TensorView packed(q.shape_span(), q.data_type(), blob.data(), q.pitch(),
                      q.layout());
    const TensorView bp[] = {b};
    TensorView opk[] = {packed};
    op->prepackWeights(bp, opk);

    // The blob carries no K (only N), so a caller who packs against one K and
    // computes against another is refused rather than read past the panels.
    auto [av, a] = test::make_random_tensor({4, 31}, -1.0f, 1.0f, 62);
    const int64_t oshape[] = {4, 19};
    std::vector<float> obuf(static_cast<size_t>(4 * 19));
    TensorView out(oshape, DataType::f32, obuf.data());
    const TensorView ins[] = {a, packed};
    NNOPS_EXPECT_EQ(op->compute(out, ins, {}, nullptr),
                    Status::invalid_argument);

    // alpha rode the per-call RHS pack: a load-time blob has no slot for it.
    MatMulAttributes scaled;
    scaled.alpha = 2.0f;
    auto sop = MatMul::create(scaled, Backend::CPU);
    const int64_t ok_a[] = {4, 37};
    std::vector<float> abuf2(static_cast<size_t>(4 * 37), 1.0f);
    TensorView a2(ok_a, DataType::f32, abuf2.data());
    const TensorView sins[] = {a2, packed};
    NNOPS_EXPECT_EQ(sop->compute(out, sins, {}, nullptr), Status::unsupported);
}
