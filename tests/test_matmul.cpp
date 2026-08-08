/// Unit tests for MatMul operator — class API with CPU reference.
///
/// All tests use the MatMul class interface:
///   1. auto op = MatMul::create(attrs, Backend::CPU);
///   2. const TensorDesc arr[] = {a_desc, b_desc}; auto descs = op->getOutputTensorDesc(arr);
///   3. Validate descs[0].rank/dims/layout/dtype
///   4. Create output via nnops::test::make_planar(descs[0], buf.data())
///   5. const TensorView ins[] = {a, b}; op->compute(out, ins);

#include "nnops/ops/matmul.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

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
