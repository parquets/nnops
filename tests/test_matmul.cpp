/// Unit tests for MatMul operator — class API with CPU reference.
///
/// All tests use the MatMul class interface:
///   1. auto op = MatMul::create(attrs, Backend::CPU);
///   2. const TensorDesc arr[] = {a_desc, b_desc}; auto descs = op->getOutputTensorDesc(arr);
///   3. Validate descs[0].rank/dims/layout/dtype
///   4. Create output via nnops::test::make_planar(descs[0], buf.data())
///   5. const TensorView ins[] = {a, b}; op->compute(out, ins);

#include "nnops/ops/matmul.hpp"
#include "backend/cpu/matmul_helper.h"   // get_matmul_plan — to pin the pack_a route
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <atomic>
#include <vector>
#include <thread>
#include <cmath>

using namespace nnops;

// Reference kernel for packed-path comparison (test-local declaration,
// following the test_concat.cpp pattern).
namespace nnops::backend::cpu::reference {
void matmul_ref(const MatMulAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* workspace);
}

// ============================================================
// f16 GEMM comparison tolerance
// ============================================================
//
// The f16 MMA kernels accumulate in different precisions per arch:
//   - x86_64:  fp32 accumulators (F16C widen → FMA → narrow), error ~fp32.
//   - aarch64: native fp16 accumulators (vfmaq_laneq_f16), so the error grows
//              with K and reaches ~0.09 for K=256 (≈3-4 fp16 ulp of the result).
// The tolerance is therefore arch-specific; the x86 value would be far too
// tight on aarch64 for any K beyond a few dozen.
#if defined(NNOPS_ARCH_AARCH64)
inline constexpr float kF16GemmTol = 2e-1f;
#else
inline constexpr float kF16GemmTol = 5e-2f;
#endif

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
    // K = 300 > KC_F32 (128): exercises multi-k-block accumulation and the
    // last-k-block epilogue/clamp logic in the packed path.
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
        NNOPS_EXPECT_NEAR(v, ref_buf[static_cast<size_t>(i)], kF16GemmTol);
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
        NNOPS_EXPECT_NEAR(v, ref_buf[static_cast<size_t>(i)], kF16GemmTol);
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
    // decomposes M with its own panel heights — MR_F32_DIRECT, {6,4,1}, so Mc
    // splits as 6a + 4b + 1c. Walk M across that boundary and K across both the
    // 4-wide inner unroll and KC_F32, so the tall tile, the 4/1 remainder
    // kernels and the scalar k-tail are each checked against the reference.
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

                // Pin the route. If a compact non-transposed A ever stops being
                // lda == K <= PACK_A_STRIDE_THRESHOLD this test would silently
                // start covering the packed kernels instead of the direct ones.
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
