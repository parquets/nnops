/// Unit tests for MatMul operator (CPU reference).

#include "nnops/ops/matmul.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"

#include <vector>
#include <cmath>

using namespace nnops;

NNOPS_TEST(matmul_basic) {
    // A: [2, 3], B: [3, 2], C: [2, 2]
    const int64_t ashape[] = {2, 3};
    const int64_t bshape[] = {3, 2};
    const int64_t cshape[] = {2, 2};

    float a_data[6] = {1, 2, 3, 4, 5, 6};
    float b_data[6] = {7, 8, 9, 10, 11, 12};
    float c_data[4] = {};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);
    TensorView c(cshape, DataType::f32, c_data);

    matmul(a, b, c);

    // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    NNOPS_EXPECT_NEAR(c_data[0], 58.0f, 1e-4f);
    // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    NNOPS_EXPECT_NEAR(c_data[1], 64.0f, 1e-4f);
    // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    NNOPS_EXPECT_NEAR(c_data[2], 139.0f, 1e-4f);
    // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    NNOPS_EXPECT_NEAR(c_data[3], 154.0f, 1e-4f);
}

NNOPS_TEST(matmul_transpose_a) {
    // A: [3, 2] transposed => [2, 3], B: [3, 2], C: [2, 2]
    const int64_t ashape[] = {3, 2};  // stored as 3x2, transposed to 2x3
    const int64_t bshape[] = {3, 2};
    const int64_t cshape[] = {2, 2};

    // A stored column-major for the same data as basic test
    float a_data[6] = {1, 4, 2, 5, 3, 6};  // [3,2] -> transpose -> [2,3] = {{1,2,3},{4,5,6}}
    float b_data[6] = {7, 8, 9, 10, 11, 12};
    float c_data[4] = {};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);
    TensorView c(cshape, DataType::f32, c_data);

    MatMulAttributes attrs;
    attrs.transpose_a = true;

    matmul(a, b, c, attrs);

    // Should match basic test results
    NNOPS_EXPECT_NEAR(c_data[0], 58.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(c_data[1], 64.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(c_data[2], 139.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(c_data[3], 154.0f, 1e-4f);
}

NNOPS_TEST(matmul_transpose_b) {
    // A: [2, 3], B: [2, 3] transposed => [3, 2], C: [2, 2]
    const int64_t ashape[] = {2, 3};
    const int64_t bshape[] = {2, 3};
    const int64_t cshape[] = {2, 2};

    float a_data[6] = {1, 2, 3, 4, 5, 6};
    // B stored as [2,3], transposed to [3,2]
    float b_data[6] = {7, 9, 11, 8, 10, 12};  // [2,3] = {{7,9,11},{8,10,12}} -> T -> [3,2] = {{7,8},{9,10},{11,12}}
    float c_data[4] = {};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);
    TensorView c(cshape, DataType::f32, c_data);

    MatMulAttributes attrs;
    attrs.transpose_b = true;

    matmul(a, b, c, attrs);

    // Should match basic test results
    NNOPS_EXPECT_NEAR(c_data[0], 58.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(c_data[1], 64.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(c_data[2], 139.0f, 1e-4f);
    NNOPS_EXPECT_NEAR(c_data[3], 154.0f, 1e-4f);
}

NNOPS_TEST(matmul_random) {
    auto [a_vec, a] = test::make_random_tensor({32, 64});
    auto [b_vec, b] = test::make_random_tensor({64, 16});
    std::vector<float> c_buf(32 * 16);
    const int64_t cshape[] = {32, 16};
    TensorView c(cshape, DataType::f32, c_buf.data());

    matmul(a, b, c);

    for (size_t i = 0; i < c_buf.size(); ++i) {
        NNOPS_EXPECT_TRUE(!std::isnan(c_buf[i]));
        NNOPS_EXPECT_TRUE(!std::isinf(c_buf[i]));
    }
}

NNOPS_TEST(matmul_class_api) {
    const int64_t ashape[] = {2, 3};
    const int64_t bshape[] = {3, 2};
    const int64_t cshape[] = {2, 2};
    float a_data[6] = {1,2,3,4,5,6};
    float b_data[6] = {7,8,9,10,11,12};
    float c1_data[4] = {};
    float c2_data[4] = {};

    TensorView a(ashape, DataType::f32, a_data);
    TensorView b(bshape, DataType::f32, b_data);
    TensorView c1(cshape, DataType::f32, c1_data);
    TensorView c2(cshape, DataType::f32, c2_data);

    // Functional
    matmul(a, b, c1);
    // Class
    auto op = MatMul::create({}, Backend::CPU);
    const TensorView ins[] = {a, b};
    op->compute(c2, ins);

    NNOPS_EXPECT_TRUE(test::allclose(c1, c2));
}
