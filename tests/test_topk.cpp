/// @file test_topk.cpp
/// @brief Unit tests for TopK operator (CPU backend).

#include "nnops/ops/topk.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"

#include <vector>
#include <algorithm>
#include <cstring>

using namespace nnops;

// ============================================================
// Hand-verified tests
// ============================================================

NNOPS_TEST(topk_1d_max_basic) {
    const int64_t shape[] = {6};
    float data[] = {3.0f, 1.0f, 7.0f, 2.0f, 9.0f, 5.0f};
    float val_buf[3] = {};
    int64_t idx_buf[3] = {-1, -1, -1};

    TensorView in(shape, DataType::f32, data);

    TopKAttributes attrs;
    attrs.type = TopKType::Max;
    attrs.axis = 0;
    attrs.k = 3;
    attrs.sorted = true;
    auto op = TopK::create(attrs, Backend::CPU);

    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    // Values output
    NNOPS_EXPECT_EQ(descs[0].rank, 1);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);
    // Indices output
    NNOPS_EXPECT_EQ(descs[1].rank, 1);
    NNOPS_EXPECT_EQ(descs[1].dims[0], 3);
    NNOPS_EXPECT_EQ(descs[1].dtype, DataType::s64);

    auto val_out = test::make_planar(descs[0], val_buf);
    auto idx_out = test::make_planar(descs[1], idx_buf);
    TensorView outs[] = {val_out, idx_out};
    const TensorView ins[] = {in};
    op->compute(outs, ins);

    // Sorted descending: 9, 7, 5
    NNOPS_EXPECT_NEAR(val_buf[0], 9.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(val_buf[1], 7.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(val_buf[2], 5.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[0], 4);  // 9 at index 4
    NNOPS_EXPECT_EQ(idx_buf[1], 2);  // 7 at index 2
    NNOPS_EXPECT_EQ(idx_buf[2], 5);  // 5 at index 5
}

NNOPS_TEST(topk_1d_min_basic) {
    const int64_t shape[] = {6};
    float data[] = {3.0f, 1.0f, 7.0f, 2.0f, 9.0f, 5.0f};
    float val_buf[3] = {};
    int64_t idx_buf[3] = {-1, -1, -1};

    TensorView in(shape, DataType::f32, data);

    TopKAttributes attrs;
    attrs.type = TopKType::Min;
    attrs.axis = 0;
    attrs.k = 3;
    attrs.sorted = true;
    auto op = TopK::create(attrs, Backend::CPU);

    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    auto val_out = test::make_planar(descs[0], val_buf);
    auto idx_out = test::make_planar(descs[1], idx_buf);
    TensorView outs[] = {val_out, idx_out};
    const TensorView ins[] = {in};
    op->compute(outs, ins);

    // Sorted ascending: 1, 2, 3
    NNOPS_EXPECT_NEAR(val_buf[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(val_buf[1], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(val_buf[2], 3.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[0], 1);  // 1 at index 1
    NNOPS_EXPECT_EQ(idx_buf[1], 3);  // 2 at index 3
    NNOPS_EXPECT_EQ(idx_buf[2], 0);  // 3 at index 0
}

NNOPS_TEST(topk_1d_unsorted) {
    const int64_t shape[] = {5};
    float data[] = {10.0f, 3.0f, 8.0f, 1.0f, 6.0f};
    float val_buf[3] = {};
    int64_t idx_buf[3] = {-1, -1, -1};

    TensorView in(shape, DataType::f32, data);

    TopKAttributes attrs;
    attrs.type = TopKType::Max;
    attrs.axis = 0;
    attrs.k = 3;
    attrs.sorted = false;
    auto op = TopK::create(attrs, Backend::CPU);

    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    auto val_out = test::make_planar(descs[0], val_buf);
    auto idx_out = test::make_planar(descs[1], idx_buf);
    TensorView outs[] = {val_out, idx_out};
    const TensorView ins[] = {in};
    op->compute(outs, ins);

    // Unsorted: top 3 values are 10, 8, 6 (any order)
    // Verify the set
    float found_vals[3] = {val_buf[0], val_buf[1], val_buf[2]};
    int64_t found_idx[3] = {idx_buf[0], idx_buf[1], idx_buf[2]};

    // Check each expected value appears
    bool has_10 = false, has_8 = false, has_6 = false;
    for (int i = 0; i < 3; ++i) {
        if (found_vals[i] == 10.0f) has_10 = true;
        if (found_vals[i] == 8.0f) has_8 = true;
        if (found_vals[i] == 6.0f) has_6 = true;
    }
    NNOPS_EXPECT_TRUE(has_10);
    NNOPS_EXPECT_TRUE(has_8);
    NNOPS_EXPECT_TRUE(has_6);

    // Check indices match values
    for (int i = 0; i < 3; ++i) {
        NNOPS_EXPECT_NEAR(data[found_idx[i]], found_vals[i], 1e-6f);
    }
}

NNOPS_TEST(topk_2d_last_axis) {
    // [2, 4], k=2, axis=-1 (last axis): top-2 per row
    const int64_t shape[] = {2, 4};
    float data[] = {3.0f, 7.0f, 1.0f, 5.0f,
                    2.0f, 0.0f, 9.0f, 4.0f};
    float val_buf[4] = {};
    int64_t idx_buf[4] = {-1};

    TensorView in(shape, DataType::f32, data);

    TopKAttributes attrs;
    attrs.type = TopKType::Max;
    attrs.axis = -1;
    attrs.k = 2;
    attrs.sorted = true;
    auto op = TopK::create(attrs, Backend::CPU);

    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 2);

    auto val_out = test::make_planar(descs[0], val_buf);
    auto idx_out = test::make_planar(descs[1], idx_buf);
    TensorView outs[] = {val_out, idx_out};
    const TensorView ins[] = {in};
    op->compute(outs, ins);

    // Row 0: values [3, 7, 1, 5] → top-2: 7 at idx 1, 5 at idx 3
    NNOPS_EXPECT_NEAR(val_buf[0], 7.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[0], 1);
    NNOPS_EXPECT_NEAR(val_buf[1], 5.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[1], 3);

    // Row 1: values [2, 0, 9, 4] → top-2: 9 at idx 2, 4 at idx 3
    NNOPS_EXPECT_NEAR(val_buf[2], 9.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[2], 2);
    NNOPS_EXPECT_NEAR(val_buf[3], 4.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[3], 3);
}

NNOPS_TEST(topk_2d_axis0) {
    // [3, 2], k=2, axis=0: top-2 per column
    const int64_t shape[] = {3, 2};
    float data[] = {5.0f, 1.0f,
                    3.0f, 8.0f,
                    7.0f, 2.0f};
    float val_buf[4] = {};   // [2, 2]
    int64_t idx_buf[4] = {-1};

    TensorView in(shape, DataType::f32, data);

    TopKAttributes attrs;
    attrs.type = TopKType::Max;
    attrs.axis = 0;
    attrs.k = 2;
    attrs.sorted = true;
    auto op = TopK::create(attrs, Backend::CPU);

    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);  // k=2
    NNOPS_EXPECT_EQ(descs[0].dims[1], 2);

    auto val_out = test::make_planar(descs[0], val_buf);
    auto idx_out = test::make_planar(descs[1], idx_buf);
    TensorView outs[] = {val_out, idx_out};
    const TensorView ins[] = {in};
    op->compute(outs, ins);

    // Col 0: values [5, 3, 7] → top-2: 7 at idx 2, 5 at idx 0
    NNOPS_EXPECT_NEAR(val_buf[0], 7.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[0], 2);
    NNOPS_EXPECT_NEAR(val_buf[1], 5.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[1], 0);

    // Col 1: values [1, 8, 2] → top-2: 8 at idx 1, 2 at idx 2
    NNOPS_EXPECT_NEAR(val_buf[2], 8.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[2], 1);
    NNOPS_EXPECT_NEAR(val_buf[3], 2.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[3], 2);
}

NNOPS_TEST(topk_k1_matches_argmax) {
    // k=1 should match ArgMax/ArgMin behavior
    const int64_t shape[] = {2, 5};
    float data[] = {3.0f, 8.0f, 1.0f, 5.0f, 2.0f,
                    7.0f, 0.0f, 9.0f, 4.0f, 6.0f};
    float val_buf[2] = {};
    int64_t idx_buf[2] = {-1, -1};

    TensorView in(shape, DataType::f32, data);

    TopKAttributes attrs;
    attrs.type = TopKType::Max;
    attrs.axis = -1;
    attrs.k = 1;
    auto op = TopK::create(attrs, Backend::CPU);

    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 2);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 1);

    auto val_out = test::make_planar(descs[0], val_buf);
    auto idx_out = test::make_planar(descs[1], idx_buf);
    TensorView outs[] = {val_out, idx_out};
    const TensorView ins[] = {in};
    op->compute(outs, ins);

    // Row 0: max is 8.0 at idx 1
    NNOPS_EXPECT_NEAR(val_buf[0], 8.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[0], 1);
    // Row 1: max is 9.0 at idx 2
    NNOPS_EXPECT_NEAR(val_buf[1], 9.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[1], 2);
}

NNOPS_TEST(topk_4d_nchw) {
    // NCHW [1, 2, 1, 4], k=2, axis=-1 (W axis)
    const int64_t shape[] = {1, 2, 1, 4};
    float data[] = {1.0f, 9.0f, 3.0f, 7.0f,   // C=0
                    4.0f, 2.0f, 8.0f, 6.0f};  // C=1
    float val_buf[4] = {};   // [1, 2, 1, 2]
    int64_t idx_buf[4] = {-1};

    TensorView in(shape, DataType::f32, data);

    TopKAttributes attrs;
    attrs.type = TopKType::Max;
    attrs.axis = -1;
    attrs.k = 2;
    attrs.sorted = true;
    auto op = TopK::create(attrs, Backend::CPU);

    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 4);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 1);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 2);
    NNOPS_EXPECT_EQ(descs[0].dims[2], 1);
    NNOPS_EXPECT_EQ(descs[0].dims[3], 2);

    auto val_out = test::make_planar(descs[0], val_buf);
    auto idx_out = test::make_planar(descs[1], idx_buf);
    TensorView outs[] = {val_out, idx_out};
    const TensorView ins[] = {in};
    op->compute(outs, ins);

    // C=0: [1, 9, 3, 7] → top-2: 9 at idx 1, 7 at idx 3
    NNOPS_EXPECT_NEAR(val_buf[0], 9.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[0], 1);
    NNOPS_EXPECT_NEAR(val_buf[1], 7.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[1], 3);

    // C=1: [4, 2, 8, 6] → top-2: 8 at idx 2, 6 at idx 3
    NNOPS_EXPECT_NEAR(val_buf[2], 8.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[2], 2);
    NNOPS_EXPECT_NEAR(val_buf[3], 6.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[3], 3);
}

NNOPS_TEST(topk_5d_ncdhw) {
    // NCDHW [1, 1, 2, 1, 3], k=2, axis=-1 (W axis)
    const int64_t shape[] = {1, 1, 2, 1, 3};
    float data[] = {10.0f, 2.0f, 6.0f,   // D=0
                    5.0f, 8.0f, 1.0f};  // D=1
    float val_buf[4] = {};   // [1, 1, 2, 1, 2]
    int64_t idx_buf[4] = {-1};

    TensorView in(shape, DataType::f32, data);

    TopKAttributes attrs;
    attrs.type = TopKType::Max;
    attrs.axis = -1;
    attrs.k = 2;
    attrs.sorted = true;
    auto op = TopK::create(attrs, Backend::CPU);

    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 5);
    NNOPS_EXPECT_EQ(descs[0].dims[4], 2);  // W dim replaced by k

    auto val_out = test::make_planar(descs[0], val_buf);
    auto idx_out = test::make_planar(descs[1], idx_buf);
    TensorView outs[] = {val_out, idx_out};
    const TensorView ins[] = {in};
    op->compute(outs, ins);

    // D=0: [10, 2, 6] → top-2: 10 at idx 0, 6 at idx 2
    NNOPS_EXPECT_NEAR(val_buf[0], 10.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[0], 0);
    NNOPS_EXPECT_NEAR(val_buf[1], 6.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[1], 2);

    // D=1: [5, 8, 1] → top-2: 8 at idx 1, 5 at idx 0
    NNOPS_EXPECT_NEAR(val_buf[2], 8.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[2], 1);
    NNOPS_EXPECT_NEAR(val_buf[3], 5.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[3], 0);
}

// ============================================================
// Random tests
// ============================================================

NNOPS_TEST(topk_random_vs_ref) {
    auto [data, in] = test::make_random_tensor({5, 20}, -10.0f, 10.0f, 42);

    TopKAttributes attrs;
    attrs.type = TopKType::Max;
    attrs.axis = -1;
    attrs.k = 5;
    attrs.sorted = true;
    auto op = TopK::create(attrs, Backend::CPU);

    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    std::vector<float> val_buf(25);
    std::vector<int64_t> idx_buf(25);
    auto val_out = test::make_planar(descs[0], val_buf.data());
    auto idx_out = test::make_planar(descs[1], idx_buf.data());
    TensorView outs[] = {val_out, idx_out};
    const TensorView ins[] = {in};
    op->compute(outs, ins);

    // Verify each row's top-5 by brute force
    for (int64_t r = 0; r < 5; ++r) {
        // Build sorted pairs for this row
        struct Pair { float v; int64_t i; };
        std::vector<Pair> pairs(20);
        for (int64_t c = 0; c < 20; ++c) {
            pairs[static_cast<size_t>(c)] = {data[r * 20 + c], c};
        }
        std::sort(pairs.begin(), pairs.end(),
            [](const Pair& a, const Pair& b) { return a.v > b.v; });

        for (int64_t j = 0; j < 5; ++j) {
            int64_t offset = r * 5 + j;
            NNOPS_EXPECT_NEAR(val_buf[offset], pairs[j].v, 1e-4f);
            NNOPS_EXPECT_EQ(idx_buf[offset], pairs[j].i);
        }
    }
}

NNOPS_TEST(topk_random_min_vs_ref) {
    auto [data, in] = test::make_random_tensor({3, 15}, -10.0f, 10.0f, 77);

    TopKAttributes attrs;
    attrs.type = TopKType::Min;
    attrs.axis = -1;
    attrs.k = 4;
    attrs.sorted = true;
    auto op = TopK::create(attrs, Backend::CPU);

    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    std::vector<float> val_buf(12);
    std::vector<int64_t> idx_buf(12);
    auto val_out = test::make_planar(descs[0], val_buf.data());
    auto idx_out = test::make_planar(descs[1], idx_buf.data());
    TensorView outs[] = {val_out, idx_out};
    const TensorView ins[] = {in};
    op->compute(outs, ins);

    for (int64_t r = 0; r < 3; ++r) {
        struct Pair { float v; int64_t i; };
        std::vector<Pair> pairs(15);
        for (int64_t c = 0; c < 15; ++c) {
            pairs[static_cast<size_t>(c)] = {data[r * 15 + c], c};
        }
        std::sort(pairs.begin(), pairs.end(),
            [](const Pair& a, const Pair& b) { return a.v < b.v; });

        for (int64_t j = 0; j < 4; ++j) {
            int64_t offset = r * 4 + j;
            NNOPS_EXPECT_NEAR(val_buf[offset], pairs[j].v, 1e-4f);
            NNOPS_EXPECT_EQ(idx_buf[offset], pairs[j].i);
        }
    }
}

// ============================================================
// Edge cases
// ============================================================

NNOPS_TEST(topk_k_equals_axis_dim) {
    // k == axis_dim: should return all elements, sorted
    const int64_t shape[] = {4};
    float data[] = {3.0f, 1.0f, 4.0f, 2.0f};
    float val_buf[4] = {};
    int64_t idx_buf[4] = {-1};

    TensorView in(shape, DataType::f32, data);

    TopKAttributes attrs;
    attrs.type = TopKType::Max;
    attrs.axis = 0;
    attrs.k = 4;
    attrs.sorted = true;
    auto op = TopK::create(attrs, Backend::CPU);

    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    auto val_out = test::make_planar(descs[0], val_buf);
    auto idx_out = test::make_planar(descs[1], idx_buf);
    TensorView outs[] = {val_out, idx_out};
    const TensorView ins[] = {in};
    op->compute(outs, ins);

    // Sorted descending: 4 at 2, 3 at 0, 2 at 3, 1 at 1
    NNOPS_EXPECT_NEAR(val_buf[0], 4.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[0], 2);
    NNOPS_EXPECT_NEAR(val_buf[1], 3.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[1], 0);
    NNOPS_EXPECT_NEAR(val_buf[2], 2.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[2], 3);
    NNOPS_EXPECT_NEAR(val_buf[3], 1.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[3], 1);
}

NNOPS_TEST(topk_negative_axis) {
    // Negative axis: -3 means C axis in NCHW (0=N, 1=C, 2=H, 3=W)
    const int64_t shape[] = {1, 3, 1, 2};  // NCHW
    float data[] = {5.0f, 1.0f,   // C=0
                    2.0f, 4.0f,   // C=1
                    3.0f, 6.0f};  // C=2
    float val_buf[4] = {};   // [1, 2, 1, 2]
    int64_t idx_buf[4] = {-1};

    TensorView in(shape, DataType::f32, data);

    TopKAttributes attrs;
    attrs.type = TopKType::Max;
    attrs.axis = -3;  // C axis
    attrs.k = 2;
    attrs.sorted = true;
    auto op = TopK::create(attrs, Backend::CPU);

    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(descs[0].dims[1], 2);  // C dim replaced by k

    auto val_out = test::make_planar(descs[0], val_buf);
    auto idx_out = test::make_planar(descs[1], idx_buf);
    TensorView outs[] = {val_out, idx_out};
    const TensorView ins[] = {in};
    op->compute(outs, ins);

    // Position (n=0, h=0, w=0): C values [5, 2, 3] → top-2: 5(idx 0), 3(idx 2)
    NNOPS_EXPECT_NEAR(val_buf[0], 5.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[0], 0);
    NNOPS_EXPECT_NEAR(val_buf[1], 3.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[1], 2);

    // Position (n=0, h=0, w=1): C values [1, 4, 6] → top-2: 6(idx 2), 4(idx 1)
    NNOPS_EXPECT_NEAR(val_buf[2], 6.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[2], 2);
    NNOPS_EXPECT_NEAR(val_buf[3], 4.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[3], 1);
}

// ============================================================
// Functional API
// ============================================================

NNOPS_TEST(topk_functional_api) {
    const int64_t shape[] = {5};
    float data[] = {2.0f, 8.0f, 4.0f, 1.0f, 6.0f};
    float val_buf[3] = {};
    int64_t idx_buf[3] = {-1};

    TensorView in(shape, DataType::f32, data);

    TopKAttributes attrs;
    attrs.type = TopKType::Max;
    attrs.axis = 0;
    attrs.k = 3;
    attrs.sorted = true;

    auto op = TopK::create(attrs, Backend::CPU);
    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    auto val_out = test::make_planar(descs[0], val_buf);
    auto idx_out = test::make_planar(descs[1], idx_buf);
    TensorView out_arr[] = {val_out, idx_out};

    topk(in, val_out, idx_out, attrs);

    NNOPS_EXPECT_NEAR(val_buf[0], 8.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[0], 1);
    NNOPS_EXPECT_NEAR(val_buf[1], 6.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[1], 4);
    NNOPS_EXPECT_NEAR(val_buf[2], 4.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[2], 2);
}

// ============================================================
// OpType
// ============================================================

NNOPS_TEST(topk_op_type) {
    auto op = TopK::create(Backend::CPU);
    NNOPS_EXPECT_EQ(static_cast<int>(op->getOpType()), static_cast<int>(OpType::TopK));
}
