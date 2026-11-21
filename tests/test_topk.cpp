/// @file test_topk.cpp
/// @brief TopK on the CPU backend: values and indices along an axis.
///
/// Each hand-checked case spells out the values it expects and where they come
/// from; the random cases rebuild the answer by sorting. Covers k, negative-axis
/// wrap, sorted and unsorted, and the (Values, Indices) outputs.

#include "nnops/ops/topk.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

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

    auto val_out = test::make_channel_first(descs[0], val_buf);
    auto idx_out = test::make_channel_first(descs[1], idx_buf);
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

    auto val_out = test::make_channel_first(descs[0], val_buf);
    auto idx_out = test::make_channel_first(descs[1], idx_buf);
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

    auto val_out = test::make_channel_first(descs[0], val_buf);
    auto idx_out = test::make_channel_first(descs[1], idx_buf);
    TensorView outs[] = {val_out, idx_out};
    const TensorView ins[] = {in};
    op->compute(outs, ins);

    // Unsorted: top 3 values are 10, 8, 6 (any order)
    float found_vals[3] = {val_buf[0], val_buf[1], val_buf[2]};
    int64_t found_idx[3] = {idx_buf[0], idx_buf[1], idx_buf[2]};

    bool has_10 = false, has_8 = false, has_6 = false;
    for (int i = 0; i < 3; ++i) {
        if (found_vals[i] == 10.0f) {
            has_10 = true;
        }
        if (found_vals[i] == 8.0f) {
            has_8 = true;
        }
        if (found_vals[i] == 6.0f) {
            has_6 = true;
        }
    }
    NNOPS_EXPECT_TRUE(has_10);
    NNOPS_EXPECT_TRUE(has_8);
    NNOPS_EXPECT_TRUE(has_6);

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

    auto val_out = test::make_channel_first(descs[0], val_buf);
    auto idx_out = test::make_channel_first(descs[1], idx_buf);
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

    auto val_out = test::make_channel_first(descs[0], val_buf);
    auto idx_out = test::make_channel_first(descs[1], idx_buf);
    TensorView outs[] = {val_out, idx_out};
    const TensorView ins[] = {in};
    op->compute(outs, ins);

    // The output is a dense [k][inner] tensor, so it is read in row-major
    // order: rank j of every column first, then the next rank.
    // Col 0: values [5, 3, 7] → top-2: 7 at idx 2, 5 at idx 0
    // Col 1: values [1, 8, 2] → top-2: 8 at idx 1, 2 at idx 2
    NNOPS_EXPECT_NEAR(val_buf[0], 7.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[0], 2);
    NNOPS_EXPECT_NEAR(val_buf[1], 8.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[1], 1);
    NNOPS_EXPECT_NEAR(val_buf[2], 5.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[2], 0);
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

    auto val_out = test::make_channel_first(descs[0], val_buf);
    auto idx_out = test::make_channel_first(descs[1], idx_buf);
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

NNOPS_TEST(topk_4d_channel_first) {
    // Channel-first [1, 2, 1, 4], k=2, axis=-1 (W axis)
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

    auto val_out = test::make_channel_first(descs[0], val_buf);
    auto idx_out = test::make_channel_first(descs[1], idx_buf);
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
    // Channel-first 3D [1, 1, 2, 1, 3], k=2, axis=-1 (W axis)
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

    auto val_out = test::make_channel_first(descs[0], val_buf);
    auto idx_out = test::make_channel_first(descs[1], idx_buf);
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
    auto val_out = test::make_channel_first(descs[0], val_buf.data());
    auto idx_out = test::make_channel_first(descs[1], idx_buf.data());
    TensorView outs[] = {val_out, idx_out};
    const TensorView ins[] = {in};
    op->compute(outs, ins);

    // Verify each row's top-5 by brute force
    for (int64_t r = 0; r < 5; ++r) {
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
    auto val_out = test::make_channel_first(descs[0], val_buf.data());
    auto idx_out = test::make_channel_first(descs[1], idx_buf.data());
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

    auto val_out = test::make_channel_first(descs[0], val_buf);
    auto idx_out = test::make_channel_first(descs[1], idx_buf);
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
    // Negative axis: -3 means C axis in dense order (0=N, 1=C, 2=H, 3=W)
    const int64_t shape[] = {1, 3, 1, 2};  // Channel-first
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

    auto val_out = test::make_channel_first(descs[0], val_buf);
    auto idx_out = test::make_channel_first(descs[1], idx_buf);
    TensorView outs[] = {val_out, idx_out};
    const TensorView ins[] = {in};
    op->compute(outs, ins);

    // The output is dense [1][k][1][2], so the two ranks come one after the
    // other: (j=0, w=0), (j=0, w=1), (j=1, w=0), (j=1, w=1).
    // w=0: C values [5, 2, 3] → top-2: 5(idx 0), 3(idx 2)
    // w=1: C values [1, 4, 6] → top-2: 6(idx 2), 4(idx 1)
    NNOPS_EXPECT_NEAR(val_buf[0], 5.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[0], 0);
    NNOPS_EXPECT_NEAR(val_buf[1], 6.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[1], 2);
    NNOPS_EXPECT_NEAR(val_buf[2], 3.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[2], 2);
    NNOPS_EXPECT_NEAR(val_buf[3], 4.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[3], 1);
}

// ============================================================
// Class API
// ============================================================

NNOPS_TEST(topk_class_api) {
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

    auto val_out = test::make_channel_first(descs[0], val_buf);
    auto idx_out = test::make_channel_first(descs[1], idx_buf);
    TensorView out_arr[] = {val_out, idx_out};

    const TensorView ins[] = {in};
    op->compute(out_arr, ins);

    NNOPS_EXPECT_NEAR(val_buf[0], 8.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[0], 1);
    NNOPS_EXPECT_NEAR(val_buf[1], 6.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[1], 4);
    NNOPS_EXPECT_NEAR(val_buf[2], 4.0f, 1e-6f);
    NNOPS_EXPECT_EQ(idx_buf[2], 2);
}

// ============================================================
// The axis-split route: fewer outer positions than workers
// ============================================================

/// A tiny thread pool: split [begin, end) over the calling thread plus three,
/// which is what makes the axis split (a single long row) reachable in a test.
static ComputeContext make_parallel_context(int nthreads) {
    ComputeContext ctx;
    auto slots = std::make_shared<int>(nthreads);
    ctx.cpu.num_threads = [slots]() { return *slots; };
    ctx.cpu.parallel_for = [slots](int64_t begin, int64_t end, const ParallelForBody& body) {
        const int nt = *slots;
        const int64_t n = end - begin;
        if (n <= 1 || nt <= 1) {
            for (int64_t i = begin; i < end; ++i) {
                body(i);
            }
            return;
        }
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(nt - 1));
        const int64_t per = (n + nt - 1) / nt;
        for (int t = 1; t < nt; ++t) {
            const int64_t lo = begin + t * per;
            const int64_t hi = std::min(lo + per, end);
            if (lo >= hi) {
                break;
            }
            workers.emplace_back([&body, lo, hi] {
                for (int64_t i = lo; i < hi; ++i) {
                    body(i);
                }
            });
        }
        const int64_t hi0 = std::min(begin + per, end);
        for (int64_t i = begin; i < hi0; ++i) {
            body(i);
        }
        for (auto& w : workers) {
            w.join();
        }
    };
    return ctx;
}

NNOPS_TEST(topk_axis_split_matches_serial) {
    constexpr int64_t V = 20000;
    constexpr int64_t K = 7;
    std::vector<float> data(static_cast<size_t>(V));
    uint32_t seed = 12345u;
    for (int64_t i = 0; i < V; ++i) {
        seed = seed * 1664525u + 1013904223u;
        data[static_cast<size_t>(i)] =
            static_cast<float>(seed >> 8) * 1e-6f - 2.0f; // distinct values
    }

    const int64_t shape[] = {V};
    TensorView in(shape, DataType::f32, data.data());

    TopKAttributes attrs;
    attrs.type = TopKType::Max;
    attrs.axis = 0;
    attrs.k = K;
    attrs.sorted = true;
    auto op = TopK::create(attrs, Backend::CPU);

    auto d_in = in.desc();
    const TensorDesc desc_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(desc_arr);

    std::vector<float> v_serial(static_cast<size_t>(K));
    std::vector<int64_t> i_serial(static_cast<size_t>(K));
    std::vector<float> v_par(static_cast<size_t>(K));
    std::vector<int64_t> i_par(static_cast<size_t>(K));

    {
        auto vo = test::make_channel_first(descs[0], v_serial.data());
        auto io = test::make_channel_first(descs[1], i_serial.data());
        TensorView outs[] = {vo, io};
        const TensorView ins[] = {in};
        op->compute(outs, ins);
    }
    {
        auto vo = test::make_channel_first(descs[0], v_par.data());
        auto io = test::make_channel_first(descs[1], i_par.data());
        TensorView outs[] = {vo, io};
        const TensorView ins[] = {in};
        op->compute(outs, ins, make_parallel_context(4));
    }

    for (int64_t j = 0; j < K; ++j) {
        NNOPS_EXPECT_NEAR(v_par[static_cast<size_t>(j)], v_serial[static_cast<size_t>(j)], 1e-6f);
        NNOPS_EXPECT_EQ(i_par[static_cast<size_t>(j)], i_serial[static_cast<size_t>(j)]);
    }

    // ... and the split answer is the brute-force one.
    std::vector<int64_t> order(static_cast<size_t>(V));
    for (int64_t i = 0; i < V; ++i) {
        order[static_cast<size_t>(i)] = i;
    }
    std::partial_sort(order.begin(), order.begin() + K, order.end(),
                      [&data](int64_t a, int64_t b) {
                          return data[static_cast<size_t>(a)] > data[static_cast<size_t>(b)];
                      });
    for (int64_t j = 0; j < K; ++j) {
        NNOPS_EXPECT_EQ(i_par[static_cast<size_t>(j)], order[static_cast<size_t>(j)]);
        NNOPS_EXPECT_NEAR(v_par[static_cast<size_t>(j)],
                          data[static_cast<size_t>(order[static_cast<size_t>(j)])], 1e-6f);
    }
}

NNOPS_TEST(topk_op_type) {
    auto op = TopK::create(Backend::CPU);
    NNOPS_EXPECT_EQ(static_cast<int>(op->getOpType()), static_cast<int>(OpType::TopK));
}
