/// Unit tests for TensorView core data structure.

#include "nnops/core/tensor_view.hpp"
#include "common/test_harness.hpp"

using namespace nnops;

NNOPS_TEST(tensor_view_dense_construction) {
    // 2x3x4 tensor, f32, NCHW
    const int64_t shape[] = {2, 3, 4};
    float data[2 * 3 * 4] = {};

    TensorView tv(shape, DataType::f32, data);

    NNOPS_EXPECT_EQ(tv.rank(), 3);
    NNOPS_EXPECT_EQ(tv.shape(0), 2);
    NNOPS_EXPECT_EQ(tv.shape(1), 3);
    NNOPS_EXPECT_EQ(tv.shape(2), 4);
    NNOPS_EXPECT_EQ(tv.stride(0), 12);   // 3*4
    NNOPS_EXPECT_EQ(tv.stride(1), 4);    // 4
    NNOPS_EXPECT_EQ(tv.stride(2), 1);    // 1
    NNOPS_EXPECT_EQ(tv.numel(), 24);
    NNOPS_EXPECT_TRUE(tv.is_contiguous());
    NNOPS_EXPECT_FALSE(tv.is_empty());
    NNOPS_EXPECT_EQ(tv.data(), static_cast<void*>(data));
    NNOPS_EXPECT_EQ(tv.data_type(), DataType::f32);
    NNOPS_EXPECT_EQ(tv.layout(), TensorLayout::NCHW);
}

NNOPS_TEST(tensor_view_explicit_strides) {
    const int64_t shape[]   = {2, 3, 4};
    const int64_t stride[]  = {24, 8, 2};  // non-contiguous
    float data[48] = {};

    TensorView tv(shape, stride, DataType::f32, data);

    NNOPS_EXPECT_EQ(tv.shape(0), 2);
    NNOPS_EXPECT_EQ(tv.stride(0), 24);
    NNOPS_EXPECT_EQ(tv.stride(1), 8);
    NNOPS_EXPECT_EQ(tv.stride(2), 2);
    NNOPS_EXPECT_FALSE(tv.is_contiguous());
}

NNOPS_TEST(tensor_view_default_empty) {
    TensorView tv;
    NNOPS_EXPECT_EQ(tv.rank(), 0);
    NNOPS_EXPECT_EQ(tv.data(), nullptr);
    NNOPS_EXPECT_TRUE(tv.is_empty());
    NNOPS_EXPECT_EQ(tv.numel(), 0);
}

NNOPS_TEST(tensor_view_nbytes) {
    const int64_t shape[] = {2, 3, 4};
    float data[24];
    TensorView tv(shape, DataType::f32, data);
    NNOPS_EXPECT_EQ(static_cast<int64_t>(tv.nbytes()), 24 * 4);  // 24 floats * 4 bytes
}

NNOPS_TEST(tensor_view_offset) {
    const int64_t shape[] = {2, 3, 4};
    float data[24];
    TensorView tv(shape, DataType::f32, data);

    // Flat index 18: n=1, c=1, h=2 (18 = 1*12 + 1*4 + 2)
    int64_t off = tv.offset(18);
    NNOPS_EXPECT_EQ(off, 18);

    // Flat index 0:
    NNOPS_EXPECT_EQ(tv.offset(0), 0);
}

NNOPS_TEST(tensor_view_data_as) {
    const int64_t shape[] = {4};
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    TensorView tv(shape, DataType::f32, data);
    auto* fp = tv.data_as<float>();
    NNOPS_EXPECT_EQ(fp[0], 1.0f);
    NNOPS_EXPECT_EQ(fp[3], 4.0f);
}
