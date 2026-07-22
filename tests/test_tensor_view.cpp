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
    // pitch = last_dim * sizeof(float) = 4 * 4 = 16
    NNOPS_EXPECT_EQ(tv.pitch(), 16);
    NNOPS_EXPECT_EQ(tv.row_stride_elems(), 4);
    // stride_elems: dim 2 (innermost) = 1, dim 1 = 4, dim 0 = 3*4 = 12
    NNOPS_EXPECT_EQ(tv.stride_elems(2), 1);
    NNOPS_EXPECT_EQ(tv.stride_elems(1), 4);
    NNOPS_EXPECT_EQ(tv.stride_elems(0), 12);
    NNOPS_EXPECT_EQ(tv.numel(), 24);
    NNOPS_EXPECT_FALSE(tv.is_empty());
    NNOPS_EXPECT_EQ(tv.ptr<void>(), static_cast<void*>(data));
    NNOPS_EXPECT_EQ(tv.data_type(), DataType::f32);
    NNOPS_EXPECT_EQ(tv.layout(), TensorLayout::NCHW);
}

NNOPS_TEST(tensor_view_explicit_pitch) {
    // 2x3x4 tensor, f32, pitch = 24 bytes (6 floats, padded from 4)
    const int64_t shape[] = {2, 3, 4};
    float data[2 * 3 * 6] = {};  // buffer is larger due to pitch padding
    const int64_t pitch = 24;  // 6 floats * 4 bytes

    TensorView tv(shape, DataType::f32, data, pitch);

    NNOPS_EXPECT_EQ(tv.shape(0), 2);
    NNOPS_EXPECT_EQ(tv.shape(1), 3);
    NNOPS_EXPECT_EQ(tv.shape(2), 4);
    NNOPS_EXPECT_EQ(tv.pitch(), 24);
    NNOPS_EXPECT_EQ(tv.row_stride_elems(), 6);  // 24/4 = 6
    NNOPS_EXPECT_EQ(tv.stride_elems(2), 1);
    NNOPS_EXPECT_EQ(tv.stride_elems(1), 6);   // pitch / sizeof = 6
    NNOPS_EXPECT_EQ(tv.stride_elems(0), 18);  // 3 * 6 = 18
}

NNOPS_TEST(tensor_view_default_empty) {
    TensorView tv;
    NNOPS_EXPECT_EQ(tv.rank(), 0);
    NNOPS_EXPECT_EQ(tv.ptr<void>(), nullptr);
    NNOPS_EXPECT_TRUE(tv.is_empty());
    NNOPS_EXPECT_EQ(tv.numel(), 0);
}

NNOPS_TEST(tensor_view_nbytes) {
    const int64_t shape[] = {2, 3, 4};
    float data[24];
    TensorView tv(shape, DataType::f32, data);
    NNOPS_EXPECT_EQ(static_cast<int64_t>(tv.nbytes()), 24 * 4);  // 24 floats * 4 bytes
}

NNOPS_TEST(tensor_view_ptr) {
    const int64_t shape[] = {4};
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    TensorView tv(shape, DataType::f32, data);
    auto* fp = tv.ptr<float>();
    NNOPS_EXPECT_EQ(fp[0], 1.0f);
    NNOPS_EXPECT_EQ(fp[3], 4.0f);
}

NNOPS_TEST(tensor_view_1d_pitch) {
    // 1D tensor: pitch = elem_size
    const int64_t shape[] = {5};
    float data[5] = {};
    TensorView tv(shape, DataType::f32, data);
    NNOPS_EXPECT_EQ(tv.pitch(), 4);   // sizeof(float)
    NNOPS_EXPECT_EQ(tv.row_stride_elems(), 1);
    NNOPS_EXPECT_EQ(tv.stride_elems(0), 1);
}

NNOPS_TEST(tensor_view_nchwc8_pitch) {
    // Simulate NCHWC8 uint8, W=7, 32-byte aligned pitch = 72
    // Shape: [N, C/8, H, W, 8] but for simplicity test a 3D case
    // Actually, test a 4D tensor with external pitch
    const int64_t shape[] = {1, 2, 7, 8};  // N=1, C8=2, H=7, Cx=8
    uint8_t data[1 * 2 * 7 * 10] = {};  // padded buffer
    const int64_t pitch = 80;  // 10 bytes per row (8 uint8 + 2 padding)

    TensorView tv(shape, DataType::u8, data, pitch);
    NNOPS_EXPECT_EQ(tv.pitch(), 80);
    NNOPS_EXPECT_EQ(tv.row_stride_elems(), 80);  // u8: pitch/elem_size = 80/1
    NNOPS_EXPECT_EQ(tv.stride_elems(3), 1);      // innermost: W*C8 dimension
    NNOPS_EXPECT_EQ(tv.stride_elems(2), 80);     // H dimension
    NNOPS_EXPECT_EQ(tv.stride_elems(1), 560);    // C8 dimension: 7 * 80
    NNOPS_EXPECT_EQ(tv.stride_elems(0), 1120);   // N dimension: 2 * 560
}
