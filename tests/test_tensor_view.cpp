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

// ============================================================
// NCHWC8 layout-aware accessors
// ============================================================

NNOPS_TEST(nchwc8_dense_construction) {
    // Logical: [N=2, C=16, H=3, W=4], NCHWC8
    // Physical row: W * 8 = 32 elements, pitch = 32 * 4 = 128 bytes
    const int64_t shape[] = {2, 16, 3, 4};
    float data[2 * 2 * 3 * 4 * 8] = {};  // N * C8 * H * W * 8
    const int64_t pitch = 4 * 8 * 4;  // W * C8 * sizeof(float)

    TensorView tv(shape, DataType::f32, data, pitch, TensorLayout::NCHWC8);
    NNOPS_EXPECT_EQ(tv.rank(), 4);
    NNOPS_EXPECT_EQ(tv.layout(), TensorLayout::NCHWC8);
    NNOPS_EXPECT_EQ(tv.channel_pack_size(), 8);
    NNOPS_EXPECT_EQ(tv.num_channel_blocks(), 2);  // ceil(16/8)
    NNOPS_EXPECT_EQ(tv.pitch(), 128);
    NNOPS_EXPECT_EQ(tv.row_stride_elems(), 32);   // 128/4 = 32 = W * 8
    // stride_elems: innermost = pack_size = 8 (NOT 1)
    NNOPS_EXPECT_EQ(tv.stride_elems(3), 8);
    NNOPS_EXPECT_EQ(tv.stride_elems(2), 32);      // H: row_stride = 32
    NNOPS_EXPECT_EQ(tv.stride_elems(1), 96);      // C: H * row_stride = 3 * 32
    NNOPS_EXPECT_EQ(tv.stride_elems(0), 192);     // N: C8 * H * row_stride = 2 * 96
    NNOPS_EXPECT_EQ(tv.numel(), 2 * 16 * 3 * 4);  // logical numel
}

NNOPS_TEST(nchwc8_partial_channel_block) {
    // C=20, not divisible by 8 → 3 blocks (2 full + 1 partial)
    const int64_t shape[] = {1, 20, 2, 3};
    float data[1 * 3 * 2 * 3 * 8] = {};
    const int64_t pitch = 3 * 8 * 4;  // W * C8 * sizeof(float)

    TensorView tv(shape, DataType::f32, data, pitch, TensorLayout::NCHWC8);
    NNOPS_EXPECT_EQ(tv.channel_pack_size(), 8);
    NNOPS_EXPECT_EQ(tv.num_channel_blocks(), 3);  // ceil(20/8) = 3
    NNOPS_EXPECT_EQ(tv.channel_block_stride_elems(), 2 * 3 * 8);  // H * W * 8 = 48
    NNOPS_EXPECT_EQ(tv.total_rows(), 1 * 3 * 2);  // N * C8 * H = 6
}

NNOPS_TEST(nchwc8_total_rows) {
    // N=2, C=8, H=4, W=5 → NCHWC8: rows = N * C8 * H = 2 * 1 * 4 = 8
    const int64_t shape[] = {2, 8, 4, 5};
    float data[2 * 1 * 4 * 5 * 8] = {};
    const int64_t pitch = 5 * 8 * 4;
    TensorView tv(shape, DataType::f32, data, pitch, TensorLayout::NCHWC8);
    NNOPS_EXPECT_EQ(tv.total_rows(), 2 * 1 * 4);
    // Compare with NCHW: rows = N * C * H = 2 * 8 * 4 = 64
    TensorView tv_nchw(shape, DataType::f32, data, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(tv_nchw.total_rows(), 2 * 8 * 4);
}

NNOPS_TEST(nchwc8_pitch_validation) {
    // pitch must be >= W * pack_size * elem_size
    const int64_t shape[] = {1, 8, 2, 3};
    std::vector<float> data(1 * 1 * 2 * 3 * 8);
    // Valid: pitch = W * 8 * 4 = 96
    const int64_t valid_pitch = 3 * 8 * static_cast<int64_t>(sizeof(float));
    TensorView tv(shape, DataType::f32, data.data(), valid_pitch, TensorLayout::NCHWC8);
    NNOPS_EXPECT_EQ(tv.pitch(), 96);
}

NNOPS_TEST(nchwc32_dense_construction) {
    const int64_t shape[] = {1, 64, 2, 3};
    float data[1 * 2 * 2 * 3 * 32] = {};
    const int64_t pitch = 3 * 32 * 4;
    TensorView tv(shape, DataType::f32, data, pitch, TensorLayout::NCHWC32);
    NNOPS_EXPECT_EQ(tv.channel_pack_size(), 32);
    NNOPS_EXPECT_EQ(tv.num_channel_blocks(), 2);  // ceil(64/32)
    NNOPS_EXPECT_EQ(tv.stride_elems(3), 32);      // innermost = pack_size
}

// ============================================================
// NCDHWC8 (3D) accessors
// ============================================================

NNOPS_TEST(ncdhwc8_dense_construction) {
    // Logical: [N=1, C=16, D=2, H=3, W=4]
    const int64_t shape[] = {1, 16, 2, 3, 4};
    float data[1 * 2 * 2 * 3 * 4 * 8] = {};
    const int64_t pitch = 4 * 8 * 4;

    TensorView tv(shape, DataType::f32, data, pitch, TensorLayout::NCDHWC8);
    NNOPS_EXPECT_EQ(tv.rank(), 5);
    NNOPS_EXPECT_EQ(tv.layout(), TensorLayout::NCDHWC8);
    NNOPS_EXPECT_EQ(tv.channel_pack_size(), 8);
    NNOPS_EXPECT_EQ(tv.num_channel_blocks(), 2);
    NNOPS_EXPECT_EQ(tv.row_stride_elems(), 32);
    NNOPS_EXPECT_EQ(tv.stride_elems(4), 8);       // W: pack_size
    NNOPS_EXPECT_EQ(tv.stride_elems(3), 32);      // H: row_stride
    NNOPS_EXPECT_EQ(tv.stride_elems(2), 96);      // D: H * row_stride = 3 * 32
    NNOPS_EXPECT_EQ(tv.channel_block_stride_elems(), 2 * 3 * 32);  // D * H * W * 8
    NNOPS_EXPECT_EQ(tv.total_rows(), 1 * 2 * 2 * 3);  // N * C8 * D * H
}

// ============================================================
// TensorDesc numel/nbytes
// ============================================================

NNOPS_TEST(tensor_desc_numel) {
    TensorDesc d;
    d.rank = 3;
    d.dims.resize(3);
    d.dims[0] = 2; d.dims[1] = 3; d.dims[2] = 4;
    d.dtype = DataType::f32;
    NNOPS_EXPECT_EQ(d.numel(), 24);
    NNOPS_EXPECT_EQ(static_cast<int64_t>(d.nbytes()), 24 * 4);
}

NNOPS_TEST(tensor_desc_row_stride_elems) {
    // Default aggregate: 0 = unknown → compact.
    TensorDesc d0;
    NNOPS_EXPECT_EQ(d0.row_stride_elems, 0);

    // Compact 2D view: row stride == last dim (W).
    const int64_t shape2d[] = {5, 7};
    float data2d[5 * 7] = {};
    TensorView compact(shape2d, DataType::f32, data2d);
    NNOPS_EXPECT_EQ(compact.desc().row_stride_elems, 7);

    // Padded 2D view (explicit pitch): row stride == pitch / elem_size.
    const int64_t pitch = 12 * 4;  // 12 floats per row, padded from 7
    float data_pad[5 * 12] = {};
    TensorView padded(shape2d, DataType::f32, data_pad, pitch);
    NNOPS_EXPECT_EQ(padded.desc().row_stride_elems, 12);

    // rank < 2: desc leaves row_stride_elems at 0.
    const int64_t shape1d[] = {9};
    float data1d[9] = {};
    TensorView vec(shape1d, DataType::f32, data1d);
    NNOPS_EXPECT_EQ(vec.desc().row_stride_elems, 0);
}
