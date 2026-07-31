/// Unit tests for layout conversion (NCHW <-> NCHWC8 packing/unpacking).

#include "backend/cpu/layout_convert.hpp"
#include "nnops/ops/layout_convert.hpp"
#include "nnops/detail/half.hpp"
#include "common/test_harness.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using namespace nnops;
using namespace nnops::backend::cpu;  // for half, float_to_half, half_to_float

namespace {

/// Fill a NCHW f32 tensor with deterministic sequential values.
inline void fill_sequential_f32(TensorView& tv) {
    float* p = tv.ptr<float>();
    const int64_t N = tv.shape(0), C = tv.shape(1);
    const int64_t H = tv.shape(2), W = tv.shape(3);
    const int64_t ch = tv.stride_elems(1);
    const int64_t row = tv.row_stride_elems();
    float val = 1.0f;
    for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < C; ++c)
            for (int64_t hh = 0; hh < H; ++hh)
                for (int64_t w = 0; w < W; ++w) {
                    p[(n * C + c) * ch + hh * row + w] = val;
                    val += 1.0f;
                }
}

/// Fill a NCHW f16 tensor with deterministic sequential values.
inline void fill_sequential_f16(TensorView& tv) {
    const int64_t N = tv.shape(0), C = tv.shape(1);
    const int64_t H = tv.shape(2), W = tv.shape(3);
    const int64_t ch = tv.stride_elems(1);
    const int64_t row = tv.row_stride_elems();
    float val = 1.0f;
    for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < C; ++c)
            for (int64_t hh = 0; hh < H; ++hh)
                for (int64_t w = 0; w < W; ++w) {
                    uint16_t* p = tv.ptr<uint16_t>()
                                  + (n * C + c) * ch + hh * row + w;
                    half h = float_to_half(val);
                    *p = h.bits;
                    val += 1.0f;
                }
}

/// Helper: compare two NCHW tensors element-by-element (bit-exact).
template <typename T>
bool tensors_equal(TensorView& a, TensorView& b) {
    if (a.rank() != b.rank()) return false;
    for (int64_t d = 0; d < a.rank(); ++d) {
        if (a.shape(d) != b.shape(d)) return false;
    }
    const int64_t N = a.shape(0), C = a.shape(1);
    const int64_t H = a.shape(2), W = a.shape(3);
    const int64_t a_ch = a.stride_elems(1), a_row = a.row_stride_elems();
    const int64_t b_ch = b.stride_elems(1), b_row = b.row_stride_elems();
    T* ap = a.ptr<T>();
    T* bp = b.ptr<T>();
    for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < C; ++c)
            for (int64_t hh = 0; hh < H; ++hh)
                for (int64_t w = 0; w < W; ++w) {
                    int64_t ai = n * C * a_ch + c * a_ch + hh * a_row + w;
                    int64_t bi = n * C * b_ch + c * b_ch + hh * b_row + w;
                    if (ap[ai] != bp[bi]) return false;
                }
    return true;
}

}  // anonymous namespace

// ============================================================
// 2D Pack / Unpack — f32
// ============================================================

NNOPS_TEST(pack_unpack_roundtrip_f32) {
    // N=2, C=16, H=3, W=4
    const int64_t shape[] = {2, 16, 3, 4};
    const int64_t C8 = (16 + 7) / 8;

    std::vector<float> nchw_data(2 * 16 * 3 * 4);
    std::vector<float> packed_data(2 * C8 * 3 * 4 * 8);
    std::vector<float> roundtrip_data(2 * 16 * 3 * 4);

    TensorView src(shape, DataType::f32, nchw_data.data(), TensorLayout::NCHW);
    fill_sequential_f32(src);

    // Pack
    const int64_t pitch = 4 * 8 * 4;
    TensorView packed(shape, DataType::f32, packed_data.data(), pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(src, packed);

    // Unpack
    TensorView dst(shape, DataType::f32, roundtrip_data.data(), TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(packed, dst);

    NNOPS_EXPECT_TRUE(tensors_equal<float>(src, dst));
}

NNOPS_TEST(pack_unpack_partial_c8_f32) {
    // C=20, not divisible by 8 → last C8 block has 4 valid channels + 4 zero-padded
    const int64_t shape[] = {1, 20, 2, 3};
    const int64_t C8 = (20 + 7) / 8;  // 3

    std::vector<float> nchw_data(1 * 20 * 2 * 3);
    std::vector<float> packed_data(1 * C8 * 2 * 3 * 8);
    std::vector<float> roundtrip_data(1 * 20 * 2 * 3);

    TensorView src(shape, DataType::f32, nchw_data.data(), TensorLayout::NCHW);
    fill_sequential_f32(src);

    const int64_t pitch = 3 * 8 * 4;
    TensorView packed(shape, DataType::f32, packed_data.data(), pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(src, packed);

    TensorView dst(shape, DataType::f32, roundtrip_data.data(), TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(packed, dst);

    NNOPS_EXPECT_TRUE(tensors_equal<float>(src, dst));
}

NNOPS_TEST(pack_unpack_single_channel_f32) {
    // C=1 — single channel, highly partial C8 block
    const int64_t shape[] = {1, 1, 4, 5};
    const int64_t C8 = 1;

    std::vector<float> nchw_data(1 * 1 * 4 * 5);
    std::vector<float> packed_data(1 * C8 * 4 * 5 * 8);
    std::vector<float> roundtrip_data(1 * 1 * 4 * 5);

    TensorView src(shape, DataType::f32, nchw_data.data(), TensorLayout::NCHW);
    fill_sequential_f32(src);

    const int64_t pitch = 5 * 8 * 4;
    TensorView packed(shape, DataType::f32, packed_data.data(), pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(src, packed);

    TensorView dst(shape, DataType::f32, roundtrip_data.data(), TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(packed, dst);

    NNOPS_EXPECT_TRUE(tensors_equal<float>(src, dst));
}

// ============================================================
// 2D Pack / Unpack — f16
// ============================================================

NNOPS_TEST(pack_unpack_roundtrip_f16) {
    const int64_t shape[] = {2, 16, 3, 4};
    const int64_t C8 = (16 + 7) / 8;

    std::vector<uint16_t> nchw_data(2 * 16 * 3 * 4);
    std::vector<uint16_t> packed_data(2 * C8 * 3 * 4 * 8);
    std::vector<uint16_t> roundtrip_data(2 * 16 * 3 * 4);

    TensorView src(shape, DataType::f16, nchw_data.data(), TensorLayout::NCHW);
    fill_sequential_f16(src);

    const int64_t pitch = 4 * 8 * 2;  // W * C8 * sizeof(half)
    TensorView packed(shape, DataType::f16, packed_data.data(), pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(src, packed);

    TensorView dst(shape, DataType::f16, roundtrip_data.data(), TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(packed, dst);

    NNOPS_EXPECT_TRUE(tensors_equal<uint16_t>(src, dst));
}

// ============================================================
// 3D Pack / Unpack — f32
// ============================================================

NNOPS_TEST(pack_unpack_3d_roundtrip_f32) {
    // N=1, C=16, D=2, H=3, W=4
    const int64_t shape[] = {1, 16, 2, 3, 4};
    const int64_t C8 = (16 + 7) / 8;

    // NCHW-style for 3D: N * C * D * H * W
    std::vector<float> ncdhw_data(1 * 16 * 2 * 3 * 4);
    std::vector<float> packed_data(1 * C8 * 2 * 3 * 4 * 8);
    std::vector<float> roundtrip_data(1 * 16 * 2 * 3 * 4);

    TensorView src(shape, DataType::f32, ncdhw_data.data(), TensorLayout::NCDHW);
    {
        // Fill with sequential values
        float* p = src.ptr<float>();
        float val = 1.0f;
        for (int64_t n = 0; n < 1; ++n)
            for (int64_t c = 0; c < 16; ++c)
                for (int64_t d = 0; d < 2; ++d)
                    for (int64_t h = 0; h < 3; ++h)
                        for (int64_t w = 0; w < 4; ++w) {
                            int64_t idx = n * 16 * 2 * 3 * 4
                                        + c * 2 * 3 * 4
                                        + d * 3 * 4
                                        + h * 4
                                        + w;
                            p[idx] = val;
                            val += 1.0f;
                        }
    }

    const int64_t pitch = 4 * 8 * 4;
    TensorView packed(shape, DataType::f32, packed_data.data(), pitch, TensorLayout::NCDHWC8);
    pack_ncdhw_to_ncdhwc8(src, packed);

    TensorView dst(shape, DataType::f32, roundtrip_data.data(), TensorLayout::NCDHW);
    unpack_ncdhwc8_to_ncdhw(packed, dst);

    // Compare element-by-element
    const float* sp = src.ptr<float>();
    const float* dp = dst.ptr<float>();
    const int64_t total = 1 * 16 * 2 * 3 * 4;
    for (int64_t i = 0; i < total; ++i) {
        if (sp[i] != dp[i]) {
            throw std::runtime_error("3D roundtrip mismatch");
        }
    }
}

// ============================================================
// Storage size helper
// ============================================================

NNOPS_TEST(storage_bytes_aligned) {
    TensorDesc desc;
    desc.rank = 4;
    desc.dims.resize(4);
    desc.dims[0] = 2;   // N
    desc.dims[1] = 16;  // C
    desc.dims[2] = 3;   // H
    desc.dims[3] = 5;   // W
    desc.dtype = DataType::f32;

    // row = W * 8 * 4 = 160 bytes
    // aligned to 32: 160 (already aligned)
    // total = N * C8 * H * aligned_row
    //       = 2 * 2 * 3 * 160 = 1920
    size_t bytes = nchwc8_storage_bytes(desc, 32);
    NNOPS_EXPECT_EQ(bytes, 1920u);
}

NNOPS_TEST(storage_bytes_needs_alignment) {
    TensorDesc desc;
    desc.rank = 4;
    desc.dims.resize(4);
    desc.dims[0] = 1;
    desc.dims[1] = 8;
    desc.dims[2] = 2;
    desc.dims[3] = 7;  // W=7, row = 7*8*4 = 224, align to 32: 224
    desc.dtype = DataType::f32;

    // row = 7 * 8 * 4 = 224 bytes (already 32-aligned: 224/32=7)
    // total = 1 * 1 * 2 * 224 = 448
    size_t bytes = nchwc8_storage_bytes(desc, 32);
    NNOPS_EXPECT_EQ(bytes, 448u);

    // With 64-byte alignment: row = align_up(224, 64) = 256
    // total = 1 * 1 * 2 * 256 = 512
    size_t bytes64 = nchwc8_storage_bytes(desc, 64);
    NNOPS_EXPECT_EQ(bytes64, 512u);
}

// ============================================================
// LayoutConvert operator — class API
// ============================================================

NNOPS_TEST(layout_convert_op_pack_f32) {
    const int64_t shape[] = {1, 16, 3, 4};
    const int64_t C8 = (16 + 7) / 8;

    std::vector<float> nchw_data(1 * 16 * 3 * 4);
    std::vector<float> packed_data(1 * C8 * 3 * 4 * 8);

    TensorView src(shape, DataType::f32, nchw_data.data(), TensorLayout::NCHW);
    fill_sequential_f32(src);

    const int64_t pitch = 4 * 8 * 4;
    TensorView dst(shape, DataType::f32, packed_data.data(), pitch, TensorLayout::NCHWC8);

    auto op = LayoutConvert::create(TensorLayout::NCHWC8);
    op->compute(dst, {&src, 1});

    // Verify roundtrip
    std::vector<float> roundtrip_data(1 * 16 * 3 * 4);
    TensorView rt(shape, DataType::f32, roundtrip_data.data(), TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(dst, rt);
    NNOPS_EXPECT_TRUE(tensors_equal<float>(src, rt));
}

NNOPS_TEST(layout_convert_op_unpack_f32) {
    const int64_t shape[] = {1, 16, 3, 4};
    const int64_t C8 = (16 + 7) / 8;

    std::vector<float> nchw_data(1 * 16 * 3 * 4);
    std::vector<float> packed_data(1 * C8 * 3 * 4 * 8);

    TensorView src(shape, DataType::f32, nchw_data.data(), TensorLayout::NCHW);
    fill_sequential_f32(src);

    const int64_t pitch = 4 * 8 * 4;
    TensorView packed(shape, DataType::f32, packed_data.data(), pitch, TensorLayout::NCHWC8);
    pack_nchw_to_nchwc8(src, packed);

    std::vector<float> unpacked_data(1 * 16 * 3 * 4);
    TensorView dst(shape, DataType::f32, unpacked_data.data(), TensorLayout::NCHW);

    auto op = LayoutConvert::create(TensorLayout::NCHW);
    op->compute(dst, {&packed, 1});

    NNOPS_EXPECT_TRUE(tensors_equal<float>(src, dst));
}

NNOPS_TEST(layout_convert_op_shape_inference) {
    TensorDesc input;
    input.rank = 4;
    input.dims.resize(4);
    input.dims[0] = 2; input.dims[1] = 16; input.dims[2] = 3; input.dims[3] = 4;
    input.dtype = DataType::f32;
    input.layout = TensorLayout::NCHW;

    auto op = LayoutConvert::create(TensorLayout::NCHWC8);
    auto outputs = op->getOutputTensorDesc({&input, 1});

    NNOPS_EXPECT_EQ(outputs.size(), 1u);
    NNOPS_EXPECT_EQ(outputs[0].rank, 4);
    NNOPS_EXPECT_EQ(outputs[0].layout, TensorLayout::NCHWC8);
    NNOPS_EXPECT_EQ(outputs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(outputs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(outputs[0].dims[1], 16);
    NNOPS_EXPECT_EQ(outputs[0].dims[2], 3);
    NNOPS_EXPECT_EQ(outputs[0].dims[3], 4);
}

// ============================================================
// LayoutConvert functional API
// ============================================================

NNOPS_TEST(layout_convert_functional_pack) {
    const int64_t shape[] = {1, 16, 3, 4};
    const int64_t C8 = (16 + 7) / 8;

    std::vector<float> nchw_data(1 * 16 * 3 * 4);
    std::vector<float> packed_data(1 * C8 * 3 * 4 * 8);

    TensorView src(shape, DataType::f32, nchw_data.data(), TensorLayout::NCHW);
    fill_sequential_f32(src);

    const int64_t pitch = 4 * 8 * 4;
    TensorView dst(shape, DataType::f32, packed_data.data(), pitch, TensorLayout::NCHWC8);

    layout_convert(src, dst, TensorLayout::NCHWC8);

    // Roundtrip verify
    std::vector<float> rt_data(1 * 16 * 3 * 4);
    TensorView rt(shape, DataType::f32, rt_data.data(), TensorLayout::NCHW);
    unpack_nchwc8_to_nchw(dst, rt);
    NNOPS_EXPECT_TRUE(tensors_equal<float>(src, rt));
}

NNOPS_TEST(layout_convert_op_3d_pack_f32) {
    const int64_t shape[] = {1, 16, 2, 3, 4};
    const int64_t C8 = (16 + 7) / 8;

    std::vector<float> ncdhw_data(1 * 16 * 2 * 3 * 4);
    std::vector<float> packed_data(1 * C8 * 2 * 3 * 4 * 8);

    TensorView src(shape, DataType::f32, ncdhw_data.data(), TensorLayout::NCDHW);
    {
        float* p = src.ptr<float>();
        float val = 1.0f;
        for (int64_t n = 0; n < 1; ++n)
            for (int64_t c = 0; c < 16; ++c)
                for (int64_t d = 0; d < 2; ++d)
                    for (int64_t h = 0; h < 3; ++h)
                        for (int64_t w = 0; w < 4; ++w) {
                            int64_t idx = n * 16 * 2 * 3 * 4
                                        + c * 2 * 3 * 4
                                        + d * 3 * 4
                                        + h * 4
                                        + w;
                            p[idx] = val;
                            val += 1.0f;
                        }
    }

    const int64_t pitch = 4 * 8 * 4;
    TensorView dst(shape, DataType::f32, packed_data.data(), pitch, TensorLayout::NCDHWC8);

    auto op = LayoutConvert::create(TensorLayout::NCDHWC8);
    op->compute(dst, {&src, 1});

    // Roundtrip
    std::vector<float> rt_data(1 * 16 * 2 * 3 * 4);
    TensorView rt(shape, DataType::f32, rt_data.data(), TensorLayout::NCDHW);
    unpack_ncdhwc8_to_ncdhw(dst, rt);

    const float* sp = src.ptr<float>();
    const float* dp = rt.ptr<float>();
    const int64_t total = 1 * 16 * 2 * 3 * 4;
    for (int64_t i = 0; i < total; ++i) {
        if (sp[i] != dp[i]) {
            throw std::runtime_error("3D operator roundtrip mismatch");
        }
    }
}

NNOPS_TEST(layout_convert_op_get_optype) {
    auto op = LayoutConvert::create(TensorLayout::NCHWC8);
    NNOPS_EXPECT_EQ(static_cast<int>(op->getOpType()), static_cast<int>(OpType::LayoutConvert));
    NNOPS_EXPECT_EQ(static_cast<int>(op->getBackend()), static_cast<int>(Backend::CPU));
}
