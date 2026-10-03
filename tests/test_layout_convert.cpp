/// Unit tests for layout conversion (NCHW <-> NCHWC8, NCDHW <-> NCDHWC8).
/// All conversions use the LayoutConvert operator class exclusively.

#include "nnops/ops/layout_convert.hpp"
#include "nnops/detail/half.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using namespace nnops;
using namespace nnops::backend::cpu;  // for half, float_to_half, half_to_float

namespace {

/// Fill a planar NCHW f32 tensor with deterministic sequential values.
inline void fill_sequential_f32(TensorView& tv) {
    float* p = tv.ptr<float>();
    const int64_t N = tv.shape(0), C = tv.shape(1);
    const int64_t H = tv.shape(2), W = tv.shape(3);
    const int64_t ch = tv.stride_elems(1);
    const int64_t row = tv.row_stride_elems();
    float val = 1.0f;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t hh = 0; hh < H; ++hh) {
                for (int64_t w = 0; w < W; ++w) {
                    p[(n * C + c) * ch + hh * row + w] = val;
                    val += 1.0f;
                }
            }
        }
    }
}

/// Fill a planar NCHW f16 tensor with deterministic sequential values.
inline void fill_sequential_f16(TensorView& tv) {
    const int64_t N = tv.shape(0), C = tv.shape(1);
    const int64_t H = tv.shape(2), W = tv.shape(3);
    const int64_t ch = tv.stride_elems(1);
    const int64_t row = tv.row_stride_elems();
    float val = 1.0f;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t hh = 0; hh < H; ++hh) {
                for (int64_t w = 0; w < W; ++w) {
                    uint16_t* p = tv.ptr<uint16_t>()
                                  + (n * C + c) * ch + hh * row + w;
                    *p = half_to_bits(float_to_half(val));
                    val += 1.0f;
                }
            }
        }
    }
}

/// Fill a planar NCDHW f32 tensor with deterministic sequential values.
inline void fill_sequential_3d_f32(TensorView& tv) {
    float* p = tv.ptr<float>();
    const int64_t N = tv.shape(0), C = tv.shape(1), D = tv.shape(2);
    const int64_t H = tv.shape(3), W = tv.shape(4);
    const int64_t ch_stride = tv.stride_elems(1);
    const int64_t depth_stride = tv.stride_elems(2);
    const int64_t row = tv.row_stride_elems();
    float val = 1.0f;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t d = 0; d < D; ++d) {
                for (int64_t hh = 0; hh < H; ++hh) {
                    for (int64_t w = 0; w < W; ++w) {
                        p[(n * C + c) * ch_stride + d * depth_stride + hh * row + w] = val;
                        val += 1.0f;
                    }
                }
            }
        }
    }
}

/// Compare two planar tensors element-by-element (bit-exact).
/// Both tensors must be densely packed (as from make_planar).
template <typename T>
bool tensors_equal(const TensorView& a, const TensorView& b) {
    if (a.rank() != b.rank()) {
        return false;
    }
    for (int64_t d = 0; d < a.rank(); ++d) {
        if (a.shape(d) != b.shape(d)) {
            return false;
        }
    }
    int64_t total = 1;
    for (int64_t d = 0; d < a.rank(); ++d) {
        total *= a.shape(d);
    }
    const T* ap = a.ptr<T>();
    const T* bp = b.ptr<T>();
    for (int64_t i = 0; i < total; ++i) {
        if (ap[i] != bp[i]) {
            return false;
        }
    }
    return true;
}

/// Build a 4D TensorDesc.
inline TensorDesc make_desc_4d(int64_t N, int64_t C, int64_t H, int64_t W,
                                DataType dtype, TensorLayout layout) {
    TensorDesc desc;
    desc.rank = 4;
    desc.dims.resize(4);
    desc.dims[0] = N;
    desc.dims[1] = C;
    desc.dims[2] = H;
    desc.dims[3] = W;
    desc.dtype = dtype;
    desc.layout = layout;
    return desc;
}

/// Build a 5D TensorDesc.
inline TensorDesc make_desc_5d(int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,
                                DataType dtype, TensorLayout layout) {
    TensorDesc desc;
    desc.rank = 5;
    desc.dims.resize(5);
    desc.dims[0] = N;
    desc.dims[1] = C;
    desc.dims[2] = D;
    desc.dims[3] = H;
    desc.dims[4] = W;
    desc.dtype = dtype;
    desc.layout = layout;
    return desc;
}

}  // anonymous namespace

// ============================================================
// 2D Roundtrip — f32, aligned C8 (divisible by 8)
// ============================================================

NNOPS_TEST(layout_convert_roundtrip_f32) {
    // N=2, C=16, H=3, W=4
    auto input_desc = make_desc_4d(2, 16, 3, 4, DataType::f32, TensorLayout::NCHW);

    std::vector<float> input_data(input_desc.numel());
    auto input = nnops::test::make_planar(input_desc, input_data.data());
    fill_sequential_f32(input);

    // Pack: NCHW -> NCHWC8
    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
    const TensorDesc pack_in_descs[] = { input_desc };
    auto packed_descs = pack_op->getOutputTensorDesc(pack_in_descs);
    NNOPS_EXPECT_EQ(packed_descs.size(), 1u);
    NNOPS_EXPECT_EQ(packed_descs[0].layout, TensorLayout::NCHWC8);

    std::vector<float> packed_data(packed_descs[0].storage_bytes() / sizeof(float));
    auto packed = nnops::test::make_packed(packed_descs[0], packed_data.data());

    const TensorView pack_ins[] = { input };
    pack_op->compute(packed, pack_ins);

    // Unpack: NCHWC8 -> NCHW
    auto unpack_op = LayoutConvert::create(TensorLayout::NCHW);
    const TensorDesc unpack_in_descs[] = { packed_descs[0] };
    auto unpacked_descs = unpack_op->getOutputTensorDesc(unpack_in_descs);
    NNOPS_EXPECT_EQ(unpacked_descs.size(), 1u);
    NNOPS_EXPECT_EQ(unpacked_descs[0].layout, TensorLayout::NCHW);

    std::vector<float> roundtrip_data(unpacked_descs[0].numel());
    auto roundtrip = nnops::test::make_planar(unpacked_descs[0], roundtrip_data.data());

    const TensorView unpack_ins[] = { packed };
    unpack_op->compute(roundtrip, unpack_ins);

    NNOPS_EXPECT_TRUE(tensors_equal<float>(input, roundtrip));
}

// ============================================================
// 2D Roundtrip — f32, partial C8 (C not divisible by 8)
// ============================================================

NNOPS_TEST(layout_convert_roundtrip_partial_c8_f32) {
    // C=20 - last C8 block has 4 valid channels + 4 zero-padded
    auto input_desc = make_desc_4d(1, 20, 2, 3, DataType::f32, TensorLayout::NCHW);

    std::vector<float> input_data(input_desc.numel());
    auto input = nnops::test::make_planar(input_desc, input_data.data());
    fill_sequential_f32(input);

    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
    const TensorDesc pack_in_descs[] = { input_desc };
    auto packed_descs = pack_op->getOutputTensorDesc(pack_in_descs);

    std::vector<float> packed_data(packed_descs[0].storage_bytes() / sizeof(float));
    auto packed = nnops::test::make_packed(packed_descs[0], packed_data.data());

    const TensorView pack_ins[] = { input };
    pack_op->compute(packed, pack_ins);

    auto unpack_op = LayoutConvert::create(TensorLayout::NCHW);
    const TensorDesc unpack_in_descs[] = { packed_descs[0] };
    auto unpacked_descs = unpack_op->getOutputTensorDesc(unpack_in_descs);

    std::vector<float> roundtrip_data(unpacked_descs[0].numel());
    auto roundtrip = nnops::test::make_planar(unpacked_descs[0], roundtrip_data.data());

    const TensorView unpack_ins[] = { packed };
    unpack_op->compute(roundtrip, unpack_ins);

    NNOPS_EXPECT_TRUE(tensors_equal<float>(input, roundtrip));
}

// ============================================================
// 2D Roundtrip — f32, single channel
// ============================================================

NNOPS_TEST(layout_convert_roundtrip_single_channel_f32) {
    // C=1 - highly partial C8 block
    auto input_desc = make_desc_4d(1, 1, 4, 5, DataType::f32, TensorLayout::NCHW);

    std::vector<float> input_data(input_desc.numel());
    auto input = nnops::test::make_planar(input_desc, input_data.data());
    fill_sequential_f32(input);

    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
    const TensorDesc pack_in_descs[] = { input_desc };
    auto packed_descs = pack_op->getOutputTensorDesc(pack_in_descs);

    std::vector<float> packed_data(packed_descs[0].storage_bytes() / sizeof(float));
    auto packed = nnops::test::make_packed(packed_descs[0], packed_data.data());

    const TensorView pack_ins[] = { input };
    pack_op->compute(packed, pack_ins);

    auto unpack_op = LayoutConvert::create(TensorLayout::NCHW);
    const TensorDesc unpack_in_descs[] = { packed_descs[0] };
    auto unpacked_descs = unpack_op->getOutputTensorDesc(unpack_in_descs);

    std::vector<float> roundtrip_data(unpacked_descs[0].numel());
    auto roundtrip = nnops::test::make_planar(unpacked_descs[0], roundtrip_data.data());

    const TensorView unpack_ins[] = { packed };
    unpack_op->compute(roundtrip, unpack_ins);

    NNOPS_EXPECT_TRUE(tensors_equal<float>(input, roundtrip));
}

// ============================================================
// 2D Roundtrip — f16
// ============================================================

NNOPS_TEST(layout_convert_roundtrip_f16) {
    auto input_desc = make_desc_4d(2, 16, 3, 4, DataType::f16, TensorLayout::NCHW);

    std::vector<uint16_t> input_data(input_desc.numel());
    auto input = nnops::test::make_planar(input_desc, input_data.data());
    fill_sequential_f16(input);

    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8);
    const TensorDesc pack_in_descs[] = { input_desc };
    auto packed_descs = pack_op->getOutputTensorDesc(pack_in_descs);

    std::vector<uint16_t> packed_data(packed_descs[0].storage_bytes() / sizeof(uint16_t));
    auto packed = nnops::test::make_packed(packed_descs[0], packed_data.data());

    const TensorView pack_ins[] = { input };
    pack_op->compute(packed, pack_ins);

    auto unpack_op = LayoutConvert::create(TensorLayout::NCHW);
    const TensorDesc unpack_in_descs[] = { packed_descs[0] };
    auto unpacked_descs = unpack_op->getOutputTensorDesc(unpack_in_descs);

    std::vector<uint16_t> roundtrip_data(unpacked_descs[0].numel());
    auto roundtrip = nnops::test::make_planar(unpacked_descs[0], roundtrip_data.data());

    const TensorView unpack_ins[] = { packed };
    unpack_op->compute(roundtrip, unpack_ins);

    NNOPS_EXPECT_TRUE(tensors_equal<uint16_t>(input, roundtrip));
}

// ============================================================
// 3D Roundtrip — f32, NCDHW <-> NCDHWC8
// ============================================================

NNOPS_TEST(layout_convert_roundtrip_3d_f32) {
    // N=1, C=16, D=2, H=3, W=4
    auto input_desc = make_desc_5d(1, 16, 2, 3, 4, DataType::f32, TensorLayout::NCDHW);

    std::vector<float> input_data(input_desc.numel());
    auto input = nnops::test::make_planar(input_desc, input_data.data());
    fill_sequential_3d_f32(input);

    // Pack: NCDHW -> NCDHWC8
    auto pack_op = LayoutConvert::create(TensorLayout::NCDHWC8);
    const TensorDesc pack_in_descs[] = { input_desc };
    auto packed_descs = pack_op->getOutputTensorDesc(pack_in_descs);
    NNOPS_EXPECT_EQ(packed_descs.size(), 1u);
    NNOPS_EXPECT_EQ(packed_descs[0].layout, TensorLayout::NCDHWC8);

    std::vector<float> packed_data(packed_descs[0].storage_bytes() / sizeof(float));
    auto packed = nnops::test::make_packed(packed_descs[0], packed_data.data());

    const TensorView pack_ins[] = { input };
    pack_op->compute(packed, pack_ins);

    // Unpack: NCDHWC8 -> NCDHW
    auto unpack_op = LayoutConvert::create(TensorLayout::NCDHW);
    const TensorDesc unpack_in_descs[] = { packed_descs[0] };
    auto unpacked_descs = unpack_op->getOutputTensorDesc(unpack_in_descs);
    NNOPS_EXPECT_EQ(unpacked_descs.size(), 1u);
    NNOPS_EXPECT_EQ(unpacked_descs[0].layout, TensorLayout::NCDHW);

    std::vector<float> roundtrip_data(unpacked_descs[0].numel());
    auto roundtrip = nnops::test::make_planar(unpacked_descs[0], roundtrip_data.data());

    const TensorView unpack_ins[] = { packed };
    unpack_op->compute(roundtrip, unpack_ins);

    NNOPS_EXPECT_TRUE(tensors_equal<float>(input, roundtrip));
}

// ============================================================
// Storage size — TensorDesc::storage_bytes()
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
    desc.layout = TensorLayout::NCHWC8;

    // row_bytes = W * pack * elem_size = 5 * 8 * 4 = 160
    // aligned_row = align_up(160, 32) = 160 (already aligned)
    // C8 = (16+7)/8 = 2
    // storage = N * C8 * H * aligned_row = 2 * 2 * 3 * 160 = 1920
    size_t bytes = desc.storage_bytes(32);
    NNOPS_EXPECT_EQ(bytes, 1920u);
}

NNOPS_TEST(storage_bytes_needs_alignment) {
    TensorDesc desc;
    desc.rank = 4;
    desc.dims.resize(4);
    desc.dims[0] = 1;
    desc.dims[1] = 8;
    desc.dims[2] = 2;
    desc.dims[3] = 7;  // W=7
    desc.dtype = DataType::f32;
    desc.layout = TensorLayout::NCHWC8;

    // row_bytes = 7 * 8 * 4 = 224, aligned to 32: 224 (already aligned)
    // storage = 1 * 1 * 2 * 224 = 448
    size_t bytes32 = desc.storage_bytes(32);
    NNOPS_EXPECT_EQ(bytes32, 448u);

    // With 64-byte alignment: aligned_row = align_up(224, 64) = 256
    // storage = 1 * 1 * 2 * 256 = 512
    size_t bytes64 = desc.storage_bytes(64);
    NNOPS_EXPECT_EQ(bytes64, 512u);
}

// ============================================================
// getOutputTensorDesc — shape and layout inference
// ============================================================

NNOPS_TEST(get_output_tensor_desc_pack_2d) {
    auto input_desc = make_desc_4d(2, 16, 3, 4, DataType::f32, TensorLayout::NCHW);

    auto op = LayoutConvert::create(TensorLayout::NCHWC8);
    const TensorDesc desc_arr[] = { input_desc };
    auto outputs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(outputs.size(), 1u);
    NNOPS_EXPECT_EQ(outputs[0].rank, 4);
    NNOPS_EXPECT_EQ(outputs[0].layout, TensorLayout::NCHWC8);
    NNOPS_EXPECT_EQ(outputs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(outputs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(outputs[0].dims[1], 16);
    NNOPS_EXPECT_EQ(outputs[0].dims[2], 3);
    NNOPS_EXPECT_EQ(outputs[0].dims[3], 4);
}

NNOPS_TEST(get_output_tensor_desc_unpack_2d) {
    auto input_desc = make_desc_4d(1, 24, 5, 6, DataType::f16, TensorLayout::NCHWC8);

    auto op = LayoutConvert::create(TensorLayout::NCHW);
    const TensorDesc desc_arr[] = { input_desc };
    auto outputs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(outputs.size(), 1u);
    NNOPS_EXPECT_EQ(outputs[0].rank, 4);
    NNOPS_EXPECT_EQ(outputs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(outputs[0].dtype, DataType::f16);
    NNOPS_EXPECT_EQ(outputs[0].dims[0], 1);
    NNOPS_EXPECT_EQ(outputs[0].dims[1], 24);
    NNOPS_EXPECT_EQ(outputs[0].dims[2], 5);
    NNOPS_EXPECT_EQ(outputs[0].dims[3], 6);
}

NNOPS_TEST(get_output_tensor_desc_pack_3d) {
    auto input_desc = make_desc_5d(3, 32, 4, 5, 6, DataType::f32, TensorLayout::NCDHW);

    auto op = LayoutConvert::create(TensorLayout::NCDHWC8);
    const TensorDesc desc_arr[] = { input_desc };
    auto outputs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(outputs.size(), 1u);
    NNOPS_EXPECT_EQ(outputs[0].rank, 5);
    NNOPS_EXPECT_EQ(outputs[0].layout, TensorLayout::NCDHWC8);
    NNOPS_EXPECT_EQ(outputs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(outputs[0].dims[0], 3);
    NNOPS_EXPECT_EQ(outputs[0].dims[1], 32);
    NNOPS_EXPECT_EQ(outputs[0].dims[2], 4);
    NNOPS_EXPECT_EQ(outputs[0].dims[3], 5);
    NNOPS_EXPECT_EQ(outputs[0].dims[4], 6);
}

// ============================================================
// OpType and Backend
// ============================================================

NNOPS_TEST(layout_convert_op_type_and_backend) {
    auto op = LayoutConvert::create(TensorLayout::NCHWC8);
    NNOPS_EXPECT_EQ(static_cast<int>(op->getOpType()), static_cast<int>(OpType::LayoutConvert));
    NNOPS_EXPECT_EQ(static_cast<int>(op->getBackend()), static_cast<int>(Backend::CPU));
}
