/// @file test_layout_convert.cpp
/// @brief LayoutConvert tests — NCHW <-> NHWC (2D) and NCDHW <-> NDHWC (3D).
///
/// The round-trip cases drive the operator in both directions and check the
/// result bit-exact against the input in logical order: f32, f16, a single
/// channel, an odd C, and 3D. One case pins the physical placement of a value
/// in a channel-last row. The rest cover the shape/layout inference done by
/// getOutputTensorDesc(), the NHWC row pitch in TensorDesc::storage_bytes(),
/// the rejections, and the op's OpType/Backend.

#include "nnops/ops/layout_convert.hpp"
#include "nnops/detail/half.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using namespace nnops;
using namespace nnops::backend::cpu;  // for float_to_half / half_to_bits

namespace {

/// Physical element offset of logical (i0, i1, ...) — works for any layout.
inline int64_t logical_offset(const TensorView& tv, std::span<const int64_t> idx) {
    int64_t off = 0;
    for (int64_t d = 0; d < tv.rank(); ++d) {
        off += idx[static_cast<size_t>(d)] * tv.stride_elems(d);
    }
    return off;
}

/// Fill in logical order with 1, 2, 3, ... so a misplaced element is obvious.
template <typename T>
void fill_sequential(TensorView& tv) {
    const int64_t rank = tv.rank();
    int64_t idx[8] = {};
    const int64_t total = tv.numel();
    float val = 1.0f;
    for (int64_t flat = 0; flat < total; ++flat) {
        const int64_t off = logical_offset(tv, std::span<const int64_t>(idx, static_cast<size_t>(rank)));
        if constexpr (sizeof(T) == 2) {
            tv.ptr<uint16_t>()[off] = half_to_bits(float_to_half(val));
        } else {
            tv.ptr<T>()[off] = static_cast<T>(val);
        }
        val += 1.0f;
        for (int64_t d = rank - 1; d >= 0; --d) {
            if (++idx[d] < tv.shape(d)) {
                break;
            }
            idx[d] = 0;
        }
    }
}

/// Bit-exact compare in logical order; both views may carry any pitch/layout.
template <typename T>
bool same_logical(const TensorView& a, const TensorView& b) {
    if (a.rank() != b.rank()) {
        return false;
    }
    for (int64_t d = 0; d < a.rank(); ++d) {
        if (a.shape(d) != b.shape(d)) {
            return false;
        }
    }
    const int64_t rank = a.rank();
    int64_t idx[8] = {};
    const int64_t total = a.numel();
    for (int64_t flat = 0; flat < total; ++flat) {
        const auto span = std::span<const int64_t>(idx, static_cast<size_t>(rank));
        if (a.ptr<T>()[logical_offset(a, span)] != b.ptr<T>()[logical_offset(b, span)]) {
            return false;
        }
        for (int64_t d = rank - 1; d >= 0; --d) {
            if (++idx[d] < a.shape(d)) {
                break;
            }
            idx[d] = 0;
        }
    }
    return true;
}

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

/// Convert planar -> channel-last -> planar and compare against the input.
template <typename T>
void roundtrip(const TensorDesc& input_desc) {
    std::vector<T> input_data(static_cast<size_t>(input_desc.numel()));
    auto input = nnops::test::make_planar(input_desc, input_data.data());
    fill_sequential<T>(input);

    const TensorLayout cl_layout = (input_desc.rank == 5)
        ? TensorLayout::NDHWC : TensorLayout::NHWC;
    const TensorLayout planar_layout = (input_desc.rank == 5)
        ? TensorLayout::NCDHW : TensorLayout::NCHW;

    auto to_cl = LayoutConvert::create(cl_layout);
    const TensorDesc cl_in_descs[] = { input_desc };
    auto cl_descs = to_cl->getOutputTensorDesc(cl_in_descs);
    NNOPS_EXPECT_EQ(cl_descs.size(), 1u);
    NNOPS_EXPECT_EQ(cl_descs[0].layout, cl_layout);

    std::vector<T> cl_data(cl_descs[0].storage_bytes() / sizeof(T));
    auto cl = nnops::test::make_packed(cl_descs[0], cl_data.data());
    const TensorView cl_ins[] = { input };
    NNOPS_EXPECT_TRUE(to_cl->compute(cl, cl_ins) == Status::ok);

    auto to_planar = LayoutConvert::create(planar_layout);
    const TensorDesc back_in_descs[] = { cl_descs[0] };
    auto back_descs = to_planar->getOutputTensorDesc(back_in_descs);
    NNOPS_EXPECT_EQ(back_descs.size(), 1u);
    NNOPS_EXPECT_EQ(back_descs[0].layout, planar_layout);

    std::vector<T> back_data(static_cast<size_t>(back_descs[0].numel()));
    auto back = nnops::test::make_planar(back_descs[0], back_data.data());
    const TensorView back_ins[] = { cl };
    NNOPS_EXPECT_TRUE(to_planar->compute(back, back_ins) == Status::ok);

    NNOPS_EXPECT_TRUE(same_logical<T>(input, back));
}

}  // anonymous namespace

// ============================================================
// 2D round-trips
// ============================================================

NNOPS_TEST(layout_convert_roundtrip_f32) {
    roundtrip<float>(make_desc_4d(2, 16, 3, 4, DataType::f32, TensorLayout::NCHW));
}

NNOPS_TEST(layout_convert_roundtrip_odd_c_f32) {
    roundtrip<float>(make_desc_4d(1, 20, 2, 3, DataType::f32, TensorLayout::NCHW));
}

NNOPS_TEST(layout_convert_roundtrip_single_channel_f32) {
    roundtrip<float>(make_desc_4d(1, 1, 4, 5, DataType::f32, TensorLayout::NCHW));
}

NNOPS_TEST(layout_convert_roundtrip_f16) {
    roundtrip<uint16_t>(make_desc_4d(2, 16, 3, 4, DataType::f16, TensorLayout::NCHW));
}

// ============================================================
// 3D round-trip
// ============================================================

NNOPS_TEST(layout_convert_roundtrip_3d_f32) {
    roundtrip<float>(make_desc_5d(1, 16, 2, 3, 4, DataType::f32, TensorLayout::NCDHW));
}

NNOPS_TEST(layout_convert_roundtrip_3d_odd_c_f32) {
    roundtrip<float>(make_desc_5d(2, 5, 2, 3, 3, DataType::f32, TensorLayout::NCDHW));
}

// ============================================================
// Physical placement inside a channel-last row
// ============================================================

NNOPS_TEST(layout_convert_nhwc_element_placement) {
    auto input_desc = make_desc_4d(1, 3, 2, 2, DataType::f32, TensorLayout::NCHW);

    std::vector<float> input_data(static_cast<size_t>(input_desc.numel()));
    auto input = nnops::test::make_planar(input_desc, input_data.data());
    fill_sequential<float>(input);  // 1..12 in logical order

    auto to_cl = LayoutConvert::create(TensorLayout::NHWC);
    const TensorDesc cl_in_descs[] = { input_desc };
    auto cl_descs = to_cl->getOutputTensorDesc(cl_in_descs);

    std::vector<float> cl_data(cl_descs[0].storage_bytes() / sizeof(float));
    auto cl = nnops::test::make_packed(cl_descs[0], cl_data.data());
    const TensorView cl_ins[] = { input };
    to_cl->compute(cl, cl_ins);

    // A pixel holds C channels: (h=0,w=0) -> values 1,5,9 (one per channel),
    // (h=0,w=1) -> 2,6,10.
    const float* row0 = cl.ptr<float>();
    NNOPS_EXPECT_NEAR(row0[0], 1.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(row0[1], 5.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(row0[2], 9.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(row0[3], 2.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(row0[4], 6.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(row0[5], 10.0f, 1e-6f);

    // The second spatial row starts at one pitch: values 3,7,11 / 4,8,12.
    const float* row1 = cl.ptr<float>(1);
    NNOPS_EXPECT_NEAR(row1[0], 3.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(row1[1], 7.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(row1[2], 11.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(row1[3], 4.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(row1[4], 8.0f, 1e-6f);
    NNOPS_EXPECT_NEAR(row1[5], 12.0f, 1e-6f);
}

// ============================================================
// Rejections
// ============================================================

NNOPS_TEST(layout_convert_rejects_same_layout) {
    auto desc = make_desc_4d(1, 3, 2, 2, DataType::f32, TensorLayout::NCHW);
    auto op = LayoutConvert::create(TensorLayout::NCHW);
    const TensorDesc ins[] = { desc };
    const TensorDesc outs[] = { desc };
    NNOPS_EXPECT_EQ(op->validate(ins, outs), Status::invalid_argument);
}

NNOPS_TEST(layout_convert_rejects_cross_rank) {
    // A 5D input cannot be converted to a 2D layout.
    auto desc = make_desc_5d(1, 3, 2, 2, 2, DataType::f32, TensorLayout::NCDHW);

    std::vector<float> in_data(static_cast<size_t>(desc.numel()));
    std::vector<float> out_data(static_cast<size_t>(desc.numel()));
    TensorView in(std::span<const int64_t>(desc.dims.data(), 5), DataType::f32,
                  in_data.data(), TensorLayout::NCDHW);
    TensorView out(std::span<const int64_t>(desc.dims.data(), 5), DataType::f32,
                   out_data.data(), TensorLayout::NHWC);

    auto op = LayoutConvert::create(TensorLayout::NHWC);
    const TensorView ins[] = { in };
    TensorView outs_view[] = { out };
    NNOPS_EXPECT_EQ(op->compute(outs_view, ins), Status::unsupported);
}

// ============================================================
// Storage size — TensorDesc::storage_bytes() / row_pitch()
// ============================================================

NNOPS_TEST(storage_bytes_channel_last) {
    // N=1, C=3, H=2, W=5 f32 channel-last: a row is 3 * 5 * 4 = 60 bytes,
    // aligned to 64; 2 rows -> 128. Planar would be 1 * 3 * 2 * 32 = 192.
    auto desc = make_desc_4d(1, 3, 2, 5, DataType::f32, TensorLayout::NHWC);
    NNOPS_EXPECT_EQ(desc.row_pitch(32), 64);
    NNOPS_EXPECT_EQ(desc.storage_bytes(32), 128u);
}

NNOPS_TEST(storage_bytes_channel_last_alignment) {
    auto desc = make_desc_4d(1, 8, 2, 7, DataType::f32, TensorLayout::NHWC);
    // row = 8 * 7 * 4 = 224 bytes: already 32-aligned, 256 at 64.
    NNOPS_EXPECT_EQ(desc.row_pitch(32), 224);
    NNOPS_EXPECT_EQ(desc.storage_bytes(32), 448u);
    NNOPS_EXPECT_EQ(desc.row_pitch(64), 256);
    NNOPS_EXPECT_EQ(desc.storage_bytes(64), 512u);
}

NNOPS_TEST(storage_bytes_3d_channel_last) {
    // N=2, C=4, D=3, H=2, W=2 f32: row = 4 * 2 * 4 = 32 bytes; rows = 2*3*2 = 12.
    auto desc = make_desc_5d(2, 4, 3, 2, 2, DataType::f32, TensorLayout::NDHWC);
    NNOPS_EXPECT_EQ(desc.row_pitch(32), 32);
    NNOPS_EXPECT_EQ(desc.storage_bytes(32), 384u);
}

// ============================================================
// getOutputTensorDesc — shape and layout inference
// ============================================================

NNOPS_TEST(get_output_tensor_desc_to_channel_last_2d) {
    auto input_desc = make_desc_4d(2, 16, 3, 4, DataType::f32, TensorLayout::NCHW);

    auto op = LayoutConvert::create(TensorLayout::NHWC);
    const TensorDesc desc_arr[] = { input_desc };
    auto outputs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(outputs.size(), 1u);
    NNOPS_EXPECT_EQ(outputs[0].rank, 4);
    NNOPS_EXPECT_EQ(outputs[0].layout, TensorLayout::NHWC);
    NNOPS_EXPECT_EQ(outputs[0].dtype, DataType::f32);
    NNOPS_EXPECT_EQ(outputs[0].dims[0], 2);
    NNOPS_EXPECT_EQ(outputs[0].dims[1], 16);
    NNOPS_EXPECT_EQ(outputs[0].dims[2], 3);
    NNOPS_EXPECT_EQ(outputs[0].dims[3], 4);
}

NNOPS_TEST(get_output_tensor_desc_to_planar_2d) {
    auto input_desc = make_desc_4d(1, 24, 5, 6, DataType::f16, TensorLayout::NHWC);

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

NNOPS_TEST(get_output_tensor_desc_to_channel_last_3d) {
    auto input_desc = make_desc_5d(3, 32, 4, 5, 6, DataType::f32, TensorLayout::NCDHW);

    auto op = LayoutConvert::create(TensorLayout::NDHWC);
    const TensorDesc desc_arr[] = { input_desc };
    auto outputs = op->getOutputTensorDesc(desc_arr);

    NNOPS_EXPECT_EQ(outputs.size(), 1u);
    NNOPS_EXPECT_EQ(outputs[0].rank, 5);
    NNOPS_EXPECT_EQ(outputs[0].layout, TensorLayout::NDHWC);
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
    auto op = LayoutConvert::create(TensorLayout::NHWC);
    NNOPS_EXPECT_EQ(static_cast<int>(op->getOpType()), static_cast<int>(OpType::LayoutConvert));
    NNOPS_EXPECT_EQ(static_cast<int>(op->getBackend()), static_cast<int>(Backend::CPU));
}
