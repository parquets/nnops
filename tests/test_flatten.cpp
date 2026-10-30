/// @file test_flatten.cpp
/// @brief Flatten tests — every accepted layout unpacks to dense contiguous
/// planar data of the input's logical rank and dtype.
///
/// Coverage: an already-dense NCHW input (the memcpy identity path), NCHWC8 with a
/// partial block, a full C8 block, a single channel and several batches, NCDHWC8
/// for the 3D case, and SIMD output against the reference flatten.
///
/// Every packed input is produced by running LayoutConvert into a make_packed
/// buffer at the default 32-byte alignment; at these row widths (128 to 256 bytes)
/// that alignment adds nothing, so no case exercises pitch padding.

#include "nnops/ops/flatten.hpp"
#include "nnops/ops/layout_convert.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"

#include <vector>

using namespace nnops;

namespace nnops::backend::cpu::reference {
    void flatten_ref(const FlattenAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);
}

NNOPS_TEST(flatten_planar_identity) {
    const int64_t shape[] = {1, 3, 4, 4};
    auto [in_vec, input] = test::make_random_tensor({1, 3, 4, 4}, -1.0f, 1.0f, 42);

    auto op = Flatten::create(Backend::CPU);
    auto d_in = input.desc();
    const TensorDesc in_arr[] = {d_in};
    auto descs = op->getOutputTensorDesc(in_arr);

    NNOPS_EXPECT_EQ(descs[0].rank, 4);
    NNOPS_EXPECT_EQ(descs[0].dims[0], 1);
    NNOPS_EXPECT_EQ(descs[0].dims[1], 3);
    NNOPS_EXPECT_EQ(descs[0].dims[2], 4);
    NNOPS_EXPECT_EQ(descs[0].dims[3], 4);
    NNOPS_EXPECT_EQ(descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(descs[0].dtype, DataType::f32);

    std::vector<float> out_buf(descs[0].numel());
    TensorView output = test::make_planar(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_TRUE(test::allclose(output, input, 1e-7f, 1e-7f));
}

NNOPS_TEST(flatten_nchwc8_to_nchw) {
    const std::vector<int64_t> shape = {2, 5, 4, 4};  // 5ch → partial C8

    auto [in_vec, in_planar] = test::make_random_tensor(shape, -1.0f, 1.0f, 123);

    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8, Backend::CPU);
    auto d = in_planar.desc();
    const TensorDesc pack_in[] = {d};
    auto pack_descs = pack_op->getOutputTensorDesc(pack_in);

    std::vector<char> packed_buf(pack_descs[0].storage_bytes());
    auto in_packed = test::make_packed(pack_descs[0], packed_buf.data());
    const TensorView pack_ins[] = {in_planar};
    pack_op->compute(in_packed, pack_ins);

    auto flatten_op = Flatten::create(Backend::CPU);
    auto pd = in_packed.desc();
    const TensorDesc flat_in[] = {pd};
    auto flat_descs = flatten_op->getOutputTensorDesc(flat_in);

    NNOPS_EXPECT_EQ(flat_descs[0].rank, 4);
    NNOPS_EXPECT_EQ(flat_descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(flat_descs[0].dtype, DataType::f32);

    std::vector<float> flat_buf(flat_descs[0].numel());
    TensorView out_planar = test::make_planar(flat_descs[0], flat_buf.data());
    const TensorView flat_ins[] = {in_packed};
    flatten_op->compute(out_planar, flat_ins);

    NNOPS_EXPECT_TRUE(test::allclose(out_planar, in_planar, 1e-5f, 1e-5f));
}

NNOPS_TEST(flatten_nchwc8_full_c8) {
    const std::vector<int64_t> shape = {1, 8, 6, 6};  // full C8 block
    auto [in_vec, in_planar] = test::make_random_tensor(shape, -1.0f, 1.0f, 456);

    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8, Backend::CPU);
    auto d = in_planar.desc();
    const TensorDesc pack_in[] = {d};
    auto pack_descs = pack_op->getOutputTensorDesc(pack_in);
    std::vector<char> packed_buf(pack_descs[0].storage_bytes());
    auto in_packed = test::make_packed(pack_descs[0], packed_buf.data());
    const TensorView pack_ins[] = {in_planar};
    pack_op->compute(in_packed, pack_ins);

    auto flatten_op = Flatten::create(Backend::CPU);
    auto pd = in_packed.desc();
    const TensorDesc flat_in[] = {pd};
    auto flat_descs = flatten_op->getOutputTensorDesc(flat_in);

    std::vector<float> flat_buf(flat_descs[0].numel());
    TensorView out_planar = test::make_planar(flat_descs[0], flat_buf.data());
    const TensorView flat_ins[] = {in_packed};
    flatten_op->compute(out_planar, flat_ins);

    NNOPS_EXPECT_TRUE(test::allclose(out_planar, in_planar, 1e-5f, 1e-5f));
}

NNOPS_TEST(flatten_nchwc8_single_channel) {
    const std::vector<int64_t> shape = {1, 1, 8, 8};  // C = 1, one packed block
    auto [in_vec, in_planar] = test::make_random_tensor(shape, -1.0f, 1.0f, 789);

    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8, Backend::CPU);
    auto d = in_planar.desc();
    const TensorDesc pack_in[] = {d};
    auto pack_descs = pack_op->getOutputTensorDesc(pack_in);
    std::vector<char> packed_buf(pack_descs[0].storage_bytes());
    auto in_packed = test::make_packed(pack_descs[0], packed_buf.data());
    const TensorView pack_ins[] = {in_planar};
    pack_op->compute(in_packed, pack_ins);

    auto flatten_op = Flatten::create(Backend::CPU);
    auto pd = in_packed.desc();
    const TensorDesc flat_in[] = {pd};
    auto flat_descs = flatten_op->getOutputTensorDesc(flat_in);

    std::vector<float> flat_buf(flat_descs[0].numel());
    TensorView out_planar = test::make_planar(flat_descs[0], flat_buf.data());
    const TensorView flat_ins[] = {in_packed};
    flatten_op->compute(out_planar, flat_ins);

    NNOPS_EXPECT_TRUE(test::allclose(out_planar, in_planar, 1e-5f, 1e-5f));
}

NNOPS_TEST(flatten_ncdhwc8_to_ncdhw) {
    const std::vector<int64_t> shape = {1, 5, 3, 4, 4};
    auto [in_vec, in_planar] = test::make_random_tensor(shape, -1.0f, 1.0f, 111);

    TensorView in_planar_tv(std::span<const int64_t>(shape), DataType::f32,
                             in_vec.data(), TensorLayout::NCDHW);

    auto pack_op = LayoutConvert::create(TensorLayout::NCDHWC8, Backend::CPU);
    auto d = in_planar_tv.desc();
    const TensorDesc pack_in[] = {d};
    auto pack_descs = pack_op->getOutputTensorDesc(pack_in);
    std::vector<char> packed_buf(pack_descs[0].storage_bytes());
    auto in_packed = test::make_packed(pack_descs[0], packed_buf.data());
    const TensorView pack_ins[] = {in_planar_tv};
    pack_op->compute(in_packed, pack_ins);

    auto flatten_op = Flatten::create(Backend::CPU);
    auto pd = in_packed.desc();
    const TensorDesc flat_in[] = {pd};
    auto flat_descs = flatten_op->getOutputTensorDesc(flat_in);

    NNOPS_EXPECT_EQ(flat_descs[0].rank, 5);
    NNOPS_EXPECT_EQ(flat_descs[0].layout, TensorLayout::NCDHW);

    std::vector<float> flat_buf(flat_descs[0].numel());
    TensorView out_planar = test::make_planar(flat_descs[0], flat_buf.data());
    const TensorView flat_ins[] = {in_packed};
    flatten_op->compute(out_planar, flat_ins);

    NNOPS_EXPECT_TRUE(test::allclose(out_planar, in_planar_tv, 1e-5f, 1e-5f));
}

NNOPS_TEST(flatten_nchwc8_multibatch) {
    const std::vector<int64_t> shape = {3, 7, 4, 4};
    auto [in_vec, in_planar] = test::make_random_tensor(shape, -1.0f, 1.0f, 222);

    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8, Backend::CPU);
    auto d = in_planar.desc();
    const TensorDesc pack_in[] = {d};
    auto pack_descs = pack_op->getOutputTensorDesc(pack_in);
    std::vector<char> packed_buf(pack_descs[0].storage_bytes());
    auto in_packed = test::make_packed(pack_descs[0], packed_buf.data());
    const TensorView pack_ins[] = {in_planar};
    pack_op->compute(in_packed, pack_ins);

    auto flatten_op = Flatten::create(Backend::CPU);
    auto pd = in_packed.desc();
    const TensorDesc flat_in[] = {pd};
    auto flat_descs = flatten_op->getOutputTensorDesc(flat_in);

    std::vector<float> flat_buf(flat_descs[0].numel());
    TensorView out_planar = test::make_planar(flat_descs[0], flat_buf.data());
    const TensorView flat_ins[] = {in_packed};
    flatten_op->compute(out_planar, flat_ins);

    NNOPS_EXPECT_TRUE(test::allclose(out_planar, in_planar, 1e-5f, 1e-5f));
}

NNOPS_TEST(flatten_simd_vs_ref) {
    const std::vector<int64_t> shape = {2, 11, 6, 8};  // 11ch → partial C8

    auto [in_vec, in_planar] = test::make_random_tensor(shape, -1.0f, 1.0f, 333);
    TensorView in_planar_tv(std::span<const int64_t>(shape), DataType::f32,
                             in_vec.data(), TensorLayout::NCHW);

    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8, Backend::CPU);
    auto d = in_planar_tv.desc();
    const TensorDesc pack_in[] = {d};
    auto pack_descs = pack_op->getOutputTensorDesc(pack_in);
    std::vector<char> packed_buf(pack_descs[0].storage_bytes());
    auto in_packed = test::make_packed(pack_descs[0], packed_buf.data());
    const TensorView pack_ins[] = {in_planar_tv};
    pack_op->compute(in_packed, pack_ins);

    auto flatten_op = Flatten::create(Backend::CPU);
    auto pd = in_packed.desc();
    const TensorDesc flat_in[] = {pd};
    auto flat_descs = flatten_op->getOutputTensorDesc(flat_in);

    std::vector<float> simd_buf(flat_descs[0].numel());
    TensorView simd_out = test::make_planar(flat_descs[0], simd_buf.data());
    const TensorView simd_ins[] = {in_packed};
    flatten_op->compute(simd_out, simd_ins);

    std::vector<float> ref_buf(flat_descs[0].numel());
    TensorView ref_out = test::make_planar(flat_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        backend::cpu::reference::flatten_ref(FlattenAttributes{}, ref_out, simd_ins, ctx, nullptr);
    }

    NNOPS_EXPECT_TRUE(test::allclose(simd_out, ref_out, 1e-5f, 1e-5f));
}

NNOPS_TEST(flatten_op_type) {
    auto op = Flatten::create(Backend::CPU);
    NNOPS_EXPECT_EQ(static_cast<int>(op->getOpType()), static_cast<int>(OpType::Flatten));
}
