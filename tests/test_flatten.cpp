/// @file test_flatten.cpp
/// @brief Flatten tests — every accepted layout unpacks to dense contiguous
/// Channel-first data of the input's logical rank and dtype.
///
/// Coverage: an already-dense channel-first input (the memcpy identity path),
/// channel-last with a partial / full / single channel plane and several batches,
/// channel-last 3D for the 3D case, and SIMD output against the reference
/// flatten.
///
/// Every channel-last input is produced by running LayoutConvert into a
/// make_packed buffer at the default 32-byte alignment, so a pitched row is
/// exercised whenever the row width is not already a multiple of 32 bytes.

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

namespace {

/// Convert a channel-first view into `target` (channel-last) and return the view; the
/// caller keeps `buf` alive.
TensorView to_channel_last(const TensorView& channel_first, TensorLayout target,
                           std::vector<char>& buf)
{
    auto op = LayoutConvert::create(target, Backend::CPU);
    const TensorDesc in[] = {channel_first.desc()};
    TensorDesc d = op->getOutputTensorDesc(in)[0];
    buf.resize(d.storage_bytes());
    TensorView cl = test::make_packed(d, buf.data());
    const TensorView ins[] = {channel_first};
    op->compute(cl, ins);
    return cl;
}

}  // anonymous namespace

NNOPS_TEST(flatten_channel_first_identity) {
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
    TensorView output = test::make_channel_first(descs[0], out_buf.data());

    const TensorView ins[] = {input};
    op->compute(output, ins);

    NNOPS_EXPECT_TRUE(test::allclose(output, input, 1e-7f, 1e-7f));
}

NNOPS_TEST(flatten_channel_last_to_channel_first) {
    const std::vector<int64_t> shape = {2, 5, 4, 4};  // odd C

    auto [in_vec, in_channel_first] = test::make_random_tensor(shape, -1.0f, 1.0f, 123);
    std::vector<char> cl_buf;
    TensorView in_cl = to_channel_last(in_channel_first, TensorLayout::NHWC, cl_buf);

    auto flatten_op = Flatten::create(Backend::CPU);
    auto pd = in_cl.desc();
    const TensorDesc flat_in[] = {pd};
    auto flat_descs = flatten_op->getOutputTensorDesc(flat_in);

    NNOPS_EXPECT_EQ(flat_descs[0].rank, 4);
    NNOPS_EXPECT_EQ(flat_descs[0].layout, TensorLayout::NCHW);
    NNOPS_EXPECT_EQ(flat_descs[0].dtype, DataType::f32);

    std::vector<float> flat_buf(flat_descs[0].numel());
    TensorView out_channel_first = test::make_channel_first(flat_descs[0], flat_buf.data());
    const TensorView flat_ins[] = {in_cl};
    flatten_op->compute(out_channel_first, flat_ins);

    NNOPS_EXPECT_TRUE(test::allclose(out_channel_first, in_channel_first, 1e-5f, 1e-5f));
}

NNOPS_TEST(flatten_channel_last_full_c) {
    const std::vector<int64_t> shape = {1, 8, 6, 6};
    auto [in_vec, in_channel_first] = test::make_random_tensor(shape, -1.0f, 1.0f, 456);

    std::vector<char> cl_buf;
    TensorView in_cl = to_channel_last(in_channel_first, TensorLayout::NHWC, cl_buf);

    auto flatten_op = Flatten::create(Backend::CPU);
    auto pd = in_cl.desc();
    const TensorDesc flat_in[] = {pd};
    auto flat_descs = flatten_op->getOutputTensorDesc(flat_in);

    std::vector<float> flat_buf(flat_descs[0].numel());
    TensorView out_channel_first = test::make_channel_first(flat_descs[0], flat_buf.data());
    const TensorView flat_ins[] = {in_cl};
    flatten_op->compute(out_channel_first, flat_ins);

    NNOPS_EXPECT_TRUE(test::allclose(out_channel_first, in_channel_first, 1e-5f, 1e-5f));
}

NNOPS_TEST(flatten_channel_last_single_channel) {
    const std::vector<int64_t> shape = {1, 1, 8, 8};
    auto [in_vec, in_channel_first] = test::make_random_tensor(shape, -1.0f, 1.0f, 789);

    std::vector<char> cl_buf;
    TensorView in_cl = to_channel_last(in_channel_first, TensorLayout::NHWC, cl_buf);

    auto flatten_op = Flatten::create(Backend::CPU);
    auto pd = in_cl.desc();
    const TensorDesc flat_in[] = {pd};
    auto flat_descs = flatten_op->getOutputTensorDesc(flat_in);

    std::vector<float> flat_buf(flat_descs[0].numel());
    TensorView out_channel_first = test::make_channel_first(flat_descs[0], flat_buf.data());
    const TensorView flat_ins[] = {in_cl};
    flatten_op->compute(out_channel_first, flat_ins);

    NNOPS_EXPECT_TRUE(test::allclose(out_channel_first, in_channel_first, 1e-5f, 1e-5f));
}

NNOPS_TEST(flatten_ndhwc_to_ncdhw) {
    const std::vector<int64_t> shape = {1, 5, 3, 4, 4};
    auto [in_vec, _] = test::make_random_tensor(shape, -1.0f, 1.0f, 111);

    TensorView in_channel_first(std::span<const int64_t>(shape), DataType::f32,
                         in_vec.data(), TensorLayout::NCDHW);

    std::vector<char> cl_buf;
    TensorView in_cl = to_channel_last(in_channel_first, TensorLayout::NDHWC, cl_buf);

    auto flatten_op = Flatten::create(Backend::CPU);
    auto pd = in_cl.desc();
    const TensorDesc flat_in[] = {pd};
    auto flat_descs = flatten_op->getOutputTensorDesc(flat_in);

    NNOPS_EXPECT_EQ(flat_descs[0].rank, 5);
    NNOPS_EXPECT_EQ(flat_descs[0].layout, TensorLayout::NCDHW);

    std::vector<float> flat_buf(flat_descs[0].numel());
    TensorView out_channel_first = test::make_channel_first(flat_descs[0], flat_buf.data());
    const TensorView flat_ins[] = {in_cl};
    flatten_op->compute(out_channel_first, flat_ins);

    NNOPS_EXPECT_TRUE(test::allclose(out_channel_first, in_channel_first, 1e-5f, 1e-5f));
}

NNOPS_TEST(flatten_channel_last_multibatch) {
    const std::vector<int64_t> shape = {3, 7, 4, 4};
    auto [in_vec, in_channel_first] = test::make_random_tensor(shape, -1.0f, 1.0f, 222);

    std::vector<char> cl_buf;
    TensorView in_cl = to_channel_last(in_channel_first, TensorLayout::NHWC, cl_buf);

    auto flatten_op = Flatten::create(Backend::CPU);
    auto pd = in_cl.desc();
    const TensorDesc flat_in[] = {pd};
    auto flat_descs = flatten_op->getOutputTensorDesc(flat_in);

    std::vector<float> flat_buf(flat_descs[0].numel());
    TensorView out_channel_first = test::make_channel_first(flat_descs[0], flat_buf.data());
    const TensorView flat_ins[] = {in_cl};
    flatten_op->compute(out_channel_first, flat_ins);

    NNOPS_EXPECT_TRUE(test::allclose(out_channel_first, in_channel_first, 1e-5f, 1e-5f));
}

NNOPS_TEST(flatten_simd_vs_ref) {
    const std::vector<int64_t> shape = {2, 11, 6, 8};

    auto [in_vec, in_channel_first] = test::make_random_tensor(shape, -1.0f, 1.0f, 333);
    TensorView in_channel_first_tv(std::span<const int64_t>(shape), DataType::f32,
                            in_vec.data(), TensorLayout::NCHW);

    std::vector<char> cl_buf;
    TensorView in_cl = to_channel_last(in_channel_first_tv, TensorLayout::NHWC, cl_buf);

    auto flatten_op = Flatten::create(Backend::CPU);
    auto pd = in_cl.desc();
    const TensorDesc flat_in[] = {pd};
    auto flat_descs = flatten_op->getOutputTensorDesc(flat_in);

    std::vector<float> simd_buf(flat_descs[0].numel());
    TensorView simd_out = test::make_channel_first(flat_descs[0], simd_buf.data());
    const TensorView simd_ins[] = {in_cl};
    flatten_op->compute(simd_out, simd_ins);

    std::vector<float> ref_buf(flat_descs[0].numel());
    TensorView ref_out = test::make_channel_first(flat_descs[0], ref_buf.data());
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
