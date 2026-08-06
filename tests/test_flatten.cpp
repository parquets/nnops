/// @file test_flatten.cpp
/// @brief Unit tests for Flatten operator (CPU backend).
///
/// Tests conversion from any layout (NCHW, NCHWC8, NCDHW, NCDHWC8,
/// with or without pitch padding) to dense contiguous planar.

#include "nnops/ops/flatten.hpp"
#include "nnops/ops/layout_convert.hpp"
#include "common/test_harness.hpp"
#include "common/random_tensor.hpp"
#include "common/compare.hpp"
#include "common/test_helpers.hpp"

#include <vector>

using namespace nnops;

// Forward declaration
namespace nnops::backend::cpu::reference {
    void flatten_ref(const FlattenAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* workspace);
}

// ============================================================
// Hand-verified: dense planar → dense planar (memcpy identity)
// ============================================================

NNOPS_TEST(flatten_planar_identity) {
    // Already-dense NCHW → should be a no-op memcpy
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

    // Output should match input exactly
    NNOPS_EXPECT_TRUE(test::allclose(output, input, 1e-7f, 1e-7f));
}

// ============================================================
// NCHWC8 → dense NCHW
// ============================================================

NNOPS_TEST(flatten_nchwc8_to_nchw) {
    // Pack → flatten → compare with original planar reference
    const std::vector<int64_t> shape = {2, 5, 4, 4};  // 5ch → partial C8

    auto [in_vec, in_planar] = test::make_random_tensor(shape, -1.0f, 1.0f, 123);

    // Pack to NCHWC8
    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8, Backend::CPU);
    auto d = in_planar.desc();
    const TensorDesc pack_in[] = {d};
    auto pack_descs = pack_op->getOutputTensorDesc(pack_in);

    std::vector<char> packed_buf(pack_descs[0].storage_bytes());
    auto in_packed = test::make_packed(pack_descs[0], packed_buf.data());
    const TensorView pack_ins[] = {in_planar};
    pack_op->compute(in_packed, pack_ins);

    // Flatten: NCHWC8 → dense NCHW
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

    // Should match original planar input
    NNOPS_EXPECT_TRUE(test::allclose(out_planar, in_planar, 1e-5f, 1e-5f));
}

NNOPS_TEST(flatten_nchwc8_full_c8) {
    // Full C8 block: [1, 8, 4, 4]
    const std::vector<int64_t> shape = {1, 8, 6, 6};
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
    // Single channel packed: [1, 1, 8, 8]
    const std::vector<int64_t> shape = {1, 1, 8, 8};
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

// ============================================================
// 3D packed → dense planar
// ============================================================

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

// ============================================================
// Multiple batches
// ============================================================

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

// ============================================================
// SIMD vs reference comparison
// ============================================================

NNOPS_TEST(flatten_simd_vs_ref) {
    const std::vector<int64_t> shape = {2, 11, 6, 8};  // 11ch → partial C8

    auto [in_vec, in_planar] = test::make_random_tensor(shape, -1.0f, 1.0f, 333);
    TensorView in_planar_tv(std::span<const int64_t>(shape), DataType::f32,
                             in_vec.data(), TensorLayout::NCHW);

    // Pack to NCHWC8
    auto pack_op = LayoutConvert::create(TensorLayout::NCHWC8, Backend::CPU);
    auto d = in_planar_tv.desc();
    const TensorDesc pack_in[] = {d};
    auto pack_descs = pack_op->getOutputTensorDesc(pack_in);
    std::vector<char> packed_buf(pack_descs[0].storage_bytes());
    auto in_packed = test::make_packed(pack_descs[0], packed_buf.data());
    const TensorView pack_ins[] = {in_planar_tv};
    pack_op->compute(in_packed, pack_ins);

    // SIMD flatten
    auto flatten_op = Flatten::create(Backend::CPU);
    auto pd = in_packed.desc();
    const TensorDesc flat_in[] = {pd};
    auto flat_descs = flatten_op->getOutputTensorDesc(flat_in);

    std::vector<float> simd_buf(flat_descs[0].numel());
    TensorView simd_out = test::make_planar(flat_descs[0], simd_buf.data());
    const TensorView simd_ins[] = {in_packed};
    flatten_op->compute(simd_out, simd_ins);

    // Reference flatten
    std::vector<float> ref_buf(flat_descs[0].numel());
    TensorView ref_out = test::make_planar(flat_descs[0], ref_buf.data());
    {
        ComputeContext ctx;
        backend::cpu::reference::flatten_ref(FlattenAttributes{}, ref_out, simd_ins, ctx, nullptr);
    }

    // SIMD should match reference
    NNOPS_EXPECT_TRUE(test::allclose(simd_out, ref_out, 1e-5f, 1e-5f));
}

// ============================================================
// Op type
// ============================================================

NNOPS_TEST(flatten_op_type) {
    auto op = Flatten::create(Backend::CPU);
    NNOPS_EXPECT_EQ(static_cast<int>(op->getOpType()), static_cast<int>(OpType::Flatten));
}
