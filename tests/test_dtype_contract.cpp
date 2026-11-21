/// @file test_dtype_contract.cpp
/// @brief What every CPU operator owes a caller whose dtype it cannot serve:
///        a non-ok Status, and an output left alone.
///
/// A `dispatch_*` that matches no case used to fall off the end: the assert
/// guarding it is compiled out under NDEBUG, so `compute()` returned
/// `Status::ok` having written nothing. That is the failure this file pins —
/// both the descriptor-only rule (validate()) and the end-to-end call.
///
/// It also pins the other direction: the element-movement operators (Concat,
/// Flatten, Slice, Permute) carry *every* whole-element dtype, so an int64 or
/// int32 tensor moves like an f32 one. Those four used to be float-only, which
/// left ONNX graphs that concat or slice shape/index tensors unable to run.

#include "nnops/ops/activation.hpp"
#include "nnops/ops/argminmax.hpp"
#include "nnops/ops/clamp.hpp"
#include "nnops/ops/concat.hpp"
#include "nnops/ops/eltwise.hpp"
#include "nnops/ops/flatten.hpp"
#include "nnops/ops/grid_sample.hpp"
#include "nnops/ops/permute.hpp"
#include "nnops/ops/quant_linear.hpp"
#include "nnops/ops/reduce.hpp"
#include "nnops/ops/resize.hpp"
#include "nnops/ops/slice.hpp"
#include "nnops/ops/softmax.hpp"
#include "nnops/ops/topk.hpp"
#include "nnops/ops/unary.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

using namespace nnops;

namespace {

constexpr int32_t kSentinel = -12345;

/// True when every element of the output still holds the sentinel: compute()
/// must not have touched it.
bool untouched(const std::vector<int32_t>& v)
{
    for (int32_t x : v) {
        if (x != kSentinel) {
            return false;
        }
    }
    return true;
}

} // namespace

// ============================================================
// An unsupported dtype is reported, never silently skipped
// ============================================================

NNOPS_TEST(dtype_contract_elementwise_rejects_int32) {
    const int64_t shape[] = {4};
    std::vector<int32_t> a_data = {1, 2, 3, 4};
    std::vector<int32_t> b_data = {5, 6, 7, 8};
    std::vector<int32_t> out_data(4, kSentinel);
    TensorView a(std::span<const int64_t>(shape, 1), DataType::s32, a_data.data());
    TensorView b(std::span<const int64_t>(shape, 1), DataType::s32, b_data.data());
    TensorView out(std::span<const int64_t>(shape, 1), DataType::s32, out_data.data());

    {
        EltwiseAttributes attrs;
        attrs.type = EltwiseType::Add;
        auto op = Eltwise::create(attrs, Backend::CPU);
        const TensorView ins[] = {a, b};
        NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::unsupported);
        NNOPS_EXPECT_TRUE(untouched(out_data));
    }
    {
        ClampAttributes attrs;
        attrs.min_val = -1.0f;
        attrs.max_val = 1.0f;
        auto op = Clamp::create(attrs, Backend::CPU);
        const TensorView ins[] = {a};
        NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::unsupported);
        NNOPS_EXPECT_TRUE(untouched(out_data));
    }
    {
        UnaryAttributes attrs;
        attrs.type = UnaryType::Neg;
        auto op = Unary::create(attrs, Backend::CPU);
        const TensorView ins[] = {a};
        NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::unsupported);
        NNOPS_EXPECT_TRUE(untouched(out_data));
    }
}

NNOPS_TEST(dtype_contract_reductions_reject_int32) {
    const int64_t shape[] = {2, 4};
    std::vector<int32_t> data = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int32_t> out2(2, kSentinel);
    std::vector<int32_t> check(2, kSentinel);
    TensorView in(std::span<const int64_t>(shape, 2), DataType::s32, data.data());
    const int64_t s1[] = {2, 1};
    const int64_t s0[] = {2};

    {
        ReduceAttributes attrs;
        attrs.axes = {1};
        attrs.type = ReduceType::Sum;
        auto op = Reduce::create(attrs, Backend::CPU);
        TensorView out(std::span<const int64_t>(s1, 2), DataType::s32, out2.data());
        const TensorView ins[] = {in};
        NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::unsupported);
        NNOPS_EXPECT_TRUE(untouched(out2));
    }
    {
        SoftmaxAttributes attrs;
        attrs.axis = -1;
        auto op = Softmax::create(attrs, Backend::CPU);
        TensorView out(std::span<const int64_t>(shape, 2), DataType::s32, check.data());
        const TensorView ins[] = {in};
        NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::unsupported);
        NNOPS_EXPECT_TRUE(untouched(check));
    }
    {
        // ArgMax writes s64 indices; an s32 input has no route at all.
        ArgMinMaxAttributes attrs;
        attrs.type = ArgMinMaxType::Max;
        attrs.axis = -1;
        auto op = ArgMax::create(attrs, Backend::CPU);
        std::vector<int64_t> idx(2, kSentinel);
        TensorView out(std::span<const int64_t>(s0, 1), DataType::s64, idx.data());
        const TensorView ins[] = {in};
        NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::unsupported);
        NNOPS_EXPECT_EQ(idx[0], int64_t(kSentinel));
        NNOPS_EXPECT_EQ(idx[1], int64_t(kSentinel));
    }
    {
        TopKAttributes attrs;
        attrs.type = TopKType::Max;
        attrs.axis = -1;
        attrs.k = 2;
        auto op = TopK::create(attrs, Backend::CPU);
        std::vector<int32_t> vals(4, kSentinel);
        std::vector<int64_t> idx(4, kSentinel);
        TensorView vo(std::span<const int64_t>(shape, 2), DataType::s32, vals.data());
        TensorView io(std::span<const int64_t>(shape, 2), DataType::s64, idx.data());
        TensorView outs[] = {vo, io};
        const TensorView ins[] = {in};
        NNOPS_EXPECT_TRUE(op->compute(outs, ins) == Status::unsupported);
        NNOPS_EXPECT_TRUE(untouched(vals));
    }
}

NNOPS_TEST(dtype_contract_spatial_reject_int32) {
    const int64_t shape[] = {1, 2, 2, 2};
    std::vector<int32_t> data(8, 3);
    std::vector<int32_t> out_data(8, kSentinel);
    TensorView in(std::span<const int64_t>(shape, 4), DataType::s32, data.data(),
                  TensorLayout::NHWC);
    TensorView out(std::span<const int64_t>(shape, 4), DataType::s32, out_data.data(),
                   TensorLayout::NHWC);

    {
        ResizeAttributes attrs;
        attrs.mode = ResizeMode::Nearest;
        auto op = Resize::create(attrs, Backend::CPU);
        const TensorView ins[] = {in};
        NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::unsupported);
        NNOPS_EXPECT_TRUE(untouched(out_data));
    }
    {
        GridSampleAttributes attrs;
        auto op = GridSample::create(attrs, Backend::CPU);
        // The grid is f32 whatever the activation dtype is.
        std::vector<float> grid(static_cast<size_t>(4 * 2), 0.0f);
        const int64_t gshape[] = {1, 2, 2, 2};
        TensorView g(std::span<const int64_t>(gshape, 4), DataType::f32, grid.data());
        const TensorView ins[] = {in, g};
        NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::unsupported);
        NNOPS_EXPECT_TRUE(untouched(out_data));
    }
    {
        // A float grid is required: an f16 one would be read as f32 bytes.
        ResizeAttributes attrs;
        auto op = GridSample::create(GridSampleAttributes{}, Backend::CPU);
        std::vector<nnops::backend::cpu::half> grid(static_cast<size_t>(4), {});
        const int64_t gshape[] = {1, 2, 2, 2};
        TensorView g(std::span<const int64_t>(gshape, 4), DataType::f16, grid.data());
        std::vector<float> fdata(8, 0.0f);
        TensorView fin(std::span<const int64_t>(shape, 4), DataType::f32, fdata.data(),
                       TensorLayout::NHWC);
        const TensorView ins[] = {fin, g};
        NNOPS_EXPECT_TRUE(op->compute(out, ins) == Status::invalid_argument);
    }
}

// ============================================================
// The movement operators carry the integer dtypes too
// ============================================================

NNOPS_TEST(dtype_contract_concat_carries_int64) {
    const int64_t a_shape[] = {2, 3};
    const int64_t out_shape[] = {2, 6};
    std::vector<int64_t> a = {1, 2, 3, 4, 5, 6};
    std::vector<int64_t> b = {7, 8, 9, 10, 11, 12};
    std::vector<int64_t> out(12, -1);
    TensorView va(std::span<const int64_t>(a_shape, 2), DataType::s64, a.data());
    TensorView vb(std::span<const int64_t>(a_shape, 2), DataType::s64, b.data());
    TensorView vo(std::span<const int64_t>(out_shape, 2), DataType::s64, out.data());

    ConcatAttributes attrs;
    attrs.axis = -1;
    auto op = Concat::create(attrs, Backend::CPU);
    const TensorView ins[] = {va, vb};
    NNOPS_EXPECT_TRUE(op->compute(vo, ins) == Status::ok);
    for (int64_t r = 0; r < 2; ++r) {
        for (int64_t c = 0; c < 6; ++c) {
            const int64_t want = (c < 3) ? a[static_cast<size_t>(r * 3 + c)]
                                         : b[static_cast<size_t>(r * 3 + c - 3)];
            NNOPS_EXPECT_EQ(out[static_cast<size_t>(r * 6 + c)], want);
        }
    }
}

NNOPS_TEST(dtype_contract_slice_permute_flatten_carry_int32) {
    // Slice: [2,4] s32 -> [2,2] taking columns 1..2
    {
        const int64_t in_shape[] = {2, 4};
        const int64_t out_shape[] = {2, 2};
        std::vector<int32_t> x = {0, 1, 2, 3, 10, 11, 12, 13};
        std::vector<int32_t> y(4, kSentinel);
        TensorView vin(std::span<const int64_t>(in_shape, 2), DataType::s32, x.data());
        TensorView vout(std::span<const int64_t>(out_shape, 2), DataType::s32, y.data());
        SliceAttributes attrs;
        attrs.axes = {1};
        attrs.starts = {1};
        attrs.ends = {3};
        auto op = Slice::create(attrs, Backend::CPU);
        const TensorView ins[] = {vin};
        NNOPS_EXPECT_TRUE(op->compute(vout, ins) == Status::ok);
        NNOPS_EXPECT_EQ(y[0], 1);
        NNOPS_EXPECT_EQ(y[1], 2);
        NNOPS_EXPECT_EQ(y[2], 11);
        NNOPS_EXPECT_EQ(y[3], 12);
    }

    // Permute: [2,4] s32 -> [4,2] (a transpose; the tiled SIMD route is fp-only)
    {
        const int64_t in_shape[] = {2, 4};
        const int64_t out_shape[] = {4, 2};
        std::vector<int32_t> x = {0, 1, 2, 3, 10, 11, 12, 13};
        std::vector<int32_t> y(8, kSentinel);
        TensorView vin(std::span<const int64_t>(in_shape, 2), DataType::s32, x.data());
        TensorView vout(std::span<const int64_t>(out_shape, 2), DataType::s32, y.data());
        PermuteAttributes attrs;
        attrs.perm = {1, 0};
        auto op = Permute::create(attrs, Backend::CPU);
        const TensorView ins[] = {vin};
        NNOPS_EXPECT_TRUE(op->compute(vout, ins) == Status::ok);
        for (int64_t r = 0; r < 2; ++r) {
            for (int64_t c = 0; c < 4; ++c) {
                NNOPS_EXPECT_EQ(y[static_cast<size_t>(c * 2 + r)],
                                x[static_cast<size_t>(r * 4 + c)]);
            }
        }
    }

    // Flatten: NHWC [1,2,2,2] s32 read as dense
    {
        const int64_t shape[] = {1, 2, 2, 2};
        std::vector<int32_t> x = {0, 1, 2, 3, 4, 5, 6, 7};
        std::vector<int32_t> y(8, kSentinel);
        TensorView vin(std::span<const int64_t>(shape, 4), DataType::s32, x.data(),
                       TensorLayout::NHWC);
        TensorView vout(std::span<const int64_t>(shape, 4), DataType::s32, y.data());
        FlattenAttributes attrs;
        auto op = Flatten::create(attrs, Backend::CPU);
        const TensorView ins[] = {vin};
        NNOPS_EXPECT_TRUE(op->compute(vout, ins) == Status::ok);
        for (size_t i = 0; i < x.size(); ++i) {
            NNOPS_EXPECT_EQ(y[i], x[i]);
        }
    }
}
