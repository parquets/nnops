/// @file test_status.cpp
/// @brief The error contract: Status values, the shared check helpers, and the
///        validate()/compute() failure paths of the operators covered here.
///
/// The point of these tests is that a malformed call *returns*, and returns a
/// specific code — no abort, no out-of-bounds access, no write to the output.
/// Each op case is one rule from one op; together they pin the split between
/// validate() (descriptors only, usable at graph-plan time) and compute()
/// (views, i.e. whether there is data behind the descriptors).

#include "nnops/core/status.hpp"
#include "nnops/detail/validate.hpp"

#include "nnops/ops/activation.hpp"
#include "nnops/ops/attention.hpp"
#include "nnops/ops/concat.hpp"
#include "nnops/ops/conv2d.hpp"
#include "nnops/ops/eltwise.hpp"
#include "nnops/ops/embed.hpp"
#include "nnops/ops/layout_convert.hpp"
#include "nnops/ops/matmul.hpp"
#include "nnops/ops/moe.hpp"
#include "nnops/ops/permute.hpp"
#include "nnops/ops/slice.hpp"
#include "nnops/ops/softmax.hpp"
#include "nnops/ops/topk.hpp"

#include "common/test_harness.hpp"

#include <cstring>
#include <span>
#include <string>
#include <vector>

using namespace nnops;
using nnops::detail::check_counts;
using nnops::detail::check_rank;
using nnops::detail::check_views;

namespace {

TensorDesc desc_of(std::initializer_list<int64_t> dims,
                   DataType dtype = DataType::f32,
                   TensorLayout layout = TensorLayout::NCHW)
{
    TensorDesc d;
    d.rank = static_cast<int64_t>(dims.size());
    for (int64_t v : dims) { d.dims.push_back(v); }
    d.dtype = dtype;
    d.layout = layout;
    return d;
}

}  // namespace

// ============================================================
// The value type itself
// ============================================================

NNOPS_TEST(status_value_type) {
    NNOPS_EXPECT_EQ(static_cast<uint8_t>(Status::ok), 0);
    NNOPS_EXPECT_TRUE(is_ok(Status::ok));
    NNOPS_EXPECT_TRUE(!is_error(Status::ok));
    NNOPS_EXPECT_TRUE(is_error(Status::invalid_argument));
    NNOPS_EXPECT_TRUE(is_error(Status::unsupported));
    NNOPS_EXPECT_TRUE(is_error(Status::not_initialized));
    NNOPS_EXPECT_TRUE(is_error(Status::out_of_memory));
    NNOPS_EXPECT_TRUE(is_error(Status::internal));

    NNOPS_EXPECT_EQ(std::string(to_string(Status::ok)), std::string("ok"));
    NNOPS_EXPECT_EQ(std::string(to_string(Status::invalid_argument)),
                    std::string("invalid_argument"));
    NNOPS_EXPECT_EQ(std::string(to_string(Status::unsupported)),
                    std::string("unsupported"));
    NNOPS_EXPECT_EQ(std::string(to_string(Status::not_initialized)),
                    std::string("not_initialized"));
    NNOPS_EXPECT_EQ(std::string(to_string(Status::out_of_memory)),
                    std::string("out_of_memory"));
    NNOPS_EXPECT_EQ(std::string(to_string(Status::internal)),
                    std::string("internal"));
}

// ============================================================
// The shared check helpers
// ============================================================

NNOPS_TEST(status_helpers) {
    // check_counts: [min, max] on inputs, a floor on outputs.
    NNOPS_EXPECT_TRUE(is_ok(check_counts(1, 1, 1, 1, 1)));
    NNOPS_EXPECT_TRUE(is_error(check_counts(0, 1, 1, 1, 1)));
    NNOPS_EXPECT_TRUE(is_error(check_counts(2, 1, 1, 1, 1)));
    NNOPS_EXPECT_TRUE(is_error(check_counts(1, 0, 1, 1, 1)));
    // An unbounded max accepts any count at or above the minimum.
    NNOPS_EXPECT_TRUE(is_ok(check_counts(9, 1, 2)));
    NNOPS_EXPECT_TRUE(is_error(check_counts(1, 1, 2)));

    // check_rank: hi < 0 means unbounded above.
    NNOPS_EXPECT_TRUE(is_ok(check_rank(2, 2)));
    NNOPS_EXPECT_TRUE(is_ok(check_rank(4, 2)));
    NNOPS_EXPECT_TRUE(is_error(check_rank(1, 2)));
    NNOPS_EXPECT_TRUE(is_ok(check_rank(3, 1, 3)));
    NNOPS_EXPECT_TRUE(is_error(check_rank(4, 1, 3)));

    // The extra parentheses keep the template's comma out of the macro's
    // argument list.
    NNOPS_EXPECT_TRUE((detail::dtype_in<DataType::f32, DataType::f16>(DataType::f16)));
    NNOPS_EXPECT_TRUE((!detail::dtype_in<DataType::f32, DataType::f16>(DataType::s8)));

    NNOPS_EXPECT_TRUE(is_ok(detail::require(true)));
    NNOPS_EXPECT_TRUE(detail::require(false) == Status::invalid_argument);
    NNOPS_EXPECT_TRUE(detail::require(false, Status::unsupported) ==
                      Status::unsupported);

    // check_views adds "there is data behind the descriptor" to the counts.
    const int64_t shape[] = {4};
    float buf[4] = {};
    TensorView live(shape, DataType::f32, buf);
    TensorView empty;
    TensorView outs[1] = {live};
    const TensorView two_live[2] = {live, live};
    const TensorView one_live[1] = {live};
    const TensorView one_empty[1] = {empty};
    NNOPS_EXPECT_TRUE(is_ok(check_views(outs, one_live, 1, 1, 1)));
    NNOPS_EXPECT_TRUE(check_views(outs, one_empty, 1, 1, 1) ==
                      Status::invalid_argument);
    NNOPS_EXPECT_TRUE(check_views(outs, two_live, 1, 1, 1) ==
                      Status::invalid_argument);
    TensorView empty_outs[1] = {empty};
    NNOPS_EXPECT_TRUE(check_views(empty_outs, one_live, 1, 1, 1) ==
                      Status::invalid_argument);
}

// ============================================================
// validate() from descriptors alone — the graph-plan path
// ============================================================

NNOPS_TEST(status_validate_descriptors_only) {
    // No data anywhere in this test: every call goes through the descriptor
    // overload, which is what a graph engine runs before allocating buffers.

    // MatMul with mismatched inner dimensions.
    {
        auto op = MatMul::create(MatMulAttributes{}, Backend::CPU);
        const TensorDesc in[] = {desc_of({4, 8}), desc_of({16, 32})};
        const TensorDesc out[] = {desc_of({4, 32})};
        NNOPS_EXPECT_TRUE(op->validate(in, out) == Status::invalid_argument);

        const TensorDesc ok_in[] = {desc_of({4, 8}), desc_of({8, 32})};
        NNOPS_EXPECT_TRUE(op->validate(ok_in, out) == Status::ok);

        // A rank-1 operand has no inner dimension to multiply.
        const TensorDesc vec_in[] = {desc_of({8}), desc_of({8, 32})};
        NNOPS_EXPECT_TRUE(op->validate(vec_in, out) == Status::invalid_argument);
    }

    // Permute: one entry per axis, and the output is the permuted input shape.
    {
        const int64_t p_short[] = {2, 1};   // the input below has rank 3
        PermuteAttributes attrs;
        attrs.perm = std::span<const int64_t>(p_short, 2);
        auto op = Permute::create(attrs, Backend::CPU);
        const TensorDesc in[] = {desc_of({2, 3, 4})};
        const TensorDesc out[] = {desc_of({3, 2, 4})};
        NNOPS_EXPECT_TRUE(op->validate(in, out) == Status::invalid_argument);
    }
    {
        const int64_t p[] = {2, 0, 1};
        PermuteAttributes attrs;
        attrs.perm = std::span<const int64_t>(p, 3);
        auto op = Permute::create(attrs, Backend::CPU);
        const TensorDesc in[] = {desc_of({2, 3, 4})};
        const TensorDesc good[] = {desc_of({4, 2, 3})};
        const TensorDesc bad[] = {desc_of({4, 3, 2})};
        NNOPS_EXPECT_TRUE(op->validate(in, good) == Status::ok);
        NNOPS_EXPECT_TRUE(op->validate(in, bad) == Status::invalid_argument);
    }

    // Slice: starts/ends/axes must be the same length, and the declared output
    // must be what those ranges produce.
    {
        SliceAttributes attrs;
        attrs.axes = {1};
        attrs.starts = {1};
        attrs.ends = {3};
        attrs.steps = {1, 2};   // one step too many
        auto op = Slice::create(attrs, Backend::CPU);
        const TensorDesc in[] = {desc_of({2, 6})};
        const TensorDesc out[] = {desc_of({2, 2})};
        NNOPS_EXPECT_TRUE(op->validate(in, out) == Status::invalid_argument);
    }
    {
        SliceAttributes attrs;
        attrs.axes = {1};
        attrs.starts = {1};
        attrs.ends = {3};
        auto op = Slice::create(attrs, Backend::CPU);
        const TensorDesc in[] = {desc_of({2, 6})};
        const TensorDesc good[] = {desc_of({2, 2})};
        const TensorDesc bad[] = {desc_of({2, 3})};
        NNOPS_EXPECT_TRUE(op->validate(in, good) == Status::ok);
        NNOPS_EXPECT_TRUE(op->validate(in, bad) == Status::invalid_argument);
    }

    // LayoutConvert: converting to the current layout is rejected, not a no-op.
    {
        auto op = LayoutConvert::create(TensorLayout::NCHW, Backend::CPU);
        const TensorDesc same[] = {desc_of({2, 3}, DataType::f32, TensorLayout::NCHW)};
        const TensorDesc other[] = {desc_of({2, 3}, DataType::f32, TensorLayout::NCHWC8)};
        NNOPS_EXPECT_TRUE(op->validate(same, same) == Status::invalid_argument);
        NNOPS_EXPECT_TRUE(op->validate(other, same) == Status::ok);
    }

    // Embed: non-integer indices are malformed; an unknown weight dtype is unsupported.
    {
        auto op = Embed::create(EmbedAttributes{}, Backend::CPU);
        const TensorDesc bad_idx[] = {desc_of({8, 4}), desc_of({2}, DataType::f32)};
        const TensorDesc out[] = {desc_of({2, 4})};
        NNOPS_EXPECT_TRUE(op->validate(bad_idx, out) == Status::invalid_argument);

        const TensorDesc s32_idx[] = {desc_of({8, 4}), desc_of({2}, DataType::s32)};
        NNOPS_EXPECT_TRUE(op->validate(s32_idx, out) == Status::ok);

        const TensorDesc odd_w[] = {desc_of({8, 4}, DataType::s32), desc_of({2}, DataType::s64)};
        NNOPS_EXPECT_TRUE(op->validate(odd_w, out) == Status::unsupported);
    }

    // Softmax dequantizes, so an integer output has no meaning.
    {
        auto op = Softmax::create(SoftmaxAttributes{}, Backend::CPU);
        // Note that a descriptor has no quant parameters — they live on the
        // view — so this rule keys off the dtype alone.
        const TensorDesc in[] = {desc_of({2, 8}, DataType::s8)};
        const TensorDesc int_out[] = {desc_of({2, 8}, DataType::s8)};
        const TensorDesc fp_out[] = {desc_of({2, 8}, DataType::f32)};
        NNOPS_EXPECT_TRUE(op->validate(in, int_out) == Status::invalid_argument);
        NNOPS_EXPECT_TRUE(op->validate(in, fp_out) == Status::ok);
    }
}

// ============================================================
// compute(): the view-level rules and the per-op contracts
// ============================================================

NNOPS_TEST(status_input_count) {
    // Every op counts its tensors before it looks at them. An empty view is a
    // valid "wrong count" probe: the count decides, nothing is dereferenced.
    const TensorView none;
    TensorView out[2] = {none, none};
    const TensorView in1[1] = {none};
    const TensorView in2[2] = {none, none};
    const TensorView in3[3] = {none, none, none};
    const TensorView in4[4] = {none, none, none, none};
    const TensorView in9[9] = {none, none, none, none, none,
                               none, none, none, none};
    const std::span<TensorView> one_out(out, 1);

    // Fixed arity for Activation (1) and Eltwise (2); MatMul and Conv2D also
    // take an optional bias.
    {
        auto op = Activation::create(ActivationAttributes{}, Backend::CPU);
        NNOPS_EXPECT_TRUE(op->compute(one_out, std::span<const TensorView>(in2)) ==
                          Status::invalid_argument);
    }
    {
        auto op = Eltwise::create(EltwiseAttributes{}, Backend::CPU);
        NNOPS_EXPECT_TRUE(op->compute(one_out, std::span<const TensorView>(in1)) ==
                          Status::invalid_argument);
    }
    {
        auto op = MatMul::create(MatMulAttributes{}, Backend::CPU);
        NNOPS_EXPECT_TRUE(op->compute(one_out, std::span<const TensorView>(in1)) ==
                          Status::invalid_argument);
        NNOPS_EXPECT_TRUE(op->compute(one_out, std::span<const TensorView>(in4)) ==
                          Status::invalid_argument);
    }
    {
        auto op = Conv2D::create(Conv2DAttributes{}, Backend::CPU);
        NNOPS_EXPECT_TRUE(op->compute(one_out, std::span<const TensorView>(in1)) ==
                          Status::invalid_argument);
        NNOPS_EXPECT_TRUE(op->compute(one_out, std::span<const TensorView>(in4)) ==
                          Status::invalid_argument);
    }
    {
        // Concat is variadic but needs at least two.
        auto op = Concat::create(ConcatAttributes{}, Backend::CPU);
        NNOPS_EXPECT_TRUE(op->compute(one_out, std::span<const TensorView>(in1)) ==
                          Status::invalid_argument);
    }
    {
        // TopK writes values and indices: one output is short.
        auto op = TopK::create(TopKAttributes{}, Backend::CPU);
        NNOPS_EXPECT_TRUE(op->compute(one_out, std::span<const TensorView>(in1)) ==
                          Status::invalid_argument);
    }
    {
        // Attention: Q, K, V [, mask].
        auto op = Attention::create(AttentionAttributes{}, Backend::CPU);
        NNOPS_EXPECT_TRUE(op->compute(one_out, std::span<const TensorView>(in2)) ==
                          Status::invalid_argument);
    }
    {
        // MoE: at least the five core inputs.
        auto op = MoE::create(MoEAttributes{}, Backend::CPU);
        NNOPS_EXPECT_TRUE(op->compute(one_out, std::span<const TensorView>(in4)) ==
                          Status::invalid_argument);
        NNOPS_EXPECT_TRUE(op->compute(one_out, std::span<const TensorView>(in9)) ==
                          Status::invalid_argument);
    }
}

NNOPS_TEST(status_empty_view) {
    // Right count, placeholder data: a descriptor may legitimately carry none
    // (an optional input, a plan), but a call that reaches a kernel may not.
    const int64_t shape[] = {4};
    float buf[4] = {};
    TensorView live(shape, DataType::f32, buf);
    TensorView empty;

    auto op = Activation::create(ActivationAttributes{}, Backend::CPU);
    TensorView out[1] = {live};
    const TensorView in_empty[1] = {empty};
    const TensorView in_live[1] = {live};
    const std::span<TensorView> outs(out, 1);

    NNOPS_EXPECT_TRUE(op->compute(outs, std::span<const TensorView>(in_empty)) ==
                      Status::invalid_argument);

    TensorView empty_out[1] = {empty};
    const std::span<TensorView> empty_outs(empty_out, 1);
    NNOPS_EXPECT_TRUE(op->compute(empty_outs, std::span<const TensorView>(in_live)) ==
                      Status::invalid_argument);

    // The same shapes with data on both sides run.
    NNOPS_EXPECT_TRUE(op->compute(outs, std::span<const TensorView>(in_live)) ==
                      Status::ok);
}

NNOPS_TEST(status_topk_rules) {
    // k and the axis are descriptor-level rules: validate() catches them, so a
    // release build never reads past the axis or sizes the output wrongly.
    const TensorDesc in[] = {desc_of({2, 5})};
    const TensorDesc out[] = {desc_of({2, 1}), desc_of({2, 1})};

    TopKAttributes attrs;
    attrs.k = 0;
    NNOPS_EXPECT_TRUE(TopK::create(attrs, Backend::CPU)->validate(in, out) ==
                      Status::invalid_argument);

    attrs.k = 6;   // the last axis holds 5
    NNOPS_EXPECT_TRUE(TopK::create(attrs, Backend::CPU)->validate(in, out) ==
                      Status::invalid_argument);

    attrs.k = 5;
    attrs.axis = 2;   // rank is 2
    NNOPS_EXPECT_TRUE(TopK::create(attrs, Backend::CPU)->validate(in, out) ==
                      Status::invalid_argument);

    attrs.axis = -1;   // wraps to axis 1
    attrs.k = 3;
    const TensorDesc out3[] = {desc_of({2, 3}), desc_of({2, 3})};
    NNOPS_EXPECT_TRUE(TopK::create(attrs, Backend::CPU)->validate(in, out3) ==
                      Status::ok);
}

NNOPS_TEST(status_unsupported_dtype) {
    // A well-formed call with no kernel is "unsupported", not malformed.
    {
        auto op = MatMul::create(MatMulAttributes{}, Backend::CPU);
        // A is f32 and B is s32: the fp route wants B in A's type (or a block
        // type), and the int8 route wants A in int8. Neither applies.
        const int64_t a_shape[] = {4, 8};
        const int64_t b_shape[] = {8, 16};
        const int64_t o_shape[] = {4, 16};
        float a[32] = {};
        std::vector<int32_t> b(128, 0);
        float out_buf[64] = {};
        TensorView a_v(a_shape, DataType::f32, a);
        TensorView b_v(b_shape, DataType::s32, b.data());
        TensorView out_v(o_shape, DataType::f32, out_buf);
        const TensorView in[] = {a_v, b_v};
        NNOPS_EXPECT_TRUE(op->compute(out_v, std::span<const TensorView>(in)) ==
                          Status::unsupported);
    }
    {
        // Embed with an s32 weight: valid shape, no kernel.
        auto op = Embed::create(EmbedAttributes{}, Backend::CPU);
        const int64_t w_shape[] = {8, 4};
        const int64_t i_shape[] = {2};
        const int64_t o_shape[] = {2, 4};
        std::vector<int32_t> w(32, 0);
        int64_t idx[2] = {0, 1};
        float out_buf[8] = {};
        TensorView w_v(w_shape, DataType::s32, w.data());
        TensorView i_v(i_shape, DataType::s64, idx);
        TensorView o_v(o_shape, DataType::f32, out_buf);
        const TensorView in[] = {w_v, i_v};
        NNOPS_EXPECT_TRUE(op->compute(o_v, std::span<const TensorView>(in)) ==
                          Status::unsupported);
    }
}

NNOPS_TEST(status_missing_backend_kernel) {
    // An op for a backend with no kernel is constructible; the failure surfaces
    // at compute() as unsupported. (Vulkan has kernels for eltwise and unary only.)
    const int64_t shape[] = {4};
    float buf[4] = {};
    TensorView live(shape, DataType::f32, buf);

    auto vulkan_op = Activation::create(ActivationAttributes{}, Backend::Vulkan);
    NNOPS_EXPECT_EQ(vulkan_op->getBackend(), Backend::Vulkan);
    const TensorView in[1] = {live};
    NNOPS_EXPECT_TRUE(vulkan_op->compute(live, std::span<const TensorView>(in)) ==
                      Status::unsupported);

    // The CPU instance of the same op runs.
    auto cpu_op = Activation::create(ActivationAttributes{}, Backend::CPU);
    NNOPS_EXPECT_TRUE(cpu_op->compute(live, std::span<const TensorView>(in)) ==
                      Status::ok);
}

NNOPS_TEST(status_output_untouched_on_error) {
    // A rejected call leaves the output exactly as it found it, so a caller can
    // retry with a corrected call without re-zeroing the destination.
    auto op = MatMul::create(MatMulAttributes{}, Backend::CPU);
    const int64_t a_shape[] = {4, 8};
    const int64_t b_shape[] = {16, 32};   // K mismatch: 8 vs 16
    const int64_t o_shape[] = {4, 32};

    float a[32], out_buf[128];
    float b[16 * 32];
    for (float& v : a) { v = 1.0f; }
    for (float& v : b) { v = 1.0f; }
    for (float& v : out_buf) { v = -7.0f; }

    TensorView a_v(a_shape, DataType::f32, a);
    TensorView b_v(b_shape, DataType::f32, b);
    TensorView out_v(o_shape, DataType::f32, out_buf);
    const TensorView in[] = {a_v, b_v};

    NNOPS_EXPECT_TRUE(op->compute(out_v, std::span<const TensorView>(in)) ==
                      Status::invalid_argument);
    for (float v : out_buf) { NNOPS_EXPECT_EQ(v, -7.0f); }
}

NNOPS_TEST(status_valid_calls_return_ok) {
    // The happy path of the ops whose checks are more than a count: the same
    // descriptors that validate() accepted must run clean.
    {
        auto op = MatMul::create(MatMulAttributes{}, Backend::CPU);
        const int64_t a_shape[] = {4, 8};
        const int64_t b_shape[] = {8, 16};
        const int64_t o_shape[] = {4, 16};
        float a[32], b[128], o[64];
        for (float& v : a) { v = 1.0f; }
        for (float& v : b) { v = 1.0f; }
        for (float& v : o) { v = 0.0f; }
        TensorView a_v(a_shape, DataType::f32, a);
        TensorView b_v(b_shape, DataType::f32, b);
        TensorView o_v(o_shape, DataType::f32, o);
        const TensorView in[] = {a_v, b_v};
        NNOPS_EXPECT_TRUE(op->compute(o_v, std::span<const TensorView>(in)) ==
                          Status::ok);
        for (float v : o) { NNOPS_EXPECT_NEAR(v, 8.0f, 1e-5f); }
    }
    {
        SliceAttributes attrs;
        attrs.axes = {1};
        attrs.starts = {1};
        attrs.ends = {3};
        auto op = Slice::create(attrs, Backend::CPU);
        const int64_t in_shape[] = {2, 6};
        const int64_t o_shape[] = {2, 2};
        float x[12], o[4];
        for (int i = 0; i < 12; ++i) { x[i] = static_cast<float>(i); }
        TensorView x_v(in_shape, DataType::f32, x);
        TensorView o_v(o_shape, DataType::f32, o);
        const TensorView in[] = {x_v};
        NNOPS_EXPECT_TRUE(op->compute(o_v, std::span<const TensorView>(in)) ==
                          Status::ok);
        NNOPS_EXPECT_EQ(o[0], 1.0f);
        NNOPS_EXPECT_EQ(o[1], 2.0f);
        NNOPS_EXPECT_EQ(o[2], 7.0f);
        NNOPS_EXPECT_EQ(o[3], 8.0f);
    }
    {
        // Permute along the same path.
        const int64_t p[] = {1, 0};
        PermuteAttributes attrs;
        attrs.perm = std::span<const int64_t>(p, 2);
        auto op = Permute::create(attrs, Backend::CPU);
        const int64_t in_shape[] = {2, 3};
        const int64_t o_shape[] = {3, 2};
        float x[6] = {0, 1, 2, 3, 4, 5};
        float o[6] = {};
        TensorView x_v(in_shape, DataType::f32, x);
        TensorView o_v(o_shape, DataType::f32, o);
        const TensorView in[] = {x_v};
        NNOPS_EXPECT_TRUE(op->compute(o_v, std::span<const TensorView>(in)) ==
                          Status::ok);
        const float expected[6] = {0, 3, 1, 4, 2, 5};
        for (int i = 0; i < 6; ++i) { NNOPS_EXPECT_EQ(o[i], expected[i]); }
    }
}
