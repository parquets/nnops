/// @file test_row_pitch.cpp
/// @brief "The second-innermost step is the row pitch" — every layout is
///        row-major, so an operator that walks rows must step by the pitch.
///
/// NHWC/NDHWC are Dense with the row redefined as one spatial line (W × C); the
/// pitch is the stride of the axis just outside that row — rank-2 for Dense,
/// rank-3 for a rank-4 channel-last view. When the row's byte size is a multiple
/// of the 32-byte alignment the two coincide, and a compact view is all a test
/// ever sees. When it is not (`TensorDesc::row_pitch()` rounds up), the step is
/// larger than the row's logical size, and an operator that computes the next
/// row's address from the logical size — softmax's contiguous fast path did —
/// reads or writes the wrong place, silently.
///
/// The first case pins the invariant on the view itself (stride_elems vs
/// pitch). Each following case builds the same logical tensor twice, compact
/// then with the row padded, runs one operator over both, and compares the
/// outputs element by element through the views' own strides.
///
/// The test harness is not flipped to padded views: most tests compute their
/// reference from the flat data vector (`data[r * cols + c]`), which is the
/// logical order only for a compact view. The coverage lives here instead.

#include "nnops/ops/activation.hpp"
#include "nnops/ops/argminmax.hpp"
#include "nnops/ops/clamp.hpp"
#include "nnops/ops/compare.hpp"
#include "nnops/ops/concat.hpp"
#include "nnops/ops/cumsum.hpp"
#include "nnops/ops/eltwise.hpp"
#include "nnops/ops/embed.hpp"
#include "nnops/ops/flatten.hpp"
#include "nnops/ops/gather.hpp"
#include "nnops/ops/grid_sample.hpp"
#include "nnops/ops/norm.hpp"
#include "nnops/ops/pad.hpp"
#include "nnops/ops/permute.hpp"
#include "nnops/ops/pixel_shuffle.hpp"
#include "nnops/ops/pooling.hpp"
#include "nnops/ops/quant_linear.hpp"
#include "nnops/ops/reduce.hpp"
#include "nnops/ops/resize.hpp"
#include "nnops/ops/scatter.hpp"
#include "nnops/ops/slice.hpp"
#include "nnops/ops/softmax.hpp"
#include "nnops/ops/split.hpp"
#include "nnops/ops/tile.hpp"
#include "nnops/ops/topk.hpp"
#include "nnops/ops/unary.hpp"
#include "nnops/ops/where.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

using namespace nnops;
using namespace nnops::test;

namespace {

using Bytes = std::vector<uint8_t>;

TensorDesc desc_of(std::initializer_list<int64_t> dims, DataType dtype = DataType::f32,
                   TensorLayout layout = TensorLayout::Dense)
{
    TensorDesc d;
    d.rank = static_cast<int64_t>(dims.size());
    for (int64_t v : dims) {
        d.dims.push_back(v);
    }
    d.dtype = dtype;
    d.layout = layout;
    return d;
}

int64_t elem_size(DataType d) { return static_cast<int64_t>(data_type_size(d)); }

/// One dense row's bytes — what the 4-argument TensorView ctor uses as the pitch.
int64_t dense_row_bytes(const TensorDesc& d)
{
    if (d.rank < 2) {
        return elem_size(d.dtype);
    }
    const int64_t row_elems = (is_channel_last(d.layout) && d.rank >= 3) ?
                                  d.dims[static_cast<size_t>(d.rank - 2)] *
                                      d.dims[static_cast<size_t>(d.rank - 1)] :
                                  d.dims[static_cast<size_t>(d.rank - 1)];
    return row_elems * elem_size(d.dtype);
}

/// A pitch deliberately different from the dense one, so the two arms cannot
/// coincide: the dense row plus a whole extra 32-byte step.
int64_t padded_pitch(const TensorDesc& d)
{
    // A rank-1 tensor has no row; one element is its storage unit.
    return d.rank < 2 ? dense_row_bytes(d) : dense_row_bytes(d) + 32;
}

struct Storage {
    std::vector<uint8_t> buf;
    TensorView view;
};

Storage make_storage(const TensorDesc& d, int64_t pitch_bytes)
{
    Storage s;
    std::vector<int64_t> shape(d.dims.begin(), d.dims.end());
    // Size the buffer from a null view at the same pitch.
    TensorView probe(std::span<const int64_t>(shape.data(), shape.size()), d.dtype, nullptr,
                     pitch_bytes, d.layout);
    s.buf.assign(probe.storage_bytes() + 64, uint8_t{0xA5});
    s.view = TensorView(std::span<const int64_t>(shape.data(), shape.size()), d.dtype,
                        s.buf.data(), pitch_bytes, d.layout);
    return s;
}

/// Byte offset of the element at `flat` (row-major over the logical shape).
int64_t logical_offset(const TensorView& v, int64_t flat)
{
    int64_t off = 0;
    int64_t rem = flat;
    for (int64_t d = v.rank() - 1; d >= 0; --d) {
        const int64_t n = v.shape(d);
        off += (rem % n) * v.stride_elems(d);
        rem /= n;
    }
    return off;
}

/// Write the logical elements (row-major, `elem_bytes` each) into the view.
void fill_logical(TensorView& v, const Bytes& logical)
{
    const int64_t es = elem_size(v.data_type());
    uint8_t* base = static_cast<uint8_t*>(v.ptr<void>());
    for (int64_t i = 0; i < v.numel(); ++i) {
        std::memcpy(base + logical_offset(v, i) * es,
                    logical.data() + static_cast<size_t>(i) * static_cast<size_t>(es),
                    static_cast<size_t>(es));
    }
}

/// Read the logical elements back out of the view.
Bytes read_logical(const TensorView& v)
{
    const int64_t es = elem_size(v.data_type());
    Bytes out(static_cast<size_t>(v.numel() * es));
    const uint8_t* base = static_cast<const uint8_t*>(v.ptr<void>());
    for (int64_t i = 0; i < v.numel(); ++i) {
        std::memcpy(out.data() + static_cast<size_t>(i) * static_cast<size_t>(es),
                    base + logical_offset(v, i) * es, static_cast<size_t>(es));
    }
    return out;
}

Bytes f32_bytes(const std::vector<float>& v)
{
    Bytes b(v.size() * sizeof(float));
    std::memcpy(b.data(), v.data(), b.size());
    return b;
}

Bytes s64_bytes(const std::vector<int64_t>& v)
{
    Bytes b(v.size() * sizeof(int64_t));
    std::memcpy(b.data(), v.data(), b.size());
    return b;
}

Bytes s32_bytes(const std::vector<int32_t>& v)
{
    Bytes b(v.size() * sizeof(int32_t));
    std::memcpy(b.data(), v.data(), b.size());
    return b;
}

/// Compare two logical byte runs as `dtype`, reporting the first mismatch.
void expect_same(const Bytes& dense, const Bytes& padded, DataType dtype, const char* what,
                 double tol)
{
    const size_t es = static_cast<size_t>(elem_size(dtype));
    if (dense.size() != padded.size()) {
        throw std::runtime_error(std::string(what) + ": size mismatch");
    }
    for (size_t i = 0; i < dense.size() / es; ++i) {
        double a = 0.0, b = 0.0;
        switch (dtype) {
        case DataType::f32: {
            float fa, fb;
            std::memcpy(&fa, dense.data() + i * 4, 4);
            std::memcpy(&fb, padded.data() + i * 4, 4);
            a = fa;
            b = fb;
            break;
        }
        case DataType::f16:
        case DataType::bf16: {
            uint16_t ha, hb;
            std::memcpy(&ha, dense.data() + i * 2, 2);
            std::memcpy(&hb, padded.data() + i * 2, 2);
            a = static_cast<double>(ha);
            b = static_cast<double>(hb);
            break;
        }
        case DataType::s64: {
            int64_t ia, ib;
            std::memcpy(&ia, dense.data() + i * 8, 8);
            std::memcpy(&ib, padded.data() + i * 8, 8);
            if (ia != ib) {
                throw std::runtime_error(std::string(what) + ": index " + std::to_string(i) +
                                         " dense=" + std::to_string(ia) +
                                         " padded=" + std::to_string(ib));
            }
            continue;
        }
        case DataType::s32: {
            int32_t ia, ib;
            std::memcpy(&ia, dense.data() + i * 4, 4);
            std::memcpy(&ib, padded.data() + i * 4, 4);
            if (ia != ib) {
                throw std::runtime_error(std::string(what) + ": element " + std::to_string(i) +
                                         " dense=" + std::to_string(ia) +
                                         " padded=" + std::to_string(ib));
            }
            continue;
        }
        default: {
            if (std::memcmp(dense.data() + i * es, padded.data() + i * es, es) != 0) {
                throw std::runtime_error(std::string(what) + ": byte mismatch at " +
                                         std::to_string(i));
            }
            continue;
        }
        }
        const double diff = std::fabs(a - b);
        const double scale = std::max(1.0, std::fabs(a));
        if (!(diff <= tol * scale)) {
            throw std::runtime_error(std::string(what) + ": element " + std::to_string(i) +
                                     " dense=" + std::to_string(a) +
                                     " padded=" + std::to_string(b));
        }
    }
}

/// Run one operator over the same logical inputs twice — dense views, then
/// pitch-padded views — and compare every output's logical elements.
template <class Fn>
void expect_pitched_same(const char* what, const std::vector<TensorDesc>& ins,
                         const std::vector<TensorDesc>& outs, const std::vector<Bytes>& in_bytes,
                         Fn&& fn, double tol = 1e-6)
{
    std::vector<Storage> dense_in, pad_in, dense_out, pad_out;
    for (const auto& d : ins) {
        dense_in.push_back(make_storage(d, dense_row_bytes(d)));
        pad_in.push_back(make_storage(d, padded_pitch(d)));
    }
    for (const auto& d : outs) {
        dense_out.push_back(make_storage(d, dense_row_bytes(d)));
        pad_out.push_back(make_storage(d, padded_pitch(d)));
    }

    const auto arm = [&](std::vector<Storage>& iv, std::vector<Storage>& ov) {
        for (size_t i = 0; i < iv.size(); ++i) {
            fill_logical(iv[i].view, in_bytes[i]);
        }
        for (auto& o : ov) {
            std::fill(o.buf.begin(), o.buf.end(), uint8_t{0xA5});
        }
        std::vector<TensorView> iviews, oviews;
        for (auto& s : iv) {
            iviews.push_back(s.view);
        }
        for (auto& s : ov) {
            oviews.push_back(s.view);
        }
        return fn(std::span<const TensorView>(iviews), std::span<TensorView>(oviews));
    };

    const Status dense_st = arm(dense_in, dense_out);
    const Status pad_st = arm(pad_in, pad_out);
    if (dense_st != Status::ok) {
        throw std::runtime_error(std::string(what) + ": compact view refused, status " +
                                 std::to_string(static_cast<int>(dense_st)));
    }
    // A padded view may sit outside an operator's contract — quant_linear's
    // kernels walk flat rows with one stride — but then it must say so rather
    // than write through the wrong stride.
    if (pad_st != Status::ok) {
        if (pad_st != Status::unsupported) {
            throw std::runtime_error(std::string(what) + ": padded view status " +
                                     std::to_string(static_cast<int>(pad_st)) +
                                     " (expected unsupported)");
        }
        return;
    }

    for (size_t i = 0; i < outs.size(); ++i) {
        expect_same(read_logical(dense_out[i].view), read_logical(pad_out[i].view),
                    outs[i].dtype, what, tol);
    }
}

std::vector<float> ramp(int64_t n, float lo, float hi)
{
    std::vector<float> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const float t = (n > 1) ? static_cast<float>(i) / static_cast<float>(n - 1) : 0.0f;
        v[static_cast<size_t>(i)] = lo + (hi - lo) * t;
    }
    return v;
}

} // namespace

// ============================================================
// The invariant: the row pitch is the step of the axis outside the row
// ============================================================

NNOPS_TEST(row_pitch_is_the_second_innermost_step) {
    // Dense: the row is the last axis, so the second-innermost step is the pitch.
    {
        const int64_t shape[] = {3, 5};
        std::vector<float> buf(static_cast<size_t>(3 * 16), 0.0f);
        const int64_t pitch_bytes = 5 * 4 + 32;  // one padded row plus a step
        TensorView v(std::span<const int64_t>(shape, 2), DataType::f32, buf.data(), pitch_bytes);
        NNOPS_EXPECT_EQ(v.pitch(), pitch_bytes);
        NNOPS_EXPECT_EQ(v.stride_elems(1), int64_t{1});
        NNOPS_EXPECT_EQ(v.stride_elems(0), pitch_bytes / 4);
        // row r starts exactly one pitch further along
        NNOPS_EXPECT_TRUE(v.ptr<float>(1) == v.ptr<float>(0) + pitch_bytes / 4);
    }
    // NHWC: the row is one spatial line (W × C), so the pitch is the H step and
    // the W step is C.
    {
        const int64_t shape[] = {1, 4, 5, 3};
        std::vector<float> buf(static_cast<size_t>(4 * 32), 0.0f);
        const int64_t pitch_bytes = 5 * 3 * 4 + 32;
        TensorView v(std::span<const int64_t>(shape, 4), DataType::f32, buf.data(), pitch_bytes,
                     TensorLayout::NHWC);
        NNOPS_EXPECT_EQ(v.stride_elems(3), int64_t{1});
        NNOPS_EXPECT_EQ(v.stride_elems(2), int64_t{3});          // W steps by C
        NNOPS_EXPECT_EQ(v.stride_elems(1), pitch_bytes / 4);     // H steps by the pitch
        NNOPS_EXPECT_EQ(v.stride_elems(0), int64_t{4} * (pitch_bytes / 4));
    }
    // The default ctor is the compact case: pitch == exactly one row.
    {
        const int64_t shape[] = {2, 7};
        std::vector<float> buf(static_cast<size_t>(14), 0.0f);
        TensorView v(std::span<const int64_t>(shape, 2), DataType::f32, buf.data());
        NNOPS_EXPECT_EQ(v.pitch(), int64_t{7 * 4});
        NNOPS_EXPECT_EQ(v.stride_elems(0), int64_t{7});
        NNOPS_EXPECT_EQ(v.stride_elems(1), int64_t{1});
    }
}

// ============================================================
// Element-wise family
// ============================================================

NNOPS_TEST(pitched_unary_activation_clamp) {
    // A Dense [3,5] f32 row is 20 B, so the padded arm's rows jump 32 B further.
    const auto in = desc_of({3, 5});
    const auto out = desc_of({3, 5});
    const Bytes data = f32_bytes(ramp(15, -2.0f, 2.0f));

    {
        UnaryAttributes attrs;
        attrs.type = UnaryType::Exp;
        auto op = Unary::create(attrs, Backend::CPU);
        expect_pitched_same("unary exp", {in}, {out}, {data},
                            [&](auto i, auto o) { return op->compute(o[0], i); });
    }
    {
        ActivationAttributes attrs;
        attrs.type = ActivationType::Gelu;
        auto op = Activation::create(attrs, Backend::CPU);
        expect_pitched_same("activation gelu", {in}, {out}, {data},
                            [&](auto i, auto o) { return op->compute(o[0], i); });
    }
    {
        ClampAttributes attrs;
        attrs.min_val = -0.5f;
        attrs.max_val = 0.5f;
        auto op = Clamp::create(attrs, Backend::CPU);
        expect_pitched_same("clamp", {in}, {out}, {data},
                            [&](auto i, auto o) { return op->compute(o[0], i); });
    }
}

NNOPS_TEST(pitched_eltwise_broadcast_where_compare) {
    const auto a_desc = desc_of({3, 4, 5});
    const auto b_desc = desc_of({5});
    const auto out_desc = desc_of({3, 4, 5});
    const Bytes a = f32_bytes(ramp(60, -1.0f, 1.0f));
    const Bytes b = f32_bytes(ramp(5, 0.5f, 1.5f));

    {
        EltwiseAttributes attrs;
        attrs.type = EltwiseType::Add;
        auto op = Eltwise::create(attrs, Backend::CPU);
        expect_pitched_same("eltwise add [3,4,5]+[5]", {a_desc, b_desc}, {out_desc}, {a, b},
                            [&](auto i, auto o) { return op->compute(o[0], i); });
    }
    {
        CompareAttributes attrs;
        attrs.type = CompareType::Greater;
        auto op = Compare::create(attrs, Backend::CPU);
        expect_pitched_same("compare >", {a_desc, b_desc}, {desc_of({3, 4, 5}, DataType::u8)},
                            {a, b}, [&](auto i, auto o) { return op->compute(o[0], i); });
    }
    {
        // A broadcast mask over a pitched Dense tensor.
        std::vector<uint8_t> mask(5);
        for (size_t i = 0; i < mask.size(); ++i) {
            mask[i] = static_cast<uint8_t>(i & 1u);
        }
        WhereAttributes attrs;
        auto op = Where::create(attrs, Backend::CPU);
        expect_pitched_same("where mask[1,1,5]", {desc_of({1, 1, 5}, DataType::u8), a_desc, a_desc},
                            {out_desc}, {mask, a, a}, [&](auto i, auto o) { return op->compute(o[0], i); });
    }
}

// ============================================================
// Reductions and normalization (the walk reads whole rows)
// ============================================================

NNOPS_TEST(pitched_reduce_argmax_cumsum) {
    const auto in = desc_of({3, 6});
    const Bytes data = f32_bytes(ramp(18, -3.0f, 3.0f));

    {
        ReduceAttributes attrs;
        attrs.axes = {1};
        attrs.type = ReduceType::Sum;
        attrs.keepdims = true;
        auto op = Reduce::create(attrs, Backend::CPU);
        expect_pitched_same("reduce sum axis1", {in}, {desc_of({3, 1})}, {data},
                            [&](auto i, auto o) { return op->compute(o[0], i); });
    }
    {
        ReduceAttributes attrs;
        attrs.axes = {0};
        attrs.type = ReduceType::Max;
        attrs.keepdims = true;
        auto op = Reduce::create(attrs, Backend::CPU);
        expect_pitched_same("reduce max axis0", {in}, {desc_of({1, 6})}, {data},
                            [&](auto i, auto o) { return op->compute(o[0], i); });
    }
    {
        ArgMinMaxAttributes attrs;
        attrs.type = ArgMinMaxType::Max;
        attrs.axis = -1;
        attrs.keepdims = true;
        auto op = ArgMax::create(attrs, Backend::CPU);
        expect_pitched_same("argmax axis -1", {in}, {desc_of({3, 1}, DataType::s64)}, {data},
                            [&](auto i, auto o) { return op->compute(o[0], i); });
    }
    {
        CumSumAttributes attrs;
        attrs.axis = -1;
        auto op = CumSum::create(attrs, Backend::CPU);
        expect_pitched_same("cumsum axis -1", {in}, {in}, {data},
                            [&](auto i, auto o) { return op->compute(o[0], i); }, 1e-6);
    }
}

NNOPS_TEST(pitched_norm_softmax) {
    // NHWC [1,2,3,4]: a row is W*C = 12 elements = 48 B, so the padded arm is
    // 48 B rows at an 80 B stride.
    const auto x_desc = desc_of({1, 2, 3, 4}, DataType::f32, TensorLayout::NHWC);
    const Bytes x = f32_bytes(ramp(24, -1.0f, 1.0f));
    const Bytes scale = f32_bytes(std::vector<float>(4, 1.0f));
    const Bytes bias = f32_bytes(std::vector<float>(4, 0.0f));
    const auto param = desc_of({4});

    {
        NormAttributes attrs;
        attrs.type = NormType::LayerNorm;
        attrs.axis = -1;
        auto op = Norm::create(attrs, Backend::CPU);
        expect_pitched_same("layer_norm axis -1", {x_desc, param, param}, {x_desc},
                            {x, scale, bias}, [&](auto i, auto o) { return op->compute(o[0], i); },
                            1e-5);
    }
    {
        SoftmaxAttributes attrs;
        attrs.axis = -1;
        auto op = Softmax::create(attrs, Backend::CPU);
        expect_pitched_same("softmax axis -1 (NHWC)", {x_desc}, {x_desc}, {x},
                            [&](auto i, auto o) { return op->compute(o[0], i); }, 1e-5);
    }
    {
        // The same softmax on a Dense view: the contiguous fast path needs a
        // compact tensor, and this one is padded.
        const auto xd = desc_of({2, 3, 4});
        const Bytes xb = f32_bytes(ramp(24, -1.0f, 1.0f));
        SoftmaxAttributes attrs;
        attrs.axis = -1;
        auto op = Softmax::create(attrs, Backend::CPU);
        expect_pitched_same("softmax axis -1 (Dense)", {xd}, {xd}, {xb},
                            [&](auto i, auto o) { return op->compute(o[0], i); }, 1e-5);
    }
}

// ============================================================
// Re-layout family: the walk writes through the output's own strides
// ============================================================

NNOPS_TEST(pitched_flatten_slice_permute) {
    // Flatten: NHWC [1,2,3,4] onto dense storage of the same dims.
    {
        const auto in = desc_of({1, 2, 3, 4}, DataType::f32, TensorLayout::NHWC);
        const auto out = desc_of({1, 2, 3, 4});
        FlattenAttributes attrs;
        auto op = Flatten::create(attrs, Backend::CPU);
        expect_pitched_same("flatten NHWC->dense", {in}, {out}, {f32_bytes(ramp(24, -1, 1))},
                            [&](auto i, auto o) { return op->compute(o[0], i); });
    }
    // Slice on a padded Dense tensor, into a padded output.
    {
        const auto in = desc_of({3, 7});
        const auto out = desc_of({3, 3});
        SliceAttributes attrs;
        attrs.axes = {1};
        attrs.starts = {2};
        attrs.ends = {5};
        auto op = Slice::create(attrs, Backend::CPU);
        expect_pitched_same("slice cols 2..4", {in}, {out}, {f32_bytes(ramp(21, -1, 1))},
                            [&](auto i, auto o) { return op->compute(o[0], i); });
    }
    // A general permutation (not the last-two swap) with a padded output.
    {
        const auto in = desc_of({2, 3, 4});
        const auto out = desc_of({3, 2, 4});
        PermuteAttributes attrs;
        attrs.perm = {1, 0, 2};
        auto op = Permute::create(attrs, Backend::CPU);
        expect_pitched_same("permute [1,0,2]", {in}, {out}, {f32_bytes(ramp(24, -1, 1))},
                            [&](auto i, auto o) { return op->compute(o[0], i); });
    }
}

NNOPS_TEST(pitched_copy_family) {
    const Bytes a = f32_bytes(ramp(12, -1.0f, 1.0f));

    // Concat along the last axis of a padded NHWC view.
    {
        const auto in = desc_of({1, 2, 3, 2}, DataType::f32, TensorLayout::NHWC);
        const auto out = desc_of({1, 2, 3, 4}, DataType::f32, TensorLayout::NHWC);
        ConcatAttributes attrs;
        attrs.axis = -1;
        auto op = Concat::create(attrs, Backend::CPU);
        expect_pitched_same("concat axis -1 (NHWC)", {in, in}, {out}, {a, a},
                            [&](auto i, auto o) { return op->compute(o[0], i); });
    }
    // Split along the last axis.
    {
        const auto in = desc_of({1, 2, 3, 4}, DataType::f32, TensorLayout::NHWC);
        const auto out = desc_of({1, 2, 3, 2}, DataType::f32, TensorLayout::NHWC);
        SplitAttributes attrs;
        attrs.axis = -1;
        attrs.split = {2, 2};
        auto op = Split::create(attrs, Backend::CPU);
        expect_pitched_same("split axis -1 (NHWC)", {in}, {out, out},
                            {f32_bytes(ramp(24, -1, 1))},
                            [&](auto i, auto o) { return op->compute(o, i); });
    }
    // Tile along H.
    {
        const auto in = desc_of({1, 2, 3, 2}, DataType::f32, TensorLayout::NHWC);
        const auto out = desc_of({1, 4, 3, 2}, DataType::f32, TensorLayout::NHWC);
        TileAttributes attrs;
        attrs.repeats = {1, 2, 1, 1};
        auto op = Tile::create(attrs, Backend::CPU);
        expect_pitched_same("tile H x2 (NHWC)", {in}, {out}, {a},
                            [&](auto i, auto o) { return op->compute(o[0], i); });
    }
    // Pad one pixel on H and W.
    {
        const auto in = desc_of({1, 2, 3, 2}, DataType::f32, TensorLayout::NHWC);
        const auto out = desc_of({1, 4, 5, 2}, DataType::f32, TensorLayout::NHWC);
        PadAttributes attrs;
        attrs.pads = {0, 1, 1, 0, 0, 1, 1, 0};
        attrs.mode = PadAttributes::Mode::Constant;
        attrs.constant_value = 0.5f;
        auto op = Pad::create(attrs, Backend::CPU);
        expect_pitched_same("pad 1px (NHWC)", {in}, {out}, {a},
                            [&](auto i, auto o) { return op->compute(o[0], i); });
    }
}

NNOPS_TEST(pitched_gather_scatter_embed_pixel_shuffle) {
    // Gather axis 0 of a padded table.
    {
        const auto table = desc_of({6, 5});
        const std::vector<int32_t> idx = {3, 0, 5};
        const auto indices = desc_of({3}, DataType::s32);
        const auto out = desc_of({3, 5});
        GatherAttributes attrs;
        attrs.axis = 0;
        auto op = Gather::create(attrs, Backend::CPU);
        expect_pitched_same("gather axis 0", {table, indices}, {out},
                            {f32_bytes(ramp(30, -1, 1)), s32_bytes(idx)},
                            [&](auto i, auto o) { return op->compute(o[0], i); });
    }
    // ScatterND over rows of a padded table.
    {
        const auto data = desc_of({5, 5});
        const std::vector<int64_t> idx = {1, 3};
        const auto indices = desc_of({2, 1}, DataType::s64);
        const auto updates = desc_of({2, 5});
        ScatterNDAttributes attrs;
        auto op = ScatterND::create(attrs, Backend::CPU);
        expect_pitched_same("scatter_nd rows", {data, indices, updates}, {data},
                            {f32_bytes(ramp(25, -1, 1)), s64_bytes(idx),
                             f32_bytes(ramp(10, 5, 9))},
                            [&](auto i, auto o) { return op->compute(o[0], i); });
    }
    // Embedding lookup out of a padded weight table.
    {
        const auto weight = desc_of({6, 5});
        const std::vector<int32_t> idx = {4, 0};
        const auto indices = desc_of({2}, DataType::s32);
        const auto out = desc_of({2, 5});
        EmbedAttributes attrs;
        auto op = Embed::create(attrs, Backend::CPU);
        expect_pitched_same("embed", {weight, indices}, {out},
                            {f32_bytes(ramp(30, -1, 1)), s32_bytes(idx)},
                            [&](auto i, auto o) { return op->compute(o[0], i); });
    }
    // PixelShuffle b=2 on a padded NHWC view.
    {
        const auto in = desc_of({1, 2, 2, 8}, DataType::f32, TensorLayout::NHWC);
        const auto out = desc_of({1, 4, 4, 2}, DataType::f32, TensorLayout::NHWC);
        PixelShuffleAttributes attrs;
        attrs.block_size = 2;
        attrs.mode = PixelShuffleMode::Dcr;
        auto op = PixelShuffle::create(attrs, Backend::CPU);
        expect_pitched_same("pixel_shuffle b2", {in}, {out}, {f32_bytes(ramp(32, -1, 1))},
                            [&](auto i, auto o) { return op->compute(o[0], i); });
    }
}

// ============================================================
// Spatial and quantized tails
// ============================================================

NNOPS_TEST(pitched_resize_grid_sample_pool) {
    const auto in = desc_of({1, 3, 4, 6}, DataType::f32, TensorLayout::NHWC);
    const Bytes x = f32_bytes(ramp(72, -1.0f, 1.0f));

    {
        const auto out = desc_of({1, 6, 8, 6}, DataType::f32, TensorLayout::NHWC);
        ResizeAttributes attrs;
        attrs.mode = ResizeMode::Linear;
        attrs.coord_mode = CoordinateTransformMode::HalfPixel;
        auto op = Resize::create(attrs, Backend::CPU);
        expect_pitched_same("resize bilinear x2", {in}, {out}, {x},
                            [&](auto i, auto o) { return op->compute(o[0], i); }, 1e-6);
    }
    {
        const auto out = desc_of({1, 3, 4, 6}, DataType::f32, TensorLayout::NHWC);
        // The grid matches the output's spatial dims [OH=3, OW=4].
        std::vector<float> grid(static_cast<size_t>(3 * 4 * 2));
        for (size_t i = 0; i < grid.size(); ++i) {
            grid[i] = ((i & 1u) ? 0.7f : -0.6f);
        }
        GridSampleAttributes attrs;
        attrs.mode = GridSampleMode::Bilinear;
        auto op = GridSample::create(attrs, Backend::CPU);
        expect_pitched_same("grid_sample bilinear", {in, desc_of({1, 3, 4, 2})}, {out},
                            {x, f32_bytes(grid)},
                            [&](auto i, auto o) { return op->compute(o[0], i); }, 1e-6);
    }
    {
        const auto out = desc_of({1, 2, 2, 6}, DataType::f32, TensorLayout::NHWC);
        PoolingAttributes attrs;
        attrs.type = PoolingType::Max;
        attrs.kernel_shape = {2, 2, 2};
        attrs.stride = {2, 2, 2};
        auto op = Pooling::create(attrs, Backend::CPU);
        expect_pitched_same("max_pool 2x2 s2", {in}, {out}, {x},
                            [&](auto i, auto o) { return op->compute(o[0], i); });
    }
}

NNOPS_TEST(pitched_quant_linear_topk) {
    // DequantizeLinear per-token out of a padded s8 view.
    {
        const int64_t M = 3, N = 5;
        std::vector<int8_t> q(static_cast<size_t>(M * N));
        for (size_t i = 0; i < q.size(); ++i) {
            q[i] = static_cast<int8_t>(static_cast<int>(i) - 7);
        }
        Bytes qb(q.size());
        std::memcpy(qb.data(), q.data(), qb.size());
        const Bytes scale = f32_bytes(std::vector<float>{0.5f, 0.25f, 0.125f});
        std::vector<int8_t> zp(static_cast<size_t>(M), 0);
        Bytes zpb(zp.size());
        std::memcpy(zpb.data(), zp.data(), zpb.size());

        const auto qdesc = desc_of({M, N}, DataType::s8);
        const auto sdesc = desc_of({M});
        const auto odesc = desc_of({M, N});
        QuantLinearAttributes attrs;
        attrs.axis = -1;
        auto op = DequantizeLinear::create(attrs, Backend::CPU);
        expect_pitched_same("dequantize per-token", {qdesc, sdesc, qdesc}, {odesc},
                            {qb, scale, zpb}, [&](auto i, auto o) { return op->compute(o[0], i); });
    }
    // TopK over the last axis of a padded Dense tensor.
    {
        const auto in = desc_of({2, 6});
        const auto vout = desc_of({2, 3});
        const auto iout = desc_of({2, 3}, DataType::s64);
        TopKAttributes attrs;
        attrs.type = TopKType::Max;
        attrs.axis = -1;
        attrs.k = 3;
        attrs.sorted = true;
        auto op = TopK::create(attrs, Backend::CPU);
        expect_pitched_same("topk 3 of 6", {in}, {vout, iout}, {f32_bytes(ramp(12, -1, 1))},
                            [&](auto i, auto o) { return op->compute(o, i); });
    }
}
