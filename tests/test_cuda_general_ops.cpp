/// @file test_cuda_general_ops.cpp
/// @brief The general (non-conv/attention/recurrent) CUDA operators against the
///        CPU backend on the same bytes.
///
/// One runner covers them all: build the host operands, ask the CPU op for the
/// output descriptors, run the CPU route as the oracle and the CUDA route on
/// uploaded copies, then compare every output element widened through its own
/// dtype (the shared `expect_same` widens with the *first* operand's dtype and
/// cannot compare a u8 mask or an s64 index).
///
/// The cases pin the contracts that differ from a plain copy: Cast's saturating
/// truncation and NaN-to-zero, Compare's NaN semantics and broadcasting,
/// Where's condition broadcast, the two pixel-shuffle channel orders, Embed's
/// index clamp and per-row dequant, and the three quantization granularities.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/nnops.hpp"
#include "nnops/ops/cast.hpp"   // not in the umbrella header
#include "nnops/ops/cumsum.hpp" // ditto
#include "nnops/ops/reduce.hpp"
#include "nnops/ops/softmax.hpp"
#include "nnops/ops/topk.hpp"
#include "nnops/ops/pool1d.hpp" // not in the umbrella header

#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

/// A dense host tensor of an integer dtype (make_host_random fills floats).
HostTensor make_host_int(std::span<const int64_t> shape, DataType dt, uint64_t seed,
                         int64_t lo, int64_t hi)
{
    HostTensor t;
    const TensorView probe(shape, dt, nullptr);
    t.bytes.assign(probe.storage_bytes(), std::byte{0});
    XorShift128 rng(seed);
    const int64_t n = probe.numel();
    if (dt == DataType::s8) {
        auto* p = reinterpret_cast<int8_t*>(t.bytes.data());
        for (int64_t i = 0; i < n; ++i) {
            p[i] = static_cast<int8_t>(lo + int64_t(rng.next_u64() % uint64_t(hi - lo + 1)));
        }
    } else if (dt == DataType::u8) {
        auto* p = reinterpret_cast<uint8_t*>(t.bytes.data());
        for (int64_t i = 0; i < n; ++i) {
            p[i] = static_cast<uint8_t>(lo + int64_t(rng.next_u64() % uint64_t(hi - lo + 1)));
        }
    } else if (dt == DataType::s32) {
        auto* p = reinterpret_cast<int32_t*>(t.bytes.data());
        for (int64_t i = 0; i < n; ++i) {
            p[i] = static_cast<int32_t>(lo + int64_t(rng.next_u64() % uint64_t(hi - lo + 1)));
        }
    } else {
        auto* p = reinterpret_cast<int64_t*>(t.bytes.data());
        for (int64_t i = 0; i < n; ++i) {
            p[i] = lo + int64_t(rng.next_u64() % uint64_t(hi - lo + 1));
        }
    }
    t.view = TensorView(shape, dt, t.bytes.data());
    return t;
}

inline HostTensor rand_t(std::initializer_list<int64_t> shape, DataType dt, bool padded,
                         uint64_t seed, float lo = -0.5f, float hi = 0.5f)
{
    const std::vector<int64_t> s(shape);
    return make_host_random(s, dt, padded, seed, lo, hi);
}

inline HostTensor int_t(std::initializer_list<int64_t> shape, DataType dt, uint64_t seed,
                        int64_t lo, int64_t hi)
{
    const std::vector<int64_t> s(shape);
    return make_host_int(s, dt, seed, lo, hi);
}

inline HostTensor as_layout(HostTensor t, TensorLayout layout)
{
    t.view = TensorView(t.view.shape_span(), t.view.data_type(), t.bytes.data(), layout);
    return t;
}

inline HostTensor as_nhwc(HostTensor t) { return as_layout(t, TensorLayout::NHWC); }
inline HostTensor as_ndhwc(HostTensor t) { return as_layout(t, TensorLayout::NDHWC); }

inline HostTensor zero_t(std::initializer_list<int64_t> shape, DataType dt)
{
    const std::vector<int64_t> s(shape);
    return make_host_zeroed(s, dt, false);
}

double load_elem(const std::byte* p, DataType dt)
{
    switch (dt) {
    case DataType::f32:
        return *reinterpret_cast<const float*>(p);
    case DataType::f16: {
        const uint16_t h = *reinterpret_cast<const uint16_t*>(p);
        return double(to_floats(&h, DataType::f16, 1)[0]);
    }
    case DataType::bf16: {
        const uint16_t h = *reinterpret_cast<const uint16_t*>(p);
        return double(to_floats(&h, DataType::bf16, 1)[0]);
    }
    case DataType::s8:
        return *reinterpret_cast<const int8_t*>(p);
    case DataType::u8:
        return *reinterpret_cast<const uint8_t*>(p);
    case DataType::s32:
        return *reinterpret_cast<const int32_t*>(p);
    case DataType::s64:
        return static_cast<double>(*reinterpret_cast<const int64_t*>(p));
    default:
        return 0.0;
    }
}

void expect_close_any(const HostTensor& got, const HostTensor& want, double tol,
                      const char* what)
{
    const DataType dt = got.view.data_type();
    const int64_t esz = data_type_size(dt);
    const int64_t rows = got.view.rank() >= 2 ? got.view.total_rows() : 1;
    const int64_t cols = rows > 0 ? got.view.numel() / rows : 0;
    for (int64_t r = 0; r < rows; ++r) {
        const std::byte* gp = got.bytes.data() + r * got.view.pitch();
        const std::byte* wp = want.bytes.data() + r * want.view.pitch();
        for (int64_t c = 0; c < cols; ++c) {
            const double a = load_elem(gp + c * esz, dt);
            const double b = load_elem(wp + c * esz, dt);
            if (std::isnan(a) && std::isnan(b)) {
                continue;
            }
            const double diff = std::abs(a - b);
            if (diff > tol) {
                char msg[320];
                std::snprintf(msg, sizeof(msg),
                              "%s: row %lld col %lld got %.17g want %.17g (|d|=%g)",
                              what, static_cast<long long>(r), static_cast<long long>(c), a,
                              b, diff);
                throw std::runtime_error(msg);
            }
        }
    }
}

/// Run `make(backend)` on both backends and compare every output. The output
/// views come from the CPU op's descriptors, so each case is just the operator,
/// its attributes and the input list.
template <class Factory>
void check(Factory make, std::initializer_list<const HostTensor*> ins, double tol,
           const char* what)
{
    std::vector<TensorDesc> descs;
    std::vector<TensorView> views;
    for (const HostTensor* t : ins) {
        descs.push_back(t != nullptr ? t->view.desc() : TensorDesc{});
        views.push_back(t != nullptr ? t->view : TensorView{});
    }

    auto op_cpu = make(Backend::CPU);
    const std::vector<TensorDesc> odescs = op_cpu->getOutputTensorDesc(descs);
    if (odescs.empty()) {
        throw std::runtime_error(std::string(what) + ": no output desc");
    }

    std::vector<HostTensor> ref;
    std::vector<HostTensor> got;
    for (const TensorDesc& d : odescs) {
        ref.push_back(make_host_zeroed(d.dims, d.dtype, false, d.layout));
        got.push_back(make_host_zeroed(d.dims, d.dtype, false, d.layout));
    }
    std::vector<TensorView> rv, gv;
    for (HostTensor& t : ref) {
        rv.push_back(t.view);
    }
    for (HostTensor& t : got) {
        gv.push_back(t.view);
    }
    const Status cpu_st = op_cpu->compute(rv, views, cpu_context(), nullptr);
    if (cpu_st != Status::ok) {
        throw std::runtime_error(std::string(what) + ": cpu compute returned " +
                                 to_string(cpu_st));
    }

    std::vector<DeviceTensor> dins(ins.size());
    std::vector<TensorView> dv;
    dv.reserve(ins.size());
    size_t di = 0;
    for (const HostTensor* t : ins) {
        if (t == nullptr) {
            dv.push_back(TensorView{});
        } else {
            dins[di] = to_device(*t);
            dv.push_back(dins[di].view);
        }
        ++di;
    }
    std::vector<DeviceTensor> douts;
    for (size_t i = 0; i < odescs.size(); ++i) {
        douts.push_back(alloc_like(gv[i], got[i]));
    }
    std::vector<TensorView> do_views;
    for (DeviceTensor& d : douts) {
        do_views.push_back(d.view);
    }

    auto op_dev = make(Backend::CUDA);
    const Status st = op_dev->compute(do_views, dv, cuda_context(), nullptr);
    if (st != Status::ok) {
        throw std::runtime_error(std::string(what) + ": cuda compute returned " +
                                 to_string(st));
    }
    CudaBuffer::sync();
    for (size_t i = 0; i < douts.size(); ++i) {
        from_device(douts[i], got[i]);
        expect_close_any(got[i], ref[i], tol, what);
    }
}

// =========================================================================
//  Reduction family: Reduce / Softmax / TopK / CumSum
// =========================================================================

namespace {

/// Values distinct along `axis` (coords * 0.5 on a 1/8 grid) so a selection or
/// a scan has no ties to disagree on.
HostTensor distinct_by_axis(std::vector<int64_t> shape, int64_t axis, DataType dt)
{
    HostTensor t = make_host_zeroed(shape, dt, false);
    const TensorView& v = t.view;
    const int64_t total = v.numel();
    const int64_t n = shape[static_cast<size_t>(axis)];
    int64_t stride = 1;
    for (size_t d = static_cast<size_t>(axis) + 1; d < shape.size(); ++d) {
        stride *= shape[d];
    }
    float* p = reinterpret_cast<float*>(t.bytes.data());
    for (int64_t i = 0; i < total; ++i) {
        const int64_t coord = (i / stride) % n;
        p[i] = static_cast<float>(coord) * 0.25f + static_cast<float>(i % 7) * 1e-3f;
    }
    if (dt == DataType::f16) {
        std::vector<float> tmp(p, p + total);
        HostTensor h = make_host_zeroed(shape, DataType::f16, false);
        auto* q = reinterpret_cast<nnops::backend::cpu::half*>(h.bytes.data());
        for (int64_t i = 0; i < total; ++i) {
            simd::s_store(&q[i], tmp[static_cast<size_t>(i)]);
        }
        return h;
    }
    return t;
}

} // anonymous namespace

NNOPS_TEST(cuda_reduce_axes) {
    const std::vector<std::vector<int64_t>> shapes = {{6, 5}, {3, 4, 5}, {2, 3, 4, 5}};
    for (const auto& shape : shapes) {
        for (int64_t axis = 0; axis < static_cast<int64_t>(shape.size()); ++axis) {
            for (ReduceType type : {ReduceType::Sum, ReduceType::Mean, ReduceType::Max,
                                    ReduceType::Min}) {
                for (bool keepdims : {false, true}) {
                    HostTensor a =
                        make_host_random(shape, DataType::f32, false, 300 + axis, -1.0f, 1.0f);
                    ReduceAttributes attrs;
                    attrs.type = type;
                    attrs.axes.push_back(axis);
                    attrs.keepdims = keepdims;
                    char what[96];
                    std::snprintf(what, sizeof(what), "reduce r%lld ax%lld t%d kd%d",
                                  static_cast<long long>(shape.size()),
                                  static_cast<long long>(axis), static_cast<int>(type),
                                  keepdims ? 1 : 0);
                    check([attrs](Backend b) { return Reduce::create(attrs, b); }, {&a}, 4e-6,
                          what);
                }
            }
        }
    }

    // A padded pitch and f16 exercise the row-stride and half paths.
    HostTensor h = make_host_random(std::vector<int64_t>{4, 33}, DataType::f16, true, 311, -1.0f,
                                    1.0f);
    for (ReduceType type : {ReduceType::Sum, ReduceType::Max}) {
        ReduceAttributes attrs;
        attrs.type = type;
        attrs.axes.push_back(-1);
        check([attrs](Backend b) { return Reduce::create(attrs, b); }, {&h}, 2e-3, "reduce f16");
    }
}

NNOPS_TEST(cuda_softmax_axes) {
    // A channel-last tensor whose softmax axis is neither the first nor the
    // last: the group walk route, in both the resident and the non-resident
    // row forms.
    HostTensor a = make_host_random(std::vector<int64_t>{2, 5, 3, 4}, DataType::f32, false, 321,
                                    -4.0f, 4.0f, TensorLayout::NHWC);
    for (int64_t axis : {1, 2}) {
        for (bool log_softmax : {false, true}) {
            SoftmaxAttributes attrs;
            attrs.axis = axis;
            attrs.log_softmax = log_softmax;
            check([attrs](Backend b) { return Softmax::create(attrs, b); }, {&a}, 2e-6,
                  "softmax mid-axis");
        }
    }
    HostTensor h = make_host_random(std::vector<int64_t>{3, 7, 11}, DataType::f16, false, 323,
                                    -4.0f, 4.0f, TensorLayout::NHWC);
    SoftmaxAttributes attrs;
    attrs.axis = 1;
    check([attrs](Backend b) { return Softmax::create(attrs, b); }, {&h}, 2e-3,
          "softmax f16 mid-axis");
}

NNOPS_TEST(cuda_topk_selection) {
    for (int64_t k : {int64_t(1), int64_t(3), int64_t(5)}) {
        for (TopKType type : {TopKType::Max, TopKType::Min}) {
            HostTensor a = distinct_by_axis({3, 9}, 1, DataType::f32);
            TopKAttributes attrs;
            attrs.k = k;
            attrs.axis = -1;
            attrs.type = type;
            // Values and the s64 indices must match exactly: the values are on a
            // grid the two backends share and distinct, so the order is fixed.
            check([attrs](Backend b) { return TopK::create(attrs, b); }, {&a}, 0.0, "topk");
        }
    }
    // A wide row with k above the register-list size, and one with many rows.
    HostTensor wide = distinct_by_axis({2, 2048}, 1, DataType::f32);
    TopKAttributes attrs;
    attrs.k = 17;
    check([attrs](Backend b) { return TopK::create(attrs, b); }, {&wide}, 0.0, "topk wide");
}

NNOPS_TEST(cuda_cumsum_modes) {
    for (int64_t axis : {0, 1}) {
        for (bool reverse : {false, true}) {
            for (bool exclusive : {false, true}) {
                HostTensor a = make_host_random(std::vector<int64_t>{17, 13}, DataType::f32, false, 331,
                                                -1.0f, 1.0f);
                CumSumAttributes attrs;
                attrs.axis = axis;
                attrs.reverse = reverse;
                attrs.exclusive = exclusive;
                check([attrs](Backend b) { return CumSum::create(attrs, b); }, {&a}, 4e-5,
                      "cumsum");
            }
        }
    }
}

// =========================================================================
//  Cast
// =========================================================================

NNOPS_TEST(cuda_cast_numeric_pairs) {
    // f32 -> f16 / f32 -> s32 (saturation and NaN) / s32 -> s8 / f32 -> u8.
    const std::vector<int64_t> shape = {3, 5};
    HostTensor x = make_host_random(shape, DataType::f32, false, 11, -40000.0f, 40000.0f);
    {
        auto* p = reinterpret_cast<float*>(x.bytes.data());
        p[0] = std::nanf("");
        p[1] = 3.9e9f;
        p[2] = -3.9e9f;
    }
    CastAttributes to_s32;
    to_s32.to = DataType::s32;
    check([&](Backend b) { return Cast::create(to_s32, b); }, {&x}, 0.0, "cast f32->s32");

    CastAttributes to_u8;
    to_u8.to = DataType::u8;
    check([&](Backend b) { return Cast::create(to_u8, b); }, {&x}, 0.0, "cast f32->u8");

    CastAttributes to_f16;
    to_f16.to = DataType::f16;
    HostTensor small = make_host_random(shape, DataType::f32, false, 13, -1000.0f, 1000.0f);
    check([&](Backend b) { return Cast::create(to_f16, b); }, {&small}, 0.0, "cast f32->f16");

    CastAttributes to_f32;
    to_f32.to = DataType::f32;
    HostTensor h16 = make_host_random(shape, DataType::f16, false, 17, -1000.0f, 1000.0f);
    check([&](Backend b) { return Cast::create(to_f32, b); }, {&h16}, 0.0, "cast f16->f32");

    HostTensor i32 = make_host_int(shape, DataType::s32, 19, -100000, 100000);
    CastAttributes to_s8;
    to_s8.to = DataType::s8;
    check([&](Backend b) { return Cast::create(to_s8, b); }, {&i32}, 0.0, "cast s32->s8");

    CastAttributes to_f32b;
    to_f32b.to = DataType::f32;
    check([&](Backend b) { return Cast::create(to_f32b, b); }, {&i32}, 0.0, "cast s32->f32");

    HostTensor i64 = make_host_int(shape, DataType::s64, 23, -(int64_t{1} << 40),
                                   int64_t{1} << 40);
    CastAttributes to_s32b;
    to_s32b.to = DataType::s32;
    check([&](Backend b) { return Cast::create(to_s32b, b); }, {&i64}, 0.0, "cast s64->s32");

    CastAttributes to_s64;
    to_s64.to = DataType::s64;
    HostTensor i8 = make_host_int(shape, DataType::s8, 29, -128, 127);
    check([&](Backend b) { return Cast::create(to_s64, b); }, {&i8}, 0.0, "cast s8->s64");
}

// =========================================================================
//  Clamp
// =========================================================================

NNOPS_TEST(cuda_clamp_and_residual) {
    const std::vector<int64_t> shape = {4, 7};
    HostTensor x = make_host_random(shape, DataType::f32, false, 31, -2.0f, 2.0f);
    ClampAttributes a;
    a.min_val = -0.5f;
    a.max_val = 0.75f;
    check([&](Backend b) { return Clamp::create(a, b); }, {&x}, 0.0, "clamp f32");

    // add_to reads the destination; the test starts both sides from zero, so
    // the check is the clamped sum against the CPU's same order.
    a.add_to = true;
    check([&](Backend b) { return Clamp::create(a, b); }, {&x}, 1e-7, "clamp f32 add_to");

    HostTensor h = make_host_random(shape, DataType::f16, false, 37, -2.0f, 2.0f);
    check([&](Backend b) { return Clamp::create(a, b); }, {&h}, 0.0, "clamp f16 add_to");
}

// =========================================================================
//  Compare / Logical
// =========================================================================

NNOPS_TEST(cuda_compare_broadcast) {
    const std::vector<int64_t> a_shape = {2, 3, 4};
    const std::vector<int64_t> b_shape = {2, 1, 4};
    HostTensor a = make_host_random(a_shape, DataType::f32, false, 41, -1.0f, 1.0f);
    HostTensor b = make_host_random(b_shape, DataType::f32, false, 43, -1.0f, 1.0f);
    reinterpret_cast<float*>(a.bytes.data())[5] = std::nanf("");

    for (CompareType t : {CompareType::Equal, CompareType::Greater, CompareType::Less,
                          CompareType::GreaterOrEqual, CompareType::LessOrEqual}) {
        CompareAttributes attrs;
        attrs.type = t;
        check([&](Backend be) { return Compare::create(attrs, be); }, {&a, &b}, 0.0,
              "compare f32");
    }

    HostTensor ia = make_host_int(a_shape, DataType::s32, 47, -5, 5);
    HostTensor ib = make_host_int(b_shape, DataType::s32, 53, -5, 5);
    CompareAttributes eq;
    check([&](Backend be) { return Compare::create(eq, be); }, {&ia, &ib}, 0.0,
          "compare s32");
}

NNOPS_TEST(cuda_logical_broadcast) {
    const std::vector<int64_t> a_shape = {2, 3, 4};
    const std::vector<int64_t> b_shape = {3, 1};
    HostTensor a = make_host_int(a_shape, DataType::u8, 59, 0, 3);
    HostTensor b = make_host_int(b_shape, DataType::u8, 61, 0, 1);
    for (LogicalType t : {LogicalType::And, LogicalType::Or, LogicalType::Xor}) {
        LogicalAttributes attrs;
        attrs.type = t;
        check([&](Backend be) { return Logical::create(attrs, be); }, {&a, &b}, 0.0,
              "logical u8");
    }
    LogicalAttributes not_a;
    not_a.type = LogicalType::Not;
    check([&](Backend be) { return Logical::create(not_a, be); }, {&a}, 0.0, "logical not");

    HostTensor ia = make_host_int(a_shape, DataType::s32, 67, -2, 2);
    HostTensor ib = make_host_int(b_shape, DataType::s32, 71, -2, 2);
    LogicalAttributes attrs;
    check([&](Backend be) { return Logical::create(attrs, be); }, {&ia, &ib}, 0.0,
          "logical s32");
}

// =========================================================================
//  Where
// =========================================================================

NNOPS_TEST(cuda_where_broadcast) {
    const std::vector<int64_t> out_shape = {2, 3, 4};
    HostTensor cond = int_t({2, 1, 4}, DataType::u8, 73, 0, 1);
    HostTensor x = make_host_random(out_shape, DataType::f32, false, 79, -1.0f, 1.0f);
    HostTensor y = rand_t({3, 4}, DataType::f32, false, 83, -1.0f, 1.0f);
    check([&](Backend b) { return Where::create(WhereAttributes{}, b); }, {&cond, &x, &y}, 0.0,
          "where f32 broadcast");

    HostTensor ix = make_host_int(out_shape, DataType::s32, 89, -9, 9);
    HostTensor iy = int_t({3, 4}, DataType::s32, 97, -9, 9);
    check([&](Backend b) { return Where::create(WhereAttributes{}, b); }, {&cond, &ix, &iy}, 0.0,
          "where s32 broadcast");
}

// =========================================================================
//  Flatten / pixel shuffle
// =========================================================================

NNOPS_TEST(cuda_flatten_pitched_and_dtypes) {
    HostTensor a = rand_t({2, 3, 5, 4}, DataType::f32, true, 101);
    check([&](Backend b) { return Flatten::create(FlattenAttributes{}, b); }, {&a}, 0.0,
          "flatten f32 pitched");
    HostTensor h = rand_t({4, 9}, DataType::f16, false, 103);
    check([&](Backend b) { return Flatten::create(FlattenAttributes{}, b); }, {&h}, 0.0,
          "flatten f16");
    HostTensor i = int_t({3, 7}, DataType::s32, 107, -100, 100);
    check([&](Backend b) { return Flatten::create(FlattenAttributes{}, b); }, {&i}, 0.0,
          "flatten s32");
}

NNOPS_TEST(cuda_pixel_shuffle_pair) {
    SpaceToDepthAttributes sd;
    sd.block_size = 2;
    HostTensor a = as_nhwc(rand_t({2, 4, 6, 3}, DataType::f32, false, 109));
    check([&](Backend b) { return SpaceToDepth::create(sd, b); }, {&a}, 0.0,
          "space_to_depth f32 b2");

    SpaceToDepthAttributes sd3;
    sd3.block_size = 3;
    HostTensor a3 = as_nhwc(rand_t({1, 6, 6, 5}, DataType::f16, false, 113));
    check([&](Backend b) { return SpaceToDepth::create(sd3, b); }, {&a3}, 0.0,
          "space_to_depth f16 b3");

    HostTensor ai = as_nhwc(int_t({1, 4, 4, 3}, DataType::s8, 127, -128, 127));
    check([&](Backend b) { return SpaceToDepth::create(sd, b); }, {&ai}, 0.0,
          "space_to_depth s8 b2");

    PixelShuffleAttributes ps;
    ps.block_size = 2;
    ps.mode = PixelShuffleMode::Dcr;
    HostTensor c = as_nhwc(rand_t({2, 3, 4, 8}, DataType::f32, false, 131));
    check([&](Backend b) { return PixelShuffle::create(ps, b); }, {&c}, 0.0,
          "pixel_shuffle dcr f32");

    ps.mode = PixelShuffleMode::Crd;
    check([&](Backend b) { return PixelShuffle::create(ps, b); }, {&c}, 0.0,
          "pixel_shuffle crd f32");

    ps.block_size = 3;
    ps.mode = PixelShuffleMode::Dcr;
    HostTensor c3 = as_nhwc(rand_t({1, 2, 3, 18}, DataType::f16, false, 137));
    check([&](Backend b) { return PixelShuffle::create(ps, b); }, {&c3}, 0.0,
          "pixel_shuffle f16 b3");

    ps.block_size = 2;
    HostTensor ci = as_nhwc(int_t({1, 3, 3, 4}, DataType::s8, 139, -128, 127));
    check([&](Backend b) { return PixelShuffle::create(ps, b); }, {&ci}, 0.0,
          "pixel_shuffle s8");
}

// =========================================================================
//  Embed
// =========================================================================

NNOPS_TEST(cuda_embed_fp_and_indices) {
    HostTensor w = rand_t({16, 8}, DataType::f32, false, 149, -1.0f, 1.0f);
    HostTensor idx = int_t({2, 3}, DataType::s64, 151, -2, 20); // clamped
    check([&](Backend b) { return Embed::create(EmbedAttributes{}, b); }, {&w, &idx}, 0.0,
          "embed f32 s64");

    HostTensor hw = rand_t({11, 5}, DataType::f16, false, 157, -1.0f, 1.0f);
    HostTensor i32 = int_t({7}, DataType::s32, 163, 0, 10);
    check([&](Backend b) { return Embed::create(EmbedAttributes{}, b); }, {&hw, &i32}, 0.0,
          "embed f16 s32");
}

NNOPS_TEST(cuda_embed_int8_per_row) {
    const int64_t V = 12, D = 6;
    HostTensor w8 = int_t({V, D}, DataType::s8, 167, -128, 127);
    std::vector<float> scale(static_cast<size_t>(V));
    std::vector<int32_t> zp(static_cast<size_t>(V));
    XorShift128 rng(173);
    for (int64_t v = 0; v < V; ++v) {
        scale[static_cast<size_t>(v)] = 0.002f + 0.006f * rng.next_float(0.0f, 1.0f);
        zp[static_cast<size_t>(v)] = static_cast<int32_t>(rng.next_u64() % 9) - 4;
    }
    QuantParams qp;
    qp.granularity = QuantGranularity::PerToken;
    qp.scale_data = scale.data();
    qp.zero_point_data = zp.data();
    qp.num_scales = V;

    const int64_t ws[2] = {V, D};
    TensorView wv(std::span<const int64_t>(ws, 2), DataType::s8, w8.bytes.data(),
                  TensorLayout::Dense, qp);

    HostTensor idx = int_t({2, 4}, DataType::s32, 179, 0, V - 1);
    // The device view has to carry the same QuantParams (the backend stages
    // the host arrays), which `to_device` does not copy.
    const int64_t os[3] = {2, 4, D};
    HostTensor out = make_host_zeroed(std::span<const int64_t>(os, 3), DataType::f32, false);

    auto op_cpu = Embed::create(EmbedAttributes{}, Backend::CPU);
    std::vector<TensorDesc> descs = {wv.desc(), idx.view.desc()};
    const std::vector<TensorDesc> odescs = op_cpu->getOutputTensorDesc(descs);
    std::vector<HostTensor> ref;
    std::vector<HostTensor> got;
    ref.push_back(make_host_zeroed(odescs[0].dims, odescs[0].dtype, false));
    got.push_back(make_host_zeroed(odescs[0].dims, odescs[0].dtype, false));
    std::vector<TensorView> rv = {ref[0].view};
    std::vector<TensorView> gv = {got[0].view};
    const std::vector<TensorView> ins = {wv, idx.view};
    if (op_cpu->compute(rv, ins, cpu_context(), nullptr) != Status::ok) {
        throw std::runtime_error("embed int8: cpu compute failed");
    }

    DeviceTensor dw = to_device(w8);
    dw.view = TensorView(std::span<const int64_t>(ws, 2), DataType::s8, dw.buf.data(),
                         TensorLayout::Dense, qp);
    DeviceTensor di = to_device(idx);
    DeviceTensor dout = alloc_like(gv[0], got[0]);
    std::vector<TensorView> dv = {dw.view, di.view};
    std::vector<TensorView> do_views = {dout.view};
    auto op_dev = Embed::create(EmbedAttributes{}, Backend::CUDA);
    const Status st = op_dev->compute(do_views, dv, cuda_context(), nullptr);
    if (st != Status::ok) {
        throw std::runtime_error(std::string("embed int8: cuda compute returned ") +
                                 to_string(st));
    }
    CudaBuffer::sync();
    from_device(dout, got[0]);
    expect_close_any(got[0], ref[0], 1e-6, "embed s8 per-row dequant");
    (void)out;
}

// =========================================================================
//  Quantize / Dequantize
// =========================================================================

NNOPS_TEST(cuda_quantize_linear_granularities) {
    const std::vector<int64_t> x_shape = {4, 6};
    HostTensor x = make_host_random(x_shape, DataType::f32, false, 181, -2.0f, 2.0f);

    // Per-tensor (a scalar scale makes the granularity per-tensor whatever the
    // axis says).
    {
        HostTensor scale = rand_t({1}, DataType::f32, false, 191, 0.01f, 0.05f);
        HostTensor zp = int_t({1}, DataType::s8, 193, -5, 5);
        QuantLinearAttributes a;
        a.output_dtype = DataType::s8;
        check([&](Backend b) { return QuantizeLinear::create(a, b); }, {&x, &scale, &zp}, 0.0,
              "quantize per-tensor");
    }
    // Per-token: one scale per row, u8 zero point and output.
    {
        HostTensor scale = rand_t({4}, DataType::f32, false, 197, 0.01f, 0.05f);
        HostTensor zp = int_t({4}, DataType::u8, 199, 100, 160);
        QuantLinearAttributes a;
        a.axis = -1;
        a.output_dtype = DataType::u8;
        check([&](Backend b) { return QuantizeLinear::create(a, b); }, {&x, &scale, &zp}, 0.0,
              "quantize per-token");
    }
    // Per-channel along axis 1.
    {
        HostTensor scale = rand_t({6}, DataType::f32, false, 211, 0.01f, 0.05f);
        HostTensor zp = int_t({6}, DataType::s8, 223, -5, 5);
        QuantLinearAttributes a;
        a.axis = 1;
        a.output_dtype = DataType::s8;
        check([&](Backend b) { return QuantizeLinear::create(a, b); }, {&x, &scale, &zp}, 0.0,
              "quantize per-channel");
    }
    // f16 activation, per-token.
    {
        HostTensor hx = make_host_random(x_shape, DataType::f16, false, 227, -2.0f, 2.0f);
        HostTensor scale = rand_t({4}, DataType::f32, false, 229, 0.01f, 0.05f);
        HostTensor zp = int_t({4}, DataType::s8, 233, -5, 5);
        QuantLinearAttributes a;
        a.axis = -1;
        a.output_dtype = DataType::s8;
        check([&](Backend b) { return QuantizeLinear::create(a, b); }, {&hx, &scale, &zp}, 0.0,
              "quantize f16 per-token");
    }
}

NNOPS_TEST(cuda_dequantize_linear_granularities) {
    const std::vector<int64_t> x_shape = {4, 6};
    HostTensor x8 = make_host_int(x_shape, DataType::s8, 239, -128, 127);
    {
        HostTensor scale = rand_t({4}, DataType::f32, false, 241, 0.01f, 0.05f);
        HostTensor zp = int_t({4}, DataType::s8, 251, -5, 5);
        QuantLinearAttributes a;
        a.axis = -1;
        a.output_dtype = DataType::f32;
        check([&](Backend b) { return DequantizeLinear::create(a, b); }, {&x8, &scale, &zp},
              1e-6, "dequantize per-token f32");
    }
    {
        HostTensor xu = make_host_int(x_shape, DataType::u8, 257, 0, 255);
        HostTensor scale = rand_t({6}, DataType::f32, false, 263, 0.01f, 0.05f);
        HostTensor zp = int_t({6}, DataType::u8, 269, 100, 160);
        QuantLinearAttributes a;
        a.axis = 1;
        a.output_dtype = DataType::f16;
        check([&](Backend b) { return DequantizeLinear::create(a, b); }, {&xu, &scale, &zp},
              0.0, "dequantize per-channel f16");
    }
    {
        HostTensor scale = rand_t({1}, DataType::f32, false, 271, 0.01f, 0.05f);
        HostTensor zp = int_t({1}, DataType::s8, 277, -5, 5);
        QuantLinearAttributes a;
        a.output_dtype = DataType::f32;
        check([&](Backend b) { return DequantizeLinear::create(a, b); }, {&x8, &scale, &zp},
              1e-6, "dequantize per-tensor");
    }
}

// =========================================================================
//  GridSample
// =========================================================================

NNOPS_TEST(cuda_grid_sample_2d) {
    const std::vector<int64_t> gshape = {1, 3, 4, 2};
    HostTensor x = as_nhwc(rand_t({1, 4, 5, 3}, DataType::f32, false, 301, -1.0f, 1.0f));
    HostTensor grid = rand_t({1, 3, 4, 2}, DataType::f32, false, 307, -1.25f, 1.25f);
    for (GridSampleMode mode : {GridSampleMode::Bilinear, GridSampleMode::Nearest}) {
        for (GridSamplePaddingMode pad :
             {GridSamplePaddingMode::Zeros, GridSamplePaddingMode::Border,
              GridSamplePaddingMode::Reflection}) {
            for (bool align : {false, true}) {
                GridSampleAttributes a;
                a.mode = mode;
                a.padding_mode = pad;
                a.align_corners = align;
                check([&](Backend b) { return GridSample::create(a, b); }, {&x, &grid}, 1e-6,
                      "grid_sample 2d f32");
            }
        }
    }
    // f16 image and a residual add.
    HostTensor hx = as_nhwc(rand_t({1, 4, 5, 3}, DataType::f16, false, 311, -1.0f, 1.0f));
    GridSampleAttributes a;
    a.add_to = true;
    check([&](Backend b) { return GridSample::create(a, b); }, {&hx, &grid}, 2e-3,
          "grid_sample 2d f16 add_to");
}

NNOPS_TEST(cuda_grid_sample_3d) {
    HostTensor x = as_ndhwc(rand_t({1, 3, 4, 5, 2}, DataType::f32, false, 313, -1.0f, 1.0f));
    HostTensor grid = rand_t({1, 2, 3, 2, 3}, DataType::f32, false, 317, -1.25f, 1.25f);
    for (GridSampleMode mode : {GridSampleMode::Bilinear, GridSampleMode::Nearest}) {
        GridSampleAttributes a;
        a.mode = mode;
        check([&](Backend b) { return GridSample::create(a, b); }, {&x, &grid}, 1e-6,
              "grid_sample 3d f32");
    }
    GridSampleAttributes a;
    a.padding_mode = GridSamplePaddingMode::Reflection;
    a.align_corners = true;
    check([&](Backend b) { return GridSample::create(a, b); }, {&x, &grid}, 1e-6,
          "grid_sample 3d reflection");
}

// =========================================================================
//  RoiAlign
// =========================================================================

NNOPS_TEST(cuda_roi_align) {
    HostTensor x = as_nhwc(rand_t({1, 6, 7, 4}, DataType::f32, false, 331, -1.0f, 1.0f));
    HostTensor rois = rand_t({3, 4}, DataType::f32, false, 337, 0.0f, 5.0f);
    HostTensor batch = int_t({3}, DataType::s32, 347, 0, 0);

    RoiAlignAttributes a;
    a.output_height = 2;
    a.output_width = 3;
    a.sampling_ratio = 0;
    check([&](Backend b) { return RoiAlign::create(a, b); }, {&x, &rois, &batch}, 1e-6,
          "roi_align avg adaptive");

    a.sampling_ratio = 2;
    a.mode = RoiAlignMode::Max;
    a.coordinate_mode = RoiAlignCoordinateMode::OutputHalfPixel;
    check([&](Backend b) { return RoiAlign::create(a, b); }, {&x, &rois}, 1e-6,
          "roi_align max");

    HostTensor hx = as_nhwc(rand_t({1, 6, 7, 4}, DataType::f16, false, 349, -1.0f, 1.0f));
    RoiAlignAttributes h;
    h.output_height = 3;
    h.output_width = 2;
    h.sampling_ratio = 1;
    h.add_to = true;
    check([&](Backend b) { return RoiAlign::create(h, b); }, {&hx, &rois}, 0.0,
          "roi_align f16 add_to");
}

// =========================================================================
//  Pool1D
// =========================================================================

NNOPS_TEST(cuda_pool1d_modes) {
    for (DataType dt : {DataType::f32, DataType::f16}) {
        HostTensor x = as_nhwc(rand_t({2, 9, 5}, dt, false, 353, -1.0f, 1.0f));
        // f16 stores the pooled value once; the CPU's sum order rounds to an
        // adjacent half now and then, so only f32 is held to the tight bound.
        const double tol = dt == DataType::f16 ? 2e-3 : 1e-6;
        for (bool max_mode : {true, false}) {
            Pool1DAttributes a;
            a.type = max_mode ? Pool1DType::Max : Pool1DType::Average;
            a.kernel_shape = 3;
            a.stride = 2;
            a.padding = 1;
            a.exclude_pad = !max_mode;
            check([&](Backend b) { return Pool1D::create(a, b); }, {&x}, tol,
                  "pool1d window");

            a.global = true;
            check([&](Backend b) { return Pool1D::create(a, b); }, {&x}, tol,
                  "pool1d global");

            a.global = false;
            a.output_size = 4;
            a.padding = 0;
            check([&](Backend b) { return Pool1D::create(a, b); }, {&x}, tol,
                  "pool1d adaptive");

            Pool1DAttributes same;
            same.type = a.type;
            same.kernel_shape = 3;
            same.stride = 2;
            same.auto_pad = Pool1DAttributes::AutoPad::SAME_UPPER;
            same.exclude_pad = !max_mode;
            check([&](Backend b) { return Pool1D::create(same, b); }, {&x}, tol,
                  "pool1d same_upper");
        }
    }
}

// =========================================================================
//  RoPE
// =========================================================================

NNOPS_TEST(cuda_rope_modes) {
    // [outer, S, D]: the last axis is the head width and the second-to-last
    // the sequence, so this is two outer rows of four positions.
    HostTensor x = rand_t({1, 2, 4, 8}, DataType::f32, false, 359, -1.0f, 1.0f);
    for (bool interleaved : {true, false}) {
        RoPEAttributes a;
        a.interleaved = interleaved;
        a.position_offset = 3;
        check([&](Backend b) { return RoPE::create(a, b); }, {&x}, 1e-6, "rope base");

        a.rotary_dim = 4;
        check([&](Backend b) { return RoPE::create(a, b); }, {&x}, 1e-6, "rope partial");

        // Fused RMSNorm before the rotation.
        HostTensor g = rand_t({8}, DataType::f32, false, 367, -1.0f, 1.0f);
        HostTensor be = rand_t({8}, DataType::f32, false, 373, -1.0f, 1.0f);
        RoPEAttributes n = a;
        n.rotary_dim = 8;
        check([&](Backend b) { return RoPE::create(n, b); }, {&x, &g, &be}, 1e-6,
              "rope fused norm");

        // A caller's cos/sin pair replaces the angle schedule.
        HostTensor cs = rand_t({4, 4}, DataType::f32, false, 379, -1.0f, 1.0f);
        HostTensor sn = rand_t({4, 4}, DataType::f32, false, 383, -1.0f, 1.0f);
        RoPEAttributes t = a;
        t.rotary_dim = 8;
        check([&](Backend b) { return RoPE::create(t, b); },
              {&x, nullptr, nullptr, &cs, &sn}, 1e-6, "rope supplied table");

        // per-batch first positions.
        HostTensor ids = int_t({1}, DataType::s64, 389, 5, 5);
        check([&](Backend b) { return RoPE::create(t, b); },
              {&x, nullptr, nullptr, nullptr, nullptr, &ids}, 1e-6, "rope position_ids");
    }

    // Long-context scaling modes.
    RoPEAttributes s;
    s.original_max_position = 16;
    s.scaling_factor = 2.0f;
    for (RoPEScalingType type :
         {RoPEScalingType::Linear, RoPEScalingType::Dynamic, RoPEScalingType::YaRN,
          RoPEScalingType::Llama3}) {
        s.scaling_type = type;
        check([&](Backend b) { return RoPE::create(s, b); }, {&x}, 1e-6, "rope scaling");
    }

    HostTensor hx = rand_t({1, 2, 4, 8}, DataType::f16, false, 397, -1.0f, 1.0f);
    RoPEAttributes h;
        check([&](Backend b) { return RoPE::create(h, b); }, {&hx}, 0.0, "rope f16");
}

// =========================================================================
//  NonMaxSuppression
// =========================================================================

NNOPS_TEST(cuda_nms_selection) {
    HostTensor boxes = rand_t({2, 6, 4}, DataType::f32, false, 401, 0.0f, 10.0f);
    HostTensor scores = rand_t({2, 3, 6}, DataType::f32, false, 409, -0.5f, 0.5f);
    // A tie must resolve by the lower box index on both backends.
    auto* sp = reinterpret_cast<float*>(scores.bytes.data());
    sp[0 * 6 + 0] = sp[0 * 6 + 1] = 0.25f;
    // A box with zero area checks the degenerate-IoU path.
    auto* bptr = reinterpret_cast<float*>(boxes.bytes.data());
    bptr[0] = bptr[2] = 1.0f; // y1 == y2, x1 == x2 for box 0 of batch 0

    HostTensor max_out = int_t({1}, DataType::s32, 419, 3, 3);
    HostTensor iou = rand_t({1}, DataType::f32, false, 421, 0.3f, 0.6f);
    HostTensor score_thr = rand_t({1}, DataType::f32, false, 431, -0.2f, 0.0f);

    for (int64_t top_k : {0, 3}) {
        for (int64_t center : {0, 1}) {
            NonMaxSuppressionAttributes a;
            a.top_k = top_k;
            a.center_point_box = center;
            check([&](Backend b) { return NonMaxSuppression::create(a, b); },
                  {&boxes, &scores, &max_out, &iou, &score_thr}, 0.0, "nms f32");
        }
    }

    HostTensor hb = rand_t({2, 6, 4}, DataType::f16, false, 433, 0.0f, 10.0f);
    HostTensor hs = rand_t({2, 3, 6}, DataType::f16, false, 439, -0.5f, 0.5f);
    NonMaxSuppressionAttributes a;
    a.top_k = 4;
    check([&](Backend b) { return NonMaxSuppression::create(a, b); },
          {&hb, &hs, &max_out, &iou}, 0.0, "nms f16 no score threshold");
}

}  // namespace

#endif  // NNOPS_HAS_CUDA
