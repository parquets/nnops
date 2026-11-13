/// @file test_cuda_argminmax.cpp
/// @brief CUDA ArgMax / ArgMin against the CPU backend.
///
/// Every case runs the operator on both backends over the same bytes and
/// compares the index tensors as integers. The CPU route is the oracle: it
/// defines the tie rule (the first of equal values keeps the index) and the NaN
/// rule, so the CUDA walk is pinned by the operator rather than by a second
/// hand-written reduction.
///
/// The cases split along the dispatch. An axis that is the last one takes the
/// warp-a-row kernel, so a lone 1D tensor, a logits-shaped [N, V] and an axis
/// longer than a warp are covered; any other axis takes the thread-per-output
/// kernel, whose inner length is what decides the coalescing. The padded input
/// covers the pitch model and the channel-last cases cover the strides the
/// reduction reads through, both of them with a batch above one so the outer
/// step is exercised rather than only the first image.
///
/// Ties and NaNs are built by hand: random data has neither, and both are the
/// places where a GPU reduction tree drifts from a sequential walk.
///
/// The walk reads strides, so a channel-last view and a pitched row are the same
/// two groups with different terms: both get a group of their own here, and the
/// CPU route for each is pinned by tests/test_argminmax.cpp against a written-out
/// oracle rather than against the same strides read back.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/argminmax.hpp"
#include "common/test_harness.hpp"
#include "common/cuda_test_helper.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

/// A dense s64 tensor laid out like the operator's own output descriptor. The
/// descriptor's dims array is longer than its rank, so the view must be built
/// from `rank` entries rather than from the array.
HostTensor make_output(const TensorDesc& od)
{
    const auto shape = std::span<const int64_t>(od.dims.data(),
                                                static_cast<size_t>(od.rank));
    HostTensor t;
    const TensorView probe(shape, od.dtype, nullptr, od.layout);
    // A rank-0 output has no elements to size the buffer from.
    t.bytes.assign(static_cast<size_t>(std::max<int64_t>(probe.storage_bytes(), 8)),
                   std::byte{0});
    t.view = TensorView(shape, od.dtype, t.bytes.data(), od.layout);
    return t;
}

struct ArgCase {
    std::vector<int64_t> shape;
    int64_t axis = -1;
    ArgMinMaxType type = ArgMinMaxType::Max;
    bool use_argmin = false;
    bool keepdims = false;
    bool padded = false;
    TensorLayout layout = TensorLayout::NCHW;
    DataType dtype = DataType::f32;
};

/// The index tensors are s64, which the shared float comparison does not widen,
/// so they are compared as integers. Both sides are dense, so the memory order is
/// the comparison: the point of the test is that the two backends agree on where
/// every index goes, and a padded or channel-last output would only hide that.
void expect_index_same(const HostTensor& got, const HostTensor& want, const char* what)
{
    const int64_t n = got.view.numel();
    const auto* gp = reinterpret_cast<const int64_t*>(got.bytes.data());
    const auto* wp = reinterpret_cast<const int64_t*>(want.bytes.data());
    for (int64_t i = 0; i < n; ++i) {
        if (gp[i] != wp[i]) {
            char msg[256];
            std::snprintf(msg, sizeof(msg), "%s: at %lld got %lld want %lld", what,
                          static_cast<long long>(i), static_cast<long long>(gp[i]),
                          static_cast<long long>(wp[i]));
            throw std::runtime_error(msg);
        }
    }
}

std::unique_ptr<OpBase> make_op(const ArgCase& c, Backend backend,
                                const ArgMinMaxAttributes& attrs)
{
    std::unique_ptr<OpBase> op;
    if (c.use_argmin) {
        op = ArgMin::create(attrs, backend);
    } else {
        op = ArgMax::create(attrs, backend);
    }
    return op;
}

/// Run one case over a given activation: CPU first (the oracle), then the same
/// bytes on the device, and compare the two index tensors.
void run_host(const HostTensor& act, const ArgCase& c, const char* what)
{
    ArgMinMaxAttributes attrs;
    attrs.type = c.type;
    attrs.axis = c.axis;
    attrs.keepdims = c.keepdims;

    auto op_cpu = make_op(c, Backend::CPU, attrs);
    const std::vector<TensorDesc> descs = {act.view.desc()};
    const std::vector<TensorDesc> outs = op_cpu->getOutputTensorDesc(descs);
    NNOPS_EXPECT_TRUE(!outs.empty());
    const TensorDesc od = outs.front();
    NNOPS_EXPECT_EQ(od.dtype, DataType::s64);

    HostTensor out_ref = make_output(od);
    HostTensor out_dev = make_output(od);

    std::vector<TensorView> cpu_ins = {act.view};
    cpu_run(*op_cpu, out_ref.view, cpu_ins);

    auto op_dev = make_op(c, Backend::CUDA, attrs);
    DeviceTensor dact = to_device(act);
    std::vector<TensorView> dev_ins = {dact.view};
    DeviceTensor dout = alloc_like(out_dev.view, out_dev);
    cuda_run(*op_dev, dout.view, dev_ins);
    from_device(dout, out_dev);

    expect_index_same(out_dev, out_ref, what);
}

void run_case(const ArgCase& c, const char* what)
{
    HostTensor act = make_host_random(c.shape, c.dtype, c.padded, 41, -1.0f, 1.0f,
                                      c.layout);
    run_host(act, c, what);
}

/// A dense f32 host tensor filled from `vals` in memory order.
HostTensor make_host_values(std::span<const int64_t> shape,
                            const std::vector<float>& vals,
                            TensorLayout layout = TensorLayout::NCHW)
{
    HostTensor t;
    const TensorView probe(shape, DataType::f32, nullptr, layout);
    t.bytes.assign(probe.storage_bytes(), std::byte{0});
    t.view = TensorView(shape, DataType::f32, t.bytes.data(), layout);
    auto* p = reinterpret_cast<float*>(t.bytes.data());
    for (size_t i = 0; i < vals.size(); ++i) { p[i] = vals[i]; }
    return t;
}

}  // namespace

// ============================================================
// The axis is the last one: the warp-a-row kernel
// ============================================================

NNOPS_TEST(cuda_argmax_axis_last) {
    {
        ArgCase c;   // one row: a single warp walks the whole axis
        c.shape = {1024};
        c.axis = 0;
        run_case(c, "cuda_argmax_axis_last");
    }
    {
        ArgCase c;   // the logits shape
        c.shape = {8, 1000};
        run_case(c, "cuda_argmax_axis_last");
    }
    {
        ArgCase c;   // the same with keepdims
        c.shape = {8, 1000};
        c.keepdims = true;
        run_case(c, "cuda_argmax_axis_last");
    }
    {
        ArgCase c;   // an axis just past a warp
        c.shape = {4, 33};
        c.axis = 1;
        run_case(c, "cuda_argmax_axis_last");
    }
    {
        ArgCase c;   // the same, on the ArgMin entry point
        c.shape = {5, 17};
        c.axis = -1;
        c.type = ArgMinMaxType::Min;
        c.use_argmin = true;
        run_case(c, "cuda_argmax_axis_last");
    }
    {
        ArgCase c;   // f16
        c.shape = {3, 64};
        c.dtype = DataType::f16;
        run_case(c, "cuda_argmax_axis_last");
    }
}

// ============================================================
// Any other axis: the thread-per-output kernel
// ============================================================

NNOPS_TEST(cuda_argmax_inner) {
    {
        ArgCase c;   // a channel plane: inner = H*W
        c.shape = {4, 16, 8, 8};
        c.axis = 1;
        run_case(c, "cuda_argmax_inner");
    }
    {
        ArgCase c;   // a narrow inner tail
        c.shape = {2, 8, 12, 10};
        c.axis = 2;
        run_case(c, "cuda_argmax_inner");
    }
    {
        ArgCase c;   // axis 0: one outer step, a wide inner run
        c.shape = {4, 6, 5};
        c.axis = 0;
        run_case(c, "cuda_argmax_inner");
    }
    {
        ArgCase c;   // keepdims keeps the reduced axis at one
        c.shape = {3, 7, 5, 9};
        c.axis = 1;
        c.keepdims = true;
        run_case(c, "cuda_argmax_inner");
    }
    {
        ArgCase c;   // ArgMin, f16
        c.shape = {2, 6, 7, 4};
        c.axis = 2;
        c.type = ArgMinMaxType::Min;
        c.use_argmin = true;
        c.dtype = DataType::f16;
        run_case(c, "cuda_argmax_inner");
    }
}

// ============================================================
// A padded pitch, both ends of the axis
// ============================================================

NNOPS_TEST(cuda_argmax_padded) {
    {
        ArgCase c;   // the padded axis is the reduced one
        c.shape = {3, 5, 7, 11};
        c.axis = 3;
        c.padded = true;
        run_case(c, "cuda_argmax_padded");
    }
    {
        ArgCase c;   // the padded pitch is the inner tail
        c.shape = {3, 5, 7, 11};
        c.axis = 2;
        c.padded = true;
        run_case(c, "cuda_argmax_padded");
    }
    {
        ArgCase c;   // a padded plane, batch above one
        c.shape = {2, 4, 9, 13};
        c.axis = 1;
        c.padded = true;
        run_case(c, "cuda_argmax_padded");
    }
}

// ============================================================
// Channel-last: the dims after the axis are interleaved
// ============================================================

NNOPS_TEST(cuda_argmax_channel_last) {
    {
        ArgCase c;   // the channel is the axis, the spatial tail is interleaved
        c.shape = {4, 16, 8, 8};
        c.axis = 1;
        c.layout = TensorLayout::NHWC;
        run_case(c, "cuda_argmax_channel_last");
    }
    {
        ArgCase c;   // a spatial axis: the channel sits inside the inner group
        c.shape = {2, 8, 12, 10};
        c.axis = 2;
        c.layout = TensorLayout::NHWC;
        run_case(c, "cuda_argmax_channel_last");
    }
    {
        ArgCase c;   // the batch axis: a single outer position, a wide tail
        c.shape = {4, 6, 5};
        c.axis = 0;
        c.layout = TensorLayout::NHWC;
        run_case(c, "cuda_argmax_channel_last");
    }
    {
        ArgCase c;   // keepdims, ArgMin, f16
        c.shape = {3, 8, 5, 7};
        c.axis = 3;
        c.keepdims = true;
        c.type = ArgMinMaxType::Min;
        c.use_argmin = true;
        c.dtype = DataType::f16;
        c.layout = TensorLayout::NHWC;
        run_case(c, "cuda_argmax_channel_last");
    }
}

// ============================================================
// The tie and NaN rules, built by hand
// ============================================================

NNOPS_TEST(cuda_argmax_ties) {
    // Four values, so most rows tie and the first of the tied indices has to
    // survive both the warp tree and the scalar walk.
    std::vector<float> vals;
    for (size_t i = 0; i < 8 * 40; ++i) {
        vals.push_back(static_cast<float>(i % 4));
    }
    {
        ArgCase c;
        c.shape = {8, 40};
        run_host(make_host_values(c.shape, vals), c, "cuda_argmax_ties");
    }
    {
        ArgCase c;
        c.shape = {8, 5, 8};
        c.axis = 1;
        run_host(make_host_values(c.shape, vals), c, "cuda_argmax_ties");
    }
    {
        ArgCase c;
        c.shape = {8, 40};
        c.type = ArgMinMaxType::Min;
        c.use_argmin = true;
        run_host(make_host_values(c.shape, vals), c, "cuda_argmax_ties");
    }
}

NNOPS_TEST(cuda_argmax_nan) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    // Row 0 starts on a NaN, row 1 holds one in the middle, row 2 is all NaN.
    const std::vector<float> vals = {nan, 1.0f, 2.0f,
                                     1.0f, nan, 3.0f,
                                     nan, nan, nan};
    ArgCase c;
    c.shape = {3, 3};

    HostTensor act = make_host_values(c.shape, vals);
    run_host(act, c, "cuda_argmax_nan");

    // ... and the same rule read back off the reference: a NaN at index 0 keeps
    // the answer at 0, a NaN anywhere else is skipped.
    ArgMinMaxAttributes attrs;
    attrs.axis = 1;
    auto op = ArgMax::create(attrs, Backend::CPU);
    const std::vector<TensorDesc> descs = {act.view.desc()};
    const TensorDesc od = op->getOutputTensorDesc(descs).front();
    HostTensor out = make_output(od);
    std::vector<TensorView> ins = {act.view};
    cpu_run(*op, out.view, ins);
    const auto* p = reinterpret_cast<const int64_t*>(out.bytes.data());
    NNOPS_EXPECT_EQ(p[0], int64_t(0));   // x[0] is NaN
    NNOPS_EXPECT_EQ(p[1], int64_t(2));   // 3.0, the NaN skipped
    NNOPS_EXPECT_EQ(p[2], int64_t(0));   // all NaN
}

#endif  // NNOPS_HAS_CUDA
