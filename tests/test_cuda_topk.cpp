/// @file test_cuda_topk.cpp
/// @brief CUDA TopK against the CPU backend on the same inputs.
///
/// The CPU route is the oracle for both outputs: the values are compared where
/// they land, and the indices element-wise. The CUDA route always emits a sorted
/// result, which the contract allows for `sorted = false`, so those cases sort
/// both sides before comparing rather than pinning the CPU's arbitrary order.
///
/// Ties are the one thing the two routes do not have to agree on (the CPU leaves
/// their order to std::sort, the CUDA route prefers the lower index), so the f32
/// cases draw from a continuous distribution and the f16 ones are filled with
/// exact multiples of 0.5 — distinct per row, where a quantized random draw is
/// not.
///
/// Channel-first with the 32 B pitch a graph hands the op, on the last axis and
/// on the two other shapes (an interior axis and an outer one). Only compiled
/// under NNOPS_HAS_CUDA.

#include "nnops/ops/topk.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"

#include <algorithm>
#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

struct TopKCase {
    std::vector<int64_t> shape;
    int64_t axis = -1;
    int64_t k = 1;
    TopKType type = TopKType::Max;
    bool sorted = true;
    DataType dtype = DataType::f32;
};

/// Exact multiples of 0.5 along the axis: every element in a row is distinct
/// even after f16 rounding, so both routes must return the same indices.
HostTensor make_distinct(const std::vector<int64_t>& shape, DataType dt,
                         int64_t axis)
{
    HostTensor t = make_host_zeroed(shape, dt, true);
    TensorView& v = t.view;
    const int64_t rank = v.rank();
    const int64_t count = v.numel();

    int64_t coords[TensorView::kMaxRank] = {};
    for (int64_t e = 0; e < count; ++e) {
        int64_t off = 0;
        for (int64_t d = 0; d < rank; ++d) { off += coords[d] * v.stride_elems(d); }
        const float val = static_cast<float>(coords[axis]) * 0.5f;
        if (dt == DataType::f16) {
            simd::s_store(&v.ptr<backend::cpu::half>()[off], val);
        } else {
            v.ptr<float>()[off] = val;
        }
        for (int64_t d = rank - 1; d >= 0; --d) {
            if (++coords[d] < v.shape(d)) { break; }
            coords[d] = 0;
        }
    }
    return t;
}

/// The logical elements in storage order, padding skipped.
std::vector<float> flatten(const HostTensor& t)
{
    const TensorView& v = t.view;
    const int64_t rows = v.rank() >= 2 ? v.total_rows() : 1;
    const int64_t cols = v.numel() / rows;
    std::vector<float> out;
    out.reserve(static_cast<size_t>(v.numel()));
    for (int64_t r = 0; r < rows; ++r) {
        const auto row = to_floats(t.bytes.data() + static_cast<size_t>(r * v.pitch()),
                                   v.data_type(), static_cast<size_t>(cols));
        out.insert(out.end(), row.begin(), row.end());
    }
    return out;
}

std::vector<int64_t> flatten_i64(const HostTensor& t)
{
    const TensorView& v = t.view;
    const int64_t rank = v.rank();
    std::vector<int64_t> out;
    out.reserve(static_cast<size_t>(v.numel()));

    int64_t coords[TensorView::kMaxRank] = {};
    for (int64_t e = 0; e < v.numel(); ++e) {
        int64_t off = 0;
        for (int64_t d = 0; d < rank; ++d) { off += coords[d] * v.stride_elems(d); }
        out.push_back(v.ptr<int64_t>()[off]);
        for (int64_t d = rank - 1; d >= 0; --d) {
            if (++coords[d] < v.shape(d)) { break; }
            coords[d] = 0;
        }
    }
    return out;
}

void run_case(const TopKCase& c, const char* what)
{
    TopKAttributes attrs;
    attrs.type = c.type;
    attrs.axis = c.axis;
    attrs.k = c.k;
    attrs.sorted = c.sorted;

    HostTensor in = c.dtype == DataType::f16
        ? make_distinct(c.shape, c.dtype, c.axis < 0
                                                ? c.axis + int64_t(c.shape.size())
                                                : c.axis)
        : make_host_random(c.shape, c.dtype, true, 29, -1.0f, 1.0f);

    auto op_cpu = TopK::create(attrs, Backend::CPU);
    const TensorDesc in_desc = in.view.desc();
    const std::vector<TensorDesc> descs = op_cpu->getOutputTensorDesc({&in_desc, 1});
    NNOPS_EXPECT_TRUE(descs.size() == 2);
    const TensorDesc vd = descs[0];
    const TensorDesc id = descs[1];
    NNOPS_EXPECT_TRUE(id.dtype == DataType::s64);

    // The outputs are dense — that is what their descriptors say (no row
    // stride); only the input carries the pitch a graph hands the op.
    HostTensor v_ref = make_host_zeroed(vd.dims, vd.dtype, false, vd.layout);
    HostTensor i_ref = make_host_zeroed(id.dims, id.dtype, false, id.layout);
    HostTensor v_dev = make_host_zeroed(vd.dims, vd.dtype, false, vd.layout);
    HostTensor i_dev = make_host_zeroed(id.dims, id.dtype, false, id.layout);

    const TensorView in_views[] = {in.view};
    TensorView cpu_outs[] = {v_ref.view, i_ref.view};
    const Status st = op_cpu->compute(std::span<TensorView>(cpu_outs),
                                      std::span<const TensorView>(in_views));
    NNOPS_EXPECT_TRUE(st == Status::ok);

    auto op_dev = TopK::create(attrs, Backend::CUDA);
    DeviceTensor din = to_device(in);
    DeviceTensor dv = alloc_like(v_dev.view, v_dev);
    DeviceTensor di = alloc_like(i_dev.view, i_dev);
    TensorView gpu_outs[] = {dv.view, di.view};
    const TensorView dins[] = {din.view};
    const Status ds = op_dev->compute(std::span<TensorView>(gpu_outs),
                                      std::span<const TensorView>(dins),
                                      cuda_context(), nullptr);
    NNOPS_EXPECT_TRUE(ds == Status::ok);
    CudaBuffer::sync();
    from_device(dv, v_dev);
    from_device(di, i_dev);

    // The values come from the same comparisons on both sides, so they match to
    // the bit; the tolerance is only there to keep the message readable.
    const auto [tol_abs, tol_rel] = c.dtype == DataType::f16
        ? std::pair<float, float>{0.0f, 0.0f}
        : std::pair<float, float>{1e-6f, 1e-6f};

    if (c.sorted) {
        expect_same(v_dev, v_ref, tol_abs, tol_rel, what);
    } else {
        std::vector<float> g = flatten(v_dev);
        std::vector<float> w = flatten(v_ref);
        std::sort(g.begin(), g.end());
        std::sort(w.begin(), w.end());
        expect_close(g, w, tol_abs, tol_rel, what);
    }

    const std::vector<int64_t> gi = flatten_i64(i_dev);
    const std::vector<int64_t> wi = flatten_i64(i_ref);
    NNOPS_EXPECT_TRUE(gi.size() == wi.size());
    if (c.sorted) {
        NNOPS_EXPECT_TRUE(gi == wi);
    } else {
        std::vector<int64_t> gs = gi;
        std::vector<int64_t> ws = wi;
        std::sort(gs.begin(), gs.end());
        std::sort(ws.begin(), ws.end());
        NNOPS_EXPECT_TRUE(gs == ws);
    }
}

}  // namespace

NNOPS_TEST(cuda_topk_last_axis) {
    TopKCase c;
    c.shape = {2, 3, 16};
    c.k = 3;
    run_case(c, "cuda_topk_last_axis");
    c.k = 1;
    run_case(c, "cuda_topk_last_axis_k1");
    c.k = 16;                      // the whole axis
    c.type = TopKType::Min;
    run_case(c, "cuda_topk_last_axis_full_min");
}

NNOPS_TEST(cuda_topk_inner_axis) {
    TopKCase c;
    c.shape = {4, 6, 5};
    c.axis = 1;
    c.k = 2;
    run_case(c, "cuda_topk_inner_axis");
    c.type = TopKType::Min;
    run_case(c, "cuda_topk_inner_axis_min");
}

NNOPS_TEST(cuda_topk_outer_axis) {
    TopKCase c;
    c.shape = {5, 4, 3};
    c.axis = 0;
    c.k = 3;
    run_case(c, "cuda_topk_outer_axis");
    c.axis = -2;
    c.k = 1;
    run_case(c, "cuda_topk_outer_axis_wrapped");
}

NNOPS_TEST(cuda_topk_unsorted) {
    // The CUDA route always emits a sorted result, which `sorted = false`
    // permits, so both sides are compared as multisets.
    TopKCase c;
    c.shape = {3, 2, 24};
    c.k = 5;
    c.sorted = false;
    run_case(c, "cuda_topk_unsorted");
    c.type = TopKType::Min;
    run_case(c, "cuda_topk_unsorted_min");
}

NNOPS_TEST(cuda_topk_f16) {
    TopKCase c;
    c.shape = {2, 4, 32};
    c.dtype = DataType::f16;
    c.k = 4;
    run_case(c, "cuda_topk_f16");
    c.k = 8;
    c.type = TopKType::Min;
    c.shape = {3, 2, 8};
    run_case(c, "cuda_topk_f16_full_min");
}

NNOPS_TEST(cuda_topk_long_axis) {
    // More elements than one block covers, so a pass needs several rounds.
    TopKCase c;
    c.shape = {2, 1000};
    c.k = 7;
    run_case(c, "cuda_topk_long_axis");
    c.axis = 0;
    c.shape = {300, 2};
    c.k = 5;
    run_case(c, "cuda_topk_long_axis_outer");
}

#endif  // NNOPS_HAS_CUDA
