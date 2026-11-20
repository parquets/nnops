/// @file test_cuda_pooling.cpp
/// @brief CUDA Pooling against the CPU backend on the same inputs.
///
/// Every case builds one random host tensor, runs the operator twice — once
/// with Backend::CPU and once with Backend::CUDA over device copies — and
/// compares the outputs. The oracle is the CPU route rather than a hand-written
/// reference so the operator's own contract (exclude_pad's divisor, the
/// all-padding window, add_to's ordering) is tested exactly as it defines it,
/// and both routes read the identical views, so nothing in the comparison
/// depends on a layout conversion.
///
/// The cases cover the channel-vector lane and its scalar tail (C = 4/8
/// exactly, then 5 and 17, which leave one or nine channels for the scalar
/// launch), plus the 3D forms, the flat fallback and global pooling.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/pooling.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"

#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

struct PoolCase {
    std::vector<int64_t> shape;
    PoolingAttributes attrs;
    DataType dtype = DataType::f32;
    TensorLayout layout = TensorLayout::NHWC;
    bool add_to = false;
};

/// f16 rounds through the kernel's accumulator and the CPU's; f32 has nothing
/// to round in max and one divide in average.
std::pair<float, float> tolerance(DataType dt)
{
    return dt == DataType::f16 ? std::pair<float, float>{kF16AccumTol, 2e-2f}
                               : std::pair<float, float>{1e-5f, 1e-5f};
}

void run_case(const PoolCase& c, const char* what)
{
    PoolingAttributes attrs = c.attrs;
    attrs.add_to = c.add_to;

    // NHWC views stay dense: their strides never consult the pitch.
    HostTensor in = make_host_random(c.shape, c.dtype, false, 13, -1.0f, 1.0f,
                                     c.layout);

    auto op_cpu = Pooling::create(attrs, Backend::CPU);
    const TensorDesc in_desc = in.view.desc();
    const std::vector<TensorDesc> outs = op_cpu->getOutputTensorDesc({&in_desc, 1});
    NNOPS_EXPECT_TRUE(!outs.empty());
    const TensorDesc od = outs.front();
    NNOPS_EXPECT_TRUE(od.layout == c.layout);

    // add_to reads the destination, so both runs must start from the same bytes.
    const auto seed = [&](uint64_t s) {
        return c.add_to ? make_host_random(od.dims, od.dtype, false, s, -1.0f, 1.0f,
                                           c.layout)
                        : make_host_zeroed(od.dims, od.dtype, false, c.layout);
    };
    HostTensor out_ref = seed(41);
    HostTensor out_dev = seed(41);

    const TensorView cpu_ins[] = {in.view};
    cpu_run(*op_cpu, out_ref.view, cpu_ins);

    auto op_dev = Pooling::create(attrs, Backend::CUDA);
    DeviceTensor din = to_device(in);
    DeviceTensor dout = alloc_like(out_dev.view, out_dev);
    const TensorView dev_ins[] = {din.view};
    cuda_run(*op_dev, dout.view, dev_ins);
    from_device(dout, out_dev);

    const auto [tol_abs, tol_rel] = tolerance(c.dtype);
    expect_same(out_dev, out_ref, tol_abs, tol_rel, what);
}

PoolingAttributes window(PoolingType type, int64_t k, int64_t stride = 1,
                         int64_t pad = 0, int64_t dilation = 1)
{
    PoolingAttributes a;
    a.type = type;
    a.kernel_shape = {k, k, k};
    a.stride = {stride, stride, stride};
    a.padding = {pad, pad, pad};
    a.dilation = {dilation, dilation, dilation};
    return a;
}

}  // namespace

// ============================================================
// NHWC
// ============================================================

NNOPS_TEST(cuda_pooling_vector_lane) {
    // C = 4 (f32) and 8 (f16) divide the 16 B vector exactly; 5 and 17 leave a
    // scalar tail, both a short one and one longer than a vector.
    for (int64_t C : {4, 5, 8, 17}) {
        for (PoolingType type : {PoolingType::Max, PoolingType::Average}) {
            PoolCase c;
            c.shape = {2, 9, 11, C};
            c.layout = TensorLayout::NHWC;
            c.attrs = window(type, 3, 2, 1);
            run_case(c, "cuda_pooling_vector_lane");
        }
    }
}

NNOPS_TEST(cuda_pooling_average_exclude_pad) {
    PoolCase c;
    c.shape = {1, 8, 10, 16};
    c.layout = TensorLayout::NHWC;
    c.attrs = window(PoolingType::Average, 3, 2, 1);
    c.attrs.exclude_pad = true;
    run_case(c, "cuda_pooling_average_exclude_pad");
    c.add_to = true;
    run_case(c, "cuda_pooling_average_exclude_pad");
}

NNOPS_TEST(cuda_pooling_f16) {
    PoolCase c;
    c.shape = {1, 7, 7, 8};
    c.dtype = DataType::f16;
    c.layout = TensorLayout::NHWC;
    c.attrs = window(PoolingType::Max, 2, 2);
    run_case(c, "cuda_pooling_f16");
    c.attrs = window(PoolingType::Average, 3, 1, 1);
    run_case(c, "cuda_pooling_f16");
}

NNOPS_TEST(cuda_pooling_rows) {
    // Unit stride and a square kernel: the kernel that walks two
    // output rows per thread. The max window pads, so its last row is short;
    // the exclude_pad average divides by a count that changes at the border.
    for (DataType dt : {DataType::f32, DataType::f16}) {
        PoolCase c;
        c.shape = {1, 10, 10, 16};
        c.dtype = dt;
        c.layout = TensorLayout::NHWC;
        c.attrs = window(PoolingType::Max, 3, 1, 1);
        run_case(c, "cuda_pooling_rows");
        c.attrs = window(PoolingType::Average, 2, 1, 1);
        c.attrs.exclude_pad = true;
        run_case(c, "cuda_pooling_rows");
    }
    // A 3D unit-stride window takes the same walk with a depth loop.
    PoolCase d;
    d.shape = {1, 3, 6, 6, 8};
    d.layout = TensorLayout::NDHWC;
    d.attrs = window(PoolingType::Average, 3, 1, 1);
    d.attrs.exclude_pad = true;
    run_case(d, "cuda_pooling_rows");
}

NNOPS_TEST(cuda_pooling_flat_fallback) {
    // (image, depth, row) past blockIdx.z's range, on a window that is neither
    // square nor of the walkable sizes: the flat script.
    PoolCase c;
    c.shape = {1, 300, 250, 2, 8};
    c.layout = TensorLayout::NDHWC;
    c.attrs = window(PoolingType::Max, 1, 1, 0);
    run_case(c, "cuda_pooling_flat_fallback");
}

NNOPS_TEST(cuda_pooling_3d) {
    PoolCase c;
    c.shape = {1, 3, 5, 6, 8};
    c.layout = TensorLayout::NDHWC;
    c.attrs = window(PoolingType::Max, 2, 2);
    run_case(c, "cuda_pooling_3d");
    c.attrs = window(PoolingType::Average, 2, 1, 1);
    c.attrs.exclude_pad = true;
    run_case(c, "cuda_pooling_3d");
}

NNOPS_TEST(cuda_pooling_global_nhwc) {
    for (int64_t C : {8, 5, 17}) {
        for (PoolingType type : {PoolingType::Max, PoolingType::Average}) {
            PoolCase c;
            c.shape = {2, 6, 7, C};
            c.layout = TensorLayout::NHWC;
            c.attrs.global = true;
            c.attrs.type = type;
            run_case(c, "cuda_pooling_global_nhwc");
        }
    }
    PoolCase d;
    d.shape = {1, 3, 4, 5, 4};
    d.layout = TensorLayout::NDHWC;
    d.attrs.global = true;
    d.attrs.type = PoolingType::Average;
    d.add_to = true;
    run_case(d, "cuda_pooling_global_nhwc");
}

#endif  // NNOPS_HAS_CUDA
