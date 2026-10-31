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
/// The cases split along the kernel's two routes. Planar cases carry a padded
/// pitch, which is the stride model a graph hands the op. Channel-last cases
/// cover the vector lane and its scalar tail (C = 4/8 exactly, then 5 and 17,
/// which leave one or nine channels for the scalar launch), plus the 3D forms
/// and global pooling on both routes.
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
    TensorLayout layout = TensorLayout::NCHW;
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

    // A planar tensor carries the 32 B pitch a graph gives it; a channel-last
    // one must stay dense, because its strides never consult the pitch.
    const bool padded = !is_channel_last(c.layout);
    HostTensor in = make_host_random(c.shape, c.dtype, padded, 13, -1.0f, 1.0f,
                                     c.layout);

    auto op_cpu = Pooling::create(attrs, Backend::CPU);
    const TensorDesc in_desc = in.view.desc();
    const std::vector<TensorDesc> outs = op_cpu->getOutputTensorDesc({&in_desc, 1});
    NNOPS_EXPECT_TRUE(!outs.empty());
    const TensorDesc od = outs.front();
    NNOPS_EXPECT_TRUE(od.layout == c.layout);

    // add_to reads the destination, so both runs must start from the same bytes.
    const auto seed = [&](uint64_t s) {
        return c.add_to ? make_host_random(od.dims, od.dtype, padded, s, -1.0f, 1.0f,
                                           c.layout)
                        : make_host_zeroed(od.dims, od.dtype, padded, c.layout);
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
// Planar
// ============================================================

NNOPS_TEST(cuda_pooling_planar_max) {
    for (int64_t k : {2, 3}) {
        for (int64_t s : {1, 2}) {
            PoolCase c;
            c.shape = {2, 3, 9, 13};
            c.attrs = window(PoolingType::Max, k, s);
            run_case(c, "cuda_pooling_planar_max");
        }
    }
}

NNOPS_TEST(cuda_pooling_planar_average) {
    for (int64_t k : {2, 3}) {
        PoolCase c;
        c.shape = {1, 4, 8, 11};
        c.attrs = window(PoolingType::Average, k, 2, 1);
        run_case(c, "cuda_pooling_planar_average");
        c.attrs.exclude_pad = true;
        run_case(c, "cuda_pooling_planar_average");
    }
    // A dilation the window walk has to step over.
    PoolCase d;
    d.shape = {1, 2, 12, 12};
    d.attrs = window(PoolingType::Average, 2, 1, 0, 2);
    run_case(d, "cuda_pooling_planar_average");
}

NNOPS_TEST(cuda_pooling_planar_f16) {
    PoolCase c;
    c.shape = {1, 3, 10, 7};
    c.dtype = DataType::f16;
    c.attrs = window(PoolingType::Max, 3, 2, 1);
    run_case(c, "cuda_pooling_planar_f16_max");
    c.attrs = window(PoolingType::Average, 3, 1, 1);
    c.attrs.exclude_pad = true;
    run_case(c, "cuda_pooling_planar_f16_avg");
}

NNOPS_TEST(cuda_pooling_planar_add_to) {
    PoolCase c;
    c.shape = {1, 3, 7, 9};
    c.attrs = window(PoolingType::Max, 2, 2);
    c.add_to = true;
    run_case(c, "cuda_pooling_planar_add_to");
    c.attrs = window(PoolingType::Average, 2, 2);
    run_case(c, "cuda_pooling_planar_add_to");
}

NNOPS_TEST(cuda_pooling_planar_3d) {
    PoolCase c;
    c.shape = {1, 3, 4, 6, 8};
    c.layout = TensorLayout::NCDHW;
    c.attrs = window(PoolingType::Max, 2, 2);
    run_case(c, "cuda_pooling_planar_3d");
    c.attrs = window(PoolingType::Average, 3, 1, 1);
    run_case(c, "cuda_pooling_planar_3d");
}

NNOPS_TEST(cuda_pooling_global_planar) {
    for (const auto& shape : {std::vector<int64_t>{2, 5, 7, 9},
                              std::vector<int64_t>{1, 1, 4, 4}}) {
        for (PoolingType type : {PoolingType::Max, PoolingType::Average}) {
            PoolCase c;
            c.shape = shape;
            c.attrs.global = true;
            c.attrs.type = type;
            run_case(c, "cuda_pooling_global_planar");
            c.add_to = true;
            run_case(c, "cuda_pooling_global_planar");
        }
    }
    PoolCase d;
    d.shape = {1, 3, 3, 4, 5};
    d.layout = TensorLayout::NCDHW;
    d.attrs.global = true;
    d.attrs.type = PoolingType::Average;
    run_case(d, "cuda_pooling_global_planar");
}

// ============================================================
// Channel-last
// ============================================================

NNOPS_TEST(cuda_pooling_nhwc_vector_lane) {
    // C = 4 (f32) and 8 (f16) divide the 16 B vector exactly; 5 and 17 leave a
    // scalar tail, both a short one and one longer than a vector.
    for (int64_t C : {4, 5, 8, 17}) {
        for (PoolingType type : {PoolingType::Max, PoolingType::Average}) {
            PoolCase c;
            c.shape = {2, C, 9, 11};
            c.layout = TensorLayout::NHWC;
            c.attrs = window(type, 3, 2, 1);
            run_case(c, "cuda_pooling_nhwc_vector_lane");
        }
    }
}

NNOPS_TEST(cuda_pooling_nhwc_average_exclude_pad) {
    PoolCase c;
    c.shape = {1, 16, 8, 10};
    c.layout = TensorLayout::NHWC;
    c.attrs = window(PoolingType::Average, 3, 2, 1);
    c.attrs.exclude_pad = true;
    run_case(c, "cuda_pooling_nhwc_average_exclude_pad");
    c.add_to = true;
    run_case(c, "cuda_pooling_nhwc_average_exclude_pad");
}

NNOPS_TEST(cuda_pooling_nhwc_f16) {
    PoolCase c;
    c.shape = {1, 8, 7, 7};
    c.dtype = DataType::f16;
    c.layout = TensorLayout::NHWC;
    c.attrs = window(PoolingType::Max, 2, 2);
    run_case(c, "cuda_pooling_nhwc_f16");
    c.attrs = window(PoolingType::Average, 3, 1, 1);
    run_case(c, "cuda_pooling_nhwc_f16");
}

NNOPS_TEST(cuda_pooling_nhwc_3d) {
    PoolCase c;
    c.shape = {1, 8, 3, 5, 6};
    c.layout = TensorLayout::NDHWC;
    c.attrs = window(PoolingType::Max, 2, 2);
    run_case(c, "cuda_pooling_nhwc_3d");
    c.attrs = window(PoolingType::Average, 2, 1, 1);
    c.attrs.exclude_pad = true;
    run_case(c, "cuda_pooling_nhwc_3d");
}

NNOPS_TEST(cuda_pooling_global_nhwc) {
    for (int64_t C : {8, 5, 17}) {
        for (PoolingType type : {PoolingType::Max, PoolingType::Average}) {
            PoolCase c;
            c.shape = {2, C, 6, 7};
            c.layout = TensorLayout::NHWC;
            c.attrs.global = true;
            c.attrs.type = type;
            run_case(c, "cuda_pooling_global_nhwc");
        }
    }
    PoolCase d;
    d.shape = {1, 4, 3, 4, 5};
    d.layout = TensorLayout::NDHWC;
    d.attrs.global = true;
    d.attrs.type = PoolingType::Average;
    d.add_to = true;
    run_case(d, "cuda_pooling_global_nhwc");
}

#endif  // NNOPS_HAS_CUDA
