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

#include <array>
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

NNOPS_TEST(cuda_pooling_adaptive) {
    // Adaptive pooling: output index o covers input [floor(o*I/O),
    // ceil((o+1)*I/O)) -- a window a fixed kernel/stride cannot express, which
    // is what this used to compute instead.
    for (PoolingType type : {PoolingType::Max, PoolingType::Average}) {
        for (std::array<int64_t, 3> os :
             {std::array<int64_t, 3>{0, 3, 4}, std::array<int64_t, 3>{0, 1, 1},
              std::array<int64_t, 3>{0, 5, 2}}) {
            PoolCase c;
            c.shape = {2, 7, 9, 8};
            c.attrs.type = type;
            c.attrs.output_size = os;
            run_case(c, type == PoolingType::Max ? "adaptive max" : "adaptive average");
        }
    }
    // 3D, and an output larger than the input on one axis.
    PoolCase d3;
    d3.shape = {1, 4, 5, 6, 8};
    d3.attrs.type = PoolingType::Max;
    d3.attrs.output_size = {2, 2, 3};
    run_case(d3, "adaptive 3d max");
    PoolCase big;
    big.shape = {1, 5, 8, 8};
    big.attrs.type = PoolingType::Average;
    big.attrs.output_size = {0, 9, 9};
    run_case(big, "adaptive larger output");
}

// ============================================================
// Quantized pooling — the channel lane over int8 storage
// ============================================================
//
// PerTensor parameters on both sides, the CPU's quantized contract. Max is
// exact in any domain, and Average divides two f32 values that came from the
// same integers, so the two routes agree within one integer step.

namespace {

QuantParams qtensor(float scale, int32_t zp)
{
    QuantParams qp;
    qp.scale = scale;
    qp.zero_point = zp;
    return qp;
}

void run_quant_pool(std::vector<int64_t> shape, PoolingAttributes attrs, DataType dt,
                    bool global, const char* what)
{
    attrs.global = global;
    const bool s8 = dt == DataType::s8;

    HostTensor x = make_host_int(shape, dt, 601, s8 ? -128 : 0, s8 ? 127 : 255);
    const TensorLayout layout = shape.size() == 5 ? TensorLayout::NDHWC : TensorLayout::NHWC;
    x.view = TensorView(x.view.shape_span(), dt, x.bytes.data(), layout);

    const QuantParams xq = qtensor(0.02f, s8 ? -5 : 128);
    const QuantParams yq = qtensor(0.05f, s8 ? 3 : 110);
    const TensorView ins[] = {with_quant_params(x.view, xq)};
    const HostTensor* owners[] = {&x};

    run_quant_op([&](Backend b) { return Pooling::create(attrs, b); },
                 std::span<const TensorView>(ins, 1),
                 std::span<const HostTensor* const>(owners, 1), yq, 1.0f, what);
}

}  // namespace

NNOPS_TEST(cuda_pooling_quant_lane) {
    // C = 16 is one vector; 5 leaves a scalar tail.
    for (int64_t C : {16, 5}) {
        for (PoolingType type : {PoolingType::Max, PoolingType::Average}) {
            PoolingAttributes a;
            a.type = type;
            a.kernel_shape = {1, 3, 3};
            a.stride = {1, 2, 2};
            a.padding = {0, 1, 1};
            a.dilation = {1, 1, 1};
            a.exclude_pad = (type == PoolingType::Average);
            run_quant_pool({2, 9, 11, C}, a, C == 16 ? DataType::s8 : DataType::u8, false,
                           "pooling quant lane");
        }
    }
}

NNOPS_TEST(cuda_pooling_quant_3d_and_global) {
    PoolingAttributes a;
    a.type = PoolingType::Max;
    a.kernel_shape = {2, 2, 2};
    a.stride = {1, 2, 2};
    a.padding = {1, 1, 1};
    a.dilation = {1, 1, 1};
    run_quant_pool({1, 4, 5, 6, 8}, a, DataType::s8, false, "pooling quant 3d");

    PoolingAttributes g;
    g.type = PoolingType::Average;
    g.kernel_shape = {1, 1, 1};
    g.stride = {1, 1, 1};
    g.padding = {0, 0, 0};
    g.dilation = {1, 1, 1};
    run_quant_pool({2, 7, 9, 16}, g, DataType::u8, true, "pooling quant global");
}

#endif  // NNOPS_HAS_CUDA
