/// @file test_cuda_linear.cpp
/// @brief CUDA Linear against the CPU backend on the same inputs.
///
/// Linear is the dense GEMM with transpose_b and a rank-2 [N, K] weight, so the
/// cases that matter here are the ones the operator adds on top: the leading
/// dims it flattens into M, the [N] bias, the epilogue, and add_to — which on
/// this operator is a residual added *after* the activation, not a beta·C term.
/// The CPU route is the oracle, so that ordering is pinned against the
/// library's own contract rather than a re-derivation of it.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/linear.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"

#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

struct LinearCase {
    std::vector<int64_t> x_shape;
    int64_t n = 0;          // output features
    DataType dtype = DataType::f32;
    bool use_bias = false;
    bool add_to = false;
    bool padded = true;
    Epilogue epilogue{};
};

void run_case(const LinearCase& c, const char* what)
{
    LinearAttributes attrs;
    attrs.epilogue = c.epilogue;
    attrs.add_to = c.add_to;
    attrs.output_dtype = c.dtype;

    const int64_t k = c.x_shape.back();
    HostTensor x = make_host_random(c.x_shape, c.dtype, c.padded, 11);
    HostTensor w = make_host_random(std::vector<int64_t>{c.n, k}, c.dtype, false, 13);
    HostTensor bias;
    if (c.use_bias) { bias = make_host_random(std::vector<int64_t>{c.n}, c.dtype, false, 19, -0.25f, 0.25f); }

    std::vector<TensorDesc> descs = {x.view.desc(), w.view.desc()};
    if (c.use_bias) { descs.push_back(bias.view.desc()); }

    auto op_cpu = Linear::create(attrs, Backend::CPU);
    const std::vector<TensorDesc> outs = op_cpu->getOutputTensorDesc(descs);
    NNOPS_EXPECT_TRUE(!outs.empty());
    const TensorDesc od = outs.front();

    const auto seed = [&](uint64_t s) {
        return c.add_to ? make_host_random(od.dims, od.dtype, c.padded, s, 0.0f, 0.5f)
                        : make_host_zeroed(od.dims, od.dtype, c.padded);
    };
    HostTensor out_ref = seed(37);
    HostTensor out_dev = seed(37);

    std::vector<TensorView> cpu_ins = {x.view, w.view};
    if (c.use_bias) { cpu_ins.push_back(bias.view); }
    cpu_run(*op_cpu, out_ref.view, cpu_ins);

    auto op_dev = Linear::create(attrs, Backend::CUDA);
    DeviceTensor dx = to_device(x);
    DeviceTensor dw = to_device(w);
    DeviceTensor dbias;
    std::vector<TensorView> dev_ins = {dx.view, dw.view};
    if (c.use_bias) {
        dbias = to_device(bias);
        dev_ins.push_back(dbias.view);
    }
    DeviceTensor dout = alloc_like(out_dev.view, out_dev);
    cuda_run(*op_dev, dout.view, dev_ins);
    from_device(dout, out_dev);

    const float tol = c.dtype == DataType::f16 ? kF16AccumTol : 1e-4f;
    expect_same(out_dev, out_ref, tol, c.dtype == DataType::f16 ? 2e-2f : 1e-4f, what);
}

}  // namespace

NNOPS_TEST(cuda_linear_2d) {
    for (int64_t m : {1, 7, 64, 65, 200}) {
        LinearCase c;
        c.x_shape = {m, 96};
        c.n = 64;
        c.use_bias = true;
        run_case(c, "cuda_linear_2d");
    }
}

NNOPS_TEST(cuda_linear_ragged_k) {
    for (int64_t k : {1, 7, 16, 17, 63, 64, 65, 130}) {
        LinearCase c;
        c.x_shape = {33, k};
        c.n = 40;
        run_case(c, "cuda_linear_ragged_k");
    }
}

NNOPS_TEST(cuda_linear_f16) {
    LinearCase c;
    c.x_shape = {65, 96};
    c.n = 63;
    c.dtype = DataType::f16;
    c.use_bias = true;
    run_case(c, "cuda_linear_f16");
}

NNOPS_TEST(cuda_linear_batched) {
    {
        LinearCase c;
        c.x_shape = {2, 3, 40};
        c.n = 33;
        c.use_bias = true;
        run_case(c, "cuda_linear_batched");
    }
    {
        LinearCase c;
        c.x_shape = {4, 2, 5, 48};
        c.n = 17;
        run_case(c, "cuda_linear_batched");
    }
}

NNOPS_TEST(cuda_linear_add_to) {
    // add_to is post-activation here, so the activation must be non-identity for
    // the ordering to be observable.
    Epilogue ep;
    ep.type = EpilogueActivateType::Relu;

    LinearCase c;
    c.x_shape = {37, 65};
    c.n = 43;
    c.use_bias = true;
    c.add_to = true;
    c.epilogue = ep;
    run_case(c, "cuda_linear_add_to");

    LinearCase d;
    d.x_shape = {37, 65};
    d.n = 43;
    d.add_to = true;
    run_case(d, "cuda_linear_add_to");
}

NNOPS_TEST(cuda_linear_dense_strides) {
    LinearCase c;
    c.x_shape = {37, 65};
    c.n = 43;
    c.use_bias = true;
    c.padded = false;
    run_case(c, "cuda_linear_dense_strides");
}

#endif  // NNOPS_HAS_CUDA
