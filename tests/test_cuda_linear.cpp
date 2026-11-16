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
#include "nnops/detail/half.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
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

NNOPS_TEST(cuda_linear_splitk_residual) {
    // A small M against a long K is what the split-K launcher cuts, and add_to
    // is the term the reduce pass has to apply itself: the residual is added
    // after the activation, once, on the summed planes.
    Epilogue ep;
    ep.type = EpilogueActivateType::Gelu;

    LinearCase c;
    c.x_shape = {8, 4096};
    c.n = 256;
    c.use_bias = true;
    c.add_to = true;
    c.epilogue = ep;
    run_case(c, "cuda_linear_splitk_residual");

    LinearCase f;
    f.x_shape = {1, 2048};
    f.n = 128;
    f.dtype = DataType::f16;
    f.add_to = true;
    f.epilogue = ep;
    run_case(f, "cuda_linear_splitk_residual");
}

NNOPS_TEST(cuda_linear_dense_strides) {
    LinearCase c;
    c.x_shape = {37, 65};
    c.n = 43;
    c.use_bias = true;
    c.padded = false;
    run_case(c, "cuda_linear_dense_strides");
}

NNOPS_TEST(cuda_linear_i8) {
    // u8|s8 activation × s8 weight, both via the shared integer GEMM: an x
    // [.., M, K] against a [N, K] weight. Integer destinations compare byte for
    // byte against the CPU; the dequantizing one uses the fp tolerance.
    const auto run = [](const std::vector<int64_t>& xshape, int64_t N, DataType adt,
                        DataType odt, bool per_token, bool per_channel,
                        bool bias, uint64_t seed) {
        const int64_t M = xshape[xshape.size() - 2];
        const int64_t K = xshape.back();
        int64_t xn = 1;
        for (auto d : xshape) { xn *= d; }
        test::XorShift128 rng(seed);

        std::vector<uint8_t> x(static_cast<size_t>(xn));
        for (auto& e : x) { e = static_cast<uint8_t>(rng.next_u64() & 0xFFu); }
        std::vector<int8_t> w(static_cast<size_t>(N * K));
        for (auto& e : w) {
            e = static_cast<int8_t>(int32_t(rng.next_u64() % 255) - 127);
        }

        std::vector<float> x_s(static_cast<size_t>(M));
        std::vector<int32_t> x_zp(static_cast<size_t>(M));
        std::vector<float> w_s(static_cast<size_t>(N));
        std::vector<int32_t> w_zp(static_cast<size_t>(N));
        for (int64_t i = 0; i < M; ++i) {
            x_s[static_cast<size_t>(i)] = 0.01f + 0.003f * float(i % 6);
            x_zp[static_cast<size_t>(i)] = int32_t(i % 7) - 3;
        }
        for (int64_t i = 0; i < N; ++i) {
            w_s[static_cast<size_t>(i)] = 0.005f + 0.001f * float(i % 4);
            w_zp[static_cast<size_t>(i)] = int32_t(i % 5) - 2;
        }

        QuantParams qx;
        qx.scale = 0.01f;
        qx.zero_point = adt == DataType::u8 ? 128 : -4;
        if (per_token) {
            qx.granularity = QuantGranularity::PerToken;
            qx.scale_data = x_s.data();
            qx.zero_point_data = x_zp.data();
            qx.num_scales = M;
        }
        QuantParams qw;
        qw.scale = 0.005f;
        qw.zero_point = 2;
        if (per_channel) {
            qw.granularity = QuantGranularity::PerChannel;
            qw.scale_data = w_s.data();
            qw.zero_point_data = w_zp.data();
            qw.num_scales = N;
        }
        QuantParams qc;
        qc.scale = 0.02f;
        qc.zero_point = odt == DataType::u8 ? 100 : 0;

        const bool int_out = odt == DataType::s32 || odt == DataType::s8 ||
                             odt == DataType::u8;
        const DataType bdt = int_out ? DataType::s32 : odt;
        std::vector<uint8_t> bias_bytes;
        if (bias) {
            bias_bytes.resize(static_cast<size_t>(N) * size_t(data_type_size(bdt)));
            for (int64_t i = 0; i < N; ++i) {
                if (bdt == DataType::s32) {
                    reinterpret_cast<int32_t*>(bias_bytes.data())[i] =
                        int32_t(rng.next_u64() % 2001) - 1000;
                } else if (bdt == DataType::f16) {
                    reinterpret_cast<nnops::backend::cpu::half*>(
                        bias_bytes.data())[i] =
                        nnops::backend::cpu::float_to_half(
                            rng.next_float(-0.5f, 0.5f));
                } else {
                    reinterpret_cast<float*>(bias_bytes.data())[i] =
                        rng.next_float(-0.5f, 0.5f);
                }
            }
        }

        LinearAttributes attrs;
        attrs.output_dtype = odt;
        const auto view = [](const std::vector<int64_t>& sh, DataType dt,
                             void* d, const QuantParams& qp = {}) {
            return TensorView(std::span<const int64_t>(sh.data(), sh.size()), dt,
                              d, TensorLayout::Dense, qp);
        };

        TensorView xv = view(xshape, adt, x.data(), qx);
        TensorView wv = view({N, K}, DataType::s8, w.data(), qw);
        TensorView biasv;
        std::vector<TensorView> ins = {xv, wv};
        std::vector<TensorDesc> descs = {xv.desc(), wv.desc()};
        if (bias) {
            biasv = view({N}, bdt, bias_bytes.data());
            ins.push_back(biasv);
            descs.push_back(biasv.desc());
        }

        auto op_cpu = Linear::create(attrs, Backend::CPU);
        const TensorDesc od = op_cpu->getOutputTensorDesc(descs).front();
        const std::vector<int64_t> odims(od.dims.begin(), od.dims.begin() + od.rank);
        int64_t on = 1;
        for (auto d : odims) { on *= d; }
        const size_t obytes = static_cast<size_t>(on) *
                              size_t(data_type_size(od.dtype));
        std::vector<uint8_t> out_cpu(obytes, 0);
        std::vector<uint8_t> out_dev(obytes, 0);
        TensorView cv = view(odims, od.dtype, out_cpu.data(), qc);
        cpu_run(*op_cpu, cv, ins);

        CudaBuffer dx(x.size()), dw(w.size());
        dx.upload(x.data(), x.size());
        dw.upload(w.data(), w.size());
        TensorView dxv = view(xshape, adt, dx.data(), qx);
        TensorView dwv = view({N, K}, DataType::s8, dw.data(), qw);
        CudaBuffer dbias;
        std::vector<TensorView> dins = {dxv, dwv};
        if (bias) {
            dbias = CudaBuffer(bias_bytes.size());
            dbias.upload(bias_bytes.data(), bias_bytes.size());
            dins.push_back(view({N}, bdt, dbias.data()));
        }
        CudaBuffer dout(obytes);
        dout.zero();
        TensorView dcv = view(odims, od.dtype, dout.data(), qc);
        auto op_dev = Linear::create(attrs, Backend::CUDA);
        cuda_run(*op_dev, dcv, dins);
        dout.download(out_dev.data(), obytes);

        if (int_out) {
            NNOPS_EXPECT_TRUE(std::memcmp(out_cpu.data(), out_dev.data(),
                                          obytes) == 0);
            return;
        }
        const size_t n = obytes / size_t(data_type_size(od.dtype));
        for (size_t i = 0; i < n; ++i) {
            float a = 0.0f, b = 0.0f;
            if (od.dtype == DataType::f16) {
                a = nnops::backend::cpu::half_to_float(
                    reinterpret_cast<const nnops::backend::cpu::half*>(
                        out_dev.data())[i]);
                b = nnops::backend::cpu::half_to_float(
                    reinterpret_cast<const nnops::backend::cpu::half*>(
                        out_cpu.data())[i]);
            } else {
                a = reinterpret_cast<const float*>(out_dev.data())[i];
                b = reinterpret_cast<const float*>(out_cpu.data())[i];
            }
            NNOPS_EXPECT_TRUE(std::fabs(a - b) <= 1e-4f * std::max(1.0f, std::fabs(b)));
        }
    };

    // Every (activation, destination) pair the integer route resolves.
    const DataType adts[] = {DataType::u8, DataType::s8};
    const DataType odts[] = {DataType::s32, DataType::s8, DataType::u8,
                             DataType::f32, DataType::f16};
    uint64_t seed = 61;
    for (DataType adt : adts) {
        for (DataType odt : odts) {
            run({33, 65}, 43, adt, odt, true, true, true, seed++);
            run({3, 8, 40}, 17, adt, odt, false, true, true, seed++);
        }
    }
}

NNOPS_TEST(cuda_linear_rejects_rank_above_six) {
    // Linear shares MatMul's batch front end, so it carries four batch axes and
    // refuses a rank-seven activation at validate time.
    const int64_t xshape[] = {2, 1, 1, 1, 1, 4, 8};
    const int64_t wshape[] = {3, 8};
    std::vector<float> x(2 * 4 * 8, 1.0f);
    std::vector<float> w(3 * 8, 1.0f);
    TensorView xv(xshape, DataType::f32, x.data());
    TensorView wv(wshape, DataType::f32, w.data());
    const int64_t cshape[] = {2, 1, 1, 1, 1, 4, 3};
    std::vector<float> out(2 * 4 * 3, 0.0f);
    TensorView cv(cshape, DataType::f32, out.data());

    auto op = Linear::create(LinearAttributes{}, Backend::CUDA);
    const TensorView ins[] = {xv, wv};
    NNOPS_EXPECT_EQ(op->compute({&cv, 1}, ins, cuda_context(), nullptr),
                    Status::unsupported);
}

#endif  // NNOPS_HAS_CUDA
