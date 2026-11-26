/// @file bench_cuda_linear_attention.cpp
/// @brief CUDA LinearAttention against the CPU route on the shapes the model
///        runs: a prefill (many tokens in one call) and a decode step (S == 1,
///        the state carried in place), with the conv and beta a gated delta net
///        layer carries.
///
/// The FLOP count is the recurrence's own — (2 + Gqq)·Dk·Dv per token per value
/// head, from the retrieval, the rank-1 update and one readout per query head —
/// so a row is comparable to any other operator's row of the same count. The
/// CPU column is the same operator on the same bytes, timed with a wall clock:
/// it is the reference the GPU route has to beat on these shapes, both being
/// issue/bandwidth bound.
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/linear_attention.hpp"
#include "common/bench_harness.hpp"
#include "common/cuda_bench_helper.hpp"
#include "common/cuda_test_helper.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

double cpu_median_ms(const std::function<void()>& fn, int warmup = 2, int samples = 5)
{
    std::vector<double> ms;
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    for (int i = 0; i < samples; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        const auto t1 = std::chrono::high_resolution_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

/// One shape, both backends. H_q == H_v (one query head reads a state), H_v may
/// be a multiple of H_k (Qwen3-Next) and Dv may differ from Dk.
void run_case(const char* tag, int64_t B, int64_t Hv, int64_t Hk, int64_t S, int64_t Dk,
              int64_t Dv, int64_t CK, DataType dt)
{
    const int64_t Gqq = 1;
    const int64_t Hq = Hv;

    HostTensor q = make_host_random(std::vector<int64_t>{B, Hq, S, Dk}, dt, false, 31, -1.0f, 1.0f);
    HostTensor k = make_host_random(std::vector<int64_t>{B, Hk, S, Dk}, dt, false, 32, -1.0f, 1.0f);
    HostTensor v = make_host_random(std::vector<int64_t>{B, Hv, S, Dv}, dt, false, 33, -1.0f, 1.0f);
    HostTensor g = make_host_random(std::vector<int64_t>{B, Hv, S, Dk}, dt, false, 34, -2.0f, 0.0f);
    HostTensor cw = make_host_random(std::vector<int64_t>{(Hq + Hk) * Dk + Hv * Dv, 1, CK}, dt,
                                     false, 35, -0.5f, 0.5f);
    HostTensor bt = make_host_random(std::vector<int64_t>{B, Hv, S}, dt, false, 36, 0.0f, 1.0f);
    HostTensor st_init =
        make_host_random(std::vector<int64_t>{B, Hv, Dk, Dv}, dt, false, 37, -1.0f, 1.0f);

    LinearAttentionAttributes attrs;
    attrs.num_heads = Hq;
    attrs.num_kv_heads = Hk;
    attrs.head_dim = Dk;
    attrs.conv_kernel_size = CK;

    HostTensor out_cpu = make_host_zeroed(std::vector<int64_t>{B, Hq, S, Dv}, dt, false);
    HostTensor out_dev = make_host_zeroed(std::vector<int64_t>{B, Hq, S, Dv}, dt, false);
    // The state is read then overwritten in place, so each backend gets its own
    // initial copy (repeating a call keeps updating the same buffer, which is
    // what a decoder does anyway).
    HostTensor st_cpu = st_init;
    HostTensor st_dev = st_init;
    st_cpu.view = TensorView(st_init.view.shape_span(), dt, st_cpu.bytes.data(),
                             st_init.view.pitch());
    st_dev.view = TensorView(st_init.view.shape_span(), dt, st_dev.bytes.data(),
                             st_init.view.pitch());

    const std::vector<TensorView> ins = {q.view, k.view, v.view, g.view, cw.view, bt.view};

    auto cpu_op = LinearAttention::create(attrs, Backend::CPU);
    std::vector<TensorView> cpu_outs = {out_cpu.view, st_cpu.view};
    const double cpu_ms = cpu_median_ms([&] {
        const Status s = cpu_op->compute(cpu_outs, ins, cpu_context(), nullptr);
        if (s != Status::ok) {
            throw std::runtime_error("cpu compute failed");
        }
    });

    std::vector<DeviceTensor> dev;
    for (const auto& t : {q, k, v, g, cw, bt}) {
        dev.push_back(to_device(t));
    }
    std::vector<TensorView> dev_ins;
    for (auto& d : dev) {
        dev_ins.push_back(d.view);
    }
    DeviceTensor dout = alloc_like(out_dev.view, out_dev);
    DeviceTensor dst = alloc_like(st_dev.view, st_dev);
    auto cuda_op = LinearAttention::create(attrs, Backend::CUDA);
    std::vector<TensorView> cuda_outs = {dout.view, dst.view};
    const double cuda_ms = cuda_median_ms([&] {
        const Status s = cuda_op->compute(cuda_outs, dev_ins, cuda_context(), nullptr);
        if (s != Status::ok) {
            throw std::runtime_error("cuda compute failed");
        }
    });

    const double flops =
        2.0 * double(B) * double(Hv) * double(S) * double(2 + Gqq) * double(Dk) * double(Dv);
    const double tf = flops / (cuda_ms * 1e9);
    char geom[200];
    std::snprintf(geom, sizeof(geom), "%-8s B=%lld S=%lld Hv=%lld Hk=%lld Dk=%lld Dv=%lld %s",
                  tag, static_cast<long long>(B), static_cast<long long>(S),
                  static_cast<long long>(Hv), static_cast<long long>(Hk),
                  static_cast<long long>(Dk), static_cast<long long>(Dv),
                  dt == DataType::f32 ? "f32" : "f16");
    std::printf("  %-58s cuda %8.3f ms (%6.2f TF)  cpu %8.3f ms  x%5.2f\n", geom, cuda_ms, tf,
                cpu_ms, cpu_ms / cuda_ms);
}

} // namespace

NNOPS_BENCH(cuda_linear_attention)
{
    std::printf("\n  == LinearAttention ==\n");
    // Qwen3-Next's linear layer: 32 value heads, 16 key heads, 128-wide state.
    run_case("decode", 1, 32, 16, 1, 128, 128, 4, DataType::f32);
    run_case("decode", 1, 32, 16, 1, 128, 128, 4, DataType::f16);
    run_case("prefill", 1, 32, 16, 2048, 128, 128, 4, DataType::f32);
    run_case("prefill", 1, 32, 16, 2048, 128, 128, 4, DataType::f16);
    // A plain Gated Delta Net layer (D_v == D_k, no head expansion).
    run_case("gdn", 1, 8, 8, 1024, 128, 128, 4, DataType::f32);
    run_case("gdn", 1, 8, 8, 1, 128, 128, 4, DataType::f32);
}

#endif // NNOPS_HAS_CUDA
