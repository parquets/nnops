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
#include "nnops/ops/matmul.hpp"
#include "backend/cuda/attention/linear_attention_wy_host.hpp"
#include "common/bench_harness.hpp"
#include "common/cuda_bench_helper.hpp"
#include "common/cuda_test_helper.hpp"
#include "detail/bench_knobs.hpp"

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
///
/// `glo`/`ghi` are the log-decay range: the default is the steep one the
/// recurrence is stress-tested with, while the mild pair a trained layer
/// produces is what lets the f16 chunked route run at all.
void run_case(const char* tag, int64_t B, int64_t Hv, int64_t Hk, int64_t S, int64_t Dk,
              int64_t Dv, int64_t CK, DataType dt, float glo = -2.0f, float ghi = 0.0f)
{
    const int64_t Gqq = 1;
    const int64_t Hq = Hv;

    HostTensor q = make_host_random(std::vector<int64_t>{B, Hq, S, Dk}, dt, false, 31, -1.0f, 1.0f);
    HostTensor k = make_host_random(std::vector<int64_t>{B, Hk, S, Dk}, dt, false, 32, -1.0f, 1.0f);
    HostTensor v = make_host_random(std::vector<int64_t>{B, Hv, S, Dv}, dt, false, 33, -1.0f, 1.0f);
    HostTensor g = make_host_random(std::vector<int64_t>{B, Hv, S, Dk}, dt, false, 34, glo, ghi);
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
    const int64_t wy_before = nnops::backend::cuda::la_wy_route_runs();
    const double cuda_ms = cuda_median_ms([&] {
        const Status s = cuda_op->compute(cuda_outs, dev_ins, cuda_context(), nullptr);
        if (s != Status::ok) {
            throw std::runtime_error("cuda compute failed");
        }
    });
    // Which route the operator took is not otherwise visible in a row.
    const bool wy = nnops::backend::cuda::la_wy_route_runs() != wy_before;

    const double flops =
        2.0 * double(B) * double(Hv) * double(S) * double(2 + Gqq) * double(Dk) * double(Dv);
    const double tf = flops / (cuda_ms * 1e9);
    char geom[200];
    std::snprintf(geom, sizeof(geom),
                  "%-8s B=%lld S=%lld Hv=%lld Hk=%lld Dk=%lld Dv=%lld %s%s", tag,
                  static_cast<long long>(B), static_cast<long long>(S),
                  static_cast<long long>(Hv), static_cast<long long>(Hk),
                  static_cast<long long>(Dk), static_cast<long long>(Dv),
                  dt == DataType::f32 ? "f32 " : "f16 ", wy ? " wy" : "");
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
    // The f16 per-token baseline at the GDN shape, and then both prefills with
    // the gate a trained layer produces (a decay of 0.98 to 1 a token): that
    // range is inside the f16 panel format, so the chunked-WY route runs, and
    // these are the rows to read it off.
    run_case("gdn", 1, 8, 8, 1024, 128, 128, 4, DataType::f16);
    run_case("prefill", 1, 32, 16, 2048, 128, 128, 4, DataType::f16, -0.02f, 0.0f);
    run_case("gdn", 1, 8, 8, 1024, 128, 128, 4, DataType::f16, -0.02f, 0.0f);
}

namespace {

/// One batched f16 product — A [b, m, k] against B [b, k, n] — through the
/// MatMul operator, which is what a chunked-WY route would issue at C = 64. A
/// row carries the operator's global traffic too, so the estimate below is an
/// upper bound on a fused kernel that keeps the chunk's panels in shared
/// memory. Returns the call's milliseconds.
double wy_gemm(int64_t b, int64_t m, int64_t n, int64_t k, const char* tag)
{
    HostTensor ha =
        make_host_random(std::vector<int64_t>{b, m, k}, DataType::f16, false, 41, -0.5f, 0.5f);
    HostTensor hb =
        make_host_random(std::vector<int64_t>{b, k, n}, DataType::f16, false, 43, -0.5f, 0.5f);
    HostTensor hc = make_host_zeroed(std::vector<int64_t>{b, m, n}, DataType::f16, false);
    DeviceTensor da = to_device(ha);
    DeviceTensor db = to_device(hb);
    DeviceTensor dc = to_device(hc);

    auto op = MatMul::create(MatMulAttributes{}, Backend::CUDA);
    const TensorView ins[] = {da.view, db.view};
    TensorView out = dc.view;
    const double ms = cuda_median_ms([&] { cuda_launch(*op, out, ins); });

    const double flops = 2.0 * double(b) * double(m) * double(n) * double(k);
    char geo[96];
    std::snprintf(geo, sizeof(geo), "b%-5lld %lldx%lldx%lld", (long long)b,
                  (long long)m, (long long)n, (long long)k);
    print_cuda_row(tag, geo, ms, cuda_gflops(ms, flops));
    return ms;
}

/// Price the two kernels the paper's single-GPU route runs. Kernel 1
/// (chunk-local) is parallel over every chunk-head; kernel 2 (the recurrence
/// and the readout) is parallel over the heads only and walks the chunks in
/// sequence. The triangular solve is ~C^3/3 MACs per chunk-head, two orders
/// below the products, so it is left out.
///
/// `splitk` is the split-K knob for the measurement: -1 leaves the heuristic
/// on, 0 turns it off. A fused WY kernel would never split (its chunk is one
/// CTA's work), and the heuristic reads a batched call's grid as one matrix's,
/// so the off arm is the honest proxy for these shapes.
void wy_estimate(int64_t Hv, int64_t S, int64_t Dk, int64_t Dv, int64_t C,
                 const char* name, int splitk)
{
    nnops::detail::cuda_splitk.store(splitk, std::memory_order_relaxed);
    const int64_t L = (S + C - 1) / C;
    const int64_t nch = Hv * L;
    std::printf("\n  -- %s: Hv=%lld S=%lld C=%lld L=%lld chunk-heads=%lld --\n", name,
                (long long)Hv, (long long)S, (long long)C, (long long)L, (long long)nch);

    // Kernel 1: KK^T, then W = A^-1 (beta K) and U = A^-1 (beta V).
    const double t_kk = wy_gemm(nch, C, C, Dk, "kernel1 KK^T");
    const double t_wu = wy_gemm(nch, C, Dk, C, "kernel1 W/U");
    const double k1 = t_kk + 2.0 * t_wu;

    // Kernel 2, one chunk at a time over the heads that exist: W*S, K^T X,
    // Q*S, QK^T and (QK^T) X. The floor is the same call on an 8^3 product, so
    // it prices the launch each of the five pays.
    const double t_ws = wy_gemm(Hv, C, Dv, Dk, "kernel2 W*S");
    const double t_kx = wy_gemm(Hv, Dk, Dv, C, "kernel2 K^T X");
    const double t_qs = wy_gemm(Hv, C, Dv, Dk, "kernel2 Q*S");
    const double t_qk = wy_gemm(Hv, C, C, Dk, "kernel2 QK^T");
    const double t_qx = wy_gemm(Hv, C, Dv, C, "kernel2 (QK^T)X");
    const double floor = wy_gemm(Hv, 8, 8, 8, "kernel2 launch floor");
    const double per_chunk = t_ws + t_kx + t_qs + t_qk + t_qx;
    const double k2 = L * per_chunk;

    std::printf("   kernel1 %.3f ms (KK^T %.3f, W/U %.3f x2)\n", k1, t_kk, t_wu);
    std::printf("   kernel2 %.3f ms (%lld chunks x %.3f ms; launch floor %.3f x5)\n",
                k2, (long long)L, per_chunk, floor);
    std::printf("   total    %.3f ms   (kernel2 net of launch floors %.3f ms)\n",
                k1 + k2, k1 + k2 - double(L) * 5.0 * floor);
    nnops::detail::cuda_splitk.store(-1, std::memory_order_relaxed);
}

} // namespace

NNOPS_BENCH(cuda_linear_attention_wy_probe)
{
    print_cuda_header("LinearAttention chunked-WY probe (f16 batched GEMM shapes)");
    // The two prefill shapes the bench above runs, at the chunk size a
    // tensor-core route would take: one chunk is a 64x64x128 product. Both
    // split-K arms are measured, because the heuristic reads a batched grid as
    // a single matrix's and a fused kernel would never split.
    for (int arm = 0; arm < 2; ++arm) {
        const int splitk = arm == 0 ? -1 : 0;
        std::printf("\n  == split-K %s ==\n", arm == 0 ? "heuristic" : "off");
        wy_estimate(32, 2048, 128, 128, 64, "Qwen3-Next prefill", splitk);
        wy_estimate(8, 1024, 128, 128, 64, "GDN prefill", splitk);
    }
    // The first shape again at other chunk sizes: 16 is the CPU route's, 32 and
    // 128 bracket the tensor-core route's 64.
    wy_estimate(32, 2048, 128, 128, 16, "Qwen3-Next prefill, C=16", 0);
    wy_estimate(32, 2048, 128, 128, 32, "Qwen3-Next prefill, C=32", 0);
    wy_estimate(32, 2048, 128, 128, 128, "Qwen3-Next prefill, C=128", 0);
}

namespace {

/// The route's second kernel on one prefill geometry. Kernel 1's panels (W, U,
/// the gated Q and K, the chunk decay) are synthesized: the row is the
/// recurrence and the readout alone, which is the half with the sequential
/// chain. Its 2·(2·C·Dk·Dv + C²·Dk + C²·Dv + Dk·Dv·C) FLOPs per chunk-head are
/// the products the kernel issues, so a row compares with the probe's model.
void wy_kernel2_case(int64_t Hv, int64_t L, const char* tag)
{
    const int64_t B = 1, C = 64, Dk = 128, Dv = 128, S = C * L;
    const DataType f16 = DataType::f16;
    HostTensor w = make_host_random(std::vector<int64_t>{B, Hv, L, C, Dk}, f16, false, 81,
                                    -0.05f, 0.05f);
    HostTensor u = make_host_random(std::vector<int64_t>{B, Hv, L, C, Dv}, f16, false, 82,
                                    -0.05f, 0.05f);
    HostTensor qe = make_host_random(std::vector<int64_t>{B, Hv, L, C, Dk}, f16, false, 83,
                                     -0.05f, 0.05f);
    HostTensor ka = make_host_random(std::vector<int64_t>{B, Hv, L, C, Dk}, f16, false, 84,
                                     -0.05f, 0.05f);
    HostTensor lc = make_host_random(std::vector<int64_t>{B, Hv, L, Dk}, DataType::f32, false,
                                     85, 0.5f, 1.0f);
    HostTensor st = make_host_random(std::vector<int64_t>{B, Hv, Dk, Dv}, f16, false, 86,
                                     -0.05f, 0.05f);
    HostTensor o = make_host_zeroed(std::vector<int64_t>{B, Hv, S, Dv}, f16, false);

    std::vector<DeviceTensor> dev;
    for (const HostTensor* t : {&w, &u, &qe, &ka, &lc}) {
        dev.push_back(to_device(*t));
    }
    DeviceTensor d_o = alloc_like(o.view, o);
    DeviceTensor d_st = alloc_like(st.view, st);

    nnops::backend::cuda::LaWyParams lp;
    lp.w = dev[0].view.ptr<void>();
    lp.u = dev[1].view.ptr<void>();
    lp.qe = dev[2].view.ptr<void>();
    lp.ka = dev[3].view.ptr<void>();
    lp.lc = dev[4].view.ptr<void>();
    lp.o = d_o.view.ptr<void>();
    lp.st = d_st.view.ptr<void>();
    lp.b = B;
    lp.hv = Hv;
    lp.s = S;
    lp.l = L;
    lp.scale = 1.0f / std::sqrt(static_cast<float>(Dk));

    if (!nnops::backend::cuda::la_wy_recurrent_cuda(lp, nullptr)) {
        std::printf("  %-18s (refused)\n", tag);
        return;
    }
    const double ms =
        cuda_median_ms([&] { nnops::backend::cuda::la_wy_recurrent_cuda(lp, nullptr); });
    const double chunk_heads = double(B * Hv * L);
    const double flops = 2.0 * chunk_heads *
                         (2.0 * C * Dk * Dv + double(C) * C * Dk + double(C) * C * Dv +
                          double(Dk) * Dv * C);
    char geo[96];
    std::snprintf(geo, sizeof(geo), "Hv=%lld L=%lld S=%lld", (long long)Hv, (long long)L,
                  (long long)S);
    print_cuda_row(tag, geo, ms, cuda_gflops(ms, flops));
}

} // namespace

NNOPS_BENCH(cuda_linear_attention_wy_kernel2)
{
    print_cuda_header("LinearAttention chunked-WY route: kernel 2 (recurrence + readout)");
    wy_kernel2_case(32, 32, "Qwen3-Next");
    wy_kernel2_case(8, 16, "GDN");
}

#endif // NNOPS_HAS_CUDA
