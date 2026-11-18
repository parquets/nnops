/// @file bench_sme_mma.cpp
/// @brief Kernel-level benchmarks for the AArch64 SME MMA micro-kernels.
///
/// The int8 SME kernel has no route yet (the I8Route integration is the next
/// step), so there is no MatMul-level number to compare. This times it where it
/// lives: packed A and B, one k-block per call, charged 2*M*N*K operations. The
/// second arm adds the per-call packs, which is what the route would pay
/// without a prepacked B.
///
/// f32/f16 are benchmarked at route level in bench_matmul (matmul_sme_ab).

#include "common/bench_harness.hpp"

#if defined(NNOPS_ARCH_SME)

#include "backend/cpu/aarch64/mma_i8_sme.hpp"
#include "backend/cpu/aarch64/mma_sme.hpp"
#include "backend/cpu/aarch64/pack_i8_sme.hpp"
#include "detail/bench_knobs.hpp"
#include "nnops/ops/matmul.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

namespace {

using namespace nnops;
using namespace nnops::backend::cpu::aarch64;

/// Deterministic fill; the values only need to be varied, not meaningful.
void fill_i8(std::vector<int8_t>& v, uint32_t seed)
{
    uint32_t st = seed;
    for (auto& x : v) {
        st = st * 1664525u + 1013904223u;
        x = static_cast<int8_t>((st >> 16) % 251 - 125);
    }
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

template <typename Fn>
double time_gops(int64_t ops, int iters, Fn&& fn)
{
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) { fn(); }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double s = std::chrono::duration<double>(t1 - t0).count();
    return static_cast<double>(ops) * iters / s * 1e-9;
}

void bench_sme_i8(const char* tag, int M, int N, int K, int rounds = 5, int iters = 100)
{
    if (!available_sme()) {
        std::printf("  SME i8 %s skipped: no SME kernels on this build or CPU\n", tag);
        return;
    }

    std::vector<int8_t> a(static_cast<size_t>(M) * K);
    std::vector<int8_t> b(static_cast<size_t>(K) * N);
    fill_i8(a, 0x5EED5EEDu);
    fill_i8(b, 0xC0FFEEu);

    std::vector<int8_t> pa(static_cast<size_t>(packed_a_elems_i8_sme(M, K)));
    std::vector<int8_t> pb(static_cast<size_t>(packed_b_elems_i8_sme(N, K)));
    std::vector<int32_t> c(static_cast<size_t>(M) * N);

    const int64_t ops = 2 * static_cast<int64_t>(M) * N * K;
    auto kernel = [&] {
        mma_i8_sme(M, N, K, c.data(), N, pa.data(), pb.data(), /*zero_mode=*/false, INT32_MIN,
                    INT32_MAX);
    };
    auto pack_and_kernel = [&] {
        pack_a_i8_sme(M, K, pa.data(), a.data(), K, false, false);
        pack_b_i8_sme(N, K, pb.data(), b.data(), N, false);
        kernel();
    };

    // Warm both arms: the first timed call otherwise inherits a cold clock and
    // reads as a spurious low round on a fanless host.
    for (int i = 0; i < 5; ++i) { pack_and_kernel(); }
    for (int i = 0; i < 20; ++i) { kernel(); }

    std::printf("  SME i8 %s  %dx%dx%d  1T  rounds=%d iters=%d  (2*M*N*K)\n", tag, M, N, K,
                rounds, iters);
    std::vector<double> kern, packed;
    for (int r = 0; r < rounds; ++r) {
        kern.push_back(time_gops(ops, iters, kernel));
        packed.push_back(time_gops(ops, iters, pack_and_kernel));
        std::printf("    round %d:  kernel=%7.1f  pack+kernel=%7.1f  GOPS   kernel/packed=%.2f\n",
                    r, kern.back(), packed.back(), kern.back() / packed.back());
    }
    std::printf("    median:   kernel=%7.1f  pack+kernel=%7.1f  GOPS   kernel/packed=%.2f\n",
                median(kern), median(packed), median(kern) / median(packed));
}

/// A whole int8 MatMul through the public op: packed on every call, or with a
/// prepacked weight. This is the number the kernel bench cannot show, because
/// every call also packs A and, without a blob, B.
struct I8Route {
    int64_t M, N, K;
    std::unique_ptr<MatMul> op;
    std::vector<int8_t> a_buf, b_buf;
    std::vector<int32_t> c_buf;
    TensorView a, b, c;
    TensorView ins[2];
    std::vector<int8_t> blob;
    TensorView packed;
    TensorView pins[2];
    bool has_blob = false;

    I8Route(int64_t m, int64_t k, int64_t n) : M(m), N(n), K(k)
    {
        const int64_t ashape[] = {M, K};
        const int64_t bshape[] = {K, N};
        a_buf.resize(static_cast<size_t>(M * K));
        b_buf.resize(static_cast<size_t>(K * N));
        fill_i8(a_buf, 0x5EED5EEDu);
        fill_i8(b_buf, 0xC0FFEEu);
        a = TensorView(ashape, DataType::s8, a_buf.data());
        b = TensorView(bshape, DataType::s8, b_buf.data());
        op = MatMul::create(MatMulAttributes{}, Backend::CPU);
        const TensorDesc d[] = {a.desc(), b.desc()};
        const auto od = op->getOutputTensorDesc(d)[0];
        c_buf.assign(static_cast<size_t>(od.numel()), 0);
        c = TensorView(std::span<const int64_t>(od.dims.data(), static_cast<size_t>(od.rank)),
                       od.dtype, c_buf.data());
        ins[0] = a;
        ins[1] = b;
    }

    /// Query, allocate, fill: the same two-step the caller runs at load time.
    void prepack()
    {
        TensorView q;
        TensorView oq[] = {q};
        const TensorView bq[] = {b};
        op->prepackWeights(bq, oq);
        q = oq[0];
        if (q.rank() != 2) {
            return;
        }
        blob.assign(static_cast<size_t>(q.desc().numel()), 0);
        packed = TensorView(q.shape_span(), q.data_type(), blob.data(), q.pitch(), q.layout(),
                            b.quant_params());
        TensorView opk[] = {packed};
        op->prepackWeights(bq, opk);
        pins[0] = a;
        pins[1] = packed;
        has_blob = packed.layout() == TensorLayout::PackedBSme;
    }

    double time(bool use_blob, int iters)
    {
        const TensorView* p = use_blob ? pins : ins;
        for (int i = 0; i < 3; ++i) { op->compute(c, {p, p + 2}, {}, nullptr); }
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) { op->compute(c, {p, p + 2}, {}, nullptr); }
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double s = std::chrono::duration<double>(t1 - t0).count();
        return 2.0 * static_cast<double>(M) * static_cast<double>(N) *
               static_cast<double>(K) * iters / s * 1e-9;
    }
};

void bench_i8_route(const char* tag, int M, int K, int N, int rounds = 4, int iters = 10)
{
    if (!available_sme()) {
        std::printf("  SME i8 route %s skipped: no SME kernels on this build or CPU\n", tag);
        return;
    }
    I8Route r(M, K, N);
    r.prepack();

    std::printf("  SME i8 route %s  %dx%dx%d  1T  rounds=%d iters=%d  blob=%s\n", tag, M, N, K,
                rounds, iters, r.has_blob ? "sme" : "none");
    std::vector<double> off, on, raw, pk;
    for (int i = 0; i < rounds; ++i) {
        detail::disable_sme.store(1, std::memory_order_relaxed);
        off.push_back(r.time(false, iters));
        detail::disable_sme.store(0, std::memory_order_relaxed);
        on.push_back(r.time(false, iters));
        raw.push_back(r.time(false, iters));
        pk.push_back(r.time(true, iters));
        std::printf("    round %d:  sme=%7.1f  generic=%7.1f  (%.2fx)   raw=%7.1f  prepacked=%7.1f"
                    "  (%.2fx)\n", i, on.back(), off.back(), on.back() / off.back(), raw.back(),
                    pk.back(), pk.back() / raw.back());
    }
    detail::disable_sme.store(0, std::memory_order_relaxed);

    const double mo = median(off), mn = median(on);
    const double mr = median(raw), mp = median(pk);
    std::printf("    median:   sme=%7.1f  generic=%7.1f  (%.2fx)   raw=%7.1f  prepacked=%7.1f"
                "  (%.2fx)\n", mn, mo, mn / mo, mr, mp, mp / mr);
}

} // namespace

NNOPS_BENCH(sme_i8_kernel) {
    // The shapes the int8 route would see: a prefill-ish 384x768 face at two
    // k-block lengths, then square products.
    bench_sme_i8("kc1024", 384, 768, 1024);
    bench_sme_i8("kc2048", 384, 768, 2048);
    bench_sme_i8("square", 1024, 1024, 1024);
}

NNOPS_BENCH(sme_i8_route) {
    // Whole products through MatMul: the SME route against the planned one, and
    // a prepacked weight against the per-call B pack.
    bench_i8_route("square", 1024, 1024, 1024, 4, 8);
    bench_i8_route("prefill", 384, 1024, 768, 4, 10);
    bench_i8_route("wide", 128, 2048, 2048, 4, 8);
    bench_i8_route("layer", 16, 1024, 1024, 4, 30);
}

#else

NNOPS_BENCH(sme_i8_kernel) {
    std::printf("  SME i8 skipped: this build has no SME kernels\n");
}

#endif // NNOPS_ARCH_SME
