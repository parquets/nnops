/// @file bench_cuda_elementwise.cpp
/// @brief Achieved bandwidth of the elementwise-shaped CUDA kernels.
///
/// Activation, Softmax and the row norms move whole tensors and do little per
/// element, so the number worth reading is bandwidth, not FLOPs: every row
/// prints the GB/s implied by the element-sized transfers the op has to make.
///
/// Two of the arms are already 128-bit accesses and serve as the machine's
/// reference — Permute's copy engine moves 2N elements, Eltwise Add 3N — so a
/// scalar-load kernel's row can be read directly against the one that has the
/// same traffic. The gap that shows up there is the vectorization headroom, and
/// it is a dtype story: a 4-byte element fills a 128 B warp transaction with
/// scalar loads, a 2-byte one only half of it.
///
/// Pass counts are stated where they differ from the traffic the row assumes,
/// since softmax reads its row three times today and the norms read it twice.

#include "nnops/ops/activation.hpp"
#include "nnops/ops/eltwise.hpp"
#include "nnops/ops/norm.hpp"
#include "nnops/ops/permute.hpp"
#include "nnops/ops/softmax.hpp"

#include "common/bench_harness.hpp"
#include "common/cuda_bench_helper.hpp"
#include "common/cuda_test_helper.hpp"

#include <cstdio>
#include <string>
#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

double gbs(double ms, double bytes)
{
    return ms > 0.0 ? bytes / (ms * 1.0e6) : 0.0;
}

const char* dn(DataType dt)
{
    return dt == DataType::f32 ? "f32" : (dt == DataType::f16 ? "f16" : "bf16");
}

double es(DataType dt)
{
    return dt == DataType::f32 ? 4.0 : 2.0;
}

/// The report row: what the op is, the geometry, its time and the bandwidth it
/// implies, given the number of element-sized transfers it has to make.
void row(const char* tag, const std::string& geom, double ms, double bytes)
{
    std::printf("  %-10s %-30s %9.4f ms %8.1f GB/s\n", tag, geom.c_str(), ms,
                gbs(ms, bytes));
}

std::string geom2(DataType dt, int64_t n)
{
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%s n=%lld", dn(dt), (long long)n);
    return buf;
}

void bench_reference(DataType dt, int64_t n, const char* tag)
{
    HostTensor a = make_host_random(std::vector<int64_t>{n}, dt, false, 11, -0.5f, 0.5f);
    HostTensor b = make_host_random(std::vector<int64_t>{n}, dt, false, 13, -0.5f, 0.5f);
    HostTensor o = make_host_zeroed(std::vector<int64_t>{n}, dt, false);
    DeviceTensor da = to_device(a);
    DeviceTensor db = to_device(b);
    DeviceTensor dout = alloc_like(o.view, o);

    // Identity permutation: the movement engine's 16 B copy run, 2N traffic.
    auto perm = Permute::create(PermuteAttributes{{0}}, Backend::CUDA);
    TensorView pout = dout.view;
    const TensorView pin[] = {da.view};
    const double pms = cuda_median_ms([&] { cuda_launch(*perm, pout, pin); });
    row(tag, std::string("permute-id ") + geom2(dt, n), pms, 2.0 * double(n) * es(dt));

    // Eltwise Add: two reads, one write, 3N traffic.
    auto add = Eltwise::create(EltwiseAttributes{EltwiseType::Add, false}, Backend::CUDA);
    TensorView aout = dout.view;
    const TensorView ain[] = {da.view, db.view};
    const double ams = cuda_median_ms([&] { cuda_launch(*add, aout, ain); });
    row(tag, std::string("eltwise-add ") + geom2(dt, n), ams, 3.0 * double(n) * es(dt));
}

void bench_activation(DataType dt, int64_t n, ActivationType type, const char* name)
{
    HostTensor a = make_host_random(std::vector<int64_t>{n}, dt, false, 11, -0.5f, 0.5f);
    HostTensor o = make_host_zeroed(std::vector<int64_t>{n}, dt, false);
    DeviceTensor da = to_device(a);
    DeviceTensor dout = alloc_like(o.view, o);

    ActivationAttributes attrs;
    attrs.type = type;
    auto op = Activation::create(attrs, Backend::CUDA);
    TensorView out = dout.view;
    const TensorView in[] = {da.view};
    const double ms = cuda_median_ms([&] { cuda_launch(*op, out, in); });
    char geom[128];
    std::snprintf(geom, sizeof(geom), "%s %s", name, geom2(dt, n).c_str());
    row("activation", geom, ms, 2.0 * double(n) * es(dt));
}

void bench_softmax(DataType dt, int64_t rows, int64_t v)
{
    HostTensor a = make_host_random(std::vector<int64_t>{rows, v}, dt, false, 11, -4.0f, 4.0f);
    HostTensor o = make_host_zeroed(std::vector<int64_t>{rows, v}, dt, false);
    DeviceTensor da = to_device(a);
    DeviceTensor dout = alloc_like(o.view, o);

    auto op = Softmax::create(SoftmaxAttributes{-1, false, 1.0f}, Backend::CUDA);
    TensorView out = dout.view;
    const TensorView in[] = {da.view};
    const double ms = cuda_median_ms([&] { cuda_launch(*op, out, in); });
    char geom[128];
    std::snprintf(geom, sizeof(geom), "%s r%lld v%lld", dn(dt), (long long)rows,
                  (long long)v);
    // Three reads (max, exp-sum, normalize) and one write, so 4N: the kernel
    // reads the row once per pass, and an online max+sum would make it 3N.
    row("softmax", geom, ms, 4.0 * double(rows) * double(v) * es(dt));
}

void bench_norm(DataType dt, NormType type, int64_t rows, int64_t h, const char* name)
{
    HostTensor a = make_host_random(std::vector<int64_t>{rows, h}, dt, false, 11, -1.0f, 1.0f);
    HostTensor s = make_host_random(std::vector<int64_t>{h}, dt, false, 81, 0.5f, 1.5f);
    HostTensor b = make_host_random(std::vector<int64_t>{h}, dt, false, 82, -0.1f, 0.1f);
    HostTensor o = make_host_zeroed(std::vector<int64_t>{rows, h}, dt, false);
    DeviceTensor da = to_device(a);
    DeviceTensor ds = to_device(s);
    DeviceTensor db = to_device(b);
    DeviceTensor dout = alloc_like(o.view, o);

    NormAttributes attrs;
    attrs.type = type;
    auto op = Norm::create(attrs, Backend::CUDA);
    TensorView out = dout.view;
    const bool with_bias = type == NormType::LayerNorm;
    std::vector<TensorView> in = {da.view, ds.view};
    if (with_bias) { in.push_back(db.view); }
    const double ms = cuda_median_ms([&] { cuda_launch(*op, out, in); });
    char geom[128];
    std::snprintf(geom, sizeof(geom), "%s %s r%lld h%lld", dn(dt), name,
                  (long long)rows, (long long)h);
    // Two reads of the row (stats, then apply) and one write.
    row(name, geom, ms, 3.0 * double(rows) * double(h) * es(dt));
}

}  // namespace

NNOPS_BENCH(cuda_bw_elementwise) {
    const DataType dts[] = {DataType::f32, DataType::f16};
    for (DataType dt : dts) {
        for (int64_t n : {int64_t(1) << 20, int64_t(8) << 20}) {
            bench_reference(dt, n, dn(dt));
            bench_activation(dt, n, ActivationType::Relu, "relu");
            bench_activation(dt, n, ActivationType::Silu, "silu");
        }
        bench_softmax(dt, 2048, 2048);
        bench_softmax(dt, 1024, 8192);
        // 20001 f32 is 80 KB and past the shared-memory budget, so this row is the
        // non-resident script: it reads the row twice instead of keeping it.
        bench_softmax(dt, 512, 20001);
        bench_norm(dt, NormType::LayerNorm, 1024, 4096, "layernorm");
        bench_norm(dt, NormType::RMSNorm, 1024, 4096, "rmsnorm");
        bench_norm(dt, NormType::LayerNorm, 512, 20001, "layernorm");
    }
}

#endif  // NNOPS_HAS_CUDA
