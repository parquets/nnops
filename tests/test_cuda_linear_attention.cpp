/// @file test_cuda_linear_attention.cpp
/// @brief CUDA LinearAttention against the CPU backend on the same bytes.
///
/// The CPU kernel is the oracle: it defines the recurrence, the optional
/// conv/beta slots, the value-head-owned state and the in-place state update.
/// Both routes read identical views — the random gates are drawn in [-2, 0], so
/// every decay factor is inside (0, 1] and the state stays bounded over the
/// test's sequence lengths.
///
/// The cases follow the kernel's surface: the plain GDN shape, the delta term
/// with beta, the causal conv (with and without beta, i.e. with an empty
/// conv_w slot), grouped query heads (Gqq == 2), Qwen3-Next's geometry
/// (H_v = 2·H_k, Dv == Dk), a value width that is neither equal to Dk nor a
/// whole column slab, a key width past a thread's 16 rows (Dk = 128), pitched
/// views, a sequence long enough for the CPU's chunked route, and incremental
/// decoding one token at a time (each call S == 1, the state carried forward
/// in place).
///
/// Only compiled under NNOPS_HAS_CUDA.

#include "nnops/ops/linear_attention.hpp"
#include "common/test_harness.hpp"
#include "common/test_helpers.hpp"
#include "common/cuda_test_helper.hpp"

#include <cmath>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#ifdef NNOPS_HAS_CUDA

using namespace nnops;
using namespace nnops::test;

namespace {

/// One problem: geometry, dtype, layout padding and the comparison budget.
struct Case {
    LinearAttentionAttributes attrs;
    int64_t B = 1, H = 1, Hk = 1, S = 8, D = 32;
    int64_t Hv = 0;   // value heads; 0 -> Hk
    int64_t Dv = 0;   // value width; 0 -> D
    int64_t CK = 0;   // conv taps; 0 -> no conv weights
    bool has_beta = false;
    bool padded = false;
    DataType dt = DataType::f32;
    uint64_t seed = 11;
    float atol = 1e-5f;
    float rtol = 1e-4f;
};

int64_t hv_of(const Case& c) { return c.Hv > 0 ? c.Hv : c.Hk; }
int64_t dv_of(const Case& c) { return c.Dv > 0 ? c.Dv : c.D; }

/// Point a host tensor's view at its own storage again. Copying a HostTensor
/// copies the bytes but leaves the view aimed at the source's buffer, so any
/// tensor that travelled by copy (or was built from a temporary in an
/// initializer list) has to be re-aimed before use.
void repoint(HostTensor& t)
{
    if (t.view.rank() == 0) {
        return;
    }
    t.view = TensorView(t.view.shape_span(), t.view.data_type(), t.bytes.data(), t.view.pitch(),
                        t.view.layout());
}

/// A copy whose view points into the copy's own storage. A plain struct copy
/// keeps pointing at the source's bytes — the state is read and overwritten in
/// place, so both backends need their own initial copy.
HostTensor clone_host(const HostTensor& src)
{
    HostTensor t;
    t.bytes = src.bytes;
    t.view = TensorView(src.view.shape_span(), src.view.data_type(), t.bytes.data(),
                        src.view.pitch(), src.view.layout());
    return t;
}

/// The problem's inputs and the two output slots, all in host memory.
struct Problem {
    LinearAttentionAttributes attrs;
    std::vector<HostTensor> ins;
    HostTensor out;
    HostTensor state;
};

Problem build(const Case& c)
{
    const int64_t hv = hv_of(c), dv = dv_of(c);

    const std::vector<int64_t> sq = {c.B, c.H, c.S, c.D};
    const std::vector<int64_t> sk = {c.B, c.Hk, c.S, c.D};
    const std::vector<int64_t> sv = {c.B, hv, c.S, dv};
    const std::vector<int64_t> sg = {c.B, hv, c.S, c.D};
    const std::vector<int64_t> so = {c.B, c.H, c.S, dv};
    const std::vector<int64_t> sst = {c.B, hv, c.D, dv};
    const std::vector<int64_t> scw = {(c.H + c.Hk) * c.D + hv * dv, 1, c.CK};
    const std::vector<int64_t> sbt = {c.B, hv, c.S};

    Problem p;
    // K's head count and the tap count are attributes, not inferred: the op
    // takes them from the caller and checks the tensors against them.
    p.attrs = c.attrs;
    if (c.Hk != c.H) {
        p.attrs.num_kv_heads = c.Hk;
    }
    if (c.CK > 0) {
        p.attrs.conv_kernel_size = c.CK;
    }

    p.ins = {make_host_random(sq, c.dt, c.padded, c.seed + 1, -1.0f, 1.0f),
             make_host_random(sk, c.dt, c.padded, c.seed + 2, -1.0f, 1.0f),
             make_host_random(sv, c.dt, c.padded, c.seed + 3, -1.0f, 1.0f),
             make_host_random(sg, c.dt, c.padded, c.seed + 4, -2.0f, 0.0f)};

    // Optional slots are positional: conv_w is 4 and beta is 5, so beta needs
    // an empty view in slot 4 when there are no taps.
    if (c.CK > 0) {
        p.ins.push_back(make_host_random(scw, c.dt, c.padded, c.seed + 6, -0.5f, 0.5f));
    } else if (c.has_beta) {
        p.ins.push_back(HostTensor{});
    }
    if (c.has_beta) {
        p.ins.push_back(make_host_random(sbt, c.dt, c.padded, c.seed + 7, 0.0f, 1.0f));
    }

    p.out = make_host_zeroed(so, c.dt, c.padded);
    p.state = make_host_random(sst, c.dt, c.padded, c.seed + 5, -1.0f, 1.0f);
    for (auto& t : p.ins) {
        repoint(t);
    }
    repoint(p.out);
    repoint(p.state);
    return p;
}

/// Logical-order floats of a view, walking its strides — a padded view's rows
/// are not contiguous, so a flat read of its bytes would compare padding.
std::vector<float> flatten(const HostTensor& t)
{
    const TensorView& v = t.view;
    const int64_t n = v.numel();
    const int64_t rank = v.rank();
    const auto esz = static_cast<int64_t>(data_type_size(v.data_type()));
    const char* base = reinterpret_cast<const char*>(t.bytes.data());
    std::vector<float> out(static_cast<size_t>(n));
    int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t k = 0; k < n; ++k) {
        int64_t off = 0;
        for (int64_t d = 0; d < rank; ++d) {
            off += idx[d] * v.stride_elems(d);
        }
        out[static_cast<size_t>(k)] = to_floats(base + off * esz, v.data_type(), 1)[0];
        for (int64_t d = rank - 1; d >= 0; --d) {
            if (++idx[d] < v.shape(d)) {
                break;
            }
            idx[d] = 0;
        }
    }
    return out;
}

void compare(const HostTensor& got, const HostTensor& want, float atol, float rtol,
             const char* what)
{
    expect_close(flatten(got), flatten(want), atol, rtol, what);
}

void run_route(Backend be, const LinearAttentionAttributes& attrs, std::vector<TensorView>& outs,
               const std::vector<TensorView>& ins, const char* what)
{
    auto op = LinearAttention::create(attrs, be);
    if (!op) {
        throw std::runtime_error(std::string(what) + ": create failed");
    }
    const Status st =
        op->compute(outs, ins, be == Backend::CUDA ? cuda_context() : cpu_context(), nullptr);
    if (st != Status::ok) {
        throw std::runtime_error(std::string(what) + ": compute returned " + to_string(st));
    }
    if (be == Backend::CUDA) {
        CudaBuffer::sync();
    }
}

/// Run the problem through the CPU backend; returns the outputs.
void run_cpu(const Problem& p, HostTensor& out, HostTensor& state)
{
    std::vector<TensorView> outs = {out.view, state.view};
    std::vector<TensorView> ins;
    for (auto& t : p.ins) {
        ins.push_back(t.view);
    }
    run_route(Backend::CPU, p.attrs, outs, ins, "cpu");
}

/// Run the problem through the CUDA backend over device copies; the outputs
/// land in `out` and `state`.
void run_cuda(const Problem& p, HostTensor& out, HostTensor& state)
{
    std::vector<DeviceTensor> dev;
    dev.reserve(p.ins.size());
    for (auto& t : p.ins) {
        dev.push_back(to_device(t));
    }
    DeviceTensor out_dev = alloc_like(out.view, out);
    DeviceTensor st_dev = alloc_like(state.view, state);
    std::vector<TensorView> outs = {out_dev.view, st_dev.view};
    std::vector<TensorView> ins;
    for (auto& d : dev) {
        ins.push_back(d.view);
    }
    run_route(Backend::CUDA, p.attrs, outs, ins, "cuda");
    from_device(out_dev, out);
    from_device(st_dev, state);
}

/// Build one problem, run both backends and compare both outputs.
void expect_case(const Case& c, const char* what)
{
    Problem p = build(c);
    HostTensor out_cpu = make_host_zeroed(p.out.view.shape_span(), c.dt, c.padded);
    HostTensor st_cpu = clone_host(p.state);   // read, then overwritten in place
    run_cpu(p, out_cpu, st_cpu);

    HostTensor out_dev = make_host_zeroed(p.out.view.shape_span(), c.dt, c.padded);
    HostTensor st_dev = clone_host(p.state);
    run_cuda(p, out_dev, st_dev);

    compare(out_dev, out_cpu, c.atol, c.rtol, (std::string(what) + " output").c_str());
    compare(st_dev, st_cpu, c.atol, c.rtol, (std::string(what) + " state").c_str());
}

/// Incremental decoding: the same single token fed `steps` times, the state
/// carried forward in place on both backends.
void expect_decode(const Case& c, int64_t steps, const char* what)
{
    Case one = c;
    one.S = 1;
    Problem p = build(one);

    HostTensor out_cpu = make_host_zeroed(p.out.view.shape_span(), c.dt, c.padded);
    HostTensor st_cpu = clone_host(p.state);
    {
        std::vector<TensorView> outs = {out_cpu.view, st_cpu.view};
        std::vector<TensorView> ins;
        for (auto& t : p.ins) {
            ins.push_back(t.view);
        }
        for (int64_t t = 0; t < steps; ++t) {
            run_route(Backend::CPU, p.attrs, outs, ins, what);
        }
    }

    HostTensor out_dev = make_host_zeroed(p.out.view.shape_span(), c.dt, c.padded);
    HostTensor st_dev = clone_host(p.state);
    std::vector<DeviceTensor> dev;
    for (auto& t : p.ins) {
        dev.push_back(to_device(t));
    }
    DeviceTensor od = alloc_like(out_dev.view, out_dev);
    DeviceTensor sd = alloc_like(st_dev.view, st_dev);
    {
        std::vector<TensorView> outs = {od.view, sd.view};
        std::vector<TensorView> ins;
        for (auto& d : dev) {
            ins.push_back(d.view);
        }
        for (int64_t t = 0; t < steps; ++t) {
            run_route(Backend::CUDA, p.attrs, outs, ins, what);
        }
    }
    from_device(od, out_dev);
    from_device(sd, st_dev);

    compare(out_dev, out_cpu, c.atol, c.rtol, (std::string(what) + " decode output").c_str());
    compare(st_dev, st_cpu, c.atol, c.rtol, (std::string(what) + " decode state").c_str());
}

} // anonymous namespace

// ============================================================
// Cases
// ============================================================

NNOPS_TEST(linear_attention_cuda_plain) { expect_case(Case{}, "plain"); }

NNOPS_TEST(linear_attention_cuda_beta)
{
    Case c;
    c.has_beta = true;
    expect_case(c, "beta");
}

NNOPS_TEST(linear_attention_cuda_conv)
{
    Case c;
    c.CK = 4;
    expect_case(c, "conv");
}

NNOPS_TEST(linear_attention_cuda_conv_beta)
{
    Case c;
    c.CK = 4;
    c.has_beta = true;
    c.S = 12;
    expect_case(c, "conv+beta");
}

NNOPS_TEST(linear_attention_cuda_conv_short)
{
    // Fewer taps than the four the register window holds: the oldest slots keep
    // their zero weights and the left-of-zero rows stay out of the sum.
    Case c;
    c.CK = 3;
    c.S = 9;
    expect_case(c, "conv 3 taps");
}

NNOPS_TEST(linear_attention_cuda_conv_one)
{
    Case c;
    c.CK = 1;
    c.S = 5;
    expect_case(c, "conv 1 tap");
}

NNOPS_TEST(linear_attention_cuda_conv_wide)
{
    // More taps than the ring holds: the leading ones come from global memory.
    Case c;
    c.CK = 6;
    c.S = 9;
    expect_case(c, "conv 6 taps");
}

NNOPS_TEST(linear_attention_cuda_gqa)
{
    Case c;
    c.H = 6;
    c.Hk = 3;
    c.S = 10;
    c.D = 32;
    expect_case(c, "gqa");
}

NNOPS_TEST(linear_attention_cuda_qwen3_next)
{
    Case c;
    c.H = 4;
    c.Hk = 2;   // H_v = 4: two value heads per key head
    c.Hv = 4;
    c.S = 8;
    c.D = 64;
    c.Dv = 64;
    c.CK = 4;
    c.has_beta = true;
    expect_case(c, "qwen3-next");
}

NNOPS_TEST(linear_attention_cuda_narrow_value)
{
    Case c;
    c.H = 2;
    c.Hk = 2;
    c.S = 8;
    c.D = 64;
    c.Dv = 40;   // not Dk, and not a whole column slab
    c.has_beta = true;
    expect_case(c, "narrow value");
}

NNOPS_TEST(linear_attention_cuda_dk128)
{
    Case c;
    c.H = 2;
    c.Hk = 2;
    c.S = 8;
    c.D = 128;
    c.Dv = 128;
    expect_case(c, "dk 128");
}

NNOPS_TEST(linear_attention_cuda_dk_odd)
{
    Case c;
    c.H = 2;
    c.Hk = 2;
    c.S = 8;
    c.D = 48;   // pads to the 16-row slice
    c.Dv = 24;
    c.CK = 4;
    expect_case(c, "odd head dim");
}

NNOPS_TEST(linear_attention_cuda_narrow_key)
{
    // Dk under 16 puts one thread in every row group, so the block's column
    // slab is wider than the block — the value staging walks more than one
    // column per thread.
    Case c;
    c.H = 1;
    c.Hk = 1;
    c.S = 6;
    c.D = 8;
    c.Dv = 256;
    expect_case(c, "narrow key");
}

NNOPS_TEST(linear_attention_cuda_padded)
{
    Case c;
    c.padded = true;
    c.CK = 4;
    c.has_beta = true;
    c.S = 10;
    expect_case(c, "padded");
}

NNOPS_TEST(linear_attention_cuda_long)
{
    Case c;
    c.H = 2;
    c.Hk = 2;
    c.S = 128;   // the CPU route chunks from 64 tokens up
    c.D = 64;
    c.CK = 4;
    c.has_beta = true;
    c.atol = 2e-5f;
    c.rtol = 2e-4f;
    expect_case(c, "long");
}

NNOPS_TEST(linear_attention_cuda_f16)
{
    Case c;
    c.dt = DataType::f16;
    c.H = 2;
    c.Hk = 2;
    c.S = 8;
    c.D = 64;
    c.Dv = 32;
    c.CK = 4;
    c.has_beta = true;
    c.atol = kF16AccumTol;
    c.rtol = 5e-2f;
    expect_case(c, "f16");
}

NNOPS_TEST(linear_attention_cuda_decode)
{
    Case c;
    c.H = 2;
    c.Hk = 2;
    c.S = 1;
    c.D = 64;
    c.Dv = 64;
    c.has_beta = true;
    expect_decode(c, 16, "decode");
}

#endif // NNOPS_HAS_CUDA
