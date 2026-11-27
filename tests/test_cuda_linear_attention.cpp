/// @file test_cuda_linear_attention.cpp
/// @brief CUDA LinearAttention against the CPU backend on the same bytes.
///
/// The CPU kernel is the oracle: it defines the recurrence, the optional
/// conv/beta slots, the value-head-owned state and the in-place state update.
/// Both routes read identical views — the random gates are drawn in [-2, 0], so
/// every decay factor is inside (0, 1] and the state stays bounded over the
/// test's sequence lengths. The f16 chunked route needs a milder gate than that
/// (its panels hold 1/L_s in f16) and its own small inputs, so its cases draw
/// both from the trained range; the steep-gate case checks the fallback.
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
#include "nnops/detail/half.hpp"
#include "backend/cuda/attention/linear_attention_wy_host.hpp"
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
    // Log-decay range. The default is steep on purpose (a decay of 0.14 to 1 a
    // token); the chunked route needs the mild range a trained gate produces,
    // because its panels hold 1/L_s in f16.
    float glo = -2.0f, ghi = 0.0f;
    // Q, K and V are drawn from [-xs, xs]. A mild gate barely contracts, so a
    // unit-scale input drives the state of a long sequence out of f16 — the
    // route's own copies of the tests draw small inputs to stay inside it.
    float xs = 1.0f;
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

    p.ins = {make_host_random(sq, c.dt, c.padded, c.seed + 1, -c.xs, c.xs),
             make_host_random(sk, c.dt, c.padded, c.seed + 2, -c.xs, c.xs),
             make_host_random(sv, c.dt, c.padded, c.seed + 3, -c.xs, c.xs),
             make_host_random(sg, c.dt, c.padded, c.seed + 4, c.glo, c.ghi)};

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

NNOPS_TEST(linear_attention_cuda_dk256_conv)
{
    // Dk past a thread's own 16-row group: the block walks two slots of raw
    // rows, and the four-tap fold's register taps belong to one channel each.
    Case c;
    c.H = 1;
    c.Hk = 1;
    c.S = 8;
    c.D = 256;
    c.Dv = 256;
    c.CK = 4;
    expect_case(c, "dk 256 conv");
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

NNOPS_TEST(linear_attention_cuda_narrow_key_conv)
{
    // The other axis of the same split: Dk under 16 puts one thread in every
    // row group, so the value slab is two slots wide and the fold walks the
    // second one -- and the ring's zero rows are wider than the thread count.
    Case c;
    c.H = 1;
    c.Hk = 1;
    c.S = 6;
    c.D = 8;
    c.Dv = 256;
    c.CK = 4;
    expect_case(c, "narrow key conv");
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

// ============================================================
// The chunked-WY prototype's recurrent kernel (kernel 2)
// ============================================================
//
// The prototype reads the panels a chunk-local pass would have built (W, U,
// the gated Q and K, and the chunk decay), so the test synthesizes random ones
// and checks the recurrence against the same algebra in f32. Three roundings
// the kernel performs are mirrored: X, Aq and the state go through f16.

namespace {

namespace nb = nnops::backend::cpu;

constexpr int64_t kWyC = 64;
constexpr int64_t kWyDk = 128;
constexpr int64_t kWyDv = 128;

float rnd16(float x)
{
    return nb::half_to_float(nb::float_to_half(x));
}

struct WyProblem {
    HostTensor w, u, qe, ka, lc, out, state;
    int64_t B = 1, Hv = 1, L = 1;
    float scale = 1.0f;
};

WyProblem build_wy(int64_t B, int64_t Hv, int64_t L, float scale, uint64_t seed)
{
    WyProblem p;
    p.B = B;
    p.Hv = Hv;
    p.L = L;
    p.scale = scale;
    const auto f16 = DataType::f16;
    const auto f32 = DataType::f32;
    // The panels are drawn small on purpose: the reference emulates the
    // kernel's f16 roundings, and a state that grows into the hundreds turns
    // every sum-order flip into a whole f16 ulp (0.5 at 832), swamping the
    // comparison with quantization noise the kernel cannot avoid either.
    p.w = make_host_random(std::vector<int64_t>{B, Hv, L, kWyC, kWyDk}, f16, false, seed + 1,
                           -0.05f, 0.05f);
    p.u = make_host_random(std::vector<int64_t>{B, Hv, L, kWyC, kWyDv}, f16, false, seed + 2,
                           -0.05f, 0.05f);
    p.qe = make_host_random(std::vector<int64_t>{B, Hv, L, kWyC, kWyDk}, f16, false, seed + 3,
                            -0.05f, 0.05f);
    p.ka = make_host_random(std::vector<int64_t>{B, Hv, L, kWyC, kWyDk}, f16, false, seed + 4,
                            -0.05f, 0.05f);
    p.lc = make_host_random(std::vector<int64_t>{B, Hv, L, kWyDk}, f32, false, seed + 5, 0.5f,
                            1.0f);
    p.out = make_host_zeroed(std::vector<int64_t>{B, Hv, kWyC * L, kWyDv}, f16, false);
    p.state = make_host_random(std::vector<int64_t>{B, Hv, kWyDk, kWyDv}, f16, false, seed + 6,
                               -0.05f, 0.05f);
    return p;
}

void wy_reference(const WyProblem& p, std::vector<float>& out, std::vector<float>& state)
{
    const auto f16 = DataType::f16;
    const std::vector<float> w = to_floats(p.w.bytes.data(), f16, p.w.view.numel());
    const std::vector<float> u = to_floats(p.u.bytes.data(), f16, p.u.view.numel());
    const std::vector<float> qe = to_floats(p.qe.bytes.data(), f16, p.qe.view.numel());
    const std::vector<float> ka = to_floats(p.ka.bytes.data(), f16, p.ka.view.numel());
    const std::vector<float> lc = to_floats(p.lc.bytes.data(), DataType::f32, p.lc.view.numel());
    const std::vector<float> st_in = to_floats(p.state.bytes.data(), f16, p.state.view.numel());

    const int64_t S = kWyC * p.L;
    out.assign(static_cast<size_t>(p.B * p.Hv * S * kWyDv), 0.0f);
    state.assign(static_cast<size_t>(p.B * p.Hv * kWyDk * kWyDv), 0.0f);

    std::vector<float> x(static_cast<size_t>(kWyC * kWyDv));
    std::vector<float> aq(static_cast<size_t>(kWyC * kWyC));
    std::vector<float> sv(static_cast<size_t>(kWyDk * kWyDv));

    for (int64_t b = 0; b < p.B; ++b) {
        for (int64_t h = 0; h < p.Hv; ++h) {
            const int64_t bh = b * p.Hv + h;
            for (int64_t i = 0; i < kWyDk * kWyDv; ++i) {
                sv[static_cast<size_t>(i)] = st_in[static_cast<size_t>(bh * kWyDk * kWyDv + i)];
            }
            for (int64_t c = 0; c < p.L; ++c) {
                const float* W = &w[static_cast<size_t>((bh * p.L + c) * kWyC * kWyDk)];
                const float* U = &u[static_cast<size_t>((bh * p.L + c) * kWyC * kWyDv)];
                const float* QE = &qe[static_cast<size_t>((bh * p.L + c) * kWyC * kWyDk)];
                const float* KA = &ka[static_cast<size_t>((bh * p.L + c) * kWyC * kWyDk)];
                const float* LC = &lc[static_cast<size_t>((bh * p.L + c) * kWyDk)];

                for (int64_t r = 0; r < kWyC; ++r) {
                    for (int64_t j = 0; j < kWyDv; ++j) {
                        float acc = 0.0f;
                        for (int64_t i = 0; i < kWyDk; ++i) {
                            acc += W[r * kWyDk + i] * sv[static_cast<size_t>(i * kWyDv + j)];
                        }
                        x[static_cast<size_t>(r * kWyDv + j)] =
                            rnd16(U[r * kWyDv + j] - acc);
                    }
                }
                for (int64_t r = 0; r < kWyC; ++r) {
                    for (int64_t s2 = 0; s2 < kWyC; ++s2) {
                        float acc = 0.0f;
                        if (s2 <= r) {
                            for (int64_t i = 0; i < kWyDk; ++i) {
                                acc += QE[r * kWyDk + i] * KA[s2 * kWyDk + i];
                            }
                            acc = rnd16(acc);
                        }
                        aq[static_cast<size_t>(r * kWyC + s2)] = acc;
                    }
                }
                float* O = &out[static_cast<size_t>((bh * S + c * kWyC) * kWyDv)];
                for (int64_t r = 0; r < kWyC; ++r) {
                    for (int64_t j = 0; j < kWyDv; ++j) {
                        float acc = 0.0f;
                        for (int64_t i = 0; i < kWyDk; ++i) {
                            acc += QE[r * kWyDk + i] * sv[static_cast<size_t>(i * kWyDv + j)];
                        }
                        for (int64_t s2 = 0; s2 <= r; ++s2) {
                            acc += aq[static_cast<size_t>(r * kWyC + s2)] *
                                   x[static_cast<size_t>(s2 * kWyDv + j)];
                        }
                        O[r * kWyDv + j] = rnd16(p.scale * acc);
                    }
                }
                for (int64_t i = 0; i < kWyDk; ++i) {
                    for (int64_t j = 0; j < kWyDv; ++j) {
                        float acc = 0.0f;
                        for (int64_t s2 = 0; s2 < kWyC; ++s2) {
                            acc += KA[s2 * kWyDk + i] * x[static_cast<size_t>(s2 * kWyDv + j)];
                        }
                        sv[static_cast<size_t>(i * kWyDv + j)] =
                            rnd16(LC[i] * (sv[static_cast<size_t>(i * kWyDv + j)] + acc));
                    }
                }
            }
            for (int64_t i = 0; i < kWyDk * kWyDv; ++i) {
                state[static_cast<size_t>(bh * kWyDk * kWyDv + i)] = sv[static_cast<size_t>(i)];
            }
        }
    }
}

void expect_wy(int64_t B, int64_t Hv, int64_t L, uint64_t seed, float atol, float rtol,
               const char* what)
{
    WyProblem p = build_wy(B, Hv, L, 1.0f / std::sqrt(static_cast<float>(kWyDk)), seed);
    std::vector<float> out_ref, st_ref;
    wy_reference(p, out_ref, st_ref);

    std::vector<DeviceTensor> dev;
    for (const HostTensor* t : {&p.w, &p.u, &p.qe, &p.ka, &p.lc}) {
        dev.push_back(to_device(*t));
    }
    DeviceTensor d_out = alloc_like(p.out.view, p.out);
    DeviceTensor d_state = alloc_like(p.state.view, p.state);

    nnops::backend::cuda::LaWyParams lp;
    lp.w = dev[0].view.ptr<void>();
    lp.u = dev[1].view.ptr<void>();
    lp.qe = dev[2].view.ptr<void>();
    lp.ka = dev[3].view.ptr<void>();
    lp.lc = dev[4].view.ptr<void>();
    lp.o = d_out.view.ptr<void>();
    lp.st = d_state.view.ptr<void>();
    lp.b = B;
    lp.hv = Hv;
    lp.s = kWyC * L;
    lp.l = L;
    lp.scale = p.scale;
    NNOPS_EXPECT_TRUE(nnops::backend::cuda::la_wy_recurrent_cuda(lp, nullptr));
    CudaBuffer::sync();

    HostTensor out_dev =
        make_host_zeroed(p.out.view.shape_span(), DataType::f16, false);
    HostTensor st_dev =
        make_host_zeroed(p.state.view.shape_span(), DataType::f16, false);
    from_device(d_out, out_dev);
    from_device(d_state, st_dev);
    const std::vector<float> out_got =
        to_floats(out_dev.bytes.data(), DataType::f16, out_dev.view.numel());
    const std::vector<float> st_got =
        to_floats(st_dev.bytes.data(), DataType::f16, st_dev.view.numel());
    expect_close(out_got, out_ref, atol, rtol, (std::string(what) + " output").c_str());
    expect_close(st_got, st_ref, atol, rtol, (std::string(what) + " state").c_str());
}

} // namespace

// ============================================================
// The chunked-WY route as the operator takes it
// ============================================================
//
// The route is only representable while the panels' 1/L_s stays inside f16, so
// the gates are drawn from a trained layer's range here and nothing like the
// [-2, 0] the other cases use. The chunk is 64 rows at Dk = Dv = 128, so the
// shapes are the ones a prefill actually has; the CPU oracle is on its own
// chunked (C = 16) route at these lengths, which is a different association of
// the same sum, and the panels and the state are f16 on the GPU side.

namespace {

/// expect_case, plus the only observable the operator offers for which route it
/// took: the route's own call count.
void expect_route(const Case& c, bool want_route, const char* what)
{
    const int64_t before = nnops::backend::cuda::la_wy_route_runs();
    expect_case(c, what);
    const int64_t ran = nnops::backend::cuda::la_wy_route_runs() - before;
    if (ran != (want_route ? 1 : 0)) {
        throw std::runtime_error(std::string(what) + ": route runs " + std::to_string(ran) +
                                 ", wanted " + (want_route ? "1" : "0"));
    }
}

Case wy_case(int64_t H, int64_t Hk, int64_t Hv, int64_t S)
{
    Case c;
    c.dt = DataType::f16;
    c.H = H;
    c.Hk = Hk;
    c.Hv = Hv;
    c.S = S;
    c.D = 128;
    c.Dv = 128;
    c.CK = 4;
    c.has_beta = true;
    c.glo = -0.05f;
    c.ghi = 0.0f;
    c.xs = 0.2f;
    return c;
}

} // namespace

NNOPS_TEST(linear_attention_cuda_wy_route)
{
    // Two chunks, the conv and the delta rate: the smallest call the route
    // takes at all.
    Case c = wy_case(2, 2, 0, 128);
    c.atol = 5e-4f;
    c.rtol = 1e-2f;
    expect_route(c, true, "wy route");
}

NNOPS_TEST(linear_attention_cuda_wy_route_qwen)
{
    // Qwen3-Next's head geometry (two value heads a key head) over three
    // chunks, and the optional slots absent: no conv, no beta. A missing beta
    // is 1.0 and a missing conv is the identity, both on the panel side.
    Case c = wy_case(4, 2, 4, 192);
    c.CK = 0;
    c.has_beta = false;
    c.atol = 5e-4f;
    c.rtol = 1e-2f;
    expect_route(c, true, "wy route qwen3-next");
}

NNOPS_TEST(linear_attention_cuda_wy_route_qwen_conv)
{
    // The head geometry with everything on: the conv channel bases are the
    // packed Q | K | V blob's, which is where a key head shared by two value
    // heads can go wrong, and four chunks cross a boundary more than once.
    Case c = wy_case(4, 2, 4, 256);
    c.atol = 5e-4f;
    c.rtol = 1e-2f;
    expect_route(c, true, "wy route qwen3-next conv");
}

NNOPS_TEST(linear_attention_cuda_wy_route_falls_back)
{
    // The same geometry with a steep gate: the panels would leave f16, so the
    // operator has to run the per-token kernel and match the oracle anyway.
    Case c = wy_case(2, 2, 0, 128);
    c.glo = -2.0f;
    c.ghi = 0.0f;
    c.atol = kF16AccumTol;
    c.rtol = 5e-2f;
    expect_route(c, false, "wy route fallback");
}

NNOPS_TEST(linear_attention_cuda_wy_kernel2)
{
    // Two heads, four chunks: the chain is what the kernel is about, and a
    // second head catches a head-offset mistake. A single head and two chunks
    // is the other extreme (one block, one recurrence step pair).
    //
    // The budget is the f16 roundings' own: Aq and X are rounded by both sides,
    // but the mma's f32 accumulation order is not the reference's, so an entry
    // sitting on a rounding boundary can flip by one ulp — 5e-4 of a term whose
    // scale is O(0.25), and a readout that cancels to 4e-3 amplifies it. A
    // layout error is O(0.1), three orders above this budget.
    expect_wy(1, 2, 4, 71, 2e-3f, 5e-3f, "wy kernel2");
    expect_wy(1, 1, 2, 73, 2e-3f, 5e-3f, "wy kernel2 single head");
}

#endif // NNOPS_HAS_CUDA
