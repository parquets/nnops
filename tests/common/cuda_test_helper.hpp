/// @file cuda_test_helper.hpp
/// @brief Device buffers, error checks and the host-oracle runner the CUDA
///        operator tests share.
///
/// A TensorView is only a pointer plus strides, so a device tensor is an
/// ordinary view over a cudaMalloc'd block: the tests build the host tensors,
/// upload them, run the op with a ComputeContext whose stream is the default
/// one, then synchronise and download before comparing. Every launch is checked
/// after the synchronise, so a kernel that faults fails the test that caused it
/// rather than the next one.
///
/// The oracle is the CPU backend running the same attributes on the same input
/// values — the contract is identical, so a difference is a CUDA bug. Rows are
/// built with make_packed (a 32 B pitch) for channel-first tensors so the padded-stride
/// path is the one under test; channel-last tensors must be dense, because their
/// strides never consult the pitch.
///
/// Only compiled under NNOPS_HAS_CUDA.

#pragma once

#ifdef NNOPS_HAS_CUDA

#include "nnops/core/compute_context.hpp"
#include "nnops/core/data_type.hpp"
#include "nnops/core/tensor_view.hpp"

#include "random_tensor.hpp"
#include "test_helpers.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace nnops::test {

/// Throws with the failing call and the runtime's message.
inline void cuda_check_impl(cudaError_t err, const char* call, const char* file,
                            int line)
{
    if (err == cudaSuccess) { return; }
    char msg[512];
    std::snprintf(msg, sizeof(msg), "%s:%d: %s failed: %s", file, line, call,
                  cudaGetErrorString(err));
    throw std::runtime_error(msg);
}

#define NNOPS_CUDA_CHECK(call) \
    ::nnops::test::cuda_check_impl((call), #call, __FILE__, __LINE__)

/// A device allocation with host-staging copies. Movable, not copyable.
class CudaBuffer {
public:
    CudaBuffer() = default;

    explicit CudaBuffer(size_t bytes) : bytes_(bytes)
    {
        NNOPS_CUDA_CHECK(cudaMalloc(&ptr_, bytes_ > 0 ? bytes_ : 1));
    }

    ~CudaBuffer() { release(); }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& o) noexcept : ptr_(o.ptr_), bytes_(o.bytes_)
    {
        o.ptr_ = nullptr;
        o.bytes_ = 0;
    }
    CudaBuffer& operator=(CudaBuffer&& o) noexcept
    {
        if (this != &o) {
            release();
            ptr_ = o.ptr_;
            bytes_ = o.bytes_;
            o.ptr_ = nullptr;
            o.bytes_ = 0;
        }
        return *this;
    }

    void* data() const { return ptr_; }
    size_t bytes() const { return bytes_; }

    void upload(const void* host, size_t n)
    {
        NNOPS_CUDA_CHECK(cudaMemcpy(ptr_, host, n, cudaMemcpyHostToDevice));
    }

    /// Blocking download; also catches a fault left behind by the launch.
    void download(void* host, size_t n) const
    {
        sync();
        NNOPS_CUDA_CHECK(cudaMemcpy(host, ptr_, n, cudaMemcpyDeviceToHost));
    }

    void zero()
    {
        NNOPS_CUDA_CHECK(cudaMemset(ptr_, 0, bytes_));
    }

    static void sync()
    {
        NNOPS_CUDA_CHECK(cudaDeviceSynchronize());
        NNOPS_CUDA_CHECK(cudaGetLastError());
    }

private:
    void release()
    {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
    }

    void* ptr_ = nullptr;
    size_t bytes_ = 0;
};

/// ComputeContext on the default stream; the tests synchronise themselves.
inline ComputeContext cuda_context()
{
    ComputeContext ctx;
    ctx.cuda_stream = nullptr;
    return ctx;
}

/// ComputeContext for the CPU oracle: the library's default worker count.
inline ComputeContext cpu_context()
{
    ComputeContext ctx;
    return ctx;
}

/// Run an op on the device and wait for it. Any launch error throws here.
template <class Op>
void cuda_run(Op& op, TensorView& out, std::span<const TensorView> ins)
{
    const Status st = op.compute({&out, 1}, ins, cuda_context(), nullptr);
    if (st != Status::ok) {
        throw std::runtime_error(std::string("cuda compute: ") + to_string(st));
    }
    CudaBuffer::sync();
}

/// Run the same op on the CPU as the ground truth.
template <class Op>
void cpu_run(Op& op, TensorView& out, std::span<const TensorView> ins)
{
    const Status st = op.compute({&out, 1}, ins, cpu_context(), nullptr);
    if (st != Status::ok) {
        const int code = static_cast<int>(st);
        throw std::runtime_error("cpu compute returned status " +
                                 std::to_string(code));
    }
}

/// Widen a raw buffer of `count` elements to float for comparison.
inline std::vector<float> to_floats(const void* data, DataType dt, size_t count)
{
    std::vector<float> out(count);
    if (dt == DataType::f16) {
        const uint16_t* src = static_cast<const uint16_t*>(data);
        for (size_t i = 0; i < count; ++i) {
            float f;
            // half -> float through the CUDA conversion, so the test's notion of
            // the value matches what the device wrote.
            const uint16_t h = src[i];
            uint32_t sign = uint32_t(h & 0x8000) << 16;
            uint32_t exp = (h >> 10) & 0x1F;
            uint32_t man = h & 0x3FF;
            uint32_t bits;
            if (exp == 0) {
                bits = sign;
            } else if (exp == 31) {
                bits = sign | 0x7F800000 | (man << 13);
            } else {
                bits = sign | ((exp + 112) << 23) | (man << 13);
            }
            std::memcpy(&f, &bits, 4);
            out[i] = f;
        }
    } else if (dt == DataType::bf16) {
        // bf16 is the upper half of the f32 pattern, so widening is a shift.
        const uint16_t* src = static_cast<const uint16_t*>(data);
        for (size_t i = 0; i < count; ++i) {
            const uint32_t bits = uint32_t(src[i]) << 16;
            float f;
            std::memcpy(&f, &bits, 4);
            out[i] = f;
        }
    } else {
        const float* src = static_cast<const float*>(data);
        for (size_t i = 0; i < count; ++i) { out[i] = src[i]; }
    }
    return out;
}

/// Elementwise comparison with a per-element budget of `tol_abs + tol_rel·|ref|`.
inline void expect_close(const std::vector<float>& got,
                         const std::vector<float>& want, float tol_abs,
                         float tol_rel, const char* what)
{
    if (got.size() != want.size()) {
        throw std::runtime_error(std::string(what) + ": size mismatch");
    }
    double worst_abs = 0.0;
    size_t worst_i = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float a = got[i];
        const float b = want[i];
        if (std::isnan(a) != std::isnan(b)) {
            char msg[256];
            std::snprintf(msg, sizeof(msg), "%s: nan mismatch at %zu (%g vs %g)",
                          what, i, double(a), double(b));
            throw std::runtime_error(msg);
        }
        const double diff = std::abs(double(a) - double(b));
        const double limit = double(tol_abs) + double(tol_rel) * std::abs(double(b));
        if (diff > limit) {
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                          "%s: at %zu got %g want %g (|d|=%g > %g)", what, i,
                          double(a), double(b), diff, limit);
            throw std::runtime_error(msg);
        }
        const double scale = std::abs(double(b)) > 1.0 ? std::abs(double(b)) : 1.0;
        if (diff / scale > worst_abs) {
            worst_abs = diff / scale;
            worst_i = i;
        }
    }
    (void)worst_i;
}

// ============================================================
// Host and device tensors
// ============================================================

/// Storage plus the view over it; the view points into `bytes`, so a HostTensor
/// must not move once its view has been taken.
struct HostTensor {
    std::vector<std::byte> bytes;
    TensorView view;
};

/// Random host tensor over `shape`. `padded` gives it the 32 B row pitch the
/// operators see in a graph; channel-last views must stay dense because their
/// strides ignore the pitch.
inline HostTensor make_host_random(std::span<const int64_t> shape, DataType dt,
                                   bool padded, uint64_t seed,
                                   float lo = -0.5f, float hi = 0.5f,
                                   TensorLayout layout = TensorLayout::NCHW)
{
    HostTensor t;
    const TensorView probe(shape, dt, nullptr, layout);
    // The padded view steps rows by row_pitch(32), so the buffer has to be
    // allocated for that pitch — allocating the dense size and then handing the
    // view a wider pitch overruns the allocation on both sides of the copy.
    const int64_t pitch = probe.desc().row_pitch(32);
    t.bytes.assign(padded ? probe.desc().storage_bytes(32) : probe.storage_bytes(),
                   std::byte{0});

    XorShift128 rng(seed);
    if (dt == DataType::f16) {
        auto* p = reinterpret_cast<nnops::backend::cpu::half*>(t.bytes.data());
        for (size_t i = 0; i < t.bytes.size() / sizeof(*p); ++i) {
            simd::s_store(&p[i], rng.next_float(lo, hi));
        }
    } else if (dt == DataType::bf16) {
        auto* p = reinterpret_cast<nnops::backend::cpu::bfloat16*>(t.bytes.data());
        for (size_t i = 0; i < t.bytes.size() / sizeof(*p); ++i) {
            simd::s_store(&p[i], rng.next_float(lo, hi));
        }
    } else {
        auto* p = reinterpret_cast<float*>(t.bytes.data());
        for (size_t i = 0; i < t.bytes.size() / sizeof(*p); ++i) {
            p[i] = rng.next_float(lo, hi);
        }
    }

    t.view = padded ? TensorView(shape, dt, t.bytes.data(), pitch, layout)
                    : TensorView(shape, dt, t.bytes.data(), layout);
    return t;
}

/// A zeroed host tensor of the given shape (the output before a run).
inline HostTensor make_host_zeroed(std::span<const int64_t> shape, DataType dt,
                                   bool padded,
                                   TensorLayout layout = TensorLayout::NCHW)
{
    HostTensor t;
    const TensorView probe(shape, dt, nullptr, layout);
    const int64_t pitch = probe.desc().row_pitch(32);
    t.bytes.assign(padded ? probe.desc().storage_bytes(32) : probe.storage_bytes(),
                   std::byte{0});
    t.view = padded ? TensorView(shape, dt, t.bytes.data(), pitch, layout)
                    : TensorView(shape, dt, t.bytes.data(), layout);
    return t;
}

/// A device tensor: the allocation and a view over it with the source's pitch.
struct DeviceTensor {
    CudaBuffer buf;
    TensorView view;
};

inline DeviceTensor to_device(const HostTensor& h)
{
    DeviceTensor d{CudaBuffer(h.bytes.size())};
    d.buf.upload(h.bytes.data(), h.bytes.size());
    d.view = TensorView(h.view.shape_span(), h.view.data_type(), d.buf.data(),
                        h.view.pitch(), h.view.layout());
    return d;
}

/// A device output, laid out like `model` (same shape, dtype, pitch, layout).
inline DeviceTensor alloc_like(const TensorView& model, const HostTensor& init)
{
    // The allocation follows the view's pitch, which is what the kernel writes
    // through; `init` is only the seed.
    const TensorView probe(model.shape_span(), model.data_type(), nullptr,
                           model.pitch(), model.layout());
    const size_t bytes = std::max(init.bytes.size(), probe.storage_bytes());
    DeviceTensor d{CudaBuffer(bytes)};
    d.buf.upload(init.bytes.data(), init.bytes.size());
    d.view = TensorView(model.shape_span(), model.data_type(), d.buf.data(),
                        model.pitch(), model.layout());
    return d;
}

/// Copy a device tensor back into its host twin.
inline void from_device(const DeviceTensor& d, HostTensor& h)
{
    d.buf.download(h.bytes.data(), h.bytes.size());
}

/// Compare two host tensors element-wise after widening to float, skipping the
/// row padding. Both views must carry the same shape.
inline void expect_same(const HostTensor& got, const HostTensor& want,
                        float tol_abs, float tol_rel, const char* what)
{
    const TensorView& g = got.view;
    const TensorView& w = want.view;
    const DataType dt = g.data_type();
    const int64_t esz = data_type_size(dt);
    const int64_t cols = g.numel() / (g.rank() >= 2 ? g.total_rows() : 1);
    const int64_t rows = g.rank() >= 2 ? g.total_rows() : 1;

    std::vector<float> flat_g(static_cast<size_t>(g.numel()));
    std::vector<float> flat_w(static_cast<size_t>(w.numel()));
    size_t k = 0;
    for (int64_t r = 0; r < rows; ++r) {
        const std::byte* gp = got.bytes.data() + static_cast<size_t>(r * g.pitch());
        const std::byte* wp = want.bytes.data() + static_cast<size_t>(r * w.pitch());
        auto gf = to_floats(gp, dt, static_cast<size_t>(cols));
        auto wf = to_floats(wp, dt, static_cast<size_t>(cols));
        for (int64_t c = 0; c < cols; ++c) {
            flat_g[k] = gf[static_cast<size_t>(c)];
            flat_w[k] = wf[static_cast<size_t>(c)];
            ++k;
        }
    }
    (void)esz;
    expect_close(flat_g, flat_w, tol_abs, tol_rel, what);
}

/// Byte-wise comparison of the live elements, skipping the row padding (which
/// neither backend writes). The movement and index operators never touch a
/// value, so they are held to identity rather than to a tolerance — which also
/// covers f16 and the integer dtypes without widening them to float.
inline void expect_same_bytes(const HostTensor& got, const HostTensor& want,
                              const char* what)
{
    const TensorView& g = got.view;
    const TensorView& w = want.view;
    const int64_t esz = data_type_size(g.data_type());
    const int64_t rows = g.rank() >= 2 ? g.total_rows() : 1;
    const int64_t cols = rows > 0 ? g.numel() / rows : 0;
    for (int64_t r = 0; r < rows; ++r) {
        const std::byte* gp = got.bytes.data() + static_cast<size_t>(r * g.pitch());
        const std::byte* wp = want.bytes.data() + static_cast<size_t>(r * w.pitch());
        if (std::memcmp(gp, wp, static_cast<size_t>(cols * esz)) == 0) { continue; }
        int64_t c = 0;
        for (; c < cols; ++c) {
            if (std::memcmp(gp + c * esz, wp + c * esz, static_cast<size_t>(esz)) != 0) {
                break;
            }
        }
        char msg[320];
        if (esz == 4) {
            float gv, wv;
            std::memcpy(&gv, gp + c * esz, 4);
            std::memcpy(&wv, wp + c * esz, 4);
            std::snprintf(msg, sizeof(msg),
                          "%s: row %lld of %lld, element %lld differs (%g vs %g)",
                          what, static_cast<long long>(r), static_cast<long long>(rows),
                          static_cast<long long>(c), double(gv), double(wv));
        } else {
            std::snprintf(msg, sizeof(msg),
                          "%s: row %lld of %lld, element %lld differs (%d vs %d)",
                          what, static_cast<long long>(r), static_cast<long long>(rows),
                          static_cast<long long>(c),
                          int(reinterpret_cast<const unsigned char*>(gp)[c * esz]),
                          int(reinterpret_cast<const unsigned char*>(wp)[c * esz]));
        }
        throw std::runtime_error(msg);
    }
}

/// The body of `run_op_both`. `exact` compares every output byte, which a kernel
/// that reproduces the CPU's arithmetic bit for bit earns; a reduction whose
/// summation order differs — softmax's block tree against the CPU's row loop —
/// takes the tolerance form instead.
template <class Factory>
void run_op_both_impl(Factory make, std::span<const HostTensor> ins,
                      const char* what, bool exact, float tol_abs, float tol_rel)
{
    std::vector<TensorDesc> descs;
    std::vector<TensorView> views;
    descs.reserve(ins.size());
    views.reserve(ins.size());
    for (const HostTensor& t : ins) {
        descs.push_back(t.view.desc());
        views.push_back(t.view);
    }

    auto op_cpu = make(Backend::CPU);
    const std::vector<TensorDesc> odescs = op_cpu->getOutputTensorDesc(descs);
    if (odescs.empty()) {
        throw std::runtime_error(std::string(what) + ": no output desc");
    }

    std::vector<HostTensor> ref;
    std::vector<HostTensor> got;
    ref.reserve(odescs.size());
    got.reserve(odescs.size());
    for (const TensorDesc& d : odescs) {
        ref.push_back(make_host_zeroed(d.dims, d.dtype, false, d.layout));
        got.push_back(make_host_zeroed(d.dims, d.dtype, false, d.layout));
    }

    std::vector<TensorView> rv, gv;
    for (HostTensor& t : ref) { rv.push_back(t.view); }
    for (HostTensor& t : got) { gv.push_back(t.view); }

    if (op_cpu->compute(rv, views, cpu_context(), nullptr) != Status::ok) {
        throw std::runtime_error(std::string(what) + ": cpu compute failed");
    }

    std::vector<DeviceTensor> dins;
    dins.reserve(ins.size());
    for (const HostTensor& t : ins) { dins.push_back(to_device(t)); }
    std::vector<TensorView> dviews;
    dviews.reserve(dins.size());
    for (DeviceTensor& d : dins) { dviews.push_back(d.view); }

    std::vector<DeviceTensor> douts;
    douts.reserve(odescs.size());
    for (size_t i = 0; i < odescs.size(); ++i) {
        douts.push_back(alloc_like(gv[i], got[i]));
    }
    std::vector<TensorView> dv;
    dv.reserve(douts.size());
    for (DeviceTensor& d : douts) { dv.push_back(d.view); }

    auto op_dev = make(Backend::CUDA);
    const Status st = op_dev->compute(dv, dviews, cuda_context(), nullptr);
    if (st != Status::ok) {
        throw std::runtime_error(std::string(what) + ": cuda compute returned " +
                                 to_string(st));
    }
    CudaBuffer::sync();

    for (size_t i = 0; i < douts.size(); ++i) {
        from_device(douts[i], got[i]);
        if (exact) {
            expect_same_bytes(got[i], ref[i], what);
        } else {
            expect_same(got[i], ref[i], tol_abs, tol_rel, what);
        }
    }
}

/// Run `make(backend)` on both backends over the same bytes and compare every
/// output byte for byte. The factory keeps a case down to one line — the
/// operator, its attributes and its input list are the whole test — and the CPU
/// route is the oracle.
template <class Factory>
void run_op_both(Factory make, std::span<const HostTensor> ins, const char* what)
{
    run_op_both_impl(make, ins, what, true, 0.0f, 0.0f);
}

/// The same run, compared to a per-element budget of `tol_abs + tol_rel·|ref|`.
template <class Factory>
void run_op_close(Factory make, std::span<const HostTensor> ins, float tol_abs,
                  float tol_rel, const char* what)
{
    run_op_both_impl(make, ins, what, false, tol_abs, tol_rel);
}

// ============================================================
// bf16
// ============================================================

/// Every element of an f32 tensor rounded to bf16 and kept in f32 storage: the
/// CPU oracle's input, so it sees exactly the numbers the device holds.
inline HostTensor bf16_rounded_host(const HostTensor& src)
{
    HostTensor t = src;
    auto* p = reinterpret_cast<float*>(t.bytes.data());
    for (size_t i = 0; i < t.bytes.size() / sizeof(float); ++i) {
        const nnops::backend::cpu::bfloat16 b(p[i]);
        p[i] = static_cast<float>(b);
    }
    // A HostTensor's view points into its own `bytes`, and the copy above left it
    // pointing at `src`'s storage — which `t` has just rounded a *copy* of. Re-aim
    // it, or the oracle reads the unrounded values.
    t.view = TensorView(t.view.shape_span(), t.view.data_type(), t.bytes.data(),
                        t.view.pitch(), t.view.layout());
    return t;
}

/// The same values in bf16 storage. A pitched source keeps the 32 B row rule
/// for the new width; a channel-last one stays dense.
inline HostTensor to_bf16_host(const HostTensor& src, bool padded)
{
    const TensorView& s = src.view;
    const int64_t rows = s.rank() >= 2 ? s.total_rows() : 1;
    const int64_t cols = rows > 0 ? s.numel() / rows : 0;

    HostTensor t;
    const TensorView probe(s.shape_span(), DataType::bf16, nullptr, s.layout());
    t.bytes.assign(padded ? probe.desc().storage_bytes(32) : probe.storage_bytes(),
                   std::byte{0});
    t.view = padded ? TensorView(s.shape_span(), DataType::bf16, t.bytes.data(),
                                 probe.desc().row_pitch(32), s.layout())
                    : TensorView(s.shape_span(), DataType::bf16, t.bytes.data(),
                                 s.layout());

    for (int64_t r = 0; r < rows; ++r) {
        const auto* sp =
            reinterpret_cast<const float*>(src.bytes.data() + r * s.pitch());
        auto* dp = reinterpret_cast<nnops::backend::cpu::bfloat16*>(
            t.bytes.data() + r * t.view.pitch());
        for (int64_t c = 0; c < cols; ++c) { dp[c] = nnops::backend::cpu::bfloat16(sp[c]); }
    }
    return t;
}

/// True when the caller gave the view a row pitch beyond its dense width.
inline bool is_pitched(const HostTensor& t)
{
    const TensorView dense(t.view.shape_span(), t.view.data_type(), nullptr,
                           t.view.layout());
    return t.view.rank() >= 2 && t.view.pitch() != dense.pitch();
}

/// True when `d` describes the same shape and layout as `in`.
inline bool same_shape(const TensorDesc& d, const TensorDesc& in)
{
    if (d.rank != in.rank || d.layout != in.layout) { return false; }
    for (int64_t i = 0; i < d.rank; ++i) {
        if (d.dims[static_cast<size_t>(i)] != in.dims[static_cast<size_t>(i)]) {
            return false;
        }
    }
    return true;
}

/// Run `make` on the CPU over bf16-rounded f32 inputs and on the device over the
/// same values as bf16, then compare every output.
///
/// Off AArch64 the CPU has no bf16 route, so it cannot be the oracle over a bf16
/// tensor: rounding the inputs on the host first makes the values identical on
/// both sides — the CPU runs f32, the device runs bf16 — and comparing the
/// outputs as bf16 (the device rounds its accumulator at the store, the oracle is
/// rounded the same way) leaves accumulation order as the only difference.
/// Non-f32 inputs are indices and pass through untouched, so an index-driven
/// operator sees the same indices on both sides. Every input keeps the pitch it
/// was built with; a same-shape output takes that pitch and any other output is
/// dense.
template <class Factory>
void run_op_bf16(Factory make, std::span<const HostTensor> ins, float tol_abs,
                 float tol_rel, const char* what)
{
    std::vector<HostTensor> cpu_src;
    std::vector<HostTensor> dev_src;
    cpu_src.reserve(ins.size());
    dev_src.reserve(ins.size());

    std::vector<TensorDesc> cpu_descs;
    std::vector<TensorDesc> dev_descs;
    cpu_descs.reserve(ins.size());
    dev_descs.reserve(ins.size());
    for (const HostTensor& t : ins) {
        if (t.view.data_type() == DataType::f32) {
            cpu_src.push_back(bf16_rounded_host(t));
            dev_src.push_back(to_bf16_host(t, is_pitched(t)));
        } else {
            cpu_src.push_back(t);
            dev_src.push_back(t);
        }
        cpu_descs.push_back(cpu_src.back().view.desc());
        dev_descs.push_back(dev_src.back().view.desc());
    }

    std::vector<TensorView> cpu_views;
    cpu_views.reserve(cpu_src.size());
    for (HostTensor& t : cpu_src) { cpu_views.push_back(t.view); }

    auto op_cpu = make(Backend::CPU);
    const std::vector<TensorDesc> odescs = op_cpu->getOutputTensorDesc(cpu_descs);
    if (odescs.empty()) {
        throw std::runtime_error(std::string(what) + ": no output desc");
    }

    // A same-shape output inherits its input's row pitch. The shape-preserving
    // kernels (softmax, the norm family, cumsum) walk the output with the input's
    // stride, and their descriptors carry no stride of their own, so the pitch
    // the caller allocated for the input is the one they assume for the output.
    // A shape-changing output has no such twin and is dense.
    std::vector<HostTensor> ref;
    ref.reserve(odescs.size());
    for (const TensorDesc& d : odescs) {
        const bool pitched = is_pitched(cpu_src[0]) && same_shape(d, cpu_descs[0]);
        ref.push_back(make_host_zeroed(d.dims, d.dtype, pitched, d.layout));
    }
    std::vector<TensorView> rv;
    for (HostTensor& t : ref) { rv.push_back(t.view); }
    if (op_cpu->compute(rv, cpu_views, cpu_context(), nullptr) != Status::ok) {
        throw std::runtime_error(std::string(what) + ": cpu compute failed");
    }

    std::vector<DeviceTensor> dins;
    dins.reserve(dev_src.size());
    for (const HostTensor& t : dev_src) { dins.push_back(to_device(t)); }
    std::vector<TensorView> dviews;
    dviews.reserve(dins.size());
    for (DeviceTensor& d : dins) { dviews.push_back(d.view); }

    auto op_dev = make(Backend::CUDA);
    const std::vector<TensorDesc> dodescs = op_dev->getOutputTensorDesc(dev_descs);
    if (dodescs.size() != odescs.size()) {
        throw std::runtime_error(std::string(what) + ": output count mismatch");
    }

    std::vector<HostTensor> got;
    std::vector<DeviceTensor> douts;
    got.reserve(dodescs.size());
    douts.reserve(dodescs.size());
    for (const TensorDesc& d : dodescs) {
        const bool pitched = is_pitched(dev_src[0]) && same_shape(d, dev_descs[0]);
        got.push_back(make_host_zeroed(d.dims, d.dtype, pitched, d.layout));
    }
    for (size_t i = 0; i < dodescs.size(); ++i) {
        douts.push_back(alloc_like(got[i].view, got[i]));
    }
    std::vector<TensorView> dv;
    dv.reserve(douts.size());
    for (DeviceTensor& d : douts) { dv.push_back(d.view); }

    const Status st = op_dev->compute(dv, dviews, cuda_context(), nullptr);
    if (st != Status::ok) {
        throw std::runtime_error(std::string(what) + ": cuda compute returned " +
                                 to_string(st));
    }
    CudaBuffer::sync();

    for (size_t i = 0; i < douts.size(); ++i) {
        from_device(douts[i], got[i]);
        const DataType dt = got[i].view.data_type();
        if (dt == DataType::f32) {
            expect_same(got[i], ref[i], tol_abs, tol_rel, what);
        } else if (dt == DataType::bf16) {
            expect_same(got[i], to_bf16_host(ref[i], is_pitched(got[i])), tol_abs,
                        tol_rel, what);
        } else {
            expect_same_bytes(got[i], ref[i], what);
        }
    }
}

}  // namespace nnops::test

#endif  // NNOPS_HAS_CUDA
