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
/// built with make_packed (a 32 B pitch) for planar tensors so the padded-stride
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

}  // namespace nnops::test

#endif  // NNOPS_HAS_CUDA
