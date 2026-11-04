/// @file eltwise.cpp
/// @brief SIMD-optimized CPU implementation of element-wise binary operations.
///
/// Supports both f32 and f16 via a single templated implementation.
/// Processing is tiled in groups of TILE_M rows and dispatched via
/// ComputeContext::cpu.run.
///
/// Quantized input/output (s8/u8 → s8/u8) is supported through a fused path: both
/// inputs are dequantized (int → f32), the binary op is applied in f32, and the
/// result is re-quantized (f32 → int). Both inputs and the output must be
/// quantized (no float↔int mixing). Only PerTensor / PerToken granularity is
/// supported; the quant/dequant work is delegated to the raw-intrinsic arch
/// kernels in `x86_64/quant.hpp` / `aarch64/quant.hpp` (NOT `simd_quant.hpp`).

#include "nnops/ops/eltwise.hpp"
#include "nnops/detail/assert.hpp"
#include "simd_kernel/simd_eltwise.hpp"
#include "common/elementwise.hpp"
#include "common/dtype_dispatch.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace nnops::backend::cpu {

namespace {

/// Apply a floating-point binary op to an M×N tile, dispatching on the op type.
/// Shared by the f32/f16 paths (strides may differ) and the quantized path
/// (a/b/out are f32 scratch, out may alias a, add_to == false).
template <typename T>
void apply_eltwise_flt(EltwiseType type, const T* a, const T* b, T* out,
                       int64_t M, int64_t N, int64_t a_stride, int64_t b_stride,
                       int64_t out_stride, bool add_to) {
    switch (type) {
    case EltwiseType::Add: kernel::add<T>(a, b, out, M, N, a_stride, b_stride, out_stride, add_to); break;
    case EltwiseType::Sub: kernel::sub<T>(a, b, out, M, N, a_stride, b_stride, out_stride, add_to); break;
    case EltwiseType::Mul: kernel::mul<T>(a, b, out, M, N, a_stride, b_stride, out_stride, add_to); break;
    case EltwiseType::Div: kernel::div<T>(a, b, out, M, N, a_stride, b_stride, out_stride, add_to); break;
    case EltwiseType::Min: kernel::min<T>(a, b, out, M, N, a_stride, b_stride, out_stride, add_to); break;
    case EltwiseType::Max: kernel::max<T>(a, b, out, M, N, a_stride, b_stride, out_stride, add_to); break;
    case EltwiseType::Pow: kernel::pow<T>(a, b, out, M, N, a_stride, b_stride, out_stride, add_to); break;
    }
}

/// True when the two inputs describe exactly the same tensor, which is the
/// case the tiled SIMD route handles.
inline bool same_shape(const TensorView& a, const TensorView& b)
{
    if (a.rank() != b.rank()) { return false; }
    for (int64_t d = 0; d < a.rank(); ++d) {
        if (a.shape(d) != b.shape(d)) { return false; }
    }
    return true;
}

/// Generic NumPy-broadcast loop: one odometer over the output's logical index
/// space, three running element offsets (a, b, out). Amortized one increment
/// per element, and no division inside the loop; the strides carry the layout,
/// so planar and channel-last need no separate code.
template <typename T, typename F>
void broadcast_loop(const TensorView& a, const TensorView& b, TensorView& out,
                    const ComputeContext& ctx, F&& op, bool add_to)
{
    const int64_t rank  = out.rank();
    const int64_t numel = out.numel();
    if (numel == 0) { return; }

    int64_t dims[8];
    int64_t as[8], bs[8], os[8];
    for (int64_t d = 0; d < rank; ++d) {
        dims[d] = out.shape(d);
        os[d]   = out.stride_elems(d);

        const int64_t ia = d - (rank - a.rank());
        as[d] = (ia < 0 || a.shape(ia) == 1) ? 0 : a.stride_elems(ia);
        const int64_t ib = d - (rank - b.rank());
        bs[d] = (ib < 0 || b.shape(ib) == 1) ? 0 : b.stride_elems(ib);
    }

    using simd::s_load;
    using simd::s_store_add;
    const T* ap = a.ptr<T>();
    const T* bp = b.ptr<T>();
    T*       op_ptr = out.ptr<T>();

    constexpr int64_t kChunk = 1 << 16;
    const int64_t chunks = (numel + kChunk - 1) / kChunk;

    ctx.cpu.run(0, chunks, [&](int64_t ci) {
        const int64_t begin = ci * kChunk;
        const int64_t end   = std::min(begin + kChunk, numel);

        int64_t idx[8] = {};
        int64_t rem = begin;
        for (int64_t d = rank - 1; d >= 0; --d) {
            idx[d] = (dims[d] > 0) ? rem % dims[d] : 0;
            rem    = (dims[d] > 0) ? rem / dims[d] : 0;
        }
        int64_t ao = 0, bo = 0, oo = 0;
        for (int64_t d = 0; d < rank; ++d) {
            ao += idx[d] * as[d];
            bo += idx[d] * bs[d];
            oo += idx[d] * os[d];
        }

        for (int64_t n = begin; n < end; ++n) {
            const float v = op(s_load(&ap[ao]), s_load(&bp[bo]));
            s_store_add(&op_ptr[oo], v, add_to);

            for (int64_t d = rank - 1; d >= 0; --d) {
                ++idx[d];
                ao += as[d];
                bo += bs[d];
                oo += os[d];
                if (idx[d] < dims[d]) { break; }
                idx[d] = 0;
                ao -= as[d] * dims[d];
                bo -= bs[d] * dims[d];
                oo -= os[d] * dims[d];
            }
        }
    });
}

}  // anonymous namespace

template <typename T>
void eltwise_impl(const EltwiseAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx)
{
    if (same_shape(inputs[0], inputs[1])) {
        tiled_float_transform2<T>(inputs[0], inputs[1], output, ctx,
            [&](const T* a, const T* b, T* out, int64_t m, int64_t n,
                int64_t a_stride, int64_t b_stride, int64_t out_stride) {
                apply_eltwise_flt<T>(attrs.type, a, b, out, m, n, a_stride, b_stride,
                                     out_stride, attrs.add_to);
            });
        return;
    }

    const bool add_to = attrs.add_to;
    switch (attrs.type) {
    case EltwiseType::Add: broadcast_loop<T>(inputs[0], inputs[1], output, ctx,
                                [](float x, float y) { return x + y; }, add_to); break;
    case EltwiseType::Sub: broadcast_loop<T>(inputs[0], inputs[1], output, ctx,
                                [](float x, float y) { return x - y; }, add_to); break;
    case EltwiseType::Mul: broadcast_loop<T>(inputs[0], inputs[1], output, ctx,
                                [](float x, float y) { return x * y; }, add_to); break;
    case EltwiseType::Div: broadcast_loop<T>(inputs[0], inputs[1], output, ctx,
                                [](float x, float y) { return x / y; }, add_to); break;
    case EltwiseType::Min: broadcast_loop<T>(inputs[0], inputs[1], output, ctx,
                                [](float x, float y) { return std::min(x, y); }, add_to); break;
    case EltwiseType::Max: broadcast_loop<T>(inputs[0], inputs[1], output, ctx,
                                [](float x, float y) { return std::max(x, y); }, add_to); break;
    case EltwiseType::Pow: broadcast_loop<T>(inputs[0], inputs[1], output, ctx,
                                [](float x, float y) { return std::pow(x, y); }, add_to); break;
    }
}

Status eltwise_quant_impl(const EltwiseAttributes& attrs,
                           TensorView& output,
                           std::span<const TensorView> inputs,
                           const ComputeContext& ctx)
{
    // add_to is only meaningful on the float (non-quantized) path.
    if (attrs.add_to) { return Status::unsupported; }

    quantized_binary_transform(inputs[0], inputs[1], output, ctx,
        [&](float* a, float* b, int h, int w) {
            apply_eltwise_flt<float>(attrs.type, a, b, a, h, w, w, w, w, false);
        });
    return Status::ok;
}

Status eltwise_cpu(const EltwiseAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx,
                   void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    const bool in_is_int  = is_quantized_dtype(dtype);
    const bool out_is_int = is_quantized_dtype(output.data_type());

    if (!in_is_int && !out_is_int) {
        dispatch_f32_f16(dtype, "eltwise_cpu", [&](auto tag) {
            using T = typename decltype(tag)::type;
            eltwise_impl<T>(attrs, output, inputs, ctx);
        });
        return Status::ok;
    }

    // Quantized path: both inputs and the output must be s8/u8 (no float↔int mixing).
    NNOPS_ASSERT(in_is_int && out_is_int);

    // The fused tile path dequantizes one element per side; a broadcast would
    // need the index arithmetic re-derived per scale row, so it stays float.
    if (!same_shape(inputs[0], inputs[1])) { return Status::unsupported; }

    return eltwise_quant_impl(attrs, output, inputs, ctx);
}

}  // namespace nnops::backend::cpu
