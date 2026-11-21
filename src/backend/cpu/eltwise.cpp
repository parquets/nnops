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
#include "common/broadcast_walk.hpp"
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
void apply_eltwise_flt(EltwiseType type, const T* a, const T* b, T* out, int64_t M, int64_t N,
                       int64_t a_stride, int64_t b_stride, int64_t out_stride, bool add_to)
{
    switch (type) {
    case EltwiseType::Add:
        kernel::add<T>(a, b, out, M, N, a_stride, b_stride, out_stride, add_to);
        break;
    case EltwiseType::Sub:
        kernel::sub<T>(a, b, out, M, N, a_stride, b_stride, out_stride, add_to);
        break;
    case EltwiseType::Mul:
        kernel::mul<T>(a, b, out, M, N, a_stride, b_stride, out_stride, add_to);
        break;
    case EltwiseType::Div:
        kernel::div<T>(a, b, out, M, N, a_stride, b_stride, out_stride, add_to);
        break;
    case EltwiseType::Min:
        kernel::min<T>(a, b, out, M, N, a_stride, b_stride, out_stride, add_to);
        break;
    case EltwiseType::Max:
        kernel::max<T>(a, b, out, M, N, a_stride, b_stride, out_stride, add_to);
        break;
    case EltwiseType::Pow:
        kernel::pow<T>(a, b, out, M, N, a_stride, b_stride, out_stride, add_to);
        break;
    }
}

/// True when the two inputs describe exactly the same tensor, which is the
/// case the tiled SIMD route handles.
inline bool same_shape(const TensorView& a, const TensorView& b)
{
    if (a.rank() != b.rank()) {
        return false;
    }
    for (int64_t d = 0; d < a.rank(); ++d) {
        if (a.shape(d) != b.shape(d)) {
            return false;
        }
    }
    return true;
}

/// NumPy-broadcast binary op. The walk hands out one contiguous output run at a
/// time, so the run's 8-lane body is a plain vector load of each side — an
/// operand that broadcasts along the run (stride 0) becomes a scalar broadcast
/// instead of a per-element load. The previous form stepped one element at a
/// time through the odometer, which left every bias add scalar.
template <typename T, typename SimdK, typename ScalarK>
void broadcast_loop(const TensorView& a, const TensorView& b, TensorView& out,
                    const ComputeContext& ctx, SimdK&& simd_op, ScalarK&& scalar_op, bool add_to)
{
    using namespace simd;
    constexpr int64_t L = simd_lane_for<T>;

    const T* ap = a.ptr<T>();
    const T* bp = b.ptr<T>();
    T* op_ptr = out.ptr<T>();
    const TensorView ins[2] = {a, b};

    broadcast_walk_run<2>(out, ins, ctx, [&](int64_t oo, const int64_t* off, const int64_t* step,
                                             int64_t run) {
        const T* a_row = ap + off[0];
        const T* b_row = bp + off[1];
        T* o_row = op_ptr + oo;

        // A broadcasting operand is read as one broadcast value, never as a lane
        // that would run past the end of its storage.
        const bool a_bcast = (step[0] == 0);
        const bool b_bcast = (step[1] == 0);
        const auto a1 = a_bcast ? v_set1(a_row, s_load(a_row)) : v_zero(a_row);
        const auto b1 = b_bcast ? v_set1(b_row, s_load(b_row)) : v_zero(b_row);
        const float a_s = a_bcast ? s_load(a_row) : 0.0f;
        const float b_s = b_bcast ? s_load(b_row) : 0.0f;

        int64_t i = 0;
        for (; i + L <= run; i += L) {
            const auto va = a_bcast ? a1 : v_load(a_row + i * step[0]);
            const auto vb = b_bcast ? b1 : v_load(b_row + i * step[1]);
            v_store_add(o_row + i, simd_op(va, vb), add_to);
        }
        for (; i < run; ++i) {
            const float va = a_bcast ? a_s : s_load(a_row + i * step[0]);
            const float vb = b_bcast ? b_s : s_load(b_row + i * step[1]);
            s_store_add(&o_row[i], scalar_op(va, vb), add_to);
        }
    });
}

} // anonymous namespace

template <typename T>
void eltwise_impl(const EltwiseAttributes& attrs, TensorView& output,
                  std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    if (same_shape(inputs[0], inputs[1])) {
        tiled_float_transform2<T>(inputs[0], inputs[1], output, ctx,
                                  [&](const T* a, const T* b, T* out, int64_t m, int64_t n,
                                      int64_t a_stride, int64_t b_stride, int64_t out_stride) {
                                      apply_eltwise_flt<T>(attrs.type, a, b, out, m, n, a_stride,
                                                           b_stride, out_stride, attrs.add_to);
                                  });
        return;
    }

    const bool add_to = attrs.add_to;
    switch (attrs.type) {
    case EltwiseType::Add:
        broadcast_loop<T>(
            inputs[0], inputs[1], output, ctx,
            [](auto x, auto y) { return simd::v_add(x, y); },
            [](float x, float y) { return x + y; }, add_to);
        break;
    case EltwiseType::Sub:
        broadcast_loop<T>(
            inputs[0], inputs[1], output, ctx,
            [](auto x, auto y) { return simd::v_sub(x, y); },
            [](float x, float y) { return x - y; }, add_to);
        break;
    case EltwiseType::Mul:
        broadcast_loop<T>(
            inputs[0], inputs[1], output, ctx,
            [](auto x, auto y) { return simd::v_mul(x, y); },
            [](float x, float y) { return x * y; }, add_to);
        break;
    case EltwiseType::Div:
        broadcast_loop<T>(
            inputs[0], inputs[1], output, ctx,
            [](auto x, auto y) { return simd::v_div(x, y); },
            [](float x, float y) { return x / y; }, add_to);
        break;
    case EltwiseType::Min:
        broadcast_loop<T>(
            inputs[0], inputs[1], output, ctx,
            [](auto x, auto y) { return simd::v_min(x, y); },
            [](float x, float y) { return std::min(x, y); }, add_to);
        break;
    case EltwiseType::Max:
        broadcast_loop<T>(
            inputs[0], inputs[1], output, ctx,
            [](auto x, auto y) { return simd::v_max(x, y); },
            [](float x, float y) { return std::max(x, y); }, add_to);
        break;
    case EltwiseType::Pow:
        broadcast_loop<T>(
            inputs[0], inputs[1], output, ctx,
            [](auto x, auto y) { return simd::v_pow(x, y); },
            [](float x, float y) { return std::pow(x, y); }, add_to);
        break;
    }
}

Status eltwise_quant_impl(const EltwiseAttributes& attrs, TensorView& output,
                          std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    // add_to is only meaningful on the float (non-quantized) path.
    if (attrs.add_to) {
        return Status::unsupported;
    }

    quantized_binary_transform(
        inputs[0], inputs[1], output, ctx, [&](float* a, float* b, int h, int w) {
            apply_eltwise_flt<float>(attrs.type, a, b, a, h, w, w, w, w, false);
        });
    return Status::ok;
}

Status eltwise_cpu(const EltwiseAttributes& attrs, TensorView& output,
                   std::span<const TensorView> inputs, const ComputeContext& ctx,
                   void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    const bool in_is_int = is_quantized_dtype(dtype);
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
    if (!same_shape(inputs[0], inputs[1])) {
        return Status::unsupported;
    }

    return eltwise_quant_impl(attrs, output, inputs, ctx);
}

} // namespace nnops::backend::cpu
