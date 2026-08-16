/// @file eltwise.cpp
/// @brief SIMD-optimized CPU implementation of element-wise binary operations.
///
/// Supports both f32 and f16 via a single templated implementation.
/// Processing is tiled in groups of TILE_M rows and dispatched via
/// ComputeContext::cpu_parallel_for when available.
///
/// Design:
///   1. Rows are grouped into tiles of TILE_M (32) for SIMD-friendly blocking
///   2. Each tile calls into tiled_eltwise kernel library (lane=8, v_f32x8/v_f16x8)
///   3. Tile iteration is parallelized via ctx.cpu_parallel_for; falls back to
///      sequential when no parallel hook is provided
///   4. Op-type dispatch lives in the shared apply_eltwise_flt helper
///
/// Quantized input/output (s8/u8 → s8/u8) is supported through a fused path: both
/// inputs are dequantized (int → f32), the binary op is applied in f32, and the
/// result is re-quantized (f32 → int). Both inputs and the output must be
/// quantized (no float↔int mixing). Only PerTensor / PerToken granularity is
/// supported; the quant/dequant work is delegated to the raw-intrinsic arch
/// kernels in `x86_64/quant.hpp` / `aarch64/quant.hpp` (NOT `simd_quant.hpp`).

#include "nnops/ops/eltwise.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "simd_kernel/simd_eltwise.hpp"

#if defined(NNOPS_ARCH_X86_64)
#include "x86_64/quant.hpp"
#elif defined(NNOPS_ARCH_AARCH64)
#include "aarch64/quant.hpp"
#else
#error "eltwise: unsupported architecture for quantization kernels"
#endif

#include <algorithm>
#include <cstdint>

namespace nnops::backend::cpu {

#if defined(NNOPS_ARCH_X86_64)
namespace quant_kernel = nnops::backend::cpu::x86_64;
#elif defined(NNOPS_ARCH_AARCH64)
namespace quant_kernel = nnops::backend::cpu::aarch64;
#endif

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

}  // anonymous namespace

// ============================================================
// Templated implementation (f32 and f16)
// ============================================================

template <typename T>
void eltwise_impl(const EltwiseAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx)
{
    const auto& A = inputs[0];
    const auto& B = inputs[1];
    const int64_t total = A.numel();
    if (total == 0) { return; }

    const int64_t rank = A.rank();
    NNOPS_ASSERT(A.numel() == B.numel());
    NNOPS_ASSERT(output.numel() == total);

    // Row-by-row layout (pitch-aware).
    const int64_t last_dim = (rank >= 1) ? A.shape(rank - 1) * A.channel_pack_size() : 1;
    const int64_t num_rows = (rank >= 2) ? A.total_rows() : total / last_dim;
    const int64_t a_row_stride = A.row_stride_elems();
    const int64_t b_row_stride = B.row_stride_elems();
    const int64_t o_row_stride = output.row_stride_elems();

    const auto* a_ptr = A.ptr<T>();
    const auto* b_ptr = B.ptr<T>();
    auto*       o_ptr = output.ptr<T>();
    const bool  add_to = attrs.add_to;

    constexpr int64_t TILE_M = 32;
    const int64_t num_tiles = (num_rows + TILE_M - 1) / TILE_M;

    // Tiled dispatch: split rows into tiles of ≤TILE_M, parallelized when
    // available. Op-type dispatch lives in apply_eltwise_flt.
    auto body = [&](int64_t ti) {
        const int64_t r = ti * TILE_M;
        const int64_t m = std::min(TILE_M, num_rows - r);
        apply_eltwise_flt<T>(attrs.type,
                             a_ptr + r * a_row_stride,
                             b_ptr + r * b_row_stride,
                             o_ptr + r * o_row_stride,
                             m, last_dim, a_row_stride, b_row_stride, o_row_stride, add_to);
    };
    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, num_tiles, body);
    }
    else {
        for (int64_t t = 0; t < num_tiles; ++t) {
            body(t);
        }
    }
}

// ============================================================
// Fused quantized eltwise — dequantize → op → quantize
// ============================================================
//
// Quantized eltwise: both inputs and the output are s8/u8 (no float mixing).
// The binary math runs in f32; each int input is dequantized and the int output
// is quantized via the arch quant.hpp per-row kernels. Granularity is PerTensor
// or PerToken only (no PerChannel for activations).
//
// The tensor is processed in TILE_H×TILE_W tiles so every intermediate buffer
// (f32 scratch, per-row scale/zero) is a fixed stack array — no heap allocation
// anywhere in the kernel. Row tiles are dispatched in parallel via
// ComputeContext::cpu_parallel_for (matching the float path).

void eltwise_quant_impl(const EltwiseAttributes& attrs,
                        TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx)
{
    const auto& A = inputs[0];
    const auto& B = inputs[1];
    const int64_t numel = A.numel();
    if (numel == 0) { return; }
    NNOPS_ASSERT(A.numel() == B.numel());

    // add_to is only meaningful on the float (non-quantized) path.
    NNOPS_ASSERT(!attrs.add_to);

    const DataType a_dtype = A.data_type();   // s8 or u8
    const DataType b_dtype = B.data_type();   // s8 or u8
    const DataType o_dtype = output.data_type();  // s8 or u8

    const QuantParams& a_qp = A.quant_params();
    const QuantParams& b_qp = B.quant_params();
    const QuantParams& o_qp = output.quant_params();

    // Inputs and output share the same quantization granularity (all PerTensor
    // or all PerToken), so a single flag drives dequant and quant alike.
    NNOPS_ASSERT(a_qp.granularity == o_qp.granularity);
    NNOPS_ASSERT(b_qp.granularity == o_qp.granularity);
    const bool per_token = a_qp.granularity == QuantGranularity::PerToken
                        && a_qp.scale_data != nullptr;

    // Row shape: PerToken = one (scale, zero_point) per innermost row;
    // PerTensor = one parameter for the whole tensor (flat M=1, N=numel).
    int64_t M = 1;
    int64_t N = numel;
    if (per_token) {
        N = A.shape(A.rank() - 1);
        M = (N == 0) ? 0 : (numel / N);
    }

    // s8/u8 are both single-byte, so row/column offsets are in elements.
    const uint8_t* a_base = A.ptr<uint8_t>();
    const uint8_t* b_base = B.ptr<uint8_t>();
    uint8_t*       o_base = output.ptr<uint8_t>();

    // Fixed stack buffers per worker — the kernel allocates nothing on the heap.
    constexpr int TILE_H = 8;
    constexpr int TILE_W = 256;
    const int64_t num_tiles = (M + TILE_H - 1) / TILE_H;

    auto body = [&](int64_t ti) {
        const int64_t h0 = ti * TILE_H;
        const int cur_h = static_cast<int>(std::min<int64_t>(TILE_H, M - h0));

        float a_scratch[TILE_H * TILE_W];
        float b_scratch[TILE_H * TILE_W];
        float a_scale[TILE_H], a_zero[TILE_H];
        float b_scale[TILE_H], b_zero[TILE_H];
        float o_scale[TILE_H], o_zero[TILE_H];

        for (int i = 0; i < cur_h; ++i) {
            const int64_t r = h0 + i;
            a_scale[i] = per_token ? a_qp.scale_data[r] : a_qp.scale;
            a_zero[i]  = per_token
                ? (a_qp.zero_point_data != nullptr ? static_cast<float>(a_qp.zero_point_data[r]) : 0.0f)
                : static_cast<float>(a_qp.zero_point);
            b_scale[i] = per_token ? b_qp.scale_data[r] : b_qp.scale;
            b_zero[i]  = per_token
                ? (b_qp.zero_point_data != nullptr ? static_cast<float>(b_qp.zero_point_data[r]) : 0.0f)
                : static_cast<float>(b_qp.zero_point);
            o_scale[i] = per_token ? o_qp.scale_data[r] : o_qp.scale;
            o_zero[i]  = per_token
                ? (o_qp.zero_point_data != nullptr ? static_cast<float>(o_qp.zero_point_data[r]) : 0.0f)
                : static_cast<float>(o_qp.zero_point);
        }

        for (int64_t w0 = 0; w0 < N; w0 += TILE_W) {
            const int cur_w = static_cast<int>(std::min<int64_t>(TILE_W, N - w0));
            const uint8_t* a_tile = a_base + (h0 * N + w0);
            const uint8_t* b_tile = b_base + (h0 * N + w0);
            uint8_t*       o_tile = o_base + (h0 * N + w0);

            // Dequantize A and B → f32.
            if (a_dtype == DataType::s8) {
                quant_kernel::dequantization<int8_t>(cur_h, cur_w, a_scratch, cur_w,
                    reinterpret_cast<const int8_t*>(a_tile), static_cast<int>(N),
                    a_scale, a_zero);
            } else {
                quant_kernel::dequantization<uint8_t>(cur_h, cur_w, a_scratch, cur_w,
                    a_tile, static_cast<int>(N), a_scale, a_zero);
            }
            if (b_dtype == DataType::s8) {
                quant_kernel::dequantization<int8_t>(cur_h, cur_w, b_scratch, cur_w,
                    reinterpret_cast<const int8_t*>(b_tile), static_cast<int>(N),
                    b_scale, b_zero);
            } else {
                quant_kernel::dequantization<uint8_t>(cur_h, cur_w, b_scratch, cur_w,
                    b_tile, static_cast<int>(N), b_scale, b_zero);
            }

            // Binary op in f32 — result overwrites a_scratch in-place.
            apply_eltwise_flt<float>(attrs.type, a_scratch, b_scratch, a_scratch,
                                     cur_h, cur_w, cur_w, cur_w, cur_w, false);

            // Quantize → output.
            if (o_dtype == DataType::s8) {
                quant_kernel::quantization<int8_t>(cur_h, cur_w,
                    reinterpret_cast<int8_t*>(o_tile), static_cast<int>(N),
                    a_scratch, cur_w, o_scale, o_zero);
            } else {
                quant_kernel::quantization<uint8_t>(cur_h, cur_w, o_tile,
                    static_cast<int>(N), a_scratch, cur_w, o_scale, o_zero);
            }
        }
    };

    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, num_tiles, body);
    } else {
        for (int64_t t = 0; t < num_tiles; ++t) {
            body(t);
        }
    }
}

// ============================================================
// Entry point with dtype dispatch
// ============================================================

void eltwise_cpu(const EltwiseAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx,
                   void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    const bool in_is_int  = is_quantized_dtype(dtype);
    const bool out_is_int = is_quantized_dtype(output.data_type());

    if (!in_is_int && !out_is_int) {
        switch (dtype) {
        case DataType::f32:
            eltwise_impl<float>(attrs, output, inputs, ctx);
            return;
        case DataType::f16:
            eltwise_impl<half>(attrs, output, inputs, ctx);
            return;
        default:
            NNOPS_ASSERT(!"eltwise_cpu: unsupported data type (only f32 and f16)");
        }
    }

    // Quantized path: both inputs and the output must be s8/u8 (no float↔int mixing).
    NNOPS_ASSERT(in_is_int && out_is_int);

    eltwise_quant_impl(attrs, output, inputs, ctx);
}

}  // namespace nnops::backend::cpu
