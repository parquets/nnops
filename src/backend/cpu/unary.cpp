/// @file unary.cpp
/// @brief SIMD-optimized CPU implementation of element-wise unary operations.
///
/// Supports both f32 and f16 via a single templated implementation.
/// Processing is tiled in groups of 8 rows and dispatched via
/// ComputeContext::cpu_parallel_for when available.
///
/// Design:
///   1. Rows are grouped into tiles of TILE_M (32) for SIMD-friendly blocking
///   2. Each tile calls into tiled_unary kernel library (lane=8, v_f32x8/v_f16x8)
///   3. Tile iteration is parallelized via ctx.cpu_parallel_for; falls back to
///      sequential when no parallel hook is provided
///   4. Op-type dispatch lives in the shared apply_unary_flt helper
///
/// Quantized input/output (s8/u8 → s8/u8) is supported through a fused path: the
/// input is dequantized (int → f32), the unary op is applied in f32, and the
/// result is re-quantized (f32 → int). Both input and output must be quantized
/// (no float↔int mixing). Only PerTensor / PerToken granularity is supported
/// (activation quantization); the quant/dequant work is delegated to the
/// raw-intrinsic arch kernels in `x86_64/quant.hpp` / `aarch64/quant.hpp` (NOT
/// `simd_kernel/simd_quant.hpp`). Float-meaning ops (Round/Ceil/Floor) do not
/// support quantized data types.

#include "nnops/ops/unary.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "nnops/detail/half.hpp"
#include "simd_kernel/simd_unary.hpp"

#if defined(NNOPS_ARCH_X86_64)
#include "x86_64/quant.hpp"
#elif defined(NNOPS_ARCH_AARCH64)
#include "aarch64/quant.hpp"
#else
#error "unary: unsupported architecture for quantization kernels"
#endif

#include <algorithm>
#include <cstdint>

namespace nnops::backend::cpu {

#if defined(NNOPS_ARCH_X86_64)
namespace quant_kernel = nnops::backend::cpu::x86_64;
#elif defined(NNOPS_ARCH_AARCH64)
namespace quant_kernel = nnops::backend::cpu::aarch64;
#endif

// ============================================================
// Fused quantized unary — helpers
// ============================================================

namespace {

/// Apply a floating-point unary op to an M×N tile, dispatching on the op type.
/// Shared by the f32/f16 paths (in/out strides may differ) and the quantized
/// path (in == out, stride == N, add_to == false). Round/Ceil/Floor are only
/// reachable from the float path.
template <typename T>
void apply_unary_flt(UnaryType type, const T* in, T* out,
                     int64_t M, int64_t N, int64_t in_stride, int64_t out_stride,
                     bool add_to) {
    switch (type) {
    case UnaryType::Exp:   kernel::exp<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Log:   kernel::log<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Sin:   kernel::sin<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Cos:   kernel::cos<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Tan:   kernel::tan<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Tanh:  kernel::tanh<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Abs:   kernel::abs<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Neg:   kernel::neg<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Sqrt:  kernel::sqrt<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Erf:   kernel::erf<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Round: kernel::round<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Ceil:  kernel::ceil<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Floor: kernel::floor<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Recip: kernel::recip<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    case UnaryType::Sign:  kernel::sign<T>(in, out, M, N, in_stride, out_stride, add_to); break;
    }
}

}  // anonymous namespace

// ============================================================
// Templated implementation (f32 and f16)
// ============================================================

template <typename T>
void unary_impl(const UnaryAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t total = input.numel();
    if (total == 0) { return; }

    const int64_t rank = input.rank();
    NNOPS_ASSERT(output.numel() == total);

    // Row-by-row layout (pitch-aware).
    const int64_t last_dim = (rank >= 1) ? input.shape(rank - 1) * input.channel_pack_size() : 1;
    const int64_t num_rows = (rank >= 2) ? input.total_rows() : total / last_dim;
    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t out_row_stride = output.row_stride_elems();

    const auto* in_ptr  = input.ptr<T>();
    auto*       out_ptr = output.ptr<T>();
    const bool  add_to  = attrs.add_to;

    constexpr int64_t TILE_M = 32;
    const int64_t num_tiles = (num_rows + TILE_M - 1) / TILE_M;

    // Tiled dispatch: split rows into tiles of ≤TILE_M, parallelized when
    // available. Op-type dispatch lives in apply_unary_flt.
    auto body = [&](int64_t ti) {
        const int64_t r = ti * TILE_M;
        const int64_t m = std::min(TILE_M, num_rows - r);
        apply_unary_flt<T>(attrs.type,
                           in_ptr  + r * in_row_stride,
                           out_ptr + r * out_row_stride,
                           m, last_dim, in_row_stride, out_row_stride, add_to);
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
// Fused quantized unary — dequantize → unary → quantize
// ============================================================
//
// Quantized unary: both input and output are s8/u8 (no float mixing). The
// unary math runs in f32; the int input is dequantized and the int output is
// quantized via the arch quant.hpp per-row kernels. Granularity is PerTensor or
// PerToken only (no PerChannel for activations).
//
// The tensor is processed in TILE_H×TILE_W tiles so every intermediate buffer
// (f32 scratch, per-row scale/zero) is a fixed stack array — no heap allocation
// or per-call deallocation anywhere in the kernel. Row tiles are dispatched in
// parallel via ComputeContext::cpu_parallel_for (matching the float path).

void unary_quant_impl(const UnaryAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t numel = input.numel();
    if (numel == 0) { return; }

    // add_to is only meaningful on the float (non-quantized) path.
    NNOPS_ASSERT(!attrs.add_to);

    const DataType in_dtype  = input.data_type();   // s8 or u8
    const DataType out_dtype = output.data_type();  // s8 or u8

    const QuantParams& in_qp  = input.quant_params();
    const QuantParams& out_qp = output.quant_params();

    // Input and output share the same quantization granularity (both PerTensor
    // or both PerToken), so a single flag drives dequant and quant alike.
    NNOPS_ASSERT(in_qp.granularity == out_qp.granularity);
    const bool per_token = in_qp.granularity == QuantGranularity::PerToken
                        && in_qp.scale_data != nullptr;

    // Row shape: PerToken = one (scale, zero_point) per innermost row;
    // PerTensor = one parameter for the whole tensor (flat M=1, N=numel).
    int64_t M = 1;
    int64_t N = input.numel();
    if (per_token) {
        N = input.shape(input.rank() - 1);
        M = (N == 0) ? 0 : (input.numel() / N);
    }

    // s8/u8 are both single-byte, so row/column offsets are in elements.
    const uint8_t* in_base  = input.ptr<uint8_t>();
    uint8_t*       out_base = output.ptr<uint8_t>();

    // Fixed stack buffers per worker — the kernel allocates nothing on the heap.
    // Row tiles are dispatched in parallel (matching the float path); column
    // tiles keep each worker's f32 scratch bounded to a small stack array.
    constexpr int TILE_H = 8;
    constexpr int TILE_W = 256;
    const int64_t num_tiles = (M + TILE_H - 1) / TILE_H;

    auto body = [&](int64_t ti) {
        const int64_t h0 = ti * TILE_H;
        const int cur_h = static_cast<int>(std::min<int64_t>(TILE_H, M - h0));

        float scratch[TILE_H * TILE_W];
        float in_scale[TILE_H],  in_zero[TILE_H];
        float out_scale[TILE_H], out_zero[TILE_H];

        for (int i = 0; i < cur_h; ++i) {
            const int64_t r = h0 + i;
            in_scale[i]  = per_token ? in_qp.scale_data[r] : in_qp.scale;
            in_zero[i]   = per_token
                ? (in_qp.zero_point_data != nullptr ? static_cast<float>(in_qp.zero_point_data[r]) : 0.0f)
                : static_cast<float>(in_qp.zero_point);
            out_scale[i] = per_token ? out_qp.scale_data[r] : out_qp.scale;
            out_zero[i]  = per_token
                ? (out_qp.zero_point_data != nullptr ? static_cast<float>(out_qp.zero_point_data[r]) : 0.0f)
                : static_cast<float>(out_qp.zero_point);
        }

        for (int64_t w0 = 0; w0 < N; w0 += TILE_W) {
            const int cur_w = static_cast<int>(std::min<int64_t>(TILE_W, N - w0));
            const uint8_t* in_tile  = in_base  + (h0 * N + w0);
            uint8_t*       out_tile = out_base + (h0 * N + w0);

            // Dequantize → f32.
            if (in_dtype == DataType::s8) {
                quant_kernel::dequantization<int8_t>(cur_h, cur_w, scratch, cur_w,
                    reinterpret_cast<const int8_t*>(in_tile), static_cast<int>(N),
                    in_scale, in_zero);
            } else {
                quant_kernel::dequantization<uint8_t>(cur_h, cur_w, scratch, cur_w,
                    in_tile, static_cast<int>(N), in_scale, in_zero);
            }

            apply_unary_flt<float>(attrs.type, scratch, scratch, cur_h, cur_w, cur_w, cur_w, false);

            // Quantize → output.
            if (out_dtype == DataType::s8) {
                quant_kernel::quantization<int8_t>(cur_h, cur_w,
                    reinterpret_cast<int8_t*>(out_tile), static_cast<int>(N),
                    scratch, cur_w, out_scale, out_zero);
            } else {
                quant_kernel::quantization<uint8_t>(cur_h, cur_w, out_tile,
                    static_cast<int>(N), scratch, cur_w, out_scale, out_zero);
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

void unary_cpu(const UnaryAttributes& attrs,
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
            unary_impl<float>(attrs, output, inputs, ctx);
            return;
        case DataType::f16:
            unary_impl<half>(attrs, output, inputs, ctx);
            return;
        default:
            NNOPS_ASSERT(!"unary_cpu: unsupported data type (only f32 and f16)");
        }
    }

    // Quantized path: both input and output must be s8/u8 (no float↔int mixing).
    NNOPS_ASSERT(in_is_int && out_is_int);

    // Round/Ceil/Floor are float-meaning ops — no quantized-dtype support.
    if (attrs.type == UnaryType::Round ||
        attrs.type == UnaryType::Ceil  ||
        attrs.type == UnaryType::Floor) {
        NNOPS_ASSERT(!"unary_cpu: Round/Ceil/Floor do not support quantized data types");
    }

    unary_quant_impl(attrs, output, inputs, ctx);
}

}  // namespace nnops::backend::cpu
