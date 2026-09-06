#pragma once
/// @file elementwise.hpp
/// @brief Shared drivers for the CPU element-wise operators (Activation, Unary,
///        Eltwise, Clamp) — row geometry, tiled float dispatch, and the fused
///        dequantize → op → quantize path.
///
/// The four operators share the same shape: a pitch-aware row loop tiled into
/// groups of TILE_M rows, dispatched via `ComputeContext::cpu.parallel_for`,
/// plus (for Activation/Unary/Eltwise) a quantized path that dequantizes the
/// int input(s) to f32, runs the float op, and requantizes the result. This
/// header is the single source of truth for that boilerplate; each operator
/// only supplies its op-type dispatch switch.
///
/// Quantized dequant/requant delegates to the raw-intrinsic arch kernels in
/// `x86_64/quant.hpp` / `aarch64/quant.hpp` (NOT `simd_kernel/simd_quant.hpp`).

#include "nnops/core/compute_context.hpp"
#include "nnops/core/data_type.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/core/quant_params.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/detail/assert.hpp"

#if defined(NNOPS_ARCH_X86_64)
#include "backend/cpu/x86_64/quant.hpp"
#elif defined(NNOPS_ARCH_AARCH64)
#include "backend/cpu/aarch64/quant.hpp"
#else
#error "elementwise: unsupported architecture for quantization kernels"
#endif

#include <algorithm>
#include <cstdint>

namespace nnops::backend::cpu {

#if defined(NNOPS_ARCH_X86_64)
namespace quant_kernel = x86_64;
#elif defined(NNOPS_ARCH_AARCH64)
namespace quant_kernel = aarch64;
#endif

// ============================================================
// Parallel dispatch + row geometry
// ============================================================

/// Run `body(i)` for i in [0, count) — parallelized when a hook is installed,
/// sequential otherwise. The canonical replacement for the per-operator
/// `ctx.cpu.run(0, count, body)` idiom.
inline void run_parallel(const ComputeContext& ctx, int64_t count,
                         const ParallelForBody& body) {
    ctx.cpu.run(0, count, body);
}

/// Row-major view of a pitch-aware tensor: `num_rows` logical rows of
/// `last_dim` elements, with input/output row strides (elements).
struct RowGeometry {
    int64_t last_dim;   ///< elements per row (already includes channel packing)
    int64_t num_rows;   ///< number of logical rows
    int64_t in_stride;  ///< input row stride (elements)
    int64_t out_stride; ///< output row stride (elements)
};

/// Derive the row geometry shared by every element-wise float kernel.
inline RowGeometry rowwise_geometry(const TensorView& input, const TensorView& output) {
    const int64_t total = input.numel();
    const int64_t rank  = input.rank();
    const int64_t last_dim = (rank >= 1) ? input.shape(rank - 1) * input.channel_pack_size() : 1;
    const int64_t num_rows = (rank >= 2) ? input.total_rows() : total / last_dim;
    return { last_dim, num_rows, input.row_stride_elems(), output.row_stride_elems() };
}

// ============================================================
// Tiled float dispatch (f32 / f16)
// ============================================================

/// Row-tiled float transform for a unary op. `apply` receives the base pointers
/// of a ≤TILE_M-row tile and its row strides, and runs the op over it.
template <typename T, typename ApplyFn>
void tiled_float_transform(const TensorView& input, TensorView& output,
                           const ComputeContext& ctx, ApplyFn&& apply,
                           int64_t tile_m = 32) {
    const int64_t total = input.numel();
    if (total == 0) { return; }
    NNOPS_ASSERT(output.numel() == total);

    const RowGeometry g = rowwise_geometry(input, output);
    const T* in  = input.ptr<T>();
    T*       out = output.ptr<T>();

    const int64_t num_tiles = (g.num_rows + tile_m - 1) / tile_m;
    auto body = [&](int64_t ti) {
        const int64_t r = ti * tile_m;
        const int64_t m = std::min(tile_m, g.num_rows - r);
        apply(in + r * g.in_stride, out + r * g.out_stride,
              m, g.last_dim, g.in_stride, g.out_stride);
    };
    run_parallel(ctx, num_tiles, body);
}

/// Row-tiled float transform for a binary op (two inputs, independent strides).
template <typename T, typename ApplyFn>
void tiled_float_transform2(const TensorView& a, const TensorView& b, TensorView& out,
                            const ComputeContext& ctx, ApplyFn&& apply,
                            int64_t tile_m = 32) {
    const int64_t total = a.numel();
    if (total == 0) { return; }
    NNOPS_ASSERT(a.numel() == b.numel());
    NNOPS_ASSERT(out.numel() == total);

    const RowGeometry g = rowwise_geometry(a, out);
    const int64_t b_stride = b.row_stride_elems();
    const T* ap = a.ptr<T>();
    const T* bp = b.ptr<T>();
    T*       op = out.ptr<T>();

    const int64_t num_tiles = (g.num_rows + tile_m - 1) / tile_m;
    auto body = [&](int64_t ti) {
        const int64_t r = ti * tile_m;
        const int64_t m = std::min(tile_m, g.num_rows - r);
        apply(ap + r * g.in_stride, bp + r * b_stride, op + r * g.out_stride,
              m, g.last_dim, g.in_stride, b_stride, g.out_stride);
    };
    run_parallel(ctx, num_tiles, body);
}

// ============================================================
// Fused quantized transform — dequantize → op → quantize
// ============================================================

/// Fill per-row scale/zero arrays for a ≤TILE_H-row tile of a quantized tensor.
inline void fill_quant_row_params(const QuantParams& qp, bool per_token,
                                  int64_t h0, int cur_h, float* scale, float* zero) {
    for (int i = 0; i < cur_h; ++i) {
        const int64_t r = h0 + i;
        scale[i] = per_token ? qp.scale_data[r] : qp.scale;
        zero[i]  = per_token
            ? (qp.zero_point_data != nullptr ? static_cast<float>(qp.zero_point_data[r]) : 0.0f)
            : static_cast<float>(qp.zero_point);
    }
}

/// Dequantize an s8/u8 tile to f32, dispatching on the storage dtype.
inline void dequantize_tile(DataType dt, int h, int w, float* dst, int dst_stride,
                            const uint8_t* src, int src_stride,
                            const float* scale, const float* zero) {
    if (dt == DataType::s8) {
        quant_kernel::dequantization<int8_t>(h, w, dst, dst_stride,
            reinterpret_cast<const int8_t*>(src), src_stride, scale, zero);
    } else {
        quant_kernel::dequantization<uint8_t>(h, w, dst, dst_stride,
            src, src_stride, scale, zero);
    }
}

/// Quantize an f32 tile to s8/u8, dispatching on the storage dtype.
inline void quantize_tile(DataType dt, int h, int w, uint8_t* dst, int dst_stride,
                          const float* src, int src_stride,
                          const float* scale, const float* zero) {
    if (dt == DataType::s8) {
        quant_kernel::quantization<int8_t>(h, w, reinterpret_cast<int8_t*>(dst), dst_stride,
            src, src_stride, scale, zero);
    } else {
        quant_kernel::quantization<uint8_t>(h, w, dst, dst_stride,
            src, src_stride, scale, zero);
    }
}

/// Fused quantized unary: dequantize input → apply op (f32, in place) → quantize.
///
/// Input and output must both be quantized and share the same granularity
/// (PerTensor or PerToken). `apply(float* scratch, int h, int w)` runs the op
/// over an h×w f32 tile in place.
template <typename ApplyFn>
void quantized_unary_transform(const TensorView& input, TensorView& output,
                               const ComputeContext& ctx, ApplyFn&& apply) {
    const int64_t numel = input.numel();
    if (numel == 0) { return; }

    const DataType in_dtype  = input.data_type();
    const DataType out_dtype = output.data_type();
    const QuantParams& in_qp  = input.quant_params();
    const QuantParams& out_qp = output.quant_params();

    NNOPS_ASSERT(in_qp.granularity == out_qp.granularity);
    const bool per_token = in_qp.granularity == QuantGranularity::PerToken
                        && in_qp.scale_data != nullptr;

    int64_t M = 1;
    int64_t N = numel;
    if (per_token) {
        N = input.shape(input.rank() - 1);
        M = (N == 0) ? 0 : (numel / N);
    }

    const uint8_t* in_base  = input.ptr<uint8_t>();
    uint8_t*       out_base = output.ptr<uint8_t>();

    constexpr int TILE_H = 8;
    constexpr int TILE_W = 256;
    const int64_t num_tiles = (M + TILE_H - 1) / TILE_H;

    auto body = [&](int64_t ti) {
        const int64_t h0 = ti * TILE_H;
        const int cur_h = static_cast<int>(std::min<int64_t>(TILE_H, M - h0));

        float scratch[TILE_H * TILE_W];
        float in_scale[TILE_H],  in_zero[TILE_H];
        float out_scale[TILE_H], out_zero[TILE_H];

        fill_quant_row_params(in_qp,  per_token, h0, cur_h, in_scale,  in_zero);
        fill_quant_row_params(out_qp, per_token, h0, cur_h, out_scale, out_zero);

        for (int64_t w0 = 0; w0 < N; w0 += TILE_W) {
            const int cur_w = static_cast<int>(std::min<int64_t>(TILE_W, N - w0));
            const uint8_t* in_tile  = in_base  + (h0 * N + w0);
            uint8_t*       out_tile = out_base + (h0 * N + w0);

            dequantize_tile(in_dtype, cur_h, cur_w, scratch, cur_w, in_tile,
                            static_cast<int>(N), in_scale, in_zero);
            apply(scratch, cur_h, cur_w);
            quantize_tile(out_dtype, cur_h, cur_w, out_tile, static_cast<int>(N),
                          scratch, cur_w, out_scale, out_zero);
        }
    };

    run_parallel(ctx, num_tiles, body);
}

/// Fused quantized binary: dequantize A and B → apply op (f32) → quantize.
///
/// Both inputs and the output must be quantized and share the same granularity.
/// `apply(float* a, float* b, int h, int w)` runs the binary op; the result is
/// written into `a` (so `a` doubles as the result scratch).
template <typename ApplyFn>
void quantized_binary_transform(const TensorView& a_in, const TensorView& b_in,
                                TensorView& output, const ComputeContext& ctx,
                                ApplyFn&& apply) {
    const int64_t numel = a_in.numel();
    if (numel == 0) { return; }
    NNOPS_ASSERT(a_in.numel() == b_in.numel());

    const DataType a_dtype = a_in.data_type();
    const DataType b_dtype = b_in.data_type();
    const DataType o_dtype = output.data_type();
    const QuantParams& a_qp = a_in.quant_params();
    const QuantParams& b_qp = b_in.quant_params();
    const QuantParams& o_qp = output.quant_params();

    NNOPS_ASSERT(a_qp.granularity == o_qp.granularity);
    NNOPS_ASSERT(b_qp.granularity == o_qp.granularity);
    const bool per_token = a_qp.granularity == QuantGranularity::PerToken
                        && a_qp.scale_data != nullptr;

    int64_t M = 1;
    int64_t N = numel;
    if (per_token) {
        N = a_in.shape(a_in.rank() - 1);
        M = (N == 0) ? 0 : (numel / N);
    }

    const uint8_t* a_base = a_in.ptr<uint8_t>();
    const uint8_t* b_base = b_in.ptr<uint8_t>();
    uint8_t*       o_base = output.ptr<uint8_t>();

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

        fill_quant_row_params(a_qp, per_token, h0, cur_h, a_scale, a_zero);
        fill_quant_row_params(b_qp, per_token, h0, cur_h, b_scale, b_zero);
        fill_quant_row_params(o_qp, per_token, h0, cur_h, o_scale, o_zero);

        for (int64_t w0 = 0; w0 < N; w0 += TILE_W) {
            const int cur_w = static_cast<int>(std::min<int64_t>(TILE_W, N - w0));
            const uint8_t* a_tile = a_base + (h0 * N + w0);
            const uint8_t* b_tile = b_base + (h0 * N + w0);
            uint8_t*       o_tile = o_base + (h0 * N + w0);

            dequantize_tile(a_dtype, cur_h, cur_w, a_scratch, cur_w, a_tile,
                            static_cast<int>(N), a_scale, a_zero);
            dequantize_tile(b_dtype, cur_h, cur_w, b_scratch, cur_w, b_tile,
                            static_cast<int>(N), b_scale, b_zero);

            apply(a_scratch, b_scratch, cur_h, cur_w);

            quantize_tile(o_dtype, cur_h, cur_w, o_tile, static_cast<int>(N),
                          a_scratch, cur_w, o_scale, o_zero);
        }
    };

    run_parallel(ctx, num_tiles, body);
}

}  // namespace nnops::backend::cpu
