/// @file softmax.cpp
/// @brief SIMD-optimized CPU implementation of softmax / log-softmax.
///
/// Supports both f32 and f16 via a single templated implementation.
///
/// Supports planar (NCHW/NCDHW) and packed (NCHWC8/NCDHWC8) layouts.
///
/// Reference: onnxruntime MlasComputeSoftmax + MlasReduceMaximumF32Kernel +
///            MlasComputeSumExpF32Kernel + MlasComputeSoftmaxOutputF32Kernel.
///
/// Three paths:
///   - Packed SIMD (axis == rank-1, pack > 1): per-lane SIMD reduction
///     within each physical row. Delegates to kernel::softmax_process_packed_row.
///   - Standard SIMD fast path (axis == rank-1, pack == 1): contiguous
///     tail. Delegates to kernel::softmax_process_standard_row.
///   - General scalar path: reference-style outer/D/inner decomposition
///     with strided access, correct for all layouts and axes.
///
/// Pitch-aware via stride_elems() / row_stride_elems().

#include "nnops/ops/softmax.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "nnops/detail/half.hpp"
#include "simd_kernel/simd_softmax.hpp"

#if defined(NNOPS_ARCH_X86_64)
#include "x86_64/quant.hpp"
#elif defined(NNOPS_ARCH_AARCH64)
#include "aarch64/quant.hpp"
#else
#error "softmax: unsupported architecture for quantization kernels"
#endif

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <vector>

namespace nnops::backend::cpu {

using namespace nnops::simd;

#if defined(NNOPS_ARCH_X86_64)
namespace quant_kernel = nnops::backend::cpu::x86_64;
#elif defined(NNOPS_ARCH_AARCH64)
namespace quant_kernel = nnops::backend::cpu::aarch64;
#endif

namespace {

// ============================================================
// General-axis softmax — SIMD strided + scalar tail
// ============================================================
//
// When the tensor is densely packed (pitch == last_dim * elem_size), inner
// positions are contiguous in memory and we can process them in SIMD groups
// of L=8. Each SIMD lane does an independent softmax over the D strided
// axis elements. The scalar tail handles the remaining inner positions
// and the case where pitch padding breaks contiguity.
//
// When pitch padding is present, inner positions are NOT contiguous
// (row boundaries have gaps), so we fall back to the scalar decomposition
// with precomputed inner offsets.

template <typename T>
void softmax_general(
    const T* x_ptr, T* y_ptr,
    const TensorView& X,
    int64_t axis, int64_t outer_size, int64_t D, int64_t inner_total,
    bool log_softmax, float inv_T,
    const ComputeContext& ctx)
{
    constexpr int L = simd_lane_for<T>;
    const int64_t rank = X.rank();
    const int64_t pack = X.channel_pack_size();  // always 1 here (packed paths handled earlier)
    const int64_t axis_stride = X.stride_elems(axis) * pack;

    // Check whether inner positions are contiguous in memory.
    // - axis == rank-2: inner = last dim only, always contiguous
    // - axis < rank-2:  contiguous iff no pitch padding
    const int64_t elem_size = static_cast<int64_t>(data_type_size(X.data_type()));
    const bool dense = (axis >= rank - 2) ||
                       (X.pitch() == X.shape(rank - 1) * elem_size);

    const auto process_row = [&](int64_t outer) {
        int64_t base = outer * D * axis_stride;

        // Scalar softmax over D strided axis elements starting at inner_off
        const auto scalar_softmax = [&](int64_t inner_off) {
            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t k = 0; k < D; ++k) {
                float v = s_load(&x_ptr[base + inner_off + k * axis_stride]);
                if (v > max_val) { max_val = v; }
            }
            float sum_exp = 0.0f;
            for (int64_t k = 0; k < D; ++k) {
                sum_exp += std::exp((s_load(&x_ptr[base + inner_off + k * axis_stride]) - max_val) * inv_T);
            }
            if (log_softmax) {
                float log_sum = std::log(sum_exp);
                for (int64_t k = 0; k < D; ++k) {
                    float val = (s_load(&x_ptr[base + inner_off + k * axis_stride]) - max_val) * inv_T - log_sum;
                    s_store(&y_ptr[base + inner_off + k * axis_stride], val);
                }
            } else {
                float inv_sum = 1.0f / sum_exp;
                for (int64_t k = 0; k < D; ++k) {
                    float val = std::exp((s_load(&x_ptr[base + inner_off + k * axis_stride]) - max_val) * inv_T) * inv_sum;
                    s_store(&y_ptr[base + inner_off + k * axis_stride], val);
                }
            }
        };

        if (dense) {
            // ---- SIMD path: process inner_total in groups of L ----
            int64_t s = 0;
            for (; s + L <= inner_total; s += L) {
                kernel::softmax_per_lane<T>(
                    x_ptr + base + s, y_ptr + base + s,
                    D, axis_stride, log_softmax, inv_T);
            }
            for (; s < inner_total; ++s) {
                scalar_softmax(s);
            }
        } else {
            // ---- Pitch-padded fallback: scalar with precomputed offsets ----
            std::vector<int64_t> inner_offsets(static_cast<size_t>(inner_total));
            for (int64_t s = 0; s < inner_total; ++s) {
                int64_t off = 0, rem = s;
                for (int64_t d = rank - 1; d > axis; --d) {
                    int64_t dim = X.shape(d);
                    off += (rem % dim) * X.stride_elems(d) * pack;
                    rem /= dim;
                }
                inner_offsets[static_cast<size_t>(s)] = off;
            }

            for (int64_t s = 0; s < inner_total; ++s) {
                scalar_softmax(inner_offsets[static_cast<size_t>(s)]);
            }
        }
    };

    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, outer_size, process_row);
    }
    else {
        for (int64_t i = 0; i < outer_size; ++i) {
            process_row(i);
        }
    }
}


}  // anonymous namespace

// ============================================================
// Templated implementation (f32 and f16)
// ============================================================

template <typename T>
void softmax_impl(const SoftmaxAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx)
{
    const auto& X = inputs[0];

    const int64_t rank = X.rank();
    NNOPS_ASSERT(rank >= 1);

    const bool log_softmax = attrs.log_softmax;
    const float temperature = attrs.temperature;
    NNOPS_ASSERT(temperature > 0.0f);
    const float inv_T = 1.0f / temperature;

    // Normalize axis
    int64_t axis = attrs.axis;
    if (axis < 0) { axis += rank; }
    NNOPS_ASSERT(axis >= 0 && axis < rank);

    const auto* x_ptr = X.ptr<T>();
    auto* y_ptr = output.ptr<T>();

    const int64_t pack = X.channel_pack_size();

    // ============================================================
    // Path 1 — Packed SIMD: axis == rank-1, pack > 1
    // ============================================================

    if (axis == rank - 1 && pack > 1) {
        const int64_t D = X.shape(axis);
        const int64_t num_rows = X.total_rows();
        const int64_t x_rs = X.row_stride_elems();
        const int64_t y_rs = output.row_stride_elems();

        const auto process_row = [&](int64_t r) {
            kernel::softmax_per_lane<T>(
                x_ptr + r * x_rs, y_ptr + r * y_rs, D, pack, log_softmax, inv_T);
        };

        if (ctx.cpu_parallel_for) {
            ctx.cpu_parallel_for(0, num_rows, process_row);
        }
        else {
            for (int64_t r = 0; r < num_rows; ++r) {
                process_row(r);
            }
        }

        return;
    }

    // ============================================================
    // Path 1b — Packed channel SIMD: axis == 1, pack > 1 (NCHWC8)
    // ============================================================

    if (axis == 1 && pack > 1) {
        const int64_t C = X.shape(1);                    // logical channel count
        const int64_t C8 = X.num_channel_blocks();        // ceil(C / pack)
        const int64_t chan_stride = X.stride_elems(1);    // stride between C8 blocks
        const int64_t valid_lanes = (C % pack == 0) ? pack : (C % pack);

        // Count spatial positions: product of all dims except C8 (axis=1)
        int64_t num_spatial = 1;
        for (int64_t d = 0; d < rank; ++d) {
            if (d != 1) {
                num_spatial *= X.shape(d);
            }
        }

        const auto process_pos = [&](int64_t s) {
            // Map flat spatial index → tensor offset (skipping C8 dim)
            int64_t off = 0;
            int64_t rem = s;
            for (int64_t d = rank - 1; d >= 0; --d) {
                if (d == 1) {
                    continue;
                }
                int64_t dim = X.shape(d);
                off += (rem % dim) * X.stride_elems(d);
                rem /= dim;
            }
            kernel::softmax_process_packed_channel<T>(
                x_ptr + off, y_ptr + off,
                chan_stride,
                C8, pack, valid_lanes, log_softmax, inv_T);
        };

        if (ctx.cpu_parallel_for) {
            ctx.cpu_parallel_for(0, num_spatial, process_pos);
        }
        else {
            for (int64_t i = 0; i < num_spatial; ++i) {
                process_pos(i);
            }
        }

        return;
    }

    // ============================================================
    // Path 1c — Packed column SIMD (pack > 1, general non-channel axis)
    //           Covers NCHWC8 H, NCDHWC8 D/H, and axis=0 for all.
    // ============================================================

    if (pack > 1) {
        const int64_t D = X.shape(axis);
        const int64_t axis_stride = X.stride_elems(axis);

        int64_t num_positions = 1;
        for (int64_t d = 0; d < rank; ++d) {
            if (d != axis) {
                num_positions *= X.shape(d);
            }
        }

        const auto process_pos = [&](int64_t s) {
            int64_t off = 0;
            int64_t rem = s;
            for (int64_t d = rank - 1; d >= 0; --d) {
                if (d == axis) {
                    continue;
                }
                int64_t dim = X.shape(d);
                off += (rem % dim) * X.stride_elems(d);
                rem /= dim;
            }
            kernel::softmax_per_lane<T>(
                x_ptr + off, y_ptr + off,
                D, axis_stride, log_softmax, inv_T);
        };

        if (ctx.cpu_parallel_for) {
            ctx.cpu_parallel_for(0, num_positions, process_pos);
        }
        else {
            for (int64_t i = 0; i < num_positions; ++i) {
                process_pos(i);
            }
        }

        return;
    }

    // ============================================================
    // Path 2 — Standard SIMD fast path: axis == rank-1, pack == 1
    // ============================================================

    if (axis == rank - 1) {
        int64_t num_rows = 1;
        for (int64_t i = 0; i < axis; ++i) {
            num_rows *= X.shape(i);
        }

        const int64_t D = X.shape(axis);
        const int64_t group_stride = (axis > 0) ? X.stride_elems(axis - 1) : D;

        const auto process_row = [&](int64_t row) {
            kernel::softmax_process_standard_row<T>(
                x_ptr + row * group_stride, y_ptr + row * group_stride, D, log_softmax, inv_T);
        };

        if (ctx.cpu_parallel_for) {
            ctx.cpu_parallel_for(0, num_rows, process_row);
        }
        else {
            for (int64_t row = 0; row < num_rows; ++row) {
                process_row(row);
            }
        }

        return;
    }

    // ============================================================
    // Path 3 — General axis: scalar, reference-style decomposition
    // ============================================================

    {
        int64_t outer_size = 1;
        for (int64_t i = 0; i < axis; ++i) {
            outer_size *= X.shape(i);
        }

        const int64_t D = X.shape(axis);

        int64_t inner_total = 1;
        for (int64_t i = axis + 1; i < rank; ++i) {
            inner_total *= X.shape(i);
        }

        softmax_general<T>(
            x_ptr, y_ptr, X,
            axis, outer_size, D, inner_total,
            log_softmax, inv_T, ctx);
    }
}

// ============================================================
// Quantized input (s8/u8) → float output (f32/f16)
// ============================================================
//
// Quantized softmax: the integer input is dequantized to f32 (via the arch
// quant.hpp per-row kernels) into a contiguous scratch buffer, softmax runs in
// f32 on that scratch, and the result is written back as f32 or f16 (softmax
// probabilities are not re-quantized). Only PerTensor / PerToken granularity
// and planar (pack == 1) layouts are supported — packed channel layouts with
// quantized input are not defined.

void softmax_quant_input_impl(const SoftmaxAttributes& attrs,
                              TensorView& output,
                              std::span<const TensorView> inputs,
                              const ComputeContext& ctx,
                              void* workspace)
{
    const auto& X = inputs[0];
    const int64_t numel = X.numel();
    const int64_t rank = X.rank();
    NNOPS_ASSERT(rank >= 1);

    const DataType in_dtype  = X.data_type();             // s8 or u8
    const DataType out_dtype = output.data_type();        // f32 or f16
    NNOPS_ASSERT(is_quantized_dtype(in_dtype));
    NNOPS_ASSERT(out_dtype == DataType::f32 || out_dtype == DataType::f16);

    // Quantized softmax is only defined over planar layouts.
    NNOPS_ASSERT(X.channel_pack_size() == 1);
    NNOPS_ASSERT(workspace != nullptr);

    const QuantParams& qp = X.quant_params();
    const bool per_token = qp.granularity == QuantGranularity::PerToken
                        && qp.scale_data != nullptr;

    // Row shape: PerToken = one (scale, zero_point) per innermost row;
    // PerTensor = one parameter for the whole tensor.
    const int64_t last_dim = X.shape(rank - 1);
    const int64_t M = numel / last_dim;  // number of `last_dim`-element rows

    // Carve the caller-provided workspace (sized by Softmax::getWorkspaceSize):
    //   [ x_f32 : numel ][ scale : M ][ zero : M ][ y_f32 : numel (f16 only) ]
    float* x_f32 = static_cast<float*>(workspace);
    float* scale = x_f32 + numel;
    float* zero  = scale + M;

    // Quant kernels take int dimensions/strides (see x86_64/quant.hpp).
    const uint8_t* in_base = X.ptr<uint8_t>();
    const int sr_i = static_cast<int>(X.row_stride_elems());
    const int m_i = static_cast<int>(M);
    const int n_i = static_cast<int>(last_dim);

    // Materialize per-row (scale, zero_point) as float for the arch kernel.
    for (int64_t i = 0; i < M; ++i) {
        if (per_token) {
            scale[i] = qp.scale_data[i];
            zero[i]  = qp.zero_point_data != nullptr
                ? static_cast<float>(qp.zero_point_data[i]) : 0.0f;
        } else {
            scale[i] = qp.scale;
            zero[i]  = static_cast<float>(qp.zero_point);
        }
    }

    // Dequantize into a contiguous f32 scratch (no pitch).
    if (in_dtype == DataType::s8) {
        quant_kernel::dequantization<int8_t>(m_i, n_i, x_f32, n_i,
            reinterpret_cast<const int8_t*>(in_base), sr_i, scale, zero);
    } else {
        quant_kernel::dequantization<uint8_t>(m_i, n_i, x_f32, n_i,
            in_base, sr_i, scale, zero);
    }

    // Run softmax over the contiguous f32 view (same logical shape).
    TensorView x_view(X.shape_span(), DataType::f32, x_f32, TensorLayout::NCHW);
    const TensorView ins[] = {x_view};

    if (out_dtype == DataType::f32) {
        softmax_impl<float>(attrs, output, ins, ctx);
    } else {
        // Compute in f32, then downcast to f16 (probabilities are not quantized).
        float* y_f32 = zero + M;
        TensorView y_view(X.shape_span(), DataType::f32, y_f32, TensorLayout::NCHW);
        softmax_impl<float>(attrs, y_view, ins, ctx);
        half* out = output.ptr<half>();
        for (int64_t i = 0; i < numel; ++i) {
            s_store(&out[i], y_f32[i]);
        }
    }
}

// ============================================================
// Entry point with dtype dispatch
// ============================================================

void softmax_cpu(const SoftmaxAttributes& attrs,
                  TensorView& output,
                  std::span<const TensorView> inputs,
                  const ComputeContext& ctx,
                  void* workspace)
{
    const auto dtype = inputs[0].data_type();
    if (is_quantized_dtype(dtype)) {
        // s8/u8 input: softmax probabilities are dequantized float, never int.
        NNOPS_ASSERT(!is_quantized_dtype(output.data_type()));
        softmax_quant_input_impl(attrs, output, inputs, ctx, workspace);
        return;
    }
    switch (dtype) {
    case DataType::f32:
        softmax_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        softmax_impl<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"softmax_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
