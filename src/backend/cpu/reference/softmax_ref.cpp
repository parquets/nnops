/// @file softmax_ref.cpp
/// @brief Naive CPU reference implementation of softmax / log-softmax.
///
/// Uses the max-subtraction trick for numerical stability.
///
/// Quantized input (s8/u8) → float output (f32/f16) is handled by a scalar
/// fused path: dequantize (int → f32), softmax in f32, write back as float.

#include "nnops/ops/softmax.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/half.hpp"

#include <cmath>
#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <vector>

namespace nnops::backend::cpu::reference {

namespace {

/// Scalar softmax over a contiguous f32 buffer with logical shape.
void softmax_scalar_contig(const float* x, float* y,
                           const int64_t* shape, int64_t rank, int64_t axis,
                           bool log_softmax, float inv_T)
{
    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) { outer *= shape[i]; }
    const int64_t D = shape[axis];
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) { inner *= shape[i]; }

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t s = 0; s < inner; ++s) {
            const int64_t base = (o * D) * inner + s;
            float max_val = -FLT_MAX;
            for (int64_t k = 0; k < D; ++k) {
                float v = x[base + k * inner];
                if (v > max_val) { max_val = v; }
            }
            float sum_exp = 0.0f;
            for (int64_t k = 0; k < D; ++k) {
                sum_exp += std::exp((x[base + k * inner] - max_val) * inv_T);
            }
            if (log_softmax) {
                float log_sum = std::log(sum_exp);
                for (int64_t k = 0; k < D; ++k) {
                    y[base + k * inner] = (x[base + k * inner] - max_val) * inv_T - log_sum;
                }
            } else {
                float inv_sum = 1.0f / sum_exp;
                for (int64_t k = 0; k < D; ++k) {
                    y[base + k * inner] = std::exp((x[base + k * inner] - max_val) * inv_T) * inv_sum;
                }
            }
        }
    }
}

}  // anonymous namespace

void softmax_quant_ref(const SoftmaxAttributes& attrs,
                       TensorView& output,
                       std::span<const TensorView> inputs)
{
    const auto& X = inputs[0];
    const int64_t numel = X.numel();
    const int64_t rank = X.rank();
    NNOPS_ASSERT(rank >= 1);

    const DataType in_dtype  = X.data_type();    // s8 or u8
    const DataType out_dtype = output.data_type();  // f32 or f16

    const auto& qp = X.quant_params();
    const bool per_token = qp.granularity == QuantGranularity::PerToken
                        && qp.scale_data != nullptr;
    const int64_t last_dim = X.shape(rank - 1);

    // Normalize axis.
    int64_t axis = attrs.axis;
    if (axis < 0) { axis += rank; }
    const float inv_T = 1.0f / attrs.temperature;

    // Dequantize (scalar) into a contiguous f32 buffer.
    std::vector<float> x_f32(static_cast<size_t>(numel));
    const int8_t* s8 = reinterpret_cast<const int8_t*>(X.ptr<uint8_t>());
    const uint8_t* u8 = X.ptr<uint8_t>();
    for (int64_t flat = 0; flat < numel; ++flat) {
        const int64_t idx = per_token ? (flat / last_dim) : 0;
        const float s = per_token ? qp.scale_data[idx] : qp.scale;
        const float z = per_token
            ? (qp.zero_point_data != nullptr ? static_cast<float>(qp.zero_point_data[idx]) : 0.0f)
            : static_cast<float>(qp.zero_point);
        const float v = (in_dtype == DataType::s8)
            ? static_cast<float>(s8[flat]) : static_cast<float>(u8[flat]);
        x_f32[static_cast<size_t>(flat)] = (v - z) * s;
    }

    // Softmax in f32.
    std::vector<float> y_f32(static_cast<size_t>(numel));
    softmax_scalar_contig(x_f32.data(), y_f32.data(), X.shape(), rank, axis,
                          attrs.log_softmax, inv_T);

    // Write back (f32 or f16).
    if (out_dtype == DataType::f32) {
        float* out = output.ptr<float>();
        for (int64_t i = 0; i < numel; ++i) {
            out[i] = y_f32[static_cast<size_t>(i)];
        }
    } else {
        nnops::backend::cpu::half* out = output.ptr<nnops::backend::cpu::half>();
        for (int64_t i = 0; i < numel; ++i) {
            out[i] = nnops::backend::cpu::float_to_half(y_f32[static_cast<size_t>(i)]);
        }
    }
}

void softmax_ref(const SoftmaxAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx,
                 void* /*workspace*/)
{
    // Quantized input (s8/u8): dequantize → softmax → float output.
    if (is_quantized_dtype(inputs[0].data_type())) {
        softmax_quant_ref(attrs, output, inputs);
        return;
    }

    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    NNOPS_ASSERT(rank >= 1);

    // Normalize axis
    int64_t axis = attrs.axis;
    if (axis < 0) { axis += rank; }
    NNOPS_ASSERT(axis >= 0 && axis < rank);

    // Compute outer size and axis stride
    int64_t outer_size = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer_size *= input.shape(i);
    }
    const int64_t D = input.shape(axis);  // normalization dimension
    const int64_t axis_elems = input.stride_elems(axis);  // element stride along axis (accounts for pitch)
    const int64_t inner_elems = input.stride_elems(rank - 1);  // always 1

    const auto* in_ptr  = input.ptr<float>();
    auto* out_ptr = output.ptr<float>();
    const bool log_softmax = attrs.log_softmax;
    const float temperature = attrs.temperature;
    NNOPS_ASSERT(temperature > 0.0f);
    const float inv_T = 1.0f / temperature;

    // Process each row of size D
    const auto process_row = [&](int64_t outer) {
        int64_t base = outer * D * axis_elems;
        // Iterate over inner elements (elements "between" axis values)
        // The inner stride is always 1 (contiguous in the innermost dimension)
        // But the "inner" here refers to the dimensions after axis.
        // We need to iterate over them properly accounting for pitch.
        //
        // For a [B, C, H, W] tensor with axis=1 (C), axis_elems = H * row_stride.
        // The elements at position (outer, k, ...) are at: base + k * axis_elems + inner_off
        // where inner_off traverses the (H, W) sub-grid.
        //
        // We compute inner_total = product of dims[axis+1..rank-1]
        // and iterate over inner using offset_within_dims.
        int64_t inner_total = 1;
        for (int64_t i = axis + 1; i < rank; ++i) {
            inner_total *= input.shape(i);
        }

        for (int64_t s = 0; s < inner_total; ++s) {
            // Compute offset of inner position s within dims[axis+1..rank-1]
            int64_t inner_off = 0;
            int64_t rem = s;
            for (int64_t i = rank - 1; i > axis; --i) {
                int64_t dim = input.shape(i);
                inner_off += (rem % dim) * input.stride_elems(i);
                rem /= dim;
            }

            // Step 1: Find max for numerical stability
            float max_val = -FLT_MAX;
            for (int64_t k = 0; k < D; ++k) {
                float v = in_ptr[base + k * axis_elems + inner_off];
                if (v > max_val) { max_val = v; }
            }

            // Step 2: Compute sum of exp((x - max) / T)
            float sum_exp = 0.0f;
            for (int64_t k = 0; k < D; ++k) {
                float v = in_ptr[base + k * axis_elems + inner_off];
                sum_exp += std::exp((v - max_val) * inv_T);
            }

            if (log_softmax) {
                // log_softmax = (x - max) / T - log(sum_exp)
                float log_sum = std::log(sum_exp);
                for (int64_t k = 0; k < D; ++k) {
                    float v = in_ptr[base + k * axis_elems + inner_off];
                    float val = (v - max_val) * inv_T - log_sum;
                    out_ptr[base + k * axis_elems + inner_off] = val;
                }
            } else {
                // softmax = exp((x - max) / T) / sum_exp
                float inv_sum = 1.0f / sum_exp;
                for (int64_t k = 0; k < D; ++k) {
                    float v = in_ptr[base + k * axis_elems + inner_off];
                    float val = std::exp((v - max_val) * inv_T) * inv_sum;
                    out_ptr[base + k * axis_elems + inner_off] = val;
                }
            }
        }
    };

    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, outer_size, process_row);
    } else {
        for (int64_t i = 0; i < outer_size; ++i) {
            process_row(i);
        }
    }
}

}  // namespace nnops::backend::cpu::reference
