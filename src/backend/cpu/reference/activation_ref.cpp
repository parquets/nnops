/// @file activation_ref.cpp
/// @brief Naive CPU reference implementation of activation functions.
///
/// Uses a functor-per-type pattern dispatched at runtime via switch.
///
/// Quantized input/output (s8/u8 → s8/u8) is handled by a scalar fused path:
/// dequantize (int → f32), apply the activation, then re-quantize (f32 → int).
/// Granularity is PerTensor / PerToken only.

#include "nnops/ops/activation.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/assert.hpp"

#include <cmath>
#include <algorithm>
#include <cstdint>

namespace nnops::backend::cpu::reference {

namespace {

/// Scalar activation (f32). Used by the quantized path.
inline float activation_scalar(ActivationType type, float x, float alpha, float beta) {
    switch (type) {
    case ActivationType::Relu:      return x > 0.0f ? x : 0.0f;
    case ActivationType::LeakyRelu: return x > 0.0f ? x : alpha * x;
    case ActivationType::Sigmoid:   return 1.0f / (1.0f + std::exp(-x));
    case ActivationType::Tanh:      return std::tanh(x);
    case ActivationType::Gelu: {
        const float c = 0.7978845608028654f;  // sqrt(2/pi)
        return 0.5f * x * (1.0f + std::tanh(c * (x + 0.044715f * x * x * x)));
    }
    case ActivationType::Silu:      return x / (1.0f + std::exp(-x));
    case ActivationType::HardSwish: {
        float relu6 = std::min(std::max(x + 3.0f, 0.0f), 6.0f);
        return x * relu6 * (beta / 6.0f);
    }
    case ActivationType::Elu:       return x > 0.0f ? x : alpha * (std::exp(x) - 1.0f);
    }
    return x;
}

}  // anonymous namespace

// Fused quantized activation (defined below activation_ref).
void activation_quant_ref(const ActivationAttributes& attrs,
                          TensorView& output,
                          std::span<const TensorView> inputs);

template <typename F>
static void element_wise_compute(TensorView& output,
                                  const TensorView& input,
                                  const ComputeContext& ctx,
                                  F func,
                                  bool add_to)
{
    const int64_t rank = input.rank();

    // Rank 0 (scalar) or rank 1 (flat): use direct indexing
    if (rank < 2) {
        const int64_t N = (rank == 0) ? 1 : input.numel();
        const auto* in_ptr  = input.ptr<float>();
        auto* out_ptr = output.ptr<float>();
        const auto write = [&](int64_t i, float v) {
            out_ptr[i] = add_to ? out_ptr[i] + v : v;
        };
        ctx.cpu.run(0, N,
                [&](int64_t idx) { write(idx, func(in_ptr[idx])); });
        return;
    }

    // Rank >= 2: row-by-row iteration, respecting pitch for packed layouts
    const int64_t num_rows = input.total_rows();
    const int64_t last_dim = input.shape(rank - 1) * input.channel_pack_size();
    const int64_t in_rs = input.row_stride_elems();
    const int64_t out_rs = output.row_stride_elems();
    const auto* in_ptr  = input.ptr<float>();
    auto* out_ptr = output.ptr<float>();

    ctx.cpu.run(0, num_rows, [&](int64_t r) {
            const float* in_row = in_ptr + r * in_rs;
            float* out_row = out_ptr + r * out_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float v = func(in_row[i]);
                out_row[i] = add_to ? out_row[i] + v : v;
            }
        });
}

void activation_ref(const ActivationAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* /*workspace*/)
{
    const auto& input = inputs[0];

    // Quantized path: both input and output must be s8/u8 (no float↔int mixing).
    const bool in_is_int  = is_quantized_dtype(input.data_type());
    const bool out_is_int = is_quantized_dtype(output.data_type());
    if (in_is_int || out_is_int) {
        activation_quant_ref(attrs, output, inputs);
        return;
    }

    switch (attrs.type) {
    case ActivationType::Relu: {
        const auto fn = [](float x) -> float { return x > 0.0f ? x : 0.0f; };
        element_wise_compute(output, input, ctx, fn, attrs.add_to);
        break;
    }
    case ActivationType::LeakyRelu: {
        const float alpha = attrs.alpha;
        const auto fn = [alpha](float x) -> float {
            return x > 0.0f ? x : alpha * x;
        };
        element_wise_compute(output, input, ctx, fn, attrs.add_to);
        break;
    }
    case ActivationType::Sigmoid: {
        const auto fn = [](float x) -> float {
            return 1.0f / (1.0f + std::exp(-x));
        };
        element_wise_compute(output, input, ctx, fn, attrs.add_to);
        break;
    }
    case ActivationType::Tanh: {
        const auto fn = [](float x) -> float {
            return std::tanh(x);
        };
        element_wise_compute(output, input, ctx, fn, attrs.add_to);
        break;
    }
    case ActivationType::Gelu: {
        const auto fn = [](float x) -> float {
            // GELU, tanh approximation — matches the kernel's SIMD path.
            const float c = 0.7978845608028654f;  // sqrt(2/pi)
            return 0.5f * x * (1.0f + std::tanh(c * (x + 0.044715f * x * x * x)));
        };
        element_wise_compute(output, input, ctx, fn, attrs.add_to);
        break;
    }
    case ActivationType::Silu: {
        const auto fn = [](float x) -> float {
            return x / (1.0f + std::exp(-x));  // x * sigmoid(x)
        };
        element_wise_compute(output, input, ctx, fn, attrs.add_to);
        break;
    }
    case ActivationType::HardSwish: {
        const float beta = attrs.beta;
        const auto fn = [beta](float x) -> float {
            float relu6 = std::min(std::max(x + 3.0f, 0.0f), 6.0f);
            return x * relu6 * (beta / 6.0f);
        };
        element_wise_compute(output, input, ctx, fn, attrs.add_to);
        break;
    }
    case ActivationType::Elu: {
        const float alpha = attrs.alpha;
        const auto fn = [alpha](float x) -> float {
            return x > 0.0f ? x : alpha * (std::exp(x) - 1.0f);
        };
        element_wise_compute(output, input, ctx, fn, attrs.add_to);
        break;
    }
    }
}

void activation_quant_ref(const ActivationAttributes& attrs,
                          TensorView& output,
                          std::span<const TensorView> inputs)
{
    const auto& input = inputs[0];
    const int64_t numel = input.numel();
    if (numel == 0) { return; }
    NNOPS_ASSERT(!attrs.add_to);

    const DataType in_dtype  = input.data_type();   // s8 or u8
    const DataType out_dtype = output.data_type();  // s8 or u8

    const auto& in_qp  = input.quant_params();
    const auto& out_qp = output.quant_params();
    const bool in_per_token  = in_qp.granularity  == QuantGranularity::PerToken  && in_qp.scale_data != nullptr;
    const bool out_per_token = out_qp.granularity == QuantGranularity::PerToken && out_qp.scale_data != nullptr;
    const int64_t last_dim = (input.rank() >= 1) ? input.shape(input.rank() - 1) : 1;

    for (int64_t flat = 0; flat < numel; ++flat) {
        const int64_t in_idx  = in_per_token  ? (flat / last_dim) : 0;
        const int64_t out_idx = out_per_token ? (flat / last_dim) : 0;

        const float in_s = in_per_token ? in_qp.scale_data[in_idx] : in_qp.scale;
        const float in_z = in_per_token
            ? (in_qp.zero_point_data != nullptr ? static_cast<float>(in_qp.zero_point_data[in_idx]) : 0.0f)
            : static_cast<float>(in_qp.zero_point);
        const float x = (in_dtype == DataType::s8)
            ? (static_cast<float>(input.ptr<int8_t>()[flat]) - in_z) * in_s
            : (static_cast<float>(input.ptr<uint8_t>()[flat]) - in_z) * in_s;

        const float y = activation_scalar(attrs.type, x, attrs.alpha, attrs.beta);

        const float out_s = out_per_token ? out_qp.scale_data[out_idx] : out_qp.scale;
        const float out_z = out_per_token
            ? (out_qp.zero_point_data != nullptr ? static_cast<float>(out_qp.zero_point_data[out_idx]) : 0.0f)
            : static_cast<float>(out_qp.zero_point);
        const int32_t qi = static_cast<int32_t>(std::nearbyintf(y / out_s) + out_z);
        if (out_dtype == DataType::s8) {
            const int32_t q = std::max<int32_t>(-128, std::min<int32_t>(127, qi));
            output.ptr<int8_t>()[flat] = static_cast<int8_t>(q);
        } else {
            const int32_t q = std::max<int32_t>(0, std::min<int32_t>(255, qi));
            output.ptr<uint8_t>()[flat] = static_cast<uint8_t>(q);
        }
    }
}

}  // namespace nnops::backend::cpu::reference
