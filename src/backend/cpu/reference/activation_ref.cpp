/// @file activation_ref.cpp
/// @brief Naive CPU reference implementation of activation functions.
///
/// Uses a functor-per-type pattern dispatched at runtime via switch.

#include "nnops/ops/activation.hpp"
#include "nnops/core/parallel_for.hpp"

#include <cmath>
#include <algorithm>

namespace nnops::backend::cpu::reference {

// ============================================================
// Functor definitions — one per activation type
// ============================================================

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
        if (ctx.cpu_parallel_for) {
            ctx.cpu_parallel_for(0, N,
                [&](int64_t idx) { write(idx, func(in_ptr[idx])); });
        } else {
            for (int64_t i = 0; i < N; ++i) {
                write(i, func(in_ptr[i]));
            }
        }
        return;
    }

    // Rank >= 2: row-by-row iteration, respecting pitch for packed layouts
    const int64_t num_rows = input.total_rows();
    const int64_t last_dim = input.shape(rank - 1) * input.channel_pack_size();
    const int64_t in_rs = input.row_stride_elems();
    const int64_t out_rs = output.row_stride_elems();
    const auto* in_ptr  = input.ptr<float>();
    auto* out_ptr = output.ptr<float>();

    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, num_rows, [&](int64_t r) {
            const float* in_row = in_ptr + r * in_rs;
            float* out_row = out_ptr + r * out_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float v = func(in_row[i]);
                out_row[i] = add_to ? out_row[i] + v : v;
            }
        });
    } else {
        for (int64_t r = 0; r < num_rows; ++r) {
            const float* in_row = in_ptr + r * in_rs;
            float* out_row = out_ptr + r * out_rs;
            for (int64_t i = 0; i < last_dim; ++i) {
                float v = func(in_row[i]);
                out_row[i] = add_to ? out_row[i] + v : v;
            }
        }
    }
}

void activation_ref(const ActivationAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    void* /*workspace*/)
{
    const auto& input = inputs[0];

    // Select the activation function at runtime
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
            // GELU approximation: x * sigmoid(1.702 * x)
            // or exact: 0.5 * x * (1 + erf(x / sqrt(2)))
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

}  // namespace nnops::backend::cpu::reference
