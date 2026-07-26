/// @file activation.cpp
/// @brief SIMD-optimized CPU implementation of activation functions.
///
/// All 8 activation types are vectorized with the nnops SIMD abstraction layer.
/// Supports both f32 (v_f32x8) and f16 (v_f16x8) via a single templated implementation
/// that uses the generic v_load/v_store/v_set1/s_load/s_store API.
///
/// On x86_64 this compiles to AVX2+FMA (f32) or F16C+AVX2 (f16); on AArch64 to
/// NEON (v_f32x8 emulated, v_f16x8 native on ARMv8.2+); on RISC-V to V extension.
///
/// Design decisions:
///   1. No blend/select needed — all conditionals are decomposed via v_max/v_min
///   2. Row-by-row processing respects pitch (non-contiguous tensors supported)
///   3. simd_lane_for<T> selects lane count (8 for both f32 and f16)
///   4. add_to is checked inside the loop body to avoid duplicating the entire
///      row-processing structure (branch predictor handles the invariant branch).

#include "nnops/ops/activation.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "activation_kernels.hpp"

#include <cmath>

namespace nnops::backend::cpu {

using namespace nnops::simd;

// ============================================================
// Shared row-processing helper — eliminates the identical
// row-loop + SIMD/scalar + add_to boilerplate across all 8
// activation types. Each case now passes just its SIMD kernel
// and scalar formula as lambdas.
// ============================================================

template <typename T, typename SimdK, typename ScalarK>
inline void process_activation_rows(
    const T* in_ptr, T* out_ptr,
    int64_t num_rows, int64_t last_dim,
    int64_t in_row_stride, int64_t out_row_stride,
    bool add_to,
    SimdK&& simd_kernel,
    ScalarK&& scalar_kernel)
{
    constexpr int L = simd_lane_for<T>;
    for (int64_t r = 0; r < num_rows; ++r) {
        const T* in_row = in_ptr + r * in_row_stride;
        T* out_row = out_ptr + r * out_row_stride;
        int64_t i = 0;
        for (; i + L <= last_dim; i += L) {
            v_store_add(out_row + i, simd_kernel(v_load(in_row + i)), add_to);
        }
        for (; i < last_dim; ++i) {
            s_store_add(&out_row[i], scalar_kernel(s_load(&in_row[i])), add_to);
        }
    }
}

// ============================================================
// Templated implementation (f32 and f16)
// ============================================================

template <typename T>
void activation_impl(const ActivationAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs)
{
    const auto& input = inputs[0];
    const int64_t total = input.numel();
    if (total == 0) { return; }

    const int64_t rank = input.rank();

    // Layout: treat the innermost dimension as contiguous "row" elements,
    // and advance by row_stride_elems between rows (accounts for pitch padding).
    const int64_t last_dim = (rank >= 1) ? input.shape(rank - 1) : 1;
    const int64_t num_rows = total / last_dim;
    const int64_t in_row_stride = input.row_stride_elems();
    const int64_t out_row_stride = output.row_stride_elems();

    const auto* in_ptr  = input.ptr<T>();
    auto* out_ptr = output.ptr<T>();
    const bool add_to = attrs.add_to;

    // Pre-computed constants used by multiple cases.
    const auto vzero = v_zero(in_ptr);
    const auto vone  = v_set1(in_ptr, 1.0f);

    switch (attrs.type) {

    case ActivationType::Relu:
        process_activation_rows(in_ptr, out_ptr, num_rows, last_dim,
            in_row_stride, out_row_stride, add_to,
            [&](auto x) { return kernel_relu(x, vzero); },
            [](float v) { return v > 0.0f ? v : 0.0f; });
        break;

    case ActivationType::LeakyRelu: {
        const auto a8 = v_set1(in_ptr, attrs.alpha);
        process_activation_rows(in_ptr, out_ptr, num_rows, last_dim,
            in_row_stride, out_row_stride, add_to,
            [&](auto x) { return kernel_leaky_relu(x, vzero, a8); },
            [&](float v) { return v > 0.0f ? v : attrs.alpha * v; });
        break;
    }

    case ActivationType::Sigmoid:
        process_activation_rows(in_ptr, out_ptr, num_rows, last_dim,
            in_row_stride, out_row_stride, add_to,
            [&](auto x) { return kernel_sigmoid(x, vone); },
            [](float v) { return 1.0f / (1.0f + std::exp(-v)); });
        break;

    case ActivationType::Tanh:
        process_activation_rows(in_ptr, out_ptr, num_rows, last_dim,
            in_row_stride, out_row_stride, add_to,
            [&](auto x) { return kernel_tanh(x); },
            [](float v) { return std::tanh(v); });
        break;

    case ActivationType::Gelu: {
        const auto half8  = v_set1(in_ptr, 0.5f);
        const auto c8     = v_set1(in_ptr, 0.7978845608028654f);
        const auto coeff8 = v_set1(in_ptr, 0.044715f);
        process_activation_rows(in_ptr, out_ptr, num_rows, last_dim,
            in_row_stride, out_row_stride, add_to,
            [&](auto x) { return kernel_gelu(x, half8, vone, c8, coeff8); },
            [](float x) {
                return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
            });
        break;
    }

    case ActivationType::Silu:
        process_activation_rows(in_ptr, out_ptr, num_rows, last_dim,
            in_row_stride, out_row_stride, add_to,
            [&](auto x) { return kernel_silu(x, vone); },
            [](float x) { return x / (1.0f + std::exp(-x)); });
        break;

    case ActivationType::HardSwish: {
        const float bd6 = attrs.beta / 6.0f;
        const auto three8 = v_set1(in_ptr, 3.0f);
        const auto six8   = v_set1(in_ptr, 6.0f);
        const auto scale8 = v_set1(in_ptr, bd6);
        process_activation_rows(in_ptr, out_ptr, num_rows, last_dim,
            in_row_stride, out_row_stride, add_to,
            [&](auto x) { return kernel_hard_swish(x, vzero, three8, six8, scale8); },
            [bd6](float x) { return x * std::min(std::max(x + 3.0f, 0.0f), 6.0f) * bd6; });
        break;
    }

    case ActivationType::Elu: {
        const auto a8 = v_set1(in_ptr, attrs.alpha);
        process_activation_rows(in_ptr, out_ptr, num_rows, last_dim,
            in_row_stride, out_row_stride, add_to,
            [&](auto x) { return kernel_elu(x, vzero, a8, vone); },
            [&](float x) { return x > 0.0f ? x : attrs.alpha * (std::exp(x) - 1.0f); });
        break;
    }

    }  // switch
}

// ============================================================
// Main entry point with dtype dispatch
// ============================================================

void activation_cpu(const ActivationAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& /*ctx*/,
                     void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        activation_impl<float>(attrs, output, inputs);
        return;
    case DataType::f16:
        activation_impl<half>(attrs, output, inputs);
        return;
    default:
        NNOPS_ASSERT(!"activation_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
