#pragma once
/// @file epilogue_impl.hpp
/// @brief CPU backend implementations of apply_epilogue.
///
/// Contains:
///   1. Scalar apply_epilogue (moved from include/nnops/core/epilogue.hpp)
///   2. SIMD vector apply_epilogue for f32x8 and f16x8
///   3. matmul_epilogue_inplace — fused bias + epilogue for MatMul output
///
/// Formulas match activation_ref.cpp exactly.

#include "nnops/core/epilogue.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "simd_kernel/activation_kernels.hpp"

#include <cmath>
#include <algorithm>

// ============================================================
// Scalar apply_epilogue (in the nnops namespace, matching
// the declarations in nnops/core/epilogue.hpp)
// ============================================================

namespace nnops {

/// Apply an epilogue to a single scalar output value (activation types).
/// Returns the value unchanged when type == None (identity).
///
/// Formulas match activation_ref.cpp exactly.
inline float apply_epilogue(const Epilogue& ep, float x) {
    switch (ep.type) {
    case EpilogueActivateType::None:
        return x;
    case EpilogueActivateType::Relu:
        return x > 0.0f ? x : 0.0f;
    case EpilogueActivateType::LeakyRelu:
        return x > 0.0f ? x : ep.alpha * x;
    case EpilogueActivateType::Sigmoid:
        return 1.0f / (1.0f + std::exp(-x));
    case EpilogueActivateType::Tanh:
        return std::tanh(x);
    case EpilogueActivateType::Gelu: {
        constexpr float c = 0.7978845608028654f;  // sqrt(2/pi)
        return 0.5f * x * (1.0f + std::tanh(c * (x + 0.044715f * x * x * x)));
    }
    case EpilogueActivateType::Silu:
        return x / (1.0f + std::exp(-x));  // x * sigmoid(x)
    case EpilogueActivateType::HardSwish: {
        float relu6 = std::min(std::max(x + 3.0f, 0.0f), 6.0f);
        return x * relu6 * (ep.beta / 6.0f);
    }
    case EpilogueActivateType::Elu:
        return x > 0.0f ? x : ep.alpha * (std::exp(x) - 1.0f);
    }
    return x;
}

/// Apply an epilogue to a single scalar output value at a given channel index.
///
/// For activation types: delegates to the scalar overload (channel is ignored).
/// For dequantize / requantize (future): uses per-channel or per-tensor
/// quantization parameters.
inline float apply_epilogue(const Epilogue& ep, float x, int64_t /*channel*/) {
    // Future quantize / dequantize types will use channel here:
    // case EpilogueActivateType::Dequantize: {
    //     int64_t idx = (ep.quant_param_count > 0)
    //         ? std::min(channel, ep.quant_param_count - 1) : 0;
    //     float scale = ep.quant_scales[idx];
    //     float zp = ep.quant_zero_points
    //         ? static_cast<float>(ep.quant_zero_points[idx]) : 0.0f;
    //     return (x - zp) * scale;
    // }
    return apply_epilogue(ep, x);
}

}  // namespace nnops

// ============================================================
// SIMD vector apply_epilogue — unified template
// ============================================================

namespace nnops::backend::cpu {

using namespace simd;  // v_load, v_zero, v_set1, etc.
using namespace nnops::kernel;  // pure SIMD kernels: relu, gelu, sigmoid, ...

/// Deduce the SIMD vector type for a given data pointer type T*.
/// v_load(float*) → v_f32x8, v_load(half*) → v_f16x8.
template <typename T>
using vec_for = decltype(v_load(std::declval<const T*>()));

/// Apply epilogue to a SIMD vector (unified f32/f16 via type tag dispatch).
///
/// The type_tag pointer is used for v_set1/v_zero overload resolution,
/// matching the pattern in activation.cpp. No per-type duplication needed.
///
/// All activation types use min/max decomposition (no comparison+blend),
/// so the kernel works uniformly across f32x8, f16x8, and all backends.
template <typename T>
inline vec_for<T> apply_epilogue_vec(const Epilogue& ep,
                                      const T* type_tag,
                                      vec_for<T> vx) {
    switch (ep.type) {
    case EpilogueActivateType::None:
        return vx;
    case EpilogueActivateType::Relu:
        return relu(vx, v_zero(type_tag));
    case EpilogueActivateType::LeakyRelu:
        return leaky_relu(vx, v_zero(type_tag),
                                 v_set1(type_tag, ep.alpha));
    case EpilogueActivateType::Sigmoid:
        return sigmoid(vx, v_set1(type_tag, 1.0f));
    case EpilogueActivateType::Tanh:
        return tanh(vx);
    case EpilogueActivateType::Gelu: {
        constexpr float c = 0.7978845608028654f;
        return gelu(vx,
            v_set1(type_tag, 0.5f),
            v_set1(type_tag, 1.0f),
            v_set1(type_tag, c),
            v_set1(type_tag, 0.044715f));
    }
    case EpilogueActivateType::Silu:
        return silu(vx, v_set1(type_tag, 1.0f));
    case EpilogueActivateType::HardSwish:
        return hard_swish(vx,
            v_zero(type_tag),
            v_set1(type_tag, 3.0f),
            v_set1(type_tag, 6.0f),
            v_set1(type_tag, ep.beta / 6.0f));
    case EpilogueActivateType::Elu:
        return elu(vx,
            v_zero(type_tag),
            v_set1(type_tag, ep.alpha),
            v_set1(type_tag, 1.0f));
    }
    return vx;
}

// ============================================================
// matmul_epilogue_inplace — fused bias + epilogue for MatMul output
// ============================================================

/// Apply bias addition and epilogue activation to a MatMul output matrix.
///
/// Processes output in-place: for each of the M rows, applies per-column
/// bias (if non-null) and then the epilogue activation function.
///
/// Uses SIMD for the bulk of each row; scalar for the tail.
///
/// @tparam T      Data type (float or half)
/// @param M        Number of rows in the output matrix
/// @param N        Number of columns in the output matrix
/// @param data     Output matrix data (row-major, ld stride between rows)
/// @param ld       Leading dimension (stride in elements between consecutive rows)
/// @param bias     Per-column bias array of length N (may be nullptr)
/// @param epilogue Epilogue descriptor (None = identity, no-op)
template <class T>
void matmul_epilogue_inplace(int M, int N, T* data, int ld,
                              const T* bias, const Epilogue& epilogue) {
    constexpr int L = simd_lane_for<T>;

    for (int m = 0; m < M; ++m) {
        T* row = data + m * ld;
        int n = 0;

        // ---- SIMD loop ----
        for (; n + L <= N; n += L) {
            auto v_data = v_load(row + n);

            // Fused bias addition
            if (bias) {
                auto v_bias = v_load(bias + n);
                v_data = v_add(v_data, v_bias);
            }

            // Epilogue activation (type dispatched via pointer overload)
            v_data = apply_epilogue_vec(epilogue, row + n, v_data);

            v_store(row + n, v_data);
        }

        // ---- Scalar tail ----
        for (; n < N; ++n) {
            float val = s_load(&row[n]);
            if (bias) {
                val += s_load(&bias[n]);
            }
            val = nnops::apply_epilogue(epilogue, val);
            s_store(&row[n], val);
        }
    }
}

}  // namespace nnops::backend::cpu
