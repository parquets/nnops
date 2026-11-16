/// @file matmul_epilogue.cuh
/// @brief The operator-level GEMM epilogue, one element at a time, free of the
///        CuTe types the engine that used to own it carried.
///
/// Order is the operator contract: scale the product, add the bias, add beta
/// times the old destination, clamp, activate, then add the residual after the
/// activation (Linear's add_to and Conv2D's residual both ask for that order).

#pragma once

#include <nnops/core/epilogue.hpp>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace nnops::backend::cuda {

/// A flat batch index resolved against up to four (size, stride) axes. Both
/// hand-written GEMMs fold the batch into blockIdx.z beside the k splits, so
/// they resolve the operand offsets with the same helper.
__device__ inline int64_t manual_batch_offset(const int64_t* size, const int64_t* stride, int rank,
                                              int64_t idx)
{
    int64_t off = 0;
    for (int d = rank - 1; d >= 0; --d) {
        const int64_t n = size[d] > 0 ? size[d] : 1;
        off += (idx % n) * stride[d];
        idx /= n;
    }
    return off;
}

struct ManualEpilogue {
    const void* bias = nullptr;
    bool bias_row = false; ///< the bias is indexed by row, else by column
    float alpha = 1.0f;
    float beta = 0.0f;
    const void* residual = nullptr;
    int64_t ldr = 0;
    EpilogueActivateType type = EpilogueActivateType::None;
    float ep_alpha = 0.0f;
    float ep_beta = 1.0f;
    float min_clip = -INFINITY;
    float max_clip = INFINITY;
};

__device__ inline float manual_apply_activation(EpilogueActivateType ep, float a, float b, float x)
{
    switch (ep) {
    case EpilogueActivateType::None:
        return x;
    case EpilogueActivateType::Relu:
        return fmaxf(x, 0.0f);
    case EpilogueActivateType::Relu6:
        return fminf(fmaxf(x, 0.0f), 6.0f);
    case EpilogueActivateType::LeakyRelu:
        return (x > 0.0f) ? x : a * x;
    case EpilogueActivateType::Sigmoid:
        return 1.0f / (1.0f + expf(-x));
    case EpilogueActivateType::Tanh:
        return tanhf(x);
    case EpilogueActivateType::Gelu: {
        constexpr float c = 0.7978845608028654f;
        return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
    }
    case EpilogueActivateType::Silu:
        return x / (1.0f + expf(-x));
    case EpilogueActivateType::HardSwish: {
        const float r = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
        return x * r * (b / 6.0f);
    }
    case EpilogueActivateType::Elu:
        return (x > 0.0f) ? x : a * (expf(x) - 1.0f);
    }
    return x;
}

template <class Elem> __device__ __forceinline__ float manual_elem_load(const Elem* p, int64_t i)
{
    return static_cast<float>(p[i]);
}

template <> __device__ __forceinline__ float manual_elem_load<__half>(const __half* p, int64_t i)
{
    return __half2float(p[i]);
}

/// One destination element: `dst` points at C[m][n] and `dst[n]` is its old
/// value for the beta term.
template <class Elem>
__device__ __forceinline__ float manual_epilogue(const ManualEpilogue& e, float acc, int64_t m,
                                                 int64_t n, const Elem* dst)
{
    float x = e.alpha * acc;
    if (e.bias != nullptr) {
        const auto* b = static_cast<const Elem*>(e.bias);
        x += manual_elem_load<Elem>(b, e.bias_row ? m : n);
    }
    if (e.beta != 0.0f) {
        x += e.beta * manual_elem_load<Elem>(dst, n);
    }
    x = fminf(fmaxf(x, e.min_clip), e.max_clip);
    x = manual_apply_activation(e.type, e.ep_alpha, e.ep_beta, x);
    if (e.residual != nullptr) {
        const auto* r = static_cast<const Elem*>(e.residual);
        x += manual_elem_load<Elem>(r, m * e.ldr + n);
    }
    return x;
}

} // namespace nnops::backend::cuda
