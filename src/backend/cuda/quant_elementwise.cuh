#pragma once
/// @file quant_elementwise.cuh
/// @brief The fused dequantize → op → requantize surface the CUDA element-wise
///        and row routes share, plus the staging a per-row parameter array needs.
///
/// The CPU's fused paths (common/elementwise.hpp, norm's row route, pooling_i8)
/// dequantize an s8/u8 element as `(x - zp) * scale`, run the op in f32, and
/// requantize as `round(v / scale) + zp`, round-to-nearest-even, clamped to the
/// storage type. The device arithmetic here is the same, with the dequantize
/// folded into the single FMA the CPU's vector path uses:
///
///     x*scale - zp*scale
///
/// One `RowParam` holds a row's four derived constants, so a kernel that walks
/// a row decodes its scale/zero_point once and then runs FMA + rint on every
/// element.
///
/// Per-tensor parameters ride by value. PerToken arrays are host-side metadata
/// on the view, as they are for Embed's weight, so a route that reads them
/// stages a device copy once per call (`stage_qparams`). PerChannel is treated
/// as per-tensor, which is what the CPU's fused transform does too.
///
/// The fused element-wise kernels walk the tensor as M dense rows of
/// shape(rank-1) — the decomposition the CPU's quantized transform uses — so a
/// pitched view is read as if it were dense, exactly as the CPU reads it.

#include "cuda_common.cuh"
#include "nnops/core/quant_params.hpp"
#include "nnops/core/status.hpp"

#include <cstdint>

namespace nnops::backend::cuda {
namespace qelem {

// ============================================================
// Per-row parameters
// ============================================================

/// One row's derived constants: the FMA addend for the dequantize, and the
/// multiply factor for the requantize.
struct RowParam {
    float scale;
    float neg_zero_scale;
    float inv_scale;
    float zero;
};

__host__ __device__ inline RowParam make_row_param(float scale, int32_t zero) noexcept
{
    RowParam p;
    p.scale = scale;
    p.neg_zero_scale = -static_cast<float>(zero) * scale;
    p.inv_scale = 1.0f / scale;
    p.zero = static_cast<float>(zero);
    return p;
}

/// A view's quantization resolved to "one scalar or a per-row array".
struct ParamRef {
    const float* scale_rows = nullptr;
    const int32_t* zp_rows = nullptr;
    float scale = 1.0f;
    int32_t zero = 0;
    bool per_row = false;
};

__host__ __device__ inline RowParam row_param(const ParamRef& r, int64_t row) noexcept
{
    if (!r.per_row) {
        return make_row_param(r.scale, r.zero);
    }
    return make_row_param(r.scale_rows[row], r.zp_rows != nullptr ? r.zp_rows[row] : 0);
}

// ============================================================
// Element conversion
// ============================================================

__device__ inline float dequant(int8_t x, const RowParam& p) noexcept
{
    return fmaf(static_cast<float>(x), p.scale, p.neg_zero_scale);
}

__device__ inline float dequant(uint8_t x, const RowParam& p) noexcept
{
    return fmaf(static_cast<float>(x), p.scale, p.neg_zero_scale);
}

/// f32 → s8/u8: scale and shift, round to nearest even, saturate.
template <class T> __device__ inline T requant(float v, const RowParam& p) noexcept
{
    const float q = rintf(fmaf(v, p.inv_scale, p.zero));
    if constexpr (std::is_same_v<T, int8_t>) {
        return static_cast<int8_t>(fminf(fmaxf(q, -128.0f), 127.0f));
    } else {
        return static_cast<uint8_t>(fminf(fmaxf(q, 0.0f), 255.0f));
    }
}

/// A quantized storage type stores through `requant`; every float type through
/// the ordinary narrow.
template <class T> __device__ inline void store_elem(T* p, float v, const RowParam& qp) noexcept
{
    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
        *p = requant<T>(v, qp);
    } else {
        s_store(p, v);
    }
}

/// 16 one-byte elements — one 128-bit lane.
template <class T> struct alignas(16) pack16 {
    T data[16];
};

// ============================================================
// Host-side staging for a per-row parameter array
// ============================================================

/// A host-to-device staging block. Async allocation when the pool allows it,
/// plain cudaMalloc otherwise — the same two arms Embed uses for its weight.
struct Stage {
    void* ptr = nullptr;
    bool async = false;

    bool alloc(int64_t bytes, cudaStream_t stream)
    {
        if (bytes <= 0) {
            return true;
        }
        if (cudaMallocAsync(&ptr, size_t(bytes), stream) == cudaSuccess) {
            async = true;
            return true;
        }
        cudaGetLastError();
        if (cudaMalloc(&ptr, size_t(bytes)) == cudaSuccess) {
            return true;
        }
        cudaGetLastError();
        return false;
    }

    bool upload(const void* src, int64_t bytes, cudaStream_t stream)
    {
        if (bytes <= 0) {
            return true;
        }
        return cudaMemcpyAsync(ptr, src, size_t(bytes), cudaMemcpyHostToDevice, stream) ==
               cudaSuccess;
    }

    void release(cudaStream_t stream)
    {
        if (ptr == nullptr) {
            return;
        }
        if (async) {
            if (cudaFreeAsync(ptr, stream) == cudaSuccess) {
                ptr = nullptr;
                return;
            }
            cudaGetLastError();
        }
        cudaFree(ptr);
        ptr = nullptr;
    }
};

/// Resolve a view's QuantParams into a `ParamRef`, staging the per-row arrays
/// when the granularity is PerToken. `rows` is the number of rows the kernel
/// will index.
inline Status stage_qparams(const QuantParams& qp, int64_t rows, cudaStream_t stream,
                            Stage& scale_stage, Stage& zp_stage, ParamRef& out)
{
    out.per_row = false;
    out.scale = qp.scale;
    out.zero = qp.zero_point;
    if (qp.granularity != QuantGranularity::PerToken || qp.scale_data == nullptr) {
        return Status::ok;
    }
    if (qp.num_scales < rows) {
        return Status::invalid_argument;
    }
    if (!scale_stage.alloc(rows * 4, stream) ||
        !scale_stage.upload(qp.scale_data, rows * 4, stream)) {
        return Status::internal;
    }
    if (qp.zero_point_data != nullptr &&
        (!zp_stage.alloc(rows * 4, stream) ||
         !zp_stage.upload(qp.zero_point_data, rows * 4, stream))) {
        return Status::internal;
    }
    out.per_row = true;
    out.scale_rows = static_cast<const float*>(scale_stage.ptr);
    out.zp_rows = static_cast<const int32_t*>(zp_stage.ptr);
    return Status::ok;
}

} // namespace qelem
} // namespace nnops::backend::cuda
